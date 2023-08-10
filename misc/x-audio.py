try:
	from common import *
except ModuleNotFoundError:
	try:
		f = open("common.py", "rb")
	except FileNotFoundError:
		import os, sys
		sys.path.append(os.path.abspath('..'))
		os.chdir("..")
		f = open("common.py", "rb")
	b = f.read()
	code = compile(b, "common.py", "exec", optimize=1)
	exec(code, globals())


tracebacksuppressor.fn = traceback.print_exc

ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
if ADDRESS == "0.0.0.0":
	ADDRESS = "127.0.0.1"

# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000


def send(*args, escape=True):
	s = " ".join(str(i) for i in args)
	if escape:
		s = "\x00" + s
	if s:
		if s[-1] != "\n":
			s += "\n"
		sys.__stdout__.buffer.write(s.encode("utf-8"))
		sys.__stdout__.flush()

@tracebacksuppressor
def request(s):
	PORT = AUTH["webserver_port"]
	token = AUTH["discord_token"]
	return reqs.next().get(f"http://{ADDRESS}:{PORT}/eval/{token}/{url_parse(s)}", verify=False, timeout=16).text

def submit(s):
	if type(s) not in (bytes, memoryview):
		s = as_str(s).encode("utf-8")
	b = b"~" + base64.b85encode(s) + b"\n"
	resp = sys.__stdout__.buffer.write(b)
	sys.__stdout__.flush()
	return resp

async def respond(s):
	# send("%" + as_str(s))
	k, c = s[1:].rstrip().split(b"~", 1)
	c = memoryview(base64.b85decode(c))
	k = k.decode("ascii")
	# send("%" + as_str(c))
	if c == b"ytdl.update()":
		with tracebacksuppressor:
			await create_future(update_cache)
		res = "None"
	elif c:
		res = None
		try:
			if c[:1] == b"!":
				c = c[1:]
				res = "None"
			if c[:6] == b"await ":
				resp = await eval(c[6:], client._globals)
			else:
				code = None
				try:
					code = compile(c, "<miza>", "eval")
				except SyntaxError:
					pass
				else:
					resp = await create_future(eval, code, client._globals)
				if code is None:
					resp = await create_future(exec, c, client._globals)
		except Exception as ex:
			traceback.print_exc()
			s = f"bot.audio.returns[{k}].set_exception(pickle.loads({repr(pickle.dumps(ex))}))"
			submit(s)
			return
	else:
		return
	if not res:
		res = repr(resp)
		if type(resp) not in (bool, int, float, str, bytes):
			try:
				compile(res, "miza2", "eval")
			except SyntaxError:
				res = repr(str(resp))
	s = f"bot.audio.returns[{k}].set_result({res})"
	# send("%" + as_str(s))
	create_future_ex(submit, s)

async def communicate():
	send("Audio client successfully connected.")
	while True:
		with tracebacksuppressor:
			s = await create_future(sys.stdin.buffer.readline)
			if not s:
				break
			if s.startswith(b"~"):
				create_task(respond(s))
	send("Audio client successfully disconnected.")

def is_strict_running(proc):
	if not proc:
		return
	try:
		if not proc.is_running():
			return False
		if proc.status() == "zombie":
			proc.wait()
			return
		return True
	except AttributeError:
		proc = psutil.Process(proc.pid)
	if not proc.is_running():
		return False
	if proc.status() == "zombie":
		proc.wait()
		return
	return True


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration(filename, _timeout=12):
	command = (
		"./ffprobe",
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"format=duration,bit_rate",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		filename,
	)
	resp = None
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = create_future_ex(proc.wait, timeout=_timeout)
		res = fut.result(timeout=_timeout)
		resp = proc.stdout.read().split()
	except:
		with suppress():
			force_kill(proc)
		with suppress():
			resp = proc.stdout.read().split()
		print_exc()
	try:
		dur = float(resp[0])
	except (IndexError, ValueError, TypeError):
		dur = None
	bps = None
	if resp and len(resp) > 1:
		with suppress(ValueError):
			bps = float(resp[1])
	return dur, bps

DUR_CACHE = {}

def get_duration(filename):
	if filename:
		with suppress(KeyError):
			return DUR_CACHE[filename]
		dur, bps = _get_duration(filename, 4)
		if not dur and is_url(filename):
			with reqs.next().get(filename, headers=Request.header(), stream=True) as resp:
				head = fcdict(resp.headers)
				if "Content-Length" not in head:
					dur = _get_duration(filename, 20)[0]
					DUR_CACHE[filename] = dur
					return dur
				if bps:
					print(head, bps, sep="\n")
					return (int(head["Content-Length"]) << 3) / bps
				ctype = [e.strip() for e in head.get("Content-Type", "").split(";") if "/" in e][0]
				if ctype.split("/", 1)[0] not in ("audio", "video") or ctype == "audio/midi":
					DUR_CACHE[filename] = nan
					return nan
				it = resp.iter_content(65536)
				data = next(it)
			ident = str(magic.from_buffer(data))
			print(head, ident, sep="\n")
			try:
				bitrate = regexp("[0-9.]+\\s.?bps").findall(ident)[0].casefold()
			except IndexError:
				dur = _get_duration(filename, 16)[0]
				DUR_CACHE[filename] = dur
				return dur
			bps, key = bitrate.split(None, 1)
			bps = float(bps)
			if key.startswith("k"):
				bps *= 1e3
			elif key.startswith("m"):
				bps *= 1e6
			elif key.startswith("g"):
				bps *= 1e9
			dur = (int(head["Content-Length"]) << 3) / bps
		DUR_CACHE[filename] = dur
		return dur


players = cdict()


class AudioPlayer(discord.AudioSource):

	vc = None
	listener = None
	listening = False
	# Empty opus packet data
	emptyopus = b"\xfc\xff\xfe"
	silent = False

	@classmethod
	async def join(cls, channel):
		channel = client.get_channel(verify_id(channel))
		self = cls(channel.guild)
		players[channel.guild.id] = concurrent.futures.Future()
		send(self, channel)
		try:
			if not self.vc:
				if channel.guild.me.voice:
					await channel.guild.change_voice_state(channel=None)
				self.vc = await channel.connect(timeout=7, reconnect=True)
		except Exception as ex:
			if channel.guild.id in players:
				players[channel.guild.id].set_exception(ex)
				players.pop(channel.guild.id)
			raise
		else:
			if channel.guild.id in players:
				players[channel.guild.id].set_result(self)
			players[channel.guild.id] = self

	@classmethod
	def from_guild(cls, guild):
		try:
			fut = players[verify_id(guild)]
		except KeyError:
			pass
		else:
			if type(fut) is cls:
				return fut
			return fut.result(timeout=7)
		self = cls(guild)
		if self.vc:
			return self

	def __init__(self, guild=None):
		self.queue = deque(maxlen=2)
		if guild:
			self.vc = client.get_guild(verify_id(guild)).voice_client

	def _listen(self):
		while self.listening:
			break

	def listen(self):
		if self.listener and not self.listener.done():
			self.listening = False
			self.listener.result()
		self.listening = True
		self.listener = create_future_ex(self._listen)
		return self.listener

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if k == "pos":
			if not self.queue or not self.queue[0] or not self.queue[0][0]:
				return 0, 0
			p = self.queue[0][0].pos / 50
			d = self.queue[0][0].duration() or inf
			return min(p, d), d
		try:
			return getattr(self.vc, k)
		except AttributeError:
			if not self.queue:
				raise
		return getattr(self.queue[0][0], k)
	
	def after(self, *args):
		if not self.queue or not self.queue[0]:
			return
		entry = self.queue.popleft()
		create_future_ex(entry[0].close)
		after = entry[1]
		if callable(after):
			after()
		if self.queue:
			with tracebacksuppressor(RuntimeError, discord.ClientException):
				self.vc.play(self, after=self.after)

	def read(self):
		if not self.queue or not self.queue[0]:
			if self.silent:
				self.vc.pause()
			self.silent = True
			return self.emptyopus * 3
		out = b""
		try:
			out = self.queue[0][0].read()
		except (StopIteration, IndexError, discord.oggparse.OggError):
			pass
		except:
			print_exc()
		if not out and self.queue:
			with tracebacksuppressor(StopIteration):
				entry = self.queue.popleft()
				create_future_ex(entry[0].close)
				after = entry[1]
				if callable(after):
					create_future_ex(after)
				if not self.queue:
					return self.emptyopus
				out = self.queue[0][0].read()
		if self.silent:
			self.silent = False
		return out or self.emptyopus

	def play(self, source, after=None):
		if not self.queue:
			self.queue.append(None)
		elif self.queue[0]:
			create_future_ex(self.queue[0][0].close)
		self.queue[0] = (source, after)
		with tracebacksuppressor(RuntimeError, discord.ClientException):
			self.vc.play(self, after=self.after)

	def enqueue(self, source, after=None):
		if not self.queue:
			return self.play(source, after=after)
		if len(self.queue) < 2:
			self.queue.append(None)
		else:
			self.queue[1][0].close()
		self.queue[1] = (source, after)

	def clear_source(self):
		if self.queue:
			source = self.queue[0][0]
			if source:
				source.close()
			self.queue[0] = None
	
	def clear_next(self):
		if len(self.queue) > 1:
			self.queue.pop()[0].close()

	def skip(self):
		if self.queue:
			entry = self.queue.popleft()
			create_future_ex(entry[0].close)
			after = entry[1]
			if callable(after):
				create_future_ex(after)

	def clear(self):
		for entry in tuple(self.queue):
			if entry:
				entry[0].close()
		self.queue.clear()

	def kill(self):
		create_task(self.vc.disconnect(force=True))
		self.clear()
		players.pop(self.guild.id, None)
		self.vc.dead = True
		self.vc = None

	is_opus = lambda self: True
	cleanup = lambda self: None

AP = AudioPlayer


cache = cdict()

def update_cache():
	for item in tuple(cache.values()):
		item.update()

ytdl = cdict(update=update_cache, cache=cache)


# Represents a cached audio file in opus format. Executes and references FFmpeg processes loading the file.
class AudioFile:

	seekable = True
	live = False
	dur = None

	def __init__(self, fn, stream=None, wasfile=False):
		self.file = fn
		self.proc = None
		self.streaming = concurrent.futures.Future()
		self.readable = concurrent.futures.Future()
		if stream is not None:
			self.streaming.set_result(stream)
		self.stream = stream
		self.wasfile = False
		self.loading = self.buffered = self.loaded = wasfile
		if wasfile:
			self.proc = cdict(is_running=lambda: False, kill=lambda: None, status=lambda: None)
		self.expired = False
		self.readers = cdict()
		self.semaphore = Semaphore(1, 1, delay=5)
		self.ensure_time()
		self.webpage_url = None
		cache[fn] = self

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>"

	def load(self, stream=None, check_fmt=False, force=False, webpage_url=None, live=False, seekable=True, duration=None, asap=False):
		self.dur = duration
		if live:
			self.loading = self.buffered = self.loaded = True
			self.live = self.stream = stream
			self.seekable = seekable
			self.proc = None
			return self
		if self.loading and not force:
			return self
		if stream is not None:
			self.stream = stream
			try:
				self.streaming.set_result(stream)
			except concurrent.futures.InvalidStateError:
				pass
		stream = self.stream
		if webpage_url is not None:
			self.webpage_url = webpage_url
		self.loading = True
		# if not asap and not live and is_youtube_stream(stream):
			# fi = "cache/" + str(time.time_ns() + random.randint(1, 1000)) + "~proxy"
			# with suppress():
			# 	stream = proxy_download(stream, fi, timeout=86400)
		ffmpeg = "./ffmpeg"
		if not os.path.exists(ffmpeg):
			ffmpeg = "./ffmpeg"
		fmt = cdc = self.file.rsplit(".", 1)[-1]
		if fmt in ("weba", "webm"):
			fmt = "webm"
			cdc = "libopus"
			cdc2 = "opus"
		if fmt == "ogg":
			cdc = "libopus"
			cdc2 = "opus"
			# cdc = "libvorbis"
			# cdc2 = "vorbis"
		elif fmt == "wav":
			cdc = "pcm_s16le"
			cdc2 = "wav"
		elif fmt == "opus":
			cdc = "libopus"
			cdc2 = "opus"
		# Collects data from source, converts to 48khz 224kbps opus format, outputting to target file
		cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-f", fmt, "-c:a", cdc, "-ar", str(SAMPLE_RATE), "-ac", "2", "-b:a", "196608", "cache/" + self.file]
		# if not stream.startswith("https://cf-hls-media.sndcdn.com/"):
		with suppress():
			if stream.startswith("https://www.yt-download.org/download/"):
				fmt2 = "mp3"
			else:
				fmt2 = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=nokey=1:noprint_wrappers=1", stream])).strip()
			if fmt2 == cdc2:
				cmd = ["./ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-c:a", "copy", "cache/" + self.file]
		self.proc = None
		try:
			try:
				self.proc = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=1048576)
			except:
				send(cmd)
				raise
			fl = 0
			# Attempt to monitor status of output file
			while fl < 4096:
				with Delay(0.1):
					if not is_strict_running(self.proc):
						err = as_str(self.proc.stderr.read())
						if check_fmt is not None:
							if self.webpage_url and ("Server returned 5XX Server Error reply" in err or "Server returned 404 Not Found" in err or "Server returned 403 Forbidden" in err):
								send(err)
								with tracebacksuppressor:
									if "https://cf-hls-media.sndcdn.com/" in stream or expired(stream):
										new_stream = request(f"VOICE.get_best_audio(VOICE.ytdl.extract_from({repr(self.webpage_url)}))")
									else:
										new_stream = request(f"VOICE.get_best_audio(VOICE.ytdl.extract_backup({repr(self.webpage_url)}))")
									if new_stream:
										return self.load(eval_json(new_stream), check_fmt=None, force=True)
							new = None
							with suppress(ValueError):
								new = request(f"VOICE.select_and_convert({repr(stream)})")
							if new not in (None, "null"):
								return self.load(eval_json(new), check_fmt=None, force=True)
						send(self.proc.args)
						if err:
							ex = RuntimeError(err)
						else:
							ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
						self.readable.set_exception(ex)
						raise ex
				try:
					fl = os.path.getsize("cache/" + self.file)
				except FileNotFoundError:
					fl = 0
			self.buffered = True
			self.ensure_time()
			# print(self.file, "buffered", fl)
		except Exception as ex:
			# File errored, remove from cache and kill corresponding FFmpeg process if possible
			ytdl.cache.pop(self.file, None)
			if self.proc is not None:
				with suppress():
					force_kill(self.proc)
			with suppress():
				os.remove("cache/" + self.file)
			self.readable.set_exception(ex)
			raise
		self.readable.set_result(self)
		return self

	# Touch the file to update its cache time.
	ensure_time = lambda self: setattr(self, "time", utc())

	# Update event run on all cached files
	def update(self):
		if not self.live:
			# Check when file has been fully loaded
			if self.buffered and not is_strict_running(self.proc):
				if not self.loaded:
					self.loaded = True
					if not is_url(self.stream):
						retry(os.remove, self.stream, attempts=3, delay=0.5)
					try:
						fl = os.path.getsize("cache/" + self.file)
					except FileNotFoundError:
						fl = 0
					# print(self.file, "loaded", fl)
			# Touch file if file is currently in use
		if self.readers:
			self.ensure_time()
			return
		# Remove any unused file that has been left for a long time
		if utc() - self.time > 86400:
			try:
				fl = os.path.getsize("cache/" + self.file)
			except FileNotFoundError:
				fl = 0
				if self.buffered:
					self.time = -inf
			ft = 86400 / (math.log2(fl / 16777216 + 1) + 1)
			if ft > 86400 * 14:
				ft = 86400 * 14
			if utc() - self.time > ft:
				self.destroy()

	# Creates a reader object that either reads bytes or opus packets from the file.
	def open(self, key=None):
		self.ensure_time()
		if self.proc is None and not self.loaded:
			raise ProcessLookupError
		f = open("cache/" + self.file, "rb")
		it = discord.oggparse.OggStream(f).iter_packets()

		reader = cdict(
			pos=0,
			byte_pos=0,
			file=f,
			it=it,
			_read=lambda self, *args: f.read(args),
			closed=False,
			advanced=False,
			is_opus=lambda self: True,
			key=key,
			duration=self.duration,
			af=self,
		)

		def read():
			try:
				out = next(reader.it, b"")
			except ValueError:
				f = open("cache/" + self.file, "rb")
				f.seek(reader.byte_pos)
				reader.file = f
				reader.it = discord.oggparse.OggStream(f).iter_packets()
				out = next(reader.it, b"")
			reader.pos += 1
			reader.byte_pos += len(out)
			return out

		def close():
			reader.closed = True
			reader.file.close()
			players.pop(reader.key, None)

		reader.read = read
		reader.close = reader.cleanup = close
		return reader

	# Destroys the file object, killing associated FFmpeg process and removing from cache.
	def destroy(self):
		self.expired = True
		if is_strict_running(self.proc):
			with suppress():
				force_kill(self.proc)
		with suppress():
			with self.semaphore:
				if not self.live:
					retry(os.remove, "cache/" + self.file, attempts=8, delay=5, exc=(FileNotFoundError,))
				# File is removed from cache data
				request(f"VOICE.ytdl.cache.pop({repr(self.file)},None)")
				ytdl.cache.pop(self.file, None)
				# print(self.file, "deleted.")

	# Creates a reader, selecting from direct opus file, single piped FFmpeg, or double piped FFmpeg.
	def create_reader(self, pos=0, auds=None, options=None, key=None):
		if self.live:
			source = self.live
		else:
			source = "cache/" + self.file
			if not os.path.exists(source):
				self.readable.result(timeout=12)
				self.load(force=True)
		stats = auds.stats
		auds.reverse = stats.speed < 0
		auds.speed = abs(stats.speed)
		if auds.speed < 0.005:
			auds.speed = 1
		players[auds.guild_id]
		stats.position = pos
		if not is_finite(stats.pitch * stats.speed):
			raise OverflowError("Speed setting out of range.")
		# Construct FFmpeg options
		if options is None:
			options = auds.construct_options(full=self.live)
		speed = 1
		if options or auds.reverse or pos or auds.stats.bitrate != 1966.08 or self.live:
			args = ["./ffmpeg", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets"]
			if (pos or auds.reverse) and self.seekable:
				arg = "-to" if auds.reverse else "-ss"
				args += [arg, str(pos)]
				if reverse and not pos:
					pos = self.duration() or 300
			speed = round_min(stats.speed * 2 ** (stats.resample / 12))
			if auds.reverse:
				speed = -speed
			args.append("-i")
			if self.loaded or self.live:
				buff = False
				args.insert(1, "-nostdin")
				args.append(source)
			else:
				buff = True
				args.append("-")
			auds.stats.bitrate = min(auds.stats.bitrate, auds.stats.max_bitrate)
			if options or auds.stats.bitrate != 1966.08:
				br = 100 * auds.stats.bitrate
				sr = SAMPLE_RATE
				while br < 4096:
					br *= 2
					sr >>= 1
				if sr < 8000:
					sr = 8000
				options.extend(("-f", "opus", "-c:a", "libopus", "-ar", str(sr), "-ac", "2", "-b:a", str(round_min(br)), "-bufsize", "8192"))
				if options:
					args.extend(options)
			else:
				args.extend(("-f", "opus"))
				if not self.live:
					args.extend(("-c:a", "copy"))
			args.append("-")
			g_id = auds.guild_id
			self.readers[g_id] = True
			callback = lambda: self.readers.pop(g_id, None)
			# print(args)
			if buff:
				self.readable.result()
				# Select buffered reader for files not yet fully loaded, convert while downloading
				player = BufferedAudioReader(self, args, callback=callback, key=key)
			else:
				# Select loaded reader for loaded files
				player = LoadedAudioReader(self, args, callback=callback, key=key)
			player.speed = speed
			auds.args = args
			reader = player.start()
		else:
			auds.args.clear()
			# Select raw file stream for direct audio playback
			reader = self.open(key)
		reader.pos = pos * 50
		players[key] = reader
		return reader		

	# Audio duration estimation: Get values from file if possible, otherwise URL
	duration = lambda self: inf if not self.seekable else getattr(self, "dur", None) or set_dict(self.__dict__, "dur", get_duration("cache/" + self.file) if self.loaded and not self.live else get_duration(self.stream), ignore=True)


# Audio reader for fully loaded files. FFmpeg with single pipe for output.
class LoadedAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args, callback=None, key=None):
		self.closed = False
		self.advanced = False
		self.args = args
		self.proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=192000)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.file = file
		self.af = file
		self.buffer = None
		self.callback = callback
		self.pos = 0
		self.key = key
		self.duration = file.duration

	def read(self):
		if self.buffer:
			b, self.buffer = self.buffer, None
			self.pos += self.speed
			return b
		for att in range(16):
			try:
				out = next(self.packet_iter, b"")
			except (OSError, BrokenPipeError):
				if self.file.seekble:
					pos = self.pos / 50
					try:
						i = self.args.index("-ss")
					except ValueError:
						try:
							i = self.args.index("-to")
						except ValueError:
							i = self.args.index("error") + 1
							self.args.insert(i, "-ss")
							self.args.insert(i + 1, str(pos))
						else:
							self.args[i + 1] = str(float(self.args[i + 1]) - pos)
					else:
						self.args[i + 1] = str(float(self.args[i + 1]) + pos)
				self.proc = psutil.Popen(self.args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=192000)
				self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
			else:
				self.pos += self.speed
				return out
		return b""

	@tracebacksuppressor
	def start(self):
		self.buffer = None
		self.buffer = self.read()
		return self

	def close(self, *void1, **void2):
		self.closed = True
		with suppress():
			force_kill(self.proc)
		players.pop(self.key, None)
		if callable(self.callback):
			self.callback()

	is_opus = lambda self: True
	cleanup = close


# Audio player for audio files still being written to. Continuously reads and sends data to FFmpeg process, only terminating when file download is confirmed to be finished.
class BufferedAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args, callback=None, key=None):
		self.closed = False
		self.advanced = False
		self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=192000)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.file = file
		self.af = file
		self.stream = open("cache/" + file.file, "rb")
		self.buffer = None
		self.callback = callback
		self.full = False
		self.pos = 0
		self.key = key
		self.duration = file.duration

	def read(self):
		if self.buffer:
			b, self.buffer = self.buffer, None
			self.pos += self.speed
			return b
		if self.full:
			fut = create_future_ex(next, self.packet_iter, b"")
			try:
				out = fut.result(timeout=0.8)
			except concurent.futures.TimeoutError:
				with suppress():
					force_kill(self.proc)
				out = b""
		else:
			out = next(self.packet_iter, b"")
		self.pos += self.speed
		return out

	# Required loop running in background to feed data to FFmpeg
	def run(self):
		self.file.readable.result(timeout=60)
		while True:
			b = bytes()
			try:
				b = self.stream.read(65536)
				if not b:
					raise EOFError
				self.proc.stdin.write(b)
				self.proc.stdin.flush()
			except (ValueError, EOFError):
				# Only stop when file is confirmed to be finished
				if self.file.loaded or self.closed:
					break
				time.sleep(0.1)
		self.full = True
		self.proc.stdin.close()

	@tracebacksuppressor
	def start(self):
		# Run loading loop in parallel thread obviously
		create_future_ex(self.run, timeout=86400)
		self.buffer = None
		self.buffer = self.read()
		return self

	def close(self):
		self.closed = True
		with suppress():
			self.stream.close()
		with suppress():
			force_kill(self.proc)
		players.pop(self.key, None)
		if callable(self.callback):
			self.callback()

	is_opus = lambda self: True
	cleanup = close


class AudioClient(discord.Client):

	intents = discord.Intents(
		guilds=True,
		members=False,
		bans=False,
		emojis=False,
		webhooks=False,
		voice_states=True,
		presences=False,
		messages=False,
		reactions=False,
		typing=False,
	)

	def __init__(self):
		super().__init__(
			loop=eloop,
			_loop=eloop,
			max_messages=1,
			heartbeat_timeout=60,
			guild_ready_timeout=5,
			status=discord.Status.idle,
			guild_subscriptions=False,
			intents=self.intents,
		)
		create_task(super()._async_setup_hook())
		self._globals = globals()

client = AudioClient()
client.http.user_agent = "Miza-Voice"


async def mobile_identify(self):
	"""Sends the IDENTIFY packet."""
	send("Overriding with mobile status...")
	payload = {
		'op': self.IDENTIFY,
		'd': {
			'token': self.token,
			'properties': {
				'os': 'Miza-OS',
				'browser': 'Discord Android',
				'device': 'Miza',
				'referrer': '',
				'referring_domain': ''
			},
			'compress': True,
			'large_threshold': 250,
			'v': 3
		}
	}

	if self.shard_id is not None and self.shard_count is not None:
		payload['d']['shard'] = [self.shard_id, self.shard_count]

	state = self._connection
	if state._activity is not None or state._status is not None:
		payload['d']['presence'] = {
			'status': state._status,
			'game': state._activity,
			'since': 0,
			'afk': False
		}

	if state._intents is not None:
		payload['d']['intents'] = state._intents.value

	await self.call_hooks('before_identify', self.shard_id, initial=self._initial_identify)
	await self.send_as_json(payload)

discord.gateway.DiscordWebSocket.identify = lambda self: mobile_identify(self)


async def kill():
	futs = deque()
	with suppress(ConnectionResetError):
		futs.append(create_task(client.change_presence(status=discord.Status.invisible)))
	for vc in client.voice_clients:
		futs.append(create_task(vc.disconnect(force=True)))
	for fut in futs:
		await fut
	sys.stdin.close()
	return await client.close()

@client.event
async def on_ready():
	with tracebacksuppressor:
		await communicate()
	await client.close()
	client.closed = True


async def ensure_parent(proc, parent):
	while not getattr(client, "closed", False):
		if not is_strict_running(parent):
			with tracebacksuppressor():
				await asyncio.wait_for(kill(), timeout=2)
			force_kill(proc)
			break
		await asyncio.sleep(12)


if __name__ == "__main__":
	pid = os.getpid()
	ppid = os.getppid()
	send(f"Audio client starting with PID {pid} and parent PID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	create_task(ensure_parent(proc, parent))
	discord.client._loop = eloop
	eloop.run_until_complete(client.start(AUTH["discord_token"]))
