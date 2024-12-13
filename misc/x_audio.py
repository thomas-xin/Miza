import asyncio
import concurrent.futures
import os
import subprocess
import sys
import time
import psutil
from collections import deque
from concurrent.futures import Future
from math import inf, log2, isfinite
from traceback import print_exc
from .asyncs import csubmit, esubmit, tsubmit, async_nop, wrap_future, cst, Semaphore, Delay, eloop
from .types import as_str, cdict, suppress, utc, ISE2, round_min, cast_id
from .util import tracebacksuppressor, is_strict_running, force_kill, retry, AUTH, TEMP_PATH, Request, EvalPipe, PipedProcess, is_url, is_youtube_stream, is_soundcloud_stream, expired, reqs, get_duration, T

# VERY HACKY removes deprecated audioop dependency for discord.py; this would cause volume transformations to fail but Miza uses FFmpeg for them anyway
sys.modules["audioop"] = sys
import discord  # noqa: E402

tracebacksuppressor.fn = print_exc

ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
if ADDRESS == "0.0.0.0":
	ADDRESS = "127.0.0.1"

# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	print = interface.print


class AudioPlayer(discord.AudioSource):

	players = {}
	waiting = {}
	sources = {}
	vc = None
	listener = None
	listening = False
	# Empty opus packet data
	emptyopus = b"\xfc\xff\xfe"
	silent = False

	@classmethod
	async def join(cls, channel):
		channel = client.get_channel(cast_id(channel))
		gid = channel.guild.id
		try:
			return cls.players[gid]
		except KeyError:
			pass
		try:
			fut = cls.waiting[gid]
		except KeyError:
			pass
		else:
			self = await wrap_future(fut)
			if self:
				return self
		self = cls(channel.guild)
		cls.waiting[gid] = Future()
		print(self, channel)
		try:
			if not self.vc:
				if channel.guild.me.voice:
					await channel.guild.change_voice_state(channel=None)
				self.vc = await channel.connect(timeout=7, reconnect=True)
		except Exception as ex:
			try:
				cst(cls.waiting[gid].set_exception, ex)
			except KeyError:
				pass
			raise
		else:
			cls.players[gid] = self
			try:
				cst(cls.waiting[gid].set_result, self)
			except KeyError:
				pass
			return self
		finally:
			cls.waiting.pop(gid, None)

	@classmethod
	def from_guild(cls, guild):
		gid = cast_id(guild)
		try:
			return cls.players[gid]
		except KeyError:
			pass
		try:
			fut = cls.waiting[gid]
		except KeyError:
			pass
		else:
			self = fut.result(timeout=7)
			if self:
				return self
		self = cls(guild)
		if self.vc:
			return self

	def __init__(self, guild=None):
		self.listening = None
		self.queue = deque(maxlen=2)
		if guild:
			self.vc = client.get_guild(cast_id(guild)).voice_client

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
		if T(self.vc).get("dead") or not self.queue or not self.queue[0]:
			return
		entry = self.queue.popleft()
		esubmit(entry[0].close)
		after = entry[1]
		if callable(after):
			after()
		sys.stderr.write(f"After {self} {self.queue} {after}\n")
		if self.queue:
			with tracebacksuppressor(RuntimeError, discord.ClientException):
				self.vc.play(self, after=self.after)

	def read(self):
		if not self.queue or not self.queue[0]:
			if self.silent:
				try:
					self.vc.pause()
				except Exception:
					pass
			self.silent = True
			self.queue.clear()
			return self.emptyopus * 3
		out = b""
		try:
			out = self.queue[0][0].read()
		except (StopIteration, IndexError, discord.oggparse.OggError):
			pass
		except Exception:
			print_exc()
		if not out and self.queue:
			with tracebacksuppressor(StopIteration):
				entry = self.queue.popleft()
				esubmit(entry[0].close)
				after = entry[1]
				if callable(after):
					after()
				sys.stderr.write(f"After2 {self} {self.queue} {after}\n")
				if not self.queue:
					return self.emptyopus
				out = self.queue[0][0].read()
		if not out:
			out = self.emptyopus
		if out == self.emptyopus:
			self.silent = True
		elif self.silent:
			self.silent = False
			try:
				self.pause()
			except Exception:
				pass
			self.resume()
		return out

	def play(self, source, after=None):
		if not self.queue:
			self.queue.append(None)
		elif self.queue[0]:
			esubmit(self.queue[0][0].close)
		self.queue[0] = (source, after)
		with tracebacksuppressor(RuntimeError, discord.ClientException):
			self.vc.play(self, after=self.after)
		if not self.is_playing():
			with suppress():
				self.vc.resume()

	def enqueue(self, source, after=None):
		if not self.queue:
			self.play(source, after=after)
		else:
			if len(self.queue) < 2:
				self.queue.append(None)
			else:
				self.queue[1][0].close()
			self.queue[1] = (source, after)
		if not self.is_playing():
			with suppress():
				self.vc.resume()

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
			esubmit(entry[0].close)
			after = entry[1]
			if callable(after):
				after()

	def clear(self):
		for entry in tuple(self.queue):
			if entry:
				entry[0].close()
		self.queue.clear()

	def kill(self):
		csubmit(self.vc.disconnect(force=True))
		self.clear()
		self.players.pop(self.guild.id, None)
		self.waiting.pop(self.guild.id, None)
		self.vc.dead = True
		self.vc = None

	def listen(self):
		if self.recording:
			self.deafen()
		self.sink = discord.sinks.PCMSink()
		self.start_recording(self.sink, async_nop)
		self.csubmit(self.listener())

	def deafen(self):
		if self.listening and not self.listening.done():
			self.listening.cancel()
		self.stop_recording()

	# Minimum 5 second recordings with 3s of silence at the end
	async def listener(self):
		while self.recording:
			await asyncio.sleep(1)
			b = self.sink.file.getbuffer()
			print(len(b))
			if len(b) > 48000 * 2 * 2:
				pass

	def is_opus(self):
		return True

	def cleanup(self):
		return None

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

	def __init__(self, fn, stream=None, wasfile=False, source=None):
		self.file = fn
		self.proc = None
		self.streaming = Future()
		self.readable = Future()
		if stream is not None:
			self.streaming.set_result(stream)
		self.stream = stream
		self.wasfile = False
		self.wasecdc = False
		self.loading = self.buffered = self.loaded = wasfile
		if wasfile:
			self.proc = cdict(is_running=lambda: False, kill=lambda: None, status=lambda: None)
		self.expired = False
		self.readers = AP.sources
		self.semaphore = Semaphore(1, 1)
		self.ensure_time()
		self.webpage_url = None
		self.source = source
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
			except ISE2:
				pass
		stream = self.stream
		if webpage_url is not None:
			self.webpage_url = webpage_url
		self.loading = True
		# if not asap and not live and is_url(stream):
		# 	fi = f"{TEMP_PATH}/" + str(time.time_ns() + random.randint(1, 1000)) + "~proxy"
		# 	with tracebacksuppressor:
		# 		stream = proxy_download(stream, fi, timeout=86400)
		ffmpeg = "./ffmpeg"
		if not os.path.exists(ffmpeg):
			ffmpeg = "./ffmpeg"
		fmt = cdc = self.file.rsplit(".", 1)[-1]
		sample_rate = SAMPLE_RATE
		if fmt in ("weba", "webm"):
			fmt = "webm"
			cdc = "libopus"
			cdc2 = "opus"
		elif fmt == "ts":
			fmt = "mpegts"
			cdc = "libopus"
			cdc2 = "opus"
		elif fmt == "ogg":
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
		elif fmt == "mp3":
			sample_rate = "44100"
		# Collects data from source, converts to 48khz 128kbps opus format, outputting to target file
		cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-f", fmt, "-c:a", cdc, "-ar", str(sample_rate), "-ac", "2", "-b:a", "192000", f"{TEMP_PATH}/audio/" + self.file]
		fixed = False
		with suppress():
			if stream.startswith("https://www.yt-download.org/download/"):
				fmt2 = "mp3"
				fixed = True
			else:
				fmt2 = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=nokey=1:noprint_wrappers=1", stream])).strip()
			if fmt2 == cdc2:
				cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-c:a", "copy", f"{TEMP_PATH}/audio/" + self.file]
				fixed = True
			elif is_youtube_stream(stream) or is_soundcloud_stream(stream):
				fixed = True
		if is_url(stream):
			cmd = [ffmpeg, "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"] + cmd[1:]
		procargs = [cmd]
		if not fixed and is_url(stream):
			with tracebacksuppressor:
				headers = Request.header()
				headers["Range"] = "bytes=0-3"
				resp = reqs.next().get(stream, headers=headers, stream=True, timeout=30)
				resp.raise_for_status()
				it = resp.iter_content(4)
				data = next(it)[:4]
				if not data:
					raise EOFError(stream)
				CONVERTERS = (
					b"MThd",
					b"Org-",
				)
				if data in CONVERTERS:
					new = None
					with suppress(ValueError):
						new = interface.run(f"VOICE.select_and_convert({repr(stream)})", timeout=120)
					if new not in (None, "null"):
						return self.load(new, check_fmt=None, force=True)
				elif data == b"ECDC":
					procargs = [
						[sys.executable, "misc/ecdc_stream.py", "-d", stream],
						["./ffmpeg", "-nostdin", "-y", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-map_metadata", "-1", "-f", fmt, "-c:a", cdc, "-ar", str(sample_rate), "-ac", "2", "-b:a", "192000", f"{TEMP_PATH}/audio/" + self.file]
					]
					self.wasecdc = True
		self.proc = None
		try:
			try:
				self.proc = PipedProcess(*procargs, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
			except:
				print(cmd)
				raise
			i = 0
			fl = 0
			# Attempt to monitor status of output file; file is sufficiently loaded if either process ends with >=4kb, or is still running with >=512kb
			while fl < 4096 or (fl < 524288 and is_strict_running(self.proc)):
				with Delay(i / 20):
					if not is_strict_running(self.proc):
						err = as_str(self.proc.stderr.read())
						if check_fmt is not None:
							if self.webpage_url and ("Server returned 5XX Server Error reply" in err or "Server returned 404 Not Found" in err or "Server returned 403 Forbidden" in err):
								print(err)
								with tracebacksuppressor:
									if "https://cf-hls-media.sndcdn.com/" in stream or expired(stream):
										new_stream = interface.run(f"VOICE.get_best_audio(VOICE.ytdl.extract_from({repr(self.webpage_url)}))")
									else:
										new_stream = interface.run(f"VOICE.get_best_audio(VOICE.ytdl.extract_backup({repr(self.webpage_url)}))")
									if new_stream:
										return self.load(new_stream, check_fmt=None, force=True)
							new = None
							with suppress(ValueError):
								new = interface.run(f"VOICE.select_and_convert({repr(stream)})", timeout=120)
							if new not in (None, "null"):
								return self.load(new, check_fmt=None, force=True)
						print(self.proc.args)
						if err:
							ex = RuntimeError(err)
						else:
							ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
						self.readable.set_exception(ex)
						raise ex
				i += 1
				try:
					fl = os.path.getsize(f"{TEMP_PATH}/audio/" + self.file)
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
				os.remove(f"{TEMP_PATH}/audio/" + self.file)
			self.readable.set_exception(ex)
			raise
		self.readable.set_result(self)
		self.live = False
		if not self.live and is_strict_running(self.proc):
			tsubmit(self.wait)
		return self

	# Touch the file to update its cache time.
	def ensure_time(self):
		return setattr(self, "time", utc())

	def wait(self):
		self.proc.wait()
		self.loaded = True
		try:
			assert os.path.getsize(f"{TEMP_PATH}/audio/" + self.file) > 0
		except (FileNotFoundError, AssertionError):
			print_exc()
		else:
			if not self.wasecdc and self.source:
				interface.run(f"VOICE.ytdl.complete({repr(self.source)},{repr(self.file)})", timeout=600)

	# Update event run on all cached files
	def update(self):
		with tracebacksuppressor:
			# Touch file if file is currently in use
			if self.readers:
				self.ensure_time()
				return
			# Remove any unused file that has been left for a long time
			if utc() - self.time > 86400:
				try:
					fl = os.path.getsize(f"{TEMP_PATH}/audio/" + self.file)
				except FileNotFoundError:
					fl = 0
					if self.buffered:
						self.time = -inf
				ft = 86400 * 7 / (log2(fl / 16777216 + 1) + 1)
				if ft > 86400 * 28:
					ft = 86400 * 28
				if utc() - self.time > ft:
					self.destroy()

	# Creates a reader object that either reads bytes or opus packets from the file.
	def open(self, key=None):
		self.ensure_time()
		if self.proc is None and not self.loaded:
			raise ProcessLookupError
		f = open(f"{TEMP_PATH}/audio/" + self.file, "rb")
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
				f = open(f"{TEMP_PATH}/audio/" + self.file, "rb")
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
			AP.sources.pop(reader.key, None)

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
			if self.semaphore.is_free():
				with self.semaphore:
					if not self.live:
						retry(os.remove, f"{TEMP_PATH}/audio/" + self.file, attempts=8, delay=5, exc=(FileNotFoundError,))
					# File is removed from cache data
					interface.submit(f"VOICE.ytdl.cache.pop({repr(self.file)},None)")
					ytdl.cache.pop(self.file, None)
					# print(self.file, "deleted.")

	# Creates a reader, selecting from direct opus file, single piped FFmpeg, or double piped FFmpeg.
	def create_reader(self, pos=0, auds=None, options=None, key=None):
		if self.live:
			source = self.live
		else:
			source = f"{TEMP_PATH}/audio/" + self.file
			if not os.path.exists(source):
				self.readable.result(timeout=12)
				self.load(force=True)
		stats = auds.stats
		auds.reverse = stats.speed < 0
		auds.speed = abs(stats.speed)
		if auds.speed < 0.005:
			auds.speed = 1
		stats.position = pos
		if not isfinite(stats.pitch * stats.speed):
			raise OverflowError("Speed setting out of range.")
		# Construct FFmpeg options
		if options is None:
			options = auds.construct_options(full=self.live)
		speed = 1
		if options or auds.reverse or pos or auds.stats.bitrate * 100 != 192000 or self.live:
			args = ["./ffmpeg", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+nobuffer+discardcorrupt+genpts+igndts+flush_packets"]
			if is_url(source):
				args = ["./ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"] + args[1:]
			if (pos or auds.reverse) and self.seekable:
				arg = "-to" if auds.reverse else "-ss"
				if auds.reverse and not pos:
					pos = self.duration() or 300
				args += [arg, str(pos)]
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
			if options or auds.stats.bitrate * 100 != 192000:
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
			if buff:
				self.readable.result()
				# Select buffered reader for files not yet fully loaded, convert while downloading
				player = BufferedAudioReader(self, args, key=key)
			else:
				# Select loaded reader for loaded files
				player = LoadedAudioReader(self, args, key=key)
			player.speed = speed
			auds.args = args
			reader = player.start()
		else:
			auds.args.clear()
			# Select raw file stream for direct audio playback
			reader = self.open(key)
		reader.pos = pos * 50
		self.readers[key] = reader
		return reader

	# Audio duration estimation: Get values from file if possible, otherwise URL
	def duration(self):
		if not self.seekable:
			return inf
		if T(self).get("dur"):
			return self.dur
		if not self.live:
			dur = get_duration(f"{TEMP_PATH}/audio/" + self.file)
			if not self.loaded:
				return dur
		else:
			dur = get_duration(self.stream)
		self.dur = dur
		return dur

	@property
	def proc_expired(self):
		return not self.proc or not self.proc.is_running()


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
		AP.sources.pop(self.key, None)
		if callable(self.callback):
			self.callback()
	cleanup = close

	def is_opus(self):
		return True


# Audio player for audio files still being written to. Continuously reads and sends data to FFmpeg process, only terminating when file download is confirmed to be finished.
class BufferedAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args, callback=None, key=None):
		self.closed = False
		self.advanced = False
		self.args = args
		self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=192000)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.file = file
		self.af = file
		self.stream = open(f"{TEMP_PATH}/audio/" + file.file, "rb")
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
		try:
			if self.full:
				fut = esubmit(next, self.packet_iter, b"")
				try:
					out = fut.result(timeout=1)
				except concurrent.futures.TimeoutError:
					with suppress():
						force_kill(self.proc)
					out = b""
			else:
				out = next(self.packet_iter, b"")
			self.pos += self.speed
			return out
		except Exception:
			print_exc()
			return b""

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
					try:
						b = self.stream.read(65536)
						if not b:
							raise EOFError
					except ValueError:
						break
					self.proc.stdin.write(b)
					self.proc.stdin.flush()
				time.sleep(0.1)
		self.full = True
		self.proc.stdin.close()

	@tracebacksuppressor
	def start(self):
		# Run loading loop in parallel thread obviously
		esubmit(self.run, timeout=86400)
		self.buffer = None
		self.buffer = self.read()
		return self

	def close(self):
		self.closed = True
		with suppress():
			self.stream.close()
		with suppress():
			force_kill(self.proc)
		AP.sources.pop(self.key, None)
		if callable(self.callback):
			self.callback()
	cleanup = close

	def is_opus(self):
		return True


class AudioClient(discord.AutoShardedClient):

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
		with suppress(AttributeError):
			csubmit(super()._async_setup_hook())
		self._globals = globals()

client = AudioClient()
client.http.user_agent = "Miza-Voice"


async def mobile_identify(self):
	"""Sends the IDENTIFY packet."""
	print("Overriding with mobile status...")
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
		futs.append(csubmit(client.change_presence(status=discord.Status.offline)))
	for vc in client.voice_clients:
		futs.append(csubmit(vc.disconnect(force=True)))
	for fut in futs:
		await fut
	sys.stdin.close()
	return await client.close()

@client.event
async def on_ready():
	with tracebacksuppressor:
		interface.start()
		print("Audio client successfully connected.")

def ensure_parent(proc, parent):
	while not getattr(client, "closed", False):
		if not is_strict_running(parent):
			with tracebacksuppressor():
				csubmit(kill)
			force_kill(proc)
			break
		time.sleep(12)


if __name__ == "__main__":
	pid = os.getpid()
	ppid = os.getppid()
	print(f"Audio client starting with PID {pid} and PPID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	tsubmit(ensure_parent, proc, parent)
	discord.client._loop = eloop
	eloop.run_until_complete(client.start(AUTH["discord_token"]))
