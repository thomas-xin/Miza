# Make linter shut up lol
if "common" not in globals():
	import common
	from common import *
print = PRINT

try:
	import yt_dlp as youtube_dl
except ModuleNotFoundError:
	try:
		youtube_dl = __import__("youtube_dl")
	except ModuleNotFoundError:
		youtube_dl = None
try:
	from misc import yt_download as ytd
	from misc.yt_download import *
except:
	print_exc()
	has_ytd = False
else:
	has_ytd = True
from bs4 import BeautifulSoup
with tracebacksuppressor:
	import openai
	import googletrans
if BOT[0]:
	bot = BOT[0]

# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000


# Gets estimated duration from duration stored in queue entry
e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else 300)


# Gets the best icon/thumbnail for a queue entry.
def get_best_icon(entry):
	with suppress(KeyError):
		return entry["thumbnail"]
	with suppress(KeyError):
		return entry["icon"]
	try:
		thumbnails = entry["thumbnails"]
	except KeyError:
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		if not url:
			return ""
		if is_discord_url(url):
			if not is_image(url):
				return "https://cdn.discordapp.com/embed/avatars/0.png"
		if is_youtube_url(url):
			if "?v=" in url:
				vid = url.split("?v=", 1)[-1]
			else:
				vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
			entry["thumbnail"] = f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
			return entry["thumbnail"]
		if ytdl.bot.is_webserver_url(url):
			return ytdl.bot.raw_webserver + "/static/mizaleaf.png"
		return url
	return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]


# Gets the best audio file download link for a queue entry.
def get_best_audio(entry):
	with suppress(KeyError):
		return entry["stream"]
	best = -inf
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	try:
		url = entry["url"]
	except KeyError:
		url = entry["webpage_url"]
	replace = True
	for fmt in fmts:
		q = fmt.get("abr", 0)
		if not isinstance(q, (int, float)):
			q = 0
		if q <= 0:
			if fmt.get("asr"):
				q = fmt["asr"] / 1000
			elif fmt.get("audio_channels"):
				q = fmt["audio_channels"]
		vcodec = fmt.get("vcodec", "none")
		if vcodec not in (None, "none"):
			q -= 1
		u = as_str(fmt["url"])
		if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = Request(url)
		fmts = alist()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = as_str(resp[:resp.index(b'"')])
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = as_str(resp[:resp.index(b'</BaseURL>')])
				fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_audio(entry)
	if not url:
		raise KeyError("URL not found.")
	return url


# Gets the best video file download link for a queue entry.
def get_best_video(entry):
	with suppress(KeyError):
		return entry["stream"]
	best = -inf
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	try:
		url = entry["url"]
	except KeyError:
		url = entry["webpage_url"]
	replace = True
	for fmt in fmts:
		q = fmt.get("height", 0)
		if not isinstance(q, (int, float)):
			q = 0
		vcodec = fmt.get("vcodec", "none")
		if vcodec in (None, "none"):
			q -= 1
		u = as_str(fmt["url"])
		if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = Request(url)
		fmts = alist()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = as_str(resp[:resp.index(b'"')])
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = as_str(resp[:resp.index(b'</BaseURL>')])
				fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_video(entry)
	if not url:
		raise KeyError("URL not found.")
	return url


# Joins a voice channel and returns the associated audio player.
async def auto_join(guild, channel, user, bot, preparing=False, vc=None, ignore=False):
	user = guild._members.get(user.id) or user
	if not ignore and not getattr(user, "voice", None):
		g = vc.guild if vc else guild
		perm = bot.get_perms(user, g)
		if perm < 1:
			raise Command.perm_error(Command, perm, 1, f"to remotely operate audio player for {g} without joining voice")
	if type(channel) in (str, int):
		channel = await bot.fetch_channel(channel)
	if guild.id not in bot.data.audio.players:
		if not channel:
			raise LookupError("Unable to find voice channel.")
		for func in bot.commands.connect:
			try:
				await func(user=user, channel=channel, vc=vc)
			except (discord.ClientException, AttributeError):
				pass
	try:
		auds = bot.data.audio.players[guild.id]
	except KeyError:
		raise LookupError("Unable to find voice channel.")
	auds.text = channel or auds.text
	return auds


# Helper function to save all items in a queue
copy_entry = lambda item: {"name": item["name"], "url": item["url"], "duration": item.get("duration")}


async def disconnect_members(bot, guild, members, channel=None):
	if bot.id in (member.id for member in members):
		with suppress(KeyError):
			auds = bot.data.audio.players[guild.id]
			await asubmit(auds.kill)
	futs = (member.move_to(None) for member in members)
	await asyncio.gather(*futs)


# Checks if the user is alone in voice chat (excluding bots).
def is_alone(auds, user):
	try:
		channel = auds.acsi.channel
	except AttributeError:
		return True
	for m in channel.members:
		if m.id != user.id and not m.bot and m.voice and m.voice.channel.id == channel.id:
			return False
	return True


# Replaces youtube search queries in youtube-dl with actual youtube search links.
def ensure_url(url):
	if url.startswith("ytsearch:"):
		url = f"https://www.youtube.com/results?search_query={verify_url(url[9:])}"
	return url


# This messy regex helps identify and remove certain words in song titles
lyric_trans = re.compile(
	(
		"[([]+"
		"(((official|full|demo|original|extended) *)?"
		"((version|ver.?) *)?"
		"((w\\/)?"
		"(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq|english) *)?"
		"((album|video|audio|cover|remix) *)?"
		"(upload|reupload|version|ver.?)?"
		"|(feat|ft)"
		".+)"
		"[)\\]]+"
	),
	flags=re.I,
)


# Audio player that wraps discord audio sources, contains a queue, and also manages audio settings.
class CustomAudio(collections.abc.Hashable):

	# Default player settings
	max_bitrate = 196608
	defaults = {
		"volume": 1,
		"reverb": 0,
		"pitch": 0,
		"speed": 1,
		"pan": 1,
		"bassboost": 0,
		"compressor": 0,
		"chorus": 0,
		"resample": 0,
		"bitrate": 131072 / 100,
		"loop": False,
		"repeat": False,
		"shuffle": False,
		"quiet": False,
		"stay": False,
	}
	paused = False
	source = None
	next = None
	timeout = 0
	seek_pos = 0
	last_play = 0
	ts = None
	player = None

	@tracebacksuppressor
	def __init__(self, text=None):
		# Class instance variables
		self.bot = bot
		self.stats = cdict(self.defaults)
		self.text = text
		self.fut = Future()
		self.acsi = None
		self.args = []
		self.queue = AudioQueue()
		self.queue._init_()
		self.queue.auds = self
		self.semaphore = Semaphore(1, 4, rate_limit=1 / 8)
		self.announcer = Semaphore(1, 1, rate_limit=1 / 3)
		self.search_sem = Semaphore(1, 1, rate_limit=1)

	def join(self, channel=None):
		if channel:
			if getattr(channel, "guild", None):
				self.guild = guild = channel.guild
				if not bot.is_trusted(guild):
					self.queue.maxitems = 8192
				bot.data.audio.players[guild.id] = self
				self.stats.update(bot.data.audiosettings.get(guild.id, {}))
			self.timeout = utc()
			return create_task(self.connect_to(channel))

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: " + "{" + f'"acsi": {self.acsi}, "queue": {len(self.queue)}, "stats": {self.stats}, "source": {self.source}' + "}"

	__hash__ = lambda self: self.guild.id

	def __getattr__(self, key):
		if key in ("reverse", "speed", "epos", "pos"):
			return self.__getattribute__("_" + key)()
		try:
			return self.__getattribute__(key)
		except AttributeError:
			pass
		try:
			return getattr(self.__getattribute__("queue"), key)
		except AttributeError:
			pass
		if not self.fut.done():
			return
		return getattr(self.__getattribute__("acsi"), key)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(("reverse", "speed", "epos", "pos"))
		data.update(dir(self.acsi))
		data.update(dir(self.queue))
		return data

	def has_options(self):
		stats = self.stats
		return stats.volume != 1 or stats.reverb != 0 or stats.pitch != 0 or stats.speed != 1 or stats.pan != 1 or stats.bassboost != 0 or stats.compressor != 0 or stats.chorus != 0 or stats.resample != 0

	def get_dump(self, position=False, paused=False, js=False):
		with self.semaphore:
			lim = 1024
			q = [copy_entry(item) for item in self.queue.verify()]
			s = {k: (v if not isinstance(v, mpf) else str(v) if len(str(v)) > 16 else float(v)) for k, v in self.stats.items()}
			d = {
				"stats": s,
				"queue": q,
			}
			if position:
				d["pos"] = self.pos
			if paused:
				d["paused"] = self.paused
			if js:
				d = orjson.dumps(d)
				if len(d) > 2097152:
					d = bytes2zip(d)
					return d, "dump.zip"
				return d, "dump.json"
			return d, None

	def _reverse(self):
		return self.stats.speed < 0

	def _speed(self):
		return abs(self.stats.speed)

	def _epos(self):
		if not self.fut.done():
			return (0, 0)
		pos = self.acsi.pos
		if not pos[1] and self.queue:
			dur = e_dur(self.queue[0].get("duration"))
			return min(dur, pos[0]), dur
		elif pos[1] is None:
			return (0, 0)
		return pos

	def _pos(self):
		return self.epos[0]

	def skip(self):
		self.acsi.skip()

	def clear_source(self):
		if self.source:
			esubmit(self.acsi.clear_source)
		self.source = None

	def clear_next(self):
		if self.next:
			esubmit(self.acsi.clear_next)
		self.next = None

	def reset(self, start=True):
		try:
			self.acsi.clear()
		except (AttributeError, TypeError):
			pass
		self.source = self.next = None
		if start:
			self.queue.update_load()

	def pause(self, unpause=False):
		if unpause and self.paused:
			self.paused = False
			self.acsi.resume()
			self.queue.update_load()
		elif not self.paused:
			self.paused = True
			if self.acsi:
				self.acsi.pause()

	def resume(self):
		if self.paused:
			self.paused = False
			self.acsi.resume()
		else:
			self.acsi.pause()
			self.acsi.resume()
			self.acsi.read()
		self.queue.update_load()

	# Stops currently playing source, closing it if possible.
	def stop(self):
		self.acsi.stop()
		self.paused = True
		return self.reset(start=False)

	# Loads and plays a new audio source, with current settings and optional song init position.
	def play(self, source=None, pos=0, update=True):
		self.seek_pos = 0
		if source is not None:
			self.source = source
			src = None
			try:
				# This call may take a while depending on the time taken by FFmpeg to start outputting
				source.readable.result(timeout=12)
				src = source.create_reader(pos, auds=self)
			except OverflowError:
				self.clear_source()
			else:
				if src:
					# Only stop and replace audio source when the next one is buffered successfully and readable
					self.acsi.play(src, after=self.queue.advance)
					self.last_play = utc()
		else:
			self.stop()

	@tracebacksuppressor
	def enqueue(self, source):
		self.next = source
		source.readable.result(timeout=12)
		src = source.create_reader(0, auds=self)
		self.acsi.enqueue(src, after=self.queue.advance)

	# Seeks current song position.
	def seek(self, pos):
		duration = self.epos[1]
		pos = max(0, pos)
		# Skip current song if position is out of range
		if (pos >= duration and not self.reverse) or (pos <= 0 and self.reverse):
			self.skip()
			return duration
		self.play(self.source, pos, update=False)
		return pos

	# Sends a deletable message to the audio player's text channel.
	def announce(self, *args, aio=False, dump=False, **kwargs):
		if self.queue and dump and (len(self.queue) > 1 or self.queue[0].get("skips") != ()):
			resp, fn = self.get_dump(js=True)
			f = CompatFile(resp, filename=fn)
		else:
			f = None
		if aio or is_main_thread():
			return create_task(send_with_react(self.text, *args, file=f, reacts="❎", **kwargs))
		with self.announcer:
			return await_fut(send_with_react(self.text, *args, file=f, reacts="❎", **kwargs))

	# Kills this audio player, stopping audio playback. Will cause bot to leave voice upon next update event.
	def kill(self, reason=None, initiator=None, wait=False, remove=True):
		if self.acsi:
			with tracebacksuppressor(AttributeError):
				self.acsi.kill()
		fut = create_task(self.bot.audio.asubmit(f"players.pop({self.guild.id},None)"))
		if wait:
			await_fut(fut)
			while self.guild.me.voice:
				await_fut(self.guild.change_voice_state(channel=None))
		elif self.guild.me and self.guild.me.voice:
			create_task(self.guild.change_voice_state(channel=None))
		popped = self.bot.data.audio.players.pop(self.guild.id, None)
		self.vc = None
		print("POPPED:", popped)
		with suppress(LookupError):
			if reason is None:
				reason = css_md(f"🎵 Successfully disconnected from {sqr_md(self.guild)}. 🎵")
			if reason:
				self.announce(reason, dump=True, reference=initiator)
		if remove:
			self.bot.data.audio.pop(self.channel.id, None)

	# Update event, ensures audio is playing correctly and moves, leaves, or rejoins voice when necessary.
	@tracebacksuppressor
	def update(self, *void1, **void2):
		guild = self.guild
		if self.fut.done() and (not guild.me or not guild.me.voice):
			return self.kill(css_md(f"🎵 Disconnected from {sqr_md(guild)}. 🎵"))
		try:
			self.fut.result(timeout=16)
		except:
			print_exc()
			return self.kill()
		t = utc()
		if getattr(self, "player", None) is not None and self.stats.speed and not self.paused:
			if t - self.player.get("time", 0) >= 0:
				self.player.time = t + 20
				create_task(bot.commands.player[0]._callback_(self.player.get("message"), guild, self.text, 0, self.bot, inf))
		if self.stats.stay:
			cnt = inf
		else:
			cnt = sum(1 for m in self.acsi.channel.members if not m.bot)
		if not self.queue and self.timeout < utc() - 3600:
			return self.kill(css_md(f"🎵 Automatically disconnected from {sqr_md(guild)}: Queue empty. 🎵"))
		if not cnt:
			# Timeout for leaving is 240 seconds
			if self.timeout < utc() - 240:
				return self.kill(css_md(f"🎵 Automatically disconnected from {sqr_md(guild)}: All channels empty. 🎵"))
			perms = self.acsi.channel.permissions_for(guild.me)
			if not perms.connect or not perms.speak:
				return self.kill(css_md(f"🎵 Automatically disconnected from {sqr_md(guild)}: No permission to connect/speak in {sqr_md(self.acsi.channel)}. 🎵"))
			# If idle for more than 20 seconds, attempt to find members in other voice channels
			elif self.timeout < utc() - 20:
				if guild.afk_channel and (guild.afk_channel.id != self.acsi.channel.id and guild.afk_channel.permissions_for(guild.me).connect):
					await_fut(self.move_unmute(self.acsi, guild.afk_channel))
				else:
					cnt = 0
					ch = None
					for channel in voice_channels(guild):
						if not guild.afk_channel or channel.id != guild.afk_channel.id:
							c = sum(1 for m in channel.members if not m.bot)
							if c > cnt:
								cnt = c
								ch = channel
					if ch:
						with tracebacksuppressor(SemaphoreOverflowError):
							await_fut(self.move_unmute(self.acsi, ch))
							self.announce(ini_md(f"🎵 Detected {sqr_md(cnt)} user{'s' if cnt != 1 else ''} in {sqr_md(ch)}, automatically joined! 🎵"), aio=False)
		elif self.queue or cnt >= 2:
			self.timeout = utc()
		self.queue.update_load()

	# Moves to the target channel, unmuting self afterwards.
	async def move_unmute(self, vc, vc_):
		await vc.move_to(vc_)
		m = self.guild.me
		perm = m.permissions_in(vc_)
		if m.voice and perm.mute_members:
			if vc_.type is discord.ChannelType.stage_voice:
				if m.voice.suppress or m.voice.requested_to_speak_at:
					await self.speak()
			elif m.voice.deaf or m.voice.mute or m.voice.afk:
				await m.edit(mute=False)

	async def connect_to(self, channel=None):
		# print(self)
		if not self.acsi:
			try:
				acsi = AudioClientSubInterface(self, channel)
				# print(acsi)
				await acsi.start()
				# print(acsi)
				self.acsi = acsi
				self.fut.set_result(self.acsi)
			except Exception as ex:
				print_exc()
				self.fut.set_exception(ex)
		self.queue._init_(auds=self)
		self.timeout = utc()
		if channel:
			self.max_bitrate = channel.bitrate
		return self.acsi

	def speak(self):
		vc = self.channel
		if not vc:
			return
		return Request(
			f"https://discord.com/api/{api}/guilds/{vc.guild.id}/voice-states/@me",
			method="PATCH",
			authorise=True,
			data={"suppress": False, "request_to_speak_timestamp": None, "channel_id": vc.id},
			aio=True,
		)

	# Constructs array of FFmpeg options using the audio settings.
	def construct_options(self, full=True):
		stats = self.stats
		# Pitch setting is in semitones, so frequency is on an exponential scale
		pitchscale = 2 ** ((stats.pitch + stats.resample) / 12)
		reverb = stats.reverb
		volume = stats.volume
		# FIR sample for reverb
		if reverb:
			args = ["-i", "misc/SNB3,0all.wav"]
		else:
			args = []
		options = deque()
		# This must be first, else the filter will not initialize properly
		if not is_finite(stats.compressor):
			options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
		# Reverses song, this may be very resource consuming
		if self.stats.speed < 0:
			options.append("areverse")
		# Adjusts song tempo relative to speed, pitch, and nightcore settings
		if pitchscale != 1 or stats.speed != 1:
			speed = abs(stats.speed) / pitchscale
			speed *= 2 ** (stats.resample / 12)
			if round(speed, 9) != 1:
				speed = max(0.005, speed)
				if speed >= 64:
					raise OverflowError
				opts = ""
				while speed > 3:
					opts += "atempo=3,"
					speed /= 3
				while speed < 0.5:
					opts += "atempo=0.5,"
					speed /= 0.5
				opts += "atempo=" + str(speed)
				options.append(opts)
		# Adjusts resample to match song pitch
		if pitchscale != 1:
			if abs(pitchscale) >= 64:
				raise OverflowError
			if full:
				options.append("aresample=" + str(SAMPLE_RATE))
			options.append("asetrate=" + str(SAMPLE_RATE * pitchscale))
		# Chorus setting, this is a bit of a mess
		if stats.chorus:
			chorus = abs(stats.chorus)
			ch = min(16, chorus)
			A = B = C = D = ""
			for i in range(ceil(ch)):
				neg = ((i & 1) << 1) - 1
				i = 1 + i >> 1
				i *= stats.chorus / ceil(chorus)
				if i:
					A += "|"
					B += "|"
					C += "|"
					D += "|"
				delay = (8 + 5 * i * tau * neg) % 39 + 19
				A += str(round(delay, 3))
				decay = (0.36 + i * 0.47 * neg) % 0.65 + 1.7
				B += str(round(decay, 3))
				speed = (0.27 + i * 0.573 * neg) % 0.3 + 0.02
				C += str(round(speed, 3))
				depth = (0.55 + i * 0.25 * neg) % max(1, stats.chorus) + 0.15
				D += str(round(depth, 3))
			b = 0.5 / sqrt(ceil(ch + 1))
			options.append(
				"chorus=0.5:" + str(round(b, 3)) + ":"
				+ A + ":"
				+ B + ":"
				+ C + ":"
				+ D
			)
		# Compressor setting, this needs a bit of tweaking perhaps
		if stats.compressor:
			comp = min(8000, abs(stats.compressor * 10 + sgn(stats.compressor)))
			while abs(comp) > 1:
				c = min(20, comp)
				try:
					comp /= c
				except ZeroDivisionError:
					comp = 1
				mult = str(round((c * math.sqrt(2)) ** 0.5, 4))
				options.append(
					"acompressor=mode=" + ("upward" if stats.compressor < 0 else "downward")
					+ ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
				)
		# Bassboost setting, the ratio is currently very unintuitive and definitely needs tweaking
		if stats.bassboost:
			opt = "firequalizer=gain_entry="
			entries = []
			high = 24000
			low = 13.75
			bars = 4
			small = 0
			for i in range(bars):
				freq = low * (high / low) ** (i / bars)
				bb = -(i / (bars - 1) - 0.5) * stats.bassboost * 64
				dB = log(abs(bb) + 1, 2)
				if bb < 0:
					dB = -dB
				if dB < small:
					small = dB
				entries.append(f"entry({round(freq, 5)},{round(dB, 5)})")
			entries.insert(0, f"entry(0,{round(small, 5)})")
			entries.append(f"entry(24000,{round(small, 5)})")
			opt += repr(";".join(entries))
			options.append(opt)
		# Reverb setting, using afir and aecho FFmpeg filters.
		if reverb:
			coeff = abs(reverb)
			wet = min(3, coeff) / 3
			# Split audio into 2 inputs if wet setting is between 0 and 1, one input passes through FIR filter
			if wet != 1:
				options.append("asplit[2]")
			volume *= 1.2
			if reverb < 0:
				volume = -volume
			options.append("afir=dry=10:wet=10")
			# Must include amix if asplit is used
			if wet != 1:
				dry = 1 - wet
				options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(-wet, 6)))
			d = [round(1 - i ** 1.3 / (i ** 1.3 + coeff), 4) for i in range(2, 18, 2)]
			options.append(f"aecho=1:1:400|630:{d[0]}|{d[1]}")
			if d[2] >= 0.05:
				options.append(f"aecho=1:1:870|1150:{d[2]}|{d[3]}")
				if d[4] >= 0.06:
					options.append(f"aecho=1:1:1410|1760:{d[4]}|{d[5]}")
					if d[6] >= 0.07:
						options.append(f"aecho=1:1:2080|2320:{d[6]}|{d[7]}")
		# Pan setting, uses extrastereo and volume filters to balance
		if stats.pan != 1:
			pan = min(10000, max(-10000, stats.pan))
			while abs(abs(pan) - 1) > 0.001:
				p = max(-10, min(10, pan))
				try:
					pan /= p
				except ZeroDivisionError:
					pan = 1
				options.append("extrastereo=m=" + str(p) + ":c=0")
				volume *= 1 / max(1, round(math.sqrt(abs(p)), 4))
		if volume != 1:
			options.append("volume=" + str(round(volume, 7)))
		# Soft clip audio using atan, reverb filter requires -filter_complex rather than -af option
		if options:
			if stats.compressor:
				options.append("alimiter")
			elif volume > 1:
				options.append("asoftclip=atan")
			args.append(("-af", "-filter_complex")[bool(reverb)])
			args.append(",".join(options))
		# print(args)
		return args


# Manages the audio queue. Has optimized insertion/removal on both ends, and linear time lookup. One instance of this class is created per audio player.
class AudioQueue(alist):

	maxitems = 262144

	def _init_(self, auds=None):
		self.lastsent = 0
		self.loading = False
		self.playlist = None
		self.sem = Semaphore(1, 0)
		self.sem2 = Semaphore(1, 0)
		self.wait = Future()
		if auds:
			self.auds = auds
			self.bot = auds.bot
			self.acsi = auds.acsi
			self.wait.set_result(auds)

	def announce_play(self, e=None):
		auds = self.auds
		if not auds.stats.quiet:
			if not e:
				if not auds.queue:
					return
				e = auds.queue[0]
			if utc() - self.lastsent > 1 and not e.get("noannounce"):
				try:
					u = self.bot.cache.users[e.u_id]
					name = u.display_name
				except KeyError:
					name = "Deleted User"
				self.lastsent = utc()
				auds.announce(italics(ini_md(f"🎵 Now playing {sqr_md(e.name)}, added by {sqr_md(name)}! 🎵")), aio=True)

	@tracebacksuppressor
	def start_queue(self):
		auds = self.auds
		try:
			auds.epos
		except (TypeError, AttributeError):
			auds.kill()
			raise
		if self.sem.is_busy():
			return
		if not auds.source and self:
			e = self[0]
			source = None
			with self.sem:
				with tracebacksuppressor:
					source = ytdl.get_stream(e, force=True, asap=True)
			if self.sem.is_busy():
				self.sem.wait()
			if not source:
				e["url"] = ""
				return self.update_load()
			with self.sem:
				self.announce_play(e)
				self.auds.play(source, pos=auds.seek_pos)
		if not auds.next and auds.source and len(self) > 1 and not self.sem2.is_busy():
			with self.sem:
				with self.sem2:
					e = self[1]
					source = ytdl.get_stream(e, asap=True)
					if source and not auds.next and auds.source:
						auds.enqueue(source)
		# if len(self) > 2 and not self.sem2.is_busy() and not auds.stats.get("shuffle"):
		#     e = self[2]
		#     sufficient = auds.epos[1] - auds.epos[0] + (self[1].get("duration") or 0) >= (self[2].get("duration") or inf) / 2
		#     if sufficient:
		#         esubmit(self.preemptive_download, e)

	def preemptive_download(self, e):
		with self.sem2:
			ytdl.get_stream(e, asap=False)

	# Update queue, loading all file streams that would be played soon
	def update_load(self):
		self.wait.result()
		q = self
		if q:
			dels = deque()
			for i, e in enumerate(q):
				if i >= len(q) or i > 64:
					break
				if "file" in e:
					e.file.ensure_time()
				if not e.get("url"):
					if not self.auds.stats.quiet:
						msg = f"A problem occured while loading {sqr_md(e.name)}, and it has been automatically removed from the queue."
						if e.get("ex"):
							msg = msg[:-1] + ":\n" + e["ex"]
						self.auds.announce(ini_md(msg))
					dels.append(i)
					continue
			q.pops(dels)
		if not q:
			if self.auds.next:
				self.auds.clear_next()
			if self.auds.source:
				self.auds.stop()
		elif not self.auds.paused:
			self.start_queue()

	# Advances queue when applicable, taking into account loop/repeat/shuffle settings.
	def advance(self, looped=True, repeated=True, shuffled=True):
		# print("Advance:", self.auds)
		self.auds.source = self.auds.next
		self.auds.next = None
		q = self
		s = self.auds.stats
		if q:
			if not (s.repeat and repeated):
				if s.loop:
					temp = q[0]
				q.popleft()
				if s.shuffle and shuffled:
					if len(q) > 1:
						temp = q.popleft()
						shuffle(q)
						q.appendleft(temp)
				if s.loop and looped:
					q.append(temp)
		# If no queue entries found but there is a default playlist assigned, load a random entry from that
		if not q:
			if not self.playlist:
				t = self.bot.data.playlists.get(self.auds.guild.id, ())
				if t:
					self.playlist = shuffle(t)
			if self.playlist:
				p = self.playlist.pop()
				e = cdict(p)
				e.u_id = self.bot.id
				e.skips = ()
				ytdl.get_stream(e)
				q.appendleft(e)
		else:
			self.announce_play(self[0])
		self.update_load()

	def verify(self):
		try:
			assert len(self) >= 0
		except (ValueError, AssertionError):
			self.clear()
		if len(self) > self.maxitems + 2048:
			self.__init__(self[1 - self.maxitems:].appendleft(self[0]), fromarray=True)
		elif len(self) > self.maxitems:
			self.rotate(-1)
			while len(self) > self.maxitems:
				self.pop()
			self.rotate(1)
		return self

	# Enqueue items at target position, starting audio playback if queue was previously empty.
	def enqueue(self, items, position=-1):
		with self.auds.semaphore:
			if len(items) > self.maxitems:
				items = astype(items, (list, alist))[:self.maxitems]
			if not self:
				self.auds.reset(start=False)
			if position == -1 or not self:
				self.extend(items)
			else:
				if position < 1:
					self.auds.reset(start=False)
				elif position < 2:
					self.auds.clear_next()
				self.rotate(-position)
				self.extend(items)
				self.rotate(len(items) + position)
			self.verify()
			esubmit(self.update_load, timeout=120)
			return self


# runs org2xm on a file, with an optional custom sample bank.
def org2xm(org, dat=None):
	# if os.name != "nt":
	#     raise OSError("org2xm is only available on Windows.")
	if not org or not isinstance(org, (bytes, memoryview)):
		if not is_url(org):
			raise TypeError("Invalid input URL.")
		org = verify_url(org)
		data = Request(org)
		if not data:
			raise FileNotFoundError("Error downloading file content.")
	else:
		if org[:4] != b"Org-":
			raise ValueError("Invalid file header.")
		data = org
	# Set compatibility option if file is not of org2 format.
	compat = data[:6] != b"Org-02"
	ts = ts_us()
	# Write org data to file.
	r_org = "cache/" + str(ts) + ".org"
	with open(r_org, "wb") as f:
		f.write(data)
	r_dat = "cache/" + str(ts) + ".dat"
	orig = False
	# Load custom sample bank if specified
	if dat is not None and is_url(dat):
		dat = verify_url(dat)
		with open(r_dat, "wb") as f:
			dat = Request(dat)
			f.write(dat)
	else:
		if type(dat) is bytes and dat:
			with open(r_dat, "wb") as f:
				f.write(dat)
		else:
			r_dat = "misc/ORG210EN.DAT"
			orig = True
	args = ["misc/org2xm", r_org, r_dat]
	if compat:
		args.append("c")
	subprocess.check_output(args)
	r_xm = f"cache/{ts}.xm"
	if not os.path.exists(r_xm):
		raise FileNotFoundError("Unable to locate converted file.")
	if not os.path.getsize(r_xm):
		raise RuntimeError("Converted file is empty.")
	for f in (r_org, r_dat)[:2 - orig]:
		with suppress():
			os.remove(f)
	return r_xm

def mid2mp3(mid):
	url = Request(
		"https://hostfast.onlineconverter.com/file/send",
		files={
			"class": (None, "audio"),
			"from": (None, "midi"),
			"to": (None, "mp3"),
			"source": (None, "file"),
			"file": mid,
			"audio_quality": (None, "192"),
		},
		method="post",
		decode=True,
	)
	fn = url.rsplit("/", 1)[-1].strip("\x00")
	for i in range(360):
		with Delay(1):
			test = Request(f"https://hostfast.onlineconverter.com/file/{fn}")
			if test == b"d":
				break
	ts = ts_us()
	r_mp3 = f"cache/{ts}.mp3"
	with open(r_mp3, "wb") as f:
		f.write(Request(f"https://hostfast.onlineconverter.com/file/{fn}/download"))
	return r_mp3

def png2wav(png):
	ts = ts_us()
	r_png = f"cache/{ts}"
	r_wav = f"cache/{ts}.wav"
	args = [sys.executable, "png2wav.py", "../" + r_png, "../" + r_wav]
	with open(r_png, "wb") as f:
		f.write(png)
	print(args)
	subprocess.run(args, cwd="misc", stderr=subprocess.PIPE)
	return r_wav

def ecdc_encode(ecdc, bitrate="24k", name=None, source=None):
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	b = await_fut(process_image("ecdc_encode", "$", [ecdc, bitrate, name, source], cap="ecdc", timeout=300))
	ts = ts_us()
	out = f"cache/{ts}.ecdc"
	with open(out, "wb") as f:
		f.write(b)
	return out

def ecdc_decode(ecdc, out=None):
	fmt = out.rsplit(".", 1)[-1] if out else "opus"
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	b = await_fut(process_image("ecdc_decode", "$", [ecdc, fmt], cap="ecdc", timeout=300))
	ts = ts_us()
	out = out or f"cache/{ts}.{fmt}"
	with open(out, "wb") as f:
		f.write(b)
	return out

CONVERTERS = {
	b"MThd": mid2mp3,
	b"Org-": org2xm,
	b"ECDC": ecdc_decode,
}

def select_and_convert(stream):
	print("Selecting and converting", stream)
	resp = reqs.next().get(stream, headers=Request.header(), timeout=8, stream=True)
	b = seq(resp)
	try:
		convert = CONVERTERS[b[:4]]
	except KeyError:
		convert = png2wav
	b = b.read()
	return convert(b)


class AudioFileLink:

	seekable = True
	live = False
	dur = None
	started = False

	def __init__(self, fn, stream=None, wasfile=None):
		self.fn = self.file = fn
		self.stream = stream
		self.streaming = Future()
		if stream:
			self.streaming.set_result(stream)
		self.readable = Future()
		if wasfile:
			self.readable.set_result(self)
			self.started = True
		source = bot.audio.submit(f"!AudioFile({repr(fn)},{repr(stream)},{repr(wasfile)})")
		self.assign = deque()
		# print("Loaded source", source)

	def __getattr__(self, k):
		if k == "__await__":
			raise AttributeError(k)
		try:
			return object.__getattribute__(self, k)
		except AttributeError:
			pass
		if not bot.audio:
			raise AttributeError("Audio client not active.")
		return bot.audio.submit(f"cache['{self.fn}'].{k}")

	def load(self, stream=None, check_fmt=False, force=False, webpage_url=None, live=False, seekable=True, duration=None, asap=True):
		if stream:
			self.stream = stream
		try:
			self.streaming.set_result(stream)
		except concurrent.futures.InvalidStateError:
			self.streaming = Future()
			self.streaming.set_result(stream)
		self.live = live
		self.seekable = seekable
		self.webpage_url = webpage_url
		timeout = 48 if asap else 300
		if not asap:
			ytdl.download_file(self.webpage_url, fmt="opus")
			self.started = True
			if duration:
				self.dur = duration
			try:
				self.readable.set_result(self)
			except concurrent.futures.InvalidStateError:
				pass
			return
		bot.audio.submit(f"!cache['{self.fn}'].load(" + ",".join(repr(i) for i in (stream, check_fmt, force, webpage_url, live, seekable, duration, asap)) + ")", timeout=timeout)
		self.started = True
		if duration:
			self.dur = duration
		try:
			self.readable.set_result(self)
		except concurrent.futures.InvalidStateError:
			pass

	def create_reader(self, pos, auds=None):
		if auds is not None:
			if auds.paused or abs(auds.stats.speed) < 0.005:
				return
			stats = cdict(auds.stats)
			stats.max_bitrate = auds.max_bitrate
			ident = cdict(stats=stats, args=[], guild_id=auds.guild.id)
			options=auds.construct_options(full=self.live)
		else:
			ident = None
			options = ()
		ts = ts_us()
		bot.audio.submit(f"!cache['{self.fn}'].create_reader({repr(pos)},{repr(ident)},{repr(options)},{ts})")
		return ts

	def duration(self):
		if not self.dur:
			self.dur = bot.audio.submit(f"cache['{self.fn}'].duration()")
		return self.dur

	def ensure_time(self):
		try:
			return bot.audio.submit(f"cache['{self.fn}'].ensure_time()")
		except KeyError:
			ytdl.cache.pop(self.fn, None)

	def update(self):
		# Newly loaded files have their duration estimates copied to all queue entries containing them
		if self.loaded:
			if not self.wasfile:
				dur = self.duration()
				if dur is not None:
					for e in self.assign:
						e["duration"] = dur
					self.assign.clear()
		return bot.audio.submit(f"cache['{self.fn}'].update()")

	def destroy(self):
		bot.audio.submit(f"cache['{self.fn}'].destroy()")
		ytdl.cache.pop(self.fn, None)

	ctime = 0
	def is_finished(self):
		if not self.ctime:
			self.ctime = utc()
		if utc() - self.ctime > 60:
			return True
		return self.started and (getattr(self, "loaded", None) or bot.audio.submit(f"bool(cache['{self.fn}'].proc)") and bot.audio.submit(f"not cache['{self.fn}'].proc.is_running()"))


class AudioClientSubInterface:

	afters = None
	bot = channel = None

	@classmethod
	@tracebacksuppressor
	def from_guild(cls, guild):
		cls.ensure_bot(cls)
		bot = cls.bot
		if bot.audio.players.get(guild.id):
			auds = bot.audio.players[guild.id]
		else:
			auds = None
		if guild.me and guild.me.voice:
			c_id = bot.audio.submit(f"getattr(getattr(client.get_guild({guild.id}).voice_client,'channel',None),'id',None)")
			if c_id:
				self = cls(auds)
				self.guild = guild
				self.channel = bot.get_channel(c_id)
				bot.audio.clients[guild.id] = self
				return self

	@classmethod
	def after(cls, key):
		cls.ensure_bot(cls)
		cls.afters[key]()

	_pos = 0
	@property
	def pos(self):
		try:
			if not self.auds or not self.auds.vc:
				raise AttributeError
			self._pos = self.bot.audio.submit(f"AP.from_guild({self.guild.id}).pos")
		except AttributeError:
			pass
		return self._pos

	def ensure_bot(self, auds=None):
		cls = self.__class__
		if type(cls) is type:
			cls = self
		if auds:
			cls.bot = auds.bot
		elif not cls.bot:
			cls.bot = bot
		if cls.bot:
			cls.afters = cls.bot.audio.__dict__.setdefault("afters", {})

	def __init__(self, auds, channel=None, reconnect=True):
		self.ensure_bot(auds)
		self.auds = auds
		bot = self.bot
		self.user = bot.user
		if channel:
			self.channel = channel
			self.guild = channel.guild

	async def start(self):
		if self.channel:
			bot = self.bot
			await bot.audio.asubmit(f"!await AP.join({self.channel.id})")
			bot.audio.clients[self.guild.id] = self

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: " + "{" + f'"guild": {self.guild}, "pos": {self.pos}' + "}"

	def __getattr__(self, k):
		if k in ("__await__"):
			raise AttributeError(k)
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not self.bot.audio:
			raise AttributeError("Audio client not active.")
		return self.bot.audio.submit(f"AP.from_guild({self.guild.id}).{k}")

	def enqueue(self, src, after=None):
		self.ensure_bot()
		if src is None:
			return
		if self.auds.source:
			s1 = self.auds.source.stream
			try:
				s2 = self.bot.audio.submit(f"AP.from_guild({self.guild.id}).queue[0][0].af.stream")
			except IndexError:
				self.auds.play(self.auds.source, self.auds.pos)
				s2 = self.bot.audio.submit(f"AP.from_guild({self.guild.id}).queue[0][0].af.stream")
			if s1 != s2:
				print(f"{self.guild} ({self.guild.id}): ACSI stream mismatch! Attempting fix...")
				if self.auds.next and s2 == self.auds.next.stream:
					self.auds.queue.advance()
				else:
					print(f"{self.guild} ({self.guild.id}): Unable to find safe position, resetting ACSI queue...")
					self.clear()
					self.auds.play(self.auds.source, self.auds.pos)
					if not self.auds.next:
						return
		key = ts_us()
		if after:
			self.afters[key] = after
			return esubmit(self.bot.audio.submit, f"AP.from_guild({self.guild.id}).enqueue(players[{repr(src)}],after=lambda *args: submit('VOICE.ACSI.after({key})'))")
		return esubmit(self.bot.audio.submit, f"AP.from_guild({self.guild.id}).enqueue(players[{repr(src)}])")

	def play(self, src, after=None):
		self.ensure_bot()
		if src is None:
			return
		key = ts_us()
		if after:
			self.afters[key] = after
			return self.bot.audio.submit(f"AP.from_guild({self.guild.id}).play(players[{repr(src)}],after=lambda *args: submit('VOICE.ACSI.after({key})'))")
		return self.bot.audio.submit(f"AP.from_guild({self.guild.id}).play(players[{repr(src)}])")

	def connect(self, reconnect=True, timeout=60):
		return create_task(self.bot.audio.asubmit(f"!await AP.from_guild({self.guild.id}).connect(reconnect={reconnect},timeout={timeout})"))

	async def disconnect(self, force=False):
		await self.bot.audio.asubmit(f"!await AP.from_guild({self.guild.id}).disconnect(force={force})")
		self.bot.audio.clients.pop(self.guild.id)
		self.channel = None

	async def move_to(self, channel=None):
		if not channel:
			return await self.disconnect(force=True)
		await self.bot.audio.asubmit(f"!await AP.from_guild({self.guild.id}).move_to(client.get_channel({channel.id}))")
		self.channel = channel

	def is_connected(self):
		try:
			return self._is_connected()
		except (AttributeError, TypeError):
			return False

	def is_paused(self):
		try:
			return self._is_paused()
		except (AttributeError, TypeError):
			return False

	def is_playing(self):
		try:
			return self._is_playing()
		except (AttributeError, TypeError):
			return False

ACSI = AudioClientSubInterface

for attr in ("read", "skip", "stop", "pause", "resume", "clear_source", "clear_next", "clear", "kill", "_is_connected", "_is_paused", "_is_playing"):
	setattr(ACSI, attr, eval("""lambda self: self.bot.audio.submit(f"AP.from_guild({self.guild.id}).""" + f"""{attr.lstrip('_')}()")"""))


# Manages all audio searching and downloading.
class AudioDownloader:

	_globals = globals()
	ydl_opts = {
		# "verbose": 1,
		"quiet": 1,
		"format": "bestvideo+bestaudio/best",
		"overwrites": 1,
		"nocheckcertificate": 1,
		"no_call_home": 1,
		"nooverwrites": 1,
		"noplaylist": 1,
		"logtostderr": 0,
		"ignoreerrors": 0,
		"default_search": "auto",
		"source_address": "0.0.0.0",
	}
	youtube_x = 0
	youtube_dl_x = 0
	spotify_x = 0
	other_x = 0
	bot = None

	def __init__(self):
		self.lastclear = 0
		self.downloading = cdict()
		self.cache = cdict()
		self.searched = Cache(timeout=3600, trash=256)
		self.semaphore = Semaphore(4, 128)
		self.download_sem = Semaphore(16, 64, rate_limit=0.5)
		esubmit(self.update_dl)
		esubmit(self.setup_pages)
		esubmit(self.set_cookie)

	# Fetches youtube playlist page codes, split into pages of 10 items
	def setup_pages(self):
		with open("misc/page_tokens.txt", "r", encoding="utf-8") as f:
			page10 = f.readlines()
		self.yt_pages = {i * 10: page10[i] for i in range(len(page10))}
	
	def set_cookie(self):
		self.youtube_base = "CONSENT=YES+cb.20210328-17-p0.en+FX"
		resp = reqs.next().get("https://www.youtube.com").text
		if "<title>Before you continue to YouTube</title>" in resp:
			resp = resp.split('<input type="hidden" name="v" value="', 1)[-1]
			resp = resp[:resp.index('">')].rsplit("+", 1)[0]
			self.youtube_base = f"CONSENT=YES+{resp}"
			self.youtube_x += 1

	def youtube_header(self):
		headers = Request.header()
		if self.youtube_base:
			headers["Cookie"] = self.youtube_base + "%03d" % random.randint(0, 999) + ";"
		return headers

	# Initializes youtube_dl object as well as spotify tokens, every 720 seconds.
	@tracebacksuppressor
	def update_dl(self):
		if utc() - self.lastclear > 720:
			self.lastclear = utc()
			self.youtube_dl_x += 1
			self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
			self.spotify_x += 1
			token = await_fut(aretry(Request, "https://open.spotify.com/get_access_token", aio=True, attempts=8, delay=5))
			self.spotify_header = {"authorization": f"Bearer {orjson.loads(token[:512])['accessToken']}"}
			self.other_x += 1

	ytd_blocked = Cache(timeout=3600)
	backup_sem = Semaphore(2, 256, rate_limit=1)
	# Gets data from yt-download.org, and adjusts the format to ensure compatibility with results from youtube-dl. Used as backup.
	def extract_backup(self, url, video=False):
		if url in self.ytd_blocked:
			raise self.ytd_blocked[url]
		entry = None
		ourl = url
		try:
			with self.backup_sem:
				url = verify_url(url)
				if is_url(url) and not is_youtube_url(url):
					with requests.head(url, headers=Request.header(), stream=True) as resp:
						url = resp.url
						name = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
						ctype = resp.headers.get("Content-Type")
						if ctype.startswith("video") or ctype.startswith("audio"):
							return dict(
								id=name,
								title=name,
								direct=True,
								url=url,
								webpage_url=url,
								extractor="generic",
							)
						elif ctype == "application/octet-stream":
							dur = get_duration(url)
							d = dict(
								id=name,
								title=name,
								direct=True,
								url=url,
								webpage_url=url,
								extractor="generic",
							)
							if dur:
								d["duration"] = dur
							return d
					raise TypeError("Unsupported URL.")
				if ":" in url:
					url = url.rsplit("/", 1)[-1].split("v=", 1)[-1].split("&", 1)[0]
				webpage_url = f"https://www.youtube.com/watch?v={url}"
				if video:
					title, stream = yt_download(webpage_url, fmt="mp4", timeout=720)
					entry = dict(
						formats=[dict(
							abr=1,
							url=stream,
							height=1080,
						)],
						title=title,
						webpage_url=webpage_url,
					)
				else:
					title, stream = yt_download(webpage_url, fmt="mp3", timeout=720)
					entry = dict(
						formats=[dict(
							abr=256,
							url=stream,
						)],
						duration=os.path.getsize(stream) / 256000 * 8,
						title=title,
						webpage_url=webpage_url,
					)
			print("Successfully resolved with yt-download.")
		except Exception as ex:
			self.ytd_blocked[ourl] = ex
			if not entry:
				raise
		return entry

	# Returns part of a spotify playlist.
	def get_spotify_part(self, url):
		try:
			d = Request(url, headers=self.spotify_header, json=True)
		except ConnectionError:
			d = None
		if not d:
			self.update_dl()
			d = Request(url, headers=self.spotify_header, json=True)
		return self.export_spotify_part(d)

	async def get_spotify_part_async(self, url):
		try:
			d = await Request(url, headers=self.spotify_header, json=True, aio=True)
		except ConnectionError:
			d = None
		if not d:
			await asubmit(self.update_dl)
			d = await Request(url, headers=self.spotify_header, json=True, aio=True)
		return self.export_spotify_part(d)

	def export_spotify_part(self, d):
		self.spotify_x += 1
		out = deque()
		with suppress(KeyError):
			d = d["tracks"]
		try:
			items = d["items"]
			total = d.get("total", 0)
		except KeyError:
			if "type" in d:
				items = (d,)
				total = 1
			else:
				items = []
				total = 0
		for item in items:
			try:
				track = item["track"]
			except KeyError:
				try:
					track = item["episode"]
				except KeyError:
					if "id" in item:
						track = item
					else:
						continue
			name = track.get("name", track["id"])
			artists = ", ".join(a["name"] for a in track.get("artists", []))
			dur = track.get("duration_ms")
			if dur:
				dur /= 1000
			temp = cdict(
				name=name,
				url="ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}"),
				id=track["id"],
				duration=dur,
				research=True,
			)
			out.append(temp)
		return out, total

	# Returns a list of formatted queue entries from a YouTube playlist renderer.
	def extract_playlist_items(self, items):
		token = None
		out = deque()
		for data in items:
			try:
				video = data["playlistVideoRenderer"]
			except KeyError:
				try:
					token = data["continuationItemRenderer"]["continuationEndpoint"]["continuationCommand"]["token"]
				except KeyError:
					print(data)
				continue
			v_id = video['videoId']
			try:
				dur = round_min(float(video["lengthSeconds"]))
			except (KeyError, ValueError):
				try:
					dur = time_parse(video["lengthText"]["simpleText"])
				except KeyError:
					dur = None
			try:
				name = video["title"]["runs"][0]["text"]
			except LookupError:
				name = v_id
			temp = cdict(
				name=name,
				url=f"https://www.youtube.com/watch?v={v_id}",
				duration=dur,
			)
			out.append(temp)
		return out, token

	# Returns a subsequent page of a youtube playlist from a page token.
	def get_youtube_continuation(self, token, ctx):
		self.youtube_x += 1
		for i in range(3):
			try:
				data = Request(
					"https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
					headers=self.youtube_header(),
					method="POST",
					data=orjson.dumps(dict(
						context=ctx,
						continuation=token,
					)),
					json=True,
				)
			except:
				print_exc()
				if i:
					time.sleep(i)
		items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		return self.extract_playlist_items(items)

	# Async version of the previous function, used when possible to minimise thread pool wastage.
	async def get_youtube_continuation_async(self, token, ctx):
		self.youtube_x += 1
		data = await Request(
			"https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
			headers=self.youtube_header(),
			method="POST",
			data=orjson.dumps(dict(
				context=ctx,
				continuation=token,
			)),
			json=True,
			aio=True,
		)
		items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		return self.extract_playlist_items(items)

	# Generates a playlist continuation token purely from ID and page number.
	def produce_continuation(self, p, i):
		if not isinstance(p, (bytes, bytearray, memoryview)):
			p = str(p).encode("ascii")
		parts = []
		if i == 1:
			parts.append(b"\xe2\xa9\x85\xb2\x02a\x12$VL")
		else:
			parts.append(b"\xe2\xa9\x85\xb2\x02_\x12$VL")
		parts.append(p)
		if i == 1:
			parts.append(b"\x1a\x14")
		else:
			parts.append(b"\x1a\x12")
		import base64
		def leb128(n):
			data = bytearray()
			while n:
				data.append(n & 127)
				n >>= 7
				if n:
					data[-1] |= 128
			if not data:
				data = b"\x00"
			return data
		key = bytes((8, i, 0x7a, (i != 1) + 6)) + b"PT:" + base64.b64encode(b"\x08" + leb128(i * 100)).rstrip(b"=")
		obj = base64.b64encode(key).replace(b"=", b"%3D")
		parts.append(obj)
		parts.append(b"\x9a\x02\x22")
		parts.append(p)
		code = b"".join(parts)
		return base64.b64encode(code).replace(b"=", b"%3D").decode("ascii")

	# Returns a full youtube playlist.
	def get_youtube_playlist(self, p_id):
		self.youtube_x += 1
		resp = Request(f"https://www.youtube.com/playlist?list={p_id}", headers=self.youtube_header())
		client = {}
		try:
			ytcfg = resp[resp.index(b"ytcfg.set"):]
			ytcfg = ytcfg[:ytcfg.index(b";")]
			ytcfg = eval(ytcfg.split(b"(", 1)[-1].rsplit(b")", 1)[0], {}, {})[-1] + "&"
			end = "&"
			start = "client.name="
			cname = ytcfg[ytcfg.index(start) + len(start):]
			client["clientName"] = cname[:cname.index(end)]
			start = "client.version="
			cversion = ytcfg[ytcfg.index(start) + len(start):]
			client["clientVersion"] = cversion[:cversion.index(end)]
		except ValueError:
			pass
		client.setdefault("clientName", "WEB")
		client.setdefault("clientVersion", "2.20211019")
		context = dict(client=client)
		try:
			try:
				resp = resp[resp.index(b'{"responseContext":{'):]
			except ValueError:
				search = b"var ytInitialData = "
				try:
					resp = resp[resp.index(search) + len(search):]
				except ValueError:
					search = b'window["ytInitialData"] = '
					resp = resp[resp.index(search) + len(search):]
			try:
				resp = resp[:resp.index(b';</script><')]
			except ValueError:
				resp = resp[:resp.index(b'window["ytInitialPlayerResponse"] = null;')]
				resp = resp[:resp.rindex(b";")]
			data = orjson.loads(resp)
		except:
			print(resp)
			raise
		count = int(data["sidebar"]["playlistSidebarRenderer"]["items"][0]["playlistSidebarPrimaryInfoRenderer"]["stats"][0]["runs"][0]["text"].replace(",", ""))
		items = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"]["contents"]
		entries, token = self.extract_playlist_items(items)
		if count > 100:
			futs = deque()
			if not token:
				token = self.produce_continuation(p_id, 1)
			for page in range(1, ceil(count / 100)):
				if is_main_thread():
					fut = esubmit(self.get_youtube_continuation, token, context)
				else:
					fut = convert_fut(self.get_youtube_continuation_async(token, context))
				futs.append(fut)
				token = self.produce_continuation(p_id, page + 1)
			for fut in futs:
				entries.extend(fut.result()[0])
		out = deque()
		urls = set()
		for entry in entries:
			if entry.url not in urls:
				urls.add(entry.url)
				out.append(entry)
		return out

	soundcloud_token = "7g7gIkrcAS05cJVf2FlIsnkOXtg4JdSe"

	def get_soundcloud_playlist(self, url):
		parts = url.split("?", 1)[0].split("/")
		if parts[0] != "https:" or parts[2] not in ("soundcloud.com", "api-v2.soundcloud.com"):
			raise TypeError("Not a SoundCloud playlist.")
		if parts[-1] == "likes":
			return self.get_soundcloud_likes(url)
		api = "https://api-v2.soundcloud.com/"

		resp = requests.get(url, headers=Request.header())
		resp.raise_for_status()
		s = resp.text
		if s[0] == "{" and s[-1] == "}":
			t = resp.json()
			return [cdict(
				name=t["title"],
				url=t["permalink_url"],
				duration=t["duration"] / 1000,
				thumbnail=t["artwork_url"],
			)]
		search = "<script>window.__sc_hydration = "
		s = s[s.index(search) + len(search):]
		s = s[:s.index(";</script>")]
		data = orjson.loads(s)

		emap = {}
		entries = []
		for hydratable in data:
			if hydratable["hydratable"] == "playlist":
				for t in hydratable["data"]["tracks"]:
					try:
						t["title"]
					except KeyError:
						tid = t["id"]
						emap[tid] = len(entries)
						entries.append(None)
					else:
						entry = cdict(
							name=t["title"],
							url=t["permalink_url"],
							duration=t["duration"] / 1000,
							thumbnail=t["artwork_url"],
						)
						entries.append(entry)

		if emap:
			ids = ",".join(map(str, emap))
			url = f"{api}tracks?ids={ids}&client_id={self.soundcloud_token}"
			resp = requests.get(url, headers=Request.header())
			if not resp.content:
				resp.raise_for_status()
			for t, p in zip(resp.json(), emap.values()):
				entry = cdict(
					name=t["title"],
					url=t["permalink_url"],
					duration=t["duration"] / 1000,
					thumbnail=t["artwork_url"],
				)
				entries[p] = entry
		return [e for e in entries if e]

	def get_soundcloud_likes(self, url):
		api = "https://api-v2.soundcloud.com/"
		lim = 1000

		uapi = api + "users/"
		if url.startswith(uapi):
			uid = url[len(uapi):].split("?", 1)[0]
		else:
			resp = requests.get(url, headers=Request.header())
			resp.raise_for_status()
			s = resp.text
			search = 'content="soundcloud://users:'
			s = s[s.index(search) + len(search):]
			uid = s[:s.index('"')]

		futs = []
		entries = []
		while True:
			url = f"{api}users/{uid}/likes?client_id={self.soundcloud_token}&limit={lim}"
			resp = requests.get(url, headers=Request.header())
			if not resp.content:
				resp.raise_for_status()
			data = resp.json()
			for e in data["collection"]:
				try:
					t = e["track"]
				except KeyError:
					p = e["playlist"]
					url = p["permalink_url"]
					if len(futs) >= 12:
						futs.pop(0).result()
					fut = esubmit(self.get_soundcloud_playlist, url)
					futs.append(fut)
					entries.append(fut)
				else:
					entry = cdict(
						name=t["title"],
						url=t["permalink_url"],
						duration=t["duration"] / 1000,
						thumbnail=t["artwork_url"],
					)
					entries.append(entry)
			url = data.get("next_href")
			if len(entries) < lim or not url:
				break
			url += f"client_id={self.soundcloud_token}&limit={lim}"

		while True:
			for i, e in enumerate(entries):
				if isinstance(e, Future):
					entries = entries[:i] + e.result() + entries[i + 1:]
					break
			else:
				break
		return entries

	def ydl_errors(self, s):
		return "this video has been removed" not in s and "private video" not in s and "has been terminated" not in s and ("Video unavailable" not in s or not self.backup_sem.active)

	blocked_yt = False

	@functools.lru_cache(maxsize=64)
	def extract_audio_video(self, url):
		with reqs.next().get(url, headers=Request.header(), stream=True) as resp:
			resp.raise_for_status()
			ct = resp.headers.get("Content-Type")
			if ct == "text/html":
				s = resp.text
				out = []
				matches = re.findall(r"""(?:audio|video)\.src ?= ?['"]https?:""", s)
				t = s
				for match in matches:
					try:
						t = t[t.index(match):]
						t = t[t.index("http"):]
						t2 = t[:re.search(r"""['"]""", t).start()]
					except (IndexError, ValueError):
						continue
					if is_url(t2):
						title = t2.split("?", 1)[0].rsplit("/", 1)[-1]
						temp = dict(url=t2, webpage_url=url, title=title, direct=True)
						out.append(temp)
				matches = re.findall(r"""<(?:audio|video) """, s)
				t = s
				for match in matches:
					try:
						t = t[t.index(match):]
						t = t[t.index("src="):]
						t = t[t.index("http"):]
						t2 = t[:re.search(r"""['"]""", t).start()]
					except (IndexError, ValueError):
						continue
					if is_url(t2):
						title = t2.split("?", 1)[0].rsplit("/", 1)[-1]
						temp = dict(url=t2, webpage_url=url, title=title, direct=True)
						out.append(temp)
				if len(out) > 1:
					return {"_type": "playlist", "entries": out}
				elif out:
					return out[0]
			elif ct.split("/", 1)[0] in ("audio", "video", "image"):
				title = url.split("?", 1)[0].rsplit("/", 1)[-1]
				return dict(url=url, webpage_url=url, title=title, direct=True)

	# Repeatedly makes calls to youtube-dl until there is no more data to be collected.
	def extract_true(self, url):
		while not is_url(url):
			try:
				resp = self.search_yt(regexp("ytsearch[0-9]*:").sub("", url, 1))[0]
			except:
				resp = self.extract_from(url)
			if "entries" in resp:
				resp = next(iter(resp["entries"]))
			if "duration" in resp and "formats" in resp:
				out = cdict(
					name=resp["title"],
					url=resp["webpage_url"],
					duration=resp["duration"],
					stream=get_best_audio(resp),
					icon=get_best_icon(resp),
					video=get_best_video(resp),
				)
				stream = out.stream
				if "googlevideo" in stream[:64]:
					durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
					if durstr:
						out.duration = round_min(durstr[0])
				return out
			try:
				url = resp["webpage_url"]
			except KeyError:
				try:
					url = resp["url"]
				except KeyError:
					url = resp["id"]
		if is_discord_url(url):
			title = url.split("?", 1)[0].rsplit("/", 1)[-1]
			if title.rsplit(".", 1)[-1] in ("ogg", "webm", "mp4", "avi", "mov"):
				url2 = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
				with reqs.next().get(url2, headers=Request.header(), stream=True) as resp:
					if resp.status_code in range(200, 400):
						url = url2
			if "." in title:
				title = title[:title.rindex(".")]
			return dict(url=url, name=title, direct=True)
		try:
			if self.blocked_yt > utc():
				raise PermissionError
			self.youtube_dl_x += 1
			entries = self.downloader.extract_info(url, download=False, process=True)
		except Exception as ex:
			s = str(ex).casefold()
			if "unsupported url:" in s:
				out = self.extract_audio_video(url)
				if out:
					return out
				raise
			elif not isinstance(ex, youtube_dl.DownloadError) or self.ydl_errors(s):
				if "429" in s:
					self.blocked_yt = utc() + 3600
				if has_ytd:
					try:
						entries = self.extract_backup(url)
					except (TypeError, youtube_dl.DownloadError):
						raise FileNotFoundError(f"Unable to fetch audio data: {repr(ex)}")
				else:
					raise
			else:
				raise
		if "entries" in entries:
			entries = entries["entries"]
		else:
			entries = [entries]
		out = deque()
		for entry in entries:
			temp = cdict(
				name=entry["title"],
				url=entry["webpage_url"],
				duration=entry.get("duration"),
				stream=get_best_audio(entry),
				icon=get_best_icon(entry),
				video=get_best_video(entry),
			)
			stream = temp.stream
			if "googlevideo" in stream[:64]:
				durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
				if durstr:
					temp.duration = round_min(durstr[0])
			out.append(temp)
		return out

	# Extracts audio information from a single URL.
	def extract_from(self, url):
		if is_discord_url(url):
			title = url.split("?", 1)[0].rsplit("/", 1)[-1]
			if title.rsplit(".", 1)[-1] in ("ogg", "webm", "mp4", "avi", "mov"):
				url2 = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
				with reqs.next().get(url2, headers=Request.header(), stream=True) as resp:
					if resp.status_code in range(200, 400):
						url = url2
			if "." in title:
				title = title[:title.rindex(".")]
			return dict(url=url, webpage_url=url, title=title, direct=True)
		try:
			if self.blocked_yt > utc():
				raise PermissionError
			if url.startswith("https://www.youtube.com/search") or url.startswith("https://www.youtube.com/results"):
				url = url.split("=", 1)[1].split("&", 1)[0]
			self.youtube_dl_x += 1
			return self.downloader.extract_info(url, download=False, process=False)
		except Exception as ex:
			s = str(ex).casefold()
			if "unsupported url:" in s:
				out = self.extract_audio_video(url)
				if out:
					return out
				raise
			elif not isinstance(ex, youtube_dl.DownloadError) or self.ydl_errors(s):
				if "429" in s:
					self.blocked_yt = utc() + 3600
				if has_ytd:
					try:
						entries = self.extract_backup(url)
					except (TypeError, youtube_dl.DownloadError):
						raise FileNotFoundError(f"Unable to fetch audio data: {repr(ex)}")
				else:
					raise
			else:
				raise

	# Extracts info from a URL or search, adjusting accordingly.
	def extract_info(self, item, count=1, search=False, mode=None):
		if (mode or search) and item[:9] not in ("ytsearch:", "scsearch:", "spsearch:", "bcsearch:") and not is_url(item):
			if count == 1:
				c = ""
			else:
				c = count
			item = item.replace(":", "-")
			if mode:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"{mode}search{c}:{item}", download=False, process=False)
			exc = ""
			try:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"ytsearch{c}:{item}", download=False, process=False)
			except Exception as ex:
				exc = repr(ex)
			try:
				self.youtube_dl_x += 1
				return self.downloader.extract_info(f"scsearch{c}:{item}", download=False, process=False)
			except Exception as ex:
				raise ConnectionError(exc + repr(ex))
		if item[:9] == "spsearch:":
			query = "https://api.spotify.com/v1/search?type=track%2Cshow_audio%2Cepisode_audio&include_external=audio&limit=1&q=" + url_parse(item[9:])
			resp = reqs.next().get(query, headers=self.spotify_header).json()
			try:
				track = resp["tracks"]["items"][0]
				name = track.get("name", track["id"])
				artists = ", ".join(a["name"] for a in track.get("artists", ()))
			except LookupError:
				return dict(_type="playlist", entries=[])
			else:
				item = "ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}")
				self.spotify_x += 1
		elif item[:9] == "bcsearch:":
			query = "https://bandcamp.com/search?q=" + url_parse(item[9:])
			resp = reqs.next().get(query).content
			try:
				resp = resp.split(b'<ul class="result-items">', 1)[1]
				tracks = resp.split(b"<!-- search result type=")
				result = cdict()
				for track in tracks:
					if track.startswith(b"track id=") or track.startswith(b"album id=") and not result:
						ttype = track[:5]
						try:
							track = track.split(b'<img src="', 1)[1]
							result.thumbnail = track[:track.index(b'">')].decode("utf-8", "replace")
						except ValueError:
							pass
						track = track.split(b'<div class="heading">', 1)[1]
						result.title = track.split(b">", 1)[1].split(b"<", 1)[0].strip().decode("utf-8", "replace")
						result.url = track.split(b'href="', 1)[1].split(b'"', 1)[0].split(b"?", 1)[0].decode("utf-8", "replace")
						if ttype == b"track":
							break
				if not result:
					raise LookupError
				return result
			except (LookupError, ValueError):
				return dict(_type="playlist", entries=[])
			else:
				item = "ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}")
				self.other_x += 1
		elif is_url(item) or not search:
			return self.extract_from(item)
		self.youtube_dl_x += 1
		return self.downloader.extract_info(item, download=False, process=False)

	# Main extract function, able to extract from youtube playlists much faster than youtube-dl using youtube API, as well as ability to follow spotify links.
	def extract(self, item, force=False, count=1, mode=None, search=True):
		try:
			page = None
			output = deque()
			if "youtube.com" in item or "youtu.be/" in item:
				p_id = None
				for x in ("?list=", "&list="):
					if x in item:
						p_id = item[item.index(x) + len(x):]
						p_id = p_id.split("&", 1)[0]
						break
				if p_id:
					with tracebacksuppressor:
						output.extend(self.get_youtube_playlist(p_id))
						# Scroll to highlighted entry if possible
						v_id = None
						for x in ("?v=", "&v="):
							if x in item:
								v_id = item[item.index(x) + len(x):]
								v_id = v_id.split("&", 1)[0]
								break
						if v_id:
							for i, e in enumerate(output):
								if v_id in e.url:
									output.rotate(-i)
									break
						return output
			elif regexp("^https:\\/\\/soundcloud\\.com\\/[A-Za-z0-9]+\\/sets\\/").search(item) or regexp("^https:\\/\\/soundcloud\\.com\\/[A-Za-z0-9]+\\/likes").search(item) or regexp("^https:\\/\\/api-v2\\.soundcloud\\.com\\/users\\/[0-9]+\\/likes").search(item):
				with tracebacksuppressor:
					return self.get_soundcloud_playlist(item)
			elif regexp("(play|open|api)\\.spotify\\.com").search(item):
				# Spotify playlist searches contain up to 100 items each
				if "playlist" in item:
					url = item[item.index("playlist"):]
					url = url[url.index("/") + 1:]
					key = url.split("/", 1)[0]
					url = f"https://api.spotify.com/v1/playlists/{key}/tracks?type=track,episode"
					page = 100
				# Spotify album searches contain up to 50 items each
				elif "album" in item:
					url = item[item.index("album"):]
					url = url[url.index("/") + 1:]
					key = url.split("/", 1)[0]
					url = f"https://api.spotify.com/v1/albums/{key}/tracks?type=track,episode"
					page = 50
				# Single track links also supported
				elif "track" in item:
					url = item[item.index("track"):]
					url = url[url.index("/") + 1:]
					key = url.split("/", 1)[0]
					url = f"https://api.spotify.com/v1/tracks/{key}"
					page = 1
				# Single episode links also supported
				elif "episode" in item:
					url = item[item.index("episode"):]
					url = url[url.index("/") + 1:]
					key = url.split("/", 1)[0]
					url = f"https://api.spotify.com/v1/episodes/{key}"
					page = 1
				else:
					raise TypeError("Unsupported Spotify URL.")
				if page == 1:
					output.extend(self.get_spotify_part(url)[0])
				else:
					futs = deque()
					maxitems = 10000
					# Optimized searching with lookaheads
					for i, curr in enumerate(range(0, maxitems, page)):
						with Delay(0.03125):
							if curr >= maxitems:
								break
							search = f"{url}&offset={curr}&limit={page}"
							if is_main_thread() or i < 8:
								fut = esubmit(self.get_spotify_part, search, timeout=90)
							else:
								fut = convert_fut(self.get_spotify_part_async(search))
							print("Sent 1 spotify search.")
							futs.append(fut)
							if not (i < 1 or math.log2(i + 1) % 1) or not i & 7:
								while futs:
									fut = futs.popleft()
									res = fut.result()
									if not i:
										maxitems = res[1] + page
									if not res[0]:
										maxitems = 0
										futs.clear()
										break
									output += res[0]
					while futs:
						output.extend(futs.popleft().result()[0])
					# Scroll to highlighted entry if possible
					v_id = None
					for x in ("?highlight=spotify:track:", "&highlight=spotify:track:"):
						if x in item:
							v_id = item[item.index(x) + len(x):]
							v_id = v_id.split("&", 1)[0]
							break
					if v_id:
						for i, e in enumerate(output):
							if v_id == e.get("id"):
								output.rotate(-i)
								break
			elif is_url(item) and tuple(reversed(item.split("/", 3)[2].encode("utf-8"))) == (109, 111, 99, 46, 117, 109, 105, 110, 97, 120):
				b = Request(item, headers=self.youtube_header(), timeout=12)
				search = b'itemprop="name" content="'
				try:
					i = b.index(search)
				except ValueError:
					pass
				else:
					s, b = b[i + len(search):].split(b'"', 1)
					name = s.decode("utf-8", "replace")
					search = b'itemprop="duration" content="'
					s, b = b.split(search, 1)[-1].split(b'"', 1)
					duration = time_parse(s.decode("utf-8", "replace").lstrip("P").rstrip("S").replace("DT", ":").replace("H", ":").replace("M", ":"))
					search = b'itemprop="thumbnailUrl" content="'
					s, b = b.split(search, 1)[-1].split(b'"', 1)
					thumbnail = s.decode("utf-8", "replace")
					search = b'itemprop="embedURL" content="'
					with requests.head(b.split(search, 1)[-1].split(b'"', 1)[0].decode("utf-8", "replace"), headers=self.youtube_header(), allow_redirects=True) as resp:
						stream = resp.url
					temp = cdict(
						name=name,
						url=item,
						duration=duration,
						stream=stream,
						icon=thumbnail,
						video=stream,
					)
					output.append(temp)
			# Only proceed if no items have already been found (from playlists in this case)
			if not len(output):
				resp = None
				# Allow loading of files output by ~dump
				if is_url(item):
					url = verify_url(item)
					utest = url.split("?", 1)[0]
					if utest[-5:] == ".json" or utest[-4:] in (".txt", ".zip"):
						s = await_fut(self.bot.get_request(url))
						try:
							d = select_and_loads(s, size=268435456)
						except orjson.JSONDecodeError:
							d = [url for url in as_str(s).splitlines() if is_url(url)]
							if not d:
								raise
							q = [dict(name=url.split("?", 1)[0].rsplit("/", 1)[-1], url=url) for url in d]
						else:
							q = d["queue"][:262144]
						return [cdict(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
				elif mode in (None, "yt"):
					with suppress(NotImplementedError):
						res = self.search_yt(item, count=count)
						res = [cdict(**e, webpage_url=e.get("url"), title=e.get("name")) for e in res]
						if res:
							resp = cdict(_type="playlist", entries=res, url=res[0].webpage_url)
				# Otherwise call automatic extract_info function
				if not resp:
					resp = self.extract_info(item, count, search=search, mode=mode)
				if resp.get("_type") == "url":
					resp = self.extract_from(resp["url"])
				if resp is None or not len(resp):
					raise LookupError(f"No results for {item}")
				# Check if result is a playlist
				if resp.get("_type") == "playlist":
					entries = list(resp["entries"])
					if force or len(entries) <= 1:
						for entry in entries:
							# Extract full data if playlist only contains 1 item
							try:
								data = self.extract_from(entry["url"])
							except KeyError:
								url = get_best_video(entry)
								temp = cdict({
									"name": resp["title"],
									"url": resp.get("webpage_url", url),
									"duration": inf,
									"stream": get_best_audio(entry),
									"icon": get_best_icon(entry),
									"video": url,
								})
							else:
								try:
									dur = round_min(data["duration"])
								except:
									dur = None
								temp = cdict({
									"name": data["title"],
									"url": data["webpage_url"],
									"duration": dur,
									"stream": get_best_audio(data),
									"icon": get_best_icon(data),
									"video": get_best_video(data),
								})
							stream = temp.stream
							if "googlevideo" in stream[:64]:
								durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
								if durstr:
									temp.duration = round_min(durstr[0])
							output.append(temp)
					else:
						found = False
						for entry in entries:
							if not found:
								# Extract full data from first item only
								try:
									if "url" in entry:
										temp = self.extract(entry["url"], search=False)[0]
									elif "formats" in entry:
										url = get_best_video(entry)
										temp = cdict({
											"name": resp["title"],
											"url": resp.get("webpage_url", url),
											"duration": inf,
											"stream": get_best_audio(entry),
											"icon": get_best_icon(entry),
											"video": url,
										})
								except:
									print_exc()
									continue
								else:
									found = True
							else:
								# Get as much data as possible from all other items, set "research" flag to have bot lazily extract more info in background
								with tracebacksuppressor:
									found = True
									if "title" in entry:
										title = entry["title"]
									else:
										title = entry["url"].rsplit("/", 1)[-1]
										if "." in title:
											title = title[:title.rindex(".")]
										found = False
									try:
										dur = round_min(entry["duration"])
									except:
										dur = None
									url = entry.get("webpage_url", entry.get("url", entry.get("id")))
									if not url:
										continue
									temp = {
										"name": title,
										"url": url,
										"duration": dur,
									}
									if not is_url(url):
										if entry.get("ie_key", "").casefold() == "youtube":
											temp["url"] = f"https://www.youtube.com/watch?v={url}"
							output.append(cdict(temp))
				else:
					# Single item results must contain full data, we take advantage of that here
					name = resp.get("title") or resp["webpage_url"].rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
					url = resp.get("webpage_url") or resp["url"]
					found = "duration" in resp
					if found:
						dur = resp["duration"]
					else:
						dur = None
						if resp.get("direct") and os.path.exists("misc/ecdc_stream.py") and url.endswith(".ecdc"):
							args = [sys.executable, "misc/ecdc_stream.py", "-i", url]
							with suppress():
								info = subprocess.check_output(args).decode("utf-8", "replace").splitlines()
								assert info
								info = cdict(line.split(": ", 1) for line in info if line)
								if info.get("Name"):
									name = orjson.loads(info["Name"]) or name
								if info.get("Duration"):
									dur = orjson.loads(info["Duration"]) or dur
								if info.get("Source"):
									url = orjson.loads(info["Source"]) or url
									resp["url"] = url
					temp = cdict({
						"name": name,
						"url": url,
						"duration": dur,
						"stream": get_best_audio(resp),
						"icon": get_best_icon(resp),
						"video": get_best_video(resp),
					})
					stream = temp.stream
					if "googlevideo" in stream[:64]:
						durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
						if durstr:
							temp.duration = round_min(durstr[0])
					output.append(temp)
			return output
		except:
			if force != "spotify":
				raise
			print_exc()
			return 0

	def item_yt(self, item):
		video = next(iter(item.values()))
		if "videoId" not in video:
			return
		try:
			dur = time_parse(video["lengthText"]["simpleText"])
		except KeyError:
			dur = None
		try:
			title = video["title"]["runs"][0]["text"]
		except KeyError:
			title = video["title"]["simpleText"]
		try:
			tn = video["thumbnail"]
		except KeyError:
			thumbnail = None
		else:
			if type(tn) is dict:
				thumbnail = sorted(tn["thumbnails"], key=lambda t: t.get("width", 0) * t.get("height", 0))[-1]["url"]
			else:
				thumbnail = tn
		try:
			views = int(video["viewCountText"]["simpleText"].replace(",", "").replace("views", "").replace(" ", ""))
		except (KeyError, ValueError):
			views = 0
		return cdict(
			name=video["title"]["runs"][0]["text"],
			url=f"https://www.youtube.com/watch?v={video['videoId']}",
			duration=dur,
			icon=thumbnail,
			views=views,
		)

	def parse_yt(self, s):
		data = orjson.loads(s)
		results = alist()
		try:
			pages = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
		except KeyError:
			pages = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
		for page in pages:
			try:
				items = next(iter(page.values()))["contents"]
			except KeyError:
				continue
			for item in items:
				if "promoted" not in next(iter(item)).casefold():
					entry = self.item_yt(item)
					if entry is not None:
						results.append(entry)
		return sorted(results, key=lambda entry: entry.views, reverse=True)

	def search_yt(self, query, skip=False, count=1):
		out = alist()
		if not skip:
			with tracebacksuppressor:
				resp = self.extract_info(query, search=True, count=count)
				if resp.get("_type", None) == "url":
					resp = self.extract_from(resp["url"])
				if resp.get("_type", None) == "playlist":
					entries = list(resp["entries"])
				else:
					entries = [resp]
				for entry in entries:
					found = True
					if "title" in entry:
						title = entry["title"]
					else:
						title = entry["url"].rsplit("/", 1)[-1]
						if "." in title:
							title = title[:title.rindex(".")]
						found = False
					url = entry.get("webpage_url", entry.get("url", entry.get("id")))
					if not url:
						continue
					if entry.get("duration"):
						dur = float(entry["duration"])
					else:
						dur = None
					temp = cdict(name=title, url=url, duration=dur)
					if not is_url(url):
						if entry.get("ie_key", "").casefold() == "youtube":
							temp["url"] = f"https://www.youtube.com/watch?v={url}"
					out.append(temp)
		if not out:
			url = f"https://www.youtube.com/results?search_query={verify_url(query)}"
			self.youtube_x += 1
			resp = Request(url, headers=self.youtube_header(), timeout=12)
			result = None
			s = resp
			with suppress(ValueError):
				with suppress(ValueError):
					s = s[s.index(b"// scraper_data_begin") + 21:s.rindex(b"// scraper_data_end")]
				s = s[s.index(b"var ytInitialData = ") + 20:]
				s = s[:s.index(b";</script>")]
				result = self.parse_yt(s)
			with suppress(ValueError):
				s = s[s.index(b'window["ytInitialData"] = ') + 26:]
				s = s[:s.index(b'window["ytInitialPlayerResponse"] = null;')]
				s = s[:s.rindex(b";")]
				result = self.parse_yt(s)
			if result is not None:
				q = to_alphanumeric(full_prune(query))
				high = alist()
				low = alist()
				for entry in result:
					if entry.duration:
						name = full_prune(entry.name)
						aname = to_alphanumeric(name)
						spl = aname.split()
						if entry.duration < 960 or "extended" in q or "hour" in q or "extended" not in spl and "hour" not in spl and "hours" not in spl:
							if fuzzy_substring(aname, q, match_length=False) >= 0.5:
								high.append(entry)
								continue
					low.append(entry)

				def key(entry):
					coeff = fuzzy_substring(to_alphanumeric(full_prune(entry.name)), q, match_length=False)
					if coeff < 0.5:
						coeff = 0
					return coeff

				out = sorted(high, key=key, reverse=True)
				out.extend(sorted(low, key=key, reverse=True))
			if not out and len(query) < 16:
				self.failed_yt = utc() + 180
				print(query)
		return out[:count]

	# Performs a search, storing and using cached search results for efficiency.
	def search(self, item, force=False, mode=None, images=False, count=1):
		item = verify_search(item)
		if not is_main_thread():
			with tracebacksuppressor:
				items = await_fut(self.bot.follow_url(item, images=images))
				if items:
					item = items[0]
		if mode is None and count == 1:
			try:
				return self.searched[item]
			except KeyError:
				pass
			try:
				output = self.searched.retrieve(item)
			except KeyError:
				pass
			else:
				esubmit(self.search, item, force=force, mode=mode, images=images, count=count)
				return output
		with self.semaphore:
			try:
				output = None
				remote = AUTH.get("remote-servers", ())
				if 0 and remote:
					with tracebacksuppressor:
						addr = choice(remote)
						resp = requests.get(f"https://{addr}/stat/?api=ytdl&q={urllib.parse.quote_plus(item)}", verify=False)
						resp.raise_for_status()
						output = [cdict(e) for e in resp.json()]
				if not output:
					output = self.extract(item, force, mode=mode, count=count)
				self.searched[item] = output
				return output
			except Exception as ex:
				print_exc()
				return repr(ex)

	# Gets the stream URL of a queue entry, starting download when applicable.
	def get_stream(self, entry, video=False, force=False, download=True, callback=None, asap=True):
		if isinstance(entry, str):
			entry = dict(url=entry)
		if not entry.get("url"):
			raise FileNotFoundError
		try:
			entry.update(self.searched[entry["url"]][0])
		except KeyError:
			pass
		if video:
			stream = entry.get("video", None)
		else:
			stream = entry.get("stream", None)
		icon = entry.get("icon", None)
		# Use SHA-256 hash of URL to avoid filename conflicts
		url = entry["url"]
		url = unyt(url)
		h = shash(url)
		if type(download) is str:
			fn = "~" + h + download
		else:
			fn = "~" + h + ".opus"
		# Use cached file if one already exists
		if self.cache.get(fn) or not download:
			if video:
				entry["video"] = stream
			else:
				entry["stream"] = stream
			entry["icon"] = icon
			# Files may have a callback set for when they are loaded
			if callback is not None:
				esubmit(callback)
			f = self.cache.get(fn)
			if f is not None:
				# Assign file duration estimate to queue entry
				# This could be done better, this current implementation is technically not thread-safe
				try:
					if not os.path.exists("cache/" + f.file):
						raise KeyError
					entry["file"] = f
					if f.loaded:
						entry["duration"] = f.duration()
					else:
						f.assign.append(entry)
					# Touch file to indicate usage
					f.ensure_time()
					f.readable.result(timeout=16)
				except KeyError:
					f = None
			if f or (not force and not download):
				return f
		# "none" indicates stream is currently loading
		if stream == "none" and not force:
			return
		if video:
			entry["video"] = "none"
		else:
			entry["stream"] = "none"
		searched = False
		# If "research" tag is set, entry does not contain full data and requires another search
		if "research" in entry:
			try:
				self.extract_single(entry)
				searched = True
				entry.pop("research", None)
			except Exception as ex:
				print_exc()
				entry.pop("research", None)
				entry["url"] = ""
				entry["ex"] = repr(ex)
				raise
			else:
				if video:
					stream = entry.get("video")
				else:
					stream = entry.get("stream")
				icon = entry.get("icon", None)
		# If stream is still not found or is a soundcloud audio fragment playlist file, perform secondary youtube-dl search
		if stream in (None, "none"):
			data = self.search(entry["url"])
			if type(data) is str:
				try:
					data = evalEX(data)
				except Exception as ex:
					entry["url"] = ""
					entry["ex"] = repr(ex)
					raise
			stream = set_dict(data[0], "stream", data[0].url)
			icon = set_dict(data[0], "icon", data[0].url)
			entry.update(data[0])
		if not searched and (stream.startswith("ytsearch:") or stream.startswith("https://cf-hls-media.sndcdn.com/") or expired(stream)):
			data = self.extract(entry["url"])
			stream = set_dict(data[0], "stream", data[0].url)
			icon = set_dict(data[0], "icon", data[0].url)
			entry.update(data[0])
		# Otherwise attempt to start file download
		try:
			if stream.startswith("ytsearch:") or stream in (None, "none"):
				self.extract_single(entry, force=True)
				if video:
					stream = entry.get("video")
				else:
					stream = entry.get("stream")
				if stream in (None, "none"):
					raise FileNotFoundError("Unable to locate appropriate file stream.")
			if video:
				entry["video"] = stream
			else:
				entry["stream"] = stream
			entry["icon"] = icon
			if "googlevideo" in stream[:64]:
				durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
				if durstr:
					entry["duration"] = round_min(durstr[0])
			if not entry.get("duration"):
				entry["duration"] = get_duration(stream)
			# print(entry.url, entry.duration)
			if entry["url"] not in self.searched:
				self.searched[entry["url"]] = [cdict(entry)]
			else:
				self.searched[entry["url"]][0].update(entry)
			if not download:
				return entry
			self.cache[fn] = f = AudioFileLink(fn)
			if type(download) is str:
				live = False
			else:
				live = not entry.get("duration") or entry["duration"] > 960 or asap > 1
			seekable = not entry.get("duration") or entry["duration"] < inf
			cf = isnan(entry.get("duration") or nan) or not (stream.startswith("https://cf-hls-media.sndcdn.com/") or is_youtube_stream(stream))
			if asap > 1 and not f.proc and not live:
				esubmit(f.load, check_fmt=cf, webpage_url=entry["url"], live=False, seekable=seekable, duration=entry.get("duration"), asap=False)
			try:
				f.load(stream, check_fmt=cf, webpage_url=entry["url"], live=live, seekable=seekable, duration=entry.get("duration"), asap=asap)
			except:
				self.cache.pop(fn, None)
				raise
			# Assign file duration estimate to queue entry
			f.assign.append(entry)
			entry["file"] = f
			f.ensure_time()
			# Files may have a callback set for when they are loaded
			if callback is not None:
				asubmit(callback)
			return f
		except Exception as ex:
			# Remove entry URL if loading failed
			print_exc()
			entry["url"] = ""
			entry["ex"] = repr(ex)

	# Video concatenation algorithm; supports different formats, codecs, resolutions, aspect ratios and framerates
	def concat_video(self, urls, fmt, start, end, auds, message=None):
		urls = list(urls)
		ts = ts_us()
		# Collect information on first video stream; use this as the baseline for all other streams to concatenate
		url = urls[0]
		res = self.search(url)
		if type(res) is str:
			raise evalex(res)
		info = res[0]
		if not (info.get("video") and info["video"].startswith("https://www.yt-download.org/download/")):
			self.get_stream(info, video=True, force=True, download=False)
		video = info["video"]
		if "yt_live_broadcast" in info["stream"] and "force_finished" in info["stream"]:
			self.downloader.params["outtmpl"]["default"] = fn
			self.downloader.download(info["url"])
			if os.path.exists(fn):
				info["stream"] = fn
				info["video"] = fn
				video = fn
		elif video == info["stream"] and is_youtube_url(info["url"]) and has_ytd:
			data = self.extract_backup(info["url"], video=True)
			video = info["video"] = get_best_video(data)
		vidinfo = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1", video])).strip()
		codec, *size, fps = vidinfo.splitlines()
		size = [int(x) for x in size]
		w2, h2 = size
		# FFprobe returns fps as a fraction
		try:
			fps = eval(fps, {}, {})
		except:
			fps = 30
		# First produce a silent video file (I would have stored it as raw, except that overflows storage really bad)
		args = ["./ffmpeg", "-nostdin", "-hide_banner", "-hwaccel", hwaccel, "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-y"]
		if len(urls) != 1:
			if str(start) != "None":
				start = round_min(float(start))
				args.extend(("-ss", str(start)))
			if str(end) != "None":
				end = round_min(min(float(end), 86400))
				args.extend(("-to", str(end)))
		args.extend(("-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", "rgb24", "-video_size", "x".join(map(str, size)), "-i", "-", "-an", "-pix_fmt", "yuv420p", "-crf", "28"))
		afile = f"cache/-{ts}-.pcm"
		if hwaccel == "cuda":
			if fmt == "mp4":
				args.extend(("-c:v", "h264_nvenc"))
			elif fmt == "webm":
				args.extend(("-c:v", "av1_nvenc"))
		if len(urls) > 1:
			outf = f"{info['name']} +{len(urls) - 1}.{fmt}"
		else:
			outf = f"{info['name']}.{fmt}"
		fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
		fnv = f"cache/V{ts}~" + outf.translate(filetrans)
		args.append(fnv)
		print(args)
		proc = psutil.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
		with suppress():
			message.__dict__.setdefault("inits", []).append(proc)
		with open(afile, "wb") as afp:
			for t, url in enumerate(urls, ts + 1):
				with tracebacksuppressor:
					# Download and convert the raw audio as pcm in background
					if len(urls) == 1:
						if str(start) != "None":
							start = round_min(float(start))
						if str(end) != "None":
							end = round_min(min(float(end), 86400))
						fut = esubmit(self.download_file, url, "pcm", start=start, end=end, auds=None, ts=t, child=False, message=message)
					else:
						fut = esubmit(self.download_file, url, "pcm", auds=None, ts=t, child=True, message=message)
					res = self.search(url)
					if type(res) is str:
						raise evalex(res)
					info = res[0]
					if not (info.get("video") and info["video"].startswith("https://www.yt-download.org/download/")):
						self.get_stream(info, video=True, force=True, download=False)
					video = info["video"]
					if video == info["stream"] and is_youtube_url(info["url"]) and has_ytd:
						data = self.extract_backup(info["url"], video=True)
						video = info["video"] = get_best_video(data)
					vidinfo = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "default=nokey=1:noprint_wrappers=1", video])).strip()
					args = alist(("./ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240", "-nostdin", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-y"))
					if hwaccel == "cuda":
						if "av1_nvenc" in args:
							devid = random.choice([i for i in range(torch.cuda.device_count()) if (torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor) >= (8, 9)])
						else:
							devid = random.randint(0, ceil(torch.cuda.device_count() / 2))
						args.extend(("-hwaccel_device", str(devid)))
					if len(urls) == 1:
						if str(start) != "None":
							start = round_min(float(start))
							args.extend(("-ss", str(start)))
						if str(end) != "None":
							end = round_min(min(float(end), 86400))
							args.extend(("-to", str(end)))
					args.extend(("-i", video))
					# Tell FFmpeg to match fps/frame count as much as possible
					vf = f"fps={fps}"
					w1, h1 = map(int, vidinfo.splitlines())
					# If video needs resizing, keep aspect ratio while adding black padding as required
					if w1 != w2 or h1 != h2:
						r = min(w2 / w1, h2 / h1)
						w, h = round(w1 * r), round(h1 * r)
						vf += f",scale={w}:{h}"
						if w != w2 or h != h2:
							vf += f",pad=width={w2}:height={h2}:x=-1:y=-1:color=black"
					args.extend(("-vf", vf))
					# Pipe to main process, as raw video is extremely bloated and easily overflows hundreds of GB disk
					args.extend(("-f", "rawvideo", "-pix_fmt", "rgb24", "-"))
					print(args)
					pin = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1048576)
					with suppress():
						message.__dict__.setdefault("inits", []).append(pin)
					# Count amount of data for the raw video input, while piping to FFmpeg to encode
					fsize = 0
					while True:
						b = pin.stdout.read(1048576)
						if not b:
							break
						proc.stdin.write(b)
						fsize += len(b)
					# Calculate duration and exact amount of audio samples to use, minimising possibility of desyncs
					duration = fsize / np.prod(size) / 3 / fps
					amax = round_random(duration * SAMPLE_RATE) * 2 * 2
					cfn = fut.result()[0]
					# Write audio to the raw pcm as desired; trim if there is too much, pad with zeros if not enough
					asize = 0
					if os.path.getsize(cfn):
						with open(cfn, "rb") as f:
							while True:
								b = f.read(1048576)
								if not b:
									break
								if asize + len(b) > amax:
									b = b[:amax - asize]
									afp.write(b)
									asize += len(b)
									break
								afp.write(b)
								asize += len(b)
					print(asize, amax)
					while asize < amax:
						if amax - asize < len(self.emptybuff):
							buf = self.emptybuff[:amax - asize]
						else:
							buf = self.emptybuff
						afp.write(buf)
						asize += len(buf)
					# with suppress():
					#     os.remove(cfn)
		proc.stdin.close()
		proc.wait()
		# Add the audio to the rendered video, without re-encoding the entire frames
		args = ["./ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-y"]
		args.extend(("-i", fnv, "-f", "s16le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-i", afile))
		ac = "aac" if fmt == "mp4" else "libopus"
		args.extend(("-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", ac, "-b:a", "192k", fn))
		print(args)
		proc = psutil.Popen(args, stderr=subprocess.PIPE)
		with suppress():
			message.__dict__.setdefault("inits", []).append(proc)
		proc.wait()
		# with suppress():
			# os.remove(fnv)
			# os.remove(afile)
		return fn, outf

	emptybuff = b"\x00" * (48000 * 2 * 2)
	# codec_map = {}
	# For ~download
	def download_file(self, url, fmt, start=None, end=None, auds=None, ts=None, copy=False, ar=SAMPLE_RATE, ac=2, size=None, container=None, child=False, silenceremove=False, message=None, rename=None):
		ffmpeg = "./ffmpeg"
		if child:
			ctx = emptyctx
		else:
			ctx = self.download_sem
		ofmt = fmt
		with ctx:
			if fmt == "ecdc":
				fmt = "opus"
				ecdc = True
			else:
				ecdc = False
			# Select a filename based on current time to avoid conflicts
			if fmt[:3] == "mid":
				mid = True
				fmt = "mp3"
			else:
				mid = False
			videos = {"webm", "mkv", "f4v", "flv", "mov", "qt", "wmv", "mp4", "m4v", "mpv", "gif", "apng", "webp"}
			vid = fmt in videos or container and container in videos
			if type(url) is str:
				urls = (url,)
			else:
				urls = url
			if vid and any(is_youtube_url(url) for url in urls):
				return self.concat_video(urls, fmt, start, end, auds)
			vst = deque()
			ast = deque()
			if not ts:
				ts = ts_us()
			if rename and os.path.exists(rename) and os.path.getsize(rename):
				return rename, rename
			enough = False
			if len(urls) == 1 and fmt in ("opus", "pcm", "wav", "mp3", "ogg") and is_youtube_url(urls[0]):
				url = unyt(url)
				info = self.search(url)
				if info and info[0] and info[0].get("duration") and info[0]["duration"] < 3600:
					enough = True
			if enough:
				h = shash(url)
				fn = "cache/~" + h + ".webm"
				out2 = "cache/~" + h + ".opus"
				if not os.path.exists(fn):
					b = await_fut(process_image("ytdl", "$", [urls[0], True], cap="ytdl", timeout=600))
					if not os.path.exists(fn) or not os.path.getsize(fn):
						with open(fn, "wb") as f:
							f.write(b)
				h2 = shash(url + ("~S" * silenceremove))
				out = "cache/~" + h2 + "." + fmt
				args = ()
				if not os.path.exists(out) or not os.path.getsize(out):
					args = [ffmpeg, "-hide_banner", "-v", "error", "-y", "-vn", "-i", fn, "-map_metadata", "-1"]
					if is_url(fn):
						args = [ffmpeg, "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"] + args[1:]
					if silenceremove:
						args.extend(("-af", "silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:start_silence=0.5:stop_periods=-9000:stop_threshold=-50dB:window=0.015625"))
						if fmt == "mp3":
							args.extend(("-b:a", "196608", out))
						elif fmt == "ogg":
							args.extend(("-b:a", "196608", "-c:a", "libopus", out))
						elif fmt == "wav":
							args.extend(("-ar", str(SAMPLE_RATE), "-ac", "2", out))
						elif fmt == "pcm":
							args.extend(("-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", out))
						else:
							args.extend(("-b:a", "196608", "-c:a", "libopus", out))
					else:
						if fmt == "mp3":
							args.extend(("-b:a", "196608", out))
						elif fmt == "ogg":
							args.extend(("-c:a", "copy", out))
						elif fmt == "wav":
							args.extend(("-ar", str(SAMPLE_RATE), "-ac", "2", out))
						elif fmt == "pcm":
							args.extend(("-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", out))
						else:
							args = None
				if args:
					print(args)
					subprocess.run(args)
				if not os.path.exists(out2) or not os.path.getsize(out2):
					args = [ffmpeg, "-hide_banner", "-v", "error", "-y", "-vn", "-i", fn, "-c:a", "copy", out2]
					print(args)
					subprocess.run(args)
				if rename:
					if os.path.exists(rename):
						return rename, rename
					os.rename(out, rename)
					return rename, rename
				file = out2.removeprefix("cache/")
				self.cache[file] = AudioFileLink(file, out2, wasfile=True)
				if ecdc:
					try:
						dur, bps = get_duration_simple(out)
					except:
						print_exc()
						bps = 196608
					if not bps:
						br = 24
					elif bps < 48000:
						br = 6
					elif bps < 96000:
						br = 12
					else:
						br = 24
						print("BPS:", bps)
					if not os.path.exists(cachedir + "/ecdc"):
						os.mkdir(cachedir + "/ecdc")
					out3 = cachedir + "/ecdc/!" + h + "~" + str(br) + ".ecdc"
					res = self.search(url)
					if type(res) is str:
						raise evalex(res)
					info = res[0]
					name = info.get("name") or "untitled"
					if not os.path.exists(out3) or not os.path.getsize(out3):
						with open(out, "rb") as f:
							b = f.read()
						fn = ecdc_encode(b, br, name, url)
						if os.path.exists(out3):
							os.remove(out3)
						try:
							os.rename(fn, out3)
						except OSError:
							with open(fn, "rb") as f:
								b = f.read()
							with open(out3, "wb") as f:
								f.write(b)
					return out3, name.translate(filetrans) + ".ecdc"
				return out, out
			outf = None
			for url in urls:
				if len(ast) > 1 and not vst:
					ast.append(url)
					continue
				try:
					res = self.search(url)
					if type(res) is str:
						raise evalex(res)
					info = res[0]
				except:
					print(url)
					print_exc()
					continue
				if not outf:
					outf = f"{info['name']}.{fmt}"
					outft = outf.translate(filetrans)
					if child:
						fn = f"cache/C{ts}~{outft}"
					else:
						fn = f"cache/\x7f{ts}~{outft}"
				if not (vid and info.get("video") and info["video"].startswith("https://www.yt-download.org/download/")):
					self.get_stream(info, video=vid, force=True, download=False)
				if "yt_live_broadcast" in info["stream"] and "force_finished" in info["stream"]:
					self.downloader.params["outtmpl"]["default"] = fn
					self.downloader.download(info["url"])
					if os.path.exists(fn):
						info["stream"] = fn
						info["video"] = fn
						if vst or vid:
							vst.append(fn)
						ast.append(fn)
						continue
				if vst or vid:
					video = info["video"]
					if video == info["stream"] and is_youtube_url(info["url"]) and has_ytd:
						data = self.extract_backup(info["url"], video=True)
						video = info["video"] = get_best_video(data)
						try:
							c = len(urls)
						except:
							c = 0
						if c == 1:
							ft = f"cache/-{ts}-.mp4"
							with requests.get(video, stream=True) as resp:
								with open(ft, "wb") as f:
									it = resp.iter_content(262144)
									try:
										while True:
											b = next(it)
											if not b:
												break
											f.write(b)
									except StopIteration:
										pass
							video = ft
							info["stream"] = video
					vst.append(video)
				ast.append(info)
			if not ast and not vst:
				raise LookupError(f"No stream URLs found for {url}")
			if len(ast) <= 1 and not vst and fmt != "pcm":
				if ast:
					# if not is_youtube_stream(ast[0]["stream"]):
					#     ffmpeg = "misc/ffmpeg-c/ffmpeg.exe"
					#     if not os.path.exists(ffmpeg):
					#         ffmpeg = "./ffmpeg"
					cdc = codec = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=format_name", "-of", "default=nokey=1:noprint_wrappers=1", ast[0]["stream"]])).strip()
					if fmt in cdc.split(","):
						copy = True
			else:
				copy = False
			args = alist((ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-hwaccel", hwaccel, "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-y", "-protocol_whitelist", "file,fd,http,https,tcp,tls"))
			if is_url(url):
				args = alist([ffmpeg, "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"]).concat(args[1:])
			if hwaccel == "cuda":
				if "av1_nvenc" in args:
					devid = random.choice([i for i in range(torch.cuda.device_count()) if (torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor) >= (8, 9)])
				else:
					devid = random.randint(0, ceil(torch.cuda.device_count() / 2))
				args.extend(("-hwaccel_device", str(devid)))
			if vst:
				if len(vst) > 1:
					codec_map = {}
					codecs = {}
					for url in vst:
						try:
							codec = codec_map[url]
						except KeyError:
							codec = as_str(subprocess.check_output(["./ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "default=nokey=1:noprint_wrappers=1", url])).strip()
							print(codec)
							codec_map[url] = codec
						add_dict(codecs, {codec: 1})
					if len(codecs) > 1:
						maxcodec = max(codecs.values())
						selcodec = [k for k, v in codecs.items() if v >= maxcodec][0]
						t = ts
						for i, url in enumerate(vst):
							if codec_map[url] != selcodec:
								selc, width, height = selcodec.splitlines()
								t += 1
								s2, w2, h2 = codec_map[url].splitlines()
								if selc == "av1" or selc.startswith("vp"):
									container = "webm"
								elif selc.startswith("h26"):
									container = "mp4"
								vst[i] = self.download_file(url, selc, size=((w2, h2), (width, height)), auds=auds, ts=t, container=container, message=message)[0].rsplit("/", 1)[-1]
					vsc = "\n".join(f"file '{i}'" for i in vst)
					vsf = f"cache/{ts}~video.concat"
					with open(vsf, "w", encoding="utf-8") as f:
						f.write(vsc)
					args.extend(("-f", "concat"))
				else:
					vsf = vsc = vst[0]
			if len(ast) > 1:
				asf = "-"
			else:
				stream = ast[0]["stream"]
				# if child and not stream.startswith("https://cf-hls-media.sndcdn.com/"):
				#     fii = f"cache/{ts}~proxy"
				#     with tracebacksuppressor:
				#         stream = proxy_download(stream, fii, timeout=86400)
				asf = asc = stream
			br = CustomAudio.max_bitrate
			if auds and br > auds.stats.bitrate:
				br = max(4096, auds.stats.bitrate)
			sr = str(SAMPLE_RATE)
			ac = "2"
			if not vst and not size:
				args.append("-vn")
			elif fmt in ("gif", "apng", "webp"):
				args.append("-an")
			if str(start) != "None":
				start = round_min(float(start))
				args.extend(("-ss", str(start)))
			else:
				start = 0
			if str(end) != "None":
				end = round_min(min(float(end), 86400))
				args.extend(("-to", str(end)))
			else:
				end = None
				if len(ast) == 1:
					args.extend(("-to", "604800"))
			if vst and vsf != asf:
				args.extend(("-i", vsf))
				if start:
					args.extend(("-ss", str(start)))
				if end is not None:
					args.extend(("-to", str(end)))
			if asf == "-":
				args.extend(("-f", "s16le"))
			if not copy and len(ast) > 1:
				args.extend(("-ar", sr, "-ac", ac))
				if vst:
					args.extend(("-pix_fmt", "yuv420p", "-crf", "28"))
			args.extend(("-i", asf, "-map_metadata", "-1"))
			if auds:
				args.extend(auds.construct_options(full=True))
			if silenceremove and len(ast) == 1 and not vid:
				args.extend(("-af", "silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:start_silence=0.5:stop_periods=-9000:stop_threshold=-50dB:window=0.015625"))
			if size:
				w1, h1 = map(int, size[0])
				w2, h2 = map(int, size[1])
				r = min(w2 / w1, h2 / h1)
				w, h = round(w1 * r), round(h1 * r)
				vf = f"scale={w}:{h}"
				if w != w2 or h != h2:
					vf += f",pad=width={w2}:height={h2}:x=-1:y=-1:color=black"
				args.extend(("-vf", vf))
			if fmt in ("vox", "adpcm"):
				args.extend(("-c:a", "adpcm_ms"))
				fmt = "wav" if fmt == "adpcm" else "vox"
				outf = f"{info['name']}.{fmt}"
				fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
			elif fmt == "ogg":
				args.extend(("-c:a", "libopus"))
			elif fmt == "weba":
				fmt = "webm"
				args.extend(("-c:a", "libopus"))
				outf = f"{info['name']}.{fmt}"
				fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
			elif fmt == "pcm":
				fmt = "s16le"
			elif fmt == "mp2":
				br = round(br / 64000) * 64000
				if not br:
					br = 64000
			elif fmt in ("aac", "m4a"):
				fmt = "adts"
			elif fmt == "8bit":
				container = "wav"
				fmt = "pcm_u8"
				sr = "24k"
				ac = "1"
				br = "256"
				outf = f"{info['name']}.wav"
				fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
			elif fmt == "mkv":
				fmt = "matroska"
			if not copy and ast:
				args.extend(("-b:a", str(br)))
				if len(ast) == 1:
					args.extend(("-ar", sr, "-ac", ac, "-b:a", str(br)))
					if vst:
						args.extend(("-pix_fmt", "yuv420p", "-crf", "28"))
			if copy:
				args.extend(("-c", "copy", fn))
			elif container:
				outf = f"{info['name']}.{container}"
				fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
				c = "-c:v" if size else "-c"
				if container == "mkv":
					container = "matroska"
				if hwaccel == "cuda":
					if fmt == "mp4":
						fmt = "h264_nvenc"
					elif fmt == "webm":
						fmt = "av1_nvenc"
				args.extend(("-f", container, c, fmt, "-strict", "-2", fn))
			else:
				if hwaccel == "cuda":
					if fmt == "mp4":
						args.extend(("-c:v", "h264_nvenc"))
					elif fmt == "webm":
						args.extend(("-c:v", "av1_nvenc"))
				args.extend(("-f", fmt, fn))
			try:
				if len(ast) > 1:
					ress = []
					futs = []
					for i, info in enumerate(ast):
						t = i + ts + 1
						cfn = None
						if type(info) is not str:
							url = info.get("url")
						else:
							url = info
						try:
							concurrent = len(self.bot.status_data["system"]["cpu"]) * 8
						except:
							print_exc()
							concurrent = 8
						while len(futs) >= concurrent:
							with tracebacksuppressor:
								cfn = futs.pop(0).result(timeout=600)[0]
								print(cfn)
								ress.append(cfn)
						fut = esubmit(self.download_file, url, ofmt, auds=None, ts=t, child=True, silenceremove=silenceremove, message=message, timeout=720)
						futs.append(fut)
					for fut in futs:
						with tracebacksuppressor:
							cfn = fut.result(timeout=600)[0]
							print(cfn)
							ress.append(cfn)
					concf = f"cache/{ts}~concat.txt"
					with open(concf, "w", encoding="utf-8") as f:
						for cfn in ress:
							f.write(f"file '{cfn.split('/', 1)[-1]}'\n")
					args = [
						"./ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
						"-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets",
						"-protocol_whitelist", "concat,tls,tcp,file,fd,http,https",
						"-f", "concat", "-safe", "0", "-vn",
						"-i", concf, "-c:a", "copy", fn,
					]
					print(args)
					proc = psutil.Popen(args)
					with suppress():
						message.__dict__.setdefault("inits", []).append(proc)
					proc.wait()
				else:
					print(args)
					proc = psutil.Popen(args, stderr=subprocess.PIPE)
					proc.wait()
			except subprocess.CalledProcessError as ex:
				# Attempt to convert file from org if FFmpeg failed
				try:
					url = ast[0]
					if type(url) is not str:
						url = url["url"]
					if is_youtube_url(url) or is_youtube_stream(url) or vid:
						raise ex
					new = select_and_convert(url)
				except ValueError:
					if resp.stderr:
						raise RuntimeError(*ex.args, resp.stderr)
					raise ex
				# Re-estimate duration if file was successfully converted from org
				args[args.index("-i") + 1] = new
				try:
					resp = subprocess.run(args, stderr=subprocess.PIPE)
					resp.check_returncode()
				except subprocess.CalledProcessError as ex:
					if resp.stderr:
						raise RuntimeError(*ex.args, resp.stderr)
					raise ex
				if not is_url(new):
					with suppress():
						os.remove(new)
			if end:
				odur = end - start
				if odur:
					dur = e_dur(get_duration(fn))
					if dur < odur - 1:
						ts += 1
						fn, fn2 = f"cache/\x7f{ts}~{outft}", fn
						times = ceil(odur / dur)
						loopf = f"cache/{ts - 1}~loop.txt"
						with open(loopf, "w", encoding="utf-8") as f:
							f.write(f"file '{fn2.split('/', 1)[-1]}'\n" * times)
						args = [
							"./ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
							"-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets",
							"-protocol_whitelist", "concat,tls,tcp,file,fd,http,https",
							"-to", str(odur), "-f", "concat", "-safe", "0",
							"-i", loopf, "-c", "copy", fn,
						]
						print(args)
						try:
							resp = subprocess.run(args)
							resp.check_returncode()
						except subprocess.CalledProcessError as ex:
							if resp.stderr:
								raise RuntimeError(*ex.args, resp.stderr)
							raise ex
						with suppress():
							if loopf:
								os.remove(loopf)
							os.remove(fn2)
			if not mid:
				assert os.path.exists(fn) and os.path.getsize(fn)
				if ecdc:
					out3 = fn.rsplit(".", 1)[0] + ".ecdc"
					outf = outf.rsplit(".", 1)[0] + ".ecdc"
					if not os.path.exists(out3) or not os.path.getsize(out3):
						with open(fn, "rb") as f:
							b = f.read()
						try:
							dur, bps = get_duration_simple(fn)
						except:
							print_exc()
							bps = 196608
						if not bps:
							br = 24
						elif bps < 48000:
							br = 6
						elif bps < 96000:
							br = 12
						else:
							br = 24
							print("BPS:", bps)
						url = urls[0]
						res = self.search(url)
						if type(res) is str:
							raise evalex(res)
						info = res[0]
						name = info.get("name")
						fn = ecdc_encode(b, br, name, url)
						try:
							os.rename(fn, out3)
						except OSError:
							with open(fn, "rb") as f:
								b = f.read()
							with open(out3, "wb") as f:
								f.write(b)
					fn = out3
				if rename:
					os.rename(fn, rename)
					return rename, outf
				return fn, outf
			self.other_x += 1
			with open(fn, "rb") as f:
				resp = Request(
					"https://cts.ofoct.com/upload.php",
					method="post",
					files={"myfile": ("temp.mp3", f)},
					timeout=32,
					decode=True
				)
				resp_fn = literal_eval(resp)[0]
			url = f"https://cts.ofoct.com/convert-file_v2.php?cid=audio2midi&output=MID&tmpfpath={resp_fn}&row=file1&sourcename=temp.ogg&rowid=file1"
			# print(url)
			with suppress():
				os.remove(fn)
			self.other_x += 1
			resp = Request(url, timeout=720)
			self.other_x += 1
			out = Request(f"https://cts.ofoct.com/get-file.php?type=get&genfpath=/tmp/{resp_fn}.mid", timeout=32)
			return out, outf[:-4] + ".mid"

	# Extracts full data for a single entry. Uses cached results for optimization.
	def extract_single(self, i, force=False):
		item = i.url
		if not force:
			if item in self.searched and not item.startswith("ytsearch:"):
				it = self.searched[item][0]
				i.update(it)
				if i.get("stream") not in (None, "none"):
					return True
		with self.semaphore:
			try:
				data = self.extract_true(item)
				if "entries" in data:
					data = data["entries"][0]
				elif not issubclass(type(data), collections.abc.Mapping):
					data = data[0]
				if data.get("research"):
					data = self.extract_true(data["url"])[0]
				out = [cdict(
					name=data.get("title") or data.get("name"),
					url=data.get("webpage_url") or data.get("url"),
					stream=data.get("stream") or get_best_audio(data),
					icon=data.get("icon") or get_best_icon(data),
					video=data.get("video") or get_best_video(data),
				)]
				try:
					out[0].duration = data["duration"]
				except KeyError:
					pass
				self.searched[item] = out
				it = out[0]
				i.update(it)
			except Exception as ex:
				i.url = ""
				i.ex = repr(ex)
				print_exc(item)
				return False
		return True

ytdl = AudioDownloader()


class Queue(Command):
	server_only = True
	name = ["▶️", "P", "Q", "Play", "Enqueue", "Search&Play"]
	alias = name + ["LS"]
	description = "Shows the music queue, or plays a song in voice."
	usage = "<url>* <force{?f}|budge{?b}|random{?r}|verbose{?v}|hide{?h}>*"
	example = ("play despacito", "queue", "enqueue https://www.youtube.com/watch?v=dQw4w9WgXcQ")
	flags = "hvfbrz"
	no_parse = True
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	_timeout_ = 2
	rate_limit = (3.5, 5)
	typing = True
	slash = ("Play", "Queue")
	msgcmd = ("Search & Play",)

	async def __call__(self, bot, user, perm, message, channel, guild, flags, name, argv, comment="", **void):
		# This command is a bit of a mess
		ytdl.bot = bot
		argv += " ".join(best_url(a) for a in message.attachments)
		if not argv:
			auds = await auto_join(guild, channel, user, bot, ignore=True)
			q = auds.queue
			v = "v" in flags
			if not v and len(q) and auds.paused & 1 and "p" in name:
				auds.resume()
				esubmit(auds.queue.update_load, timeout=120)
				return css_md(f"Successfully resumed audio playback in {sqr_md(guild)}."), 1
			if not len(q):
				auds.preparing = False
				esubmit(auds.update, timeout=180)
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-voice-queue-"
				+ str(user.id) + "_0_" + str(int(v))
				+ "-\nLoading Queue...```*",
				buttons=buttons,
			)
			return
		# Get audio player as fast as possible, scheduling it to join asynchronously if necessary
		try:
			auds = bot.data.audio.players[guild.id]
			auds.channel = channel
			future = None
		except KeyError:
			future = create_task(auto_join(guild, channel, user, bot, preparing=True))
		# Start typing event asynchronously to avoid delays
		async with discord.context_managers.Typing(channel):
			# Perform search concurrently, may contain multiple URLs
			out = None
			urls = await bot.follow_url(argv, allow=True, images=False)
			if urls:
				if len(urls) == 1:
					argv = urls[0]
					if is_discord_message_link(argv):
						spl = argv.rsplit("/", 2)
						channel = await bot.fetch_channel(spl[-2])
						msg = await bot.fetch_message(spl[-1], channel)
						argv = msg.content
				else:
					out = [asubmit(ytdl.search, url) for url in urls]
			if out is None:
				resp = await asubmit(ytdl.search, argv, timeout=180)
			else:
				resp = deque()
				for fut in out:
					temp = await fut
					# Ignore errors when searching with multiple URLs
					if type(temp) not in (str, bytes):
						resp.extend(temp)
			# Wait for audio player to finish loading if necessary
			if future is not None:
				auds = await future
		if "f" in flags or "b" in flags:
			if not is_alone(auds, user) and perm < 1:
				raise self.perm_error(perm, 1, "to force play while other users are in voice")
		if auds.stats.quiet & 2:
			set_dict(flags, "h", 1)
		elapsed, length = auds.epos
		q = auds.queue
		# Raise exceptions returned by searches
		if type(resp) is str:
			raise evalEX(resp)
		if "r" in flags:
			resp = (choice(resp),)
		# Assign search results to queue entries
		added = deque()
		names = []
		for i, e in enumerate(resp, 1):
			if i > 262144:
				break
			try:
				name = e["name"]
			except:
				print(e)
				raise
			url = e["url"]
			temp = {
				# "hash": e.hash,
				"name": name,
				"url": url,
				"duration": e.get("duration"),
				"u_id": user.id,
				"skips": deque(),
			}
			if "research" in e:
				temp["research"] = True
			added.append(cdict(temp))
			names.append(no_md(name))
			if not i & 16383:
				await asyncio.sleep(0.2)
		# Prepare to enqueue entries
		if "b" not in flags:
			total_duration = 0
			for e in q:
				total_duration += e_dur(e.duration)
			if auds.reverse and auds.queue:
				total_duration += elapsed - length
			else:
				total_duration -= elapsed
		if auds.stats.shuffle:
			added = shuffle(added)
		tdur = 3
		if "f" in flags:
			# Force play moves currently playing item, which means we have to unset the "played" tag from currently playing entries to allow them to play again
			auds.queue.enqueue(added, 0)
			total_duration = tdur
		elif "b" in flags:
			auds.queue.enqueue(added, 1)
			total_duration = max(3, length - elapsed if q else 0)
		else:
			auds.queue.enqueue(added, -1)
			total_duration = max(total_duration / auds.speed, tdur)
		if not names:
			raise LookupError(f"No results for {argv}.")
		if "v" in flags:
			names = no_md(alist(i.name + ": " + time_disp(e_dur(i.duration)) for i in added))
		elif len(names) == 1:
			names = names[0]
		else:
			names = f"{len(names)} items"
		if "h" not in flags:
			if auds.paused:
				p = f"\nNote: Player is currently paused. Use {bot.get_prefix(guild)}resume to resume!"
			else:
				p = ""
			s = comment + "\n" + css_md(f"🎶 Added {sqr_md(names)} to the queue! Estimated time until playing: {sqr_md(time_until(utc() + total_duration))}. 🎶{p}")
			return await send_with_react(channel, s, reference=message, reacts="❎")

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos, v = list(map(int, vals.split("_", 2)))
		if reaction and u_id != user.id and perm < 1:
			return
		if reaction not in self.directions and reaction is not None:
			return
		user = await bot.fetch_user(u_id)
		guild = message.guild
		auds = await auto_join(guild, None, user, bot, ignore=True)
		q = auds.queue
		last = max(0, len(q) - 10)
		if reaction is not None:
			i = self.directions.index(reaction)
			if i == 0:
				new = 0
			elif i == 1:
				new = max(0, pos - 10)
			elif i == 2:
				new = min(last, pos + 10)
			elif i == 3:
				new = last
			else:
				new = pos
			pos = new
		content = message.content
		if not content:
			content = message.embeds[0].description
		i = content.index("callback")
		content = "*```" + "\n" * ("\n" in content[:i]) + (
			"callback-voice-queue-"
			+ str(u_id) + "_" + str(pos) + "_" + str(int(v))
			+ "-\nQueue for " + guild.name.replace("`", "") + ": "
		)
		elapsed, length = auds.epos
		startTime = 0
		if not q:
			stime = "0"
		elif auds.stats.loop:
			stime = "inf (loop)"
		elif auds.stats.repeat:
			stime = "inf (repeat)"
		elif auds.paused:
			stime = "inf (paused)"
		else:
			if auds.reverse and q:
				totalTime = elapsed - length
			else:
				totalTime = -elapsed
			i = 0
			for e in q:
				totalTime += e_dur(e.duration)
				if i < pos:
					startTime += e_dur(e.duration)
				if not 1 + i & 32767:
					await asyncio.sleep(0.1)
				i += 1
			stime = time_until(utc() + totalTime / auds.speed)
		cnt = len(q)
		info = (
			str(cnt) + " item" + "s" * (cnt != 1) + "\nEstimated total duration: "
			+ stime + "```*"
		)
		if not q:
			duration = 0
		else:
			duration = length
		if duration == 0:
			elapsed = 0
			duration = 0.0001
		bar = await bot.create_progress_bar(18, elapsed / duration)
		if not q:
			countstr = "Queue is currently empty.\n"
		else:
			countstr = f'{"[`" + no_md(q[0].name) + "`]"}({q[0].url})'
		countstr += f"` ({uni_str(time_disp(elapsed))}/{uni_str(time_disp(duration))})`\n{bar}\n"
		emb = discord.Embed(
			description=content + info + countstr,
			colour=rand_colour(),
		)
		emb.set_author(**get_author(user))
		if q:
			icon = get_best_icon(q[0])
		else:
			icon = ""
		if icon:
			emb.set_thumbnail(url=icon)
		async with auds.semaphore:
			embstr = ""
			currTime = startTime
			i = pos
			maxlen = 40 if icon else 48
			maxlen = maxlen - int(math.log10(len(q))) if q else maxlen
			while i < min(pos + 10, len(q)):
				e = q[i]
				space = int(math.log10(len(q))) - int(math.log10(max(1, i)))
				curr = "`" + " " * space
				ename = no_md(e.name)
				curr += f'【{i}】`{"[`" + no_md(lim_str(ename + " " * (maxlen - len(ename)), maxlen)) + "`]"}({ensure_url(e.url)})` ({time_disp(e_dur(e.duration))})`'
				if v:
					try:
						u = bot.cache.users[e.u_id]
						name = u.display_name
					except KeyError:
						name = "Deleted User"
						with suppress():
							u = await bot.fetch_user(e.u_id)
							name = u.display_name
					curr += "\n" + css_md(sqr_md(name))
				if auds.reverse and len(auds.queue):
					estim = currTime + elapsed - length
				else:
					estim = currTime - elapsed
				if v:
					if estim > 0:
						curr += "Time until playing: "
						estimate = time_until(utc() + estim / auds.speed)
						if i <= 1 or not auds.stats.shuffle:
							curr += "[" + estimate + "]"
						else:
							curr += "{" + estimate + "}"
					else:
						curr += "Remaining time: [" + time_until(utc() + (estim + e_dur(e.duration)) / auds.speed) + "]"
					curr += "```"
				curr += "\n"
				if len(embstr) + len(curr) > 4096 - len(emb.description):
					break
				embstr += curr
				if i <= 1 or not auds.stats.shuffle:
					currTime += e_dur(e.duration)
				if not 1 + 1 & 4095:
					await asyncio.sleep(0.3)
				i += 1
		emb.description += embstr
		more = len(q) - i
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Playlist(Command):
	server_only = True
	name = ["DefaultPlaylist", "PL"]
	min_display = "0~2"
	description = "Shows, appends, or removes from the default playlist."
	usage = "(add|remove)? <search_links>*"
	example = ("playlist add https://www.youtube.com/watch?v=wDgQdr8ZkTw", "playlist remove 6")
	flags = "aedzf"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (7, 11)
	typing = True
	slash = True

	async def __call__(self, user, argv, guild, flags, channel, message, perm, **void):
		update = self.bot.data.playlists.update
		bot = self.bot
		if argv or "d" in flags:
			req = 2
			if perm < req:
				reason = f"to modify default playlist for {guild.name}"
				raise self.perm_error(perm, req, reason)
		pl = set_dict(bot.data.playlists, guild.id, [])
		if not argv:
			if "d" in flags:
				# This deletes all default playlist entries for the current guild
				if "f" not in flags and len(pl) > 1:
					raise InterruptedError(css_md(sqr_md(f"WARNING: {len(pl)} ENTRIES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True))
				bot.data.playlists.pop(guild.id)
				return italics(css_md(f"Successfully removed all {sqr_md(len(pl))} entries from the default playlist for {sqr_md(guild)}."))
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-voice-playlist-"
				+ str(user.id) + "_0"
				+ "-\nLoading Playlist database...```*",
				buttons=buttons,
			)
			return
		if "d" in flags:
			# Can only remove by index atm
			i = await bot.eval_math(argv)
			temp = pl[i]
			pl.pop(i)
			update(guild.id)
			return italics(css_md(f"Removed {sqr_md(temp.name)} from the default playlist for {sqr_md(guild)}."))
		lim = 8 << self.bot.is_trusted(guild.id) * 2 + 1
		if len(pl) >= lim:
			raise OverflowError(f"Playlist for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
		urls = await bot.follow_url(argv, allow=True, images=False)
		if urls:
			argv = urls[0]
		async with discord.context_managers.Typing(channel):
		# Unlike ~queue this command only supports a single URL/search
			resp = await asubmit(ytdl.search, argv, timeout=180)
		if type(resp) is str:
			raise evalEX(resp)
		# Assign search results to default playlist entries
		names = []
		for e in resp:
			name = e.name
			names.append(no_md(name))
			pl.append(cdict(
				name=name,
				url=e.url,
				duration=e.duration,
			))
		if not names:
			raise LookupError(f"No results for {argv}.")
		pl.sort(key=lambda x: x["name"].casefold())
		update(guild.id)
		stuff = str(len(names)) + " items" if len(names) > 3 else ', '.join(names)
		return css_md(f"Added {sqr_md(stuff)} to the default playlist for {sqr_md(guild)}.")

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		pl = bot.data.playlists.get(guild.id, [])
		page = 12
		last = max(0, len(pl) - page)
		if reaction is not None:
			i = self.directions.index(reaction)
			if i == 0:
				new = 0
			elif i == 1:
				new = max(0, pos - page)
			elif i == 2:
				new = min(last, pos + page)
			elif i == 3:
				new = last
			else:
				new = pos
			pos = new
		content = message.content
		if not content:
			content = message.embeds[0].description
		i = content.index("callback")
		content = "*```" + "\n" * ("\n" in content[:i]) + (
			"callback-voice-playlist-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not pl:
			content += f"No currently enabled default playlist for {str(guild).replace('`', '')}.```*"
			msg = ""
		else:
			pl.sort(key=lambda x: x["name"].casefold())
			content += f"{len(pl)} item(s) in default playlist for {str(guild).replace('`', '')}:```*"
			key = lambda x: lim_str(sqr_md(x["name"]) + "(" + x["url"] + ")", 1900 / page)
			msg = iter2str(pl[pos:pos + page], key=key, offset=pos, left="`【", right="】`")
		colour = await self.bot.get_colour(guild)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		)
		emb.set_author(**get_author(user))
		more = len(pl) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Connect(Command):
	server_only = True
	name = ["📲", "🎤", "🎵", "🎶", "📴", "📛", "Summon", "J", "Join", "DC", "Disconnect", "Leave", "Yeet", "Move", "Reconnect"]
	# Because Rythm also has this alias :P
	alias = name + ["FuckOff"]
	description = "Summons the bot into a voice channel, or advises it to leave."
	example = ("join", "connect 247184721262411777", "leave")
	usage = "<channel>?"
	rate_limit = (3, 4)
	slash = ("Join", "Leave")

	async def __call__(self, user, channel, name="join", argv="", vc=None, message=None, **void):
		bot = self.bot
		joining = False
		if name in ("dc", "disconnect", "leave", "yeet", "fuckoff", "📴", "📛"):
			vc_ = None
		elif argv or name == "move":
			c_id = verify_id(argv)
			if not isinstance(c_id, int):
				raise TypeError(f"Invalid channel identifier: {c_id}")
			if not c_id > 0:
				vc_ = None
			else:
				vc_ = await bot.fetch_channel(c_id)
		else:
			# If voice channel is already selected, use that
			if vc is not None:
				vc_ = vc
			else:
				vc_ = select_voice_channel(user, channel)
		# target guild may be different from source guild
		if vc_ is None:
			guild = channel.guild
		else:
			guild = vc_.guild
		# Use permission level in target guild to make sure user is able to perform command
		perm = bot.get_perms(user, guild)
		if perm < 0:
			raise self.perm_error(perm, 0, f"for command {self.name} in {guild}")
		# If no voice channel is selected, perform disconnect
		if vc_ is None:
			if argv:
				if perm < 2:
					raise self.perm_error(perm, 2, f"for command {self.name} in {guild}")
				u_id = verify_id(argv)
				try:
					t_user = await bot.fetch_user_member(u_id, guild)
				except (LookupError):
					t_role = guild.get_role(u_id)
					if t_role is None:
						raise LookupError(f"No results for {u_id}.")
					members = [member for member in t_role.members if member.voice is not None]
					if not members:
						return code_md("No members to disconnect.")
					await disconnect_members(bot, guild, members)
					if len(members) == 1:
						return ini_md(f"Disconnected {sqr_md(members[0])} from {sqr_md(guild)}."), 1
					return ini_md(f"Disconnected {sqr_md(str(members) + ' members')} from {sqr_md(guild)}."), 1
				member = guild.get_member(t_user.id)
				if not member or member.voice is None:
					return code_md("No members to disconnect.")
				await disconnect_members(bot, guild, (member,))
				return ini_md(f"Disconnected {sqr_md(member)} from {sqr_md(guild)}."), 1
			try:
				auds = bot.data.audio.players[guild.id]
			except KeyError:
				raise LookupError("Unable to find voice channel.")
			auds.text = channel
			if not is_alone(auds, user) and perm < 1:
				raise self.perm_error(perm, 1, "to disconnect while other users are in voice")
			return await asubmit(auds.kill, initiator=message)
		if not vc_.permissions_for(guild.me).connect:
			raise ConnectionError("Insufficient permissions to connect to voice channel.")
		# if vc_.permissions_for(guild.me).manage_channels:
			# if guild.id in bot.data.audio.players:
				# br = round(bot.data.audio.players[guild.id].stats.bitrate * 100)
			# else:
				# br = CustomAudio.max_bitrate
			# if vc_.type is discord.ChannelType.stage_voice:
				# bitrate = min(br, 64000)
			# else:
				# bitrate = min(br, guild.bitrate_limit)
			# if vc_.bitrate < bitrate:
				# try:
					# await vc_.edit(bitrate=bitrate, reason="I deliver maximum quality audio only! :3")
				# except:
					# print_exc()
		# Create audio source if none already exists
		if guild.id not in bot.data.audio.players:
			globals()["bot"] = bot
			auds = CustomAudio(channel)
			auds.join(vc_)
			joining = True
		else:
			auds = bot.data.audio.players[guild.id]
			if not auds.acsi or auds.acsi.channel != vc_:
				await auds.move_unmute(auds.acsi, vc_)
				joining = True
		if not guild.me:
			raise RuntimeError("Server not detected!")
		if guild.me.voice is None:
			for i in range(1, 17):
				try:
					await bot.wait_for("voice_state_update", check=lambda member, before, after: member.id == bot.id and after, timeout=1)
				except (T0, T1, T2):
					if guild.me.voice is None and auds.acsi is None:
						if i >= 16:
							auds.kill(reason="")
							raise
						continue
				break
		member = guild.me
		if getattr(member, "voice", None) is not None and vc_.permissions_for(member).mute_members:
			if vc_.type is discord.ChannelType.stage_voice:
				if member.voice.suppress or member.voice.requested_to_speak_at:
					auds.speak()
			elif member.voice.deaf or member.voice.mute or member.voice.afk:
				create_task(member.edit(mute=False))
		if joining:
			# Send update event to bot audio database upon joining
			create_task(bot.data.audio(guild=guild))
			return css_md(f"🎵 Successfully connected to {sqr_md(vc_)} in {sqr_md(guild)}. 🎵"), 1


class Skip(Command):
	server_only = True
	name = ["⏭", "🚫", "S", "SK", "FS", "CQ", "ForceSkip", "Remove", "Rem", "ClearQueue", "Clear"]
	min_display = "0~1"
	description = "Removes an entry or range of entries from the voice channel queue."
	usage = "<queue_positions(0)>* <force{?f}|vote{?v}|hide{?h}>*"
	example = ("skip", "fs", "skip 1 2 3", "skip 3..9", "skip 90:50:-1", "remove 9", "clear")
	flags = "fhv"
	rate_limit = (3.5, 5)
	slash = True

	async def __call__(self, bot, channel, user, perm, name, args, argv, guild, flags, message, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		auds.text = channel
		# ~clear is an alias for ~skip -f inf
		if name.startswith("c"):
			argv = "inf"
			args = [argv]
		if name[0] in "rcf" or "f" in flags:
			if not is_alone(auds, user) and perm < 1:
				raise self.perm_error(perm, 1, "to force skip while other users are in voice")
		count = len(auds.queue)
		if not count:
			raise IndexError("Queue is currently empty.")
		# Default to currently playing item
		if not argv:
			elems = [0]
		# Accept range/slice inputs
		elif ":" in argv or ".." in argv:
			while "..." in argv:
				argv = regexp("\\.{3,}").sub("..", argv)
			l = argv.replace("..", ":").split(":")
			it = None
			if len(l) > 3:
				raise ArgumentError("Too many arguments for range input.")
			elif len(l) > 2:
				num = await bot.eval_math(l[0])
				it = int(round(num))
			if l[0]:
				num = await bot.eval_math(l[0])
				if num > count:
					num = count
				else:
					num = round(num) % count
				left = num
			else:
				left = 0
			if l[1]:
				num = await bot.eval_math(l[1])
				if num > count:
					num = count
				else:
					num = round(num) % count
				right = num
			else:
				right = count
			elems = xrange(left, right, it)
		else:
			# Accept multiple single indices
			elems = [0] * len(args)
			for i in range(len(args)):
				elems[i] = await bot.eval_math(args[i])
		if not "f" in flags:
			valid = True
			for e in elems:
				if not is_finite(e):
					valid = False
					break
			if not valid:
				elems = range(count)
		async with auds.semaphore:
			# Calculate required vote count based on amount of non-bot members in voice
			members = sum(1 for m in auds.acsi.channel.members if not m.bot)
			required = 1 + members >> 1
			response = ""
			i = 1
			for pos in elems:
				pos = float(pos)
				try:
					# If infinite entries are selected and force flag is set, remove all items
					if not is_finite(pos):
						if name[0] in "rcf" or "f" in flags:
							auds.queue.clear()
							await asubmit(auds.reset, start=False)
							if "h" not in flags:
								return italics(fix_md("Removed all items from the queue.")), 1
							return
						raise LookupError
					curr = auds.queue[pos]
				except LookupError:
					response += "\n" + repr(IndexError(f"Entry {pos} is out of range."))
					continue
				# Add skips if voting
				if issubclass(type(curr.skips), collections.abc.MutableSequence):
					if name[0] in "rcf" or "f" in flags or user.id == curr["u_id"] and not "v" in flags:
						curr.skips = None
					elif user.id not in curr.skips:
						curr.skips.append(user.id)
				elif "v" in flags:
					curr.skips = deque([user.id])
				else:
					curr.skips = None
				if curr.skips is not None:
					if len(response) > 1200:
						response = lim_str(response, 1200)
					else:
						response += f"Voted to remove {sqr_md(curr.name)}: {sqr_md(str(len(curr.skips)) + '/' + str(required))}.\n"
				if not i & 8191:
					await asyncio.sleep(0.2)
				i += 1
			# Get list of items to remove from the queue, based on whether they have sufficient amount of skips
			pops = set()
			count = 0
			i = 0
			while i < len(auds.queue):
				song = auds.queue[i]
				if song.skips is None or len(song.skips) >= required:
					pops.add(i)
					if count < 4:
						response += f"{sqr_md(song.name)} has been removed from the queue.\n"
					count += 1
				i += 1
			if pops:
				auds.queue.pops(pops)
				if 1 in pops:
					auds.clear_next()
				# If first item is skipped, advance queue and update audio player
				if 0 in pops:
					auds.clear_source()
					await asubmit(auds.reset)
			if "h" not in flags:
				if count >= 4:
					return italics(css_md(f"{sqr_md(count)} items have been removed from the queue."))
				return css_md(response), 1


class Pause(Command):
	server_only = True
	name = ["⏸️", "⏯️", "⏹️", "Resume", "Unpause", "Stop"]
	min_display = "0~1"
	description = "Pauses, stops, or resumes audio playing."
	usage = "<hide{?h}>?"
	example = ("pause", "resume", "stop")
	flags = "h"
	rate_limit = (3, 4)
	slash = True

	async def __call__(self, bot, name, guild, user, perm, channel, message, flags, comment="", **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		auds.text = channel
		if name in ("pause", "stop", "⏸️", "⏯️", "⏹️"):
			if not is_alone(auds, user) and perm < 1:
				raise self.perm_error(perm, 1, f"to {name} while other users are in voice")
		if name in ("resume", "unpause"):
			await asubmit(auds.resume)
			word = name + "d"
		elif name in ("⏹️", "stop"):
			await asubmit(auds.stop)
			word = "stopped"
		elif name in ("⏸️", "pause"):
			await asubmit(auds.pause)
			word = "paused"
		else:
			await asubmit(auds.pause, unpause=True)
			word = "paused" if auds.paused else "resumed"
		if "h" not in flags:
			s = css_md(f"Successfully {word} audio playback in {sqr_md(guild)}.")
			if comment:
				s = comment + "\n" + s
			return await send_with_react(channel, s, reference=message, reacts="❎")


class Seek(Command):
	server_only = True
	name = ["↔️", "Replay"]
	min_display = "0~1"
	description = "Seeks to a position in the current audio file."
	usage = "<position(0)>? <hide{?h}>?"
	example = ("replay", "seek 5m30", "seek 3:41", "seek 123")
	flags = "h"
	rate_limit = (0.5, 3)
	slash = True

	async def __call__(self, argv, bot, guild, user, perm, channel, name, flags, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		auds.text = channel
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to seek while other users are in voice")
		# ~replay always seeks to position 0
		if name == "replay":
			num = 0
		elif not argv:
			return ini_md(f"Current audio position: {sqr_md(sec2time(auds.pos))}."), 1
		else:
			# ~seek takes an optional time input
			orig = auds.pos
			expr = argv
			num = await bot.eval_time(expr, orig)
		pos = await asubmit(auds.seek, num)
		if "h" not in flags:
			return italics(css_md(f"Successfully moved audio position to {sqr_md(sec2time(pos))}.")), 1


class Dump(Command):
	server_only = True
	time_consuming = True
	name = ["Export", "Import", "Save", "Load"]
	alias = name + ["Dujmpö"]
	min_display = "0~1"
	description = "Saves or loads the currently playing audio queue state."
	usage = "<data>? <append{?a}|song_positions{?x}|hide{?h}>*"
	example = ("save", "dump https://cdn.discordapp.com/attachments/731709481863479436/1052210287303999528/dump.json")
	flags = "ahx"
	rate_limit = (1, 2)
	slash = True

	async def __call__(self, guild, channel, user, bot, perm, name, argv, flags, message, vc=None, noannouncefirst=False, **void):
		auds = await auto_join(guild, channel, user, bot, vc=vc)
		# ~save is the same as ~dump without an argument
		if argv == "" and not message.attachments or name in ("save", "export"):
			if name in ("load", "import"):
				raise ArgumentError("Please input a file or URL to load.")
			async with discord.context_managers.Typing(channel):
				x = "x" in flags
				resp, fn = await asubmit(auds.get_dump, x, paused=x, js=True, timeout=18)
				f = CompatFile(resp, filename=fn)
			create_task(bot.send_with_file(channel, f"Queue data for {bold(str(guild))}:", f, reference=message))
			return
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to load new queue while other users are in voice")
		if isinstance(argv, str):
			if message.attachments:
				url = message.attachments[0].url
			else:
				url = argv
			urls = await bot.follow_url(argv, allow=True, images=False)
			try:
				url = urls[0]
			except IndexError:
				raise ArgumentError("Input must be a valid URL or attachment.")
			s = await self.bot.get_request(url)
			try:
				d = await asubmit(select_and_loads, s, size=268435456)
			except orjson.JSONDecodeError:
				d = [url for url in as_str(s).splitlines() if is_url(url)]
				if not d:
					raise
				d = [dict(name=url.split("?", 1)[0].rsplit("/", 1)[-1], url=url) for url in d]
		else:
			# Queue may already be in dict form if loaded from database
			d = argv
		if type(d) is list:
			d = dict(queue=d, stats={})
		elif "stats" not in d:
			d["stats"] = {}
		q = d["queue"][:262144]
		try:
			ctx = discord.context_managers.Typing(channel) if q else emptyctx
		except AttributeError:
			ctx = emptyctx
		async with ctx:
			# Copy items and cast to cdict queue entries
			for i, e in enumerate(q, 1):
				if type(e) is not cdict:
					e = q[i - 1] = cdict(e)
				e.duration = e.get("duration")
				e.u_id = user.id
				e.skips = deque()
				if not i & 8191:
					await asyncio.sleep(0.1)
			# Shuffle newly loaded dump if autoshuffle is on
			if auds.stats.shuffle and not vc:
				shuffle(q)
			for k, v in deque(d["stats"].items()):
				if k not in auds.stats:
					d["stats"].pop(k, None)
				if k in "loop repeat shuffle quiet stay":
					d["stats"][k] = bool(v)
				elif isinstance(v, str):
					d["stats"][k] = mpf(v)
		if noannouncefirst:
			d["queue"][0]["noannounce"] = True
		if "a" not in flags:
			# Basic dump, replaces current queue
			if auds.queue:
				auds.queue.clear()
			auds.stats.update(d["stats"])
			auds.seek_pos = d.get("pos", 0)
			if d.get("paused"):
				await asubmit(auds.pause)
			auds.queue.enqueue(q, -1)
			if "h" not in flags:
				return italics(css_md(f"Successfully loaded audio data for {sqr_md(guild)}.")), 1
		else:
			# append dump, adds items without replacing
			auds.queue.enqueue(q, -1)
			auds.stats.update(d["stats"])
			if "h" not in flags:
				return italics(css_md(f"Successfully appended loaded data to queue for {sqr_md(guild)}.")), 1


class AudioSettings(Command):
	server_only = True
	# Aliases are a mess lol
	aliasMap = {
		"Volume": "volume",
		"Speed": "speed",
		"Pitch": "pitch",
		"Pan": "pan",
		"BassBoost": "bassboost",
		"Reverb": "reverb",
		"Compressor": "compressor",
		"Chorus": "chorus",
		"NightCore": "resample",
		"Resample": "resample",
		"Bitrate": "bitrate",
		"LoopQueue": "loop",
		"Repeat": "repeat",
		"ShuffleQueue": "shuffle",
		"AutoShuffle": "shuffle",
		"Quiet": "quiet",
		"Stay": "stay",
		"Reset": "reset",
	}
	aliasExt = {
		"AudioSettings": None,
		"Audio": None,
		# "A": None,
		"Vol": "volume",
		"V": "volume",
		"🔉": "volume",
		"🔊": "volume",
		"📢": "volume",
		"SP": "speed",
		"⏩": "speed",
		"rewind": "rewind",
		"⏪": "rewind",
		"PI": "pitch",
		"↕️": "pitch",
		"PN": "pan",
		"BB": "bassboost",
		"🥁": "bassboost",
		"RV": "reverb",
		"📉": "reverb",
		"CO": "compressor",
		"🗜": "compressor",
		"CH": "chorus",
		"📊": "chorus",
		"NC": "resample",
		"BPS": "bitrate",
		"BR": "bitrate",
		"LQ": "loop",
		"🔁": "loop",
		"LoopOne": "repeat",
		"🔂": "repeat",
		"L1": "repeat",
		"SQ": "shuffle",
		"🤫": "quiet",
		"🔕": "quiet",
		"24/7": "stay",
		"♻": "reset",
	}
	rate_limit = (3.5, 5)
	slash = True

	def __init__(self, *args):
		self.alias = list(self.aliasMap) + list(self.aliasExt)[1:]
		self.name = list(self.aliasMap)
		self.min_display = "0~2"
		self.description = "Changes the current audio settings for this server. Some settings are very flexible; volume and bassboost are unlimited, speed and nightcore can be negative, etc."
		self.usage = "<value>? <volume(?v)|speed(?s)|pitch(?p)|pan(?e)|bassboost(?b)|reverb(?r)|compressor(?c)|chorus(?u)|nightcore(?n)|bitrate(?i)|loop(?l)|repeat(?1)|shuffle(?x)|quiet(?q)|stay(?t)|force_permanent(?f)|disable(?d)|hide(?h)>*"
		self.example = ("volume 150", "speed 200", "pitch -400", "reverb -f 320", "chorus -d", "bitrate 19600", "repeat 1", "stay 1")
		self.flags = "vspebrcunil1xqtfdh"
		self.map = {k.casefold(): self.aliasMap[k] for k in self.aliasMap}
		add_dict(self.map, {k.casefold(): self.aliasExt[k] for k in self.aliasExt})
		super().__init__(*args)

	async def __call__(self, bot, channel, user, guild, flags, name, argv, perm, message, comment="", **void):
		auds = await auto_join(guild, channel, user, bot)
		ops = alist()
		op1 = self.map[name]
		if op1 == "reset":
			flags.clear()
			flags["d"] = True
		elif op1 is not None:
			ops.append(op1)
		disable = "d" in flags
		# yanderedev code moment 🙃🙃🙃
		if "v" in flags:
			ops.append("volume")
		if "s" in flags:
			ops.append("speed")
		if "p" in flags:
			ops.append("pitch")
		if "e" in flags:
			ops.append("pan")
		if "b" in flags:
			ops.append("bassboost")
		if "r" in flags:
			ops.append("reverb")
		if "c" in flags:
			ops.append("compressor")
		if "u" in flags:
			ops.append("chorus")
		if "n" in flags:
			ops.append("resample")
		if "i" in flags:
			ops.append("bitrate")
		if "l" in flags:
			ops.append("loop")
		if "1" in flags:
			ops.append("repeat")
		if "x" in flags:
			ops.append("shuffle")
		if "q" in flags:
			ops.append("quiet")
		if "t" in flags:
			ops.append("stay")
		# If no number input given, show audio setting
		if not disable and not argv and (len(ops) != 1 or ops[-1] not in "rewind loop repeat shuffle quiet stay"):
			if len(ops) == 1:
				op = ops[0]
			else:
				key = lambda x: x if type(x) is bool else round_min(100 * x)
				d = dict(auds.stats)
				d.pop("position", None)
				return f"Current audio settings for **{escape_markdown(guild.name)}**:\n{ini_md(iter2str(d, key=key))}"
			orig = auds.stats[op]
			num = round_min(100 * orig)
			return css_md(f"Current audio {op} setting in {sqr_md(guild)}: [{num}].")
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to modify audio settings while other users are in voice")
		# No audio setting selected
		if not ops:
			if disable:
				# Disables all audio settings
				pos = auds.pos
				res = False
				for k, v in auds.defaults.items():
					if k != "volume" and auds.stats.get(k) != v:
						res = True
						break
				auds.stats = cdict(auds.defaults)
				if "f" in flags:
					bot.data.audiosettings.pop(guild.id, None)
				if auds.queue and res:
					auds.clear_next()
					await asubmit(auds.play, auds.source, pos, timeout=18)
				succ = "Permanently" if "f" in flags else "Successfully"
				return italics(css_md(f"{succ} reset all audio settings for {sqr_md(guild)}."))
			else:
				# Default to volume
				ops.append("volume")
		s = ""
		for op in ops:
			# These audio settings automatically invert when used
			if type(op) is str:
				if op in "loop repeat shuffle quiet stay" and not argv:
					argv = str(not auds.stats[op])
				elif op == "rewind":
					argv = "100"
			if op == "rewind":
				op = "speed"
				argv = "- " + argv
			# This disables one or more audio settings
			if disable:
				val = auds.defaults[op]
				if type(val) is not bool:
					val *= 100
				argv = str(val)
			# Values should be scaled by 100 to indicate percentage
			origStats = auds.stats
			orig = round_min(origStats[op] * 100)
			if argv.endswith("%"):
				argv = argv[:-1]
			num = await bot.eval_math(argv, orig)
			new = round_min(num)
			val = round_min(num / 100)
			if op in "loop repeat shuffle quiet stay":
				origStats[op] = new = bool(val)
				orig = bool(orig)
				if "f" in flags:
					bot.data.audiosettings.setdefault(guild.id, {})[op] = new
					bot.data.audiosettings.update(guild.id)
			else:
				if op == "bitrate":
					if val > CustomAudio.max_bitrate:
						raise PermissionError(f"Maximum permitted bitrate is {CustomAudio.max_bitrate}.")
					elif val < 5.12:
						raise ValueError("Bitrate must be equal to or above 512.")
				elif op == "speed":
					if abs(val * 2 ** (origStats.get("resample", 0) / 12)) > 16:
						raise OverflowError("Maximum permitted speed is 1600%.")
				elif op == "resample":
					if abs(origStats.get("speed", 1) * 2 ** (val / 12)) > 16:
						raise OverflowError("Maximum permitted speed is 1600%.")
				origStats[op] = val
				if "f" in flags:
					bot.data.audiosettings.setdefault(guild.id, {})[op] = val
					bot.data.audiosettings.update(guild.id)
			if auds.queue:
				if type(op) is str and op not in "loop repeat shuffle quiet stay":
					# Attempt to adjust audio setting by re-initializing FFmpeg player
					auds.clear_next()
					try:
						await asubmit(auds.play, auds.source, auds.pos, timeout=12)
					except (T0, T1, T2):
						if auds.source:
							print(auds.args)
						await asubmit(auds.stop, timeout=18)
						raise RuntimeError("Unable to adjust audio setting.")
			changed = "Permanently changed" if "f" in flags else "Changed"
			s += f"\n{changed} audio {op} setting from {sqr_md(orig)} to {sqr_md(new)}."
		if "h" not in flags:
			s = css_md(s)
			if comment:
				s = comment + "\n" + s
			return await send_with_react(channel, s, reference=message, reacts="❎")


class Jump(Command):
	server_only = True
	name = ["🔄", "Roll", "Next", "RotateQueue"]
	min_display = "0~1"
	description = "Rotates the queue to the left by a certain amount of steps."
	usage = "<position>? <hide{?h}>?"
	example = ("jump 6", "roll -3")
	flags = "h"
	rate_limit = (4, 9)

	async def __call__(self, perm, argv, flags, guild, channel, user, bot, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		auds.text = channel
		if not argv:
			amount = 1
		else:
			amount = await bot.eval_math(argv)
		if len(auds.queue) > 1 and amount:
			if not is_alone(auds, user) and perm < 1:
				raise self.perm_error(perm, 1, "to rotate queue while other users are in voice")
			async with auds.semaphore:
				# Clear "played" tag of current item
				auds.queue.rotate(-amount)
				await asubmit(auds.reset)
		if "h" not in flags:
			return italics(css_md(f"Successfully rotated queue [{amount}] step{'s' if amount != 1 else ''}.")), 1


class Shuffle(Command):
	server_only = True
	name = ["🔀", "Scramble"]
	min_display = "0~1"
	description = "Shuffles the audio queue. Leaves the current song untouched unless ?f is specified."
	usage = "<force_full_shuffle{?f}|hide{?h}>*"
	flags = "fsh"
	rate_limit = (4, 9)
	slash = True

	async def __call__(self, perm, flags, guild, channel, user, bot, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		if not auds.queue:
			raise IndexError("Queue is currently empty.")
		auds.text = channel
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to shuffle queue while other users are in voice")
		async with auds.semaphore:
			if "f" in flags or "s" in flags:
				# Clear "played" tag of current item
				shuffle(auds.queue)
				await asubmit(auds.reset)
			else:
				temp = auds.queue.popleft()
				shuffle(auds.queue)
				auds.queue.appendleft(temp)
		if "h" not in flags:
			return italics(css_md(f"Successfully shuffled queue for {sqr_md(guild)}.")), 1


class Dedup(Command):
	server_only = True
	name = ["Unique", "Deduplicate", "RemoveDuplicates"]
	min_display = "0~1"
	description = "Removes all duplicate items from the audio queue."
	usage = "<hide{?h}>*"
	flags = "h"
	rate_limit = (4, 9)

	async def __call__(self, perm, flags, guild, channel, user, bot, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		if not auds.queue:
			raise IndexError("Queue is currently empty.")
		auds.text = channel
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to removed duplicate items from queue while other users are in voice")
		async with auds.semaphore:
			if auds.queue:
				queue = auds.queue
				orig = queue[0]
				pops = deque()
				found = set()
				for i, e in enumerate(queue):
					if e["url"] in found:
						pops.append(i)
					else:
						found.add(e["url"])
				queue.pops(pops)
				if orig != queue[0]:
					await asubmit(auds.reset)
		if "h" not in flags:
			return italics(css_md(f"Successfully removed duplicate items from queue for {sqr_md(guild)}.")), 1


class Reverse(Command):
	server_only = True
	min_display = "0~1"
	description = "Reverses the audio queue direction."
	usage = "<hide{?h}>?"
	flags = "h"
	rate_limit = (4, 9)

	async def __call__(self, perm, flags, guild, channel, user, bot, **void):
		if guild.id not in bot.data.audio.players:
			raise LookupError("Currently not playing in a voice channel.")
		auds = bot.data.audio.players[guild.id]
		if not auds.queue:
			raise IndexError("Queue is currently empty.")
		auds.text = channel
		if not is_alone(auds, user) and perm < 1:
			raise self.perm_error(perm, 1, "to reverse queue while other users are in voice")
		async with auds.semaphore:
			reverse(auds.queue)
			auds.queue.rotate(-1)
		if "h" not in flags:
			return italics(css_md(f"Successfully reversed queue for {sqr_md(guild)}.")), 1


class UnmuteAll(Command):
	server_only = True
	time_consuming = True
	min_level = 3
	description = "Disables server mute/deafen for all members."
	usage = "<hide{?h}>?"
	flags = "h"
	rate_limit = 10

	async def __call__(self, guild, flags, **void):
		for vc in guild.voice_channels:
			for user in vc.members:
				if user.voice is not None:
					if user.voice.deaf or user.voice.mute or user.voice.afk:
						create_task(user.edit(mute=False, deafen=False))
		if "h" not in flags:
			return italics(css_md(f"Successfully unmuted all users in voice channels in {sqr_md(guild)}.")), 1


class VoiceNuke(Command):
	server_only = True
	time_consuming = True
	min_level = 3
	name = ["☢️"]
	description = "Removes all users from voice channels in the current server."
	usage = "<hide{?h}>?"
	flags = "h"
	rate_limit = 10

	async def __call__(self, guild, flags, **void):
		connected = set()
		for vc in voice_channels(guild):
			for user in vc.members:
				if user.id != self.bot.id:
					if user.voice is not None:
						connected.add(user)
		await disconnect_members(self.bot, guild, connected)
		if "h" not in flags:
			return italics(css_md(f"Successfully removed all users from voice channels in {sqr_md(guild)}.")), 1


class Radio(Command):
	name = ["FM"]
	description = "Searches for a radio station livestream on https://worldradiomap.com that can be played on ⟨MIZA⟩."
	usage = "<0:country>? <2:state>? <1:city>?"
	example = ("radio", "radio australia", "radio Canada Ottawa,_on")
	rate_limit = (6, 8)
	slash = True
	countries = fcdict()

	def country_repr(self, c):
		out = io.StringIO()
		start = None
		for w in c.split("_"):
			if len(w) > 1:
				if start:
					out.write("_")
				if len(w) > 3 or not start:
					if len(w) < 3:
						out.write(w.upper())
					else:
						out.write(w.capitalize())
				else:
					out.write(w.lower())
			else:
				out.write(w.upper())
			start = True
		out.seek(0)
		return out.read().strip("_")

	def get_countries(self):
		with tracebacksuppressor:
			resp = Request("https://worldradiomap.com", timeout=24)
			search = b'<option value="selector/_blank.htm">- Select a country -</option>'
			resp = resp[resp.index(search) + len(search):]
			resp = resp[:resp.index(b"</select>")]
			with suppress(ValueError):
				while True:
					search = b'<option value="'
					resp = resp[resp.index(search) + len(search):]
					search = b'">'
					href = as_str(resp[:resp.index(search)])
					if not href.startswith("http"):
						href = "https://worldradiomap.com/" + href.lstrip("/")
					if href.endswith(".htm"):
						href = href[:-4]
					resp = resp[resp.index(search) + len(search):]
					country = single_space(as_str(resp[:resp.index(b"</option>")]).replace(".", " ")).replace(" ", "_")
					try:
						self.countries[country].url = href
					except KeyError:
						self.countries[country] = cdict(name=country, url=href, cities=fcdict(), states=False)
					data = self.countries[country]
					alias = href.rsplit("/", 1)[-1].split("_", 1)[-1]
					self.countries[alias] = data

					def get_cities(country):
						resp = Request(country.url, decode=True)
						search = '<img src="'
						resp = resp[resp.index(search) + len(search):]
						icon, resp = resp.split('"', 1)
						icon = icon.replace("../", "https://worldradiomap.com/")
						country.icon = icon
						search = '<option selected value="_blank.htm">- Select a city -</option>'
						try:
							resp = resp[resp.index(search) + len(search):]
						except ValueError:
							search = '<option selected value="_blank.htm">- State -</option>'
							resp = resp[resp.index(search) + len(search):]
							country.states = True
							with suppress(ValueError):
								while True:
									search = '<option value="'
									resp = resp[resp.index(search) + len(search):]
									search = '">'
									href = as_str(resp[:resp.index(search)])
									if not href.startswith("http"):
										href = "https://worldradiomap.com/selector/" + href
									if href.endswith(".htm"):
										href = href[:-4]
									search = "<!--"
									resp = resp[resp.index(search) + len(search):]
									city = single_space(resp[:resp.index("-->")].replace(".", " ")).replace(" ", "_")
									country.cities[city] = cdict(url=href, cities=fcdict(), icon=icon, states=False, get_cities=get_cities)
									country.cities[city.rsplit(",", 1)[0]] = cdict(url=href, cities=fcdict(), icon=icon, states=False, get_cities=get_cities)
									self.bot.data.radiomaps[full_prune(city)] = country.name
									self.bot.data.radiomaps[full_prune(city.rsplit(",", 1)[0])] = country.name
						else:
							resp = resp[:resp.index("</select>")]
							with suppress(ValueError):
								while True:
									search = '<option value="'
									resp = resp[resp.index(search) + len(search):]
									search = '">'
									href = as_str(resp[:resp.index(search)])
									if href.startswith("../"):
										href = "https://worldradiomap.com/" + href[3:]
									if href.endswith(".htm"):
										href = href[:-4]
									resp = resp[resp.index(search) + len(search):]
									city = single_space(resp[:resp.index("</option>")].replace(".", " ")).replace(" ", "_")
									country.cities[city] = href
									country.cities[city.rsplit(",", 1)[0]] = href
									self.bot.data.radiomaps[full_prune(city)] = country.name
									self.bot.data.radiomaps[full_prune(city.rsplit(",", 1)[0])] = country.name
						return country

					data.get_cities = get_cities
		return self.countries

	async def __call__(self, bot, channel, message, args, **void):
		if not self.countries:
			await asubmit(self.get_countries)
		path = deque()
		if not args:
			fields = msdict()
			for country in self.countries:
				if len(country) > 2:
					fields.add(country[0].upper(), self.country_repr(country))
			bot.send_as_embeds(channel, title="Available countries", fields={k: "\n".join(v) for k, v in fields.items()}, author=get_author(bot.user), reference=message)
			return
		c = args.pop(0)
		if c not in self.countries:
			await asubmit(self.get_countries)
			if c not in self.countries:
				d = full_prune(c)
				if d in bot.data.radiomaps:
					args.insert(0, c)
					c = bot.data.radiomaps[d]
				else:
					raise LookupError(f"Country {c} not found.")
		path.append(c)
		country = self.countries[c]
		if not country.cities:
			await asubmit(country.get_cities, country)
		if not args:
			fields = msdict()
			desc = deque()
			for city in country.cities:
				desc.append(self.country_repr(city))
			t = "states" if country.states else "cities"
			bot.send_as_embeds(channel, title=f"Available {t} in {self.country_repr(c)}", thumbnail=country.icon, description="\n".join(desc), author=get_author(bot.user), reference=message)
			return
		c = args.pop(0)
		if c not in country.cities:
			await asubmit(country.get_cities, country)
			if c not in country.cities:
				d = full_prune(c)
				if d in bot.data.radiomaps:
					args.insert(0, c)
					c = bot.data.radiomaps[d]
				else:
					raise LookupError(f"Country {c} not found.")
		path.append(c)
		city = country.cities[c]
		if type(city) is not str:
			state = city
			if not state.cities:
				await asubmit(state.get_cities, state)
			if not args:
				fields = msdict()
				desc = deque()
				for city in state.cities:
					desc.append(self.country_repr(city))
				bot.send_as_embeds(channel, title=f"Available cities in {self.country_repr(c)}", thumbnail=country.icon, description="\n".join(desc), author=get_author(bot.user), reference=message)
				return
			c = args.pop(0)
			if c not in state.cities:
				await asubmit(state.get_cities, state)
				if c not in state.cities:
					raise LookupError(f"City {c} not found.")
			path.append(c)
			city = state.cities[c]
		resp = await Request(city, aio=True)
		title = "Radio stations in " + ", ".join(self.country_repr(c) for c in reversed(path)) + ", by frequency (MHz)"
		fields = deque()
		search = b'<table class=fix cellpadding="0" cellspacing="0">'
		resp = as_str(resp[resp.index(search) + len(search):resp.index(b"</p></div><!--end rightcontent-->")])
		for section in resp.split("<td class=tr31><b>")[1:]:
			try:
				i = regexp(r"(?:Hz|赫|هرتز|Гц)</td>").search(section).start()
				scale = section[section.index("</b>,") + 5:i].upper()
			except:
				print(section)
				print_exc()
				scale = ""
			coeff = 0.000001
			if any(n in scale for n in ("M", "兆", "مگا", "М")):
				coeff = 1
			elif any(n in scale for n in ("K", "千", "کیلو", "к")):
				coeff = 0.001
			# else:
				# coeff = 1
			with tracebacksuppressor:
				while True:
					search = "<td class=freq>"
					search2 = "<td class=dxfreq>"
					i = j = inf
					with suppress(ValueError):
						i = section.index(search) + len(search)
					with suppress(ValueError):
						j = section.index(search2) + len(search2)
					if i > j:
						i = j
					if type(i) is not int:
						break
					section = section[i:]
					freq = round_min(round(float(section[:section.index("<")].replace("&nbsp;", "").strip()) * coeff, 6))
					field = [freq, ""]
					curr, section = section.split("</tr>", 1)
					for station in regexp('(?:<td class=(?:dx)?fsta2?>|\s{2,})<a href="').split(curr)[1:]:
						if field[1]:
							field[1] += "\n"
						href, station = station.split('"', 1)
						if not href.startswith("http"):
							href = "https://worldradiomap.com/" + href.lstrip("/")
							if href.endswith(".htm"):
								href = href[:-4]
						search = "class=station>"
						station = station[station.index(search) + len(search):]
						name = station[:station.index("<")]
						field[1] += f"[{name.strip()}]({href.strip()})"
					fields.append(field)
		bot.send_as_embeds(channel, title=title, thumbnail=country.icon, fields=sorted(fields), author=get_author(bot.user), reference=message)


class UpdateRadioMaps(Database):
	name = "radiomaps"


class Party(Command):
	server_only = True
	min_level = 2
	name = ["YTT"]
	description = "Generates an activity party link in the nearest voice channel!"
	usage = "(poker|betrayal|youtube|fishington|chess)?"
	example = ("ytt", "party poker")
	rate_limit = (10, 14)
	sem = Semaphore(2, 8, rate_limit=8)
	names = fcdict((
		("Poker", 755827207812677713),
		("Poker Night", 755827207812677713),
		("YouTube", 755600276941176913),
		("YouTube Together", 755600276941176913),
		("Betrayal", 773336526917861400),
		("Betrayal.io", 773336526917861400),
		("Fishington", 814288819477020702),
		("Fishington.io", 814288819477020702),
		("Chess", 832012586023256104),
	))

	async def __call__(self, bot, guild, channel, user, argv, **void):
		vc = select_voice_channel(user, channel)
		if not argv:
			argv = "YouTube"
		try:
			aid = self.names[argv]
		except KeyError:
			if not argv.isnumeric():
				raise KeyError(f"Unsupported party application: {argv}")
			aid = int(argv)
		# invites = await bot.get_full_invites(guild)
		# for invite in invites:
		#     if invite.max_age == invite.max_uses == 0:
		#         c = invite.get("channel", {})
		#         if c.get("id") == vc.id:
		#             app = invite.get("target_application")
		#             if app:
		#                 if app.get("id") == aid:
		#                     return f"https://discord.gg/{invite.code}"
		async with self.sem:
			data = await Request(
				f"https://discord.com/api/{api}/channels/{vc.id}/invites",
				method="POST",
				data=dict(
					max_age=0,
					max_uses=0,
					target_application_id=aid,
					target_type=2,
					temporary=False,
					validate=None,
				),
				authorise=True,
				aio=True,
				json=True
			)
		return f"https://discord.gg/{data['code']}"


class Player(Command):
	server_only = True
	buttons = demap({
		b'\xe2\x8f\xaf\xef\xb8\x8f': 0,
		b'\xf0\x9f\x94\x81': 1,
		b'\xf0\x9f\x94\x80': 2,
		b'\xe2\x8f\xae': 3,
		b'\xe2\x8f\xad': 4,
		b'\xf0\x9f\x94\x8a': 5,
		b'\xf0\x9f\xa5\x81': 6,
		b'\xf0\x9f\x93\x89': 7,
		b'\xf0\x9f\x93\x8a': 8,
		b'\xe2\x99\xbb': 9,
		# b'\xe2\x8f\xaa': 10,
		# b'\xe2\x8f\xa9': 11,
		# b'\xe2\x8f\xab': 12,
		# b'\xe2\x8f\xac': 13,
		b'\xe2\x8f\x8f': 14,
		b'\xe2\x9c\x96': 15,
	})
	barsize = 24
	name = ["NP", "NowPlaying", "Playing"]
	min_display = "0~3"
	description = "Creates an auto-updating virtual audio player for the current server."
	usage = "<enable{?e}|disable{?d}>?"
	example = ("player", "np -d")
	flags = "adez"
	rate_limit = (6, 9)

	async def show(self, auds):
		q = auds.queue
		if q:
			s = q[0].skips
			if s is not None:
				skips = len(s)
			else:
				skips = 0
			output = "Playing " + str(len(q)) + " item" + "s" * (len(q) != 1) + " "
			output += skips * "🚫"
		else:
			output = "Queue is currently empty. "
		if auds.stats.repeat:
			output += "🔂"
		else:
			if auds.stats.loop:
				output += "🔁"
			if auds.stats.shuffle:
				output += "🔀"
		if auds.stats.quiet:
			output += "🔕"
		if q:
			p = auds.epos
		else:
			p = [0, 1]
		output += "```"
		output += await self.bot.create_progress_bar(18, p[0] / p[1])
		if q:
			output += "\n[`" + no_md(q[0].name) + "`](" + ensure_url(q[0].url) + ")"
		output += "\n`"
		if auds.paused or not auds.stats.speed:
			output += "⏸️"
		elif auds.stats.speed > 0:
			output += "▶️"
		else:
			output += "◀️"
		if q:
			p = auds.epos
		else:
			p = [0, 0.25]
		output += uni_str(f" ({time_disp(p[0])}/{time_disp(p[1])})`\n")
		if auds.has_options():
			v = abs(auds.stats.volume)
			if v == 0:
				output += "🔇"
			if v <= 0.5:
				output += "🔉"
			elif v <= 1.5:
				output += "🔊"
			elif v <= 5:
				output += "📢"
			else:
				output += "🌪️"
			b = auds.stats.bassboost
			if abs(b) > 1 / 6:
				if abs(b) > 5:
					output += "💥"
				elif b > 0:
					output += "🥁"
				else:
					output += "🎻"
			r = auds.stats.reverb
			if r:
				if abs(r) >= 1:
					output += "📈"
				else:
					output += "📉"
			u = auds.stats.chorus
			if u:
				output += "📊"
			c = auds.stats.compressor
			if c:
				output += "🗜️"
			e = auds.stats.pan
			if abs(e - 1) > 0.25:
				output += "♒"
			s = auds.stats.speed * 2 ** (auds.stats.resample / 12)
			if s < 0:
				output += "⏪"
			elif s > 1:
				output += "⏩"
			elif s > 0 and s < 1:
				output += "🐌"
			p = auds.stats.pitch + auds.stats.resample
			if p > 0:
				output += "⏫"
			elif p < 0:
				output += "⏬"
		return output

	async def _callback_(self, message, guild, channel, reaction, bot, perm, **void):
		if not guild.id in bot.data.audio.players:
			return
		auds = bot.data.audio.players[guild.id]
		if reaction is None:
			return
		elif reaction == 0:
			auds.player.time = inf
		elif auds.player is None or auds.player.message.id != message.id:
			return
		if perm < 1:
			return
		if not message:
			content = "```callback-voice-player-\n"
		elif message.content:
			content = message.content
		else:
			content = message.embeds[0].description
		orig = content.split("\n", 1)[0] + "\n"
		if reaction:
			if type(reaction) is bytes:
				emoji = reaction
			else:
				try:
					emoji = reaction.emoji
				except:
					emoji = str(reaction)
			if type(emoji) is str:
				emoji = reaction.encode("utf-8")
			if emoji in self.buttons:
				if hasattr(message, "int_token"):
					create_task(bot.ignore_interaction(message))
				i = self.buttons[emoji]
				if i == 0:
					await asubmit(auds.pause, unpause=True)
				elif i == 1:
					if auds.stats.loop:
						auds.stats.loop = False
						auds.stats.repeat = True
					elif auds.stats.repeat:
						auds.stats.loop = False
						auds.stats.repeat = False
					else:
						auds.stats.loop = True
						auds.stats.repeat = False
				elif i == 2:
					auds.stats.shuffle = bool(auds.stats.shuffle ^ 1)
				elif i == 3 or i == 4:
					if i == 3:
						auds.seek(0)
					else:
						auds.queue.pop(0)
						auds.clear_source()
						await asubmit(auds.reset)
					return
				elif i == 5:
					v = abs(auds.stats.volume)
					if v < 0.25 or v >= 2:
						v = 1 / 3
					elif v < 1:
						v = 1
					else:
						v = 2
					auds.stats.volume = v
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 6:
					b = auds.stats.bassboost
					if abs(b) < 1 / 3:
						b = 1
					elif b < 0:
						b = 0
					else:
						b = -1
					auds.stats.bassboost = b
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 7:
					r = auds.stats.reverb
					if r >= 1:
						r = 0
					elif r < 0.5:
						r = 0.5
					else:
						r = 1
					auds.stats.reverb = r
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 8:
					c = abs(auds.stats.chorus)
					if c:
						c = 0
					else:
						c = 1 / 3
					auds.stats.chorus = c
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 9:
					pos = auds.pos
					auds.stats = cdict(auds.defaults)
					auds.stats.quiet = True
					await asubmit(auds.play, auds.source, pos, timeout=18)
				elif i == 10 or i == 11:
					s = 0.25 if i == 11 else -0.25
					auds.stats.speed = round(auds.stats.speed + s, 5)
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 12 or i == 13:
					p = 1 if i == 13 else -1
					auds.stats.pitch -= p
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 14:
					await asubmit(auds.kill)
					await bot.silent_delete(message)
					return
				else:
					auds.player = None
					await bot.silent_delete(message)
					return
		other = await self.show(auds)
		text = lim_str(orig + other, 4096)
		last = await self.bot.get_last_message(channel)
		emb = discord.Embed(
			description=text,
			colour=rand_colour(),
			timestamp=utc_dt(),
		).set_author(**get_author(self.bot.user))
		if message and last and message.id == last.id:
			await message.edit(
				embed=emb,
			)
		else:
			buttons = [[] for _ in loop(3)]
			for s, i in self.buttons.a.items():
				s = as_str(s)
				if i < 5:
					buttons[0].append(cdict(emoji=s, custom_id=s, style=3))
				elif i < 14:
					j = 1 if len(buttons[1]) < 5 else 2
					buttons[j].append(cdict(emoji=s, custom_id=s, style=1))
				else:
					buttons[-1].append(cdict(emoji=s, custom_id=s, style=4))
			auds.player.time = inf
			temp = message
			message = await send_with_reply(
				channel,
				reference=None,
				embed=emb,
				buttons=buttons,
			)
			auds.player.message = message
			await bot.silent_delete(temp, no_log=True)
		if auds.queue and not auds.paused & 1:
			p = auds.epos
			maxdel = p[1] - p[0] + 2
			delay = min(maxdel, p[1] / self.barsize / 2 / abs(auds.stats.speed))
			if delay > 10:
				delay = 10
			elif delay < 5:
				delay = 5
		else:
			delay = inf
		auds.player.time = utc() + delay
		auds.stats.quiet = True

	async def __call__(self, guild, channel, user, bot, flags, perm, **void):
		auds = await auto_join(channel.guild, channel, user, bot)
		auds.player = cdict(time=0, message=None)
		esubmit(auds.update)


# Small helper function to fetch song lyrics from json data, because sometimes genius.com refuses to include it in the HTML
def extract_lyrics(s):
	s = s[s.index("JSON.parse(") + len("JSON.parse("):]
	s = s[:s.index("</script>")]
	if "window.__" in s:
		s = s[:s.index("window.__")]
	s = s[:s.rindex(");")]
	data = literal_eval(s)
	d = eval_json(data)
	lyrics = d["songPage"]["lyricsData"]["body"]["children"][0]["children"]
	newline = True
	output = ""
	while lyrics:
		line = lyrics.pop(0)
		if type(line) is str:
			if line:
				if line.startswith("["):
					output += "\n"
					newline = False
				if "]" in line:
					if line == "]":
						if output.endswith(" ") or output.endswith("\n"):
							output = output[:-1]
					newline = True
				output += line + ("\n" if newline else (" " if not line.endswith(" ") else ""))
		elif type(line) is dict:
			if "children" in line:
				# This is a mess, the children objects may or may not represent single lines
				lyrics = line["children"] + lyrics
	return output


# Main helper function to fetch song lyrics from genius.com searches
async def get_lyrics(item):
	url = f"https://genius.com/api/search/multi?q={item}"
	for i in range(2):
		header = {"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
		data = {"q": item}
		rdata = await Request(url, data=data, headers=header, aio=True, json=True, timeout=18)
		hits = chain(*(sect["hits"] for sect in rdata["response"]["sections"]))
		name = None
		path = None
		for h in hits:
			with tracebacksuppressor:
				name = h["result"]["title"]
				path = h["result"]["api_path"]
				break
		if path and name:
			s = "https://genius.com" + path
			page = await Request(s, headers=header, decode=True, aio=True)
			text = page
			html = await asubmit(BeautifulSoup, text, "html.parser", timeout=18)
			lyricobj = html.find('div', class_='lyrics')
			if lyricobj is not None:
				lyrics = lyricobj.get_text().strip()
				print("lyrics_html", s)
				return name, lyrics
			try:
				lyrics = extract_lyrics(text).strip()
				print("lyrics_json", s)
				return name, lyrics
			except:
				if i:
					raise
				print_exc()
				print(s)
				print(text)
	raise LookupError(f"No results for {item}.")


class Lyrics(Command):
	time_consuming = True
	name = ["SongLyrics"]
	description = "Searches genius.com for lyrics of a song."
	usage = "<search_link>* <verbose{?v}>?"
	example = ("lyrics", "lyrics despacito", "lyrics -v viva la vida")
	flags = "v"
	rate_limit = (7, 12)
	typing = True
	slash = True

	async def __call__(self, bot, guild, channel, message, argv, flags, user, **void):
		for a in message.attachments:
			argv = a.url + " " + argv
		if not argv:
			try:
				auds = bot.data.audio.players[guild.id]
				if not auds.queue:
					raise LookupError
				argv = auds.queue[0].name
			except LookupError:
				raise IndexError("Queue not found. Please input a search term, URL, or file.")
		# Extract song name if input is a URL, otherwise search song name directly
		urls = await bot.follow_url(argv, allow=True, images=False)
		if urls:
			resp = await asubmit(ytdl.search, urls[0], timeout=18)
			if type(resp) is str:
				raise evalEX(resp)
			search = resp[0].name
		else:
			search = argv
		search = search.translate(self.bot.mtrans)
		# Attempt to find best query based on the song name
		item = verify_search(to_alphanumeric(lyric_trans.sub("", search)))
		ic = item.casefold()
		if ic.endswith(" with lyrics"):
			item = item[:-len(" with lyrics")]
		elif ic.endswith(" lyrics"):
			item = item[:-len(" lyrics")]
		elif ic.endswith(" acoustic"):
			item = item[:-len(" acoustic")]
		item = item.rsplit(" ft ", 1)[0].strip()
		if not item:
			item = verify_search(to_alphanumeric(search))
			if not item:
				item = search
		async with discord.context_managers.Typing(channel):
			try:
				name, lyrics = await get_lyrics(item)
			except KeyError:
				print_exc()
				raise KeyError(f"Invalid response from genius.com for {item}")
		# Escape colour markdown because that will interfere with the colours we want
		text = clr_md(lyrics.strip()).replace("#", "♯")
		msg = f"Lyrics for **{escape_markdown(name)}**:"
		s = msg + ini_md(text)
		# Directly return lyrics in a code box if it fits
		if "v" not in flags and len(s) <= 2000:
			return s
		title = f"Lyrics for {name}:"
		if len(text) > 54000:
			return (title + "\n\n" + text).strip()
		bot.send_as_embeds(channel, text, author=dict(name=title), colour=(1024, 128), md=ini_md, reference=message)


class Download(Command):
	time_consuming = True
	_timeout_ = 75
	name = ["📥", "Search", "YTDL", "DownloadAsMP3", "Youtube_DL", "AF", "AudioFilter", "Trim", "Concat", "Concatenate", "🌽🐱", "ConvertORG", "Org2xm", "Convert"]
	description = "Searches and/or downloads a song from a YouTube/SoundCloud query or audio file link. Will extend (loop) if trimmed past the end. The \"-\" character is used to omit parameters for ~trim."
	usage = "<0:search_links>* <multi{?m}> <trim{?t}>? <-3:trim_start|->? <-2:trim_end|->? <-1:out_format(mp4)>? <concatenate{?c}|remove_silence{?r}|apply_settings{?a}|verbose_search{?v}>*"
	example = ("download https://www.youtube.com/watch?v=kJQP7kiw5Fk mp3", "trim https://www.youtube.com/watch?v=dQw4w9WgXcQ 1m 3m as mp4", "concatenate https://www.youtube.com/watch?v=kJQP7kiw5Fk https://www.youtube.com/watch?v=dQw4w9WgXcQ webm")
	flags = "avtzcrm"
	rate_limit = (30, 45)
	typing = True
	slash = True
	msgcmd = ("Download as mp3",)

	async def __call__(self, bot, channel, guild, message, name, argv, flags, user, **void):
		ytdl.bot = bot
		fmt = default_fmt = "mp3"
		if name in ("af", "audiofilter"):
			set_dict(flags, "a", 1)
		# Prioritize attachments in message
		for a in message.attachments:
			argv = a.url + " " + argv
		direct = getattr(message, "simulated", None) or name == "org2xm"
		concat = "concat" in name or "c" in flags or name == "🌽🐱"
		multi = "m" in flags
		start = end = None
		# Attempt to download items in queue if no search query provided
		if not argv:
			try:
				auds = bot.data.audio.players[guild.id]
				if not auds.queue:
					raise LookupError
				res = [{"name": e.name, "url": e.url} for e in auds.queue[:10]]
				fmt = "ogg"
				desc = f"Current items in queue for {guild}:"
			except:
				raise IndexError("Queue not found. Please input a search term, URL, or file.")
		else:
			# Parse search query, detecting file format selection if possible
			if " " in argv:
				spl = smart_split(argv)
				if len(spl) >= 1:
					fmt = spl[-1].lstrip(".")
					if fmt.casefold() not in ("mp3", "ecdc", "opus", "ogg", "m4a", "flac", "wav", "wma", "mp2", "weba", "vox", "adpcm", "pcm", "8bit", "mid", "midi", "webm", "mp4", "avi", "mov", "m4v", "mkv", "f4v", "flv", "wmv", "gif", "apng", "webp"):
						fmt = default_fmt
					else:
						if spl[-2] in ("as", "to"):
							spl.pop(-1)
						argv = " ".join(spl[:-1])
			if name == "trim" or "t" in flags:
				try:
					argv, start, end = argv.rsplit(None, 2)
				except ValueError:
					raise ArgumentError("Please input search term followed by trim start and end.")
				if start == "-":
					start = None
				else:
					start = await bot.eval_time(start)
				if end == "-":
					end = None
				else:
					end = await bot.eval_time(end)
			argv = verify_search(argv)
			res = []
			# Input may be a URL or set of URLs, in which case we attempt to find the first one
			urls = await bot.follow_url(argv, allow=True, images=False)
			if urls:
				if not concat and not multi:
					urls = (urls[0],)
				futs = deque()
				for e in urls:
					futs.append(asubmit(ytdl.extract, e, timeout=120))
				for fut in futs:
					temp = await fut
					res.extend(temp)
				direct = len(res) == 1 or concat or multi
			if not res:
				# 2 youtube results per soundcloud result, increased with verbose flag, followed by 1 spotify and 1 bandcamp
				sc = min(4, flags.get("v", 0) + 1)
				yt = min(6, sc << 1)
				futs = deque()
				futs.append(asubmit(ytdl.search, argv, mode="yt", count=yt))
				futs.append(asubmit(ytdl.search, argv, mode="sc", count=sc))
				futs.append(asubmit(ytdl.search, "spsearch:" + argv.split("spsearch:", 1)[-1].replace(":", "-"), mode="sp"))
				futs.append(asubmit(ytdl.search, "bcsearch:" + argv.split("bcsearch:", 1)[-1].replace(":", "-"), mode="bc"))
				for fut in futs:
					temp = await fut
					if type(temp) is not str:
						res.extend(temp)
			if not res:
				raise LookupError(f"No results for {argv}.")
			if not concat and not multi:
				res = res[:10]
			desc = f"Search results for {argv}:"
		a = flags.get("a", 0)
		b = flags.get("r", 0)
		if multi:
			entry = [e["url"] for e in res]
			print(entry)
			futs = []
			async with discord.context_managers.Typing(channel):
				try:
					if a:
						auds = bot.data.audio.players[guild.id]
					else:
						auds = None
				except LookupError:
					auds = None
				for i, url in enumerate(entry):
					if i >= 12:
						await wrap_future(futs[i - 12])
					futs.append(esubmit(
						ytdl.download_file,
						url,
						fmt=fmt,
						start=start,
						end=end,
						auds=auds,
						silenceremove=b,
						child=len(entry) > 1,
						message=message,
					))
				fn = f"cache/{ts_us()}.zip"
				with zipfile.ZipFile(fn, "w", zipfile.ZIP_STORED, allowZip64=True, strict_timestamps=False) as z:
					for fut in futs:
						f, out = await wrap_future(fut)
						z.write(f, arcname=f.rsplit("/", 1)[-1])
				create_task(bot.send_with_file(
					channel=channel,
					msg="",
					file=fn,
					filename="download.zip",
					rename=True,
					reference=message,
				))
			return
		if concat:
			entry = [e["url"] for e in res]
			print(entry)
			async with discord.context_managers.Typing(channel):
				try:
					if a:
						auds = bot.data.audio.players[guild.id]
					else:
						auds = None
				except LookupError:
					auds = None
				f, out = await asubmit(
					ytdl.download_file,
					entry,
					fmt=fmt,
					start=start,
					end=end,
					auds=auds,
					silenceremove=b,
					message=message,
				)
				create_task(bot.send_with_file(
					channel=channel,
					msg="",
					file=f,
					filename=out,
					rename=True,
					reference=message,
				))
			return
		desc += "\nDestination format: {." + fmt + "}"
		if start is not None or end is not None:
			desc += f"\nTrim: [{'-' if start is None else start} ~> {'-' if end is None else end}]"
		if b:
			desc += ", Silence remover: {ON}"
		if a:
			desc += ", Audio settings: {ON}"
		desc += "```*"
		# Encode URL list into bytes and then custom base64 representation, hide in code box header
		url_bytes = orjson.dumps([e["url"] for e in res])
		url_enc = as_str(bytes2b64(url_bytes, True))
		vals = f"{user.id}_{len(res)}_{fmt}_{int(bool(a))}_{start}_{end}_{int(bool(b))}"
		msg = "*```" + "\n" * ("z" in flags) + "callback-voice-download-" + vals + "-" + url_enc + "\n" + desc
		emb = discord.Embed(colour=rand_colour())
		emb.set_author(**get_author(user))
		emb.description = "\n".join(f"`【{i}】` [{escape_markdown(e['name'])}]({ensure_url(e['url'])})" for i, e in enumerate(res))
		if getattr(message, "simulated", None):
			sent = bot._globals["SimulatedMessage"](bot, msg, ts_us(), message.name, message.nick)
			sent.embeds = [emb]
			sent.edit = sent.send = message.send
		else:
			sent = await send_with_reply(channel, message, msg, embed=emb)
		if direct:
			# Automatically proceed to download and convert immediately
			await self._callback_(
				message=sent,
				guild=guild,
				channel=channel,
				reaction=b"0\xef\xb8\x8f\xe2\x83\xa3",
				bot=bot,
				perm=3,
				vals=vals,
				argv=url_enc,
				user=user,
			)
			return
		# Add reaction numbers corresponding to search results for selection
		for i in range(len(res)):
			await sent.add_reaction(str(i) + as_str(b"\xef\xb8\x8f\xe2\x83\xa3"))

	async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, argv, user, **void):
		if reaction is None or user.id == bot.id:
			return
		spl = vals.split("_")
		u_id = int(spl[0])
		if user.id != u_id and perm < 3:
			return
		# Make sure reaction is a valid number
		if b"\xef\xb8\x8f\xe2\x83\xa3" not in reaction:
			return
		simulated = getattr(message, "simulated", None)
		with bot.ExceptionSender(channel):
			# Make sure selected index is valid
			num = int(as_str(reaction)[0])
			if num >= int(spl[1]):
				return
			# Reconstruct list of URLs from hidden encoded data
			data = orjson.loads(b642bytes(argv, True))
			url = data[num]
			# Perform all these tasks asynchronously to save time
			async with discord.context_managers.Typing(channel):
				f = out = None
				fmt = spl[2]
				try:
					if int(spl[3]):
						auds = bot.data.audio.players[guild.id]
					else:
						auds = None
				except LookupError:
					auds = None
				silenceremove = False
				try:
					if int(spl[6]):
						silenceremove = True
				except IndexError:
					pass
				start = end = None
				if len(spl) >= 6:
					start, end = spl[4:6]
				if not simulated:
					download = None
					if tuple(map(str, (start, end))) == ("None", "None") and not silenceremove and not auds and fmt in ("mp3", "opus", "ogg", "wav", "weba"):
						# view = bot.raw_webserver + "/ytdl?fmt=" + fmt + "&view=" + url
						download =  f"http://127.0.0.1:{PORT}/ytdl?fmt={fmt}&download={url}"
						entries = await asubmit(ytdl.search, url)
						if entries:
							name = entries[0].get("name")
						else:
							name = None
						name = name or url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
						# name = f"【{num}】{name}"
						# sem = getattr(message, "sem", None)
						# if not sem:
						#     try:
						#         sem = EDIT_SEM[message.channel.id]
						#     except KeyError:
						#         sem = EDIT_SEM[message.channel.id] = Semaphore(5.15, 256, rate_limit=5)
						# async with sem:
						#     return await Request(
						#         f"https://discord.com/api/{api}/channels/{message.channel.id}/messages/{message.id}",
						#         data=dict(
						#             components=restructure_buttons([[
						#                 cdict(emoji="🔊", name=name, url=view),
						#                 cdict(emoji="📥", name=name, url=download),
						#             ]]),
						#         ),
						#         method="PATCH",
						#         authorise=True,
						#         aio=True,
						#     )
					if len(data) <= 1:
						create_task(message.edit(
							content=ini_md(f"Downloading and converting {sqr_md(ensure_url(url))}..."),
							embed=None,
						))
					else:
						message = await message.channel.send(
							ini_md(f"Downloading and converting {sqr_md(ensure_url(url))}..."),
						)
					if download:
						f = await bot.get_request(download, timeout=3600)
						out = name + "." + (fmt if fmt != "weba" else "webm")
				if not f:
					f, out = await asubmit(
						ytdl.download_file,
						url,
						fmt=fmt,
						start=start,
						end=end,
						auds=auds,
						silenceremove=silenceremove,
						message=message.reference.cached_message if message.reference else None,
					)
				if not simulated:
					create_task(message.edit(
						content=css_md(f"Uploading {sqr_md(out)}..."),
						embed=None,
					))
					create_task(bot._state.http.send_typing(channel.id))
			reference = getattr(message, "reference", None)
			if reference:
				r_id = getattr(reference, "message_id", None) or getattr(reference, "id", None)
				reference = bot.cache.messages.get(r_id)
			resp = await bot.send_with_file(
				channel=channel,
				msg="",
				file=f,
				filename=out,
				rename=True,
				reference=reference,
			)
			if resp.attachments and type(f) is str and "~" not in f and "!" not in f and os.path.exists(f):
				with suppress():
					os.remove(f)
			if not simulated:
				create_task(bot.silent_delete(message, no_log=True))


class Transcribe(Command):
	time_consuming = True
	_timeout_ = 75
	name = ["Whisper", "TranscribeAudio", "Caption"]
	description = "Downloads a song from a link, automatically transcribing to English, or a provided language if applicable."
	usage = "<1:language(en)>? <0:search_link>"
	example = ("transcribe https://www.youtube.com/watch?v=kJQP7kiw5Fk", "transcribe Chinese https://www.youtube.com/watch?v=dQw4w9WgXcQ")
	rate_limit = (30, 45)
	typing = True
	slash = True

	async def __call__(self, bot, channel, guild, message, name, argv, flags, user, **void):
		if max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1) < 2:
			raise PermissionError(f"Sorry, this feature is currently for premium users only. Please make sure you have a subscription level of minimum 1 from {bot.kofi_url}, or try out ~trial if you would like to manage/fund your own usage!")
		for a in message.attachments:
			argv = a.url + " " + argv
		dest = "English"
		# Attempt to download items in queue if no search query provided
		if not argv:
			try:
				auds = bot.data.audio.players[guild.id]
				if not auds.queue:
					raise LookupError
				url = auds.queue[0].get("url")
			except:
				raise IndexError("Queue not found. Please input a search term, URL, or file.")
		else:
			# Parse search query, detecting file format selection if possible
			if " " in argv:
				spl = smart_split(argv)
				if len(spl) >= 1:
					tr = bot.commands.translate[0]
					arg = spl[0]
					if (dest := (tr.renamed.get(c := arg.casefold()) or (tr.languages.get(c) and c))):
						dest = (googletrans.LANGUAGES.get(dest) or dest).capitalize()
						# curr.languages.append(dest)
						argv = " ".join(spl[1:])
			argv = verify_search(argv)
			# Input must be a URL
			urls = await bot.follow_url(argv, allow=True, images=False)
			if not urls:
				raise TypeError("Input must be a valid URL.")
			url = urls[0]
		simulated = getattr(message, "simulated", None)
		async with discord.context_managers.Typing(channel):
			entries = await asubmit(ytdl.search, url)
			if entries:
				name = entries[0].get("name")
			else:
				name = None
			if not simulated:
				m = await message.reply(
					ini_md(f"Downloading and converting {sqr_md(ensure_url(url))}..."),
				)
			else:
				m = None
			await asubmit(ytdl.get_stream, entries[0], force=True, download=False)
			name, url = entries[0].get("name"), entries[0].get("url")
			if not name or not url:
				raise FileNotFoundError(500, argv)
			url = unyt(url)
			fn, out = await asubmit(ytdl.download_file, entries[0].get("stream") or entries[0].url, fmt="weba")
			fni = fn#.rsplit(".", 1)[0] + ".webm"
			if bot.is_trusted(guild) >= 2:
				for uid in bot.data.trusted[guild.id]:
					if uid and bot.premium_level(uid, absolute=True) >= 2:
						break
				else:
					uid = next(iter(bot.data.trusted[guild.id]))
				u = await bot.fetch_user(uid)
			else:
				u = user
			data = bot.data.users.get(u.id, {})
			oai = data.get("trial") and data.get("openai_key") or AUTH.get("openai_key")
			bals = {k: v for k, v in bot.data.token_balances.items() if v < 0}
			if bals:
				openai.api_key = uoai = sorted(bals, key=bals.get)[0]
			else:
				openai.api_key = uoai = oai
			if m:
				create_task(m.edit(
					content=css_md(f"Transcribing {fni}..."),
					embed=None,
				))
			with open(fni, "rb") as f:
				resp = await bot.oai.audio.translations.create(
					model="whisper-1",
					temperature=0,
					file=f,
				)
			# resp.raise_for_status()
		# text = resp.json()["text"].strip()
		text = resp.text.strip()
		tr = bot.commands.translate[0]
		if not tr.equiv(dest, "English"):
			if m:
				create_task(m.edit(
					content=css_md(f"Translating {fni}..."),
					embed=None,
				))
				create_task(bot._state.http.send_typing(channel.id))
			translated = {}
			comments = {}
			await tr.chatgpt_translate(bot, guild, channel, user, text, "English", [dest], translated, comments)
			text = "\n".join(translated.values()).strip()
		emb = discord.Embed(description=text)
		emb.title = name
		emb.colour = await bot.get_colour(user)
		emb.set_author(**get_author(user))
		if m:
			create_task(bot.silent_delete(m, no_log=True))
		# reference = getattr(m, "reference", None)
		# if reference:
		#     r_id = getattr(m, "message_id", None) or getattr(m, "id", None)
		#     reference = bot.cache.messages.get(r_id)
		bot.send_as_embeds(channel, text, author=get_author(user), reference=message)


class UpdateAudio(Database):
	name = "audio"
	destroyed = False

	def __load__(self):
		self.players = cdict()

	# Searches for and extracts incomplete queue entries
	@tracebacksuppressor
	async def research(self, auds):
		if auds.search_sem.is_busy():
			return
		async with auds.search_sem:
			searched = 0
			q = auds.queue
			async with Delay(2):
				for i, e in enumerate(q, 1):
					if searched >= 1 or i > 12:
						break
					if "research" in e:
						try:
							await asubmit(ytdl.extract_single, e, timeout=18)
							e.pop("research", None)
							searched += 1
						except:
							e.pop("research", None)
							print_exc()
							break
						e.pop("id", None)
					if "research" not in e and not e.get("duration") and "stream" in e:
						e["duration"] = await asubmit(get_duration, e["stream"])

	# Delays audio player display message by 15 seconds when a user types in the target channel
	async def _typing_(self, channel, user, **void):
		if getattr(channel, "guild", None) is None:
			return
		if channel.guild.id in self.players and user.id != self.bot.id:
			auds = self.players[channel.guild.id]
			if auds.player is not None and channel.id == auds.channel.id:
				t = utc() + 15
				if auds.player.time < t:
					auds.player.time = t

	# Delays audio player display message by 10 seconds when a user sends a message in the target channel
	async def _send_(self, message, **void):
		if message.guild.id in self.players and message.author.id != self.bot.id:
			auds = self.players[message.guild.id]
			if auds.player is not None and message.channel.id == auds.channel.id:
				t = utc() + 10
				if auds.player.time < t:
					auds.player.time = t

	# Makes 1 attempt to disconnect a single member from voice.
	@tracebacksuppressor(discord.Forbidden)
	async def _dc(self, member):
		await member.move_to(None)

	def update_vc(self, guild):
		m = guild.me
		if m:
			if guild.id not in self.players:
				if m.voice is not None:
					acsi = AudioClientSubInterface.from_guild(guild)
					if acsi is not None:
						return asubmit(acsi.kill)
					return guild.change_voice_state(channel=None)
			else:
				if m.voice is not None:
					vc_ = m.voice.channel
					perm = m.permissions_in(vc_)
					if perm.mute_members:
						if vc_.type is discord.ChannelType.stage_voice:
							if m.voice.suppress or m.voice.requested_to_speak_at:
								return self.bot.audio.players[guild.id].speak()
						elif m.voice.deaf or m.voice.mute or m.voice.afk:
							return m.edit(mute=False)
		return emptyfut

	# Updates all voice clients
	@tracebacksuppressor(SemaphoreOverflowError)
	async def __call__(self, guild=None, **void):
		bot = self.bot
		async with self._semaphore:
			# Ensure all voice clients are not muted, disconnect ones without matching audio players
			if guild is not None:
				create_task(self.update_vc(guild))
			else:
				[create_task(self.update_vc(g)) for g in bot.cache.guilds.values()]
		# Update audio players
		if guild is not None:
			if guild.id in self.players:
				auds = self.players[guild.id]
				esubmit(auds.update)
		else:
			async with Delay(0.5):
				a = 1
				futs = deque()
				for g in tuple(self.players):
					with tracebacksuppressor(KeyError):
						auds = self.players[g]
						futs.append(asubmit(auds.update, priority=True))
						futs.append(create_task(self.research(auds)))
						if auds.queue and not auds.paused and "dailies" in bot.data:
							if auds.ts is not None and auds.acsi:
								for member in auds.acsi.channel.members:
									if member.id != bot.id:
										vs = member.voice
										if vs is not None and not vs.deaf and not vs.self_deaf:
											bot.data.users.add_gold(member, 0.25)
											bot.data.dailies.progress_quests(member, "music", utc() - auds.ts)
							auds.ts = utc()
						else:
							auds.ts = None
					if not a & 15:
						await asyncio.sleep(0.2)
					a += 1
				for fut in futs:
					with tracebacksuppressor:
						await fut
			await bot.audio.asubmit("ytdl.update()")
		esubmit(ytd.update)
		esubmit(ytdl.update_dl, priority=True)
		if not self.backup_sem.busy and not self.destroyed:
			async with self.backup_sem:
				await self.backup()

	def _announce_(self, *args, **kwargs):
		for auds in self.players.values():
			if auds.queue and not auds.paused:
				esubmit(auds.announce, *args, aio=True, **kwargs)

	backup_sem = Semaphore(1, 0, rate_limit=30)
	async def backup(self, force=False):
		if self.bot.ready:
			if not ("blacklist" in self.bot.data and self.bot.data.blacklist.get(0)):
				self.clear()
		for auds in tuple(self.players.values()):
			if not auds.acsi or not auds.queue:
				continue
			d, _ = await asubmit(auds.get_dump, True, True)
			self.data[auds.acsi.channel.id] = dict(dump=d, channel=auds.text.id)
			self.update(auds.acsi.channel.id)
		if force:
			await asubmit(self.update, force=True, priority=True)

	# Stores all currently playing audio data to temporary database when bot shuts down
	async def _destroy_(self, **void):
		self.destroyed = True
		await self.backup(force=True)
		for file in tuple(ytdl.cache.values()):
			if not file.loaded:
				await asubmit(file.destroy)
		for auds in tuple(self.players.values()):
			if auds.queue and not auds.paused:
				reason = "🎵 Temporarily disconnecting for maintenance"
				if auds.queue:
					reason += " (Queue saved)."
				else:
					reason += "."
				reason += " Apologies for any inconvenience! 🎵"
			else:
				reason = ""
			with tracebacksuppressor:
				await asubmit(auds.kill, reason=css_md(reason) if reason else None, remove=False)

	# Restores all audio players from temporary database when applicable
	async def _bot_ready_(self, bot, **void):
		globals()["bot"] = bot
		ytdl.bot = bot
		try:
			await asubmit(subprocess.check_output, ("./ffmpeg",))
		except subprocess.CalledProcessError:
			pass
		except FileNotFoundError:
			print("WARNING: FFmpeg not found. Unable to convert and play audio.")
		while bot.audio is None:
			await asyncio.sleep(0.5)
		await wrap_future(bot.audio.fut)
		count = 0
		for file in os.listdir("cache"):
			if file.startswith("~") and file not in ytdl.cache:
				ytdl.cache[file] = f = await asubmit(AudioFileLink, file, "cache/" + file, wasfile=True)
				count += 1
		if count:
			print(f"Successfully reinstated {count} audio file{'s' if count != 1 else ''}")
		if "blacklist" in self.bot.data and self.bot.data.blacklist.get(0):
			return
		for k, v in self.data.items():
			with tracebacksuppressor:
				vc = await bot.fetch_channel(k)
				channel = await bot.fetch_channel(v["channel"])
				guild = channel.guild
				bot = bot
				user = bot.user
				perm = inf
				name = "dump"
				argv = v["dump"]
				flags = "h"
				message = cdict(attachments=None)
				for dump in bot.commands.dump:
					if argv.get("queue"):
						loading = True
					else:
						stats = dict(CustomAudio.defaults)
						stats.update(argv.get("stats", {}))
						loading = stats != CustomAudio.defaults
					if loading:
						print("Auto-loading queue of", len(argv["queue"]), "items to", guild)
						create_task(dump(guild, channel, user, bot, perm, name, argv, flags, message, vc=vc, noannouncefirst=True))
		# self.data.clear()


class UpdatePlaylists(Database):
	name = "playlists"


class UpdateAudioSettings(Database):
	name = "audiosettings"
