import asyncio
from collections import deque
import concurrent.futures
from concurrent.futures import Future
import diskcache
import io
import itertools
from math import inf, tau, isfinite, sqrt, ceil
import os
import shutil
import subprocess
import sys
import threading
from traceback import print_exc
from urllib.parse import quote_plus, unquote_plus
import orjson
import niquests
import numpy as np
import psutil
from .asyncs import csubmit, esubmit, asubmit, wrap_future, cst, eloop, Delay
from .types import utc, as_str, alist, cdict, suppress, round_min, cast_id, lim_str, astype
from .smath import log2lin
from .util import (
	tracebacksuppressor, force_kill, AUTH, CACHE_PATH, EvalPipe, Request, api,
	italics, ansi_md, colourise, colourise_brackets, maybe_json, select_and_loads,
	is_url, unyt, url2fn, get_duration, get_duration_2, CachingTeeFile,
	rename, uhash, expired, is_youtube_stream, b64,  # noqa: F401
)
from .audio_downloader import AudioDownloader
VC_TIMEOUT = 13

# VERY HACKY removes deprecated audioop dependency for discord.py; this would cause volume transformations to fail but we use FFmpeg for them anyway
sys.modules["audioop"] = sys.__class__("audioop")
import discord  # noqa: E402

tracebacksuppressor.fn = print_exc

ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
if ADDRESS == "0.0.0.0":
	ADDRESS = "127.0.0.1"

# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000
MAX_BITRATE = 160000
MAX_QUEUE = 262144


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	interface2 = EvalPipe.from_stdin(glob=globals())
	print = interface.print

ytdl = None
client_fut = Future()
ytdl_fut = esubmit(AudioDownloader, workers=1)

class AudioPlayer(discord.AudioSource):
	"""
	Custom audio player for Discord voice channels, extending discord.AudioSource.
	This class manages audio playback, queueing, voice connections, and various audio effects/settings
	for a Discord bot. It handles joining/leaving voice channels, audio streaming, queue manipulation,
	and application of effects like pitch, speed, reverb, etc.
	Attributes:
		cache (FileHashDict): Cache for voice channel players.
		defaults (dict): Default audio settings (volume, reverb, pitch, etc.).
		players (dict): Mapping of guild IDs to AudioPlayer instances.
		waiting (dict): Futures for pending voice connections.
		futs (dict): Additional futures for operations.
		sources (dict): Audio sources.
		users (dict): Cached users.
		fetched (dict): Fetched user voice states per guild.
		vc (discord.VoiceClient): The voice client connection.
		args: Additional arguments (not explicitly used).
		emptyopus (bytes): Empty Opus packet for silence.
		silent (bool): Indicates if currently silent.
		last_played (float): Timestamp of last audio playback.
		last_activity (float): Timestamp of last activity (set to infinity).
	Class Methods:
		join(cls, vcc, channel=None, user=None, announce=0): Joins or reconnects to a voice channel.
		join_into(self, vcc, announce=0): Internal method to connect to voice.
		find_user(cls, guild, user): Finds and fetches user voice state.
		ensure_speak(cls, vcc): Ensures the bot can speak in stage channels.
		speak(cls, vcc): Requests to speak in stage channels.
		force_disconnect(cls, guild): Forcefully disconnects from voice.
		from_guild(cls, guild): Retrieves or creates player from guild.
		disconnect(cls, guild, announce=False, cid=None): Disconnects from voice.
		fetch_user(cls, u_id): Fetches and caches a user.
	Instance Methods:
		__init__(self, vcc=None, channel=None, queue=[], settings={}): Initializes the player.
		__getattr__(self, k): Attribute getter with fallback to voice client or playing source.
		epos (property): Current position and duration of playing audio.
		reverse (property): Whether audio is playing in reverse.
		construct_options(self, full=True): Builds FFmpeg options from settings.
		announce_play(self, entry): Announces the currently playing entry.
		announce(self, s, dump=False, channel=None): Sends announcement message.
		get_dump(self): Dumps queue and settings to JSON.
		load_dump(self, b, uid=None, universal=False): Loads queue from dump.
		update_activity(self): Updates activity and handles timeouts for empty channels.
		update_streaming(self): Updates streaming status and handles empty queue timeouts.
		read(self): Reads audio data, handles skipping and silence.
		enqueue(self, items, start=-1, stride=1): Adds items to the queue.
		skip(self, indices=0, loop=False, repeat=False, shuffle=False): Skips queue items.
		seek(self, pos=0): Seeks to position in current audio.
		ensure_play(self, force=0): Ensures playback is active.
		clear(self): Clears queue and stops audio.
		refresh(self): Resets update tasks.
		backup(self): Backs up player state.
		is_opus(self): Returns True (Opus audio).
		cleanup(self): Cleans up resources (calls clear).
	Note: This class integrates with Discord.py for voice functionality and uses custom utilities
	like alist, cdict, and audio effects processing.
	"""

	# cache = FileHashDict(path="cache/vc.players")
	cache = diskcache.Cache(directory=f"{CACHE_PATH}/audio", expiry=86400 * 7)
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
		"bitrate": 160000,
		"pause": False,
		"loop": False,
		"repeat": False,
		"shuffle": False,
		"quiet": False,
		"stay": False,
	}
	players = {}
	waiting = {}
	futs = {}
	sources = {}
	users = {}
	fetched = {}
	vc = None
	args = None
	# Empty opus packet data
	emptyopus = b"\xfc\xff\xfe"
	silent = False
	last_read = None
	last_played = 0
	last_activity = inf

	@classmethod
	async def join(cls, vcc, channel=None, user=None, announce=0, force=False):
		globals()["client"] = await wrap_future(client_fut)
		cid = cast_id(vcc)
		vcc = client.get_channel(cid)
		if not vcc:
			vcc = await client.fetch_channel(cid)
		if channel is not None:
			cid = cast_id(channel)
			channel = client.get_channel(cid)
			if not channel:
				channel = await client.fetch_channel(cid)
		gid = vcc.guild.id
		if user is not None:
			await cls.find_user(vcc.guild, user)
		self = None
		try:
			self = cls.players[gid]
		except KeyError:
			pass
		else:
			if not self or self.vc and (self.vc.channel.id != vcc.id and (force or getattr(user, "voice", None)) or not self.vc.is_connected()):
				await cls.force_disconnect(vcc.guild)
				await asyncio.sleep(1)
				if self:
					self.vc = None
		if self and self.vc:
			await cls.ensure_speak(vcc)
			self.refresh()
			self.channel = channel
			return self
		try:
			fut = cls.waiting[gid]
		except KeyError:
			pass
		else:
			try:
				self = await asyncio.wait_for(wrap_future(fut), timeout=VC_TIMEOUT)
			except asyncio.TimeoutError:
				if self:
					self.vc = None
				raise
		if self and self.vc:
			await cls.ensure_speak(vcc)
			self.refresh()
			self.channel = channel
			return self
		if not self:
			self = cls(vcc, channel)
		else:
			self.channel = channel
		self.refresh()
		cls.waiting[gid] = Future()
		cls.futs[gid] = csubmit(self.join_into(vcc, announce=announce))
		return self

	async def join_into(self, vcc, announce=0):
		print(self, vcc)
		if self.fut.done():
			self.fut = Future()
		gid = vcc.guild.id
		member = vcc.guild.me
		try:
			await self.ensure_speak(vcc)
			if not self.vc or not self.vc.is_connected():
				try:
					self.vc = await vcc.connect(timeout=VC_TIMEOUT, reconnect=True)
				except (discord.ClientException, asyncio.TimeoutError):
					await self.force_disconnect(vcc.guild)
					self.vc = await vcc.connect(timeout=VC_TIMEOUT, reconnect=True)
		except Exception as ex:
			try:
				cst(self.waiting[gid].set_exception, ex)
			except KeyError:
				pass
			self.fut.set_exception(ex)
			self.players.pop(gid, None)
			raise
		else:
			self.last_played = utc()
			self.players[gid] = self
			try:
				cst(self.waiting[gid].set_result, self)
			except KeyError:
				pass
			self.fut.set_result(None)
			await self.ensure_speak(vcc)
			if member and member.voice is not None and vcc.permissions_for(member).mute_members:
				if member.voice.deaf or member.voice.self_deaf or member.voice.mute or member.voice.afk:
					csubmit(member.edit(mute=False))
			if announce:
				connected = "connected" if announce == 1 else "reconnected"
				s = ansi_md(
					f"{colourise('🎵', fg='blue')}{colourise()} Successfully {connected} to {colourise(self.channel.guild, fg='magenta')}{colourise()}. {colourise('🎵', fg='blue')}{colourise()}"
				)
				csubmit(self.announce(s))
			return self
		finally:
			self.waiting.pop(gid, None)
			self.futs.pop(gid, None)

	@classmethod
	async def find_user(cls, guild, user):
		uid = cast_id(user)
		if uid == client.user.id:
			return
		gid = cast_id(guild)
		guild = client.get_guild(gid) or await client.fetch_guild(gid)
		member = guild.get_member(uid)
		if uid not in cls.fetched.get(gid, ()) and (not member or not member.voice):
			print(uid, member, cls.fetched.get(gid))
			cls.fetched.setdefault(gid, set()).add(uid)
			# Manually fetch voice state, as discord.py does not do this automatically
			try:
				data = await Request(
					f"https://discord.com/api/{api}/guilds/{gid}/voice-states/{uid}",
					authorise=True,
					json=True,
					aio=True,
					timeout=VC_TIMEOUT,
				)
			except ConnectionError:
				pass
			else:
				print(data)
				client._connection.parse_voice_state_update(data)
				return data

	@classmethod
	async def ensure_speak(cls, vcc):
		member = vcc.guild.me
		if not member:
			return
		if member.voice is not None and vcc.permissions_for(member).mute_members and vcc.type is discord.ChannelType.stage_voice:
			if member.voice.suppress or member.voice.requested_to_speak_at:
				await cls.speak(vcc)

	@classmethod
	async def speak(cls, vcc):
		"""Overrides speaking permissions in a stage voice channel."""
		return await Request(
			f"https://discord.com/api/{api}/guilds/{vcc.guild.id}/voice-states/@me",
			method="PATCH",
			authorise=True,
			data={"suppress": False, "request_to_speak_timestamp": None, "channel_id": vcc.id},
			aio=True,
		)

	@classmethod
	async def force_disconnect(cls, guild):
		"""Forcibly disconnects the bot from a voice channel, regardless of the current state of discord.py's cache."""
		AP = cls.players.pop(guild.id, None)
		if AP:
			AP.vc = None
		fut = cls.waiting.pop(guild.id, None)
		if fut:
			fut.cancel()
		resp = await Request(
			f"https://discord.com/api/{api}/guilds/{guild.id}/members/{client.user.id}",
			method="PATCH",
			authorise=True,
			data={"channel_id": None},
			aio=True,
		)
		# This removes the cached voice state, which is necessary for reconnecting to a voice channel after a disconnect
		vc = client._connection._voice_clients.pop(guild.id, None)
		if vc and vc.ws:
			try:
				await asyncio.wait_for(vc.ws.close(), timeout=VC_TIMEOUT)
			except Exception:
				print_exc()
		if vc and vc.socket:
			vc.socket.close()
		client._connection._remove_voice_client(guild.id)
		return resp

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
			try:
				self = fut.result(timeout=VC_TIMEOUT)
			except concurrent.futures.TimeoutError:
				self = None
			if self and self.vc:
				return self
		guild = client.get_guild(gid)
		if not guild or not guild.me or not guild.me.voice or not guild.me.voice.channel:
			raise KeyError(gid)
		vcc = guild.me.voice.channel
		self = cls(vcc)
		cls.waiting[gid] = Future()
		# csubmit(self.join_into(vcc))
		return self

	@classmethod
	async def disconnect(cls, guild, announce=False, cid=None):
		gid = cast_id(guild)
		guild = client.get_guild(gid)
		# if not guild or not guild.me or not guild.me.voice:
		# 	raise KeyError(gid)
		si = StopIteration("Voice disconnected.")
		wait = cls.waiting.pop(gid, None)
		if wait and not wait.done():
			wait.set_exception(si)
		self = None
		try:
			self = cls.players.pop(gid)
		except KeyError:
			await cls.force_disconnect(guild)
		else:
			if not self.fut.done():
				self.fut.set_exception(si)
			if self.vc:
				self.vc.stop()
				await self.vc.disconnect()
		if not self:
			if not cid or not client.get_channel(cid):
				return
		else:
			print("Clearing queue to disconnect...")
			self.settings.update(self.defaults)
			self.clear()
		if announce:
			channel = self.channel if self else client.get_channel(cid)
			s = ansi_md(
				f"{colourise('🎵', fg='blue')}{colourise()} Successfully disconnected from {colourise(channel.guild, fg='magenta')}{colourise()}. {colourise('🎵', fg='blue')}{colourise()}"
			)
			return await cls.announce(self, s, channel=channel)

	async def leave(self, reason=None, dump=False):
		csubmit(self.disconnect(self.vcc.guild))
		if not self.channel:
			return
		r = f": {colourise(reason, fg='yellow')}{colourise()}" if reason else ""
		if dump:
			r += " (use ~load to restore)"
		s = ansi_md(
			f"{colourise('🎵', fg='blue')}{colourise()} Automatically disconnected from {colourise(self.channel.guild, fg='magenta')}{colourise()}{r}. {colourise('🎵', fg='blue')}{colourise()}"
		)
		return await self.announce(s, dump=dump)

	@classmethod
	async def fetch_user(cls, u_id):
		globals()["client"] = await wrap_future(client_fut)
		try:
			return cls.users[u_id]
		except KeyError:
			cls.users[u_id] = user = await client.fetch_user(u_id)
		return user

	def __init__(self, vcc=None, channel=None, queue=[], settings={}):
		self.listening = None
		self.last_played = utc()
		self.queue = alist(queue)
		self.settings = cdict(self.defaults)
		self.settings.update(settings)
		self.playing = deque(maxlen=2)
		self.fut = Future()
		self.ensure_lock = threading.RLock()
		self.vcc = vcc
		if vcc:
			self.vc = client.get_guild(cast_id(vcc.guild)).voice_client
		self.channel = channel

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not self.vc and not self.fut.done():
			self.fut.result(timeout=60)
		try:
			return getattr(self.vc, k)
		except AttributeError:
			if not self.playing:
				raise
		return getattr(self.playing[0], k)

	@property
	def epos(self):
		if not self.playing or not self.playing[0]:
			return 0, 0
		p = self.playing[0].pos / 50
		d = self.playing[0].af.duration or inf
		return min(p, d), d

	@property
	def reverse(self):
		return self.settings.speed < 0

	# Constructs array of FFmpeg options using the audio settings. FFmpeg's arguments are not directly exposed to the user, as they are not very user-friendly.
	def construct_options(self, full=True):
		settings = self.settings
		for k, v in settings.items():
			settings[k] = round_min(v)
		# Pitch setting is in semitones, so frequency is on an exponential scale (2^(1/12) per semitone)
		pitchscale = 2 ** ((settings.pitch + settings.resample) / 12)
		reverb = settings.reverb
		volume = settings.volume
		args = []
		# FIR sample for reverb
		if reverb:
			args.extend(["-i", "misc/SNB3,0all.wav"])
		options = deque()
		# This must be first, else the filter will not initialize properly
		if not isfinite(settings.compressor):
			options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
		# Reverses song, this may be very resource consuming as FFmpeg will read the entire file into memory
		# TODO: Implement chunked reverse
		if self.reverse:
			options.append("areverse")
		# Adjusts song tempo relative to speed, pitch, and nightcore settings
		if pitchscale != 1 or settings.speed != 1:
			speed = abs(settings.speed) / pitchscale
			speed *= 2 ** (settings.resample / 12)
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
		# TODO: Make this sound better
		if settings.chorus:
			chorus = abs(settings.chorus)
			ch = min(16, chorus)
			A = B = C = D = ""
			for i in range(ceil(ch)):
				neg = ((i & 1) << 1) - 1
				i = 1 + i >> 1
				i *= settings.chorus / ceil(chorus)
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
				depth = (0.55 + i * 0.25 * neg) % max(1, settings.chorus) + 0.15
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
		if settings.compressor:
			comp = min(8000, abs(settings.compressor * 10))
			while abs(comp) > 1:
				c = min(20, comp)
				try:
					comp /= c
				except ZeroDivisionError:
					comp = 1
				mult = str(round((c * sqrt(2)) ** 0.5, 4))
				options.append(
					"acompressor=mode=" + ("upward" if settings.compressor < 0 else "downward")
					+ ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
				)
		# Bassboost setting, the ratio is currently very unintuitive and definitely needs tweaking
		if settings.bassboost:
			opt = "firequalizer=gain_entry="
			entries = []
			high = 24000
			low = 13.75
			bars = 4
			small = 0
			for i in range(bars):
				freq = low * (high / low) ** (i / bars)
				dB = -(i / (bars - 1) - 0.5) * settings.bassboost * 64
				# dB = log(abs(bb) + 1, 2)
				# if bb < 0:
				# 	dB = -dB
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
			# We pick up to four arbitrary but spaced-out pairs of delay and decay values to produce a natural-sounding effect
			options.append(f"aecho=1:1:400|630:{d[0]}|{d[1]}")
			if d[2] >= 0.05:
				options.append(f"aecho=1:1:870|1150:{d[2]}|{d[3]}")
				if d[4] >= 0.06:
					options.append(f"aecho=1:1:1410|1760:{d[4]}|{d[5]}")
					if d[6] >= 0.07:
						options.append(f"aecho=1:1:2080|2320:{d[6]}|{d[7]}")
		# Pan setting, uses extrastereo and volume filters to balance (as extrastereo will otherwise mess with the overall volume)
		if settings.pan != 1:
			pan = min(10000, max(-10000, settings.pan))
			while abs(abs(pan) - 1) > 0.001:
				p = max(-10, min(10, pan))
				try:
					pan /= p
				except ZeroDivisionError:
					pan = 1
				options.append("extrastereo=m=" + str(p) + ":c=0")
				volume *= 1 / max(1, round(sqrt(abs(p)), 4))
		if volume != 1:
			v = round(log2lin(volume), 7)
			if not isfinite(v):
				v = 1e308
			options.append("volume=" + str(v))
		# Soft clip audio using atan, reverb filter requires -filter_complex rather than -af option
		# Clipping is bypassed if compressor is enabled; instead we use a limiter such that if the volume is also set higher than 100%, the limiter will take effect instead.
		if options:
			if settings.compressor:
				options.append("alimiter")
			elif volume > 1:
				options.append("asoftclip=atan")
			args.append(("-af", "-filter_complex")[bool(reverb)])
			args.append(",".join(options))
		return args

	async def announce_play(self, entry):
		if not self.channel or not self.channel.permissions_for(self.channel.guild.me).send_messages:
			return
		try:
			u = await self.fetch_user(entry.u_id)
			if not u:
				raise KeyError(entry.u_id)
			name = u.display_name
		except (KeyError, AttributeError, discord.NotFound):
			name = "Unknown User"
		s = italics(ansi_md(
			f"{colourise('🎵', fg='blue')}{colourise()} Now playing {colourise_brackets(entry.name, 'red', 'green', 'magenta')}{colourise()}, added by {colourise(name, fg='blue')}{colourise()}! {colourise('🎵', fg='blue')}{colourise()}"
		))
		return await self.announce(s)

	async def announce(self, s, dump=False, channel=None):
		channel = channel or self.channel
		if not dump and self and (not channel or self.settings.get("quiet") or not channel.permissions_for(channel.guild.me).send_messages):
			return
		if dump:
			assert self, "No accessible queue to save!"
			b = self.get_dump()
			dump = discord.File(io.BytesIO(b), filename="dump.json")
		message = await channel.send(lim_str(s, 2000), file=dump or None)
		if channel.permissions_for(channel.guild.me).add_reactions:
			csubmit(message.add_reaction("❎"))
		return message

	def get_dump(self):
		data = dict(queue=self.queue)
		setts = dict(self.settings)
		for k, v in tuple(setts.items()):
			if k not in self.defaults or v == self.defaults[k]:
				setts.pop(k)
		if setts:
			data["settings"] = setts
		elapsed, _length = self.epos
		if elapsed:
			data.setdefault("settings", {})["pos"] = elapsed
		return maybe_json(data)

	def load_dump(self, b, uid=None, universal=False):
		try:
			d = select_and_loads(b, size=268435456)
		except orjson.JSONDecodeError:
			d = [url for url in as_str(b).splitlines() if is_url(url)]
			if not d:
				raise
			d = [dict(name=url2fn(url), url=url) for url in d]
		if isinstance(d, list):
			d = dict(queue=d)
		for e in d["queue"]:
			if uid:
				e.setdefault("u_id", uid)
			e["url"] = unyt(e["url"])
		settings = d.get("settings")
		if not settings:
			settings = d.get("stats", {})
			if settings.get("bitrate"):
				settings["bitrate"] *= 100
		# In universal mode, all settings are copied to the player
		if not universal:
			settings.pop("pause", None)
			settings.pop("pos", None)
		pos = settings.pop("pos", None)
		self.settings.update({k: v for k, v in settings.items() if k in self.defaults})
		self.queue.fill(map(cdict, d["queue"]))
		if pos is not None:
			esubmit(self.seek, pos)
		else:
			esubmit(self.ensure_play, 2)
		return list(self.queue)

	updating_activity = None
	def update_activity(self):
		"""Updates whether there are people listening; timeout after 6 minutes of inactivity."""
		if self.updating_activity:
			self.updating_activity.cancel()
			self.updating_activity = None
		if self.settings.stay:
			return
		if not self.vc and not self.fut.done():
			return
		connected = self.vcc.guild.me.voice or interface.run(f"bool(client.get_channel({self.vcc.id}).guild.me.voice)")
		if connected:
			# Handle special case of only deafened users; they are not counted as listeners but will still keep the bot in the channel, paused instead
			listeners = sum(not m.bot and bool(m.voice) and not (m.voice.deaf or m.voice.self_deaf) for m in self.vcc.members)
			if listeners == 0:
				self.updating_activity = csubmit(self._updating_activity())
			elif not self.settings.pause:
				self.resume()
	async def _updating_activity(self):
		self.pause()
		await asyncio.sleep(360)
		if self is not self.players.get(self.vcc.guild.id):
			return
		connected = self.vcc.guild.me.voice or interface.run(f"bool(client.get_channel({self.vcc.id}).guild.me.voice)")
		if connected:
			listeners = sum(not m.bot and bool(m.voice) for m in self.vcc.members)
			if listeners == 0:
				await self.leave("Channel empty", dump=len(self.queue) > 0)

	updating_streaming = None
	def update_streaming(self):
		"""Updates whether we're streaming audio; timeout after 16 minutes of inactivity."""
		if self.updating_streaming:
			self.updating_streaming.cancel()
			self.updating_streaming = None
		if self.settings.stay:
			return
		if not self.vc and not self.fut.done():
			return
		connected = self.vcc.guild.me.voice or interface.run(f"bool(client.get_channel({self.vcc.id}).guild.me.voice)")
		if len(self.queue) == 0 and connected:
			self.updating_streaming = csubmit(self._updating_streaming())
	async def _updating_streaming(self):
		await asyncio.sleep(960)
		if self is not self.players.get(self.vcc.guild.id):
			return
		connected = self.vcc.guild.me.voice or interface.run(f"bool(client.get_channel({self.vcc.id}).guild.me.voice)")
		if len(self.queue) == 0 and connected:
			await self.leave("Queue empty")

	def read(self):
		"""Overrides discord.AudioSource read method to read audio data from the current playing source. Handles EOF, seamlessly skipping songs that have ended, and removing the need for a new play() call."""
		if not self.playing or self.settings.pause:
			if self.silent:
				try:
					self.vc.pause()
				except Exception:
					pass
			self.silent = True
			return self.emptyopus * 3
		new = False
		out = self.emptyopus
		try:
			if not self.queue or not self.playing:
				raise IndexError
			new = self.playing[0].new
			self.last_read = self.last_read or esubmit(self.playing[0].read)
			out = self.last_read.result(timeout=0.05)
		except concurrent.futures.TimeoutError:
			pass
		except (StopIteration, IndexError, discord.oggparse.OggError):
			self.last_read = None
		except Exception:
			print_exc()
			self.last_read = None
		else:
			self.last_read = None
			if out and new:
				self.playing[0].new = False
				csubmit(self.announce_play(self.queue[0]))
			self.last_played = utc()
		if (self.playing and self.queue) and (not out or self.playing[0].pos / 50 >= (self.queue[0].get("end") or inf)):
			self.skip(0, loop=self.settings.loop, repeat=self.settings.repeat, shuffle=self.settings.shuffle)
			if not out and self.playing:
				return self.read()
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

	def enqueue(self, items, start=-1, stride=1):
		"""Inserts items into the queue, respecting the start and stride parameters where applicable."""
		if len(items) > MAX_QUEUE:
			items = astype(items, (list, alist))[:MAX_QUEUE]
		items = list(astype(e, cdict) for e in items)
		if stride == 1 and (start == -1 or start > len(self.queue) or not self.queue):
			self.queue.extend(items)
		else:
			self.queue.rotate(-start)
			rotpos = start
			if stride == 1:
				self.queue.extend(items)
				rotpos += len(items)
			else:
				temp = alist([None] * (len(items) * abs(stride)))
				temp[::stride] = items
				sli = temp.view == None  # noqa: E711
				inserts = self[:len(items) * (abs(stride) - 1)]
				i = -1
				while not sli[i] or np.sum(sli) > len(inserts):
					sli[i] = False
					i -= 1
				print(len(temp), len(sli), len(inserts))
				temp.view[sli] = inserts
				temp = temp.view[temp.view != None]  # noqa: E711
				temp = np.concatenate([temp, self[len(items) * (abs(stride) - 1):]])
				self.queue.fill(temp)
			self.queue.rotate(rotpos)
		self.last_played = utc()
		esubmit(self.ensure_play)
		return self

	shuffler = 0
	def skip(self, indices=0, loop=False, repeat=False, shuffle=False):
		"""Skips items in the queue, respecting the loop, repeat, and shuffle parameters where applicable."""
		with self.ensure_lock:
			if isinstance(indices, int):
				indices = [indices]
			resp = []
			if not repeat:
				resp = self.queue.pops(indices)
			if 1 in indices:
				if len(self.playing) > 1:
					self.playing.pop().close()
			if 0 in indices:
				if self.playing:
					self.playing.popleft().close()
			if loop:
				if shuffle:
					if not self.shuffler:
						self.queue[2:].shuffle()
					shuffler = self.shuffler + 1
					if shuffler >= len(self.queue):
						shuffler = 0
					self.shuffler = shuffler
				self.queue.extend(resp)
		esubmit(self.ensure_play)
		return resp

	def seek(self, pos=0):
		"""Seeks to a specific position in the current playing audio."""
		if not self.queue:
			return
		with self.ensure_lock:
			self.pause()
			source = AF.load(self.queue[0], asap=pos <= 8).create_reader(self, pos=pos)
			source.new = False
			if self.playing:
				self.playing[0], _ = source, self.playing[0].close()
			else:
				self.playing.append(source)
			if not self.settings.pause:
				self.resume()
		return self.ensure_play()

	# force=0: normal (only play if not currently playing)
	# force=1: always regenerate readers (this updates audio settings)
	# force=2: always restarts songs from beginning
	def ensure_play(self, force=0):
		"""Ensures that the player is playing audio when it should be. Called when the queue is modified, reloaded, or resumed."""
		with self.ensure_lock:
			pos = None
			if len(self.queue) > MAX_QUEUE + 2048:
				self.queue.fill(self.queue[1 - MAX_QUEUE:].appendleft(self.queue[0]))
			elif len(self.queue) > MAX_QUEUE:
				self.queue.rotate(-1)
				while len(self.queue) > MAX_QUEUE:
					self.queue.pop()
				self.queue.rotate(1)
			if len(self.playing) > 1 and (force or (len(self.queue) > 1 and self.playing[1].af.url != self.queue[1].url)):
				self.playing.pop().close()
			if self.playing and (force or self.queue and self.playing[0].af.url != self.queue[0].url):
				temp = self.playing.popleft().close()
				if force == 1 and temp.af.url == self.queue[0].url:
					pos = temp.pos / 50
			if not self.playing and self.queue:
				entry = self.queue[0]
				try:
					source = AF.load(entry).create_reader(self, pos=pos if pos is not None else entry.get("start", 0))
				except Exception as ex:
					print_exc()
					s = italics(ansi_md(
						f"{colourise('❗', fg='blue')}{colourise()} An error occured while loading {colourise_brackets(entry.name, 'red', 'green', 'magenta')}{colourise()}, and it has been removed automatically. {colourise('❗', fg='blue')}{colourise()}\n"
						+ f"Exception: {colourise_brackets(lim_str(repr(ex), 1024), 'red', 'magenta', 'yellow')}"
					))
					csubmit(self.announce(s))
					return self.skip()
				if pos is not None:
					source.new = False
				self.playing.append(source)
			if self.playing and not self.settings.pause:
				self.fut.result(timeout=60)
				self.last_played = utc()
				if self.vc and not self.vc.is_playing():
					self.vc.play(self)
				self.resume()
			elif self.settings.pause and self.vc.is_playing():
				self.vc.pause()
			if not self.ensuring or self.ensuring.done():
				self.ensuring = esubmit(self.ensure_next)
		if not self.queue:
			self.update_streaming()
		else:
			self.update_activity()

	ensuring = None
	def ensure_next(self):
		with tracebacksuppressor:
			if len(self.playing) == 1 and len(self.queue) > 1:
				entry = self.queue[1]
				if not entry.get("duration") or not entry.duration <= 3960:
					return
				asap = entry["duration"] > (self.epos[1] - self.epos[0]) * 8
				try:
					source = AF.load(entry, asap=asap).create_reader(self, pos=entry.get("start", 0))
				except Exception as ex:
					print_exc()
					s = italics(ansi_md(
						f"{colourise('❗', fg='blue')}{colourise()} An error occured while loading {colourise_brackets(entry.name, 'red', 'green', 'magenta')}{colourise()}, and it has been removed automatically. {colourise('❗', fg='blue')}{colourise()}\n"
						+ f"Exception: {colourise_brackets(repr(ex), 'red', 'magenta', 'yellow')}"
					))
					csubmit(self.announce(s))
					return self.skip(1)
				if len(self.playing) == 1 and len(self.queue) > 1 and self.queue[1].url == source.af.url:
					self.playing.append(source)

	def clear(self):
		"""Clears the queue and stops all audio."""
		for entry in tuple(self.playing):
			try:
				entry.close()
			except Exception:
				print_exc()
		self.playing.clear()
		self.queue.clear()
		self.refresh()

	def refresh(self):
		if self.updating_activity:
			self.updating_activity.cancel()
		if self.updating_streaming:
			self.updating_streaming.cancel()

	def backup(self):
		if self.queue and self.vcc:
			AP.cache[self.vcc.guild.id] = [self.vcc.id, self.channel and self.channel.id, self.get_dump(), [m.id for m in self.vcc.members]]
		return len(self.queue)

	def is_opus(self):
		return True

	def cleanup(self):
		return
		# print("Clearing queue for cleanup...")
		# return self.clear()

AP = AudioPlayer


class AudioFile:
	"""
	A class to represent an audio file with caching and streaming capabilities.
	Attributes
	----------
	cached : dict
		A class-level dictionary to cache audio files.
	Methods
	-------
	__str__():
		Returns a string representation of the AudioFile object.
	load(entry, asap=True):
		Class method to load an audio file from a given entry.
	live():
		Property to check if the audio file is a live stream.
	open():
		Creates a reader object that reads bytes or opus packets from the file.
	create_reader(auds, pos=0):
		Creates a reader, selecting from direct opus file, single piped FFmpeg, or double piped FFmpeg.
	proc_expired():
		Property to check if the FFmpeg process has expired.
	"""

	temporary = None
	duration = None
	cached = {}

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}, linked to {self.path}>"

	@classmethod
	def load(cls, entry, asap=True):
		url = unyt(entry["url"])
		self = None
		try:
			fut = cls.cached[url]
		except KeyError:
			pass
		else:
			try:
				self = fut.result(timeout=60)
			except Exception:
				pass
			else:
				if isinstance(self.stream, str):
					if not is_url(self.stream) and not os.path.exists(self.stream):
						self = None
					if is_url(self.stream) and expired(self.stream):
						self = None
				if self:
					if isinstance(self.stream, str) and not is_url(self.stream) and os.path.exists(self.stream):
						if not entry.get("duration"):
							entry["duration"] = get_duration(self.stream) or self.duration
							name, _url = map(unquote_plus, self.stream.rsplit("/", 1)[-1].rsplit(" ", 1))
							entry["name"] = name
					return self
		cls.cached[url] = Future()
		try:
			self = cls()
			self.url = url
			# Sometimes entries may have a corrupted name, so we need to fetch a new search using the URL
			if not entry.get("duration") and (not entry.get("name") or entry["name"] == entry["url"].rsplit("/", 1)[-1].split("?", 1)[0]):
				results = ytdl.search(entry["url"])
				if results:
					entry.update(results[0])
			name = lim_str(quote_plus(entry.get("name") or url2fn(url)), 80)
			self.path = f"{CACHE_PATH}/audio/{name} {uhash(url)}.opus"
			if os.path.exists(self.path) and os.path.getsize(self.path):
				self.stream = self.path
				if not entry.get("duration"):
					entry["duration"] = self.duration = get_duration(self.stream)
					return self
				self.duration = get_duration(self.stream)
				if abs(entry["duration"] - self.duration) < 1:
					entry["duration"] = self.duration
					return self
			stream, codec, duration, channels = ytdl.get_audio(entry, asap=asap)
			name = lim_str(quote_plus(entry.get("name") or url2fn(url)), 80)
			self.path = f"{CACHE_PATH}/audio/{name} {uhash(url)}.opus"
			if not is_url(stream) and codec == "opus" and channels == 2:
				print("DL:", stream, self.path)
				rename(stream, self.path)
				self.stream = self.path
				self.duration = entry["duration"] = get_duration(self.stream) or duration
				return self
			if not duration or duration > 36960:
				self.stream = stream
				self.duration = entry["duration"] = duration
				return self
			ffmpeg = "ffmpeg"
			sample_rate = SAMPLE_RATE
			ba = "160k" if is_url(stream) or os.path.getsize(stream) <= 10485760 else "108k"
			cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-f", "opus", "-c:a", "libopus", "-ar", str(sample_rate), "-ac", "2", "-b:a", ba, "-vbr", "on", "-frame_duration", "20", "-"]
			if ba == "160k" and codec == "opus" and channels == 2:
				cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-vn", "-i", stream, "-map_metadata", "-1", "-f", "opus", "-c:a", "copy", "-"]
			if is_url(stream):
				cmd = [ffmpeg, "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"] + cmd[1:]
			print(cmd)
			proc = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			assert proc.is_running(), "FFmpeg process failed to start"
			err = CachingTeeFile(proc.stderr).open()

			def callback():
				self.stream = self.path
				self.duration = entry["duration"] = get_duration(self.stream) or self.duration
				if self.temporary:
					try:
						os.remove(self.temporary)
					except Exception:
						pass

			stream = self.stream = CachingTeeFile(proc.stdout, self.path, callback=callback)
			if not stream.open().read(1024):
				proc.terminate()
				ex = as_str(err.read())
				if asap and "Server returned 403 Forbidden (access denied)" in str(ex):
					stream, codec, duration, channels = ytdl.get_audio(entry, asap=False)
					name = lim_str(quote_plus(entry.get("name") or url2fn(url)), 80)
					self.path = f"{CACHE_PATH}/audio/{name} {uhash(url)}.opus"
					print("DL:", self.path)
					assert not is_url(stream) and codec == "opus" and channels == 2, f"Unexpected stream format: {stream} {codec} {channels}"
					rename(stream, self.path)
					self.stream = self.path
					self.duration = entry["duration"] = get_duration(self.stream) or duration
					return self
				raise RuntimeError("File was empty!")
			print("DL:", self.path)
			self.duration = entry["duration"] = duration
			return self
		except Exception as ex:
			cls.cached[url].set_exception(ex)
			raise
		finally:
			if not cls.cached[url].done():
				cls.cached[url].set_result(self)

	@property
	def live(self):
		return isinstance(self.stream, str) and is_url(self.stream) and self.stream

	# Creates a reader object that either reads bytes or opus packets from the file.
	def open(self):
		if isinstance(self.stream, str):
			assert not is_url(self.stream) and os.path.exists(self.stream) and os.path.getsize(self.stream), "File not found or empty"
			f = open(self.stream, "rb")
		elif hasattr(self.stream, "open"):
			f = self.stream.open()
		else:
			f = self.stream
		it = discord.oggparse.OggStream(f).iter_packets()

		reader = cdict(
			pos=0,
			byte_pos=0,
			file=f,
			it=it,
			_read=lambda self, *args: f.read(args),
			closed=False,
			is_opus=lambda self: True,
			duration=self.duration,
			af=self,
			new=True,
		)

		def read():
			try:
				out = next(reader.it, b"")
			except ValueError:
				f = open(f"{CACHE_PATH}/audio/" + self.file, "rb")
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
			return reader

		reader.read = read
		reader.close = reader.cleanup = close
		return reader

	# Creates a reader, selecting from direct opus file, single piped FFmpeg, or double piped FFmpeg.
	def create_reader(self, auds, pos=0):
		source = self.stream
		# Construct FFmpeg options
		if self.live and not self.duration:
			pos = 0
		options = auds.construct_options(full=self.live)
		speed = 1
		if options or pos or auds.settings.bitrate < auds.defaults["bitrate"] or self.live or not isinstance(self.stream, (str, CachingTeeFile)):
			fflags = "+discardcorrupt+genpts+igndts+flush_packets"
			if self.live:
				fflags += "+bitexact"
			args = ["ffmpeg", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", fflags]
			if pos or auds.reverse:
				arg = "-to" if auds.reverse else "-ss"
				if auds.reverse and not pos:
					pos = self.duration or 300
				args += [arg, str(pos)]
			speed = round_min(auds.settings.speed * 2 ** (auds.settings.resample / 12))
			if auds.reverse:
				speed = -speed
			if isinstance(source, str) and not is_url(source) and source.endswith(".concat"):
				args.extend(("-f", "concat"))
			args.append("-i")
			buff = False
			live = False
			if isinstance(self.stream, str):
				if pos > 60 or not self.live or not is_youtube_stream(source):
					if is_url(source):
						args = ["ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "250"] + args[1:]
					args.insert(1, "-nostdin")
					args.append(source)
				else:
					live = True
					args.append("-")
			else:
				buff = True
				args.append("-")
			auds.settings.bitrate = min(auds.settings.bitrate, MAX_BITRATE)
			same_codec = False
			if not options:
				if not isinstance(source, str):
					args.extend(("-f", "opus", "-c:a", "copy"))
					same_codec = True
				if isinstance(source, str) and not auds.settings.bitrate < auds.defaults["bitrate"]:
					_dur, _bps, cdc, ac = get_duration_2(source)
					if ac and cdc in ("opus", "libopus"):
						args.extend(("-f", "opus", "-c:a", "copy"))
						same_codec = True
			if not same_codec:
				br = auds.settings.bitrate
				if self.live:
					br = min(96000, br)
					options.extend(("-rtbufsize", "1M"))
				sr = SAMPLE_RATE
				while br < 512:
					br *= 2
					sr >>= 1
				if sr < 8000:
					sr = 8000
				options.extend(("-f", "opus", "-c:a", "libopus", "-ar", str(sr), "-ac", "2", "-b:a", str(round(br)), "-vbr", "on", "-bufsize", "16k", "-frame_duration", "20"))
				args.extend(options)
			if self.live:
				args.extend(("-map_metadata", "-1", "-thread_queue_size", "512", "-application", "lowdelay"))
			args.append("-")
			print(args)
			if buff:
				# Select buffered reader for files not yet fully loaded, convert while downloading
				player = BufferedAudioReader(self, args, stream=self.stream)
			elif live:
				# Select live reader for streamed links
				player = LiveAudioReader(self, args)
			else:
				# Select loaded reader for loaded files
				player = LoadedAudioReader(self, args)
			print(player)
			player.speed = speed
			auds.args = args
			reader = player.start()
		else:
			auds.args = []
			# Select raw file stream for direct audio playback
			reader = self.open()
		reader.pos = pos * 50
		return reader

	@property
	def proc_expired(self):
		return not self.proc or not self.proc.is_running()

AF = AudioFile


class LoadedAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args):
		self.closed = False
		self.args = args
		self.proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=192000)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.af = file
		self.file = file
		self.pos = 0
		self.new = True

	def read(self):
		if self.buffer:
			b, self.buffer = self.buffer, None
			self.pos += self.speed
			return b
		out = b""
		for att in range(16):
			try:
				out = next(self.packet_iter, b"")
			except (OSError, BrokenPipeError):
				if getattr(self.file, "seekable", False):
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
		return out

	@tracebacksuppressor
	def start(self):
		self.buffer = None
		self.buffer = self.read()
		return self

	def close(self, *void1, **void2):
		self.closed = True
		force_kill(self.proc)
		return self
	cleanup = close

	def is_opus(self):
		return True


class BufferedAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args, stream):
		self.closed = False
		self.args = args
		self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=192000)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.file = file
		self.af = file
		self.stream = stream.open()
		self.pos = 0
		self.new = True

	def read(self):
		if self.buffer:
			b, self.buffer = self.buffer, None
			self.pos += self.speed
			return b
		try:
			fut = esubmit(next, self.packet_iter, b"")
			try:
				out = fut.result(timeout=60)
			except concurrent.futures.TimeoutError:
				print_exc()
				with suppress():
					force_kill(self.proc)
				out = b""
			self.pos += self.speed
			return out
		except Exception:
			print_exc()
			return b""

	# Required loop running in background to feed data to FFmpeg
	def run(self):
		try:
			for i in itertools.count(1):
				b = self.stream.read(i * 1024)
				if not b:
					break
				self.proc.stdin.write(b)
				self.proc.stdin.flush()
			self.proc.stdin.close()
		except BrokenPipeError:
			return
		except Exception:
			print_exc()

	@tracebacksuppressor
	def start(self):
		# Run loading loop in parallel thread obviously
		esubmit(self.run, timeout=86400)
		self.buffer = None
		self.buffer = self.read()
		return self

	def close(self):
		self.closed = True
		self.stream.close()
		force_kill(self.proc)
		return self
	cleanup = close

	def is_opus(self):
		return True


class LiveAudioReader(discord.AudioSource):

	speed = 1

	def __init__(self, file, args):
		self.closed = False
		self.args = args
		self.resp = None
		self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=192000)
		self.gen = esubmit(self.generate, file.stream, self.proc)
		self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
		self.af = file
		self.file = file
		self.pos = 0
		self.new = True

	def generate(self, stream, proc):
		with tracebacksuppressor:
			try:
				self.resp = niquests.get(stream, headers=Request.header(), stream=True, timeout=24)
				it = self.resp.iter_content(65536)
				b = next(it)
				fut = esubmit(next, it)
				proc.stdin.write(b)
				proc.stdin.flush()
				try:
					while True:
						b, fut = fut.result(), esubmit(next, it)
						proc.stdin.write(b)
						proc.stdin.flush()
				except StopIteration:
					pass
			finally:
				if self.resp:
					self.resp.close()
					self.resp = None
				proc.stdin.close()

	def read(self):
		if self.buffer:
			b, self.buffer = self.buffer, None
			self.pos += self.speed
			return b
		for att in range(16):
			try:
				out = next(self.packet_iter, b"")
			except (OSError, BrokenPipeError):
				if getattr(self.file, "seekable", False):
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
		if self.resp:
			self.resp.close()
		force_kill(self.proc)
		return self
	cleanup = close

	def is_opus(self):
		return True


class AudioClient(discord.AutoShardedClient):
	"""
	AudioClient is a subclass of discord.AutoShardedClient designed to handle audio-related functionalities.
	Attributes:
		intents (discord.Intents): The intents configuration for the client, specifying which events the bot should receive.
	Methods:
		__init__(): Initializes the AudioClient instance with specific configurations such as event loop, heartbeat timeout, and status.
	Note:
		- The client is configured with specific intents to optimize performance and limit the events received.
		- The initialization includes setting up the event loop and other configurations necessary for the client's operation.
	"""

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

async def reload_player(gid):
	with tracebacksuppressor:
		vcci, ci, dump, mis = AP.cache[gid]
		print(vcci, ci, len(dump), mis)
		try:
			a = await AP.join(vcci, ci, announce=2)
		except discord.HTTPException as ex:
			if "403" in str(ex):
				AP.cache.pop(gid, None)
			raise
		a.load_dump(dump, universal=True)
		# Annoying quirk of Discord API where list of members in voice is not directly retrievable; we must have a list of member IDs (or fetch every single user in the server, which is not feasible). This of course means members that join while the bot is offline will not be loaded into the audio player. However, this is a rare occurrence and can be mitigated by affected users simply rejoining the voice channel, or using the join or play commands, which notify the bot of their presence.
		for mid in mis:
			await a.find_user(gid, mid)
		# AP.cache.pop(gid, None)

async def unload_player(gid):
	with tracebacksuppressor:
		a = AP.players[gid]
		a.backup()
		await a.leave(
			reason="Temporary maintenance",
			dump=bool(a.queue),
		)

async def autosave_loop(start=True):
	print("Autosave loop initialised.")
	while start or not client.is_closed():
		start = False
		with tracebacksuppressor:
			await client.wait_until_ready()
			async with Delay(60):
				for guild in client.guilds:
					a = AP.players.get(guild.id)
					if not a or not len(a.queue):
						AP.cache.pop(guild.id, None)
						continue
					a.backup()
					if a.updating_activity is None:
						a.update_activity()
					if a.updating_streaming is None:
						a.update_streaming()
		await asyncio.sleep(1)
	print("Exiting...")

@client.event
async def on_connect():
	with tracebacksuppressor:
		if not client_fut.done():
			client_fut.set_result(client)
			# Restore audio players from our cache on disk
			keys = set(AP.cache.keys())
			if keys:
				print("Reloading players:", keys)
				await asyncio.gather(*(reload_player(gid) for gid in keys))
			await asyncio.gather(*(AP.force_disconnect(guild.id) for guild in client.guilds if guild.me and guild.me.voice is not None and guild.id not in keys))
			csubmit(autosave_loop())
		print("Audio client successfully connected.")

@client.event
async def on_ready():
	print("Audio client ready.")

@client.event
async def on_voice_state_update(member, before, after):
	if member.id == client.user.id:
		return
	guild = member.guild
	try:
		a = await asubmit(AP.from_guild, guild.id)
	except KeyError:
		return
	if member.bot:
		return
	a.update_activity()

async def terminate():
	# Unload all audio players and preserve their state in our cache on disk
	await asyncio.gather(*(unload_player(gid) for gid in AP.players.keys()))
	await asubmit(ytdl.close)
	# AP.cache.sync()
	return await client.close()


if __name__ == "__main__":
	pid = os.getpid()
	ppid = os.getppid()
	print(f"Audio client starting with PID {pid} and PPID {ppid}...")

	def startup(fut):
		interface.start()
		interface2.start()
		globals()["ytdl"] = fut.result()

	ytdl_fut.add_done_callback(startup)
	discord.client._loop = eloop
	eloop.run_until_complete(client.start(AUTH["discord_token"]))
