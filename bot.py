# Coco invaded, on the 1st August 2020 this became my territory *places flag* B3
# 11th of September 2023 they invaded again and stole your baby niece WUAHAHA *places updated flag* B3
# It's been 5 years it's like 2025 now why am I still here *blows up the past 2 flags with propane bomb and places new updated flag* B3

#!/usr/bin/python3

import os
print("BOT:", __name__)
if __name__ != "__mp_main__":
	os.environ["IS_BOT"] = "1"
from misc import common, asyncs
from misc.common import * # noqa: F403
import pathlib
import pdb
import ddgs

# import asyncio
# import collections
# import contextlib
# import datetime
# import json
# import pdb
# import subprocess
# import sys
# import time
# import discord
# import orjson
# import psutil
# import misc.common as common
# from collections import deque
# from concurrent.futures import Future
# from math import inf, ceil, log10
# from misc.asyncs import asubmit, csubmit, esubmit, tsubmit, gather, eloop, get_event_loop, Semaphore, SemaphoreOverflowError
# from misc.smath import xrand
# from misc.types import cdict, fdict, fcdict, mdict, alist, azero, round_min, full_prune, suppress, tracebacksuppressor
# from misc.util import AUTH, TEMP_PATH, FAST_PATH, PORT, PROC, EvalPipe, python, utc, T, lim_str, regexp, Request, reqs, is_strict_running, force_kill
# from misc.common import api, load_colour_list, load_emojis, touch, BASE_LOGO, closing, MemoryTimer


# import tracemalloc
# tracemalloc.start()

ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
if ADDRESS == "0.0.0.0":
	ADDRESS = "127.0.0.1"

if __name__ != "__mp_main__":
	esubmit(load_colour_list)
	esubmit(load_emojis)
	esubmit(download_binary_dependencies)


class Bot(discord.AutoShardedClient, contextlib.AbstractContextManager, collections.abc.Callable):
	"Main class containing all global bot data."

	github = AUTH.get("github") or "https://github.com/thomas-xin/Miza"
	rcc_invite = AUTH.get("rcc_invite") or "https://discord.gg/cbKQKAr"
	discord_icon = BASE_LOGO
	twitch_url = "https://www.twitch.tv/-"
	webserver = AUTH.get("webserver") or "https://mizabot.xyz"
	kofi_url = AUTH.get("kofi_url") or "https://ko-fi.com/waveplasma/tiers"
	rapidapi_url = AUTH.get("rapidapi_url") or "https://rapidapi.com/thomas-xin/api/miza"
	raw_webserver = AUTH.get("raw_webserver") or "https://api.mizabot.xyz"
	heartbeat_file = "heartbeat.tmp"
	restart = "restart.tmp"
	shutdown = "shutdown.tmp"
	activity = 0
	caches = ("guilds", "channels", "users", "roles", "emojis", "messages", "members", "attachments", "banned", "colours")
	statuses = (discord.Status.online, discord.Status.idle, discord.Status.dnd, discord.Streaming, discord.Status.invisible)
	# Default command prefix
	prefix = AUTH.get("prefix", "~")
	# This is a fixed ID apparently
	deleted_user = 456226577798135808
	_globals = globals()
	intents = discord.Intents(
		guilds=True,
		members=True,
		bans=True,
		emojis=True,
		webhooks=True,
		voice_states=True,
		presences=False,
		messages=True,
		reactions=True,
		typing=True,
	)
	intents.value |= 32768 # message content intent because discord dumb
	allowed_mentions = discord.AllowedMentions(
		everyone=False,
		users=True,
		roles=False,
		replied_user=False,
	)
	collecting = None
	connect_ready = Future()
	full_ready = Future()
	socket_responses = deque(maxlen=256)
	premium_server = 247184721262411776
	premium_roles = {
		1052645637033824346: 1,
		1052645761638215761: 2,
		1052647823188967444: 3,
	}
	active_categories = set(AUTH.setdefault("active_categories", ["MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "WEBHOOK", "FUN"]))

	def __init__(self, cache_size=65536, timeout=24):
		"Initializes client (first in __mro__ of class inheritance)"
		self.start_time = utc()
		shard_fut = esubmit(
			Request,
			f"https://discord.com/api/{api}/gateway/bot",
			authorise=True,
			json=True,
		)
		self.cache_size = cache_size
		# Base cache: contains all other caches
		self.cache = fcdict((c, fdict()) for c in self.caches)
		self.timeout = timeout
		self.set_classes()
		self.bot = self
		self.client = super()
		self.closing = False
		self.closed = False
		self.loaded = False
		# Channel-Webhook cache: for accessing all webhooks for a channel.
		self.cw_cache = cdict()
		self.usernames = {}
		self.events = mdict()
		self.react_sem = cdict()
		self.mention = ()
		self.user_loader = set()
		self.users_updated = True
		self.guilds_updated = True
		self.update_semaphore = Semaphore(2, 1)
		self.ready_semaphore = Semaphore(1, inf)
		self.guild_semaphore = Semaphore(5, inf, rate_limit=5)
		self.load_semaphore = Semaphore(5, inf, rate_limit=1)
		self.user_semaphore = Semaphore(64, inf, rate_limit=8)
		self.cache_semaphore = Semaphore(1, 1, rate_limit=30)
		self.command_semaphore = Semaphore(262144, 16384, rate_limit=30)
		print("Time:", datetime.datetime.now())
		print("Initializing...")
		# O(1) time complexity for searching directory
		directory = frozenset(os.listdir())
		if "saves" not in directory:
			os.mkdir("saves")
		for k in ("attachments", "audio", "filehost"):
			if not os.path.exists(f"{CACHE_PATH}/{k}"):
				os.mkdir(f"{CACHE_PATH}/{k}")
			if not os.path.exists(f"{TEMP_PATH}/{k}"):
				os.mkdir(f"{TEMP_PATH}/{k}")
			if not os.path.exists(f"{FAST_PATH}/{k}"):
				os.mkdir(f"{FAST_PATH}/{k}")
		try:
			self.token = AUTH["discord_token"]
		except KeyError:
			print("ERROR: discord_token not found. Unable to login.")
			self.setshutdown(force=True)
		try:
			owner_id = AUTH["owner_id"]
			if type(owner_id) not in (list, tuple):
				owner_id = (owner_id,)
			self.owners = alist(int(i) for i in owner_id)
		except KeyError:
			self.owners = alist()
			print("WARNING: owner_id not found. Unable to locate owner.")
		# Initialize rest of bot variables
		self.proc = PROC
		self.guild_count = 0
		self.updated = False
		self.started = False
		self.bot_ready = False
		self.ready = False
		self.initialisation_complete = False
		self.status_iter = xrand(4)
		self.curr_state = azero(4)
		self.ip = "127.0.0.1"
		self.audio = None
		self.embed_senders = cdict()
		# Assign bot cache to global variables for convenience
		globals().update(self.cache)
		modload = self.get_modules()
		self.modload = csubmit(gather(*modload))
		tsubmit(self.heartbeat_loop)
		data = shard_fut.result()
		self.wss = data["url"]
		x = AUTH.get("guild_count", 1)
		s = max(1, data["shards"])
		shards = max(1, ceil(x / log10(x) / 100 / s)) * s
		print("Automatic shards:", shards)
		assert data["session_start_limit"]["remaining"] > shards, "Insufficient session quota."
		self.monkey_patch()
		super().__init__(
			loop=eloop,
			_loop=eloop,
			max_messages=256,
			heartbeat_timeout=64,
			chunk_guilds_at_startup=False,
			guild_ready_timeout=16,
			intents=self.intents,
			allowed_mentions=self.allowed_mentions,
			assume_unsync_clock=True,
		)
		self.shard_count = shards
		self.set_client_events()
		with suppress(AttributeError):
			csubmit(super()._async_setup_hook())
		globals()["messages"] = self.messages = self.MessageCache()

	__str__ = lambda self: str(self.user) if T(self).get("user") else object.__str__(self)
	__repr__ = lambda self: repr(self.user) if T(self).get("user") else object.__repr__(self)
	__call__ = lambda self: self
	__exit__ = lambda self, *args, **kwargs: self.close()

	def __getattr__(self, key):
		try:
			return object.__getattribute__(self, key)
		except AttributeError:
			pass
		if key == "user":
			return self.__getattribute__("_user")
		for attr in ("_connection", "user", "proc"):
			this = self.__getattribute__(attr)
			try:
				return getattr(this, key)
			except AttributeError:
				pass
		raise AttributeError(key)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(dir(self._connection))
		data.update(dir(self.user))
		data.update(dir(self.proc))
		return data

	@property
	def maintenance(self):
		return "blacklist" in self.data and self.data.blacklist.get(0)

	def guild_shard(self, g_id):
		return (g_id >> 22) % self.shard_count

	# Waits an amount of seconds and shuts down.
	def setshutdown(self, delay=None, force=False):
		if delay:
			time.sleep(delay)
		if force:
			pathlib.Path.touch(self.shutdown)
			with suppress():
				os.remove(self.heartbeat_file)
		# force_kill(self.proc)

	def command_options(self, command):
		accepts_attachments = False
		out = []
		extra = []
		if command.schema:
			for k, v in command.schema.items():
				if not isinstance(v, cdict):
					raise TypeError(k, v)
				desc = lim_str((v.get("description") or v.type) + (f', e.g. "{v.example}"' if v.get("example") else ""), 100)
				arg = cdict(
					type=3,
					name=k,
					description=desc,
				)
				if v.type in ("url", "image", "visual", "video", "audio", "media"):
					accepts_attachments = True
					arg.pop("required", None)
				elif v.get("required") or v.get("required_slash"):
					arg.required = True
				if v.type == "enum":
					options = sorted(v.validation.get("enum") or v.validation.accepts)
					if len(options) <= 25:
						arg.choices = [dict(name=opt, value=opt) for opt in options]
					elif len(arg.description) < 100:
						argf = ",".join(map(str, options))
						arg.description = lim_str(arg.description + f"; one of ({argf})", 100, mode="left")
				elif v.type == "integer":
					arg.type = 4
					if v.get("validation") and isinstance(v.validation, str):
						lx, rx = v.validation.split(",")
						mx, Mx = round_min(lx[1:]), round_min(rx[:-1])
						arg.min_value = mx
						arg.max_value = Mx
				elif v.type == "bool":
					arg.type = 5
				elif v.type == "user":
					arg.type = 6
				elif v.type == "channel":
					arg.type = 7
				elif v.type == "role":
					arg.type = 8
				elif v.type == "mentionable":
					arg.type = 9
				elif v.type == "number":
					arg.type = 10
					if v.get("validation") and isinstance(v.validation, str):
						lx, rx = v.validation.split(",")
						mx, Mx = round_min(lx[1:]), round_min(rx[:-1])
						arg.min_value = mx
						arg.max_value = Mx
				out.append(arg)
				if v.get("multiple"):
					for i in range(5):
						a2 = cdict(arg)
						a2.name = arg.name + f"-{i + 2}"
						a2.description = f"Extension of {arg.name}"
						extra.append(a2)
		else:
			for i in command.usage.split():
				with tracebacksuppressor:
					arg = dict(type=3, name=i, description=i)
					if i.endswith("?"):
						arg["description"] = "[optional] " + arg["description"][:-1]
					elif i.endswith("*"):
						arg["description"] = "[zero or more] " + arg["description"][:-1]
					else:
						if i.endswith("+"):
							arg["description"] = "[one or more] " + arg["description"][:-1]
						arg["required"] = True
						# if not default and usage.count(" "):
						#     arg["default"] = default = True
					arg["description"] = lim_str(arg["description"], 100)
					if i.startswith("<"):
						s = i[1:].split(":", 1)[-1].rsplit(">", 1)[0]
						formats = regexp(r"[\w\-\[\]]+(?:\((?:\?:)?[\w\'\-\|\[\]]+\))?").findall(s)
						for fmt in formats:
							a = dict(arg)
							if "(" not in fmt:
								name = fmt
								if fmt == "user":
									a["type"] = 9
									a["description"] = "user"
								elif fmt == "id":
									a["type"] = 4
									a["description"] = "integer"
								else:
									a["description"] = "string"
							else:
								name, opts = fmt.split("(", 1)
								if "|" not in opts:
									a["type"] = 5
									a["description"] = "bool"
								elif opts.startswith("?:"):
									a["description"] = "(" + opts[2:].rstrip(")") + ")"
								else:
									opts = opts.rstrip(")").split("|")
									a["choices"] = [dict(name=opt, value=opt) for opt in opts]
									a["description"] = "choice"
							if "[" in name:
								name, d = name.split("[", 1)
								a["description"] += " [" + d
							a["name"] = name
							out.append(a)
						continue
					if arg["name"] == "user":
						arg["type"] = 9
					elif arg["name"] == "url":
						accepts_attachments = True
						arg.pop("required", None)
					out.append(arg)
		if getattr(command, "ephemeral", None):
			arg = dict(type=5, name="ephemeral", description="Display response only for you")
			out.append(arg)
		if accepts_attachments:
			arg = dict(type=11, name="attachment", description="Attachment in place of URL")
			out.append(arg)
		out.extend(extra)
		return sorted(out, key=lambda arg: not arg.get("required"))

	slash_sem = Semaphore(5, 256, rate_limit=5)
	@tracebacksuppressor
	def create_command(self, data):
		with self.slash_sem:
			attempts = 16
			for i in range(attempts):
				resp = reqs.next().post(
					f"https://discord.com/api/{api}/applications/{self.id}/commands",
					headers={"Content-Type": "application/json", "Authorization": "Bot " + self.token},
					data=json_dumps(data),
					timeout=30,
				)
				if resp.status_code == 429 and i < attempts - 1:
					time.sleep(2 ** (i + 4))
					continue
				resp.raise_for_status()
				print("SLASH CREATE:", resp.text)
				return

	def update_slash_commands(self):
		print("Updating global slash commands...")
		with tracebacksuppressor:
			resp = reqs.next().get(
				f"https://discord.com/api/{api}/applications/{self.id}/commands",
				headers=dict(Authorization="Bot " + self.token),
				timeout=30,
			)
			resp.raise_for_status()
			commands = dict((int(c["id"]), c) for c in resp.json() if str(c.get("application_id")) == str(self.id))
			if commands:
				print(f"Successfully loaded {len(commands)} application command{'s' if len(commands) != 1 else ''}.")

		def recursively_equal(opt1, opt2):
			for k, v in opt1.items():
				if isinstance(v, dict):
					if not opt2.get(k) or not recursively_equal(v, opt2[k]):
						return False
				elif isinstance(v, list):
					if not v:
						if opt2.get(k):
							return False
					elif not opt2.get(k):
						return False
					elif isinstance(v[0], dict):
						if not all(recursively_equal(x, y) for x, y in zip(v, opt2[k])):
							return False
					elif not all(x == y for x, y in zip(v, opt2[k])):
						return False
				elif opt2.get(k) != v:
					return False
			return True

		for catg in self.categories.values():
			if not AUTH.get("slash_commands"):
				break
			for command in catg:
				with tracebacksuppressor:
					if T(command).get("msgcmd"):
						aliases = command.msgcmd if type(command.msgcmd) is tuple else (command.parse_name(),)
						for name in aliases:
							command_data = dict(name=name, type=3)
							found = False
							for i, curr in list(commands.items()):
								if curr["name"] == name and curr["type"] == command_data["type"]:
									found = True
									commands.pop(i)
									break
							if not found:
								print(f"creating new message command {command_data['name']}...")
								print(command_data)
								esubmit(self.create_command, command_data, priority=True)
					if T(command).get("usercmd"):
						aliases = command.usercmd if type(command.usercmd) is tuple else (command.parse_name(),)
						for name in aliases:
							command_data = dict(name=name, type=2)
							found = False
							for i, curr in list(commands.items()):
								if curr["name"] == name and curr["type"] == command_data["type"]:
									found = True
									commands.pop(i)
									break
							if not found:
								print(f"creating new user command {command_data['name']}...")
								print(command_data)
								esubmit(self.create_command, command_data, priority=True)
					if T(command).get("slash"):
						aliases = command.slash if type(command.slash) is tuple else (command.parse_name(),)
						for name in (full_prune(i) for i in aliases):
							description = lim_str(command.parse_description(), 100)
							options = self.command_options(command)
							command_data = dict(name=name, description=description, type=1, nsfw=getattr(command, "nsfw", False))
							if getattr(command, "server_only", False):
								command_data["contexts"] = [0]
							else:
								command_data["contexts"] = [0, 1, 2]
							command_data["integration_types"] = [0, 1]
							if options:
								command_data["options"] = options
							found = False
							for i, curr in list(commands.items()):
								if curr["name"] == name and curr["type"] == command_data["type"]:
									if not recursively_equal(command_data, curr):
										print(command_data)
										print(curr)
										print(f"{curr['name']}'s slash command does not match, removing...")
										with self.slash_sem:
											attempts = 16
											for j in range(attempts):
												resp = reqs.next().delete(
													f"https://discord.com/api/{api}/applications/{self.id}/commands/{curr['id']}",
													headers=dict(Authorization="Bot " + self.token),
													timeout=30,
												)
												if resp.status_code == 429 and j < attempts - 1:
													time.sleep(2 ** (j + 4))
													continue
												resp.raise_for_status()
												break
									else:
										found = True
									commands.pop(i, None)
									break
							if not found:
								print(f"creating new slash command {command_data['name']}...")
								print(command_data)
								esubmit(self.create_command, command_data)
		with self.slash_sem:
			time.sleep(1)
		for curr in commands.values():
			with tracebacksuppressor:
				print(curr)
				print(f"{curr['name']}'s application command does not exist, removing...")
				resp = reqs.next().delete(
					f"https://discord.com/api/{api}/applications/{self.id}/commands/{curr['id']}",
					headers=dict(Authorization="Bot " + self.token),
					timeout=30,
				)
				resp.raise_for_status()

	async def create_main_website(self, first=False):
		if first:
			print("Generating command json...")
			j = {}
			for category in ("MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "FUN", "AI", "NSFW", "MISC", "OWNER"):
				k = j[category] = {}
				if category not in self.categories:
					continue
				for command in self.categories[category]:
					c = k[command.parse_name()] = dict(
						aliases=[n.strip("_") for n in command.alias],
						description=command.parse_description(),
						usage=command.usage,
						level=str(command.min_level),
						rate_limit=str(command.rate_limit),
						example=T(command).get("example", []),
						timeout=str(T(command).get("_timeout_", 1) * self.timeout),
					)
					for attr in ("flags", "server_only", "slash"):
						with suppress(AttributeError):
							c[attr] = command.attr
			with open("misc/web/static/HELP.json", "w", encoding="utf-8") as f:
				json.dump(j, f, indent="\t")

	server = None
	server_start_sem = Semaphore(1, 0, rate_limit=5)
	def start_webserver(self):
		with self.server_start_sem:
			if self.server:
				try:
					self.server.run("terminate()", timeout=8)
				except Exception:
					print_exc()
				self.server.terminate()
			if os.path.exists("misc/x_server.py") and PORT:
				print("Starting webserver...")
				self.server = EvalPipe.connect(
					[python, "-m", "misc.x_server", "6562"],
					6562,
					glob=globals(),
				)
			else:
				self.server = None

	def start_audio_client(self):
		if self.audio:
			try:
				self.audio.run("terminate()", timeout=8)
			except Exception:
				print_exc()
			self.audio.terminate()
		if os.path.exists("misc/x_audio.py"):
			print("Starting audio client...")
			self.audio = EvalPipe.connect(
				[python, "-m", "misc.x_audio", "6561"],
				6561,
				glob=globals(),
			)
		else:
			self.audio = None

	@tracebacksuppressor
	def run(self):
		"Starts up client."
		t = utc()
		for i in itertools.count(1):
			try:
				requests.head(f"https://discord.com/api/{api}/users/@me", timeout=5).close()
			except Exception:
				print_exc()
				time.sleep(i ** 2)
				print(f"Unable to reach Discord API after {sec2time(utc() - t)}, retrying...")
			else:
				break
		print("Logging in...")
		try:
			self.audio_client_start = asubmit(self.start_audio_client, priority=1)
			loop = get_event_loop()
			with contextlib.closing(loop):
				with tracebacksuppressor:
					loop.run_until_complete(self.start(self.token))
				with tracebacksuppressor:
					loop.run_until_complete(self.close())
				for t in asyncio.all_tasks(loop):
					with tracebacksuppressor:
						t.cancel()
		finally:
			self.setshutdown()

	def print(self, *args, sep=" ", end="\n"):
		"A reimplementation of the print builtin function."
		sys.__stdout__.write(str(sep).join(str(i) for i in args) + end)

	def close(self):
		"Closes the bot, preventing all events."
		self.closing = True
		self.closed = True
		return super().close()

	@tracebacksuppressor(SemaphoreOverflowError)
	async def garbage_collect(self, obj):
		"A garbage collector for empty and unassigned objects in the database."
		if not self.ready or hasattr(obj, "no_delete") or not any(hasattr(obj, i) for i in ("guild", "user", "channel", "garbage")) and not getattr(obj, "garbage_collect", None):
			return
		with MemoryTimer(f"{obj.name}-gc"):
			async with obj._garbage_semaphore:
				data = obj.data
				if getattr(obj, "garbage_collect", None):
					return await obj.garbage_collect()
				if len(data) <= 1024:
					keys = data.keys()
				else:
					low = xrand(ceil(len(data) / 1024)) << 10
					keys = astype(data, alist).view[low:low + 1024]
				for key in keys:
					if getattr(obj, "unloaded", False):
						return
					if not key or isinstance(key, str):
						continue
					try:
						# Database keys may be user, guild, or channel IDs
						if getattr(obj, "channel", False):
							d = self.get_channel(key)
						elif getattr(obj, "user", False):
							d = await self.fetch_user(key)
						else:
							if not data[key]:
								raise LookupError
							with suppress(KeyError):
								d = self.cache.guilds[key]
								continue
							d = await self.fetch_messageable(key)
						if d is not None:
							continue
					except Exception:
						print_exc()
					print(f"Deleting {key} from {obj}...")
					data.pop(key, None)

	@tracebacksuppressor
	async def send_event(self, ev, *args, exc=False, **kwargs):
		"Calls a bot event, triggered by client events or others, across all bot databases. Calls may be sync or async."
		if self.closed:
			return
		with MemoryTimer(f"{ev}-event"):
			ctx = emptyctx if exc else tracebacksuppressor
			events = self.events.get(ev, ())
			if len(events) == 1:
				with ctx:
					return await asubmit(events[0](*args, **kwargs))
				return
			for func in filter(bool, events):
				fut = func(*args, **kwargs)
				if fut:
					await fut
			# futs = [asubmit(func(*args, **kwargs)) for func in events]
			# with ctx:
			# 	return await gather(*futs)

	@tracebacksuppressor(default=[])
	async def get_full_invites(self, guild):
		"Gets the full list of invites from a guild, if applicable."
		member = guild.get_member(self.id)
		if member.guild_permissions.create_instant_invite:
			invitedata = await Request(
				f"https://discord.com/api/{api}/guilds/{guild.id}/invites",
				authorise=True,
				aio=True,
				json=True,
			)
			invites = [cdict(invite) for invite in invitedata]
			return sorted(invites, key=lambda invite: (invite.max_age == 0, -abs(invite.max_uses - invite.uses), len(invite.url)))
		return []

	def get_first_sendable(self, guild, member):
		"Gets the first accessable text channel in the target guild."
		if member is None:
			return guild.owner
		found = {}
		for channel in sorted(guild.text_channels, key=lambda c: c.id):
			if channel.permissions_for(member).send_messages:
				with suppress(ValueError):
					rname = full_prune(channel.name).replace("-", " ").replace("_", " ").split(maxsplit=1)[0]
					i = ("miza", "bots", "bot", "general").index(rname)
					if i < min(found):
						found[i] = channel
		if found:
			return found[min(found)]
		channel = guild.system_channel
		if channel is None or not channel.permissions_for(member).send_messages:
			channel = guild.rules_channel
			if channel is None or not channel.permissions_for(member).send_messages:
				for channel in sorted(guild.text_channels, key=lambda c: c.id):
					if channel.permissions_for(member).send_messages:
						return channel
				return guild.owner
		return channel

	def in_cache(self, o_id):
		"Returns a discord object if it is in any of the internal cache."
		cache = self.cache
		try:
			return cache.users[o_id]
		except KeyError:
			pass
		try:
			return cache.channels[o_id]
		except KeyError:
			pass
		try:
			return cache.guilds[o_id]
		except KeyError:
			pass
		try:
			return cache.roles[o_id]
		except KeyError:
			pass
		try:
			return cache.emojis[o_id]
		except KeyError:
			pass
		try:
			return self.data.mimics[o_id]
		except KeyError:
			pass

	async def fetch_messageable(self, s_id):
		"Fetches either a user or channel object from ID, using the bot cache when possible."
		if not isinstance(s_id, int):
			try:
				s_id = int(s_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid messageable identifier: {s_id}")
		with suppress(KeyError):
			return self.get_user(s_id)
		with suppress(KeyError):
			return self.cache.channels[s_id]
		try:
			user = await super().fetch_user(s_id)
		except (LookupError, discord.NotFound):
			channel = await super().fetch_channel(s_id)
			self.cache.channels[s_id] = channel
			return channel
		self.cache.users[s_id] = user
		return user

	async def _fetch_user(self, u_id):
		"Fetches a user from ID, using the bot cache when possible."
		async with self.user_semaphore:
			user = await super().fetch_user(u_id)
			self.cache.users[u_id] = user
			return user
	def fetch_user(self, u_id):
		with suppress(KeyError):
			user = as_fut(self.get_user(u_id))
			if user and T(user).get("_avatar") != self.discord_icon:
				return user
		u_id = verify_id(u_id)
		if not isinstance(u_id, int):
			raise TypeError(f"Invalid user identifier: {u_id}")
		return self._fetch_user(u_id)

	async def auser2cache(self, u_id):
		with suppress(discord.NotFound):
			self.cache.users[u_id] = await super().fetch_user(u_id)

	def user2cache(self, data):
		users = self.cache.users
		u_id = int(data["id"])
		if u_id not in users:
			if isinstance(data, dict):
				with tracebacksuppressor:
					if "s" in data:
						s = data.pop("s")
						if "#" in s:
							data["username"], data["discriminator"] = s.rsplit("#", 1)
						else:
							data["username"] = s
							data["discriminator"] = 0
					else:
						if data.get("discriminator") not in (None, 0, "0"):
							s = data["username"] + "#" + data["discriminator"]
						else:
							s = data["username"]
					self.usernames[s] = users[u_id] = self._state.store_user(data)
					return
			self.user_loader.add(u_id)

	def update_users(self):
		if self.user_loader:
			if not self.user_semaphore.busy:
				u_id = self.user_loader.pop()
				if u_id not in self.cache.users:
					csubmit(self.auser2cache(u_id))

	def get_user(self, u_id, replace=False):
		"Gets a user from ID, using the bot cache."
		if not isinstance(u_id, int):
			try:
				u_id = int(u_id)
			except (ValueError, TypeError):
				user = self.user_from_identifier(u_id)
				if user is not None:
					return user
				if "#" in u_id:
					raise LookupError(f"User identifier not found: {u_id}")
				u_id = verify_id(u_id)
				if not isinstance(u_id, int):
					raise TypeError(f"Invalid user identifier: {u_id}")
		with suppress(KeyError):
			return self.cache.users[u_id]
		if u_id == self.deleted_user:
			user = self.GhostUser()
			user.system = True
			user.name = "Deleted User"
			user.nick = "Deleted User"
			user.id = u_id
		else:
			try:
				user = super().get_user(u_id)
				if user is None:
					raise LookupError
			except LookupError:
				if replace:
					return self.get_user(self.deleted_user)
				raise KeyError("Target user ID not found.")
		self.cache.users[u_id] = user
		return user

	async def find_users(self, argl, args, user, guild, roles=False):
		if not argl and not args:
			return (user,)
		if argl:
			users = {}
			for u_id in argl:
				u = await self.fetch_user_member(u_id, guild)
				users[u.id] = u
			return users.values()
		u_id = verify_id(args.pop(0))
		if isinstance(u_id, int) and guild:
			role = guild.get_role(u_id)
			if role is not None:
				if roles:
					return (role,)
				try:
					return role.members
				except AttributeError:
					return [member for member in guild._members.values() if u_id in member._roles]
		if isinstance(u_id, str) and "@" in u_id and ("everyone" in u_id or "here" in u_id):
			return await self.get_full_members(guild)
		u = await self.fetch_user_member(u_id, guild)
		return (u,)

	def user_from_identifier(self, u_id):
		spl = u_id.split()
		for i in range(len(spl)):
			uid = " ".join(spl[i:])
			try:
				return self.usernames[uid]
			except KeyError:
				pass

	async def fetch_user_member(self, u_id, guild=None):
		u_id = verify_id(u_id)
		if isinstance(u_id, int):
			try:
				user = self.cache.users[u_id]
			except KeyError:
				try:
					user = await self.fetch_user(u_id)
				except discord.NotFound:
					if guild and "webhooks" in self.data:
						for channel in guild.text_channels:
							webhooks = await self.data.webhooks.get(channel)
							try:
								return [w for w in webhooks if w.id == u_id][0]
							except IndexError:
								pass
					raise
			with suppress():
				if guild:
					member = guild.get_member(user.id)
					if member is not None:
						return member
			with suppress():
				return self.get_member(u_id, guild, find_others=False)
			return user
		user = self.user_from_identifier(u_id)
		if user is not None:
			if guild is None:
				return user
			member = guild.get_member(user.id)
			if member is not None:
				return member
			return user
		return await self.fetch_member_ex(u_id, guild)

	async def get_full_members(self, guild):
		members = guild._members.values()
		if "bans" in self.data:
			members = set(members)
			for b in self.data.bans.get(guild.id, ()):
				try:
					user = await self.fetch_user(b.get("u", self.deleted_user))
				except LookupError:
					user = self.cache.users[self.deleted_user]
				members.add(user)
		return members

	async def query_members(self, members, query, fuzzy=0.5):
		query = str(query)
		fuz_base = 0
		with suppress(LookupError):
			return await str_lookup(
				members,
				query,
				qkey=userQuery1,
				ikey=userIter1,
				loose=False,
				fuzzy=fuz_base,
			)
		with suppress(LookupError):
			return await str_lookup(
				members,
				query,
				qkey=userQuery2,
				ikey=userIter2,
				fuzzy=fuzzy,
			)
		raise LookupError(f"No results for {query}.")

	async def fetch_member_ex(self, u_id, guild=None, allow_banned=True, fuzzy=0.5):
		"Fetches a member in the target server by ID or name lookup."
		if not isinstance(u_id, int) and u_id.isnumeric():
			with suppress(TypeError, ValueError):
				u_id = int(u_id)
		member = None
		if isinstance(u_id, int) and guild:
			member = guild.get_member(u_id)
		if member is None:
			if isinstance(u_id, int):
				with suppress(LookupError):
					if guild:
						member = await self.fetch_member(u_id, guild)
				if member is None:
					with suppress(LookupError):
						member = await self.fetch_user(u_id)
			try:
				return self.usernames[u_id]
			except KeyError:
				pass
			if member is None:
				if not guild:
					raise LookupError("No guild or unique identifier provided.")
				elif allow_banned:
					members = await self.get_full_members(guild)
				else:
					members = guild.members
				if not members:
					members = guild.members = await guild.fetch_members(limit=None)
					guild._members.update({m.id: m for m in members})
				return await self.query_members(members, u_id, fuzzy=fuzzy)
		return member

	def fetch_member(self, u_id, guild=None, find_others=False):
		"Fetches the first seen instance of the target user as a member in any shared server."
		return asubmit(self.get_member, u_id, guild, find_others)

	def get_member(self, u_id, guild=None, find_others=True):
		if not isinstance(u_id, int):
			try:
				u_id = int(u_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid user identifier: {u_id}")
		if find_others:
			with suppress(LookupError):
				member = self.cache.members[u_id].guild.get_member(u_id)
				if member is None:
					raise LookupError
				return member
		g = self.cache.guilds
		if guild is None:
			if find_others:
				guilds = deque(self.cache.guilds.values())
			else:
				return self.cache.users[u_id]
		else:
			if find_others:
				guilds = deque(g[i] for i in g if g[i].id != guild.id)
				guilds.appendleft(guild)
			else:
				guilds = [guild]
		member = None
		for guild in guilds:
			member = guild.get_member(u_id)
			if member is not None:
				break
		if member is None:
			raise LookupError("Unable to find member data.")
		if find_others:
			self.cache.members[u_id] = member
		return member

	async def fetch_guild(self, g_id, follow_invites=True):
		"Fetches a guild from ID, using the bot cache when possible."
		if not isinstance(g_id, int):
			try:
				g_id = int(g_id)
			except (ValueError, TypeError):
				if follow_invites:
					try:
						# Parse and follow invites to get partial guild info
						invite = await super().fetch_invite(g_id.strip("< >"))
						g = invite.guild
						with suppress(KeyError):
							return self.cache.guilds[g.id]
						if not hasattr(g, "member_count"):
							guild = cdict(ghost=True, member_count=invite.approximate_member_count)
							for at in ('banner', 'created_at', 'description', 'features', 'icon', 'id', 'name', 'splash', 'verification_level'):
								setattr(guild, at, getattr(g, at))
							guild.member_count = getattr(invite, "approximate_member_count", None)
							guild.icon_url = str(guild.icon)
						else:
							guild = g
						return guild
					except (discord.NotFound, discord.HTTPException) as ex:
						raise LookupError(str(ex))
				raise TypeError(f"Invalid server identifier: {g_id}")
		with suppress(KeyError):
			return self.cache.guilds[g_id]
		try:
			guild = super().get_guild(g_id)
			if guild is None:
				raise LookupError
		except LookupError:
			guild = await super().fetch_guild(g_id)
		# self.cache.guilds[g_id] = guild
		return guild

	async def _fetch_channel(self, c_id):
		"Fetches a channel from ID, using the bot cache when possible."
		channel = await super().fetch_channel(c_id)
		self.cache.channels[c_id] = channel
		return channel
	def fetch_channel(self, c_id):
		if not isinstance(c_id, int):
			try:
				c_id = int(c_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid channel identifier: {c_id}")
		with suppress(KeyError):
			return as_fut(self.cache.channels[c_id])
		return self._fetch_channel(c_id)

	def force_channel(self, data):
		if isinstance(data, dict):
			c_id = data["channel_id"]
		else:
			c_id = verify_id(data)
		if not isinstance(c_id, int):
			try:
				c_id = int(c_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid channel identifier: {c_id}")
		with suppress(KeyError):
			return self.cache.channels[c_id]
		channel, _ = bot._get_guild_channel(dict(channel_id=c_id))
		if channel and type(channel) is not discord.Object:
			if not isinstance(channel, discord.abc.PrivateChannel):
				self.cache.channels[c_id] = channel
			return channel
		raise KeyError(f"Channel {c_id} not found.")

	async def refresh_message(self, message):
		message = await self._fetch_message(message.id, message.channel)
		self.add_message(message, force=True)
		return message

	async def edit_message(self, message, **kwargs):
		resp = await manual_edit(message, **kwargs)
		self.add_message(resp, force=True)
		return resp

	# Fetches a message from ID and channel, using the bot cache when possible.
	async def _fetch_message(self, m_id, channel=None):
		if channel is None:
			raise LookupError("Message data not found.")
		with suppress(TypeError):
			int(channel)
			channel = await self.fetch_channel(channel)
		# This method is only called when a message is not found in the cache. This means that it is likely either out of range of cache or was produced during downtime, meaning surrounding messages may not be cached either. We preemtively fetch the surrounding messages to prepare for future requests, or requests that may involve the surrounding messages.
		messages = await flatten(discord.abc.Messageable.history(channel, limit=101, around=cdict(id=m_id)))
		data = {m.id: m for m in messages}
		min_id = min(data, default=m_id)
		if min_id != m_id:
			left = await flatten(discord.abc.Messageable.history(channel, limit=100, before=cdict(id=min_id)))
			data.update({m.id: m for m in left})
		max_id = max(data, default=m_id)
		if max_id != m_id:
			right = await flatten(discord.abc.Messageable.history(channel, limit=100, after=cdict(id=max_id)))
			data.update({m.id: m for m in right})
		self.cache.messages.update(data)
		return data[m_id]
	async def fetch_message(self, m_id, channel=None, old=False):
		if not isinstance(m_id, int):
			try:
				m_id = int(m_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid message identifier: {m_id}")
		m = None
		with suppress(KeyError):
			m = self.cache.messages[m_id]
		if not m and "message_cache" in self.data:
			with suppress(KeyError):
				m = await asubmit(self.data.message_cache.load_message, m_id)
		if m and not old:
			if m.attachments:
				if any(discord_expired(str(a.url)) for a in m.attachments):
					if channel:
						if isinstance(channel, int):
							channel = self.cache.channels.get(channel)
						if channel:
							return await channel.fetch_message(m_id)
					m = None
		if m:
			return m
		return await self._fetch_message(m_id, channel)

	async def fetch_reference(self, message):
		if not getattr(message, "reference", None):
			raise LookupError("Message has no reference.")
		reference = message.reference
		try:
			mid = getattr(reference, "message_id", None) or getattr(reference, "id", None)
			if not mid:
				print("Reference not found!")
				print_class(reference)
				raise ValueError(reference)
			reference = await self.fetch_message(mid, message.channel)
			if reference.webhook_id:
				# Force a re-fetch to refresh possibly changed information such as author.
				return await reference.channel.fetch_message(reference.id)
			return reference
		except AttributeError as ex:
			raise LookupError(*ex.args)

	def as_file(self, file, filename=None):
		url = await_fut(self.data.exec.lproxy(file, filename=filename))
		print("AS_FILE:", url)
		return url

	@functools.lru_cache(maxsize=64)
	def preserve_attachment(self, a_id, fn=None):
		if fn and "://" in fn:
			u = fn.split("?", 1)[0].rsplit("/", 1)[-1]
			if "." in u:
				fn = "." + u.rsplit(".", 1)[-1]
			else:
				fn = ""
		elif not fn:
			fn = ""
		elif "." not in fn:
			fn = "." + fn
		if is_url(a_id):
			url = a_id
			if is_discord_attachment(url):
				_c_id = int(url.split("?", 1)[0].rsplit("/", 3)[-3])
				a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
				if a_id in self.data.attachments:
					return self.webserver + "/u/" + base64.urlsafe_b64encode(a_id.to_bytes(8, "big")).rstrip(b"=").decode("ascii") + fn
			a_id = ts_us()
			while a_id in self.data.attachments:
				a_id += 1
			self.data.attachments[a_id] = url
		return self.webserver + "/u/" + base64.urlsafe_b64encode(a_id.to_bytes(8, "big")).rstrip(b"=").decode("ascii") + fn

	def preserve_into(self, c, m, a, fn=None):
		if fn and "://" in fn:
			u = fn.split("?", 1)[0].rsplit("/", 1)[-1]
			if "." in u:
				fn = "." + u.rsplit(".", 1)[-1]
			else:
				fn = ""
		elif not fn:
			fn = ""
		elif "." not in fn:
			fn = "." + fn
		a_id = verify_id(a)
		self.data.attachments[a_id] = (verify_id(c), verify_id(m))
		return self.webserver + "/u/" + base64.urlsafe_b64encode(a_id.to_bytes(8, "big")).rstrip(b"=").decode("ascii") + fn

	def preserve_as_long(self, c_id, m_id, a_id, fn=None):
		if fn and is_url(fn) and ("exec" not in self.data or not is_discord_attachment(fn)):
			return self.webserver + "/u?url=" + quote_plus(fn.split("?", 1)[0])
		if fn and "://" in fn:
			fn = fn.split("?", 1)[0].rsplit("/", 1)[-1]
		else:
			fn = ""
		return self.webserver + "/u/" + encode_attachment(c_id, m_id, a_id, fn)

	async def renew_from_long(cself, c, m, a):
		c_id = int.from_bytes(base64.urlsafe_b64decode(c + "=="), "big")
		m_id = int.from_bytes(base64.urlsafe_b64decode(m + "=="), "big")
		a_id = int.from_bytes(base64.urlsafe_b64decode(a + "=="), "big")
		with tracebacksuppressor:
			channel = await self.fetch_channel(c_id)
			message = await self.fetch_message(m_id, channel)
			for attachment in message.attachments:
				if attachment.id == a_id:
					url = str(attachment.url).rstrip("&")
					if discord_expired(url):
						return await self.renew_attachment(url, m_id)
					return url
		return "https://mizabot.xyz/notfound.png"

	def try_attachment(self, url, m_id=None) -> str:
		if not isinstance(url, int):
			_c_id = int(url.split("?", 1)[0].rsplit("/", 3)[-3])
			a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
		else:
			a_id = url
		if a_id in self.data.attachments:
			return self.webserver + "/u/" + base64.urlsafe_b64encode(a_id.to_bytes(8, "big")).rstrip(b"=").decode("ascii")
		return url

	async def delete_attachment(self, url, m_id=None):
		"Deletes a cached attachment by URL or ID."
		if isinstance(url, int):
			a_id = url
			tup = self.data.attachments.get(a_id)
			if is_url(tup):
				return tup
			if not tup:
				return False
			c_id, m_id = tup
		else:
			c_id = int(url.split("?", 1)[0].rsplit("/", 3)[-3])
			a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
		if not m_id:
			m_id = self.data.attachments.get(a_id)
			if is_url(m_id):
				return m_id
			if isinstance(m_id, (tuple, list)):
				c_id, m_id = m_id
		if not m_id:
			return False
		channel = await self.fetch_channel(c_id)
		message = await self.fetch_message(channel, m_id)
		if message.author.id == self.user.id:
			await self.silent_delete(message)
		return True

	async def renew_attachment(self, url, m_id=None):
		"Renews a cached attachment URL by either re-fetching the message, or failing that, proxying its embed preview."
		if isinstance(url, int):
			a_id = url
			tup = self.data.attachments.get(a_id)
			if is_url(tup):
				return await self.backup_url(tup)
			if not tup:
				return "https://mizabot.xyz/notfound.png"
			c_id, m_id = tup
		else:
			try:
				c_id = int(url.split("?", 1)[0].rsplit("/", 3)[-3])
			except ValueError:
				return url # Either not a discord attachment, or an attachment in DMs which cannot be renewed.
			a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
		if not m_id:
			m_id = self.data.attachments.get(a_id)
			if is_url(m_id):
				return await self.backup_url(m_id)
			if isinstance(m_id, (tuple, list)):
				c_id, m_id = m_id
		if not m_id:
			return await self.backup_url(url)
		channel = await self.fetch_channel(c_id)
		try:
			message = self.data.message_cache.load_message(m_id)
			for attachment in message.attachments:
				if attachment.id == a_id:
					url = str(attachment.url)
					if not discord_expired(url):
						return url.rstrip("&")
		except (AttributeError, LookupError):
			pass
		try:
			message = await channel.fetch_message(m_id)
		except discord.NotFound:
			print_exc()
			return await self.backup_url(url)
		self.add_message(message, force=True)
		for attachment in message.attachments:
			if attachment.id == a_id:
				return str(attachment.url).rstrip("&")
		return await self.backup_url(url)

	async def backup_url(self, url):
		a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
		u = url.rstrip("&")
		if discord_expired(u):
			u2 = None
			with tracebacksuppressor:
				u2 = await attachment_cache.obtain(url=u.split("?", 1)[0], m_id=0)
			if u2:
				self.data.attachments[a_id] = url
				return u2
		return u

	async def delete_attachments(self, aids=()):
		"Deletes a list of cached attachments asynchronously."
		futs = []
		for a_id in aids:
			fut = self.delete_attachment(a_id)
			futs.append(fut)
		return await gather(*futs)

	async def renew_attachments(self, aids=()):
		"Renews a list of cached attachments asynchronously."
		futs = []
		for a_id in aids:
			if is_url(a_id):
				if not is_discord_attachment(a_id) or not discord_expired(a_id):
					futs.append(as_fut(a_id))
					continue
			fut = self.renew_attachment(a_id)
			futs.append(fut)
		return await gather(*futs)

	async def fetch_role(self, r_id, guild=None):
		"Fetches a role from ID and guild, using the bot cache when possible."
		if not isinstance(r_id, int):
			try:
				r_id = int(r_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid role identifier: {r_id}")
		with suppress(KeyError):
			return self.cache.roles[r_id]
		try:
			role = guild.get_role(r_id)
			if role is None:
				raise LookupError
		except LookupError:
			if len(guild.roles) <= 1:
				roles = await guild.fetch_roles()
				guild.roles = sorted(roles)
				role = utils.get(roles, id=r_id)
			if role is None:
				raise LookupError("Role not found.")
		self.cache.roles[r_id] = role
		return role

	async def fetch_emoji(self, e_id, guild=None):
		"Fetches an emoji from ID and guild, using the bot cache when possible."
		if not isinstance(e_id, int):
			try:
				e_id = int(e_id)
			except (ValueError, TypeError):
				raise TypeError(f"Invalid emoji identifier: {e_id}")
		with suppress(KeyError):
			return self.cache.emojis[e_id]
		try:
			emoji = super().get_emoji(e_id)
			if emoji is None:
				raise LookupError
		except LookupError:
			if guild is not None:
				emoji = await guild.fetch_emoji(e_id)
			else:
				raise LookupError("Emoji not found or not usable.")
		self.cache.emojis[e_id] = emoji
		return emoji

	# Searches the bot database for a webhook proxy from ID.
	def get_mimic(self, m_id, user=None):
		if user is not None:
			with suppress(KeyError):
				mimics = self.data.mimics[user.id]
				mlist = mimics[m_id]
				return self.get_mimic(choice(mlist))
		with suppress(KeyError):
			if isinstance(m_id, int) or m_id.isnumeric():
				m_id = "&" + str(int(m_id))
			mimic = self.data.mimics[m_id]
			if not isinstance(mimic, cdict):
				self.data.mimics[m_id] = mimic = cdict(mimic)
			return mimic
		raise LookupError("Unable to find target proxy.")

	# Gets the DM channel for the target user, creating a new one if none exists.
	async def get_dm(self, user):
		if isinstance(user, discord.abc.PrivateChannel):
			return user
		with suppress(TypeError):
			int(user)
			user = await self.fetch_user(user)
		channel = user.dm_channel
		if channel is None:
			channel = await user.create_dm()
		return channel

	def get_available_guild(self, animated=True, return_all=False):
		gids = AUTH.get("emoji_servers", ())
		found = [{} for i in range(6)]
		if gids:
			guilds = [self.cache.guilds[gid] for gid in gids]
		else:
			guilds = self.guilds
		for guild in guilds:
			m = guild.me
			if m is not None and m.guild_permissions.manage_emojis:
				owners_in = self.owners.intersection(guild._members)
				if guild.owner_id == self.id:
					x = 0
				elif len(owners_in) == len(self.owners):
					x = 1
				elif guild.id == self.premium_server:
					x = 2
				elif owners_in:
					x = 3
				elif m.guild_permissions.administrator and len(deque(member for member in guild.members if not member.bot)) <= 5:
					x = 4
				else:
					x = 5
				rem = guild.emoji_limit - len(deque(e for e in guild.emojis if e.animated == animated))
				rem /= len(guild.members)
				if rem > 0:
					found[x][rem] = guild
		if return_all:
			valids = list(itertools.chain.from_iterable(v.values() for v in found))
			return valids, [guild.emoji_limit - len(deque(e for e in guild.emojis if e.animated == animated)) for guild in valids]
		for i, f in enumerate(found):
			if f:
				return f[max(f.keys())]
		raise LookupError("Unable to find suitable guild.")

	@tracebacksuppressor
	async def create_progress_bar(self, length, ratio=0.5):
		if "emojis" in self.data:
			return await self.data.emojis.create_progress_bar(length, ratio)
		position = min(length, round(length * ratio))
		return "⬜" * position + "⬛" * (length - position)

	def permissions_in(self, obj):
		user = self.user
		guild = obj
		if hasattr(guild, "members"):
			user = ([m for m in guild.members if m.id == user.id] or [user])[0]
		if hasattr(guild, "guild"):
			guild = guild.guild
		if hasattr(guild, "get_member"):
			try:
				user = guild.get_member(user.id) or user
			except Exception:
				pass
		if hasattr(obj, "permissions_for"):
			try:
				return obj.permissions_for(user)
			except (AttributeError, discord.errors.ClientException):
				pass
		if hasattr(user, "permissions_in"):
			try:
				return user.permissions_in(obj)
			except Exception:
				try:
					return obj.permissions_in(guild)
				except discord.errors.ClientException:
					pass
		if hasattr(obj, "recipient") or hasattr(obj, "dm_channel"):
			return discord.Permissions(2147483647)
		return discord.Permissions(0)

	hiscache = {}
	async def history(self, channel, limit=200, before=None, after=None, use_cache=True, full=True):
		c_id = verify_id(channel)
		c = self.in_cache(c_id)
		if c is None:
			c = channel
		if c is None:
			return
		if not is_channel(c):
			c = await self.get_dm(c)
		channel = c
		c_id = c.id
		if limit and not isfinite(limit):
			limit = None
		if type(before) in (int, float):
			if not isfinite(before):
				before = None
			else:
				before = cdict(id=before)
		if type(after) in (int, float):
			if not isfinite(after):
				after = None
			else:
				after = cdict(id=after)
		if T(channel).get("simulated"):
			return
		found = set()
		if use_cache:
			if "channel_cache" in self.data and (not limit or limit > self.hiscache.get(c_id, 0)) and self.permissions_in(channel).read_message_history:
				hist = await flatten(discord.abc.Messageable.history(channel, limit=limit, before=before, after=after, oldest_first=False))
				await self.data.channel_cache.splice(channel, hist)
				self.hiscache[c_id] = len(hist)
				for message in hist:
					self.add_message(message, files=False, force=True)
					yield message
				return
			if "channel_cache" in self.data:
				async for message in self.data.channel_cache.grab(c_id, as_message=full, force=False):
					if isinstance(message, int):
						message = cdict(id=message)
					if before:
						if message.id > time_snowflake(before):
							continue
					if after:
						if message.id < time_snowflake(after):
							break
					found.add(message.id)
					yield message
					if limit is not None and len(found) >= limit:
						return
		if self.permissions_in(channel).read_message_history:
			async for message in discord.abc.Messageable.history(channel, limit=limit, before=before, after=after, oldest_first=False):
				if message.id in found:
					continue
				self.data.deleted.cache.pop(message.id, None)
				self.hiscache.pop(c_id, None)
				self.add_message(message, files=False, force=True)
				found.add(message.id)
				yield message

	async def get_last_message(self, channel, key=None):
		m_id = T(channel).get("last_message_id")
		if m_id:
			try:
				return await self.fetch_message(m_id, channel)
			except (LookupError, discord.NotFound):
				pass
		if key:
			async for message in self.history(channel):
				if key(message):
					return message
		async for message in self.history(channel):
			return message

	async def get_last_image(self, channel):
		async for message in self.history(channel):
			try:
				return get_last_image(message)
			except FileNotFoundError:
				pass
		raise FileNotFoundError("Image file not found.")

	mime = magic.Magic(mime=True, mime_encoding=True)
	mimes = {}

	async def id_from_message(self, m_id):
		if not m_id:
			return m_id
		m_id = as_str(m_id)
		links = await self.follow_url(m_id)
		m_id = links[0] if links else m_id
		if is_url(m_id.strip("<>")) and "/emojis/" in m_id:
			return int(m_id.strip("<>").split("/emojis/", 1)[-1].split(".", 1)[0])
		if m_id[0] == "<" and m_id[-1] == ">" and ":" in m_id:
			n = m_id.rsplit(":", 1)[-1][:-1]
			if n.isnumeric():
				return int(n)
		if m_id.isnumeric():
			m_id = int(m_id)
			if m_id in self.cache.messages:
				return await self.id_from_message(self.cache.messages[m_id].content)
		return verify_id(m_id)

	followed = AutoCache(f"{CACHE_PATH}/follow", stale=3600, timeout=86400 * 7)
	async def _follow_url(self, urls, it=None, best=False, preserve=True, images=True, emojis=True, reactions=False, allow=False, limit=None, no_cache=False, ytd=True):
		if images:
			medias = ("video", "image", "thumbnail")
		else:
			medias = "video"
		out = deque()
		for url in urls:
			if discord_expired(url):
				url = await self.renew_attachment(url)
			u = T(url).get("url")
			if u:
				url = u
			if is_discord_message_link(url):
				found = deque()
				try:
					spl = url[url.index("channels/") + 9:].replace("?", "/").split("/", 2)
					c = await self.fetch_channel(spl[1])
					m = await self.fetch_message(spl[2], c)
				except Exception:
					print_exc()
				else:
					if preserve:
						for a in m.attachments:
							self.preserve_into(c.id, m.id, a.id, fn=a.url)
					# All attachments should be valid URLs
					if best:
						found.extend(best_url(a) for a in m.attachments)
					else:
						found.extend(a.url for a in m.attachments)
					found.extend(find_urls(m.content) if allow else find_urls_ex(m.content))
					for s in T(m).get("stickers", ()):
						found.append(s.url)
					# Attempt to find URLs in embed contents
					for e in m.embeds:
						for a in medias:
							obj = T(e).get(a)
							if obj:
								if best:
									url = best_url(obj)
								else:
									url = obj.url
								if url:
									found.append(url)
									break
					# Attempt to find URLs in embed descriptions
					[found.extend(find_urls(e.description) if allow else find_urls_ex(e.description)) for e in m.embeds if e.description]
					if emojis:
						temp = await self.follow_to_image(m.content, follow=reactions)
						found.extend(filter(is_url, temp))
					if images:
						if reactions:
							m = await self.ensure_reactions(m)
							for r in m.reactions:
								e = r.emoji
								if hasattr(e, "url"):
									found.append(as_str(e.url))
								else:
									u = translate_emojis(e)
									if is_url(u):
										found.append(u)
					if found:
						for u in filter(bool, found):
							# Do not attempt to find the same URL twice
							if u in it:
								continue
							it[u] = True
							if not len(it) & 255:
								await asyncio.sleep(0.2)
							found2 = await self.follow_url(u, it, best=best, preserve=preserve, images=images, emojis=emojis, reactions=reactions, allow=allow, limit=limit, ytd=ytd)
							if len(found2):
								out.extend(found2)
							# elif allow and m.content:
							# 	lost.append(m.content)
							# elif preserve:
							# 	lost.append(u)
			elif is_discord_attachment(url):
				out.append(url)
			elif is_miza_attachment(url):
				out.append(url)
			else:
				if (match := scraper_blacklist.search(url)):
					print("Interrupted:", match)
					return [url]
				try:
					resp = await create_future(
						reqs.next().head,
						unyt(url),
						headers=Request.header(),
						verify=False,
						allow_redirects=True,
						_timeout_=5,
					)
				except Exception:
					print_exc()
					out.append(url)
					continue
				if resp.headers.get("Content-Type", "").split(";", 1)[0] not in ("text/html", "application/json"):
					url = resp.url
				elif ytd and self.audio:
					try:
						resp = await self.audio.asubmit(f"ytdl.search({repr(url)})")
						if not resp:
							raise FileNotFoundError(url)
					except Exception as ex:
						print(repr(ex))
					else:
						resp = resp[0]
						if resp.get("video") and resp["video"][0]:
							url = resp["video"][0]
						elif images and resp.get("icon"):
							url = resp["icon"]
						elif resp["url"] != url:
							url = resp["url"]
						else:
							url = resp.get("stream") or resp.get("icon") or resp.get("url") or url
				out.append(url)
		# if lost:
		# 	out.extend(lost)
		if not out:
			out = urls
		return tuple(out)

	async def follow_url(self, url, it=None, best=False, preserve=True, images=True, emojis=True, reactions=False, allow=False, limit=None, no_cache=False, ytd=True) -> list:
		"Finds URLs in a string, following any discord message links found. Traces all the way to raw file stream if \"ytd\" parameter is set."
		if not url or limit is not None and limit <= 0:
			return []
		if not isinstance(url, str) and hasattr(url, "channel"):
			url = message_link(url)
		if it is None and " " in url or not is_url(url):
			urls = find_urls(url) if allow else find_urls_ex(url)
			if not urls:
				if images or emojis or reactions:
					return await self.follow_to_image(url, follow=reactions)
				return []
		else:
			urls = [url]
		it = it or {}
		urls = tuple(urls)

		if no_cache:
			return await self._follow_url(urls, it=it, best=best, preserve=preserve, images=images, emojis=emojis, reactions=reactions, allow=allow, limit=limit, no_cache=no_cache, ytd=ytd)

		tup = shash((urls, best, preserve, images, emojis, reactions, allow, ytd))
		out = await self.followed.aretrieve(
			tup, self._follow_url,
			urls, it=it, best=best, preserve=preserve, images=images,
			emojis=emojis, reactions=reactions, allow=allow, limit=limit,
			no_cache=no_cache, ytd=ytd,
		)
		for i, url in enumerate(out):
			if discord_expired(url):
				out = list(out)
				out[i] = await self.renew_attachment(url)
				self.followed[tup] = out
		return out[:limit]

	@functools.lru_cache(maxsize=64)
	def detect_mime(self, url) -> tuple:
		resp = reqs.next().get(url, stream=True, timeout=30, allow_redirects=True)
		head = fcdict(resp.headers)
		try:
			return tuple(t.strip() for t in head.get("Content-Type", "").split(";"))
		except KeyError:
			it = resp.iter_content(65536)
			data = next(it)
			return tuple(t.strip() for t in self.mime.from_buffer(data).split(";"))

	emoji_stuff = {}
	def is_animated(self, e, verify=False):
		"Detects whether an emoji is usable, and if so, animated. Returns a ternary True/False/None value where True represents an emoji that is both usable and animated, False for a usable non-animated emoji, and None for unusable emojis."
		if type(e) in (int, str):
			try:
				emoji = self.cache.emojis[e]
			except KeyError:
				e = int(e)
				if e <= 0 or e > time_snowflake(dtn(), high=True):
					return
				try:
					return self.emoji_stuff[e]
				except KeyError:
					pass
				base = f"https://cdn.discordapp.com/emojis/{e}."
				if verify:
					fut = esubmit(Request, base + "png", method="HEAD")
				url = base + "gif"
				with reqs.next().head(url, headers=Request.header(), timeout=30) as resp:
					if resp.status_code in range(400, 500):
						if not verify:
							return False
						try:
							fut.result()
						except ConnectionError:
							self.emoji_stuff[e] = None
							return
						self.emoji_stuff[e] = False
						return False
				self.emoji_stuff[e] = True
				return True
		else:
			emoji = e
		return emoji.animated

	async def proxy_emojis(self, msg, guild=None, user=None, is_webhook=False, return_pops=False, lim=400):
		"Retrieves and maps user's emoji list on target string. Used for ~AutoEmoji and compatibility with other commands."
		orig = self.bot.data.emojilists.get(user.id, {}) if user else {}
		emojis = emoji = None
		regex = regexp("(?:^|^[^<\\\\`]|[^<][^\\\\`]|.[^a\\\\`])(:[A-Za-z0-9\\-~_]{1,32}:)(?:(?![^0-9]).)*(?:$|[^0-9>`])")
		pops = set()
		offs = 0
		replaceds = []
		while offs < len(msg):
			matched = regex.search(msg[offs:])
			if not matched:
				break
			substitutes = None
			s = matched.group()
			start = matched.start()
			while s and not regexp(":[A-Za-z0-9\\-~_]").fullmatch(s[:2]):
				s = s[1:]
				start += 1
			while s and not regexp("[A-Za-z0-9\\-~_]:").fullmatch(s[-2:]):
				s = s[:-1]
			offs = start = offs + start
			offs += len(s)
			if not s:
				continue
			name = s[1:-1]
			if emojis is None:
				emojis = self.data.autoemojis.guild_emoji_map(guild, user, dict(orig))
			emoji = emojis.get(name)
			if not emoji:
				if name.isnumeric():
					emoji = int(name)
				else:
					t = name[::-1].replace("~", "-", 1)[::-1].rsplit("-", 1)
					if t[-1].isnumeric():
						i = int(t[-1])
						if i < 1000:
							if not emoji:
								name = t[0]
								emoji = emojis.get(name)
							while i > 1 and not emoji:
								i -= 1
								name = t[0] + "-" + str(i)
								emoji = emojis.get(name)
			if isinstance(emoji, int):
				e_id = await self.id_from_message(emoji)
				emoji = self.cache.emojis.get(e_id)
				if not emoji:
					animated = await asubmit(self.is_animated, e_id, verify=True)
					if animated is not None:
						emoji = cdict(id=e_id, animated=animated, name=self.data.emojinames.get(e_id))
				if not emoji and not is_webhook and user:
					self.data.emojilists.get(user.id, {}).pop(name, None)
			if emoji:
				pops.add((str(name), emoji.id))
				if len(msg) < lim:
					sub = "<"
					if emoji.animated:
						sub += "a"
					name = T(emoji).get("name") or "_"
					sub += f":{name}:{emoji.id}>"
				else:
					sub = min_emoji(emoji)
				substitutes = (start, sub, start + len(s))
				if T(emoji).get("name"):
					if not is_webhook and user:
						orig = self.data.emojilists.setdefault(user.id, {})
						orig.setdefault(name, emoji.id)
						self.data.emojinames[emoji.id] = name
				replaceds.append(emoji)
			if substitutes:
				msg = msg[:substitutes[0]] + substitutes[1] + msg[substitutes[2]:]
		if return_pops:
			return msg, pops, replaceds
		return msg

	async def emoji_to_url(self, e):
		if isinstance(e, str) and not e.isnumeric():
			e = e[3:]
			i = e.index(":")
			e_id = int(e[i + 1:e.rindex(">")])
			try:
				url = str(self.cache.emojis[e_id].url)
			except KeyError:
				anim = await asubmit(self.is_animated, e_id)
				if anim is None:
					return
				if anim:
					fmt = "gif"
				else:
					fmt = "png"
				url = f"https://cdn.discordapp.com/emojis/{e_id}.{fmt}"
		elif type(e) in (int, str):
			anim = self.is_animated(e)
			if anim is None:
				return
			fmt = "gif" if anim else "png"
			url = f"https://cdn.discordapp.com/emojis/{e}.{fmt}"
		else:
			anim = self.is_animated(e)
			if anim is None:
				return
			fmt = "gif" if anim else "png"
			url = f"https://cdn.discordapp.com/emojis/{e.id}.{fmt}"
		return url

	def emoji_exists(self, e):
		if type(e) in (int, str):
			url = f"https://cdn.discordapp.com/emojis/{e}.png"
			with reqs.next().head(url, headers=Request.header(), timeout=30) as resp:
				if resp.status_code in range(400, 500):
					self.emoji_stuff.pop(int(e), None)
					return
		else:
			if e.id not in self.cache.emojis:
				return
		return True

	async def min_emoji(self, e, full=False):
		animated = await asubmit(self.is_animated, e, verify=True)
		if animated is None:
			raise LookupError(f"Emoji {e} does not exist.")
		if type(e) in (int, str):
			e = cdict(id=e, animated=animated)
		return min_emoji(e, full=full)

	async def emoji_i2s(self, id, full=False):
		if hasattr(id, "id"):
			id = id.id
		if isinstance(id, int) or id.isnumeric():
			return await self.min_emoji(id, full=full)
		return id

	async def optimise_image(self, image, fsize=DEFAULT_FILESIZE, msize=None, fmt="auto", duration=None, anim=True, timeout=3600):
		"Optimises the target image or video file to fit within the \"fsize\" size, or \"msize\" resolution. Optional format and duration parameters."
		args = [[], None, None, "max", msize, None, "-o"]
		if not anim:
			args.insert(0, "-nogif")
		elif duration is not None:
			args += ["-d", duration]
		args += ["-fs", fsize, "-f", fmt]
		# print(args)
		return await process_image(image, "resize_map", args, timeout=timeout, retries=2)

	# Map of search engine locations for browsing the internet. As we do not have access to the user's IP address, we estimate their timezone and use that to approximate their location.
	browse_locations = {
		-11: "nz-en",	# New Zealand
		-10: "nz-en",
		-9: "us-en",	# United States
		-8: "us-en",
		-7: "us-en",
		-6: "us-en",
		-5: "ca-en",	# Canada
		-4: "ca-en",
		-3: "ca-en",
		-2: "uk-en",	# United Kingdom
		-1: "uk-en",
		0: "uk-en",
		1: "uk-en",
		2: "za-en",		# South Africa
		3: "xa-en",		# Arabia
		4: "xa-en",
		5: "in-en",		# India
		6: "in-en",
		7: "id-en",		# Indonesia
		8: "sg-en",		# Singapore
		9: "au-en",		# Australia
		10: "au-en",
		11: "au-en",
		12: "nz-en",	# New Zealand
	}
	ddgs = ddgs.DDGS()
	async def browse(self, argv, uid=0, timezone=None, region=None, timeout=60, screenshot=False, include_hrefs=False, best=True):
		"Browses the internet using DuckDuckGo or Microsoft Edge. Returns an image if screenshot is set to True."
		if not region:
			if timezone is None:
				if "users" in self.data:
					timezone = self.data.users.any_timezone(uid)
			if timezone is not None:
				timezone = round(get_offset(timezone) / 3600)
			region = self.browse_locations.get(timezone, "wt-wt")
		async def retrieval(argv, region="wt-wt", screenshot=False, best=False):
			if not is_url(argv):
				print("Browse query:", repr(argv))
				with tracebacksuppressor:
					data = None
					if best:
						try:
							data = await asubmit(self.ddgs.text, argv, safesearch="off", region=region, max_results=20 if best else 10)
						except ddgs.exceptions.DuckDuckGoSearchException:
							print_exc()
					if not data:
						headers = Request.header()
						headers.Referer = "https://duckduckgo.com"
						headers.Accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
						resp = await asubmit(niquests.post, "https://lite.duckduckgo.com/lite/", data=dict(q=argv), headers=headers)
						from bs4 import BeautifulSoup
						data = BeautifulSoup(resp.content)
						text = re.sub("\n{2,}", "\n\n", data.get_text().split("Any Time\n", 1)[-1].split("Past Year\n", 1)[-1].strip())
						return text
					if include_hrefs:
						return "\n\n".join("[" + (e.get("title", "") + "](" + e.get("href", "") + ")\n" + e.get("body", "")).strip() for e in data).strip()
					return "\n\n".join((e.get("title", "") + "\n" + e.get("body", "")).strip() for e in data).strip()
			return await process_image("browse", "$", [argv, not screenshot], cap="browse", timeout=timeout, retries=2)
		urls = find_urls(argv)
		if not urls:
			urls.append(argv)
		futs = [csubmit(ai.cache.aretrieve(shash((argv, screenshot, best)), retrieval, argv, region, screenshot, best)) for argv in urls]
		resp = await gather(*futs)
		return "\n\n\n".join(resp) if isinstance(resp[0], str) else b"\n\n\n".join(resp)

	async def function_call(self, *args, is_nsfw=None, backup_models=True, assistant_name=None, stream=False, models=[], model=None, **kwargs):
		h = shash((args, kwargs))
		if not stream:
			try:
				return ai.cache[h]
			except KeyError:
				pass
		if backup_models:
			models.extend((
				"grok-4-fast",
			))
		if model:
			if model in models:
				models.remove(model)
			models.insert(0, model)
		model = models[0]
		fut = csubmit(self.caption_into(kwargs["messages"], model=model, premium_context=kwargs.get("premium_context", [])))
		if is_nsfw is None:
			mod1 = json_dumps(kwargs.get("tools"))
			mod2 = kwargs.get("messages")
			pc = kwargs.get("premium_context", [])
			futs = [ai.moderate(mod1, premium_context=pc), ai.moderate(mod2, premium_context=pc)]
			r1, r2 = await gather(*futs)
			is_nsfw = nsfw_flagged(r1) or nsfw_flagged(r2)
		if is_nsfw:
			models = reversed(models)
		kwargs["messages"], _model = await fut
		exc = None
		for model in models:
			kwargs["model"] = model
			if model in ai.is_reasoning:
				kwargs["reasoning_effort"] = "low"
			else:
				kwargs.pop("reasoning_effort", None)
			try:
				resp = await ai.llm("chat.completions.create", *args, stream=False, **kwargs)
			except Exception as ex:
				if not exc:
					exc = ex
				print(repr(ex))
			else:
				if not stream:
					try:
						m = resp.choices[0].message
					except Exception:
						print(resp)
						raise
					if m.content and assistant_name:
						content = m.content.strip()
						if (content.startswith("name=") or content.startswith("Name=")) and "\n" in content:
							content = content.split("\n", 1)[-1]
						elif content.startswith(assistant_name + ":"):
							content = content.split(":", 1)[-1]
						m.content = content.strip()
					ai.cache[h] = resp
				return resp
		raise (exc or RuntimeError("Unknown error occured."))

	async def force_completion(self, model, prompt, stream=True, max_tokens=1024, strip=True, **kwargs):
		ctx = ai.contexts.get(model, 4096)
		is_question = prompt.endswith(".") or prompt.endswith("?")
		if model in ai.is_completion and (not model in ai.is_chat or not is_question) or model not in ai.is_chat:
			count = await tcount(prompt, model="llamav2")
			max_tokens = min(max_tokens, ctx - count - 64)
			if "max_completion_tokens" not in kwargs:
				kwargs["max_tokens"] = max_tokens
			resp = await ai.llm("completions.create", model=model, prompt=prompt, stream=True, **kwargs)
			async def _completion(resp, strip):
				async for r in resp:
					if not r.choices:
						continue
					s = r.choices[0].text or ""
					if s and strip:
						yield s.lstrip()
						strip = False
						continue
					yield s
				return
			return CloseableAsyncIterator(_completion(resp, strip), resp.close)
		messages = [cdict(role="user", content=prompt)]
		count = await count_to(messages)
		max_tokens = min(max_tokens, ctx - count - 64)
		if "max_completion_tokens" not in kwargs:
			kwargs["max_tokens"] = max_tokens
		resp = await ai.llm("chat.completions.create", model=model, messages=messages, stream=True, **kwargs)
		async def _completion(resp, strip):
			async for r in resp:
				if not r.choices:
					continue
				s = r.choices[0].delta.content or ""
				if s and strip:
					yield s.lstrip()
					strip = False
					continue
				yield s
		return CloseableAsyncIterator(_completion(resp, strip), resp.close)

	async def force_chat(self, model, messages, text=None, assistant_name=None, stream=False, max_tokens=1024, vision_model=None, **kwargs):
		ctx = ai.contexts.get(model, 4096)
		messages, vision_model = await self.caption_into(messages, model=model, backup_model=vision_model, premium_context=kwargs.get("premium_context", []))
		if vision_model in ai.is_chat:
			count = await count_to(messages)
			max_tokens = min(max_tokens, ctx - count - 64)
			if "max_completion_tokens" not in kwargs:
				kwargs["max_tokens"] = max_tokens
			return await ai.llm("chat.completions.create", model=vision_model, messages=messages, stream=stream, **kwargs)
		fmt = ai.instruct_formats.get(model, "chatml")
		assistant_messages = [m for m in messages if m.get("content") and m.get("role") == "assistant"]
		if assistant_name:
			bot_name = assistant_name
		elif not assistant_messages:
			bot_name = None
		else:
			assistant_names = [(m.get("name") or (m["content"].split(":", 1)[0] if ":" in m["content"] else "")) for m in assistant_messages]
			bot_names = [n for n in assistant_names if n]
			if not bot_names:
				bot_name = None
			else:
				bot_name = bot_names[-1]
		prompt, stopn = instruct_structure(messages, fmt=fmt, assistant=bot_name)
		if text:
			prompt += " " + text
		kwargs["stop"] = list(set(tuple(kwargs.get("stop", ())) + tuple(stopn)))
		if model in ai.is_reasoning:
			kwargs["reasoning_effort"] = "low"
		data = dict(
			model=model,
			prompt=prompt,
			**kwargs,
		)
		print("CC:", data)
		count = await tcount(prompt, model="llamav2")
		max_tokens = min(max_tokens, ctx - count - 64)
		if "max_completion_tokens" not in kwargs:
			kwargs["max_tokens"] = max_tokens
		resp = await ai.llm("completions.create", stream=stream, **data)
		if stream:
			async def stream_iter(resp):
				name = None
				found = deque()
				nt = await tcount(assistant_name, model="llamav2")
				async for chunk in resp:
					if not chunk.choices:
						if getattr(chunk, "error_message"):
							e = orjson.loads(chunk.error_message.split(":", 1)[-1])
							raise ConnectionError(e.get("code", 510), e.get("message"))
						continue
					choice = chunk.choices[0]
					text = choice.text
					found.append(text)
					if len(found) < 5 + nt:
						continue
					if not name:
						temp = ""
						while f"name={assistant_name}".startswith(temp) or f"{assistant_name}:".startswith(temp):
							text = found.popleft()
							temp += text
						if temp and temp != text and "\n" in temp:
							name, text = temp.split("\n", 1)
							name = name.removeprefix("name=").strip()
						elif temp and temp != text and ":" in temp:
							name, text = temp.split(":", 1)
							name = name.strip()
							text = text.strip()
					yield cdict(
						id=chunk.id,
						choices=[cdict(
							finish_reason=choice.finish_reason,
							index=0,
							logprobs=None,
							# text=text,
							delta=cdict(role="assistant", content=text, tool_calls=None),
						)],
						created=T(chunk).get("created") or floor(utc()),
						model=T(chunk).get("model") or model,
						object="chat.completion.chunk",
					)
				text = "".join(found).rstrip()
				text = text.removesuffix("###").removesuffix("|").removesuffix("im_end").removesuffix("<|").rstrip()
				if not text:
					return
				yield cdict(
					id=chunk.id,
					choices=[cdict(
						finish_reason=choice.finish_reason,
						index=0,
						logprobs=None,
						refusal=T(choice).get("refusal"),
						# text=text,
						delta=cdict(role="assistant", name=name, content=text, tool_calls=None),
					)],
					created=T(chunk).get("created") or floor(utc()),
					model=T(chunk).get("model") or model,
					object="chat.completion.chunk",
				)
			return CloseableAsyncIterator(stream_iter(resp), resp.close)
		choice = resp.choices[0]
		text = choice.text.strip().removesuffix("###").removesuffix("|").removesuffix("im_start").removesuffix("im_end").removesuffix("<|").strip()
		if assistant_name:
			text = text.removeprefix("name=" + assistant_name).removeprefix(assistant_name + ":").strip()
		return cdict(
			id=resp.id,
			choices=[cdict(
				finish_reason=choice.finish_reason,
				index=0,
				logprobs=None,
				refusal=T(choice).get("refusal"),
				# text=text,
				message=cdict(role="assistant", content=text, tool_calls=None),
			)],
			created=T(chunk).get("created") or floor(utc()),
			model=T(chunk).get("model") or model,
			object="chat.completion",
			usage=resp.usage,
		)

	async def caption_into(self, _messages, model=None, backup_model=None, premium_context=[]):
		# print("Resolving Images:", model, backup_model, lim_str(_messages, 1024))
		context = ai.contexts.get(model, 4096)
		messages = [cdict(m) for m in _messages]
		follows = [None] * len(messages)
		for j, m in enumerate(reversed(messages)):
			i = len(messages) - j - 1
			if isinstance(m.get("content"), list):
				cont = m.content
				m.content = ""
				urls = []
				for c in cont:
					if c.get("type") == "text":
						if m.content:
							m.content += "\n\n"
						m.content += c.get("text", "")
					elif c.get("type") == "image":
						d = c["data"]
						url = "data:" + (c.get("media_type") or magic.from_buffer(d)) + ";base64," + d
						urls.append(url)
					elif c.get("type") == "image_url":
						url = c["image_url"]["url"]
						urls.append(url)
					else:
						raise TypeError(c["type"])
				follows[i] = as_fut(urls)
			elif sum(f is not None for f in follows) < 4 and m.get("url") and j < 8:
				extract = not is_discord_message_link(m.url) and not is_discord_attachment(m.url) and m.get("new")
				follows[i] = csubmit(self.follow_url(m.url, ytd=extract))
			m.pop("url", None)
		for i, fut in enumerate(follows):
			if not fut:
				continue
			try:
				urls = await fut
			except Exception:
				print_exc()
				continue
			urls = [url for url in urls if not is_discord_message_link(url)]
			if not urls:
				continue
			follows[i] = urls
		extracts = [None] * len(messages)
		for i, (m, urls) in enumerate(zip(messages, follows)):
			if not urls:
				continue
			if m.get("url"):
				for url in list(urls) + [m.get("url")]:
					if not url:
						continue
					m.content = m.content.replace(url, "", 1).strip()
			if model in ai.is_vision and m.get("role") != "assistant":
				futs = [self.to_data_url(url, small=not m.get("new")) for url in urls]
				extracts[i] = csubmit(gather(*futs))
			elif m.get("new") and backup_model and backup_model in ai.is_vision and m.get("role") != "assistant":
				futs = [self.to_data_url(url, small=False) for url in urls]
				extracts[i] = csubmit(gather(*futs))
				if futs and extracts:
					model = backup_model
					print("Auto-Selecting:", backup_model, m, urls)
			else:
				best = 2 if model in ai.is_premium and m.get("new") else 0
				futs = [self.caption(url, best=best, premium_context=premium_context) for url in urls]
				extracts[i] = csubmit(gather(*futs, return_exceptions=True))
		for i, (m, fut) in enumerate(zip(messages, extracts)):
			if not fut:
				continue
			best = 2 if model in ai.is_premium and m.get("new") else 0
			try:
				captions = await fut
			except Exception as ex:
				print("Caption Error:", repr(ex))
				continue
			images = []
			for caption in captions:
				if isinstance(caption, BaseException):
					print("Caption Error:", repr(caption))
					continue
				if isinstance(caption, tuple):
					caption = "<" + ":".join(caption) + ">"
				if not caption.startswith("data:"):
					if not m.get("new"):
						caption = lim_tokens(caption, 256)
					else:
						caption = await ai.summarise(caption, min_length=context / 3, best=True, premium_context=premium_context)
					m.content += "\n\n" + caption
				else:
					im = cdict(type="image_url", image_url=cdict(url=caption, detail="auto" if best else "low"))
					images.append(im)
			if images:
				m.content = [cdict(type="text", text=m.content)]
				m.content.extend(images)
		for m in messages:
			m.pop("new", None)
		return messages, model

	async def classify(self, content, examples=[], model="embed-multilingual-v3.0", premium_context=[]):
		if model == "embed-multilingual-v3.0" and len(content) > 1024:
			if content.isascii():
				model = "embed-english-light-v3.0"
			else:
				model = "embed-multilingual-light-v3.0"
		inputs = dict(
			inputs=[content],
			examples=examples,
			model=model,
		)
		data = await Request(
			"https://api.cohere.ai/v1/classify",
			method="POST",
			headers={
				"Authorization": "Bearer " + AUTH["cohere_key"],
				"Content-Type": "application/json",
			},
			data=json_dumps(inputs),
			aio=True,
			json=True,
			timeout=24,
		)
		print("EVALUATION:", data)
		premium_context.append(["cohere", model, "0.00005"])
		return data["classifications"][0]["prediction"]

	async def evaluate(self, messages, premium_context=[]):
		contents = "\n\n".join(map(m_str, messages))
		try:
			label = await self.classify(contents, model="c31ba1d3-2e68-4f70-954b-43cc93846b2d-ft", premium_context=premium_context)
		except Exception:
			print_exc()
			return "ok"
		return "refusal" if label == "Assistant Refused" else "insufficient" if label == "Assistant Misunderstood" else "ok"

	model_levels = {
		0: cdict(
			reasoning="gpt-5-mini",
			instructive="gpt-oss-120b",
			casual="gemini-2.5-flash",
			nsfw="grok-4-fast",
			backup="deepseek-v3",
			retry="gpt-5-mini",
			function="gpt-5-nano",
			vision="mistral-24b",
			target="auto",
		),
		1: cdict(
			reasoning="gpt-5-mini",
			instructive="gpt-5-mini",
			casual="gemini-2.5-flash-t",
			nsfw="grok-4-fast",
			backup="kimi-k2",
			retry="claude-4.5-haiku",
			function="grok-4-fast",
			vision="gemini-2.5-flash-t",
			target="auto",
		),
		2: cdict(
			reasoning="gpt-5",
			instructive="gpt-5",
			casual="grok-4",
			nsfw="grok-4",
			backup="kimi-k2",
			retry="claude-4.5-sonnet",
			function="grok-4-fast",
			vision="gpt-5",
			target="auto",
		),
	}
	async def chat_completion(self, messages, model="miza-1", system=None, frequency_penalty=None, presence_penalty=None, repetition_penalty=None, max_tokens=256, temperature=0.7, top_p=0.9, tools=None, tool_choice=None, model_router=None, tool_router=None, stop=(), user=None, props=None, stream=True, tinfo=None, allow_nsfw=False, predicate=None, premium_context=[], **void):
		"OpenAI-compatible Chat Completion function. Autoselects model using a function call, then routes to tools and target model as required."
		await require_predicate(predicate)
		modlvl = ["miza-1", "miza-2", "miza-3"].index(model.rsplit("/", 1)[-1])
		modelist = self.model_levels[modlvl]
		messages = [cdict(m) for m in messages]
		if system:
			messages.insert(0, cdict(role="system", content=system))
		prompt = [m.content for m in messages if m.get("role") == "user"][-1]
		if modlvl >= 2:
			maxlim = 196608
			minlim = 2500
			snip = 800
			best = 2
		elif modlvl >= 1:
			maxlim = 98304
			minlim = 1000
			snip = 320
			best = 1
		else:
			maxlim = 3000
			minlim = 500
			snip = 240
			best = 0
		messages = await ai.cut_to(messages, maxlim, minlim, best=best, prompt=prompt, premium_context=premium_context)
		length = await count_to(messages)
		length = ceil(length * 1.1) + 4 * len(messages)
		tmp = temperature
		tpp = top_p
		fp = frequency_penalty
		pp = presence_penalty
		rp = repetition_penalty
		if not rp:
			if not fp and not pp:
				fp = 0.6
				pp = 0.4
			rp = ai.cast_rp(fp, pp, model=model)
		elif not fp and not pp:
			fp = rp - 1
			pp = 0
		def force_ua(r):
			if r == "assistant":
				return r
			return "user"
		raws = [cdict(role=force_ua(m.get("role")), content=m.content) for m in messages]
		snippet = await ai.cut_to(raws, snip, snip, best=False, simple=True)
		sniplen = await count_to(snippet)
		text = ""
		ustr = str(hash(str(user) or self.user.name))
		if tool_choice == "auto":
			tool_choice = None
		cid = hex(ts_us()).removeprefix("0x") + "-Miza"
		if not props:
			props = {}
		assistant_name = props.get("name")
		cargs = props.get("cargs") or {}
		is_nsfw = cargs.get("nsfw")
		message = None
		if not cargs:
			content = messages[-1].content
			mod = await ai.moderate(messages[-3:], premium_context=premium_context)
			cargs["nsfw"] = is_nsfw = nsfw_flagged(mod)
			toolscan = tools
			if isinstance(toolscan, dict):
				temp = []
				for tooln in toolscan.values():
					for tc in tooln:
						if tc not in temp:
							temp.append(tc)
				toolscan = temp
			if modelist.instructive != modelist.casual:
				users = 0
				toolcheck = []
				for m in reversed(snippet):
					toolcheck.append(m)
					if m.get("role") == "user":
						users += 1
						if users > 1:
							break
				toolcheck.append(messages[0])
				toolcheck.reverse()
				vision_alt = modelist.vision if modelist.function not in ai.is_vision and modelist.vision in ai.is_function else modelist.function
				toolcheck, toolmodel = await self.caption_into(toolcheck, model=modelist.function, backup_model=vision_alt, premium_context=premium_context)
				mode = None
				label = "instructive"
				try:
					resp = await self.function_call(
						model=toolmodel,
						messages=toolcheck,
						temperature=tmp,
						top_p=tpp,
						frequency_penalty=fp,
						presence_penalty=pp,
						repetition_penalty=rp,
						tools=list(toolscan) + [f_default],
						tool_choice="required",
						require_message=False,
						max_tokens=min(256, max_tokens),
						user=ustr,
						stop=stop,
						assistant_name=assistant_name,
						is_nsfw=is_nsfw,
						premium_context=premium_context,
					)
					message = resp.choices[0].message
				except Exception:
					print_exc()
					message = None
				print("SCAN:", cargs, message)
		if message:
			directly_answer = True
			for tc in tuple(message.tool_calls or ()):
				if tc.function.name == "directly_answer":
					try:
						args = cdict(eval_json(tc.function.arguments))
					except Exception:
						print(tc.function.arguments)
						print_exc()
						args = {}
					print(args)
					if args.get("format"):
						mode = args["format"]
					message.tool_calls.remove(tc)
					break
				else:
					directly_answer = False
			if not directly_answer and message.tool_calls:
				print("Immediate call:", message)
				choice = resp.choices[0]
				st = await count_to(messages)
				ct = await tcount(message.content)
				if is_nsfw:
					label = "nsfw"
				cargs["mode"] = mode = label
				yield cdict(
					id=cid,
					choices=[cdict(
						finish_reason=choice.finish_reason,
						index=0,
						logprobs=None,
						delta=cdict(
							content=getattr(message, "content", None) or None,
							role=getattr(message, "role", "assistant"),
							**(dict(name=message.name) if getattr(message, "name", None) else {}),
							tool_calls=getattr(message, "tool_calls", None),
						)
					)],
					created=getattr(resp, "created", None) or floor(utc()),
					source_model=getattr(resp, "model", None) or model,
					model=f"Miza/{model}",
					object="chat.completion.chunk",
					usage=cdict(
						completion_tokens=ct,
						prompt_tokens=st,
						total_tokens=ct + st,
					),
					cargs=cargs,
					modelist=modelist,
				)
				return
			if mode:
				model_router = None
				label = mode
				cargs["mode"] = label
			else:
				if model_router:
					label = await self.classify(content, examples=model_router, premium_context=premium_context)
			if tool_router:
				if not isinstance(tools, dict):
					tools = {f["function"]["name"]: [f] for f in tools if "function" in f}
				try:
					label = await self.classify(content, examples=tool_router, premium_context=premium_context)
				except Exception:
					print_exc()
					tools = toolscan
				else:
					tools = tools[label]
			elif isinstance(tools, dict):
				tools = toolscan
			else:
				tools = tools or None
			cargs["tools"] = tools
			if is_nsfw:
				print(mod, allow_nsfw)
				label = "nsfw" if allow_nsfw else "casual"
			cargs["mode"] = label
		decensor = not is_nsfw or allow_nsfw
		tools = cargs.get("tools")
		mode = cargs.get("mode", "casual")
		if mode not in ("instructive", "casual", "nsfw"):
			mode = "instructive"
		# if mode != "nsfw":
		# 	ps = [m for m in messages if m.get("new")]
		# 	for m in ps:
		# 		url = m.get("url")
		# 		if url:
		# 			urls = await self.follow_url(url)
		# 			if urls:
		# 				url = urls[0]
		# 				if is_discord_message_link(url):
		# 					url = None
		# 					m.pop("url")
		# 			else:
		# 				url = None
		# 				m.pop("url")
		# 		if url:
		# 			mode = "vision"
		# 			break
		mA = 4 if not allow_nsfw else 6 if model == "miza-3" else 5
		draft = monologue = None
		last_successful = None
		finish_reason = "end"
		result = cdict(
			id=cid,
			choices=[cdict(
				finish_reason=None,
				index=0,
				logprobs=None,
			)],
			created=0,
			object="chat.completion.chunk",
			cargs=cargs,
		)
		ex = None
		print("Chat completions:", model, ai.overview(messages), cargs, sep="\n")
		tmpcut = None
		for attempts in range(mA):
			await require_predicate(predicate)
			assistant = modelist[mode]
			ctx = ai.contexts.get(assistant, 4096)
			ml = min(max(32, min(128, ctx - length)), max_tokens)
			resp = None
			insufficient = False
			refusal = False
			result.model = result.get("model") or assistant
			ctx = ai.contexts.get(assistant, 4096)
			passable = not modelist.target or assistant == modelist.target or modelist.target == "auto" or attempts >= mA - 1
			if not passable:
				temp = snippet
				tlen = sniplen
			elif length >= ctx * 2 / 3:
				if tmpcut:
					temp = tmpcut
					tlen = tmplen
				else:
					temp = tmpcut = await ai.cut_to(messages, 65536, ctx // 3, best=True, premium_context=premium_context)
					tmplen = await count_to(tmpcut)
					tlen = tmplen = ceil(tmplen * 1.1) + 4 * len(tmpcut)
			else:
				temp = messages
				tlen = length
			ml = min(max(256, min(8192, ctx - tlen)), max_tokens)
			data = dict(
				model=assistant,
				vision_model=modelist.vision,
				messages=temp,
				assistant_name=assistant_name,
				temperature=tmp,
				top_p=tpp,
				frequency_penalty=fp,
				presence_penalty=pp,
				repetition_penalty=rp,
				max_tokens=ml,
				user=ustr,
				stop=stop,
			)
			if tools and assistant in ai.is_function:
				data["tools"] = tools
				if text:
					yield "\r"
					text = ""
			elif assistant in ai.is_chat:
				if text:
					yield "\r"
					text = ""
			else:
				if text.startswith("\r"):
					yield "\r"
				text = text.strip()
				data["text"] = text
			try:
				resp = await self.force_chat(**data, premium_context=premium_context, stream=True, timeout=90)
			except openai.BadRequestError:
				raise
			except Exception as e:
				ex = e
				print_exc()
				refusal = True
			else:
				print("Response chosen:", T(resp).get("model", assistant), tlen, resp)
				message = None
				written = False
				try:
					async for chunk in resp:
						await require_predicate(predicate)
						if not chunk.choices:
							if getattr(chunk, "error_message"):
								e = orjson.loads(chunk.error_message.split(":", 1)[-1])
								raise ConnectionError(e.get("code", 510), e.get("message"))
							continue
						finish_reason = chunk.choices[0].finish_reason or finish_reason
						delta = chunk.choices[0].delta
						if not message:
							message = cdict(delta)
							text += message.content or ""
						else:
							if delta.content:
								content = (message.content or "") + delta.content
								message.content = content
								if delta.content[0] == "\r":
									text = delta.content[1:]
								else:
									text += delta.content
							if delta.tool_calls:
								message.tool_calls = message.tool_calls or []
								for tc in delta.tool_calls:
									if tc.index >= len(message.tool_calls):
										message.tool_calls.append(tc)
									else:
										of = message.tool_calls[tc.index].function
										if tc.function.name:
											of.name = (of.name or "") + tc.function.name
										if tc.function.arguments:
											of.arguments = (of.arguments or "") + tc.function.arguments
						if T(delta).get("refusal") or text and attempts < mA - 1 and decensor and len(text) < 512 and ai.decensor.search(text) or text.rstrip(": \n") == assistant_name:
							refusal = True
							break
						if delta.content and not message.tool_calls:
							choice = result.choices[0]
							result.update(chunk)
							choice.update(cdict(chunk.choices[0]))
							result.choices[0] = choice
							if not T(choice.delta).get("name") and (text.startswith("name") or text.startswith("Name") or text.startswith(assistant_name)):
								if text.startswith(assistant_name + ": "):
									text = text.removeprefix(assistant_name + ": ")
									naming = assistant_name
								else:
									if "\n" not in text:
										continue
									naming, text = text.split("\n", 1)
									if "=" not in naming:
										continue
								result.choices[0].delta.content = text
								result.choices[0].delta.name = naming.split("=", 1)[-1].rstrip()
							if passable:
								yield result
							written = True
						elif written and message.tool_calls:
							if passable:
								yield "\r"
							written = False
				except (httpx.RemoteProtocolError, ConnectionError, openai.APIError):
					print_exc()
					insufficient = True
				finally:
					if getattr(resp, "close", None):
						await resp.close()
				if message:
					if getattr(message, "tool_calls", None):
						print("Output call:", message)
						st = tlen
						ct = await tcount(text)
						yield cdict(
							id=cid,
							choices=[cdict(
								finish_reason=finish_reason,
								index=0,
								logprobs=None,
								delta=cdict(
									content=text or None,
									role=getattr(message, "role", "assistant"),
									**(dict(name=message.name) if getattr(message, "name", None) else {}),
									tool_calls=getattr(message, "tool_calls", None),
								)
							)],
							created=getattr(resp, "created", None) or floor(utc()),
							source_model=getattr(resp, "model", None) or model,
							model=f"Miza/{model}",
							object="chat.completion.chunk",
							usage=result.get("usage"),
							cargs=cargs,
						)
						return
			eval1 = None
			eval2 = None
			if not text:
				insufficient = True
			if decensor and attempts < mA - 1:
				if ai.decensor.search(text):
					refusal = True
				# if not passable and not insufficient and not refusal and modlvl >= 1:
				# 	eval1 = await ai.moderate(text, premium_context=premium_context)
				# 	if not nsfw_flagged(eval1):
				# 		for m in reversed(messages):
				# 			if m.get("role") == "user":
				# 				break
				# 		else:
				# 			m = messages[-1]
				# 		m = cdict(m)
				# 		if m.content and isinstance(m.content, str):
				# 			m.content = lim_tokens(m.content, 512)
				# 		ms = [m, cdict(role="assistant", content=text)]
				# 		# ms = await ai.cut_to(ms, 400, 400, simple=True)
				# 		# ms.append(cdict(role="assistant", content=text))
				# 		arg = await self.evaluate(ms, premium_context=premium_context)
				# 		if arg == "refusal":
				# 			refusal = True
				# 		if arg == "insufficient":
				# 			insufficient = True
				if not last_successful:
					last_successful = text
				elif not refusal or not insufficient:
					last_successful = text
			elif not text and last_successful:
				finish_reason = "attempts"
				text = draft.content if draft else last_successful
				insufficient = refusal = False
			if insufficient or refusal:
				print("Evaluation:", attempts, lim_str(text, 128), eval2, insufficient, refusal)
			if not insufficient and not refusal and passable:
				text = (text or "").rstrip().removesuffix("### End").removesuffix("### Response").removesuffix("<|im_end|>").rstrip().removesuffix("###").rstrip()
				ct = await tcount(text)
				usage = resp.usage if attempts == 0 else cdict(
					completion_tokens=ct,
					prompt_tokens=length,
					total_tokens=ct + length,
				)
				result.update(dict(
					id=cid,
					choices=[cdict(
						finish_reason=finish_reason,
						index=0,
						logprobs=None,
						delta=cdict(
							role="assistant",
							name=assistant_name,
							content="",
						),
					)],
					created=getattr(result, "created", None) or floor(utc()),
					source_model=getattr(result, "model", None) or model,
					model=f"Miza/{model}",
					usage=usage,
				))
				result.choices[0].delta.content = "\r" + text
				yield result
				return
			if refusal:
				if attempts < 1 and mA > 2:
					mode = "instructive" if mode == "casual" else "backup"
				else:
					mode = "backup"
				text = "\r"
			elif insufficient:
				if attempts < 1 and mA > 2:
					mode = "retry" if mode == "instructive" else "instructive"
				else:
					mode = "backup" if mode == "retry" else "retry"
				if text and mode in ("instructive", "casual", "retry"):
					content = f"### Instruction:\nThe above assistant's response was deemed insufficient. Please rewrite the message, ensuring to answer in a more accurate and helpful way. Remember to stay in character as {assistant_name}, and to use the same language the user last spoke in, unless directed otherwise!\n\n### Response:"
					if draft:
						draft.content = text
						monologue.content = content
					else:
						draft = cdict(role="assistant", content=text)
						messages.append(draft)
						monologue = cdict(role="user", content=content)
						messages.append(monologue)
					tmpcut = None
					length = await count_to(messages)
					length = ceil(length * 1.1) + 4 * len(messages)
				text = "\r"
			else:
				mode = "target"
				if text:
					content = f"### Instruction:\nAbove is a sample response from another automated assistant. Please rewrite the message, ensuring to better stay in character as {assistant_name}. Remember to use the same language the user last spoke in, unless directed otherwise!\n\n### Response:"
					if draft:
						draft.content = text
						monologue.content = content
					else:
						draft = cdict(role="assistant", content=text)
						messages.append(draft)
						monologue = cdict(role="user", content=content)
						messages.append(monologue)
					tmpcut = None
					length = await count_to(messages)
					length = ceil(length * 1.1) + 4 * len(messages)
				text = "\r"
		raise ex or RuntimeError("Maximum inference attempts exceeded (model likely encountered an infinite loop).")

	async def req_data(self, url, screenshot=False, timeout=20):
		resp = None
		if isinstance(url, str):
			if url.startswith("data:") and "base64," in url:
				durl = url.split("base64,", 1)[-1].encode("ascii")
				d = base64.b64decode(durl + b"==")
			else:
				d = await self.get_request(url)
			name = url.rsplit("/", 1)[-1].split("?", 1)[0]
		else:
			d = url
			name = None
		mime = resp.headers.get("Content-Type", "") if resp else magic.from_buffer(d)
		if mime == "text/html" and screenshot:
			d = await self.browse(url, best=True, timeout=timeout + 8, screenshot=True)
			mime = magic.from_buffer(d)
		return resp, mime, name, d

	analysed = AutoCache(f"{CACHE_PATH}/caption", stale=86400, timeout=86400 * 7)
	async def caption(self, url, best=False, screenshot=False, timeout=24, premium_context=[]):
		"Produces an AI-generated caption for an image. Model used is determined by \"best\" argument."
		h = shash((url, best))
		s = await self.analysed.aretrieve(h, self.vision, url, best=best, timeout=timeout)
		return ("Image", s)

	caption_prompt = "Please describe this image in detail; be descriptive but concise!"
	description_prompt = "Please describe this <IMAGE> in detail:\n- The image may be a collage of frames representing a video, in which case it should be analysed as if it were one\n- Transcribe text if present, but do not mention there not being text\n- Note details especially for people/characters if present\n- Be descriptive but concise!"

	async def to_data_url(self, url, small=False, fmt="jpg", timeout=8):
		sizelim = 82944 if small else 1638400
		dimlim = 256 if small else 1024
		if isinstance(url, str):
			if url.startswith("data:"):
				return url
			if not is_url(url):
				if not os.path.exists(url):
					return url
				with open(url, "rb") as f:
					d = await asubmit(f.read)
				mime = magic.from_buffer(d)
				name = url.replace("\\", "/").rsplit("/", 1)
			else:
				_resp, mime, name, d = await self.req_data(url, timeout=timeout, screenshot=True)
			lim = 5 * 1048576 * 3 / 4
			p = 2 if len(d) > 1048576 else 0
			if mime not in ("image/jpg", "image/jpeg", "image/png") or len(d) > lim or np.prod(await asubmit(get_image_size, d, priority=p)) > sizelim:
				if mime.split("/", 1)[0] not in ("image", "video"):
					if len(d) > 288 and mime not in ("text/plain", "text/html"):
						d = d[:128] + b".." + d[-128:]
					s = as_str(d)
					return f'<file name="{name}">' + s + "</file>"
				assert isinstance(d, (str, bytes)), d
				d = await process_image(d, "resize_map", [[], None, None, "rel", dimlim, "-", "auto", "-bg", "-oz", "-fs", lim, "-f", "jpg"], timeout=24)
		else:
			d = url
		mime = magic.from_buffer(d)
		return "data:" + mime + ";base64," + base64.b64encode(d).decode("ascii")

	async def vision(self, url, name=None, best=True, model=None, question=None, premium_context=[], timeout=12):
		"Requests an image description from a vision-supporting LLM."
		if name:
			iname = f'image "{name}"'
		elif isinstance(url, str) and is_url(url):
			iname = f"image {url2fn(url)}"
		else:
			iname = "image"
		data_url = await self.to_data_url(url, timeout=timeout)
		if not data_url.startswith("data:"):
			return data_url
		content = (question or self.description_prompt).replace("<IMAGE>", iname)
		messages = [
			cdict(role="user", content=[
				cdict(type="text", text=content),
				cdict(type="image_url", image_url=cdict(url=data_url, detail="auto" if best else "low")),
			]),
		]
		model = model or ("gpt-5" if best else "gemini-2.5-flash")
		messages, _model = await self.caption_into(messages, model=model, premium_context=premium_context)
		data = cdict(
			model=model,
			messages=messages,
			temperature=0.5,
			max_tokens=2048,
			top_p=0.9,
			frequency_penalty=0.6,
			presence_penalty=0.8,
			user=str(hash(self.name)),
		)
		print("Vision Input:", lim_str(data, 1024))
		async with asyncio.timeout(timeout):
			response = await ai.llm("chat.completions.create", premium_context=premium_context, **data, timeout=timeout)
		out = response.choices[0].message.content.strip()
		if ai.decensor.search(out):
			raise ValueError(f"Failed or censored response: {repr(out)}.")
		print("Vision Response:", out)
		return out

	async def ocr(self, url):
		resp = await process_image(url, "caption", ["-nogif"], cap="caption", timeout=3600)
		data = await self.to_data_url(url)
		mistral_key = AUTH.get("mistral_key")
		s = None
		if mistral_key:
			mistral_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {mistral_key}"}
			resp = await Request(
				"https://api.mistral.ai/v1/ocr",
				method="POST",
				headers=mistral_headers,
				data=orjson.dumps(dict(
					model="mistral-ocr-latest",
					document=dict(type="image_url", image_url=data)
				)),
				json=True,
				aio=True,
			)
			print(resp)
			s = "\n\n".join(page["markdown"] for page in resp["pages"]).strip()
			if s == "![img-0.jpeg](img-0.jpeg)":
				s = None
		if not s:
			s = await self.vision(
				data,
				name=url2fn(url),
				question="Please transcribe all text within this <IMAGE>, as accurately as possible. Leave all text in their original language, using unicode if necessary, and do NOT attempt to describe any other elements within the picture.",
				model="mistral-24b",
			)
		if not s:
			fut = asubmit(__import__, "pytesseract")
			resp = await process_image(url, "resize_max", ["-nogif", 4096, 0, "auto", "-bg", "-oz", "-f", "jpg"], timeout=60)
			im = await asubmit(Image.open, io.BytesIO(resp))
			pytesseract = await fut
			s = await asubmit(pytesseract.image_to_string, im, config="--psm 1", timeout=8)
		return s

	async def follow_to_image(self, url, follow=True):
		"Follows a message link, replacing emojis and user mentions with their icon URLs."
		temp = find_urls(url)
		if temp:
			return temp
		users = find_users(url)
		if follow:
			emojis = find_emojis_ex(url)
		else:
			emojis = find_emojis(url)
		out = deque()
		if users and follow:
			futs = [csubmit(self.fetch_user(verify_id(u))) for u in users]
			for fut in futs:
				with suppress(LookupError):
					res = await fut
					out.append(best_url(res))
		for s in emojis:
			if is_url(s):
				url = s
			else:
				url = await self.emoji_to_url(s)
			out.append(url.strip())
		if not out and follow:
			out = [url.rstrip() for url in find_urls(translate_emojis(replace_emojis(url)))]
		return out

	async def send_with_file(self, channel, msg=None, file=None, filename=None, embed=None, best=False, rename=True, reference=None, reacts="", tts=False):
		"Sends a message to a channel, then edits to add links to all attached files. Automatically transfers excessively large files to filehost."
		if not msg:
			msg = ""
		f = None
		fsize = 0
		size = CACHE_FILESIZE
		with suppress(AttributeError):
			if channel.guild.filesize_limit > 26214400:
				size = channel.guild.filesize_limit
		if getattr(channel, "simulated", None) or getattr(channel, "guild", None) and not channel.permissions_for(channel.guild.me).attach_files:
			size = -1
		data = file
		if file and not hasattr(file, "fp"):
			if isinstance(file, str):
				if not os.path.exists(file):
					raise FileNotFoundError(file)
				fsize = os.path.getsize(file)
				f = file
			else:
				data  = file
				file = CompatFile(data, filename)
				fsize = len(data)
		if fsize <= size:
			if not hasattr(file, "fp"):
				f2 = CompatFile(file, filename)
			else:
				f2 = file
			if not filename:
				filename = getattr(file, "filename", None) or getattr(file, "name", None) or file
			file = f2
			fp = file.fp
			fp.seek(0)
			data = fp.read()
			fsize = len(data)
			fp.seek(0)
			with suppress(AttributeError):
				fp.clear()
		# if reference and getattr(reference, "slash", None) or getattr(reference, "simulated", None):
		#     reference = None
		try:
			if fsize > size:
				if not f:
					f = filename if filename and not hasattr(file, "fp") else getattr(file, "_fp", None) or data
				if not isinstance(f, str):
					f = as_str(f)
				url = await self.data.exec.lproxy(file._fp if getattr(file, "_fp", None) else f, filename=filename)
				message = await send_with_reply(channel, reference, (msg + ("" if msg.endswith("```") else "\n") + url).strip(), embed=embed, tts=tts)
			else:
				message = await send_with_reply(channel, reference, msg, embed=embed, file=file, tts=tts)
				if filename is not None:
					if hasattr(filename, "filename"):
						filename = filename.filename
					# with suppress():
					# 	os.remove(filename)
		except:
			if filename is not None:
				if not isinstance(filename, str):
					filename = getattr(filename, "filename", None) or filename.name
				if not os.path.exists(filename):
					raise
				print(filename, os.path.getsize(filename))
			raise
		if not getattr(reference, "slash", None) and message.attachments:
			def temp_url(url):
				if is_discord_attachment(url):
					a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
					if a_id in self.data.attachments:
						u = self.preserve_attachment(a_id, fn=url)
						if filename and not isinstance(filename, str) or filename.endswith(".gif"):
							u += ".gif"
						return u
					if best:
						return self.preserve_into(channel.id, message.id, a_id, fn=url)
					return self.preserve_as_long(channel.id, message.id, a_id, fn=url)
				return url
			content = message.content + ("" if message.content.endswith("```") else "\n") + "\n".join("<" + temp_url(a.url) + ">" for a in message.attachments)
			message = await bot.edit_message(message, content=content.strip())
		if not message:
			print("No message detected.")
		elif reacts:
			for react in reacts:
				try:
					await message.add_reaction(react)
				except CE:
					await asyncio.sleep(1)
					with tracebacksuppressor:
						await message.add_reaction(react)
		return message

	async def tag_message(self, message):
		always_log = "logM" in self.data and message.guild is not None and message.guild.id in self.data.logM
		for attachment in message.attachments:
			if always_log or url2ext(attachment.url) in IMAGE_FORMS:
				csubmit(self.add_and_test(message, attachment))
		urls = [url for url in find_urls_ex(message.content) if message.content[message.content.index(url) - 1] != "<"]
		urls = await self.renew_attachments(urls)
		for url in urls:
			if is_discord_attachment(url) and not discord_expired(url) or self.is_webserver_url(url) and (always_log or url2ext(url) in IMAGE_FORMS):
				resp = await asubmit(reqs.next().get, url, headers=Request.header(), verify=False, stream=True, timeout=30, allow_redirects=True)
				url = resp.headers.get("Location") or resp.url
				if is_discord_attachment(url):
					uid = url.rsplit("/", 2)[-2]
				else:
					uid = url.split("?", 1)[0].rsplit("/", 1)[-1]
				attachment = cdict(id=uid, name=url2fn(url), url=url, size=int(resp.headers.get("Content-Length", 1)), read=lambda: self.get_request(url))
				resp.close()
				csubmit(self.add_and_test(message, attachment))

	def add_message(self, message, files=True, cache=True, force=False):
		"Inserts a message into the bot cache, discarding existing ones if full."
		if self.closed:
			return message
		try:
			m = self.cache.messages[message.id]
		except KeyError:
			m = None
		else:
			if force < 2 and isinstance(m, self.ExtendedMessage):
				with suppress(AttributeError):
					if not object.__getattribute__(m, "replaceable"):
						return m
			if getattr(m, "slash", False):
				message.slash = m.slash
		if cache and not m or force:
			created_at = message.created_at
			if created_at.tzinfo:
				created_at = created_at.replace(tzinfo=None)
			if not getattr(message, "simulated", None) and "channel_cache" in self.data:
				self.data.channel_cache.add(message.channel.id, message.id)
				if message.guild and hasattr(message.author, "guild"):
					guild = message.guild
					author = message.author
					try:
						if guild._members[author.id] != author:
							guild._members[author.id] = author
							if "guilds" in self.data:
								self.data.guilds.register(guild, force=False)
					except KeyError:
						pass
			if files and (not message.author.bot or message.webhook_id):
				if (utc_dt() - created_at).total_seconds() < 7200:
					csubmit(self.tag_message(message))
			self.cache.messages[message.id] = message
			if (utc_dt() - created_at).total_seconds() < 86400 * 14 and "message_cache" in self.data and not getattr(message, "simulated", None):
				self.data.message_cache.save_message(message)
		# if "attachments" in self.data:
		# 	for a in message.attachments:
		# 		self.data.attachments[a.id] = (message.channel.id, message.id)
		return message

	def remove_message(self, message):
		"Deletes a message from the bot cache."
		self.cache.messages.pop(message.id, None)
		if not message.author.bot:
			s = message_repr(message, username=True)
			ch = f"deleted/{message.channel.id}.txt"
			print(s, file=ch)

	async def add_attachment(self, attachment):
		if int(attachment.size) <= 64 * 1048576:
			return await self.get_attachment(attachment.url, size=attachment.size)

	async def add_and_test(self, message, attachment):
		n = 0
		fn = await self.add_attachment(attachment)
		if fn and "prot" in self.data:
			known = self.cache.attachments.get(attachment.id, None)
			if get_mime(fn).startswith("image/"):
				res, n = await self.data.prot.scan(message, fn, known=known)
			else:
				res = ""
			self.cache.attachments[attachment.id] = res

	async def get_attachment(self, url, data=False, size=0):
		if not data:
			filename = f"{CACHE_PATH}/attachments/{uhash(url.split('//', 1)[-1])}"
			if not os.path.exists(filename):
				await attachment_cache.download(url, filename=filename)
			return filename
		return await attachment_cache.download(url)

	async def get_request(self, url, limit=None, full=True, data=True, size=0, timeout=12):
		if (match := scraper_blacklist.search(url)):
			raise InterruptedError(match)
		if not full:
			if is_discord_url(url):
				try:
					return await asubmit(reqs.next().get, url, headers=Request.header(), stream=True, _timeout_=3, allow_redirects=True)
				except Exception:
					print_exc()
			if is_miza_attachment(url):
				ids = expand_attachment(url)
				url = await attachment_cache.obtain(*ids)
			return await Request.asession.get(url, headers=Request.header(), _timeout_=timeout, verify=False, allow_redirects=True)
		return await self.get_attachment(url, data=data)

	def get_colour(self, user) -> int:
		if user is None:
			return as_fut(16777214)
		if hasattr(user, "icon_url"):
			user = astype(user.icon_url, str)
		url = worst_url(user)
		return self.data.colours.get(url)

	async def get_proxy_url(self, user, force=False) -> str:
		if hasattr(user, "webhook"):
			url = user.webhook.avatar_url_as(format="webp", size=4096)
		else:
			with tracebacksuppressor:
				url = best_url(user)
		if not url:
			return self.discord_icon
		if "proxies" in self.data:
			with tracebacksuppressor:
				url = (await self.data.exec.uproxy(url, force=force, optimise=True)) or url
		return url

	async def as_embed(self, message, link=False, colour=False, refresh=True, proxy_images=True) -> discord.Embed:
		if refresh:
			message = await self.ensure_reactions(message)
		emb = discord.Embed(description="").set_author(**get_author(message.author))
		if colour:
			col = await self.get_colour(message.author)
			emb.colour = col
		content = message.content or message.system_content
		if proxy_images and content:
			urls = await self.follow_url(content)
			if urls:
				with tracebacksuppressor:
					url = urls[0]
					resp = await asubmit(reqs.next().head, url, headers=Request.header(), _timeout_=12, allow_redirects=True)
					headers = fcdict(resp.headers)
					if headers.get("Content-Type", "").split("/", 1)[0] == "image":
						if float(headers.get("Content-Length", inf)) < CACHE_FILESIZE:
							url = await self.data.exec.uproxy(url)
						else:
							url = await self.data.exec.aproxy(url)
						emb.url = url
						# url = allow_gif(url)
						emb.set_image(url=url)
						if url != content:
							emb.description = content
						if link:
							link = message_link(message)
							emb.description = lim_str(f"{emb.description}\n\n[View Message]({link})", 4096)
							emb.timestamp = message.edited_at or message.created_at
						return emb
		emb.description = content
		image = None
		thumbnail = None
		for a in message.attachments:
			url = a.url
			if is_image(url) is not None:
				if not image:
					if proxy_images:
						image = await self.data.exec.uproxy(url)
					else:
						image = url
				else:
					if proxy_images:
						thumbnail = await self.data.exec.uproxy(url)
					else:
						thumbnail = url
		for s in T(message).get("stickers", ()):
			if not image:
				image = s.url
			else:
				thumbnail = s.url
		for e in message.embeds:
			if not content:
				if e.description and e.description != EmptyEmbed:
					content = e.description
				for f in T(e).get("_fields", ()):
					if f:
						emb.add_field(name=f["name"], value=f["value"], inline=f.get("inline", True))
			if e.image:
				if not image:
					url = e.image.url
					if proxy_images:
						image = await self.data.exec.uproxy(url)
					else:
						image = url
				else:
					url = e.image.url
					if proxy_images:
						thumbnail = await self.data.exec.uproxy(url)
					else:
						thumbnail = url
				break
			if e.thumbnail:
				url = e.thumbnail.url
				if proxy_images:
					thumbnail = await self.data.exec.uproxy(url)
				else:
					thumbnail = url
		if image:
			emb.url = image
			emb.set_image(url=image)
		if thumbnail:
			emb.set_thumbnail(url=thumbnail)
		for e in message.embeds:
			if len(T(emb).get("_fields", ())) >= 25:
				break
			if not emb.description or emb.description == EmptyEmbed:
				title = e.title or ""
				if title:
					emb.title = title
				emb.url = e.url or ""
				description = e.description or e.url or ""
				if description:
					emb.description = description
			else:
				if e.title or e.description:
					emb.add_field(name=e.title or e.url or "\u200b", value=lim_str(e.description, 1024) or e.url or "\u200b", inline=False)
			for f in T(e).get("_fields", ()):
				if len(T(emb).get("_fields", ())) >= 25:
					break
				if f:
					emb.add_field(name=f["name"], value=f["value"], inline=f.get("inline", True))
			if len(emb) >= 6000:
				while len(emb) > 6000:
					emb.remove_field(-1)
				break
		urls = [e.url for e in message.embeds if e.url] + [best_url(a) for a in message.attachments]
		items = []
		for i in range((len(urls) + 9) // 10):
			temp2 = temp = urls[i * 10:i * 10 + 10]
			if proxy_images:
				try:
					temp2 = await self.data.exec.uproxy(*temp, collapse=False)
				except Exception:
					print_exc()
			def as_link(url1, url2):
				if image in (url1, url2) or thumbnail in (url1, url2):
					return
				if url1 == url2:
					return "(" + url2 + ")"
				return "[" + url2 + "]"
			items.extend(filter(bool, (as_link(u1, u2) for u1, u2 in zip(temp, temp2))))
		if items:
			if emb.description in items:
				emb.description = lim_str("\n".join(items), 4096)
			elif emb.description or items:
				emb.description = lim_str(emb.description + "\n" + "\n".join(items), 4096)
		if link:
			link = message_link(message)
			emb.description = lim_str(f"{emb.description}\n\n[View Message]({link})", 4096)
			emb.timestamp = message.edited_at or message.created_at
		return emb

	async def coloured_embed(self, url):
		colour = await self.get_colour(url)
		return discord.Embed(colour=colour).set_image(url=url)

	async def random_embed(self, url):
		colour = rand_colour()
		return discord.Embed(colour=colour).set_image(url=url)

	def limit_cache(self, cache=None, limit=None):
		"Limits a cache to a certain amount, discarding oldest entries first."
		if limit is None:
			limit = self.cache_size
		if cache is not None:
			caches = (self.cache[cache],)
		else:
			caches = self.cache.values()
		for c in caches:
			if len(c) >= limit * 2:
				items = tuple(c.items())
				c.clear()
				c.update(dict(items[-limit:]))

	def cache_reduce(self):
		self.limit_cache("messages")
		self.limit_cache("attachments", 256)
		# self.limit_cache("guilds")
		self.limit_cache("channels")
		self.limit_cache("users")
		self.limit_cache("members")
		self.limit_cache("roles")
		self.limit_cache("emojis")
		self.limit_cache("deleted", limit=16384)
		self.limit_cache("banned", 4096)

	def update_cache_feed(self):
		"Updates bot cache from the discord.py client cache, using automatic feeding to mitigate the need for slow dict.update() operations."
		self.cache.guilds._feed = (self._guilds, getattr(self, "sub_guilds", {}))
		self.cache.emojis._feed = (self._emojis,)
		self.cache.users._feed = (self._users,)
		g = self._guilds.values()
		self.cache.members._feed = lambda: (guild._members for guild in g)
		self.cache.channels._feed = lambda: chain(
			(guild._channels for guild in g),
			(guild._threads for guild in g),
			(self._private_channels,),
			(getattr(self, "sub_channels", {}),),
		)
		self.cache.roles._feed = lambda: (guild._roles for guild in g)

	def update_usernames(self):
		if self.users_updated:
			self.usernames = {str(user): user for user in self.cache.users.values()}
			self.users_updated = False
		if self.guilds_updated:
			has_guilds = getattr(self, "_guilds", None) or self.cache.guilds
			nf = [k for k in self.data.guilds if k not in has_guilds]
			for k in nf:
				self.cache.guilds.pop(k, None)
				self.data.guilds.pop(k, None)
			self.guilds_updated = False

	sub_channels = {}
	def update_subs(self):
		self.sub_guilds = dict(self._guilds) or self.sub_guilds
		sc = self.sub_channels
		self.sub_channels = dict(chain.from_iterable(guild._channels.items() for guild in self.sub_guilds.values())) or sc
		self.sub_channels.update(sc)
		if not hasattr(self, "guilds_ready") or not self.guilds_ready.done():
			return
		for guild in self.guilds:
			if len(guild._members) != guild.member_count:
				print("Incorrect member count:", guild, len(guild._members), guild.member_count)
				csubmit(self.load_guild(guild))

	def get_prefix(self, guild):
		"Gets the target bot prefix for the target guild, return the default one if none exists."
		try:
			g_id = guild.id
		except AttributeError:
			try:
				g_id = int(guild)
			except TypeError:
				g_id = 0
		with suppress(KeyError):
			return self.data.prefixes[g_id]
		return self.prefix

	def get_perms(self, user, guild=None):
		"Gets effective permission level for the target user in a certain guild, taking into account roles."
		try:
			u_id = user.id
		except AttributeError:
			u_id = int(user)
		if self.is_owner(u_id):
			return nan
		if self.is_blacklisted(u_id):
			return -inf
		if u_id == self.id:
			return inf
		if guild is None or hasattr(guild, "ghost"):
			return inf
		if hasattr(guild, "guild"):
			guild = guild.guild
		if u_id == guild.owner_id:
			return inf
		with suppress(KeyError):
			perm = self.data.perms[guild.id][u_id]
			if isnan(perm):
				return -inf
			return perm
		m = guild.get_member(u_id)
		if m is None:
			r = guild.get_role(u_id)
			if r is None:
				with suppress(KeyError):
					return self.data.perms[guild.id][guild.id]
				return -inf
			return self.get_role_perms(r, guild)
		if m.bot:
			try:
				return self.data.perms[guild.id][guild.id]
			except KeyError:
				return 0
		try:
			p = m.guild_permissions
		except AttributeError:
			return -inf
		if p.administrator:
			return inf
		perm = -inf
		for role in m.roles:
			rp = self.get_role_perms(role, guild)
			if rp > perm:
				perm = rp
		if isnan(perm):
			perm = -inf
		return perm

	def get_role_perms(self, role, guild):
		"Gets effective permission level for the target role in a certain guild, taking into account permission values."
		if role.permissions.administrator:
			return inf
		with suppress(KeyError):
			perm = self.data.perms[guild.id][role.id]
			if isnan(perm):
				return -inf
			return perm
		if guild.id == role.id:
			return 0
		p = role.permissions
		if all((p.ban_members, p.manage_channels, p.manage_guild, p.manage_roles, p.manage_messages)):
			return 4
		elif any((p.ban_members, p.manage_channels, p.manage_guild)):
			return 3
		elif any((p.kick_members, p.manage_messages, p.manage_nicknames, p.manage_roles, p.manage_webhooks, p.manage_emojis, p.value & 0x200000000, p.value & 0x400000000)):
			return 2
		elif any((p.view_audit_log, p.priority_speaker, p.mention_everyone, p.move_members, p.mute_members, p.deafen_members, p.value & 0x80000)):
			return 1
		return -1

	def set_perms(self, user, guild, value):
		"Sets the permission value for a snowflake in a guild to a value."
		perms = self.data.perms
		try:
			u_id = user.id
		except AttributeError:
			u_id = user
		g_perm = perms.setdefault(guild.id, {})
		g_perm[u_id] = round_min(value)

	def remove_perms(self, user, guild):
		"Removes the permission value for a snowflake in a guild."
		perms = self.data.perms
		try:
			u_id = user.id
		except AttributeError:
			u_id = user
		g_perm = perms.get(guild.id, {})
		g_perm.pop(u_id, None)
		if not g_perm:
			self.data.perms.pop(guild.id, None)

	def get_enabled(self, channel):
		"Retrieves the list of enabled command categories for a channel."
		guild = getattr(channel, "guild", None)
		if not guild and hasattr(channel, "member_count"):
			guild = channel
		if guild:
			try:
				enabled = self.data.enabled[channel.id]
			except KeyError:
				try:
					enabled = self.data.enabled[guild.id]
				except KeyError:
					name = getattr(channel, "name", "")
					is_bot = full_prune(name) in ("bot", "bots", "bot-spam", "bot-commands", full_prune(self.name))
					enabled = (visible_commands if self.is_nsfw(channel) else default_commands) if len(guild._members) <= 100 or is_bot else basic_commands
		else:
			enabled = self.categories.keys()
		return enabled

	def status_changed(self, before, after):
		"Checks whether a member's status was changed."
		if before.activity != after.activity:
			return True
		for attr in ("status", "desktop_status", "web_status", "mobile_status"):
			b, a = getattr(before, attr), getattr(after, attr)
			if b == a:
				return False
		return True

	def status_updated(self, before, after):
		"Checks whether a member's status was updated by themselves."
		if before.activity != after.activity:
			return True
		for attr in ("status", "desktop_status", "web_status", "mobile_status"):
			b, a = getattr(before, attr), getattr(after, attr)
			if b == discord.Status.online and a == discord.Status.idle:
				if utc() - self.data.users.get(after.id, {}).get("last_seen", 0) < 900:
					return False
			elif a == discord.Status.offline:
				return False
			elif b == a:
				return False
		return True

	def is_deleted(self, message):
		"Checks if a message has been flagged as deleted by the deleted cache. 1 = regular delete, 2 = bot delete, 3 = silent delete"
		try:
			m_id = int(message.id)
		except AttributeError:
			m_id = int(message)
		if "deleted" not in self.data:
			return False
		try:
			return self.data.deleted.cache[m_id]
		except KeyError:
			return False

	async def verify_integrity(self, message):
		v = self.is_deleted(message)
		if v:
			return v != 1
		if hasattr(message, "simulated") or hasattr(message, "slash"):
			curr_message = message
		else:
			try:
				curr_message = await self.fetch_message(message.id, message.channel)
			except Exception:
				print_exc()
				return False
		if getattr(message, "deleted", None) or getattr(curr_message, "deleted", None):
			return False
		return True

	async def require_integrity(self, message):
		if not await self.verify_integrity(message):
			print(message, message.id, message.content)
			raise CommandCancelledError("Reference message was deleted.")
		return message

	def log_delete(self, message, value=1):
		"Logs if a message has been deleted."
		if not message:
			return
		try:
			m_id = int(message.id)
		except AttributeError:
			m_id = int(message)
		self.data.deleted.cache[m_id] = value

	async def silent_delete(self, message, keep_log=False, exc=False, delay=None):
		"Silently deletes a message, bypassing logs whilst enabling connected commands to continue executing."
		if not message:
			return
		if delay:
			await asyncio.sleep(float(delay))
		v = 2 if keep_log else 3
		if isinstance(message, list_like) and len(message) > 1:
			channel = None
			messages = message
			for m in messages:
				if channel is None:
					channel = m.channel
				elif channel.id != m.channel.id:
					futs = [self.silent_delete(m, keep_log=keep_log, exc=exc) for m in messages]
					return await gather(*futs)
			for m in messages:
				self.log_delete(m, v)
			return await channel.delete_messages(messages)
		elif isinstance(message, list_like):
			message = message[0]
		if isinstance(message, int):
			message = await self.fetch_message(message)
		try:
			self.log_delete(message, v)
			await discord.Message.delete(message)
		except Exception:
			self.data.deleted.cache.pop(message.id, None)
			if exc:
				raise

	async def verified_ban(self, user, guild, reason=None):
		self.cache.banned[(guild.id, user.id)] = utc()
		try:
			await guild.ban(user, delete_message_days=0, reason=reason)
		except:
			self.cache.banned.pop((guild.id, user.id), None)
			raise
		self.cache.banned[(guild.id, user.id)] = utc()

	def recently_banned(self, user, guild, duration=20):
		return utc() - self.cache.banned.get((verify_id(guild), verify_id(user)), 0) < duration

	def is_mentioned(self, message, user, guild=None):
		if not message.content and not (message.attachments or message.embeds):
			return
		if message.content:
			c = no_md(message.content)
			if c and not c.startswith(self.get_prefix(guild)):
				if c[0] in COMM or c[:2] in ("//", "/*"):
					return False
		u_id = verify_id(user)
		if u_id in (member.id for member in message.mentions):
			return True
		if guild is None and getattr(message.channel, "recipient", None) and self.ready:
			return True
		if "exec" in self.data and self.data.exec.get(message.channel.id, 0) & 64:
			return True
		if not guild:
			return False
		member = guild.get_member(u_id)
		if member is None:
			return False
		if message.content.count("`") > 1:
			return False
		for role in member.roles:
			if not role.mentionable:
				if role.mention in message.content:
					return True
		return False

	# Checks if a user is an owner of the bot.
	is_owner = lambda self, user: verify_id(user) in self.owners

	def is_trusted(self, guild):
		"Checks if a guild is trusted by the bot."
		try:
			trusted = self.data.trusted
		except (AttributeError, KeyError):
			return 0
		if not guild:
			return 0
		i = verify_id(guild)
		if i not in trusted:
			return 0
		if not isinstance(trusted[i], set):
			try:
				trusted[i] = set(trusted[i])
			except TypeError:
				trusted[i] = set()
		for u in tuple(trusted[i]):
			if u is None:
				continue
			if u in self.data.premiums and self.data.premiums[u]["lv"] >= 2:
				pass
			else:
				print(i, "trusted lost from", u)
				trusted[i].remove(u)
		trusted[i].add(None)
		return min(2, len(trusted[i]))

	def premium_level(self, user, absolute=False):
		"Retrieves a user's premium subscription level."
		if not user:
			return 0
		if self.is_owner(user):
			return inf
		try:
			premiums = self.data.premiums
		except (AttributeError, KeyError):
			return 0
		uid = verify_id(user)
		lv = 0
		if self.premium_server in self.cache.guilds:
			u = self.cache.guilds[self.premium_server].get_member(uid)
			if u:
				for role in u.roles:
					if role.id in self.premium_roles:
						lv = max(lv, self.premium_roles[role.id])
		elif not self.ready:
			return 0
		else:
			return 3
		if not absolute:
			if not self.data.users:
				return lv
			data = self.data.users.get(uid)
			if data and data.get("payg"):
				return 3
			if data and data.get("credit"):
				return 2
			premiums.subscribe(user, lv)
		return lv

	def premium_multiplier(self, pl):
		if not isfinite(pl):
			return pl
		if pl < 0:
			return 0
		if pl < 1:
			return 1
		if pl < 2:
			return 1.5
		if pl < 3:
			return 2
		return 3

	def premium_limit(self, pl2):
		if not isfinite(pl2):
			return pl2
		if pl2 < 0:
			return 0
		if pl2 < 1:
			return 25
		if pl2 < 2:
			return 50
		if pl2 < 3:
			return 200
		if pl2 < 4:
			return 300
		if pl2 < 5:
			return 625
		if pl2 < 6:
			return 1250
		if pl2 < 7:
			return 2000
		return 3000

	class PremiumContext(contextlib.AbstractContextManager, contextlib.ContextDecorator, collections.abc.Callable):

		quota_expiry = 84600

		def __init__(self, user, target=None, value=0, cost=0):
			self.user = user
			self.target = target or user
			self.value = value
			self.cost = cost
			self.embed = None
			self.description = None
			self.costtup = []

		@property
		def value_approx(self):
			data = bot.data.users.setdefault(self.user.id, {})
			freebies = T(data).coercedefault("freebies", list, [])
			freelim = bot.premium_limit(self.value)
			rem = max(0, freelim - len(freebies))
			if rem < 25:
				return 1
			return self.value

		def require(self, value=0, cost=None):
			data = bot.data.users.setdefault(self.user.id, {})
			if data.get("payg") or data.get("credit"):
				return self
			if self.value < value:
				if value > 2:
					raise PermissionError(f"Premium level {value // 2} or higher required; please see {bot.kofi_url} for more info!")
			ts = utc()
			freebies = T(data).coercedefault("freebies", list, [])
			while freebies and ts - freebies[0] >= self.quota_expiry:
				freebies.pop(0)
			if value <= 0:
				return self
			freelim = bot.premium_limit(self.value)
			rem = max(0, freelim - len(freebies))
			if rem <= 0:
				s = " (next refresh " + time_repr(self.quota_expiry + freebies[0]) + ")" if freebies else ""
				raise PermissionError(f"Apologies, you have exceeded your quota of {freelim} for today{s}. Please see /premium or {bot.kofi_url} for more info!")
			self.cost = cost or self.cost
			return self

		def __enter__(self):
			pass

		def __exit__(self, exc_type=None, exc_value=None, exc_tb=None):
			target = self.target
			try:
				for tup in self.costtup:
					s = str(utc_dt().date())
					costs = bot.data.costs.setdefault(s, {})
					t = (tup[1], tup[2])
					try:
						costs[t] = str(mpf(costs[t]) + mpf(tup[-1]))
					except KeyError:
						costs[t] = tup[-1]
					self.add(mpf(tup[-1]) * 1000)
				if exc_type and exc_value:
					return
				data = bot.data.users.setdefault(target.id, {})
				cost = round_random(self.cost)
				dcost = self.cost / 1000
				print("QCost:", target, self.cost, cost)
				rem = inf
				if data.get("payg"):
					pass
				elif data.get("credit"):
					c = mpf(data["credit"]) - dcost
					if c <= 0:
						data.pop("credit", None)
					else:
						data["credit"] = str(c)
				else:
					ts = utc()
					freebies =  T(data).coercedefault("freebies", list, [])
					while freebies and ts - freebies[0] >= self.quota_expiry:
						freebies.pop(0)
					if cost:
						freebies.extend([ts] * cost)
					freelim = bot.premium_limit(self.value)
					rem = max(0, freelim - len(freebies))
					print("Remaining:", target, f"{rem}/{freelim}")
				if data.get("payg") and data.get("usages"):
					tcost = sum(mpf(t[-1]) for t in data["usages"])
				if data.get("logging") == "none":
					return
				elif data.get("logging", "auto") == "auto":
					if data.get("payg") and data.get("usages"):
						if floor(tcost - dcost) < floor(tcost):
							pass
						else:
							return
					elif data.get("credit"):
						c = mpf(data["credit"])
						if floor(c) < floor(c + dcost):
							pass
						else:
							return
					elif not rem or isfinite(rem) and isfinite(cost) and int(math.log(rem, 3)) < int(math.log(rem + cost, 3)):
						pass
					else:
						return
				or_adjust = " or adjust logging" if data.get("logging") == "auto" else ""
				if data.get("payg") and data.get("usages"):
					desc = f"Command incurred cost of `${dcost}` (`${tcost}` total pending). See /premium to check usage stats{or_adjust}!"
				elif not cost:
					return
				else:
					if data.get("credit"):
						c = data["credit"]
						q = round(mpf(c) * 1000)
						s = f"Command incurred cost of `{cost}` (`${dcost}`); `{q}` premium credits remaining."
					else:
						s = " (next refresh " + time_repr(self.quota_expiry + freebies[0]) + ")" if freebies else ""
						s = f"Command incurred cost of `{cost}`; `{rem}/{freelim}`{' free' if self.value <= 1 else ''} quota remaining today{s}."
					if self.value >= 2 or data.get("credit"):
						desc = f"{s}\nSee /premium to check usage stats{or_adjust}."
					else:
						desc = f"{s}\nIf you're able to contribute towards [funding](<{bot.kofi_url}>) my hosting costs it would mean the world to us, and ensure that I can continue providing up-to-date tools and entertainment.\nEvery little bit helps due to the size of my audience!\nSee /premium to check usage stats{or_adjust}."
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = desc
				self.description = desc
				self.embed = emb
			finally:
				self.cost = 0
				self.costtup.clear()

		def add(self, cost=0):
			self.cost += cost

		def apply(self, cost=None):
			self.cost = cost or self.cost
			self.__exit__()
			return self.description

		def append(self, tup):
			data = bot.data.users.setdefault(self.target.id, {})
			if data.get("payg"):
				data.setdefault("usages", []).append(tup)
			self.costtup.append(tup)
			return tup

	def premium_context(self, user, guild=None):
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		return self.PremiumContext(user, value=premium)

	def is_nsfw(self, channel):
		if is_nsfw(channel):
			return True
		if "nsfw" in self.data:
			if getattr(channel, "recipient", None):
				return self.data.nsfw.get(channel.recipient.id, False)
			return self.data.nsfw.get(channel.id, False)

	async def donate(self, name, uid, amount, msg):
		channel = self.get_channel(320915703102177293)
		if not channel:
			return
		if msg:
			emb = discord.Embed(colour=rand_colour())
			emb.set_author(**get_author(self.user))
			emb.description = msg
		else:
			emb = None
		if not uid:
			await channel.send(f"Failed to locate donation of ${amount} from user {name}!", embed=emb)
			return
		await asubmit(self.update_usernames)
		try:
			user = await self.fetch_user(uid)
		except Exception:
			print_exc()
			await channel.send(f"Failed to locate donation of ${amount} from user {name}/{uid}!", embed=emb)
			return
		dias = round_min(amount * 300)
		self.data.users.add_diamonds(user, dias, multiplier=False)
		csubmit(channel.send(f"Thank you {user_mention(user.id)} for donating ${amount}! Your account has been credited 💎 {dias}!", embed=emb))
		await user.send(f"Thank you for donating ${amount}! Your account has been credited 💎 {dias}!")
		return True

	def is_blacklisted(self, user):
		"Checks if a user is blacklisted from the bot."
		u_id = verify_id(user)
		if self.is_owner(u_id) or u_id == self.id:
			return False
		with suppress(KeyError):
			return (self.data.blacklist.get(u_id) or 0) > 1
		return True

	dangerous_command = bold(ansi_md(colourise(uni_str('[WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. REPEAT COMMAND WITH "?f" FLAG TO CONFIRM.]'), fg="red")))

	mmap = {
		"“": '"',
		"”": '"',
		"„": '"',
		"‘": "'",
		"’": "'",
		"‚": "'",
		"〝": '"',
		"〞": '"',
		"⸌": "'",
		"⸍": "'",
		"⸢": "'",
		"⸣": "'",
		"⸤": "'",
		"⸥": "'",
		"⸨": "((",
		"⸩": "))",
		"⟦": "[",
		"⟧": "]",
		"〚": "[",
		"〛": "]",
		"「": "[",
		"」": "]",
		"『": "[",
		"』": "]",
		"【": "[",
		"】": "]",
		"〖": "[",
		"〗": "]",
		"（": "(",
		"）": ")",
		"［": "[",
		"］": "]",
		"｛": "{",
		"｝": "}",
		"⌈": "[",
		"⌉": "]",
		"⌊": "[",
		"⌋": "]",
		"⦋": "[",
		"⦌": "]",
		"⦍": "[",
		"⦐": "]",
		"⦏": "[",
		"⦎": "]",
		"⁅": "[",
		"⁆": "]",
		"〔": "[",
		"〕": "]",
		"«": "<<",
		"»": ">>",
		"❮": "<",
		"❯": ">",
		"❰": "<",
		"❱": ">",
		"❬": "<",
		"❭": ">",
		"＜": "<",
		"＞": ">",
		"⟨": "<",
		"⟩": ">",
	}
	mtrans = "".maketrans(mmap)

	cmap = {
		"<": "alist((",
		">": "))",
	}
	ctrans = "".maketrans(cmap)

	op = {
		"=": None,
		":=": None,
		"+=": "__add__",
		"-=": "__sub__",
		"*=": "__mul__",
		"/=": "__truediv__",
		"//=": "__floordiv__",
		"**=": "__pow__",
		"^=": "__pow__",
		"%=": "__mod__",
	}
	consts = {
		"k": 1 << 10,
		"M": 1 << 20,
		"G": 1 << 30,
		"T": 1 << 40,
		"P": 1 << 50,
		"E": 1 << 60,
		"Z": 1 << 70,
		"Y": 1 << 80,
	}

	async def eval_math(self, expr, default=0, op=True):
		"Evaluates a math formula to a float value, using a math process from the subprocess pool when necessary."
		expr = as_str(expr)
		if op:
			# Allow mathematical operations on a default value
			_op = None
			for op, at in self.op.items():
				if expr.startswith(op):
					expr = expr[len(op):].strip()
					_op = at
			num = await self.eval_math(expr, op=False)
			if _op is not None:
				num = getattr(mpf(default), _op)(num)
			return num
		f = expr.strip()
		if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\/\d*\.?\d+)?", f):
			return round_min(f)
		try:
			if not f:
				return 0
			else:
				s = f.casefold()
				if len(s) < 5 and s in ("t", "true", "y", "yes", "on"):
					return True
				elif len(s) < 6 and s in ("f", "false", "n", "no", "off"):
					return False
				else:
					try:
						return round_min(f)
					except Exception:
						try:
							return orjson.loads(f)
						except orjson.JSONDecodeError:
							return eval_json(f)
		except (ValueError, TypeError, SyntaxError):
			r = await self.solve_math(f, 128, 0, variables=self.consts)
		x = r[0]
		if x and isinstance(x, str):
			if x[0] != "[" and "[" in x and "]" in x:
				try:
					x = float(sympy.sympify(x))
				except Exception:
					print_exc()
				else:
					return x
			if x.isnumeric():
				return int(x)
		if isinstance(x, (int, float, np.number)):
			return round_min(x)
		if x in (None, "None"):
			return
		try:
			if "/" in x:
				raise ValueError
			x = round_min(x)
		except ValueError:
			try:
				x = round_min(eval_json(x))
			except Exception:
				x = None
		if x is None:
			raise ValueError(f'Could not evaluate expression "{expr}" as number.')
		if not isinstance(x, int) and len(str(x)) <= 16:
			return round_min(float(x))
		return x

	async def solve_math(self, f, prec=128, r=False, timeout=16, variables=None, nlp=False):
		"Evaluates a math formula to a list of answers, using a math process from the subprocess pool when necessary."
		res = await process_math(f.strip(), int(prec), int(r), timeout=timeout, variables=variables)
		if nlp:
			return f"{f} = {res[0]}"
		return res

	def update_ip(self, ip):
		"Updates the bot's stored external IP address."
		if regexp("^([0-9]{1,3}\\.){3}[0-9]{1,3}$").search(ip):
			self.ip = ip
			# new_ip = f"https://{self.ip}:{PORT}"
			# if self.raw_webserver != self.webserver and self.raw_webserver != new_ip:
			#     csubmit(self.create_main_website())
			# self.raw_webserver = new_ip

	def is_webserver_url(self, url):
		if url.startswith(self.webserver) or url.startswith(self.raw_webserver) or url.startswith("https://" + self.raw_webserver.split("//", 1)[-1]):
			return (url,)
		return regexp("^https?:\\/\\/(?:[A-Za-z]+\\.)?mizabot\\.xyz").findall(url)

	ip_sem = Semaphore(1, 1, rate_limit=60)
	async def get_ip(self):
		"Gets the external IP address from api.ipify.org"
		if not self.ip_sem.busy:
			async with self.ip_sem:
				self.ip = await Request("https://api.ipify.org", bypass=False, decode=True, timeout=3, aio=True)
		return self.ip

	total_hosted = 0
	async def get_hosted(self):
		size = 0
		try:
			filehost = os.listdir("saves/filehost")
		except FileNotFoundError:
			return size
		for fn in filehost:
			with tracebacksuppressor(ValueError):
				if "$" in fn and fn.split("$", 1)[0].endswith("~.forward"):
					size += int(fn.split("$", 2)[1])
				else:
					p = "saves/filehost/" + fn
					size += os.path.getsize(p)
		self.total_hosted = size
		return size

	caps = set()
	capfrom = AutoCache(stale=0, timeout=60)
	last_pings = {}
	compute_queue = {}
	compute_wait = {}
	def distribute(self, caps, stat=None, resp=None, ip="127.0.0.1"):
		self.last_pings[ip] = utc()
		if stat:
			for k, v in stat.items():
				self.status_data.system[k].update(v)
		if resp:
			for k, v in resp.items():
				k = int(k)
				# print("END TASK:", k, bot.compute_wait, lim_str(str(v), 64), frand())
				if k not in self.compute_wait:
					if not isinstance(v, BaseException):
						print("MISSING:", k, lim_str(str(v), 256))
					continue
				task = self.compute_wait.pop(k)
				if isinstance(v, BaseException):
					print(repr(v), ip, k)
					# v2 = v.__class__(*v.args, ip, k)
					task.set_exception(v)
				else:
					task.set_result(v)
				# print("TASK:", k, task, v)
		tasks = []
		for cap in caps:
			self.capfrom[(cap, ip)] = None
			misc = self.compute_queue.get(cap)
			if not misc:
				continue
			task = misc.pop()
			if not task:
				continue
			tasks.append(task)
		prompts = []
		for task in tasks:
			i = ts_us() + id(task)
			while i in self.compute_wait:
				i += 1
			task.ts = i
			self.compute_wait[i] = task
			prompt = [i, task.cap, task.command, task.timeout]
			prompts.append(prompt)
		self.caps = set(c for c, i in self.capfrom)
		return prompts

	def worker_count(self, cap="image"):
		return sum(c == cap for c, i in self.capfrom)

	async def lambdassert(self, cap="image", timeout=2):
		if cap not in self.caps:
			raise NotImplementedError(f"Capability {cap} required.")
		await process_image(lambdassert, "$", (), cap=cap, timeout=timeout)
		return sum(c == cap for c, i in self.capfrom)

	_cpuinfo = None
	api_latency = inf
	lll = inf
	llc = 0
	async def get_system_stats(self):
		fut = asubmit(get_current_stats, self.up_bps, self.down_bps, priority=2)
		t = utc()
		latency = self.latency
		if latency != self.lll and isfinite(latency):
			self.api_latency = self.lll = latency
			self.llc = t
		elif t - self.llc < 5:
			pass
		else:
			try:
				await Request.sessions.next().head(f"https://discord.com/api/{api}/users/@me", timeout=5)
				if self.api_latency >= 300:
					print(f"API latency {self.api_latency} exceeded 300s, restarting...")
					return await self.commands.shutdown[0].confirm_shutdown()
				self.api_latency = self.api_latency * 2 / 3 + (utc() - t) / 3
			except Exception as ex:
				print(repr(ex))
				self.api_exc = ex
				self.api_latency += 5
			else:
				self.llc = utc()
		try:
			audio_players, playing_players = await asyncio.wait_for(self.audio.asubmit("len(AP.players),sum(bool(p.vc) and bool(p.queue) and p.is_playing() for p in AP.players.values())"), timeout=2)
		except (AssertionError, AttributeError, asyncio.TimeoutError):
			audio_players = playing_players = playing_audio_players = "N/A"
		files = os.listdir("misc")
		for f in files:
			path = "misc/" + f
			if is_code(path):
				self.size2[f] = line_count(path)
		size = (
			np.sum(list(self.size.values()), dtype=np.uint32, axis=0)
			+ np.sum(list(self.size2.values()), dtype=np.uint32, axis=0)
		)
		with tracebacksuppressor:
			system = await fut
			for k, v in system.items():
				self.status_data.system[k].update(v)
		for k, v in self.status_data.system.items():
			for i, e in tuple(v.items()):
				if t - e.get("time", 0) > 30:
					v.pop(i)
		self.status_data.update({
			"discord": {
				"Shard count": self.shard_count + bool(self.audio),
				"Server count": len(self._guilds),
				"User count": len(self.cache.users),
				"Channel count": len(self.cache.channels),
				"Role count": len(self.cache.roles),
				"Emoji count": len(self.cache.emojis),
				"Cached messages": len(self.cache.messages),
				"API latency": self.api_latency,
				**({"Website URL": self.webserver} if self.webserver else {}),
			},
			"misc": {
				"Active commands": self.command_semaphore.active,
				"Voice (connected, playing)": f"{audio_players}|{playing_players}",
				"Total data transmitted": self.total_bytes,
				"Hosted storage": self.total_hosted,
				"System time": datetime.datetime.now(),
				"Uptime (past week)": self.uptime,
				"Uptime (current)": time_disp(utc() - self.start_time),
				"Command count": len(set(itertools.chain.from_iterable(self.commands.values()))),
				"Code size": [x.item() for x in size],
			},
		})
		return self.status_data

	status_sem = Semaphore(1, inf, rate_limit=1, sync=True)
	status_data = cdict(
		system=cdict(
			cpu={},
			gpu={},
			memory={},
			disk={},
			network={},
			power={},
			temperature={},
		),
		discord=cdict(),
		misc=cdict(),
	)
	async def status(self, interval=None, simplified=False):
		if not self.status_sem.busy:
			async with self.status_sem:
				self.status_fut = csubmit(self.get_system_stats())
				self.status_data = await self.status_fut
		if interval:
			ninter = self.ninter
			it = int(utc() // ninter) * ninter
			out = []
			for i in range(ninter, interval + ninter, ninter):
				out.append(self.uptime_db.get(i - interval + it, {}))
			return out
		status = self.status_data
		if simplified:
			return self.simplify_status(status)
		return status

	def simplify_status(self, status):
		system = status.system
		cpu_usage = sum(e["usage"] * e["count"] for e in system.cpu.values()) / sum(e["max"] * e["count"] for e in system.cpu.values()) if system.cpu else 0
		gpu_usage = sum(e["usage"] * e["count"] for e in system.gpu.values()) / sum(e["max"] * e["count"] for e in system.gpu.values()) if system.gpu else 0
		cpu_cores = sum(e["count"] for e in system.cpu.values()) if system.cpu else 0
		gpu_cores = sum(e["count"] for e in system.gpu.values()) if system.gpu else 0
		memory_usage = sum(e["usage"] * e["count"] for e in system.memory.values()) if system.memory else 0
		memory_max = sum(e["max"] * e["count"] for e in system.memory.values()) if system.memory else 0
		disk_usage = sum(e["usage"] * e["count"] for e in system.disk.values()) if system.disk else 0
		disk_max = sum(e["max"] * e["count"] for e in system.disk.values()) if system.disk else 0
		network_usage = sum(e["usage"] * e["count"] for e in system.network.values()) if system.network else 0
		power_usage = sum(e["usage"] * e["count"] for e in system.power.values()) if system.power else 0
		power_max = sum(e["max"] * e["count"] for e in system.power.values()) if system.power else 0
		temp_usage = max(e["usage"] * e["count"] for e in system.temperature.values()) if system.temperature else 0
		discord_stats = dict(status.discord)
		discord_stats["API latency"] = sec2time(discord_stats["API latency"])
		misc_stats = dict(status.misc)
		misc_stats["Total data transmitted"] = byte_scale(misc_stats["Total data transmitted"]) + "B"
		misc_stats["Hosted storage"] = byte_scale(misc_stats["Hosted storage"]) + "B"
		misc_stats["Uptime (past week)"] = f'{round(misc_stats["Uptime (past week)"] * 100, 3)}%'
		return {
			"System info": {
				"CPU usage": f"{round(cpu_usage * 100, 3)}% *{cpu_cores}",
				"GPU usage": f"{round(gpu_usage * 100, 3)}% *{gpu_cores}",
				"Memory usage": byte_scale(memory_usage) + "B/" + byte_scale(memory_max) + "B",
				"Disk usage": byte_scale(disk_usage) + "B/" + byte_scale(disk_max) + "B",
				"Network usage": byte_scale(network_usage) + "bps",
				"Power usage": f"{round(power_usage, 3)} W/{round(power_max, 3)} W",
				"Internal Temperature": f"{round(temp_usage, 3)} °C",
			},
			"Discord info": discord_stats,
			"Misc info": misc_stats,
		}

	@tracebacksuppressor
	def get_module(self, module):
		"Loads a module containing commands and databases by name."
		f = module
		if "." in f:
			f = f[:f.rindex(".")]
		path, module = module, f
		new = False
		reloaded = False
		if module in self._globals:
			reloaded = True
			print(f"Reloading module {module}...")
			if module in self.categories:
				self.unload(module)
			# mod = importlib.reload(self._globals[module])
		else:
			print(f"Loading module {module}...")
			new = True
			# mod = __import__(module)
		if not new:
			mod = self._globals.pop(module, {})
		# else:
		#     mod = self._globals
		else:
			mod = cdict(common.__dict__)
		assert isinstance(mod, dict)
		fn = f"commands/{module}.py"
		with open(fn, "rb") as f:
			b = f.read()
		code = compile(b, fn, "exec")
		exec(code, mod)
		print(f"Evaluated module {module}...")
		self._globals[module] = mod
		commands = deque()
		dataitems = deque()
		for cls in all_subclasses(Importable):
			if mod.get(cls.__name__) is cls and cls not in (Importable, Command, Database):
				obj = cls(self, module)
				if issubclass(cls, Command):
					commands.append(obj)
				elif issubclass(cls, Database):
					dataitems.append(obj)
		commands = alist(commands)
		dataitems = alist(dataitems)
		for u in dataitems:
			for c in commands:
				c.data[u.name] = u
		self.categories[module] = commands
		self.dbitems[module] = dataitems
		self.size[module] = line_count("commands/" + path)
		if commands:
			print(f"{module}: Successfully loaded {len(commands)} command{'s' if len(commands) != 1 else ''}.")
		if dataitems:
			print(f"{module}: Successfully loaded {len(dataitems)} database{'s' if len(dataitems) != 1 else ''}.")
		if reloaded:
			while not self.ready:
				time.sleep(0.5)
			print(f"Resending _ready_ event to module {module}...")
			futs = []
			for db in dataitems:
				for f in dir(db):
					if f.startswith("_") and f[-1] == "_" and f[1] != "_":
						func = getattr(db, f, None)
						if callable(func):
							self.events.append(f, func)
				for e in ("_bot_ready_", "_ready_"):
					func = getattr(db, e, None)
					if callable(func):
						fut = asubmit(func, bot=self)
						futs.append(fut)
			await_fut(gather(*futs))
		print(f"Successfully loaded module {module}.")
		return True

	@tracebacksuppressor
	def unload(self, mod=None):
		if mod is None:
			mods = deque(self.categories)
		else:
			mod = mod.casefold()
			if mod not in self.categories:
				raise KeyError
			mods = [mod]
		for mod in mods:
			for command in self.categories[mod]:
				command.unload()
			for database in self.dbitems[mod]:
				database.unload()
			self.categories.pop(mod)
			self.dbitems.pop(mod)
			self.size.pop(mod)
		return True

	def reload(self, mod=None):
		if not mod:
			modload = deque()
			files = [i for i in os.listdir("commands") if is_code(i) and i.rsplit(".", 1)[0] in self.active_categories]
			for f in files:
				modload.append(esubmit(self.get_module, f, priority=True))
			esubmit(self.start_audio_client)
			csubmit(self.create_main_website())
			return all(fut.result() for fut in modload)
		return self.get_module(mod + ".py")

	size = fcdict()
	size2 = fcdict()
	def get_modules(self):
		"Loads all modules in the commands folder and initializes bot commands and databases."
		files = [i for i in os.listdir("commands") if is_code(i) and i.rsplit(".", 1)[0] in self.active_categories]
		self.categories = fcdict()
		self.dbitems = fcdict()
		self.commands = fcdict()
		self.data = fcdict()
		self.database = fcdict()
		for f in os.listdir():
			if is_code(f):
				self.size[f] = line_count(f)
		modload = deque()
		for f in files:
			modload.append(asubmit(self.get_module, f, priority=True))
		self.loaded = True
		return modload

	@tracebacksuppressor
	def clear_cache(self):
		if self.cache_semaphore.busy:
			return 0
		with self.cache_semaphore:
			i = 0
			t = utc()
			for path, limit in zip((CACHE_PATH, TEMP_PATH, FAST_PATH), (16384, 1024, 4096)):
				for k in ("", "audio"):
					atts = os.listdir(f"{path}/{k}")
					if len(atts) > limit:
						atts = sorted((f"{path}/{k}/{a}" for a in atts), key=lambda p: (t - os.path.getatime(p)) * os.path.getsize(p), reverse=True)
						for p in atts[:-limit]:
							with tracebacksuppressor(PermissionError):
								os.remove(p)
								i += 1
			with tracebacksuppressor(FileNotFoundError):
				c = len(os.listdir("misc/cache"))
				shutil.rmtree("misc/cache")
				i += c + 1
			if i > 1:
				print(f"{i} cached files flagged for deletion.")
			return i

	@tracebacksuppressor
	def backup(self):
		backup = AUTH.get("backup_path") or os.getcwd() + "/backup"
		self.clear_cache()
		date = utc_dt().date()
		if not os.path.exists(backup):
			os.mkdir(backup)
		fn = f"{backup}/saves.{date}.tar"
		if os.path.exists(fn):
			if utc() - os.path.getmtime(fn) < 60:
				return fn
			os.remove(fn)
		for i in range(30):
			d2 = date - datetime.timedelta(days=i + 7)
			f2 = f"{backup}/saves.{d2}.tar"
			if os.path.exists(f2):
				os.remove(f2)
				continue
			break
		args = ["tar", "-cvf", fn, "saves"]
		subprocess.run(args)
		if os.path.exists(fn):
			print("Backup database created in", fn)
		else:
			print("Backup database failed!", fn)
		return fn

	def update(self, force=False):
		"Autosaves modified bot databases. Called once every minute and whenever the bot is about to shut down."
		if force:
			self.update_embeds(True)
		saved = alist()
		with tracebacksuppressor:
			for i, u in self.data.items():
				if getattr(u, "sync", None):
					with MemoryTimer(f"{u}-sync"):
						if u.sync(force=True):
							saved.append(i)
							# time.sleep(0.05)
		backup = AUTH.get("backup_path") or "backup"
		if not os.path.exists(backup):
			os.mkdir(backup)
		fn = f"{backup}/saves.{DynamicDT.utcnow().date()}.tar"
		day = not os.path.exists(fn)
		if day:
			await_fut(self.send_event("_day_"))
			self.users_updated = True
		if force or day:
			fut = self.send_event("_save_")
			await_fut(fut)
		if day and (not self.collecting or self.collecting.done()):
			self.backup()
			futs = deque()
			for u in self.data.values():
				if not xrand(5) and not u._garbage_semaphore.busy:
					futs.append(self.garbage_collect(u))
			self.collecting = csubmit(gather(*futs))

	async def as_rewards(self, diamonds, gold=Dummy):
		if diamonds and not isinstance(diamonds, int):
			with suppress(OverflowError, ValueError):
				diamonds = floor(diamonds)
		if gold is Dummy:
			gold = diamonds
			diamonds = 0
		if gold and not isinstance(gold, int):
			with suppress(OverflowError, ValueError):
				gold = floor(gold)
		out = deque()
		if diamonds:
			out.append(f"💎 {diamonds}")
		if gold:
			coin = await self.data.emojis.emoji_as("miza_coin.gif", full=True)
			out.append(f"{coin} {gold}")
		if out:
			return " ".join(out)
		return

	zw_callback = zwencode("callback")

	async def react_callback(self, message, reaction, user):
		"Operates on reactions on special messages, calling the _callback_ methods of commands when necessary."
		if message.author.id == self.id:
			if self.closed:
				return
			u_perm = self.get_perms(user.id, message.guild)
			if u_perm <= -inf:
				return
			if reaction is not None:
				reacode = str(reaction).encode("utf-8")
			else:
				reacode = None
			interaction = hasattr(message, "int_token")
			# m = self.cache.messages.get(message.id)
			if getattr(message, "_react_callback_", None):
				resp = await message._react_callback_(
					message=message,
					channel=message.channel,
					guild=message.guild,
					reaction=reacode,
					user=user,
					perm=u_perm,
					vals="",
					argv="",
					bot=self,
				)
				if isinstance(resp, cdict):
					csubmit(self.ignore_interaction(message))
					await self.edit_message(
						message,
						**resp,
						allowed_mentions=discord.AllowedMentions.none(),
					)
				return
			msg = message.content.strip("*")
			if not msg and message.embeds:
				msg = str(message.embeds[0].description).strip("*")
			if msg[:3] != "```" or len(msg) <= 3:
				msg = None
				if message.embeds and message.embeds[0].footer:
					s = message.embeds[0].footer.text
					if is_zero_enc(s):
						msg = s
				if not msg:
					return
			else:
				msg = msg[3:].lstrip("\n")
				check = "callback-"
				with suppress(ValueError):
					msg = msg[:msg.index("\n")]
				if not msg.startswith(check):
					return
			while len(self.react_sem) > 65536:
				with suppress(RuntimeError):
					self.react_sem.pop(next(iter(self.react_sem)))
			while utc() - self.react_sem.get(message.id, 0) < 30:
				# Ignore if more than 2 reactions already queued for target message
				if self.react_sem.get(message.id, 0) - utc() > 1:
					return
				await asyncio.sleep(0.2)
			msg = message.content.strip("*")
			if not msg and message.embeds:
				msg = str(message.embeds[0].description).strip("*")
			if msg[:3] != "```" or len(msg) <= 3:
				msg = None
				if message.embeds and message.embeds[0].footer:
					s = message.embeds[0].footer.text
					if is_zero_enc(s):
						msg = s
				if not msg:
					return
				# Experimental zero-width invisible character encoded message (unused)
				try:
					msg = msg[msg.index(self.zw_callback) + len(self.zw_callback):]
				except ValueError:
					return
				msg = zwdecode(msg)
				args = msg.split("q")
			else:
				msg = msg[3:].lstrip("\n")
				check = "callback-"
				msg = msg.splitlines()[0]
				msg = msg[len(check):]
				args = msg.split("-")
			catn, func, vals = args[:3]
			func = func.casefold()
			argv = "-".join(args[3:])
			catg = self.categories[catn]
			# Force a rate limit on the reaction processing for the message
			self.react_sem[message.id] = max(utc(), self.react_sem.get(message.id, 0) + 1)
			for f in catg:
				if f.parse_name().casefold() == func:
					with self.ExceptionSender(message.channel, reference=message):
						timeout = getattr(f, "_timeout_", 1) * self.timeout
						if timeout >= inf:
							timeout = None
						elif self.is_trusted(message.guild):
							timeout *= 3
						self.data.usage.add(f)
						async with asyncio.timeout(timeout):
							future = csubmit(f._callback_(
								message=message,
								channel=message.channel,
								guild=message.guild,
								reaction=reacode,
								user=user,
								perm=u_perm,
								vals=vals,
								argv=argv,
								bot=self,
							))
							if interaction:
								fut = csubmit(delayed_callback(future, 1, self.defer_interaction, message, mode="patch", ephemeral=getattr(message, "ephemeral", False), exc=False))
							resp = await future
						await self.send_event("_command_", user=user, command=f, loop=False, message=message)
						if isinstance(resp, cdict):
							if interaction and not resp.get("file"):
								r, d = await fut
								if not d:
									await self.defer_interaction(message, mode="patch")
								await interaction_patch(
									bot=self,
									message=message,
									**resp,
								)
								break
							csubmit(self.ignore_interaction(message))
							await self.edit_message(
								message,
								**resp,
								allowed_mentions=discord.AllowedMentions.none(),
							)
						break
			self.react_sem.pop(message.id, None)

	status_cycle = Semaphore(1, 1, rate_limit=60, sync=True)
	@tracebacksuppressor
	async def update_status(self, force=False):
		guild_count = len(self.guilds)
		changed = force or guild_count != self.guild_count
		sem = emptyctx if changed else self.status_cycle
		if getattr(sem, "busy", False):
			return
		async with sem:
			self.guild_count = guild_count
			status_changes = list(range(self.status_iter))
			status_changes.extend(range(self.status_iter + 1, len(self.statuses) - (not self.audio)))
			if not status_changes:
				status_changes = range(len(self.statuses))
			self.status_iter = choice(status_changes)
			with suppress(discord.NotFound):
				if self.maintenance:
					if getattr(self, "laststat", None) == discord.Status.invisible:
						return
					text = "Currently under maintenance, please stay tuned!"
				elif not self.ready:
					text = "Currently loading, please wait..."
				elif AUTH.get("status"):
					text = AUTH["status"]
				else:
					text = f"{self.webserver}, to {uni_str(guild_count)} server{'s' if guild_count != 1 else ''}"
					if self.owners:
						u = await self.fetch_user(next(iter(self.owners)))
						n = u.display_name
						text += f", from {belongs(uni_str(n))} place!"
					else:
						text += "!"
				# Status iterates through 5 possible choices
				status = self.statuses[self.status_iter]
				if 0:
					status = None
					activity = discord.Game(name=text)
				elif status is discord.Streaming:
					activity = discord.Streaming(name=text, url=self.twitch_url)
					status = discord.Status.dnd
				else:
					activity = discord.Game(name=text)
				# if changed:
				# 	print(repr(activity))
				if self.audio:
					audio_status = "await client.change_presence(status=discord.Status."
					if status is None:
						status = discord.Status.offline
						csubmit(self.audio.asubmit(audio_status + "offline)"))
						await self.seen(self.user, event="misc", raw="Changing their status")
					elif status == discord.Status.invisible:
						status = discord.Status.idle
						esubmit(self.audio.submit(audio_status + "online)"))
						await self.seen(self.user, event="misc", raw="Changing their status")
					else:
						# if status == discord.Status.online:
						esubmit(self.audio.submit(audio_status + "dnd)"))
						csubmit(self.seen(self.user, event="misc", raw="Changing their status"))
				elif status is None:
					status = discord.Status.offline
				elif status == discord.Status.invisible:
					status = discord.Status.idle
				with suppress(ConnectionResetError):
					await self.change_presence(activity=activity, status=status)
				self.laststat = status
				# Member update events are not sent through for the current user, so manually send a _seen_ event
				await self.seen(self.user, event="misc", raw="Changing their status")

	async def handle_update(self, force=False):
		"Handles all updates to the bot. Manages the bot's status and activity on discord, and updates all databases."
		sem = self.update_semaphore if not force else emptyctx
		if T(sem).get("busy") and not force:
			return
		async with sem:
			if self.bot_ready:
				# Update databases
				futs = []
				for u in self.data.values():
					if not u._semaphore.busy:
						async def call_into(u):
							with MemoryTimer(f"{u}-call"):
								return await asubmit(u, priority=None)
						fut = call_into(u)
						futs.append(fut)
				await gather(*futs)

	async def extract_kwargs(self, argv, command, u_perm, user, message, channel, guild, command_check, o_kwargs) -> dict:
		schema = command.schema
		argv = argv or ""
		args, ws = smart_split(argv, rws=True)
		append_lws = None
		if schema is None:
			flags = {}
			for f in getattr(command, "flags", ()):
				f2 = f"-{f}"
				if f2 in args:
					args.remove(f2)
					argv = argv.replace(f2, "", 1).strip()
					add_dict(flags, {f: 1})
			return dict(
				perm=u_perm,
				user=user,
				message=message,
				channel=channel,
				guild=guild,
				name=command_check,
				looped=loop,
				argv=argv,
				args=args,
				argl=[],
				flags=flags,
			)
		oargs = tuple(args)
		if message and message.attachments:
			args = [best_url(a) for a in message.attachments] + args
		tz = None
		parser = getattr(command, "parser", None)
		if not parser:
			used = set()
			chars = set("abcdefghijklmnopqrstuvwxyz")
			parser = command.parser = argparse.ArgumentParser(prog=command.__name__, description=command.description, prefix_chars="-", exit_on_error=False, add_help=False)
			parser.error = lambda msg: throw(ArgumentError(msg))
			for k, v in reversed(schema.items()):
				all_aliases = [k, *v.get("aliases", ())]
				names = [("-" if len(x) == 1 else "--") + x for x in all_aliases]
				for a in all_aliases:
					if "_" in a:
						names.append("--" + a.replace("_", "-"))
				shortened = "-" + "".join(w[0] for w in k.split("_"))
				if k[0] in chars and shortened not in used:
					names.append(shortened)
					chars.remove(k[0])
				used.update(names)
				if v.get("type") == "bool":
					parser.add_argument(*names, action=argparse.BooleanOptionalAction)
					continue
				action = "append" if v.get("multiple") else "store"
				parser.add_argument(*names, action=action)
			parser.has_string = []
			for k, v in schema.items():
				if v.get("type") == "string" and v.get("greedy", True):
					parser.has_string.append(k)
				if v.get("type") == "enum":
					for e in v.validation.enum:
						names = ["--" + e, "-" + "".join(w[0] for w in e.split("_"))]
						if "_" in e:
							names.append("--" + e.replace("_", "-"))
						names = [n for n in names if n not in used]
						if not names:
							continue
						used.update(names)
						action = "append_const" if v.get("multiple") else "store_const"
						parser.add_argument(*names, dest=k, action=action, const=e)
		spl = parser.parse_known_args(args)
		kwargs = cdict((k, v) for k, v in spl[0]._get_kwargs() if v is not None)
		if o_kwargs:
			for k, v in o_kwargs.items():
				kwargs.setdefault(k, v)
		for k, v in tuple(kwargs.items()):
			if "-" in k:
				kwargs[k.replace("-", "_")] = kwargs.pop(k)
		args = alist(spl[0]._get_args() + spl[1])
		if args:
			for k, v in schema.items():
				if k in kwargs or not v.get("greedy", True):
					continue
				r = None
				# Datetime and timedelta are greedy arguments that attempt to match as many contiguous words as possible; e.g. "2021 12 31" turns into a single datetime object
				if v.type == "datetime":
					def try_parse_date(s):
						try:
							DynamicDT.parse(s)
						except Exception:
							return False
						return True
					extracted, index = longest_sublist(args, lambda x: try_parse_date(" ".join(x)))
					if extracted:
						check = " ".join(extracted)
						if not check.replace(" ", "").isnumeric():
							if user and "users" in self.data:
								tz = tz or self.data.users.any_timezone(user.id)
							r = DynamicDT.parse(check, timezone=tz)
							args.pops(range(index, index + len(extracted)))
				elif v.type == "timedelta":
					def try_parse_timedelta(s):
						try:
							DynamicDT.parse_delta(s)
						except Exception:
							return False
						return True
					extracted, index = longest_sublist(args, lambda x: try_parse_timedelta(" ".join(x)))
					if extracted:
						check = " ".join(extracted)
						if not check.replace(" ", "").isnumeric():
							if user and "users" in self.data:
								tz = tz or self.data.users.any_timezone(user.id)
							r = DynamicDT.parse_delta(" ".join(extracted))
							args.pops(range(index, index + len(extracted)))
				if r:
					kwargs[k] = [r] if v.get("multiple") else r
					continue
		oj = 0
		pops = []
		for i, a in enumerate(args):
			for k, v in schema.items():
				if k in kwargs and not v.get("multiple"):
					if v.type in ("text", "string"):
						if a in oargs:
							j = oargs.index(a, oj)
							oj = j
							kwargs[k] = (kwargs.get(k) or "") + ws[j] + a
							append_lws = (k, j + 1)
							pops.append(i)
						else:
							kwargs[k] = (kwargs.get(k) or "") + " " + a
							pops.append(i)
						break
					continue
				hs = parser.has_string
				taken = False
				if not hs and v.type == "bool" and full_prune(a) in ("true", "false", "t", "f", "1", "0"):
					taken = True
				elif not hs and v.type == "enum" and (full_prune(a) in v.validation.enum or full_prune(a) in v.validation.get("accepts", ())):
					taken = True
				elif v.type == "emoji" and (find_emojis_ex(a) or is_discord_emoji(a)):
					taken = True
				elif v.type in ("url", "image", "visual", "video", "audio", "media") and (is_url(a) or find_emojis_ex(a) or a[0] == "<" and a[-1] == ">"):
					taken = True
				elif v.type == "message" and is_discord_message_link(a):
					taken = True
				elif not hs and v.type == "filesize" and re.fullmatch(r"[\.0-9]+[A-Za-z]?[Bb]", a):
					taken = True
				elif not hs and v.type == "resolution" and re.fullmatch(r"-?[0-9]+[:x*]-?[0-9]+", a):
					taken = True
				elif not hs and v.type == "index" and (a.casefold() == "all" or re.fullmatch(r"(?:[\-0-9]+|[:\-]|\.{2,}){1,5}", a)):
					taken = True
				elif not hs and v.type in ("number", "integer") and re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\/\d*\.?\d+)?", a):
					taken = True
				elif hs and v.type == "string":
					taken = True
				if not taken:
					continue
				if k in kwargs:
					kwargs[k].append(a)
				else:
					kwargs[k] = [a] if v.get("multiple") else a
				pops.append(i)
				break
		args.pops(pops)
		oj = 0
		pops = []
		for i, a in enumerate(args):
			for k, v in schema.items():
				if k in kwargs and not v.get("multiple"):
					if v.type in ("text", "string"):
						if a in oargs:
							j = oargs.index(a, oj)
							oj = j
							kwargs[k] = (kwargs.get(k) or "") + ws[j] + a
							append_lws = (k, j + 1)
							pops.append(i)
						else:
							kwargs[k] = (kwargs.get(k) or "") + " " + a
							pops.append(i)
						break
					continue
				taken = False
				if v.type in ("word", "string"):
					taken = True
				elif v.type == "message" and a.isnumeric():
					taken = True
				elif v.type in ("datetime", "timedelta"):
					taken = True
				elif v.type in ("number", "integer") and any(c.isnumeric() for c in a):
					taken = True
				elif v.type in ("mentionable", "user", "channel", "guild", "role"):
					taken = True
				elif v.type == "colour":
					taken = True
				if not taken:
					continue
				if k in kwargs:
					kwargs[k].append(a)
				else:
					kwargs[k] = [a] if v.get("multiple") else a
				pops.append(i)
				break
		args.pops(pops)
		oj = 0
		pops = []
		for i, a in enumerate(args):
			for k, v in schema.items():
				if k in kwargs and not v.get("multiple"):
					if v.type in ("word", "text", "string"):
						if a in oargs:
							j = oargs.index(a, oj)
							oj = j
							kwargs[k] = (kwargs.get(k) or "") + ws[j] + a
							append_lws = (k, j + 1)
							pops.append(i)
						else:
							kwargs[k] = (kwargs.get(k) or "") + " " + a
							pops.append(i)
						break
					continue
		args.pops(pops)
		if args:
			for k, v in schema.items():
				if k in kwargs:
					continue
				kwargs[k] = args.pop(0)
				if not args:
					break
		if not args:
			for k, v in schema.items():
				if k in kwargs or not v.get("required"):
					continue
				r = None
				if v.type in ("url", "image", "visual", "video"):
					url = None
					if getattr(message, "reference", None):
						urls = await self.follow_url(message, ytd=False)
						if urls and not is_discord_message_link(urls[0]):
							url = urls[0]
					if not url:
						try:
							url = await bot.get_last_image(message.channel)
						except FileNotFoundError:
							pass
					if url:
						r = url
				elif v.type == "message":
					reference = getattr(message, "reference", None)
					if reference:
						r = await self.fetch_message(reference.message_id, message.channel)
					else:
						r = message
				elif v.type in ("mentionable", "user"):
					r = user
				elif v.type == "channel":
					r = channel
				elif v.type == "guild":
					r = guild
				elif v.type == "role":
					if getattr(user, "roles", None):
						r = user.roles[-1]
				elif v.type in ("media", "audio") and guild.me.voice:
					auds = bot.data.audio.players[guild.id]
					if auds.queue and auds.queue[0].url:
						r = auds.queue[0].url
				if r:
					kwargs[k] = [r] if v.get("multiple") else r
					continue
				print(kwargs)
				if v:
					raise ArgumentError(f"Argument {k} ({v.description}) is required.")
				raise ArgumentError(f"Argument {k} ({v.type}) is required.")
		if append_lws:
			k, j = append_lws
			if j < len(ws):
				kwargs[k] = (kwargs[k] + ws[j]).strip()
		return await self.validate_schema(kwargs, schema, command_check=command_check, argv=argv, args=args, guild=guild)

	async def validate_into(self, k, v, info, guild):
		if not isinstance(v, str):
			return v
		err = lambda e, k, v: e(f'Unable to parse input {json.dumps(v)} for {k}.')
		if info.type in ("mentionable", "user", "channel", "guild", "role"):
			m = verify_id(v)
			if info.type == "mentionable":
				if isinstance(m, int):
					v = self.in_cache(m)
				elif isinstance(m, int):
					v = await self.fetch_messageable(m)
				else:
					v2 = verify_id(m)
					v = None
					if isinstance(v2, int):
						try:
							v = await self.fetch_messageable(v2)
						except Exception:
							pass
					if v is None:
						v = await self.fetch_member_ex(m, guild)
			elif info.type == "user":
				v2 = verify_id(m)
				v = None
				if isinstance(v2, int):
					v = guild.get_member(v2)
					if v is None:
						try:
							v = await self.fetch_user(v2)
						except Exception:
							pass
				if v is None:
					v = await self.fetch_member_ex(m, guild)
			elif info.type == "channel":
				if isinstance(m, int):
					v = await self.fetch_channel(m)
				elif not guild:
					raise TypeError("Channels must be specified by ID outside of servers.")
				else:
					v2 = verify_id(m)
					v = None
					if isinstance(v2, int):
						try:
							v = await self.fetch_channel(v2)
						except Exception:
							pass
					if v is None:
						v = await str_lookup(
							guild.channels,
							m,
							qkey=userQuery1,
							ikey=userIter1,
							fuzzy=1 / 3,
						)
			elif info.type == "guild":
				v = await self.fetch_guild(m)
			elif info.type == "role":
				if isinstance(m, int):
					v = await self.fetch_role(m, guild)
				elif not guild:
					raise TypeError("Roles must be specified by ID outside of servers.")
				else:
					v2 = verify_id(m)
					v = None
					if isinstance(v2, int):
						try:
							v = await self.fetch_role(v2, guild)
						except Exception:
							pass
					if v is None:
						v = await str_lookup(
							guild.roles,
							m,
							qkey=userQuery1,
							ikey=userIter1,
							fuzzy=1 / 3,
						)
		elif info.type == "emoji":
			if isinstance(v, string_like):
				v = await self.id_from_message(v)
		elif info.type in ("url", "image", "visual", "video", "audio", "media"):
			ytd = info.type in ("image", "visual", "video", "audio")
			urls = await self.follow_url(v, ytd=ytd, reactions=True, allow=True)
			if not urls or is_discord_message_link(urls[0]):
				raise err(TypeError, k, v)
			v = urls[0]
		elif info.type == "message":
			if isinstance(v, str):
				assert is_discord_message_link(v), f"{k}: Expected valid message link."
				p1, p2, p3, p4, gid, cid, mid, *_ = v.split("/", 7)
				channel = await self.fetch_channel(cid)
				v = await self.fetch_message(mid, channel)
			elif isinstance(v, int):
				v = await self.fetch_message(v)
		elif info.type == "filesize":
			if not isinstance(v, (int, float, np.number)):
				try:
					v = byte_unscale(full_prune(v).removesuffix("b"))
				except (IndexError, OverflowError, ValueError) as ex:
					raise err(ex.__class__, k, v)
		elif info.type == "bool":
			if not isinstance(v, bool):
				v = full_prune(v)
				if v in ("true", "t", "1"):
					v = True
				elif v in ("false", "f", "0"):
					v = False
				else:
					raise err(TypeError, k, v)
		elif info.type in ("number", "integer"):
			if not isinstance(v, (int, float, np.number)):
				try:
					v = await self.eval_math(full_prune(v))
					if info.type == "integer" and isfinite(v):
						v = int(v)
				except Exception as ex:
					raise err(ex.__class__, k, v)
		elif info.type == "datetime":
			try:
				v = DynamicDT.parse(v)
			except Exception as ex:
				raise err(ex.__class__, k, v)
		elif info.type == "timedelta":
			try:
				v = DynamicDT.parse_delta(v)
			except Exception as ex:
				raise err(ex.__class__, k, v)
		elif info.type == "resolution":
			try:
				def round_or_omit(s):
					if s == "-":
						return s
					return round_min(s)
				v = tuple(map(round_or_omit, regexp(r"[*x:]").split(v, 1)))
			except Exception as ex:
				raise err(ex.__class__, k, v)
		elif info.type == "index":
			if v.casefold == "all":
				v = [None, None]
			else:
				try:
					if re.fullmatch(r"[0-9]+-[0-9]+", v):
						v = tuple(map(int, v.split("-", 1)))
					else:
						v = tuple(int(x) if x else None for x in re.split(r"(?:\.{2,}|:)", v))
				except Exception as ex:
					raise err(ex.__class__, k, v)
		elif info.type == "colour":
			try:
				v = parse_colour(full_prune(v))
			except Exception as ex:
				raise err(ex.__class__, k, v)
		validation = info.get("validation")
		if not validation:
			return v
		if isinstance(validation, str):
			if info.type == "resolution":
				verifs = [n for n in v if n != "-"]
			elif info.type == "colour":
				verifs = list(v)
			else:
				verifs = [v]
			for x in verifs:
				lx, rx = validation.split(",")
				mx, Mx = lx[1:].strip(), rx[:-1].strip()
				if not mx:
					mx = -inf
				else:
					mx = float(round_min(mx))
				if not Mx:
					Mx = inf
				else:
					Mx = float(round_min(Mx))
				valid = True
				if lx[0] == "(" and not mx < x:
					valid = False
				elif lx[0] == "[" and not mx <= x:
					valid = False
				elif rx[-1] == ")" and not x < Mx:
					valid = False
				elif rx[-1] == "]" and not x <= Mx:
					valid = False
				if not valid:
					raise OverflowError(f'{k} value "{x}" must be in range {validation}.')
		elif validation.get("mapping"):
			try:
				return validation.mapping(v)
			except Exception:
				e = validation.get("error")
				if not e:
					raise
				if "{}" in E:
					raise ValueError(e.format(v))
				raise ValueError(e)
		elif validation.get("function"):
			try:
				assert validation.function(v)
			except Exception:
				e = validation.get("error")
				if not e:
					raise
				if "{}" in E:
					raise ValueError(e.format(v))
				raise ValueError(e)
		elif validation.get("enum") or validation.get("accepts"):
			v = v if validation.get("strict_case") else full_prune(v)
			enum = validation.get("enum", ())
			accepts = validation.get("accepts", ())
			if v not in enum and v not in accepts:
				enum = set(enum)
				accepts = set(accepts)
				if enum and accepts:
					raise EnumError(f'{k} value "{v}" must be one of {enum} or aliases {accepts}.')
				raise EnumError(f'{k} value "{v}" must be one of {enum.union(accepts)}.')
			if v not in enum:
				return validation.accepts[v]
		return v

	async def run_command(self, command, kwargs=None, message=None, argv=None, comment=None, slash=False, command_check=None, user=None, channel=None, guild=None, min_perm=None, respond=False, allow_recursion=True):
		command_check = command_check or command.name[0].casefold()
		user = user or (message.author if message else self.GhostUser())
		if message and user:
			print(f"{message.channel.id}: {user} ({user.id}) issued command {command_check} {kwargs or argv}")
		soon_indicator = False
		if not self.ready:
			# If the bot is not currently ready (either loading or in maintenance), send an indicator and wait
			if message:
				soon_indicator = csubmit(message.add_reaction("🔜"))
				if slash or getattr(message, "slash", False):
					await self.defer_interaction(message, ephemeral=getattr(message, "ephemeral", False))
			await wrap_future(self.connect_ready)
		channel = channel or (message.channel if message else None)
		guild = guild or getattr(channel, "guild", None)
		if user.id == self.id:
			prefix = self.prefix
		else:
			prefix = self.get_prefix(guild)
		if getattr(command, "nsfw", False) and not self.is_nsfw(channel):
			if hasattr(channel, "recipient"):
				raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels. Please verify your age using ~verify within a NSFW channel to enable NSFW in DMs.")
			raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels.")
		# Make sure server-only commands can only be run in servers.
		if guild is None or getattr(guild, "ghost", None):
			if channel:
				if channel.id not in self.cache.channels:
					self.cache.channels[channel.id] = channel
				channel = await self.fetch_channel(channel.id)
				guild = guild or getattr(channel, "guild", None)
			if getattr(command, "server_only", False) and (guild is None or getattr(guild, "ghost", None)):
				raise ReferenceError("This command is only available in servers.")
		req = command.min_level
		sem = emptyctx
		if not self.ready:
			await wrap_future(self.full_ready)
			if channel:
				channel = await self.fetch_channel(channel.id)
				guild = getattr(channel, "guild", None) or guild
		if soon_indicator:
			# Remove the "soon" indicator since the bot is now ready
			await soon_indicator
			csubmit(message.remove_reaction("🔜", self.user))
		u_perm = max(min_perm, self.get_perms(user.id, guild)) if min_perm is not None else self.get_perms(user.id, guild)
		if not isnan(u_perm):
			enabled = self.get_enabled(channel)
			if full_prune(command.category) not in enabled and isfinite(u_perm):
				raise PermissionError(f"This command is not enabled here. Use {prefix}ec to view or modify the list of enabled commands")
			if getattr(command, "maintenance", False):
				raise NotImplementedError("This command is disabled due to pending or ongoing maintenance, sorry!")
			if not allow_recursion and T(command).get("recursive", False):
				raise PermissionError("Nested recursive commands are not permitted.")
			min_perm = None
			gid = self.data.blacklist.get(0)
			if gid and gid != guild.id and not isnan(u_perm):
				print("BOUNCED:", user, message.content)
				csubmit(send_with_react(
					channel,
					f"I am currently under maintenance, please [stay tuned](<{self.rcc_invite}>)!",
					reacts="❎",
					reference=message,
				))
				return
			elif u_perm <= -inf:
				print("REFUSED:", user, message.content)
				csubmit(send_with_react(
					channel,
					"Sorry, you are currently not permitted to request my services.",
					reacts="❎",
					reference=message,
				))
				return
		# Make sure target has permission to use the target command, rate limit the command if necessary.
		if not isnan(u_perm):
			if not u_perm >= req:
				raise command.perm_error(u_perm, req, "for command " + command_check)
			rl = command.rate_limit
			if rl:
				rl = rl[bool(self.is_trusted(guild))] if isinstance(rl, (tuple, list)) else rl
				pm = self.premium_multiplier(self.premium_level(user))
				uid = user.id if user else None
				rl /= pm
				burst = ceil(pm + 2)
				rlv = ceil(rl * burst)
				sem = command.used.get(uid)
				if sem is None or sem.rate_limit > rlv or not sem.active and sem.rate_limit < rlv:
					sem = command.used[uid] = Semaphore(burst, burst, rate_limit=rlv)
				if sem.full and sem.reset_after:
					raise TooManyRequests(f"Command has a rate limit of {sec2time(rl)} with a burst+queue of {burst}; please wait {sec2time(sem.reset_after)}.")
		# Assign "guild" as an object that mimics the discord.py guild if there is none
		if guild is None:
			guild = self.UserGuild(
				user=user,
				channel=channel,
			)
			channel = guild.channel
		elif channel and guild.me and hasattr(guild.me, "timed_out") and (guild.me.timed_out or not channel.permissions_for(guild.me).send_messages):
			raise PermissionError("Unable to send message.")
		if getattr(sem, "busy", None):
			csubmit(message.add_reaction("🌡️"))
		if command_check in command.macromap:
			kv = command.macromap[command_check].copy()
			if kwargs:
				kv.update(kwargs)
			kwargs = kv
		kwargs = await self.extract_kwargs(argv, command, u_perm, user, message, channel, guild, command_check, kwargs)
		comment = comment or ""
		fut = None
		async with sem:
			# Automatically start typing if the command is time consuming
			tc = getattr(command, "time_consuming", False)
			if not loop and tc and not getattr(message, "simulated", False):
				fut = csubmit(self._state.http.send_typing(channel.id))
			# Get maximum time allowed for command to process
			if isnan(u_perm):
				timeout = None
			else:
				timeout = getattr(command, "_timeout_", 1) * self.timeout
				if timeout >= inf:
					timeout = None
				elif message and self.is_trusted(message.guild):
					timeout *= 2
				timeout *= self.premium_multiplier(self.premium_level(user))
			premium = self.premium_context(user, guild=guild)
			# Create a future to run the command
			future = asubmit(
				command,						# command is a callable object, may be async or not
				bot=self,						# for interfacing with bot's database
				_prefix=prefix,
				_premium=premium,
				_perm=u_perm,					# permission level
				_nsfw=self.is_nsfw(channel),
				_user=user,						# user that invoked the command
				_message=message,				# message data
				_channel=channel,				# channel data
				_guild=guild,					# guild data
				_name=command_check,			# alias the command was called as
				_comment=comment,
				_slash=slash,
				_looped=loop,					# whether this command was invoked as part of a loop
				_timeout=timeout,				# timeout delay assigned to the command
				**kwargs,						# Keyword arguments for schema-specified commands
				timeout=timeout and timeout + 1,# timeout delay for the whole function
			)
			if not T(command).get("no_cancel"):
				try:
					message.__dict__.setdefault("inits", []).append(future)
				except Exception:
					pass
			self.data.usage.add(command)
			# Add a callback to typing in the channel if the command takes too long
			if fut is None and not hasattr(command, "typing") and channel and not getattr(message, "simulated", False):
				csubmit(delayed_callback(future, sqrt(3), self._state.http.send_typing, channel.id, repeat=7, exc=True))
			if slash or getattr(message, "slash", None):
				csubmit(delayed_callback(future, 1, self.defer_interaction, message, ephemeral=getattr(message, "ephemeral", False), exc=False))
			csem = emptyctx if isnan(command.min_level) else self.command_semaphore
		async with csem:
			response = await future
			await self.send_event("_command_", user=user, command=command, loop=loop, message=message)
			if isinstance(response, str):
				response = cdict(content=response)
			if not respond:
				return response
			fut = csubmit(self.respond_with(response, message=message, command=command))
			try:
				message.__dict__.setdefault("inits", []).append(fut)
			except Exception:
				pass
			return await fut

	async def validate_schema(self, kwargs, schema, command_check="", argv="", args=(), guild=None):
		if args:
			for arg in args:
				for k, v in schema.items():
					if v.get("multiple"):
						args = kwargs.setdefault(k, [])
						if not isinstance(args, list):
							args = kwargs[k] = [args]
						args.append(arg)
						break
					elif v.type in ("text", "string"):
						kwargs[k] = (kwargs.get(k) or "") + arg
						break
		for k, v in schema.items():
			if k in kwargs and v.get("multiple"):
				ks = kwargs[k]
				if len(ks) == 1 and not v.get("required") and ks[0] is None:
					kwargs[k] = None
		if len(kwargs) == 1:
			k = next(iter(kwargs))
			if argv and schema[k].type in ("string",) and not schema[k].get("multiple"):
				kwargs[k] = argv
		for k, v in schema.items():
			if v.get("required", 0) > 1 and len(kwargs[k]) < v.required:
				raise ArgumentError(f'{k} requires a minimum amount of {v.required} inputs.')
		for k, info in schema.items():
			if k not in kwargs or kwargs[k] == "-":
				if k == next(iter(schema)) and info.type == "enum" and (command_check in info.validation.get("enum", ()) or command_check in info.validation.get("accepts", ())):
					kwargs[k] = info.validation.accepts[command_check] if command_check not in info.validation.get("enum", ()) else command_check
				elif info.get("default") is not None:
					kwargs[k] = info.default
				elif info.get("required"):
					raise ArgumentError(f"Required input {k} was not found.")
			v = kwargs.get(k)
			if v is None:
				kwargs[k] = None
				continue
			if info.get("multiple"):
				futs = [self.validate_into(k, a, info=info, guild=guild) for a in v]
				kwargs[k] = await gather(*futs)
				continue
			kwargs[k] = await self.validate_into(k, v, info=info, guild=guild)
		return kwargs

	async def parse_command(self, message):
		if utc() - message.created_at.timestamp() > 14 * 86400:
			return
		user = message.author
		if getattr(user, "bot", None) and getattr(user, "webhook_id", None):
			return
		from_mention = False
		comm = message.content
		prefix = self.get_prefix(message.guild)
		# Mentioning the bot serves as an alias for the prefix.
		for check in self.mention:
			if comm.startswith(check):
				prefix = self.prefix
				comm = comm[len(check):].strip()
				from_mention = True
				break
		if comm.startswith(prefix):
			comm = comm[len(prefix):].strip()
		elif not from_mention:
			return
		# Special case: the ? alias for the ~help command, since ? would not otherwise be an alias
		if len(comm) and comm[0] == "?":
			command_check = comm[0]
			i = 1
		else:
			# Parse message to find command.
			i = len(comm)
			for end in " ?\t\n":
				with suppress(ValueError):
					i2 = comm.index(end)
					if i2 < i:
						i = i2
			command_check = full_prune(comm[:i]).replace("*", "").replace("_", "").replace("||", "")
		# Hash table lookup for target command: O(1) average time complexity.
		if command_check in self.commands:
			# Multiple commands may have the same alias, run all of them
			for command in self.commands[command_check]:
				if from_mention and getattr(command, "exact", True) and full_prune(comm) != command_check:
					continue
				# argv is the raw parsed argument data
				argv = comm[i:].strip()
				argv = await self.proxy_emojis(argv, guild=message.guild, user=user, is_webhook=getattr(message, "webhook_id", None), lim=inf)
				yield command, command_check, argv, from_mention

	async def process_message(self, message, before=None, min_perm=None, kwargs=None):
		"Processes a message, runs all necessary commands and bot events. May be called from another source."
		if self.closing:
			return 0
		user = message.author
		channel = message.channel
		guild = message.guild
		u_id = user.id
		if u_id == self.id:
			return 0

		truemention = True
		if self.id in (member.id for member in message.mentions) and not isinstance(before, self.GhostMessage):
			try:
				m = await self.fetch_reference(message)
			except (LookupError, discord.NotFound):
				pass
			else:
				truemention = m.author.id != self.id and all(s not in message.content for s in self.mention)
			if truemention:
				try:
					await self.send_event("_mention_", user=user, message=message, exc=True)
				except CommandCancelledError:
					return 0

		run = 0
		async for command, command_check, argv, from_mention in self.parse_command(message):
			run += 1
			out_fut = None
			try:
				await self.run_command(command, kwargs, message=message, argv=argv, command_check=command_check, min_perm=min_perm, respond=True)
			# Represents any timeout error that occurs
			except CE:
				print(command, argv)
				raise
			except (T0, T1, T2, ArgumentError, TooManyRequests) as ex:
				out_fut = self.send_exception(channel, ex, reference=message, comm=command)
				return
			# Represents all other errors
			except Exception as ex:
				print_exc()
				if from_mention:
					prefix = self.get_prefix(guild)
					op = ("Unintentional command?", f"If you meant to chat with me instead, use {prefix}ask or one of its aliases to avoid accidentally triggering a command in the future!")
				else:
					op = None
				out_fut = self.send_exception(channel, ex, reference=message, op=op, comm=command)
			if out_fut is not None and getattr(message, "simulated", None):
				await out_fut
			elif getattr(message, "simulated", None):
				return -1
		if not run:
			# If message was not processed as a command, send a _nocommand_ event with the parsed message data.
			with self.command_semaphore:
				await self.send_event("_nocommand2_", message=message)
				not_self = True
				if u_id == self.id:
					not_self = False
				elif T(message).get("webhook_id") and guild and user.name == guild.me.display_name:
					cola = await self.get_colour(self)
					colb = await self.get_colour(user)
					not_self = cola != colb
				if not_self:
					msg = message.content
					temp = to_alphanumeric(msg).casefold()
					temp2 = to_alphanumeric(message.clean_content or msg).casefold()
					await self.send_event("_nocommand_", text=temp, text2=temp2, edit=bool(before), before=before, msg=msg, message=message, perm=self.get_perms(user, guild), truemention=truemention)
			return 0
		# Return the delay before the message can be called again. This is calculated by the rate limit of the command.
		return 0
	
	async def respond_with(self, response, message=None, command=None, manager=None, done=True, timeout=3600):
		with self.command_semaphore:
			# Limit for regular Discord messages, beyond which we'll need to split the text into multiple messages
			msglen = 2000
			# Maximum length we'll allow for a set of messages, beyond which the text will instead be uploaded as a file
			maxlen = 12000
			force = bool(manager)
			channel = manager.channel if manager else message.channel if hasattr(message, "channel") else response.get("channel")
			guild = (manager.guild if manager else message.guild if hasattr(message, "guild") else response.get("guild")) or getattr(channel, "guild", None)
			# Process response to command if there is one
			if response and isinstance(response, dict):
				bypass_prefix = response.pop("bypass_prefix", None)
				bypass_suffix = response.pop("bypass_suffix", None)
				prefix = response.pop("prefix", None) or ""
				suffix = response.pop("suffix", None) or ""
				content = response.pop("content", None) or ""
				tts = response.pop("tts", False) or False
				def get_prefix():
					return prefix if not bypass_prefix or none(content.startswith(b) for b in bypass_prefix) else ""
				def get_suffix():
					return suffix if not bypass_suffix or none(content.endswith(b) for b in bypass_suffix) else ""
				reference = response["reference"] if "reference" in response else message or response.get("message")
				file = response.get("file")
				if manager or not getattr(reference, "simulated", False) and not isinstance(content, (str, bytes)):
					if manager:
						old_content = manager.content
					else:
						manager = self.StreamedMessage(channel, reference=reference, msglen=msglen, maxlen=maxlen)
						old_content = ""

					def add_content(old_content, content):
						if not old_content:
							return content
						elif old_content.endswith("```") or content.startswith("```"):
							return old_content + content
						return old_content + "\n" + content

					if isinstance(content, collections.abc.AsyncIterator):
						it = content
					elif isinstance(content, str | dict):
						async def iterator():
							yield content
						it = iterator()
					else:
						raise NotImplementedError(content)
					start = utc()
					ct = start + 1
					async with discord.context_managers.Typing(channel):
						content = await anext(it)
						if isinstance(content, dict):
							response.update(content)
							content = response.pop("content", "")
						task = None
						blocked = False
						edit = True
						fut = None
						resp = None
						try:
							while True:
								if reference:
									await self.require_integrity(reference)
								try:
									d = ct - utc() + 1
									if d <= 0:
										raise TimeoutError
									if not task:
										task = csubmit(anext(it))
									wait = asyncio.shield(task) if utc() < start + timeout else task
									resp = await asyncio.wait_for(wait, timeout=d)
								except (T0, T1, CE):
									if not blocked and edit and (not fut or fut.done()):
										if fut:
											try:
												await fut
											except InterruptedError:
												# If the StreamedMessage is interrupted, it is most likely from another user sending a new message. We block the current stream and buffer all other content until we have the full message.
												blocked = True
										try:
											new_content = add_content(old_content, content)
											if len(new_content) >= 32 or "." in new_content:
												fut = csubmit(manager.update(new_content, prefix=prefix, suffix=suffix, bypass=(bypass_prefix, bypass_suffix), force=False, done=False))
										except OverflowError:
											blocked = True
										else:
											edit = False
									ct = utc()
								else:
									if isinstance(resp, dict):
										response.update(resp)
										resp = response.pop("content", "")
									if "\r" in content:
										content = content.rsplit("\r", 1)[-1]
									if resp.startswith("\r"):
										content = resp.lstrip("\r")
									elif resp:
										content += resp
									edit = True
									task = None
						except StopAsyncIteration:
							pass
					content = content.strip()
					embeds = manager.embeds + (response.get("embeds") or ([response["embed"]] if response.get("embed") else None) or [])
					files = manager.files + (response.get("files") or ([response["file"]] if response.get("file") else None) or [])
					reacts = response.get("reacts")
					buttons = manager.buttons + (response.get("buttons") or [])
					print("STOP:", content, embeds, files, reacts, buttons)
					total_length = len(get_prefix()) + len(content) + len(get_suffix())
					if fut:
						with tracebacksuppressor:
							await fut
					try:
						new_content = add_content(old_content, content)
						await manager.update(new_content, embeds=embeds, files=files, buttons=buttons, prefix=prefix, suffix=suffix, bypass=(bypass_prefix, bypass_suffix), reacts=reacts, done=done, force=force)
					except (OverflowError, InterruptedError):
						# If the StreamedMessage was interrupted or exceeded the maximum length, we wipe the original and force a new message. This ensures the list of messages stays contiguous, which improves readability.
						csubmit(manager.delete())
					else:
						messages = await manager.collect()
						for m in messages:
							self.add_message(m, force=2)
						return manager
				if isinstance(content, collections.abc.AsyncIterator):
					async for cc in content:
						if isinstance(cc, dict):
							response.update(cc)
						else:
							response["content"] = cc
					content = (response.pop("content", None) or "").lstrip("\r")
				original_nickname = None
				try:
					total_length = len(get_prefix()) + len(content) + len(get_suffix())
					if total_length > msglen or tts:
						if total_length > maxlen:
							data = content.encode("utf-8")
							file2 = CompatFile(data, filename="message.txt")
							if file:
								response["files"] = [file, file2]
								file = None
							else:
								response.pop("files", None)
								file = file2
							content = "Response too long for message."
						else:
							if tts:
								# For TTS mode, we split the message into smaller chunks to ensure the message is read out correctly. Unlike the regular limit of 2000 characters which is known, the TTS limit is not well documented and may vary between clients. We impose an arbitrary limit of 150 characters instead, which should be safe for most clients.
								ms = split_text(content, prefix=prefix, suffix=suffix, max_length=150)
							else:
								ms = split_text(content, prefix=prefix, suffix=suffix, max_length=msglen)
							content = ms[-1] if ms else "\xad"
							futs = []
							for i, t in enumerate(ms[:-1]):
								if tts and i == 1 and channel and guild and guild.me.permissions_in(channel).change_nickname:
									# If we've got more than one message in TTS mode, we automatically backup the bot's nickname and replace it with a backtick (silent character) to avoid it being read out alongside every message.
									original_nickname = guild.me.nick or "" # The "" is crucial to differentiate between None and an empty string.
									await guild.me.edit(nick="`")
								fut = csubmit(send_with_react(channel, t, reference=reference, tts=tts))
								futs.append(fut)
								reference = None
								await asyncio.sleep(0.125)
							await gather(*futs)
					else:
						content = get_prefix() + content + get_suffix() or None
					if file and not response.get("files") and not response.get("buttons"):
						return await self.send_with_file(
							response.get("channel") or channel,
							msg=content,
							file=file,
							filename=getattr(file, "filename", None),
							embed=response.get("embed"),
							reference=reference,
							reacts=response.get("reacts"),
							tts=tts,
						)
					return await send_with_react(
						response.get("channel") or channel,
						content=content,
						file=file,
						files=response.get("files"),
						embed=response.get("embed"),
						embeds=response.get("embeds"),
						reference=reference,
						buttons=response.get("buttons"),
						reacts=response.get("reacts"),
						ephemeral=getattr(message, "ephemeral", False),
						tts=tts,
					)
				finally:
					if original_nickname is not None:
						# If we've changed the bot's nickname, we restore it to its original state.
						await guild.me.edit(nick=original_nickname)
			return response

	async def run_simulate(self, ip, command):
		message = SimulatedMessage(self, command, utc(), ip, "user")
		self.cache.users[message.author.id] = message.author
		resp = {}
		async for command, command_check, argv, from_mention in self.parse_command(message):
			resp = await self.run_command(command, message=message, argv=argv, slash=True, respond=False)
			break
		return resp

	chunk_guild_sems = None
	load_guild_sem = Semaphore(48, inf, rate_limit=1)
	async def load_guild(self, guild):
		if not self.chunk_guild_sems:
			self.chunk_guild_sems = [Semaphore(3, inf, rate_limit=0.5) for i in range(self.shard_count)]
		if "guilds" in self.data:
			self.data.guilds.update_guild(guild)
		finished = False
		async with self.load_guild_sem:
			member_count = getattr(guild, "_member_count", None) or len(guild._members)
			sid = self.guild_shard(guild.id)
			if member_count in range(3, 250) and not self.shards[sid].is_ws_ratelimited() and not self.chunk_guild_sems[sid].busy:
				try:
					async with self.chunk_guild_sems[sid]:
						await asyncio.wait_for(self._connection.chunk_guild(guild), timeout=30)
				except Exception:
					print_exc()
				else:
					finished = True
			if not finished:
				await asubmit(self.load_guild_http, guild, priority=1)
		guild._member_count = len(guild._members)
		if "guilds" in self.data:
			self.data.guilds.register(guild)
		return guild.members

	def load_guild_http(self, guild):
		_members = {}
		x = 0
		i = 1000
		while i >= 1000:
			memberdata = []
			for r in range(64):
				try:
					with self.load_semaphore:
						memberdata = Request(
							f"https://discord.com/api/{api}/guilds/{guild.id}/members?limit=1000&after={x}",
							authorise=True,
							json=True,
							timeout=32,
						)
				except Exception as ex:
					if isinstance(ex, ConnectionError) and T(ex).get("errno") in (401, 403, 404):
						break
					print_exc()
					time.sleep(r + 2)
				else:
					break
			else:
				raise RuntimeError("Max retries exceeded in loading guild members via http.")
			members = {int(m["user"]["id"]): discord.Member(guild=guild, data=m, state=self._connection) for m in memberdata}
			guild._members.update(members)
			_members.update(members)
			i = len(memberdata)
			x = max(members) if members else 0
		guild._members = _members
		return guild

	@tracebacksuppressor
	async def load_guilds(self, guilds=None):
		guilds = guilds or self.client.guilds
		if "guilds" in self.data:
			for guild in guilds:
				with tracebacksuppressor:
					self.data.guilds.load_guild(guild)
		async def load_guilds_into():
			while self.maintenance:
				await asyncio.sleep(5)
			futs = deque()
			for guild in sorted(guilds, key=lambda guild: getattr(guild, "_member_count", None) or len(guild._members), reverse=True):
				fut = csubmit(asyncio.wait_for(self.load_guild(guild), timeout=3600))
				futs.append(fut)
				fut.guild = guild
				await asyncio.sleep(0.05)
			for fut in futs:
				try:
					await fut
				except (T0, T1, T2, CE):
					print("Error loading", fut.guild)
					print_exc()
					await self.load_guild(fut.guild)
		if not self.maintenance:
			await load_guilds_into()
		else:
			csubmit(load_guilds_into())
		self.users_updated = True
		print(len(guilds), "guilds loaded.")

	inter_cache = AutoCache(stale=0, timeout=900)
	async def defer_interaction(self, message, ephemeral=False, mode="post"):
		with tracebacksuppressor:
			if hasattr(message, "int_id"):
				int_id, int_token = message.int_id, message.int_token
			elif hasattr(message, "slash"):
				int_id, int_token = message.id, message.slash
			else:
				return
			data = await Request(
				f"https://discord.com/api/{api}/interactions/{int_id}/{int_token}/callback",
				method="POST",
				authorise=True,
				data='{"type":5,"data":{"flags":64}}' if ephemeral else '{"type":5}' if mode == "post" else '{"type":6}',
				aio=True,
			)
			print("Deferred:", message.id, data)
			self.inter_cache[int_id] = int_token
			self.inter_cache[message.id] = int_token
			message.deferred = int_token
			message.method = mode
			if self.cache.messages.get(message.id):
				self.cache.messages[message.id].deferred = int_token
			else:
				self.cache.messages[message.id] = message
			return message

	async def ignore_interaction(self, message, skip=False):
		with tracebacksuppressor:
			if getattr(message, "method", None) == "patch":
				return
			if hasattr(message, "int_id"):
				int_id, int_token = message.int_id, message.int_token
				if getattr(message, "deferred", None):
					int_token = message.deferred
					self.inter_cache[int_id] = int_token
				self.inter_cache[message.id] = int_token
			elif hasattr(message, "slash"):
				int_id, int_token = message.id, message.slash
			else:
				return
			m = None
			try:
				if skip:
					raise ConnectionError(400)
				m = await Request(
					f"https://discord.com/api/{api}/interactions/{int_id}/{int_token}/callback",
					method="POST",
					authorise=True,
					data='{"type":6}',
					aio=True,
				)
			except ConnectionError:
				m = await interaction_response(self, message, "\xad")
				# m = await send_with_reply(getattr(message, "channel", None), message, "\xad", ephemeral=False)
				print("Ignored:", m)
				if m and not getattr(m, "ephemeral", False):
					await self.silent_delete(m)
			else:
				message.method = "patch"

	def add_webhook(self, w):
		"Inserts a webhook into the bot's user and webhook cache."
		return self.data.webhooks.add(w)

	def load_channel_webhooks(self, channel, force=False, bypass=False):
		"Loads all webhooks in the target channel."
		return self.data.webhooks.get(channel, force=force, bypass=bypass)

	avatar_data = None
	async def ensure_webhook(self, channel, force=False, bypass=False, fill=False):
		"Gets a valid webhook for the target channel, creating a new one when necessary."
		wlist = await self.load_channel_webhooks(channel, force=force, bypass=bypass)
		data = self.avatar_data
		try:
			if fill:
				while len(wlist) < fill:
					# data = self.avatar_data = data or await self.optimise_image(get_author(self.user).url, 1048576)
					w = await channel.create_webhook(name=self.name, avatar=None, reason="Auto Webhook")
					w = self.add_webhook(w)
					wlist.append(w)
			if not wlist:
				# data = self.avatar_data = data or await self.optimise_image(get_author(self.user).url, 1048576)
				w = await channel.create_webhook(name=self.name, avatar=None, reason="Auto Webhook")
				w = self.add_webhook(w)
			else:
				wlist.sort(key=lambda w: (getattr(w, "user", None) != self.user, random.random()), reverse=True)
				w = wlist[0]
		except discord.HTTPException as ex:
			if "maximum" in str(ex).lower():
				print_exc()
				wlist = await self.load_channel_webhooks(channel, force=True, bypass=bypass)
				wlist.sort(key=lambda w: (getattr(w, "user", None) != self.user, random.random()), reverse=True)
				w = wlist[0]
		# if not w.avatar or str(w.avatar) == "https://cdn.discordapp.com/embed/avatars/0.png":
		# 	data = self.avatar_data = data or await self.optimise_image(get_author(self.user).url, 1048576, fmt="webp", anim=False)
		# 	return await w.edit(name=self.name, avatar=data)
		return w

	async def send_as_webhook(self, channel, *args, recurse=True, **kwargs):
		"Sends a message to the target channel, using a random webhook from that channel."
		if recurse and "exec" in self.data:
			try:
				avatar_url = kwargs.pop("avatar_url")
			except KeyError:
				pass
			else:
				with tracebacksuppressor:
					kwargs["avatar_url"] = await self.data.exec.uproxy(avatar_url)
		reacts = kwargs.pop("reacts", None)
		if hasattr(channel, "simulated") or not getattr(channel, "guild", None) or hasattr(channel, "recipient") or not hasattr(channel, "send"):
			kwargs.pop("username", None)
			kwargs.pop("avatar_url", None)
			message = await discord.abc.Messageable.send(channel, *args, **kwargs)
		else:
			if args and args[0] and args[0].count(":") >= 2 and channel.guild.me.guild_permissions.manage_roles:
				everyone = channel.guild.default_role
				permissions = everyone.permissions
				if not permissions.use_external_emojis:
					permissions.use_external_emojis = True
					await everyone.edit(permissions=permissions, reason="I need to send emojis 🙃")
			mchannel = None
			for i in range(25):
				mchannel = channel.parent if hasattr(channel, "thread") or isinstance(channel, discord.Thread) else channel
				if not mchannel:
					await asyncio.sleep(0.2)
				else:
					break
			w = await self.ensure_webhook(mchannel, bypass=True)
			kwargs.pop("wait", None)
			args = list(args)
			content = args.pop(0) if args else kwargs.get("content")
			if kwargs.get("reference"):
				content = fake_reply(kwargs.pop("reference")) + "\n" + (content or "")
				content = content.strip()
			else:
				kwargs.pop("reference", None)
			try:
				async with getattr(w, "semaphore", emptyctx):
					w = getattr(w, "webhook", w)
					if hasattr(channel, "thread") or isinstance(channel, discord.Thread):
						if kwargs.get("files"):
							kwargs.pop("avatar_url", None)
							kwargs.pop("username", None)
							message = await channel.send(*args, **kwargs)
						else:
							username = kwargs.get("username")
							if not username and getattr(channel, "guild", None):
								username = guild.me.display_name
							data = dict(
								content=content,
								username=username,
								avatar_url=kwargs.get("avatar_url"),
								tts=kwargs.get("tts"),
								embeds=[emb.to_dict() for emb in kwargs.get("embeds", ())] or ([kwargs["embed"].to_dict()] if kwargs.get("embed") is not None else None),
							)
							try:
								resp = await Request(
									f"https://discord.com/api/{api}/webhooks/{w.id}/{w.token}?wait=True&thread_id={channel.id}",
									method="POST",
									authorise=True,
									data=data,
									aio=True,
									json=True,
								)
							except:
								print("Errored:", kwargs, data)
								raise
							message = self.ExtendedMessage.new(resp)
					else:
						message = await w.send(content, *args, wait=True, **kwargs)
			except (discord.NotFound, discord.Forbidden):
				w = await self.ensure_webhook(mchannel, force=True)
				async with getattr(w, "semaphore", emptyctx):
					w = getattr(w, "webhook", w)
					if hasattr(channel, "thread"):
						resp = await Request(
							f"https://discord.com/api/{api}/webhooks/{w.id}/{w.token}?wait=True&thread_id={channel.id}",
							method="POST",
							authorise=True,
							data=data,
							aio=True,
							json=True,
						)
						message = self.ExtendedMessage.new(resp)
					else:
						message = await w.send(content, *args, wait=True, **kwargs)
			except discord.HTTPException as ex:
				if "400 Bad Request" in repr(ex):
					if "embeds" in kwargs:
						print(sum(len(e) for e in kwargs["embeds"]))
						for embed in kwargs["embeds"]:
							print(embed.to_dict())
					print(args, kwargs)
				raise
			await self.seen(self.user, channel.guild, event="message", count=len(kwargs.get("embeds", (None,))), raw="Sending a message")
		if reacts:
			await add_reacts(message, reacts)
		return message

	async def _send_embeds(self, sendable, embeds, reacts=None, reference=None, force=True, exc=True):
		"Sends a list of embeds to the target sendable, using a webhook when possible."
		s_id = verify_id(sendable)
		sendable = await self.fetch_messageable(s_id)
		if exc:
			ctx = self.ExceptionSender(sendable, reference=reference)
		else:
			ctx = tracebacksuppressor
		with ctx:
			if not embeds:
				return
			guild = getattr(sendable, "guild", None)
			# Determine whether to send embeds individually or as blocks of up to 10, based on whether it is possible to use webhooks
			if not guild:
				return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
			single = len(embeds) == 1
			if not hasattr(guild, "simulated") and hasattr(guild, "ghost"):
				single = True
			else:
				m = guild.me
				if not hasattr(guild, "simulated"):
					if m is None:
						m = self.user
						single = True
					else:
						with suppress(AttributeError):
							if not m.guild_permissions.manage_webhooks:
								return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
			if single:
				futs = []
				for emb in embeds:
					async with Delay(1 / 3):
						if type(emb) is not discord.Embed:
							r2 = emb.pop("reacts", None)
							if not reacts:
								reacts = astype(r2, list) if r2 else reacts
							elif r2:
								reacts.extend(r2)
							r2 = emb.pop("reference", None)
							if not reference:
								reference = r2
							emb = discord.Embed.from_dict(emb)
						fut = csubmit(send_with_react(sendable, embed=emb, reacts=reacts, reference=reference))
						futs.append(fut)
				return await gather(*futs)
			if force:
				return await send_with_react(sendable, embeds=embeds, reacts=reacts, reference=reference)
			embs = deque()
			for emb in embeds:
				if type(emb) is not discord.Embed:
					r2 = emb.pop("reacts", None)
					if not reacts:
						reacts = astype(r2, list) if r2 else reacts
					elif r2:
						reacts.extend(r2)
					r2 = emb.pop("reference", None)
					if not reference:
						reference = r2
					emb = discord.Embed.from_dict(emb)
				if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
					url = await self.get_proxy_url(m)
					await self.send_as_webhook(sendable, embeds=embs, username=m.display_name, avatar_url=url, reacts=reacts, reference=reference)
					embs.clear()
				embs.append(emb)
				reacts = None
			if embs:
				url = await self.get_proxy_url(m)
				await self.send_as_webhook(sendable, embeds=embs, username=m.display_name, avatar_url=url, reacts=reacts, reference=reference)

	def send_embeds(self, channel, embeds=None, embed=None, reacts=None, reference=None, exc=True, bottleneck=False):
		"Adds embeds to the embed sender, waiting for the next update event."
		fut = fut_nop
		if embeds is not None and not issubclass(type(embeds), collections.abc.Collection):
			embeds = (embeds,)
		elif embeds:
			embeds = tuple(embeds)
		if embed is not None:
			if embeds:
				embeds += (embed,)
			else:
				embeds = (embed,)
		elif not embeds:
			return fut_nop
		if reference:
			# print(reference, reference.__dict__)
			if getattr(reference, "slash", False):
				embs, embeds = embeds[:10], embeds[10:]
				print("Sending embeds directly due to interaction token")
				fut = csubmit(self._send_embeds(channel, embs, reacts, reference, exc=exc))
				if not embeds:
					return fut
			csubmit(self.ignore_interaction(reference, skip=True))
		c_id = verify_id(channel)
		user = self.cache.users.get(c_id)
		if user is not None:
			print("Sending embeds directly due to specified user")
			return csubmit(self._send_embeds(user, embeds, reacts, reference, exc=exc))
		if not self.initialisation_complete:
			embs, embeds = embeds[:10], embeds[10:]
			print("Sending embeds directly due to incomplete initialisation")
			fut = csubmit(self._send_embeds(channel, embs, reacts, reference, exc=exc))
			if not embeds:
				return fut
		if reacts or reference:
			embeds = [e.to_dict() for e in embeds]
			for e in embeds:
				e["reacts"] = reacts
				e["reference"] = reference
		embs = set_dict(self.embed_senders, c_id, [])
		lim = 16 if bottleneck else 2048
		if len(embs) >= lim:
			return fut
		embs.extend(embeds)
		if len(embs) > 2048:
			self.embed_senders[c_id] = embs[-2048:]
		return fut

	def send_as_embeds(self, channel=None, description=None, title=None, fields=None, md=nofunc, author=None, footer=None, thumbnail=None, image=None, images=None, colour=None, reacts=None, reference=None, exc=True, bottleneck=False):
		if type(description) is discord.Embed:
			emb = description
			description = emb.description or None
			title = emb.title or None
			fields = emb.fields or None
			author = emb.author or None
			footer = emb.footer or None
			thumbnail = emb.thumbnail or None
			image = emb.image or None
			if emb.colour:
				colour = colorsys.rgb_to_hsv(*(alist(raw2colour(verify_id(emb.colour))) / 255))[0] * 1536
		if description and not isinstance(description, str):
			description = as_str(description)
		elif not description:
			description = None
		if not description and not fields and not thumbnail and not image and not images:
			return fut_nop
		if not channel:
			if not reference:
				raise ValueError("Channel not specified.")
			channel = reference.channel
		return csubmit(self._send_as_embeds(channel, description, title, fields, md, author, footer, thumbnail, image, images, colour, reacts, reference, exc=exc, bottleneck=bottleneck))

	async def _send_as_embeds(self, channel, description=None, title=None, fields=None, md=nofunc, author=None, footer=None, thumbnail=None, image=None, images=None, colour=None, reacts=None, reference=None, exc=True, bottleneck=False):
		fin_col = col = None
		if colour is None:
			if author:
				try:
					url = author.icon_url
				except AttributeError:
					url = author.get("icon_url")
				if url:
					with suppress():
						fin_col = await self.data.colours.get(url)
		if fin_col is None:
			if type(colour) is discord.Colour:
				fin_col = colour
			elif colour is None:
				try:
					colour = self.cache.colours[channel.id]
				except KeyError:
					colour = xrand(12)
				self.cache.colours[channel.id] = (colour + 1) % 12
				fin_col = colour2raw(hue2colour(colour * 1536 / 12))
			else:
				col = colour if not issubclass(type(colour), collections.abc.Sequence) else colour[0]
				# off = 128 if not issubclass(type(colour), collections.abc.Sequence) else colour[1]
				fin_col = colour2raw(hue2colour(col))
		embs = deque()
		emb = discord.Embed(colour=fin_col)
		if title:
			emb.title = title
		if author:
			try:
				emb.set_author(**author)
			except TypeError:
				emb.set_author(name=author.name, url=author.url, icon_url=author.icon_url)
		if description:
			# Separate text into paragraphs, then lines, then words, then characters and attempt to add them one at a time, adding extra embeds when necessary
			paragraphs = split_text(description, max_length=4080)
			for para in paragraphs[:-1]:
				emb.description = md(para)
				embs.append(emb)
				emb = discord.Embed(colour=fin_col)
				if col is not None:
					col += 128
					emb.colour = colour2raw(hue2colour(col))
			if paragraphs:
				para = paragraphs[-1]
				emb.description = md(para)
		if fields:
			if issubclass(type(fields), collections.abc.Mapping):
				fields = fields.items()
			for field in fields:
				if issubclass(type(field), collections.abc.Mapping):
					field = tuple(field.values())
				elif not issubclass(type(field), collections.abc.Sequence):
					try:
						field = tuple(field)
					except TypeError:
						field = (field.name, field.value, getattr(field, "inline", None))
				n = lim_str(field[0], 256)
				v = lim_str(md(field[1]), 1024)
				i = True if len(field) < 3 else bool(field[2])
				if len(emb) + len(n) + len(v) > 6000 or len(T(emb).get("_fields", ())) > 24:
					embs.append(emb)
					emb = discord.Embed(colour=fin_col)
					if col is not None:
						col += 128
						emb.colour = colour2raw(hue2colour(col))
					else:
						try:
							colour = self.cache.colours[channel.id]
						except KeyError:
							colour = xrand(12)
						self.cache.colours[channel.id] = (colour + 1) % 12
						emb.colour = colour2raw(hue2colour(colour * 1536 / 12))
				emb.add_field(name=n, value=v if v else "\u200b", inline=i)
		if len(emb):
			embs.append(emb)
		if thumbnail:
			if not isinstance(thumbnail, str):
				thumbnail = thumbnail.url
			embs[0].set_thumbnail(url=thumbnail)
		if footer and embs:
			embs[-1].set_footer(**footer)
		if image:
			if not isinstance(image, str):
				image = image.url
			if images:
				images = deque(images)
				images.appendleft(image)
			else:
				images = (image,)
		if images:
			for i, img in enumerate(images):
				if is_video(img):
					csubmit(channel.send(escape_roles(img)))
				else:
					if i >= len(embs):
						emb = discord.Embed(colour=fin_col)
						if col is not None:
							col += 128
							emb.colour = colour2raw(hue2colour(col))
						else:
							try:
								colour = self.cache.colours[channel.id]
							except KeyError:
								colour = xrand(12)
							self.cache.colours[channel.id] = (colour + 1) % 12
							emb.colour = colour2raw(hue2colour(colour * 1536 / 12))
						embs.append(emb)
					embs[i].set_image(url=img)
					embs[i].url = img
		return await self.send_embeds(channel, embeds=embs, reacts=reacts, reference=reference, exc=exc, bottleneck=bottleneck)

	def update_embeds(self, force=False):
		"Updates all embed senders."
		sent = False
		for s_id in self.embed_senders:
			embeds = self.embed_senders[s_id]
			if not force and len(embeds) <= 10 and sum(len(e) for e in embeds) <= 6000:
				continue
			reacts = []
			reference = None
			embs = deque()
			for emb in embeds:
				if type(emb) is not discord.Embed:
					r2 = emb.pop("reacts", None)
					if not reacts:
						reacts = astype(r2, list) if r2 else reacts
					elif r2:
						reacts.extend(r2)
					r2 = emb.pop("reference", None)
					if not reference:
						reference = r2
					elif r2 and r2.id != reference.id:
						break
					emb = discord.Embed.from_dict(emb)
				# Send embeds in groups of up to 10, up to 6000 characters
				if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
					break
				embs.append(emb)
			# Left over embeds are placed back in embed sender
			self.embed_senders[s_id] = embeds = embeds[len(embs):]
			if not embeds:
				self.embed_senders.pop(s_id)
			csubmit(self._send_embeds(s_id, embs, force=force, exc=False, reacts=reacts, reference=reference))
			sent = True
		return sent

	def fast_loop(self):
		"The fast update loop that runs almost 24 times per second. Used for events where timing is important."
		fps = 24
		delay = 0.51
		freq = fps * delay

		def event_call():
			f = round_random(freq)
			for i in range(f):
				with Delay(delay / f):
					await_fut(self.send_event("_call_"))

		sent = 0
		while not self.closed:
			while self.api_latency > 6:
				time.sleep(1)
			with tracebacksuppressor:
				sent = self.update_embeds(utc() % 1 < 0.5)
				if sent:
					event_call()
				else:
					with Delay(delay / freq):
						await_fut(self.send_event("_call_"))
				self.update_users()

	uptime_db = diskcache.Cache(f"{CACHE_PATH}/uptime", expiry=86400 * 7)
	def update_uptime(self, data):
		uptimes = self.uptime_db
		ninter = self.ninter
		it = int(utc() // ninter) * ninter
		interval = 86400 * 7
		if it not in uptimes:
			uptimes[it] = copy.deepcopy(data)
			if min(uptimes) <= it - interval - 3600:
				sl = sorted(uptimes)
				while sl[0] <= it - interval:
					uptimes.pop(sl.pop(0), None)
				while sl[-1] > it:
					uptimes.pop(sl.pop(-1), None)
				skipto = 0
				for i in sl[:-3600 // ninter]:
					if i * ninter % 3600 == 0:
						continue
					if skipto >= i:
						continue
					if i * ninter % 3600 == ninter and uptimes.get(i * ninter + 3600 - ninter * 2) == {}:
						skipto = i * ninter + 3600 - ninter * 2
					if uptimes[i]:
						uptimes[i] = {}
		uptimea = np.array(list(uptimes.iterkeys()))
		if len(uptimea):
			uptimea.sort(kind="stable")
		i = np.searchsorted(uptimea, it - interval + ninter)
		j = np.searchsorted(uptimea, it + ninter)
		ut = j - i
		self.uptime = ut / interval * ninter
		return self.uptime_db

	uptime = 0
	up_bps = down_bps = 0
	total_bytes = 0
	ninter = 3
	slsem = Semaphore(1, 1, rate_limit=ninter, sync=True)
	async def slow_loop(self):
		"The slow update loop that runs once every 3 seconds."
		await asyncio.sleep(2)
		ninter = self.ninter
		while not self.closed:
			async with self.slsem:
				with tracebacksuppressor:
					csubmit(self.update_status())
					with MemoryTimer("network_usage"):
						fut = asubmit(psutil.net_io_counters)
						data = await self.status()
						await asubmit(self.update_uptime, data)
						resp = await fut
						sent, recv = resp.bytes_sent, resp.bytes_recv
						if not hasattr(self, "up_bytes"):
							self.up_bytes = deque(maxlen=ninter)
							self.down_bytes = deque(maxlen=ninter)
							self.start_up = max(0, self.data.insights.get("up_bytes", 0) - sent)
							self.start_down = max(0, self.data.insights.get("down_bytes", 0) - recv)
						self.up_bytes.append(sent)
						self.down_bytes.append(recv)
						self.up_bps = (self.up_bytes[-1] - self.up_bytes[0]) * 8 / len(self.up_bytes) / ninter
						self.down_bps = (self.down_bytes[-1] - self.down_bytes[0]) * 8 / len(self.down_bytes) / ninter
						self.bitrate = self.up_bps + self.down_bps
						self.data.insights["up_bytes"] = up_bytes = self.up_bytes[-1] + self.start_up
						self.data.insights["down_bytes"] = down_bytes = self.down_bytes[-1] + self.start_down
						self.total_bytes = up_bytes + down_bytes

	async def lazy_loop(self):
		"The lazy update loop that runs once every 3~11 seconds."
		await asyncio.sleep(5)
		while not self.closed:
			async with Delay(random.random() * 8 + 3):
				async with tracebacksuppressor:
					# self.var_count = await asubmit(var_count)
					with MemoryTimer("handle_update"):
						await self.handle_update()

	@tracebacksuppressor
	async def worker_heartbeat(self):
		futs = []
		key = AUTH.get("discord_secret") or ""
		# uri = f"http://IP:{PORT}"
		uri = "https://api.mizabot.xyz"
		dc = pk = ""
		if DOMAIN_CERT and PRIVATE_KEY:
			async with aiofiles.open(DOMAIN_CERT, "r") as f:
				dc = await f.read()
			async with aiofiles.open(PRIVATE_KEY, "r") as f:
				pk = await f.read()
		for addr in AUTH.get("remote_servers", ()):
			token = AUTH.get("alt_token") or self.token
			channels = [k for k, v in self.data.exec.items() if v & 16]
			data = orjson.dumps(dict(
				domain_cert=dc,
				private_key=pk,
				channels=channels,
				encryption_key=AUTH["encryption_key"],
				token=self.token,
				alt_token=AUTH.get("alt_token") or self.token,
			))
			encoded = base64.b64encode(encrypt(data)).rstrip(b"=").decode("ascii")
			async def external_heartbeat():
				if not Request.sessions:
					return
				return await Request(
					f"https://{addr}/authorised-heartbeat?key={quote_plus(key)}&uri={quote_plus(uri)}",
					method="POST",
					headers={"content-type": "application/json"},
					data=orjson.dumps(dict(data=encoded)),
					aio=True,
					timeout=5,
					session=Request.sessions.next(),
				)
			fut = csubmit(external_heartbeat())
			futs.append(fut)
		await gather(*futs)
		return futs

	async def global_loop(self):
		"The slowest update loop that runs once every 5 minutes. Used for slow operations, such as the bot database autosave event."
		while not self.closed:
			async with Delay(300):
				with tracebacksuppressor:
					await self.worker_heartbeat()
					await asyncio.sleep(1)
					with MemoryTimer("update_file_cache"):
						await asubmit(update_file_cache)
					await asyncio.sleep(1)
					# with MemoryTimer("get_disk"):
					#     await self.get_disk()
					with MemoryTimer("get_hosted"):
						await self.get_hosted()
					await asyncio.sleep(1)
					with MemoryTimer("update_subs"):
						await asubmit(self.update_subs, priority=True)
					await asyncio.sleep(1)
					await self.send_event("_minute_loop_")
					esubmit(self.cache_reduce, priority=True)
					with MemoryTimer("update"):
						await asubmit(self.update, priority=True)

	@tracebacksuppressor
	async def heartbeat(self):
		await asyncio.sleep(0.5)
		if self.closed:
			return
		pathlib.Path.touch(self.heartbeat_file)

	def heartbeat_loop(self):
		"Heartbeat loop: Repeatedly renames a file to inform the watchdog process that the bot's event loop is still running."
		print("Heartbeat Loop initiated.")
		fut = None
		lt = utc()
		seen = set()
		while not self.closed:
			with Delay(8):
				with tracebacksuppressor:
					t = utc()
					# at = asyncio.all_tasks(eloop)
					# sat = set(at)
					# sat.difference_update(seen)
					# seen.update(sat)
					# print(t, t - lt, len(seen), len(at), len(sat), sat)
					if not fut or fut.done():
						fut = csubmit(self.heartbeat())
						lt = t
					if t - lt > 30:
						cf = sys._current_frames()
						cf = sorted(cf.items())
						cf += list(asyncio.all_tasks(eloop))
						out = Flush(sys.__stdout__)
						out.write(f"STACK TRACE:\n{cf}\n\n\n")
						out.write(f"CRASHED AT {t}: {t - self.start_time}: {t - lt}\n")
						pdb.set_trace()
						break

	async def seen(self, *args, delay=0, event=None, **kwargs):
		"User seen event"
		for arg in args:
			if arg:
				await self.send_event("_seen_", user=arg, delay=delay, event=event, **kwargs)

	async def ensure_reactions(self, message):
		if not message.reactions or isinstance(message, self.CachedMessage | self.LoadedMessage) or isinstance(message.author, cdict | self.GhostUser):
			if self.permissions_in(message.channel).read_message_history:
				try:
					message = await discord.abc.Messageable.fetch_message(message.channel, message.id)
				except (LookupError, discord.NotFound):
					pass
				else:
					self.add_message(message, files=False, force=True)
		return message

	async def react_with(self, message, name):
		react = await self.data.emojis.grab(name)
		if isinstance(react, dict):
			react = f"{react['name']}:{react['id']}"
		return await message.add_reaction(react)

	async def check_to_delete(self, message, reaction, user):
		"Deletes own messages if any of the \"X\" emojis are reacted by a user with delete message permission level. Spoilers own messages if any of the square button emojis are reacted by a user with spoiler message permission level. Messages replying to a user automatically grants them permission to perform these actions."
		if user.id == self.id:
			return
		if str(reaction) not in "❌✖️🇽❎🔳🔲":
			return
		if str(reaction) in "🔳🔲" and (not message.attachments and not message.embeds or "exec" not in self.data):
			return
		if message.author.id == self.id or getattr(message, "webhook_id", None):
			pass
		else:
			return
		print(message, reaction, user)
		with tracebacksuppressor(discord.NotFound):
			u_perm = self.get_perms(user.id, message.guild)
			check = False
			if not u_perm < 3:
				check = True
			else:
				# Handle special case where a message may be part of a set of StreamedMessage instances, where only the first will actually have the reference
				message2 = message
				async for temp in self.history(message.channel, before=message, limit=10):
					if temp.author.id != self.id:
						break
					if temp.reference:
						message2 = temp
				try:
					reference = await self.fetch_reference(message2)
				except (LookupError, discord.NotFound):
					for react in message.reactions:
						if str(reaction) == str(react) and react.me:
							check = True
							break
				else:
					if reference.author.id == user.id:
						check = True
			if not check:
				return
			if str(reaction) in "🔳🔲":
				if message.content.startswith("||"):
					content = message.content.replace("||", "").strip()
				else:
					def temp_url(url, mid=None):
						if is_discord_attachment(url):
							a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
							if a_id in self.data.attachments:
								return self.preserve_attachment(a_id, fn=url)
							channel = message.channel
							return self.preserve_as_long(channel.id, message.id, a_id, fn=url)
						return url
					futs = deque()
					for a in message.attachments:
						futs.append(csubmit(self.get_request(a.url)))
					urls = set()
					for e in message.embeds:
						if e.image:
							urls.add(temp_url(e.image.url))
						if e.thumbnail:
							urls.add(temp_url(e.thumbnail.url))
					symrem = "".maketrans({c: "" for c in "<>|*"})
					spl = [word.translate(symrem) for word in message.content.split() if not word.startswith("<")]
					content = " ".join(word for word in spl if word and not is_url(word))
					urls.update(word for word in spl if is_url(word))
					if futs:
						datas = await gather(*futs)
						fut = csubmit(self.edit_message(message, content="`LOADING...`", attachments=(), embeds=()))
						urli = await self.data.exec.uproxy(*(CompatFile(d, filename=url2fn(a.url)) for d, a in zip(datas, message.attachments)), collapse=False, keep=False)
						urli.extend(urls)
						urls = urli
					if urls:
						content += "\n" + "\n".join(f"||{url} ||" for url in urls)
					if futs:
						await fut
				# before = copy.copy(message)
				message = await self.edit_message(message, content=content, attachments=(), embeds=())
				# await self.send_event("_edit_", before=before, after=message, force=True)
			else:
				await self.silent_delete(message, exc=True)
				await self.send_event("_delete_", message=message)

	async def handle_message(self, message, before=None):
		"Handles a new sent message, calls process_message and sends an error if an exception occurs."
		if message.author.id != self.user.id:
			for i, a in enumerate(message.attachments):
				if a.filename == "message.txt":
					b = await self.get_request(message.attachments.pop(i).url)
					if message.content:
						message.content += " "
					message.content += as_str(b)
		await self.process_message(message, before=before)

	def set_classes(self):
		bot = self

		class UserGuild(discord.Object):
			"For compatibility with guild objects, takes a user and DM channel."

			class UserChannel(discord.abc.PrivateChannel):

				def __init__(self, channel, **void):
					self.channel = channel
					self.id = channel.id if channel else 0

				def __dir__(self):
					data = set(object.__dir__(self))
					data.update(dir(self.channel))
					return data

				def __getattr__(self, key):
					try:
						return self.__getattribute__(key)
					except AttributeError:
						pass
					return getattr(self.__getattribute__("channel"), key)

				def fetch_message(self, id):
					return bot.fetch_message(id, self.channel)

				@property
				def me(self):
					return bot.user
				name = "DM"
				topic = None
				is_nsfw = lambda self: bot.is_nsfw(self.channel)
				is_news = lambda *self: False
				is_channel = True
				__str__ = lambda self: self.channel.recipient.display_name if self.channel.recipient else "channel"

			def __init__(self, user=None, channel=None, **void):
				self.channel = self.system_channel = self.rules_channel = self.UserChannel(channel) if channel else None
				self.members = [bot.user]
				if user:
					self.members.append(user)
				self._members = {m.id: m for m in self.members if m}
				self.channels = self.text_channels = [self.channel] if channel else []
				self.voice_channels = []
				self.roles = []
				self.emojis = []
				self.get_channel = lambda _id: self.channel
				self.owner_id = bot.user.id
				self.owner = bot.user
				self.fetch_member = bot.fetch_user
				self.get_member = self._members.get
				self.voice_client = None
				self.id = channel.id if channel else user.id if user else 0

			def __dir__(self):
				data = set(object.__dir__(self))
				data.update(dir(self.channel))
				return data

			def __getattr__(self, key):
				try:
					return self.__getattribute__(key)
				except AttributeError:
					pass
				return getattr(self.__getattribute__("channel"), key)

			@property
			def me(self):
				return bot.user
			@me.setter
			def me(self, value):
				return
			
			get_role = lambda *args: None
			filesize_limit = CACHE_FILESIZE
			bitrate_limit = 98304
			emoji_limit = 0
			large = False
			description = ""
			max_members = 2
			unavailable = False
			ghost = True
			is_channel = True
			is_nsfw = lambda self: bot.is_nsfw(self.channel)
			__str__ = lambda self: self.channel.recipient.display_name if self.channel.recipient else "channel"

		class GhostUser(discord.abc.Snowflake):
			"Represents a deleted/not found user."

			__repr__ = lambda self: f"<Ghost User id={self.id} name='{self.name}' discriminator='{self.discriminator}' bot=False>"
			__str__ = discord.user.BaseUser.__str__
			system = False
			history = lambda *void1, **void2: fut_nop
			dm_channel = None
			create_dm = lambda self: fut_nop
			relationship = None
			is_friend = lambda self: None
			is_blocked = lambda self: None
			is_migrated = lambda self: None
			colour = color = discord.Colour(16777215)
			_avatar = _avatar_decoration = None
			name = "[USER DATA NOT FOUND]"
			nick = None
			global_name = None
			discriminator = "0"
			id = 0
			guild = None
			mutual_guilds = []
			status = None
			voice = None
			display_avatar = avatar = "0"
			avatar_url = icon_url = url = bot.discord_icon
			joined_at = premium_since = None
			timed_out_until = None
			communication_disabled_until = None
			_primary_guild = None
			_avatar_decoration_data = None
			_client_status = client_status = _status = cdict({None: "offline", "_status": "offline", "desktop": "false", "mobile": "false", "web": "false"})
			pending = False
			ghost = True
			roles = ()
			_roles = ()
			activities = ()
			_activities = ()
			flags = _flags = public_flags = _public_flags = discord.flags.PublicUserFlags()
			banner = None
			_banner = None
			accent_colour = None
			_accent_colour = None
			_permissions = discord.Permissions(0)
			is_timed_out = lambda *args: False

			def __getattr__(self, k):
				if k == "member":
					return self.__getattribute__(k)
				elif hasattr(self, "member"):
					try:
						return getattr(self.member, k)
					except AttributeError:
						pass
				return self.__getattribute__(k)

			def send(self, *args, **kwargs):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.send(self, *args, **kwargs)

			def edit(self, *args, **kwargs):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.edit(self, *args, **kwargs)

			def add_roles(self, *args, **kwargs):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.add_roles(self, *args, **kwargs)

			def remove_roles(self, *args, **kwargs):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.remove_roles(self, *args, **kwargs)

			def kick(self, reason=None):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.kick(self, reason=reason)

			def ban(self, reason=None):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.ban(self, reason=reason)

			def timeout(self, duration, reason=None):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.timeout(self, duration, reason=reason)

			def move_to(self, duration, reason=None):
				if not getattr(self, "guild", None):
					raise AttributeError("Member is not in a guild.")
				return discord.Member.move_to(self, duration, reason=reason)

			@property
			def display_name(self):
				return self.nick or self.name

			@property
			def mention(self):
				return f"<@{self.id}>"

			@property
			def created_at(self):
				return snowflake_time_3(self.id)

			@property
			def _state(self):
				return bot._state

			@property
			def _user(self):
				return bot.cache.users.get(self.id) or self
			@_user.setter
			def _user(self, user):
				bot.cache.users[user.id] = user

			def _to_minimal_user_json(self):
				return cdict(
					username=self.name,
					id=self.id,
					avatar=self.avatar,
					discriminator=self.discriminator,
					bot=self.bot,
				)

			def _update(self, data):
				m = discord.Member._copy(self)
				m._update(data)
				self.member = m
				return m

		GhostUser.bot = False
		user = GhostUser()
		user.bot = True
		name = AUTH.get("name") or "Unknown User"
		user.name = name
		user.nick = name
		user.id = AUTH.get("discord_id", 0)
		bot._user = user

		class GhostMessage(discord.abc.Snowflake):
			"Represents a deleted/not found message."

			content = bold(css_md(uni_str("[MESSAGE DATA NOT FOUND]"), force=True))

			def __init__(self):
				self.author = bot.get_user(bot.deleted_user)
				self.channel = None
				self.guild = None
				self.id = 0

			async def delete(self, *void1, **void2):
				pass

			@property
			def _state(self):
				return bot._state

			__repr__ = lambda self: f"<GhostMessage id={self.id}>"
			tts = False
			type = "default"
			nonce = False
			embeds = ()
			call = None
			mention_everyone = False
			mentions = ()
			webhook_id = None
			attachments = ()
			pinned = False
			flags = None
			reactions = ()
			reference = None
			activity = None
			system_content = clean_content = ""
			edited_at = None
			is_system = lambda self: None
			slash = None

			@property
			def created_at(self):
				return snowflake_time_3(self.id)

			@property
			def jump_url(self):
				channel = getattr(self, "channel", None)
				guild = getattr(self, "guild", None) or getattr(channel, "guild", None)
				if not guild or not channel:
					return "https://discord.com/channels/-1/-1/-1"
				return f"https://discord.com/channels/{self.guild.id}/{self.channel.id}/{self.id}"

			edit = delete
			publish = delete
			pin = delete
			unpin = delete
			add_reaction = delete
			remove_reaction = delete
			clear_reaction = delete
			clear_reactions = delete
			ack = delete
			ghost = True
			deleted = True

		class ExtendedMessage:
			"discord.py-compatible message object that enables insertion of additional attributes."

			__slots__ = ("__weakref__", "__dict__")

			def __repr__(self):
				return "<" + self.__class__.__name__ + " object @ " + str(self.id) + ">"

			@classmethod
			def new(cls, data, channel=None, **void):
				if not channel:
					try:
						channel = bot.force_channel(data)
					except Exception:
						print_exc()
				message = discord.Message(channel=channel, data=copy.deepcopy(data), state=bot._state)
				self = cls(message)
				self._data = data
				return self

			async def edit(self, *args, **kwargs):
				if "file" in kwargs:
					kwargs["attachments"] = [kwargs.pop("file")]
				if "files" in kwargs:
					kwargs["attachments"] = kwargs.pop("files")
				if not self.webhook_id:
					try:
						if args:
							kwargs["content"] = " ".join(args)
						return await discord.Message.edit(self, **kwargs)
					except discord.HTTPException:
						print(self)
						print(args)
						print(kwargs)
						raise
				if self.webhook_id == bot.id:
					if args:
						kwargs["content"] = " ".join(args)
					return await discord.Message.edit(self, **kwargs)
				try:
					w = bot.cache.users[self.webhook_id]
					webhook = getattr(w, "webhook", w)
					if not getattr(webhook, "token", None):
						raise KeyError("Webhook token not found.")
				except KeyError:
					webhook = await bot.fetch_webhook(self.webhook_id)
					bot.data.webhooks.add(webhook)
				data = kwargs
				if args:
					data["content"] = " ".join(args)
				if "embed" in data:
					data["embeds"] = [data.pop("embed").to_dict()]
				elif "embeds" in data:
					data["embeds"] = [emb.to_dict() for emb in data["embeds"]]
				resp = await Request(
					f"https://discord.com/api/{api}/webhooks/{webhook.id}/{webhook.token}/messages/{self.id}",
					data=data,
					authorise=True,
					method="PATCH",
					aio=True,
					json=True,
				)
				return self.__class__.new(channel=self.channel, data=resp)

			def __init__(self, message):
				self.message = message
			
			def __copy__(self):
				return self.__class__(copy.copy(self.message))

			def __getattr__(self, k):
				if k == "channel":
					m = object.__getattribute__(self, "message")
					c = getattr(m, "channel")
					if isinstance(c, discord.abc.PrivateChannel):
						try:
							c = bot.cache.channels[c.id]
						except KeyError:
							pass
						else:
							m.channel = c
					return c
				elif k == "guild":
					m = object.__getattribute__(self, "message")
					g = getattr(m, "guild")
					if not g:
						c = self.channel
						m.guild = getattr(c, "guild", None)
					return m.guild
				try:
					return self.__getattribute__(k)
				except AttributeError:
					pass
				m = object.__getattribute__(self, "message")
				if not isinstance(m, self.__class__):
					v = getattr(m, k)
				else:
					v = object.__getattribute__(m, k)
				if v and k in ("content", "system_content", "clean_content"):
					return readstring(v)
				return v

		class StreamedMessage:
			"Semi discord.py-compatible message object that enables management of multiple messages in sequence."
			"Extensively used by LLM features."

			def __init__(self, channel=None, reference=None, msglen=2000, maxlen=10000, obfuscate=False):
				self.content = ""
				self.id = 0
				self.channel = channel
				self.guild = getattr(channel, "guild", None)
				self.me = self.author = channel.guild.me if getattr(channel, "guild", None) else bot.user
				self.created_at = to_utc(utc_dt())
				self.edited_at = None
				self.reference = reference
				self.msglen = msglen
				self.maxlen = maxlen
				self.obfuscate = obfuscate
				self.messages = []
				self.removed = set()
				self.embeds = []
				self.files = []
				self.attachments = []
				self.reacts = []
				self.buttons = []

			@classmethod
			async def attach(cls, message, replace=False, **kwargs):
				"Creates a StreamedMessage object from a discord.Message object."
				if isinstance(message, cls):
					message.msglen = kwargs.get("msglen", 2000)
					message.maxlen = kwargs.get("maxlen", 2000)
					message.obfuscate = kwargs.get("obfuscate", False)
					if replace:
						message.content = ""
						message.attachments = []
						message.embeds = []
						message.files = []
					return message
				channel = message.channel
				try:
					reference = await bot.fetch_reference(message)
				except (LookupError, discord.NotFound):
					reference = None
				self = cls(channel, reference=reference, **kwargs)
				self.messages.append(message)
				self.id = message.id
				self.author = message.author
				self.created_at = message.edited_at
				self.edited_at = message.edited_at
				if replace:
					self.content = ""
					self.attachments = []
					self.embeds = []
					self.files = []
				else:
					self.content = message.content
					self.attachments = message.attachments
					self.embeds = message.embeds
					if hasattr(message, "files"):
						self.files = message.files
					else:
						futs = [csubmit(bot.get_attachment(a.url, size=a.size)) for a in message.attachments]
						files = await gather(*futs)
						self.files = [CompatFile(b, filename=a.filename) for a, b in zip(message.attachments, files)]
				return self

			def update(self, content, embeds=[], files=[], buttons=[], prefix="", suffix="", bypass=((), ()), reacts=[], force=True, done=True):
				"Updates the StreamedMessage object with new content, embeds, files, and buttons. If force is True, the message will be sent even if it exceeds the maximum length, as a file."
				if not done and self.content and not content.strip() and not self.content.endswith(" <<<"):
					content = self.content + " <<<"
				content = readstring(content).strip()
				if not done and content == readstring(self.content):
					return as_fut(self)
				if self.obfuscate:
					content = obfuscate(content)
				size = len(prefix) + len(content) + len(suffix)
				if size > self.maxlen:
					if force:
						file = CompatFile(content.encode("utf-8"), filename="message.txt")
						files.append(file)
						content = ""
					else:
						raise OverflowError(size)
				self.content = content
				self.files = files
				self.embeds = embeds
				return self._update(content, embeds, files, buttons, prefix, suffix, bypass, reacts, force, done)

			async def _update(self, content, embeds, files, buttons, prefix, suffix, bypass, reacts, force, done):
				ms = split_text(content, max_length=self.msglen, prefix=prefix, suffix=suffix)
				futs = []
				while len(ms) < len(self.messages):
					fut = csubmit(bot.silent_delete(self.messages.pop(-1)))
					futs.append(fut)
				for i, s in enumerate(ms):
					left = (i - len(ms)) * 10
					right = (i + 1 - len(ms)) * 10 or None
					embs = embeds[left:right]
					fils = files[left:right]
					buts = buttons or [] if i == len(ms) - 1 else []
					ref = self.reference if not i else None
					reac = reacts if done else "💬" if i == len(ms) - 1 else None
					if i >= len(self.messages):
						await self.assert_last(assertion=not force)
						fut = csubmit(send_with_react(self.channel, s, reference=ref, embeds=embs, files=fils, buttons=buts, reacts=reac))
						futs.append(fut)
						self.messages.append(fut)
						await asyncio.sleep(0.25)
						continue
					m = self.messages[i]
					if isinstance(m, (asyncio.Future, asyncio.Task)):
						m = self.messages[i] = await m
					if not await bot.verify_integrity(m):
						self.messages.pop(i)
						continue
					if i == len(self.messages) - 1:
						if (not reac or "💬" not in reac) and m.id not in self.removed:
							fut = csubmit(m.remove_reaction("💬", self.me))
							self.removed.add(m.id)
							futs.append(fut)
					if s != m.content or embs or fils or buts:
						fut = csubmit(bot.edit_message(m, content=s, embeds=embs, attachments=fils, buttons=buts))
						futs.append(fut)
						self.messages[i] = fut
				for fut in futs:
					try:
						await fut
					except discord.errors.NotFound:
						pass
				if self.messages:
					m = self.messages[-1]
					if isinstance(m, (asyncio.Future, asyncio.Task)):
						m = self.messages[-1] = await m
					self.id = m.id
					self.author = m.author
					self.attachments = m.attachments
					self.edited_at = m.created_at or m.edited_at
				return self

			async def assert_last(self, assertion=True):
				"Asserts that the last message in the StreamedMessage object is the same as the last message in the channel. If assertion is True, an error will be raised if the last message is not the same."
				if not self.messages:
					return
				await self.collect()
				if assertion:
					async for m in bot.history(self.channel, limit=ceil(self.maxlen / self.msglen)):
						if m.id == self.messages[0].id:
							break
						if m.id not in (m2.id for m2 in self.messages):
							raise InterruptedError(m.id)

			async def delete(self):
				messages = await self.collect()
				csubmit(bot.silent_delete(messages))
				self.messages.clear()
				self.removed.clear()

			async def collect(self):
				messages = []
				for i, m in enumerate(self.messages):
					if isinstance(m, asyncio.Future):
						m = self.messages[i] = await m
					messages.append(m)
				return messages

		class LoadedMessage(discord.Message):
			"discord.py-compatible message object that enables dynamic creation."

			def __getattr__(self, k):
				if k not in ("mentions", "role_mentions"):
					return super().__getattribute__(k)
				try:
					return super().__getattribute__(k)
				except AttributeError:
					return []

		class CachedMessage(discord.abc.Snowflake):
			"discord.py-compatible message object that enables fast loading."

			__slots__ = ("_data", "id", "created_at", "author", "channel", "guild", "channel_id", "deleted", "attachments", "sem", "cached")

			def __init__(self, data):
				self._data = data
				self.id = int(data["id"])
				self.created_at = snowflake_time_3(self.id)
				author = data["author"]
				if author["id"] not in bot.cache.users:
					bot.user2cache(author)

			def __copy__(self):
				d = dict(self.__getattribute__("_data"))
				channel = self.channel
				if "channel_id" not in d:
					d["channel_id"] = channel.id
				author = self.author
				if "tts" not in d:
					d["tts"] = False
				if "message_reference" in d:
					with tracebacksuppressor:
						ref = d["message_reference"]
						if "channel_id" not in ref:
							ref["channel_id"] = d["channel_id"]
						if "guild_id" not in ref:
							if getattr(channel, "guild", None):
								ref["guild_id"] = channel.guild.id
						if d.get("referenced_message"):
							m = d["referenced_message"]
							m["channel_id"] = d["channel_id"]
							if "message_reference" in m:
								m["message_reference"]["channel_id"] = d["channel_id"]
				for k in ("tts", "pinned", "mention_everyone"):
					if k not in d:
						d[k] = None
				if "edited_timestamp" not in d:
					d["edited_timestamp"] = ""
				if "type" not in d:
					d["type"] = 0
				if "content" not in d:
					d["content"] = ""
				for k in ("reactions", "attachments", "embeds"):
					if k not in d:
						d[k] = []
				if d["reactions"]:
					for r in d["reactions"]:
						r.setdefault("me", False)
				try:
					message = bot.LoadedMessage(state=bot._state, channel=channel, data=d)
				except:
					print(d)
					raise
				if not getattr(message, "author", None):
					message.author = author
				return message

			def __getattr__(self, k):
				if k in self.__slots__:
					try:
						return self.__getattribute__(k)
					except AttributeError:
						if k == "deleted":
							raise
				if k in ("simulated", "slash"):
					raise AttributeError(k)
				d = self.__getattribute__("_data")
				if k in ("content", "system_content", "clean_content"):
					return readstring(d.get(k) or d.get("content", ""))
				if k == "channel":
					try:
						channel, _ = bot._get_guild_channel(d)
						if channel is None:
							raise LookupError
					except LookupError:
						pass
					else:
						return channel
					cid = int(d["channel_id"])
					channel = bot.cache.channels.get(cid)
					if channel:
						self.channel = channel
						return channel
					else:
						return cdict(id=cid)
				if k == "guild":
					return getattr(self.channel, "guild", None)
				if k == "author":
					self.author = bot.get_user(d["author"]["id"], replace=True)
					guild = getattr(self.channel, "guild", None)
					if guild is not None:
						member = guild.get_member(self.author.id)
						if member is not None:
							self.author = member
					return self.author
				if k == "type":
					return discord.enums.try_enum(discord.MessageType, d.get("type", 0))
				if k == "attachments":
					self.attachments = attachments = [discord.Attachment(data=a, state=bot._state) for a in d.get("attachments", ())]
					return attachments
				if k == "embeds":
					return [discord.Embed.from_dict(a) for a in d.get("embeds", ())]
				if k == "system_content" and not d.get("type"):
					return self.content
				try:
					m = self.__getattribute__("cached")
				except AttributeError:
					m = bot.cache.messages.get(d["id"])
				if m is None or m is self or not isinstance(m, bot.LoadedMessage):
					message = self.cached = self.__copy__()
					if type(m) not in (discord.Message, bot.ExtendedMessage):
						bot.add_message(message, files=False, force=True)
					try:
						return getattr(message, k)
					except (AttributeError, KeyError):
						if k == "mentions":
							return ()
						raise
				try:
					return getattr(m, k)
				except AttributeError:
					if k == "mentions":
						return ()
					raise

		class MessageCache(collections.abc.Mapping):
			data = bot.cache.messages

			def __getitem__(self, k):
				with suppress(KeyError):
					return self.data[k]
				if "message_cache" in bot.data:
					return bot.data.message_cache.load_message(k)
				raise KeyError(k)
			__call__ = __getitem__

			def __setitem__(self, k, v):
				bot.add_message(v, force=True)

			__delitem__ = data.pop
			__bool__ = lambda self: bool(self.data)
			__iter__ = data.__iter__
			__reversed__ = data.__reversed__
			__len__ = data.__len__
			__str__ = lambda self: f"<MessageCache ({len(self)} items)>"
			__repr__ = data.__repr__

			def __contains__(self, k):
				with suppress(KeyError):
					self[k]
					return True
				return False

			get = data.get
			pop = data.pop
			popitem = data.popitem
			clear = data.clear

		class ExceptionSender(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
			"A context manager that sends exception tracebacks to a messageable."

			def __init__(self, sendable, *args, reference=None, **kwargs):
				self.sendable = sendable
				self.reference = reference
				self.exceptions = args + tuple(kwargs.values())

			def __exit__(self, exc_type, exc_value, exc_tb):
				if exc_type and exc_value:
					for exception in self.exceptions:
						if issubclass(type(exc_value), exception):
							bot.send_exception(self.sendable, exc_value, reference=self.reference)
							return True
					bot.send_exception(self.sendable, exc_value, reference=self.reference)
					with tracebacksuppressor:
						raise exc_value
				return True
			
			__aexit__ = lambda self, *args: as_fut(self.__exit__(*args))
			__call__ = lambda self, *args, **kwargs: self.__class__(*args, **kwargs)


		bot.UserGuild = UserGuild
		bot.GhostUser = GhostUser
		bot.GhostMessage = GhostMessage
		bot.ExtendedMessage = ExtendedMessage
		bot.StreamedMessage = StreamedMessage
		bot.LoadedMessage = LoadedMessage
		bot.CachedMessage = CachedMessage
		bot.MessageCache = MessageCache
		bot.ExceptionSender = ExceptionSender

	def monkey_patch(self):
		"Extends discord.py's features using our bot class, allowing access to caches/databases on disk and fixing several issues."
		bot = self

		discord.http.Route.BASE = f"https://discord.com/api/{api}"
		discord.Member.permissions_in = lambda self, channel: discord.Permissions.none() if not getattr(self, "roles", None) else discord.Permissions(fold(int.__or__, (role.permissions.value for role in self.roles))) if not getattr(channel, "permissions_for", None) else channel.permissions_for(self)
		discord.VoiceChannel._get_channel = lambda self: as_fut(self)
		discord.user.BaseUser.__str__ = lambda self: self.name if self.discriminator in (None, 0, "", "0") else f"{self.name}#{self.discriminator}"

		recv_message = discord.gateway.DiscordWebSocket.received_message
		async def received_message(self, msg, /):
			if isinstance(msg, byte_like):
				msg = self._decompressor.decompress(msg)
				if msg is None:
					return
			res = orjson.loads(msg)
			bot.socket_responses.append(res)
			self._dispatch("socket_response", res)
			return await recv_message(self, as_str(msg))
		discord.gateway.DiscordWebSocket.received_message = received_message

		def _get_guild_channel(self, data, guild_id=None):
			channel_id = int(data["channel_id"])
			try:
				channel = bot.cache.channels[channel_id]
			except KeyError:
				pass
			else:
				return channel, getattr(channel, "guild", None)
			try:
				guild = self._get_guild(int(guild_id or data["guild_id"]))
			except KeyError:
				channel = discord.DMChannel._from_message(self, channel_id)
				guild = None
			else:
				channel = guild and guild._resolve_channel(channel_id)
			return channel or discord.PartialMessageable(state=self, id=channel_id), guild
		discord.state.ConnectionState._get_guild_channel = _get_guild_channel

		# async def get_gateway(self, *, encoding="json", zlib=True):
		# 	try:
		# 		data = await self.request(discord.http.Route("GET", "/gateway"))
		# 	except discord.HTTPException as exc:
		# 		raise discord.GatewayNotFound() from exc
		# 	value = "{0}?encoding={1}&v=" + api[1:]
		# 	if zlib:
		# 		value += "&compress=zlib-stream"
		# 	return value.format(data["url"], encoding)
		# discord.http.HTTPClient.get_gateway = get_gateway

		async def history(self, limit=100, before=None, after=None, around=None, oldest_first=None):
			if not getattr(self, "channel", None):
				try:
					self.channel = await self.messageable._get_channel()
				except AttributeError:
					try:
						self.channel = await self._get_channel()
					except AttributeError:
						try:
							self.channel = self
						except AttributeError:
							pass
			channel = getattr(self, "channel", self)

			async def _around_strategy(retrieve, around=None, limit=None):
				if not around:
					return [], None, 0

				around_id = around.id if around else None
				data = await self._state.http.logs_from(channel.id, retrieve, around=around_id)

				return data, None, 0

			async def _after_strategy(retrieve, after=None, limit=None):
				after_id = after.id if after else None
				data = await self._state.http.logs_from(channel.id, retrieve, after=after_id)

				if data:
					if limit is not None:
						limit -= len(data)

					after = cdict(id=int(data[0]['id']))

				return data, after, limit

			async def _before_strategy(retrieve, before=None, limit=None):
				before_id = before.id if before else None
				data = await self._state.http.logs_from(channel.id, retrieve, before=before_id)

				if data:
					if limit is not None:
						limit -= len(data)

					before = cdict(id=int(data[-1]['id']))

				return data, before, limit

			if isinstance(before, datetime.datetime):
				before = cdict(id=utils.time_snowflake(before, high=False))
			elif isinstance(before, (str, int)):
				before = cdict(id=int(before))
			if isinstance(after, datetime.datetime):
				after = cdict(id=utils.time_snowflake(after, high=True))
			elif isinstance(after, (str, int)):
				after = cdict(id=int(after))
			if isinstance(around, datetime.datetime):
				around = cdict(id=utils.time_snowflake(around))
			elif isinstance(around, (str, int)):
				around = cdict(id=int(around))

			if oldest_first is None:
				reverse = after is not None
			else:
				reverse = oldest_first

			after = after or 0
			predicate = None

			if around:
				if limit is None:
					raise ValueError('history does not support around with limit=None')
				if limit > 101:
					raise ValueError("history max limit 101 when specifying around parameter")

				# Strange Discord quirk
				limit = 100 if limit == 101 else limit

				strategy, state = _around_strategy, around

				if before and after:
					predicate = lambda m: after.id < int(m['id']) < before.id
				elif before:
					predicate = lambda m: int(m['id']) < before.id
				elif after:
					predicate = lambda m: after.id < int(m['id'])
			elif reverse:
				strategy, state = _after_strategy, after
				if before:
					predicate = lambda m: int(m['id']) < before.id
			else:
				strategy, state = _before_strategy, before
				if after and after != 0:
					predicate = lambda m: int(m['id']) > after.id

			channel = await self._get_channel()

			CM = bot.CachedMessage
			while True:
				retrieve = 100 if limit is None else min(limit, 100)
				if retrieve < 1:
					return

				data, state, limit = await strategy(retrieve, state, limit)

				if reverse:
					data = reversed(data)
				if predicate:
					data = filter(predicate, data)

				count = 0

				for count, raw_message in enumerate(data, 1):
					raw_message["channel_id"] = raw_message.get("channel_id") or channel.id
					message = CM(raw_message)
					message.channel = channel
					yield message
					# yield self._state.create_message(channel=channel, data=raw_message)

				if count < 100:
					# There's no data left after this
					break
		discord.abc.Messageable.history = history

		def parse_message_reaction_add(self, data):
			emoji = data["emoji"]
			emoji_id = utils._get_as_snowflake(emoji, "id")
			emoji = discord.PartialEmoji.with_state(self, id=emoji_id, animated=emoji.get("animated", False), name=emoji["name"])
			raw = discord.RawReactionActionEvent(data, emoji, "REACTION_ADD")

			member_data = data.get("member")
			if member_data:
				guild = self._get_guild(raw.guild_id)
				raw.member = discord.Member(data=member_data, guild=guild, state=self)
			else:
				raw.member = None

			self.dispatch("raw_reaction_add", raw)
			csubmit(bot.reaction_add(raw, data))
		discord.state.ConnectionState.parse_message_reaction_add = parse_message_reaction_add

		def parse_message_reaction_remove(self, data):
			emoji = data["emoji"]
			emoji_id = utils._get_as_snowflake(emoji, "id")
			emoji = discord.PartialEmoji.with_state(self, id=emoji_id, animated=emoji.get("animated", False), name=emoji["name"])
			raw = discord.RawReactionActionEvent(data, emoji, "REACTION_REMOVE")

			member_data = data.get("member")
			if member_data:
				guild = self._get_guild(raw.guild_id)
				raw.member = discord.Member(data=member_data, guild=guild, state=self)
			else:
				raw.member = None

			self.dispatch("raw_reaction_remove", raw)
			csubmit(bot.reaction_remove(raw, data))
		discord.state.ConnectionState.parse_message_reaction_remove = parse_message_reaction_remove

		def parse_message_reaction_remove_all(self, data):
			raw = discord.RawReactionClearEvent(data)
			self.dispatch("raw_reaction_clear", raw)
			csubmit(bot.reaction_clear(raw, data))
		discord.state.ConnectionState.parse_message_reaction_remove_all = parse_message_reaction_remove_all

		def parse_message_create(self, data, *_):
			message = bot.ExtendedMessage.new(data)
			self.dispatch("message", message)
			if self._messages is not None:
				self._messages.append(message)
			channel = message.channel
			if channel:
				try:
					if not getattr(channel, "guild", None):
						channel.guild = message.guild
					channel.last_message_id = message.id
				except AttributeError:
					pass
		discord.state.ConnectionState.parse_message_create = parse_message_create

		def create_message(self, *, channel, data=None):
			try:
				data["channel_id"] = channel.id
			except:
				print("CREATE_MESSAGE:", data)
				raise
			return bot.ExtendedMessage.new(data)
		discord.state.ConnectionState.create_message = create_message

		@property
		def cached_message(self):
			m_id = self.message_id
			m = bot.cache.messages.get(m_id)
			if m:
				return m
			return bot.data.message_cache.load_message(m_id)
		discord.message.MessageReference.cached_message = cached_message

		async def do_typing(self) -> None:
			channel = await self._get_channel()
			typing = channel._state.http.send_typing

			while True:
				async with Delay(7):
					await typing(channel.id)
		async def __aenter__(self):
			self.task = csubmit(self.do_typing())
			self.task.add_done_callback(discord.context_managers._typing_done_callback)
		discord.context_managers.Typing.do_typing = do_typing
		discord.context_managers.Typing.__aenter__ = __aenter__

		discord.Embed.__hash__ = lambda self: len(self)

		def _get_mime_type_for_audio(data: bytes):
			if data.startswith(b'\x49\x44\x33') or data.startswith(b'\xff\xfb'):
				return 'audio/mpeg'
			if data.startswith(b"OggS"):
				return 'audio/ogg'
			raise ValueError('Unsupported audio type given')
		discord.utils._get_mime_type_for_audio = _get_mime_type_for_audio

	def send_exception(self, messageable, ex, reference=None, op=None, comm=None):
		if self.maintenance and not (reference and self.is_owner(reference.author)):
			print(reference)
			print_exc()
			return fut_nop
		if getattr(ex, "no_react", None):
			reacts = ""
		else:
			reacts="❎"
		footer = None
		fields = None
		if isinstance(ex, TooManyRequests):
			fields = (("Running into the rate limit often?", f"Consider donating using one of the subscriptions from my [ko-fi]({self.kofi_url}), which will grant shorter rate limits amongst many feature improvements!"),)
		elif isinstance(ex, discord.Forbidden):
			fields = (("403", "This error usually indicates that I am missing one or more necessary Discord permissions to perform this command!"),)
		elif isinstance(ex, (discord.HTTPException, ConnectionError)):
			fields = ((lim_str(str(ex.args[0]).split(None, 1)[0], 1024), "This error usually indicates that the remote server (possibly Discord) is rejecting the response. Please double check your inputs, or try again later!"),)
		elif isinstance(ex, (CE, CE2)):
			fields = (("Response disconnected.", "If this error occurs during a command, it is likely due to maintenance!"),)
		elif hasattr(ex, "footer"):
			fields = (ex.footer,) if isinstance(ex.footer, tuple) else (("Note", ex.footer))
		elif isinstance(op, tuple):
			fields = (op,)
		elif comm and (not comm.schema or getattr(comm, "maintenance", False)):
			fields = (("Unexpected or confusing error?", f"This command may currently be under maintenance. Consider joining the [support server]({self.rcc_invite}) for bug reports!"),)
		else:
			fields = (("Unexpected or confusing error?", f"Use {self.get_prefix(getattr(messageable, 'guild', None))}help for help, or consider joining the [support server]({self.rcc_invite}) for bug reports!"),)
		if reference and isinstance(ex, discord.Forbidden) and reference.guild and not messageable.permissions_for(reference.guild.me).send_messages:
			return csubmit(self.missing_perms(messageable, reference))
		title = f"⚠ {type(ex).__name__} ⚠"
		description = "\n".join(as_str(i) for i in T(ex).get("args", ()))
		if (guild := getattr(messageable, "guild", None)) and not messageable.permissions_for(guild.me).embed_links:
			content = (title + "\n" + description).strip()
			if fields:
				content += "\n> " + "\n> ".join((t := ((f["name"], f["value"]) if isinstance(f, dict) else f)) and ("### " + t[0] + "\n" + t[1]) for f in fields if f)
			return csubmit(send_with_react(
				messageable,
				lim_str(content, 2000),
				reacts=reacts,
				reference=reference,
			))
		print(reference)
		return self.send_as_embeds(
			messageable,
			description=lim_str(description, 3000),
			title=title,
			fields=fields,
			reacts=reacts,
			reference=reference,
			footer=footer,
			exc=False,
		)

	async def missing_perms(self, messageable, reference):
		message = reference
		user = message.author
		link = message_link(message)
		s = f"Oops, it appears I do not have permission to reply to your command [here]({link}).\nPlease contact an admin of the server if you believe this is a mistake!"
		colour = await self.get_colour(self.user)
		emb = discord.Embed(colour=colour)
		emb.set_author(**get_author(self.user))
		emb.description = s
		return await user.send(embed=emb)

	async def reaction_add(self, raw, data):
		with tracebacksuppressor:
			channel = await self.fetch_channel(raw.channel_id)
			user = await self.fetch_user(raw.user_id)
			emoji = self._upgrade_partial_emoji(raw.emoji)
			try:
				message = await self.fetch_message(raw.message_id)
			except LookupError:
				message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
				reaction = message._add_reaction(data, emoji, user.id)
				if reaction.count > 1:
					reaction.count -= 1
			else:
				reaction = message._add_reaction(data, emoji, user.id)
			message = await self.ensure_reactions(message)
			self.dispatch("reaction_add", reaction, user)
			self.add_message(message, files=False, force=True)

	async def reaction_remove(self, raw, data):
		with tracebacksuppressor:
			channel = await self.fetch_channel(raw.channel_id)
			user = await self.fetch_user(raw.user_id)
			emoji = self._upgrade_partial_emoji(raw.emoji)
			reaction = None
			try:
				message = await self.fetch_message(raw.message_id)
			except LookupError:
				message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
				reaction = message._add_reaction(data, emoji, user.id)
				if reaction.count > 1:
					reaction.count -= 1
			else:
				with tracebacksuppressor(ValueError):
					reaction = message._remove_reaction(data, emoji, user.id)
			if not reaction:
				message = await discord.abc.Messageable.fetch_message(channel, raw.message_id)
				reaction = message._add_reaction(data, emoji, user.id)
				if reaction.count > 1:
					reaction.count -= 1
				reaction = message._remove_reaction(data, emoji, user.id)
			self.dispatch("reaction_remove", reaction, user)
			self.add_message(message, files=False, force=True)
	
	async def reaction_clear(self, raw, data):
		channel = await self.fetch_channel(raw.channel_id)
		message = await self.fetch_message(raw.message_id, channel=channel)
		old_reactions = message.reactions.copy()
		message.reactions.clear()
		self.dispatch("reaction_clear", message, old_reactions)
		self.add_message(message, files=False, force=True)

	@tracebacksuppressor
	async def init_ready(self):
		await asubmit(self.start_webserver)
		await self.modload
		print(f"Mapped command count: {len(self.commands)}")
		commands = set()
		for command in self.commands.values():
			commands.update(command)
		print(f"Unique command count: {len(commands)}")
		# Assign all bot database events to their corresponding keys.
		for db in self.data.values():
			for f in dir(db):
				if f.startswith("_") and f[-1] == "_" and f[1] != "_":
					func = getattr(db, f, None)
					if callable(func):
						self.events.append(f, func)
		print(f"Database event count: {sum(len(v) for v in self.events.values())}")
		await self.fetch_user(self.deleted_user)
		self.gl = csubmit(self.global_loop())
		self.sl = csubmit(self.slow_loop())
		self.ll = csubmit(self.lazy_loop())
		print("Update loops initiated.")
		futs = alist()
		if commands:
			futs.add(asubmit(self.update_slash_commands, priority=True))
		futs.add(csubmit(self.create_main_website(first=True)))
		futs.add(self.audio_client_start)
		await self.wait_until_ready()
		print("Bot ready.")
		self.bot_ready = True
		# Send bot_ready event to all databases.
		await self.send_event("_bot_ready_", bot=self)
		print("Waiting on workers...")
		for fut in futs:
			with tracebacksuppressor:
				await fut
		print("Workers ready.")
		await wrap_future(self.connect_ready)
		self.start_time = utc()
		print("Connect ready.")
		self.ready = True
		await asubmit(self.update_usernames)
		# Send ready event to all databases.
		await self.send_event("_ready_", bot=self)
		await self.update_status(force=True)
		tsubmit(self.fast_loop)
		print("Database ready.")
		await wrap_future(self.full_ready)
		await self.guilds_ready
		await asubmit(self.update_usernames)
		print("Guilds ready.")
		self.initialisation_complete = True
		print("Initialisation complete.")

	async def flatten_into_cache(self, history):
		data = {}
		async for m in history:
			data[m.id] = m
		esubmit(self.cache.messages.update, data)
		return data

	def set_guilds(self):
		AUTH["guild_count"] = len(self._guilds)
		save_auth(AUTH)

	def set_client_events(self):

		print("Setting client events...")
		self.states = [None] * self.shard_count

		async def on_full_connect():
			await asubmit(self.set_guilds)
			print("Successfully connected as " + str(self.user))

		@self.event
		async def on_shard_connect(shard_id):
			print("Shard", shard_id, "connected.")
			if all(s is None for s in self.states):
				AUTH["name"] = self.user.display_name
				AUTH["discord_id"] = self.user.id
				save_auth(AUTH)
				self.invite = f"https://discordapp.com/oauth2/authorize?permissions=8&client_id={self.id}&scope=bot%20applications.commands"
				self.mention = (user_mention(self.id), user_pc_mention(self.id))
				if not self.started:
					self.started = True
					csubmit(self.init_ready())
				else:
					print("Reconnected.")
			self.states[shard_id] = self.states[shard_id] or False
			if none(s is None for s in self.states):
				await on_full_connect()

		async def on_ready():
			print("discord.py ready.")

		async def on_full_ready():
			print("All clients ready.")
			if not self.full_ready.done():
				self.full_ready.set_result(True)
			csubmit(aretry(self.get_ip, delay=10))
			with tracebacksuppressor:
				for guild in self.guilds:
					if guild.unavailable:
						print(f"Warning: Guild {guild.id} is not available.")
				await self.handle_update(force=True)

		@self.event
		async def on_shard_ready(shard_id):
			print("Shard", shard_id, "ready.")
			if none(s is True for s in self.states):
				self.guilds_loading = []
				if not self.connect_ready.done():
					self.connect_ready.set_result(True)
			self.states[shard_id] = self.states[shard_id] or True
			guilds = [g for g in self.client.guilds if self.guild_shard(g.id) == shard_id]
			for g in guilds:
				self.cache.guilds.pop(g.id, None)
				self._guilds[g.id] = g
			await self.modload
			fut = csubmit(self.load_guilds(guilds))
			self.guilds_loading.append(fut)
			await asubmit(self.update_subs, priority=True)
			self.update_cache_feed()
			with tracebacksuppressor:
				await self.handle_update(force=True)
			if all(s is True for s in self.states):
				self.guilds_ready = csubmit(gather(*self.guilds_loading))
				await on_full_ready()

		# Server join message
		@self.event
		async def on_guild_join(guild):
			self.guilds_updated = True
			self.users_updated = True
			print(f"New server: {guild}")
			guild = await self.fetch_guild(guild.id)
			self.sub_guilds[guild.id] = guild
			m = guild.me
			await self.send_event("_join_", user=m, guild=guild)
			channel = self.get_first_sendable(guild, m)
			emb = discord.Embed(colour=discord.Colour(8364031))
			emb.set_author(**get_author(self.user))
			emb.description = f"```callback-fun-wallet-{utc()}-\nHi there!```- I'm {self.name}, a multipurpose discord bot created by <@201548633244565504>. Thanks for adding me"
			user = None
			if guild.me.guild_permissions.view_audit_log:
				with suppress(discord.Forbidden):
					a = guild.audit_logs(limit=5, action=discord.AuditLogAction.bot_add)
					async for e in a:
						if e.target.id == self.id:
							user = e.user
							break
			if user is not None:
				emb.description += f", {user_mention(user.id)}"
				if "dailies" in self.data:
					self.data.dailies.progress_quests(user, "invite")
			emb.description += (
				f"!\n-My default prefix is `{self.prefix}`, which can be changed as desired on a per-server basis. Mentioning me also serves as an alias for all prefixes.\n"
				+ (f"- As this appears to be a large server, all except the most basic commands will automatically be disabled for non-bot channels. Please view the `{self.prefix}enabled` command for specifics.\n" if len(guild._members) > 100 else "")
				+ f"- For more information, use the `{self.prefix}help` command, "
				+ (f"I have a website at {self.webserver}, " if self.webserver else "")
				+ f"and my source code is available at {self.github} for those who are interested.\n"
				+ "Pleased to be at your service 🙂"
			)
			if not m.guild_permissions.administrator:
				emb.add_field(name="Psst!", value=(
					"I noticed you haven't given me administrator permissions here.\n"
					+ "That's completely understandable if intentional, but please note that some features may not function well, or not at all, without the required permissions."
				))
			message = await channel.send(embed=emb)
			await message.add_reaction("✅")
			await self.load_guild(guild)
			for member in guild.members:
				name = str(member)
				self.usernames[name] = self.cache.users[member.id]
			await asubmit(self.set_guilds)

		# Guild destroy event: Remove guild from bot cache.
		@self.event
		async def on_guild_remove(guild):
			self.guilds_updated = True
			self.users_updated = True
			self.cache.guilds.pop(guild.id, None)
			self.sub_guilds.pop(guild.id, None)
			await asubmit(self.set_guilds)
			print("Server lost:", guild, "removed.")

		# Reaction add event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
		@self.event
		async def on_reaction_add(reaction, user):
			message = reaction.message
			channel = message.channel
			emoji = reaction.emoji
			if user.id == self.deleted_user:
				print("Deleted user RAW_REACTION_ADD", channel, user, message, emoji, channel.id, message.id)
			await self.seen(user, channel, message.guild, event="reaction", raw="Adding a reaction")
			if user.id != self.id:
				if "users" in self.data:
					self.data.users.add_xp(user, xrand(4, 7))
					self.data.users.add_gold(user, xrand(1, 5))
				reaction = str(emoji)
				await self.send_event("_reaction_add_", message=message, react=reaction, user=user)
				await self.react_callback(message, reaction, user)
				await self.check_to_delete(message, reaction, user)

		# Reaction remove event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
		@self.event
		async def on_reaction_remove(reaction, user):
			message = reaction.message
			channel = message.channel
			emoji = reaction.emoji
			if user.id == self.deleted_user:
				print("Deleted user RAW_REACTION_REMOVE", channel, user, message, emoji, channel.id, message.id)
			await self.seen(user, channel, message.guild, event="reaction", raw="Removing a reaction")
			if user.id != self.id:
				reaction = str(emoji)
				await self.send_event("_reaction_remove_", message=message, react=reaction, user=user)
				await self.react_callback(message, reaction, user)
				await self.check_to_delete(message, reaction, user)

		# Voice state update event: automatically unmutes self if server muted, calls _seen_ bot database event.
		@self.event
		async def on_voice_state_update(member, before, after):
			if member.id == self.id:
				after = member.voice
				if after is not None:
					if (after.mute or after.deaf) and member.permissions_in(after.channel).deafen_members:
						# print("Unmuted self in " + member.guild.name)
						await member.edit(mute=False, deafen=False)
					await self.handle_update()
			# Check for users with a voice state.
			if after is not None and not after.afk:
				if before is None:
					if "users" in self.data:
						self.data.users.add_xp(after, xrand(6, 12))
						self.data.users.add_gold(after, xrand(2, 5))
					await self.seen(member, member.guild, event="misc", raw="Joining a voice channel")
				elif any((getattr(before, attr) != getattr(after, attr) for attr in ("self_mute", "self_deaf", "self_stream", "self_video"))):
					await self.seen(member, member.guild, event="misc", raw="Updating their voice settings")

		# Typing event: calls _typing_ and _seen_ bot database events.
		@self.event
		async def on_typing(channel, user, when):
			await self.send_event("_typing_", channel=channel, user=user)
			if user.id == self.deleted_user:
				print("Deleted user TYPING", channel, user, channel.id)
			await self.seen(user, T(channel).get("guild"), delay=10, event="typing", raw="Typing")

		# Message send event: processes new message. calls _send_ and _seen_ bot database events.
		@self.event
		async def on_message(message):
			channel = message.channel
			if T(channel).get("guild") is None and T(channel).get("recipient") is None:
				try:
					channel = await self._fetch_channel(channel.id)
				except discord.NotFound:
					print(f"Channel {channel.id} not found.")
				else:
					message.channel = channel
			self.add_message(message, force=True)
			guild = message.guild
			if guild:
				csubmit(self.send_event("_send_", message=message))
			user = message.author
			if user.id == self.deleted_user:
				print("Deleted user MESSAGE", channel, user, message, channel.id, message.id)
			fut = csubmit(self.seen(user, channel, guild, event="message", raw="Sending a message"))
			await self.react_callback(message, None, user)
			await fut
			await self.handle_message(message)

		# Socket response event: if the event was an interaction, create a virtual message with the arguments as the content, then process as if it were a regular command.
		@self.event
		async def on_socket_response(data):
			if not data.get("op") and data.get("t") == "INTERACTION_CREATE" and "d" in data:
				try:
					# dt = utc_dt()
					message = self.GhostMessage()
					d = data["d"]
					message.id = int(d["id"])
					message.slash = d["token"]
					cdata = d.get("data")
					if d["type"] == 2:
						# print("SLASH:", cdata)
						name = cdata["name"].replace(" ", "")
						command = self.commands[name][0]
						try:
							usage = command.usage
						except LookupError:
							usage = ""
						arguments = sorted(cdata.get("options", ()), key=lambda arg: ((i := usage.find(arg.get("name") or "")) < 0, i))
						kwargs = {arg["name"]: arg["value"] for arg in arguments}
						if kwargs and command.schema:
							for k in tuple(kwargs):
								if k not in command.schema:
									continue
								if command.schema[k].get("multiple"):
									argl = [kwargs[k]]
									for i in range(5):
										try:
											argl.append(kwargs.pop(f"{k}-{i + 2}"))
										except KeyError:
											pass
						mdata = d.get("member")
						if not mdata:
							mdata = d.get("user")
						else:
							mdata = mdata.get("user")
						author = self._state.store_user(mdata)
						message.author = author
						channel = None
						if cdata.get("type") == 3 and "resolved" in cdata:
							res = cdata.get("resolved", {})
							for mdata in res.get("users", {}).values():
								self._state.store_user(mdata)
							for mdata in res.get("messages", {}).values():
								msg = self.ExtendedMessage.new(mdata)
								self.add_message(msg, force=True)
								message.channel = channel = msg.channel or message.channel
						try:
							channel = self.force_channel(d["channel_id"])
							try:
								guild = await self.fetch_guild(d["guild_id"])
							except (LookupError, discord.NotFound):
								raise KeyError
							message.guild = guild
							author = guild.get_member(author.id)
							if author:
								message.author = author
						except KeyError:
							if author is None:
								raise
							if channel is None:
								channel = await self.get_dm(author)
							guild = T(channel).get("guild")
						if not T(message).get("guild"):
							message.guild = guild
						message.content = f"/{name} " + " ".join(map(json.dumps, kwargs.values()))
						message.channel = channel
						message.noref = True
						message.deleted = False
						message.ephemeral = kwargs.pop("ephemeral", False) and getattr(command, "ephemeral", False)
						try:
							await self.run_command(command, kwargs, message=message, slash=True, respond=True)
						except Exception as ex:
							if not getattr(message, "deferred", False):
								await self.defer_interaction(message)
							resp = await self.send_exception(channel, ex, reference=message, comm=command)
							print("SC:", resp)
						finally:
							message.deleted = True
					elif d["type"] == 3:
						custom_id = cdata.get("custom_id", "")
						if "?" in custom_id:
							custom_id = custom_id.rsplit("?", 1)[0]
						if custom_id.startswith("▪️"):
							return await self.ignore_interaction(message)
						mdata = d.get("member")
						if not mdata:
							mdata = d.get("user")
						else:
							mdata = mdata.get("user")
						user = self._state.store_user(mdata)
						if not user:
							user = self.GhostUser()
						channel = None
						try:
							channel = self.force_channel(d["channel_id"])
							try:
								guild = await self.fetch_guild(d["guild_id"])
							except (LookupError, discord.NotFound):
								raise KeyError
							user = (guild.get_member(user.id) if guild else None) or user
						except KeyError:
							if user is None:
								raise
							if channel is None:
								channel = await self.get_dm(user)
						message.channel = channel
						if custom_id.startswith("\x7f"):
							custom_id = cdata.get("values") or custom_id
							if isinstance(custom_id, list_like):
								custom_id = " ".join(custom_id)
						if isinstance(custom_id, str) and custom_id.startswith("~"):
							m_id, custom_id = custom_id[1:].split("~", 1)
							custom_id = "~" + custom_id
						else:
							m_id = d["message"]["id"]
						m = await self.fetch_message(m_id, channel)
						if getattr(m, "method", None):
							del m.method
						add = False
						if type(m) is not self.ExtendedMessage:
							m = self.ExtendedMessage(m)
							add = True
						if "```callback" not in m.content and not m.embeds:
							m = await channel.fetch_message(m_id)
							add = True
						if add:
							self.add_message(m, force=True)
						m.int_id = message.id
						m.int_token = message.slash
						# print(custom_id, user)
						await self.react_callback(m, custom_id, user)
					else:
						print("Unknown interaction:\n" + str(data))
					print("Deleting interaction:", message)
					for attr in ("slash", "int_id", "int_token"):
						try:
							delattr(message, attr)
						except AttributeError:
							pass
				except Exception:
					print_exc()
					print("Failed interaction:\n" + str(data))

		# Message edit event: processes edited message, uses raw payloads rather than discord.py message cache. calls _edit_ and _seen_ bot database events.
		@self.event
		async def on_raw_message_edit(payload):
			data = payload.data
			m_id = int(data["id"])
			raw = False
			if payload.cached_message:
				before = payload.cached_message
				after = await self.fetch_message(m_id, payload.channel_id)
			else:
				try:
					before = await self.fetch_message(m_id, old=True)
				except LookupError:
					# If message was not in cache, create a ghost message object to represent old message.
					c_id = data.get("channel_id")
					if not c_id:
						return
					before = self.GhostMessage()
					before.deleted = False
					before.channel = channel = self.force_channel(c_id)
					before.guild = guild = T(channel).get("guild")
					before.id = payload.message_id
					try:
						u_id = data["author"]["id"]
					except KeyError:
						u_id = None
						before.author = None
					else:
						if guild is not None:
							user = guild.get_member(u_id)
						else:
							user = None
						if not user:
							user = self._state.store_user(data["author"])
							# user = await self.fetch_user(u_id)
						before.author = user
					try:
						after = await discord.abc.Messageable.fetch_message(channel, before.id)
					except LookupError:
						after = copy.copy(before)
						after._update(data)
					else:
						before.author = after.author
					if len(after.embeds) == 1 and after.embeds[0].type != "rich" and find_urls(after.content) or any((fmt := url2ext(url)) in IMAGE_FORMS or fmt in VIDEO_FORMS for a in message.attachments):
						print(f"Possible embed-only update on message {after.id}, ignoring...")
						return
					raw = True
				else:
					if type(before) is self.CachedMessage:
						before = copy.copy(before)
					after = copy.copy(before)
					after._update(data)
			with suppress(AttributeError):
				before.deleted = True
			if after.channel is None:
				after.channel = self.force_channel(payload.channel_id)
			sem = T(before).get("sem")
			if not sem and after.edited_at and (utc_ddt() - after.created_at).total_seconds() >= 3590:
				try:
					after.sem = Semaphore(3, 1, rate_limit=20.09)
					after.sem.delay_for(20)
				except AttributeError:
					after = self.ExtendedMessage(after)
					after.sem = Semaphore(3, 1, rate_limit=20.09)
					after.sem.delay_for(20)
			with tracebacksuppressor:
				inits = T(before).get("inits")
				if inits:
					print("Cancel:", inits)
					for fut in inits:
						with tracebacksuppressor:
							try:
								fut.cancel()
							except AttributeError:
								try:
									fut.close()
								except AttributeError:
									force_kill(fut)
			self.add_message(after, files=False, force=2)
			if before.author.id == self.deleted_user or after.author.id == self.deleted_user:
				print("Deleted User RAW_MESSAGE_EDIT", after.channel, before.author, after.author, before, after, after.channel.id, after.id)
			if raw or before.content != after.content:
				if "users" in self.data:
					self.data.users.add_xp(after.author, xrand(1, 4))
				if T(after).get("guild"):
					fut = csubmit(self.send_event("_edit_", before=before, after=after))
				else:
					fut = None
				await self.seen(after.author, after.channel, after.guild, event="message", raw="Editing a message")
				if fut:
					with tracebacksuppressor:
						await fut
				await self.handle_message(after, before=before)

		self.deletes = {}
		self.bulk_deletes = {}
		self.delete_waits = {}
		@self.event
		async def on_audit_log_entry_create(data):
			# print("AUDIT:", data)
			with tracebacksuppressor:
				if data.action is discord.AuditLogAction.message_delete:
					# print("Audited delete:", data)
					guild = data.guild
					T(self.delete_waits.get(guild.id)).if_instance(Future, lambda t: t.obj.set_result(None))
					self.deletes.setdefault(guild.id, deque(maxlen=256)).append(data)
					self.delete_waits.pop(guild.id, None)
				elif data.action is discord.AuditLogAction.message_bulk_delete:
					# print("Audited bulk delete:", data)
					guild = data.guild
					T(self.delete_waits.get(guild.id)).if_instance(Future, lambda t: t.obj.set_result(None))
					self.bulk_deletes.setdefault(guild.id, deque(maxlen=256)).append(data)
					self.delete_waits.pop(guild.id, None)

		# Message delete event: uses raw payloads rather than discord.py message cache. calls _delete_ bot database event.
		@self.event
		async def on_raw_message_delete(payload):
			# print("DELETE:", payload)
			with tracebacksuppressor:
				fut = self.delete_waits.setdefault(payload.guild_id, Future())
				fut2 = csubmit(_on_raw_message_delete(payload))
				with suppress(T1, CE):
					await asyncio.wait_for(wrap_future(fut), timeout=1)
				message = await fut2
				guild = message.guild
				if guild:
					fut = csubmit(self.send_event("_delete_", message=message))
					await self.send_event("_raw_delete_", message=message)
					await fut

		async def _on_raw_message_delete(payload):
			if "deleted" in self.data:
				self.data.deleted.cache[payload.message_id] = max(1, self.is_deleted(payload.message_id))
			try:
				message = payload.cached_message
				if not message:
					raise LookupError
			except (AttributeError, LookupError):
				channel = await self.fetch_channel(payload.channel_id)
				try:
					message = await self.fetch_message(payload.message_id, channel, old=True)
					if message is None:
						raise LookupError
				except (AttributeError, LookupError, discord.NotFound):
					# If message was not in cache, create a ghost message object to represent old message.
					message = self.GhostMessage()
					message.channel = channel
					try:
						message.guild = channel.guild
					except AttributeError:
						message.guild = None
					message.id = payload.message_id
					message.author = await self.fetch_user(self.deleted_user)
					message.author.name = "Unknown User"
					history = discord.abc.Messageable.history(channel, limit=101, around=message)
					csubmit(self.flatten_into_cache(history))
			try:
				message.deleted = True
			except AttributeError:
				message = self.ExtendedMessage(message)
				self.add_message(message, force=True)
				message.deleted = True
			with tracebacksuppressor:
				inits = T(message).get("inits")
				if inits:
					print("Cancel:", inits)
					for fut in inits:
						with tracebacksuppressor:
							try:
								fut.cancel()
							except AttributeError:
								try:
									fut.close()
								except AttributeError:
									force_kill(fut)
			return message

		# Message bulk delete event: uses raw payloads rather than discord.py message cache. calls _bulk_delete_ and _delete_ bot database events.
		@self.event
		async def on_raw_bulk_message_delete(payload):
			# print("BULK_DELETE:", payload)
			with tracebacksuppressor:
				fut = self.delete_waits.setdefault(payload.guild_id, Future())
				fut2 = csubmit(_on_raw_bulk_message_delete(payload))
				with suppress(T1, CE):
					await asyncio.wait_for(wrap_future(fut), timeout=1)
				messages = await fut2
				await self.send_event("_bulk_delete_", messages=messages)
				for message in messages:
					guild = T(message).get("guild")
					if guild:
						await self.send_event("_delete_", message=message, bulk=True)

		async def _on_raw_bulk_message_delete(payload):
			try:
				messages = payload.cached_messages
				if not messages or len(messages) < len(payload.message_ids):
					raise LookupError
			except (AttributeError, LookupError):
				messages = alist()
				channel = await self.fetch_channel(payload.channel_id)
				for m_id in payload.message_ids:
					try:
						message = await self.fetch_message(m_id, channel)
						if message is None:
							raise LookupError
					except (AttributeError, LookupError, discord.NotFound):
						# If message was not in cache, create a ghost message object to represent old message.
						message = self.GhostMessage()
						message.channel = channel
						try:
							message.guild = channel.guild
						except AttributeError:
							message.guild = None
						message.id = m_id
						message.author = await self.fetch_user(self.deleted_user)
					messages.add(message)
			out = deque()
			for message in sorted(messages, key=lambda m: m.id):
				try:
					message.deleted = True
				except AttributeError:
					message = self.ExtendedMessage(message)
					self.add_message(message, force=True)
					message.deleted = True
				out.append(message)
			messages = alist(out)
			return messages

		@self.event
		async def on_guild_update(before, after):
			await self.send_event("_guild_update_", before=before, after=after)

		# User update event: calls _user_update_ and _seen_ bot database events.
		@self.event
		async def on_user_update(before, after):
			b = str(before)
			a = str(after)
			if b != a:
				self.usernames.pop(b, None)
				self.usernames[a] = after
			if not isinstance(before, self.GhostUser):
				await self.send_event("_user_update_", before=before, after=after)
			if before.id == self.deleted_user or after.id == self.deleted_user:
				print("Deleted user USER_UPDATE", before, after, before.id, after.id)
			await self.seen(after, event="misc", raw="Editing their profile")

		# Member update event: calls _member_update_ and _seen_ bot database events.
		@self.event
		async def on_member_update(before, after):
			if not isinstance(before, self.GhostUser):
				await self.send_event("_member_update_", before=before, after=after)
			if self.status_changed(before, after):
				# A little bit of a trick to make sure this part is only called once per user event.
				# This is necessary because on_member_update is called once for every member object.
				# By fetching the first instance of a matching member object,
				# this ensures the event will not be called multiple times if the user shares multiple guilds with the bot.
				try:
					member = self.get_member(after.id)
				except LookupError:
					member = None
				if member is None or member.guild == after.guild:
					if self.status_updated(before, after):
						if "dailies" in self.data:
							self.data.dailies.progress_quests(after, "status")
						await self.seen(after, event="misc", raw="Changing their status")
					elif after.status == discord.Status.offline:
						await self.send_event("_offline_", user=after)

		# Member join event: calls _join_ and _seen_ bot database events.
		@self.event
		async def on_member_join(member):
			name = str(member)
			self.usernames[name] = self.cache.users[member.id]
			if member.guild.id in self._guilds:
				member.guild._member_count = len(member.guild._members)
				if "guilds" in self.data:
					self.data.guilds.register(member.guild, force=False)
			await self.send_event("_join_", user=member, guild=member.guild)
			await self.seen(member, member.guild, event="misc", raw="Joining a server")

		# Member leave event: calls _leave_ bot database event.
		@self.event
		async def on_member_remove(before):
			after = self.cache.users[before.id] = await self._fetch_user(before.id)
			b = str(before)
			a = str(after)
			if b != a:
				self.usernames.pop(b, None)
				self.usernames[a] = after
				await self.send_event("_user_update_", before=before, after=after)
			if hasattr(before, "_user"):
				before._user = after
			if after.id not in self.cache.members:
				name = str(after)
				self.usernames.pop(name, None)
			if before.guild.id in self._guilds:
				before.guild._members.pop(after.id, None)
				before.guild._member_count = len(before.guild._members)
				if "guilds" in self.data:
					self.data.guilds.register(before.guild, force=False)
			await self.send_event("_leave_", user=before, guild=before.guild)

		# Channel create event: calls _channel_create_ bot database event.
		@self.event
		async def on_guild_channel_create(channel):
			self.sub_channels[channel.id] = channel
			guild = channel.guild
			if guild:
				await self.send_event("_channel_create_", channel=channel, guild=guild)

		# Channel update event: calls _channel_update_ bot database event.
		@self.event
		async def on_guild_channel_update(before, after):
			self.sub_channels[after.id] = after
			guild = after.guild
			if guild and (before.name != after.name or before.position != after.position):
				await self.send_event("_channel_update_", before=before, after=after, guild=guild)

		# Channel delete event: calls _channel_delete_ bot database event.
		@self.event
		async def on_guild_channel_delete(channel):
			self.sub_channels.pop(channel.id, None)
			self.cache.channels.pop(channel.id, None)
			guild = channel.guild
			if guild:
				await self.send_event("_channel_delete_", channel=channel, guild=guild)

		# Thread delete event: calls _channel_delete_ bot database event.
		@self.event
		async def on_thread_delete(channel):
			self.sub_channels.pop(channel.id, None)
			self.cache.channels.pop(channel.id, None)
			guild = channel.guild
			if guild:
				await self.send_event("_channel_delete_", channel=channel, guild=guild)

		# Thread update event: calls _thread_update_ bot database event.
		@self.event
		async def on_thread_update(before, after):
			await self.send_event("_thread_update_", before=before, after=after)

		# Webhook update event: updates the bot's webhook cache if there are new webhooks.
		@self.event
		async def on_webhooks_update(channel):
			self.data.webhooks.pop(channel.id, None)

		# User ban event: calls _ban_ bot database event.
		@self.event
		async def on_member_ban(guild, user):
			await self.send_event("_ban_", user=user, guild=guild)


@tracebacksuppressor
def update_file_cache():
	attachments = {t for t in bot.cache.attachments.items() if isinstance(t[-1], bytes)}
	while len(attachments) > 512:
		a_id = next(iter(attachments))
		self.cache.attachments[a_id] = a_id
		attachments.discard(a_id)


class SimulatedMessage:
	"discord.py-compatible message object used for simulated messages, either from slash commands or API."

	def __init__(self, bot, content, t, name, nick=None, recursive=True):
		self._state = bot._state
		self.created_at = DynamicDT.utcfromtimestamp(t)
		self.ip = name
		self.id = time_snowflake(self.created_at, high=True) - 1
		self.content = content
		self.response = deque()
		if recursive:
			author = self.__class__(bot, content, (ip2int(name) + MIZA_EPOCH) * 1000 + 1, name, nick, recursive=False)
			author.response = self.response
			author.message = self
			author.dm_channel = author
		else:
			author = self
		self.author = author
		self.channel = self
		self.is_channel = True
		self.guild = author
		self.dm_channel = author
		self.name = name
		disc = str(xrand(10000))
		self.discriminator = "0" * (4 - len(disc)) + disc
		self.nick = self.display_name = nick or name
		self.owner_id = self.id
		self.mention = f"<@{self.id}>"
		self.recipient = author
		self.me = bot.user
		self.channels = self.text_channels = self.voice_channels = [author]
		self.members = self._members = [author, bot.user]
		self.message = self
		self.owner = author

	system_content = clean_content = ""
	display_avatar = avatar_url = icon_url = "https://mizabot.xyz/u/EHrvYWbEUD0"
	roles = []
	emojis = []
	mentions = []
	attachments = []
	embeds = []
	reactions = []
	position = 0
	voice = None
	bot = False
	ghost = True
	slash = True
	simulated = True
	deleted = False
	reference = None
	__str__ = lambda self: self.name

	async def send(self, *args, **kwargs):
		if args:
			try:
				kwargs["content"] = args[0]
			except IndexError:
				kwargs["content"] = ""
		try:
			embed = kwargs.pop("embed")
		except KeyError:
			pass
		else:
			if embed is not None:
				e = embed.to_dict()
				if e:
					kwargs["embed"] = e
		try:
			embeds = kwargs.pop("embeds")
		except KeyError:
			pass
		else:
			embeds = [e for e in ((embed if isinstance(embed, dict) else embed.to_dict()) for embed in embeds if embed is not None) if e]
			if embeds:
				kwargs["embeds"] = embeds
		try:
			files = kwargs.pop("files", None) or [kwargs.pop("file")]
		except KeyError:
			pass
		else:
			ofiles = []
			for file in files:
				f = await asubmit(self.as_file, file)
				ofiles.append(f)
			kwargs["files"] = ofiles
		self.response.append(kwargs)
		return self

	def edit(self, **kwargs):
		self.response[-1].update(kwargs)
		return as_fut(self)

	async def history(self, *args, **kwargs):
		yield self

	get_member = lambda self, *args: None
	delete = async_nop
	add_reaction = async_nop
	remove_reaction = async_nop
	delete_messages = async_nop
	trigger_typing = async_nop
	webhooks = lambda self: []
	guild_permissions = discord.Permissions((1 << 32) - 1)
	permissions_for = lambda self, member=None: self.guild_permissions
	permissions_in = lambda self, channel=None: self.guild_permissions
	invites = lambda self: exec("raise LookupError")


async def desktop_identify(self):
	"""Sends the IDENTIFY packet."""
	print("Overriding with desktop status...")
	payload = {
		'op': self.IDENTIFY,
		'd': {
			'token': self.token,
			'properties': {
				'os': 'Miza-OS',
				'browser': 'Discord Client',
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

discord.gateway.DiscordWebSocket.identify = lambda self: desktop_identify(self)


# If this is the module being run and not imported, create a new Bot instance and run it.
if __name__ == "__main__":
	# Redirects all output to the main log manager (PRINT).
	_print = print
	with contextlib.redirect_stdout(PRINT):
		with contextlib.redirect_stderr(PRINT):
			orig_warning = asyncio.base_events.logger.warning
			task_re = re.compile(r"<Task.*name='([^']+)'.*?>")

			def custom_warning(msg, *args, **kwargs):
				orig_warning(msg, *args, **kwargs)

				m = task_re.search(msg % args)
				if not m:
					return

				task_name = m.group(1)
				loop = asyncio.get_event_loop()
				for task in asyncio.all_tasks(loop):
					if task.get_name() == task_name:
						print(f"\n[Slow task detected] {task}")
						for line in format_async_stack(task.get_coro()):
							print(line.strip())
						print("[End slow task dump]\n")

			# Monkey-patch
			asyncio.base_events.logger.warning = custom_warning
			eloop.slow_callback_duration = 0.375
			eloop.set_debug(True)
			with tracebacksuppressor:
				PRINT.start()
				sys.stdout = sys.stderr = print = PRINT
				print("Logging started.")
				initialise_ppe()
				esubmit(proc_start)
				discord.client._loop = eloop
				self = miza = bot = client = BOT[0] = Bot()
				miza.http.user_agent = "Miza"
				miza.miza = miza
				with miza:
					miza.run()
			sys.__stdout__.write("MAIN PROCESS EXITING...")
			common.MEM_LOCK.close()
			asyncs.athreads.shutdown(wait=False)
			asyncs.bthreads.shutdown(wait=False)
			asyncs.pthreads.shutdown(wait=False)
			asyncs.mthreads.shutdown(wait=False)
	print = _print
	sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
