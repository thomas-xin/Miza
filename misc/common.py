# ruff: noqa: E401 E402 E731 F401 F403 F405

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

print("COMMON:", __name__)

import misc
from misc.types import *
from misc.util import *

if __name__ != "__mp_main__":
	from misc import smath
	from misc.smath import *
	from misc.caches import *

import psutil, subprocess, weakref, zipfile, urllib, asyncio, json, pickle, functools, orjson, aiohttp, threading, shutil, filetype, inspect, sqlite3, argparse, bisect, httpx

# VERY HACKY removes deprecated audioop dependency for discord.py; this would cause volume transformations to fail but Miza uses FFmpeg for them anyway
sys.modules["audioop"] = sys
import discord, discord.utils, discord.file  # noqa: E402

from misc.asyncs import *

openai = None
if os.environ.get("IS_BOT") and os.environ.get("AI_FEATURES", True):
	import misc.ai as ai
	from misc.ai import *

BOT = [None]

from zipfile import ZipFile
import urllib.request, urllib.parse

DC = 0
torch = None
try:
	import pynvml
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
	if not os.environ.get("AI_FEATURES", True):
		raise StopIteration("AI features disabled.")
	if __name__ != "__mp_main__":
		import torch
except Exception:
	print_exc()
hwaccel = "cuda" if DC else "d3d11va" if os.name == "nt" else "auto"

utils = discord.utils
escape_markdown = utils.escape_markdown
escape_mentions = utils.escape_mentions
escape_everyone = lambda s: s#s.replace("@everyone", "@\xadeveryone").replace("@here", "@\xadhere")
escape_roles = lambda s: s#escape_everyone(s).replace("<@&", "<@\xad&")

standard_roles = lambda member: [role for role in member.roles if not role.is_default()]


class MemoryTimer(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	"A context manager that monitors the amount of time taken for a designated section of code."

	timers = cdict()

	@classmethod
	def list(cls):
		return "\n".join(str(name) + ": " + str(duration) for duration, name in sorted(((mean(v), k) for k, v in cls.timers.items()), reverse=True))

	def __init__(self, name=None):
		self.name = name
		self.start = utc()

	def __call__(self):
		return self.exit()

	__enter__ = lambda self: self
	def __exit__(self, *args):
		taken = utc() - self.start
		try:
			self.timers[self.name].append(taken)
		except KeyError:
			self.timers[self.name] = t = deque(maxlen=100)
			t.append(taken)

	__aenter__ = lambda self: emptyfut
	def __aexit__(self, *args):
		self.__exit__()
		return emptyfut


def restructure_buttons(buttons):
	"Combines unstructured list of buttons into groups compatible with Discord API."
	if not buttons:
		return buttons
	if issubclass(type(buttons[0]), collections.abc.Mapping):
		b = alist()
		if len(buttons) <= 3:
			b.append(buttons)
		elif len(buttons) <= 5:
			b.append(buttons[:-2])
			b.append(buttons[-2:])
		elif len(buttons) <= 7:
			b.append(buttons[:-3])
			b.append(buttons[-3:])
		elif len(buttons) == 8:
			b.append(buttons[:4])
			b.append(buttons[4:])
		elif len(buttons) == 9:
			b.append(buttons[:3])
			b.append(buttons[3:6])
			b.append(buttons[6:])
		elif len(buttons) == 10:
			b.append(buttons[:5])
			b.append(buttons[5:])
		elif len(buttons) <= 12:
			b.append(buttons[:4])
			b.append(buttons[4:8])
			b.append(buttons[8:])
		elif len(buttons) <= 15:
			b.append(buttons[:5])
			b.append(buttons[5:-5])
			b.append(buttons[-5:])
		elif len(buttons) == 16:
			b.append(buttons[:4])
			b.append(buttons[4:8])
			b.append(buttons[8:12])
			b.append(buttons[12:])
		elif len(buttons) <= 20:
			b.append(buttons[:5])
			b.append(buttons[5:10])
			b.append(buttons[10:15])
			b.append(buttons[15:])
		else:
			while buttons:
				b.append(buttons[:5])
				buttons = buttons[5:]
		buttons = b
	used_custom_ids = set()
	for row in buttons:
		for button in row:
			if "type" not in button:
				button["type"] = 2
			if "name" in button:
				button["label"] = button.pop("name")
			if "label" in button:
				button["label"] = lim_str(button["label"], 34)
			try:
				if isinstance(button["emoji"], str):
					s = button["emoji"]
					if find_emojis(s):
						left, right = s.strip("<>").rsplit(":", 1)
						animated, left = left.split(":", 1)
						button["emoji"] = cdict(id=int(right), name=left, animated=bool(animated))
					else:
						button["emoji"] = cdict(id=None, name=button["emoji"])
				elif not issubclass(type(button["emoji"]), collections.abc.Mapping):
					emoji = button["emoji"]
					button["emoji"] = cdict(name=emoji.name, id=emoji.id, animated=getattr(emoji, "animated", False))
			except KeyError:
				pass
			if "url" in button:
				button["style"] = 5
			elif "custom_id" not in button:
				if "id" in button:
					button["custom_id"] = button["id"]
				else:
					button["custom_id"] = custom_id = button.get("label")
					if not custom_id:
						if button.get("emoji"):
							button["custom_id"] = min_emoji(button["emoji"])
						else:
							button["custom_id"] = 0
			elif not isinstance(button["custom_id"], str):
				button["custom_id"] = as_str(button["custom_id"])
			if "custom_id" in button:
				while button["custom_id"] in used_custom_ids:
					if "?" in button["custom_id"]:
						spl = button["custom_id"].rsplit("?", 1)
						button["custom_id"] = spl[0] + f"?{int(spl[-1]) + 1}"
					else:
						button["custom_id"] = button["custom_id"] + "?0"
				used_custom_ids.add(button["custom_id"])
			if "style" not in button:
				button["style"] = 1
			if button.get("emoji"):
				if button["emoji"].get("label") == "▪️":
					button["disabled"] = True
			button.pop("id", None)
	return [dict(type=1, components=row) for row in buttons]

async def interaction_response(bot, message, content=None, embed=None, embeds=(), components=None, buttons=None, ephemeral=False):
	"Uses the raw Discord HTTP API to post/send an interaction message."
	if getattr(message, "deferred", False):
		return await interaction_post(bot, message, content=content, embed=embed, embeds=embeds, components=components, buttons=buttons, ephemeral=ephemeral)
	if hasattr(embed, "to_dict"):
		embed = embed.to_dict()
	if embed:
		embeds = astype(embeds, list)
		embeds.append(embed)
	if not getattr(message, "int_id", None):
		message.int_id = message.id
	if not getattr(message, "int_token", None):
		message.int_token = message.slash
	ephemeral = ephemeral and 64
	resp = await Request(
		f"https://discord.com/api/{api}/interactions/{message.int_id}/{message.int_token}/callback",
		data=json_dumps(dict(
			type=4,
			data=dict(
				flags=ephemeral,
				content=content,
				embeds=embeds,
				components=components or restructure_buttons(buttons),
			),
		)),
		method="POST",
		authorise=True,
		aio=True,
	)
	# print("INTERACTION_RESPONSE", resp)
	bot = BOT[0]
	if resp:
		if bot:
			M = bot.ExtendedMessage.new
		else:
			M = discord.Message
		message = M(state=bot._state, channel=message.channel, data=eval_json(resp))
		bot.add_message(message, files=False, force=True)
	return message

async def interaction_post(bot, message, content=None, embed=None, embeds=(), components=None, buttons=None, ephemeral=False):
	"Uses the raw Discord HTTP API to post/send a deferred interaction message."
	if hasattr(embed, "to_dict"):
		embed = embed.to_dict()
	if embed:
		embeds = astype(embeds, list)
		embeds.append(embed)
	if not getattr(message, "int_id", None):
		message.int_id = message.id
	if not getattr(message, "int_token", None):
		message.int_token = message.slash
	ephemeral = ephemeral and 64
	resp = await Request(
		f"https://discord.com/api/{api}/webhooks/{bot.id}/{message.int_token}",
		data=json_dumps(dict(
			flags=ephemeral,
			content=content,
			embeds=embeds,
			components=components or restructure_buttons(buttons),
		)),
		method="POST",
		authorise=True,
		aio=True,
	)
	# print("INTERACTION_POST", resp)
	bot = BOT[0]
	if resp:
		if bot:
			M = bot.ExtendedMessage.new
		else:
			M = discord.Message
		message = M(state=bot._state, channel=message.channel, data=eval_json(resp))
		bot.add_message(message, files=False, force=True)
	elif getattr(message, "simulated", False):
		message.content = content or message.content
		message.embeds = [discord.Embed.from_dict(embed)] if embed else message.embeds
	return message

async def interaction_patch(bot, message, content=None, embed=None, embeds=(), attachments=None, components=None, buttons=None, ephemeral=False):
	"Uses the raw Discord HTTP API to patch/edit an interaction message."
	if hasattr(embed, "to_dict"):
		embed = embed.to_dict()
	if embed:
		embeds = astype(embeds, list)
		embeds.append(embed)
	if not getattr(message, "int_id", None):
		message.int_id = message.id
	if not getattr(message, "int_token", None):
		message.int_token = message.slash
	mid = message.id or "@original"
	extra = {} if attachments is None else dict(attachments=attachments)
	if components or buttons:
		extra["components"] = components or restructure_buttons(buttons)
	resp = await Request(
		f"https://discord.com/api/{api}/webhooks/{bot.id}/{message.int_token}/messages/{mid}",
		data=json_dumps(dict(
			content=content,
			embeds=embeds,
			**extra,
		)),
		method="PATCH",
		authorise=True,
		aio=True,
	)
	# print("INTERACTION_PATCH", resp)
	bot = BOT[0]
	if resp:
		if bot:
			M = bot.ExtendedMessage.new
		else:
			M = discord.Message
		message = M(state=bot._state, channel=message.channel, data=eval_json(resp))
		bot.add_message(message, files=False, force=True)
	elif getattr(message, "simulated", False):
		message.content = content or message.content
		message.embeds = [discord.Embed.from_dict(embed)] if embed else message.embeds
	return message


channel_repr = lambda s: as_str(s) if not isinstance(s, discord.abc.GuildChannel) else str(s)


def line_count(fn):
	"Counts the number of lines in a file."
	with open(fn, "r", encoding="utf-8") as f:
		data = f.read()
	return (len(data), data.count("\n") + 1)

# Checks if a file is a python code file using its filename extension.
is_code = lambda fn: str(fn).endswith(".py") or str(fn).endswith(".pyw")

if os.name == "nt":
	def get_folder_size(path="."):
		s = subprocess.check_output(f'dir /a /w /s "{path}"', shell=True)
		spl = s.splitlines()
		finfo = spl[-2].strip().decode("ascii")
		# print(finfo)
		fc, fs = finfo.split("File(s)")
		fc = int(fc)
		fs = int(fs.removesuffix("bytes").replace(",", "").strip())
		return fc, fs
else:
	def get_folder_size(path="."):
		fc = fs = 0
		for dirpath, dirnames, filenames in os.walk(path):
			fc += len(filenames)
			for f in filenames:
				fs += os.path.getsize(os.path.join(dirpath, f))
		return fc, fs


# Recursively iterates through an iterable finding coroutines and executing them.
async def recursive_coro(item):
	if not issubclass(type(item), collections.abc.MutableSequence):
		return item
	for i, obj in enumerate(item):
		if awaitable(obj):
			if not issubclass(type(obj), asyncio.Task):
				item[i] = csubmit(obj)
		elif issubclass(type(obj), collections.abc.MutableSequence):
			item[i] = csubmit(recursive_coro(obj))
	for i, obj in enumerate(item):
		if hasattr(obj, "__await__"):
			with suppress():
				item[i] = await obj
	return item


is_channel = lambda channel: isinstance(channel, discord.abc.GuildChannel) or isinstance(channel, discord.abc.PrivateChannel) or isinstance(channel, discord.Thread) or getattr(channel, "is_channel", False)
is_guild = lambda guild: isinstance(guild, discord.Guild) or isinstance(guild, discord.PartialInviteGuild)

def is_nsfw(channel):
	try:
		return channel.is_nsfw()
	except AttributeError:
		return False

class CompatFile(discord.File):
	"A discord.File implementation that is compatible with files, bytes, memoryview, and buffer objects."

	def __init__(self, fp, filename=None, description=None, spoiler=False):
		if type(fp) in (bytes, memoryview):
			fp = io.BytesIO(fp)
		self.fp = self._fp = fp
		if isinstance(fp, io.IOBase):
			self.fp = fp
			self._original_pos = fp.tell()
			self._owner = False
		else:
			self.fp = open2(fp, "rb")
			self._original_pos = 0
			self._owner = True
		self._closer = self.fp.close
		self.fp.close = lambda: None
		if filename is None:
			if isinstance(fp, str):
				_, self.filename = os.path.split(fp)
			else:
				try:
					self.filename = fp.name
				except AttributeError:
					self.filename = ""
		else:
			self.filename = filename
		self.description = lim_str(description or self.filename or "", 1024) or None
		fn = self.filename
		if not fn:
			fn = "untitled." + get_ext(self.fp)
			self.reset()
		self.filename = lim_str(fn.strip().replace(" ", "_").translate(filetrans), 64)
		if spoiler:
			if self.filename:
				if not self.filename.startswith("SPOILER_"):
					self.filename = "SPOILER_" + self.filename
			else:
				self.filename = "SPOILER_" + "UNKNOWN"
		elif self.filename and self.filename.startswith("SPOILER_"):
			self.filename = self.filename[8:]
		self.name = self.filename
		self.clear = getattr(self.fp, "clear", lambda self: None)

	def reset(self, seek=True):
		if seek:
			try:
				self.fp.seek(self._original_pos)
			except ValueError:
				if not self._owner:
					raise
				self.fp = open2(self._fp, "rb")
				self._original_pos = 0
				self.fp.seek(self._original_pos)

	def close(self):
		self.fp.close = self._closer
		if self._owner:
			self._closer()


REPLY_SEM = cdict()
EDIT_SEM = cdict()
# noreply = discord.AllowedMentions(replied_user=False)

async def send_with_reply(channel, reference=None, content="", embed=None, embeds=None, tts=None, file=None, files=None, buttons=None, mention=False, ephemeral=False):
	if not channel:
		channel = reference.channel
	bot = BOT[0]
	if embed:
		embeds = (embed,) + tuple(embeds or ())
	if file:
		files = (file,) + tuple(files or ())
	if buttons:
		components = restructure_buttons(buttons)
	else:
		components = ()
	guild = getattr(channel, "guild", None)
	if guild and guild.me and not getattr(reference, "simulated", None) and not channel.permissions_for(guild.me).read_message_history:
		reference = None
	if getattr(reference, "slash", None):
		ephemeral = ephemeral and 64
		sem = emptyctx
		inter = True
		if getattr(reference, "deferred", False) or getattr(reference, "int_id", reference.id) in bot.inter_cache:
			url = f"https://discord.com/api/{api}/webhooks/{bot.id}/{bot.inter_cache.get(reference.id, reference.slash)}"
		else:
			url = f"https://discord.com/api/{api}/interactions/{reference.id}/{reference.slash}/callback"
		data = dict(
			type=4,
			data=dict(
				flags=ephemeral or 0,
			),
		)
		if content:
			data["data"]["content"] = content
		if embeds:
			data["data"]["embeds"] = [embed.to_dict() for embed in embeds]
		if components:
			data["data"]["components"] = components
	else:
		ephemeral = False
		fields = {}
		if embeds:
			fields["embeds"] = [embed.to_dict() for embed in embeds]
		if tts:
			fields["tts"] = tts
		if not (not reference or getattr(reference, "noref", None) or getattr(bot.messages.get(verify_id(reference)), "deleted", None) or getattr(channel, "simulated", None)): 
			if not getattr(reference, "to_message_reference_dict", None):
				if isinstance(reference, int):
					reference = cdict(to_message_reference_dict=eval(f"lambda: dict(message_id={reference})"))
				else:
					reference.to_message_reference_dict = lambda message: dict(message_id=message.id)
			fields["reference"] = reference
		if files:
			fields["files"] = files
		if not buttons and (not embeds or len(embeds) <= 1) and getattr(channel, "send", None):
			if embeds:
				fields["embed"] = next(iter(embeds))
			fields.pop("embeds", None)
			try:
				return await channel.send(content, **fields)
			except discord.HTTPException as ex:
				if fields.get("reference") and "Unknown message" in str(ex):
					fields.pop("reference")
					if fields.get("files"):
						for file in fields["files"]:
							file.reset()
					return await channel.send(content, **fields)
				raise
			except (aiohttp.client_exceptions.ClientOSError):
				await asyncio.sleep(random.random() * 2 + 1)
				if fields.get("files"):
					for file in fields["files"]:
						file.reset()
				return await channel.send(content, **fields)
			except discord.Forbidden:
				print(channel.id, channel)
				raise
		try:
			sem = REPLY_SEM[channel.id]
		except KeyError:
			sem = None
		if not sem:
			# g_id = channel.guild.id if getattr(channel, "guild", None) else None
			# bucket = f"{channel.id}:{g_id}:" + "/channels/{channel_id}/messages"
			try:
				try:
					sem = REPLY_SEM[channel.id]
				except KeyError:
					# bucket = f"{channel.id}:None:" + "/channels/{channel_id}/messages"
					sem = REPLY_SEM[channel.id]
			except KeyError:
				# print_exc()
				sem = REPLY_SEM[channel.id] = Semaphore(5, buffer=256, rate_limit=5.15)
		inter = False
		url = f"https://discord.com/api/{api}/channels/{channel.id}/messages"
		if getattr(channel, "dm_channel", None):
			channel = channel.dm_channel
		elif getattr(channel, "send", None) and getattr(channel, "guild", None) and channel.guild.me and not channel.permissions_for(channel.guild.me).read_message_history:
			fields = {}
			if embeds:
				fields["embeds"] = [embed.to_dict() for embed in embeds]
			if tts:
				fields["tts"] = tts
			return await channel.send(content, **fields)
		data = dict(
			content=content,
			allowed_mentions=dict(parse=["users"], replied_user=mention)
		)
		if reference:
			data["message_reference"] = dict(message_id=verify_id(reference))
		if components:
			data["components"] = components
		if embeds:
			data["embeds"] = [embed.to_dict() for embed in embeds]
		if tts is not None:
			data["tts"] = tts
		if getattr(channel, "simulated", False):
			return await channel.send(content, **fields)
	body = json_dumps(data)
	exc = RuntimeError("Unknown error occured.")
	if bot:
		M = bot.ExtendedMessage.new
	else:
		M = discord.Message
	method = "post"
	for i in range(xrand(3, 6)):
		try:
			if getattr(reference, "slash", False) and "webhooks" in url:
				body = json_dumps(data["data"])
			if files:
				form = aiohttp.FormData()
				for i, f in enumerate(files):
					f.reset()
					b = f.fp.read()
					form.add_field(
						name=f"files[{i}]",
						filename=f.filename,
						value=io.BytesIO(b),
						content_type=magic.from_buffer(b),
					)
					f.reset()
					# if "data" in data:
					#     data["data"].setdefault("attachments", []).append(dict(id=i, description=".", filename=f.filename))
				form.add_field(
					name="payload_json",
					value=json_dumps(data).decode("utf-8", "replace"),
					content_type="application/json",
				)
				body = form
			async with sem:
				resp = await Request(
					url,
					method=method,
					data=body,
					authorise=True,
					aio=True,
				)
		except Exception as ex:
			exc = ex
			if isinstance(ex, ConnectionError) and int(ex.args[0]) in range(400, 500):
				if not inter:
					print_exc()
				elif ex.errno == 404:
					continue
				elif ex.errno == 400 and "Interaction has already been acknowledged." in repr(ex):
					slash = bot.inter_cache.get(reference.id, reference.slash)
					url = f"https://discord.com/api/{api}/webhooks/{bot.id}/{slash}/messages/@original"
					method = "patch"
					body = json_dumps(data["data"])
					print("Retrying interaction:", url, method, body)
					resp = await Request(
						url,
						method=method,
						data=body,
						authorise=True,
						aio=True,
					)
					message = M(state=bot._state, channel=channel, data=eval_json(resp))
					if ephemeral:
						message.id = reference.id
						message.slash = getattr(reference, "slash", None)
						message.ephemeral = True
					for a in message.attachments:
						print("<attachment>", a.url)
					return message
				print_exc()
				print("Broken interaction:", url, repr(ex), data)
				fields = {}
				if files:
					fields["files"] = files
				if embeds:
					fields["embeds"] = embeds
				if tts:
					fields["tts"] = tts
				message = await discord.abc.Messageable.send(channel, content, **fields)
				for a in message.attachments:
					print("<attachment>", a.url)
				return message
			if isinstance(ex, SemaphoreOverflowError):
				print("send_with_reply:", repr(ex))
			else:
				print_exc()
		else:
			if not resp:
				if url.endswith("/callback") and hasattr(reference, "slash"):
					url = f"https://discord.com/api/{api}/webhooks/{bot.id}/{reference.slash}/messages/@original"
					resp = await Request(
						url,
						method="GET",
						authorise=True,
						aio=True,
					)
				if not resp:
					return
			message = M(state=bot._state, channel=channel, data=eval_json(resp))
			if ephemeral:
				message.id = reference.id
				message.slash = getattr(reference, "slash", None)
				message.ephemeral = True
			for a in message.attachments:
				print("<attachment>", a.url)
			return message
		await asyncio.sleep(i + 1)
	print("Maximum attempts exceeded:", url, method)
	raise exc

async def manual_edit(message, **fields):
	if not fields.get("buttons"):
		fields.pop("buttons", None)
		return await message.edit(**fields)
	if fields.get("embeds"):
		fields["embeds"] = [embed.to_dict() for embed in fields["embeds"]]
		if fields.get("embed"):
			fields["embeds"].insert(0, fields["embed"].to_dict())
	elif fields.get("embed"):
		fields["embed"] = fields["embed"].to_dict()
	if fields.get("buttons"):
		fields["components"] = restructure_buttons(fields.pop("buttons"))
	if fields.get("files"):
		files = fields.pop("files")
		if fields.get("file"):
			files.insert(0, fields.pop("file"))
	elif fields.get("file"):
		files = [fields.pop("file")]
	else:
		files = None
	if fields.get("allowed_mentions"):
		mentions = fields["allowed_mentions"]
		parse = []
		if mentions.users:
			parse.append("users")
		if mentions.roles:
			parse.append("roles")
		if mentions.everyone:
			parse.append("everyone")
		fields["allowed_mentions"] = dict(
			parse=parse,
			replied_user=mentions.replied_user,
		)
	channel = message.channel
	try:
		sem = EDIT_SEM[channel.id]
	except KeyError:
		sem = None
	if not sem:
		try:
			try:
				sem = EDIT_SEM[channel.id]
			except KeyError:
				sem = EDIT_SEM[channel.id]
		except KeyError:
			sem = EDIT_SEM[channel.id] = Semaphore(5, buffer=256, rate_limit=5.15)
	url = f"https://discord.com/api/{api}/channels/{channel.id}/messages/{message.id}"
	data = fields
	body = json_dumps(data)
	method = "patch"
	if files:
		form = aiohttp.FormData()
		for i, f in enumerate(files):
			f.reset()
			b = f.fp.read()
			form.add_field(
				name=f"files[{i}]",
				filename=f.filename,
				value=io.BytesIO(b),
				content_type=magic.from_buffer(b),
			)
			f.reset()
			# if "data" in data:
			#     data["data"].setdefault("attachments", []).append(dict(id=i, description=".", filename=f.filename))
		form.add_field(
			name="payload_json",
			value=json_dumps(data).decode("utf-8", "replace"),
			content_type="application/json",
		)
		body = form
	async with sem:
		resp = await Request(
			url,
			method=method,
			data=body,
			authorise=True,
			aio=True,
		)
	message._update(eval_json(resp))
	return message

async def add_reacts(message, reacts):
	if not reacts or not getattr_chain(message, "guild.me.guild_permissions.add_reactions", True):
		return message
	futs = []
	if reacts and not getattr(message, "ephemeral", False):
		tempsem = Semaphore(5, inf, rate_limit=5)
		for react in reacts:
			async with tempsem:
				with tracebacksuppressor:
					await message.add_reaction(react)
				# async with Delay(0.25):
				# 	fut = csubmit(aretry(message.add_reaction, react))
				# 	futs.append(fut)
	for fut in futs:
		await fut
	return message

# Sends a message to a channel, then adds reactions accordingly.
async def send_with_react(channel, *args, reacts=None, reference=None, mention=False, **kwargs):
	try:
		if reference or "buttons" in kwargs or "embeds" in kwargs:
			sent = await send_with_reply(channel, reference, *args, mention=mention, **kwargs)
		elif getattr(channel, "simulated", False):
			sent = await channel.send(*args, **kwargs)
		else:
			sent = await discord.abc.Messageable.send(channel, *args, **kwargs)
		await add_reacts(sent, reacts)
		return sent
	except:
		print_exc()
		raise


voice_channels = lambda guild: [channel for channel in guild.channels if getattr(channel, "type", None) in (discord.ChannelType.voice, discord.ChannelType.stage_voice)]

async def select_voice_channel(user, channel):
	# Attempt to match user's currently connected voice channel
	if getattr(user, "voice", None):
		return user.voice.channel
	if not user.guild:
		raise LookupError("Unable to find voice channel.")
	user = await user.guild.fetch_member(user.id)
	user.guild._members[user.id] = user
	voice = user.voice
	if voice:
		return voice.channel
	# Otherwise attempt to find closest voice channel to current text channel
	catg = channel.category
	if catg is not None:
		channels = voice_channels(catg)
	else:
		channels = None
	if not channels:
		member = user.guild.me
		pos = 0 if channel.category is None else channel.category.position
		# Sort by distance from text channel
		channels = sorted(tuple(channel for channel in voice_channels(channel.guild) if channel.permissions_for(member).connect and channel.permissions_for(member).speak and channel.permissions_for(member).use_voice_activation), key=lambda channel: (abs(pos - (channel.position if channel.category is None else channel.category.position)), abs(channel.position)))
	if channels:
		vc = channels[0]
	else:
		raise LookupError("Unable to find voice channel.")
	return vc


# Creates and starts a coroutine for typing in a channel.
typing = lambda self: csubmit(self.trigger_typing())


# Gets the string representation of a url object with the maximum allowed image size for discord, replacing png with webp format when possible.
def to_webp(url):
	if not isinstance(url, str):
		url = str(url)
	if url.startswith("https://cdn.discordapp.com/embed/avatars/"):
		return url.replace("/media.discordapp.net/", "/cdn.discordapp.com/").replace(".webp", ".png")
	if url.endswith("?size=1024"):
		url = url[:-10] + "?size=4096"
	if "/embed/" not in url[:48]:
		url = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
	return url.replace(".png", ".webp")

def to_webp_ex(url):
	if not isinstance(url, str):
		url = str(url)
	if url.startswith("https://cdn.discordapp.com/embed/avatars/"):
		return url.replace("/media.discordapp.net/", "/cdn.discordapp.com/").replace(".webp", ".png")
	if url.endswith("?size=1024"):
		url = url[:-10] + "?size=256"
	if "/embed/" not in url[:48]:
		url = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
	return url.replace(".png", ".webp")

BASE_LOGO = "https://cdn.discordapp.com/embed/avatars/0.png"
def get_url(obj, f=to_webp) -> str:
	if isinstance(obj, str):
		return obj
	found = False
	for attr in ("display_avatar", "avatar_url", "icon_url", "icon", "avatar"):
		try:
			url = getattr(obj, attr)
		except AttributeError:
			continue
		found = True
		if url:
			if url == "0":
				return BASE_LOGO
			return f(url)
	if found:
		return BASE_LOGO

# Finds the best URL for a Discord object's icon, prioritizing proxy_url for images if applicable.
proxy_url = lambda obj: get_url(obj) or (obj.proxy_url if is_image(obj.proxy_url) else obj.url)
# Finds the best URL for a Discord object's icon.
best_url = lambda obj: get_url(obj) or getattr(obj, "url", None) or BASE_LOGO
# Finds the worst URL for a Discord object's icon.
worst_url = lambda obj: get_url(obj, to_webp_ex) or getattr(obj, "url", None) or BASE_LOGO

allow_gif = lambda url: url + ".gif" if "." not in url.rsplit("/", 1)[-1] and "?" not in url else url

def get_author(user, uid=None):
	url = best_url(user)
	bot = BOT[0]
	if bot and "proxies" in bot.data:
		url2 = bot.data.proxies.get(uuhash(url))
		if url2:
			url = url2
		else:
			bot.data.exec.cproxy(url)
	name = getattr(user, "display_name", None) or user.name
	if uid:
		name = f"{name} ({user.id})"
	return cdict(name=name, icon_url=url, url=url)

# Finds emojis and user mentions in a string.
find_emojis = lambda s: regexp(r"<a?:[A-Za-z0-9\-~_]+:[0-9]+>").findall(s)
find_users = lambda s: regexp(r"<@!?[0-9]+>").findall(s)


def min_emoji(emoji, full=False) -> str:
	if getattr(emoji, "unicode", None):
		return emoji.unicode
	if not getattr(emoji, "id", None):
		if getattr(emoji, "name", None):
			return emoji.name
		emoji = as_str(emoji)
		if emoji.isnumeric():
			return f"<:_:{emoji}>"
		return emoji
	name = T(emoji).get("name", "_") if full else "_"
	if emoji.animated:
		return f"<a:{name}:{emoji.id}>"
	return f"<:{name}:{emoji.id}>"


smileys = (
	"😀,😃,😄,😁,😆,🥹,😅,😂,🤣,🥲,☺️,😊,😇,🙂,🙃,😉,😌,😍,🥰,😘,😗,😙,😚,😋,😛,😝,😜,🤪,🤨,🧐,🤓,😎,🥸,🤩,🥳,"
	"😏,😒,😞,😔,😟,😕,🙁,☹️,😣,😖,😫,😩,🥺,😢,😭,😤,😠,😡,🤬,🤯,😳,🥵,🥶,😶‍🌫️,😱,😨,😰,😥,😓,🤗,🤔,🫣,🤭,🫢,🫡,"
	"🤫,🫠,🤥,😶,🫥,😐,🫤,😑,🫨,🙂‍↔️,🙂‍↕️,😬,🙄,😯,😦,😧,😮,😲,🥱,😴,🤤,😪,😮‍💨,😵,😵‍💫,🤐,🥴,🤢,🤮,🤧,😷,🤒,🤕,🤑,🤠,"
	"😈,👿,🤡"
).split(",")
def get_random_smiley():
	return choice(smileys)


def replace_map(s, mapping):
	temps = {k: chr(65535 - i) for i, k in enumerate(mapping.keys())}
	trans = "".maketrans({chr(65535 - i): mapping[k] for i, k in enumerate(mapping.keys())})
	for key, value in temps.items():
		s = s.replace(key, value)
	for key, value in mapping.items():
		s = s.replace(value, key)
	return s.translate(trans)


# You can easily tell I was the one to name this thing. 🍻 - smudgedpasta
def grammarly_2_point_0(string):
	s = " " + string.lower().replace("am i", "are y\uf000ou").replace("i am", "y\uf000ou are") + " "
	s = s.replace(" yours ", " mine ").replace(" mine ", " yo\uf000urs ").replace(" your ", " my ").replace(" my ", " yo\uf000ur ")
	s = replace_map(s.strip(), {
		"yourself": "myself",
		"are you": "am I",
		"you are": "I am",
		"you're": "i'm",
		"you'll": "i'll"
	})
	modal_verbs = "shall should shan't shalln't shouldn't must mustn't can could couldn't may might mightn't will would won't wouldn't have had haven't hadn't do did don't didn't"
	r1 = re.compile(f"(?:{modal_verbs.replace(' ', '|')}) you")
	r2 = re.compile(f"you (?:{modal_verbs.replace(' ', '|')})")
	while True:
		m = r1.search(s)
		if not m:
			m = r2.search(s)
			if not m:
				break
			s = s[:m.start()] + "I" + s[m.start() + 3:]
		else:
			s = s[:m.end() - 3] + "I" + s[m.end():]
	res = alist(s.split())
	for sym in "!.,'":
		if sym in s:
			for word, rep in {"you": "m\uf000e", "me": "you", "i": "I"}.items():
				src = word + sym
				dest = rep + sym
				if res[0] == src:
					res[0] = dest
				res.replace(src, dest)
	if res[0] == "you":
		res[0] = "I"
	s = " ".join(res.replace("you", "m\uf000e").replace("i", "you").replace("me", "you").replace("i", "I").replace("i'm", "I'm").replace("i'll", "I'll"))
	return s.replace("\uf000", "")

def grammarly_2_point_1(string):
	s = grammarly_2_point_0(string)
	return s[0].upper() + s[1:]


# Gets the last image referenced in a message.
def get_last_image(message, embeds=True):
	for a in reversed(message.attachments):
		url = a.url
		if is_image(url) is not None:
			return url
	if embeds:
		for e in reversed(message.embeds):
			if e.video:
				return e.video.url
			if e.image:
				return e.image.url
			if e.thumbnail:
				return e.thumbnail.url
	raise FileNotFoundError("Message has no image.")


# Gets the length of a message.
def get_message_length(message):
	return len(message.system_content or message.content) + sum(len(e) for e in message.embeds) + sum(len(a.url) for a in message.attachments)

def get_message_words(message):
	return word_count(message.system_content or message.content) + sum(word_count(e.description) if e.description else sum(word_count(f.name) + word_count(f.value) for f in e.fields) if e.fields else 0 for e in message.embeds) + len(message.attachments)

# Returns a string representation of a message object.
def message_repr(message, limit=1024, username=False, link=False):
	c = message.content
	s = getattr(message, "system_content", None)
	if s and len(s) > len(c):
		c = s
	if link:
		c = message_link(message) + "\n" + c
	if username:
		c = user_mention(message.author.id) + ":\n" + c
	data = lim_str(c, limit)
	if message.attachments:
		data += "\n[" + ", ".join(i.url for i in message.attachments) + "]"
	if message.embeds:
		data += "\n⟨" + ", ".join(str(i.to_dict()) for i in message.embeds) + "⟩"
	if message.reactions:
		data += "\n{" + ", ".join(str(i) for i in message.reactions) + "}"
	with suppress(AttributeError):
		t = message.created_at
		if message.edited_at:
			t = message.edited_at
		data += f"\n`({t})`"
	if not data:
		data = css_md(uni_str("[EMPTY MESSAGE]"), force=True)
	return lim_str(data, limit)

def message_link(message):
	try:
		return message.jump_url
	except AttributeError:
		pass
	guild = getattr(message, "guild", None)
	g_id = getattr(guild, "id", -1)
	return f"https://discord.com/channels/{g_id}/{message.channel.id}/{message.id}"

def fake_reply(message):
	"Simulates the appearance of a reply to a message. Required as Discord does not support replies for webhook messages, or messages from different channels."
	content = message.content
	if content.startswith("-# ⮣ ") and "\n" in content:
		content = content.split("\n", 1)[-1]
	if len(content) >= 100:
		content = content[:100] + "…"
	content = content.strip()
	if not content:
		content = "…" if not message.attachments and not message.embeds else "*Click to see attachment*"
	encoded = f"**{message.author.display_name}** {content}"
	encoded = no_links(encoded) # Disallow HTTP string prefixes as Discord blacklists them
	return f"-# ⮣ [{encoded}]({message_link(message)})"


def apply_stickers(message, data=None):
	"Applies stickers to a message based on its Discord data. Each individual sticker is treated as a separate embed."
	if not data and not getattr(message, "stickers", None):
		return message
	has = set()
	for e in getattr(message, "embeds", ()):
		if e.image:
			has.add(e.image.url)
		if e.thumbnail:
			has.add(e.thumbnail.url)
		has.add(e.url)
	for a in getattr(message, "attachments", ()):
		has.add(a.url)
	for s in getattr(message, "stickers", ()):
		url = s.url.replace("media.discordapp.net", "cdn.discordapp.com").replace(".webp", ".png")
		if url in has:
			continue
		has.add(url)
		emb = discord.Embed()
		emb.set_image(url=url)
		message.embeds.append(emb)
	if data and data.get("sticker_items"):
		for s in data["sticker_items"]:
			if s.get("format_type") == 3:
				url = f"https://discord.com/stickers/{s['id']}.json"
			else:
				url = f"https://cdn.discordapp.com/stickers/{s['id']}.png"
			if url in has:
				continue
			has.add(url)
			emb = discord.Embed()
			emb.set_image(url=url)
			message.embeds.append(emb)
	return message

try:
	EmptyEmbed = discord.embeds._EmptyEmbed
except AttributeError:
	EmptyEmbed = None

def add_embed_fields(emb, fields):
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
		v = lim_str(field[1], 1024)
		i = True if len(field) < 3 else bool(field[2])
		emb.add_field(name=n, value=v if v else "\u200b", inline=i)
	return emb

@functools.lru_cache(maxsize=4)
def as_embed(message, link=False):
	emb = discord.Embed(description="").set_author(**get_author(message.author))
	content = message.content or message.system_content
	if not content:
		if len(message.attachments) == 1:
			url = message.attachments[0].url
			if is_image(url):
				emb.url = url
				emb.set_image(url=url)
				if link:
					link = message_link(message)
					emb.description = lim_str(f"{emb.description}\n\n[View Message]({link})", 4096)
					emb.timestamp = message.edited_at or message.created_at
				return emb
		elif not message.attachments and len(message.embeds) == 1:
			emb2 = message.embeds[0]
			if emb2.description != EmptyEmbed and emb2.description:
				emb.description = emb2.description
			if emb2.title:
				emb.title = emb2.title
			if emb2.url:
				emb.url = emb2.url
			if emb2.image:
				emb.set_image(url=emb2.image.url)
			if emb2.thumbnail:
				emb.set_thumbnail(url=emb2.thumbnail.url)
			for f in emb2.fields:
				if f:
					emb.add_field(name=f.name, value=f.value, inline=getattr(f, "inline", True))
			if link:
				link = message_link(message)
				emb.description = lim_str(f"{emb.description}\n\n[View Message]({link})", 4096)
				emb.timestamp = message.edited_at or message.created_at
			return emb
	else:
		urls = find_urls(content)
	emb.description = content
	if len(message.embeds) > 1 or content:
		urls = chain(("(" + e.url + ")" for e in message.embeds[1:] if e.url), ("[" + best_url(a) + "]" for a in message.attachments))
		items = list(urls)
	else:
		items = None
	if items:
		if emb.description in items:
			emb.description = lim_str("\n".join(items), 4096)
		elif emb.description or items:
			emb.description = lim_str(emb.description + "\n" + "\n".join(items), 4096)
	image = None
	for a in message.attachments:
		url = a.url
		if is_image(url) is not None:
			image = url
	if not image and message.embeds:
		for e in message.embeds:
			if e.image:
				image = e.image.url
			if e.thumbnail:
				image = e.thumbnail.url
	if image:
		emb.url = image
		emb.set_image(url=image)
	for e in message.embeds:
		if len(emb.fields) >= 25:
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
		for f in e.fields:
			if len(emb.fields) >= 25:
				break
			if f:
				emb.add_field(name=f.name, value=f.value, inline=getattr(f, "inline", True))
		if len(emb) >= 6000:
			while len(emb) > 6000:
				emb.remove_field(-1)
			break
	if not emb.description:
		urls = chain(("(" + e.url + ")" for e in message.embeds if e.url), ("[" + best_url(a) + "]" for a in message.attachments))
		emb.description = lim_str("\n".join(urls), 4096)
	if link:
		link = message_link(message)
		emb.description = lim_str(f"{emb.description}\n\n[View Message]({link})", 4096)
		emb.timestamp = message.edited_at or message.created_at
	return emb

exc_repr = lambda ex: lim_str(py_md(f"Error: {repr(ex).replace('`', '')}"), 2000)

# Returns a string representation of an activity object.
def activity_repr(activity):
	if hasattr(activity, "type") and activity.type != discord.ActivityType.custom:
		t = activity.type.name
		if t == "listening":
			t += " to"
		return f"{t.capitalize()} {activity.name}"
	return str(activity)


# Alphanumeric string regular expression.
is_alphanumeric = lambda string: string.replace(" ", "").isalnum()
to_alphanumeric = lambda string: single_space(regexp("[^a-z 0-9]+", re.I).sub(" ", unicode_prune(string)))
is_numeric = lambda string: regexp("[0-9]").search(string) and not regexp("[a-z]", re.I).search(string)


# Strips code box from the start and end of a message.
def strip_code_box(s):
	if s.startswith("```") and s.endswith("```"):
		s = s[s.index("\n") + 1:-3]
	return s


# A string lookup operation with an iterable, multiple attempts, and sorts by priority.
async def str_lookup(it, query, ikey=lambda x: [str(x)], qkey=lambda x: [str(x)], loose=True, fuzzy=0.5):
	queries = qkey(query)
	qlist = [q for q in queries if q]
	if not qlist:
		qlist = list(queries)
	cache = [[[nan, None], [nan, None]] for _ in qlist]
	for x, i in enumerate(shuffle(it), 1):
		for c in ikey(i):
			if not c and i:
				continue
			if fuzzy:
				for a, b in enumerate(qkey(c)):
					match = string_similarity(qlist[a], b)
					if match >= 1:
						return i
					elif match >= fuzzy and not match <= cache[a][0][0]:
						cache[a][0] = [match, i]
			else:
				for a, b in enumerate(qkey(c)):
					if b == qlist[a]:
						return i
		if not x & 2047:
			await asyncio.sleep(0.1)
	for c in cache:
		if c[0][0] < inf:
			return c[0][1]
	if loose and not fuzzy:
		for c in cache:
			if c[1][0] < inf:
				return c[1][1]
	raise LookupError(f"No results for {query}.")


# Queries for searching members

def userQuery1(x):
	yield str(x)

def userIter1(x):
	yield str(x)

def userQuery2(x):
	yield str(x)

def userIter2(x):
	yield str(x.name)
	if T(x).get("global_name"):
		yield str(x.global_name)
	if T(x).get("nick"):
		yield str(x.nick)


# Generates a random colour across the spectrum, in intervals of 128.
rand_colour = lambda: colour2raw(hue2colour(xrand(12) * 128))


base_colours = cdict(
	black=(0,) * 3,
	white=(255,) * 3,
	grey=(127,) * 3,
	gray=(127,) * 3,
	dark_grey=(64,) * 3,
	dark_gray=(64,) * 3,
	light_grey=(191,) * 3,
	light_gray=(191,) * 3,
	silver=(191,) * 3,
)
primary_secondary_colours = cdict(
	red=(255, 0, 0),
	green=(0, 255, 0),
	blue=(0, 0, 255),
	yellow=(255, 255, 0),
	cyan=(0, 255, 255),
	aqua=(0, 255, 255),
	magenta=(255, 0, 255),
	fuchsia=(255, 0, 255),
)
tertiary_colours = cdict(
	orange=(255, 127, 0),
	chartreuse=(127, 255, 0),
	lime=(127, 255, 0),
	lime_green=(127, 255, 0),
	spring_green=(0, 255, 127),
	azure=(0, 127, 255),
	violet=(127, 0, 255),
	rose=(255, 0, 127),
	dark_red=(127, 0, 0),
	maroon=(127, 0, 0),
)
colour_shades = cdict(
	dark_green=(0, 127, 0),
	dark_blue=(0, 0, 127),
	navy_blue=(0, 0, 127),
	dark_yellow=(127, 127, 0),
	dark_cyan=(0, 127, 127),
	teal=(0, 127, 127),
	dark_magenta=(127, 0, 127),
	dark_orange=(127, 64, 0),
	brown=(127, 64, 0),
	dark_chartreuse=(64, 127, 0),
	dark_spring_green=(0, 127, 64),
	dark_azure=(0, 64, 127),
	dark_violet=(64, 0, 127),
	dark_rose=(127, 0, 64),
	light_red=(255, 127, 127),
	peach=(255, 127, 127),
	light_green=(127, 255, 127),
	light_blue=(127, 127, 255),
	light_yellow=(255, 255, 127),
	light_cyan=(127, 255, 255),
	turquoise=(127, 255, 255),
	light_magenta=(255, 127, 255),
	light_orange=(255, 191, 127),
	light_chartreuse=(191, 255, 127),
	light_spring_green=(127, 255, 191),
	light_azure=(127, 191, 255),
	sky_blue=(127, 191, 255),
	light_violet=(191, 127, 255),
	purple=(191, 127, 255),
	light_rose=(255, 127, 191),
	pink=(255, 127, 191),
)
colour_types = (
	colour_shades,
	base_colours,
	primary_secondary_colours,
	tertiary_colours,
)

colourlist_cache = AutoCache(f"{CACHE_PATH}/colourlist", stale=86400 * 7, timeout=86400 * 30)
@tracebacksuppressor
def get_colour_list():
	global colour_names
	colour_names = cdict()
	resp = Request("https://en.wikipedia.org/wiki/List_of_colors_(compact)", decode=True, timeout=None)
	resp = resp.split('<span class="mw-headline" id="List_of_colors">List of colors</span>', 1)[-1].split("</h3>", 1)[-1].split("<h2>", 1)[0]
	n = len("background-color:rgb")
	while resp:
		try:
			i = resp.index("background-color:rgb")
		except ValueError:
			break
		colour, resp = resp[i + n:].split(";", 1)
		colour = literal_eval(colour)
		resp = resp.split("<a ", 1)[-1].split(">", 1)[-1]
		name, resp = resp.split("<", 1)
		name = full_prune(name).strip().replace(" ", "_")
		if "(" in name and ")" in name:
			name = (name.split("(", 1)[0] + name.rsplit(")", 1)[-1]).strip("_")
			if name in colour_names:
				continue
		colour_names[name] = colour
	for colour_group in colour_types:
		if colour_group:
			if not colour_names:
				colour_names = cdict(colour_group)
			else:
				colour_names.update(colour_group)
	print(f"Successfully loaded {len(colour_names)} colour names.")
def load_colour_list():
	global colour_names
	colour_names = colourlist_cache.retrieve("map", get_colour_list)
	return colour_names

def parse_colour(s, default=None):
	ishex = isdec = False
	if none(c.isnumeric() for c in s):
		pass
	elif s.startswith("0x"):
		s = s[2:]
		ishex = True
	elif s.startswith("#"):
		s = s[1:]
		ishex = True
	elif s.endswith("d"):
		s = s[:-1]
		isdec = True
	else:
		s = s.removeprefix("rgb").removeprefix("a")
	s = single_space(s.replace(",", " ").strip("()[]{}<>")).strip()
	# Try to parse as colour tuple first
	if not s:
		if default is None:
			raise ArgumentError("Missing required colour argument.")
		return default
	try:
		return colour_names[full_prune(s).replace(" ", "_")]
	except KeyError:
		pass
	if not ishex and not isdec and " " in s:
		channels = [min(255, max(0, round_min(i))) for i in s.split(" ")[:5] if i]
		if len(channels) not in (3, 4):
			raise ArgumentError("Please input 3 or 4 channels for colour input.")
		return channels
	try:
		raw = int(s, 16) if not isdec else round(float(s))
		if len(s) <= 3:
			channels = [raw >> 8 & 15, raw >> 4 & 15, raw & 15]
			channels = [x << 4 | x for x in channels]
		elif len(s) <= 6:
			channels = [raw >> 16 & 255, raw >> 8 & 255, raw & 255]
		elif len(s) <= 8:
			channels = [raw >> 16 & 255, raw >> 8 & 255, raw & 255, raw >> 24 & 255]
		else:
			raise ValueError
	except ValueError:
		raise ArgumentError("Please input a valid colour identifier.")
	return channels


# A translator to stip all characters from mentions.
__imap = {
	"#": "",
	"<": "",
	">": "",
	"@": "",
	"!": "",
	"&": "",
	":": "",
}
__itrans = "".maketrans(__imap)

def verify_id(obj):
	if isinstance(obj, int):
		return obj
	if isinstance(obj, str):
		with suppress(ValueError):
			return int(obj.rsplit(">", 1)[0].rsplit(":", 1)[-1].translate(__itrans))
		return obj
	with suppress(AttributeError):
		return obj.recipient.id
	with suppress(AttributeError):
		return obj.id
	with suppress(AttributeError):
		return obj.value
	return int(obj)


status_text = {
	discord.Status.online: "Online",
	discord.Status.idle: "Idle",
	discord.Status.dnd: "DND",
	discord.Status.invisible: "Invisible",
	discord.Status.offline: "Offline",
}
status_icon = {
	discord.Status.online: "🟢",
	discord.Status.idle: "🟡",
	discord.Status.dnd: "🔴",
	discord.Status.invisible: "⚫",
	discord.Status.offline: "⚫",
}
status_order = tuple(status_text)


# Subprocess pool for resource-consuming operations.
PROCS = {}
PROCS_BY_CAPS = {}

# Gets amount of processes running in pool.
sub_count = lambda: sum(is_strict_running(p) for p in PROCS.values())


proc_args = (python, "-m", "misc.x_compute")

COMPUTE_LOAD = AUTH.get("compute_load", []).copy()
COMPUTE_POT = COMPUTE_LOAD.copy()
COMPUTE_ORDER = AUTH.get("compute_order", []).copy()
if len(COMPUTE_LOAD) < DC:
	handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
	gcore = [pynvml.nvmlDeviceGetNumGpuCores(d) for d in handles]
	COMPUTE_LOAD = AUTH["compute_load"] = gcore
	COMPUTE_POT = [i * 100 for i in gcore]
	COMPUTE_ORDER = list(range(DC))
else:
	COMPUTE_LOAD = COMPUTE_LOAD[:DC]
	COMPUTE_POT = COMPUTE_POT[:DC]
	COMPUTE_ORDER = COMPUTE_ORDER[:DC]
if COMPUTE_LOAD:
	total = sum(COMPUTE_LOAD)
	if total != 1:
		COMPUTE_LOAD = [i / total for i in COMPUTE_LOAD]
	if __name__ == "__main__":
		print("Compute load distribution:", COMPUTE_LOAD)
		print("Compute pool order:", COMPUTE_ORDER)

async def start_proc(n, di=(), caps="image", it=0, wait=False, timeout=None):
	if hasattr(n, "caps"):
		n, di, caps, it = n.n, n.di, n.caps, it + 1
	if n in PROCS:
		proc = PROCS[n]
		if is_strict_running(proc):
			it = max(it, proc.it + 1)
			proc.pipe.kill()
		elif PROCS[n] is False:
			return
		for c in caps:
			PROCS_BY_CAPS[c].remove(proc)
		PROCS[n] = False
	port = await asubmit(get_free_port)
	args = proc_args
	for c in caps:
		args = AUTH.get("cap_versions", {}).get(c) or args
	args = list(args)
	args.append(str(port))
	args.append(",".join(map(str, di)))
	args.append(",".join(caps))
	args.append(json_dumps(COMPUTE_LOAD).decode("ascii"))
	properties = [torch.cuda.get_device_properties(i) for i in range(DC)]
	args.append(json_dumps([(p.major, p.minor) for p in properties]).decode("ascii"))
	args.append(json_dumps(COMPUTE_ORDER).decode("ascii"))
	args.append(str(it))
	pipe = await asubmit(
		EvalPipe.connect,
		args,
		port,
		glob=globals(),
		independent=False,
	)
	proc = pipe.proc
	proc.n = n
	proc.di = di
	proc.caps = caps
	proc.it = it
	# proc.is_running = lambda: not proc.returncode
	proc.sem = Semaphore(8, inf)
	PROCS[n] = proc
	proc.pipe = pipe
	for c in caps:
		PROCS_BY_CAPS.setdefault(c, []).append(proc)
	return proc

async def restart_workers():
	futs = []
	for proc in PROCS.values():
		futs.append(start_proc(proc))
	return await gather(*futs)


IS_MAIN = True
FIRST_LOAD = True
def spec2cap(skip=False):
	"Automatically calculates list of capabilities from device specs. Uses benchmark results if available."
	global FIRST_LOAD
	g = 1073741824
	if not skip and FIRST_LOAD:
		try:
			from multiprocessing import shared_memory
			globals()["MEM_LOCK"] = shared_memory.SharedMemory(name="X-DISTRIBUTE", create=True, size=1)
		except FileExistsError:
			if IS_MAIN:
				raise
			return
		FIRST_LOAD = False
	caps = [[]]
	cc = psutil.cpu_count()
	mc = cc + 1 >> 1
	ram = psutil.virtual_memory().total
	try:
		subprocess.run("ffmpeg")
	except FileNotFoundError:
		ffmpeg = False
	else:
		ffmpeg = True
	done = []
	try:
		import pynvml
		pynvml.nvmlInit()
		handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
		rrams = [pynvml.nvmlDeviceGetMemoryInfo(d).total for d in handles]
		bwidths = [g / 8 / 8 * 2 ** pynvml.nvmlDeviceGetCurrPcieLinkGeneration(d) * pynvml.nvmlDeviceGetCurrPcieLinkWidth(d) for d in handles]
	except Exception:
		rrams = bwidths = []
	vrams = tuple(rrams)
	if vrams and AUTH.get("discord_token") and AUTH.get("reserved_vram"):
		rrams = [max(0, v - r * g) for v, r in zip(vrams, AUTH["reserved_vram"])]
	if os.name == "nt" and ram > 3 * g:
		caps.append("browse")
	if len(caps) > 1:
		if not IS_MAIN:
			yield caps + ["remote"]
		else:
			yield caps + ["host"]
		if cc > 1:
			yield caps
	rm = 1
	while mc > 1:
		caps = [[], "math"]
		if cc > 3 and ram > (rm * 8 - 2) * g and ffmpeg:
			caps.append("image")
		if cc > 5 and ram > (rm * 14 - 2) * g:
			caps.append("caption")
		mc -= 1
		rm += 1
		yield caps
	if not DC:
		return
	for i, (v, w) in enumerate(zip(rrams, bwidths)):
		c = COMPUTE_POT[i]
		caps = [[i]]
		if c > 100000 and vrams[i] > 1 * g and ffmpeg and w > 64 * 1048576:
			caps.append("video")
			caps.append("ecdc")
		if c > 400000 and v >= 7 * g and v < 13 * g and w < 256 * 1048576:
			caps.append("sdxl")
			v -= 11 * g
		elif c > 400000 and v > 11 * g and (v <= 19 * g or "scc" not in done):
			# if "sdxl" not in done or c <= 600000:
			caps.append("scc")
			done.append("scc")
			v -= 11 * g
		elif c > 400000 and v > 9 * g:
			caps.append("sdxl")
			if v > 19 * g:
				caps.append("scc")
				done.append("scc")
				caps.append("sd")
				v -= 19 * g
			else:
				v -= 9 * g
		# elif c > 400000 and IS_MAIN and vrams[i] > 15 * g:
		# 	caps.append("sdxl")
		# 	if vrams[i] > 19 * g:
		# 		caps.append("scc")
		# 	caps.append("nvram")
		# 	v -= 15 * g
		# elif c > 400000 and IS_MAIN and "scc" not in done and vrams[i] > 11 * g:
		# 	caps.append("scc")
		# 	caps.append("nvram")
		# 	done.append("scc")
		# 	v -= 11 * g
		# if c > 200000 and v > 6 * g:
		# 	if "whisper" not in done or c <= 600000:
		# 		caps.append("whisper")
		# 		done.append("whisper")
		# 		v -= 6 * g
		if c > 200000 and v > 5 * g:
			if "sd" not in done or c <= 600000:
				caps.append("sd")
				done.append("sd")
				v -= 5 * g
		# if c > 200000 and vrams[i] > 4 * g and rrams[i] > g:
		# 	caps.append("summ")
		# 	done.append("summ")
			# v -= 1 * g
		# if v <= 4 * g:
			# v = 0
		# vrams[i] = v
		if len(caps) > 1:
			yield caps


def proc_start():
	"Starts designated subprocesses using computed device specifications and capabilities."
	if torch and os.environ.get("AI_FEATURES", True):
		globals()["DC"] = torch.cuda.device_count()
		COMPUTE_LOAD = AUTH.get("compute_load", []).copy()
		if len(COMPUTE_LOAD) < DC:
			COMPUTE_LOAD = AUTH["compute_load"] = [torch.cuda.get_device_properties(i).multi_processor_count for i in range(torch.cuda.device_count())]
		elif len(COMPUTE_LOAD) > DC:
			COMPUTE_LOAD = COMPUTE_LOAD[:DC]
		if COMPUTE_LOAD:
			total = sum(COMPUTE_LOAD)
			if total != 1:
				COMPUTE_LOAD = [i / total for i in COMPUTE_LOAD]
			print("Compute load distribution:", COMPUTE_LOAD)
	else:
		COMPUTE_LOAD = ()
		globals()["DC"] = 0
	CAPS = globals().get("SCAP", [])
	# if not CAPS:
	with tracebacksuppressor:
		CAPS = globals()["SCAP"] = list(spec2cap())
	print("CAPS:", CAPS)
	for n, (di, *caps) in enumerate(tuple(CAPS)):
		csubmit(start_proc(n, di, caps))
		time.sleep(2)
		if "load" in caps:
			CAPS.pop(n)

def device_cap(i, resolve=False):
	"Gets scaled CUDA compute capability."
	di = torch.cuda.get_device_capability(i)
	if resolve:
		return 1.15 ** (di[0] * 10 + di[1])
	return di


async def proc_eval(s, caps=["math"], priority=False, timeout=12):
	procs = PROCS_BY_CAPS[caps[0]]
	for p in procs:
		if not set(caps).difference(p.caps):
			break
	else:
		raise RuntimeError("No suitable worker process for task.")
	try:
		fut = csubmit(p.pipe.asubmit(s, priority=priority))
		return await asyncio.wait_for(fut, timeout=timeout)
	except (T0, T1, T2):
		print(f"Process {p} timed out, restarting!")
		csubmit(start_proc(p))
		raise

def process_math(expr, prec=64, rational=False, timeout=12, variables=None, retries=0):
	"Sends an operation to the math subprocess pool."
	return proc_eval(f"x_math.procResp(x_math.evalSym({repr(expr)},{repr(prec)},{repr(rational)},{repr(variables)}))", caps=["math"], timeout=timeout)

def process_image(image, operation="$", args=[], cap="image", priority=False, timeout=60, retries=1):
	"Sends an operation to the image subprocess pool."
	args = astype(args, list)
	for i, a in enumerate(args):
		if type(a) is mpf:
			args[i] = float(a)
		elif type(a) in (list, deque, np.ndarray, dict):
			try:
				args[i] = "orjson.loads(" + as_str(json_dumps(as_str(json_dumps(a)))) + ")"
			except (TypeError, orjson.JSONDecodeError):
				args[i] = "pickle.loads(" + repr(pickle.dumps(a)) + ")"

	def as_arg(arg):
		if isinstance(arg, str) and (arg.startswith("pickle.loads(") or arg.startswith("orjson.loads(")):
			return arg
		return repr(arg)

	argi = "[" + ",".join(map(as_arg, args)) + "]"
	return proc_eval(f"evaluate_image([{repr(image)},{repr(operation)},{argi}])", caps=[cap], priority=priority, timeout=timeout)


@tracebacksuppressor
def exec_tb(s, *args, **kwargs):
	"Executes a string as code, providing a traceback upon exception."
	exec(s, *args, **kwargs)


emoji_translate = {}
emoji_replace = {}
em_trans = {}
discord_stripped = RangeSet([range(0x2000, 0x2070), range(0xfe00, 0xffff)])
discord_stripmap = "".maketrans({k: "" for k in discord_stripped})
emoji_cache = AutoCache(f"{CACHE_PATH}/follow", stale=86400 * 7, timeout=86400 * 30)
_eop = "\n    query vendorHistoricEmojiV1(\n      $slug: Slug!\n      $version: Slug = null\n      $status: VendorHistoricEmojiStatus = null\n      $lang: Language\n    ) {\n      vendorHistoricEmoji_v1(slug: $slug, version: $version, status: $status, lang: $lang) {\n        ...vendorHistoricEmojiResource\n      }\n    }\n    \n  fragment vendorHistoricEmojiImageFragment on VendorHistoricEmojiImage {\n    slug\n    image {\n      source\n      description\n      useOriginalImage\n    }\n    status\n  }\n\n    \n  fragment vendorHistoricEmojiResource on VendorHistoricEmoji {\n    items {\n      category {\n        slug\n        title\n\n        representingEmoji {\n          code\n        }\n      }\n      images {\n        ...vendorHistoricEmojiImageFragment\n      }\n    }\n    statuses\n  }\n\n  "
def request_emojis():
	emojimap = {}
	for slug, version in (("twitter", "twemoji-15.0.3"), ("discord", "15.1")):
		data = Request(
			"https://emojipedia.org/api/graphql",
			data=json_dumps(dict(
				operationName="vendorHistoricEmojiV1",
				query=_eop,
				variables=dict(
					lang="EN",
					slug="twitter",
					version="twemoji-15.0.3",
				),
			)),
			headers={"Content-Type": "application/json"},
			method="POST",
			json=True,
			timeout=None,
		)
		for category in data["data"]["vendorHistoricEmoji_v1"]["items"]:
			for emoji in category["images"]:
				name = emoji["slug"]
				source = emoji["image"]["source"]
				url = "https://em-content.zobj.net/" + source
				e_id = source.rsplit(".", 1)[0].rsplit("_", 1)[-1]
				e = "".join(chr(int(i, 16)) for i in e_id.split("-"))
				emojimap[e] = (name, url)
				e2 = e.translate(discord_stripmap)
				if e2 and not e2.isascii():
					emojimap[e2] = (name, url)
	return emojimap
def load_emojilist():
	return emoji_cache.retrieve("map", request_emojis)
@tracebacksuppressor
def load_emojis():
	global emoji_translate, emoji_replace, em_trans
	emap = load_emojilist()
	for k, v in emap.items():
		if len(k) == 1:
			emoji_translate[k] = v[1] + " "
		else:
			emoji_replace[k] = v[1] + " "
	em_trans = "".maketrans(emoji_translate)
	print(f"Successfully loaded {len(emap)} unicode emojis.")


@functools.lru_cache(maxsize=4)
def translate_emojis(s):
	res = s.translate(em_trans)
	if res in emoji_replace:
		return emoji_replace[res]
	return res

@functools.lru_cache(maxsize=4)
def replace_emojis(s):
	"Replaces all emojis in a string with their respective URLs"
	for emoji, url in emoji_replace.items():
		if emoji in s:
			s = s.replace(emoji, url)
	return s

@functools.lru_cache(maxsize=64)
def find_emojis_ex(s, cast_urls=True):
	"Finds all emojis in a string, both unicode and discord-exclusive representations. Prioritises multi-character emojis if possible."
	out = {}
	for emoji, url in emoji_replace.items():
		try:
			i = s.index(emoji)
		except ValueError:
			continue
		if cast_urls:
			out[i] = url.rstrip()
		else:
			out[i] = emoji
		for j in range(1, len(out[i])):
			out[i + j] = None
	for i, c in enumerate(s):
		if i in out:
			continue
		try:
			url = emoji_translate[c]
		except KeyError:
			continue
		if cast_urls:
			out[i] = url.rstrip()
		else:
			out[i] = c
	found = find_emojis(s)
	for e in found:
		i = s.index(e)
		if cast_urls:
			eid = verify_id(e)
			url = f"https://cdn.discordapp.com/emojis/{eid}.webp"
			if e.startswith("<:a:"):
				url += "?animated=true"
			out[i] = url
			continue
		out.setdefault(i, e)
	return [t[1] for t in sorted(out.items()) if t[1]]

HEARTS = ["❤️", "🧡", "💛", "💚", "💙", "💜", "💗", "💞", "🤍", "🖤", "🤎", "❣️", "💕", "💖"]

class SimulatedEmoji(cdict):

	def __str__(self):
		if self.get("unicode"):
			return self.unicode
		return min_emoji(self, full=True)

	@property
	def url(self):
		if self.get("unicode"):
			urls = find_emojis_ex(self.unicode)
			return urls[0] if urls else "https://mizabot.xyz/notfound.png"
		return f"https://cdn.discordapp.com/emojis/{self.id}.{'gif' if self.animated else 'png'}"


readstring = lambda s: deobfuscate(zwremove(s))


# Default and standard command categories to enable.
basic_commands = frozenset(("main", "string", "admin"))
standard_commands = default_commands = basic_commands.union(("voice", "image", "webhook", "fun", "ai"))
visible_commands = default_commands.union(("nsfw",))


class Importable(collections.abc.Hashable, collections.abc.Callable):
	pass


class Command(Importable):
	"Basic abstract inheritable class for all bot commands."
	description = ""
	schema = None
	server_only = False
	nsfw = False
	usage = ""
	min_level = 0
	rate_limit = (2, 3)

	@classmethod
	def perm_error(cls, perm, req=None, reason=None):
		if req is None:
			req = cls.min_level
		if reason is None:
			reason = f"for command {cls.__name__}"
		if isinstance(req, str):
			pass
		elif not req <= inf:
			req = "nan (Bot Owner)"
		elif req >= inf:
			req = "inf (Administrator)"
		elif req >= 3:
			req = f"{req} (Moderator: Ban Members or Manage Channels/Server)"
		elif req >= 2:
			req = f"{req} (Helper: Manage Messages/Threads/Nicknames/Roles/Webhooks/Emojis/Events)"
		elif req >= 1:
			req = f"{req} (Trusted: View Audit Log/Server Insights or Move/Mute/Deafen Members or Mention Everyone)"
		elif req >= 0:
			req = f"{req} (Member)"
		else:
			req = f"{req} (Guest)"
		return PermissionError(f"Insufficient priviliges {reason}. Required level: {req}, Current level: {perm}.")

	@classmethod
	def convert_name(cls, name):
		return name.replace("*", "").replace("_", "").replace("||", "")

	def __init__(self, bot, catg):
		self.used = {}
		if not hasattr(self, "data"):
			self.data = cdict()
		if not hasattr(self, "min_display"):
			self.min_display = self.min_level
		if not hasattr(self, "name"):
			self.name = []
		self.__name__ = self.__class__.__name__
		if not hasattr(self, "alias"):
			self.alias = self.name
		else:
			self.alias.insert(0, self.parse_name())
		self.macros = getattr(self, "macros", {})
		self.macromap = fcdict(self.macros)
		self.name.insert(0, self.parse_name())
		self.name.extend(self.macros)
		self.aliases = {full_prune(alias).replace("*", "").replace("_", "").replace("||", ""): alias for alias in self.alias}
		self.aliases.pop("", None)
		for a in self.aliases:
			if a in bot.commands:
				bot.commands[a].add(self)
			else:
				bot.commands[a] = alist((self,))
		self.catg = self.category = catg
		self.bot = bot
		self._globals = bot._globals
		f = getattr(self, "__load__", None)
		if callable(f):
			try:
				f()
			except Exception:
				print_exc()
				self.data.clear()
				f()

	__hash__ = lambda self: hash(self.parse_name()) ^ hash(self.category)
	__str__ = lambda self: f"Command <{self.parse_name()}>"
	__call__ = lambda self, **void: None

	parse_name = lambda self: self.__name__.strip("_")
	parse_description = lambda self: self.description.replace('⟨BOT⟩', self.bot.user.name).replace('⟨WEBSERVER⟩', self.bot.webserver)

	def parse_usage(self):
		schema = self.schema
		if not schema:
			return f"{self.parse_name()} {self.usage}"
		s = self.parse_name()
		for k, v in schema.items():
			s += f"\n{colourise('<', fg='white')}{colourise(k, fg='yellow')}{colourise(':', fg='white')}{colourise(v.type, fg='magenta')}{colourise('>', fg='white')}"
			desc = []
			if v.get("required"):
				desc.append(colourise("Required", fg="red"))
			if v.get("description"):
				desc.append(colourise(v.description))
				if v.get("validation"):
					valid = v.validation if isinstance(v.validation, str) else (colourise(",") + colourise(" ", fg="cyan")).join(v.validation.enum)
					desc.append(f"{colourise('Allowed', fg='red')}{colourise(':', fg='white')} {colourise(valid, fg='cyan')}")
				elif v.get("default"):
					desc.append(f"{colourise('Default', fg='red')}{colourise(':', fg='white')} {colourise(json_if(v.default), fg='cyan')}")
			if desc:
				d = colourise("; ", fg="white").join(desc)
				s += " " + "\n".join(split_text(f"{colourise('(', fg='blue')}{d}{colourise(')', fg='blue')}", max_length=936))
		return s

	def parse_example(self):
		schema = self.schema
		if not schema:
			if getattr(self, "example", None):
				if isinstance(self.example, str):
					return self.example
				return choice(self.example)
			return self.parse_name().casefold()
		has_string = any(v.get("type") == "string" for v in schema.values())
		s = self.parse_name().casefold()
		values = []
		u = set()
		hgs = self.has_greedy_string()
		for k, v in reversed(schema.items()):
			val = v.get("example") or v.get("default")
			if not val:
				if v.get("type") not in ("bool", "enum"):
					continue
				if v["type"] == "enum":
					val = v["validation"]["enum"][-1]
			k2 = k[0]
			if v.get("type") == "bool":
				if v.get("default"):
					values.append(f"{colourise('--no-' + k, fg='blue')}")
				else:
					values.append(f"{colourise('--' + k, fg='blue')}")
				u.add(k2)
			elif v.get("type") == "enum":
				values.append(f"{colourise('--' + val if hgs else val, fg='blue' if hgs else 'magenta')}")
				u.add(k2)
			elif not has_string and v.get("type") in ("url", "image", "visual", "video", "audio", "media", "filesize", "resolution", "index", "text", "string"):
				values.append(f"{colourise(json_if(val))}")
				u.add(k2)
			# elif k2 not in u:
			# 	values.append(f"{colourise('-' + k2, fg='cyan')} {colourise(val)}")
			# 	u.add(k2)
			else:
				values.append(f"{colourise('--' + k, fg='cyan')} {colourise(json_if(val))}")
		values.append(s)
		return " ".join(reversed(values))

	def has_greedy_string(self):
		schema = self.schema
		if not schema:
			return False
		for v in schema.values():
			if v.get("type") in ("string", "datetime", "timedelta") and v.get("greedy", True):
				return True
		return False

	def unload(self):
		bot = self.bot
		for alias in itertools.chain(self.alias, self.macromap):
			alias = self.convert_name(alias)
			coms = bot.commands.get(alias)
			if coms:
				coms.remove(self)
				print("unloaded", alias, "from", self)
			if not coms:
				bot.commands.pop(alias, None)

	def encode(self, *args):
		raise NotImplementedError

	def decode(self, *args):
		raise NotImplementedError


def default_pagination_key(curr, pos=0, page=16):
	return iter2str(
		tuple(curr)[pos:pos + page],
		left="`【",
		right="】`",
		offset=pos,
	).strip()

class PaginationCommand(Command):

	directions = [b'\xe2\x8f\xaa', b'\xe2\x97\x80', b'\xe2\x96\xb6', b'\xe2\x8f\xa9', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	dirmap = dict(zip(dirnames, map(as_str, directions)))

	def encode(self, uid: int, data: bytes, s: str) -> str:
		s = s.strip()
		code = invisicode.encode(self.__name__.encode("utf-8") + b"\x00" + leb128(uid) + as_bytes(data))
		if (s.startswith("```") or s.startswith("*```")) and "\n" in s:
			x, y = s.split("\n", 1)
			if not y or not y[0].isascii():
				code += "\u200b"
			return x + "\n" + code + y
		if not s or not s[0].isascii():
			code += "\u200b"
		return code + s

	@staticmethod
	def decode(s: str) -> tuple:
		if s.startswith("```") or s.startswith("*```"):
			s = s.split("\n", 2)[1]
		if not s:
			raise ValueError(s)
		for i, c in enumerate(s):
			n = ord(c)
			if not invisicode.BASE <= n < invisicode.BASE + invisicode.RANGE and n != invisicode.STRINGPREFIX:
				break
		code = invisicode.decode(s[:i])
		name, code = code.split(b"\x00", 1)
		return (as_str(name), *decode_leb128(code))

	def react_perms(self, perm: int):
		return None

	def construct(self, uid, data, content="", **kwargs):
		if not content:
			embed = kwargs.get("embed")
			if embed:
				embed.description = self.encode(uid, data, embed.description or "")
			else:
				content = self.encode(uid, data, content)
		else:
			content = self.encode(uid, data, content)
		if "buttons" not in kwargs:
			kwargs["buttons"] = [[cdict(emoji=self.dirmap[k], name=k, custom_id=self.dirmap[k]) for k in tup] for tup in [("First", "Prev", "Refresh", "Next", "Last")]]
			#[cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
		return cdict(
			content=content or None,
			**kwargs,
		)

	async def _callback_(self, _user, **void):
		return self.construct(_user.id, b"", "`PLACEHOLDER`")

	async def default_display(self, name, uid, pos, curr, diridx=-1, extra=b"", key=default_pagination_key, akey=None, page_size=16):
		bot = self.bot
		user = await bot.fetch_user(uid)
		page = page_size
		pos = self.paginate(pos, len(curr), page, diridx)
		colour = await self.bot.get_colour(user)
		emb = discord.Embed(
			colour=colour,
		).set_author(**get_author(user))
		s = "s" if len(curr) != 1 else ""
		if not curr:
			emb.title = f"No {name}{s}."
		else:
			emb.title = f"{len(curr)} {name}{s}:"
			if akey:
				emb.description = await akey(curr, pos, page)
			else:
				emb.description = key(curr, pos, page)
		max_page = len(curr) // page + 1
		if max_page > 1:
			curr_page = min(max_page, ceil(pos / page)) + 1
			emb.set_footer(text=f"Page {curr_page}/{max_page}")
		return self.construct(uid, leb128(pos) + extra, embed=emb)

	def paginate(self, pos, size, page, diridx=-1):
		last = max(0, size - page)
		match diridx:
			case 0:
				pos = 0
			case 1:
				pos = max(0, pos - page)
			case 2:
				pos = min(last, pos + page)
			case 3:
				pos = last
		return pos


class Database(Importable, collections.abc.MutableMapping):
	"Basic abstract inheritable class for all bot databases."
	bot = None
	rate_limit = 3
	name = "data"
	encode = None
	decode = None
	automut = True
	no_file = False

	def __init__(self, bot, catg):
		self.name
		name = self.name
		self.__name__ = self.__class__.__name__
		fhp = "saves/" + name
		if self.no_file:
			self.data = {}
		else:
			self.data = FileHashDict(path=fhp, encode=self.encode, decode=self.decode, automut=self.automut)
		self.fixup()
		bot.database[name] = bot.data[name] = self
		self.catg = self.category = catg
		self.bot = bot
		self._semaphore = Semaphore(1, 1, rate_limit=self.rate_limit)
		self._garbage_semaphore = Semaphore(1, 0, rate_limit=self.rate_limit * 3 + 30)
		self._globals = globals()
		f = getattr(self, "__load__", None)
		if callable(f):
			try:
				f()
			except Exception:
				print_exc()

	def fixup(self):
		pass

	__hash__ = lambda self: hash(self.__name__)
	__str__ = lambda self: f"Database <{self.__name__}>"
	__call__ = lambda self: None
	__len__ = lambda self: len(self.data)
	__iter__ = lambda self: iter(self.data)
	__contains__ = lambda self, k: k in self.data
	__eq__ = lambda self, other: self.data == other
	__ne__ = lambda self, other: self.data != other

	def __setitem__(self, k, v):
		self.data[k] = v
		return self
	def __getitem__(self, k):
		return self.data[k]
	def __delitem__(self, k):
		return self.data.__delitem__(k)
	@property
	def db(self):
		return self.data.db

	keys = lambda self: self.data.keys()
	items = lambda self: self.data.items()
	values = lambda self: self.data.values()
	get = lambda self, *args, **kwargs: self.data.get(*args, **kwargs)
	coerce = lambda self, *args, **kwargs: self.data.coerce(*args, **kwargs)
	pop = lambda self, *args, **kwargs: self.data.pop(*args, **kwargs)
	popitem = lambda self, *args, **kwargs: self.data.popitem(*args, **kwargs)
	fill = lambda self, other: self.data.fill(other)
	clear = lambda self: self.data.clear()
	setdefault = lambda self, k, v: self.data.setdefault(k, v)
	coercedefault = lambda self, *args, **kwargs: coercedefault(self.data, *args, **kwargs)
	updatedefault = lambda self, *args, **kwargs: updatedefault(self.data, *args, **kwargs)
	keys = lambda self: self.data.keys()
	discard = lambda self, k: self.data.pop(k, None)
	vacuum = lambda self: self.data.vacuum() if hasattr(self.data, "vacuum") else None

	def sync(self, modified=None, force=False):
		if self.no_file:
			return
		if force:
			try:
				limit = getattr(self, "limit", None)
				if limit and len(self) > limit:
					print(f"{self} overflowed by {len(self) - limit}, dropping...")
					with tracebacksuppressor:
						while len(self) > limit:
							self.pop(next(iter(self)))
				self.data.sync()
			except Exception:
				print(self, traceback.format_exc(), sep="\n", end="")
		else:
			if modified is None:
				self.data.modified.update(self.data.keys())
			else:
				if issubclass(type(modified), collections.abc.Sized) and type(modified) not in (str, bytes):
					self.data.modified.update(modified)
				else:
					self.data.modified.add(modified)
			self.data.modify()
		return False

	def unload(self):
		self.unloaded = True
		bot = self.bot
		func = getattr(self, "_destroy_", None)
		if callable(func):
			await_fut(asubmit(func, priority=True))
		for f in dir(self):
			if f.startswith("_") and f[-1] == "_" and f[1] != "_":
				func = getattr(self, f, None)
				if callable(func):
					bot.events[f].remove(func)
					print("unloaded", f, "from", self)
		self.sync(force=True)
		bot.data.pop(self, None)
		bot.database.pop(self, None)
		if isinstance(self.data, FileHashDict):
			self.data.unload()


class ImagePool:
	schema = cdict(
		embed=cdict(
			type="bool",
			description="Whether to send the message as an embed",
			default=True,
		),
	)
	rate_limit = (0.05, 0.25)
	threshold = 1024

	async def __call__(self, bot, embed=True, **void):
		url = await bot.data.imagepools.get(self.database, self.fetch_one, self.threshold)
		return await self.send(url, embed=embed)

	async def send(self, url, embed=True):
		if embed:
			emb = await self.bot.random_embed(url)
			return cdict(embed=emb)
		return cdict(url=url)


def get_nvml():
	import pynvml
	pynvml.nvmlInit()
	dc = pynvml.nvmlDeviceGetCount()
	handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
	gname = [pynvml.nvmlDeviceGetName(d) for d in handles]
	gcore = [pynvml.nvmlDeviceGetNumGpuCores(d) for d in handles]
	gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) for d in handles]
	gutil = [pynvml.nvmlDeviceGetUtilizationRates(d) for d in handles]
	gpowa = [pynvml.nvmlDeviceGetPowerUsage(d) for d in handles]
	gpowb = [pynvml.nvmlDeviceGetEnforcedPowerLimit(d) for d in handles]
	gtempa = [pynvml.nvmlDeviceGetTemperature(d, 0) for d in handles]
	gtempb = [pynvml.nvmlDeviceGetTemperatureThreshold(d, 0) for d in handles]
	return gname, gcore, gmems, gutil, gpowa, gpowb, gtempa, gtempb

WMT = 0
WMV = 0
def get_wmem(mused=0):
	global WMT, WMV
	t = utc()
	if t - WMT > 60:
		try:
			f1 = esubmit(subprocess.check_output, "wmic OS get TotalVirtualMemorySize /Value")
			fvms = subprocess.check_output("wmic OS get FreeVirtualMemory /Value")
			tvms = f1.result()
			tvms = int(tvms.strip().decode("ascii").removeprefix("TotalVirtualMemorySize="))
			fvms = int(fvms.strip().decode("ascii").removeprefix("FreeVirtualMemory="))
			WMV = (tvms - fvms) * 1024 - mused
		except Exception:
			WMV = 0
		WMT = utc()
	return WMV

_cpuinfo = _diskinfo = None
_ctime = _dtime = 0
def get_current_stats(up_bps, down_bps):
	global WMI, _cpuinfo, _ctime, _diskinfo, _dtime
	import psutil
	t = utc()
	cinfo = _cpuinfo
	if t - _ctime > 3600:
		_ctime = t
		import cpuinfo
		cinfo = _cpuinfo = cpuinfo.get_cpu_info()
	f1 = psutil.cpu_percent()
	f2 = psutil.virtual_memory()
	f3 = psutil.swap_memory()
	try:
		gname, gcore, gmems, gutil, gpowa, gpowb, gtempa, gtempb = get_nvml()
	except Exception:
		gname = []
	dinfo = _diskinfo
	if t - _dtime > 60:
		_dtime = t
		dinfo = _diskinfo = {}
		for p in psutil.disk_partitions(all=False):
			try:
				dinfo[p.mountpoint] = psutil.disk_usage(p.mountpoint)
			except OSError:
				pass
	cpercent, minfo, sinfo = f1, f2, f3
	ip = "127.0.0.1"
	if os.name == "nt":
		cswap = get_wmem(minfo.used)
		if cswap > sinfo.used:
			class mtemp:
				def __init__(self, used, total):
					self.used, self.total = used, total
			sinfo = mtemp(used=cswap, total=sinfo.total)
	ram_name = globals().get("RAM_NAME") or "RAM"
	if os.name == "nt" and not globals().get("WMI"):
		try:
			import wmi
			globals()["WMI"] = WMI = wmi.WMI()
		except Exception:
			traceback.print_exc()
			globals()["WMI"] = False
	if globals().get("WMI") is not False:
		if ram_name == "RAM":
			if not globals().get("wRAM"):  
				ram = globals()["wRAM"] = WMI.Win32_PhysicalMemory()[0]
			else:
				ram = globals()["wRAM"]
			ram_speed = ram.ConfiguredClockSpeed
			ram_type = ram.SMBIOSMemoryType
			try:
				ram_class = {
					2: "DRAM",
					5: "EDO",
					9: "RAM",
					10: "ROM",
					20: "DDR1",
					21: "DDR2",
					24: "DDR3",
					26: "DDR4",
					34: "DDR5",
					35: "DDR5",
				}[ram_type]
			except KeyError:
				ram_class = "DDR" + str(max(1, ceil(log2(ram_speed / 250))))
			ram_name = globals()["RAM_NAME"] = f"{ram_class}-{ram_speed}"
	return dict(
		cpu={ip: dict(name=cinfo["brand_raw"], count=cinfo["count"], usage=cpercent / 100, max=1, time=t)},
		gpu={f"{ip}-{i}": dict(
			name=name,
			count=gcore[i],
			usage=gutil[i].gpu / 100,
			max=1,
			time=t,
		) for i, name in enumerate(gname)},
		memory={
			f"{ip}-v": dict(name=ram_name, count=1, usage=minfo.used, max=minfo.total, time=t),
			f"{ip}-s": dict(name="Swap", count=1, usage=sinfo.used, max=sinfo.total, time=t),
			**{f"{ip}-{i}": dict(
				name=name,
				count=1,
				usage=gmems[i].used,
				max=gmems[i].total,
				time=t,
			) for i, name in enumerate(gname)},
		},
		disk={f"{ip}-{k}": dict(name=k, count=1, usage=v.used, max=v.total, time=t) for k, v in dinfo.items()},
		network={
			f"{ip}-u": dict(name="Upstream", count=1, usage=up_bps, max=-1, time=t),
			f"{ip}-d": dict(name="Downstream", count=1, usage=down_bps, max=-1, time=t),
		},
		power={
			**{f"{ip}-{i}": dict(
				name=name,
				count=1,
				usage=gpowa[i] / 1000,
				max=gpowb[i] / 1000,
				time=t,
			) for i, name in enumerate(gname)},
		},
		temperature={
			**{f"{ip}-{i}": dict(
				name=name,
				count=1,
				usage=gtempa[i],
				max=gtempb[i],
				time=t,
			) for i, name in enumerate(gname)},
		},
	)


if __name__ != "__mp_main__":
	class __logPrinter:
		"Forwards all print operations to target files/callables, limiting the amount of operations that can occur in any given amount of time for efficiency."

		def __init__(self, file=None, archive=None, archive_size=8 * 1048576):
			self.buffer = self
			self.data = {}
			self.history = {}
			self.counts = collections.Counter()
			self.funcs = alist()
			self.file = file
			self.archive = archive
			self.archive_size = archive_size
			self.closed = True
			self.strbuff = []

		def start(self):
			self.thread = tsubmit(self.update_print)
			self.closed = False

		def file_print(self, fn, b):
			try:
				trunc = False
				if isinstance(b, (tuple, list)):
					b = ("" if isinstance(b, str) else b"").join(b)
				if self.archive and isinstance(fn, str) and os.path.exists(fn) and os.path.getsize(fn) >= self.archive_size:
					arcname = str(datetime.datetime.now()).replace(" ", "_", 1).rsplit(".", 1)[0].replace(":", ".", 2) + ".log"
					with zipfile.ZipFile(self.archive, "a", compression=zipfile.ZIP_LZMA) as z:
						if arcname in z.namelist():
							pass
						else:
							z.write(fn, arcname)
							trunc = True
				if not isinstance(fn, (str, bytes)):
					f = fn
				elif isinstance(b, byte_like):
					f = open(fn, "ab+")
				elif isinstance(b, str):
					f = open(fn, "a+", encoding="utf-8")
				else:
					f = fn
				with contextlib.closing(f):
					if trunc:
						f.truncate(0)
					try:
						f.write(b)
					except TypeError:
						try:
							f.write(as_str(b))
						except ValueError:
							pass
			except Exception:
				sys.__stdout__.write(traceback.format_exc())

		def flush(self):
			outfunc = lambda s: self.file_print(self.file, s)
			enc = lambda x: x.encode("utf-8")
			try:
				for f, v in tuple(self.data.items()):
					s = "".join(v)
					v.clear()
					out = lim_str(s, 65536)
					data = enc(s)
					if self.funcs and out.strip():
						[func(out) for func in self.funcs]
					if f == self.file:
						outfunc(data)
					else:
						self.file_print(f, data)
			except Exception:
				sys.__stdout__.write(traceback.format_exc())

		def update_print(self):
			if self.file is None:
				return
			while True:
				for i in range(40):
					s, self.strbuff = self.strbuff, []
					sys.__stdout__.write("".join(s))
					time.sleep(0.25)
				self.flush()
				while not os.path.exists("misc/common.py") or self.closed:
					time.sleep(1)

		def __call__(self, *args, sep=" ", end="\n", prefix="", file=None, **void):
			out = str(sep).join(i if isinstance(i, str) else str(i) for i in args) + str(end) + str(prefix)
			if not out:
				return
			temp = out.strip()
			if self.closed:
				return self.strbuff.append(out)
			if file is None:
				file = self.file
			if temp:
				if file in self.history and self.history.get(file).strip() == temp:
					self.counts[file] += 1
					return
				elif self.counts.get(file):
					count = self.counts.pop(file)
					times = "s" if count != 1 else ""
					out, self.history[file] = f"<Last message repeated {count} time{times}>\n{out}", out
				else:
					self.history[file] = out
					self.counts.pop(file, None)
			try:
				self.data[file].append(out)
			except KeyError:
				self.data[file] = [out]
			return self.strbuff.append(out)

		def write(self, *args, end="", **kwargs):
			args2 = [as_str(arg) for arg in args]
			return self.__call__(*args2, end=end, **kwargs)

		read = lambda self, *args, **kwargs: bytes()
		close = lambda self, force=False: self.__setattr__("closed", force)
		isatty = lambda self: False

	PRINT = __logPrinter(AUTH["log_path"], AUTH["log_store"])