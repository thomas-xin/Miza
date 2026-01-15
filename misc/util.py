import ast
import asyncio
import base64
import base65536
import collections
from collections import deque, defaultdict
import concurrent.futures
import contextlib
import datetime
import diskcache
import fractions
import functools
import hashlib
import html
import io
import itertools
import json
from math import ceil, comb, inf, isfinite, isqrt, log10
import multiprocessing.connection
import multiprocessing.shared_memory
import os
os.environ["PYTHONUTF8"] = "1"
import pickle
import random
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from traceback import format_exc, print_exc
from urllib.parse import quote_plus, unquote_plus
import urllib.request
import zipfile
import aiofiles
import aiohttp
from dynamic_dt import DynamicDT
import filetype
import invisicode
import nacl.secret
import numpy as np
import orjson
import psutil
import pynvml
import niquests
import requests
from misc.smath import predict_next, display_to_precision, unicode_prune, full_prune
from misc.types import ISE, CCE, Dummy, PropagateTraceback, is_exception, alist, cdict, fcdict, as_bytes, as_str, lim_str, single_space, try_int, round_min, regexp, suppress, loop, safe_eval, number, byte_like, json_like, hashable_args, always_copy, astype, MemoryBytes, ts_us, utc, tracebacksuppressor, T, coerce, coercedefault, updatedefault, json_dumps, json_dumpstr, pretty_json, MultiEncoder # noqa: F401
from misc.asyncs import await_fut, wrap_future, awaitable, reflatten, asubmit, csubmit, esubmit, tsubmit, Future, Semaphore

print("UTIL:", __name__)


python = sys.executable

with open("auth.json", "rb") as f:
	AUTH = cdict(eval(f.read(), dict(true=True, false=False, null=None)))
cachedir = AUTH.get("cache_path") or None
if cachedir:
	# print(f"Setting model cache {cachedir}...")
	os.environ["HF_HOME"] = f"{cachedir}/huggingface"
	os.environ["TORCH_HOME"] = f"{cachedir}/torch"
	os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cachedir}/huggingface/hub"
	os.environ["TRANSFORMERS_CACHE"] = f"{cachedir}/huggingface/transformers"
	os.environ["HF_DATASETS_CACHE"] = f"{cachedir}/huggingface/datasets"
else:
	cachedir = os.path.expanduser("~") + "/.cache"
	os.makedirs(cachedir, exist_ok=True)

CACHE_PATH = cachedir + "/cache"
os.makedirs(CACHE_PATH, exist_ok=True)
TEMP_PATH = AUTH.get("temp_path")
if not TEMP_PATH or not os.path.exists(TEMP_PATH):
	TEMP_PATH = os.path.abspath("cache")
	os.makedirs(TEMP_PATH, exist_ok=True)
FAST_PATH = os.path.abspath("cache")
os.makedirs(FAST_PATH, exist_ok=True)
assert isinstance(CACHE_PATH, str)
assert isinstance(TEMP_PATH, str)
assert isinstance(FAST_PATH, str)

persistdir = AUTH.get("persist_path") or cachedir
ecdc_dir = persistdir + "/ecdc/"

DEBUG = astype(AUTH.get("debug", ()), frozenset)

PORT = AUTH.get("webserver_port", 80)
if PORT:
	PORT = int(PORT)
IND = "\x7f"

compat_python = AUTH.get("compat_python") or python
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0 AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36 Edg/134.0.3124.85"

@tracebacksuppressor
def save_file(data, fn):
	if isinstance(data, byte_like):
		data = io.BytesIO(data)
	with open(fn, "wb") as f:
		shutil.copyfileobj(data, f)

async def save_file_a(data, fn):
	async with aiofiles.open(fn, "wb") as f:
		if hasattr(data, "read"):
			data = await asubmit(data.read)
		await f.write(data)

async def read_file_a(fn):
	async with aiofiles.open(fn, "rb") as f:
		return await f.read()

_globals = globals()
def save_auth(auth):
	globals()["AUTH"].update(auth)
	_globals["AUTH"].update(auth)
	data = pretty_json(AUTH)
	with open("auth.json", "w", encoding="utf-8") as f:
		f.write(data)


DC = 0
try:
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
except Exception:
	print_exc()
hwaccel = "cuda" if DC else "d3d11va" if os.name == "nt" else "auto"


api = "v10"


def print_class(obj):
	s = f"<{obj.__class__.__name__} object at 0x{hex(id(obj))[2:].upper()}"
	atts = []
	for k in dir(obj):
		if k.startswith("__") and k.endswith("__"):
			continue
		try:
			v = getattr(obj, k)
		except Exception as ex:
			v = repr(ex)
		if callable(v):
			continue
		atts.append(f"{k}={v}")
	if not atts:
		return s + ">"
	return s + "; " + ", ".join(atts) + ">"

async def require_predicate(predicate):
	if not predicate:
		return
	res = predicate()
	if not res:
		raise CCE("Predicate cancelled.")
	if awaitable(res):
		res = await res
		if not res:
			raise CCE("Predicate cancelled.")
	return res

def is_strict_running(proc):
	"Detects if a process is truly running. Zombie processes are treated as dead."
	if not proc:
		return
	try:
		if T(proc).get("returncode") is not None:
			return False
		if hasattr(proc, "poll") and proc.poll() is not None:
			return False
		if not proc.is_running():
			return False
		try:
			if os.name != "nt" and proc.status() == "zombie":
				proc.wait()
				return
		except (ProcessLookupError, psutil.NoSuchProcess):
			return
		return True
	except AttributeError:
		try:
			proc = psutil.Process(proc.pid)
		except (ProcessLookupError, psutil.NoSuchProcess):
			return
		except Exception:
			print_exc()
			return
	if not proc.is_running():
		return False
	try:
		if os.name != "nt" and proc.status() == "zombie":
			proc.wait()
			return
	except (ProcessLookupError, psutil.NoSuchProcess):
		return
	return True

@tracebacksuppressor(psutil.NoSuchProcess)
def force_kill(proc):
	"Force kills a process, trying all available methods. Does not error if process does not exist."
	if not proc:
		return
	if T(proc).get("fut") and not proc.fut.done():
		with tracebacksuppressor:
			proc.fut.set_exception(CCE("Response disconnected. If this error occurs during a command, it is likely due to maintenance!"))
	if isinstance(proc, EvalPipe):
		return proc.terminate()
	killed = deque()
	if not callable(T(proc).get("children")):
		proc = psutil.Process(proc.pid)
	for child in proc.children(recursive=True):
		with suppress(Exception):
			child.terminate()
			killed.append(child)
			print(child, "killed.")
	proc.terminate()
	print(proc, "killed.")
	with tracebacksuppressor:
		_, alive = psutil.wait_procs(killed, timeout=2)
	for child in alive:
		with suppress(Exception):
			child.kill()
	try:
		proc.wait(timeout=2)
	except psutil.TimeoutExpired:
		proc.kill()

def get_free_port():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[-1]
	finally:
		sock.close()

def esafe(s, binary=False):
	return (d := as_str(s).replace("\n", "\uffff")) and (d.encode("utf-8") if binary else d)
def dsafe(s):
	return as_str(s).replace("\uffff", "\n")
def nop2(s):
	return s

PROC = psutil.Process()
def quit(*args, **kwargs): force_kill(PROC)


# SHA256 operations: base64 and base16.
def shash(s): return e64(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), out=str)
def uhash(s): return min([shash(s), quote_plus(s.removeprefix("https://"))], key=len)
def uuhash(s): return uhash(unyt(s))
def hhash(s): return hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).hexdigest()
def ihash(s): return int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()[:8], "little")
def nhash(s): return int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little")


# Strips <> characters from URLs.
def strip_acc(url):
	if url.startswith("<") and url[-1] == ">":
		s = url[1:-1]
		if is_url(s):
			return s
	return url

__smap = {"|": "", "*": ""}
__strans = "".maketrans(__smap)
def verify_search(f):
	return unyt(strip_acc(single_space(f.strip().translate(__strans))))
COMM = "\\#$%"


@functools.lru_cache(maxsize=64)
def get_encoding(e):
	global tiktoken, cl100k_base, cl100k_im
	import tiktoken
	cl100k_base = tiktoken.get_encoding("cl100k_base")
	cl100k_im = tiktoken.Encoding(
		name="cl100k_im",
		pat_str=cl100k_base._pat_str,
		mergeable_ranks=cl100k_base._mergeable_ranks,
		special_tokens={
			**cl100k_base._special_tokens,
			"<|im_start|>": 100264,
			"<|im_end|>": 100265,
		},
	)
	if e == "cl100k_im":
		return cl100k_im
	try:
		return tiktoken.get_encoding(e)
	except (KeyError, ValueError):
		pass
	return tiktoken.encoding_for_model(e)

@functools.lru_cache(maxsize=1024)
def encode_to(s, enc) -> tuple:
	try:
		special_tokens = enc._special_tokens
	except AttributeError:
		return tuple(enc.encode(s))
	return tuple(enc.encode(s, allowed_special=set(special_tokens)))

def tik_encode(s, encoding="cl100k_im", allowed=65536) -> list:
	enc = get_encoding(encoding)
	out = []
	while s:
		temp, s = s[:allowed], s[allowed:]
		tokens = encode_to(temp, enc)
		out.extend(tokens)
	return out

def tik_decode(t, encoding="cl100k_im") -> str:
	enc = get_encoding(encoding)
	return enc.decode(t)

@hashable_args
@functools.lru_cache(maxsize=64)
def lim_tokens(s, maxlen=10, mode="centre", encoding="cl100k_im") -> str:
	"Limits a string to a maximum amount of tokens, cutting from the middle and replacing with \"..\" when possible."
	if maxlen is None:
		return s
	if maxlen <= 1:
		return "..."
	if not isinstance(s, str):
		s = str(s)
	if len(s) > maxlen * 8:
		s = lim_str(s, maxlen * 6, mode=mode)
	enc = get_encoding(encoding)
	tokens = tik_encode(s, encoding=encoding)
	over = (len(tokens) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(tokens) / 2
			s = enc.decode(tokens[:ceil(half - over - 1)]) + ".." + enc.decode(tokens[ceil(half + over + 1):])
		elif mode == "right":
			s = "..." + enc.decode(tokens[1 - maxlen:])
		else:
			s = enc.decode(tokens[:maxlen - 1]) + "..."
	return s.strip()

async def tik_encode_a(s, encoding="cl100k_im") -> list:
	if len(s) > 1024:
		return await asubmit(tik_encode, s, encoding=encoding, priority=2)
	return tik_encode(s, encoding=encoding)

async def tik_decode_a(t, encoding="cl100k_im") -> str:
	if len(t) > 256:
		return await asubmit(tik_decode, t, encoding=encoding, priority=2)
	return tik_decode(t, encoding=encoding)

@functools.lru_cache(maxsize=65536)
def _tlen(s, model="cl100k_im") -> int:
	if not s:
		return 0
	return len(tik_encode(s, encoding=model))

async def tcount(s, model="cl100k_im") -> int:
	if not s:
		return 0
	if len(s) <= 65536:
		return _tlen(s)
	tokens = await tik_encode_a(s, encoding=model)
	return len(tokens)

def tlen(s, model="cl100k_im") -> int:
	if not s:
		return 0
	if len(s) <= 65536:
		return _tlen(s)
	return len(tik_encode(s, encoding=model))

# Escapes syntax in code highlighting markdown.
ESCAPE_T = {
	"[": "⦍",
	"]": "⦎",
	"@": "＠",
	"`": "",
	";": ";",
}
__emap = "".maketrans(ESCAPE_T)

ESCAPE_T2 = {
	"@": "＠",
	"`": "",
	"#": "♯",
	";": ";",
}
__emap2 = "".maketrans(ESCAPE_T2)

breaks = "".maketrans({
	"\n": " ",
	"\r": " ",
	"\t": " ",
	"\v": " ",
	"\f": " ",
})

# Discord markdown format helper functions
def no_md(s):
	return str(s).translate(__emap)
def clr_md(s):
	return str(s).translate(__emap2)
def sqr_md(s):
	return f"[#{no_md(s)}]" if hasattr(s, "send") and not hasattr(s, "bot") else f"[{no_md(s)}]"
def no_links(s):
	return re.sub(r"`|https?:\/\/", "", s.translate(breaks)).strip()

def italics(s):
	if not isinstance(s, str):
		s = str(s)
	if "*" not in s:
		s = f"*{s}*"
	return s

def bold(s):
	if not isinstance(s, str):
		s = str(s)
	if "**" not in s:
		s = f"**{s}**"
	return s

def single_md(s):
	return f"`{s}`"
def code_md(s):
	return f"```\n{s}```" if s else "``` ```"
def py_md(s):
	return f"```py\n{s}```" if s else "``` ```"
def ini_md(s):
	return f"```ini\n{s}```" if s else "``` ```"
def css_md(s, force=False):
	return (f"```css\n{s}```".replace("'", "’").replace('"', "”") if force else ini_md(s)) if s else "``` ```"
def fix_md(s):
	return f"```fix\n{s}```" if s else "``` ```"
def ansi_md(s, max_length=4096):
	if not s:
		return "``` ```"
	s2 = s.replace("[[", colourise("[", fg="cyan") + colourise("", fg="blue")).replace("]]", colourise("]", fg="cyan") + colourise())
	if len(s2) <= max_length:
		s = s2
	else:
		s = decolourise(s.replace("[[", "[").replace("]]", "]"))
	return f"```ansi\n{s}```"

fgmap = cdict(
	black=30,
	red=31,
	green=32,
	yellow=33,
	blue=34,
	magenta=35,
	cyan=36,
	white=37,
)
bgmap = cdict(
	black=40,
	red=41,
	green=42,
	yellow=43,
	blue=44,
	magenta=45,
	cyan=46,
	white=47,
)
autocolours = dict(
	r="red",
	g="green",
	b="blue",
	c="cyan",
	m="magenta",
	y="yellow",
	k="black",
	w="white",
)
def colourise(s=None, fg=None, bg=None):
	fg = autocolours.get(fg, fg)
	bg = autocolours.get(bg, bg)
	s = as_str(s) if s is not None else ""
	if not bg:
		if not fg:
			return "\033[0m" + s
		return f"\033[0;{fgmap.get(fg, fg)}m" + s
	if not fg:
		return f"\033[0;{bgmap.get(bg, bg)}m" + s
	return f"\033[{fgmap.get(fg, fg)};{bgmap.get(bg, bg)}m" + s
colourised_quotes = r"""'"`“”"""
colourised_splits = r"""'\/\\|\-~:#@"""
def decolourise(s):
	return re.sub("\033" + r"\[((?:\d+;)*\d+)?m", "", s)
def colourise_brackets(s=None, a=None, b=None, c=None):
	out = ""
	while s:
		match = re.search(r"""[\(\[\{<⟨⟪【『⌊⌈](?:.*?)[\)\]\}>⟩⟫】』⌋⌉]|["`“](?:.*?)["`”]|['\/\\|\-~:#@]""", s)
		if not match:
			break
		if match.end() - match.start() == 1 and match.group() in colourised_splits:
			out += colourise(s[:match.start()], fg=a)
			out += colourise(match.group(), fg=c or b)
			s = s[match.end():]
			continue
		out += colourise(s[:match.start()], fg=a)
		opening = s[match.start()]
		out += colourise(opening, fg=b if opening in colourised_quotes else (c or b))
		out += colourise(s[match.start() + 1:match.end() - 1], fg=b)
		closing = s[match.end() - 1]
		out += colourise(closing, fg=b if closing in colourised_quotes else (c or b))
		s = s[match.end():]
	return out + colourise(s, fg=a)
def colourise_auto(s):
	while s:
		match = re.search(r"\$[rgbcmykwRGBCMYKW]*<[^\$]*>", s)
		if not match:
			break
		left, mid, right = s[:match.start()], match.group(), s[match.end():]
		code, mid = mid[1:-1].split("<", 1)
		kwargs = dict(
			fg=None,
			bg=None,
		)
		for c in code:
			target = "fg" if c.islower() else "bg"
			kwargs[target] = c.lower()
		s = left + colourise(mid, **kwargs) + colourise() + right
	return s

# Discord object mention formatting
def user_mention(uid):
	return f"<@{uid}>"
def user_pc_mention(uid):
	return f"<@!{uid}>"
def channel_mention(cid):
	return f"<#{cid}>"
def role_mention(rid):
	return f"<@&{rid}>"
def auto_mention(obj):
	if getattr(obj, "mention", None):
		return obj.mention
	if getattr(obj, "avatar", None):
		return user_mention(obj.id)
	if getattr(obj, "permissions", None):
		return role_mention(obj.id)
	if getattr(obj, "send", None):
		return channel_mention(obj.id)
	return str(obj.id)

@functools.lru_cache(maxsize=64)
def html_decode(s) -> str:
	"Decodes HTML encoded characters in a string."
	return html.unescape(s)

number_emojis = "0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟".split()

@hashable_args
@functools.lru_cache(maxsize=256)
def split_across(s, lim=2000, prefix="", suffix="", mode="len", bypass=((), ()), close_codeboxes=True) -> list:
	"""
	Splits a string into segments that fit within a specified length limit, considering prefixes, suffixes and code blocks.
	Args:
		s (str): The string to split.
		lim (int, optional): Maximum length limit for each segment. Defaults to 2000.
		prefix (str, optional): String to prepend to each segment. Defaults to "".
		suffix (str, optional): String to append to each segment. Defaults to "".
		mode (str, optional): Length calculation mode - "len" for character count or "tlen" for token length. Defaults to "len".
		bypass (tuple, optional): Two tuples containing strings - if segment starts with any string in first tuple, 
								prefix is bypassed; if segment ends with any string in second tuple, suffix is bypassed. 
								Defaults to ((), ()).
		close_codeboxes (bool, optional): Whether to automatically close unclosed code blocks. Defaults to True.
	Returns:
		list: List of string segments, each within the specified length limit including prefix/suffix.
	Examples:
		>>> split_across("Long text here", lim=10)
		['Long text', 'here']
		>>> split_across("```code\nstuff```", prefix="Pre: ")
		['Pre: ```code\nstuff```']
	Notes:
		- Attempts to split on natural boundaries (paragraphs, sentences, spaces) when possible
		- Handles code blocks (```) intelligently, ensuring they're properly closed
		- Can bypass prefix/suffix for specific string patterns
		- Supports both character length and token length counting modes
	"""
	cb = "```"
	if cb in suffix:
		close_codeboxes = False
	current_codebox = ""
	s = s.replace("\r\n", "\n").replace("\f", "\n\n").replace("\v", "\n\n")
	# Natural boundaries in order of preference
	splitters = ["\n\n", "\n", "\t", "? ", "! ", ". ", " "]

	if mode == "len":
		raw_len = len
	elif mode == "tlen":
		@functools.lru_cache(maxsize=64)
		def raw_len(s):
			# Borrows the tiktoken encoding for token length calculation
			return tlen(s)
	else:
		raise NotImplementedError(f"split_across: Unsupported mode {mode}")
	def required_len(s):
		c = raw_len(s)
		if bypass[0] and any(s.startswith(w) for w in bypass[0]):
			pass
		else:
			c += raw_len(prefix)
		if bypass[1] and any(s.endswith(w) for w in bypass[1]):
			pass
		else:
			c += raw_len(suffix)
		if current_codebox:
			c += raw_len(current_codebox)
		if close_codeboxes and not s.endswith(cb) and s.count(cb) + bool(current_codebox) & 1:
			c += raw_len(cb)
		return c
	def complete(s):
		out = ""
		cc = ""
		if current_codebox:
			s = current_codebox + s
		s = s.strip()
		if bypass[0] and any(s.startswith(w) for w in bypass[0]):
			pass
		else:
			out += prefix
		out += s
		if close_codeboxes and not s.endswith(cb) and s.count(cb) & 1:
			out += cb
			cc = cb + (s.rsplit(cb, 1)[-1].split("\n", 1)[0].split(None, 1) or [""])[0] + "\n"
		if bypass[1] and any(s.endswith(w) for w in bypass[1]):
			pass
		else:
			out += suffix
		return out, cc
	def tries(s):
		yield s, ""
		for cut in splitters:
			try:
				i = s.rindex(cut)
			except ValueError:
				pass
			else:
				yield s[:i], cut
	def try_one(s, cut):
		try:
			i = s.rindex(cut)
		except ValueError:
			return s, ""
		else:
			return s[:i], cut
	def from_budget(s, budget):
		if mode == "len":
			return s[:budget]
		# Assume 8 characters per token, then actually cut down after encoding. This means tokens over 8 characters will cut into the budget, but this is a reasonable approximation that doesn't require reencoding the entire string each time.
		return tik_decode(tik_encode(s[:budget * 8])[:budget])

	if close_codeboxes:
		if "\n" in s and "#" in s and cb in s:
			pattern = re.compile(rf"^((?:[#\-\s]|[0-9]+\.\s).*{cb})", re.MULTILINE)
			s = pattern.sub(lambda x: x.group(1).lstrip()[:-3] + ("\n" + cb), s)
	if required_len(s) <= lim:
		# If the entire string fits within the limit, return it as a single segment
		return [complete(s)[0]]
	out = []
	temp = ""
	found = None
	while s:
		# Keep track of a "budget" for how much we can add to the current segment. Note that we need the `required_len` function to account for the prefix/suffix, and possible code blocks that need to be closed.
		budget = max(1, lim - required_len(temp) + 1)
		checker = from_budget(s, budget)
		if found is not None:
			# If we already found a natural boundary, try to cut more segments using that same boundary
			cur, cut = try_one(checker, found)
			if required_len(new := temp + cur) > lim:
				# If a single higher-priority segment exceeds half the limit, we just cut it off and move on, otherwise try to add more smaller segments until we reach the limit. This eliminates the case where an extremely short segment is separated by a high priority delimiter, such as a double newline, which would lead to very short segments being present in the output. In essence, don't be too conservative with splitting.
				if required_len(temp) > lim / 2:
					text, current_codebox = complete(temp)
					out.append(text)
					temp = ""
					found = None
					s = s.lstrip()
					continue
			else:
				s = s[len(cur + cut):]
				found = cut
				temp = new + cut
				continue
		for cur, cut in tries(checker):
			if required_len(new := temp + cur) > lim:
				continue
			s = s[len(cur + cut):]
			found = cut
			temp = new + cut
			break
		else:
			n_required = lim - required_len(temp)
			if n_required >= lim / 2:
				if mode == "len":
					temp = temp + checker[:n_required]
					s = s[n_required:]
				else:
					tokens = tik_encode(checker)
					temp = temp + tik_decode(tokens[:n_required])
					s = s[len(tik_decode(tokens[n_required:])):]
			text, current_codebox = complete(temp)
			out.append(text)
			temp = ""
			found = None
			s = s.lstrip()
			continue
	if temp:
		out.append(complete(temp)[0])
	return out

def find_split_position(text, max_length, priority):
	substring = text[:max_length]
	for delimiter in priority:
		index = substring.rfind(delimiter)
		if index != -1 and index >= max_length / 2:
			return index + len(delimiter)
	return max_length

def close_markdown(text):
	text = text.rstrip()
	closed = []
	# Handle bold (**)
	bold_count = len(re.findall(r'(?:^|\s)\*\*(?:$|\s)', text))
	if bold_count & 1:
		closed.append('**')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]

	# Handle italic (*) - single asterisks not part of bold and code block
	if "`" not in text:
		italic_count = len(re.findall(r'(?<!\*)(?:^|\s)\*(?:$|\s)(?!\*)', text))
		if italic_count & 1:
			closed.append('*')
			if text.endswith(closed[-1]):
				text = text.removesuffix(closed[-1])
			else:
				text += closed[-1]

	# Handle code blocks (```)
	code_block_count = len(re.findall(r'```', text))
	if code_block_count & 1:
		# Code blocks are special because they may contain a format indicator; we need to copy the last detected opening sequence
		last_open = text.rfind('```')
		assert last_open != -1, "Unmatched closing code block detected"
		closed.append(text[last_open:].split("\n", 1)[0] + "\n")
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += "```"

	# Handle inline code (`) - single backticks not part of code block
	inline_code_count = len(re.findall(r'(?<!`)`(?!`)', text))
	if inline_code_count & 1:
		closed.append('`')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]

	# Handle spoiler tags (||) - double pipes not part of code block
	spoiler_count = len(re.findall(r'(?<!`)\|\|(?!`)', text))
	if spoiler_count & 1:
		closed.append('||')
		if text.endswith(closed[-1]):
			text = text.removesuffix(closed[-1])
		else:
			text += closed[-1]
	return text, "".join(reversed(closed))

def split_text(text, max_length=2000, priority=("\n\n", "\n", "\t", "? ", "! ", "။", "。", ". ", ", ", " "), prefix="", suffix=""):
	chunks = []
	opening = ""
	while text:
		if len(text) <= max_length:
			chunks.append(close_markdown(opening + text)[0])
			break

		for adjusted in range(max_length):
			adjusted_max_length = max(max_length - adjusted, 1)

			split_pos = find_split_position(text, adjusted_max_length, priority)
			current_part = opening + text[:split_pos]
			if not current_part.startswith(prefix):
				current_part = prefix + current_part
			remaining_text = text[split_pos:]

			processed_part, new_opening = close_markdown(current_part)
			if not processed_part.endswith(suffix):
				processed_part += suffix
			if len(processed_part) <= max_length:
				opening = new_opening
				break
		chunks.append(processed_part)
		text = remaining_text
	return chunks

smart_re = re.compile(r"""
	(?:^|\s)
	(['"`])
	(?:
		\\[\s\S]
	| (?!\1)"[^"]*"
	| (?!\1)'[^']*'
	| (?!\1)`[^`]*`
	| [^"'`]
	)*
	\1
	(?:$|\s)
""", re.VERBOSE)
leftspace_re = re.compile(r"^\s+")
rightspace_re = re.compile(r"\s+$")
def smart_split(s, rws=False):
	"""
	Split a string into words while preserving whitespace information.
	This function intelligently splits a string into words, handling quoted strings
	and various whitespace patterns. It can optionally return both the words and
	the whitespace that separates them.
	Args:
		s (str): The input string to split.
		rws (bool, optional): If True, return both words and whitespace. 
			If False, return only words. Defaults to False.
	Returns:
		list or tuple: If rws is False, returns a list of words (strings).
			If rws is True, returns a tuple of (words, whites) where:
			- words (list): List of extracted words/tokens
			- whites (list): List of whitespace strings separating the words
	Note:
		This function uses regex patterns (smart_re, leftspace_re, rightspace_re)
		that should be defined in the module scope. It handles:
		- Quoted strings (preserving content between matching quotes)
		- Leading and trailing whitespace
		- Multiple consecutive whitespace characters
	"""
	words, whites = [], []

	def process_token(token):
		if not token:
			return words.append(token)
		if len(token) > 1 and token[0] == token[-1] and token[0] in (r"""`"`"""):
			return words.append(token[1:-1])
		spl = token.split()
		for i, w in enumerate(spl):
			words.append(w)
			token = token[len(w):]
			if i < len(spl) - 1:
				wss = token.index(spl[i + 1])
				whites.append(token[:wss])
				token = token[wss:]

	while s:
		match = smart_re.search(s)
		if not match:
			match = leftspace_re.search(s)
			if not match:
				whites.append("")
				process_token(s)
				break
			whites.append(match.group())
			process_token(s[match.end():])
			break
		token = s[:match.start()]
		sub = token.lstrip()
		if len(sub) < len(token):
			whites.append(token[:len(token) - len(sub)])
			if sub:
				process_token(sub)
			else:
				words.append("")
		elif sub:
			whites.append("")
			process_token(sub)
		grouped = match.group()
		match2 = leftspace_re.search(grouped)
		match3 = rightspace_re.search(grouped)
		end = match.end()
		if not match2:
			whites.append("")
			if not match3:
				process_token(grouped)
			else:
				process_token(grouped[:match3.start()])
				end -= len(match3.group())
		else:
			whites.append(match2.group())
			if not match3:
				process_token(grouped[match2.end():])
			else:
				process_token(grouped[match2.end():match3.start()])
				end -= len(match3.group())
		s = s[end:]
		if s and not s.lstrip():
			whites.append(s)
			break
	if rws:
		return words, whites
	return words

@hashable_args
@functools.lru_cache(maxsize=256)
def longest_prefix(s1, s2):
	length = min(len(s1), len(s2))
	for i, (x, y) in enumerate(zip(s1, s2)):
		if x != y:
			break
	else:
		return length
	return i

@hashable_args
@functools.lru_cache(maxsize=256)
def longest_common_substring(s1, s2):
	m = len(s1)
	n = len(s2)
	max_len = 0
	ending_index = m
	lcsuff = np.empty((m + 1, n + 1), dtype=np.uint32)
	for i in range(m + 1):
		for j in range(n + 1):
			if i == 0 or j == 0:
				lcsuff[i][j] = 0
			elif s1[i-1] == s2[j-1]:
				lcsuff[i][j] = lcsuff[i-1][j-1] + 1
				if max_len < lcsuff[i][j]:
					max_len = lcsuff[i][j]
					ending_index = i
			else:
				lcsuff[i][j] = 0
	return s1[ending_index - max_len:ending_index]

def capwords(s, spl=None):
	return (" " if spl is None else spl).join(w.capitalize() for w in s.split(spl))

# This reminds me of Perl - Smudge
def find_urls(url): return url and regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+").findall(url)
def find_urls_ex(url):
	no_triple = re.sub(r'```.*?```', '', url, flags=re.DOTALL)
	no_code = re.sub(r'`[^`]*`', '', no_triple, flags=re.DOTALL)
	return re.findall(r'''https?://[^\s`|"'\])>]+''', no_code)
def is_url(url): return url and isinstance(url, (str, bytes)) and regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\\])>]+$").fullmatch(url)
def is_discord_url(url): return url and regexp("^https?:\\/\\/(?:\\w{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/").findall(url) + regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").findall(url)
def is_discord_attachment(url): return url and regexp("^https?:\\/\\/(?:\\w{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/attachments\\/").search(str(url))
def is_discord_emoji(url): return url and regexp("^https?:\\/\\/(?:\\w{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/emojis\\/").search(str(url))
def is_tenor_url(url): return url and regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[\\w\\-]+-[0-9]+").findall(url)
def is_imgur_url(url): return url and regexp("^https?:\\/\\/(?:\\w\\.)?imgur.com/[\\w\\-]+").findall(url)
def is_giphy_url(url): return url and regexp("^https?:\\/\\/giphy.com/gifs/[\\w\\-]+").findall(url)
def is_miza_url(url): return url and regexp("^https?:\\/\\/(?:\\w+\\.)?mizabot.xyz").findall(url)
def is_miza_attachment(url): return url and regexp("^https?:\\/\\/(?:\\w+\\.)?mizabot.xyz\\/\\w\\/").findall(url)
def is_youtube_url(url): return url and regexp("^https?:\\/\\/(?:\\w{1,5}\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+").findall(url)
def is_youtube_stream(url): return url and regexp("^https?:\\/\\/r+[0-9]+---.{2}-[\\w\\-]{4,}\\.googlevideo\\.com").findall(url)
def is_soundcloud_stream(url): return url and regexp("^https?:\\/\\/(?:[\\w\\-]*)?media\\.sndcdn\\.com\\/[^\\s<>`|\"']+").findall(url)
def is_deviantart_url(url): return url and regexp("^https?:\\/\\/(?:www\\.)?deviantart\\.com\\/[^\\\\s<>`|\"']+").findall(url)
def is_reddit_url(url): return url and regexp("^https?:\\/\\/(?:\\w{2,3}\\.)?reddit.com\\/r\\/[^/\\W]+\\/").findall(url)
def is_emoji_url(url): return url and url.startswith("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/")
def is_spotify_url(url): return url and regexp("^https?:\\/\\/(?:play|open|api)\\.spotify\\.com\\/").findall(url)
def _unyt(s):
	s = re.sub(r"[\?&]pp=[\w\-]+", "", re.sub(r"[\?&]si=[\w\-]+", "", s))
	return re.sub(r"https?:\/\/(?:\w{1,5}\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/)|youtu\.be\/)", "https://youtu.be/", s)
def unyt(s):
	"Produces a unique URL, such as converting all instances of https://www.youtube.com/watch?v=video to https://youtu.be/video. This is useful for caching and deduplication."
	if not is_url(s):
		return s
	if (s.startswith("https://mizabot.xyz/u") or s.startswith("https://api.mizabot.xyz/u")) and ("?url=" in s or "&url=" in s):
		s = unquote_plus(s.replace("&url=", "?url=", 1).split("?url=", 1)[-1])
	if s.startswith("https://mizabot.xyz/ytdl") or s.startswith("https://api.mizabot.xyz/ytdl"):
		if "?d=" in s or "?v=" in s:
			s = unquote_plus(s.replace("?v=", "?d=", 1).split("?d=", 1)[-1])
		else:
			s = re.sub(r"https?:\/\/(?:api\.)?mizabot\.xyz\/ytdl\?[vd]=(?:https:\/\/youtu\.be\/|https%3A%2F%2Fyoutu\.be%2F)", "https://youtu.be/", s)
		s = s.split("&", 1)[0]
	if is_discord_attachment(s) or is_spotify_url(s) or s.startswith("https://i.ytimg.com"):
		s = s.split("?", 1)[0]
	return _unyt(s)
def is_discord_message_link(url) -> bool:
	"Detects whether a Discord link represents a channel or message link."
	check = url[:64]
	return "channels/" in check and "discord" in check
def discord_expired(url, early=21600 + 60):
	if is_discord_attachment(url):
		if "?ex=" not in url and "&ex=" not in url:
			return True
		temp = url.replace("?ex=", "&ex=").split("&ex=", 1)[-1].split("&", 1)[0]
		try:
			ts = int(temp, 16)
		except ValueError:
			return True
		return ts < utc() + early
def expired(stream):
	if not stream:
		return True
	if stream == "none":
		return True
	if discord_expired(stream):
		return True
	if is_youtube_url(stream):
		return True
	if stream.startswith("ytsearch:"):
		return True
	if stream.startswith("https://open.spotify.com/track/"):
		return True
	if stream.startswith("https://www.yt-download.org/download/"):
		if int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60:
			return True
	elif re.match(r"https?:\/\/cdn[0-9]*\.tik\.live\/api\/stream", stream):
		if float(stream.replace("/", "=").replace("&e=", "?e=").split("?e=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) / 1000 < utc() + 60:
			return True
	elif is_youtube_stream(stream):
		if int(stream.replace("/", "=").split("expire=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 60:
			return True
	elif is_soundcloud_stream(stream):
		if int(stream.replace("/", "=").split("expires=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 30:
			return True

def url2fn(url) -> str:
	return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1].translate(filetrans)
def url2ext(url) -> str:
	fn = url2fn(url)
	return fn.rsplit(".", 1)[-1] if "." in fn else "bin"

def replace_ext(fn, ext="") -> str:
	if "." not in fn:
		return fn + "." + ext
	return fn.rsplit(".", 1)[0] + "." + ext

scraper_blacklist = re.compile("|".join(map(re.escape, (
	"ko-fi.com",
	"spotify.com",
	"artfight.net",
	"discord.com/invite",
))))

def bytes2hex(b, space=True) -> str:
	"Converts a bytes object to a hex string."
	if type(b) is str:
		b = b.encode("utf-8")
	if space:
		return b.hex(" ").upper()
	return b.hex().upper()
def hex2bytes(b) -> bytes:
	"Converts a hex string to a bytes object."
	s = as_str(b).replace(" ", "")
	if len(s) & 1:
		s = s[:-1] + "0" + s[-1]
	return bytes.fromhex(s)


def maps(funcs, *args, **kwargs):
	"A map-compatible function that takes and iterates through multiple functions in a list as the first argument."
	for func in funcs:
		yield func(*args, **kwargs)

def temporary_file(fmt="bin", name=""):
	return f"{TEMP_PATH}/{name or ts_us()}.{fmt}" if fmt else f"{TEMP_PATH}/{ts_us()}"

def get_image_size(b):
	if not b:
		raise FileNotFoundError("Input data empty or not received!")
	if isinstance(b, io.BytesIO):
		input = b
		input.seek(0)
	elif isinstance(b, (bytes, bytearray, memoryview)):
		input = io.BytesIO(b)
	elif isinstance(b, (io.IOBase)):
		try:
			b.seek(0)
		except Exception:
			print_exc()
		input = io.BytesIO(b.read(4096))
	elif isinstance(b, str):
		input = b
	else:
		raise TypeError(type(b))
	try:
		import imagesize
		w, h = imagesize.get(input)
		if w < 0 or h < 0:
			raise IndexError(w, h)
		return w, h
	except Exception as ex:
		if not isinstance(ex, IndexError):
			print_exc()
		if not getsize(input):
			raise FileNotFoundError("Input data empty or not received!")
	if isinstance(b, io.BytesIO):
		input = b
		input.seek(0)
	elif isinstance(b, (bytes, bytearray, memoryview)):
		input = io.BytesIO(b)
	elif isinstance(b, (io.IOBase)):
		try:
			b.seek(0)
		except Exception:
			print_exc()
		input = b
	elif isinstance(b, str):
		input = open(input, "rb")
	from PIL import Image
	input.seek(0)
	try:
		with Image.open(input) as im:
			return im.size
	except Exception as ex:
		print("Image size error:", repr(ex))
		temp = temporary_file()
		input.seek(0)
		save_file(input, temp)
		cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", temp)
		print(cmd)
		try:
			out = subprocess.check_output(cmd)
		finally:
			os.remove(temp)
		if not out or b"x" not in out:
			raise
	return list(map(int, out.split(b"x")))

def rename(src, dst):
	"""
	Rename a file or directory from src to dst.

	This function attempts to rename a file or directory from the source path
	(src) to the destination path (dst). If the destination already exists,
	it will be removed before renaming. If the rename operation fails due to
	an OSError with error code 18 (cross-device link), the function will
	manually copy the file contents from src to dst.

	Args:
		src (str): The source path of the file or directory to be renamed.
		dst (str): The destination path where the file or directory should be renamed to.

	Returns:
		None

	Raises:
		PermissionError: If the destination exists and cannot be removed.
		OSError: If any other error occurs during the rename operation.
	"""
	try:
		return os.replace(src, dst)
	except FileExistsError:
		if os.path.exists(dst):
			os.remove(dst)
		return os.rename(src, dst)
	except OSError as ex:
		if ex.args and ex.args[0] == 18:
			shutil.copyfile(src, dst)

archive_formats = ("7z", "zip", "tar", "gz", "bz", "xz")
archive_mimes = ("application/zip", "application/gzip", "application/x-gzip", "application/zstd", "application/vnd.rar", "application/tar", "application/x-tar", "application/x-tar+xz", "application/x-7z-compressed", "application/x-bzip2")

def unpack_gz(archive_name, extract_dir):
	import gzip
	with gzip.open(archive_name, "rb") as f_in:
		with open(os.path.join(extract_dir, os.path.basename(archive_name).replace(".gz", "")), "wb") as f_out:
			f_out.write(f_in.read())
shutil.register_unpack_format("gz", [".gz"], unpack_gz)
def unpack_xz(filename, extract_dir):
	import tarfile
	with tarfile.open(filename, "r:xz") as tar:
		tar.extractall(path=extract_dir)
shutil.register_unpack_format("xz", [".xz"], unpack_xz)

def extract_archive(archive_path, format=None, excludes=()):
	path = f"{TEMP_PATH}/{ts_us()}"
	os.mkdir(path)
	with open(archive_path, "rb") as f:
		b = f.read(6)
	if b == b'7z\xbc\xaf\x27\x1c':
		import py7zr
		with py7zr.SevenZipFile(archive_path, mode="r") as z:
			z.extractall(path=path)
	else:
		shutil.unpack_archive(archive_path, extract_dir=path, format=format or get_ext(archive_path))
	return [f"{path}/{fn}" for fn in os.listdir(path) if fn.rsplit(".", 1)[-1] not in excludes]

IMAGE_FORMS = {
	"auto": None,
	"gif": True,
	"png": True,
	"apng": True,
	"bmp": False,
	"jpg": True,
	"jpeg": True,
	"jp2": False,
	"jpx": False,
	"jxl": False,
	"tiff": False,
	"webp": True,
	"heic": False,
	"heif": True,
	"avif": True,
	"ico": False,
}
def is_image(url):
	"Checks whether a url or filename ends with an image file extension. Returns a ternary True/False/None value where True indicates a positive match, False indicates a possible match, and None indicates no match."
	if url:
		return IMAGE_FORMS.get(url2ext(url))

VIDEO_FORMS = {
	"auto": None,
	"ts": True,
	"webm": True,
	"mkv": True,
	"f4v": False,
	"flv": True,
	"ogv": True,
	"gif": False,
	"gifv": True,
	"apng": False,
	"avi": True,
	"avif": False,
	"mov": True,
	"qt": True,
	"wmv": True,
	"mp4": True,
	"m4v": True,
	"mpg": True,
	"mpeg": True,
	"mpv": True,
	"m3u8": True,
	"tar": False,
	"zip": False,
}
def is_video(url):
	"Checks whether a url or filename ends with a video file extension. Returns a ternary True/False/None value where True indicates a positive match, False indicates a possible match, and None indicates no match."
	if url:
		return VIDEO_FORMS.get(url2ext(url))

AUDIO_FORMS = {
	"auto": None,
	"mp3": True,
	"mp2": True,
	"ogg": True,
	"opus": True,
	"wav": True,
	"flac": True,
	"m4a": True,
	"aac": True,
	"wma": True,
	"vox": True,
	"ts": False,
	"webm": False,
	"weba": True,
	"mp4": False,
	"pcm": False,
}
def is_audio(url):
	"Checks whether a url or filename ends with an audio file extension. Returns a ternary True/False/None value where True indicates a positive match, False indicates a possible match, and None indicates no match."
	if url:
		return AUDIO_FORMS.get(url2ext(url))

VISUAL_FORMS = {
	"auto": None,
	"gif": True,
	"png": True,
	"apng": True,
	"bmp": False,
	"jpg": True,
	"jpeg": True,
	"jp2": False,
	"jpx": False,
	"jxl": False,
	"tiff": False,
	"webp": True,
	"heic": False,
	"ico": False,
	"ts": True,
	"webm": True,
	"mkv": True,
	"f4v": False,
	"flv": True,
	"ogv": True,
	"ogg": False,
	"gifv": True,
	"avi": True,
	"avif": True,
	"mov": True,
	"qt": True,
	"wmv": True,
	"mp4": True,
	"m4v": True,
	"av1": True,
	"h266": True,
	"h265": True,
	"h264": True,
	"mpg": True,
	"mpeg": True,
	"mpv": True,
	"m3u8": True,
	**{k: False for k in archive_formats},
}
MEDIA_FORMS = IMAGE_FORMS.copy()
MEDIA_FORMS.update(VIDEO_FORMS)
MEDIA_FORMS.update(AUDIO_FORMS)

CODEC_FFMPEG = {
	"auto": "av1_nvenc",
	"x264": "h264_nvenc",
	"h264": "h264_nvenc",
	"avc": "h264_nvenc",
	"x265": "hevc_nvenc",
	"h265": "hevc_nvenc",
	"hevc": "hevc_nvenc",
	"mkv": "hevc_nvenc",
	"x266": "libvvenc",
	"h266": "libvvenc",
	"vvc": "libvvenc",
	"av1": "av1_nvenc",
}
CODEC_PIX = {
	"h264_nvenc": "nv12",
	"h265_nvenc": "nv12",
	"av1_nvenc": "nv12",
	"libvvcenc": "yuv444p",
}
CODECS = {
	"auto": "auto",
	"x264": "mp4",
	"h264": "mp4",
	"avc": "mp4",
	"x265": "mp4",
	"h265": "mp4",
	"hevc": "mp4",
	"x266": "mp4",
	"h266": "mp4",
	"vvc": "mp4",
	"vp8": "webm",
	"vp9": "webm",
	"av1": "mp4",
	"quicktime": "qt",
	"mpegts": "ts",
	"libopus": "opus",
	"matroska": "mkv",
	"s16le": "pcm",
}
CODECS_INV = {v: k for k, v in CODECS.items()}
MIMES = cdict(
	bin="application/octet-stream",
	css="text/css",
	json="application/json",
	js="application/javascript",
	txt="text/plain",
	html="text/html",
	svg="image/svg+xml",
	ico="image/x-icon",
	png="image/png",
	jpg="image/jpeg",
	gif="image/gif",
	webp="image/webp",
	avif="image/avif",
	ts="video/ts",
	webm="video/webm",
	weba="audio/weba",
	qt="video/quicktime",
	mp3="audio/mpeg",
	ogg="audio/ogg",
	opus="audio/opus",
	flac="audio/flac",
	wav="audio/x-wav",
	mp4="video/mp4",
	mkv="video/x-matroska",
	tar="application/tar",
	zip="application/zip",
)

def load_mimes():
	with open("misc/mimes.txt") as f:
		mimedata = f.read().splitlines()
		globals()["mimesplitter"] = mimesplitter = {}
		for line in mimedata:
			dat, ext, mime = line.split("\t")
			data = hex2bytes(dat)
			try:
				mimesplitter[len(data)][data] = (ext, mime)
			except KeyError:
				mimesplitter[len(data)] = {}
				mimesplitter[len(data)][data] = (ext, mime)

mime_wait = esubmit(load_mimes)

def simple_mimes(b, mime=True):
	"Low-latency function that detects mimetype from first few bytes. Less accurate than mime_from_file."
	mimesplitter = globals()["mimesplitter"]
	for k, v in reversed(mimesplitter.items()):
		out = v.get(b[:k])
		if out:
			return out[mime]
	try:
		_s = b.decode("utf-8")
	except UnicodeDecodeError:
		return "application/octet-stream" if mime else "bin"
	return "text/plain" if mime else "txt"

special_mimes = {
	"avi": "x-msvideo",
	"bin": "octet-stream",
	"bz": "x-bzip",
	"bz2": "x-bzip2",
	"ico": "x-icon",
	"jpg": "jpeg",
	"js": "javascript",
	"mpg": "mpeg",
	"php": "x-httpd-php",
	"rar": "vnd.rar",
	"svg": "svg+xml",
	"tar": "tar",
	"qt": "quicktime",
	"ts": "mp2t",
	"txt": "plain",
	"7z": "x-7z-compressed",
}
inv_mimes = {v: k for k, v in special_mimes.items()}

def mime_into(mime: str) -> str:
	ext = mime.split("/", 1)[-1]
	return inv_mimes.get(ext) or ext.rsplit("/", 1)[-1]

def mime_equiv(a, b):
	"Checks if a mimetype matches a given file extension. Required as some do not match."
	if a == b or a == "auto" or b == "auto":
		return True
	a = a.split("/", 1)[-1]
	b = b.split("/", 1)[-1]
	if a == b:
		return True
	a, b = sorted((a, b))
	if a == "jpeg" and b == "jpg":
		return True
	return special_mimes.get(a) == b

__filetrans = {
	"\\": "_",
	"/": "_",
	" ": "%20",
	"\n": "%20",
	":": "=",
	"*": "-",
	"?": "&",
	'"': "^",
	"<": "{",
	">": "}",
	"|": "!",
}
filetrans = "".maketrans(__filetrans)

def filetransd(fn):
	out = []
	for c in fn:
		if c.casefold() not in "abcdefghijklmnopqrstuvwxyz0123456789-_.":
			out.append("_")
			continue
		out.append(c)
	return "".join(out)

def get_ext(f):
	mime = mime_from_file(f)
	return mime_into(mime)

def mime_from_file(path, filename=None, mime=True):
	"Detects mimetype of file or buffer. Includes custom .jar, .ecdc, .m3u8 detection."
	data = filetype.get_bytes(path)
	if mime:
		out = filetype.guess_mime(data)
	else:
		out = filetype.guess_extension(data)
	filename = filename or (path if isinstance(path, str) else "")
	if out and out.split("/", 1)[-1] == "zip" and isinstance(filename, str) and filename.endswith(".jar"):
		return "application/java-archive"
	if not out:
		if not isinstance(data, bytes):
			if isinstance(data, str):
				raise TypeError(data)
			data = bytes(data)
		out = simple_mimes(data, mime)
	if out in ("application/octet-stream", "application/vnd.lotus-organizer"):
		if data.startswith(b'ECDC'):
			return "audio/x-ecdc"
		if data.startswith(b"MThd"):
			return "audio/midi"
		if data.startswith(b"Org-"):
			return "audio/x-org"
		if data[4:8] == b"ftyp":
			if data[8:12] in (b"avis", b"avif"):
				return "image/avif"
			if data[8:12] in (b"heic", "heix", "hevc", "hevx"):
				return "image/heic"
			if data[8:12] in (b"mif1", "msf1"):
				return "image/heif"
	if out == "text/plain" and data.startswith(b"#EXTM3U"):
		return "video/m3u8"
	return out

magic = cdict(
	from_file=mime_from_file,
	from_buffer=mime_from_file,
	Magic=lambda mime, *args, **kwargs: cdict(
		from_file=lambda b: mime_from_file(b, mime),
		from_buffer=lambda b: mime_from_file(b, mime),
	),
)

def get_mime(path):
	if not isinstance(path, str) or os.path.getsize(path) < 1048576:
		try:
			mime = magic.from_file(path, mime=True)
		except Exception:
			print_exc()
			mime = "cannot open `"
		if not isinstance(path, str):
			path = path.name if hasattr(path, "name") else ".txt"
	else:
		mime = "cannot open `"
	if mime.startswith("cannot open `"):
		with open(path, "rb") as f:
			b = f.read(262144)
		mime = magic.from_buffer(b, mime=True)
		if mime == "application/octet-stream":
			if path.endswith(".txt"):
				return "text/plain"
			try:
				_s = b.decode("utf-8")
			except UnicodeDecodeError:
				pass
			else:
				return "text/plain"
	if mime.startswith("text/plain"):
		ext = path.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
		mime2 = MIMES.get(ext, "")
		if mime2:
			return mime2
	elif mime.split("/", 1)[-1] == "zip" and path.endswith(".jar"):
		return "application/java-archive"
	return mime

def find_file(path, cwd="saves/filehost", ind="\x7f"):
	"Finds a matching file starting with the given path. Mostly used for filehost."
	if not path:
		raise EOFError
	# do not include "." in the path name
	path = str(path).rsplit(".", 1)[0]
	fn = f"{ind}{path}"
	if not isinstance(cwd, (tuple, list)):
		cwd = (cwd,)
	for wd in cwd:
		for fi in reversed(os.listdir(wd)):
			# file cache is stored as "{timestamp}~{name}", search for file via timestamp
			if fi[-1] != ind and fi.rsplit(".", 1)[0].split("~", 1)[0] == fn:
				return wd + "/" + fi
	raise FileNotFoundError(404, path)

url_parse = quote_plus
url_unparse = unquote_plus

def stream_exists(url, fmt="opus"):
	url = unyt(url)
	h = shash(url)
	fn = f"{TEMP_PATH}/audio/~" + h + "." + fmt
	return os.path.exists(fn) and os.path.getsize(fn)

def verify_url(url):
	return url if is_url(url) else quote_plus(url)

__scales = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
__uscales = [s.lower() for s in __scales]
def byte_scale(n, ratio=1024):
	e = 0
	while n >= ratio:
		n /= ratio
		e += 1
		if e >= len(__scales) - 1:
			break
	return f"{round(n, 4)} {__scales[e]}"
def byte_unscale(s, ratio=1024):
	num_part = regexp(r"^[\.0-9]+").findall(s)
	if not num_part:
		n = 1
	else:
		n = num_part[0]
		s = s[len(n):]
		n = round_min(n)
	return round_min(n * ratio ** __uscales.index(s.lower()))

def e64(b, out=bytes):
	b = as_bytes(b)
	e = base64.urlsafe_b64encode(b).rstrip(b"=")
	if out is str:
		e = e.decode("ascii")
	return e
def b64(b):
	b = as_bytes(b)
	if len(b) & 3:
		b += b"=="
	return base64.urlsafe_b64decode(b)

def b64_or_uni(b):
	b = b.strip()
	if b.isascii():
		try:
			return b64(b)
		except Exception:
			pass
	return base65536.decode(as_str(b))

def cantor(*x):
	n = len(x)
	p = 0
	for k in range(n):
		q = k + sum(x[j] for j in range(k))
		p += comb(q, k + 1)
	return p
def icantor(z):
	w = isqrt(8 * z + 1) - 1 >> 1
	t = w * w + w >> 1
	y = z - t
	return w - y, y

def encode_filename(fn):
	fn = fn.encode("ascii").replace(b"-", b"--").replace(b".", b"-_")
	if fn.endswith(b"A"):
		fn += b"_"
	return b64(fn)
def decode_filename(b):
	fn = e64(b).removesuffix(b"A")
	fn = fn.replace(b"-_", b".").replace(b"--", b"-")
	return fn.rstrip(b"_").decode("ascii")

def b2n(b, mode="big"):
	return int.from_bytes(b, mode)
def n2b(n, mode="big"):
	c = n.bit_length() + 7 >> 3
	return n.to_bytes(c, mode)

def leb128(n: int) -> bytearray:
	"Encodes an integer using a custom LEB128 algorithm. Supports a sign for negative integers via an additional 00 byte, maintaining compatibility with standard LEB128 (unlike SLEB128)."
	if n <= 0:
		was_negative = True
		n = -n
	else:
		was_negative = False
	data = bytearray()
	while n > 0:
		data.append(n & 0x7F)
		n >>= 7
		if n:
			data[-1] |= 0x80
	if was_negative:
		if len(data):
			data[-1] |= 0x80
		data.append(0)
	return data
def decode_leb128(data: byte_like) -> tuple[int, byte_like]:
	"Decodes an integer from LEB128 encoded data; returns a tuple of decoded and remaining data."
	i = n = 0
	shift = 0
	for i, byte in enumerate(data):
		n |= (byte & 0x7F) << shift
		if byte & 0x80 == 0:
			if byte == 0:
				n = -n
			break
		else:
			shift += 7
	return n, data[i + 1:]

def szudzik(x, y):
	"Szudzik's pairing function."
	if x < y:
		return y * y + x
	return x * x + x + y
def iszudzik(z):
	"Inverse of Szudzik's pairing function."
	w = isqrt(z)
	t = z - w * w
	if t < w:
		return t, w
	return w, t - w

def interleave(*X):
	"Bit interleaves a list of integers."
	m = max(x.bit_length() for x in X)
	z = 0
	X2 = list(X)
	n = len(X)
	for e in range(m):
		for i in range(n):
			b = X2[i] & 1
			X2[i] >>= 1
			z |= b << e * n + i
	return z
def deinterleave(z, n=1):
	"Unpacks a bit interleaved integer into a list of integers."
	m = z.bit_length()
	X = [0] * n
	for i in range(m):
		if (z >> i) & 1:
			X[i % n] |= 1 << i // n
	return X

def encode_snowflake(*args, store_count=False, minimise=False):
	timestamps = []
	ids = []
	increments = []
	for n in args:
		timestamp = n >> 22
		worker_id = (n & 0x3E0000) >> 17
		process_id = (n & 0x1F000) >> 12
		increment = n & 0xFFF
		timestamps.append(timestamp)
		ids.append(szudzik(worker_id, process_id))
		increments.append(increment)
	timestamp = interleave(*timestamps)
	z1 = interleave(*ids)
	increment = interleave(*increments)
	z2 = szudzik(z1, increment)
	encoded = leb128(z2) + n2b(timestamp)
	# 127 indicates extra padding byte
	# <127 indicates snowflake count
	# >127 indicates predetermined count
	if store_count:
		assert 1 < len(args) < 127, "Snowflake count must be between 2 and 126."
		encoded = bytes([len(args)]) + encoded
	elif encoded[0] < 128:
		encoded = b"\x7f" + encoded
	if minimise:
		return base65536.encode(encoded)
	return e64(encoded, out=str)
def decode_snowflake(data, n=1):
	decoded = b64_or_uni(data)
	if 1 < decoded[0] < 127:
		n, decoded = decoded[0], decoded[1:]
	elif decoded[0] == 127:
		decoded = decoded[1:]
	z2, b = decode_leb128(decoded)
	timestamp = b2n(b)
	z1, increment = iszudzik(z2)
	increments = deinterleave(increment, n)
	ids = deinterleave(z1, n)
	timestamps = deinterleave(timestamp, n)
	args = []
	for timestamp, id, increment in zip(timestamps, ids, increments):
		worker_id, process_id = iszudzik(id)
		n = (timestamp << 22) | (worker_id << 17) | (process_id << 12) | increment
		args.append(n)
	return args

def split_url(url, mid):
	_, cid, aid, fn = url.split("?", 1)[0].rsplit("/", 3)
	return (int(cid), int(mid) if mid is not None else None, int(aid), fn)
def merge_url(cid, mid, aid, fn):
	return f"https://cdn.discordapp.com/attachments/{cid}/{aid}/{fn}", mid
def merge_attachment(func) -> collections.abc.Callable:
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		if len(args) != 2:
			args = merge_url(*args)
		return func(*args, **kwargs)
	return wrapper
def split_attachment(func) -> collections.abc.Callable:
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		if len(args) != 4:
			args = split_url(*args)
		return func(*args, **kwargs)
	return wrapper

def serialise_nums(nums: list) -> bytearray:
	data = bytearray()
	m = 0
	for n in nums:
		data.extend(leb128(n ^ m))
		m = n
	return data
def deserialise_nums(b: bytearray) -> list:
	nums = []
	b = memoryview(b)
	m = 0
	while b:
		n, b = decode_leb128(b)
		m ^= n
		nums.append(m)
	return nums

@split_attachment
def encode_attachment(cid, mid, aid, fn, minimise=False):
	if is_url(fn):
		fn = url2fn(fn)
	if minimise:
		aid = 0
	if aid == 0:
		url = encode_snowflake(*map(int, (cid, mid)), store_count=True, minimise=minimise)
		if not minimise:
			url += f"/{fn}"
		return url
	return encode_snowflake(*map(int, (cid, mid, aid))) + f"/{fn}"
def decode_attachment(encoded):
	data, *fn = encoded.split("/", 1)
	ids = list(decode_snowflake(data, 3))
	while len(ids) < 3:
		ids.append(0)
	ids.extend(fn)
	if len(ids) < 4:
		ids.append("")
	return ids

@split_attachment
def shorten_attachment(cid, mid, aid, fn, mode="u", size=0, base="https://mizabot.xyz", minimise=False):
	url = f"{base}/{mode}/" + encode_attachment(cid, mid, aid, fn, minimise=minimise)
	if size:
		url += f"?size={size}"
	return url
def expand_attachment(url):
	assert "//" in url, "Expected shortened URL."
	regs = regexp(r"\/\w\/").split(unquote_plus(url.split("?", 1)[0]), 1)
	assert len(regs) == 2, "Invalid shortened URL."
	encoded = regs[-1]
	return decode_attachment(encoded)

def group_attachments(size_mb, cid, mids, minimise=False):
	i = cid
	b = leb128(size_mb) + leb128(i)
	for m in mids:
		m, i = m ^ i, m
		b += leb128(m)
	return base65536.encode(b) if minimise else e64(b, out=str)
def ungroup_attachments(b):
	b = MemoryBytes(b64_or_uni(b))
	size_mb, b = decode_leb128(b)
	i, b = decode_leb128(b)
	ids = [i]
	while b:
		m, b = decode_leb128(b)
		i ^= m
		ids.append(i)
	return size_mb, ids.pop(0), ids

def shorten_chunks(size_mb, cid, mids, fn, mode="c", base="https://mizabot.xyz", minimise=False):
	encoded = group_attachments(size_mb, cid, mids, minimise=minimise)
	url = f"{base}/{mode}/{encoded}"
	if not minimise:
		url += f"/{fn}"
	return url
def expand_chunks(url):
	path = url.split("?", 1)[0].split("/c/", 1)[-1]
	if "/" not in path:
		path += "/"
	path, fn = path.split("/", 1)
	size_mb, cid, mids = ungroup_attachments(path)
	return size_mb, cid, mids, fn

def p2n(b):
	"Converts a urlsafe-base64 string to big-endian integer."
	return b2n(b64(b))
def n2p(n):
	"Converts a big-endian integer to unpadded urlsafe-base64 representation."
	return e64(n2b(n))


enc_key = None
with tracebacksuppressor:
	enc_key = AUTH["encryption_key"]

if not enc_key:
	enc_key = (AUTH.get("discord_secret") or AUTH.get("discord_token") or e64(randbytes(32), out=str)).replace(".", "A").replace("_", "a").replace("-", "a")[:43]
	while len(enc_key) < 43:
		enc_key += "A"
	AUTH["encryption_key"] = enc_key 
	save_auth(AUTH)

enc_key += "=="
enc_box = nacl.secret.SecretBox(b64(enc_key)[:32])

def encrypt(s): 
	if not isinstance(s, byte_like):
		s = str(s).encode("utf-8")
	return b">~MIZA~>" + enc_box.encrypt(s)
def decrypt(s):
	if not isinstance(s, byte_like):
		s = str(s).encode("utf-8")
	assert s[:8] == b">~MIZA~>", "Data header not found."
	return enc_box.decrypt(s[8:])
estream_size = 1048576
def encrypt_stream(s):
	if not isinstance(s, byte_like):
		s = str(s).encode("utf-8")
	yield b">~MIZA+~>" + leb128(estream_size)
	for i in range(0, len(s), estream_size):
		yield enc_box.encrypt(s[i:i + estream_size])
def decrypt_stream(b):
	s = next(b)
	while len(s) < 9:
		s += next(b)
	assert s[:9] == b">~MIZA+~>", "Data header not found."
	s = s[9:]
	try:
		while len(s) < estream_size:
			s += next(b)
	except StopIteration:
		pass
	es, s = decode_leb128(s)
	ss = es + enc_box.NONCE_SIZE + enc_box.MACBYTES
	try:
		while True:
			while len(s) < ss:
				s += next(b)
			yield enc_box.decrypt(s[:ss])
			s = s[ss:]
	except StopIteration:
		yield enc_box.decrypt(s)


DOMAIN_CERT = AUTH.get("domain_cert")
PRIVATE_KEY = AUTH.get("private_key")


def zip2bytes(data):
	if data[:1] == b"~":
		import lzma
		return lzma.decompress(memoryview(data)[1:])
	if data[:1] == b"!":
		import zlib
		return zlib.decompress(memoryview(data)[1:])
	if not hasattr(data, "read"):
		data = io.BytesIO(data)
	with zipfile.ZipFile(data, allowZip64=True, strict_timestamps=False) as z:
		return z.read(z.namelist()[0])

def bytes2zip(data, lzma=True):
	import zlib
	if lzma:
		a = b"!" + zlib.compress(data)
		import lzma
		b = b"~" + lzma.compress(data)
		if len(b) < len(a):
			return b
		return a
	return b"!" + zlib.compress(data)

def eval_json(s):
	"Safer than raw eval, more powerful than json.loads. No global variables are provided."
	if not isinstance(s, str | bytes):
		s = bytes(s)
	if isinstance(s, byte_like) and s.startswith(b'b64("') and s.endswith(b'")'):
		return b64(s[5:-2])
	try:
		if len(s) > 24 * 1048576:
			return orjson.loads(s)
		else:
			return json.loads(s)
	except orjson.JSONDecodeError:
		try:
			import json5
			return json5.loads(s)
		except ValueError:
			pass
		cond = ("__" not in s or "." not in s) if isinstance(s, str) else (b"__" not in s or b"." not in s)
		if cond:
			try:
				if len(s) > 1048576:
					return esubmit(safe_eval, s, priority=2).result()
				return safe_eval(s)
			except Exception:
				pass
		raise

bidict = __builtins__
if not isinstance(bidict, dict):
	bidict = bidict.__dict__
def maybe_json(d):
	if isinstance(d, BaseException):
		cn = d.__class__.__name__
		if cn not in bidict:
			return f"RuntimeError({repr(cn)},{','.join(map(repr, d.args))})".encode("utf-8")
		return repr(d).encode("utf-8")
	if isinstance(d, byte_like):
		if not isinstance(d, (bytes, memoryview, bytearray)):
			d = bytes(d)
		return (b'b64("' + e64(d) + b'")')
	try:
		return json.dumps(d, cls=MultiEncoder).encode("ascii")
	except TypeError:
		return json_dumps(repr(d))

def json_if(s):
	if isinstance(s, str) and len(s.split(None, 1)) > 1:
		return json.dumps(s)
	return s

@always_copy
@functools.lru_cache(maxsize=256)
def select_and_loads(s, encrypted=False, size=None, safe=False):
	"Automatically decodes data from JSON or Pickle, decompressing if necessary."
	if not s:
		raise ValueError("Data must not be empty.")
	if size and size < len(s):
		raise OverflowError("Data input size too large.")
	if isinstance(s, str):
		s = s.encode("utf-8")
	if encrypted:
		try:
			s = decrypt(s)
		except ValueError:
			pass
		except:
			raise
	if s[0] == 128 and not safe:
		return pickle.loads(s)
	if s[:1] in (b"~", b"!") or zipfile.is_zipfile(io.BytesIO(s)):
		s = zip2bytes(s)
	data = None
	if not s:
		return data
	if s[0] == 128 and not safe:
		return pickle.loads(s)
	if data is None:
		tcls = None
		if s[0] in b"$" and s[1] in b"[":
			s = memoryview(s)[1:]
			tcls = set
		elif s[0] not in b"{[" and b"{" in s:
			s = memoryview(s)[s.index(b"{"):s.rindex(b"}") + 1]
		data = orjson.loads(s)
		if tcls:
			data = tcls(data)
	return data

def select_and_dumps(data, safe=True, compress=True):
	"Automatically serialises data as JSON or Pickle, compressing if beneficial."
	if isinstance(data, Future):
		data = data.result(timeout=2)
	if not safe:
		s = pickle.dumps(data)
		if len(s) > 32768 and compress:
			t = bytes2zip(s, lzma=len(s) > 16777216)
			if len(t) < len(s) * 0.9:
				s = t
		return s
	try:
		s = maybe_json(data)
	except (TypeError, orjson.JSONEncodeError):
		s = None
	if len(s) > 262144:
		t = bytes2zip(s, lzma=False)
		if len(t) < len(s) * 0.9:
			s = t
	return s


def safe_save(fn, s):
	"Writes data to a file, creating a temporary backup which is then swapped with the destination file. This operation is less susceptible to corruption upon a crash."
	if os.path.exists(fn):
		save_file(s, fn + "\x7f")
		with open(fn + "\x7f", "rb") as f:
			if f.read(1) in (b"\x00", b" ", b""):
				raise ValueError
		with tracebacksuppressor(FileNotFoundError):
			os.remove(fn + "\x7f\x7f")
	if os.path.exists(fn) and not os.path.exists(fn + "\x7f\x7f"):
		os.rename(fn, fn + "\x7f\x7f")
		os.rename(fn + "\x7f", fn)
	else:
		save_file(s, fn)


reqs = alist(requests.Session() for i in range(6))

class open2(io.IOBase):
	"A file-compatible open function that wraps already open files."

	__slots__ = ("fp", "fn", "mode", "filename")

	def __init__(self, fn, mode="rb", filename=None):
		self.fp = None
		self.fn = fn
		self.mode = mode
		self.filename = filename or T(fn).get("name") or fn

	def __getattribute__(self, k):
		if k in object.__getattribute__(self, "__slots__") or k == "clear":
			return object.__getattribute__(self, k)
		if k == "name":
			return object.__getattribute__(self, "filename")
		if self.fp is None:
			self.fp = open(self.fn, self.mode)
		if k[0] == "_" and (len(k) < 2 or k[1] != "_"):
			k = k[1:]
		return getattr(self.fp, k)

	def clear(self):
		with suppress(Exception):
			self.fp.close()
		self.fp = None


class RNGFile(io.IOBase):

	def __init__(self, count=1073741824):
		self.pos = 0
		self.count = count
		assert isinstance(self.count, int)

	def seek(self, offset=0, whence=0):
		match whence:
			case 0:
				self.pos = offset
			case 1:
				self.pos += offset
			case 2:
				self.pos = self.count + offset
			case _:
				raise NotImplementedError(whence)
		return self.pos

	def read(self, count=-1):
		if count < 0:
			b = random.randbytes(self.count - self.pos)
			self.pos = self.count
			return b
		n = min(count, self.count - self.pos)
		self.pos += n
		return random.randbytes(n)

	def tell(self):
		return self.pos


class TeeBuffer:
	def __init__(self, src, chunk_size=8192):
		self.src = src
		self.chunk_size = chunk_size
		self.buffer = bytearray()
		self.eof = False
		self.lock = threading.Lock()
		self.data_ready = threading.Condition(self.lock)

	def _fill_to(self, size):
		"""Ensure buffer has at least `size` bytes (if possible)."""
		if self.eof:
			return
		with self.lock:
			while len(self.buffer) < size and not self.eof:
				chunk = self.src.read(self.chunk_size)
				if not chunk:
					self.eof = True
					self.data_ready.notify_all()
					break
				self.buffer.extend(chunk)
				self.data_ready.notify_all()

	def open(self):
		"""Return a new independent file-like reader."""
		return _TeeReader(self)

class _TeeReader(io.RawIOBase):
	def __init__(self, tee):
		self.tee = tee
		self.pos = 0

	def seek(self, offset=0, whence=0):
		match whence:
			case 0:
				self.pos = offset
			case 1:
				self.pos += offset
			case 2:
				for i in itertools.count(1):
					if self.tee.eof:
						break
					self.tee._fill_to(len(self.tee.buffer) + i * 1024)
				self.pos = len(self.tee.buffer) + offset
			case _:
				raise NotImplementedError(whence)
		return self.pos

	def read(self, n=-1):
		if n == -1:
			out = bytearray()
			for i in itertools.count(1):
				chunk = self.read(i * 1024)
				if not chunk:
					break
				out.extend(chunk)
			return out

		target = self.pos + n
		self.tee._fill_to(target)
		with self.tee.lock:
			while len(self.tee.buffer) < target and not self.tee.eof:
				self.tee.data_ready.wait()
			data = memoryview(self.tee.buffer).toreadonly()[self.pos:min(target, len(self.tee.buffer))]
			self.pos += len(data)
			return data


class CachingTeeFile:
	def __init__(self, src, cache_path=None, chunk_size=1024, callback=None):
		self.src = src
		self.chunk_size = chunk_size
		self.cache_path = cache_path or temporary_file("bin")
		self._cache_file = open(self.cache_path, "wb")
		self._pos = 0
		self._written = 0
		self._eof = False
		self._error = None
		self._callback = callback

		self._lock = threading.Lock()
		self._data_ready = threading.Condition(self._lock)
		self._stop_event = threading.Event()

		self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
		self._writer_thread.start()

	def _writer_loop(self):
		i = 1
		try:
			with self._cache_file:
				while not self._stop_event.is_set():
					# Read a chunk; may block
					chunk = self.src.read(self.chunk_size * i)
					if not chunk:
						break

					# Snapshot current written position under lock
					with self._lock:
						pos = self._written

					# Do I/O outside the lock
					self._cache_file.seek(pos)
					self._cache_file.write(chunk)
					self._cache_file.flush()

					# Publish the new bytes and notify
					with self._lock:
						self._written += len(chunk)
						self._data_ready.notify_all()

					i += 1
		except Exception as e:
			# If we were asked to stop, treat exceptions as normal shutdown
			if not self._stop_event.is_set():
				with self._lock:
					self._error = e
		finally:
			# Always mark terminal state and wake readers
			with self._lock:
				self._eof = True
				self._data_ready.notify_all()
			if self._callback:
				try:
					self._callback()
				except Exception:
					pass

	def open(self):
		return _CachingReader(self)

	def wait_until_complete(self, timeout=None):
		self._writer_thread.join(timeout=timeout)

	def close(self):
		# Signal stop and forcibly close the source to unblock read()
		self._stop_event.set()
		try:
			self.src.close()
		except Exception:
			pass
		# Wait for the background writer to exit
		self._writer_thread.join()

class _CachingReader(io.RawIOBase):
	def __init__(self, parent):
		self.parent = parent
		self.file = open(parent.cache_path, "rb")
		self.pos = 0

	def readable(self):
		return True

	def seek(self, offset=0, whence=0):
		match whence:
			case 0:
				self.pos = offset
			case 1:
				self.pos += offset
			case _:
				raise NotImplementedError(whence)
		return self.pos

	def read(self, n=-1):
		if n == 0:
			return b""

		parent = self.parent
		with parent._lock:
			while True:
				available = parent._written - self.pos

				if n is None or n < 0:
					# read-all: wait until EOF or error (or any available if already eof)
					if parent._eof or parent._error:
						break
				else:
					# Wait until we have at least n bytes, or EOF, or error
					if available >= n or parent._eof or parent._error:
						break

				parent._data_ready.wait()

			# Propagate writer error to readers
			if parent._error is not None:
				raise parent._error

			# Determine how much to read
			if n is None or n < 0:
				read_len = parent._written - self.pos
			else:
				read_len = min(n, parent._written - self.pos)

		if read_len <= 0:
			return b""

		self.file.seek(self.pos)
		data = self.file.read(read_len)
		self.pos += len(data)
		return data

	def close(self):
		self.file.close()


class FileStreamer(io.BufferedRandom, contextlib.AbstractContextManager):
	"A buffer-compatible file object that treats multiple files or buffers as a single concatenated one."

	def __init__(self, *objs, filename=None, name=None):
		self.pos = 0
		self.data = []
		self.filename = filename or name
		i = 0
		objs = list(objs)
		while objs:
			f = objs.pop(0)
			if isinstance(f, (list, tuple)):
				objs.extend(f)
				continue
			if isinstance(f, (bytes, memoryview)):
				f = io.BytesIO(f)
			elif isinstance(f, str):
				f = open(f, "rb")
			elif T(f).get("fp"):
				f = f.fp
			self.data.append((i, f))
			i += f.seek(0, os.SEEK_END)
			self.filename = self.filename or T(f).get("filename") or T(f).get("name")

	def seek(self, pos=0):
		self.pos = pos
		return self.pos

	def read(self, size=None):
		out = []
		size = size if size is not None else inf
		while size:
			t = None
			for i, f in self.data:
				if i > self.pos:
					break
				t = (i, f)
			if not t:
				break
			p = self.pos - t[0]
			t[1].seek(p)
			b = t[1].read(min(size, 4294967296))
			if not b:
				break
			out.append(b)
			size -= len(b)
			self.pos += len(b)
		return b"".join(out)

	def close(self):
		data = [f for i, f in self.data]
		self.data.clear()
		for f in data:
			with tracebacksuppressor:
				f.close()

	isatty = lambda self: False					# noqa: E731
	flush = lambda self: None					# noqa: E731
	writable = lambda self: False				# noqa: E731
	seekable = lambda self: True				# noqa: E731
	readable = lambda self: True				# noqa: E731
	tell = lambda self: self.pos				# noqa: E731
	__enter__ = lambda self: self				# noqa: E731
	__exit__ = lambda self, *args: self.close()	# noqa: E731

class PipedProcess:
	"""A class for managing piped subprocesses.
	This class allows creating and managing a chain of subprocess where the output of each process
	is piped to the input of the next process in the chain.
	Attributes:
		procs (tuple): Tuple of running processes.
		stdin: Standard input stream of first process.
		stdout: Standard output stream of last process.
		stderr: Standard error stream of last process.
		pid: Process ID of the first process in the chain.
	Examples:
		>>> # Chain 'cat file.txt' and 'grep pattern'
		>>> p = PipedProcess(['cat', 'file.txt'], ['grep', 'pattern'], stdout=subprocess.PIPE)
		>>> p.wait()  # Wait for all processes to complete
		>>> print(p.stdout.read().decode()) # Print the output of the last process
	Args:
		*args: Command arguments for each process in the chain.
		stdin: Custom stdin for first process. Defaults to None.
		stdout: Custom stdout for last process. Defaults to None.
		stderr: Custom stderr for last process. Defaults to None.
		cwd (str): Working directory for processes. Defaults to current directory.
		bufsize (int): Buffer size for pipe operations. Defaults to 4096.
	"""

	procs = ()
	stdin = stdout = stderr = None

	def __init__(self, *args, stdin=None, stdout=None, stderr=None, cwd=".", bufsize=4096, shell=False):
		if not args:
			return
		self.procs = []
		proc = None
		for i, arg in enumerate(args):
			first = not i
			last = i >= len(args) - 1
			si = stdin if first else proc.stdout
			so = stdout if last else subprocess.PIPE
			se = stderr if last else None
			proc = psutil.Popen(arg, stdin=si, stdout=so, stderr=se, cwd=cwd, bufsize=bufsize * 256, shell=shell or isinstance(arg, str))
			if first:
				self.stdin = proc.stdin
				self.args = arg
			if last:
				self.stdout = proc.stdout
				self.stderr = proc.stderr
			self.procs.append(proc)
		self.pid = self.procs[0].pid

	def is_running(self):
		for proc in self.procs:
			if proc.is_running():
				return True
		return False

	def terminate(self):
		for proc in self.procs:
			proc.terminate()

	def kill(self):
		for proc in self.procs:
			proc.kill()

	def wait(self):
		for proc in self.procs:
			proc.wait()

	def status(self):
		try:
			return self.procs[-1].status()
		except psutil.NoSuchProcess:
			return "terminated"

class seq(io.BufferedRandom, collections.abc.Sequence, contextlib.AbstractContextManager):
	"A Sequence implementation that attempts to turn buffer objects into indexable array-like objects."

	BUF = 262144
	iter = None

	def __init__(self, obj, filename=None, buffer_size=None):
		if buffer_size:
			self.BUF = buffer_size
		self.closer = T(obj).get("close")
		self.high = 0
		self.finished = False
		if isinstance(obj, io.IOBase) or hasattr(obj, "read"):
			if isinstance(obj, io.BytesIO):
				self.data = obj
				self.finished = True
			elif hasattr(obj, "getbuffer"):
				self.data = io.BytesIO(obj.getbuffer())
				self.finished = True
			else:
				if hasattr(obj, "seek"):
					obj.seek(0)
				def obj_iter(fp):
					b = fp.read(self.BUF)
					if not b:
						raise StopIteration
					yield b
				self.iter = obj_iter(obj)
				self.data = io.BytesIO()
		elif isinstance(obj, bytes) or isinstance(obj, bytearray) or isinstance(obj, memoryview):
			self.data = io.BytesIO(obj)
			self.high = len(obj)
			self.finished = True
		elif isinstance(obj, collections.abc.Iterable):
			self.iter = iter(obj)
			self.data = io.BytesIO()
		elif T(obj).get("iter_content"):
			self.iter = obj.iter_content(self.BUF)
			self.data = io.BytesIO()
		else:
			raise TypeError(f"a bytes-like object is required, not '{type(obj)}'")
		self.filename = filename
		self.buffer = {}
		self.pos = 0
		self.limit = None

	def __len__(self):
		return self.limit or max(k + len(v) for k, v in self.buffer.items()) if self.buffer else 0

	seekable = lambda self: True	# noqa: E731
	readable = lambda self: True	# noqa: E731
	writable = lambda self: False	# noqa: E731
	isatty = lambda self: False		# noqa: E731
	flush = lambda self: None		# noqa: E731
	tell = lambda self: self.pos	# noqa: E731

	def seek(self, pos=0):
		self.pos = pos
		return self.pos

	def read(self, size=None):
		out = self.peek(size)
		self.pos += len(out)
		return out

	def peek(self, size=None):
		if not size:
			if self.limit is not None:
				return self[self.pos:self.limit]
			return self[self.pos:]
		if self.limit is not None:
			return self[self.pos:min(self.pos + size, self.limit)]
		return self[self.pos:self.pos + size]

	def truncate(self, limit=None):
		self.limit = limit

	def fileno(self):
		raise OSError

	def __getitem__(self, k):
		if self.finished:
			return self.data.getbuffer()[k]
		if type(k) is slice:
			start = k.start or 0
			stop = k.stop or inf
			step = k.step or 1
			rev = step < 0
			if rev:
				start, stop, step = stop + 1, start + 1, -step
			curr = start // self.BUF * self.BUF
			out = deque()
			out.append(self.load(curr))
			curr += self.BUF
			while curr < stop:
				temp = self.load(curr)
				if not temp:
					break
				out.append(temp)
				curr += self.BUF
			b = memoryview(b"".join(out))
			b = b[start % self.BUF:]
			if isfinite(stop):
				b = b[:stop - start]
			if step != 1:
				b = b[::step]
			if rev:
				b = b[::-1]
			return b
		base = k // self.BUF
		with suppress(Exception):
			return self.load(base)[k % self.BUF]
		raise IndexError("seq index out of range")

	def __str__(self):
		if self.filename is None:
			return str(self.data)
		if self.filename:
			return f"<seq name='{self.filename}'>"
		return f"<seq object at {hex(id(self))}"

	def __iter__(self):
		i = 0
		while True:
			try:
				x = self[i]
			except IndexError:
				break
			if x:
				yield x
			else:
				break
			i += 1

	def __getattribute__(self, k):
		if k in ("name", "filename"):
			try:
				return object.__getattribute__(self, "filename")
			except AttributeError:
				k = "name"
		else:
			try:
				return object.__getattribute__(self, k)
			except AttributeError:
				pass
		return object.__getattribute__(self.data, k)

	close = lambda self: self.closer() if self.closer else None		# noqa: E731
	__enter__ = lambda self: self									# noqa: E731
	__exit__ = lambda self, *args: self.close()						# noqa: E731

	def load(self, k):
		if self.finished:
			return self.data.getbuffer()[k:k + self.BUF]
		try:
			return self.buffer[k]
		except KeyError:
			pass
		seek = T(self.data).get("seek")
		if seek:
			if self.iter is not None and k + self.BUF >= self.high:
				out = deque()
				try:
					while k + self.BUF >= self.high:
						temp = next(self.iter)
						if not temp:
							raise StopIteration
						out.append(temp)
						self.high += len(temp)
				except StopIteration:
					out.appendleft(self.data.getbuffer())
					self.data = io.BytesIO(b"".join(out))
					self.finished = True
					return self.data.getbuffer()[k:k + self.BUF]
				out.appendleft(self.data.getbuffer())
				self.data = io.BytesIO(b"".join(out))
			self.buffer[k] = b = self.data.getbuffer()[k:k + self.BUF]
			return b
		try:
			while self.high < k:
				temp = next(self.data)
				if not temp:
					raise StopIteration
				if self.high in self.buffer:
					self.buffer[self.high] += temp
				else:
					self.buffer[self.high] = temp
				self.high += self.BUF
		except StopIteration:
			self.data = io.BytesIO(b"".join(self.buffer.values()))
			self.finished = True
			return self.data.getbuffer()[k:k + self.BUF]
		return self.buffer.get(k, b"")

class Stream(io.IOBase):

	BUF = 262144
	resp = None

	def __init__(self, url):
		self.url = url
		self.buflen = 0
		self.buf = io.BytesIO()
		self.reset()
		self.refill()

	def reset(self):
		if self.resp:
			with suppress(Exception):
				self.resp.close()
		self.resp = requests.get(self.url, stream=True)
		self.iter = self.resp.iter_content(self.BUF)

	def refill(self):
		att = 0
		while self.buflen < self.BUF * 4:
			try:
				b = next(self.iter)
				self.buf.write(b)
			except StopIteration:
				with suppress(Exception):
					self.resp.close()
				return
			except Exception:
				if att > 16:
					raise
				att += 1
				self.reset()
			else:
				self.buflen += len(b)
		with suppress(Exception):
			self.resp.close()


class FileHashDict(collections.abc.MutableMapping):
	"A dictionary-compatible object that represents a mapping to SQL tables stored on disk."

	sem = Semaphore(64, 128, 0.3, 1)
	db_sems = {}
	cache_size = 4096
	encoder = [None, None]
	max_concurrency = 8

	def __init__(self, *args, path="", encode=None, decode=None, automut=True, autosave=60, safe=False, **kwargs):
		if not kwargs and len(args) == 1:
			self.data = args[0]
		else:
			self.data = dict(*args, **kwargs)
		if encode:
			self.encoder[0] = encode
		if decode:
			self.encoder[1] = decode
		path = path.rstrip("/")
		self.internal = path
		self.path = os.path.abspath(path + ".sql3")
		self.modified = set()
		self.deleted = set()
		self.iter = None
		if os.path.exists(path) and os.path.isdir(path) and os.path.exists(path + "/~~"):
			os.rename(path + "/~~", self.path)
			shutil.rmtree(path)
		self.load_cursor()
		self.db.commit()
		self.db_sem = self.db_sems.setdefault(path, Semaphore(self.max_concurrency, inf))
		self.c_updated = False
		self.modifying = None
		self.automut = automut
		self.autosave = autosave
		self.safe = safe
		self.tp = concurrent.futures.ThreadPoolExecutor(max_workers=2)

	@property
	def encode(self):
		return self.encoder[0] or nop2

	@property
	def decode(self):
		return self.encoder[1] or nop2

	db = None
	def load_cursor(self):
		if self.db:
			self.db.close()
		self.db = sqlite3.connect(self.path, check_same_thread=False)
		self.cur = alist([self.db.cursor() for i in range(self.max_concurrency)])
		try:
			self.cur[-1].execute(f"CREATE TABLE IF NOT EXISTS '{self.internal}' (key VARCHAR(256) PRIMARY KEY, value BLOB)")
		except Exception:
			print("Error in database", self)
			raise
		self.cur[-1].execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON '{self.internal}' (key)")
		self.codb = set(try_int(r[0]) for r in self.cur[-1].execute(f"SELECT key FROM '{self.internal}'") if r)
		return self.codb

	def __hash__(self):
		return lambda self: hash(self.path)
	def __str__(self):
		s = self.__class__.__name__ + "(" + str(self.data)
		if self.path:
			s += f", path={json.dumps(self.path)}"
		return s + ")"
	def __repr__(self):
		s = self.__class__.__name__ + "(" + str(self.full)
		if self.path:
			s += f", path={json.dumps(self.path)}"
		return s + ")"
	def __call__(self, k):
		return self.__getitem__(k)
	def __len__(self):
		return len(self.keys())
	def __contains__(self, k):
		return k in self.keys().to_frozenset()
	def __eq__(self, other):
		return self.data == other
	def __ne__(self, other):
		return self.data != other

	rmap = {
		"\\": "\U0001fb00",
		"/": "\U0001fb01",
		":": "\U0001fb02",
		"*": "\U0001fb03",
		"?": "\U0001fb04",
		'"': "\U0001fb05",
		"'": "\U0001fb06",
		"<": "\U0001fb07",
		">": "\U0001fb08",
		"|": "\U0001fb09",
		"%": "\U0001fb0a",
	}
	rtrans = "".maketrans(rmap)
	dtrans = "".maketrans({v: k for k, v in rmap.items()})
	def remap(self, k):
		return k.translate(self.rtrans)
	def demap(self, k):
		return k.translate(self.dtrans)

	@property
	def full(self):
		out = {}
		waits = set()
		for k in self.keys():
			try:
				out[k] = self.data[k]
			except KeyError:
				out[k] = esubmit(self.__getitem__, k)
				waits.add(k)
		for k in waits:
			out[k] = out[k].result()
		return out

	def keys(self):
		if self.iter is None:
			gen = set()
			if self.modified:
				gen.update(self.modified)
			if self.deleted:
				gen.difference_update(self.deleted)
			if self.codb:
				gen.update(self.codb)
			gen.discard("--datestr")
			self.iter = alist(gen)
		return self.iter

	def values(self):
		for k in self.keys():
			with suppress(KeyError):
				yield self[k]

	def items(self):
		for k in self.keys():
			with suppress(KeyError):
				yield (k, self[k])

	def __iter__(self):
		return iter(self.keys())

	def __reversed__(self):
		return reversed(self.keys())

	def __getitem__(self, k):
		if k in self.deleted:
			raise KeyError(k)
		try:
			value = self.data[k]
		except KeyError:
			pass
		else:
			if self.automut and not isinstance(value, collections.abc.Hashable):
				self.modified.add(k)
				self.modify()
			return value
		if k in self.codb:
			try:
				with self.db_sem:
					s = next(self.cur.next().execute(f"SELECT value FROM '{self.internal}' WHERE key=?", [k]))[0]
			except StopIteration:
				raise KeyError(k)
			if not s:
				self.deleted.add(k)
				raise KeyError(k)
			d = self.decode(s)
			value = select_and_loads(d)
			self.data[k] = value
			if self.automut and not isinstance(value, collections.abc.Hashable):
				self.modified.add(k)
				self.modify()
			return value
		if isinstance(k, str) and k.startswith("~"):
			raise RuntimeError("Attempted to load SQL database inappropriately")
		raise KeyError(k)

	def __setitem__(self, k, v):
		k = try_int(k)
		self.deleted.discard(k)
		self.data[k] = v
		self.modified.add(k)
		self.modify()

	def get(self, k, default=None):
		with suppress(KeyError):
			return self[k]
		return default

	def coerce(self, k, cls=None, default=Dummy):
		return coerce(self, k, cls, default)

	def coercedefault(self, k, cls=None, default=Dummy):
		return coercedefault(self, k, cls, default)

	def updatedefault(self, other):
		return updatedefault(self, other)

	def pop(self, k, *args, force=False, remove=True):
		self.modify()
		if remove:
			if k in self.codb:
				self.db_sem.pause()
				try:
					self.cur.next().execute(f"DELETE FROM '{self.internal}' WHERE key=?", [k])
				finally:
					self.db_sem.resume()
				self.codb.discard(k)
				self.c_updated = True
			self.deleted.add(k)
		if force:
			out = self[k]
			return self.data.pop(k, out)
		return self.data.pop(k, None)

	__delitem__ = pop

	def popitem(self, k):
		try:
			return self.data.popitem(k)
		except KeyError:
			out = self[k]
		self.pop(k)
		return (k, out)

	def discard(self, k):
		with suppress(KeyError):
			return self.pop(k)

	def setdefault(self, k, v):
		try:
			return self[k]
		except KeyError:
			self[k] = v
		return v

	def _modify(self):
		if self.autosave is None:
			return
		time.sleep(self.autosave)
		self.sync()

	def modify(self):
		self.iter = None
		if not self.modifying or self.modifying.done():
			self.modifying = self.tp.submit(self._modify)
		return self

	def parallel(self, func, *args, **kwargs):
		if os.environ.get("IS_BOT"):
			return esubmit(func, *args, priority=2, **kwargs)
		return self.tp.submit(func, *args, **kwargs)

	def update(self, other):
		"Updates this database with data from a dictionary-compatible object."
		self.modified.update(other)
		self.deleted.difference_update(other)
		self.data.update(other)
		return self.modify()

	def fill(self, other):
		"Replaces all the data in this database with data from a dictionary-compatible object."
		if not other:
			return self.clear()
		self.modified.update(other)
		self.deleted.difference_update(other)
		self.data.update(other)
		temp = set(self)
		temp.difference_update(other)
		self.deleted.update(temp)
		return self.modify()

	def clear(self):
		"Empties the database. Affects both memory and disk layers."
		if len(self):
			print("WARNING: Clearing", self)
		self.modified.clear()
		self.deleted.clear()
		self.data.clear()
		self.db_sem.pause()
		try:
			self.cur.next().execute(f"DELETE FROM '{self.internal}'")
		finally:
			self.db_sem.resume()
		self.unload()
		self.db_sem.pause()
		try:
			self.codb.clear()
			try:
				os.remove(self.path)
			except (PermissionError, FileNotFoundError):
				pass
			self.load_cursor()
			self.db.commit()
		finally:
			self.db_sem.resume()
		return self.modify()

	def sync(self):
		"Saves all changes to the database."
		vac = False
		datestr = str(datetime.datetime.now(tz=datetime.timezone.utc).date())
		if datestr != self.get("--datestr", None):
			vac = True
			self.data["--datestr"] = datestr
			self.modified.add("--datestr")
			self.deleted.discard("--datestr")
		modified = set(self.modified)
		self.modified.clear()
		deleted = set(self.deleted)
		self.deleted.clear()
		inter = modified.intersection(deleted)
		modified.difference_update(deleted)
		if modified or deleted:
			self.iter = None
		ndel = deleted and deleted.intersection(self.codb)
		if ndel:
			self.db_sem.pause()
			try:
				for k in ndel:
					self.cur.next().execute(f"DELETE FROM '{self.internal}' WHERE key=?", [k])
			finally:
				self.db_sem.resume()
			self.codb.difference_update(ndel)
			self.c_updated = True
		if modified:
			mods = {}
			futs = []
			for k in modified:
				try:
					d = self.data[k]
				except KeyError:
					self.deleted.add(k)
					continue
				futs.append((k, self.parallel(select_and_dumps, d, safe=self.safe, compress=True)))
			fut2 = []
			for k, fut in futs:
				b = fut.result()
				if self.encoder[0] is None:
					f = Future()
					f.set_result(b)
					fut2.append((k, f))
				else:
					fut2.append((k, self.parallel(self.encode, b)))
			for k, fut in fut2:
				s = fut.result()
				mods[k] = s
			self.db_sem.pause()
			try:
				for k, d in mods.items():
					self.cur.next().execute(f"INSERT INTO '{self.internal}' ('key', 'value') VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET 'value' = ?", [k, d, d])
			finally:
				self.db_sem.resume()
			self.c_updated = True
			deleted.update(modified)
		if self.c_updated:
			self.c_updated = False
			self.db_sem.pause()
			try:
				self.db.commit()
				self.codb = set(try_int(r[0]) for r in self.cur.next().execute(f"SELECT key FROM '{self.internal}'") if r)
			finally:
				self.db_sem.resume()
		if len(self.data) > self.cache_size * 2:
			self.data.clear()
		else:
			while len(self.data) > self.cache_size:
				with suppress(KeyError, RuntimeError):
					self.data.pop(next(iter(self.data)))
		if vac:
			self.vacuum()
		return inter

	def unload(self):
		self.db_sem.pause()
		try:
			self.db.commit()
			self.db.close()
		finally:
			self.db_sem.resume()

	def vacuum(self):
		"Commits and vacuums the database in its current state, freeing up disk space."
		self.unload()
		self.db_sem.pause()
		try:
			args = [python, "misc/vacuum.py", self.path]
			print(args)
			subprocess.run(args)
			return self.load_cursor()
		finally:
			self.db_sem.resume()


cachecls = diskcache.FanoutCache
class AutoCache(cachecls, collections.abc.MutableMapping):
	"""
	A disk-based cache with automatic staleness detection and background refresh capabilities.
	AutoCache extends a disk cache implementation with automatic expiration, staleness checking,
	and concurrent retrieval handling. It supports both synchronous and asynchronous operations
	for retrieving and caching values.
	Attributes:
		_path (str or None): Directory path for the disk cache storage.
		_stale (float): Time in seconds after which cached items are considered stale and
			should be refreshed in the background. Defaults to 60 seconds.
		_stimeout (float): Time in seconds after which cached items are expired and must be
			refreshed before returning. Defaults to 86400 seconds (1 day).
		_retrieving (dict): Dictionary tracking ongoing retrieval operations using Futures
			to prevent duplicate concurrent retrievals of the same key.
		_kwargs (dict): Additional keyword arguments passed to the parent cache class.
		version (str): Cache version identifier used to invalidate incompatible cache formats.
	Args:
		directory (str, optional): Path to the cache directory. Defaults to None.
		shards (int, optional): Number of database shards for the cache. Defaults to 6.
		stale (float, optional): Staleness threshold in seconds. Defaults to 60.
		timeout (float, optional): Expiration timeout in seconds. Defaults to 86400.
		**kwargs: Additional arguments passed to the parent cache implementation.
	Methods:
		retrieve: Synchronously retrieve or compute a cached value with staleness/expiration handling.
		aretrieve: Asynchronously retrieve or compute a cached value with staleness/expiration handling.
		age: Get the age of a cached item in seconds.
		validate_or_clear: Check cache version compatibility and clear if mismatched.
	Notes:
		- Items are tagged with timestamps for staleness detection.
		- Stale items trigger background refresh while returning cached value.
		- Expired items block until refresh completes.
		- Handles database timeout and operational errors by reinitializing.
		- Supports concurrent retrievals with Future-based deduplication.
	"""
	__slots__ = ("_initialised", "_path", "_shardcount", "_stale", "_stimeout", "_desync", "_retrieving", "_kwargs")

	def __init__(self, directory=None, shards=6, stale=60, timeout=86400, desync=0, **kwargs):
		self._path = directory or None
		self._shardcount = shards
		self._stale = stale or inf
		self._stimeout = timeout or inf
		self._desync = desync
		self._retrieving = {}
		self._kwargs = kwargs
		self._initialised = None

	def base_init(self, force=False):
		if not force and self._initialised:
			return self._initialised.result()
		self._initialised = fut = concurrent.futures.Future()
		fut.set_result(super().__init__(self._path, shards=self._shardcount, **self._kwargs))
		print("Loaded Cache:", self._path, len(self), self._stale, self._stimeout, self._kwargs)

	def __getattr__(self, k):
		if k not in ("base_init", "__class__", "__slots__", *self.__slots__):
			self.base_init()
		try:
			return object.__getattribute__(self, k)
		except AttributeError:
			return super().__getattribute__(k)

	version = "1.0.1"
	def validate_or_clear(self):
		if self.get("__version__") != self.version:
			print(tuple(self.keys()))
			self.clear()
		self.set("__version__", self.version)

	@property
	def expire_offset(self):
		if isfinite(self._stale):
			return 2. ** 52
		return self._stimeout * ((random.random() - 0.5) * self._desync + 1)

	def __iter__(self):
		return (i for i in super().__iter__() if i != "__version__")

	def __len__(self):
		return super().__len__()

	def __contains__(self, k):
		try:
			return super().__contains__(k)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			self.base_init(force=True)
			return super().__contains__(k)

	def __getitem__(self, k):
		if (fut := self._retrieving.get(k)):
			return fut.result()
		try:
			return super().__getitem__(k)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			self.base_init(force=True)
			return super().__getitem__(k)

	def __setitem__(self, k, v, read=False):
		if isinstance(v, memoryview):
			v = bytes(v)
		super().set(k, v, expire=self.expire_offset, tag=utc(), read=read)

	def update(self, other):
		t = utc()
		if hasattr(other, "items"):
			other = other.items()
		for k, v in other:
			if isinstance(v, memoryview):
				v = bytes(v)
			super().set(k, v, expire=self.expire_offset, tag=t)
		return self

	def pop(self, k, v=Dummy):
		self._retrieving.pop(k, None)
		v = super().pop(k, default=(v,))
		if v is Dummy:
			raise KeyError(k)
		return v

	def age(self, k):
		return utc() - super().get(k, (-inf,))[-1]

	def setdefault(self, k, v):
		if (fut := self._retrieving.get(k)):
			return fut.result()
		try:
			return super().__getitem__(k)
		except KeyError:
			self[k] = v
		return v

	def _retrieve(self, k, func, *args, read=False, **kwargs):
		try:
			self._retrieving[k] = fut = concurrent.futures.Future()
			v = func(*args, **kwargs)
			self.__setitem__(k, v, read=read)
		except Exception as ex:
			fut.set_exception(ex)
			raise
		else:
			fut.set_result(v)
		finally:
			self._retrieving.pop(k, None)
		return v
	def retrieve(self, k, func, *args, _read=False, _force=False, **kwargs):
		if (fut := self._retrieving.get(k)):
			resp = fut.result()
			if _read and hasattr(resp, "name") and os.path.exists(resp.name):
				return open(resp.name, "rb")
			return resp
		try:
			v, t = super().get(k, read=_read, tag=True)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			super().__init__(self._path, shards=len(self._shards), **self._kwargs)
			self.base_init(force=True)
			v, t = super().get(k, read=_read, tag=True)
		except TypeError:
			t = None
		if t is not None and v is not Dummy:
			delay = utc() - t
			if delay > self._stimeout or _force and delay > self._stale:
				try:
					v = self._retrieve(k, func, *args, read=_read, **kwargs)
				except Exception:
					pass
			elif delay > self._stale:
				esubmit(self._retrieve, k, func, *args, read=_read, **kwargs)
			elif isinstance(v, Exception):
				raise v
		else:
			return self._retrieve(k, func, *args, read=_read, **kwargs)
		return v

	async def _aretrieve(self, k, func, *args, read=False, **kwargs):
		await asubmit(self.base_init)
		try:
			self._retrieving[k] = fut = concurrent.futures.Future()
			v = await asubmit(func, *args, **kwargs)
			await asubmit(self.__setitem__, k, v, read=read)
		except Exception as ex:
			fut.set_exception(ex)
			raise
		else:
			fut.set_result(v)
		finally:
			self._retrieving.pop(k, None)
		return v
	async def aretrieve(self, k, func, *args, _read=False, _force=False, **kwargs):
		await asubmit(self.base_init)
		if (fut := self._retrieving.get(k)):
			resp = await wrap_future(fut)
			if _read and hasattr(resp, "name") and os.path.exists(resp.name):
				return open(resp.name, "rb")
			return resp
		try:
			v, t = super().get(k, read=_read, tag=True)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			super().__init__(self._path, shards=len(self._shards), **self._kwargs)
			await asubmit(self.base_init, force=True)
			v, t = super().get(k, read=_read, tag=True)
		except TypeError:
			t = None
		if t is not None and v is not Dummy:
			delay = utc() - t
			if delay > self._stimeout or _force and delay > self._stale:
				try:
					v = await self._aretrieve(k, func, *args, read=_read, **kwargs)
				except Exception:
					pass
			elif delay > self._stale:
				csubmit(self._aretrieve(k, func, *args, read=_read, **kwargs))
			elif isinstance(v, Exception):
				raise v
		else:
			return await self._aretrieve(k, func, *args, read=_read, **kwargs)
		return v

	def keys(self):
		return iter(self)
	iterkeys = keys

	def values(self):
		for k in iter(self):
			try:
				yield cachecls.__getitem__(self, k)
			except KeyError:
				pass
	itervalues = values

	def items(self):
		for k in iter(self):
			try:
				yield k, cachecls.__getitem__(self, k)
			except KeyError:
				pass
	iteritems = items

	def clear(self):
		self._retrieving.clear()
		super().clear()
		if self._path and os.path.exists(self._path) and os.path.isdir(self._path) and os.listdir(self._path):
			self.close()
			try:
				shutil.rmtree(self._path)
				os.mkdir(self._path)
			except PermissionError:
				pass
			self.base_init(force=True)


class AutoDatabase(cachecls, collections.abc.MutableMapping):
	__slots__ = ("_path", "_shardcount", "_retrieving", "_kwargs")

	def __init__(self, directory=None, shards=6, **kwargs):
		self._path = directory or None
		self._shardcount = shards
		self._retrieving = {}
		self._kwargs = kwargs
		self.base_init()

	def base_init(self):
		super().__init__(self._path, shards=self._shardcount, **self._kwargs)
		print("Loaded Database:", self._path, len(self), self._kwargs)

	def __getattr__(self, k):
		try:
			return object.__getattribute__(self, k)
		except AttributeError:
			return super().__getattribute__(k)

	version = "1.0.0"
	def validate_or_clear(self):
		if self.get("__version__") != self.version:
			if len(self):
				print(tuple(self.keys()))
				raise NotImplementedError("Database must be updated.")
		self.set("__version__", self.version)

	def __iter__(self):
		return (i for i in super().__iter__() if i != "__version__")

	def __len__(self):
		return super().__len__()

	def __contains__(self, k):
		try:
			return super().__contains__(k)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			self.base_init()
			return super().__contains__(k)

	def __getitem__(self, k):
		if (fut := self._retrieving.get(k)):
			return fut.result()
		try:
			return super().__getitem__(k)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			self.base_init()
			return super().__getitem__(k)

	def __setitem__(self, k, v, read=False):
		if isinstance(v, memoryview):
			v = bytes(v)
		super().set(k, v, read=read)

	def update(self, other):
		t = utc()
		if hasattr(other, "items"):
			other = other.items()
		for k, v in other:
			if isinstance(v, memoryview):
				v = bytes(v)
			super().set(k, v)
		return self

	def pop(self, k, v=Dummy):
		self._retrieving.pop(k, None)
		v = super().pop(k, default=(v,))
		if v is Dummy:
			raise KeyError(k)
		return v

	def setdefault(self, k, v):
		if (fut := self._retrieving.get(k)):
			return fut.result()
		try:
			return super().__getitem__(k)
		except KeyError:
			self[k] = v
		return v

	def _retrieve(self, k, func, *args, read=False, **kwargs):
		try:
			self._retrieving[k] = fut = concurrent.futures.Future()
			v = func(*args, **kwargs)
			self.__setitem__(k, v, read=read)
		except Exception as ex:
			fut.set_exception(ex)
			raise
		else:
			fut.set_result(v)
		finally:
			self._retrieving.pop(k, None)
		if read:
			v.seek(0)
		return v
	def retrieve(self, k, func, *args, _read=False, **kwargs):
		if (fut := self._retrieving.get(k)):
			resp = fut.result()
			if _read and hasattr(resp, "name") and os.path.exists(resp.name):
				return open(resp.name, "rb")
			return resp
		try:
			v = super().get(k, read=_read)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			super().__init__(self._path, shards=len(self._shards), **self._kwargs)
			self.base_init()
			v = super().get(k, read=_read)
		if v is Dummy:
			return self._retrieve(k, func, *args, read=_read, **kwargs)
		if _read:
			v.seek(0)
		return v

	async def _aretrieve(self, k, func, *args, read=False, **kwargs):
		try:
			self._retrieving[k] = fut = concurrent.futures.Future()
			v = await asubmit(func, *args, **kwargs)
			await asubmit(self.__setitem__, k, v, read=read)
		except Exception as ex:
			fut.set_exception(ex)
			raise
		else:
			fut.set_result(v)
		finally:
			self._retrieving.pop(k, None)
		if read:
			v.seek(0)
		return v
	async def aretrieve(self, k, func, *args, _read=False, **kwargs):
		if (fut := self._retrieving.get(k)):
			resp = await wrap_future(fut)
			if _read and hasattr(resp, "name") and os.path.exists(resp.name):
				return open(resp.name, "rb")
			return resp
		try:
			v = super().get(k, read=_read)
		except (diskcache.core.Timeout, sqlite3.OperationalError):
			super().__init__(self._path, shards=len(self._shards), **self._kwargs)
			self.base_init()
			v = super().get(k, read=_read)
		if v is Dummy:
			return await self._aretrieve(k, func, *args, read=_read, **kwargs)
		if _read:
			v.seek(0)
		return v

	def keys(self):
		return iter(self)
	iterkeys = keys

	def values(self):
		return (cachecls.__getitem__(self, k) for k in iter(self))
	itervalues = values

	def items(self):
		return ((k, cachecls.__getitem__(self, k)) for k in iter(self))
	iteritems = items

	def clear(self):
		self._retrieving.clear()
		super().clear()
		if self._path and os.path.exists(self._path) and os.path.isdir(self._path) and os.listdir(self._path):
			self.close()
			try:
				shutil.rmtree(self._path)
				os.mkdir(self._path)
			except PermissionError:
				pass
			self.base_init()


DISCORD_EPOCH = 1420070400000 # 1 Jan 2015
MIZA_EPOCH = 1577797200000 # 1 Jan 2020

def id2ts(id):
	i = (id >> 22) + (id & 0xFFF)
	try:
		j = i + (id & 0xFFF) / 0x1000
	except OverflowError:
		return (i + DISCORD_EPOCH) // 1000
	return (j + DISCORD_EPOCH) / 1000
def id2td(id):
	i = (id >> 22) + (id & 0xFFF)
	try:
		j = i + (id & 0xFFF) / 0x1000
	except OverflowError:
		return i // 1000
	return j / 1000

def snowflake_time(id):
	i = getattr(id, "id", None)
	if i is None:
		i = id
	if isinstance(i, int):
		return datetime.datetime.utcfromtimestamp(id2ts(i))
	return i
def snowflake_time_2(id):
	return datetime.datetime.fromtimestamp(id2ts(id))
def snowflake_time_3(id):
	return datetime.datetime.fromtimestamp(id2ts(id), tz=datetime.timezone.utc)

def time_snowflake(dt, high=None):
	if getattr(dt, "id", None) is not None:
		return dt.id
	if not isinstance(dt, (int, float)):
		discord_millis = int(dt.timestamp() * 1000 - DISCORD_EPOCH)
		return (discord_millis << 22) + (2**22 - 1 if high else 0)
	return dt

def ip2int(ip):
	return int.from_bytes(b"\x00" + bytes(int(i) for i in ip.split(".")), "big")


def choice(*args):
	"Custom random.choice implementation that also accepts non-ordered sequences."
	if not args:
		return
	it = args if len(args) > 1 or not issubclass(type(args[0]), collections.abc.Sized) else args[0]
	if not issubclass(type(it), collections.abc.Sequence):
		if not issubclass(type(it), collections.abc.Sized):
			it = tuple(it)
		else:
			size = len(it)
			it = iter(it)
			i = random.randint(0, size - 1)
			for _ in loop(i):
				next(it)
			return next(it)
	return random.choice(it)

def shuffle(it) -> collections.abc.Iterable:
	"Shuffles an iterable, in-place if possible, returning it."
	if type(it) is list:
		random.shuffle(it)
		return it
	elif type(it) is tuple:
		it = list(it)
		random.shuffle(it)
		return it
	elif type(it) is dict:
		ir = shuffle(list(it))
		new = {}
		for i in ir:
			new[i] = it[i]
		it.clear()
		it.update(new)
		return it
	elif type(it) is deque:
		it = list(it)
		random.shuffle(it)
		return deque(it)
	elif isinstance(it, alist):
		return it.shuffle()
	else:
		try:
			it = list(it)
			random.shuffle(it)
			return it
		except TypeError:
			raise TypeError(f"Shuffling {type(it)} is not supported.")

# Reverses an iterable, in-place if possible, returning it.
def reverse(it) -> collections.abc.Iterable:
	if type(it) is list:
		return list(reversed(it))
	elif type(it) is tuple:
		return list(reversed(it))
	elif type(it) is dict:
		ir = tuple(reversed(it))
		new = {}
		for i in ir:
			new[i] = it[i]
		it.clear()
		it.update(new)
		return it
	elif type(it) is deque:
		return deque(reversed(it))
	elif isinstance(it, alist):
		temp = it.reverse()
		it.data = temp.data
		it.offs = temp.offs
		it.hash = None
		del temp
		return it
	else:
		try:
			return list(reversed(it))
		except TypeError:
			raise TypeError(f"Reversing {type(it)} is not supported.")

# Sorts an iterable with an optional key, in-place if possible, returning it.
def sort(it, key=None, reverse=False) -> collections.abc.Iterable:
	if type(it) is list:
		it.sort(key=key, reverse=reverse)
		return it
	elif type(it) is tuple:
		it = sorted(it, key=key, reverse=reverse)
		return it
	elif issubclass(type(it), collections.abc.Mapping):
		keys = sorted(it, key=it.get if key is None else lambda x: key(it.get(x)))
		if reverse:
			keys = reversed(keys)
		items = tuple((i, it[i]) for i in keys)
		it.clear()
		it.__init__(items)
		return it
	elif type(it) is deque:
		it = sorted(it, key=key, reverse=reverse)
		return deque(it)
	elif isinstance(it, alist):
		it.fill(sorted(it, key=key, reverse=reverse))
		return it
	else:
		try:
			it = list(it)
			it.sort(key=key, reverse=reverse)
			return it
		except TypeError:
			raise TypeError(f"Sorting {type(it)} is not supported.")


class Flush(io.IOBase):

	__slots__ = ("obj", "_flush")

	def __init__(self, obj: io.IOBase, flush: callable = None):
		if not hasattr(obj, "flush") and flush is None:
			raise ValueError("The wrapped object must have a flush method or a flush method must be provided.")
		self.obj = obj
		self._flush = flush or self.obj.flush

	def write(self, data: bytes) -> int:
		try:
			return self.obj.write(data)
		finally:
			self.flush()

	def read(self, size: int = -1) -> bytes:
		return self.obj.read(size)

	def seek(self, index: int = 0, whence: int = io.SEEK_SET) -> int:
		return self.obj.seek(index, whence)

	def flush(self) -> None:
		return self._flush()

	def close(self) -> None:
		return

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		return


# Repeatedly retries a synchronous operation, with optional break exceptions.
def retry(func, *args, attempts=5, delay=1, exc=(), **kwargs):
	for i in range(attempts):
		t = utc()
		try:
			return func(*args, **kwargs)
		except BaseException as ex:
			if i >= attempts - 1 or ex in exc:
				raise
		remaining = delay - utc() + t
		if remaining > 0:
			time.sleep(remaining)

# Repeatedly retries a asynchronous operation, with optional break exceptions.
async def aretry(func, *args, attempts=5, delay=1, exc=(), **kwargs):
	for i in range(attempts):
		t = utc()
		try:
			return await func(*args, **kwargs)
		except BaseException as ex:
			if i >= attempts - 1 or ex in exc:
				raise
		remaining = delay - utc() + t
		if remaining > 0:
			await asyncio.sleep(remaining)
		delay += delay / 2


# Evaluates an an expression, returning it if it is an exception.
def evalex(exc, glob=None, loc=None):
	if not isinstance(exc, str | bytes):
		exc = bytes(exc)
	try:
		ex = eval(exc, glob, loc)
	except (SyntaxError, NameError):
		exc = as_str(exc)
		s = exc[exc.index("(") + 1:exc.rindex(")")]
		with suppress(TypeError, SyntaxError, ValueError):
			s = ast.literal_eval(s)
		s = lim_str(s, 4096)
		ex = RuntimeError(s)
	if isinstance(ex, tuple):
		print(ex[1])
		return ex[0]
	return ex

# Evaluates an an expression, raising it if it is an exception.
def evalEX(exc):
	ex = evalex(exc)
	if is_exception(ex):
		raise ex
	return ex

# Evaluates an an expression, raising it regardless of if it is an exception.
def evalFX(exc):
	ex = evalEX(exc)
	raise RuntimeError(ex)

# Much more powerful exec function that can evaluate async expressions, is thread-safe, and always returns the value on the last line even without a return statement specified.
__eval__ = eval
__exec__ = exec
def aexec(s, glob=None, filename="<aexec>"):
	glob = glob or globals()
	s = as_str(s)
	if s.startswith("!"):
		proc = subprocess.run(s[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output = (as_str(proc.stdout) + "\n" + as_str(proc.stderr)).strip()
		if output:
			glob["_"] = output
		return output or None
	try:
		parsed = ast.parse(s)
	except Exception as ex:
		raise PropagateTraceback.cast(ex, format_exc())
	if not parsed.body:
		return
	ret = parsed.body[-1]
	while isinstance(ret, (ast.Return, ast.Expr, ast.Await)):
		ret = ret.value
	try:
		expr = ast.Expression(body=ret)
		end = compile(expr, filename=filename, mode="eval")
	except (SyntaxError, TypeError):
		end = None
	except Exception as ex:
		raise PropagateTraceback.cast(ex, format_exc())
	else:
		parsed.body.pop(-1)
	if parsed.body:
		try:
			code = compile(parsed, filename=filename, mode="exec")
		except SyntaxError as ex:
			if ex.args and ("outside async function" in ex.args[0] or "outside function" in ex.args[0]):
				code = None
			else:
				raise PropagateTraceback.cast(ex, format_exc())
		except Exception as ex:
			raise PropagateTraceback.cast(ex, format_exc())
		else:
			__exec__(code, glob)
		if code is None:
			_ = glob.get("_")
			outside = ast.parse("""async def _():
	try:
		pass
	except Exception as ex:
		raise PropagateTraceback.cast(ex, format_exc())""")
			outside.body[0].body[0].body.clear()
			outside.body[0].body[0].body.extend(parsed.body)
			finalise = ast.Expr(value=ast.parse("globals().update(locals())", mode="eval").body, lineno=0, col_offset=0)
			outside.body[0].body[0].finalbody.append(finalise)
			try:
				code = compile(outside, filename=filename, mode="exec")
				__exec__(code, glob)
			except Exception as ex:
				raise PropagateTraceback.cast(ex, format_exc())
			else:
				await_fut(glob["_"]())
			glob["_"] = _
	if end:
		try:
			fut = __eval__(end, glob)
			if awaitable(fut):
				fut = await_fut(asubmit(fut))
		except Exception as ex:
			raise PropagateTraceback.cast(ex, format_exc())
		return fut
aeval = aexec


def wrap_iter(resp):
	it = resp.iter_content(24)
	try:
		while True:
			yield next(it)
	except StopIteration as ex:
		raise CCE(*ex.args)

def strnum(num):
	return str(round(num, 6))
def time_disp(s, rounded=True):
	"Returns a representation of a time interval using days:hours:minutes:seconds."
	if not isfinite(s):
		return str(s)
	if rounded:
		s = round(s)
	output = strnum(s % 60)
	if len(output) < 2:
		output = "0" + output
	if s >= 60:
		temp = strnum((s // 60) % 60)
		if len(temp) < 2 and s >= 3600:
			temp = "0" + temp
		output = temp + ":" + output
		if s >= 3600:
			temp = strnum((s // 3600) % 24)
			if len(temp) < 2 and s >= 86400:
				temp = "0" + temp
			output = temp + ":" + output
			if s >= 86400:
				output = strnum(s // 86400) + ":" + output
	else:
		output = "0:" + output
	return output
def time_parse(ts):
	"Converts a time interval represented using days:hours:minutes:seconds, to a value in seconds."
	if ts == "N/A":
		return inf
	data = ts.split(":")
	if len(data) >= 5: 
		raise TypeError("Too many time arguments.")
	mults = (1, 60, 3600, 86400)
	return round_min(sum(float(count) * mult for count, mult in zip(data, reversed(mults[:len(data)]))))

def _normalize(s: str) -> str:
	s = re.sub(r"\s+", " ", s).strip()
	return s

def _tokens(s: str):
	return re.findall(r"\b\w+\b", s.lower())

def _common_prefix_len(a: str, b: str) -> int:
	n = min(len(a), len(b))
	i = 0
	while i < n and a[i] == b[i]:
		i += 1
	return i

def _weighted_edit_distance(a: str, b: str) -> float:
    # Costs
    ins_cost = 1.0
    del_cost = 1.0
    sub_cost_alnum = 1.2
    sub_cost_other = 1.4
    case_cost = 0.2  # cheap substitution when only case differs

    n, m = len(a), len(b)
    if n == 0:
        return m * ins_cost
    if m == 0:
        return n * del_cost

    # dp rows
    prev = [j * ins_cost for j in range(m + 1)]
    curr = [0.0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i * del_cost
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]

            if ca == cb:
                sub_cost = 0.0
            elif ca.lower() == cb.lower():
                sub_cost = case_cost
            else:
                if ca.isalnum() and cb.isalnum():
                    sub_cost = sub_cost_alnum
                else:
                    sub_cost = sub_cost_other

            curr[j] = min(
                prev[j] + del_cost,      # deletion
                curr[j - 1] + ins_cost,  # insertion
                prev[j - 1] + sub_cost   # substitution
            )

        prev, curr = curr, prev

    return prev[m]

# Thank you to gpt-5.2-codex for improving on the old algorithm.
@functools.lru_cache(maxsize=4096)
def string_similarity(a: str, b: str) -> float:
	a_norm = _normalize(a)
	b_norm = _normalize(b)

	# --- Component 1: weighted edit similarity ---
	dist = _weighted_edit_distance(a_norm, b_norm)
	max_len = max(len(a_norm), len(b_norm), 1)
	max_sub_cost = 1.4
	avg_len = max((len(a_norm) + len(b_norm)) / 2, 1)
	edit_sim = 1.0 - (dist / (avg_len * max_sub_cost))

	# --- Component 2: token overlap (Jaccard) ---
	ta = set(_tokens(a_norm))
	tb = set(_tokens(b_norm))
	if ta or tb:
		token_sim = len(ta & tb) / len(ta | tb)
	else:
		token_sim = 1.0

	# --- Component 3: prefix similarity ---
	prefix_len = _common_prefix_len(full_prune(a_norm), full_prune(b_norm))
	prefix_sim = prefix_len / max_len

	# Weighted blend
	w_edit, w_token, w_prefix = 0.3, 0.3, 0.4
	score = (w_edit * edit_sim) + (w_token * token_sim) + (w_prefix * prefix_sim)

	# Clamp to $[0, 1]$
	return max(0.0, min(1.0, score))

# A string lookup operation with an iterable, multiple attempts, and sorts by priority.
def str_lookup(objs, query, key=lambda obj: obj, fuzzy=0, compare=string_similarity):
	objs = astype(objs, (tuple, list))
	query = query.strip()
	keys = [key(obj).strip() for obj in objs]
	try:
		return objs[keys.index(query)]
	except ValueError:
		pass
	closest = (-inf, None, "")
	for s, obj in zip(keys, objs):
		match = compare(query, s)
		if match > closest[0]:
			closest = (match, obj, s)
	if closest[0] >= 1 - fuzzy:
		return closest[1]
	err = f'No results for "{query}".'
	if closest[0] > -inf:
		err += f' Did you mean: "{closest[2]}"?'
	raise LookupError(err)

def longest_sublist(lst, predicate):
	"Returns the longest contiguous sublist of a list that satisfies a predicate. For example, if the predicate is `lambda a: all(a[i] < a[i + 1] for i in range(len(a) - 1))`, the function will return the longest sorted sublist. Note that for our implementation, we may sometimes need to perform backtracking with the sliding window, as a contiguous sublist may not fulfil the predicate if cut off; for example, if our predicate is instead a function which parses a string and returns a time delta, it may consider the strings `2 hours` and `3 minutes` as valid, but not `2 hours 3`, so we would need to backtrack at `3 minutes` to the previous predicate-satisfying sublist to be able to correctly identify the string `2 hours 3 minutes`."
	if not lst:
		return [], -1
	max_len = 0
	max_start = 0
	curr_len = 0
	curr_start = 0
	curr_buf = []
	prev_bufs = []
	for i in range(len(lst)):
		curr_buf.append(lst[i])
		if predicate(curr_buf):
			# Since the predicate is satisfied, we backtrack to see if we can extend the current buffer with any of the previous buffers
			for prev_buf in reversed(prev_bufs):
				if predicate(prev_buf + curr_buf):
					curr_buf = prev_buf + curr_buf
					curr_start -= len(prev_buf)
				else:
					break
			prev_bufs.clear()
			curr_len = len(curr_buf)
			if curr_len > max_len:
				max_len = curr_len
				max_start = curr_start
		else:
			prev_bufs.append(curr_buf[:-1])
			curr_buf = [lst[i]]
			curr_start = i
	if len(curr_buf) > max_len and predicate(curr_buf):
		return curr_buf, curr_start
	while prev_bufs and predicate(prev_bufs[-1] + curr_buf):
		prev_buf = prev_bufs.pop(-1)
		curr_buf = prev_buf + curr_buf
		curr_start -= len(prev_buf)
	final = lst[max_start:max_start + max_len]
	if predicate(final):
		return final, max_start
	return [], -1

# Thank you to deepseek-r1 for once again improving on another algorithm.
def predict_continuation(posts, min_score=0.5):
	"""
	Predict the next post in a sequence of posts based on:
	- The most common and closest matching non-numeric string.
	- The continuation of the numeric sequence.
	
	Args:
		posts (list): A list of strings representing the sequence of posts.
		min_score (float): The minimum similarity score for a string to be considered a match.
	
	Returns:
		str: The predicted next post, or None if no valid continuation is found.
	"""
	if not posts:
		return None
	
	# Regular expression to match numeric substrings as individual words
	numeric_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
	
	# Split each post into tokens (numeric and non-numeric parts)
	split_posts = []
	for post in posts:
		tokens = []
		last_end = 0
		for match in numeric_pattern.finditer(post):
			# Add non-numeric part before the numeric match
			if match.start() > last_end:
				tokens.append(post[last_end:match.start()])
			# Add numeric part
			tokens.append(fractions.Fraction(match.group()))
			last_end = match.end()
		# Add non-numeric part after the last numeric match
		if last_end < len(post):
			tokens.append(post[last_end:])
		split_posts.append(tokens)
	
	# Extract non-numeric and numeric parts separately
	non_numeric_parts = [' '.join(str(token) for token in tokens if not isinstance(token, number)) for tokens in split_posts]
	numeric_parts = [[token for token in tokens if isinstance(token, number)] for tokens in split_posts]
	
	# Find the most common and closest matching non-numeric string
	counts = defaultdict(int)
	for part in non_numeric_parts:
		counts[part] += 1
	
	# Ensure the most common string appears at least twice
	most_common = max(counts.keys(), key=lambda x: counts[x])
	if counts[most_common] < 2:
		return None  # Require at least two occurrences for a valid prediction
	
	# Check if the most common string is sufficiently similar to all non-numeric parts
	for part in non_numeric_parts:
		if string_similarity(most_common, part) < min_score:
			return None  # A non-numeric part doesn't match sufficiently
	
	# Predict the next numeric values for each sequence
	next_numerics = []
	for i in range(len(numeric_parts[0])):
		sequence = [parts[i] for parts in numeric_parts if i < len(parts)]
		next_numeric = predict_next(sequence)
		if next_numeric is None:
			return None  # No valid numeric continuation for this sequence
		next_numerics.append(next_numeric)
	
	# Find the structure of the most common non-numeric string
	most_common_tokens = None
	for tokens in split_posts:
		non_numeric_part = ' '.join(str(token) for token in tokens if not isinstance(token, number))
		if non_numeric_part == most_common:
			most_common_tokens = tokens
			break
	
	# Reconstruct the predicted post by replacing numeric tokens with predicted values
	predicted_tokens = []
	numeric_index = 0
	for token in most_common_tokens:
		if isinstance(token, number):
			if numeric_index < len(next_numerics):
				predicted_tokens.append(next_numerics[numeric_index])
				numeric_index += 1
			else:
				return None  # Not enough numeric values to replace; mismatched sequence lengths
		else:
			predicted_tokens.append(token)

	def show_token(token):
		if isinstance(token, number):
			scaled = float(token)
			if not scaled:
				return "0"  # Avoid log(0) in the next line
			return display_to_precision(token, ceil(abs(log10(abs(scaled)))) + 6)  # max 6 decimal places or significant figures, whichever is higher
		return token
	return ''.join(map(show_token, predicted_tokens)).strip()


RAINBOW = [
	"\x1b[38;5;196m",
    "\x1b[38;5;208m",
    "\x1b[38;5;226m",
    "\x1b[38;5;118m",
    "\x1b[38;5;46m",
    "\x1b[38;5;48m",
    "\x1b[38;5;51m",
    "\x1b[38;5;33m",
    "\x1b[38;5;21m",
    "\x1b[38;5;93m",
    "\x1b[38;5;201m",
    "\x1b[38;5;198m",
]
WHITE = "\u001b[37m"
RESET = "\u001b[0m"

if os.name == "nt":
	os.system("color")


MEMS = {}

@tracebacksuppressor
def share_bytes(sender, b):
	"""Shares bytes with a sender, either directly or through shared memory.

	This function attempts to send bytes to a sender. If the bytes are small enough
	(less than 65535 bytes), they are sent directly. For larger data, the function
	uses shared memory to transfer the bytes.

	Args:
		sender: A callable that accepts bytes as an argument and sends them.
		b: The bytes to be shared/sent.

	Returns:
		The result of the sender function call.

	Raises:
		Any exceptions that may occur during the sending process.

	Note:
		- For direct sending (small data), prepends b'\x01' to the data
		- For shared memory (large data), prepends b'\x02' followed by size and memory name
		- Shared memory objects are stored in a MEMS dictionary using the memory name as key
	"""
	if len(b) < 65536:
		return sender(b"\x01" + b)
	mem = multiprocessing.shared_memory.SharedMemory(create=True, size=len(b))
	# print(mem)
	MEMS[mem.name] = mem
	mem.buf[:len(b)] = b
	try:
		return sender(b"\x02" + len(b).to_bytes(8, "little") + mem.name.encode("utf-8"))
	except:
		mem.close()
		raise

@tracebacksuppressor(ConnectionResetError, BrokenPipeError)
def receive_bytes(receiver, unlink):
	"""Receives and processes bytes from a receiver function, handling both regular and shared memory data.

	This function processes incoming bytes and supports two modes of data transfer:
	- Mode 1: Direct memory bytes transfer
	- Mode 2: Shared memory transfer using multiprocessing.shared_memory

	Args:
		receiver (callable): A function that returns bytes data when called
		unlink (callable): A function to handle cleanup of shared memory resources

	Returns:
		bytes: The processed bytes data from either direct transfer or shared memory.
		If the receiver returns empty data, returns the empty data directly.

	Note:
		For Mode 2 (shared memory):
		- First byte indicates mode
		- Next 8 bytes represent size in little-endian
		- Remaining bytes contain the shared memory name
		The shared memory is automatically unlinked after reading.
	"""
	b = receiver()
	if not b:
		return b
	if b[0] == 1:
		return MemoryBytes(b)[1:]
	mode, size, name = b[:1], b[1:9], b[9:]
	if mode[0] != 2:
		return b
	size = int.from_bytes(size, "little")
	name = as_str(name)
	mem = multiprocessing.shared_memory.SharedMemory(create=False, name=name)
	print(mem)
	try:
		return mem.buf[:size].tobytes()
	finally:
		mem.unlink()
		esubmit(unlink, name)


class PipeableIterator(collections.abc.Iterator):
	"""A threaded iterator class that allows for dynamic item appending and termination.
	This iterator can be used as a pipe, allowing items to be appended while iterating.
	The iterator will wait for new items when the current items are exhausted, unless terminated.
	Attributes:
		items (list): The list of items to iterate over
		i (int): The current iteration index
		stop_index (float): The index at which iteration should stop
		condition (threading.Condition): Thread synchronization condition
		buffer (bytes): Buffer for read operations
	Methods:
		append(item): Adds an item to the iterator
		terminate(): Stops the iterator from accepting new items
		close(): Alias for terminate()
		read(n=None): Reads n bytes from the iterator, or all bytes if n is None
	"""

	def __init__(self, items=None):
		self.items = items or []
		self.i = 0
		self.stop_index = inf
		self.condition = threading.Condition()
		self.buffer = None

	def __next__(self):
		while self.i >= len(self.items) and self.condition is not None:
			with self.condition:
				self.condition.wait()
		if self.i >= self.stop_index:
			raise StopIteration
		try:
			return self.items[self.i]
		finally:
			self.i += 1

	def append(self, item):
		self.items.append(item)
		with self.condition:
			self.condition.notify()

	def terminate(self):
		self.stop_index = len(self.items)
		if self.condition is None:
			return self
		condition, self.condition = self.condition, None
		with condition:
			condition.notify_all()
		return self
	close = terminate

	def read(self, n=None):
		if n is None:
			return (self.buffer or b"") + b"".join(self)
		b = self.buffer or b""
		while len(b) < n:
			try:
				b += next(self)
			except StopIteration:
				break
		b, self.buffer = b[:n], b[n:]
		return b

class EvalPipe:
	"""
	EvalPipe: A bidirectional, asynchronous evaluation and messaging channel between a controller
	(process or thread) and a worker process (or another endpoint). It supports:
	- Remote code execution (sync and async) with structured request/response framing.
	- Iterator/async-iterator streaming of results.
	- Transparent in-band error propagation with reconstructed exceptions.
	- Optional automatic spawning/listening of worker processes via sockets or stdio.
	- Multiplexed concurrent requests managed via numeric ids (positive = background thread, negative = priority/main thread).
	- Local fast-path zero-copy transfer using shared memory where available (127.0.0.1 optimization).
	- Caching of completed results (opt-in per call).
	- Graceful shutdown and forced termination semantics.
	Communication framing protocol (byte-oriented):
		Requests (controller -> worker):
			~>{id}:{utf8_code}
				id > 0  => execute in background via thread pool / executor.
				id < 0  => execute synchronously (priority) in the main thread of the worker.
		Responses (worker -> controller):
			<~{id}:@:{json_value}    Single successful result.
			<~{id}:#:{json_value}    Streamed/iterator item (zero or more).
			<~{id}:$                 Iterator completion sentinel.
			<~{id}:!:{serialized_ex} Exception: tuple-like encoding of (repr(e), RuntimeError(traceback)).
		Logging / passthrough lines may be prefixed with ASCII NUL (0x00) and are surfaced as debug/info.
	Key internal structures:
		self.responses : dict[int, concurrent.futures.Future]
				Futures awaiting a single terminal value or first iterator element.
		self.iterators : dict[int, PipeableIterator]
				Active streaming iterators being incrementally filled.
		self.cache     : dict[str, tuple[Any, float]]
				Optional time-bounded caching of run() results (code -> (value, timestamp)).
	Construction variants (classmethods):
		connect(...)     : Connect to an already-listening EvalPipe server (auto-spawn if allowed).
		listen(...)      : Create a passive listener; accepts a single client at a time (re-arms after disconnect).
		from_proc(proc)  : Wrap an existing subprocess with stdin/stdout pipes.
		from_stdin(...)  : Treat current process as a worker reading from its stdin.
	Public high-level methods:
		start(background=True):
				Begin the communication loop (creates a background thread unless background=False).
		submit(code, priority=False) -> Future:
				Enqueue code for execution; returns a Future whose result is decoded from JSON (or raises remote exception).
		run(code, timeout=30, cache=None, priority=False):
				Synchronous helper around submit(). Optional cache (seconds) to reuse prior result.
		asubmit(code, priority=False) -> Awaitable:
				Async wrapper returning an awaitable for the Future.
		print(*args, sep=' ', end='\\n'):
				Mirror of print that also forwards output through the pipe/log channel.
		kill() / terminate():
				Forcefully abort outstanding work, reject Futures, close connections.
		join():
				Block until underlying worker has exited (or listener has been closed).
	Execution semantics:
		- Code strings are evaluated via aexec(...) against the mutable namespace self.glob.
		- The constructor injects JSON-like aliases (true/false/null) for convenience.
		- Return values are JSON-serialized (via maybe_json / eval_json helpers).
		- Iterators (sync or async) are flattened into a sequence of <~# events followed by <~$.
	Thread-safety:
		- Submission index allocation guarded by self.rlock.
		- Output send path guarded by self.wlock.
		- Iterator accumulation guarded during mutation.
		- Futures are resolved exactly once; late or unknown response ids are ignored defensively.
	Priority execution:
		- Negative ids are executed inline via compute() without deferral, ensuring libraries that
			require main-thread execution (UI/toolkits) retain correctness.
	Error handling:
		- Remote exceptions are reconstructed locally with evalex(); original traceback text is embedded.
		- compute() distinguishes between normal, streaming, and exceptional paths and emits the correct marker.
		- kill() synthesizes a RuntimeError for all unresolved Futures/iterators.
	Zero-copy optimization (loopback only):
		- When address == '127.0.0.1', large payloads may be transferred through shared memory segments
			referenced indirectly (share_bytes / receive_bytes); cleanup is scheduled after use.
	Lifecycle:
		1. Instantiate via one of the factory classmethods or directly (advanced).
		2. start() the communication thread (unless start=True was supplied to constructor/factory).
		3. submit()/run()/asubmit() code strings for evaluation.
		4. Iterate over streaming results where applicable.
		5. kill() or let p_alive() become False to exit; join() to ensure cleanup.
	Parameters (constructor):
		p_alive   : Callable[[], bool]
				Liveness probe for the remote endpoint/process.
		p_in      : Callable[[bytes], Any] | None
				Low-level writer to the transport (bytes in).
		p_out     : Callable[[], bytes] | None
				Low-level reader from the transport (bytes out).
		p_kill    : Callable[[], Any] | None
				Termination hook (process kill / close).
		p_join    : Callable[[], Any] | None
				Blocking wait hook until remote fully exits.
		writable  : bool
				Initial readiness state for submission; listener mode starts False until a client connects.
		start     : bool
				Auto-start communication thread upon creation.
		glob      : dict
				Execution namespace (mutated in-place).
		id        : int
				Cosmetic identifier used in debug coloring and ordering.
		server    : multiprocessing.connection.Listener | None
				Listener object when acting as a server.
		address   : str | None
				Bound or connected address (used for optimization decisions).
		port      : int | None
				Bound or connected port (used as part of identity/logging).
	Important invariants / notes:
		- ensure_writable() blocks until a client connects (listener mode) or remote is ready.
		- Negative ids (priority) are never offloaded to thread pools.
		- Responses for unknown ids are safely ignored to tolerate race conditions on teardown.
		- All externally surfaced values are deserialized; remote-side JSON conversion must succeed.
	Example (controller side pseudo-usage):
		pipe = EvalPipe.connect(["python", "worker.py"], port=5001)
		result = pipe.run("1 + 2")                # => 3
		fut = pipe.submit("sum(range(1000000))")  # Future; do other work
		stream = pipe.run("(x for x in range(3))")# PipeableIterator -> iterate for 0,1,2
				val = fut.result(timeout=10)
		except Exception as e:
				...
		pipe.kill()
	Example (worker side pseudo-usage):
		if __name__ == '__main__':
				ep = EvalPipe.listen(port=5001, start=True)
				ep.join()
	Caveats:
		- Security: This is an arbitrary code execution channel; do NOT expose to untrusted clients.
		- Serialization: Non-JSON-serializable objects rely on helper maybe_json; custom types may degrade.
		- Backpressure: Streaming relies on in-memory buffering (iterators[i].append); large streams should
			be consumed promptly.

	- This was written by Coco btw, trust, I'm so smart I'm a whole 10 on an IQ test
	"""
	MEMS = globals()["MEMS"]

	def __init__(self, p_alive, p_in, p_out, p_kill=None, p_join=None, writable=True, start=True, glob=globals(), id=0, server=None, address=None, port=None, is_main=False):
		self.rlock = threading.Lock()
		self.wlock = threading.Lock()
		self.responses = {}
		self.iterators = {}
		self.p_alive = p_alive
		self.p_in = p_in
		self.p_out = p_out
		self.p_kill = p_kill
		self.p_join = p_join
		self.writable = writable
		self.cache = {}
		glob.update(dict(true=True, false=False, null=None))
		self.glob = glob
		self.id = id
		self.thread = None
		self.server = server
		self.address = address
		self.port = port
		self.is_main = is_main
		if start:
			self.start()

	key = b64(enc_key)[:32]
	@classmethod
	def connect(cls, args, port, address="127.0.0.1", independent=True, glob=globals(), timeout=60):
		addr = (address, port)
		print(f"{DynamicDT.now()}: EvalPipe connecting to", addr)
		try:
			conn = multiprocessing.connection.Client(addr, authkey=cls.key)
		except ConnectionRefusedError:
			if independent:
				script = None
				if os.name == "nt":
					argstr = " ".join(map(json_dumpstr, args)) + "\nexit /b"
					script = temporary_file("bat")
					with open(script, "w") as f:
						f.write(argstr)
					args = f"wt -w 0 -d %cd% cmd /s/c {script}"
				else:
					argstr = " ".join(map(json_dumpstr, args))
					args = ["xterm", "-e", argstr]
			print(args)
			subprocess.Popen(args, shell=isinstance(args, str), stdin=subprocess.DEVNULL, stdout=None, stderr=None)
			t = utc()
			for i in range(timeout):
				if utc() - t > timeout:
					raise TimeoutError(timeout)
				try:
					conn = multiprocessing.connection.Client(addr, authkey=cls.key)
				except ConnectionRefusedError:
					time.sleep(1)
				else:
					break
			else:
				raise
		print(f"{DynamicDT.now()}: Connection established:", conn)
		assert conn.readable and conn.writable, "Connection must be readable and writable."
		pid = conn.recv()
		proc = psutil.Process(pid)
		self = cls(
			proc.is_running,
			(lambda b: share_bytes(conn.send_bytes, b)) if address == "127.0.0.1" else conn.send_bytes,
			None,
			proc.terminate,
			proc.wait,
			glob=glob,
			id=port,
			address=address,
			port=port,
			is_main=True,
		)
		self.p_out = (lambda: receive_bytes(conn.recv_bytes, unlink=lambda name: self.submit(f"EvalPipe.MEMS.pop({repr(name)}).close()"))) if address == "127.0.0.1" else conn.recv_bytes
		self.proc = proc
		return self

	@classmethod
	def listen(cls, port=0, address="127.0.0.1", glob=globals(), start=False):
		addr = (address, port)
		attempts = 4
		for i in range(attempts):
			try:
				server = multiprocessing.connection.Listener(addr, authkey=cls.key)
			except PermissionError:
				if i < attempts - 1:
					time.sleep(i ** 2 + 1)
					continue
				print("Failed address:", address, port)
				raise
			except Exception:
				print("Failed address:", address, port)
				raise
			else:
				break
		print(f"{DynamicDT.now()}: EvalPipe listening on", server.address)
		return cls(
			lambda: True,
			None,
			None,
			None,
			None,
			writable=False,
			glob=glob,
			id=port + 3,
			server=server,
			start=start,
			address=address,
			port=port,
			is_main=False,
		)

	@classmethod
	def from_proc(cls, proc, glob=globals()):
		flushin = Flush(proc.stdin)
		print(f"{DynamicDT.now()}: EvalPipe connecting on", proc.pid)
		return cls(
			proc.is_running,
			lambda b: flushin.write(b.rstrip(b"\n") + b"\n"),
			proc.stdout.readline,
			proc.terminate,
			proc.wait,
			glob=glob,
			id=proc.pid,
			is_main=True,
		)

	@classmethod
	def from_stdin(cls, start=False, glob=globals()):
		flushout = Flush(sys.__stdout__.buffer, flush=sys.__stdout__.flush)
		print(f"{DynamicDT.now()}: EvalPipe listening from", os.getpid())
		return cls(
			lambda: not sys.__stdin__.closed,
			lambda b: flushout.write(b.rstrip(b"\n") + b"\n"),
			sys.__stdin__.buffer.readline,
			sys.exit,
			start=start,
			glob=glob,
			id=os.getpid() + 3,
			is_main=False,
		)

	def start(self, background=True):
		if self.thread:
			return self.thread
		if not background:
			return self.communicating()
		self.thread = tsubmit(self.communicating)
		return self.thread

	def ensure_writable(self, timeout=12):
		assert self.p_alive(), "No running worker process."
		t = utc()
		while not self.writable:
			time.sleep(1)
			if timeout and utc() - t > timeout:
				raise TimeoutError(timeout)
		assert self.p_alive(), "No running worker process."

	async def a_ensure_writable(self, timeout=12):
		assert self.p_alive(), "No running worker process."
		t = utc()
		while not self.writable:
			await asyncio.sleep(1)
			if timeout and utc() - t > timeout:
				raise TimeoutError(timeout)
		assert self.p_alive(), "No running worker process."

	def submit(self, s, priority=False) -> Future:
		self.ensure_writable()
		with self.rlock:
			mr = bool(self.responses) and max(max(self.responses), -min(self.responses))
			mi = bool(self.iterators) and max(max(self.iterators), -min(self.iterators))
			i = max(mr, mi) + 1
			if priority:
				# For high-priority tasks, use negative indices; the worker will assign these to its main thread.
				# Also required for compatibility with libraries that fail if the main thread is not used.
				i = -i
			fut = self.responses[i] = Future()
		if isinstance(s, str):
			s = s.encode("utf-8")
		b = f"~>{i}:".encode("ascii") + s
		esubmit(self.send, b)
		return fut

	def run(self, s, timeout=30, cache=None, priority=False):
		self.ensure_writable(timeout=timeout)
		if cache:
			t = utc()
			try:
				res, t2 = self.cache[s]
			except KeyError:
				pass
			else:
				if t - t2 <= cache:
					return res
			res = self.submit(s, priority=priority).result(timeout=timeout)
			self.cache[s] = (res, t)
			return res
		return self.submit(s, priority=priority).result(timeout=timeout)

	async def asubmit(self, s, priority=False):
		await self.a_ensure_writable()
		return await wrap_future(self.submit(s, priority=priority))

	def print(self, *args, sep=" ", end="\n"):
		b = ("\x00" + sep.join(map(str, args)) + end).encode("utf-8")
		Flush(sys.__stderr__.buffer, flush=sys.__stderr__.flush).write(b)
		if self.p_alive() and self.writable:
			return self.send(b)

	def debug(self, s):
		if self.is_main:
			return print(s, end="")
		sys.__stderr__.write(s)
		sys.__stderr__.flush()

	def send(self, b):
		# self.debug(RAINBOW[self.id % len(RAINBOW)] + str(b) + WHITE + "\n")
		with self.wlock:
			return self.p_in(b)

	def compute(self, i, s):
		self.ensure_writable()
		try:
			resp = aexec(s, self.glob)
		except BaseException as ex:
			s = maybe_json(ex)
			b = f"<~{i}:!:".encode("ascii") + b"(" + s + f",RuntimeError({repr(ex)}))".encode("utf-8")
			self.send(b)
			return
		if isinstance(resp, collections.abc.AsyncIterator):
			resp = reflatten(resp)
		if isinstance(resp, collections.abc.Iterator):
			for x in resp:
				s = maybe_json(x)
				b = f"<~{i}:#:".encode("ascii") + s
				self.send(b)
			b = f"<~{i}:$".encode("ascii")
			self.send(b)
			return
		s = maybe_json(resp)
		b = f"<~{i}:@:".encode("ascii") + s
		self.send(b)

	# Communication; `~>{id}:{msg}` input, `<~{id}:!:{msg}` error output, `<~{id}:@:{msg}` success output, `<~{id}:#:{msg}` iterator output, `<~{id}:$` iterator end
	def communicating(self):
		with tracebacksuppressor:
			while self.p_alive():
				if self.server:
					self.writable = False
					self.p_in = self.p_out = self.p_kill = self.p_join = None
					conn = self.server.accept()
					print("Connection established:", conn)
					conn.send(os.getpid())
					self.p_in = (lambda b: share_bytes(conn.send_bytes, b)) if self.address == "127.0.0.1" else conn.send_bytes
					self.p_out = (lambda: receive_bytes(conn.recv_bytes, unlink=lambda name: self.run(f"EvalPipe.MEMS.pop({repr(name)}).close()"))) if self.address == "127.0.0.1" else conn.recv_bytes
					self.p_kill = conn.close
					self.p_join = lambda: None
					self.writable = True
				while self.p_alive():
					try:
						b = self.p_out()
					except Exception:
						break
					if not b:
						break
					b = MemoryBytes(b)
					if b.startswith(b"~>"):
						b = b[2:]
						sep = b.index(b":")
						i, s = int(b[:sep]), b[sep + 1:].decode("utf-8")
						if i < 0:
							self.compute(i, s)
						else:
							esubmit(self.compute, i, s, priority=1)
						continue
					if b.startswith(b"<~"):
						b = b[2:]
						sep = b.index(b":")
						i, ind, s = int(b[:sep]), chr(b[sep + 1]), b[sep + 3:]
						if i not in self.responses and i not in self.iterators:
							continue
						if ind == "#":
							cur = eval_json(s)
							with self.rlock:
								try:
									self.iterators[i].append(cur)
								except KeyError:
									res = PipeableIterator([cur])
									self.responses.pop(i).set_result(res)
									self.iterators[i] = res
						elif ind == "@":
							res = eval_json(s)
							with self.rlock:
								self.responses.pop(i).set_result(res)
						elif ind == "!":
							ex = evalex(s)
							with self.rlock:
								if i in self.responses:
									self.responses.pop(i).set_exception(ex)
								elif i in self.iterators:
									self.iterators.pop(i).terminate()
								else:
									raise KeyError("Uncaught error", ex)
						elif ind == "$":
							with self.rlock:
								try:
									self.iterators.pop(i).terminate()
								except KeyError:
									self.responses.pop(i).set_result(PipeableIterator([]).terminate())
						else:
							raise NotImplementedError("Unrecognised output", lim_str(s, 262144))
						continue
					b = b.removeprefix(b"\x00")
					if b.startswith(b"INFO:"):
						continue
					b = b.strip()
					if b:
						self.debug(RAINBOW[self.id % len(RAINBOW)] + "INFO:" + WHITE + " " + lim_str(as_str(b), 262144) + "\n")

	def join(self):
		try:
			while not self.p_join:
				time.sleep(1)
			self.p_join()
		except Exception:
			with tracebacksuppressor:
				self.kill()
			raise
		finally:
			self.thread.result()

	def kill(self):
		ex = RuntimeError("Subprocess killed (likely timeout).")
		for resp in tuple(self.responses.values()):
			with suppress(ISE):
				resp.set_exception(ex)
		self.responses.clear()
		for resp in tuple(self.iterators.values()):
			resp.terminate()
		if self.server:
			self.server.close()
		self.iterators.clear()
		if self.p_kill:
			with tracebacksuppressor:
				self.p_kill()
		self.p_alive = lambda: False
		self.thread.result()
	terminate = kill


def prioritise(proc):
	try:
		if os.name == "nt":
			proc.ionice(psutil.IOPRIO_HIGH)
		else:
			proc.ionice(psutil.IOPRIO_CLASS_RT, value=7)
	except psutil.AccessDenied:
		pass
	return proc


def parse_ratelimit_header(headers) -> number:
	try:
		reset = headers.get('X-Ratelimit-Reset')
		if reset:
			delta = float(reset) - utc()
		else:
			reset_after = headers.get('X-Ratelimit-Reset-After')
			delta = float(reset_after)
		if not delta:
			raise
	except (TypeError, ValueError, OverflowError):
		delta = float(headers['retry_after'])
	return max(0.001, delta)


# Manages both sync and async web requests.
class RequestManager(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, collections.abc.Callable):

	ts = 0
	semaphore = Semaphore(512, 256)
	sessions = ()
	session = niquests.Session()
	compat_session = requests.Session()

	@classmethod
	def header(cls, base=(), **fields) -> cdict:
		"Creates a custom HTTP request header with randomised properties that spoof anti-scraping sites."
		head = {
			"User-Agent": USER_AGENT,
			"DNT": "1",
			"X-Forwarded-For": "34." + ".".join(str(random.randint(1, 254)) for _ in range(3)),
			"X-Real-Ip": "34." + ".".join(str(random.randint(1, 254)) for _ in range(3)),
		}
		if base:
			head.update(base)
		if fields:
			head.update(fields)
		return cdict(head)
	headers = header

	async def _init_(self):
		if self.sessions and utc() - self.ts < 86400:
			return self
		if self.sessions:
			for session in self.sessions:
				await session.close()
			await self.nossl.close()
		self.sessions = alist(aiohttp.ClientSession() for i in range(3))
		self.nossl = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
		self.ts = utc()
		return self

	async def aio(self, url, headers=None, files=None, data=None, method="GET", decode=False, json=False, bypass=True, session=None, ssl=None, timeout=24, authorise=False) -> bytes | str | json_like:
		if headers is None:
			headers = {}
		if authorise:
			token = AUTH["discord_token"]
			headers["Authorization"] = f"Bot {token}"
			if data:
				if not isinstance(data, aiohttp.FormData):
					if not isinstance(data, (str, bytes, memoryview)):
						data = json_dumps(data)
			if data and (isinstance(data, (list, dict)) or (data[:1] in '[{"') if isinstance(data, str) else (data[:1] in b'[{"' if isinstance(data, (bytes, memoryview)) else False)):
				headers["Content-Type"] = "application/json"
			session = None
		elif bypass:
			if "user-agent" not in headers and "User-Agent" not in headers:
				headers["User-Agent"] = USER_AGENT
				headers["X-Forwarded-For"] = ".".join(str(random.randint(1, 254)) for _ in loop(4))
			headers["DNT"] = "1"
		method = method.upper()
		verify = True if ssl is not False else False
		if not self.ts:
			await self._init_()
		if isinstance(data, aiohttp.FormData):
			session = self.sessions.next()
		elif not session:
			async with niquests.AsyncSession() as asession:
				try:
					resp = await asession.request(method, url, headers=headers, files=files, data=data, timeout=timeout, verify=verify)
				except niquests.exceptions.SSLError:
					if ssl is not None:
						raise
					resp = await asession.request(method, url, headers=headers, files=files, data=data, timeout=timeout, verify=False)
				if resp.status_code >= 400:
					raise ConnectionError(resp.status_code, (url, as_str(resp.content)))
				if json:
					return resp.json()
				if decode:
					return resp.text
				return resp.content
		async with self.semaphore:
			req = session or (self.sessions.next() if ssl else self.nossl)
			resp = await req.request(method, url, headers=headers, data=data, timeout=timeout)
			status = T(resp).get("status_code") or getattr(resp, "status", 400)
			if status >= 400:
				try:
					data = await resp.read()
				except (TypeError, AttributeError):
					data = resp.text
				if not isinstance(data, bytes):
					data = bytes(data, "utf-8")
				if not data or magic.from_buffer(data).startswith("text/"):
					raise ConnectionError(status, (url, as_str(data)))
			if json and resp.headers.get("Content-Type") == "application/json":
				data = resp.json()
				if awaitable(data):
					return await data
				return data
			try:
				data = await resp.read()
			except (AttributeError, TypeError):
				return resp.content
			if json:
				data = orjson.loads(data)
			if decode:
				return as_str(data)
			return data

	def __call__(self, url, headers=None, files=None, data=None, raw=False, timeout=8, method="get", decode=False, json=False, bypass=True, session=None, ssl=None, authorise=False) -> bytes | str | json_like:
		"Creates and executes a HTTP request, returning the body in bytes, string or JSON format. Raises an exception if status code is below 200 or above 399"
		if headers is None:
			headers = {}
		if authorise:
			token = AUTH["discord_token"]
			headers["Authorization"] = f"Bot {token}"
			if data:
				if not isinstance(data, aiohttp.FormData):
					if not isinstance(data, (str, bytes, memoryview)):
						data = json_dumps(data)
			if data and (isinstance(data, (list, dict)) or (data[:1] in '[{"') if isinstance(data, str) else (data[:1] in b'[{"' if isinstance(data, (bytes, memoryview)) else False)):
				headers["Content-Type"] = "application/json"
			session = self.session
		elif bypass:
			if "user-agent" not in headers and "User-Agent" not in headers:
				headers["User-Agent"] = USER_AGENT
				headers["X-Forwarded-For"] = ".".join(str(random.randint(1, 254)) for _ in loop(4))
			headers["DNT"] = "1"
		method = method.casefold()
		with self.semaphore:
			req = self.session
			verify = True if ssl is not False else False
			try:
				resp = getattr(req, method)(url, headers=headers, files=files, data=data, timeout=timeout, verify=verify)
			except (niquests.exceptions.SSLError, requests.exceptions.SSLError):
				if ssl is not None:
					raise
				resp = getattr(req, method)(url, headers=headers, files=files, data=data, timeout=timeout, verify=False)
			if resp.status_code >= 400:
				if not resp.content or magic.from_buffer(resp.content).startswith("text/"):
					raise ConnectionError(resp.status_code, (url, resp.text))
			if json:
				return resp.json()
			if raw and T(resp).get("raw"):
				data = resp.raw.read()
			else:
				data = resp.content
			if decode:
				return as_str(data)
			return data

	def __enter__(self) -> "RequestManager":
		return self

	def __exit__(self, *args):
		self.session.close()

	async def __aexit__(self, *args):
		self.session.close()

Request = RequestManager()
get_request = Request.__call__

def header_test(url, timeout=12):
	req = urllib.request.Request(url, method="HEAD", headers=Request.header())
	try:
		resp = urllib.request.urlopen(req, timeout=timeout / 2)
	except urllib.error.HTTPError as ex:
		if ex.getcode() not in (400, 405):
			raise ConnectionError(ex.getcode(), ex.msg)
	except urllib.error.URLError:
		pass
	else:
		resp.close()
		return resp.headers
	with requests.get(url, headers=Request.header(), stream=True, verify=False, timeout=timeout) as resp:
		try:
			resp.raise_for_status()
		except requests.exceptions.HTTPError as ex:
			raise ConnectionError(ex.response.status_code, ex.response.reason)
		return resp.headers

def download_file(*urls, filename=None, timeout=12, return_headers=False):
	if filename is None:
		file = io.BytesIO()
	else:
		file = open(filename, "wb")
	headers = {}
	for url in urls:
		req = urllib.request.Request(url, method="GET", headers=Request.header())
		try:
			resp = urllib.request.urlopen(req, timeout=timeout)
		except urllib.error.HTTPError as ex:
			raise ConnectionError(ex.getcode(), ex.msg)
		else:
			if not headers:
				headers = dict(resp.headers)
		shutil.copyfileobj(resp, file, 65536)
		resp.close()
	if filename is None:
		if return_headers:
			return file.getbuffer(), resp.headers
		return file.getbuffer()
	file.close()
	if return_headers:
		return filename, resp.headers
	return filename

def getsize(fp):
	if isinstance(fp, byte_like):
		return len(fp)
	if isinstance(fp, str):
		return os.path.getsize(fp)
	if hasattr(fp, "seek"):
		p = fp.tell()
		try:
			return fp.seek(0, os.SEEK_END)
		finally:
			fp.seek(p)
	raise NotImplementedError(fp)

def update_headers(headers, **fields):
	"Updates a dictionary of HTTP headers with new fields. Case-insensitive."
	lowers = {k.lower(): k for k in headers}
	for k, v in fields.items():
		k2 = k.lower()
		if k2 in lowers:
			headers.pop(lowers[k2])
		headers[k] = v
	return headers

sps = {}
browsers = {}
def new_playwright_page(browser="firefox", viewport=dict(width=480, height=320), headless=True):
	tid = threading.get_ident()
	h = f"{browser}~{tid}~{int(headless)}"
	try:
		browser = browsers[h]
	except KeyError:
		try:
			sp = sps[tid]
		except KeyError:
			from playwright.sync_api import sync_playwright
			sp = sps[tid] = sync_playwright().start()
		browser = browsers[h] = getattr(sp, browser).launch(headless=headless)
	context = browser.new_context(
		device_scale_factor=1,
		is_mobile=False,
		has_touch=True,
		viewport=viewport,
		default_browser_type="firefox",
		user_agent=USER_AGENT,
	)
	return context.new_page()


CACHE_FILESIZE = 10 * 1048576
DEFAULT_FILESIZE = 50 * 1048576

mime_wait.result(timeout=8)
