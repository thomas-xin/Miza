import aiohttp
import ast
import asyncio
import base64
import collections
from collections import deque
import concurrent.futures
import contextlib
import datetime
import functools
import hashlib
import io
import json
from math import ceil, comb, inf, isfinite, isqrt
import multiprocessing.connection
import multiprocessing.shared_memory
import os
import pickle
import psutil
import random
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from traceback import format_exc, print_exc
from urllib.parse import quote_plus, unquote_plus
import zipfile
from dynamic_dt import DynamicDT
import filetype
import nacl.secret
import numpy as np
import orjson
try:
	import pynvml
except Exception:
	pynvml = None
import requests
from misc.types import ISE, CCE, Dummy, PropagateTraceback, is_exception, alist, cdict, fcdict, as_str, lim_str, single_space, try_int, round_min, regexp, suppress, loop, T2, safe_eval, number, byte_like, json_like, hashable_args, always_copy, astype, MemoryBytes, ts_us, utc, tracebacksuppressor, T, coerce, coercedefault, updatedefault, json_dumps, json_dumpstr, MultiEncoder, sublist_index # noqa: F401
from misc.asyncs import await_fut, wrap_future, awaitable, reflatten, asubmit, csubmit, esubmit, tsubmit, waited_sync, Future, Semaphore

try:
	from random import randbytes
except ImportError:
	def randbytes(size):
		return np.random.randint(0, 256, size=size, dtype=np.uint8).data

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
	if not os.path.exists(cachedir):
		os.mkdir(cachedir)

CACHE_PATH = cachedir + "/cache"
if not os.path.exists(CACHE_PATH):
	os.mkdir(CACHE_PATH)
TEMP_PATH = AUTH.get("temp_path")
if not TEMP_PATH or not os.path.exists(TEMP_PATH):
	TEMP_PATH = "cache"
	if not os.path.exists(TEMP_PATH):
		os.mkdir(TEMP_PATH)
FAST_PATH = "cache"
if not os.path.exists(FAST_PATH):
	os.mkdir(FAST_PATH)

persistdir = AUTH.get("persist_path") or cachedir
ecdc_dir = persistdir + "/ecdc/"

DEBUG = astype(AUTH.get("debug", ()), frozenset)

PORT = AUTH.get("webserver_port", 80)
if PORT:
	PORT = int(PORT)
IND = "\x7f"

compat_python = AUTH.get("compat_python") or python

_globals = globals()
def save_auth(auth):
	globals()["AUTH"].update(auth)
	_globals["AUTH"].update(auth)
	with open("auth.json", "w", encoding="utf-8") as f:
		json.dump(AUTH, f, indent="\t")


DC = 0
try:
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
except Exception:
	print_exc()
hwaccel = "cuda" if DC else "d3d11va" if os.name == "nt" else "auto"


api = "v10"


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
	engaged = set(s.laddr.port for s in psutil.net_connections())
	choices = set(range(32768, 65536)).difference(engaged)
	if not choices:
		raise ValueError("No inbound TCP ports available.")
	return random.choice(tuple(choices))

def esafe(s, binary=False):
	return (d := as_str(s).replace("\n", "\uffff")) and (d.encode("utf-8") if binary else d)
def dsafe(s):
	return as_str(s).replace("\uffff", "\n")
def nop2(s):
	return s

PROC = psutil.Process()
def quit(*args, **kwargs): force_kill(PROC)


# SHA256 operations: base64 and base16.
def shash(s): return e64(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).decode("ascii")
def uhash(s): return sorted([shash(s), quote_plus(s.removeprefix("https://"))], key=len)[0]
def hhash(s): return hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).hexdigest()
def ihash(s): return int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little") % 4294967296 - 2147483648
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
	if e == "llamav2":
		from transformers import AutoTokenizer
		return AutoTokenizer.from_pretrained("wolfram/miquliz-120b-v2.0", use_fast=True)
	if e == "cohere":
		from transformers import AutoTokenizer
		return AutoTokenizer.from_pretrained("alpindale/c4ai-command-r-plus-GPTQ", use_fast=True)
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

# Discord markdown format helper functions
def no_md(s):
	return str(s).translate(__emap)
def clr_md(s):
	return str(s).translate(__emap2)
def sqr_md(s):
	return f"[#{no_md(s)}]" if hasattr(s, "send") and not hasattr(s, "bot") else f"[{no_md(s)}]"

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
def ansi_md(s):
	if not s:
		return "``` ```"
	s = s.replace("[[", colourise("[", fg="cyan") + colourise("", fg="blue")).replace("]]", colourise("]", fg="cyan") + colourise())
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
def colourise(s=None, fg=None, bg=None):
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

# Discord object mention formatting
def user_mention(u_id):
	return f"<@{u_id}>"
def user_pc_mention(u_id):
	return f"<@!{u_id}>"
def channel_mention(c_id):
	return f"<#{c_id}>"
def role_mention(r_id):
	return f"<@&{r_id}>"
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
	while len(s) > 7:
		try:
			i = s.index("&#")
		except ValueError:
			break
		try:
			if s[i + 2] == "x":
				base = 16
				p = i + 3
			else:
				base = 10
				p = i + 2
			for a in range(p, p + 16):
				c = s[a]
				if c == ";":
					v = int(s[p:a], base)
					break
				elif not c.isnumeric() and c not in "abcdefABCDEF":
					break
			c = chr(v)
			s = s[:i] + c + s[a + 1:]
		except (ValueError, NameError, IndexError):
			s = s[:i + 1] + "\u200b" + s[i + 1:]
			continue
	s = s.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*").replace("<u>", "*").replace("</u>", "*")
	s = s.replace("\u200b", "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
	return s.replace("&quot;", '"').replace("&apos;", "'")

@hashable_args
@functools.lru_cache(maxsize=256)
def split_across(s, lim=2000, prefix="", suffix="", mode="len", bypass=((), ()), close_codeboxes=True) -> list:
	"Splits a string into segments according to given length limit. Prefers splitting across paragraphs/lines than words/characters."
	cb = "```"
	if cb in suffix:
		close_codeboxes = False
	current_codebox = ""
	s = s.replace("\r\n", "\n")
	splitters = ["\n\n", "\n", "\t", "? ", "! ", ". ", " "]

	if mode == "len":
		raw_len = len
	elif mode == "tlen":
		@functools.lru_cache(maxsize=64)
		def raw_len(s):
			return tlen(s)
	else:
		raise NotImplementedError(f"split_across: Unsupported mode {mode}")
	@functools.lru_cache(maxsize=64)
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
				pass #splitters.remove(cut)
			else:
				yield s[:i], cut
	def try_one(s, cut):
		try:
			i = s.rindex(cut)
		except ValueError:
			return s, ""
		else:
			return s[:i], cut

	if close_codeboxes:
		if "\n" in s and "#" in s and cb in s:
			pattern = re.compile(rf"^((?:[#\-\s]|[0-9]+\.\s).*{cb})", re.MULTILINE)
			s = pattern.sub(lambda x: x.group(1).lstrip()[:-3] + ("\n" + cb), s)
	if required_len(s) <= lim:
		return [complete(s)[0]]
	out = []
	temp = ""
	found = None
	while s:
		budget = max(1, lim - required_len(temp) + 1)
		# print(budget)
		if mode == "len":
			checker = s[:budget]
		else:
			checker = tik_decode(tik_encode(s[:budget * 8])[:budget])
		if found is not None:
			cur, cut = try_one(checker, found)
			if required_len(temp) + raw_len(cur) > lim or required_len(new := temp + cur) > lim:
				if required_len(temp) > lim / 2:
					text, current_codebox = complete(temp)
					out.append(text)
					temp = ""
					found = None
					continue
			else:
				s = s[len(cur + cut):]
				found = cut
				temp = new + cut
				continue
		for cur, cut in tries(checker):
			if required_len(temp) + raw_len(cur) > lim or required_len(new := temp + cur) > lim:
				continue
			s = s[len(cur + cut):]
			found = cut
			temp = new + cut
			break
		else:
			if mode == "len":
				n_required = lim - required_len(temp)
				temp = temp + checker[:n_required]
				s = s[n_required:]
			else:
				n_required = lim - required_len(temp)
				tokens = tik_encode(checker)
				temp = temp + tik_decode(tokens[:n_required])
				s = s[len(tik_decode(tokens[n_required:])):]
			text, current_codebox = complete(temp)
			out.append(text)
			temp = ""
			found = None
			continue
	if temp:
		out.append(complete(temp)[0])
	return out

def apply_translator(s, t, reverse=False):
	if reverse:
		for k, v in t.items():
			s = s.replace(v, k)
		return s
	for k, v in t.items():
		s = s.replace(k, v)
	return s

smart_map = {
	"'''": "\ufffd",
	"```": "\ufffe",
	'"""': "\uffff",
}
def smart_shlex(s):
	t = shlex.shlex(s, posix=True)
	t.whitespace_split = True
	t.escape = "\\"
	t.quotes = t.escapedquotes = "'\"`" + "".join(smart_map.values())
	return t
def smart_split(s, rws=False):
	si = apply_translator(s, smart_map)
	try:
		t = smart_shlex(si)
		out = deque()
		while True:
			try:
				w = t.get_token()
			except ValueError:
				remainder = si[len(si) - len(t.token) - len(t.state):]
				tup = remainder.split(None, 1)
				out.append(tup[0])
				if len(tup) == 1:
					break
				t = smart_shlex(tup[-1])
				continue
			if w is None or w == t.eof:
				break
			out.append(w)
	except ValueError:
		out = si.split()
	out = [apply_translator(w, smart_map, reverse=True) for w in out]
	if rws:
		whites = []
		for w in out:
			if not w:
				continue
			if w not in s:
				whites.append("")
				continue
			left, s = s.split(w, 1)
			whites.append(left)
		whites.append(s)
		return out, whites
	return alist(out)

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
def is_url(url): return url and isinstance(url, (str, bytes)) and regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+$").fullmatch(url)
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
def unyt(s):
	if not is_url(s):
		return s
	if (s.startswith("https://mizabot.xyz/u") or s.startswith("https://api.mizabot.xyz/u")) and ("?url=" in s or "&url=" in s):
		s = unquote_plus(s.replace("&url=", "?url=", 1).split("?url=", 1)[-1])
	if s.startswith("https://mizabot.xyz/ytdl") or s.startswith("https://api.mizabot.xyz/ytdl"):
		if "?d=" in s or "?v=" in s:
			s = unquote_plus(s.replace("?v=", "?d=", 1).split("?d=", 1)[-1])
		else:
			s = re.sub(r"https?:\/\/(?:\w{1,5}\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/)|youtu\.be\/)|https?:\/\/(?:api\.)?mizabot\.xyz\/ytdl\?[vd]=(?:https:\/\/youtu\.be\/|https%3A%2F%2Fyoutu\.be%2F)", "https://youtu.be/", re.sub(r"[\?&]si=[\w\-]+", "", s))
		s = s.split("&", 1)[0]
	if is_discord_attachment(s):
		s = s.split("?", 1)[0]
	return re.sub(r"https?:\/\/(?:\w{1,5}\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/)|youtu\.be\/)", "https://youtu.be/", re.sub(r"[\?&]si=[\w\-]+", "", s))
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
	if is_soundcloud_stream(stream):
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

def url2fn(url) -> str:
	return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]

def replace_ext(fn, ext="") -> str:
	if "." not in fn:
		return fn + "." + ext
	return fn.rsplit(".", 1)[0] + "." + ext

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

def get_image_size(b):
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
		input = io.BytesIO(b.read(256))
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
	if isinstance(input, str):
		input = open(input, "rb")
	from PIL import Image
	input.seek(0)
	try:
		with Image.open(input) as im:
			return im.size
	except Exception as ex:
		print(repr(ex))
		ts = ts_us()
		temp = TEMP_PATH + f"/{ts}"
		input.seek(0)
		with open(temp, "wb") as f:
			shutil.copyfileobj(input, f, 65536)
		cmd = ("./ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", temp)
		print(cmd)
		try:
			out = subprocess.check_output(cmd)
		finally:
			os.remove(temp)
		if not out or b"x" not in out:
			raise
	return list(map(int, out.split(b"x")))

def rename(src, dst):
	try:
		return os.replace(src, dst)
	except FileExistsError:
		if os.path.exists(dst):
			os.remove(dst)
		return os.rename(src, dst)
	except OSError as ex:
		if ex.args and ex.args[0] == 18:
			with open(dst, "wb") as fdst:
				with open(src, "rb") as fsrc:
					shutil.copyfileobj(fsrc, fdst)

IMAGE_FORMS = {
	"auto": None,
	"gif": True,
	"png": True,
	"apng": True,
	"bmp": False,
	"jpg": True,
	"jpeg": True,
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
		fn = url2fn(url)
		if "." in fn:
			return IMAGE_FORMS.get(fn.rsplit(".", 1)[-1])

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
	"zip": False,
}
def is_video(url):
	"Checks whether a url or filename ends with a video file extension. Returns a ternary True/False/None value where True indicates a positive match, False indicates a possible match, and None indicates no match."
	if url:
		fn = url2fn(url)
		if "." in fn:
			return VIDEO_FORMS.get(fn.rsplit(".", 1)[-1])

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
		fn = url2fn(url)
		if "." in fn:
			return AUDIO_FORMS.get(fn.rsplit(".", 1)[-1])

VISUAL_FORMS = {
	"auto": None,
	"gif": True,
	"png": True,
	"apng": True,
	"bmp": False,
	"jpg": True,
	"jpeg": True,
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
	"mpg": True,
	"mpeg": True,
	"mpv": True,
	"m3u8": True,
	"zip": False,
}
MEDIA_FORMS = IMAGE_FORMS.copy()
MEDIA_FORMS.update(VIDEO_FORMS)
MEDIA_FORMS.update(AUDIO_FORMS)

CODECS = {
	"auto": "auto",
	"x264": "mp4",
	"h264": "mp4",
	"avc": "mp4",
	"av1": "webm",
	"vp9": "webm",
	"vp8": "webm",
	"quicktime": "qt",
	"x265": "ts",
	"h265": "ts",
	"hevc": "ts",
	"mpegts": "ts",
	"libopus": "opus",
	"matroska": "mkv",
	"s16le": "pcm",
}
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
	ts="video/ts",
	webm="video/mp2t",
	weba="audio/weba",
	qt="video/quicktime",
	mp3="audio/mpeg",
	ogg="audio/ogg",
	opus="audio/opus",
	flac="audio/flac",
	wav="audio/x-wav",
	mp4="video/mp4",
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

def simple_mimes(b, mime=True):
	"Low-latency function that detects mimetype from first few bytes. Less accurate than from_file."
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
	"tar": "x-tar",
	"qt": "quicktime",
	"ts": "mp2t",
	"txt": "plain",
	"7z": "x-7z-compressed",
}
inv_mimes = {v: k for k, v in special_mimes.items()}

def mime_into(mime):
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

def get_ext(f):
	mime = from_file(f)
	return mime_into(mime)

def from_file(path, filename=None, mime=True):
	"Detects mimetype of file or buffer. Includes custom .jar, .ecdc, .m3u8 detection."
	data = filetype.get_bytes(path)
	if mime:
		out = filetype.guess_mime(data)
	else:
		out = filetype.guess_extension(data)
	filename = filename or (path if isinstance(path, str) else "")
	if out and out.split("/", 1)[-1] == "zip" and filename.endswith(".jar"):
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
		if data.startswith(b'\x00\x00\x00,ftypavis'):
			return "image/avif"
	if out == "text/plain" and data.startswith(b"#EXTM3U"):
		return "video/m3u8"
	return out

magic = cdict(
	from_file=from_file,
	from_buffer=from_file,
	Magic=lambda mime, *args, **kwargs: cdict(
		from_file=lambda b: from_file(b, mime),
		from_buffer=lambda b: from_file(b, mime),
	),
)

def get_mime(path):
	if os.path.getsize(path) < 1048576:
		try:
			mime = magic.from_file(path, mime=True)
		except Exception:
			print_exc()
			mime = "cannot open `"
	else:
		mime = "cannot open `"
	if mime.startswith("cannot open `"):
		with open(path, "rb") as f:
			b = f.read(1048576)
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
		mime2 = MIMES.get(path.rsplit("/", 1)[-1].rsplit(".", 1)[-1], "")
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

def ecdc_exists(url, exc=False, force=None):
	url = unyt(url)
	h = shash(url)
	elist = {}
	for fn in os.listdir(ecdc_dir):
		if not os.path.getsize(ecdc_dir + fn):
			os.remove(ecdc_dir + fn)
			continue
		if fn.startswith("!" + h + "~"):
			if force and fn.rsplit("/", 1)[-1] != force.rsplit("/", 1)[-1]:
				os.remove(ecdc_dir + fn)
				continue
			brs = fn.rsplit("~", 1)[-1].rsplit(".", 1)[0]
			if brs.endswith(".0"):
				fn2 = fn.rsplit("~", 1)[0] + "~" + brs[:-2] + "." + fn.rsplit(".", 1)[-1]
				if os.path.exists(ecdc_dir + fn2):
					os.remove(ecdc_dir + fn)
					continue
				else:
					os.rename(ecdc_dir + fn, ecdc_dir + fn2)
					fn = fn2
			br = float(brs)
			elist[br] = fn
	if not elist:
		if not exc:
			return
		raise FileNotFoundError(h)
	n = max(elist)
	fn = elist.pop(n)
	if elist:
		for f2 in elist.values():
			os.remove(ecdc_dir + f2)
	return ecdc_dir + fn

@functools.lru_cache(maxsize=None)
def ecdc_info(fn):
	if is_url(fn):
		fn = ecdc_exists(fn, exc=True)
	from misc import ecdc_stream
	return cdict(ecdc_stream.get_info(fn))

def ecdc_br(fn):
	bps = None
	try:
		_dur, bps, cdc, *_ = get_duration_2(fn)
	except Exception:
		print_exc()
	else:
		if bps:
			if cdc in ("wav", "flac"):
				bps //= 4
			if cdc not in ("opus", "vorbis"):
				bps //= 2
	bps = bps or 192000
	br = min(32, round(bps / 6000))
	if 18 < br < 24:
		br = 24
	elif 9 < br < 12:
		br = 12
	elif 4.5 < br < 6:
		br = 6
	return br

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

def e64(b):
	if isinstance(b, str):
		b = b.encode("utf-8")
	elif isinstance(b, MemoryBytes):
		b = bytes(b)
	return base64.urlsafe_b64encode(b).rstrip(b"=")

def b64(b):
	if isinstance(b, str):
		b = b.encode("ascii")
	elif isinstance(b, MemoryBytes):
		b = bytes(b)
	if len(b) & 3:
		b += b"=="
	return base64.urlsafe_b64decode(b)

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

def b2n(b):
	return int.from_bytes(b, "big")
def n2b(n):
	c = n.bit_length() + 7 >> 3
	return n.to_bytes(c, "big")

def leb128(n):
	"Encodes an integer using LEB128."
	data = bytearray()
	while n:
		data.append(n & 127)
		n >>= 7
		if n:
			data[-1] |= 128
	return data or b"\x00"
def decode_leb128(data, mode="cut"): # mode: cut | index
	"Decodes an integer from LEB128 encoded data; optionally returns the remaining data."
	i = n = 0
	shift = 0
	for i, byte in enumerate(data):
		n |= (byte & 0x7F) << shift
		if byte & 0x80 == 0:
			break
		else:
			shift += 7
	if mode == "cut":
		return n, data[i + 1:]
	return n, i + 1

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

def encode_snowflake(*args, store_count=False):
	"""
	Encodes a list of snowflake IDs into a compact string representation.

	Args:
		*args: Variable length argument list of snowflake IDs to encode.
		store_count (bool): If True, stores the count of snowflake IDs in the encoded string.
							If False, uses a padding byte or predetermined count.

	Returns:
		str: The encoded string representation of the snowflake IDs.

	Raises:
		AssertionError: If store_count is True and the number of snowflake IDs is not between 2 and 126 inclusive.

	Notes:
		- The function extracts the timestamp, worker ID, process ID, and increment from each snowflake ID.
		- It interleaves these components and encodes them using LEB128 and Szudzik's pairing function.
		- The encoded string is then base64 encoded and returned.
		- If store_count is True, the count of snowflake IDs is stored in the encoded string.
		- If store_count is False and the first byte of the encoded string is less than 128, a padding byte (0x7f) is added.
	"""
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
		assert 1 < len(args) < 127
		encoded = bytes([len(args)]) + encoded
	elif encoded[0] < 128:
		encoded = b"\x7f" + encoded
	return e64(encoded).decode("ascii")
def decode_snowflake(data, n=1):
	"""
	Decodes a snowflake ID into its constituent parts.

	Args:
		data (str): The base64 encoded snowflake ID.
		n (int, optional): The number of IDs to decode. Defaults to 1.

	Returns:
		list: A list of decoded snowflake IDs, each represented as an integer.

	Raises:
		ValueError: If the input data is not valid base64 or if decoding fails.

	Note:
		The function assumes the input data is a valid base64 encoded string.
		It decodes the string, extracts the timestamp, worker ID, process ID,
		and increment values, and then reconstructs the original snowflake IDs.
	"""
	decoded = b64(data)
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

def split_url(url, m_id):
	_, c_id, a_id, fn = url.split("?", 1)[0].rsplit("/", 3)
	return (int(c_id), int(m_id) if m_id is not None else None, int(a_id), fn)
def merge_url(c_id, m_id, a_id, fn):
	return f"https://cdn.discordapp.com/attachments/{c_id}/{a_id}/{fn}", m_id
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

@split_attachment
def encode_attachment(c_id, m_id, a_id, fn):
	if a_id == 0:
		return encode_snowflake(*map(int, (c_id, m_id)), store_count=True) + "/" + fn
	return encode_snowflake(*map(int, (c_id, m_id, a_id))) + "/" + fn
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
def shorten_attachment(c_id, m_id, a_id, fn, mode="u", size=0, base="https://mizabot.xyz"):
	url = f"{base}/{mode}/" + encode_attachment(c_id, m_id, a_id, fn)
	if size:
		url += f"?size={size}"
	return url
def expand_attachment(url):
	assert "//" in url
	regs = regexp(r"\/\w\/").split(url.split("?", 1)[0], 1)
	assert len(regs) == 2
	encoded = regs[-1]
	return decode_attachment(encoded)

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
	enc_key = (AUTH.get("discord_secret") or AUTH.get("discord_token") or e64(randbytes(32)).decode("ascii")).replace(".", "A").replace("_", "a").replace("-", "a")[:43]
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
	if s[:8] == b">~MIZA~>":
		return enc_box.decrypt(s[8:])
	raise ValueError("Data header not found.")

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

def maybe_json(d):
	if isinstance(d, BaseException):
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
def select_and_loads(s, encrypted=False, size=None):
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
	if s[0] == 128:
		return pickle.loads(s)
	if s[:1] in (b"~", b"!") or zipfile.is_zipfile(io.BytesIO(s)):
		s = zip2bytes(s)
	data = None
	if not s:
		return data
	if s[0] == 128:
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

def select_and_dumps(data, mode="safe", compress=True):
	"Automatically serialises data as JSON or Pickle, compressing if beneficial."
	if isinstance(data, Future):
		data = data.result(timeout=2)
	if mode == "unsafe":
		# try:
		# 	if len(data) and isinstance(data, dict) and not isinstance(next(iter(data)), str):
		# 		raise TypeError
		# 	if isinstance(data, (set, frozenset)):
		# 		s = b"$" + json_dumps(list(data))
		# 	else:
		# 		s = json_dumps(data)
		# except TypeError:
		# 	s = pickle.dumps(data)
		s = pickle.dumps(data)
		if len(s) > 32768 and compress:
			t = bytes2zip(s, lzma=len(s) > 16777216)
			if len(t) < len(s) * 0.9:
				s = t
		return s
	try:
		s = json_dumps(data)
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
		with open(fn + "\x7f", "wb") as f:
			f.write(s)
		with open(f + "\x7f", "rb") as f:
			if f.read(1) in (b"\x00", b" ", b""):
				raise ValueError
		with tracebacksuppressor(FileNotFoundError):
			os.remove(fn + "\x7f\x7f")
	if os.path.exists(fn) and not os.path.exists(fn + "\x7f\x7f"):
		os.rename(fn, fn + "\x7f\x7f")
		os.rename(fn + "\x7f", fn)
	else:
		with open(fn, "wb") as f:
			f.write(s)


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

class DownloadingFile(io.IOBase):
	"A buffer indicating a file that is currently being written. Calls to read() when there is no more data but the write is not yet complete will block until either condition is met."

	__slots__ = ("fp", "fn", "mode", "filename", "af")
	min_buffer = 4096

	def __init__(self, fn, af, mode="rb", filename=None, min_buffer=4096):
		self.fp = None
		self.fn = fn
		self.mode = mode
		self.filename = filename or T(fn).get("name") or fn
		self.af = af
		self.last_size = 0
		self.min_buffer = min_buffer
		for _ in loop(720):
			if os.path.exists(fn) and os.path.getsize(fn) > 3:
				break
			if af():
				raise FileNotFoundError
			time.sleep(0.1)

	def __getattribute__(self, k):
		if k in object.__getattribute__(self, "__slots__") or k in ("seek", "read", "clear", "min_buffer", "scan_size", "last_size"):
			return object.__getattribute__(self, k)
		if k == "name":
			return object.__getattribute__(self, "filename")
		if self.fp is None:
			self.fp = open(self.fn, self.mode)
		if k[0] == "_" and (len(k) < 2 or k[1] != "_"):
			k = k[1:]
		return getattr(self.fp, k)

	def scan_size(self, req=0):
		while req > self.last_size:
			time.sleep(0.1)
			if self.af():
				break
			self.last_size = os.path.exists(self.fn) and os.path.getsize(self.fn)
			print(self.last_size, req)

	def seek(self, pos):
		self.scan_size(pos)
		self._seek(pos)

	def read(self, size):
		pos = self.tell()
		self.scan_size(pos + size + self.min_buffer)
		b = self._read(size)
		s = len(b)
		if s < size:
			buf = deque([b])
			n = 1
			while s < size:
				time.sleep(n / 3)
				n += 1
				b = self._read(size - s)
				if not b:
					if self.af():
						print("AF Complete.")
						b = self._read(size - s)
						if not b:
							break
					else:
						continue
				s += len(b)
				buf.append(b)
			b = b"".join(buf)
		return b

	def clear(self):
		with suppress(Exception):
			self.fp.close()
		self.fp = None


def get_duration_2(filename, _timeout=12):
	command = (
		"./ffprobe",
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"stream=codec_name,channels,duration",
		"-show_entries",
		"format=duration,bit_rate",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		filename,
	)
	resp = None
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = esubmit(proc.wait, timeout=_timeout)
		_res = fut.result(timeout=_timeout)
		resp = proc.stdout.read().splitlines()
	except Exception:
		with suppress():
			proc.kill()
		print_exc()
	try:
		cdc = as_str(resp[0].rstrip())
	except (IndexError, ValueError, TypeError):
		cdc = "auto"
	try:
		ac = int(resp[1].rstrip())
	except (IndexError, ValueError, TypeError):
		ac = 0
	try:
		dur = float(resp[2])
	except (IndexError, ValueError, TypeError):
		try:
			dur = float(resp[3])
		except (IndexError, ValueError, TypeError):
			dur = None
	bps = None
	if resp and len(resp) > 4:
		with suppress(ValueError):
			bps = float(resp[4])
	return dur, bps, cdc, ac

def get_duration_simple(filename, _timeout=12):
	"Runs ffprobe on a file or url, returning the duration if possible."
	command = (
		"./ffprobe",
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"stream=duration",
		"-show_entries",
		"format=duration,bit_rate",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		filename,
	)
	resp = None
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = esubmit(proc.wait, timeout=_timeout)
		fut.result(timeout=_timeout)
		resp = proc.stdout.read().split()
	except Exception:
		with suppress():
			force_kill(proc)
		with suppress():
			resp = proc.stdout.read().split()
		print_exc()
	try:
		dur = float(resp[0])
	except (IndexError, ValueError, TypeError):
		try:
			dur = float(resp[1])
		except (IndexError, ValueError, TypeError):
			dur = None
	bps = None
	if resp and len(resp) > 2:
		with suppress(ValueError):
			bps = float(resp[2])
	return dur, bps

def get_duration(filename):
	"Gets the duration of an audio/video file using metadata, bitrate, filesize etc. Falls back to FFMPEG"
	if not filename:
		return
	dur, bps = get_duration_simple(filename, 4)
	if not dur and is_url(filename):
		with reqs.next().get(filename, headers=Request.header(), stream=True) as resp:
			head = fcdict(resp.headers)
			if "Content-Length" not in head:
				dur = get_duration_simple(filename, 20)[0]
				return dur
			if bps:
				print(head, bps, sep="\n")
				return (int(head["Content-Length"]) << 3) / bps
			ctype = [e.strip() for e in head.get("Content-Type", "").split(";") if "/" in e][0]
			if ctype.split("/", 1)[0] not in ("audio", "video") or ctype == "audio/midi":
				return
			it = resp.iter_content(65536)
			data = next(it)
		ident = str(magic.from_buffer(data))
		print(head, ident, sep="\n")
		try:
			bitrate = regexp("[0-9.]+\\s.?bps").findall(ident)[0].casefold()
		except IndexError:
			dur = get_duration_simple(filename, 16)[0]
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
	return dur


class ForwardedRequest(io.IOBase):
	"A requests-compatible buffer that caches read data to enable seeking."

	__slots__ = ("fp", "resp", "size", "pos", "it")

	def __init__(self, resp, buffer=65536):
		self.resp = resp
		self.it = resp.iter_content(buffer)
		self.fp = io.BytesIO()
		self.size = 0
		self.pos = 0

	def __getattribute__(self, k):
		if k in object.__getattribute__(self, "__slots__") or k in ("seek", "read", "clear"):
			return object.__getattribute__(self, k)
		if k == "name":
			return object.__getattribute__(self, "filename")
		if self.fp is None:
			self.fp = open(self.fn, self.mode)
		if k[0] == "_" and (len(k) < 2 or k[1] != "_"):
			k = k[1:]
		return getattr(self.fp, k)

	def seek(self, pos):
		while self.size < pos:
			try:
				n = next(self.it)
			except StopIteration:
				n = b""
			if not n:
				self.resp.close()
				break
			self.fp.seek(self.size)
			self.size += len(n)
			self.fp.write(n)
		self.fp.seek(pos)
		self.pos = pos

	def read(self, size):
		b = self.fp.read(size)
		s = len(b)
		self.pos += s
		while s < size:
			try:
				n = next(self.it)
			except StopIteration:
				n = b""
			if not n:
				self.resp.close()
				break
			self.fp.seek(self.size)
			self.size += len(n)
			self.fp.write(n)
			self.fp.seek(self.pos)
			b += self.fp.read(size - s)
			s += len(b)
			self.pos += len(b)
		return b

	def clear(self):
		with suppress(Exception):
			self.fp.close()
		self.fp = None

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
	"A custom subprocess/psutil-compatible Process implementation that invokes multiple processes simultaneously, automatically piping their stdin/stdout data. All stderr output is captured in the destination stdout."

	procs = ()
	stdin = stdout = stderr = None

	def __init__(self, *args, stdin=None, stdout=None, stderr=None, cwd=".", bufsize=4096):
		if not args:
			return
		self.exc = concurrent.futures.ThreadPoolExecutor(max_workers=len(args) - 1) if len(args) > 1 else None
		self.procs = []
		for i, arg in enumerate(args):
			first = not i
			last = i >= len(args) - 1
			si = stdin if first else subprocess.PIPE
			so = stdout if last else subprocess.PIPE
			se = stderr if last else None
			proc = psutil.Popen(arg, stdin=si, stdout=so, stderr=se, cwd=cwd, bufsize=bufsize * 256)
			if first:
				self.stdin = proc.stdin
				self.args = arg
			if last:
				self.stdout = proc.stdout
				self.stderr = proc.stderr
			self.procs.append(proc)
		for i in range(len(args) - 1):
			self.exc.submit(self.pipe, i, bufsize=bufsize)
		self.pid = self.procs[0].pid

	def pipe(self, i, bufsize=4096):
		try:
			proc = self.procs[i]
			proc2 = self.procs[i + 1]
			si = 0
			while proc.is_running() and proc2.is_running():
				b = proc.stdout.read(si * (si + 1) * bufsize // 8 + bufsize)
				if not b:
					break
				proc2.stdin.write(b)
				proc2.stdin.flush()
				si += 1
			if proc2.is_running():
				proc2.stdin.close()
		except Exception:
			print_exc()
			if not proc.is_running() or not proc2.is_running():
				self.terminate()
		if self.exc:
			self.exc.shutdown(wait=False)

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

	def __del__(self):
		del self.buffer
		del self.data

	seekable = lambda self: True	# noqa: E731
	readable = lambda self: True	# noqa: E731
	writable = lambda self: False	# noqa: E731
	isatty = lambda self: False		# noqa: E731
	flush = lambda self: None		# noqa: E731
	tell = lambda self: self.pos	# noqa: E731

	def seek(self, pos=0):
		self.pos = pos

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
		# print("Loading cursor", self)
		if self.db:
			self.db.close()
			self.db = None
		self.db = sqlite3.connect(self.path, check_same_thread=False)
		self.cur = alist([self.db.cursor() for i in range(self.max_concurrency)])
		try:
			self.cur[-1].execute(f"CREATE TABLE IF NOT EXISTS '{self.internal}' (key VARCHAR(256) PRIMARY KEY, value BLOB)")
		except Exception:
			print("Error in database", self)
			raise
		self.cur[-1].execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON '{self.internal}' (key)")
		self.codb = set(try_int(r[0]) for r in self.cur[-1].execute(f"SELECT key FROM '{self.internal}'") if r)
		# print("Loaded cursor", self)
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
			with self.db_sem:
				s = next(self.cur.next().execute(f"SELECT value FROM '{self.internal}' WHERE key=?", [k]))[0]
			if not s:
				self.deleted.add(k)
				raise RuntimeError(f"Found empty value for key {k} in {self.internal}")
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
				futs.append((k, self.parallel(select_and_dumps, d, mode="safe" if self.safe else "unsafe", compress=True)))
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
			self.db = None
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


class CacheItem:
	__slots__ = ("value")

	def __init__(self, value):
		self.value = value

class Cache(dict):
	"A dictionary-compatible object where the key-value pairs expire after a specified delay. Implements stale-while-revaluate protocol."
	# __slots__ = ("timeout", "timeout2", "tmap", "soonest", "waiting", "lost", "trash", "db")

	def __init__(self, *args, timeout=60, timeout2=8, trash=4096, persist=None, autosave=3600, **kwargs):
		super().__init__(*args, **kwargs)
		object.__setattr__(self, "timeout", timeout)
		object.__setattr__(self, "timeout2", timeout2)
		object.__setattr__(self, "lost", {})
		object.__setattr__(self, "trash", trash)
		object.__setattr__(self, "db", None)
		if self:
			ts = time.time()
			tmap = {k: ts for k in self}
			object.__setattr__(self, "soonest", ts)
		else:
			tmap = {}
			object.__setattr__(self, "soonest", inf)
		object.__setattr__(self, "tmap", tmap)
		if self:
			fut = esubmit(waited_sync, self._update, timeout)
		else:
			fut = None
		object.__setattr__(self, "waiting", fut)
		if persist is not None:
			path = f"{persistdir}/{persist}"
			db = FileHashDict(path=path, automut=False, autosave=autosave)
			self.attach(db)

	def attach(self, db):
		db.setdefault("__lost", {}).update(self.lost)
		object.__setattr__(self, "lost", db["__lost"])
		db.setdefault("__tmap", {}).update(self.tmap)
		db.pop("tmap", None)
		object.__setattr__(self, "db", db)
		object.__setattr__(self, "tmap", db["__tmap"])
		super().clear()
		if self.waiting:
			self.waiting.cancel()
		fut = esubmit(waited_sync, self._update, self.timeout)
		object.__setattr__(self, "waiting", fut)
		t = time.time()
		for k in self.db:
			if k not in self.tmap:
				self.tmap[k] = t

	def _update(self):
		tmap = self.tmap
		timeout = self.timeout
		lost = self.lost
		popcount = 0
		t = time.time()
		for k in sorted(tmap, key=tmap.__getitem__):
			if not isfinite(timeout):
				continue
			if t < tmap.get(k, inf) + timeout:
				ts = tmap.get(k, inf) + timeout
				object.__setattr__(self, "soonest", ts)
				if self.waiting:
					self.waiting.cancel()
				if isfinite(ts):
					fut = esubmit(waited_sync, self._update, ts - t)
					object.__setattr__(self, "waiting", fut)
				break
			popcount += 1
			tmap.pop(k)
			v = self.db.pop(k) if self.db else super().pop(k)
			if self.trash >= popcount:
				while len(lost) > self.trash:
					with suppress(KeyError, RuntimeError):
						lost.pop(next(iter(lost)))
				lost[k] = v
		else:
			object.__setattr__(self, "soonest", inf)
		if self.db:
			self.db.sync()
		return self

	def __iter__(self):
		if self.db is not None:
			return iter(self.db)
		return super().__iter__()

	def __len__(self):
		if self.db is not None:
			return len(self.db)
		return super().__len__()

	def __getitem__(self, k):
		if self.db is not None:
			try:
				return self.db.__getitem__(k)
			except KeyError:
				pass
		return super().__getitem__(k)

	def __setitem__(self, k, v):
		if self.db is None:
			super().__setitem__(k, v)
		else:
			self.db.__setitem__(k, v)
		self.lost.pop(k, None)
		timeout = self.timeout2 if is_exception(v) else self.timeout
		ts = time.time()
		if isfinite(timeout) and ts < self.soonest:
			object.__setattr__(self, "soonest", ts)
			if self.waiting:
				self.waiting.cancel()
			fut = esubmit(waited_sync, self._update, timeout)
			if self.waiting:
				self.waiting.cancel()
			object.__setattr__(self, "waiting", fut)
		self.tmap[k] = ts

	def update(self, other):
		super().update(self, other)
		t = time.time()
		for k, v in other.items():
			self.tmap[k] = t
		return self

	def pop(self, k, v=Dummy):
		try:
			v = self[k]
		except KeyError:
			if v is Dummy:
				raise KeyError(k)
			return v
		super().pop(k, None)
		if self.db is not None:
			self.db.pop(k, None)
		self.lost.pop(k, None)
		self.tmap.pop(k, None)
		return v

	def age(self, k):
		try:
			self[k]
		except KeyError:
			self.tmap.pop(k, None)
			return inf
		return utc() - self.tmap.get(k, -inf)

	def setdefault(self, k, v, timeout=None):
		try:
			resp = self[k]
		except KeyError:
			self[k] = v
			return v
		if timeout is not None and self.age(k) > timeout:
			self[k] = v
			return v
		return resp

	def retrieve(self, k):
		return self.lost.pop(k)

	async def retrieve_into(self, k, func, *args, **kwargs):
		resp = await asubmit(func, *args, **kwargs)
		self[k] = resp
		return resp

	async def retrieve_from(self, k, func, *args, **kwargs):
		try:
			resp = self[k]
		except KeyError:
			pass
		else:
			if isinstance(resp, CacheItem):
				try:
					resp = await resp.value
				except BaseException as ex:
					resp = ex
				if is_exception(resp):
					ti = self.tmap.get(k)
					if not ti or utc() - ti > self.timeout2:
						resp = Dummy
			if resp is not Dummy:
				if is_exception(resp):
					raise resp
				return resp
		fut = csubmit(self.retrieve_into(k, func, *args, **kwargs))
		super().__setitem__(k, CacheItem(fut))
		resp = Dummy
		try:
			resp = await fut
		except Exception:
			try:
				resp = self.retrieve(k)
			except KeyError:
				pass
			if is_exception(resp):
				raise
		super().__delitem__(k)
		return resp

	def clear(self):
		db = self.db
		super().clear()
		db.clear()
		db.setdefault("__tmap", {})
		object.__setattr__(self, "tmap", db["__tmap"])


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
		it.__init__(sorted(it, key=key, reverse=reverse))
		it.hash = None
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

@functools.lru_cache(maxsize=256)
def fuzzy_substring(sub, s, match_start=False, match_length=True):
	"A fuzzy substring search that returns the ratio of characters matched between two strings."
	if not match_length and s in sub:
		return 1
	if s.startswith(sub):
		return len(sub) / len(s) * 2
	match = 0
	if not match_start or sub and s.startswith(sub[0]):
		found = [0] * len(s)
		x = 0
		for i, c in enumerate(sub):
			temp = s[x:]
			if temp.startswith(c):
				if found[x] < 1:
					match += 1
					found[x] = 1
				x += 1
			elif c in temp:
				y = temp.index(c)
				x += y
				if found[x] < 1:
					found[x] = 1
					match += 1 - y / len(s)
				x += 1
			else:
				temp = s[:x]
				if c in temp:
					y = temp.rindex(c)
					if found[y] < 1:
						match += 1 - (x - y) / len(s)
						found[y] = 1
					x = y + 1
		if len(sub) > len(s) and match_length:
			match *= len(s) / len(sub)
	# ratio = match / len(s)
	ratio = max(0, match / len(s))
	return ratio


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
	if len(b) <= 65536 - 1:
		return sender(b"\x01" + b)
	mem = multiprocessing.shared_memory.SharedMemory(create=True, size=len(b))
	print(mem)
	MEMS[mem.name] = mem
	mem.buf[:len(b)] = b
	try:
		return sender(b"\x02" + len(b).to_bytes(8, "little") + mem.name.encode("utf-8"))
	except:
		mem.close()
		raise

@tracebacksuppressor
def receive_bytes(receiver, unlink):
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

	MEMS = globals()["MEMS"]

	def __init__(self, p_alive, p_in, p_out, p_kill=None, p_join=None, writable=True, start=True, glob=globals(), id=0, server=None):
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
		if start:
			self.start()

	key = b"EvalPipe"
	@classmethod
	def connect(cls, args, port, independent=True, glob=globals(), timeout=60):
		addr = ("127.0.0.1", port)
		print(f"{DynamicDT.now()}: EvalPipe connecting to", addr)
		try:
			conn = multiprocessing.connection.Client(addr, authkey=cls.key)
		except ConnectionRefusedError:
			if independent:
				script = None
				ts = ts_us()
				if os.name == "nt":
					argstr = " ".join(map(json_dumpstr, args)) + "\nexit /b"
					script = f"cache\\{ts}.bat"
					with open(script, "w") as f:
						f.write(argstr)
					args = f"wt -w 0 -d %cd% cmd /s/c {script}"
				else:
					argstr = " ".join(map(json_dumpstr, args))
					args = ["xterm", "-e", argstr]
			print(args)
			subprocess.Popen(args, shell=isinstance(args, str))
			for i in range(timeout):
				try:
					conn = multiprocessing.connection.Client(addr, authkey=cls.key)
				except ConnectionRefusedError:
					time.sleep(1)
				else:
					break
			else:
				raise
		print(f"{DynamicDT.now()}: New connection:", conn)
		assert conn.readable and conn.writable
		pid = conn.recv()
		proc = psutil.Process(pid)
		self = cls(
			proc.is_running,
			lambda b: share_bytes(conn.send_bytes, b),
			None,
			proc.terminate,
			proc.wait,
			glob=glob,
			id=port,
		)
		self.p_out = lambda: receive_bytes(conn.recv_bytes, unlink=lambda name: self.submit(f"EvalPipe.MEMS.pop({repr(name)}).close()"))
		self.proc = proc
		return self

	@classmethod
	def listen(cls, port=0, glob=globals(), start=False):
		addr = ("127.0.0.1", port)
		server = multiprocessing.connection.Listener(addr, authkey=cls.key)
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
				i = -i
			fut = self.responses[i] = Future()
		if isinstance(s, str):
			s = s.encode("utf-8")
		b = f"~>{i}:".encode("ascii") + s
		self.send(b)
		return fut

	def run(self, s, timeout=30, cache=None):
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
			res = self.submit(s).result(timeout=timeout)
			self.cache[s] = (res, t)
			return res
		return self.submit(s).result(timeout=timeout)

	async def asubmit(self, s):
		await self.a_ensure_writable()
		return await wrap_future(self.submit(s))

	def print(self, *args, sep=" ", end="\n"):
		b = ("\x00" + sep.join(map(str, args)) + end).encode("utf-8")
		Flush(sys.__stderr__.buffer, flush=sys.__stderr__.flush).write(b)
		try:
			self.ensure_writable()
			return self.send(b)
		except Exception:
			print_exc()

	def debug(self, s):
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
			print_exc()
			s = maybe_json(ex)
			b = f"<~{i}:!:".encode("ascii") + s
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
					print("New connection:", conn)
					conn.send(os.getpid())
					self.p_in = lambda b: share_bytes(conn.send_bytes, b)
					self.p_out = lambda: receive_bytes(conn.recv_bytes, unlink=lambda name: self.run(f"EvalPipe.MEMS.pop({repr(name)}).close()"))
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
		ex = RuntimeError("Subprocess killed.")
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


class ProxyManager:

	def __init__(self):
		try:
			from fp.fp import FreeProxy
		except Exception:
			print_exc()
			self.fp = None
		else:
			self.fp = FreeProxy()
		self.ctime = 0
		self.proxies = set()
		self.ptime = 0
		self.bad_proxies = set()
		self.btime = 0
		self.sem = Semaphore(2, inf, rate_limit=1)

	@property
	def fresh(self) -> bool:
		return bool(self.proxies) and utc() - self.ctime <= 120

	def get_proxy(self, retry=True) -> str:
		if self.fresh:
			return random.choice(tuple(self.proxies))
		if not self.fp:
			return
		with self.sem:
			while not self.proxies or utc() - self.ptime > 240:
				i = random.randint(1, 3)
				if i == 1:
					repeat = False
					self.fp.country_id = ["US"]
				elif i == 2:
					repeat = True
					self.fp.country_id = None
				else:
					repeat = False
					self.fp.country_id = None
				proxies = self.fp.get_proxy_list(repeat)
				self.proxies.update(proxies)
				if self.proxies:
					self.ptime = utc()
					break
				else:
					time.sleep(1)
			proxies = list(self.proxies)
			if utc() - self.btime > 480:
				self.bad_proxies.clear()
				self.btime = utc()
			else:
				self.proxies.difference_update(self.bad_proxies)
			futs = [esubmit(self.check_proxy, p) for p in proxies]
			for i, (p, fut) in enumerate(zip(proxies, futs)):
				try:
					assert fut.result(timeout=6)[0] == 105
				except (IndexError, AssertionError, T2):
					self.proxies.discard(p)
					self.bad_proxies.add(p)
			if not self.proxies:
				if not retry:
					return
				return self.get_proxy(retry=False)
			self.ctime = utc()
		return random.choice(tuple(self.proxies))

	def check_proxy(self, p):
		url = "https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/deleter.py"
		with reqs.next().get(url, timeout=5, proxies=dict(http=p, https=p), verify=False) as resp:
			return resp.content

	def request(self, url, headers={}, timeout=None, stream=True, method="get", **void):
		if is_discord_url(url):
			return getattr(reqs.next(), method.casefold())(url, headers=headers, timeout=timeout, stream=stream, verify=False)
		fut = None
		if not self.fresh and method == "get" and not stream:
			fut = esubmit(self.get_proxy)
			loc = random.choice(("eu", "us"))
			i = random.randint(1, 17)
			stream = f"https://{loc}{i}.proxysite.com/includes/process.php?action=update"
			try:
				resp = reqs.next().post(
					stream,
					data=dict(d=url, allowCookies="off"),
					timeout=timeout,
					stream=True,
				)
			except Exception:
				print_exc()
			else:
				if resp.status_code in range(200, 400):
					b = resp.content
					if b[:15] != b"<!DOCTYPE html>":
						return resp
		p = fut.result() if fut else self.get_proxy()
		if p:
			try:
				resp = getattr(reqs.next(), method.casefold())(url, headers=headers, timeout=timeout, stream=stream, proxies=dict(http=p, https=p), verify=False)
			except Exception:
				print_exc()
			else:
				if resp.status_code in range(200, 400):
					return resp
		loc = random.choice(("eu", "us"))
		i = random.randint(1, 17)
		stream = f"https://{loc}{i}.proxysite.com/includes/process.php?action=update"
		return reqs.next().post(
			stream,
			data=dict(d=url, allowCookies="on"),
			timeout=timeout,
			stream=stream,
		)
	get = request

	def content_or(self, url, headers={}, timeout=None, method="get", **void):
		resp = self.request(url, headers=headers, timeout=timeout, stream=False, method=method)
		if resp.status_code not in range(200, 400) or not resp.content or resp.content[:15] == b"<!DOCTYPE html>":
			resp = self.request(url, headers=headers, timeout=timeout, stream=False, method=method)
		if resp.status_code in range(200, 400) and resp.content and resp.content[:15] != b"<!DOCTYPE html>":
			return resp
		return getattr(reqs.next(), method.casefold())(url, headers=headers, timeout=timeout, verify=False)
		
proxy = ProxyManager()

def proxy_download(url, fn=None, timeout=720):
	"Downloads a file through proxysite.com; possibly outdated implementation."
	downloading = globals().setdefault("proxy-download", {})
	try:
		fut = downloading[url]
	except KeyError:
		downloading[url] = fut = Future()
	else:
		return fut.result(timeout=timeout)
	with proxy.get(url, timeout=timeout) as resp:
		if resp.status_code not in range(200, 400):
			raise ConnectionError(resp.status_code, (url, resp.content[:1024]))
		if not fn:
			b = resp.content
			return b
		it = resp.iter_content(65536)
		if isinstance(fn, str):
			f = open(fn, "wb")
		else:
			f = fn
		try:
			while True:
				b = next(it)
				if not b:
					break
				f.write(b)
		except StopIteration:
			pass
		return fn


# Manages both sync and async web requests.
class RequestManager(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, collections.abc.Callable):

	ts = 0
	semaphore = Semaphore(512, 256)
	sessions = ()

	@classmethod
	def header(cls, base=(), **fields) -> cdict:
		"Creates a custom HTTP request header with randomised properties that spoof anti-scraping sites."
		head = {
			"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in loop(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in loop(4)),
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
		self.sessions = alist(aiohttp.ClientSession() for i in range(6))
		self.nossl = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
		self.ts = utc()
		return self

	@property
	def session(self) -> aiohttp.ClientSession:
		return choice(self.sessions)

	async def aio_call(self, url, headers, files, data, method, decode=False, json=False, session=None, ssl=True, timeout=24) -> bytes | str | json_like:
		await self._init_()
		async with self.semaphore:
			req = session or (self.sessions.next() if ssl else self.nossl)
			resp = await req.request(method.upper(), url, headers=headers, data=data, timeout=timeout)
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

	def __call__(self, url, headers=None, files=None, data=None, raw=False, timeout=8, method="get", decode=False, json=False, bypass=True, proxy=False, aio=False, session=None, ssl=True, authorise=False) -> bytes | str | json_like:
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
			if aio:
				session = None
			else:
				session = requests
		elif bypass:
			if "user-agent" not in headers and "User-Agent" not in headers:
				headers["User-Agent"] = f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
				headers["X-Forwarded-For"] = ".".join(str(random.randint(1, 254)) for _ in loop(4))
			headers["DNT"] = "1"
		method = method.casefold()
		if aio:
			return csubmit(asyncio.wait_for(self.aio_call(url, headers, files, data, method, decode, json, session, ssl, timeout=timeout), timeout=timeout))
		with self.semaphore:
			if proxy:
				data = proxy_download(url)
				if json:
					return orjson.loads(data)
				if decode:
					return as_str(data)
				return data
			if session:
				req = session
				resp = req.request(method.upper(), url, headers=headers, files=files, data=data, timeout=timeout, verify=ssl)
			elif bypass:
				req = reqs.next()
				resp = req.request(method.upper(), url, headers=headers, files=files, data=data, timeout=timeout, verify=ssl)
			else:
				req = requests
				resp = getattr(req, method)(url, headers=headers, files=files, data=data, timeout=timeout, verify=ssl)
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

sp = None
browsers = {}
def new_playwright_page(browser="firefox", viewport=dict(width=480, height=320)):
	try:
		return browsers[browser].new_page()
	except KeyError:
		if sp is None:
			from playwright.sync_api import sync_playwright
			globals()["sp"] = sync_playwright().start()
		browsers[browser] = getattr(sp, browser).launch(headless=True)
	return browsers[browser].new_page(viewport=viewport)

esubmit(load_mimes)