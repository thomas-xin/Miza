if __name__ != "__mp_main__":
	import smath
	from smath import *

with open("auth.json", "rb") as f:
	AUTH = cdict(eval(f.read()))
cachedir = AUTH.get("cache_path") or None
if cachedir:
	print(f"Setting model cache {cachedir}...")
	os.environ["HF_HOME"] = f"{cachedir}/huggingface"
	os.environ["TORCH_HOME"] = f"{cachedir}/torch"
	os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cachedir}/huggingface/hub"
	os.environ["TRANSFORMERS_CACHE"] = f"{cachedir}/huggingface/transformers"
	os.environ["HF_DATASETS_CACHE"] = f"{cachedir}/huggingface/datasets"
else:
	cachedir = os.path.expanduser("~") + "/.cache"
	if not os.path.exists(cachedir):
		os.mkdir(cachedir)

common_modules = (
	"asyncio",
	"psutil",
	"subprocess",
	"weakref",
	"tracemalloc",
	"zipfile",
	"urllib",
	"nacl",
	"discord",
	"json",
	"functools",
	"orjson",
	"aiohttp",
	"threading",
	"shutil",
	"filetype",
	"inspect",
	"sqlite3",
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
MultiAutoImporter(
	*common_modules,
	pool=import_exc,
	_globals=globals(),
)
if __name__ == "__lint__":
	import psutil, subprocess, weakref, tracemalloc, zipfile, urllib, nacl,discord, asyncio, json, functools, orjson, aiohttp, threading, shutil, filetype, inspect, sqlite3

PROC = psutil.Process()
quit = lambda *args, **kwargs: force_kill(PROC)
BOT = [None]

tracemalloc.start()

from zipfile import ZipFile
import urllib.request, urllib.parse
import nacl.secret

DC = 0
torch = None
if __name__ != "__mp_main__":
	import pynvml
	try:
		pynvml.nvmlInit()
		DC = pynvml.nvmlDeviceGetCount()
		if not os.environ.get("AI_FEATURES", True):
			raise StopIteration("AI features disabled.")
		import torch
	except:
		pass
hwaccel = "cuda" if DC else "d3d11va" if os.name == "nt" else "auto"

utils = discord.utils
reqs = alist(requests.Session() for i in range(6))
url_parse = urllib.parse.quote_plus
url_unparse = urllib.parse.unquote_plus
escape_markdown = utils.escape_markdown
escape_mentions = utils.escape_mentions
escape_everyone = lambda s: s#s.replace("@everyone", "@\xadeveryone").replace("@here", "@\xadhere")
escape_roles = lambda s: s#escape_everyone(s).replace("<@&", "<@\xad&")

DISCORD_EPOCH = 1420070400000 # 1 Jan 2015
MIZA_EPOCH = 1577797200000 # 1 Jan 2020

time_snowflake = lambda dt, high=None: utils.time_snowflake(dt, high=high) if type(dt) is not int else getattr(dt, "id", None) or dt

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
	if type(i) is int:
		return utc_dft(id2ts(i))
	return i

snowflake_time_2 = lambda id: datetime.datetime.fromtimestamp(id2ts(id))
snowflake_time_3 = utils.snowflake_time

ip2int = lambda ip: int.from_bytes(b"\x00" + bytes(int(i) for i in ip.split(".")), "big")

api = "v10"

# Main event loop for all asyncio operations.
try:
	eloop = asyncio.get_event_loop()
except:
	eloop = asyncio.new_event_loop()
__setloop__ = lambda: asyncio.set_event_loop(eloop)
__setloop__()

emptyfut = fut_nop = asyncio.Future(loop=eloop)
fut_nop.set_result(None)
Future = concurrent.futures.Future
newfut = nullfut = Future()
newfut.set_result(None)

def as_fut(obj):
	if obj is None:
		return emptyfut
	fut = asyncio.Future()
	eloop.call_soon_threadsafe(fut.set_result, obj)
	return fut


class EmptyContext(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	__enter__ = lambda self, *args: self
	__exit__ = lambda *args: None
	__aenter__ = lambda self, *args: as_fut(self)
	__aexit__ = lambda *args: emptyfut
	__call__ = lambda self, *args: self

emptyctx = EmptyContext()


SEMS = {}

# Manages concurrency limits, similar to asyncio.Semaphore, but has a secondary threshold for enqueued tasks, as well as an optional rate limiter.
class Semaphore(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

	__slots__ = ("limit", "buffer", "fut", "active", "passive", "rate_limit", "rate_bin", "last", "trace", "weak")

	def __init__(self, limit=256, buffer=32, delay=0.05, rate_limit=None, sync=False, randomize_ratio=2, last=False, trace=False, weak=False):
		self.limit = limit
		self.buffer = buffer
		self.active = 0
		self.passive = 0
		self.rate_limit = rate_limit
		self.rate_bin = deque()
		self.fut = Future()
		self.fut.set_result(None)
		self.last = last
		self.trace = trace and inspect.stack()[1]
		self.weak = weak
		self.sync = sync
		if rate_limit and sync:
			self.delay_for(rate_limit - 1)

	def __str__(self):
		classname = str(self.__class__).replace("'>", "")
		classname = classname[classname.index("'") + 1:]
		s = f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: {self.active}/{self.limit}, {self.passive}/{self.buffer}"
		if self.rate_limit:
			s += f", {round(self.reset_after, 1)}/{self.rate_limit}"
		return s

	@property
	def reset_after(self):
		if not self.rate_limit or not self.rate_bin:
			return 0
		t = time.time()
		if t - self.rate_bin[0] <= self.rate_limit:
			return self.rate_limit - (t - self.rate_bin[0])
		return 0

	async def _update_bin_after_a(self, t):
		await asyncio.sleep(t)
		self._update_bin()

	def _update_bin_after(self, t):
		time.sleep(t)
		self._update_bin()

	def _update_bin(self):
		if self.rate_limit:
			try:
				if self.last:
					if self.rate_bin and time.time() - self.rate_bin[-1] >= self.rate_limit:
						self.rate_bin.clear()
				else:
					while self.rate_bin and time.time() - self.rate_bin[0] >= self.rate_limit:
						self.rate_bin.popleft()
			except IndexError:
				pass
			if len(self.rate_bin) < self.limit:
				try:
					self.fut.set_result(None)
				except concurrent.futures.InvalidStateError:
					pass
		if self.weak and not self.rate_bin:
			SEMS.pop(id(self), None)
		return self.rate_bin

	def delay_for(self, seconds=0):
		t = time.time() + seconds
		if self.sync and self.rate_limit:
			t -= t % self.rate_limit
		for i in range(self.limit):
			self.rate_bin.append(t)
		for i in range(len(self.rate_bin) - self.limit):
			self.rate_bin.popleft()
		return self

	def enter(self):
		if self.trace:
			self.trace = inspect.stack()[2]
		self.active += 1
		if self.rate_limit:
			t = time.time()
			if self.sync and self.rate_limit:
				t -= t % self.rate_limit
			self._update_bin().append(t)
			if self.weak:
				SEMS[id(self)] = self
		return self

	def check_overflow(self):
		if self.is_full():
			raise SemaphoreOverflowError(f"Semaphore object of limit {self.limit} overloaded by {self.passive}")

	def __enter__(self):
		if self.is_busy():
			self.check_overflow()
			self.passive += 1
			while self.is_busy():
				if self.fut.done():
					time.sleep(0.08)
				else:
					self.fut.result()
			self.passive -= 1
		return self.enter()

	def __exit__(self, *args):
		self.active = max(0, self.active - 1)
		if self.rate_bin:
			t = self.rate_bin[0 - self.last] + self.rate_limit - time.time()
			if t > 0:
				if get_event_loop().is_running():
					csubmit(self._update_bin_after_a(t))
				else:
					esubmit(self._update_bin_after, t)
			else:
				self._update_bin()
		elif self.active < self.limit:
			try:
				self.fut.set_result(None)
			except concurrent.futures.InvalidStateError:
				pass

	async def __aenter__(self):
		if self.is_busy():
			self.check_overflow()
			self.passive += 1
			while self.is_busy():
				if self.fut.done():
					await asyncio.sleep(0.08)
				else:
					await wrap_future(self.fut)
			self.passive -= 1
		self.enter()
		return self

	def __aexit__(self, *args):
		self.__exit__()
		return emptyfut

	def wait(self):
		while self.is_busy():
			if self.fut.done():
				time.sleep(0.08)
			else:
				self.fut.result()

	async def __call__(self):
		while self.is_busy():
			if self.fut.done():
				await asyncio.sleep(0.08)
			else:
				await wrap_future(self.fut)
	
	acquire = __call__

	def finish(self):
		while self.active or self.is_busy():
			if self.fut.done():
				time.sleep(0.08)
			else:
				self.fut.result()

	async def afinish(self):
		while self.active or self.is_busy():
			if self.fut.done():
				await asyncio.sleep(0.08)
			else:
				await wrap_future(self.fut)

	def is_active(self):
		return self.active or self.passive

	def is_busy(self):
		return self.rate_limit and len(self._update_bin()) >= self.limit or self.active >= self.limit

	def is_full(self):
		return self.passive >= self.buffer

	def clear(self):
		self.rate_bin.clear()
		self._update_bin()

	@property
	def full(self):
		return self.is_full()

	@property
	def busy(self):
		return self.is_busy()

	@property
	def free(self):
		return not self.is_busy()

	def is_inactive(self):
		return not self.is_busy()

class SemaphoreOverflowError(RuntimeError):
	__slots__ = ()


# A context manager that sends exception tracebacks to stdout.
class TracebackSuppressor(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

	def __init__(self, *args, fn=print_exc, **kwargs):
		self.fn = fn
		self.exceptions = args + tuple(kwargs.values())

	__enter__ = lambda self: self
	def __exit__(self, exc_type, exc_value, exc_tb):
		if exc_type and exc_value:
			for exception in self.exceptions:
				if issubclass(type(exc_value), exception):
					return True
			self.fn()
		return True

	__aenter__ = lambda self: emptyfut
	def __aexit__(self, *args):
		return as_fut(self.__exit__(*args))

	def __call__(self, *ins, default=None):
		if len(ins) == 1 and callable(ins[0]) and (not isinstance(ins[0], type) or not issubclass(ins[0], BaseException)):
			def decorator(*args, **kwargs):
				with self:
					return ins[0](*args, **kwargs)
				return default
			return decorator
		return self.__class__(*ins)

tracebacksuppressor = TracebackSuppressor()


# A context manager that delays the return of a function call.
class Delay(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

	def __init__(self, duration=0):
		self.duration = duration
		self.start = utc()

	def __call__(self):
		return self.exit()

	__enter__ = lambda self: self
	def __exit__(self, *args):
		remaining = self.duration - utc() + self.start
		if remaining > 0:
			time.sleep(remaining)

	async def __aexit__(self, *args):
		remaining = self.duration - utc() + self.start
		if remaining > 0:
			await asyncio.sleep(remaining)


# A context manager that monitors the amount of time taken for a designated section of code.
class MemoryTimer(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

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
			self.timers[self.name] = t = deque(maxlen=8)
			t.append(taken)

	__aenter__ = lambda self: emptyfut
	def __aexit__(self, *args):
		self.__exit__()
		return emptyfut


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
			time.sleep(delay)

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
			await asyncio.sleep(delay)
		delay += delay / 2


# For compatibility with versions of asyncio and concurrent.futures that have the exceptions stored in a different module
T0 = TimeoutError
try:
	T1 = asyncio.exceptions.TimeoutError
except AttributeError:
	try:
		T1 = asyncio.TimeoutError
	except AttributeError:
		T1 = TimeoutError
try:
	T2 = concurrent.futures._base.TimeoutError
except AttributeError:
	try:
		T2 = concurrent.futures.TimeoutError
	except AttributeError:
		T2 = TimeoutError

try:
	ISE = asyncio.exceptions.InvalidStateError
except AttributeError:
	ISE = asyncio.InvalidStateError
try:
	CE = asyncio.exceptions.CancelledError
except AttributeError:
	CE = asyncio.CancelledError
try:
	CE2 = concurrent.futures._base.CancelledError
except AttributeError:
	CE2 = concurrent.futures.CancelledError


class ArgumentError(LookupError):
	__slots__ = ()

class TooManyRequests(PermissionError):
	__slots__ = ()

class CommandCancelledError(Exception):
	__slots__ = ()

AE = ArgumentError
TMR = TooManyRequests
CCE = CommandCancelledError


python = sys.executable

class MultiEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (set, frozenset, alist, np.ndarray)):
			return list(obj)
		return json.JSONEncoder.default(self, obj)


enc_key = None
with tracebacksuppressor:
	enc_key = AUTH["encryption_key"]

if not enc_key:
	enc_key = (AUTH.get("discord_secret") or AUTH.get("discord_token") or as_str(base64.b64encode(randbytes(32)).rstrip(b"="))).replace(".", "A").replace("_", "a").replace("-", "a")[:43]
	while len(enc_key) < 43:
		enc_key += "A"
	AUTH["encryption_key"] = enc_key 
	with open("auth.json", "w", encoding="utf-8") as f:
		json.dump(AUTH, f, indent="\t")

if os.environ.get("AI_FEATURES", True):
	import openai

enc_key += "=="
enc_box = nacl.secret.SecretBox(base64.b64decode(enc_key)[:32])

def encrypt(s): 
	if type(s) not in (bytes, memoryview):
		s = str(s).encode("utf-8")
	return b">~MIZA~>" + enc_box.encrypt(s)
def decrypt(s):
	if type(s) not in (bytes, memoryview):
		s = str(s).encode("utf-8")
	if s[:8] == b">~MIZA~>":
		return enc_box.decrypt(s[8:])
	raise ValueError("Data header not found.")

PORT = AUTH.get("webserver_port", 80)
if PORT:
	PORT = int(PORT)
IND = "\x7f"


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

def bytes2zip(data, lzma=False):
	if lzma:
		import lzma
		return b"~" + lzma.compress(data)
	import zlib
	return b"!" + zlib.compress(data)
	# b = io.BytesIO()
	# ctype = zipfile.ZIP_LZMA if lzma else zipfile.ZIP_DEFLATED
	# with ZipFile(b, "w", compression=ctype, allowZip64=True) as z:
	#     z.writestr("D", data=data)
	# return b.getbuffer()


# Safer than raw eval, more powerful than json.loads
eval_json = lambda s: copy.deepcopy(_eval_json(s))
@functools.lru_cache(maxsize=2)
def _eval_json(s):
	if type(s) is memoryview:
		s = bytes(s)
	try:
		return orjson.loads(s)
	except:
		pass
		try:
			return safe_eval(s)
		except:
			pass
		raise

@functools.lru_cache(maxsize=8)
def select_and_loads(s, mode="safe", size=None):
	if not s:
		raise ValueError("Data must not be empty.")
	if size and size < len(s):
		raise OverflowError("Data input size too large.")
	if type(s) is str:
		s = s.encode("utf-8")
	if mode != "unsafe":
		try:
			s = decrypt(s)
		except ValueError:
			pass
		except:
			raise
	if s[:1] in (b"~", b"!") or zipfile.is_zipfile(io.BytesIO(s)):
		s = zip2bytes(s)
	# b = io.BytesIO(s)
	# if zipfile.is_zipfile(b):
	#     if len(s) > 1048576:
	#         print(f"Loading zip file of size {len(s)}...")
	#     b.seek(0)
	#     with ZipFile(b, allowZip64=True, strict_timestamps=False) as z:
	#         n = z.namelist()[0]
	#         if size:
	#             x = z.getinfo(n).file_size
	#             if size < x:
	#                 raise OverflowError(f"Data input size too large ({x} > {size}).")
	#         s = z.read(n)
	data = None
	if not s:
		return data
	with tracebacksuppressor:
		if s[0] == 128:
			return pickle.loads(s)
	if data is None:
		# if mode == "unsafe":
		#     if not s:
		#         raise FileNotFoundError
		#     data = eval(compile(s, "<loader>", "eval", optimize=2, dont_inherit=False))
		# else:
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
	if mode == "unsafe":
		try:
			if len(data) and isinstance(data, dict) and not isinstance(next(iter(data)), str):
				raise TypeError
			if isinstance(data, (set, frozenset)):
				s = b"$" + orjson.dumps(list(data))
			else:
				s = orjson.dumps(data)
		except TypeError:
			s = pickle.dumps(data)
		if len(s) > 32768 and compress:
			t = bytes2zip(s, lzma=len(s) > 16777216)
			if len(t) < len(s) * 0.9:
				s = t
		return s
	try:
		s = orjson.dumps(data)
	except:
		s = None
	if len(s) > 262144:
		t = bytes2zip(s, lzma=False)
		if len(t) < len(s) * 0.9:
			s = t
	return s


def nop2(s):
	return s

class FileHashDict(collections.abc.MutableMapping):

	sem = Semaphore(64, 128, 0.3, 1)
	cache_size = 256
	encoder = [nop2, nop2]

	def __init__(self, *args, path="", encode=None, decode=None, **kwargs):
		if not kwargs and len(args) == 1:
			self.data = args[0]
		else:
			self.data = dict(*args, **kwargs)
		if encode:
			self.encoder[0] = encode
		if decode:
			self.encoder[1] = decode
		self.path = path.rstrip("/")
		self.modified = set()
		self.deleted = set()
		self.iter = None
		if self.path and not os.path.exists(self.path):
			os.mkdir(self.path)
			self.iter = alist()
		self.load_cursor()
		self.db_sem = Semaphore(1, 64)
		self.c_updated = False

	@property
	def encode(self):
		return self.encoder[0]

	@property
	def decode(self):
		return self.encoder[1]

	db = None
	def load_cursor(self):
		if self.db:
			self.db.close()
			self.db = None
		self.db = sqlite3.connect(f"{self.path}/~~", check_same_thread=False)
		self.cur = self.db.cursor()
		self.cur.execute(f"CREATE TABLE IF NOT EXISTS '{self.path}' (key VARCHAR(256) PRIMARY KEY, value BLOB)")
		self.codb = set(try_int(r[0]) for r in self.cur.execute(f"SELECT key FROM '{self.path}'") if r)
		return self.codb

	__hash__ = lambda self: lambda self: hash(self.path)
	__str__ = lambda self: self.__class__.__name__ + "(" + str(self.data) + ")"
	__repr__ = lambda self: self.__class__.__name__ + "(" + str(self.full) + ")"
	__call__ = lambda self, k: self.__getitem__(k)
	__len__ = lambda self: len(self.keys())
	__contains__ = lambda self, k: k in self.keys().to_frozenset()
	__eq__ = lambda self, other: self.data == other
	__ne__ = lambda self, other: self.data != other

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

	def key_path(self, k):
		return f"{self.path}/{k}"

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
			gen = set(try_int(self.demap(i)) for i in os.listdir(self.path) if not i.endswith("\x7f"))
			if self.modified:
				gen.update(self.modified)
			if self.deleted:
				gen.difference_update(self.deleted)
			if self.codb:
				gen.update(self.codb)
			self.iter = alist(i for i in gen if not isinstance(i, str) or not i.startswith("~"))
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
		with suppress(KeyError):
			return self.data[k]
		fn = self.key_path(k)
		if not os.path.exists(fn):
			if k != "~":
				if k in self.codb:
					with self.db_sem:
						s = next(self.cur.execute(f"SELECT value FROM '{self.path}' WHERE key=?", (k,)))[0]
					if s:
						with tracebacksuppressor:
							data = select_and_loads(self.decode(s), mode="unsafe")
							self.data[k] = data
							return data
				elif isinstance(k, str) and k.startswith("~"):
					raise TypeError("Attempted to load SQL database inappropriately")
			self.deleted.add(k)
			raise KeyError(k)
		with self.sem:
			with open(fn, "rb") as f:
				s = f.read()
		data = BaseException
		with tracebacksuppressor:
			data = select_and_loads(self.decode(s), mode="unsafe")
		# if data is BaseException:
		# 	fn = fn.rstrip("\x7f")
		# 	backup = AUTH.get("backup_path") or "backup"
		# 	for file in sorted(os.listdir(backup), reverse=True):
		# 		with tracebacksuppressor:
		# 			if file.endswith(".wb"):
		# 				if ":" not in backup:
		# 					backup = "../" + backup
		# 				s = subprocess.check_output([sys.executable, "neutrino.py", backup + "/" + file, "-f", fn.split("/", 1)[-1]], cwd="misc")
		# 			else:
		# 				with zipfile.ZipFile(backup + "/" + file, allowZip64=True, strict_timestamps=False) as z:
		# 					s = z.read(fn)
		# 			data = select_and_loads(self.decode(s), mode="unsafe")
		# 			self.modified.add(k)
		# 			self.iter = None
		# 			print(f"Successfully recovered backup of {fn} from {file}.")
		# 			break
		if data is BaseException:
			# self.deleted.add(k)
			raise KeyError(k)
		self.data[k] = data
		return data

	def __setitem__(self, k, v):
		with suppress(TypeError, ValueError):
			k = int(k)
		if k not in self:
			self.iter = None
		self.deleted.discard(k)
		self.data[k] = v
		self.modified.add(k)

	def get(self, k, default=None):
		with suppress(KeyError):
			return self[k]
		return default

	def pop(self, k, *args, force=False, remove=True):
		fn = self.key_path(k)
		try:
			if remove:
				if k in self.codb:
					with self.db_sem:
						self.cur.execute(f"DELETE FROM '{self.path}' WHERE key=?", (k,))
					self.codb.discard(k)
					self.c_updated = True
				self.deleted.add(k)
			if force:
				out = self[k]
				return self.data.pop(k, out)
			return self.data.pop(k, None)
		except KeyError:
			if not os.path.exists(fn):
				if args:
					return args[0]
				raise
			self.deleted.add(k)
			if args:
				return self.data.pop(k, args[0])
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

	def update(self, other):
		self.modified.update(other)
		self.deleted.difference_update(other)
		self.data.update(other)
		self.iter = None
		return self

	def fill(self, other):
		if not other:
			return self.clear()
		self.iter = None
		self.modified.update(other)
		self.deleted.difference_update(other)
		self.data.update(other)
		temp = set(self)
		temp.difference_update(other)
		self.deleted.update(temp)
		return self

	def clear(self):
		self.iter = None
		self.modified.clear()
		self.deleted.clear()
		self.data.clear()
		with self.db_sem:
			self.cur.execute(f"DELETE FROM '{self.path}'")
			self.db.commit()
			self.db.close()
			self.db = None
			self.codb.clear()
			try:
				shutil.rmtree(self.path)
			except (PermissionError, FileNotFoundError):
				pass
			else:
				os.mkdir(self.path)
			self.load_cursor()
		return self

	def __update__(self):
		# print("DATABASE:", self.path)
		modified = set(self.modified)
		self.modified.clear()
		deleted = set(self.deleted)
		self.deleted.clear()
		inter = modified.intersection(deleted)
		modified.difference_update(deleted)
		if modified or deleted:
			self.iter = None
		ndel = deleted.intersection(self.codb)
		if ndel:
			with self.db_sem:
				for k in ndel:
					self.cur.execute(f"DELETE FROM '{self.path}' WHERE key=?", (k,))
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
				futs.append((k, esubmit(select_and_dumps, d, mode="unsafe", compress=True, priority=2)))
			fut2 = []
			for k, fut in futs:
				b = fut.result()
				if self.encode is nop2:
					fut2.append((k, esubmit(self.encode, b, priority=1)))
				else:
					fut2.append((k, esubmit(self.encode, b, priority=2)))
			for k, fut in fut2:
				s = fut.result()
				mods[k] = s
			with self.db_sem:
				for k, d in mods.items():
					self.cur.execute(
						f"INSERT INTO '{self.path}' ('key', 'value') VALUES ('{k}', ?) ON CONFLICT(key) DO UPDATE SET 'value' = ?",
						[d, d],
					)
			self.c_updated = True
			deleted.update(modified)
		if self.c_updated:
			self.c_updated = False
			with self.db_sem:
				self.db.commit()
				self.codb = set(try_int(r[0]) for r in self.cur.execute(f"SELECT key FROM '{self.path}'") if r)
		# if any(not f.path.rsplit("/", 1)[-1].startswith("~") for f in os.scandir(self.path)):
			# for k in deleted:
				# self.data.pop(k, None)
				# fn = self.key_path(k)
				# with suppress(OSError):
					# with suppress(FileNotFoundError):
						# os.remove(fn)
					# with suppress(FileNotFoundError):
						# os.remove(fn + "\x7f")
					# with suppress(FileNotFoundError):
						# os.remove(fn + "\x7f\x7f")
			# t = utc()
			# for fn in (f.path for f in os.scandir(self.path) if f.name.endswith("\x7f") and t - f.stat().st_mtime > 3600):
				# with suppress(FileNotFoundError, PermissionError):
					# os.remove(fn)
		if len(self.data) > self.cache_size * 2:
			self.data.clear()
		else:
			while len(self.data) > self.cache_size:
				with suppress(RuntimeError):
					self.data.pop(next(iter(self.data)))
		return inter

	def vacuum(self):
		with self.db_sem:
			self.db.commit()
			self.db.close()
			self.db = None
			args = [python, "misc/vacuum.py", f"{self.path}/~~"]
			subprocess.run(args)
			return self.load_cursor()


def safe_save(fn, s):
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


# Decodes HTML encoded characters in a string.
@functools.lru_cache(maxsize=8)
def html_decode(s):
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


def restructure_buttons(buttons):
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
				button["label"] = lim_str(button["label"], 80)
			try:
				if type(button["emoji"]) is str:
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
			elif type(button["custom_id"]) is not str:
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
	return [dict(type=1, components=row) for row in buttons]


async def interaction_response(bot, message, content=None, embed=None, embeds=(), components=None, buttons=None, ephemeral=False):
	if getattr(message, "deferred", False):
		return interaction_patch(bot, message, content, embed, embeds, components, buttons, ephemeral)
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
		data=orjson.dumps(dict(
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
	# else:
	#     m = bot.GhostMessage()
	#     m.id = message.id
	#     m.content = content
	#     m.embeds = embeds
	#     m.ephemeral = ephemeral
	#     if getattr(message, "slash", False):
	#         m.slash = message.slash
	#     bot.add_message(message, files=False, force=True)
	return message

async def interaction_patch(bot, message, content=None, embed=None, embeds=(), components=None, buttons=None, ephemeral=False):
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
		data=orjson.dumps(dict(
			type=7,
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
no_md = lambda s: str(s).translate(__emap)
clr_md = lambda s: str(s).translate(__emap2)
sqr_md = lambda s: f"[{no_md(s)}]" if not isinstance(s, discord.abc.GuildChannel) else f"[#{no_md(s)}]"

def italics(s):
	if type(s) is not str:
		s = str(s)
	if "*" not in s:
		s = f"*{s}*"
	return s

def bold(s):
	if type(s) is not str:
		s = str(s)
	if "**" not in s:
		s = f"**{s}**"
	return s

single_md = lambda s: f"`{s}`"
code_md = lambda s: f"```\n{s}```" if s else "``` ```"
py_md = lambda s: f"```py\n{s}```" if s else "``` ```"
ini_md = lambda s: f"```ini\n{s}```" if s else "``` ```"
css_md = lambda s, force=False: (f"```css\n{s}```".replace("'", "\u2019").replace('"', "\u201d") if force else ini_md(s)) if s else "``` ```"
fix_md = lambda s: f"```fix\n{s}```" if s else "``` ```"

# Discord object mention formatting
user_mention = lambda u_id: f"<@{u_id}>"
user_pc_mention = lambda u_id: f"<@!{u_id}>"
channel_mention = lambda c_id: f"<#{c_id}>"
role_mention = lambda r_id: f"<@&{r_id}>"

channel_repr = lambda s: as_str(s) if not isinstance(s, discord.abc.GuildChannel) else str(s)


# Counts the number of lines in a file.
def line_count(fn):
	with open(fn, "r", encoding="utf-8") as f:
		data = f.read()
	return (len(data), data.count("\n") + 1)


def split_across(s, lim=2000, prefix=""):
	state = 0
	n = len(prefix)
	out = []
	while len(s) > lim - n:
		t = []
		while s:
			cl = sum(map(len, t))
			spl = s.split("\n\n", 1)
			if len(spl) > 1 and cl + len(spl[0]) < lim - n - 2:
				t.append(spl[0])
				t.append("\n\n")
				s = spl[1]
				state = 2
				continue
			if t and state >= 2:
				break
			spl = s.split("\n", 1)
			if len(spl) > 1 and cl + len(spl[0]) < lim - n - 2:
				t.append(spl[0])
				t.append("\n")
				s = spl[1]
				state = 1
				continue
			if t and state >= 1:
				break
			spl = s.split(None, 1)
			if len(spl) > 1 and cl + len(spl[0]) < lim - n - 2:
				t.append(spl[0])
				t.append(" ")
				s = spl[1]
				state = 0
				continue
			if t:
				break
			t.append(s[:lim - n - cl])
			s = s[lim - n - cl:]
		if prefix:
			t.insert(0, prefix)
		out.append("".join(t).strip())
	if s:
		out.append(s.strip())
	return out


# Checks if a file is a python code file using its filename extension.
is_code = lambda fn: str(fn).endswith(".py") or str(fn).endswith(".pyw")

def touch(file):
	with open(file, "ab"):
		pass

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


# Checks if an object can be used in "await" operations.
awaitable = lambda obj: hasattr(obj, "__await__") or issubclass(type(obj), asyncio.Future) or issubclass(type(obj), asyncio.Task) or inspect.isawaitable(obj)

# Async function that waits for a given time interval if the result of the input coroutine is None.
async def wait_on_none(coro, seconds=0.5):
	resp = await coro
	if resp is None:
		await asyncio.sleep(seconds)
	return resp


# Recursively iterates through an iterable finding coroutines and executing them.
async def recursive_coro(item):
	if not issubclass(type(item), collections.abc.MutableSequence):
		return item
	for i, obj in enumerate(item):
		if awaitable(obj):
			if not issubclass(type(obj), asyncio.Task):
				item[i] = create_task(obj)
		elif issubclass(type(obj), collections.abc.MutableSequence):
			item[i] = create_task(recursive_coro(obj))
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
	if getattr(reference, "slash", None):
		ephemeral = ephemeral and 64
		sem = emptyctx
		inter = True
		if getattr(reference, "deferred", False) or getattr(reference, "int_id", reference.id) in bot.inter_cache:
			url = f"https://discord.com/api/{api}/webhooks/{bot.id}/{bot.inter_cache.get(reference.id, reference.slash)}/messages/@original"
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
				if type(reference) is int:
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
			except (aiohttp.client_exceptions.ClientOSError, CE):
				await asyncio.sleep(random.random() * 2 + 1)
				if fields.get("files"):
					for file in fields["files"]:
						file.reset()
				return await channel.send(content, **fields)
		try:
			sem = REPLY_SEM[channel.id]
		except KeyError:
			sem = None
		if not sem:
			g_id = channel.guild.id if getattr(channel, "guild", None) else None
			bucket = f"{channel.id}:{g_id}:" + "/channels/{channel_id}/messages"
			try:
				try:
					sem = REPLY_SEM[channel.id]
				except KeyError:
					bucket = f"{channel.id}:None:" + "/channels/{channel_id}/messages"
					sem = REPLY_SEM[channel.id]
			except KeyError:
				# print_exc()
				sem = REPLY_SEM[channel.id] = Semaphore(5, buffer=256, delay=0.1, rate_limit=5.15)
		inter = False
		url = f"https://discord.com/api/{api}/channels/{channel.id}/messages"
		if getattr(channel, "dm_channel", None):
			channel = channel.dm_channel
		elif getattr(channel, "send", None) and getattr(channel, "guild", None) and not channel.permissions_for(channel.guild.me).read_message_history:
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
	body = orjson.dumps(data)
	exc = RuntimeError
	if bot:
		M = bot.ExtendedMessage.new
	else:
		M = discord.Message
	for i in range(xrand(3, 6)):
		try:
			method = "patch" if getattr(reference, "deferred", False) else "post"
			if method == "patch":
				url = f"https://discord.com/api/{api}/webhooks/{bot.id}/{reference.slash}/messages/@original"
				body = orjson.dumps(data["data"])
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
					value=orjson.dumps(data).decode("utf-8", "replace"),
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
					body = orjson.dumps(data["data"])
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

# Sends a message to a channel, then adds reactions accordingly.
async def send_with_react(channel, *args, reacts=None, reference=None, mention=False, **kwargs):
	try:
		if reference or "buttons" in kwargs or "embeds" in kwargs:
			sent = await send_with_reply(channel, reference, *args, mention=mention, **kwargs)
		elif getattr(channel, "simulated", False):
			sent = await channel.send(*args, **kwargs)
		else:
			sent = await discord.abc.Messageable.send(channel, *args, **kwargs)
		if reacts and not getattr(sent, "ephemeral", False):
			if len(reacts) > 5:
				for react in reacts:
					async with Delay(1):
						await aretry(sent.add_reaction, react)
			else:
				for react in reacts:
					async with Delay(0.5):
						create_task(aretry(sent.add_reaction, react))
		return sent
	except:
		print_exc()
		raise


voice_channels = lambda guild: [channel for channel in guild.channels if getattr(channel, "type", None) in (discord.ChannelType.voice, discord.ChannelType.stage_voice)]

def select_voice_channel(user, channel):
	# Attempt to match user's currently connected voice channel
	voice = user.voice
	member = user.guild.me
	if voice is None:
		# Otherwise attempt to find closest voice channel to current text channel
		catg = channel.category
		if catg is not None:
			channels = voice_channels(catg)
		else:
			channels = None
		if not channels:
			pos = 0 if channel.category is None else channel.category.position
			# Sort by distance from text channel
			channels = sorted(tuple(channel for channel in voice_channels(channel.guild) if channel.permissions_for(member).connect and channel.permissions_for(member).speak and channel.permissions_for(member).use_voice_activation), key=lambda channel: (abs(pos - (channel.position if channel.category is None else channel.category.position)), abs(channel.position)))
		if channels:
			vc = channels[0]
		else:
			raise LookupError("Unable to find voice channel.")
	else:
		vc = voice.channel
	return vc


# Creates and starts a coroutine for typing in a channel.
typing = lambda self: create_task(self.trigger_typing())


# Gets the string representation of a url object with the maximum allowed image size for discord, replacing png with webp format when possible.
def to_webp(url):
	if type(url) is not str:
		url = str(url)
	if url.startswith("https://cdn.discordapp.com/embed/avatars/"):
		return url.replace("/media.discordapp.net/", "/cdn.discordapp.com/").replace(".webp", ".png")
	if url.endswith("?size=1024"):
		url = url[:-10] + "?size=4096"
	if "/embed/" not in url[:48]:
		url = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
	return url.replace(".png", ".webp")

def to_webp_ex(url):
	if type(url) is not str:
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
	if BOT[0] and isinstance(obj, discord.Attachment):
		return BOT[0].try_attachment(obj.url)
	found = False
	for attr in ("display_avatar", "avatar_url", "icon_url", "icon", "avatar"):
		try:
			url = getattr(obj, attr)
		except AttributeError:
			continue
		found = True
		if url:
			return f(url)
	if found:
		return BASE_LOGO

# Finds the best URL for a discord object's icon, prioritizing proxy_url for images if applicable.
proxy_url = lambda obj: get_url(obj) or (obj.proxy_url if is_image(obj.proxy_url) else obj.url)
# Finds the best URL for a discord object's icon.
best_url = lambda obj: get_url(obj) or getattr(obj, "url", None) or BASE_LOGO
# Finds the worst URL for a discord object's icon.
worst_url = lambda obj: get_url(obj, to_webp_ex) or getattr(obj, "url", None) or BASE_LOGO

allow_gif = lambda url: url + ".gif" if "." not in url.rsplit("/", 1)[-1] and "?" not in url else url

def get_author(user, u_id=None):
	url = best_url(user)
	bot = BOT[0]
	if bot and "proxies" in bot.data:
		url2 = bot.data.proxies.get(uhash(url))
		if url2:
			url = url2
		else:
			bot.data.exec.cproxy(url)
	name = getattr(user, "display_name", None) or user.name
	if u_id:
		name = f"{name} ({user.id})"
	return cdict(name=name, icon_url=allow_gif(url), url=url)

# Finds emojis and user mentions in a string.
find_emojis = lambda s: regexp("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>").findall(s)
find_users = lambda s: regexp("<@!?[0-9]+>").findall(s)


def min_emoji(emoji):
	if not getattr(emoji, "id", None):
		if getattr(emoji, "name", None):
			return emoji.name
		emoji = as_str(emoji)
		if emoji.isnumeric():
			return f"<:_:{emoji}>"
		return emoji
	if emoji.animated:
		return f"<a:_:{emoji.id}>"
	return f"<:_:{emoji.id}>"


def get_random_emoji():
	d = [chr(c) for c in range(128512, 128568)]
	d.extend(chr(c) for c in range(128577, 128580))
	d.extend(chr(c) for c in range(129296, 129302))
	d.extend(chr(c) for c in range(129312, 129318))
	d.extend(chr(c) for c in range(129319, 129328))
	d.extend(chr(c) for c in range(129392, 129399))
	d.extend(chr(c) for c in (129303, 129400, 129402))
	return random.choice(d)


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
	g_id = getattr(guild, "id", 0)
	return f"https://discord.com/channels/{g_id}/{message.channel.id}/{message.id}"


# Applies stickers to a message based on its discord data.
def apply_stickers(message, data=None):
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
async def str_lookup(it, query, ikey=lambda x: [str(x)], qkey=lambda x: [str(x)], loose=True, fuzzy=0):
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
					match = fuzzy_substring(qlist[a], b)
					if match >= 1:
						return i
					elif match >= fuzzy and not match <= cache[a][0][0]:
						cache[a][0] = [match, i]
			elif fuzzy == 0:
				for a, b in enumerate(qkey(c)):
					if b == qlist[a]:
						return i
					elif b.startswith(qlist[a]):
						if not len(b) >= cache[a][0][0]:
							cache[a][0] = [len(b), i]
					elif loose and qlist[a] in b:
						if not len(b) >= cache[a][1][0]:
							cache[a][1] = [len(b), i]
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

def parse_colour(s, default=None):
	if s.startswith("0x"):
		s = s[2:].rstrip()
	else:
		s = single_space(s.replace("#", "").replace(",", " ")).strip()
	# Try to parse as colour tuple first
	if not s:
		if default is None:
			raise ArgumentError("Missing required colour argument.")
		return default
	try:
		return colour_names[full_prune(s).replace(" ", "_")]
	except KeyError:
		pass
	if " " in s:
		channels = [min(255, max(0, int(round(float(i.strip()))))) for i in s.split(" ")[:5] if i]
		if len(channels) not in (3, 4):
			raise ArgumentError("Please input 3 or 4 channels for colour input.")
	else:
		# Try to parse as hex colour value
		try:
			raw = int(s, 16)
			if len(s) <= 6:
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
	if type(obj) is int:
		return obj
	if type(obj) is str:
		with suppress(ValueError):
			return int(obj.translate(__itrans))
		return obj
	with suppress(AttributeError):
		return obj.recipient.id
	with suppress(AttributeError):
		return obj.id
	with suppress(AttributeError):
		return obj.value
	return int(obj)


# Strips <> characters from URLs.
def strip_acc(url):
	if url.startswith("<") and url[-1] == ">":
		s = url[1:-1]
		if is_url(s):
			return s
	return url

__smap = {"|": "", "*": ""}
__strans = "".maketrans(__smap)
verify_search = lambda f: strip_acc(single_space(f.strip().translate(__strans)))
# This reminds me of Perl - Smudge
find_urls = lambda url: url and regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+").findall(url)
is_url = lambda url: url and isinstance(url, (str, bytes)) and regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+$").fullmatch(url)
is_discord_url = lambda url: url and regexp("^https?:\\/\\/(?:[A-Za-z]{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/").findall(url) + regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").findall(url)
is_discord_attachment = lambda url: url and regexp("^https?:\\/\\/(?:[A-Za-z]{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/attachments\\/").search(str(url))
is_tenor_url = lambda url: url and regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[a-zA-Z0-9\\-_]+-[0-9]+").findall(url)
is_imgur_url = lambda url: url and regexp("^https?:\\/\\/(?:[A-Za-z]\\.)?imgur.com/[a-zA-Z0-9\\-_]+").findall(url)
is_giphy_url = lambda url: url and regexp("^https?:\\/\\/giphy.com/gifs/[a-zA-Z0-9\\-_]+").findall(url)
is_youtube_url = lambda url: url and regexp("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+").findall(url)
is_youtube_stream = lambda url: url and regexp("^https?:\\/\\/r+[0-9]+---.{2}-[A-Za-z0-9\\-_]{4,}\\.googlevideo\\.com").findall(url)
is_deviantart_url = lambda url: url and regexp("^https?:\\/\\/(?:www\\.)?deviantart\\.com\\/[^\\s<>`|\"']+").findall(url)
is_reddit_url = lambda url: url and regexp("^https?:\\/\\/(?:www\\.)?reddit.com\\/r\\/[^/]+\\/").findall(url)
is_emoji_url = lambda url: url and url.startswith("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/")
unyt = lambda s: re.sub(r"https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)|https?:\/\/(?:api\.)?mizabot\.xyz\/ytdl\?[vd]=(?:https:\/\/youtu\.be\/|https%3A%2F%2Fyoutu\.be%2F)", "https://youtu.be/", s)
COMM = "\\#$%"

def discord_expired(url):
	if is_discord_attachment(url):
		if "?ex=" not in url and "&ex=" not in url:
			return True
		temp = url.replace("?ex=", "&ex=").split("&ex=", 1)[-1].split("&", 1)[0]
		try:
			ts = int(temp, 16)
		except ValueError:
			return True
		return ts < utc() + 60

def expired(stream):
	if is_youtube_url(stream):
		return True
	if discord_expired(stream):
		return True
	if stream.startswith("https://www.yt-download.org/download/"):
		if int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60:
			return True
	elif is_youtube_stream(stream):
		if int(stream.replace("/", "=").split("expire=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 60:
			return True

def is_discord_message_link(url):
	check = url[:64]
	return "channels/" in check and "discord" in check

verify_url = lambda url: url if is_url(url) else url_parse(url)


def maps(funcs, *args, **kwargs):
	for func in funcs:
		yield func(*args, **kwargs)


# Checks if a URL contains a valid image extension, and removes it if possible.
IMAGE_FORMS = {
	".gif": True,
	".png": True,
	".bmp": False,
	".jpg": True,
	".jpeg": True,
	".tiff": False,
	".webp": True,
}
def is_image(url):
	if url:
		url = url.split("?", 1)[0]
		if "." in url:
			url = url[url.rindex("."):]
			url = url.casefold()
			return IMAGE_FORMS.get(url)

VIDEO_FORMS = {
	".ts": True,
	".webm": True,
	".mkv": True,
	".f4v": False,
	".flv": True,
	".ogv": True,
	".ogg": False,
	".gif": False,
	".gifv": True,
	".avi": True,
	".mov": True,
	".qt": True,
	".wmv": True,
	".mp4": True,
	".m4v": True,
	".mpg": True,
	".mpeg": True,
	".mpv": True,
}
def is_video(url):
	if "." in url:
		url = url[url.rindex("."):]
		url = url.casefold()
		return VIDEO_FORMS.get(url)

AUDIO_FORMS = {
	".mp3": True,
	".mp2": True,
	".ogg": True,
	".opus": True,
	".wav": True,
	".flac": True,
	".m4a": True,
	".aac": True,
	".wma": True,
	".vox": True,
	".ts": False,
	".webm": False,
	".mp4": False,
}
def is_audio(url):
	if url:
		url = url.split("?", 1)[0]
		if "." in url:
			url = url[url.rindex("."):]
			url = url.casefold()
			return AUDIO_FORMS.get(url)


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
	mimesplitter = globals()["mimesplitter"]
	for k, v in reversed(mimesplitter.items()):
		out = v.get(b[:k])
		if out:
			return out[mime]
	try:
		s = b.decode("utf-8")
	except UnicodeDecodeError:
		return "application/octet-stream" if mime else "bin"
	return "text/plain" if mime else "txt"


def from_file(path, mime=True):
	path = filetype.get_bytes(path)
	if mime:
		out = filetype.guess_mime(path)
	else:
		out = filetype.guess_extension(path)
	if out and out.split("/", 1)[-1] == "zip" and type(path) is str and path.endswith(".jar"):
		return "application/java-archive"
	if not out:
		if type(path) is not bytes:
			if type(path) is str:
				raise TypeError(path)
			path = bytes(path)
		out = simple_mimes(path, mime)
	if out == "application/octet-stream" and path.startswith(b'ECDC'):
		return "audio/ecdc"
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
		except:
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
				s = b.decode("utf-8")
			except:
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
PROC_RESP = {}#weakref.WeakValueDictionary()

# Gets amount of processes running in pool.
sub_count = lambda: sum(is_strict_running(p) for p in PROCS.values())

def is_strict_running(proc):
	if not proc:
		return
	try:
		if not proc.is_running():
			return False
		try:
			if proc.status() == "zombie":
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
		except:
			print_exc()
			return
	if not proc.is_running():
		return False
	try:
		if proc.status() == "zombie":
			proc.wait()
			return
	except (ProcessLookupError, psutil.NoSuchProcess):
		return
	return True

@tracebacksuppressor(psutil.NoSuchProcess)
def force_kill(proc):
	if not proc:
		return
	if getattr(proc, "fut", None) and not proc.fut.done():
		with tracebacksuppressor:
			proc.fut.set_exception(CCE("Response disconnected. If this error occurs during a command, it is likely due to maintenance!"))
	killed = deque()
	if not callable(getattr(proc, "children", None)):
		proc = psutil.Process(proc.pid)
	for child in proc.children(recursive=True):
		with suppress():
			child.terminate()
			killed.append(child)
			print(child, "killed.")
	proc.terminate()
	print(proc, "killed.")
	with tracebacksuppressor:
		_, alive = psutil.wait_procs(killed, timeout=2)
	for child in alive:
		with suppress():
			child.kill()
	try:
		proc.wait(timeout=2)
	except psutil.TimeoutExpired:
		proc.kill()

async def proc_communicate(proc):
	b = await proc.stdout.readline()
	if b == b"#R\n":
		return create_task(start_proc(proc))
	while proc:
		with tracebacksuppressor:
			if not is_strict_running(proc):
				return
			b = await proc.stdout.readline()
			if b == b"#R\n":
				return create_task(start_proc(proc))
			if not b:
				return
			# s = as_str(b.rstrip())
			# if s and s[0] == "~":
			#     c = as_str(evalEX(s[1:]))
			#     exec_tb(c, globals())
		s = b.rstrip()
		try:
			if s and s[:1] == b"$":
				s, r = s.split(b"~", 1)
				# print("PROC_RESP:", s, PROC_RESP.keys())
				d = {"_x": base64.b64decode(r)}
				c = evalex(memoryview(s)[1:], globals(), d)
				if isinstance(c, (str, bytes, memoryview)):
					exec_tb(c, globals(), d)
			elif s and s[:1] == b"~":
				c = evalex(memoryview(s)[1:], globals())
				if isinstance(c, (str, bytes, memoryview)):
					exec_tb(c, globals())
			else:
				print(lim_str(as_str(s), 262144))
		except:
			print_exc()
			print(lim_str(as_str(s), 262144))

async def proc_distribute(proc):
	bot = BOT[0]
	tasks = ()
	frozen = False
	while True:
		exc = None
		with tracebacksuppressor:
			if not is_strict_running(proc):
				return
			if not tasks:
				try:
					async with asyncio.timeout(60):
						await wrap_future(proc.fut)
				except (T0, T1, T2):
					pass
				except CCE as ex:
					exc = ex
				else:
					proc.fut = Future()
				tasks = bot.distribute(proc.caps, {}, {}, ip=f"127.0.0.1-{proc.n}")
				if not tasks:
					await asyncio.sleep(1 / 6)
					continue
			if exc:
				raise exc
			resps = {}
			futs = []
			while tasks or futs:
				if not is_strict_running(proc):
					return
				for task in tasks:
					i, cap, command, timeout = task
					# print("NEW TASK:", proc, i, bot.compute_wait, lim_str(str(command), 64), frand())
					if "nvram" in proc.caps:
						for p2 in PROCS.values():
							if p2 and p2.pid != proc.pid and p2.used and "vram" in p2.caps and set(p2.di).intersection(proc.di):
								if utc() - p2.used < 10:
									await asyncio.sleep(5)
								try:
									async with asyncio.timeout(timeout):
										await start_proc(p2, wait=True)
								except:
									create_task(asyncio.shield(start_proc(p2, wait=True, timeout=3600)))
									raise
					elif "vram" in proc.caps:
						for p2 in PROCS.values():
							if p2 and p2.pid != proc.pid and p2.used and "nvram" in p2.caps and set(p2.di).intersection(proc.di):
								if utc() - p2.used < 10:
									await asyncio.sleep(5)
								try:
									async with asyncio.timeout(timeout):
										await start_proc(p2, wait=True)
								except:
									create_task(asyncio.shield(start_proc(p2, wait=True, timeout=3600)))
									raise
					fut = create_task(_sub_submit(proc, command, _timeout=timeout))
					fut.ts = i
					futs.append(fut)
					if proc.used:
						proc.used = utc()
					if len(command) > 1 and command[1] == "&":
						with suppress():
							await fut
				# print(proc, tasks, [fut.ts for fut in futs], futs)
				futd = {i for i, fut in enumerate(futs) if fut.done()}
				for i in futd:
					fut = futs[i]
					while is_strict_running(proc):
						try:
							async with asyncio.timeout(6):
								resp = await asyncio.shield(fut)
						except Exception as ex:
							if not fut.done():
								continue
							resps[fut.ts] = ex
						else:
							resps[fut.ts] = resp
						break
					if not is_strict_running(proc):
						for i in futd:
							fut = futs[i]
							if fut.done():
								resps[fut.ts] = fut.result()
						break
					proc.used = utc()
				futs = [futs[i] for i in range(len(futs)) if i not in futd]
				if futs and not resps:
					delay = 1 + proc.sem.active
					with tracebacksuppressor(T1):
						async with asyncio.timeout(delay):
							await asyncio.shield(futs[0])
				if not is_strict_running(proc) or proc.sem.busy:
					caps = ()
				else:
					caps = proc.caps
				tasks = bot.distribute(caps, {}, resps, ip=f"127.0.0.1-{proc.n}")
				resps.clear()
		await asyncio.sleep(0.01)

proc_args = (python, "misc/x-compute.py")

COMPUTE_LOAD = AUTH.get("compute_load", [])
COMPUTE_POT = COMPUTE_LOAD.copy()
COMPUTE_ORDER = AUTH.get("compute_order", [])
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
		COMPUTE_LOAD = AUTH["compute_load"] = [i / total for i in COMPUTE_LOAD]
	if __name__ == "__main__":
		print("Compute load distribution:", COMPUTE_LOAD)
		print("Compute pool order:", COMPUTE_ORDER)

RESTARTING = {}
async def start_proc(n, di=(), caps="ytdl", it=0, wait=False, timeout=None):
	if hasattr(n, "caps"):
		n, di, caps, it = n.n, n.di, n.caps, it + 1
	if n in PROCS:
		proc = PROCS[n]
		if is_strict_running(proc):
			PROCS[n] = False
			RESTARTING[n] = proc
			it = max(it, proc.it + 1)
			if wait:
				with tracebacksuppressor:
					if timeout is None:
						await proc.sem.afinish()
					else:
						async with asyncio.timeout(timeout):
							await proc.sem.afinish()
				if timeout and proc.sem.active and utc() - proc.used < 60:
					raise CCE("Process schedule conflict.")
			await create_future(force_kill, proc)
		elif PROCS[n] is False:
			return
	args = list(proc_args)
	args.append(",".join(map(str, di)))
	args.append(",".join(caps))
	args.append(orjson.dumps(COMPUTE_LOAD).decode("ascii"))
	properties = [torch.cuda.get_device_properties(i) for i in range(DC)]
	args.append(orjson.dumps([(p.major, p.minor) for p in properties]))
	args.append(orjson.dumps(COMPUTE_ORDER).decode("ascii"))
	args.append(str(it))
	proc = await asyncio.create_subprocess_exec(
		*args,
		limit=1073741824,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=None,
	)
	if "load" in caps:
		return
	proc.n = n
	proc.di = di
	proc.caps = caps
	proc.it = it
	proc.is_running = lambda: not proc.returncode
	proc.sem = Semaphore(8, inf)
	proc.comm = create_task(proc_communicate(proc))
	proc.dist = create_task(proc_distribute(proc))
	proc.fut = newfut
	proc.used = 0
	PROCS[n] = proc
	return proc

IS_MAIN = True
FIRST_LOAD = True
# Spec requirements:
# ytdl			FFMPEG							anything with internet
# math			CPU >1							multithreading support
# image			FFMPEG, CPU >3, RAM >6GB		multiprocessing support
# browse		Windows, CPU >1, RAM >3GB		webdriver support
# caption		Tesseract, CPU >5, RAM >14GB	cpu inference
# video			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# ecdc			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# summ			GPU >200k, VRAM >4GB			GTX970, M60, GTX1050ti, P4, GTX1630
# sd			GPU >200k, VRAM >5GB			RTX2060, T4, RTX3050, RTX3060m, A16
# whisper		GPU >200k, VRAM >6GB			RTX2070, T4, RTX3060, A16, RTX4060
# sdxl			GPU >400k, VRAM >9GB			GTX1080ti, RTX2080ti, RTX3060, RTX3080, A2000
# sdxlr			GPU >400k, VRAM >15GB			V100, RTX3090, A4000, RTX4080, L4
# exl2			GPU >700k, VRAM >44GB			2xV100, 5xRTX3080, 2xRTX3090, A6000, A40, A100, 2xRTX4090, L6000, L40
def spec2cap():
	global FIRST_LOAD
	try:
		from multiprocessing import shared_memory
		globals()["MEM_LOCK"] = shared_memory.SharedMemory(name="X-DISTRIBUTE", create=True, size=1)
	except FileExistsError:
		if IS_MAIN:
			raise
		return
	caps = [[]]
	if not IS_MAIN:
		caps.append("remote")
	cc = psutil.cpu_count()
	mc = cc
	ram = psutil.virtual_memory().total
	try:
		subprocess.run("ffmpeg")
	except FileNotFoundError:
		ffmpeg = False
	else:
		ffmpeg = True
	try:
		subprocess.run("tesseract")
	except FileNotFoundError:
		tesseract = False
	else:
		tesseract = True
	if ffmpeg:
		caps.append("ytdl")
	done = []
	try:
		import pynvml
		pynvml.nvmlInit()
		handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
		rrams = [pynvml.nvmlDeviceGetMemoryInfo(d).total for d in handles]
	except:
		rrams = []
	vrams = tuple(rrams)
	cut = 0
	tdid = []
	if AUTH.get("discord_token") and any(v > 6 * 1073741824 and c > 700000 for v, c in zip(rrams, COMPUTE_POT)):
		vrs = [69]
		using = False
		for v in vrs:
			vram = sum(rrams[i] for i in range(DC) if COMPUTE_POT[i] > 400000)
			if vram > v * 1073741824:
				using = True
				cut = v * 1073741824
				did = []
				for i in COMPUTE_ORDER:
					vi = rrams[i]
					if vi < 2 * 1073741824 or (vi < v / 2 * 1073741824 and i in tdid):
						continue
					if cut > 0:
						red = min(cut, vi)
						rrams[i] -= red
						cut -= red
						did.append(i)
					else:
						break
				yield [did, "exl2", f"vr{v}", "vram"]
				tdid.extend(did)
				done.append("exl2")
		if using and FIRST_LOAD:
			FIRST_LOAD = False
			yield [[], "load", "exl2", "sdxlr"]
	if os.name == "nt" and ram > 3 * 1073741824:
		caps.append("browse")
	if len(caps) > 1:
		yield caps
		if cc > 1:
			yield caps
	rm = 1
	while mc > 1:
		caps = [[], "math"]
		if cc > 3 and ram > (rm * 8 - 2) * 1073741824 and ffmpeg:
			caps.append("image")
		if cc > 5 and ram > (rm * 32 - 2) * 1073741824 and tesseract:
			caps.append("caption")
		mc -= 1
		rm += 1
		yield caps
	if not DC:
		return
	for i, v in enumerate(rrams):
		c = COMPUTE_POT[i]
		caps = [[i]]
		if c > 100000 and v > 3 * 1073741824 and ffmpeg:
			caps.append("video")
			caps.append("ecdc")
		if c > 400000 and v > 15 * 1073741824:
			caps.append("sdxlr")
			caps.append("sdxl")
			# done.append("sdxlr")
			done.append("sdxl")
			v -= 15 * 1073741824
		elif c > 400000 and IS_MAIN and vrams[i] > 15 * 1073741824:
			caps.append("sdxlr")
			caps.append("sdxl")
			caps.append("nvram")
			if vrams[i] > 19 * 1073741824:
				caps.append("sd")
			# done.append("sdxlr")
			v -= 15 * 1073741824
		elif c > 400000 and v > 9 * 1073741824 and "sdxl" not in done:
			# if "sdxl" not in done or c <= 600000:
			caps.append("sdxl")
			# caps.append("sd")
			done.append("sdxl")
			v -= 9 * 1073741824
		elif c > 400000 and IS_MAIN and "sdxl" not in done and vrams[i] > 9 * 1073741824:
			caps.append("sdxl")
			caps.append("nvram")
			done.append("sdxl")
			v -= 9 * 1073741824
		if c > 200000 and v > 6 * 1073741824:
			if "whisper" not in done or c <= 600000:
				caps.append("whisper")
				done.append("whisper")
				v -= 7 * 1073741824
		if c > 200000 and v > 5 * 1073741824:
			if "sd" not in done or c <= 600000:
				caps.append("sd")
				done.append("sd")
				v -= 5 * 1073741824
		if c > 200000 and vrams[i] > 4 * 1073741824 and rrams[i] > 1073741824:
			caps.append("summ")
			done.append("summ")
			# v -= 1 * 1073741824
		# if v <= 4 * 1073741824:
			# v = 0
		# vrams[i] = v
		if i not in tdid and "nvram" in caps:
			caps.remove("nvram")
		if "sdxl" in caps and FIRST_LOAD:
			FIRST_LOAD = False
			yield [[], "load", "sdxlr"]
		if len(caps) > 1:
			yield caps

# print(list(spec2cap()))
# raise

def proc_start():
	if torch and os.environ.get("AI_FEATURES", True):
		globals()["DC"] = torch.cuda.device_count()
		COMPUTE_LOAD = AUTH.get("compute_load", [])
		if len(COMPUTE_LOAD) < DC:
			COMPUTE_LOAD = AUTH["compute_load"] = [torch.cuda.get_device_properties(i).multi_processor_count for i in range(torch.cuda.device_count())]
		elif len(COMPUTE_LOAD) > DC:
			COMPUTE_LOAD = COMPUTE_LOAD[:DC]
		if COMPUTE_LOAD:
			total = sum(COMPUTE_LOAD)
			if total != 1:
				COMPUTE_LOAD = AUTH["compute_load"] = [i / total for i in COMPUTE_LOAD]
			print("Compute load distribution:", COMPUTE_LOAD)
	else:
		COMPUTE_LOAD = ()
		globals()["DC"] = 0
	CAPS = globals().get("SCAP", [])
	if not CAPS:
		with tracebacksuppressor:
			CAPS = globals()["SCAP"] = list(spec2cap())
	print("CAPS:", CAPS)
	for n, (di, *caps) in enumerate(tuple(CAPS)):
		create_task(start_proc(n, di, caps))
		time.sleep(2)
		if "load" in caps:
			CAPS.pop(n)

def device_cap(i, resolve=False):
    di = torch.cuda.get_device_capability(i)
    if resolve:
        return 1.15 ** (di[0] * 10 + di[1])
    return di

last_task_time = 0
async def sub_submit(cap, command, _timeout=12, retries=1):
	t = utc()
	td = t - last_task_time
	globals()["last_task_time"] = t
	bot = BOT[0]
	ex2 = RuntimeError("Maximum compute attempts exceeded.")
	for i in range(round_random(retries) + 1):
		task = Future()
		task.cap = cap
		task.command = command
		task.timeout = _timeout
		queue = bot.compute_queue.setdefault(cap, set())
		queue.add(task)
		procs = filter(bool, PROCS.values())
		for proc in sorted(procs, key=lambda proc: (proc.sem.active, (0 in proc.di or "nvram" in proc.caps) and not proc.used and td < 60, -device_cap(proc.di[0], resolve=True) * COMPUTE_POT[proc.di[0]] if COMPUTE_POT and proc.di else random.random())):
			if not proc:
				continue
			if cap in proc.caps and not proc.fut.done():
				proc.fut.set_result(None)
		try:
			async with asyncio.timeout((_timeout or inf) + 1):
				return await wrap_future(task)
		except (T1, CE) as ex:
			task.cancel()
			queue.discard(task)
			ts = getattr(task, "ts", None)
			if ts:
				bot.compute_wait.pop(ts, None)
			elif isinstance(ex, CE):
				raise CCE(*ex.args)
			else:
				raise EnvironmentError(repr(ex))
			ex2 = ex
			print(lim_str((task, task.cap, task.command, task.timeout), 256))
			print_exc()
			await asyncio.sleep(i)
			continue
	raise ex2

def sub_kill(start=True, force=False):
	for p in PROCS.values():
		if is_strict_running(p):
			if not force:
				with tracebacksuppressor:
					p.sem.finish()
			force_kill(p)
	PROCS.clear()
	PROC_RESP.clear()
	bot = BOT[0]
	for k, v in bot.compute_wait.items():
		v.cancel()
	for k, v in bot.compute_queue.items():
		for w in v:
			w.cancel()
	bot.compute_wait.clear()
	bot.compute_queue.clear()
	if start:
		return proc_start()

lambdassert = "lambda:1+1"

async def _sub_submit(proc, command, _timeout=12):
	ts = ts_us()
	while ts in PROC_RESP:
		ts += 1
	PROC_RESP[ts] = fut = Future()
	comm = "[" + ",".join(map(repr, command[:2])) + "," + ",".join(map(str, command[2:])) + "]"
	s = f"~{ts}~".encode("ascii") + base64.b64encode(comm.encode("utf-8")) + b"\n"
	sem = proc.sem
	await sem()
	async with sem:
		proc.last_task = s
		try:
			proc.stdin.write(s)
			await proc.stdin.drain()
			fut = PROC_RESP[ts]
			tries = ceil(_timeout / 3) if _timeout and is_finite(_timeout) else 3600
			ex2 = None
			for i in range(tries):
				if not is_strict_running(proc):
					raise CCE("Retrying as process disappeared.")
				if ts not in PROC_RESP:
					raise CCE("Response disconnected. If this error occurs during a command, it is likely due to maintenance!")
				try:
					async with asyncio.timeout(min(4, i + 1)):
						resp = await wrap_future(fut)
				except T1 as ex:
					if command[0] == lambdassert and command[1] in "$&" and command[2] in ("[]", "()"):
						raise StopIteration("Temporary wait cancelled.")
					if i >= tries - 1:
						ex2 = ex
						break
				except Exception as ex:
					ex2 = ex
					break
				else:
					break
			else:
				raise OSError("Max waits exceeded.")
			if ex2:
				raise ex2
		except (BrokenPipeError, OSError, RuntimeError) as ex:
			with suppress(ISE):
				fut.set_exception(ex)
			if isinstance(ex, RuntimeError):
				if "OutOfMemoryError" in repr(ex):
					pass
				else:
					raise
			print("Killing process", lim_str((proc, command, ex), 384))
			create_task(start_proc(proc))
			raise
		except StopIteration as ex:
			with suppress(ISE):
				fut.set_exception(ex)
			raise TimeoutError(*ex.args)
		finally:
			PROC_RESP.pop(ts, None)
	return resp

SUB_WAITING = None


# Sends an operation to the math subprocess pool.
def process_math(expr, prec=64, rat=False, timeout=12, variables=None, retries=0):
	return sub_submit("math", (expr, "%", prec, rat, variables), _timeout=timeout, retries=retries)

# Sends an operation to the image subprocess pool.
def process_image(image, operation="$", args=[], cap="image", timeout=36, retries=1):
	args = astype(args, list)
	for i, a in enumerate(args):
		if type(a) is mpf:
			args[i] = float(a)
		elif type(a) in (list, deque, np.ndarray, dict):
			try:
				args[i] = "orjson.loads(" + as_str(orjson.dumps(as_str(orjson.dumps(a)))) + ")"
			except (TypeError, orjson.JSONDecodeError):
				args[i] = "pickle.loads(" + repr(pickle.dumps(a)) + ")"

	def as_arg(arg):
		if isinstance(arg, str) and (arg.startswith("pickle.loads(") or arg.startswith("orjson.loads(")):
			return arg
		return repr(arg)

	command = "[" + ",".join(map(as_arg, args)) + "]"
	return sub_submit(cap, (image, operation, command), _timeout=timeout, retries=retries)


def evalex(exc, g=None, l=None):
	try:
		ex = eval(exc, g, l)
	except (SyntaxError, NameError):
		exc = as_str(exc)
		s = exc[exc.index("(") + 1:exc.rindex(")")]
		with suppress(TypeError, SyntaxError, ValueError):
			s = ast.literal_eval(s)
		s = lim_str(s, 4096)
		ex = RuntimeError(s)
		if exc.startswith("PROC_RESP["):
			ex = eval(exc.split("(", 1)[0] + f"({repr(ex)})", g, l)
	return ex

# Evaluates an an expression, raising it if it is an exception.
def evalEX(exc):
	ex = evalex(exc)
	if issubclass(type(ex), BaseException):
		raise ex
	return ex


ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor

# Thread pool manager for multithreaded operations.
class MultiThreadPool(collections.abc.Sized, concurrent.futures.Executor):

	def __init__(self, pool_count=1, thread_count=8, initializer=None):
		self.pools = alist()
		self.pool_count = max(1, pool_count)
		self.thread_count = max(1, thread_count)
		self.initializer = initializer
		self.position = -1
		self.update()

	__len__ = lambda self: sum(len(pool._threads) for pool in self.pools)

	# Adjusts pool count if necessary
	def _update(self):
		if self.pool_count != len(self.pools):
			self.pool_count = max(1, self.pool_count)
			self.thread_count = max(1, self.thread_count)
			while self.pool_count > len(self.pools):
				pool = ThreadPoolExecutor(
					max_workers=self.thread_count,
					initializer=self.initializer,
				)
				self.pools.append(pool)
			while self.pool_count < len(self.pools):
				func = self.pools.popright().shutdown
				self.pools[-1].submit(func, wait=True)

	def update(self):
		if not self.pools:
			self._update()
		self.position = (self.position + 1) % len(self.pools)
		self.pools.next().submit(self._update)

	def map(self, func, *args, **kwargs):
		self.update()
		return self.pools[self.position].map(func, *args, **kwargs)

	def submit(self, func, *args, **kwargs):
		self.update()
		return self.pools[self.position].submit(func, *args, **kwargs)

	shutdown = lambda self, wait=True: [exc.shutdown(wait) for exc in self.pools].append(self.pools.clear())

if os.environ.get("IS_BOT"):
	pthreads = ProcessPoolExecutor(4)
else:
	pthreads = ThreadPoolExecutor(4, initializer=__setloop__)
bthreads = ThreadPoolExecutor(32, initializer=__setloop__)
athreads = concurrent.futures.exc_worker = MultiThreadPool(pool_count=2, thread_count=64, initializer=__setloop__)
athreads.pools.append(import_exc)

def get_event_loop():
	return eloop
	try:
		return asyncio.get_running_loop()
	except:
		return eloop

# Creates an asyncio Future that waits on a multithreaded one.
def wrap_future(fut, loop=None, shield=False, thread_safe=True):
	if getattr(fut, "done", None) and fut.done():
		res = fut.result()
		if res is None:
			return emptyfut
		return as_fut(res)
	if loop is None:
		loop = get_event_loop()
	wrapper = None
	if not thread_safe:
		try:
			wrapper = asyncio.wrap_future(fut, loop=loop)
		except (AttributeError, TypeError):
			pass
	if wrapper is None:
		wrapper = loop.create_future()

		def set_suppress(res, is_exception=False):
			try:
				if is_exception:
					wrapper.set_exception(res)
				else:
					wrapper.set_result(res)
			except (RuntimeError, asyncio.InvalidStateError):
				pass

		def on_done(*void):
			try:
				res = fut.result()
			except Exception as ex:
				loop.call_soon_threadsafe(set_suppress, ex, True)
			else:
				loop.call_soon_threadsafe(set_suppress, res)

		fut.add_done_callback(on_done)
	if shield:
		wrapper = asyncio.shield(wrapper)
	return wrapper

def shutdown_thread_after(thread, fut):
	fut.result()
	return thread.shutdown(wait=True)

def create_thread(func, *args, **kwargs):
	target = func
	if args or kwargs:
		target = lambda: func(*args, **kwargs)
	t = threading.Thread(
		target=target,
		daemon=True,
	)
	t.start()
	return t
tsubmit = create_thread

# Runs a function call in a parallel thread, returning a future object waiting on the output.
def create_future_ex(func, *args, timeout=None, priority=False, **kwargs):
	try:
		kwargs["timeout"] = kwargs.pop("_timeout_")
	except KeyError:
		pass
	fut = (athreads, bthreads, pthreads)[priority].submit(func, *args, **kwargs)
	if timeout is not None:
		fut = (athreads, bthreads, pthreads)[priority].submit(fut.result, timeout=timeout)
	return fut
esubmit = create_future_ex

# Forces the operation to be a coroutine regardless of whether it is or not. Regular functions are executed in the thread pool.
async def _create_future(obj, *args, loop, timeout, priority, **kwargs):
	for i in range(256):
		if not asyncio.iscoroutinefunction(obj):
			break
		obj = obj(*args, **kwargs)
	if callable(obj):
		if asyncio.iscoroutinefunction(obj.__call__) or not is_main_thread():
			obj = obj.__call__(*args, **kwargs)
		else:
			obj = await wrap_future(esubmit(obj, *args, timeout=timeout, priority=priority, **kwargs), loop=loop)
	for i in range(256):
		if not awaitable(obj):
			break
		if timeout is not None:
			async with asyncio.timeout(timeout):
				obj = await obj
		else:
			obj = await obj
	return obj

# High level future asyncio creation function that takes both sync and async functions, as well as coroutines directly.
def create_future(obj, *args, loop=None, timeout=None, priority=False, **kwargs):
	if loop is None:
		loop = get_event_loop()
	fut = _create_future(obj, *args, loop=loop, timeout=timeout, priority=priority, **kwargs)
	if not isinstance(fut, asyncio.Task):
		fut = create_task(fut, loop=loop)
	return fut
asubmit = create_future

# Creates an asyncio Task object from an awaitable object.
def create_task(fut, *args, loop=None, **kwargs):
	if loop is None:
		loop = get_event_loop()
	return asyncio.ensure_future(fut, *args, loop=loop, **kwargs)
fsubmit = csubmit = create_task

async def _await_fut(fut, ret):
	out = await fut
	ret.set_result(out)
	return ret

# Blocking call that waits for a single asyncio future to complete, do *not* call from main asyncio loop
def await_fut(fut, timeout=None):
	return convert_fut(fut).result(timeout=timeout)

def convert_fut(fut):
	loop = get_event_loop()
	if is_main_thread():
		if not isinstance(fut, asyncio.Task):
			fut = create_task(fut, loop=loop)
		raise RuntimeError("This function must not be called from the main thread's asyncio loop.")
	try:
		ret = asyncio.run_coroutine_threadsafe(fut, loop=loop)
	except:
		ret = Future()
		loop.create_task(_await_fut(fut, ret))
	return ret

is_main_thread = lambda: threading.current_thread() is threading.main_thread()

# A dummy coroutine that returns None.
async_nop = lambda *args, **kwargs: emptyfut

async def delayed_coro(fut, duration=None):
	async with Delay(duration):
		return await fut

async def waited_coro(fut, duration=None):
	await asyncio.sleep(duration)
	return await fut

async def traceback_coro(fut, *args):
	with tracebacksuppressor(*args):
		return await fut

def trace(fut, *args):
	return create_task(traceback_coro(fut, *args))

# A function that takes a coroutine, and calls a second function if it takes longer than the specified delay.
async def delayed_callback(fut, delay, func, *args, repeat=False, exc=False, **kwargs):
	await asyncio.sleep(delay / 2)
	if not fut.done():
		await asyncio.sleep(delay / 2)
	try:
		return fut.result()
	except ISE:
		while not fut.done():
			if hasattr(func, "__call__"):
				res = func(*args, **kwargs)
			else:
				res = func
			if awaitable(res):
				await res
			if not repeat:
				break
		return await fut
	except:
		if exc:
			raise


@tracebacksuppressor
def exec_tb(s, *args, **kwargs):
	exec(s, *args, **kwargs)


def p2n(b):
	try:
		if isinstance(b, str):
			b = b.encode("ascii")
		if len(b) % 4:
			b += b"=="
		return int.from_bytes(base64.urlsafe_b64decode(b), "big")
	except Exception as ex:
		raise FileNotFoundError(*ex.args)

def n2p(n):
	c = n.bit_length() + 7 >> 3
	return base64.urlsafe_b64encode(n.to_bytes(c, "big")).rstrip(b"=").decode("ascii")

def find_file(path, cwd="saves/filehost", ind="\x7f"):
	# if no file name is inputted, return no content
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


class open2(io.IOBase):

	__slots__ = ("fp", "fn", "mode", "filename")

	def __init__(self, fn, mode="rb", filename=None):
		self.fp = None
		self.fn = fn
		self.mode = mode
		self.filename = filename or getattr(fn, "name", None) or fn

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
		with suppress():
			self.fp.close()
		self.fp = None

class CompatFile(discord.File):

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
				self.filename = getattr(fp, "name", None)
		else:
			self.filename = filename
		self.description = lim_str(description or self.filename or "", 1024) or None
		self.filename = lim_str((self.filename or "untitled").strip().replace(" ", "_").translate(filetrans), 64)
		if spoiler:
			if self.filename is not None:
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

class DownloadingFile(io.IOBase):

	__slots__ = ("fp", "fn", "mode", "filename", "af")

	def __init__(self, fn, af, mode="rb", filename=None):
		self.fp = None
		self.fn = fn
		self.mode = mode
		self.filename = filename or getattr(fn, "name", None) or fn
		self.af = af
		for _ in loop(720):
			if os.path.exists(fn) and os.path.getsize(fn):
				break
			if af():
				raise FileNotFoundError
			time.sleep(0.1)

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
		while os.path.getsize(self.fn) < pos:
			if self.af():
				break
			time.sleep(0.1)
		self._seek(pos)

	def read(self, size):
		b = self._read(size)
		s = len(b)
		if s < size:
			buf = deque()
			buf.append(b)
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
		with suppress():
			self.fp.close()
		self.fp = None

class ForwardedRequest(io.IOBase):

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
		with suppress():
			self.fp.close()
		self.fp = None

class FileStreamer(io.BufferedRandom, contextlib.AbstractContextManager):

	def __init__(self, *objs, filename=None):
		self.pos = 0
		self.data = []
		self.filename = filename
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
			elif isinstance(f, discord.File):
				f = f.fp
			self.data.append((i, f))
			i += f.seek(0, os.SEEK_END)
			self.filename = filename or getattr(f, "filename", None) or getattr(f, "name", None)

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

	isatty = lambda self: False
	flush = lambda self: None
	writable = lambda self: False
	seekable = lambda self: True
	readable = lambda self: True
	tell = lambda self: self.pos
	__enter__ = lambda self: self
	__exit__ = lambda self, *args: self.close()

class PipedProcess:

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
		except:
			import traceback
			traceback.print_exc()
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
		return self.procs[-1].status()

class seq(io.BufferedRandom, collections.abc.MutableSequence, contextlib.AbstractContextManager):

	BUF = 262144
	iter = None

	def __init__(self, obj, filename=None, buffer_size=None):
		if buffer_size:
			self.BUF = buffer_size
		self.closer = getattr(obj, "close", None)
		self.high = 0
		self.finished = False
		if isinstance(obj, io.IOBase):
			if isinstance(obj, io.BytesIO):
				self.data = obj
			elif hasattr(obj, "getbuffer"):
				self.data = io.BytesIO(obj.getbuffer())
			else:
				obj.seek(0)
				self.data = io.BytesIO(obj.read())
				obj.seek(0)
			self.finished = True
		elif isinstance(obj, bytes) or isinstance(obj, bytearray) or isinstance(obj, memoryview):
			self.data = io.BytesIO(obj)
			self.high = len(obj)
			self.finished = True
		elif isinstance(obj, collections.abc.Iterable):
			self.iter = iter(obj)
			self.data = io.BytesIO()
		elif getattr(obj, "iter_content", None):
			self.iter = obj.iter_content(self.BUF)
			self.data = io.BytesIO()
		else:
			raise TypeError(f"a bytes-like object is required, not '{type(obj)}'")
		self.filename = filename
		self.buffer = {}
		self.pos = 0
		self.limit = None

	seekable = lambda self: True
	readable = lambda self: True
	writable = lambda self: False
	isatty = lambda self: False
	flush = lambda self: None
	tell = lambda self: self.pos

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
			if is_finite(stop):
				b = b[:stop - start]
			if step != 1:
				b = b[::step]
			if rev:
				b = b[::-1]
			return b
		base = k // self.BUF
		with suppress(KeyError):
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
			x = self[i]
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

	close = lambda self: self.closer() if self.closer else None
	__enter__ = lambda self: self
	__exit__ = lambda self, *args: self.close()

	def load(self, k):
		if self.finished:
			return self.data.getbuffer()[k:k + self.BUF]
		with suppress(KeyError):
			return self.buffer[k]
		seek = getattr(self.data, "seek", None)
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
			with suppress():
				self.resp.close()
		self.resp = reqs.next().get(self.url, stream=True)
		self.iter = self.resp.iter_content(self.BUF)

	def refill(self):
		att = 0
		while self.buflen < self.BUF * 4:
			try:
				b = next(self.iter)
				self.buf.write(b)
			except StopIteration:
				with suppress():
					self.resp.close()
				return
			except:
				if att > 16:
					raise
				att += 1
				self.reset()
			else:
				self.buflen += len(b)
		with suppress():
			self.resp.close()


def parse_ratelimit_header(headers):
	try:
		reset = headers.get('X-Ratelimit-Reset')
		if reset:
			delta = float(reset) - utc()
		else:
			reset_after = headers.get('X-Ratelimit-Reset-After')
			delta = float(reset_after)
		if not delta:
			raise
	except:
		delta = float(headers['retry_after'])
	return max(0.001, delta)


def proxy_download(url, fn=None, refuse_html=True, timeout=720):
	downloading = globals().setdefault("proxy-download", {})
	try:
		fut = downloading[url]
	except KeyError:
		downloading[url] = fut = Future()
	else:
		return fut.result(timeout=timeout)
	o_url = url
	loc = random.choice(("eu", "us"))
	i = random.randint(1, 17)
	stream = f"https://{loc}{i}.proxysite.com/includes/process.php?action=update"
	with reqs.next().post(
		stream,
		data=dict(d=url, allowCookies="on"),
		timeout=timeout,
		stream=True,
	) as resp:
		if resp.status_code not in range(200, 400):
			raise ConnectionError(resp.status_code, (url, resp.content))
		if not fn:
			b = resp.content
			if refuse_html and b[:15] == b"<!DOCTYPE html>":
				raise ValueError(b[:256])
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
		with open(fn, "rb") as f:
			if refuse_html and f.read(15) == b"<!DOCTYPE html>":
				f.seek(0)
				raise ValueError(f.read(256))
		return fn
	# if fn:
	# 	return o_url
	# fut.set_result(o_url)
	# return o_url


# Runs ffprobe on a file or url, returning the duration if possible.
def get_duration_simple(filename, _timeout=12):
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
		fut = esubmit(proc.wait, timeout=_timeout)
		fut.result(timeout=_timeout)
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
		dur, bps = get_duration_simple(filename, 4)
		if not dur and is_url(filename):
			with reqs.next().get(filename, headers=Request.header(), stream=True) as resp:
				head = fcdict(resp.headers)
				if "Content-Length" not in head:
					dur = get_duration_simple(filename, 20)[0]
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
				dur = get_duration_simple(filename, 16)[0]
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


# Manages both sync and async web requests.
class RequestManager(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, collections.abc.Callable):

	ts = 0
	semaphore = Semaphore(512, 256, delay=0.25)
	sessions = ()

	@classmethod
	def header(cls, base=(), **fields):
		head = {
			"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(xrand(1, 255)) for _ in loop(4)),
			"X-Real-Ip": ".".join(str(xrand(1, 255)) for _ in loop(4)),
		}
		if base:
			head.update(base)
		if fields:
			head.update(fields)
		return head
	headers = header

	@tracebacksuppressor
	async def _init_(self):
		if self.sessions:
			for session in self.sessions:
				await session.close()
			await self.nossl.close()
		self.sessions = alist(aiohttp.ClientSession() for i in range(6))
		self.nossl = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
		self.ts = utc()
		return self

	@property
	def session(self):
		return choice(self.sessions)

	async def aio_call(self, url, headers, files, data, method, decode=False, json=False, session=None, ssl=True, timeout=24):
		async with self.semaphore:
			req = session or (self.sessions.next() if ssl else self.nossl)
			resp = await req.request(method.upper(), url, headers=headers, data=data, timeout=timeout)
			status = getattr(resp, "status_code", None) or getattr(resp, "status", 400)
			if status >= 400:
				try:
					data = await resp.read()
				except (TypeError, AttributeError):
					data = resp.text
				if not isinstance(data, bytes):
					data = bytes(data, "utf-8")
				if not data or magic.from_buffer(data).startswith("text/"):
					raise ConnectionError(status, (url, as_str(data)))
			if json:
				data = resp.json()
				if awaitable(data):
					return await data
				return data
			try:
				data = await resp.read()
				if decode:
					return as_str(data)
				return data
			except (AttributeError, TypeError):
				return resp.content

	def __call__(self, url, headers=None, files=None, data=None, raw=False, timeout=8, method="get", decode=False, json=False, bypass=True, proxy=False, aio=False, session=None, ssl=True, authorise=False):
		if headers is None:
			headers = {}
		if authorise:
			token = AUTH["discord_token"]
			headers["Authorization"] = f"Bot {token}"
			if data:
				if not isinstance(data, aiohttp.FormData):
					if not isinstance(data, (str, bytes, memoryview)):
						data = orjson.dumps(data)
			if data and (isinstance(data, (list, dict)) or (data[:1] in '[{"') if isinstance(data, str) else (data[:1] in b'[{"' if isinstance(data, (bytes, memoryview)) else False)):
				headers["Content-Type"] = "application/json"
			if aio:
				session = self.sessions.next()
			else:
				session = requests
		elif bypass:
			if "user-agent" not in headers and "User-Agent" not in headers:
				headers["User-Agent"] = f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
				headers["X-Forwarded-For"] = ".".join(str(xrand(1, 255)) for _ in loop(4))
			headers["DNT"] = "1"
		method = method.casefold()
		if aio:
			return create_task(asyncio.wait_for(self.aio_call(url, headers, files, data, method, decode, json, session, ssl, timeout=timeout), timeout=timeout))
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
			if raw and getattr(resp, "raw", None):
				data = resp.raw.read()
			else:
				data = resp.content
			if decode:
				return as_str(data)
			return data

	__enter__ = lambda self: self
	def __exit__(self, *args):
		self.session.close()

	def __aexit__(self, *args):
		self.session.close()
		return emptyfut

Request = RequestManager()


emoji_translate = {}
emoji_replace = {}
em_trans = {}
def reload_emojis():
	global emoji_translate, emoji_replace, em_trans
	if not os.path.exists("misc/emojis.json"):
		return
	with open("misc/emojis.json", "rb") as f:
		b = f.read()
	etrans = orjson.loads(b)
	emoji_translate = {k: v for k, v in etrans.items() if len(k) == 1}
	emoji_replace = {k: v for k, v in etrans.items() if len(k) > 1}
	em_trans = "".maketrans(emoji_translate)

@tracebacksuppressor
def load_emojis():
	global emoji_translate, emoji_replace, em_trans
	if os.path.exists("misc/emojis.json") and utc() - os.path.getmtime("misc/emojis.json") < 86400:
		return
	data = Request("https://api.github.com/repos/twitter/twemoji/git/trees/master?recursive=1", json=True, timeout=None)
	ems = [e for e in data["tree"] if e["path"].startswith("assets/svg/")]
	e_ids = [e["path"].rsplit("/", 1)[-1].split(".", 1)[0] for e in ems]
	urls = [f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/{e_id}.svg" for e_id in e_ids]
	emojis = ["".join(chr(int(i, 16)) for i in e_id.split("-")) for e_id in e_ids]
	etrans = dict(zip(emojis, urls))
	b = orjson.dumps(etrans)
	with open("misc/emojis.json", "wb") as f:
		f.write(b)
	emoji_translate = {k: v for k, v in etrans.items() if len(k) == 1}
	emoji_replace = {k: v for k, v in etrans.items() if len(k) > 1}
	em_trans = "".maketrans(emoji_translate)
	print(f"Successfully loaded {len(etrans)} unicode emojis.")

@functools.lru_cache(maxsize=4)
def translate_emojis(s):
	res = s.translate(em_trans)
	if res in emoji_replace:
		return emoji_replace[res]
	return res

@functools.lru_cache(maxsize=4)
def replace_emojis(s):
	for emoji, url in emoji_replace.items():
		if emoji in s:
			s = s.replace(emoji, url)
	return s

@functools.lru_cache(maxsize=4)
def find_emojis_ex(s):
	out = deque()
	for emoji, url in emoji_replace.items():
		if emoji in s:
			out.append(url[1:-1])
	for emoji, url in emoji_translate.items():
		if emoji in s:
			out.append(url[1:-1])
	return list(set(out))

HEARTS = ["❤️", "🧡", "💛", "💚", "💙", "💜", "💗", "💞", "🤍", "🖤", "🤎", "❣️", "💕", "💖"]


# Stores and manages timezones information.
TIMEZONES = cdict()

@tracebacksuppressor
def load_timezones():
	with open("misc/timezones.txt", "rb") as f:
		data = as_str(f.read())
		for line in data.splitlines():
			info = line.split("\t")
			abb = info[0].casefold()
			if len(abb) >= 3 and (abb not in TIMEZONES or "(unofficial)" not in info[1]):
				temp = info[-1].replace("\\", "/")
				curr = sorted([round((1 - (i[3] == "−") * 2) * (time_parse(i[4:]) if ":" in i else float(i[4:]) * 60) * 60) for i in temp.split("/") if i.startswith("UTC")])
				if len(curr) == 1:
					curr = curr[0]
				TIMEZONES[abb] = curr
		print(f"Successfully loaded {len(TIMEZONES)} timezones.")

def is_dst(dt=None, timezone="UTC"):
	if dt is None:
		dt = utc_dt()
	timezone = pytz.timezone(timezone)
	timezone_aware_date = timezone.localize(dt, is_dst=None)
	return timezone_aware_date.tzinfo._dst.seconds != 0

def get_timezone(tz):
	s = TIMEZONES[tz]
	if issubclass(type(s), collections.abc.Collection):
		return s[is_dst(timezone=tz.upper())]
	return s

def as_timezone(tz):
	if not tz:
		raise KeyError
	with suppress(KeyError):
		return round((city_time(tz).timestamp() - utc()) / 60) * 60
	a = tz
	h = 0
	for op in ("+-"):
		with suppress(ValueError):
			i = a.index(op)
			h += float(a[i:])
			a = a[:i]
			break
	tz = a.casefold()
	return round_min(get_timezone(tz) + h * 3600)

def timezone_repr(tz):
	if tz in ZONES:
		return capwords(tz)
	return tz.upper()

def time_repr(t, mode=None):
	if hasattr(t, "timestamp"):
		t = t.timestamp
		if callable(t):
			t = t()
	t = round(t)
	if not mode:
		mode = "R"
	return f"<t:{t}:{mode}>"

def parse_with_now(expr):
	if not expr or expr.strip().casefold() == "now":
		return utc_ddt()
	bc = False
	if expr[-3:].casefold() == " ad":
		expr = expr[:-3]
	elif expr[-5:].casefold() == " a.d.":
		expr = expr[:-5]
	if expr[-3:].casefold() == " bc":
		expr = expr[:-3]
		bc = True
	elif expr[-5:].casefold() == " b.c.":
		expr = expr[:-5]
		bc = True
	try:
		dt = tparser.parse(expr).replace(tzinfo=datetime.timezone.utc)
	except Exception as ex:
		print(ex)
		s = str(ex).split(":", 1)[0]
		if s.startswith("year "):
			s = s[5:]
			if s.endswith(" is out of range"):
				s = s[:-16]
				y = int(s)
				if bc:
					y = -y
				offs, year = divmod(y, 400)
				offs = offs * 400 - 2000
				year += 2000
				expr = regexp("0*" + s).sub(str(year), expr, 1)
				dt = tparser.parse(expr).replace(tzinfo=datetime.timezone.utc)
				return DynamicDT.fromdatetime(dt).set_offset(offs)
		elif s.startswith("Python int too large to convert to C"):
			y = int(regexp("[0-9]{10,}").findall(expr)[0])
			if bc:
				y = -y
			offs, year = divmod(y, 400)
			offs = offs * 400 - 2000
			year += 2000
			expr = regexp("[0-9]{10,}").sub(str(year), expr, 1)
			dt = tparser.parse(expr).replace(tzinfo=datetime.timezone.utc)
			return DynamicDT.fromdatetime(dt).set_offset(offs)
		elif s.startswith("Unknown string format") or s.startswith("month must be in"):
			try:
				y = int(regexp("[0-9]{5,}").findall(expr)[0])
			except IndexError:
				y = None
			if y is None:
				raise
			if bc:
				y = -y
			offs, year = divmod(y, 400)
			offs = offs * 400 - 2000
			year += 2000
			expr = regexp("[0-9]{5,}").sub(str(year), expr, 1)
			dt = tparser.parse(expr).replace(tzinfo=datetime.timezone.utc)
			return DynamicDT.fromdatetime(dt).set_offset(offs)
		raise
	if bc:
		y = -dt.year
		offs, year = divmod(y, 400)
		offs = offs * 400 - 2000
		year += 2000
		return DynamicDT.fromdatetime(dt.replace(year=year)).set_offset(offs)
	return DynamicDT.fromdatetime(dt)

# Parses a time expression, with an optional timezone input at the end.
def tzparse(expr):
	try:
		s = float(expr)
	except ValueError:
		expr = expr.strip()
		day = None
		if "today" in expr:
			day = 0
			expr = expr.replace("today", "")
		elif "tomorrow" in expr:
			day = 1
			expr = expr.replace("tomorrow", "")
		elif "yesterday" in expr:
			day = -1
			expr = expr.replace("yesterday", "")
		if " " in expr:
			t = 0
			try:
				args = shlex.split(expr)
			except ValueError:
				args = expr.split()
			for i in (0, -1):
				arg = args[i]
				with suppress(KeyError):
					t = as_timezone(arg)
					args.pop(i)
					expr = " ".join(args)
					break
				h = 0
			t = parse_with_now(expr) - (h * 3600 + t)
		else:
			t = parse_with_now(expr)
		if day is not None:
			curr = utc_ddt() + day * 86400
			one_day = 86400
			while t < curr:
				t += one_day
			while (t - curr).total_seconds() > one_day:
				t -= one_day
		return t
	if not is_finite(s) or abs(s) >= 1 << 31:
		try:
			s = int(expr.split(".", 1)[0])
		except (TypeError, ValueError):
			pass
	return utc_dft(s)


smart_split = lambda s: _smart_split(s).copy()

@functools.lru_cache(maxsize=64)
def _smart_split(s):
	s = s.replace("#", "\uffff")
	try:
		t = shlex.shlex(s)
		t.whitespace_split = True
		out = deque()
		while True:
			try:
				w = t.get_token()
			except ValueError:
				out.append(t.token.strip(t.quotes))
				break
			if not w:
				break
			out.extend(shlex.split(w))
	except ValueError:
		out = s.split()
	return alist(w.replace("\uffff", "#") for w in out)


import tiktoken

def get_encoding(e):
	try:
		return tiktoken.get_encoding(e)
	except (KeyError, ValueError):
		pass
	return tiktoken.encoding_for_model(e)

def tik_encode(s, encoding="cl100k_base"):
	enc = get_encoding(encoding)
	return enc.encode(s)

def tik_decode(t, encoding="cl100k_base"):
	enc = get_encoding(encoding)
	return enc.decode(t)

@functools.lru_cache(maxsize=64)
def lim_tokens(s, maxlen=10, mode="centre", encoding="cl100k_base"):
	if maxlen is None:
		return s
	if type(s) is not str:
		s = str(s)
	enc = get_encoding(encoding)
	tokens = enc.encode(s)
	over = (len(tokens) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(tokens) / 2
			s = enc.decode(tokens[:ceil(half - over - 1)]) + ".." + enc.decode(tokens[ceil(half + over + 1):])
		else:
			s = enc.decode(tokens[:maxlen - 3]) + "..."
	return s.strip()

async def tik_encode_a(s, encoding="cl100k_base"):
	if len(s) > 1024:
		return await asubmit(tik_encode, s, encoding=encoding, priority=2)
	return tik_encode(s, encoding=encoding)

async def tik_decode_a(t, encoding="cl100k_base"):
	if len(t) > 256:
		return await asubmit(tik_decode, t, encoding=encoding, priority=2)
	return tik_decode(t, encoding=encoding)

async def tcount(s, model="gpt-3.5-turbo"):
	enc = await tik_encode_a(s, encoding=model)
	return len(enc)


class CacheItem:
	__slots__ = ("value")

	def __init__(self, value):
		self.value = value

class Cache(cdict):
	__slots__ = ("timeout", "tmap", "soonest", "sooning", "lost", "trash", "db")

	def __init__(self, *args, timeout=60, trash=8, **kwargs):
		super().__init__(*args, **kwargs)
		object.__setattr__(self, "timeout", timeout)
		object.__setattr__(self, "lost", {})
		object.__setattr__(self, "trash", trash)
		if self:
			ts = utc() + timeout
			tmap = {k: ts for k in self}
			object.__setattr__(self, "soonest", ts)
		else:
			tmap = {}
			object.__setattr__(self, "soonest", inf)
		object.__setattr__(self, "tmap", tmap)
		if self:
			fut = create_task(waited_coro(self._update(), timeout))
		else:
			fut = None
		object.__setattr__(self, "sooning", fut)
		object.__setattr__(self, "db", None)

	def attach(self, db):
		db.setdefault("__lost", {}).update(self.lost)
		object.__setattr__(self, "lost", db["__lost"])
		db.setdefault("__tmap", {}).update(self.tmap)
		db.pop("tmap", None)
		object.__setattr__(self, "db", db)

	async def _update(self):
		tmap = self.tmap
		timeout = self.timeout
		lost = self.lost
		t = utc()
		for k in sorted(tmap, key=tmap.__getitem__):
			if t <= tmap[k] + timeout:
				ts = tmap[k] + timeout
				object.__setattr__(self, "soonest", ts)
				fut = create_task(waited_coro(self._update(), ts - t))
				object.__setattr__(self, "sooning", fut)
				break
			tmap.pop(k)
			if self.trash > 0:
				while len(lost) > self.trash:
					with suppress(KeyError, RuntimeError):
						lost.pop(next(iter(lost)))
				lost[k] = self.db.pop(k) if self.db else super().pop(k)
		else:
			object.__setattr__(self, "soonest", inf)
		return self

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
			self.db.update("__tmap")
			self.db.update("__lost")
		timeout = self.timeout
		t = utc()
		ts = t + timeout
		if ts < self.soonest:
			object.__setattr__(self, "soonest", ts)
			fut = create_task(waited_coro(self._update(), timeout))
			sooning = self.sooning
			if sooning:
				sooning.cancel()
			object.__setattr__(self, "sooning", fut)
		self.tmap[k] = ts

	retrieve = lambda self, k: self.lost.pop(k)

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
				return await resp.value
		fut = create_task(self.retrieve_into(k, func, *args, **kwargs))
		try:
			resp = self.retrieve(k)
		except KeyError:
			pass
		else:
			if isinstance(resp, CacheItem):
				return await resp.value
		super().__setitem__(k, CacheItem(fut))
		return await fut


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


# Basic inheritable class for all bot commands.
class Command(collections.abc.Hashable, collections.abc.Callable):
	description = ""
	usage = ""
	min_level = 0
	rate_limit = (2, 3)

	def perm_error(self, perm, req=None, reason=None):
		if req is None:
			req = self.min_level
		if reason is None:
			reason = f"for command {self.name[-1]}"
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
			self.alias.append(self.parse_name())
		self.name.append(self.parse_name())
		self.aliases = {full_prune(alias).replace("*", "").replace("_", "").replace("||", ""): alias for alias in self.alias}
		self.aliases.pop("", None)
		for a in self.aliases:
			if a in bot.commands:
				bot.commands[a].add(self)
			else:
				bot.commands[a] = alist((self,))
		self.catg = catg
		self.bot = bot
		self._globals = bot._globals
		f = getattr(self, "__load__", None)
		if callable(f):
			try:
				f()
			except:
				print_exc()
				self.data.clear()
				f()

	__hash__ = lambda self: hash(self.parse_name()) ^ hash(self.catg)
	__str__ = lambda self: f"Command <{self.parse_name()}>"
	__call__ = lambda self, **void: None

	parse_name = lambda self: self.__name__.strip("_")
	parse_description = lambda self: self.description.replace('⟨MIZA⟩', self.bot.user.name).replace('⟨WEBSERVER⟩', self.bot.webserver)

	def unload(self):
		bot = self.bot
		for alias in self.alias:
			alias = alias.replace("*", "").replace("_", "").replace("||", "")
			coms = bot.commands.get(alias)
			if coms:
				coms.remove(self)
				print("unloaded", alias, "from", self)
			if not coms:
				bot.commands.pop(alias, None)


# Basic inheritable class for all bot databases.
class Database(collections.abc.MutableMapping, collections.abc.Hashable, collections.abc.Callable):
	bot = None
	rate_limit = 3
	name = "data"
	encode = None
	decode = None

	def __init__(self, bot, catg):
		name = self.name
		self.__name__ = self.__class__.__name__
		fhp = "saves/" + name
		if not getattr(self, "no_file", False):
			if os.path.exists(fhp):
				data = self.data = FileHashDict(path=fhp, encode=self.encode, decode=self.decode)
			else:
				self.file = fhp + ".json"
				self.updated = False
				try:
					with open(self.file, "rb") as f:
						s = f.read()
					if not s:
						raise FileNotFoundError
					try:
						data = select_and_loads(s, mode="unsafe")
					except:
						print(self.file)
						print_exc()
						raise FileNotFoundError
					data = FileHashDict(data, path=fhp, encode=self.encode, decode=self.decode)
					data.modified.update(data.data.keys())
					self.iter = None
					self.data = data
				except FileNotFoundError:
					data = None
		else:
			data = self.data = {}
		if data is None:
			self.data = FileHashDict(path=fhp, encode=self.encode, decode=self.decode)
		if not issubclass(type(self.data), collections.abc.MutableMapping):
			self.data = FileHashDict(dict.fromkeys(self.data), path=fhp, encode=self.encode, decode=self.decode)
		bot.database[name] = bot.data[name] = self
		self.catg = catg
		self.bot = bot
		self._semaphore = Semaphore(1, 1, delay=0.5, rate_limit=self.rate_limit)
		self._garbage_semaphore = Semaphore(1, 0, delay=3, rate_limit=self.rate_limit * 3 + 30)
		self._globals = globals()
		f = getattr(self, "__load__", None)
		if callable(f):
			try:
				f()
			except:
				print_exc()
				# self.data.clear()
				# f()

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

	keys = lambda self: self.data.keys()
	items = lambda self: self.data.items()
	values = lambda self: self.data.values()
	get = lambda self, *args, **kwargs: self.data.get(*args, **kwargs)
	pop = lambda self, *args, **kwargs: self.data.pop(*args, **kwargs)
	popitem = lambda self, *args, **kwargs: self.data.popitem(*args, **kwargs)
	fill = lambda self, other: self.data.fill(other)
	clear = lambda self: self.data.clear()
	setdefault = lambda self, k, v: self.data.setdefault(k, v)
	keys = lambda self: self.data.keys()
	discard = lambda self, k: self.data.pop(k, None)
	vacuum = lambda self: self.data.vacuum() if hasattr(self.data, "vacuum") else None

	def update(self, modified=None, force=False):
		if hasattr(self, "no_file"):
			return
		if force:
			try:
				limit = getattr(self, "limit", None)
				if limit and len(self) > limit:
					print(f"{self} overflowed by {len(self) - limit}, dropping...")
					with tracebacksuppressor:
						while len(self) > limit:
							self.pop(next(iter(self)))
				self.data.__update__()
			except:
				print(self, traceback.format_exc(), sep="\n", end="")
		else:
			if modified is None:
				self.data.modified.update(self.data.keys())
			else:
				if issubclass(type(modified), collections.abc.Sized) and type(modified) not in (str, bytes):
					self.data.modified.update(modified)
				else:
					self.data.modified.add(modified)
			self.data.iter = None
		return False

	def unload(self):
		self.unloaded = True
		bot = self.bot
		func = getattr(self, "_destroy_", None)
		if callable(func):
			await_fut(create_future(func, priority=True))
		for f in dir(self):
			if f.startswith("_") and f[-1] == "_" and f[1] != "_":
				func = getattr(self, f, None)
				if callable(func):
					bot.events[f].remove(func)
					print("unloaded", f, "from", self)
		self.update(force=True)
		bot.data.pop(self, None)
		bot.database.pop(self, None)
		self.data.clear()


class ImagePool:
	usage = "<verbose{?v}>?"
	flags = "v"
	rate_limit = (0.05, 0.25)
	threshold = 1024

	async def __call__(self, bot, channel, flags, message, **void):
		url = await bot.data.imagepools.get(self.database, self.fetch_one, self.threshold)
		if "v" in flags:
			return escape_roles(url)
		self.bot.send_as_embeds(channel, image=url, reference=message)


if __name__ != "__mp_main__":
	# Redirects all print operations to target files, limiting the amount of operations that can occur in any given amount of time for efficiency.
	class __logPrinter:

		ignored_messages = {
			"A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.",
		}

		def __init__(self, file=None):
			self.buffer = self
			self.data = {}
			self.history = {}
			self.counts = {}
			self.funcs = alist()
			self.file = file
			self.closed = True

		def start(self):
			# self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
			# self.future = self.exec.submit(self.update_print)
			threading.Thread(target=self.update_print).start()
			self.closed = False

		def file_print(self, fn, b):
			try:
				if type(fn) not in (str, bytes):
					f = fn
				elif type(b) in (bytes, bytearray):
					f = open(fn, "ab")
				elif type(b) is str:
					f = open(fn, "a", encoding="utf-8")
				else:
					f = fn
				with closing(f):
					try:
						f.write(b)
					except TypeError:
						try:
							f.write(as_str(b))
						except ValueError:
							pass
			except:
				sys.__stdout__.write(traceback.format_exc())

		def flush(self):
			outfunc = lambda s: self.file_print(self.file, s)
			enc = lambda x: bytes(x, "utf-8")
			try:
				for f in tuple(self.data):
					if not self.data[f]:
						self.data.pop(f)
						continue
					out = lim_str(self.data[f], 65536)
					data = enc(self.data[f])
					self.data[f] = ""
					if self.funcs and out.strip():
						[func(out) for func in self.funcs]
					if f == self.file:
						outfunc(data)
					else:
						self.file_print(f, data)
			except:
				sys.__stdout__.write(traceback.format_exc())

		def update_print(self):
			if self.file is None:
				return
			while True:
				with Delay(10):
					self.flush()
				while not os.path.exists("common.py") or self.closed:
					time.sleep(1)

		def __call__(self, *args, sep=" ", end="\n", prefix="", file=None, **void):
			out = str(sep).join(i if type(i) is str else str(i) for i in args) + str(end) + str(prefix)
			if not out:
				return
			temp = out.strip()
			if self.closed or temp.rsplit("\n", 1)[-1] in self.ignored_messages:
				return sys.__stdout__.write(out)
			if file is None:
				file = self.file
			if file not in self.data:
				self.data[file] = ""
			if temp:
				if file in self.history and self.history.get(file).strip() == temp:
					add_dict(self.counts, {file:1})
					return
				elif self.counts.get(file):
					count = self.counts.pop(file)
					times = "s" if count != 1 else ""
					out, self.history[file] = f"<Last message repeated {count} time{times}>\n{out}", out
				else:
					self.history[file] = out
					self.counts.pop(file, None)
			self.data[file] += out
			return sys.__stdout__.write(out)

		def write(self, *args, end="", **kwargs):
			args2 = [as_str(arg) for arg in args]
			return self.__call__(*args2, end=end, **kwargs)

		read = lambda self, *args, **kwargs: bytes()
		close = lambda self, force=False: self.__setattr__("closed", force)
		isatty = lambda self: False


	esubmit(load_mimes)
	esubmit(reload_emojis)
	PRINT = __logPrinter("log.txt")

	# Sets all instances of print to the custom print implementation.

	# sys.stdout = sys.stderr = print
	# for mod in (discord, concurrent.futures, asyncio.futures, asyncio, psutil, subprocess, tracemalloc):
	#     builtins = getattr(mod, "__builtins__", None)
	#     if builtins:
	#         builtins["print"] = print
