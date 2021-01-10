import smath
from smath import *

with MultiThreadedImporter(globals()) as importer:
    importer.__import__(
        "os",
        "importlib",
        "inspect",
        "tracemalloc",
        "psutil",
        "subprocess",
        "asyncio",
        "discord",
        "json",
        "requests",
        "aiohttp",
        "psutil",
        "threading",
        "urllib",
        "zipfile",
        "nacl",
        "shutil",
        "magic",
    )

PROC = psutil.Process()
quit = lambda *args, **kwargs: PROC.kill()

tracemalloc.start()

from zipfile import ZipFile
import urllib.request, urllib.parse
import nacl.secret

url_parse = urllib.parse.quote_plus
escape_markdown = discord.utils.escape_markdown
escape_mentions = discord.utils.escape_mentions
escape_everyone = lambda s: s.replace("@everyone", "@\xadeveryone").replace("@here", "@\xadhere").replace("<@&", "<@\xad&")

DISCORD_EPOCH = 1420070400000
MIZA_EPOCH = 1577797200000
time_snowflake = discord.utils.time_snowflake
id2ts = lambda id: ((id >> 22) + (id & 0xFFF) / 0x1000 + DISCORD_EPOCH) / 1000
snowflake_time = lambda id: utc_ft(id2ts(id))
snowflake_time_2 = lambda id: datetime.datetime.fromtimestamp(id2ts(id))

ip2int = lambda ip: int.from_bytes(b"\x00" + bytes(int(i) for i in ip.split(".")), "big")


class EmptyContext(contextlib.AbstractContextManager):
    __enter__ = lambda self, *args: self
    __exit__ = lambda *args: None

    async def __aenter__(self, *args):
        pass
    async def __aexit__(self, *args):
        pass

emptyctx = EmptyContext()


# Manages concurrency limits, similar to asyncio.Semaphore, but has a secondary threshold for enqueued tasks.
class Semaphore(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    __slots__ = ("limit", "buffer", "delay", "active", "passive", "last", "ratio")

    def __init__(self, limit=256, buffer=32, delay=0.05, rate_limit=None, randomize_ratio=2):
        self.limit = limit
        self.buffer = buffer
        self.delay = delay
        self.active = 0
        self.passive = 0
        self.rate_limit = rate_limit
        self.rate_bin = alist()
        self.last = utc()
        self.ratio = randomize_ratio

    def _update_bin(self):
        while self.rate_bin and utc() - self.rate_bin[0] >= self.rate_limit:
            self.rate_bin.popleft()
        return self.rate_bin
    
    def __enter__(self):
        self.last = utc()
        if self.is_busy():
            if self.passive >= self.buffer:
                raise SemaphoreOverflowError(f"Semaphore object of limit {self.limit} overloaded by {self.passive}")
            self.passive += 1
            while self.is_busy():
                time.sleep(self.delay if not self.ratio else (random.random() * self.ratio + 1) * self.delay)
                self._update_bin()
            self.passive -= 1
        if self.rate_limit:
            self.rate_bin.append(utc())
        self.active += 1
        return self
    
    async def __aenter__(self):
        self.last = utc()
        if self.is_busy():
            if self.passive >= self.buffer:
                raise SemaphoreOverflowError(f"Semaphore object of limit {self.limit} overloaded by {self.passive}")
            self.passive += 1
            while self.is_busy():
                await asyncio.sleep(self.delay)
                self._update_bin()
            self.passive -= 1
        if self.rate_limit:
            self.rate_bin.append(utc())
        self.active += 1
        return self

    def __exit__(self, *args):
        self.active -= 1
        self.last = utc()

    async def __aexit__(self, *args):
        self.active -= 1
        self.last = utc()

    async def __call__(self):
        while self.value >= self.limit:
            await asyncio.sleep(self.delay)

    def is_active(self):
        return self.active or self.passive

    def is_busy(self):
        return self.active >= self.limit or len(self._update_bin()) >= self.limit

    @property
    def busy(self):
        return self.is_busy()

class SemaphoreOverflowError(RuntimeError):
    __slots__ = ()


# A context manager that sends exception tracebacks to stdout.
class TracebackSuppressor(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, *args, **kwargs):
        self.exceptions = args + tuple(kwargs.values())
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    return True
            try:
                raise exc_value
            except:
                print_exc()
        return True

    async def __aexit__(self, *args):
        return self.__exit__(*args)

    __call__ = lambda self, *args, **kwargs: self.__class__(*args, **kwargs)

tracebacksuppressor = TracebackSuppressor()


# Sends an exception into the target discord sendable, with the autodelete react.
send_exception = lambda sendable, ex, reference=None: send_with_react(sendable, exc_repr(ex), reacts="❎", reference=reference)


# A context manager that sends exception tracebacks to a sendable.
class ExceptionSender(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, sendable, *args, reference=None, **kwargs):
        self.sendable = sendable
        self.reference = reference
        self.exceptions = args + tuple(kwargs.values())
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    create_task(send_exception(self.sendable, exc_value, self.reference))
                    return True
            create_task(send_exception(self.sendable, exc_value, self.reference))
            with tracebacksuppressor:
                raise exc_value
        return True

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    await send_exception(self.sendable, exc_value, self.reference)
                    return True
            await send_exception(self.sendable, exc_value, self.reference)
            with tracebacksuppressor:
                raise exc_value
        return True

    __call__ = lambda self, *args, **kwargs: self.__class__(*args, **kwargs)


# A context manager that delays the return of a function call.
class delay(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, duration=0):
        self.duration = duration
        self.start = utc()

    def __call__(self):
        return self.exit()
    
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
    
    def __exit__(self, *args):
        taken = utc() - self.start
        try:
            self.timers[self.name].append(taken)
        except KeyError:
            self.timers[self.name] = t = deque(maxlen=8)
            t.append(taken)

    async def __aexit__(self, *args):
        taken = utc() - self.start
        try:
            self.timers[self.name].append(taken)
        except KeyError:
            self.timers[self.name] = t = deque(maxlen=8)
            t.append(taken)


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


class ArgumentError(LookupError):
    __slots__ = ()

class TooManyRequests(PermissionError):
    __slots__ = ()

class CommandCancelledError(RuntimeError):
    __slots__ = ()


python = ("python3", "python")[os.name == "nt"]
python_path = ""


with open("auth.json") as f:
    AUTH = eval(f.read())

enc_key = None
with tracebacksuppressor:
    py = AUTH.get("python_path", "")
    while py.endswith("\\") or py.endswith("/"):
        py = py[:-1]
    if py:
        python_path = py + "/"
        python = python_path + "python"

with tracebacksuppressor:
    enc_key = AUTH["encryption_key"]

if not enc_key:
    enc_key = AUTH["encryption_key"] = as_str(base64.b64encode(randbytes(32)))
    try:
        s = json.dumps(AUTH, indent=4)
    except:
        print_exc()
        s = repr(AUTH)
    with open("auth.json", "w", encoding="utf-8") as f:
        f.write(s)

enc_box = nacl.secret.SecretBox(base64.b64decode(enc_key)[:32])


def zip2bytes(data):
    if not hasattr(data, "read"):
        data = io.BytesIO(data)
    z = ZipFile(data, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
    b = z.open("DATA").read()
    z.close()
    return b

def bytes2zip(data):
    b = io.BytesIO()
    z = ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    z.writestr("DATA", data=data)
    z.close()
    b.seek(0)
    return b.read()


# Safer than raw eval, more powerful than json.decode
def eval_json(s):
    if type(s) is memoryview:
        s = bytes(s)
    with suppress(json.JSONDecodeError):
        return json.loads(s)
    return safe_eval(s)

encrypt = lambda s: b">~MIZA~>" + enc_box.encrypt(s if type(s) is bytes else str(s).encode("utf-8"))
def decrypt(s):
    if type(s) is not bytes:
        s = str(s).encode("utf-8")
    if s[:8] == b">~MIZA~>":
        return enc_box.decrypt(s[8:])
    raise ValueError("Data header not found.")

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
        else:
            time.sleep(0.1)
    b = io.BytesIO(s)
    if zipfile.is_zipfile(b):
        print(f"Loading zip file of size {len(s)}...")
        b.seek(0)
        z = ZipFile(b, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
        if size:
            x = z.getinfo("DATA").file_size
            if size < x:
                raise OverflowError(f"Data input size too large ({x} > {size}).")
        s = z.open("DATA").read()
        z.close()
    data = None
    with tracebacksuppressor:
        if s[0] == 128:
            data = pickle.loads(s)
    # if data and type(data) in (str, bytes):
    #     s, data = data, None
    if data is None:
        if mode == "unsafe":
            data = eval(compile(s.replace(b"\0", b""), "<loader>", "eval", optimize=2, dont_inherit=False))
        else:
            if b"{" in s:
                s = s[s.index(b"{"):s.rindex(b"}") + 1]
            data = eval_json(s)
    return data

def select_and_dumps(data, mode="safe"):
    if mode == "unsafe":
        s = pickle.dumps(data)
        if len(s) > 65536:
            s = bytes2zip(s)
        return s
    try:
        s = json.dumps(data)
    except:
        s = None
    if not s or len(s) > 262144:
        s = pickle.dumps(data)
        if len(s) > 1048576:
            s = bytes2zip(s)
        return encrypt(s)
    return s.encode("utf-8")


class FileHashDict(collections.abc.MutableMapping):

    sem = Semaphore(64, 128, 0.3, 1)

    def __init__(self, *args, path="", **kwargs):
        if not kwargs and len(args) == 1:
            self.data = args[0]
        else:
            self.data = dict(*args, **kwargs)
        self.path = path.rstrip("/")
        self.modified = set()
        self.deleted = set()
        self.iter = None
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)
            self.iter = []
    
    __hash__ = lambda self: lambda self: hash(self.path)
    __str__ = lambda self: self.__class__.__name__ + "(" + str(self.data) + ")"
    __repr__ = lambda self: self.__class__.__name__ + "(" + str(self.full) + ")"
    __call__ = lambda self, k: self.__getitem__(k)
    __len__ = lambda self: len(self.keys())
    __contains__ = lambda self, k: (k in self.data or k in self.keys()) and k not in self.deleted
    __eq__ = lambda self, other: self.data == other
    __ne__ = lambda self, other: self.data != other

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
                out[k] = create_future_ex(self.__getitem__, k)
                waits.add(k)
        for k in waits:
            out[k] = out[k].result()
        return out

    def keys(self):
        if self.iter is None or self.modified or self.deleted:
            gen = (try_int(i) for i in os.listdir(self.path) if i not in self.deleted)
            if self.modified:
                gen = set(gen)
                gen.update(self.modified)
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
        with suppress(KeyError):
            return self.data[k]
        fn = self.key_path(k)
        if not os.path.exists(fn):
            fn += "\x7f"
            if not os.path.exists(fn):
                raise KeyError(k)
        with self.sem:
            with open(fn, "rb") as f:
                s = f.read()
        data = BaseException
        with tracebacksuppressor:
            data = select_and_loads(s, mode="unsafe")
        if data is BaseException:
            for file in sorted(os.listdir("backup"), reverse=True):
                with tracebacksuppressor:
                    z = zipfile.ZipFile("backup/" + file, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
                    time.sleep(0.03)
                    s = z.open(fn).read()
                    z.close()
                    data = select_and_loads(s, mode="unsafe")
                    print(f"Successfully recovered backup of {fn} from {file}.")
                    break
        if data is BaseException:
            raise BaseException(k)
        self.data[k] = data
        return data

    def __setitem__(self, k, v):
        with suppress(ValueError):
            k = int(k)
        self.data[k] = v
        self.modified.add(k)

    def get(self, k, default=None):
        with suppress(KeyError):
            return self[k]
        return default

    def pop(self, k, *args, force=False):
        fn = self.key_path(k)
        try:
            if force:
                out = self[k]
                del self.data[k]
                self.deleted.add(k)
                return out
            self.deleted.add(k)
            return self.data.pop(k)
        except KeyError:
            if not os.path.exists(fn):
                if args:
                    return args[0]
                raise

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
        self.update(other)
        return self

    def clear(self):
        self.iter.clear()
        self.modified.clear()
        self.data.clear()
        with suppress(FileNotFoundError):
            shutil.rmtree(self.path)
        os.mkdir(self.path)
        return self

    def __update__(self):
        modified = frozenset(self.modified)
        if modified:
            self.iter = None
        self.modified.clear()
        for k in modified:
            fn = self.key_path(k)
            try:
                d = self.data[k]
            except KeyError:
                self.deleted.add(k)
                continue
            s = select_and_dumps(d, mode="unsafe")
            with self.sem:
                safe_save(fn, s)
        deleted = frozenset(self.deleted)
        if deleted:
            self.iter = None
        self.deleted.clear()
        for k in deleted:
            with suppress(FileNotFoundError):
                os.remove(self.key_path(k))
        while len(self.data) > 1048576:
            self.data.pop(next(iter(self.data)), None)
        return modified.union(deleted)


def safe_save(fn, s):
    if os.path.exists(fn):
        with open(fn + "\x7f", "wb") as f:
            f.write(s)
        if os.path.exists(fn + "\x7f\x7f"):
            with tracebacksuppressor:
                os.remove(fn + "\x7f\x7f")
    if os.path.exists(fn) and not os.path.exists(fn + "\x7f\x7f"):
        os.rename(fn, fn + "\x7f\x7f")
        os.rename(fn + "\x7f", fn)
        if os.path.exists(fn + "\x7f\x7f"):
            create_future_ex(os.remove, fn + "\x7f\x7f", priority=True)
    else:
        with open(fn, "wb") as f:
            f.write(s)


# Decodes HTML encoded characters in a string.
def html_decode(s):
    while len(s) > 7:
        try:
            i = s.index("&#")
        except ValueError:
            break
        try:
            if s[i + 2] == "x":
                h = "0x"
                p = i + 3
            else:
                h = ""
                p = i + 2
            for a in range(4):
                if s[p + a] == ";":
                    v = int(h + s[p:p + a])
                    break
            c = chr(v)
            s = s[:i] + c + s[p + a + 1:]
        except ValueError:
            continue
        except IndexError:
            continue
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return s.replace("&quot;", '"').replace("&apos;", "'")


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
sqr_md = lambda s: f"[{no_md(s)}]" if not issubclass(type(s), discord.abc.GuildChannel) else f"[#{no_md(s)}]"

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
code_md = lambda s: f"```\n{s}```"
py_md = lambda s: f"```py\n{s}```"
ini_md = lambda s: f"```ini\n{s}```"
css_md = lambda s: f"```css\n{s}```".replace("'", "\u2019").replace('"', "\u201d")
fix_md = lambda s: f"```fix\n{s}```"

# Discord object mention formatting
user_mention = lambda u_id: f"<@{u_id}>"
user_pc_mention = lambda u_id: f"<@!{u_id}>"
channel_mention = lambda c_id: f"<#{c_id}>"
role_mention = lambda r_id: f"<@&{r_id}>"


# Counts the number of lines in a file.
def line_count(fn):
    with open(fn, "r", encoding="utf-8") as f:
        data = f.read()
        return alist((len(data), data.count("\n") + 1))


# Checks if a file is a python code file using its filename extension.
is_code = lambda fn: str(fn).endswith(".py") or str(fn).endswith(".pyw")

def touch(file):
    with open(file, "ab"):
        pass


def get_folder_size(path="."):
    return sum(get_folder_size(f.path) if f.is_dir() else f.stat().st_size for f in os.scandir(path))


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


is_channel = lambda channel: issubclass(type(channel), discord.abc.GuildChannel) or type(channel) is discord.abc.PrivateChannel


REPLY_SEM = cdict()

async def send_with_reply(channel, reference, content="", embed=None, tts=None, mention=False):
    try:
        sem = REPLY_SEM[channel.id]
    except KeyError:
        sem = REPLY_SEM[channel.id] = Semaphore(5, buffer=256, delay=0.1, rate_limit=5)
    if getattr(reference, "slash", None) and not embed:
        inter = True
        # try:
        #     discord_id = AUTH['discord_id']
        # except KeyError:
        #     url = f"https://discord.com/api/v8/interactions/{reference.id}/{reference.slash}/callback"
        # else:
        #     url = f"https://discord.com/api/v8/webhooks/{discord_id}/{reference.slash}"
        url = f"https://discord.com/api/v8/interactions/{reference.id}/{reference.slash}/callback"
        data = dict(
            type=4,
            data=dict(
                flags=64,
                content=content,
            ),
        )
    else:
        inter = False
        url = f"https://discord.com/api/v8/channels/{channel.id}/messages"
        if not reference or getattr(reference, "noref", None) or getattr(channel, "simulated", None):
            fields = {}
            if embed:
                fields["embed"] = embed
            if tts:
                fields["tts"] = tts
            return await channel.send(content, **fields)
        if not is_channel(channel):
            c = channel.dm_channel
            if c is None:
                c = await channel.create_dm()
            channel = c
        data = dict(
            content=content,
            message_reference=dict(message_id=str(verify_id(reference))),
            allowed_mentions=dict(parse=["users", "roles", "everyone"], replied_user=mention)
        )
        if embed is not None:
            data["embed"] = embed.to_dict()
        if tts is not None:
            data["tts"] = tts
    body = json.dumps(data)
    exc = RuntimeError
    for i in range(xrand(12, 17)):
        try:
            async with sem:
                # if inter:
                #     print(url, body)
                resp = await Request.aio_call(
                    url,
                    method="post",
                    data=body,
                    headers={"Content-Type": "application/json", "authorization": f"Bot {channel._state.http.token}"},
                    decode=False,
                    files=None,
                )
            return discord.Message(state=channel._state, channel=channel, data=eval_json(resp))
        except Exception as ex:
            exc = ex
            if ex.args and "400" in str(ex.args[0]) or "401" in str(ex.args[0]) or "403" in str(ex.args[0]) or "404" in str(ex.args[0]):
                if not inter:
                    print_exc()
                elif "404" in str(ex.args[0]):
                    continue
                fields = {}
                if embed:
                    fields["embed"] = embed
                if tts:
                    fields["tts"] = tts
                return await channel.send(content, **fields)
        await asyncio.sleep(i + 1)
    raise exc

# Sends a message to a channel, then adds reactions accordingly.
async def send_with_react(channel, *args, reacts=None, reference=None, mention=False, **kwargs):
    with tracebacksuppressor:
        if reference:
            sent = await send_with_reply(channel, reference, *args, mention=mention, **kwargs)
        else:
            sent = await channel.send(*args, **kwargs)
        if reacts:
            for react in reacts:
                async with delay(1 / 3):
                    create_task(sent.add_reaction(react))


# Creates and starts a coroutine for typing in a channel.
typing = lambda self: create_task(self.trigger_typing())


# Finds the best URL for a discord object's icon, prioritizing proxy_url for images if applicable.
proxy_url = lambda obj: obj if type(obj) is str else (to_png(obj.avatar_url) if getattr(obj, "avatar_url", None) else (obj.proxy_url if is_image(obj.proxy_url) else obj.url))
# Finds the best URL for a discord object's icon.
best_url = lambda obj: obj if type(obj) is str else (to_png(obj.avatar_url) if getattr(obj, "avatar_url", None) else obj.url)
# Finds the worst URL for a discord object's icon.
worst_url = lambda obj: obj if type(obj) is str else (to_png_ex(obj.avatar_url) if getattr(obj, "avatar_url", None) else obj.url)


get_author = lambda user, u_id=None: cdict(name=f"{user}" + "" if not u_id else f" ({user.id})", icon_url=best_url(user), url=best_url(user))


# Finds emojis and user mentions in a string.
find_emojis = lambda s: regexp("<.?:[^<>:]+:[0-9]+>").findall(s)
find_users = lambda s: regexp("<@!?[0-9]+>").findall(s)


def get_message_length(message):
    return len(message.system_content or message.content) + sum(len(e) for e in message.embeds) + sum(len(a.url) for a in message.attachments)

def get_message_words(message):
    return len((message.system_content or message.content).split()) + sum(len(e.description.split()) if e.description else 0 + 2 * len(e.fields) if e.fields else 0 for e in message.embeds) + len(message.attachments)

# Returns a string representation of a message object.
def message_repr(message, limit=1024, username=False):
    c = message.content
    s = getattr(message, "system_content", None)
    if s and len(s) > len(c):
        c = s
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
        data = css_md(uni_str("[EMPTY MESSAGE]"))
    return lim_str(data, limit)

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
to_alphanumeric = lambda string: single_space(regexp("[^a-z 0-9]", re.I).sub(" ", unicode_prune(string)))
is_numeric = lambda string: regexp("[0-9]", re.I).search(string)


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
            else:
                for a, b in enumerate(qkey(c)):
                    if b == qlist[a]:
                        return i
                    elif b.startswith(qlist[a]):
                        if not len(b) >= cache[a][0][0]:
                            cache[a][0] = [len(b), i]
                    elif loose and qlist[a] in b:
                        if not len(b) >= cache[a][1][0]:
                            cache[a][1] = [len(b), i]
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

def parse_colour(s, default=None):
    s = single_space(s.replace("#", "").replace(",", " ")).strip()
    # Try to parse as colour tuple first
    if not s:
        if default is None:
            raise ArgumentError("Missing required colour argument.")
        return default
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
            raise ArgumentError("Please input a valid hex colour.")
    return channels


# Gets the string representation of a url object with the maximum allowed image size for discord, replacing webp with png format when possible.
def to_png(url):
    if type(url) is not str:
        url = str(url)
    if url.endswith("?size=1024"):
        url = url[:-10] + "?size=4096"
    return url.replace("/cdn.discordapp.com/", "/media.discordapp.net/").replace(".webp", ".png")

def to_png_ex(url):
    if type(url) is not str:
        url = str(url)
    if url.endswith("?size=1024"):
        url = url[:-10] + "?size=256"
    return url.replace("/cdn.discordapp.com/", "/media.discordapp.net/").replace(".webp", ".png")


# A translator to stip all characters from mentions.
__imap = {
    "#": "",
    "<": "",
    ">": "",
    "@": "",
    "!": "",
    "&": "",
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
find_urls = lambda url: regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+").findall(url)
is_url = lambda url: regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+$").findall(url)
is_discord_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/").findall(url)
is_tenor_url = lambda url: regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[a-zA-Z0-9\\-_]+-[0-9]+").findall(url)
is_imgur_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]\\.)?imgur.com/[a-zA-Z0-9\\-_]+").findall(url)
is_giphy_url = lambda url: regexp("^https?:\\/\\/giphy.com/gifs/[a-zA-Z0-9\\-_]+").findall(url)
is_youtube_url = lambda url: regexp("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+").findall(url)

def is_discord_message_link(url):
    check = url[:64]
    return "channels/" in check and "discord" in check

verify_url = lambda url: url if is_url(url) else url_parse(url)


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
SUBS = cdict(math=cdict(procs=alist(), busy=cdict()), image=cdict(procs=alist(), busy=cdict()))

# Gets amount of processes running in pool.
sub_count = lambda: sum(1 for ptype in SUBS.values() for proc in ptype.procs if proc.is_running())

def force_kill(proc):
    for child in proc.children(recursive=True):
        with suppress():
            child.kill()
            print(child, "killed.")
    print(proc, "killed.")
    return proc.kill()

# Kills all subprocesses in the pool, then restarts it.
def sub_kill():
    for ptype in SUBS.values():
        for proc in ptype.procs:
            with suppress(psutil.NoSuchProcess):
                force_kill(proc)
        ptype.procs.clear()
        ptype.busy.clear()
    proc_update()

# Updates process pools once every 120 seconds.
def proc_updater():
    while True:
        with delay(120):
            proc_update()

# Updates process pool by killing off processes when not necessary, and spawning new ones when required.
def proc_update():
    for pname, ptype in SUBS.items():
        procs = ptype.procs
        b = len(ptype.busy)
        count = sum(1 for proc in procs if utc() >= proc.sem.last)
        if count > 16:
            return
        if b + 1 > count:
            if pname == "math":
                proc = psutil.Popen(
                    [python, "misc/math.py"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            elif pname == "image":
                proc = psutil.Popen(
                    [python, "misc/image.py"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                raise TypeError(f"invalid subpool {pname}.")
            proc.sem = Semaphore(1, 3)
            print(proc)
            procs.append(proc)
        att = 0
        while count >= b + 5:
            found = False
            for p, proc in enumerate(procs):
                # Busy variable indicates when the last operation finished;
                # processes that are idle longer than 1 hour are automatically terminated
                if utc() - proc.sem.last > 3600:
                    force_kill(proc)
                    procs.pop(p)
                    found = True
                    count -= 1
                    break
            att += 1
            if att >= 16 or not found:
                break


def proc_start():
    proc_exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    proc_exc.submit(proc_updater)


# Sends an operation to the math subprocess pool.
async def process_math(expr, prec=64, rat=False, key=None, timeout=12, variables=None):
    if type(key) is not int:
        if key is None:
            key = random.random()
        else:
            key = verify_id(key)
    procs, busy = SUBS.math.procs, SUBS.math.busy
    sem = set_dict(busy, key, Semaphore(2, 3, delay=0.1))
    async with sem:
        with suppress(StopIteration):
            while True:
                for proc in procs:
                    if utc() > proc.sem.last:
                        raise StopIteration
                await create_future(proc_update, priority=True)
                await asyncio.sleep(0.5)
        if variables:
            args = (expr, prec, rat, variables)
        else:
            args = (expr, prec, rat)
        d = repr(bytes("`".join(i if type(i) is str else str(i) for i in args), "utf-8")).encode("utf-8") + b"\n"
        try:
            async with proc.sem:
                await create_future(proc_update, priority=True)
                await create_future(proc.stdin.write, d)
                await create_future(proc.stdin.flush, timeout=timeout)
                resp = await create_future(proc.stdout.readline, timeout=timeout)
        except (T0, T1, T2, OSError):
            create_future_ex(force_kill, proc, priority=True)
            procs.remove(proc)
            busy.pop(key, None)
            create_future_ex(proc_update, priority=True)
            raise
        busy.pop(key, None)
        output = evalEX(evalEX(resp))
        return output

# Sends an operation to the image subprocess pool.
async def process_image(image, operation, args, key=None, timeout=24):
    if type(key) is not int:
        if key is None:
            key = random.random()
        else:
            key = verify_id(key)
    procs, busy = SUBS.image.procs, SUBS.image.busy
    sem = set_dict(busy, key, Semaphore(2, 3, delay=0.1))
    async with sem:
        with suppress(StopIteration):
            while True:
                for proc in procs:
                    if utc() > proc.sem.last:
                        raise StopIteration
                await create_future(proc_update)
                await asyncio.sleep(0.5)
        d = repr(bytes("`".join(str(i) for i in (image, operation, args)), "utf-8")).encode("utf-8") + b"\n"
        try:
            async with proc.sem:
                await create_future(proc_update, priority=True)
                await create_future(proc.stdin.write, d)
                await create_future(proc.stdin.flush, timeout=timeout)
                resp = await create_future(proc.stdout.readline, timeout=timeout)
        except (T0, T1, T2, OSError):
            create_future_ex(force_kill, proc, priority=True)
            procs.remove(proc)
            busy.pop(key, None)
            create_future_ex(proc_update, priority=True)
            raise
        busy.pop(key, None)
        output = evalEX(evalEX(resp))
        print(output)
        return output


# Evaluates an an expression, raising it if it is an exception.
def evalEX(exc):
    try:
        ex = eval(exc)
    except NameError:
        exc = as_str(exc)
        s = exc[exc.index("(") + 1:exc.index(")")]
        with suppress(TypeError, SyntaxError, ValueError):
            s = ast.literal_eval(s)
        ex = RuntimeError(s)
    except:
        print(exc)
        raise
    if issubclass(type(ex), BaseException):
        raise ex
    return ex


# Main event loop for all asyncio operations.
eloop = asyncio.get_event_loop()
__setloop__ = lambda: asyncio.set_event_loop(eloop)


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
                self.pools.append(concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count, initializer=self.initializer))
            while self.pool_count < len(self.pools):
                func = self.pools.popright().shutdown
                self.pools[-1].submit(func, wait=True)

    def update(self):
        if not self.pools:
            self._update()
        self.position = (self.position + 1) % len(self.pools)
        choice(self.pools).submit(self._update)

    def map(self, func, *args, **kwargs):
        self.update()
        return self.pools[self.position].map(func, *args, **kwargs)

    def submit(self, func, *args, **kwargs):
        self.update()
        return self.pools[self.position].submit(func, *args, **kwargs)

    shutdown = lambda self, wait=True: [exc.shutdown(wait) for exc in self.pools].append(self.pools.clear())

pthreads = MultiThreadPool(pool_count=2, thread_count=48, initializer=__setloop__)
athreads = MultiThreadPool(pool_count=2, thread_count=64, initializer=__setloop__)

def get_event_loop():
    with suppress(RuntimeError):
        return asyncio.get_event_loop()
    return eloop

# Creates an asyncio Future that waits on a multithreaded one.
def wrap_future(fut, loop=None):
    if loop is None:
        loop = get_event_loop()
    wrapper = loop.create_future()

    def on_done(*void):
        try:
            res = fut.result()
        except Exception as ex:
            wrapper.set_exception(ex)
        else:
            wrapper.set_result(res)

    fut.add_done_callback(on_done)
    return wrapper

def shutdown_thread_after(thread, fut):
    fut.result()
    return thread.shutdown(wait=True)

def create_thread(func, *args, wait=False, **kwargs):
    thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = thread.submit(func, *args, **kwargs)
    if wait:
        create_future_ex(shutdown_thread_after, thread, fut, priority=True)
    return thread

# Runs a function call in a parallel thread, returning a future object waiting on the output.
def create_future_ex(func, *args, timeout=None, priority=False, **kwargs):
    fut = (athreads, pthreads)[priority].submit(func, *args, **kwargs)
    if timeout is not None:
        fut = (athreads, pthreads)[priority].submit(fut.result, timeout=timeout)
    return fut

# Forces the operation to be a coroutine regardless of whether it is or not. Regular functions are executed in the thread pool.
async def _create_future(obj, *args, loop, timeout, priority, **kwargs):
    if asyncio.iscoroutinefunction(obj):
        obj = obj(*args, **kwargs)
    elif callable(obj):
        if asyncio.iscoroutinefunction(obj.__call__) or not is_main_thread():
            obj = obj.__call__(*args, **kwargs)
        else:
            obj = await wrap_future(create_future_ex(obj, *args, timeout=timeout, **kwargs), loop=loop)
    while awaitable(obj):
        if timeout is not None:
            obj = await asyncio.wait_for(obj, timeout=timeout)
        else:
            obj = await obj
    return obj

# High level future asyncio creation function that takes both sync and async functions, as well as coroutines directly.
def create_future(obj, *args, loop=None, timeout=None, priority=False, **kwargs):
    if loop is None:
        loop = get_event_loop()
    fut = _create_future(obj, *args, loop=loop, timeout=timeout, priority=priority, **kwargs)
    return create_task(fut, loop=loop)

# Creates an asyncio Task object from an awaitable object.
def create_task(fut, *args, loop=None, **kwargs):
    if loop is None:
        loop = get_event_loop()
    return asyncio.ensure_future(fut, *args, loop=loop, **kwargs)

async def _await_fut(fut, ret):
    out = await fut
    ret.set_result(out)
    return ret

# Blocking call that waits for a single asyncio future to complete, do *not* call from main asyncio loop
def await_fut(fut, timeout=None):
    if is_main_thread():
        raise RuntimeError("This function must not be called from the main thread's asyncio loop.")
    ret = concurrent.futures.Future()
    create_task(_await_fut(fut, ret))
    return ret.result(timeout=timeout)

is_main_thread = lambda: threading.current_thread() is threading.main_thread()

# A dummy coroutine that returns None.
async def async_nop(*args, **kwargs):
    return

async def delayed_coro(fut, duration=None):
    async with delay(duration):
        return await fut

async def traceback_coro(fut, *args):
    with tracebacksuppressor(*args):
        return await fut

# A function that takes a coroutine, and calls a second function if it takes longer than the specified delay.
async def delayed_callback(fut, delay, func, *args, exc=False, **kwargs):
    await asyncio.sleep(delay)
    try:
        return fut.result()
    except ISE:
        res = func(*args, **kwargs)
        if awaitable(res):
            await res
        try:
            return await fut
        except:
            if exc:
                raise
    except:
        if exc:
            raise


class open2(io.IOBase):

    __slots__ = ("fp", "fn", "mode")

    def __init__(self, fn, mode="rb"):
        self.fp = None
        self.fn = fn
        self.mode = mode
    
    def __getattribute__(self, k):
        if k in object.__getattribute__(self, "__slots__") or k == "clear":
            return object.__getattribute__(self, k)
        if self.fp is None:
            self.fp = open(self.fn, self.mode)
        return getattr(self.fp, k)

    def clear(self):
        with suppress():
            self.fp.close()
        self.fp = None

class CompatFile(discord.File):

    def __init__(self, fp, filename=None, spoiler=False):
        self.fp = self._fp = fp
        if issubclass(type(fp), io.IOBase):
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
                self.filename = getattr(fp, 'name', None)
        else:
            self.filename = filename
        if spoiler:
            if self.filename is not None:
                if not self.filename.startswith("SPOILER_"):
                    self.filename = "SPOILER_" + self.filename
            else:
                self.filename = "SPOILER_" + "UNKNOWN"
        self.clear = getattr(self.fp, "clear", lambda self: None)

    def reset(self, seek=True):
        if seek:
            self.fp.seek(self._original_pos)

    def close(self):
        self.fp.close = self._closer
        if self._owner:
            self._closer()


class seq(io.IOBase, collections.abc.MutableSequence, contextlib.AbstractContextManager):

    BUF = 262144

    def __init__(self, obj, filename=None):
        self.iter = None
        self.closer = getattr(obj, "close", None)
        if issubclass(type(obj), io.IOBase):
            if issubclass(type(obj), io.BytesIO):
                self.data = obj
            else:
                obj.seek(0)
                self.data = io.BytesIO(obj.read())
                obj.seek(0)
        elif issubclass(type(obj), bytes) or issubclass(type(obj), bytearray) or issubclass(type(obj), memoryview):
            self.data = io.BytesIO(obj)
        elif issubclass(type(obj), collections.abc.Iterator):
            self.iter = iter(obj)
            self.data = io.BytesIO()
            self.high = 0
        elif issubclass(type(obj), requests.models.Response):
            self.iter = obj.iter_content(self.BUF)
            self.data = io.BytesIO()
            self.high = 0
        else:
            raise TypeError(f"a bytes-like object is required, not '{type(obj)}'")
        self.filename = filename
        self.buffer = {}

    def __getitem__(self, k):
        if type(k) is slice:
            out = io.BytesIO()
            start = k.start or 0
            stop = k.stop or inf
            step = k.step or 1
            if step < 0:
                start, stop, step = stop + 1, start + 1, -step
                rev = True
            else:
                rev = False
            curr = start // self.BUF * self.BUF
            offs = start % self.BUF
            out.write(self.load(curr))
            curr += self.BUF
            while curr < stop:
                temp = self.load(curr)
                if not temp:
                    break
                out.write(temp)
                curr += self.BUF
            out.seek(0)
            return out.read()[k]
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

    def __getattr__(self, k):
        if k in ("data", "filename"):
            return self.data
        return object.__getattribute__(self.data, k)

    close = lambda self: self.closer() if self.closer else None
    __exit__ = lambda self, *args: self.close()

    def load(self, k):
        with suppress(KeyError):
            return self.buffer[k]
        seek = getattr(self.data, "seek", None)
        if seek:
            if self.iter is not None and k + self.BUF >= self.high:
                seek(self.high)
                with suppress(StopIteration):
                    while k + self.BUF >= self.high:
                        temp = next(self.iter)
                        self.data.write(temp)
                        self.high += len(temp)
            seek(k)
            self.buffer[k] = self.data.read(self.BUF)
        else:
            with suppress(StopIteration):
                while self.high < k:
                    temp = next(self.data)
                    if not temp:
                        return b""
                    self.buffer[self.high] = temp
                    self.high += self.BUF
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
        self.resp = requests.get(url, stream=True)
        self.iter = self.resp.iter_content(self.BUF)

    def refill(self):
        att = 0
        while self.buflen < self.BUF * 4:
            try:
                self.buf.write(next(self.iter))
            except StopIteration:
                return
            except:
                if att > 16:
                    raise
                att += 1
                self.reset()


# Manages both sync and async get requests.
class RequestManager(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, collections.abc.Callable):

    session = None
    semaphore = emptyctx

    @classmethod
    def header(cls):
        return {
            "User-Agent": f"Mozilla/5.{xrand(1, 10)}",
            "DNT": "1",
        }
    headers = header

    async def _init_(self):
        self.session = aiohttp.ClientSession(loop=eloop)
        self.semaphore = Semaphore(512, 256, delay=0.25)

    async def aio_call(self, url, headers, files, data, method, decode=False, json=False):
        if files is not None:
            raise NotImplementedError("Unable to send multipart files asynchronously.")
        async with self.semaphore:
            async with getattr(self.session, method)(url, headers=headers, data=data) as resp:
                if resp.status >= 400:
                    data = await resp.read()
                    raise ConnectionError(f"Error {resp.status}", url, as_str(data))
                if json:
                    return await resp.json()
                data = await resp.read()
                if decode:
                    return as_str(data)
                return data

    def __call__(self, url, headers={}, files=None, data=None, raw=False, timeout=8, method="get", decode=False, json=False, bypass=True, aio=False):
        if bypass:
            if "user-agent" not in headers:
                headers["User-Agent"] = f"Mozilla/5.{xrand(1, 10)}"
            headers["DNT"] = "1"
        method = method.casefold()
        if aio:
            return create_task(asyncio.wait_for(self.aio_call(url, headers, files, data, method, decode, json), timeout=timeout))
        with self.semaphore:
            with getattr(requests, method)(url, headers=headers, files=files, data=data, stream=True, timeout=timeout) as resp:
                if resp.status_code >= 400:
                    raise ConnectionError(f"Error {resp.status_code}", url, resp.text)
                if json:
                    return resp.json()
                if raw:
                    data = resp.raw.read()
                else:
                    data = resp.content
                if decode:
                    return as_str(data)
                return data

    def __exit__(self, *args):
        self.session.close()

    def __aexit__(self, *args):
        self.session.close()
        return async_nop()

Request = RequestManager()
create_task(Request._init_())


def load_emojis():
    global emoji_translate, emoji_replace, em_trans
    with tracebacksuppressor:
        resp = Request("https://raw.githubusercontent.com/BreadMoirai/DiscordEmoji/master/src/main/java/com/github/breadmoirai/Emoji.java", decode=True, timeout=None)
        e_resp = [line.strip()[:-1] for line in resp[resp.index("public enum Emoji {") + len("public enum Emoji {"):resp.index("private static final Emoji[] SORTED;")].strip().split("\n")]
        e_data = {safe_eval(words[0]).encode("utf-16", "surrogatepass").decode("utf-16"): f" {safe_eval(words[2][:-1])} " for emoji in e_resp for words in (emoji.strip(";")[emoji.index("\\u") - 1:].split(","),) if words[2][:-1].strip() != "null"}
        with open("misc/emojis.txt", "r", encoding="utf-8") as f:
            resp = f.read()
        e_data.update({k: v for k, v in (line.split(" ", 1) for line in resp.splitlines())})
        emoji_translate = {k: v for k, v in e_data.items() if len(k) == 1}
        emoji_replace = {k: v for k, v in e_data.items() if len(k) > 1}
        em_trans = "".maketrans(emoji_translate)

def translate_emojis(s):
    return s.translate(em_trans)

def replace_emojis(s):
    for emoji, url in emoji_replace.items():
        if emoji in s:
            s = s.replace(emoji, url)
    return s

def find_emojis_ex(s):
    out = deque()
    for emoji, url in emoji_replace.items():
        if emoji in s:
            out.append(url[1:-1])
    for emoji, url in emoji_translate.items():
        if emoji in s:
            out.append(url[1:-1])
    return list(set(out))

create_future_ex(load_emojis, priority=True)


# Stores and manages timezones information.
TIMEZONES = cdict()

def load_timezones():
    with tracebacksuppressor():
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

create_future_ex(load_timezones, priority=True)

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
                return DynamicDT.fromdatetime(tparser.parse(expr)).set_offset(offs)
        elif s.startswith("Python int too large to convert to C"):
            y = int(regexp("[0-9]{10,}").findall(expr)[0])
            if bc:
                y = -y
            offs, year = divmod(y, 400)
            offs = offs * 400 - 2000
            year += 2000
            expr = regexp("[0-9]{10,}").sub(str(year), expr, 1)
            return DynamicDT.fromdatetime(tparser.parse(expr)).set_offset(offs)
        elif s.startswith("Unknown string format") or s.startswith("month must be in"):
            y = int(regexp("[0-9]{5,}").findall(expr)[0])
            if bc:
                y = -y
            offs, year = divmod(y, 400)
            offs = offs * 400 - 2000
            year += 2000
            expr = regexp("[0-9]{5,}").sub(str(year), expr, 1)
            return DynamicDT.fromdatetime(tparser.parse(expr)).set_offset(offs)
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
        s = int(expr.split(".", 1)[0])
    return utc_dft(s)


__filetrans = {
    "\\": "_",
    "/": "_",
    " ": "%20",
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
    rate_limit = 0

    def perm_error(self, perm, req=None, reason=None):
        if req is None:
            req = self.min_level
        if reason is None:
            reason = f"for command {self.name[-1]}"
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

    def __init__(self, bot, catg):
        name = self.name
        self.__name__ = self.__class__.__name__
        fhp = "saves/" + name
        if not getattr(self, "no_file", False):
            if os.path.exists(fhp):
                data = self.data = FileHashDict(path=fhp)
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
                    data = FileHashDict(data, path=fhp)
                    data.modified.update(data.data.keys())
                    self.data = data
                except FileNotFoundError:
                    data = None
        else:
            data = self.data = {}
        if data is None:
            self.data = FileHashDict(path=fhp)
        if not issubclass(type(self.data), collections.abc.MutableMapping):
            self.data = FileHashDict(dict.fromkeys(self.data), path=fhp)
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
                self.data.clear()
                f()

    __hash__ = lambda self: hash(self.__name__)
    __str__ = lambda self: f"Database <{self.__name__}>"
    __repr__ = lambda self: repr(self.data)
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
    clear = lambda self: self.data.clear()
    setdefault = lambda self, k, v: self.data.setdefault(k, v)
    keys = lambda self: self.data.keys()
    discard = lambda self, k: self.data.pop(k, None)

    def update(self, modified=None, force=False):
        if hasattr(self, "no_file"):
            return
        if force:
            try:
                self.data.__update__()
            except:
                print(self, self.data, traceback.format_exc(), sep="\n", end="")
        else:
            if modified is None:
                self.data.modified.update(self.data.keys())
            else:
                if issubclass(type(modified), collections.abc.Sized) and type(modified) not in (str, bytes):
                    self.data.modified.update(modified)
                else:
                    self.data.modified.add(modified)
        return False
    
    def unload(self):
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


# Redirects all print operations to target files, limiting the amount of operations that can occur in any given amount of time for efficiency.
class __logPrinter:

    def __init__(self, file=None):
        self.buffer = self
        self.data = {}
        self.history = {}
        self.counts = {}
        self.funcs = alist()
        self.file = file
        self.closed = True

    def start(self):
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.exec.submit(self.update_print)
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
                    f.write(as_str(b))
        except:
            sys.__stdout__.write(traceback.format_exc())
    
    def update_print(self):
        if self.file is None:
            return
        outfunc = lambda s: self.file_print(self.file, s)
        enc = lambda x: bytes(x, "utf-8")
        while True:
            with delay(1):
                try:
                    for f in tuple(self.data):
                        if not self.data[f]:
                            self.data.pop(f)
                            continue
                        out = lim_str(self.data[f], 65536)
                        data = enc(self.data[f])
                        self.data[f] = ""
                        if self.funcs:
                            [func(out) for func in self.funcs]
                        if f == self.file:
                            outfunc(data)
                        else:
                            self.file_print(f, data)
                except:
                    sys.__stdout__.write(traceback.format_exc())
            while not os.path.exists("common.py") or self.closed:
                time.sleep(0.5)

    def __call__(self, *args, sep=" ", end="\n", prefix="", file=None, **void):
        out = str(sep).join(i if type(i) is str else str(i) for i in args) + str(end) + str(prefix)
        if not out:
            return
        if args and type(args[0]) is str and args[0].startswith("WARNING:"):
            return sys.__stdout__.write(out)
        if file is None:
            file = self.file
        if file not in self.data:
            self.data[file] = ""
        temp = out.strip()
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
    flush = open = lambda self: (self, self.__setattr__("closed", False))[0]
    close = lambda self: self.__setattr__("closed", True)
    isatty = lambda self: False


PRINT = __logPrinter("log.txt")

# Sets all instances of print to the custom print implementation.

# sys.stdout = sys.stderr = print
# for mod in (discord, concurrent.futures, asyncio.futures, asyncio, psutil, subprocess, tracemalloc):
#     builtins = getattr(mod, "__builtins__", None)
#     if builtins:
#         builtins["print"] = print