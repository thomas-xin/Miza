import smath
from smath import *

with MultiThreadedImporter(globals()) as importer:
    importer.__import__("os", "importlib", "inspect", "tracemalloc", "psutil", "subprocess", "asyncio", "discord", "json", "pytz", "requests", "aiohttp", "psutil")

PROC = psutil.Process()
quit = lambda *args, **kwargs: PROC.kill()

tracemalloc.start()

import urllib.request, urllib.parse

python = ("python3", "python")[os.name == "nt"]
urlParse = urllib.parse.quote
escape_markdown = discord.utils.escape_markdown
escape_everyone = lambda s: s.replace("@everyone", "@\xadeveryone").replace("@here", "@\xadhere").replace("<@&", "<@\xad&")
time_snowflake = discord.utils.time_snowflake
snowflake_time = discord.utils.snowflake_time


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


class ArgumentError(LookupError):
    pass
class TooManyRequests(PermissionError):
    pass


# Safer than raw eval, more powerful than json.decode
def eval_json(s):
    with suppress(json.JSONDecodeError):
        return json.loads(s)
    return safe_eval(s)


# Decodes HTML encoded characters in a string.
def htmlDecode(s):
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

noHighlight = lambda s: str(s).translate(__emap)
clrHighlight = lambda s: str(s).translate(__emap2)
sbHighlight = lambda s: "[" + noHighlight(s) + "]"


# Counts the number of lines in a file.
def getLineCount(fn):
    with open(fn, "rb") as f:
        count = 1
        size = 0
        while True:
            try:
                i = f.read(32768)
                if not i:
                    raise EOFError
                size += len(i)
                count += i.count(b"\n")
            except EOFError:
                return hlist((size, count))


# Checks if a file is a python code file using its filename extension.
iscode = lambda fn: str(fn).endswith(".py") or str(fn).endswith(".pyw")


# Checks if an object can be used in "await" operations.
awaitable = lambda obj: hasattr(obj, "__await__") or issubclass(type(obj), asyncio.Future) or issubclass(type(obj), asyncio.Task) or inspect.isawaitable(obj)

# Async function that waits for a given time interval if the result of the input coroutine is None.
async def waitOnNone(coro, seconds=0.5):
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
            item[i] = create_task(recursiveCoro(obj))
    for i, obj in enumerate(item):
        if hasattr(obj, "__await__"):
            try:
                item[i] = await obj
            except:
                pass
    return item


# Sends a message to a channel, then adds reactions accordingly.
async def sendReact(channel, *args, reacts=(), **kwargs):
    with tracebacksuppressor:
        sent = await channel.send(*args, **kwargs)
        for react in reacts:
            await sent.add_reaction(react)

# Sends a message to a channel, then edits to add links to all attached files.
async def sendFile(channel, msg, file, filename=None, best=False):
    try:
        message = await channel.send(msg, file=file)
        if filename is not None:
            create_future_ex(os.remove, filename, priority=True)
    except:
        if filename is not None:
            create_future_ex(os.remove, filename, priority=True)
        raise
    if message.attachments:
        await message.edit(content=message.content + ("" if message.content.endswith("```") else "\n") + ("\n".join("<" + bestURL(a) + ">" for a in message.attachments) if best else "\n".join("<" + a.url + ">" for a in message.attachments)))


# Finds the best URL for a discord object's icon.
bestURL = lambda obj: obj if type(obj) is str else (strURL(obj.avatar_url) if getattr(obj, "avatar_url", None) else (obj.proxy_url if obj.proxy_url else obj.url))


# Finds emojis and user mentions in a string.
emojiFind = re.compile("<.?:[^<>:]+:[0-9]+>")
findEmojis = lambda s: re.findall(emojiFind, s)
userFind = re.compile("<@!?[0-9]+>")
findUsers = lambda s: re.findall(userFind, s)


# Returns a string representation of a message object.
def strMessage(message, limit=1024, username=False):
    c = message.content
    s = getattr(message, "system_content", None)
    if s and len(s) > len(c):
        c = s
    if username:
        c = "<@" + str(message.author.id) + ">:\n" + c
    data = limStr(c, limit)
    if message.attachments:
        data += "\n[" + ", ".join(i.url for i in message.attachments) + "]"
    if message.embeds:
        data += "\n⟨" + ", ".join(str(i.to_dict()) for i in message.embeds) + "⟩"
    if message.reactions:
        data += "\n{" + ", ".join(str(i) for i in message.reactions) + "}"
    try:
        t = message.created_at
        if message.edited_at:
            t = message.edited_at
        data += "\n`(" + str(t) + ")`"
    except AttributeError:
        pass
    if not data:
        data = "```css\n" + uniStr("[EMPTY MESSAGE]") + "```"
    return limStr(data, limit)

# Returns a string representation of an activity object.
def strActivity(activity):
    if hasattr(activity, "type") and activity.type != discord.ActivityType.custom:
        t = activity.type.name
        return t[0].upper() + t[1:] + " " + activity.name
    return str(activity)


# Alphanumeric string regular expression.
atrans = re.compile("[^a-z 0-9]", re.I)
ntrans = re.compile("[0-9]", re.I)
is_alphanumeric = lambda string: not re.search(atrans, string)
to_alphanumeric = lambda string: singleSpace(re.sub(atrans, " ", reconstitute(string)))
is_numeric = lambda string: re.search(ntrans, string)


# Strips code box from the start and end of a message.
def noCodeBox(s):
    if s.startswith("```") and s.endswith("```"):
        s = s[s.index("\n") + 1:-3]
    return s


# A string lookup operation with an iterable, multiple attempts, and sorts by priority.
async def strLookup(it, query, ikey=lambda x: [str(x)], qkey=lambda x: [str(x)], loose=True):
    queries = qkey(query)
    qlist = [q for q in queries if q]
    if not qlist:
        qlist = list(queries)
    cache = [[[inf, None], [inf, None]] for _ in qlist]
    for x, i in enumerate(shuffle(it), 1):
        for c in ikey(i):
            if not c and i:
                continue
            for a, b in enumerate(qkey(c)):
                if b == qlist[a]:
                    return i
                elif b.startswith(qlist[a]):
                    if len(b) < cache[a][0][0]:
                        cache[a][0] = [len(b), i]
                elif loose and qlist[a] in b:
                    if len(b) < cache[a][1][0]:
                        cache[a][1] = [len(b), i]
        if not x & 1023:
            await asyncio.sleep(0.1)
    for c in cache:
        if c[0][0] < inf:
            return c[0][1]
    if loose:
        for c in cache:
            if c[1][0] < inf:
                return c[1][1]
    raise LookupError("No results for " + str(query) + ".")


# Generates a random colour across the spectrum, in intervals of 128.
randColour = lambda: colour2Raw(colourCalculation(xrand(12) * 128))


# Gets the string representation of a url object with the maximum allowed image size for discord, replacing webp with png format when possible.
def strURL(url):
    if type(url) is not str:
        url = str(url)
    if url.endswith("?size=1024"):
        url = url[:-10] + "?size=4096"
    return url.replace(".webp", ".png")


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

def verifyID(value):
    with suppress(ValueError):
        return int(str(value).translate(__itrans))
    return value


# Strips <> characters from URLs.
def stripAcc(url):
    if url.startswith("<") and url[-1] == ">":
        s = url[1:-1]
        if isURL(s):
            return s
    return url
__smap = {"|": "", "*": ""}
__strans = "".maketrans(__smap)
verifySearch = lambda f: stripAcc(singleSpace(f.strip().translate(__strans)))
urlFind = re.compile("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+")
urlIs = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
findURLs = lambda url: re.findall(urlFind, url)
isURL = lambda url: re.search(urlIs, url)
verifyURL = lambda url: url if isURL(url) else urllib.parse.quote(url)


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
    if "." in url:
        url = url[url.rindex("."):]
    url = url.casefold()
    return IMAGE_FORMS.get(url)


# Subprocess pool for resource-consuming operations.
SUBS = cdict(math=cdict(procs=hlist(), busy=cdict()), image=cdict(procs=hlist(), busy=cdict()))

# Gets amount of processes running in pool.
subCount = lambda: sum(1 for ptype in SUBS.values() for proc in ptype.procs if proc.is_running())

def forceKill(proc):
    for child in proc.children(recursive=True):
        try:
            child.kill()
        except:
            pass
        else:
            print(child, "killed.")
    print(proc, "killed.")
    return proc.kill()

# Kills all subprocesses in the pool, then restarts it.
def subKill():
    for ptype in SUBS.values():
        for proc in ptype.procs:
            with suppress(psutil.NoSuchProcess):
                forceKill(proc)
        ptype.procs.clear()
        ptype.busy.clear()
    procUpdate()

# Updates process pools once every 120 seconds.
def procUpdater():
    while True:
        procUpdate()
        time.sleep(120)

# Updates process pool by killing off processes when not necessary, and spawning new ones when required.
def procUpdate():
    for pname, ptype in SUBS.items():
        procs = ptype.procs
        b = len(ptype.busy)
        count = sum(1 for proc in procs if utc() > proc.busy)
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
                raise TypeError("invalid subpool " + pname)
            proc.busy = nan
            x = bytes(random.randint(0, 255) for _ in loop(32))
            if random.randint(0, 1):
                x = hashlib.sha256(x).digest()
            x = base64.b64encode(x)
            proc.stdin.write(bytes(repr(x) + "\n", "utf-8"))
            proc.key = x.decode("utf-8", "replace")
            proc.busy = utc()
            print(proc, "initialized with key", proc.key)
            procs.append(proc)
        att = 0
        while count > b + 2:
            found = False
            for p, proc in enumerate(procs):
                # Busy variable indicates when the last operation finished;
                # processes that are idle longer than 1 hour are automatically terminated
                if utc() - proc.busy > 3600:
                    forceKill(proc)
                    procs.pop(p)
                    found = True
                    count -= 1
                    break
            att += 1
            if att >= 16 or not found:
                break


def proc_start():
    proc_exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    proc_exc.submit(procUpdater)


# Sends an operation to the math subprocess pool.
async def mathProc(expr, prec=64, rat=False, key=None, timeout=12, authorize=False):
    if type(key) is not int:
        if key is None:
            key = random.random()
        else:
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = key.id
    procs, busy = SUBS.math.procs, SUBS.math.busy
    while utc() - busy.get(key, 0) < 60:
        await asyncio.sleep(0.5)
    try:
        while True:
            for p in range(len(procs)):
                if p < len(procs):
                    proc = procs[p]
                    if utc() > proc.busy:
                        raise StopIteration
                else:
                    break
            await create_future(procUpdate, priority=True)
            await asyncio.sleep(0.5)
    except StopIteration:
        pass
    if authorize:
        args = (expr, prec, rat, proc.key)
    else:
        args = (expr, prec, rat)
    d = repr(bytes("`".join(i if type(i) is str else str(i) for i in args), "utf-8")).encode("utf-8") + b"\n"
    try:
        proc.busy = inf
        busy[key] = utc()
        await create_future(procUpdate, priority=True)
        await create_future(proc.stdin.write, d)
        await create_future(proc.stdin.flush)
        resp = await create_future(proc.stdout.readline, timeout=timeout)
        proc.busy = utc()
    except (T0, T1):
        create_future_ex(forceKill, proc, priority=True)
        procs.pop(p, None)
        busy.pop(key, None)
        create_future_ex(procUpdate, priority=True)
        raise
    busy.pop(key, None)
    output = evalEX(evalEX(resp))
    return output

# Sends an operation to the image subprocess pool.
async def imageProc(image, operation, args, key=None, timeout=24):
    if type(key) is not int:
        if key is None:
            key = random.random()
        else:
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = key.id
    procs, busy = SUBS.image.procs, SUBS.image.busy
    while utc() - busy.get(key, 0) < 60:
        await asyncio.sleep(0.5)
    with suppress(StopIteration):
        while True:
            for p in range(len(procs)):
                if p < len(procs):
                    proc = procs[p]
                    if utc() > proc.busy:
                        raise StopIteration
                else:
                    break
            await create_future(procUpdate)
            await asyncio.sleep(0.5)
    d = repr(bytes("`".join(str(i) for i in (image, operation, args)), "utf-8")).encode("utf-8") + b"\n"
    try:
        proc.busy = inf
        busy[key] = utc()
        await create_future(procUpdate, priority=True)
        await create_future(proc.stdin.write, d)
        await create_future(proc.stdin.flush)
        resp = await create_future(proc.stdout.readline, timeout=timeout)
        proc.busy = utc()
    except (T0, T1):
        create_future_ex(forceKill, proc, priority=True)
        procs.pop(p, None)
        busy.pop(key, None)
        create_future_ex(procUpdate, priority=True)
        raise
    busy.pop(key, None)
    output = evalEX(evalEX(resp))
    return output


# Evaluates an an expression, raising it if it is an exception.
def evalEX(exc):
    try:
        ex = eval(exc)
    except NameError:
        if type(exc) is bytes:
            exc = exc.decode("utf-8", "replace")
        s = exc[exc.index("(") + 1:exc.index(")")]
        try:
            s = ast.literal_eval(s)
        except:
            pass
        ex = RuntimeError(s)
    except:
        print(exc)
        raise
    if issubclass(type(ex), Exception):
        raise ex
    return ex


# Main event loop for all asyncio operations.
eloop = asyncio.get_event_loop()
__setloop__ = lambda: asyncio.set_event_loop(eloop)


# Thread pool manager for multithreaded operations.
class MultiThreadPool(collections.abc.Sized, concurrent.futures.Executor):

    def __init__(self, pool_count=3, thread_count=64, initializer=None):
        self.pools = hlist()
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
        random.choice(self.pools).submit(self._update)

    def map(self, func, *args, **kwargs):
        self.update()
        return self.pools[self.position].map(func, *args, **kwargs)

    def submit(self, func, *args, **kwargs):
        self.update()
        return self.pools[self.position].submit(func, *args, **kwargs)

    shutdown = lambda self, wait=True: [exc.shutdown(wait) for exc in self.pools].append(self.pools.clear())

pthreads = MultiThreadPool(thread_count=48, initializer=__setloop__)
athreads = MultiThreadPool(thread_count=64, initializer=__setloop__)
__setloop__()

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
            loop.call_soon_threadsafe(wrapper.set_exception, ex)
        else:
            loop.call_soon_threadsafe(wrapper.set_result, res)

    fut.add_done_callback(on_done)
    return wrapper

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
        if asyncio.iscoroutinefunction(obj.__call__):
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
    fut = _create_future(obj, *args, loop=loop, timeout=None, priority=priority, **kwargs)
    return create_task(fut, loop=loop)

# Creates an asyncio Task object from an awaitable object.
def create_task(fut, *args, loop=None, **kwargs):
    if loop is None:
        loop = get_event_loop()
    return asyncio.ensure_future(fut, *args, loop=loop, **kwargs)

# A dummy coroutine that returns None.
async def retNone(*args, **kwargs):
    return

# A function that takes a coroutine, and calls a second function if it takes longer than the specified delay.
async def delayed_callback(fut, delay, func, *args, exc=False, **kwargs):
    await asyncio.sleep(delay)
    try:
        return fut.result()
    except asyncio.exceptions.InvalidStateError:
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

# Manages both sync and async get requests.
class AutoRequest:

    async def _init_(self):
        self.session = aiohttp.ClientSession()
        self.semaphore = asyncio.Semaphore(512)

    async def aio_call(self, url, headers, data, decode):
        async with self.semaphore:
            async with self.session.get(url, headers=headers, data=data) as resp:
                if resp.status >= 400:
                    data = await resp.read()
                    raise ConnectionError("Error " + str(resp.status) + ": " + data.decode("utf-8", "replace"))
                data = await resp.read()
                if decode:
                    return data.decode("utf-8", "replace")
                return data

    def __call__(self, url, headers={}, data=None, raw=False, timeout=8, bypass=True, decode=False, aio=False):
        if bypass and "user-agent" not in headers:
            headers["user-agent"] = "Mozilla/5." + str(xrand(1, 10))
        if aio:
            return create_task(asyncio.wait_for(self.aio_call(url, headers, data, decode), timeout=timeout))
        with requests.get(url, headers=headers, data=data, stream=True, timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise ConnectionError("Error " + str(resp.status_code) + ": " + resp.text)
            if raw:
                data = resp.raw.read()
            else:
                data = resp.content
            if decode:
                return data.decode("utf-8", "replace")
            return data

Request = AutoRequest()
create_task(Request._init_())


# Stores and manages timezones information.
TIMEZONES = cdict()

def load_timezones():
    with open("misc/timezones.txt", "rb") as f:
        data = f.read().decode("utf-8", "replace")
        for line in data.split("\n"):
            info = line.split("\t")
            abb = info[0].casefold()
            if len(abb) >= 3 and (abb not in TIMEZONES or "(unofficial)" not in info[1]):
                temp = info[-1].replace("\\", "/")
                curr = sorted([round((1 - (i[3] == "−") * 2) * (rdhms(i[4:]) if ":" in i else float(i[4:]) * 60) * 60) for i in temp.split("/") if i.startswith("UTC")])
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
    if issubclass(type(s), collections.abc.Sequence):
        return s[is_dst(timezone=tz.upper())]
    return s

create_future_ex(load_timezones, priority=True)

def parse_with_now(expr):
    if expr.strip().casefold() == "now":
        return datetime.datetime.utcnow()
    return tparser.parse(expr)

# Parses a time expression, with an optional timezone input at the end.
def tzparse(expr):
    try:
        s = float(expr)
    except ValueError:
        expr = expr.strip()
        if " " in expr:
            t = 0
            try:
                args = shlex.split(expr)
            except ValueError:
                args = expr.split()
            for arg in (args[0], args[-1]):
                a = arg
                h = 0
                for op in "+-":
                    try:
                        i = arg.index(op)
                    except ValueError:
                        continue
                    a = arg[:i]
                    h += float(arg[i:])
                tz = a.casefold()
                if tz in TIMEZONES:
                    t = get_timezone(tz)
                    expr = expr.replace(arg, "")
                    break
                h = 0
            return parse_with_now(expr) - datetime.timedelta(hours=h, seconds=t)
        return parse_with_now(expr)
    return datetime.datetime.utcfromtimestamp(s)


# Basic inheritable class for all bot commands.
class Command(collections.abc.Hashable, collections.abc.Callable):
    min_level = -inf
    rate_limit = 0
    description = ""
    usage = ""

    def permError(self, perm, req=None, reason=None):
        if req is None:
            req = self.min_level
        if reason is None:
            reason = "for command " + self.name[-1]
        return PermissionError(
            "Insufficient priviliges " + str(reason)
            + ". Required level: " + str(req)
            + ", Current level: " + str(perm) + "."
        )

    def __init__(self, bot, catg):
        self.used = {}
        if not hasattr(self, "data"):
            self.data = cdict()
        if not hasattr(self, "name"):
            self.name = []
        self.__name__ = self.__class__.__name__
        if not hasattr(self, "alias"):
            self.alias = self.name
        else:
            self.alias.append(self.__name__)
        self.name.append(self.__name__)
        if not hasattr(self, "min_display"):
            self.min_display = self.min_level
        for a in self.alias:
            b = a.replace("*", "").replace("_", "").replace("||", "")
            if b:
                a = b
            a = a.casefold()
            if a in bot.commands:
                bot.commands[a].append(self)
            else:
                bot.commands[a] = hlist([self])
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

    __hash__ = lambda self: hash(self.__name__)
    __str__ = lambda self: self.__name__
    __call__ = lambda self, **void: None


# Basic inheritable class for all bot databases.
class Database(collections.abc.Hashable, collections.abc.Callable):
    bot = None
    rate_limit = 3
    name = "data"

    def __init__(self, bot, catg):
        self.used = utc()
        name = self.name
        self.__name__ = self.__class__.__name__
        if not getattr(self, "no_file", False):
            self.file = "saves/" + name + ".json"
            self.updated = False
            try:
                with open(self.file, "rb") as f:
                    s = f.read()
                if not s:
                    raise FileNotFoundError
                data = None
                with suppress(pickle.UnpicklingError):
                    data = pickle.loads(s)
                if type(data) in (str, bytes):
                    data = eval(compile(data, "<database>", "eval", optimize=2))
                if data is None:
                    try:
                        data = eval(compile(s, "<database>", "eval", optimize=2))
                    except:
                        print(self.file)
                        print_exc()
                        raise FileNotFoundError
                bot.data[name] = self.data = data
            except FileNotFoundError:
                data = None
        else:
            data = None
        if not data:
            bot.data[name] = self.data = cdict()
        bot.database[name] = self
        self.catg = catg
        self.bot = bot
        self.busy = self.checking = False
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
    __str__ = lambda self: self.__name__
    __call__ = lambda self, **void: None

    def update(self, force=False):
        if not hasattr(self, "updated"):
            self.updated = False
        if force:
            name = getattr(self, "name", None)
            if name:
                if self.updated:
                    self.updated = False
                    try:
                        s = str(self.data)
                    except:
                        s = None
                    if not s or len(s) > 262144:
                        # print("Pickling " + name + "...")
                        s = pickle.dumps(self.data)
                    else:
                        s = s.encode("utf-8")
                    with open(self.file, "wb") as f:
                        f.write(s)
                    return True
        else:
            self.updated = True
        return False


# Redirects all print operations to target files, limiting the amount of operations that can occur in any given amount of time for efficiency.
class _logPrinter:

    def __init__(self, file=None):
        self.buffer = self
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.data = {}
        self.funcs = hlist()
        self.file = file
        self.future = self.exec.submit(self.updatePrint)
        self.closed = False

    def filePrint(self, fn, b):
        try:
            if type(fn) not in (str, bytes):
                f = fn
            if type(b) in (bytes, bytearray):
                f = open(fn, "ab")
            elif type(b) is str:
                f = open(fn, "a", encoding="utf-8")
            else:
                f = fn
            with closing(f):
                f.write(b)
        except:
            sys.__stdout__.write(traceback.format_exc())
    
    def updatePrint(self):
        if self.file is None:
            outfunc = sys.__stdout__.write
            enc = lambda x: x
        else:
            outfunc = lambda s: (sys.__stdout__.buffer.write(s), self.filePrint(self.file, s))
            enc = lambda x: bytes(x, "utf-8")
        while True:
            try:
                for f in tuple(self.data):
                    if not self.data[f]:
                        self.data.pop(f)
                        continue
                    out = limStr(self.data[f], 8192)
                    self.data[f] = ""
                    data = enc(out)
                    if self.funcs:
                        [func(out) for func in self.funcs]
                    if f == self.file:
                        outfunc(data)
                    else:
                        self.filePrint(f, data)
            except:
                sys.__stdout__.write(traceback.format_exc())
            time.sleep(1)
            while not os.path.exists("common.py") or self.closed:
                time.sleep(0.5)

    def __call__(self, *args, sep=" ", end="\n", prefix="", file=None, **void):
        if file is None:
            file = self.file
        if file not in self.data:
            self.data[file] = ""
        self.data[file] += str(sep).join(i if type(i) is str else str(i) for i in args) + str(end) + str(prefix)

    def write(self, *args, end="", **kwargs):
        args2 = [arg if type(arg) is str else arg.decode("utf-8", "replace") for arg in args]
        return self.__call__(*args2, end=end, **kwargs)

    read = lambda self, *args, **kwargs: bytes()
    flush = open = lambda self: (self, self.__setattr__("closed", False))[0]
    close = lambda self: self.__setattr__("closed", True)
    isatty = lambda self: False


PRINT = _logPrinter("log.txt")

# Sets all instances of print to the custom print implementation.

# sys.stdout = sys.stderr = print
# for mod in (discord, concurrent.futures, asyncio.futures, asyncio, psutil, subprocess, tracemalloc):
#     builtins = getattr(mod, "__builtins__", None)
#     if builtins:
#         builtins["print"] = print