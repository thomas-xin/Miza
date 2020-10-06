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
        "pytz",
        "requests",
        "aiohttp",
        "psutil",
        "threading",
        "urllib",
        "zipfile",
    )

PROC = psutil.Process()
quit = lambda *args, **kwargs: PROC.kill()

tracemalloc.start()

from zipfile import ZipFile
import urllib.request, urllib.parse

python = ("python3", "python")[os.name == "nt"]
url_parse = urllib.parse.quote
escape_markdown = discord.utils.escape_markdown
escape_mentions = discord.utils.escape_mentions
escape_everyone = lambda s: s.replace("@everyone", "@\xadeveryone").replace("@here", "@\xadhere").replace("<@&", "<@\xad&")
time_snowflake = discord.utils.time_snowflake
snowflake_time = discord.utils.snowflake_time


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
        return self.active >= self.limit or len(self.rate_bin) >= self.limit

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
send_exception = lambda sendable, ex: send_with_react(sendable, exc_repr(ex), reacts="❎")


# A context manager that sends exception tracebacks to a sendable.
class ExceptionSender(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, sendable, *args, **kwargs):
        self.sendable = sendable
        self.exceptions = args + tuple(kwargs.values())
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    create_task(send_exception(self.sendable, exc_value))
                    return True
            create_task(send_exception(self.sendable, exc_value))
            with tracebacksuppressor:
                raise exc_value
        return True

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    await send_exception(self.sendable, exc_value)
                    return True
            await send_exception(self.sendable, exc_value)
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
    with suppress(json.JSONDecodeError):
        return json.loads(s)
    return safe_eval(s)


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


def get_folder_size(folder):
    folder = folder.strip("/") + "/"
    size = 0
    for file in os.listdir(folder):
        with suppress(FileNotFoundError, PermissionError):
            size += os.path.getsize(folder + file)
    return size


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


# Sends a message to a channel, then adds reactions accordingly.
async def send_with_react(channel, *args, reacts=(), **kwargs):
    with tracebacksuppressor:
        sent = await channel.send(*args, **kwargs)
        for react in reacts:
            async with delay(1 / 3):
                create_task(sent.add_reaction(react))


# Creates and starts a coroutine for typing in a channel.
typing = lambda self: create_task(self.trigger_typing())


# Finds the best URL for a discord object's icon.
best_url = lambda obj: obj if type(obj) is str else (to_png(obj.avatar_url) if getattr(obj, "avatar_url", None) else (obj.proxy_url if obj.proxy_url else obj.url))


get_author = lambda user, u_id=None: cdict(name=f"{user}" + "" if not u_id else f" ({user.id})", icon_url=best_url(user), url=best_url(user))


# Finds emojis and user mentions in a string.
find_emojis = lambda s: regexp("<.?:[^<>:]+:[0-9]+>").findall(s)
find_users = lambda s: regexp("<@!?[0-9]+>").findall(s)


def get_message_length(message):
    return len(message.system_content) + sum(len(e) for e in message.embeds) + sum(len(a.url) for a in message.attachments)

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
        if not x & 1023:
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

def verify_id(obj):
    if type(obj) is int:
        return obj
    if type(obj) is str:
        with suppress(ValueError):
            return int(str(obj).translate(__itrans))
        return obj
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
find_urls = lambda url: regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+").findall(url)
is_url = lambda url: regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$").findall(url)
is_discord_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/").findall(url)
is_tenor_url = lambda url: regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[a-zA-Z0-9\\-_]+-[0-9]+").findall(url)
is_imgur_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]\\.)?imgur.com/[a-zA-Z0-9\\-_]+").findall(url)
is_giphy_url = lambda url: regexp("^https?:\\/\\/giphy.com/gifs/[a-zA-Z0-9\\-_]+").findall(url)

def is_discord_message_link(url):
    check = url[:64]
    return "channels/" in check and "discord" in check

verify_url = lambda url: url if is_url(url) else urllib.parse.quote(url)


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

VIDEO_FORMS = {
    ".webm": True,
    ".mkv": True,
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
            with proc.sem:
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
            with proc.sem:
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


# Evaluates an an expression, raising it if it is an exception.
def evalEX(exc):
    try:
        ex = eval(exc)
    except NameError:
        if type(exc) is bytes:
            exc = exc.decode("utf-8", "replace")
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
            loop.call_soon_threadsafe(wrapper.set_exception, ex)
        else:
            loop.call_soon_threadsafe(wrapper.set_result, res)

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

def _await_fut(fut, delay):
    if not issubclass(type(fut), asyncio.Task):
        fut = create_task(fut)
    while True:
        with suppress(ISE):
            return fut.result()
        time.sleep(delay)

def await_fut(fut, delay=0.05, timeout=None, priority=False):
    if is_main_thread():
        return create_future_ex(_await_fut, fut, delay, timeout=timeout, priority=priority).result(timeout=timeout)
    else:
        return _await_fut(fut, delay)

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

# Manages both sync and async get requests.
class RequestManager(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, collections.abc.Callable):

    session = None
    semaphore = emptyctx

    async def _init_(self):
        self.session = aiohttp.ClientSession(loop=eloop)
        self.semaphore = Semaphore(512, 256, delay=0.25)

    async def aio_call(self, url, headers, files, data, method, decode):
        if files is not None:
            raise NotImplementedError("Unable to send multipart files asynchronously.")
        async with self.semaphore:
            async with getattr(self.session, method)(url, headers=headers, data=data) as resp:
                if resp.status >= 400:
                    data = await resp.read()
                    raise ConnectionError(f"Error {resp.status}: {data.decode('utf-8', 'replace')}")
                data = await resp.read()
                if decode:
                    return data.decode("utf-8", "replace")
                return data

    def __call__(self, url, headers={}, files=None, data=None, raw=False, timeout=8, method="get", decode=False, bypass=True, aio=False):
        if bypass:
            if "user-agent" not in headers:
                headers["user-agent"] = f"Mozilla/5.{xrand(1, 10)}"
            headers["DNT"] = "1"
        method = method.casefold()
        if aio:
            return create_task(asyncio.wait_for(self.aio_call(url, headers, files, data, method, decode), timeout=timeout))
        with self.semaphore:
            with getattr(requests, method)(url, headers=headers, files=files, data=data, stream=True, timeout=timeout) as resp:
                if resp.status_code >= 400:
                    raise ConnectionError(f"Error {resp.status_code}: {resp.text}")
                if raw:
                    data = resp.raw.read()
                else:
                    data = resp.content
                if decode:
                    return data.decode("utf-8", "replace")
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
            data = f.read().decode("utf-8", "replace")
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
            t = parse_with_now(expr) - datetime.timedelta(hours=h, seconds=t)
        else:
            t = parse_with_now(expr)
        if day is not None:
            curr = utc_dt() + datetime.timedelta(days=day)
            one_day = datetime.timedelta(days=1)
            while t < curr:
                t += one_day
            while t - curr > one_day:
                t -= one_day
        return t
    return datetime.datetime.utcfromtimestamp(s)


# Basic inheritable class for all bot commands.
class Command(collections.abc.Hashable, collections.abc.Callable):
    min_level = -inf
    rate_limit = 0
    description = ""
    usage = ""

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
            self.alias.append(self.__name__)
        self.name.append(self.__name__)
        self.aliases = {full_prune(alias).replace("*", "").replace("_", "").replace("||", ""): alias for alias in self.alias}
        self.aliases.pop("", None)
        for a in self.aliases:
            if a in bot.commands:
                bot.commands[a].append(self)
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

    __hash__ = lambda self: hash(self.__name__) ^ hash(self.catg)
    __str__ = lambda self: f"Command <{self.__name__}>"
    __call__ = lambda self, **void: None

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
class Database(collections.abc.Hashable, collections.abc.Callable):
    bot = None
    rate_limit = 3
    name = "data"

    def __init__(self, bot, catg):
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
                b = io.BytesIO(s)
                if zipfile.is_zipfile(b):
                    b.seek(0)
                    z = ZipFile(b, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
                    s = z.open("DATA").read()
                    z.close()
                data = None
                with tracebacksuppressor:
                    if s[0] == 128:
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
        self._semaphore = Semaphore(1, 1, delay=0.5, rate_limit=self.rate_limit)
        self._garbage_semaphore = Semaphore(1, 0, delay=3, rate_limit=self.rate_limit + 20)
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
    __call__ = lambda self: None

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
                        if len(s) > 1048576:
                            s = bytes2zip(s)
                    else:
                        s = s.encode("utf-8")
                    with open(self.file, "wb") as f:
                        f.write(s)
                    return True
        else:
            self.updated = True
        return False
    
    def unload(self):
        bot = self.bot
        func = getattr(self, "_destroy_", None)
        if callable(func):
            await_fut(create_future(func, priority=True), priority=True)
        for f in dir(self):
            if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                func = getattr(self, f, None)
                if callable(func):
                    bot.events[f].remove(func)
                    print("unloaded", f, "from", self)
        self.update(True)
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
                        self.data[f] = ""
                        data = enc(out)
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
        if type(args[0]) is str and args[0].startswith("WARNING:"):
            return sys.__stdout__.write(out)
        if file is None:
            file = self.file
        if file not in self.data:
            self.data[file] = ""
        if self.history.get(file) == out:
            add_dict(self.counts, {file:1})
            return
        elif self.counts.get(file):
            count = self.counts.pop(file)
            times = "s" if count != 1 else ""
            out = f"<Last message repeated {count} time{times}>\n{out}"
        else:
            self.history[file] = out
            self.counts.clear()
        self.data[file] += out
        return sys.__stdout__.write(out)

    def write(self, *args, end="", **kwargs):
        args2 = [arg if type(arg) is str else arg.decode("utf-8", "replace") for arg in args]
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