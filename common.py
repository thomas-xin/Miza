import os, sys, subprocess, psutil, asyncio, discord, json, requests
import urllib.request, urllib.parse, concurrent.futures
from smath import *

python = ("python3", "python")[os.name == "nt"]
CalledProcessError = subprocess.CalledProcessError
Process = psutil.Process()
urlParse = urllib.parse.quote
escape_markdown = discord.utils.escape_markdown
time_snowflake = discord.utils.time_snowflake
snowflake_time = discord.utils.snowflake_time
getattr(discord, "__builtins__", {})["print"] = print
getattr(concurrent.futures, "__builtins__", {})["print"] = print
getattr(asyncio.futures, "__builtins__", {})["print"] = print
getattr(asyncio, "__builtins__", {})["print"] = print


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


ESCAPE_T = {
    "[": "⦍",
    "]": "⦎",
    "@": "＠",
    "`": "",
}
__emap = "".maketrans(ESCAPE_T)

noHighlight = lambda s: str(s).translate(__emap)

sbHighlight = lambda s: "[" + noHighlight(s) + "]"


def getLineCount(fn):
    f = open(fn, "rb")
    count = 1
    size = 0
    while True:
        try:
            i = f.read(1024)
            if not i:
                raise EOFError
            size += len(i)
            count += i.count(b"\n")
        except EOFError:
            f.close()
            return hlist((size, count))


iscode = lambda fn: str(fn).endswith(".py") or str(fn).endswith(".pyw")

awaitable = lambda obj: asyncio.iscoroutine(obj) or isinstance(obj, asyncio.Future)


class returns:

    def __init__(self, data=None):
        self.data = data

    __call__ = lambda self: self.data
    __bool__ = lambda self: self.data is not None

async def parasync(coro, rets):
    try:
        resp = await coro
        rets.data = returns(resp)
    except Exception as ex:
        rets.data = repr(ex)
    return returns()

async def recursiveCoro(item):
    rets = hlist()
    for i in range(len(item)):
        try:
            if type(item[i]) in (str, bytes, dict):
                raise TypeError
            if isinstance(item[i], freeClass):
                raise TypeError
            if awaitable(item[i]):
                raise TypeError
            item[i] = tuple(item[i])
        except TypeError:
            pass
        if type(item[i]) is tuple:
            rets.append(returns())
            create_task(parasync(recursiveCoro(item[i]), rets[-1]))
        elif awaitable(item[i]):
            rets.append(returns())
            create_task(parasync(item[i], rets[-1]))
        else:
            rets.append(returns(item[i]))
    full = False
    while not full:
        full = True
        for i in rets:
            if not i:
                full = False
        await asyncio.sleep(0.2)
    output = hlist()
    for i in rets:
        while isinstance(i, returns):
            i = i()
        output.append(i)
    return output

async def sendReact(channel, *args, reacts=(), **kwargs):
    try:
        sent = await channel.send(*args, **kwargs)
        for react in reacts:
            await sent.add_reaction(react)
    except:
        print(traceback.format_exc())

async def sendFile(channel, msg, file, filename=None):
    message = await channel.send(msg, file=file)
    if filename is not None:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
        except:
            print(traceback.format_exc())
    if message.attachments:
        await message.edit(content=message.content + "\n" + "\n".join("<" + a.url + ">" for a in message.attachments))


def strMessage(message, limit=1024, username=False):
    c = message.content
    s = getattr(message, "system_content", "")
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

def strActivity(activity):
    if hasattr(activity, "type") and activity.type != discord.ActivityType.custom:
        t = activity.type.name
        return t[0].upper() + t[1:] + " " + activity.name
    return str(activity)

def hasSymbol(string):
    for c in string.lower():
        x = ord(c)
        if x > 122 or (x < 97 and x > 57) or x < 48:
            return False
    return True


async def strLookup(it, query, ikey=lambda x: [str(x)], qkey=lambda x: [str(x)]):
    qlist = qkey(query)
    cache = [[[inf, None], [inf, None]] for _ in qlist]
    x = 1
    for i in shuffle(it):
        for c in ikey(i):
            for a, b in enumerate(qkey(c)):
                if b == qlist[a]:
                    return i
                elif b.startswith(qlist[a]):
                    if len(b) < cache[a][0][0]:
                        cache[a][0] = [len(b), i]
                elif qlist[a] in b:
                    if len(b) < cache[a][1][0]:
                        cache[a][1] = [len(b), i]
        if not x & 1023:
            await asyncio.sleep(0.1)
        x += 1
    for c in cache:
        if c[0][0] < inf:
            return c[0][1]
    for c in cache:
        if c[1][0] < inf:
            return c[1][1]
    raise LookupError("No results for " + str(query) + ".")


randColour = lambda: colour2Raw(colourCalculation(xrand(12) * 128))

strURL = lambda url: str(url).replace(".webp", ".png")

shash = lambda s: bytes2Hex(hashlib.sha256(s.encode("utf-8")).digest(), space=False)

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
    try:
        return int(str(value).translate(__itrans))
    except ValueError:
        return value

__umap = {
    "<": "",
    ">": "",
    "|": "",
    "*": "",
    " ": "%20",
}
__utrans = "".maketrans(__umap)

def verifyURL(f):
    if "file:" in f:
        raise PermissionError("Unable to open local file " + f + ".")
    return f.strip().translate(__utrans)

__smap = {
    "<": "",
    ">": "",
    "|": "",
    "*": "",
}
__strans = "".maketrans(__smap)

verifySearch = lambda f: f.strip().translate(__strans)

DOMAIN_FORMAT = re.compile(
    r"(?:^(\w{1,255}):(.{1,255})@|^)"
    r"(?:(?:(?=\S{0,253}(?:$|:))"
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:[a-z0-9]{1,63})))"
    r"|localhost)"
    r"(:\d{1,5})?",
    re.IGNORECASE
)
SCHEME_FORMAT = re.compile(
    r"^(http|hxxp|ftp|fxp)s?$",
    re.IGNORECASE
)

def isURL(url):
    url = url.strip()
    if not url:
        return None
    result = urllib.parse.urlparse(url)
    scheme = result.scheme
    domain = result.netloc
    if not scheme:
        return False
    if not re.fullmatch(SCHEME_FORMAT, scheme):
        return False
    if not domain:
        return False
    if not re.fullmatch(DOMAIN_FORMAT, domain):
        return False
    return True


class urlBypass(urllib.request.FancyURLopener):
    version = "Mozilla/5." + str(xrand(1, 10))

    __call__ = lambda self, url: self.open(url)

def urlOpen(url):
    opener = urlBypass()
    resp = opener(verifyURL(url))
    if resp.getcode() != 200:
        raise ConnectionError("Error " + str(resp.code))
    return resp


SUBS = freeClass(procs=hlist(), busy=freeClass())

subCount = lambda: sum(1 for proc in SUBS.procs if proc.is_running())

def subKill():
    for sub in SUBS.procs:
        sub.kill()
    SUBS.procs.clear()
    SUBS.busy.clear()

def procUpdate():
    procs = SUBS.procs
    b = len(SUBS.busy)
    count = sum(1 for proc in procs if not proc.busy)
    if count > 16:
        return
    if b + 1 > count:
        proc = psutil.Popen(
            [python, "misc/math.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.busy = False
        procs.append(proc)
    att = 0
    while count > b + 2:
        for p in range(len(procs)):
            if p < len(procs):
                proc = procs[p]
                if not proc.busy:
                    proc.kill()
                    procs.pop(p)
                    break
            else:
                break
        att += 1
        if att >= 16:
            break

procUpdate()

async def mathProc(data, key=-1, timeout=12):
    procs, busy = SUBS.procs, SUBS.busy
    while time.time() - busy.get(key, 0) < 60:
        await asyncio.sleep(0.5)
    try:
        while True:
            for p in range(len(procs)):
                if p < len(procs):
                    proc = procs[p]
                    if not proc.busy:
                        raise StopIteration
                else:
                    break
            procUpdate()
            await asyncio.sleep(0.5)
    except StopIteration:
        pass
    d = repr(bytes(data, "utf-8")).encode("utf-8") + b"\n"
    print(d)
    try:
        proc.busy = True
        busy[key] = time.time()
        procUpdate()
        proc.stdin.write(d)
        proc.stdin.flush()
        resp = await asyncio.wait_for(create_future(proc.stdout.readline), timeout=timeout)
        proc.busy = False
    except (TimeoutError, asyncio.exceptions.TimeoutError):
        proc.kill()
        try:
            procs.pop(p)
        except LookupError:
            pass
        try:
            busy.pop(key)
        except KeyError:
            pass
        procUpdate()
        raise
    try:
        busy.pop(key)
    except KeyError:
        pass
    output = evalEX(evalEX(resp))
    return output


def evalEX(exc):
    is_ex = False
    try:
        ex = eval(exc)
    except NameError:
        if type(exc) is bytes:
            exc = exc.decode("utf-8")
        ex = RuntimeError(exc[exc.index("(") + 1:exc.index(")")].strip("'"))
    except:
        print(exc)
        raise
    try:
        if issubclass(ex.__class__, Exception):
            is_ex = True
    except AttributeError:
        pass
    if is_ex:
        raise ex
    return ex


def funcSafe(func, *args, print_exc=False, **kwargs):
    try:
        return [func(*args, **kwargs)]
    except Exception as ex:
        if print_exc:
            print(traceback.format_exc())
        return repr(ex)


pthreads = concurrent.futures.ThreadPoolExecutor(max_workers=128)
athreads = concurrent.futures.ThreadPoolExecutor(max_workers=64)
eloop = asyncio.new_event_loop()
asyncio.set_event_loop(eloop)

def wrap_future(fut, loop=None):
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = eloop
    new_fut = loop.create_future()

    def on_done(*void):
        try:
            result = fut.result()
        except Exception as ex:
            loop.call_soon_threadsafe(new_fut.set_exception, ex)
        else:
            loop.call_soon_threadsafe(new_fut.set_result, result)

    fut.add_done_callback(on_done)
    return new_fut

create_future = lambda func, *args, loop=None, priority=False, **kwargs: wrap_future((athreads, pthreads)[priority].submit(func, *args, **kwargs), loop=loop)

def create_task(fut, *args, loop=None, **kwargs):
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = eloop
    return asyncio.ensure_future(fut, *args, loop=loop, **kwargs)


logClear = lambda: os.system(("clear", "cls")[os.name == "nt"])

class __logPrinter:
    
    def updatePrint(self):

        def filePrint(fn, b):
            f = open(fn, "ab")
            f.write(b)
            f.close()

        if self.file is None:
            outfunc = sys.stdout.write
            enc = lambda x: x
        else:
            outfunc = lambda s: filePrint(self.file, s)
            enc = lambda x: bytes(x, "utf-8")
        outfunc(enc("Logging started...\n"))
        while True:
            try:
                for f in self.data:
                    if not self.data[f]:
                        self.data.pop(f)
                        continue
                    out = limStr(self.data[f], 8192)
                    self.data[f] = ""
                    data = enc(out)
                    if f == self.file:
                        outfunc(data)
                    else:
                        filePrint(f, data)
            except:
                print(traceback.format_exc())
            time.sleep(1)

    def logPrint(self, *args, sep=" ", end="\n", prefix="", file=None, **void):
        if file is None:
            file = self.file
        if file not in self.data:
            self.data[file] = ""
        self.data[file] += str(sep).join(str(i) for i in args) + str(end) + str(prefix)

    def __init__(self, file=None):
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.data = freeClass()
        self.file = file
        self.future = self.exec.submit(self.updatePrint)

__printer = __logPrinter("log.txt")
print = __printer.logPrint


class Command:
    min_level = -inf
    description = ""
    usage = ""

    def permError(self, perm, req=None, reason=None):
        if req is None:
            req = self.min_level
        if reason is None:
            reason = "for command " + self.name[-1]
        raise PermissionError(
            "Insufficient priviliges " + str(reason)
            + ". Required level: " + str(req)
            + ", Current level: " + str(perm) + "."
        )

    def __init__(self, _vars, catg):
        if not hasattr(self, "data"):
            self.data = freeClass()
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
            a = a.lower()
            if a in _vars.commands:
                _vars.commands[a].append(self)
            else:
                _vars.commands[a] = hlist([self])
        self.catg = catg
        self._vars = _vars
        self._globals = _vars._globals
    
    async def __call__(self, **void):
        pass


class Database:
    _vars = None
    name = "data"

    def __init__(self, _vars, catg):
        name = self.name
        self.__name__ = self.__class__.__name__
        if not getattr(self, "no_file", False):
            self.file = "saves/" + name + ".json"
            self.updated = False
            try:
                f = open(self.file, "rb")
                s = f.read()
                f.close()
                if not s:
                    raise FileNotFoundError
                data = None
                try:
                    data = pickle.loads(s)
                except pickle.UnpicklingError:
                    pass
                if data is None:
                    try:
                        data = eval(s)
                    except:
                        print(self.file)
                        print(traceback.format_exc())
                        raise FileNotFoundError
                _vars.data[name] = self.data = data
            except FileNotFoundError:
                _vars.data[name] = self.data = {}
        else:
            _vars.data[name] = self.data = {}
        _vars.database[name] = self
        self.catg = catg
        self._vars = _vars
        self.busy = self.checking = False
        self._globals = globals()
        # print(name, self.__name__)

    async def __call__(self, **void):
        pass

    def update(self, force=False):
        if not hasattr(self, "updated"):
            self.updated = False
        if force:
            name = getattr(self, "name", None)
            if name:
                if self.updated:
                    self.updated = False
                    data = repr(self.data)
                    if len(data) > 262144:
                        print("Pickling " + name + "...")
                        data = pickle.dumps(data)
                    else:
                        data = data.encode("utf-8")
                    f = open(self.file, "wb")
                    f.write(data)
                    f.close()
                    return True
        else:
            self.updated = True
        return False