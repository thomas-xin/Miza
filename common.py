import asyncio, discord, json
from smath import *
import urllib.request, urllib.parse, requests

urlParse = urllib.parse.quote
escape_markdown = discord.utils.escape_markdown
time_snowflake = discord.utils.time_snowflake
snowflake_time = discord.utils.snowflake_time

if hasattr(asyncio, "create_task"):
    create_task = asyncio.create_task
else:
    create_task = asyncio.ensure_future


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

def noHighlight(s):
    return str(s).translate(__emap)

def sbHighlight(s):
    return "[" + noHighlight(s) + "]"


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


def iscode(fn):
    fn = str(fn)
    return fn.endswith(".py") or fn.endswith(".pyw")# or fn.endswith(".c") or fn.endswith(".cpp")


class returns:

    def __init__(self, data=None):
        self.data = data

    __call__ = lambda self: self.data
    __bool__ = lambda self: self.data is not None

async def parasync(coro, returns):
    try:
        resp = await coro
        returns.data = returns(resp)
    except Exception as ex:
        returns.data = repr(ex)
    return returns()

async def recursiveCoro(item):
    returns = hlist()
    for i in range(len(item)):
        try:
            if type(item[i]) in (str, bytes, dict) or isinstance(item[i], freeClass):
                raise TypeError
            item[i] = tuple(item[i])
        except TypeError:
            pass
        if type(item[i]) is tuple:
            returns.append(returns())
            create_task(parasync(recursiveCoro(item[i]), returns[-1]))
        elif asyncio.iscoroutine(item[i]):
            returns.append(returns())
            create_task(parasync(item[i], returns[-1]))
        else:
            returns.append(returns(item[i]))
    full = False
    while not full:
        full = True
        for i in returns:
            if not i:
                full = False
        await asyncio.sleep(0.2)
    output = hlist()
    for i in returns:
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
        await message.edit(content=message.content + "\n" + "\n".join(tuple("<" + a.url + ">" for a in message.attachments)))


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


def randColour():
    return colour2Raw(colourCalculation(xrand(12) * 128))

def strURL(url):
    return str(url).replace(".webp", ".png")

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

def verifySearch(f):
    return f.strip().translate(__strans)

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


def logClear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class __logPrinter():

    print_temp = ""
    
    def updatePrint(self, file):
        if file is None:
            outfunc = sys.stdout.write
            enc = lambda x: str(x)
        else:
            def filePrint(b):
                f = open(file, "ab+")
                f.write(b)
                f.close()
            outfunc = filePrint
            enc = lambda x: bytes(str(x), "utf-8")
        outfunc(enc("Logging started...\n"))
        while True:
            if self.print_temp:
                self.print_temp = limStr(self.print_temp, 4096)
                data = enc(self.print_temp)
                #sys.stdout.write(repr(data))
                outfunc(data)
                self.print_temp = ""
            time.sleep(1)
            #sys.stdout.write(str(f))

    def logPrint(self, *args, sep=" ", end="\n", prefix="", **void):
        self.print_temp += str(sep).join((str(i) for i in args)) + str(end) + str(prefix)

    def __init__(self, file=None):
        doParallel(self.updatePrint, [file], name="printer", killable=False)

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

    def __init__(self, _vars):
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
        self._vars = _vars
        self._globals = _vars._globals
    
    async def __call__(self, **void):
        pass


class Database:
    _vars = None
    name = "data"

    def __init__(self, _vars):
        name = self.name
        self.__name__ = self.__class__.__name__
        if not getattr(self, "no_file", False):
            self.file = "saves/" + name + ".json"
            self.updated = False
            try:
                f = open(self.file, "rb")
                s = f.read()
                if not s:
                    raise FileNotFoundError
                data = None
                try:
                    data = pickle.loads(s)
                except pickle.UnpicklingError:
                    pass
                if data is None:
                    data = eval(s)
                _vars.data[name] = self.data = data
                f.close()
            except FileNotFoundError:
                _vars.data[name] = self.data = freeClass()
        else:
            _vars.data[name] = self.data = freeClass()
        _vars.database[name] = self
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