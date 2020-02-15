import discord, os, sys, datetime, json
import urllib.request
from smath import *

from matplotlib import use as plot_sys
plot_sys("Agg")
from matplotlib import pyplot as plt

sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

client = discord.Client(
    max_messages=2000,
)


def tryFunc(func, *args, force=False, amax, **kwargs):
    try:
        ans = nan
        ans = func(*args, **kwargs)
        if not (ans > -amax and ans < amax):
            raise OverflowError
    except:
        if force:
            ans = 0
        else:
            if ans.imag:
                ans = nan
            elif ans > amax:
                ans = inf
            elif ans < amax:
                ans = -inf
            else:
                ans = nan
    return ans

def plot(*args,**kwargs):
    args = list(args)
    flip = False
    if type(args[0]) is str:
        s = args[0]
        if s[0] == "y":
            try:
                t = s.index("=")
                s = s[t + 1:]
            except ValueError:
                pass
        f = _vars.evalMath("lambda x: " + s)
        try:
            f(0)
        except ArithmeticError:
            pass
        except NameError as ex1:
            if s[0] == "x":
                try:
                    t = s.index("=")
                    s = s[t + 1:]
                except ValueError:
                    pass
            f = _vars.evalMath("lambda y: " + s)
            try:
                f(0)
            except ArithmeticError:
                pass
            except NameError as ex2:
                raise NameError(str(ex1) + ", " + str(ex2))
            flip = True
        args = (f,) + args[1:]
    if callable(args[0]):
        amax = 100
        if len(args) < 2:
            r = float(2 * tau)
            c = float(-tau)
        elif len(args) < 3:
            r = args[1]*2
            c = -args[1]
        else:
            r = abs(args[2]-args[1])
            c = min(args[2], args[1])
            if len(args) >= 4:
                amax = args[3]
        size = 1024
        array1 = array(range(size)) / size * r + c
        array2 = [tryFunc(args[0], array1[i], force=(i == 0 or i == len(array1) - 1), amax=amax) for i in range(len(array1))]
        if flip:
            args = [array2, array1]
        else:
            args = [array1, array2]
    if len(args) < 3:
        cols = "rgbcmy"
        args.append("-" + cols[xrand(len(cols))])
    return plt.plot(*args)


fig = plt.figure()


class main_data:
    timeout = 24
    min_suspend = 3
    heartbeat = "heartbeat.json"
    suspected = "suspected.json"
    shutdown = "shutdown.json"
    savedata = "data.json"
    authdata = "auth.json"
    client = client
    deleted = [
        "discord",
        "client",
        "urllib",
        "os",
        "sys",
        "asyncio",
        "ast",
        "threading",
        "processes",
        "printVars",
        "printGlobals",
        "printLocals",
        "origPrint",
        "logPrint",
        "doParallel",
        "updatePrint",
        "waitParallel",
        "killThreads",
        "performAction",
        "dynamicFunc",
        "dumpLogData",
        "setPrint",
        "processMessage",
        "updateLoop",
        "outputLoop",
        "heartbeatLoop",
        "handleMessage",
        "checkDelete",
        "on_ready",
        "on_reaction_add",
        "on_reaction_remove",
        "on_message",
        "on_message_edit",
        "on_raw_reaction_add",
        "on_raw_message_edit",
        "on_voice_state_update",
        "on_raw_message_delete",
        "on_typing",
        "_vars",
        "fig",
        "plt",
    ]
    disabled = [
        "__",
        ".load",
        ".save",
        ".dump",
        ".fromfile",
    ]
    builtins_list = [
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    ]
    builtins = {i: getattr(__builtins__, i) for i in builtins_list}

    class customGuild:

        def __init__(self, user, channel, **void):
            self.id = channel.id
            self.name = "DM"
            self.members = [user]
            self.channels = [self]
            self.me = client.user
            self.roles = []
            self.unavailable = False

    def __init__(self):
        print("Initializing...")
        if not os.path.exists("cache/"):
            os.mkdir("cache/")
        if not os.path.exists("saves/"):
            os.mkdir("saves/")
        self.lastCheck = time.time()
        self.queue = {}
        self.fig = fig
        self.plt = plt
        f = open(self.authdata)
        auth = ast.literal_eval(f.read())
        f.close()
        self.owner_id = int(auth["owner_id"])
        self.token = auth["discord_token"]
        self.resetGlobals()
        self.data = {}
        doParallel(self.getModules)
        self.current_channel = None
        self.guilds = 0
        self.blocked = 0
        self.doUpdate = False
        self.updated = False
        self.audiocache = {}
        self.message_cache = {}
        print("Initialized.")

    async def fetch_user(self, u_id):
        u_id = int(u_id)
        try:
            user = client.get_user(u_id)
            if user is None:
                raise EOFError
        except:
            user = await client.fetch_user(u_id)
        return user

    async def fetch_guild(self, g_id):
        g_id = int(g_id)
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            guild = await client.fetch_guild(g_id)
        return guild

    async def fetch_channel(self, c_id):
        c_id = int(c_id)
        try:
            channel = client.get_channel(c_id)
            if channel is None:
                raise EOFError
        except:
            channel = await client.fetch_channel(c_id)
        return channel

    async def fetch_message(self, m_id, channel=None, user=None):
        m_id = int(m_id)
        message = None
        if m_id in self.message_cache:
            message = self.message_cache[m_id]
        if message is None and user is not None and user.id != client.user.id:
            try:
                message = await user.fetch_message(m_id)
                if message is not None:
                    self.message_cache[m_id] = message
            except discord.NotFound:
                pass
        if message is None and channel is not None:
            try:
                message = await channel.fetch_message(m_id)
                if message is not None:
                    self.message_cache[m_id] = message
            except discord.NotFound:
                pass
        lim = 10000
        while len(self.message_cache) > lim:
            i = iter(self.message_cache)
            self.message_cache.pop(next(i))
        return message

    async def getDM(self, user):
        try:
            int(user)
            user = await self.fetch_user(user)
        except:
            pass
        channel = user.dm_channel
        if channel is None:
            channel = await user.create_dm()
        return channel

    def isSuspended(self, u_id):
        u_id = int(u_id)
        if u_id in (self.owner_id, client.user.id):
            return False
        return self.data["suspended"].get(u_id, False) >= time.time() + self.min_suspend * 86400

    def updatePart(self, force=False):
        if force:
            name = getattr(self, "name", None)
            if name:
                if self.updated:
                    #print(self.file)
                    self.updated = False
                    f = open(self.file, "wb")
                    f.write(bytes(repr(self.data), "utf-8"))
                    f.close()
                    return True
        else:
            self.updated = True
        return False

    def getModule(self, module):
        #print(main_data)
        rename = module.lower()
        print("Loading module " + rename + "...")
        mod = __import__(module)
        commands = hlist()
        updates = hlist()
        vd = mod.__dict__
        for k in vd:
            var = vd[k]
            try:
                var.is_command
                var._vars = self
                obj = var()
                obj.data = {}
                obj.__name__ = var.__name__
                obj.name.append(obj.__name__)
                commands.append(obj)
                #print("Successfully loaded command " + obj.__name__ + ".")
            except AttributeError:
                try:
                    var.is_update
                    if getattr(var, "name", None):
                        name = var.name
                        var.file = "saves/" + name + ".json"
                        var.update = main_data.updatePart
                        var.updated = False
                        try:
                            f = open(var.file, "rb")
                            self.data[name] = var.data = eval(f.read())
                            f.close()
                        except FileNotFoundError:
                            self.data[name] = var.data = {}
                    var._vars = self
                    obj = var()
                    self.updaters[obj.name] = obj
                    updates.append(obj)
                    #print("Successfully loaded updater " + obj.__name__ + ".")
                except AttributeError:
                    pass
        for u in updates:
            for c in commands:
                c.data[u.name] = u
        self.categories[rename] = commands

    def getModules(self):
        files = [f for f in os.listdir("commands/") if f.endswith(".py") or f.endswith(".pyw")]
        self.categories = {}
        self.updaters = {}
        for f in files:
            if f.endswith(".py"):
                f = f[:-3]
            else:
                f = f[:-4]
            doParallel(self.getModule, [f])

    def update(self):
        count = 0
        try:
            for u in self.updaters.values():
                if getattr(u, "update", None) is not None:
                    count += u.update(True)
            self.updated = False
        except Exception as ex:
            print(traceback.format_exc())
        if count:
            print("Autosaved " + str(count) + " save file" + "s" * (count != 1) + ".")

    def verifyID(self, value):
        return int(str(value).replace("<", "").replace(">", "").replace("@", "").replace("!", ""))

    def getPerms(self, user, guild):
        perms = self.data["perms"]
        try:
            u_id = int(user.id)
        except AttributeError:
            u_id = int(user)
        if guild:
            g_id = guild.id
            g_perm = perms.setdefault(g_id, {})
            if u_id in (self.owner_id, client.user.id):
                u_perm = nan
            else:
                u_perm = g_perm.get(u_id, perms.setdefault("defaults", {}).get(g_id, 0))
        elif u_id in (self.owner_id, client.user.id):
            u_perm = nan
        else:
            u_perm = 1
        if u_perm is not nan and u_id == getattr(guild, "owner_id", 0):
            u_perm = inf
        return u_perm

    def setPerms(self, user, guild, value):
        perms = self.data["perms"]
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = perms.setdefault(guild.id, {})
        g_perm.update({u_id: value})
        self.updaters["perms"].update()

    def resetGlobals(self):
        self.stored_vars = dict(globals())
        self.verifyGlobals()
        return self.stored_vars

    def updateGlobals(self):
        self.stored_vars.update(dict(globals()))
        self.verifyGlobals()
        return self.stored_vars

    def verifyGlobals(self):
        for i in self.deleted:
            try:
                self.stored_vars.pop(i)
            except KeyError:
                pass
        self.stored_vars["__builtins__"] = self.builtins
        return self.stored_vars
    
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
        "<": "hlist((",
        ">": "))",
    }
    ctrans = "".maketrans(cmap)

    umap = {
        "<": "",
        ">": "",
        "|": "",
        "*": "",
        "_": "",
        "~": "",
        " ": "%20",
    }
    utrans = "".maketrans(umap)

    def verifyCommand(self, func):
        f1 = func.translate(self.mtrans)
        f2 = f1.translate(self.ctrans)
        for d in self.disabled:
            if d in f1:
                raise PermissionError("\"" + d + "\" is not enabled.")
        return f2, f1

    def verifyURL(self, f):
        return f.strip(" ").translate(self.utrans)

    def doMath(self, f, returns):
        try:
            att = 0
            for f in self.verifyCommand(f):
                try:
                    answer = eval(f, self.stored_vars)
                    break
                except:
                    try:
                        exec(f, self.stored_vars)
                        answer = None
                        break
                    except:
                        if att >= 1:
                            raise
                att += 1
        except Exception as ex:
            answer = "\nError: " + repr(ex)
        if answer is not None:
            answer = str(answer)
        returns[0] = answer

    def evalMath(self, f):
        f1, f2 = self.verifyCommand(f)
        try:
            return eval(f1, self.stored_vars)
        except:
            return eval(f2, self.stored_vars)

    async def reactCallback(self, message, reaction, user):
        if message.author.id == client.user.id:
            suspended = _vars.isSuspended(user.id)
            if suspended:
                return
            u_perm = self.getPerms(user.id, message.guild)
            msg = message.content
            if msg[:3] != "```" or len(msg) <= 3:
                return
            msg = msg[3:]
            while msg[0] == "\n":
                msg = msg[1:]
            check = "callback-"
            msg = msg.split("\n")[0]
            if msg[:len(check)] == check:
                msg = msg[len(check):]
                args = msg.split("-")
                catx = args[0]
                func = args[1]
                vals = args[2]
                argv = "-".join(args[3:])
                catg = self.categories[catx]
                for f in catg:
                    if f.__name__ == func:
                        try:
                            await asyncio.wait_for(
                                f._callback_(
                                    client=client,
                                    message=message,
                                    channel=message.channel,
                                    guild=message.guild,
                                    reaction=reaction,
                                    user=user,
                                    perm=u_perm,
                                    vals=vals,
                                    argv=argv,
                                    _vars=self,
                                ),
                                timeout=self.timeout)
                            return
                        except Exception as ex:
                            print(traceback.format_exc())
                            killThreads()
                            sent = await message.channel.send("```py\nError: " + repr(ex) + "\n```")
                            await sent.add_reaction("❎")

    async def handleUpdate(self, force=False):
        if force or time.time() - self.lastCheck > 0.5:
            #print("Sending update...")
            guilds = len(client.guilds)
            if guilds != self.guilds:
                self.guilds = guilds
                u = await self.fetch_user(self.owner_id)
                n = u.name
                gamestr = (
                    "live from " + uniStr(n) + "'" + "s" * (n[-1] != "s")
                    + " place, to " + uniStr(guilds) + " server" + "s" * (guilds != 1) + "!"
                )
                print("Playing " + gamestr)
                game = discord.Game(gamestr)
                await client.change_presence(
                    activity=game,
                )
            self.lastCheck = time.time()
            for u in self.updaters.values():
                asyncio.create_task(u())


async def processMessage(message, msg, edit=True, orig=None, cb_argv=None, cb_flags=None, loop=False):
    global client
    perms = _vars.data["perms"]
    bans = _vars.data["bans"]
    categories = _vars.categories
    stored_vars = _vars.stored_vars
    if msg[:2] == "> ":
        msg = msg[2:]
    elif msg[:2] == "||" and msg[-2:] == "||":
        msg = msg[2:-2]
    msg = msg.replace("`", "")
    while len(msg):
        if msg[0] == "\n" or msg[0] == "\r":
            msg = msg[1:]
        else:
            break
    user = message.author
    guild = message.guild
    u_id = user.id
    if guild:
        g_id = guild.id
    else:
        g_id = 0
    channel = message.channel
    c_id = channel.id
    if g_id:
        try:
            enabled = _vars.data["enabled"][c_id]
        except KeyError:
            enabled = _vars.data["enabled"][c_id] = ["string", "admin"]
            _vars.update()
    else:
        enabled = list(_vars.categories)
    u_perm = _vars.getPerms(u_id, guild)
    channel = message.channel

    check = "<@!" + str(client.user.id) + ">"

    if len(msg) >= 2 and msg[0] == "~" and msg[1] != "~":
        comm = msg[1:]
        op = True
    elif msg[:len(check)] == check:
        comm = msg[len(check):]
        while comm[0] == " ":
            comm = comm[1:]
        op = True
    else:
        op = False

    suspended = _vars.isSuspended(u_id)
    if (suspended and op) or msg.replace(" ", "") == check:
        if not u_perm < 0 and not suspended:
            sent = await channel.send(
		"Hi, did you require my services for anything? Use `~?` or `~help` for help."
	    )
        else:
            print("Ignoring command from suspended user " + user.name + " (" + str(u_id) + ").")
            sent = await channel.send("Sorry, you are currently not permitted to request my services.")
        await sent.add_reaction("❎")
        return
    if op:
        commands = hlist()
        for catg in categories:
            if catg in enabled or catg == "main":
                commands.extend(categories[catg])
        for command in commands:
            for alias in command.name:
                alias = alias.lower()
                length = len(alias)
                check = comm[:length].lower()
                argv = comm[length:]
                if check == alias and (len(comm) == length or comm[length] == " " or comm[length] == "?"):
                    print(user.name + " (" + str(u_id) + ") issued command " + msg)
                    req = command.min_level
                    try:
                        if req > u_perm or (u_perm is not nan and req is nan):
                            raise PermissionError(
                                "Insufficient priviliges for command " + alias
                                + ".\nRequred level: " + uniStr(req)
                                + ", Current level: " + uniStr(u_perm) + "."
                            )
                        if cb_argv is not None:
                            argv = cb_argv
                            flags = cb_flags
                            if loop:
                                addDict(flags, {"h": 1})
                        else:
                            flags = {}
                            if argv:
                                while argv[0] == " ":
                                    argv = argv[1:]
                            if "?" in argv:
                                for c in range(26):
                                    char = chr(c + 97)
                                    flag = "?" + char
                                    for r in (flag, flag.upper()):
                                        if len(argv) >= 4 and r in argv:
                                            i = argv.index(r)
                                            if i == 0 or argv[i - 1] == " " or argv[i - 2] == "?":
                                                try:
                                                    if argv[i + 2] == " " or argv[i + 2] == "?":
                                                        argv = argv[:i] + argv[i + 2:]
                                                        addDict(flags, {char: 1})
                                                except:
                                                    pass
                            if "?" in argv:
                                for c in range(26):
                                    char = chr(c + 97)
                                    flag = "?" + char
                                    for r in (flag, flag.upper()):
                                        if len(argv) >= 2 and r in argv:
                                            for check in (r + " ", " " + r):
                                                if check in argv:
                                                    argv = argv.replace(check, "")
                                                    addDict(flags, {char: 1})
                                            if argv == r:
                                                argv = ""
                                                addDict(flags, {char: 1})
                        if argv:
                            while argv[0] == " ":
                                argv = argv[1:]
                        if not len(argv.replace(" ", "")):
                            argv = ""
                            args = []
                        else:
                            a = argv.replace('"', "\0")
                            b = a.replace("'", "")
                            c = b.replace("<", "'")
                            d = c.replace(">", "'")
                            try:
                                args = shlex.split(d)
                            except ValueError:
                                args = d.split(" ")
                        for a in range(len(args)):
                            args[a] = args[a].replace("", "'").replace("\0", '"')
                        if guild is None:
                            guild = main_data.customGuild(
                                user=user,
                                channel=channel,
                            )
                            if getattr(command, "server_only", False):
                                raise ReferenceError("This command is only available in servers.")
                        tc = getattr(command, "time_consuming", False)
                        if not loop and tc:
                            asyncio.create_task(channel.trigger_typing())
                        for u in _vars.updaters.values():
                            f = getattr(u, "_command_", None)
                            if f is not None:
                                await f(user, command)
                        response = await command(
                            client=client,          # for interfacing with discord
                            _vars=_vars,            # for interfacing with bot's database
                            argv=argv,              # raw text argument
                            args=args,              # split text arguments
                            flags=flags,            # special flags
                            perm=u_perm,            # permission level
                            user=user,              # user that invoked the command
                            message=message,        # message data
                            channel=channel,        # channel data
                            guild=guild,            # guild data
                            name=alias,             # alias the command was called as
                            callback=processMessage,# function that called the command
                        )
                        if response is not None and len(response):
                            if type(response) is tuple:
                                response, react = response
                                if react == 1:
                                    react = "❎"
                            else:
                                react = False
                            sent = None
                            if type(response) is list:
                                for r in response:
                                    asyncio.create_task(channel.send(r))
                            elif type(response) is dict:
                                if react:
                                    sent = await channel.send(**response)
                                else:
                                    asyncio.create_task(channel.send(**response))
                            else:
                                if len(response) <= 2000:
                                    if react:
                                        sent = await channel.send(response)
                                    else:
                                        asyncio.create_task(channel.send(response))
                                else:
                                    fn = "cache/temp.txt"
                                    f = open(fn, "wb")
                                    f.write(bytes(response, "utf-8"))
                                    f.close()
                                    f = discord.File(fn)
                                    print("Created file " + fn)
                                    asyncio.create_task(channel.send("Response too long for message.", file=f))
                            if sent is not None:
                                await sent.add_reaction(react)
                    except TimeoutError:
                        killThreads()
                        raise TimeoutError("Request timed out.")
                    except Exception as ex:
                        errmsg = limStr("```py\nError: " + repr(ex) + "\n```", 2000)
                        print(traceback.format_exc())
                        sent = await channel.send(errmsg)
                        await sent.add_reaction("❎")
    elif message.guild and u_id != client.user.id and orig:
        s = "0123456789abcdefghijklmnopqrstuvwxyz"
        temp = list(orig.lower())
        for i in range(len(temp)):
            if not(temp[i] in s):
                temp[i] = " "
        temp = "".join(temp)
        while "  " in temp:
            temp = temp.replace("  ", " ")
        for u in _vars.updaters.values():
            f = getattr(u, "_nocommand_", None)
            if f is not None:
                await f(
                    text=temp,
                    edit=edit,
                    orig=orig,
                    message=message,
                )
    elif not message.guild:
        print(
            "\nDM " + str(message.channel.id) + ": "
            + message.author.name + ": "
            + limStr(message.content, 512)
        )


async def heartbeatLoop():
    print("Heartbeat Loop initiated.")
    try:
        while True:
            try:
                _vars
            except NameError:
                sys.exit()
            if _vars.heartbeat in os.listdir():
                try:
                    os.remove(_vars.heartbeat)
                except:
                    print(traceback.format_exc())
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        sys.exit()


def sendInput(output):
    while True:
        output[0] = input()


async def outputLoop():
    print("Output Loop initiated.")
    msg = [None]
    doParallel(sendInput, [msg], name="inputter")
    while True:
        try:
            msg[0] = None
            ch = _vars.current_channel
            if ch is not None:
                chan = str(ch.id)
            else:
                chan = ""
            printed = chan + ">>> "
            setPrint(printed)
            while msg[0] is None:
                await asyncio.sleep(0.1)
            proc = msg[0]
            if not proc:
                continue
            if proc[0] == "!":
                proc = proc[1:]
                if not proc:
                    ch = _vars.current_channel = None
                    print()
                    continue
                try:
                    _vars.current_channel = await _vars.fetch_channel(proc)
                    print()
                except ValueError:
                    sent = await ch.send("*** ***")
                    await processMessage(sent, reconstitute(proc))
                    try:
                        await sent.delete()
                    except discord.NotFound:
                        pass
                    print()
            elif proc[0] == "&":
                proc = proc[1:]
                hist = await ch.history(limit=1).flatten()
                message = hist[0]
                await message.add_reaction(proc)
                print()
            else:
                if ch:
                    await ch.send(proc)
                    print()
                else:
                    try:
                        output = await eval(proc)
                        print(output)
                    except:
                        #print(traceback.format_exc())
                        try:
                            output = eval(proc)
                            print(output)
                        except:
                            #print(traceback.format_exc())
                            try:
                                exec(proc)
                                print(None)
                            except:
                                print(traceback.format_exc())
        except:
            print(traceback.format_exc())


async def updateLoop():
    global _vars
    print("Update loop initiated.")
    autosave = 0
    while True:
        try:
            if time.time() - autosave > 30:
                autosave = time.time()
                _vars.update()
            while _vars.blocked > 0:
                print("Blocked...")
                _vars.blocked -= 1
                await asyncio.sleep(1)
            await _vars.handleUpdate()
            t = time.time()
            while time.time() - t < frand(2) + 2:
                await asyncio.sleep(0.01)
                if _vars.doUpdate:
                    await _vars.handleUpdate(True)
                    _vars.doUpdate = False
        except:
            print(traceback.format_exc())
        

@client.event
async def on_ready():
    print("Successfully connected as " + str(client.user))
    print("Servers: ")
    for guild in client.guilds:
        if guild.unavailable:
            print("> " + str(guild.id) + " is not available.")
        else:
            print("> " + guild.name)
    await _vars.handleUpdate()
    asyncio.create_task(updateLoop())
    asyncio.create_task(outputLoop())
    asyncio.create_task(heartbeatLoop())
##    print("Users: ")
##    for guild in client.guilds:
##        print(guild.members)


async def checkDelete(message, reaction, user):
    if message.author.id == client.user.id:
        u_perm = _vars.getPerms(user.id, message.guild)
        check = False
        if not u_perm < 1:
            check = True
        else:
            for reaction in message.reactions:
                async for u in reaction.users():
                    if u.id == client.user.id:
                        check = True
        if check:
            if user.id != client.user.id:
                s = str(reaction)
                if s in "❌✖️🇽❎":
                    try:
                        temp = message.content
                        await message.delete()
                        print(temp + " deleted by " + user.name)
                    except discord.NotFound:
                        pass
            await _vars.handleUpdate()


@client.event
async def on_raw_reaction_add(payload):
    try:
        channel = await _vars.fetch_channel(payload.channel_id)
        user = await _vars.fetch_user(payload.user_id)
        message = await _vars.fetch_message(payload.message_id, channel=channel, user=user)
    except discord.NotFound:
        return
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        await _vars.reactCallback(message, reaction, user)
        asyncio.create_task(checkDelete(message, reaction, user))


@client.event
async def on_raw_reaction_remove(payload):
    try:
        channel = await _vars.fetch_channel(payload.channel_id)
        user = await _vars.fetch_user(payload.user_id)
        message = await _vars.fetch_message(payload.message_id, channel=channel, user=user)
    except discord.NotFound:
        return
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        await _vars.reactCallback(message, reaction, user)
        asyncio.create_task(checkDelete(message, reaction, user))


@client.event
async def on_raw_message_delete(payload):
    m_id = payload.message_id
    if m_id in _vars.message_cache:
        _vars.message_cache.pop(m_id)
    await _vars.handleUpdate()


@client.event
async def on_typing(channel, user, when):
    await _vars.handleUpdate()


@client.event
async def on_voice_state_update(member, before, after):
    if member.id == client.user.id:
        if after.mute or after.deaf:
            print("Unmuted self in " + member.guild.name)
            await member.edit(mute=False, deafen=False)
    await _vars.handleUpdate()


async def handleMessage(message, edit=True):
    msg = message.content
    try:
        await asyncio.wait_for(processMessage(message, reconstitute(msg), edit, msg), timeout=_vars.timeout)
    except Exception as ex:
        print(traceback.format_exc())
        killThreads()
        errmsg = limStr("```py\nError: " + repr(ex) + "\n```", 2000)
        sent = await message.channel.send(errmsg)
        await sent.add_reaction("❎")
    return


@client.event
async def on_message(message):
    await _vars.reactCallback(message, None, message.author)
    await handleMessage(message, False)
    await _vars.handleUpdate(True)


@client.event
async def on_message_edit(before, after):
    await _vars.handleUpdate()
    if before.content != after.content:
        message = after
        await handleMessage(message)
        await _vars.handleUpdate(True)


@client.event
async def on_raw_message_edit(payload):
    message = None
    if payload.cached_message is None:
        try:
            channel = await _vars.fetch_channel(payload.data["channel_id"])
            message = await _vars.fetch_message(payload.message_id, channel=channel)
        except:
            for guild in client.guilds:
                for channel in guild.text_channels:
                    try:
                        message = await _vars.fetch_message(payload.message_id, channel=channel)
                    except Exception as ex:
                        print(traceback.format_exc())
    if message:
        await handleMessage(message)
        await _vars.handleUpdate(True)


if __name__ == "__main__":
    _vars = main_data()
    print("Attempting to authorize with token " + _vars.token + ":")
    try:
        client.run(_vars.token)
    except KeyboardInterrupt:
        sys.exit()
    except SystemExit:
        sys.exit()
