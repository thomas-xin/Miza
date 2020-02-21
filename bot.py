import discord, os, sys, datetime, json
import urllib.request
from smath import *

sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

client = discord.Client(
    max_messages=2000,
)


class main_data:
    
    timeout = 24
    min_suspend = 3
    heartbeat = "heartbeat.json"
    suspected = "suspected.json"
    shutdown = "shutdown.json"
    savedata = "data.json"
    authdata = "auth.json"
    client = client
    cache = {
        "guilds": {},
        "channels": {},
        "users": {},
        "messages": {},
    }
    cachelim = 16384

    class userGuild:

        class userChannel:

            def __init__(self, channel, **void):
                self.recipient = channel.recipient
                self.me = client.user.id
                self.name = "DM"
                self.id = channel.id
                self.topic = None
                self.is_nsfw = lambda: True
                self.is_news = lambda: False
                self.send = channel.send
                self.history = channel.history
                self.trigger_typing = channel.trigger_typing

        def __init__(self, user, channel, **void):
            self.channel = self.userChannel(channel)
            self.id = self.channel.id
            self.name = self.channel.name
            self.members = [user]
            self.channels = [self.channel]
            self.me = self.channel.me
            self.roles = []
            self.unavailable = False
            
    def __init__(self):
        print("Initializing...")
        if not os.path.exists("cache/"):
            os.mkdir("cache/")
        if not os.path.exists("saves/"):
            os.mkdir("saves/")
        f = open(self.authdata)
        auth = ast.literal_eval(f.read())
        f.close()
        self.owner_id = int(auth["owner_id"])
        self.token = auth["discord_token"]
        self.data = {}
        self.proc = psutil.Process()
        doParallel(self.getModules, state=2)
        self.current_channel = None
        self.guilds = 0
        self.blocked = 0
        self.lastCheck = 0
        self.doUpdate = False
        self.busy = False
        self.updated = False
        self.suffix = ">>> "
        print("Initialized.")

    def run(self):
        print("Attempting to authorize with token " + self.token + ":")
        try:
            self.client.run(self.token)
        except KeyboardInterrupt:
            sys.exit()
        except SystemExit:
            sys.exit()

    def print(self, *args, sep=" ", end="\n", suffix=None):
        if suffix is None:
            suffix = self.suffix
        sys.stdout.write(str(sep).join(str(i) for i in args) + end + suffix)

    async def fetch_user(self, u_id):
        u_id = int(u_id)
        try:
            user = client.get_user(u_id)
            if user is None:
                raise EOFError
        except:
            if u_id in self.cache["users"]:
                return self.cache["users"][u_id]
            user = await client.fetch_user(u_id)
        self.cache["users"][u_id] = user
        self.limitCache()
        return user

    async def fetch_guild(self, g_id):
        g_id = int(g_id)
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            if g_id in self.cache["guilds"]:
                return self.cache["guilds"][g_id]
            guild = await client.fetch_guild(g_id)
        self.cache["guilds"][g_id] = guild
        self.limitCache()
        return guild

    async def fetch_channel(self, c_id):
        c_id = int(c_id)
        try:
            channel = client.get_channel(c_id)
            if channel is None:
                raise EOFError
        except:
            if c_id in self.cache["channels"]:
                return self.cache["channels"][c_id]
            channel = await client.fetch_channel(c_id)
        self.cache["channels"][c_id] = channel
        self.limitCache()
        return channel

    async def fetch_message(self, m_id, channel=None, user=None):
        m_id = int(m_id)
        message = None
        if m_id in self.cache["messages"]:
            return self.cache["messages"][m_id]
        if message is None and user is not None and user.id != client.user.id:
            try:
                message = await user.fetch_message(m_id)
            except discord.NotFound:
                pass
        if message is None and channel is not None:
            try:
                message = await channel.fetch_message(m_id)
            except discord.NotFound:
                pass
        if message is not None:
            self.cache["messages"][m_id] = message
            self.limitCache()
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

    def limitCache(self, cache=None):
        if cache is not None:
            cache = [self.cache[cache]]
        else:
            cache = self.cache.values()
        for c in cache:
            if len(c) > self.cachelim:
                i = iter(c)
                c.pop(next(i))

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
        try:
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
                                s = f.read()
                                if not s:
                                    raise FileNotFoundError
                                self.data[name] = var.data = eval(s)
                                f.close()
                            except FileNotFoundError:
                                self.data[name] = var.data = {}
                        var._vars = self
                        obj = var()
                        obj.busy = False
                        self.updaters[obj.name] = obj
                        updates.append(obj)
                        #print("Successfully loaded updater " + obj.__name__ + ".")
                    except AttributeError:
                        pass
            for u in updates:
                for c in commands:
                    c.data[u.name] = u
            self.categories[rename] = commands
            self.modules[module] = mod
        except:
            print(traceback.format_exc())

    def getModules(self, reload=False):
        files = (i for i in os.listdir("commands") if iscode(i))
        self.categories = {}
        self.updaters = {}
        totalsize = [0,0]
        totalsize += sum(getLineCount(i) for i in os.listdir() if iscode(i))
        totalsize += sum(getLineCount(p) for i in os.listdir("misc") for p in ["misc/" + i] if iscode(p))
        if not hasattr(self, "modules"):
            self.modules = {}
        for f in files:
            totalsize += getLineCount("commands/" + f)
            if f.endswith(".py"):
                f = f[:-3]
            else:
                f = f[:-4]
            if f in self.modules:
                importlib.reload(self.modules[f])
            else:
                doParallel(self.getModule, [f], state=2)
        self.codeSize = totalsize
        __import__("smath", globals())
        subKill()

    def update(self):
        saved = hlist()
        try:
            for i in self.updaters:
                u = self.updaters[i]
                if getattr(u, "update", None) is not None:
                    if u.update(True):
                        saved.append(i)
        except Exception as ex:
            print(traceback.format_exc())
        if saved:
            print("Autosaved " + str(saved) + ".")

    def verifyID(self, value):
        return int(str(value).replace("<", "").replace(">", "").replace("@", "").replace("!", ""))

    def getPerms(self, user, guild):
        perms = self.data["perms"]
        try:
            u_id = user.id
        except AttributeError:
            u_id = int(user)
        if guild:
            try:
                g_id = guild.id
            except AttributeError:
                g_id = int(guild)
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
        " ": "%20",
    }
    utrans = "".maketrans(umap)

    def verifyURL(self, f):
        return f.strip(" ").translate(self.utrans)

    def getVar(self, g_id):
        var = self.updaters["variables"]
        data = var.data
        if g_id not in data:
            data[g_id] = var.create()
        if data[g_id] is None:
            data.pop(g_id)
            var.update()
            raise OverflowError(
                "Unfortunately the variable cache for this server has become "
                + "too large, and has been deleted. Please try again."
            )
        return data[g_id].data

    async def evalMath(self, f, guild):
        try:
            r = [ast.literal_eval(f)]
        except ValueError:
            r = await self.solveMath(f, guild, 16, 0)
        return roundMin(float(r[0]))

    async def solveMath(self, f, guild, prec, r):
        try:
            g_id = guild.id
        except AttributeError:
            g_id = int(guild)
        args = [
            g_id,
            "python misc/math.py",
            str(f) + "`" + str(int(prec)) + "`" + str(int(r)) + "`" + str(g_id),
            self.timeout / 2,
        ]
        print(args)
        returns = [None]
        doParallel(subFunc, args, returns, state=2)
        t = time.time()
        while returns[0] is None:
            await asyncio.sleep(0.21)
        resp = returns[0]
        print(resp)
        if type(resp) is str:
            raise eval(resp)
        resp = eval(resp[0].replace("\n", "").replace("\r", ""))
        if type(resp) is str:
            raise eval(resp)
        return resp

    def getActive(self):
        procs = 2 + sum(1 for c in self.proc.children(True))
        thrds = self.proc.num_threads()
        coros = sum(1 for i in asyncio.all_tasks())
        return hlist((procs, thrds, coros))

    async def getState(self):
        stats = hlist((0, 0))
        if getattr(self, "currState", None) is None:
            self.currState = stats
        proc = self.proc
        proc.cpu_percent()
        await asyncio.sleep(0.5)
        stats += (proc.cpu_percent(), proc.memory_percent())
        for child in proc.children(True):
            child.cpu_percent()
            await asyncio.sleep(0.5)
            try:
                stats += (child.cpu_percent(), child.memory_percent())
            except psutil.NoSuchProcess:
                pass
        stats[0] /= psutil.cpu_count()
        stats[1] *= psutil.virtual_memory().total / 100
        self.currState = stats
        return stats

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
            if msg.startswith(check):
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
                            sent = await message.channel.send(
                                "```py\nError: " + repr(ex) + "\n```"
                            )
                            await sent.add_reaction("❎")

    async def handleUpdate(self, force=False):
        if force or time.time() - self.lastCheck > 0.5:
            while self.busy:
                await asyncio.sleep(0.1)
            self.busy = True
            asyncio.create_task(self.getState())
            try:
                #print("Sending update...")
                guilds = len(client.guilds)
                if guilds != self.guilds:
                    self.guilds = guilds
                    u = await self.fetch_user(self.owner_id)
                    n = u.name
                    gamestr = (
                        "live from " + uniStr(n) + "'" + "s" * (n[-1] != "s")
                        + " place, to " + uniStr(guilds) + " server"
                        + "s" * (guilds != 1) + "!"
                    )
                    print("Playing " + gamestr)
                    game = discord.Game(gamestr)
                    await client.change_presence(activity=game)
                self.lastCheck = time.time()
                for u in self.updaters.values():
                    asyncio.create_task(u())
            except:
                print(traceback.format_exc())
            self.busy = False

    def printMessage(self, message, channel, suffix=False):
        data = (
            "[" + channel.name + "] "
            + str(channel.id) + ": "
            + message.author.name + ": "
            + limStr(message.content, 512)
        )
        if message.reactions:
            data += " {" + ", ".join(str(i) for i in message.reactions) + "}"
        if message.embeds:
            data += " <" + ", ".join(str(i.to_dict()) for i in message.embeds) + ">"
        if message.attachments:
            data += " <" + ", ".join(i.url for i in message.attachments) + ">"
        t = message.created_at
        if message.edited_at:
            t = message.edited_at
        data += " (" + str(t) + ")"
        if suffix:
            suffix = None
        else:
            suffix = ""
        self.print(data, suffix=suffix)


async def processMessage(message, msg, edit=True, orig=None, cb_argv=None, cb_flags=None, loop=False):
    perms = _vars.data["perms"]
    bans = _vars.data["bans"]
    categories = _vars.categories
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

    comm = msg
    if len(msg) >= 2 and msg[0] == "~" and msg[1] != "~":
        comm = msg[1:]
        op = True
    elif msg.startswith(check):
        comm = msg[len(check):]
        while len(comm) and (comm[0] in " ~"):
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
            print(
                "Ignoring command from suspended user "
                + user.name + " (" + str(u_id) + "): "
                + limStr(message.content, 256)
            )
            sent = await channel.send(
                "Sorry, you are currently not permitted to request my services."
            )
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
                match = check == alias and (
                    len(comm) == length or comm[length] == " " or comm[length] == "?"
                )
                if match:
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
                                        found = False
                                        while len(argv) >= 4 and r in argv:
                                            i = argv.index(r)
                                            if i == 0 or argv[i - 1] == " " or argv[i - 2] == "?":
                                                try:
                                                    if argv[i + 2] == " " or argv[i + 2] == "?":
                                                        argv = argv[:i] + argv[i + 2:]
                                                        addDict(flags, {char: 1})
                                                        found = True
                                                except (IndexError, KeyError):
                                                    pass
                                            if not found:
                                                break
                            if "?" in argv:
                                for c in range(26):
                                    char = chr(c + 97)
                                    flag = "?" + char
                                    for r in (flag, flag.upper()):
                                        found = False
                                        while len(argv) >= 2 and r in argv:
                                            for check in (r + " ", " " + r):
                                                if check in argv:
                                                    argv = argv.replace(check, "")
                                                    addDict(flags, {char: 1})
                                                    found = True
                                            if argv == r:
                                                argv = ""
                                                addDict(flags, {char: 1})
                                                found = True
                                            if not found:
                                                break
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
                            guild = main_data.userGuild(
                                user=user,
                                channel=channel,
                            )
                            channel = guild.channel
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
                                    asyncio.create_task(
                                        channel.send("Response too long for message.", file=f)
                                    )
                            if sent is not None:
                                await sent.add_reaction(react)
                    except TimeoutError:
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
    if guild is None or _vars.current_channel and (channel.id == _vars.current_channel.id and not edit):
        if guild is None:
            guild = main_data.userGuild(
                user=user,
                channel=channel,
            )
            channel = guild.channel
        _vars.print(suffix="")
        _vars.printMessage(message, channel, suffix=True)


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
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        sys.exit()


def sendInput(output):
    while True:
        output[0] = input()


async def inputLoop():

    def updateChannel():
        ch = _vars.current_channel
        if ch is not None:
            chan = str(ch.id)
        else:
            chan = ""
        suffix = chan + ">>> "
        _vars.suffix = suffix
        return ch

    def verifyChannel(channel):
        if channel is None:
            return None
        try:
            if channel.guild is None:
                raise TypeError
        except:
            channel = _vars.userGuild(channel.recipient, channel).channel
        return channel
        
    print("Input Loop initiated.")
    msg = [None]
    doParallel(sendInput, [msg], name="inputter", killable=False)
    prev = hlist()
    _vars.print(end="")
    while True:
        try:
            msg[0] = None
            ch = updateChannel()
            while msg[0] is None:
                await asyncio.sleep(0.2)
            proc = msg[0]
            if not proc:
                _vars.print(end="")
                continue
            if proc[0] == "!":
                proc = proc[1:]
                if not proc:
                    prev.append(_vars.current_channel)
                    _vars.current_channel = None
                    ch = updateChannel()
                    _vars.print(end="")
                    continue
                elif proc[0] == "!":
                    _vars.current_channel = verifyChannel(prev.popright())
                    ch = updateChannel()
                    _vars.print(end="")
                    continue
                elif proc[0] == "~":
                    sent = await ch.send("_ _")
                    await processMessage(sent, reconstitute(proc))
                    try:
                        await sent.delete()
                    except discord.NotFound:
                        pass
                    _vars.print(end="")
                    continue
                try:
                    prev.append(_vars.current_channel)
                    try:
                        channel = await _vars.fetch_channel(proc)
                        _vars.current_channel = verifyChannel(channel)
                        ch = updateChannel()
                    except:
                        user = await _vars.fetch_user(proc)
                        temp = await _vars.getDM(user)
                        _vars.current_channel = _vars.userGuild(user, temp).channel
                        ch = updateChannel()
                    logClear()
                    hist = await ch.history(limit=512).flatten()
                    hist = hlist(hist)
                    for m in reversed(hist):
                        _vars.printMessage(m, _vars.current_channel)
                except:
                    pass
                _vars.print(end="")
            elif proc[0] == "&":
                proc = proc[1:]
                hist = await ch.history(limit=1).flatten()
                message = hist[0]
                await message.add_reaction(proc)
                _vars.print(end="")
            else:
                if ch:
                    await ch.send(proc)
                    _vars.print(end="")
                else:
                    try:
                        output = await eval(proc)
                        _vars.print(output)
                    except:
                        #print(traceback.format_exc())
                        try:
                            output = eval(proc)
                            _vars.print(output)
                        except:
                            #print(traceback.format_exc())
                            try:
                                exec(proc)
                                _vars.print(None)
                            except:
                                _vars.print(traceback.format_exc())
        except:
            _vars.print(traceback.format_exc())


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
                await asyncio.sleep(0.03)
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
    asyncio.create_task(inputLoop())
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
    if m_id in _vars.cache["messages"]:
        _vars.cache["messages"].pop(m_id)
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
        await asyncio.wait_for(
            processMessage(message, reconstitute(msg), edit, msg), timeout=_vars.timeout
        )
    except Exception as ex:
        print(traceback.format_exc())
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
    _vars.run()
