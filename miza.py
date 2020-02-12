import discord, ast, os, sys, asyncio, datetime, json, shlex, traceback
import urllib.request
from smath import *


sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

client = discord.Client(
    max_messages=2000,
    )
    

from matplotlib import use as plot_sys
plot_sys("Agg")
from matplotlib import pyplot as plt


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


class __globals:
    timeout = 24
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
        "customAudio",
        "dynamicFunc",
        "dumpLogData",
        "setPrint",
        "processMessage",
        "updateLoop",
        "outputLoop",
        "handleMessage",
        "handleUpdate",
        "changeColour",
        "checkDelete",
        "sendUpdateRequest",
        "research",
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
    savedata = "data.json"
    authdata = "auth.json"
    client = client

    def __init__(self):
        if not os.path.exists("cache/"):
            os.mkdir("cache/")
        self.lastCheck = time.time()
        self.queue = {}
        self.loadSave()
        self.fig = fig
        self.plt = plt
        f = open(self.authdata)
        auth = ast.literal_eval(f.read())
        f.close()
        self.owner_id = int(auth["owner_id"])
        self.token = auth["discord_token"]
        self.resetGlobals()
        doParallel(self.getModules)
        self.current_channel = None
        self.guilds = 0
        self.blocked = 0
        self.doUpdate = False
        self.msgFollow = {}
        self.audiocache = {}
        self.clearAudioCache()
        self.message_cache = {}

    async def fetch_user(self, u_id):
        try:
            user = client.get_user(u_id)
            if user is None:
                raise EOFError
        except:
            user = await client.fetch_user(u_id)
        return user

    async def fetch_guild(self, g_id):
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            guild = await client.fetch_guild(g_id)
        return guild

    async def fetch_channel(self, c_id):
        try:
            channel = client.get_channel(c_id)
            if channel is None:
                raise EOFError
        except:
            channel = await client.fetch_channel(c_id)
        return channel

    async def fetch_message(self, m_id, channel=None, user=None):
        message = None
        if m_id in self.message_cache:
            message = self.message_cache[m_id]
        if message is None and user is not None and user.id != client.user.id:
            message = await user.fetch_message(m_id)
            if message is not None:
                self.message_cache[m_id] = message
        if message is None and channel is not None:
            message = await channel.fetch_message(m_id)
            if message is not None:
                self.message_cache[m_id] = message
        lim = 10000
        i = iter(self.message_cache)
        while len(self.message_cache) > lim:
            self.message_cache.pop(next(i))
        return message

    def loadSave(self):
        try:
            f = open(self.savedata, "rb")
            savedata = eval(f.read())
        except:
            print("Creating new save data...")
            self.perms = {}
            self.bans = {}
            self.enabled = {}
            self.scheduled = {}
            self.special = {}
            self.following = {}
            self.playlists = {}
            self.imglists = {}
            self.bans[0] = {}
            self.update()
            f = open(self.savedata)
            savedata = eval(f.read())
            
        f.close()
        self.perms = savedata.get("perms", {})
        self.bans = savedata.get("bans", {})
        self.enabled = savedata.get("enabled", {})
        self.scheduled = savedata.get("scheduled", {})
        self.special = savedata.get("special", {})
        self.following = savedata.get("following", {})
        self.playlists = savedata.get("playlists", {})
        self.imglists = savedata.get("imglists", {})

    def getModule(self, module):
        rename = module.lower()
        print("Loading module " + rename + "...")
        mod = __import__(module)
        commands = []
        vd = mod.__dict__
        for k in vd:
            var = vd[k]
            try:
                var.is_command
                obj = var()
                obj.__name__ = var.__name__
                obj.name.append(obj.__name__)
                commands.append(obj)
                #print("Successfully loaded function " + obj.__name__ + ".")
            except AttributeError:
                pass
        self.categories[rename] = commands

    def getModules(self):
        files = [f for f in os.listdir("commands/") if f.endswith(".py") or f.endswith(".pyw")]
        self.categories = {}
        for f in files:
            if f.endswith(".py"):
                f = f[:-3]
            else:
                f = f[:-4]
            doParallel(self.getModule, [f])

    def update(self, force=False):
        if force:
            try:
                f = open(self.savedata, "wb")
                savedata = {
                    "perms": self.perms,
                    "bans": self.bans,
                    "enabled": self.enabled,
                    "scheduled": self.scheduled,
                    "special": self.special,
                    "following": self.following,
                    "playlists": self.playlists,
                    "imglists": self.imglists,
                    }
                s = bytes(repr(savedata), "utf-8")
                f.write(s)
                f.close()
            except Exception as ex:
                print(traceback.format_exc())

    def clearAudioCache(self):
        should_cache = []
        for g in self.playlists:
            for i in self.playlists[g]:
                s = i["id"] + ".mp3"
                should_cache.append(s)
        for path in os.listdir("cache/"):
            found = False
            for i in should_cache:
                if i in path:
                    found = True
                    break
            if not found:
                try:
                    os.remove("cache/" + path)
                except Exception as ex:
                    print(traceback.format_exc())

    def verifyID(self, value):
        return int(str(value).replace("<", "").replace(">", "").replace("@", "").replace("!", ""))

    def getPerms(self, user, guild):
        try:
            u_id = int(user.id)
        except AttributeError:
            u_id = int(user)
        if guild:
            g_id = guild.id
            g_perm = self.perms.setdefault(g_id, {})
            if u_id == self.owner_id or u_id == client.user.id:
                u_perm = nan
            else:
                u_perm = g_perm.get(u_id, self.perms.setdefault("defaults", {}).get(g_id, 0))
        elif u_id == self.owner_id or u_id == client.user.id:
            u_perm = nan
        else:
            u_perm = 1
        if u_perm is not nan and u_id == getattr(guild, "owner_id", 0):
            u_perm = inf
        return u_perm

    def setPerms(self, user, guild, value):
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = self.perms.setdefault(guild.id, {})
        g_perm.update({u_id: value})
        self.update()

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

    def verifyCommand(self, func):
        f = func.lower()
        for d in self.disabled:
            if d in f:
                raise PermissionError("\"" + d + "\" is not enabled.")
        return func

    def verifyURL(self, _f):
        _f = _f.replace("<", "").replace(">", "").replace("|", "").replace("*", "").replace("_", "").replace("`", "")
        return _f

    def doMath(self, f, returns):
        try:
            self.verifyCommand(f)
            try:
                answer = eval(f, self.stored_vars)
            except:
                exec(f, self.stored_vars)
                answer = None
        except Exception as ex:
            answer = "\nError: " + repr(ex)
        if answer is not None:
            answer = str(answer)
        returns[0] = answer

    def evalMath(self, f):
        self.verifyCommand(f)
        return eval(f, self.stored_vars)

    async def reactCallback(self, message, reaction, user):
        if message.author.id == client.user.id:
            suspended = self.bans[0].get(user.id, False)
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
                            sent = await message.channel.send("```python\nError: " + repr(ex) + "\n```")
                            await sent.add_reaction("❎")


async def processMessage(message, msg, edit=True, orig=None, cb_argv=None, cb_flags=None, loop=False):
    global client
    perms = _vars.perms
    bans = _vars.bans
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
            enabled = _vars.enabled[c_id]
        except KeyError:
            enabled = _vars.enabled[c_id] = ["string", "admin"]
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

    suspended = _vars.bans[0].get(u_id, False)
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
        commands = []
        for catg in categories:
            if catg in enabled or catg == "main":
                commands += categories[catg]
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
                        if guild is None and getattr(command, "server_only", False):
                            raise ReferenceError("This command is only available in servers.")
                        if not loop and getattr(command, "time_consuming", False):
                            asyncio.create_task(channel.trigger_typing())
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
                    except Exception as ex:
                        rep = repr(ex)
                        if len(rep) > 1950:
                            errmsg = "```fix\nError: Error message too long.\n```"
                        else:
                            errmsg = "```python\nError: " + rep + "\n```"
                        print(traceback.format_exc())
                        sent = await channel.send(errmsg)
                        await sent.add_reaction("❎")
    elif u_id != client.user.id and g_id in _vars.following and orig:
        if not edit:
            if _vars.following[g_id]["follow"]:
                checker = orig
                curr = _vars.msgFollow.get(g_id)
                if curr is None:
                    curr = [checker, 1, 0]
                    _vars.msgFollow[g_id] = curr
                elif checker == curr[0] and u_id != curr[2]:
                    curr[1] += 1
                    if curr[1] >= 3:
                        curr[1] = xrand(-3) + 1
                        if len(checker):
                            asyncio.create_task(channel.send(checker))
                else:
                    if len(checker) > 100:
                        checker = ""
                    curr[0] = checker
                    curr[1] = xrand(-1, 2)
                curr[2] = u_id
                #print(curr)
        s = "0123456789abcdefghijklmnopqrstuvwxyz"
        temp = list(reconstitute(orig.lower()))
        for i in range(len(temp)):
            if not(temp[i] in s):
                temp[i] = " "
        temp = "".join(temp).split(" ")
        for i in temp:
            if not len(i.replace(" ", "")):
                temp.remove(i)
        try:
            for r in _vars.following[g_id]["reacts"]:
                if r in temp:
                    await message.add_reaction(_vars.following[g_id]["reacts"][r])
        except discord.Forbidden:
            pass
        currentSchedule = _vars.scheduled.get(channel.id, {})
        for k in currentSchedule:
            if k in " ".join(temp):
                curr = currentSchedule[k]
                role = curr["role"]
                deleter = curr["deleter"]
                try:
                    perm = float(role)
                    currPerm = _vars.getPerms(user, guild)
                    if perm > currPerm:
                        _vars.setPerms(user, guild, perm)
                    print("Granted perm " + str(perm) + " to " + user.name + ".")
                except ValueError:
                    for r in guild.roles:
                        if r.name.lower() == role:
                            await user.add_roles(
                                r,
                                reason="Verified.",
                                atomic=True,
                                )
                            print("Granted role " + r.name + " to " + user.name + ".")
                if deleter:
                    try:
                        await message.delete()
                    except discord.NotFound:
                        pass


async def outputLoop():
    global client, _vars
    print("Output Loop initiated.")
    while True:
        try:
            msg = [None]
            ch = _vars.current_channel
            if ch is not None:
                chan = str(ch.id)
            else:
                chan = ""
            printed = chan + ">>> "
            setPrint(printed)
            doParallel(input, [printed], msg, name="inputter")
            while msg[0] is None:
                await asyncio.sleep(0.1)
            proc = msg[0]
            if not proc:
                continue
            if proc[0] == "!":
                proc = proc[1:]
                if not proc:
                    ch = _vars.current_channel = None
                    continue
                try:
                    chanID = int(proc)
                    _vars.current_channel = await _vars.fetch_channel(chanID)
                except ValueError:
                    sent = await ch.send("*** ***")
                    await processMessage(sent, reconstitute(proc))
                    try:
                        await sent.delete()
                    except discord.NotFound:
                        pass
            elif proc[0] == "&":
                proc = proc[1:]
                hist = await ch.history(limit=1).flatten()
                message = hist[0]
                await message.add_reaction(proc)
            else:
                if ch:
                    await ch.send(proc)
                else:
                    try:
                        print(eval(proc))
                    except:
                        try:
                            exec(proc)
                            print(proc + " successfully executed!")
                        except:
                            print(traceback.format_exc())
        except:
            print(traceback.format_exc())


async def updateLoop():
    global _vars
    print("Update loop initiated.")
    autosave = 0
    counter = 0
    while True:
        try:
            if time.time() - autosave > 60:
                autosave = time.time()
                _vars.update(True)
            while _vars.blocked > 0:
                print("Blocked...")
                _vars.blocked -= 1
                await asyncio.sleep(1)
            for g in _vars.special:
                asyncio.create_task(changeColour(g, _vars.special[g], counter))
            await handleUpdate()
            t = time.time()
            while time.time() - t < frand(2) + 2:
                await asyncio.sleep(0.001)
                if _vars.doUpdate:
                    await handleUpdate(True)
                    _vars.doUpdate = False
        except:
            print(traceback.format_exc())
        counter = counter + 1 & 65535


async def changeColour(g_id, roles, counter):
    guild = await _vars.fetch_guild(g_id)
    colTime = 12
    l = list(roles)
    for r in l:
        try:
            role = guild.get_role(r)
            delay = roles[r]
            if not (counter + r) % delay:
                col = colour2Raw(colourCalculation(xrand(1536)))
                await role.edit(colour=discord.Colour(col))
                #print("Edited role " + role.name)
            await asyncio.sleep(frand(2))
        except discord.HTTPException as ex:
            print(traceback.format_exc())
            _vars.blocked += 20
            break


@client.event
async def on_ready():
    print("Successfully connected as " + str(client.user))
    print("Servers: ")
    for guild in client.guilds:
        if guild.unavailable:
            print("> " + str(guild.id) + " is not available.")
        else:
            print("> " + guild.name)
    await handleUpdate()
    asyncio.create_task(updateLoop())
    asyncio.create_task(outputLoop())
##    print("Users: ")
##    for guild in client.guilds:
##        print(guild.members)


def sendUpdateRequest(error=False):
    #print("Sending update request...")
    _vars.doUpdate = True


async def handleUpdate(force=False):
    if force or time.time() - _vars.lastCheck > 0.5:
        #print("Sending update...")
        guilds = len(client.guilds)
        if guilds != _vars.guilds:
            _vars.guilds = guilds
            u = await _vars.fetch_user(_vars.owner_id)
            n = u.name
            gamestr = (
                "live from " + uniStr(n) + "'" + "s" * (n[-1] != "s")
                + " place, to " + uniStr(guilds) + " servers!"
                )
            print("Playing " + gamestr)
            game = discord.Game(gamestr)
            await client.change_presence(
                activity=game,
                )
        _vars.lastCheck = time.time()
        dtime = datetime.datetime.utcnow().timestamp()
        bans = _vars.bans
        if bans:
            changed = False
            for g in bans:
                if g:
                    bl = list(bans[g])
                    for b in bl:
                        if type(bans[g][b]) is list and dtime >= bans[g][b][0]:
                            try:
                                u_target = await _vars.fetch_user(b)
                                g_target = await _vars.fetch_guild(g)
                                try:
                                    await g_target.unban(u_target)
                                    c_target = await _vars.fetch_channel(bans[g][b][1])
                                    await c_target.send(
                                        "```css\n" + uniStr(u_target.name)
                                        + " has been unbanned from " + uniStr(g_target.name) + ".```"
                                        )
                                    changed = True
                                except:
                                    c_target = await _vars.fetch_channel(bans[g][b][1])
                                    await c_target.send(
                                        "```css\nUnable to unban " + uniStr(u_target.name)
                                        + " from " + uniStr(g_target.name) + ".```"
                                        )
                                    print(traceback.format_exc())
                                bans[g].pop(b)
                            except:
                                print(traceback.format_exc())
            if changed:
                _vars.update()
        ytdl = None
        for func in _vars.categories.get("voice", []):
            if "queue" in func.name:
                ytdl = func.ytdl
        if ytdl is not None:
            should_cache = []
            for g in _vars.playlists:
                for i in _vars.playlists[g]:
                    should_cache.append(i["id"])
            for vc in client.voice_clients:
                if not vc.is_connected():
                    continue
                channel = vc.channel
                guild = channel.guild
                try:
                    auds = _vars.queue[guild.id]
                    playing = auds.is_playing and vc.is_playing() or auds.is_loading
                    membs = channel.members
                    for memb in membs:
                        if memb.id == client.user.id:
                            membs.remove(memb)
                    cnt = len(membs)
                except KeyError:
                    continue
                if not cnt or getattr(auds, "dead", 0):
                    try:
                        channel = auds.channel
                        _vars.queue.pop(guild.id)
                        msg = "```css\n🎵 Successfully disconnected from "+ uniStr(guild.name) + ". 🎵```"
                        sent = await channel.send(msg)
                        await sent.add_reaction("❎")
                        #print(msg)
                    except KeyError:
                        pass
                    await vc.disconnect(force=False)
                else:
                    try:
                        asyncio.create_task(auds.updatePlayer())
                        try:
                            q = auds.queue
                        except NameError:
                            continue
                        asyncio.create_task(research(auds, ytdl))
                        for e in q:
                            e_id = e["id"]
                            if not e_id:
                                q.remove(e)
                                continue
                            if e_id in _vars.audiocache:
                                e["duration"] = _vars.audiocache[e_id][0]
                        if len(q):
                            for i in range(2):
                                if i < len(q):
                                    e_id = q[i]["id"]
                                    should_cache.append(e_id)
                                    if not q[i].get("download", 0):
                                        q[i]["download"] = 1
                                        if e_id not in _vars.audiocache:
                                            search = e_id + ".mp3"
                                            found = False
                                            for path in os.listdir("cache"):
                                                if search in path:
                                                    found = True
                                            if not found:
                                                durc = [q[i]["duration"]]
                                                _vars.audiocache[e_id] = durc
                                                doParallel(
                                                    ytdl.downloadSingle,
                                                    [q[i], durc],
                                                    )
                                            else:
                                                q[i]["duration"] = ytdl.getDuration("cache/" + search)
                            if not q[0].get("download", 0) > 1 and not playing:
                                try:
                                    path = "cache/" + q[0]["id"] + ".mp3"
                                    f = open(path, "rb")
                                    minl = 64
                                    b = f.read(minl)
                                    f.close()
                                    if len(b) < minl:
                                        raise FileNotFoundError
                                    q[0]["download"] = 2
                                    name = q[0]["name"]
                                    added_by = q[0]["added by"]
                                    auds = _vars.queue[guild.id]
                                    auds.new(path)
                                    if not vc.is_playing():
                                        vc.play(auds, after=sendUpdateRequest)
                                    if not auds.stats["quiet"]:
                                        channel = auds.channel
                                        sent = await channel.send(
                                            "```css\n🎵 Now playing "
                                            + uniStr(noSquareBrackets(name))
                                            + ", added by " + uniStr(added_by) + "! 🎵```"
                                            )
                                        await sent.add_reaction("❎")
                                except FileNotFoundError:
                                    pass
                                auds.preparing = False
                            elif not playing and auds.source is None:
                                auds.advance()
                        if not len(q) and not auds.preparing:
                            t = _vars.playlists.get(guild.id, ())
                            if len(t):
                                d = None
                                while d is None or d["id"] == auds.prev:
                                    p = t[xrand(len(t))]
                                    d = {
                                        "name": p["name"],
                                        "url": p["url"],
                                        "duration": p["duration"],
                                        "added by": client.user.name,
                                        "u_id": client.user.id,
                                        "id": p["id"],
                                        "skips": (),
                                        }
                                    if len(t) <= 1:
                                        break
                                q.append(d)
                    except KeyError as ex:
                        print(traceback.format_exc())
            l = list(_vars.audiocache)
            for i in l:
                if not i in should_cache:
                    path = "cache/" + i + ".mp3"
                    try:
                        os.remove(path)
                        _vars.audiocache.pop(i)
                    except PermissionError:
                        pass
                    except FileNotFoundError:
                        _vars.audiocache.pop(i)


async def research(auds, ytdl):
    if auds.searching >= 1:
        #print("researching blocked.")
        return
    auds.searching += 1
    #print("researching...")
    q = auds.queue
    for i in q:
        if i in auds.queue and "research" in i:
            try:
                print(i["name"])
                i.pop("research")
                returns = [None]
                t = time.time()
                doParallel(ytdl.extractSingle, [i], returns)
                while returns[0] is None and time.time() - t < 10:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.1)
            except:
                print(traceback.format_exc())
    await asyncio.sleep(1)
    auds.searching = max(auds.searching - 1, 0)


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
            await handleUpdate()


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
    await handleUpdate()


@client.event
async def on_typing(channel, user, when):
    await handleUpdate()


@client.event
async def on_voice_state_update(member, before, after):
    if member.id == client.user.id:
        if after.mute or after.deaf:
            print("Unmuted self in " + member.guild.name)
            await member.edit(mute=False, deafen=False)
    await handleUpdate()


async def handleMessage(message, edit=True):
    msg = message.content
    try:
        await asyncio.wait_for(processMessage(message, reconstitute(msg), edit, msg), timeout=_vars.timeout)
    except Exception as ex:
        print(traceback.format_exc())
        killThreads()
        errmsg = "```python\nError: " + repr(ex) + "\n```"
        sent = await message.channel.send(errmsg)
        await sent.add_reaction("❎")
    return


@client.event
async def on_message(message):
    await _vars.reactCallback(message, None, message.author)
    await handleUpdate()
    await handleMessage(message, False)
    await handleUpdate(True)


@client.event
async def on_message_edit(before, after):
    await handleUpdate()
    if before.content != after.content:
        message = after
        await handleMessage(message)
        await handleUpdate(True)


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
        await handleUpdate()
        await handleMessage(message)
        await handleUpdate(True)


if __name__ == "__main__":
    _vars = __globals()
    print("Attempting to authorize with token " + _vars.token + ":")
    client.run(_vars.token)
