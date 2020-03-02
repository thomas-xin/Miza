import discord, os, sys, datetime, json
from smath import *

sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

client = discord.AutoShardedClient(
    max_messages=4096,
    heartbeat_timeout=30,
)


class main_data:
    
    timeout = 24
    min_suspend = 3
    website = "https://github.com/thomas-xin/Miza"
    heartbeat = "heartbeat.json"
    restart = "restart.json"
    shutdown = "shutdown.json"
    suspected = "suspected.json"
    savedata = "data.json"
    authdata = "auth.json"
    client = client
    cache = {
        "guilds": {},
        "channels": {},
        "users": {},
        "messages": {},
        "deleted": {},
    }
    cachelim = 262144
    deleted_user = 456226577798135808
    _globals = globals()
    python = ("python3", "python")[os.name == "nt"]
            
    def __init__(self):
        print("Initializing...")
        if not os.path.exists("cache/"):
            os.mkdir("cache/")
        if not os.path.exists("saves/"):
            os.mkdir("saves/")
        try:
            f = open(self.authdata)
        except FileNotFoundError:
            f = open(self.authdata, "w")
            f.write(
                '{\n'
                + '"discord_token":"",\n'
                + '"owner_id":""\n'
                + '}'
            )
            f.close()
            print("ERROR: Please fill in details for " + self.authdata + " to continue.")
            self.shutdown()
        auth = ast.literal_eval(f.read())
        f.close()
        try:
            self.token = auth["discord_token"]
        except KeyError:
            print("ERROR: discord_token not found. Unable to login.")
            self.shutdown()
        try:
            self.owner_id = int(auth["owner_id"])
        except KeyError:
            self.owner_id = 0
            print("WARNING: owner_id not found. Unable to locate owner.")
        self.data = {}
        self.proc = psutil.Process()
        doParallel(self.getModules, state=2)
        self.current_channel = None
        self.guilds = 0
        self.blocked = 0
        self.updated = False
        self.suffix = ">>> "
        print("Initialized.")

    def shutdown(self):
        time.sleep(2)
        f = open(self.shutdown, "wb")
        f.close()
        sys.exit(1)

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

    async def verifyDelete(self, obj):
        started = hasattr(self, "started")
        if obj.checking:
            return
        obj.checking = True
        data = obj.data
        for key in tuple(data):
            if key != 0 and type(key) is not str:
                try:
                    if getattr(obj, "user", None):
                        d = await self.fetch_user(key)
                    else:
                        if not data[key] and not started:
                            raise LookupError
                        try:
                            d = await self.fetch_guild(key)
                            if d is not None:
                                continue
                        except:
                            pass
                        d = await self.fetch_channel(key)
                    if d is not None:
                        continue
                except:
                    pass
                print("Deleting " + str(key) + " from " + str(obj) + "...")
                data.pop(key)
                obj.update()
            if random.random() > .9:
                await asyncio.sleep(0.2)
        await asyncio.sleep(2)
        obj.checking = False
        self.started = True

    async def fetch_user(self, u_id):
        try:
            u_id = int(u_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid user identifier: " + uniStr(u_id))
        if u_id == self.deleted_user:
            user = self.ghostUser()
            user.id = u_id
            user.avatar_url = "https://cdn.discordapp.com/embed/avatars/0.png"
        else:
            try:
                user = client.get_user(u_id)
                if user is None:
                    raise EOFError
            except:
                if u_id in self.cache["users"]:
                    return self.cache["users"][u_id]
                user = await client.fetch_user(u_id)
        self.cache["users"][u_id] = user
        self.limitCache("users")
        return user

    async def fetch_guild(self, g_id):
        try:
            g_id = int(g_id)
        except (ValueError, TypeError):
            try:
                invite = await client.fetch_invite(g_id)
                g = invite.guild
                if not hasattr(g, "member_count"):
                    guild = freeClass(
                        member_count=invite.approximate_member_count,
                        created_at="N/A",
                    )
                    for at in g.__slots__:
                        setattr(guild, at, getattr(g, at))
                    icon = str(guild.icon)
                    guild.icon_url = (
                        "https://cdn.discordapp.com/icons/"
                        + str(guild.id) + "/" + icon
                        + ".gif" * icon.startswith("a_")
                    )
                else:
                    guild = g
                return guild
            except (discord.NotFound, discord.HTTPException) as ex:
                raise LookupError(str(ex))
            except:
                raise TypeError("Invalid server identifier: " + uniStr(g_id))
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            if g_id in self.cache["guilds"]:
                return self.cache["guilds"][g_id]
            guild = await client.fetch_guild(g_id)
        self.cache["guilds"][g_id] = guild
        self.limitCache("guilds")
        return guild

    async def fetch_channel(self, c_id):
        try:        
            c_id = int(c_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid channel identifier: " + uniStr(c_id))
        try:
            channel = client.get_channel(c_id)
            if channel is None:
                raise EOFError
        except:
            if c_id in self.cache["channels"]:
                return self.cache["channels"][c_id]
            channel = await client.fetch_channel(c_id)
        self.cache["channels"][c_id] = channel
        self.limitCache("channels")
        return channel

    async def fetch_message(self, m_id, channel=None):
        try:
            m_id = int(m_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid message identifier: " + uniStr(m_id))
        if m_id in self.cache["messages"]:
            return self.cache["messages"][m_id]
        message = await channel.fetch_message(m_id)
        if message is not None:
            self.cache["messages"][m_id] = message
            self.limitCache("messages")
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

    def getPrefix(self, guild):
        try:
            g_id = guild.id
        except:
            try:
                g_id = int(guild)
            except TypeError:
                g_id = 0
        return self.data["prefixes"].get(g_id, "~")

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
            u_perm = inf
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

    def limitCache(self, cache=None):
        if cache is not None:
            cache = [self.cache[cache]]
        else:
            cache = self.cache.values()
        for c in cache:
            if len(c) > self.cachelim:
                i = iter(c)
                c.pop(next(i))

    def isDeleted(self, message):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        return self.cache["deleted"].get(m_id, False)

    def logDelete(self, message):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        self.cache["deleted"][m_id] = True
        self.limitCache("deleted")

    def isSuspended(self, u_id):
        u_id = int(u_id)
        if u_id in (self.owner_id, client.user.id):
            return False
        return self.data["users"].get(
            u_id, {"suspended": 0}
        )["suspended"] >= time.time() + self.min_suspend * 86400

    def updatePart(self, force=False):
        if force:
            name = getattr(self, "name", None)
            if name:
                if self.updated:
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
            f = module
            f = ".".join(f.split(".")[:-1])
            path, module = module, f
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
                        name = var.name
                        if not getattr(var, "no_file", False):
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
                        else:
                            self.data[name] = var.data = {}
                        var._vars = self
                        obj = var()
                        obj.busy = False
                        obj.checking = False
                        self.updaters[obj.name] = obj
                        updates.append(obj)
                        #print("Successfully loaded updater " + obj.__name__ + ".")
                    except AttributeError:
                        pass
            for u in updates:
                for c in commands:
                    c.data[u.name] = u
            self.categories[rename] = commands
            self.codeSize += getLineCount("commands/" + path)
        except:
            print(traceback.format_exc())

    def getModules(self, reload=False):
        files = (i for i in os.listdir("commands") if iscode(i))
        self.categories = {}
        self.updaters = {}
        totalsize = [0,0]
        totalsize += sum(getLineCount(i) for i in os.listdir() if iscode(i))
        totalsize += sum(getLineCount(p) for i in os.listdir("misc") for p in ["misc/" + i] if iscode(p))
        self.codeSize = totalsize
        for f in files:
            doParallel(self.getModule, [f], state=2)
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

    imap = {
        "#": "",
        "<": "",
        ">": "",
        "@": "",
        "!": "",
        "&": "",
    }
    itrans = "".maketrans(imap)
    
    def verifyID(self, value):
        try:
            return int(str(value).translate(self.itrans))
        except ValueError:
            return value
    
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

    async def evalMath(self, f, guild):
        try:
            if f in ("t", "T", "true", "TRUE"):
                r = [True]
            elif f in ("f", "F", "false", "FALSE"):
                r = [False]
            else:
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
            self.python + " misc/math.py",
            str(f) + "`" + str(int(prec)) + "`" + str(int(r)) + "`" + str(g_id),
            self.timeout / 2,
        ]
        print(args)
        returns = [None]
        doParallel(subFunc, args, returns, state=2)
        t = time.time()
        while returns[0] is None:
            await asyncio.sleep(0.25)
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
            try:
                child.cpu_percent()
                await asyncio.sleep(0.25)
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
            if reaction is not None:
                reacode = str(reaction).encode("utf-8")
            else:
                reacode = None
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
                                    reaction=reacode,
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
        if not hasattr(self, "stat_timer"):
            self.stat_timer = 0
        if not hasattr(self, "lastCheck"):
            self.lastCheck = 0
        if not hasattr(self, "busy"):
            self.busy = False
        if force or time.time() - self.lastCheck > 0.5:
            while self.busy:
                await asyncio.sleep(0.1)
            self.busy = True
            asyncio.create_task(self.getState())
            try:
                #print("Sending update...")
                guilds = len(client.guilds)
                changed = guilds != self.guilds
                if changed or time.time() - self.stat_timer > 60:
                    self.stat_timer = time.time()
                    self.guilds = guilds
                    u = await self.fetch_user(self.owner_id)
                    n = u.name
                    place = "live from " + uniStr(n) + "'" + "s" * (n[-1] != "s") + " place, "
                    activity = discord.Streaming(
                        name=(
                            place + "to " + uniStr(guilds) + " server"
                            + "s" * (guilds != 1) + "!"
                        ),
                        url=self.website,
                    )
                    activity.game = self.website
                    if changed:
                        print(repr(activity))
                    await client.change_presence(activity=activity)
                self.lastCheck = time.time()
                for u in self.updaters.values():
                    asyncio.create_task(u())
                    asyncio.create_task(self.verifyDelete(u))
            except:
                print(traceback.format_exc())
            self.busy = False

    def randColour(self):
        return colour2Raw(colourCalculation(xrand(12) * 128))

    def strMessage(self, message):
        data = limStr(message.content, 512)
        if message.reactions:
            data += "\n{" + ", ".join(str(i) for i in message.reactions) + "}"
        if message.embeds:
            data += "\n⟨" + ", ".join(str(i.to_dict()) for i in message.embeds) + "⟩"
        if message.attachments:
            data += "\n[" + ", ".join(i.url for i in message.attachments) + "]"
        try:
            t = message.created_at
            if message.edited_at:
                t = message.edited_at
            data += "\n`(" + str(t) + ")`"
        except AttributeError:
            pass
        if not data:
            data = "```css\n" + uniStr("[EMPTY MESSAGE]") + "```"
        return limStr(data, 1024)

    class userGuild(discord.abc.Snowflake):

        class userChannel(discord.abc.PrivateChannel):

            def __init__(self, channel, **void):
                self.channel = channel
                self.recipient = channel.recipient
                self.me = client.user.id
                self.id = channel.id
                self.send = channel.send
                self.history = channel.history
                self.created_at = channel.created_at
                self.trigger_typing = channel.trigger_typing
                self.pins = channel.pins

            def fetch_message(id):
                return _vars.fetch_message(id, self.channel)

            me = client.user
            name = "DM"
            topic = None
            is_nsfw = lambda: True
            is_news = lambda: False

        def __init__(self, user, channel, **void):
            self.channel = self.system_channel = self.rules_channel = self.userChannel(channel)
            self.id = self.channel.id
            self.name = self.channel.name
            self.members = [user, client.user]
            self.channels = self.text_channels = [self.channel]
            self.voice_channels = []
            self.me = self.channel.me
            self.created_at = self.channel.created_at
            self.roles = []
            self.emojis = []
            self.get_channel = lambda *void1, **void2: self.channel
            self.owner_id = client.user.id
            fetch_member = _vars.fetch_user

        filesize_limit = 8388608
        bitrate_limit = 98304
        emoji_limit = 0
        large = False
        description = ""
        max_members = 2
        unavailable = False
        isDM = True
        ghost = True

    class ghostUser(discord.abc.User):
        
        def __init__(self):
            self.id = 0
            self.name = "[USER DATA NOT FOUND]"
            self.discriminator = "0000"
            self.avatar = "0"

        bot = False
        system = False
        history = lambda *void1, **void2: collections.namedtuple(
            "GhostIterator",
            "flatten",
            lambda: [],
        )
        dm_channel = None
        create_dm = lambda self: None
        relationship = None
        is_friend = lambda self: None
        is_blocked = lambda self: None
        avatar_url = ""
        color = discord.Colour(16777215)
        colour = color
        created_at = 0
        display_name = ""
        mention = "<@0>"
        ghost = True

    class ghostMessage(discord.abc.Snowflake):
        
        def __init__(self):
            self.author = _vars.ghostUser()
            self.content = "```css\n" + uniStr("[MESSAGE DATA NOT FOUND]") + "```"
            self.channel = None
            self.guild = None
            self.id = 0

        tts = False
        type = "default"
        nonce = False
        embeds = ()
        call = None
        mention_everyone = False
        mentions = ()
        webhook_id = None
        attachments = ()
        pinned = False
        flags = None
        reactions = ()
        activity = None
        clean_content = ""
        system_content = ""
        edited_at = None
        jump_url = "https://discordapp.com/channels/-1/-1/-1"
        is_system = lambda self: None
        created_at = 0
        
        async def delete(*void1, **void2):
            pass
        edit = delete
        publish = delete
        pin = delete
        unpin = delete
        add_reaction = delete
        remove_reaction = delete
        clear_reaction = delete
        clear_reactions = delete
        ack = delete
        ghost = True


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
            enabled = _vars.data["enabled"][c_id] = ["main", "string", "admin"]
            _vars.update()
    else:
        enabled = list(_vars.categories)
    u_perm = _vars.getPerms(u_id, guild)
    admin = not inf > u_perm

    mention = (
        "<@" + str(client.user.id) + ">",
        "<@!" + str(client.user.id) + ">",
    )
    prefix = _vars.getPrefix(guild)
    op = False
    comm = msg
    for check in (prefix, *mention):
        if comm.startswith(check):
            comm = comm[len(check):]
            op = True
        while len(comm) and comm[0] == " ":
            comm = comm[1:]
    suspended = _vars.isSuspended(u_id)
    if (suspended and op) or msg.replace(" ", "") in mention:
        if not u_perm < 0 and not suspended:
            sent = await channel.send(
		"Hi, did you require my services for anything? Use `"
                + prefix + "?` or `" + prefix + "help` for help."
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
    run = False
    if op:
        commands = hlist()
        for catg in categories:
            if catg in enabled or admin:
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
                    run = True
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
                                        while len(argv) >= 4 and r in argv:
                                            found = False
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
                                        while len(argv) >= 2 and r in argv:
                                            found = False
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
                                args = shlex.split(d.replace("\n", " ").replace("\r", "").replace("\t", " "))
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
    if not run and u_id != client.user.id and orig:
        s = "0123456789abcdefghijklmnopqrstuvwxyz"
        temp = list(reconstitute(orig).lower())
        for i in range(len(temp)):
            if not(temp[i] in s):
                temp[i] = " "
        temp = "".join(temp)
        while "  " in temp:
            temp = temp.replace("  ", " ")
        for u in _vars.updaters.values():
            f = getattr(u, "_nocommand_", None)
            if f is not None:
                try:
                    await f(
                        text=temp,
                        edit=edit,
                        orig=orig,
                        message=message,
                    )
                except:
                    print(traceback.format_exc())


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


async def updateLoop():
    print("Update loop initiated.")
    autosave = 0
    _vars.doUpdate = False
    while True:
        try:
            if time.time() - autosave > 60:
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
    await _vars.getState()
    print("Servers: ")
    for guild in client.guilds:
        if guild.unavailable:
            print("> " + str(guild.id) + " is not available.")
        else:
            print("> " + guild.name)
    await _vars.handleUpdate()
    asyncio.create_task(updateLoop())
    asyncio.create_task(heartbeatLoop())


async def checkDelete(message, reaction, user):
    if message.author.id == client.user.id:
        u_perm = _vars.getPerms(user.id, message.guild)
        check = False
        if not u_perm < 3:
            check = True
        else:
            for react in message.reactions:
                if str(reaction) == str(react):
                    users = await react.users().flatten()
                    for u in users:
                        if u.id == client.user.id:
                            check = True
                            break
        if check:
            if user.id != client.user.id:
                s = str(reaction)
                if s in "❌✖️🇽❎":
                    try:
                        d_id = message.id
                        await message.delete()
                        _vars.logDelete(d_id)
                    except discord.NotFound:
                        pass
            await _vars.handleUpdate()


@client.event
async def on_raw_reaction_add(payload):
    try:
        channel = await _vars.fetch_channel(payload.channel_id)
        user = await _vars.fetch_user(payload.user_id)
        message = await _vars.fetch_message(payload.message_id, channel=channel)
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
        message = await _vars.fetch_message(payload.message_id, channel=channel)
    except discord.NotFound:
        return
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        await _vars.reactCallback(message, reaction, user)
        asyncio.create_task(checkDelete(message, reaction, user))


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
    _vars.cache["messages"][message.id] = message
    guild = message.guild
    if guild:
        for u in _vars.updaters.values():
            f = getattr(u, "_send_", None)
            if f is not None:
                try:
                    await f(message=message)
                except:
                    print(traceback.format_exc())
    await _vars.reactCallback(message, None, message.author)
    await handleMessage(message, False)
    await _vars.handleUpdate(True)


@client.event
async def on_raw_message_delete(payload):
    try:
        message = payload.cached_message
        if message is None:
            raise LookupError
    except:
        channel = await _vars.fetch_channel(payload.channel_id)
        try:
            message = await _vars.fetch_message(payload.message_id, channel)
            if message is None:
                raise LookupError
        except:
            message = _vars.ghostMessage()
            message.channel = channel
            try:
                message.guild = channel.guild
            except AttributeError:
                message.guild = None
            message.id = payload.message_id
    guild = message.guild
    if guild:
        for u in _vars.updaters.values():
            f = getattr(u, "_delete_", None)
            if f is not None:
                try:
                    await f(message=message)
                except:
                    print(traceback.format_exc())


@client.event
async def on_raw_bulk_message_delete(payload):
    try:
        messages = payload.cached_messages
        if messages is None:
            raise LookupError
    except:
        messages = deque()
        channel = await _vars.fetch_channel(payload.channel_id)
        for m_id in payload.message_ids:
            try:
                message = await _vars.fetch_message(m_id, channel)
                if message is None:
                    raise LookupError
            except:
                message = _vars.ghostMessage()
                message.channel = channel
                try:
                    message.guild = channel.guild
                except AttributeError:
                    message.guild = None
                message.id = m_id
            messages.append(message)
    for message in messages:
        guild = message.guild
        if guild:
            for u in _vars.updaters.values():
                f = getattr(u, "_delete_", None)
                if f is not None:
                    try:
                        await f(message=message, bulk=True)
                    except:
                        print(traceback.format_exc())


async def updateEdit(before, after):
    if before.content == after.content:
        before = _vars.ghostMessage()
        before.channel = after.channel
        before.guild = after.guild
        before.author = after.author
        before.id = after.id
    guild = after.guild
    if guild:
        for u in _vars.updaters.values():
            f = getattr(u, "_edit_", None)
            if f is not None:
                try:
                    await f(before=before, after=after)
                except:
                    print(traceback.format_exc())


@client.event
async def on_message_edit(before, after):
    if before.content != after.content:
        message = after
        _vars.cache["messages"][message.id] = message
        await handleMessage(message)
        await _vars.handleUpdate(True)
        await updateEdit(before, after)


@client.event
async def on_raw_message_edit(payload):
    if payload.cached_message is not None:
        return
    try:
        c_id = payload.data.get("channel_id", 0)
        channel = await _vars.fetch_channel(c_id)
        message = await _vars.fetch_message(payload.message_id, channel)
        if message is None:
            raise LookupError
    except:
        message = _vars.ghostMessage()
        message.channel = await _vars.fetch_channel(c_id)
        message.guild = channel.guild
        message.id = payload.message_id
    if message:
        _vars.cache["messages"][message.id] = message
        await handleMessage(message)
        await _vars.handleUpdate(True)
        await updateEdit(message, message)


if __name__ == "__main__":
    _vars = main_data()
    _vars.run()
