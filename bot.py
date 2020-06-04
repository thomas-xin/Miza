#!/usr/bin/python3

try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

client = discord.AutoShardedClient(
    max_messages=4096,
    heartbeat_timeout=30,
)


class Bot:
    
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
    prefix = "~"
    deleted_user = 456226577798135808
    _globals = globals()
            
    def __init__(self):
        self.bot = self
        self.closed = False
        self.loaded = False
        self.cache = freeClass(
            guilds={},
            channels={},
            users={},
            roles={},
            emojis={},
            messages={},
            deleted={},
        )
        self.proc_call = {}
        print("Time: " + str(datetime.datetime.now()))
        print("Initializing...")
        if not os.path.exists("cache/"):
            os.mkdir("cache")
        if not os.path.exists("saves/"):
            os.mkdir("saves")
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
            self.setshutdown()
        auth = ast.literal_eval(f.read())
        f.close()
        try:
            self.token = auth["discord_token"]
        except KeyError:
            print("ERROR: discord_token not found. Unable to login.")
            self.setshutdown()
        try:
            self.owner_id = int(auth["owner_id"])
        except KeyError:
            self.owner_id = 0
            print("WARNING: owner_id not found. Unable to locate owner.")
        self.proc = psutil.Process()
        self.getModules()
        self.guilds = 0
        self.blocked = 0
        self.updated = False
        print("Initialized.")
        create_future_ex(self.clearcache, priority=True)

    __call__ = lambda self: self

    def setshutdown(self):
        time.sleep(2)
        f = open(self.shutdown, "wb")
        f.close()
        sys.exit(1)

    def run(self):
        print("Attempting to authorize with token " + self.token + ":")
        try:
            eloop.run_until_complete(client.start(self.token))
        except (KeyboardInterrupt, SystemExit):
            eloop.run_until_complete(client.logout())
            eloop.close()
            sys.exit()

    def clearcache(self):
        for path in os.listdir("cache"):
            try:
                os.remove("cache/" + path)
            except:
                print(traceback.format_exc())

    def print(self, *args, sep=" ", end="\n"):
        sys.stdout.write(str(sep).join(str(i) for i in args) + end)

    async def verifyDelete(self, obj):
        started = hasattr(self, "started")
        if obj.checking:
            return
        if hasattr(obj, "no_delete"):
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
            if random.random() > .99:
                await asyncio.sleep(0.2)
        await asyncio.sleep(10)
        obj.checking = False
        self.started = True

    async def get_sendable(self, guild, member):
        if member is None:
            return guild.owner
        channel = guild.system_channel
        if channel is None or not channel.permissions_for(member).send_messages:
            channel = guild.rules_channel
            if channel is None or not channel.permissions_for(member).send_messages:
                found = False
                if guild.text_channels:
                    for channel in guild.text_channels:
                        if channel.permissions_for(member).send_messages:
                            found = True
                            break
                if not found:
                    return guild.owner
        return channel
                    
    async def fetch_user(self, u_id):
        try:
            u_id = int(u_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid user identifier: " + str(u_id))
        if u_id in self.cache.users:
            return self.cache.users[u_id]
        if u_id == self.deleted_user:
            user = self.ghostUser()
            user.system = True
            user.name = "Deleted User"
            user.display_name = "Deleted User"
            user.id = u_id
            user.avatar_url = "https://cdn.discordapp.com/embed/avatars/0.png"
            user.created_at = snowflake_time(u_id)
        else:
            try:
                user = client.get_user(u_id)
                if user is None:
                    raise EOFError
            except:
                user = await client.fetch_user(u_id)
        self.cache.users[u_id] = user
        self.limitCache("users")
        return user

    async def fetch_member(self, u_id, guild=None):
        try:
            u_id = int(u_id)
        except:
            pass
        member = None
        if type(u_id) is int:
            member = guild.get_member(u_id)
        if member is None:
            if type(u_id) is int:
                try:
                    member = await guild.fetch_member(u_id)
                except discord.NotFound:
                    pass
            if member is None:
                members = guild.members
                if not members:
                    members = guild.members = await guild.fetch_members(limit=None)
                try:
                    member = await strLookup(
                        members,
                        u_id,
                        qkey=lambda x: [str(x), reconstitute(x).replace(" ", "").lower()],
                        ikey=lambda x: [str(x), reconstitute(x.name), reconstitute(x.display_name)],
                    )
                except LookupError:
                    raise LookupError("Unable to find member data.")
        return member

    async def fetch_whuser(self, u_id, guild=None):
        try:
            if u_id in self.cache.users:
                return self.cache.users[u_id]
            try:
                g_id = guild.id
            except AttributeError:
                g_id = guild
            guild = await self.fetch_guild(g_id)
            try:
                webhooks = await guild.webhooks()
            except AttributeError:
                raise EOFError
            for w in webhooks:
                if w.id == u_id:
                    user = bot.ghostUser()
                    user.id = u_id
                    user.name = w.name
                    user.display_name = w.name
                    user.created_at = user.joined_at = w.created_at
                    user.avatar = w.avatar
                    user.avatar_url = w.avatar_url
                    user.bot = True
                    user.webhook = w
                    raise StopIteration
            raise EOFError
        except StopIteration:
            self.cache.users[u_id] = user
            self.limitCache("users")
            return user
        except EOFError:
            raise LookupError("Unable to find target from ID.")

    async def fetch_guild(self, g_id):
        try:
            g_id = int(g_id)
        except (ValueError, TypeError):
            try:
                invite = await client.fetch_invite(g_id)
                g = invite.guild
                if not hasattr(g, "member_count"):
                    guild = freeClass(member_count=invite.approximate_member_count)
                    for at in g.__slots__:
                        setattr(guild, at, getattr(g, at))
                    guild.created_at = snowflake_time(guild.id)
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
                raise TypeError("Invalid server identifier: " + str(g_id))
        if g_id in self.cache.guilds:
            return self.cache.guilds[g_id]
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            guild = await client.fetch_guild(g_id)
        self.cache.guilds[g_id] = guild
        self.limitCache("guilds", limit=65536)
        return guild

    async def fetch_channel(self, c_id):
        try:
            c_id = int(c_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid channel identifier: " + str(c_id))
        if c_id in self.cache.channels:
            return self.cache.channels[c_id]
        try:
            channel = client.get_channel(c_id)
            if channel is None:
                raise EOFError
        except:
            channel = await client.fetch_channel(c_id)
        self.cache.channels[c_id] = channel
        self.limitCache("channels")
        return channel

    async def fetch_message(self, m_id, channel=None):
        try:
            m_id = int(m_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid message identifier: " + str(m_id))
        if m_id in self.cache.messages:
            return self.cache.messages[m_id]
        if channel is None:
            raise LookupError("Message data not found.")
        try:
            int(channel)
            channel = await self.fetch_channel(channel)
        except TypeError:
            pass
        message = await channel.fetch_message(m_id)
        if message is not None:
            self.cache.messages[m_id] = message
            self.limitCache("messages")
        return message

    async def fetch_role(self, r_id, guild):
        try:
            r_id = int(r_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid role identifier: " + str(r_id))
        if r_id in self.cache.roles:
            return self.cache.roles[r_id]
        try:
            role = guild.get_role(r_id)
            if role is None:
                raise EOFError
        except:
            if len(guild.roles) <= 1:
                roles = await guild.fetch_roles()
                guild.roles = sorted(roles)
                role = discord.utils.get(roles, id=r_id)
            if role is None:
                raise discord.NotFound("Role not found.")
        self.cache.roles[r_id] = role
        self.limitCache("roles")
        return role

    async def fetch_emoji(self, e_id, guild=None):
        try:
            e_id = int(e_id)
        except (ValueError, TypeError):
            raise TypeError("Invalid emoji identifier: " + str(e_id))
        if e_id in self.cache.emojis:
            return self.cache.emojis[e_id]
        try:
            emoji = client.get_emoji(e_id)
            if emoji is None:
                raise EOFError
        except:
            if guild is not None:
                emoji = await guild.fetch_emoji(e_id)
            else:
                raise discord.NotFound("Emoji not found.")
        self.cache.emojis[e_id] = emoji
        self.limitCache("emojis")
        return emoji
    
    def get_mimic(self, m_id, user=None):
        try:
            try:
                m_id = "&" + str(int(m_id))
            except (ValueError, TypeError):
                pass
            mimic = self.data.mimics[m_id]
            return mimic
        except KeyError:
            pass
        if user is not None:
            try:
                mimics = self.data.mimics[user.id]
                mlist = mimics[m_id]
                return self.get_mimic(random.choice(mlist))
            except KeyError:
                pass
        raise LookupError("Unable to find target mimic.")

    async def getDM(self, user):
        try:
            int(user)
            user = await self.fetch_user(user)
        except TypeError:
            pass
        channel = user.dm_channel
        if channel is None:
            channel = await user.create_dm()
        return channel

    async def followURL(self, url, it=None):
        if it is None:
            url = stripAcc(url.strip().strip("`"))
            it = {}
        if "channels/" in url and ("discord" in url or url.startswith("channels/")):
            spl = url[url.index("channels/") + 9:].split("/")
            c = await self.fetch_channel(spl[1])
            m = await self.fetch_message(spl[2], c)
            if m.attachments:
                url = m.attachments[0].url
            else:
                url = m.content
                if " " in url or "\n" in url or not isURL(url):
                    for i in url.replace("\n", " ").split(" "):
                        u = verifyURL(i)
                        if isURL(u):
                            url = u
                            break
                url = verifyURL(url)
                if not url:
                    url = " ".join(e.description for e in m.embeds if type(e.description) is str)
                    if url:
                        if " " in url or "\n" in url or not isURL(url):
                            for i in url.replace("\n", " ").replace("(", " ").replace(")", " ").split(" "):
                                u = verifyURL(i)
                                if isURL(u):
                                    url = u
                                    break
                    if not url:
                        url = " ".join(e.image if type(e.image) is str else e.thumbnail if type(e.thumbnail) is str else e.video for e in m.embeds)
                if url in it:
                    return url
                it[url] = True
                if not len(it) & 255:
                    await asyncio.sleep(0.2)
                url = await self.followURL(url, it)
        return url

    def cacheMessage(self, message):
        self.cache.messages[message.id] = message
        self.limitCache("messages")

    def deleteMessage(self, message):
        try:
            self.cache.messages.pop(message.id)
        except KeyError:
            pass

    def limitCache(self, cache=None, limit=1048576):
        if cache is not None:
            caches = [self.cache[cache]]
        else:
            caches = self.cache.values()
        for c in caches:
            while len(c) > limit:
                c.pop(next(iter(c)))

    def getPrefix(self, guild):
        try:
            g_id = guild.id
        except AttributeError:
            try:
                g_id = int(guild)
            except TypeError:
                g_id = 0
        try:
            return self.data.prefixes[g_id]
        except KeyError:
            return bot.prefix

    def getPerms(self, user, guild=None):
        try:
            u_id = user.id
        except AttributeError:
            u_id = int(user)
        if u_id == self.owner_id:
            return nan
        if self.isSuspended(u_id):
            return -inf
        if u_id == client.user.id:
            return inf
        if guild is None or hasattr(guild, "ghost"):
            return inf
        if u_id == guild.owner_id:
            return inf
        try:
            perm = self.data.perms[guild.id][u_id]
            if isnan(perm):
                return -inf
            return perm
        except KeyError:
            pass
        m = guild.get_member(u_id)
        if m is None:
            r = guild.get_role(u_id)
            if r is None:
                return -inf
            return self.getRolePerms(r, guild)
        p = m.guild_permissions
        if p.administrator:
            return inf
        perm = -inf
        for role in m.roles: 
            rp = self.getRolePerms(role, guild)
            if rp > perm:
                perm = rp
        if isnan(perm):
            perm = -inf
        return perm
    
    def getRolePerms(self, role, guild):
        if role.permissions.administrator:
            return inf
        try:
            perm = self.data.perms[guild.id][role.id]
            if isnan(perm):
                return -inf
            return perm
        except KeyError:
            pass
        if guild.id == role.id:
            return 0
        p = role.permissions
        if all((p.ban_members, p.manage_channels, p.manage_guild, p.manage_roles, p.manage_messages)):
            return 4
        elif any((p.ban_members, p.manage_channels, p.manage_guild)):
            return 3
        elif any((p.kick_members, p.manage_messages, p.manage_nicknames, p.manage_roles, p.manage_webhooks, p.manage_emojis)):
            return 2
        elif any((p.view_audit_log, p.priority_speaker, p.mention_everyone, p.move_members)):
            return 1
        return -1

    def setPerms(self, user, guild, value):
        perms = self.data.perms
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = setDict(perms, guild.id, {})
        g_perm.update({u_id: value})
        self.database.perms.update()

    def isDeleted(self, message):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        return self.cache.deleted.get(m_id, False)

    def logDelete(self, message, no_log=False):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        self.cache.deleted[m_id] = no_log + 1
        self.limitCache("deleted", limit=4096)
    
    async def silentDelete(self, message, exc=False, no_log=False, delay=None):
        if delay:
            await asyncio.sleep(float(delay))
        try:
            self.logDelete(message, no_log)
            await message.delete()
        except:
            try:
                self.cache.deleted.pop(message.id)
            except KeyError:
                pass
            if exc:
                raise

    def isTrusted(self, g_id):
        try:
            trusted = self.data.trusted
        except (AttributeError, KeyError):
            return False
        return g_id in trusted

    def isSuspended(self, u_id):
        u_id = int(u_id)
        if u_id in (self.owner_id, client.user.id):
            return False
        try:
            return self.data.blacklist.get(
                u_id, 0
            ) >= utc() + self.min_suspend * 86400
        except KeyError:
            return True

    def getModule(self, module):
        try:
            f = module
            f = ".".join(f.split(".")[:-1])
            path, module = module, f
            rename = module.lower()
            print("Loading module " + rename + "...")
            if module in self._globals:
                mod = importlib.reload(self._globals[module])
            else:
                mod = __import__(module)
            self._globals[module] = mod
            commands = hlist()
            dataitems = hlist()
            vd = mod.__dict__
            for k in tuple(vd):
                var = vd[k]
                if var not in (Command, Database):
                    load_type = 0
                    try:
                        if issubclass(var, Command):
                            load_type = 1
                        elif issubclass(var, Database):
                            load_type = 2
                    except TypeError:
                        pass
                    if load_type:
                        obj = var(self, rename)
                        if load_type == 1:
                            commands.append(obj)
                            print("Successfully loaded command " + obj.__name__ + ".")
                        elif load_type == 2:
                            dataitems.append(obj)
                            print("Successfully loaded database " + obj.__name__ + ".")
            for u in dataitems:
                for c in commands:
                    c.data[u.name] = u
            self.categories[rename] = commands
            self.codeSize += getLineCount("commands/" + path)
        except:
            print(traceback.format_exc())

    def getModules(self, reload=False):
        if reload:
            subKill()
        files = [i for i in os.listdir("commands") if iscode(i)]
        self.categories = freeClass()
        self.commands = freeClass()
        self.database = freeClass()
        self.data = freeClass()
        totalsize = [0, 0]
        totalsize += sum(getLineCount(i) for i in os.listdir() if iscode(i))
        totalsize += sum(getLineCount(p) for i in os.listdir("misc") for p in ["misc/" + i] if iscode(p))
        self.codeSize = totalsize
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(files))
        for f in files:
            executor.submit(self.getModule, f)
        executor.shutdown(wait=False)
        self.loaded = True

    def update(self):
        saved = hlist()
        try:
            for i in self.database:
                u = self.database[i]
                if getattr(u, "update", None) is not None:
                    if u.update(True):
                        saved.append(i)
        except:
            print(traceback.format_exc())
        if saved:
            print("Autosaved " + str(saved) + ".")
    
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

    op = {
        "=": None,
        ":=": None,
        "+=": "__add__",
        "-=": "__sub__",
        "*=": "__mul__",
        "/=": "__truediv__",
        "//=": "__floordiv__",
        "**=": "__pow__",
        "^=": "__pow__",
        "%=": "__mod__",
    }

    async def evalMath(self, expr, guild, default=0, op=True):
        if op:
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip()
                    _op = at
            num = await self.evalMath(expr, guild, op=False)
            if _op is not None:
                num = getattr(float(default), _op)(num)
            return num
        f = expr.strip()
        try:
            if not f:
                r = [0]
            elif f.lower() in ("t", "true", "y", "yes", "on"):
                r = [True]
            elif f.lower() in ("f", "false", "n", "no", "off"):
                r = [False]
            elif f.lower() == "inf":
                r = [inf]
            elif f.lower() == "-inf":
                r = [-inf]
            elif f.lower() in ("nan", "-nan"):
                r = [nan]
            else:
                r = [ast.literal_eval(f)]
        except (ValueError, TypeError, SyntaxError):
            r = await self.solveMath(f, guild, 16, 0)
        x = r[0]
        try:
            while True:
                if type(x) is str:
                    raise TypeError
                x = tuple(x)[0]
        except TypeError:
            pass
        return roundMin(float(x))

    async def solveMath(self, f, guild, prec, r):
        f = f.strip()
        try:
            if guild is None or hasattr(guild, "ghost"):
                g_id = self.deleted_user
            else:
                g_id = guild.id
        except AttributeError:
            g_id = int(guild)
        return await mathProc(f, int(prec), int(r), g_id)

    timeChecks = {
        "galactic year": ("gy", "galactic year", "galactic years"),
        "millenium": ("ml", "millenium", "millenia"),
        "century": ("c", "century", "centuries"),
        "decade": ("dc", "decade", "decades"),
        "year": ("y", "year", "years"),
        "month": ("mo", "mth", "month", "mos", "mths", "months"),
        "week": ("w", "wk", "week", "wks", "weeks"),
        "day": ("d", "day", "days"),
        "hour": ("h", "hr", "hour", "hrs", "hours"),
        "minute": ("m", "min", "minute", "mins", "minutes"),
        "second": ("s", "sec", "second", "secs", "seconds"),
    }
    andcheck = re.compile("[^a-z](and)[^a-z]", re.I)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    async def evalTime(self, expr, guild, default=0, op=True):
        if op:
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip(" ")
                    _op = at
            num = await self.evalTime(expr, guild, op=False)
            if _op is not None:
                num = getattr(float(default), _op)(num)
            return num
        t = 0
        if expr:
            try:
                if ":" in expr:
                    data = expr.split(":")
                    mult = 1
                    while len(data):
                        t += await self.evalMath(data[-1], guild.id) * mult
                        data = data[:-1]
                        if mult <= 60:
                            mult *= 60
                        elif mult <= 3600:
                            mult *= 24
                        elif len(data):
                            raise TypeError("Too many time arguments.")
                else:
                    f = re.sub(self.andcheck, " ", expr).lower()
                    for tc in self.timeChecks:
                        for check in reversed(self.timeChecks[tc]):
                            if check in f:
                                i = f.index(check)
                                isnt = i + len(check) < len(f) and f[i + len(check)] in self.alphabet
                                if not i or f[i - 1] in self.alphabet or isnt:
                                    continue
                                n = await self.evalMath(f[:i], guild.id)
                                s = TIMEUNITS[tc]
                                if type(s) is list:
                                    s = s[0]
                                t += s * n
                                f = f[i + len(check):]
                    if f.strip():
                        t += await self.evalMath(f, guild.id)
            except:
                t = utc_ts(tparser.parse(f)) - utc_ts(tparser.parse("0s"))
        if type(t) is not float:
            t = float(t)
        return t

    def getActive(self):
        procs = 2 + sum(1 for c in self.proc.children(True))
        thrds = self.proc.num_threads()
        coros = sum(1 for i in asyncio.all_tasks())
        return hlist((procs, thrds, coros))

    async def getProcState(self, proc):
        try:
            create_future_ex(proc.cpu_percent, priority=True)
            await asyncio.sleep(1)
            c = await create_future(proc.cpu_percent, priority=True)
            m = await create_future(proc.memory_percent, priority=True)
            return float(c), float(m)
        except psutil.NoSuchProcess:
            return 0, 0
        
    getCacheState = lambda self: sum(os.path.getsize("cache/" + fn) for fn in os.listdir("cache"))

    async def getState(self):
        stats = hlist(0, 0, 0)
        if getattr(self, "currState", None) is None:
            self.currState = stats
        procs = await create_future(self.proc.children, recursive=True, priority=True)
        procs.append(self.proc)
        tasks = [self.getProcState(p) for p in procs]
        resp = await recursiveCoro(tasks)
        stats += [sum(st[0] for st in resp), sum(st[1] for st in resp), 0]
        cpu = await create_future(psutil.cpu_count, priority=True)
        mem = await create_future(psutil.virtual_memory, priority=True)
        disk = await create_future(self.getCacheState, priority=True)
        stats[0] /= cpu
        stats[1] *= mem.total / 100
        stats[2] = disk
        self.currState = stats
        return stats

    async def reactCallback(self, message, reaction, user):
        if message.author.id == client.user.id:
            if self.closed:
                return
            u_perm = self.getPerms(user.id, message.guild)
            if u_perm <= -inf:
                return
            msg = message.content
            if not msg:
                msg = message.embeds[0].description
            if msg[:3] != "```" or len(msg) <= 3:
                return
            msg = msg[3:]
            while msg.startswith("\n"):
                msg = msg[1:]
            check = "callback-"
            try:
                msg = msg[:msg.index("\n")]
            except ValueError:
                pass
            if msg.startswith(check):
                while len(self.proc_call) > 65536:
                    self.proc_call.pop(next(iter(self.proc_call)))
                while utc() - self.proc_call.get(message.id, 0) < 30:
                    await asyncio.sleep(0.2)
                if reaction is not None:
                    reacode = str(reaction).encode("utf-8")
                else:
                    reacode = None
                msg = message.content
                if not msg:
                    msg = message.embeds[0].description
                if msg[:3] != "```" or len(msg) <= 3:
                    return
                msg = msg[3:]
                while msg[0] == "\n":
                    msg = msg[1:]
                check = "callback-"
                msg = msg.split("\n")[0]
                self.proc_call[message.id] = utc()
                msg = msg[len(check):]
                args = msg.split("-")
                catn, func, vals = args[:3]
                func = func.lower()
                argv = "-".join(args[3:])
                catg = self.categories[catn]
                for f in catg:
                    if f.__name__.lower() == func:
                        try:
                            timeout = getattr(f, "_timeout_", 1) * self.timeout
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
                                    bot=self,
                                ),
                                timeout=timeout)
                            break
                        except Exception as ex:
                            print(traceback.format_exc())
                            create_task(sendReact(
                                message.channel,
                                "```py\nError: " + repr(ex).replace("`", "") + "\n```",
                                reacts="❎",
                            ))
                try:
                    self.proc_call.pop(message.id)
                except KeyError:
                    pass

    async def handleUpdate(self, force=False):
        if not hasattr(self, "stat_timer"):
            self.stat_timer = 0
        if not hasattr(self, "lastCheck"):
            self.lastCheck = 0
        if not hasattr(self, "busy"):
            self.busy = False
        if not hasattr(self, "status_iter"):
            self.status_iter = xrand(3)
        if utc() - self.lastCheck > 0.5 or force:
            while self.busy and not force:
                await asyncio.sleep(0.1)
            self.busy = True
            if not force:
                create_task(self.getState())
            try:
                #print("Sending update...")
                guilds = len(client.guilds)
                changed = guilds != self.guilds
                if changed or utc() > self.stat_timer:
                    self.stat_timer = utc() + float(frand(5)) + 12
                    self.guilds = guilds
                    try:
                        u = await self.fetch_user(self.owner_id)
                        n = u.name
                        place = ", from " + uniStr(n) + "'" + "s" * (n[-1] != "s") + " place!"
                        activity = discord.Streaming(
                            name=(
                                "live to " + uniStr(guilds) + " server"
                                + "s" * (guilds != 1) + place
                            ),
                            url=self.website,
                        )
                        activity.game = self.website
                        if changed:
                            print(repr(activity))
                        status = (discord.Status.online, discord.Status.dnd, discord.Status.idle)[self.status_iter]
                        try:
                            await client.change_presence(activity=activity, status=status)
                        except discord.HTTPException:
                            print(traceback.format_exc())
                            await asyncio.sleep(3)
                        except:
                            pass
                    except discord.NotFound:
                        pass
                    self.status_iter = (self.status_iter + 1) % 3
            except:
                print(traceback.format_exc())
            for u in self.database.values():
                if utc() - u.used > u.rate_limit or force:
                    create_task(u())
                    create_task(self.verifyDelete(u))
            self.busy = False

    async def ensureWebhook(self, channel, force=False):
        if not hasattr(self, "cw_cache"):
            self.cw_cache = freeClass()
        wlist = None
        if channel.id in self.cw_cache:
            if utc() - self.cw_cache[channel.id].time > 300 or force:
                self.cw_cache.pop(channel.id)
            else:
                self.cw_cache[channel.id].time = utc()
                wlist = [self.cw_cache[channel.id].webhook]
        if not wlist:
            wlist = await channel.webhooks()
        if not wlist:
            w = await channel.create_webhook(name=bot.client.user.name)
        else:
            w = random.choice(wlist)
        self.cw_cache[channel.id] = freeClass(time=utc(), webhook=w)
        return w

    class userGuild(discord.Object):

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

            def fetch_message(self, id):
                return bot.fetch_message(id, self.channel)

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
            self.roles = []
            self.emojis = []
            self.get_channel = lambda *void1, **void2: self.channel
            self.owner_id = client.user.id
            self.owner = client.user
            self.fetch_member = bot.fetch_user

        filesize_limit = 8388608
        bitrate_limit = 98304
        emoji_limit = 0
        large = False
        description = ""
        max_members = 2
        unavailable = False
        isDM = True
        ghost = True

    class ghostUser(discord.abc.Snowflake):
        
        def __init__(self):
            self.id = 0
            self.name = "[USER DATA NOT FOUND]"
            self.discriminator = "0000"
            self.avatar = ""
            self.avatar_url = ""
            self.bot = False
            self.display_name = ""

        __repr__ = lambda self: str(self.name) + "#" + str(self.discriminator)
        system = False
        history = lambda *void1, **void2: None
        dm_channel = None
        create_dm = lambda self: None
        relationship = None
        is_friend = lambda self: None
        is_blocked = lambda self: None
        colour = color = discord.Colour(16777215)
        mention = "<@0>"
        created_at = 0
        ghost = True

    class ghostMessage(discord.abc.Snowflake):
        
        def __init__(self):
            self.author = bot.ghostUser()
            self.content = "```css\n" + uniStr("[MESSAGE DATA NOT FOUND]") + "```"
            self.channel = None
            self.guild = None
            self.id = 0

        async def delete(self, *void1, **void2):
            pass

        __repr__ = lambda self: str(
            (
                self.system_content,
                self.content
            )[len(self.system_content) > len(self.content)]
        )
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


async def processMessage(message, msg, edit=True, orig=None, cb_argv=None, loop=False):
    if bot.closed:
        return
    cpy = msg
    if msg[:2] == "> ":
        msg = msg[2:]
    elif msg[:2] == "||" and msg[-2:] == "||":
        msg = msg[2:-2]
    msg = msg.replace("`", "")
    while len(msg):
        if msg[0] in (" ", "\n", "\t", "\r"):
            msg = msg[1:]
        else:
            break
    user = message.author
    guild = message.guild
    u_id = user.id
    channel = message.channel
    c_id = channel.id
    if guild:
        g_id = guild.id
    else:
        g_id = 0
    if g_id:
        try:
            enabled = bot.data.enabled[c_id]
        except KeyError:
            try:
                enabled = bot.data.enabled[c_id] = ["main", "string", "admin"]
                bot.update()
            except KeyError:
                enabled = ["main", "admin"]
    else:
        enabled = list(bot.categories)
    u_perm = bot.getPerms(u_id, guild)
    admin = not inf > u_perm
    mention = (
        "<@" + str(client.user.id) + ">",
        "<@!" + str(client.user.id) + ">",
    )
    if u_id == client.user.id:
        prefix = bot.prefix
    else:
        prefix = bot.getPrefix(guild)
    op = False
    comm = msg
    for check in (prefix, *mention):
        if comm.startswith(check):
            comm = comm[len(check):]
            op = True
        while len(comm) and comm[0] == " ":
            comm = comm[1:]
    if (u_perm <= -inf and op) or msg.replace(" ", "") in mention:
        if not u_perm < 0 and not u_perm <= -inf:
            create_task(sendReact(
                channel,
                (
                    "Hi, did you require my services for anything? Use `"
                    + prefix + "?` or `" + prefix + "help` for help."
                ),
                reacts="❎",
            ))
        else:
            print(
                "Ignoring command from suspended user "
                + user.name + " (" + str(u_id) + "): "
                + limStr(message.content, 256)
            )
            create_task(sendReact(
                channel,
                "Sorry, you are currently not permitted to request my services.",
                reacts="❎",
            ))
        return
    run = False
    if op:
        if len(comm) and comm[0] == "?":
            check = comm[0]
            i = 1
        else:
            i = len(comm)
            for end in " ?-+":
                if end in comm:
                    i2 = comm.index(end)
                    if i2 < i:
                        i = i2
            check = reconstitute(comm[:i]).lower().replace("*", "").replace("_", "").replace("||", "")
        if check in bot.commands:
            for command in bot.commands[check]:
                if command.catg in enabled or admin:
                    alias = command.__name__
                    for a in command.alias:
                        if a.lower() == check:
                            alias = a
                    alias = alias.lower()
                    argv = comm[i:]
                    run = True
                    print(str(guild.id) + ": " + str(user) + " (" + str(u_id) + ") issued command " + msg)
                    req = command.min_level
                    try:
                        if u_perm is not nan:
                            if not u_perm >= req:
                                raise command.permError(u_perm, req, "for command " + alias)
                            x = command.rate_limit
                            if x:
                                d = command.used
                                t = d.get(u_id, -inf)
                                wait = utc() - t - x
                                if wait > -1:
                                    if wait < 0:
                                        w = max(0.2, -wait)
                                        d[u_id] = max(t, utc()) + w
                                        await asyncio.sleep(w)
                                    if len(d) >= 4096:
                                        d.pop(next(iter(d)))
                                    d[u_id] = max(t, utc())
                                else:
                                    raise TooManyRequests("Command has a rate limit of " + sec2Time(x) + "; please wait " + sec2Time(-wait) + ".")
                        flags = {}
                        if cb_argv is not None:
                            argv = cb_argv
                            if loop:
                                addDict(flags, {"h": 1})
                        if argv:
                            if not hasattr(command, "no_parse"):
                                argv = reconstitute(argv)
                            argv = argv.strip()
                            if hasattr(command, "flags"):
                                flaglist = command.flags
                                for q in "?-+":
                                    if q in argv:
                                        for char in flaglist:
                                            flag = q + char
                                            for r in (flag, flag.upper()):
                                                while len(argv) >= 4 and r in argv:
                                                    found = False
                                                    i = argv.index(r)
                                                    if i == 0 or argv[i - 1] == " " or argv[i - 2] == q:
                                                        try:
                                                            if argv[i + 2] == " " or argv[i + 2] == q:
                                                                argv = argv[:i] + argv[i + 2:]
                                                                addDict(flags, {char: 1})
                                                                found = True
                                                        except (IndexError, KeyError):
                                                            pass
                                                    if not found:
                                                        break
                                    if q in argv:
                                        for char in flaglist:
                                            flag = q + char
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
                            argv = argv.strip()
                        if not argv:
                            args = []
                        else:
                            argv = argv.replace("\n", " ").replace("\r", "").replace("\t", " ")
                            try:
                                args = shlex.split(argv)
                            except ValueError:
                                args = argv.split(" ")
                        if guild is None:
                            if getattr(command, "server_only", False):
                                raise ReferenceError("This command is only available in servers.")
                            guild = bot.userGuild(
                                user=user,
                                channel=channel,
                            )
                            channel = guild.channel
                        tc = getattr(command, "time_consuming", False)
                        if not loop and tc:
                            create_task(channel.trigger_typing())
                        for u in bot.database.values():
                            f = getattr(u, "_command_", None)
                            if f is not None:
                                await f(user, command)
                        timeout = getattr(f, "_timeout_", 1) * bot.timeout
                        response = await asyncio.wait_for(command(
                            client=client,          # for interfacing with discord
                            bot=bot,                # for interfacing with bot's database
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
                        ), timeout=timeout)
                        if response is not None:
                            if issubclass(type(response), Exception):
                                raise response
                            elif bool(response) is not False:
                                if type(response) is tuple:
                                    response, react = response
                                    if react == 1:
                                        react = "❎"
                                else:
                                    react = False
                                sent = None
                                if type(response) is list:
                                    for r in response:
                                        create_task(channel.send(r))
                                elif type(response) is dict or isinstance(response, freeClass):
                                    if react:
                                        sent = await channel.send(**response)
                                    else:
                                        create_task(channel.send(**response))
                                else:
                                    if type(response) is str and len(response) <= 2000:
                                        if react:
                                            sent = await channel.send(response)
                                        else:
                                            create_task(channel.send(response))
                                    else:
                                        if type(response) is not bytes:
                                            response = bytes(str(response), "utf-8")
                                            filemsg = "Response too long for message."
                                        else:
                                            filemsg = "Response data:"
                                        if len(response) <= guild.filesize_limit:
                                            b = io.BytesIO(response)
                                            f = discord.File(b, filename="message.txt")
                                            create_task(sendFile(channel, filemsg, f))
                                        else:
                                            raise OverflowError("Response too long for file upload.")
                                if sent is not None:
                                    await sent.add_reaction(react)
                    except (TimeoutError, asyncio.exceptions.TimeoutError):
                        print(msg)
                        raise TimeoutError("Request timed out.")
                    except Exception as ex:
                        errmsg = limStr("```py\nError: " + repr(ex).replace("`", "") + "\n```", 2000)
                        print(traceback.format_exc())
                        create_task(sendReact(
                            channel,
                            errmsg,
                            reacts="❎",
                        ))
    if not run and u_id != client.user.id and not u_perm <= -inf:
        temp = to_alphanumeric(reconstitute(cpy).lower())
        for u in bot.database.values():
            f = getattr(u, "_nocommand_", None)
            if f is not None:
                create_task(safeCoro(f(
                    text=temp,
                    edit=edit,
                    orig=orig,
                    message=message,
                    perm=u_perm,
                )))


async def heartbeatLoop():
    print("Heartbeat Loop initiated.")
    try:
        while True:
            try:
                bot.client
            except NameError:
                sys.exit()
            if bot.heartbeat in os.listdir():
                try:
                    os.remove(bot.heartbeat)
                except:
                    print(traceback.format_exc())
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        sys.exit(1)        


async def updateLoop():
    print("Update loop initiated.")
    autosave = 0
    while True:
        try:
            if utc() - autosave > 60:
                autosave = utc()
                bot.update()
            while bot.blocked > 0:
                print("Update event blocked.")
                bot.blocked -= 1
                await asyncio.sleep(1)
            await bot.handleUpdate()
            await asyncio.sleep(frand(2) + 2)
        except:
            print(traceback.format_exc())
        

@client.event
async def on_ready():
    print("Successfully connected as " + str(client.user))
    try:
        await bot.getState()
    except:
        print(traceback.format_exc())
    print("Servers: ")
    for guild in client.guilds:
        if guild.unavailable:
            print("> " + str(guild.id) + " is not available.")
        else:
            print("> " + guild.name)
    await bot.handleUpdate()
    if not hasattr(bot, "started"):
        bot.started = True
        create_task(updateLoop())
        create_task(heartbeatLoop())


@client.event
async def on_guild_join(guild):
    print("New server: " + str(guild))
    g = await bot.fetch_guild(guild.id)
    m = guild.get_member(client.user.id)
    channel = await bot.get_sendable(g, m)
    emb = discord.Embed(colour=discord.Colour(8364031))
    url = strURL(client.user.avatar_url)
    emb.set_author(name=client.user.name, url=url, icon_url=url)
    emb.description = (
        "Hi there! I'm " + client.user.name
        + ", a multipurpose discord bot created by <@"
        + str(bot.owner_id) + ">. Thanks for adding me"
    )
    user = None
    try:
        a = await guild.audit_logs(limit=5, action=discord.AuditLogAction.bot_add).flatten()
    except discord.Forbidden:
        pass
    else:
        for e in a:
            if e.target.id == client.user.id:
                user = e.user
                break
    if user is not None:
        emb.description += ", <@" + str(user.id) + ">"
    emb.description += (
        "!\nMy default prefix is `" + bot.prefix + "`, which can be changed as desired on a per-server basis. Mentioning me also serves as an alias for all prefixes.\n"
        + "For more information, use the `" + bot.prefix + "help` command, and my source code is available at " + bot.website + " for those who are interested.\n"
        + "Pleased to be at your service 🙂"
    )
    if not m.guild_permissions.administrator:
        emb.add_field(name="Psst!", value=(
            "I noticed you haven't given me administrator permissions here.\n"
            + "That's completely understandable if intentional, but please note that without it, some features may not function well, or not at all."
        ))
    await channel.send(embed=emb)

    
async def seen(user, delay=0):
    for u in bot.database.values():
        f = getattr(u, "_seen_", None)
        if f is not None:
            try:
                await f(user=user, delay=delay)
            except:
                print(traceback.format_exc())


async def checkDelete(message, reaction, user):
    if message.author.id == client.user.id:
        u_perm = bot.getPerms(user.id, message.guild)
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
                        await bot.silentDelete(message, exc=True)
                    except discord.NotFound:
                        pass


@client.event
async def on_raw_reaction_add(payload):
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        user = await bot.fetch_user(payload.user_id)
        message = await bot.fetch_message(payload.message_id, channel=channel)
    except discord.NotFound:
        return
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        create_task(seen(user))
        await bot.reactCallback(message, reaction, user)
        create_task(checkDelete(message, reaction, user))


@client.event
async def on_raw_reaction_remove(payload):
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        user = await bot.fetch_user(payload.user_id)
        message = await bot.fetch_message(payload.message_id, channel=channel)
    except discord.NotFound:
        return
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        create_task(seen(user))
        await bot.reactCallback(message, reaction, user)
        create_task(checkDelete(message, reaction, user))


@client.event
async def on_voice_state_update(member, before, after):
    if member.id == client.user.id:
        after = member.voice
        if after is not None:
            if after.mute or after.deaf:
                print("Unmuted self in " + member.guild.name)
                await member.edit(mute=False, deafen=False)
            await bot.handleUpdate()
    if member.voice is not None and not member.voice.afk:
        create_task(seen(member))


async def handleMessage(message, edit=True):
    cpy = msg = message.content
    try:
        if msg and msg[0] == "\\":
            cpy = msg[1:]
        await processMessage(message, cpy, edit, msg)
    except Exception as ex:
        errmsg = limStr("```py\nError: " + repr(ex).replace("`", "") + "\n```", 2000)
        print(traceback.format_exc())
        create_task(sendReact(
            message.channel,
            errmsg,
            reacts="❎",
        ))
    return


@client.event
async def on_typing(channel, user, when):
    for u in bot.database.values():
        f = getattr(u, "_typing_", None)
        if f is not None:
            try:
                await f(channel=channel, user=user)
            except:
                print(traceback.format_exc())
    create_task(seen(user, delay=10))


@client.event
async def on_message(message):
    bot.cacheMessage(message)
    guild = message.guild
    if guild:
        for u in bot.database.values():
            f = getattr(u, "_send_", None)
            if f is not None:
                try:
                    await f(message=message)
                except:
                    print(traceback.format_exc())
    create_task(seen(message.author))
    await bot.reactCallback(message, None, message.author)
    await handleMessage(message, False)


@client.event
async def on_user_update(before, after):
    for u in bot.database.values():
        f = getattr(u, "_user_update_", None)
        if f is not None:
            try:
                await f(before=before, after=after)
            except:
                print(traceback.format_exc())
    create_task(seen(after))


@client.event
async def on_member_update(before, after):
    for u in bot.database.values():
        f = getattr(u, "_member_update_", None)
        if f is not None:
            try:
                await f(before=before, after=after)
            except:
                print(traceback.format_exc())
    if str(before.status) != str(after.status) or str(before.activity) != str(after.activity):
        create_task(seen(after))


@client.event
async def on_member_join(member):
    for u in bot.database.values():
        f = getattr(u, "_join_", None)
        if f is not None:
            try:
                await f(user=member, guild=member.guild)
            except:
                print(traceback.format_exc())
    create_task(seen(member))

            
@client.event
async def on_member_remove(member):
    for u in bot.database.values():
        f = getattr(u, "_leave_", None)
        if f is not None:
            try:
                await f(user=member, guild=member.guild)
            except:
                print(traceback.format_exc())


@client.event
async def on_raw_message_delete(payload):
    try:
        message = payload.cached_message
        if message is None:
            raise LookupError
    except:
        channel = await bot.fetch_channel(payload.channel_id)
        try:
            message = await bot.fetch_message(payload.message_id, channel)
            if message is None:
                raise LookupError
        except:
            message = bot.ghostMessage()
            message.channel = channel
            try:
                message.guild = channel.guild
            except AttributeError:
                message.guild = None
            message.id = payload.message_id
            message.created_at = snowflake_time(message.id)
            message.author = await bot.fetch_user(bot.deleted_user)
    guild = message.guild
    if guild:
        for u in bot.database.values():
            f = getattr(u, "_delete_", None)
            if f is not None:
                try:
                    await f(message=message)
                except:
                    print(traceback.format_exc())
    bot.deleteMessage(message)


@client.event
async def on_raw_bulk_message_delete(payload):
    try:
        messages = payload.cached_messages
        if messages is None:
            raise LookupError
    except:
        messages = deque()
        channel = await bot.fetch_channel(payload.channel_id)
        for m_id in payload.message_ids:
            try:
                message = await bot.fetch_message(m_id, channel)
                if message is None:
                    raise LookupError
            except:
                message = bot.ghostMessage()
                message.channel = channel
                try:
                    message.guild = channel.guild
                except AttributeError:
                    message.guild = None
                message.id = m_id
                message.created_at = snowflake_time(message.id)
                message.author = await bot.fetch_user(bot.deleted_user)
            messages.append(message)
    for message in messages:
        guild = message.guild
        if guild:
            for u in bot.database.values():
                f = getattr(u, "_delete_", None)
                if f is not None:
                    try:
                        await f(message=message, bulk=True)
                    except:
                        print(traceback.format_exc())
        bot.deleteMessage(message)


@client.event
async def on_guild_channel_delete(channel):
    print(channel, "was deleted from", channel.guild)
    guild = channel.guild
    if guild:
        for u in bot.database.values():
            f = getattr(u, "_channel_delete_", None)
            if f is not None:
                try:
                    await f(channel=channel, guild=guild)
                except:
                    print(traceback.format_exc())


@client.event
async def on_member_ban(guild, user):
    print(user, "was banned from", guild)
    if guild:
        for u in bot.database.values():
            f = getattr(u, "_ban_", None)
            if f is not None:
                try:
                    await f(user=user, guild=guild)
                except:
                    print(traceback.format_exc())


@client.event
async def on_guild_remove(guild):
    if guild.id in bot.cache.guilds:
        bot.cache.guilds.pop(guild.id)
    print(guild, "removed.")


async def updateEdit(before, after):
    if before.content == after.content:
        before = bot.ghostMessage()
        before.channel = after.channel
        before.guild = after.guild
        before.author = after.author
        before.id = after.id
        before.created_at = snowflake_time(before.id)
    guild = after.guild
    if guild:
        for u in bot.database.values():
            f = getattr(u, "_edit_", None)
            if f is not None:
                try:
                    await f(before=before, after=after)
                except:
                    print(traceback.format_exc())
    create_task(seen(after.author))


@client.event
async def on_message_edit(before, after):
    if before.content != after.content:
        bot.cacheMessage(after)
        await handleMessage(after)
        await updateEdit(before, after)


@client.event
async def on_raw_message_edit(payload):
    if payload.cached_message is not None:
        return
    try:
        c_id = payload.data.get("channel_id", 0)
        channel = await bot.fetch_channel(c_id)
        before = await bot.fetch_message(payload.message_id, channel)
        if before is None:
            raise LookupError
    except:
        before = bot.ghostMessage()
        before.channel = await bot.fetch_channel(c_id)
        before.guild = channel.guild
        before.id = payload.message_id
        before.created_at = snowflake_time(before.id)
    if before:
        after = await before.channel.fetch_message(payload.message_id)
        bot.cacheMessage(after)
        if not hasattr(before, "ghost"):
            if before.content == after.content:
                return
        await handleMessage(after)
        await updateEdit(before, after)


if __name__ == "__main__":
    bot = Bot()
    bot.run()