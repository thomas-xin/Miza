#!/usr/bin/python3

from common import *

# Allows importing from commands and misc directories.
sys.path.insert(1, "commands")
sys.path.insert(1, "misc")

# discord.py client object.
client = discord.AutoShardedClient(
    max_messages=1024,
    heartbeat_timeout=30,
)


# Main class containing all global data
class Bot:
    
    timeout = 24
    website = "https://github.com/thomas-xin/Miza"
    discord_icon = "https://cdn.discordapp.com/embed/avatars/0.png"
    heartbeat = "heartbeat.tmp"
    restart = "restart.tmp"
    shutdown = "shutdown.tmp"
    savedata = "data.json"
    authdata = "auth.json"
    caches = ("guilds", "channels", "users", "roles", "emojis", "messages", "members", "deleted")
    client = client
    prefix = "~"
    # This is a fixed ID apparently
    deleted_user = 456226577798135808
    _globals = globals()
            
    def __init__(self):
        self.bot = self
        self.closed = False
        self.loaded = False
        self.cache = cdict({c: cdict() for c in self.caches})
        self.cw_cache = cdict()
        self.events = mdict()
        self.proc_call = cdict()
        self.mention = ()
        print("Time: " + str(datetime.datetime.now()))
        print("Initializing...")
        # O(1) time complexity for searching directory
        directory = dict.fromkeys(os.listdir())
        [os.mkdir(folder) for folder in ("cache", "saves", "deleted") if folder not in directory]
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
            owner_id = auth["owner_id"]
            if type(owner_id) not in (list, tuple):
                owner_id = [owner_id]
            self.owners = cdict({int(i): True for i in owner_id})
        except KeyError:
            self.owners = ()
            print("WARNING: owner_id not found. Unable to locate owner.")
        # Initialize rest of bot variables
        create_task(heartbeatLoop())
        self.proc = psutil.Process()
        self.getModules()
        self.guilds = 0
        self.blocked = 0
        self.updated = False
        self.started = False
        self.ready = False
        self.embedSenders = cdict()
        # Assign bot cache to global variables for convenience
        globals().update(self.cache)

    __call__ = lambda self: self

    # Waits 2 seconds and shuts down.
    def setshutdown(self):
        time.sleep(2)
        f = open(self.shutdown, "wb")
        f.close()
        sys.exit(1)

    # Starts up client.
    def run(self):
        print("Attempting to authorize with token " + self.token + ":")
        try:
            eloop.run_until_complete(client.start(self.token))
        except (KeyboardInterrupt, SystemExit):
            eloop.run_until_complete(client.logout())
            eloop.close()
            raise SystemExit

    # A reimplementation of the print builtin function.
    def print(self, *args, sep=" ", end="\n"):
        sys.__stdout__.write(str(sep).join(str(i) for i in args) + end)

    # A garbage collector for empty and unassigned objects in the database.
    async def verifyDelete(self, obj):
        if not self.ready or hasattr(obj, "no_delete"):
            return
        if obj.checking > utc():
            return
        obj.checking = utc() + 30
        data = obj.data
        for key in tuple(data):
            if key != 0 and type(key) is not str:
                try:
                    # Database keys may be user, guild, or channel IDs
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
        obj.checking = utc() + 10
        self.started = True

    # Calls a bot event, triggered by client events or others, across all bot databases. Calls may be sync or async.
    async def event(self, ev, *args, **kwargs):
        events = self.events.get(ev, ())
        if len(events) == 1:
            try:
                return await forceCoro(events[0](*args, **kwargs))
            except:
                print(traceback.format_exc())
            return
        futs = [create_task(forceCoro(func(*args, **kwargs))) for func in events]
        out = deque()
        for fut in futs:
            try:
                res = await fut
                out.append(res)
            except:
                print(traceback.format_exc())
        return out

    # Gets the first accessable text channel in the target guild.
    async def get_first_sendable(self, guild, member):
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

    # Fetches either a user or channel object from ID, using the bot cache when possible.
    async def fetch_sendable(self, s_id):
        if type(s_id) is not int:
            try:
                s_id = int(s_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid user identifier: " + str(s_id))
        try:
            return self.get_user(s_id)
        except KeyError:
            try:
                return self.cache.channels[s_id]
            except KeyError:
                try:
                    user = await client.fetch_user(s_id)
                except LookupError:
                    channel = await client.fetch_channel(s_id)
                    self.cache.channels[s_id] = channel
                    self.limitCache("channels")
                    return channel
                self.cache.users[u_id] = user
                self.limitCache("users")
                return user

    # Fetches a user from ID, using the bot cache when possible.
    async def fetch_user(self, u_id):
        try:
            return self.get_user(u_id)
        except KeyError:
            user = await client.fetch_user(u_id)
        self.cache.users[u_id] = user
        self.limitCache("users")
        return user

    # Gets a user from ID, using the bot cache.
    def get_user(self, u_id, replace=False):
        if type(u_id) is not int:
            try:
                u_id = int(u_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid user identifier: " + str(u_id))
        try:
            return self.cache.users[u_id]
        except KeyError:
            pass
        if u_id == self.deleted_user:
            user = self.ghostUser()
            user.system = True
            user.name = "Deleted User"
            user.display_name = "Deleted User"
            user.id = u_id
            user.mention = "<@" + str(u_id) + ">"
            user.avatar_url = self.discord_icon
            user.created_at = snowflake_time(u_id)
        else:
            try:
                user = client.get_user(u_id)
                if user is None:
                    raise TypeError
            except:
                if replace:
                    return self.get_user(self.deleted_user)
                raise KeyError("Target user ID not found.")
        self.cache.users[u_id] = user
        self.limitCache("users")
        return user

    # Fetches a member in the target server by ID or name lookup.
    async def fetch_member_ex(self, u_id, guild=None):
        if type(u_id) is not int:
            try:
                u_id = int(u_id)
            except (TypeError, ValueError):
                pass
        member = None
        if type(u_id) is int:
            member = guild.get_member(u_id)
        if member is None:
            if type(u_id) is int:
                try:
                    member = await self.fetch_member(u_id, guild)
                except LookupError:
                    pass
            if member is None:
                members = guild.members
                if not members:
                    members = guild.members = await guild.fetch_members(limit=None)
                    guild._members = {m.id: m for m in members}
                if type(u_id) is not str:
                    u_id = str(u_id)
                try:
                    member = await strLookup(
                        members,
                        u_id,
                        qkey=userQuery1,
                        ikey=userIter1,
                        loose=False,
                    )
                except LookupError:
                    try:
                        member = await strLookup(
                        members,
                        u_id,
                        qkey=userQuery2,
                        ikey=userIter2,
                    )
                    except LookupError:
                        try:
                            member = await strLookup(
                                members,
                                u_id,
                                qkey=userQuery3,
                                ikey=userIter3,
                            )
                        except LookupError:
                            raise LookupError("Unable to find member data.")
        return member

    # Fetches the first seen instance of the target user as a member in any shared server.
    async def fetch_member(self, u_id, guild=None, find_others=False):
        if type(u_id) is not int:
            try:
                u_id = int(u_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid user identifier: " + str(u_id))
        if find_others:
            try:
                member = self.cache.members[u_id].guild.get_member(u_id)
                if member is None:
                    raise LookupError
                return member
            except LookupError:
                pass
        g = bot.cache.guilds
        if guild is None:
            guilds = list(bot.cache.guilds.values())
        else:
            if find_others:
                guilds = [g[i] for i in g if g[i].id != guild.id]
                guilds.insert(0, guild)
            else:
                guilds = [guild]
        member = None
        for i, guild in enumerate(guilds, 1):
            member = guild.get_member(u_id)
            if member is not None:
                break
            if not i & 4095:
                await asyncio.sleep(0.2)
        if member is None:
            raise LookupError("Unable to find member data.")
        self.cache.members[u_id] = member
        self.limitCache("members")
        return member

    # Fetches a guild from ID, using the bot cache when possible.
    async def fetch_guild(self, g_id, follow_invites=True):
        if type(g_id) is not int:
            try:
                g_id = int(g_id)
            except (ValueError, TypeError):
                if follow_invites:
                    try:
                        # Parse and follow invites to get partial guild info
                        invite = await client.fetch_invite(g_id.strip("< >"))
                        g = invite.guild
                        if not hasattr(g, "member_count"):
                            guild = cdict(member_count=invite.approximate_member_count)
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
                raise TypeError("Invalid server identifier: " + str(g_id))
        try:
            return self.cache.guilds[g_id]
        except KeyError:
            pass
        try:
            guild = client.get_guild(g_id)
            if guild is None:
                raise EOFError
        except:
            guild = await client.fetch_guild(g_id)
        self.cache.guilds[g_id] = guild
        self.limitCache("guilds", limit=65536)
        return guild

    # Fetches a channel from ID, using the bot cache when possible.
    async def fetch_channel(self, c_id):
        if type(c_id) is not int:
            try:
                c_id = int(c_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid channel identifier: " + str(c_id))
        try:
            return self.cache.channels[c_id]
        except KeyError:
            pass
        channel = await client.fetch_channel(c_id)
        self.cache.channels[c_id] = channel
        self.limitCache("channels")
        return channel

    # Fetches a message from ID and channel, using the bot cache when possible.
    async def fetch_message(self, m_id, channel=None):
        if type(m_id) is not int:
            try:
                m_id = int(m_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid message identifier: " + str(m_id))
        try:
            return self.cache.messages[m_id]
        except KeyError:
            pass
        if channel is None:
            raise LookupError("Message data not found.")
        try:
            int(channel)
            channel = await self.fetch_channel(channel)
        except TypeError:
            pass
        message = await channel.fetch_message(m_id)
        if message is not None:
            self.cacheMessage(message)
        return message

    # Fetches a role from ID and guild, using the bot cache when possible.
    async def fetch_role(self, r_id, guild):
        if type(r_id) is not int:
            try:
                r_id = int(r_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid role identifier: " + str(r_id))
        try:
            return self.cache.roles[r_id]
        except KeyError:
            pass
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
                raise LookupError("Role not found.")
        self.cache.roles[r_id] = role
        self.limitCache("roles")
        return role

    # Fetches an emoji from ID and guild, using the bot cache when possible.
    async def fetch_emoji(self, e_id, guild=None):
        if type(e_id) is not int:
            try:
                e_id = int(e_id)
            except (ValueError, TypeError):
                raise TypeError("Invalid emoji identifier: " + str(e_id))
        try:
            return self.cache.emojis[e_id]
        except KeyError:
            pass
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
    
    # Searches the bot database for a webhook mimic from ID.
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

    # Gets the DM channel for the target user, creating a new one if none exists.
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

    # Finds URLs in a string, following any discord message links found.
    async def followURL(self, url, it=None, best=False, preserve=True, images=True, allow=False, limit=None):
        if limit is not None and limit <= 0:
            return []
        if it is None:
            urls = findURLs(url)
            if not urls:
                return []
            it = {}
        else:
            urls = [url]
        out = deque()
        if preserve or allow:
            lost = deque()
        else:
            lost = None
        if images:
            medias = ("video", "image", "thumbnail")
        else:
            medias = "video"
        for url in urls:
            # An attempt at checking for discord message links
            check = url[:64]
            if "channels/" in check and "discord" in check:
                found = deque()
                spl = url[url.index("channels/") + 9:].replace("?", "/").split("/")
                c = await self.fetch_channel(spl[1])
                m = await self.fetch_message(spl[2], c)
                # All attachments should be valid URLs
                if best:
                    found.extend(bestURL(a) for a in m.attachments)
                else:
                    found.extend(a.url for a in m.attachments)
                found.extend(findURLs(m.content))
                # Attempt to find URLs in embed contents
                for e in m.embeds:
                    for a in medias:
                        obj = getattr(e, a, None)
                        if obj:
                            if best:
                                url = bestURL(obj)
                            else:
                                url = obj.url
                            if url:
                                found.append(url)
                                break
                # Attempt to find URLs in embed descriptions
                [found.extend(findURLs(e.description)) for e in m.embeds if e.description]
                for u in found:
                    # Do not attempt to find the same URL twice
                    if u not in it:
                        it[u] = True
                        if not len(it) & 255:
                            await asyncio.sleep(0.2)
                        found2 = await self.followURL(u, it, best=best, preserve=preserve, images=images, allow=allow, limit=limit)
                        if len(found2):
                            out.extend(found2)
                        elif allow and m.content:
                            lost.append(m.content)
                        elif preserve:
                            lost.append(u)
            else:
                out.append(url)
        if lost:
            out.extend(lost)
        if not out:
            return urls
        if limit is not None:
            return list(out)[:limit]
        return out

    # Follows a message link, replacing emojis and user mentions with their icon URLs.
    async def followImage(self, url):
        temp = findURLs(url)
        if temp:
            return temp
        users = findUsers(url)
        emojis = findEmojis(url)
        out = deque()
        if users:
            futs = [create_task(self.fetch_user(verifyID(u))) for u in users]
            for fut in futs:
                try:
                    res = await fut
                    out.append(bestURL(res))
                except LookupError:
                    pass
        for s in emojis:
            s = s[3:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            out.append("https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1")
        return out

    # Inserts a message into the bot cache, discarding existing ones if full.
    def cacheMessage(self, message):
        self.cache.messages[message.id] = message
        self.limitCache("messages", 4194304)
        return message

    # Deletes a message from the bot cache.
    def deleteMessage(self, message):
        self.cache.messages.pop(message.id, None)
        if not message.author.bot:
            s = strMessage(message, username=True)
            ch = "deleted/" + str(message.channel.id) + ".txt"
            print(s, file=ch)

    # Limits a cache to a certain amount, discarding oldest entries first.
    def limitCache(self, cache=None, limit=1048576):
        if cache is not None:
            caches = [self.cache[cache]]
        else:
            caches = self.cache.values()
        for c in caches:
            while len(c) > limit:
                c.pop(next(iter(c)))
    
    # Updates bot cache from the discord.py client cache.
    def updateClient(self):
        self.cache.guilds.update(client._connection._guilds)
        self.cache.emojis.update(client._connection._emojis)
        self.cache.users.update(client._connection._users)
        self.cache.channels.update(client._connection._private_channels)

    # Updates bot cache from the discord.py guild objects.
    def cacheFromGuilds(self):
        for i, guild in enumerate(client.guilds, 1):
            self.cache.channels.update(guild._channels)
            self.cache.roles.update(guild._roles)
            if not i & 63:
                time.sleep(0.2)

    # Gets the target bot prefix for the target guild, return the default one if none exists.
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

    # Gets effective permission level for the target user in a certain guild, taking into account roles.
    def getPerms(self, user, guild=None):
        try:
            u_id = user.id
        except AttributeError:
            u_id = int(user)
        if u_id in self.owners:
            return nan
        if self.isBlacklisted(u_id):
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
    
    # Gets effective permission level for the target role in a certain guild, taking into account permission values.
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

    # Sets the permission value for a snowflake in a guild to a value.
    def setPerms(self, user, guild, value):
        perms = self.data.perms
        try:
            u_id = user.id
        except AttributeError:
            u_id = user
        g_perm = setDict(perms, guild.id, {})
        g_perm.update({u_id: roundMin(value)})
        self.database.perms.update()

    # Checks if a message has been flagged as deleted by the deleted cache.
    def isDeleted(self, message):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        return self.cache.deleted.get(m_id, False)

    # Logs if a message has been deleted.
    def logDelete(self, message, no_log=False):
        try:
            m_id = int(message.id)
        except AttributeError:
            m_id = int(message)
        self.cache.deleted[m_id] = no_log + 2
        self.limitCache("deleted", limit=4096)
    
    # Silently deletes a message, bypassing logs.
    async def silentDelete(self, message, exc=False, no_log=False, delay=None):
        if delay:
            await asyncio.sleep(float(delay))
        try:
            self.logDelete(message, no_log)
            await message.delete()
        except:
            self.cache.deleted.pop(message.id, None)
            if exc:
                raise

    # Checks if a guild is trusted.
    def isTrusted(self, g_id):
        try:
            trusted = self.data.trusted
        except (AttributeError, KeyError):
            return False
        return g_id in trusted

    # Checks if a user is blacklisted from the bot.
    def isBlacklisted(self, u_id):
        u_id = int(u_id)
        if u_id in self.owners or u_id == client.user.id:
            return False
        try:
            return self.data.blacklist.get(u_id, False)
        except KeyError:
            return True
    
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

    # Evaluates a math formula to a float value, using a math process from the subprocess pool when necessary.
    async def evalMath(self, expr, obj, default=0, op=True):
        if op:
            # Allow mathematical operations on a default value
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip()
                    _op = at
            num = await self.evalMath(expr, obj, op=False)
            if _op is not None:
                num = getattr(float(default), _op)(num)
            return num
        f = expr.strip()
        try:
            if not f:
                r = [0]
            else:
                s = f.casefold()
                if s in ("t", "true", "y", "yes", "on"):
                    r = [True]
                elif s in ("f", "false", "n", "no", "off"):
                    r = [False]
                elif s == "inf":
                    r = [inf]
                elif s == "-inf":
                    r = [-inf]
                elif s in ("nan", "-nan"):
                    r = [nan]
                else:
                    r = [ast.literal_eval(f)]
        except (ValueError, TypeError, SyntaxError):
            r = await self.solveMath(f, obj, 16, 0)
        x = r[0]
        try:
            while True:
                if type(x) is str:
                    raise TypeError
                x = tuple(x)[0]
        except TypeError:
            pass
        return roundMin(float(x))

    # Evaluates a math formula to a list of answers, using a math process from the subprocess pool when necessary.
    async def solveMath(self, f, obj, prec, r, timeout=12, authorize=False):
        f = f.strip()
        try:
            if obj is None:
                key = None
            elif hasattr(obj, "ghost"):
                key = self.deleted_user
            else:
                key = obj.id
        except AttributeError:
            key = int(obj)
        # Bot owners have no semaphore limit
        if key in self.owners:
            key = None
        return await mathProc(f, int(prec), int(r), key, timeout=12, authorize=authorize)

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
    connectors = re.compile("[^a-z](and|at)[^a-z]", re.I)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # Evaluates a time input, using a math process from the subprocess pool when necessary.
    async def evalTime(self, expr, obj, default=0, op=True):
        if op:
            # Allow mathematical operations on a default value
            _op = None
            for op, at in self.op.items():
                if expr.startswith(op):
                    expr = expr[len(op):].strip(" ")
                    _op = at
            num = await self.evalTime(expr, obj, op=False)
            if _op is not None:
                num = getattr(float(default), _op)(num)
            return num
        t = 0
        if expr:
            f = None
            if " " in expr:
                # Parse timezones first
                try:
                    args = shlex.split(expr)
                except ValueError:
                    args = expr.split()
                for a in (args[0], args[-1]):
                    tz = a.casefold()
                    if tz in TIMEZONES:
                        t = -get_timezone(tz)
                        expr = expr.replace(a, "")
                        break
            try:
                # Try to evaluate time inputs
                if ":" in expr:
                    data = expr.split(":")
                    mult = 1
                    while len(data):
                        t += await self.evalMath(data[-1], obj) * mult
                        data = data[:-1]
                        if mult <= 60:
                            mult *= 60
                        elif mult <= 3600:
                            mult *= 24
                        elif len(data):
                            raise TypeError("Too many time arguments.")
                else:
                    # Otherwise move on to main parser
                    f = re.sub(self.connectors, " ", expr).casefold()
                    for tc in self.timeChecks:
                        for check in reversed(self.timeChecks[tc]):
                            if check in f:
                                i = f.index(check)
                                isnt = i + len(check) < len(f) and f[i + len(check)] in self.alphabet
                                if not i or f[i - 1] in self.alphabet or isnt:
                                    continue
                                n = await self.evalMath(f[:i], obj)
                                s = TIMEUNITS[tc]
                                if type(s) is list:
                                    s = s[0]
                                t += s * n
                                f = f[i + len(check):]
                    if f.strip():
                        t += await self.evalMath(f, obj)
            except:
                # Use datetime parser if regular parser fails
                t = utc_ts(tparser.parse(f if f is not None else expr)) - utc_ts(tparser.parse("0s"))
        if type(t) is not float:
            t = float(t)
        return t

    ipCheck = re.compile("^([0-9]{1,3}\\.){3}[0-9]{1,3}$")

    # Updates the bot's stored external IP address.
    def updateIP(self, ip):
        if re.search(self.ipCheck, ip):
            self.ip = ip

    # Gets the external IP address from api.ipify.org
    async def getIP(self):
        resp = await Request("https://api.ipify.org", decode=True, aio=True)
        self.updateIP(resp)

    # Gets the amount of active processes, threads, coroutines.
    def getActive(self):
        procs = 2 + sum(1 for c in self.proc.children(True))
        thrds = self.proc.num_threads()
        coros = sum(1 for i in asyncio.all_tasks())
        return hlist((procs, thrds, coros))

    # Gets the CPU and memory usage of a process over a period of 1 second.
    async def getProcState(self, proc):
        try:
            create_future_ex(proc.cpu_percent, priority=True)
            await asyncio.sleep(1)
            c = await create_future(proc.cpu_percent, priority=True)
            m = await create_future(proc.memory_percent, priority=True)
            return float(c), float(m)
        except psutil.NoSuchProcess:
            return 0, 0

    # Gets the total size of the cache folder.
    getCacheSize = lambda self: sum(os.path.getsize("cache/" + fn) for fn in os.listdir("cache"))

    # Gets the status of the bot.
    async def getState(self):
        stats = hzero(3)
        if getattr(self, "currState", None) is None:
            self.currState = stats
        procs = await create_future(self.proc.children, recursive=True, priority=True)
        procs.append(self.proc)
        tasks = [self.getProcState(p) for p in procs]
        resp = await recursiveCoro(tasks)
        stats += [sum(st[0] for st in resp), sum(st[1] for st in resp), 0]
        cpu = await create_future(psutil.cpu_count, priority=True)
        mem = await create_future(psutil.virtual_memory, priority=True)
        disk = await create_future(self.getCacheSize, priority=True)
        # CPU is totalled across all cores
        stats[0] /= cpu
        # Memory is in %
        stats[1] *= mem.total / 100
        stats[2] = disk
        self.currState = stats
        self.size2 = cdict()
        files = await create_future(os.listdir, "misc")
        for f in files:
            path = "misc/" + f
            if iscode(path):
                self.size2[f] = getLineCount(path)
        return stats

    # Loads a module containing commands and databases by name.
    def getModule(self, module):
        try:
            f = module
            f = ".".join(f.split(".")[:-1])
            path, module = module, f
            rename = module.casefold()
            print("Loading module " + rename + "...")
            if module in self._globals:
                if rename in self.categories:
                    self.unload(module)
                mod = importlib.reload(self._globals[module])
            else:
                mod = __import__(module)
            self._globals[module] = mod
            commands = hlist()
            dataitems = hlist()
            items = mod.__dict__
            for var in items.values():
                if callable(var) and var not in (Command, Database):
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
            self.dbitems[rename] = dataitems
            self.size[rename] = getLineCount("commands/" + path)
            if self.ready:
                for db in dataitems:
                    for f in dir(db):
                        if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                            func = getattr(db, f, None)
                            if callable(func):
                                self.events.append(f, func)
                    func = getattr(db, "_ready_", None)
                    if callable(func):
                        fut = create_task(forceCoro(func, bot=self))
                        time.sleep(0.05)
                        while True:
                            try:
                                fut.result()
                            except asyncio.InvalidStateError:
                                time.sleep(0.1)
                            else:
                                break
        except:
            print(traceback.format_exc())

    def unload(self, mod=None):
        if mod is None:
            mods = list(self.categories)
        else:
            mod = mod.casefold()
            if mod not in self.categories:
                raise KeyError
            mods = [mod]
        for mod in mods:
            rename = mod.casefold()
            for command in self.categories[rename]:
                for alias in command.alias:
                    alias = alias.replace("*", "").replace("_", "").replace("||", "").casefold()
                    coms = self.commands.get(alias)
                    if coms:
                        coms.remove(command)
                        print(alias, command)
                    if not coms:
                        self.commands.pop(alias)
            for db in self.dbitems[rename]:
                func = getattr(db, "_destroy_", None)
                if callable(func):
                    fut = create_task(forceCoro(func))
                    time.sleep(0.05)
                    while True:
                        try:
                            fut.result()
                        except asyncio.InvalidStateError:
                            time.sleep(0.1)
                        else:
                            break
                for f in dir(db):
                    if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                        func = getattr(db, f, None)
                        if callable(func):
                            bot.events[f].remove(func)
                            print(f, db)
                db.update(True)
                self.data.pop(db, None)
                self.database.pop(db, None)
            self.categories.pop(rename)
            self.dbitems.pop(rename)
            self.size.pop(rename)

    def reload(self, mod=None):
        if not mod:
            subKill()
            self.modload = deque()
            files = [i for i in os.listdir("commands") if iscode(i)]
            for f in files:
                self.modload.append(self.executor.submit(self.getModule, f))
            return [fut.result() for fut in self.modload]           
        return self.getModule(mod + ".py")

    # Loads all modules in the commands folder and initializes bot commands and databases.
    def getModules(self):
        files = [i for i in os.listdir("commands") if iscode(i)]
        self.categories = cdict()
        self.dbitems = cdict()
        self.commands = cdict()
        self.database = cdict()
        self.data = cdict()
        self.size = cdict()
        for f in os.listdir():
            if iscode(f):
                self.size[f] = getLineCount(f)
        self.modload = deque()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(files))
        for f in files:
            self.modload.append(self.executor.submit(self.getModule, f))
        self.loaded = True

    # Autosaves modified bot databases. Called once every minute and whenever the bot is about to shut down.
    def update(self):
        create_task(self.updateEmbeds())
        saved = hlist()
        try:
            for i in self.database:
                u = self.database[i]
                if getattr(u, "update", None) is not None:
                    if u.update(True):
                        saved.append(i)
        except:
            print(traceback.format_exc())
        # if saved:
        #     print("Autosaved " + str(saved) + ".")

    zwCallback = zwencode("callback")

    # Operates on reactions on special messages, calling the _callback_ methods of commands when necessary.
    async def reactCallback(self, message, reaction, user):
        if message.author.id == client.user.id:
            if self.closed:
                return
            u_perm = self.getPerms(user.id, message.guild)
            if u_perm <= -inf:
                return
            msg = message.content
            if not msg and message.embeds:
                msg = message.embeds[0].description
            if msg[:3] != "```" or len(msg) <= 3:
                msg = None
                if message.embeds:
                    s = message.embeds[0].footer.text
                    if isZeroEnc(s):
                        msg = s
                if not msg:
                    return
            else:
                msg = msg[3:]
                while msg.startswith("\n"):
                    msg = msg[1:]
                check = "callback-"
                try:
                    msg = msg[:msg.index("\n")]
                except ValueError:
                    pass
                if not msg.startswith(check):
                    return
            while len(self.proc_call) > 65536:
                self.proc_call.pop(next(iter(self.proc_call)))
            while utc() - self.proc_call.get(message.id, 0) < 30:
                # Ignore if more than 2 reactions already queued for target message
                if self.proc_call.get(message.id, 0) - utc() > 1:
                    return
                await asyncio.sleep(0.2)
            if reaction is not None:
                reacode = str(reaction).encode("utf-8")
            else:
                reacode = None
            msg = message.content
            if not msg and message.embeds:
                msg = message.embeds[0].description
            if msg[:3] != "```" or len(msg) <= 3:
                msg = None
                if message.embeds:
                    s = message.embeds[0].footer.text
                    if isZeroEnc(s):
                        msg = s
                if not msg:
                    return
                # Experimental zero-width invisible character encoded message (unused)
                try:
                    msg = msg[msg.index(self.zwCallback) + len(self.zwCallback):]
                except ValueError:
                    return
                msg = zwdecode(msg)
                args = msg.split("q")
            else:
                msg = msg[3:]
                while msg[0] == "\n":
                    msg = msg[1:]
                check = "callback-"
                msg = msg.split("\n")[0]
                msg = msg[len(check):]
                args = msg.split("-")
            catn, func, vals = args[:3]
            func = func.casefold()
            argv = "-".join(args[3:])
            catg = self.categories[catn]
            # Force a rate limit on the reaction processing for the message
            self.proc_call[message.id] = max(utc(), self.proc_call.get(message.id, 0) + 1)
            for f in catg:
                if f.__name__.casefold() == func:
                    try:
                        timeout = getattr(f, "_timeout_", 1) * self.timeout
                        if timeout >= inf:
                            timeout = None
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
            self.proc_call.pop(message.id, None)

    # Handles all updates to the bot. Manages the bot's status and activity on discord, and updates all databases.
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
                guilds = len(client.guilds)
                changed = guilds != self.guilds
                if changed or utc() > self.stat_timer:
                    # Status changes every 12-21 seconds
                    self.stat_timer = utc() + float(frand(5)) + 12
                    self.guilds = guilds
                    try:
                        u = await self.fetch_user(tuple(self.owners)[0])
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
                        # Status iterates through 3 possible choices
                        status = (discord.Status.online, discord.Status.dnd, discord.Status.idle)[self.status_iter]
                        try:
                            await client.change_presence(activity=activity, status=status)
                            # Member update events are not sent through for the current user, so manually send a _seen_ event
                            await seen(client.user, event="misc", raw="Changing their status")
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
            # Update databases
            for u in self.database.values():
                if utc() - u.used > u.rate_limit or force:
                    create_task(forceCoro(u))
                    create_task(self.verifyDelete(u))
            self.busy = False

    # Adds a webhook to the bot's user and webhook cache.
    def add_webhook(self, w):
        user = bot.ghostUser()
        user.id = w.id
        user.mention = "<@" + str(w.id) + ">"
        user.name = w.name
        user.display_name = w.name
        user.created_at = user.joined_at = w.created_at
        user.avatar = w.avatar
        user.avatar_url = w.avatar_url
        user.bot = True
        user.send = w.send
        user.dm_channel = w.channel
        user.webhook = w
        self.cache.users[w.id] = user
        self.limitCache("users")
        if w.token:
            webhooks = setDict(self.cw_cache, w.channel.id, cdict())
            webhooks[w.id] = w
        return user

    # Loads all webhooks in the target guild.
    async def load_webhooks(self, guild):
        try:
            webhooks = await guild.webhooks()
        except discord.Forbidden:
            webhooks = deque()
            futs = [channel.webhooks() for channel in guild.text_channels]
            for fut in futs:
                try:
                    temp = await fut
                except discord.Forbidden:
                    pass
                except discord.HTTPException:
                    print(traceback.format_exc())
                    await asyncio.sleep(5)
                    temp = await fut
                except:
                    print(traceback.format_exc())
                else:
                    webhooks.extend(temp)
        except discord.HTTPException:
            print(traceback.format_exc())
            await asyncio.sleep(10)
            for _ in loop(5):
                try:
                    webhooks = await guild.webhooks()
                    break
                except discord.HTTPException:
                    print(traceback.format_exc())
                    await asyncio.sleep(15)
        return deque(self.add_webhook(w) for w in webhooks)

    # Gets a valid webhook for the target channel, creating a new one when necessary.
    async def ensureWebhook(self, channel, force=False):
        while not self.ready:
            await asyncio.sleep(2)
        wlist = None
        if channel.id in self.cw_cache:
            if force:
                self.cw_cache.pop(channel.id)
            else:
                wlist = list(self.cw_cache[channel.id].values())
        if not wlist:
            w = await channel.create_webhook(name=bot.client.user.name, reason="Auto Webhook")
            self.add_webhook(w)
        else:
            w = random.choice(wlist)
        return w

    # Sends a list of embeds to the target channel, using a webhook when possible.
    async def sendEmbeds(self, channel, embeds):
        try:
            if not embeds:
                return
            guild = getattr(channel, "guild", None)
            # Determine whether to send embeds individually or as blocks of up to 10, based on whether it is possible to use webhooks
            single = False
            if guild is None or hasattr(guild, "ghost") or len(embeds) == 1:
                single = True
            else:
                try:
                    m = guild.get_member(client.user.id)
                except (AttributeError, LookupError):
                    m = None
                if m is None:
                    m = client.user
                    single = True
                else:
                    if not m.guild_permissions.manage_webhooks:
                        single = True
            if single:
                for emb in embeds:
                    create_task(channel.send(embed=emb))
                    await asyncio.sleep(0.5)
                return
            w = await self.bot.ensureWebhook(channel)
            try:
                await w.send(embeds=embeds, username=m.display_name, avatar_url=bestURL(m))
            except (discord.NotFound, discord.InvalidArgument, discord.Forbidden):
                w = await self.bot.ensureWebhook(channel, force=True)
                await w.send(embeds=embeds, username=m.display_name, avatar_url=bestURL(m))
            await seen(client.user, event="message", count=len(embeds), raw="Sending a message")
        except Exception as ex:
            print(traceback.format_exc())
            await sendReact(channel, "```py\n" + repr(ex) + "```", reacts="❎")
    
    async def sendEmbedsTo(self, s_id, embs):
        sendable = await self.fetch_sendable(s_id)
        await self.sendEmbeds(sendable, embs)

    # Adds embeds to the embed sender, waiting for the next update event.
    def embedSender(self, channel, embeds=None, embed=None):
        if embeds is not None and not issubclass(type(embeds), collections.abc.Sequence):
            embeds = (embeds,)
        if embed is not None:
            if embeds is not None:
                embeds += (embed,)
            else:
                embeds = (embed,)
        elif not embeds:
            return
        for e in embeds:
            if len(e) > 6000:
                print(e.to_dict())
                raise OverflowError
        try:
            c_id = int(channel)
        except (TypeError, ValueError):
            c_id = channel.id
        user = self.cache.users.get(c_id)
        if user is not None:
            create_task(self.sendEmbeds(user, embeds))
        else:
            embs = setDict(self.embedSenders, c_id, [])
            embs.extend(embeds)

    # Updates all embed senders.
    async def updateEmbeds(self):
        if not self.ready:
            return
        sent = False
        for s_id in tuple(self.embedSenders):
            embeds = self.embedSenders[s_id]
            embs = deque()
            for emb in embeds:
                # Send embeds in groups of up to 10, up to 6000 characters
                if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
                    break
                embs.append(emb)
            # Left over embeds are placed back in embed sender
            self.embedSenders[s_id] = embeds = embeds[len(embs):]
            if not embeds:
                self.embedSenders.pop(s_id)
            create_task(self.sendEmbedsTo(s_id, embs))
            sent = True
        return sent

    # For compatibility with guild objects, takes a user and DM channel.
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
                self.trigger_typing = lambda *args: channel.trigger_typing()
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

    # Represents a deleted/not found user.
    class ghostUser(discord.abc.Snowflake):
        
        def __init__(self):
            self.id = 0
            self.name = "[USER DATA NOT FOUND]"
            self.discriminator = "0000"
            self.avatar = ""
            self.avatar_url = ""
            self.bot = False
            self.display_name = ""

        __repr__ = lambda self: "<Ghost User id=" + str(self.id) + " name='" + str(self.name) + "' discriminator='" + str(self.discriminator) + "' bot=False>"
        __str__ = lambda self: str(self.name) + "#" + str(self.discriminator)
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

    # Represents a deleted/not found message.
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
        jump_url = "https://discord.com/channels/-1/-1/-1"
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


# Creates and starts a coroutine for typing in a channel.
typing = lambda self: create_task(self.trigger_typing())


# Queries for searching members
# Order of priority:
"""
ID (Full literal match)
Username + Discriminator (Full literal match)
Username (Full case-insensitive match)
Nickname (Full case-insensitive match)
Username + Discriminator (Full alphanumeric match)
Nickname (Full alphanumeric match)
Username + Discriminator (Starting literal match)
Username (Starting case-insensitive match)
Nickname (Starting case-insensitive match)
Username + Discriminator (Starting alphanumeric match)
Nickname (Starting alphanumeric match)
Username + Discriminator (Substring literal match)
Username (Substring case-insensitive match)
Nickname (Substring case-insensitive match)
Username + Discriminator (Substring alphanumeric match)
Nickname (Substring alphanumeric match)
"""
# Results are automatically sorted by match length, randomized if a tie occurs.

def userQuery1(x):
    yield x
    
def userIter1(x):
    yield str(x)

def userQuery2(x):
    yield x
    yield reconstitute(x).casefold()

def userIter2(x):
    yield str(x)
    yield reconstitute(x.name).casefold()
    yield reconstitute(x.display_name).casefold()

def userQuery3(x):
    yield to_alphanumeric(x).replace(" ", "").casefold()

def userIter3(x):
    yield to_alphanumeric(x).replace(" ", "").casefold()
    yield to_alphanumeric(x.display_name).replace(" ", "").casefold()


# Processes a message, runs all necessary commands and bot events. May be called from another source.
async def processMessage(message, msg, edit=True, orig=None, cb_argv=None, loop=False):
    if bot.closed:
        return
    cpy = msg
    # Strip quote from message.
    if msg[:2] == "> ":
        msg = msg[2:]
    # Strip spoiler from message.
    elif msg[:2] == "||" and msg[-2:] == "||":
        msg = msg[2:-2]
    # Strip code boxes from message.
    msg = msg.replace("`", "").strip()
    # Get user, channel, guild that the message belongs to
    user = message.author
    guild = message.guild
    u_id = user.id
    channel = message.channel
    c_id = channel.id
    if guild:
        g_id = guild.id
    else:
        g_id = 0
    # Get list of enabled commands for the channel.
    if g_id:
        try:
            enabled = bot.data.enabled[c_id]
        except KeyError:
            enabled = ["main", "string", "admin"]
    else:
        enabled = list(bot.categories)
    u_perm = bot.getPerms(u_id, guild)
    admin = not inf > u_perm
    # Gets prefix for current guild.
    if u_id == client.user.id:
        prefix = bot.prefix
    else:
        prefix = bot.getPrefix(guild)
    op = False
    comm = msg
    # Mentioning the bot serves as an alias for the prefix.
    for check in (*bot.mention, prefix):
        if comm.startswith(check):
            comm = comm[len(check):].strip()
            op = True
    # Respond to blacklisted users attempting to use a command, or when mentioned without a command.
    if (u_perm <= -inf and op) or msg in bot.mention:
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
                "Ignoring command from blacklisted user "
                + str(user) + " (" + str(u_id) + "): "
                + limStr(message.content, 256)
            )
            create_task(sendReact(
                channel,
                "Sorry, you are currently not permitted to request my services.",
                reacts="❎",
            ))
        return
    delay = 0
    run = False
    if op:
        # Special case: the ? alias for the ~help command, since ? is an argument flag indicator and will otherwise be parsed as one.
        if len(comm) and comm[0] == "?":
            check = comm[0]
            i = 1
        else:
            # Parse message to find command.
            i = len(comm)
            for end in " ?-+":
                if end in comm:
                    i2 = comm.index(end)
                    if i2 < i:
                        i = i2
            check = reconstitute(comm[:i]).casefold().replace("*", "").replace("_", "").replace("||", "")
        # Hash table lookup for target command: O(1) average time complexity.
        if check in bot.commands:
            # Multiple commands may have the same alias, run all of them
            for command in bot.commands[check]:
                # Make sure command is enabled, administrators bypass this
                if command.catg in enabled or admin:
                    alias = command.__name__
                    for a in command.alias:
                        if a.casefold() == check:
                            alias = a
                    alias = alias.casefold()
                    # argv is the raw parsed argument data
                    argv = comm[i:]
                    run = True
                    print(str(getattr(guild, "id", 0)) + ": " + str(user) + " (" + str(u_id) + ") issued command " + msg)
                    req = command.min_level
                    fut = None
                    try:
                        # Make sure server-only commands can only be run in servers.
                        if guild is None:
                            if getattr(command, "server_only", False):
                                raise ReferenceError("This command is only available in servers.")
                        # Make sure target has permission to use the target command, rate limit the command if necessary.
                        if u_perm is not nan:
                            if not u_perm >= req:
                                raise command.permError(u_perm, req, "for command " + alias)
                            x = command.rate_limit
                            if x:
                                if issubclass(type(x), collections.abc.Sequence):
                                    x = x[not bot.isTrusted(getattr(guild, "id", 0))]
                                delay += x
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
                                incDict(flags, h=1)
                        if argv:
                            # Commands by default always parse unicode fonts as regular text unless otherwise specified.
                            if not hasattr(command, "no_parse"):
                                argv = reconstitute(argv)
                            argv = argv.strip()
                            # Parse command flags (this is a bit of a mess)
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
                        # args is a list of arguments parsed from argv, using shlex syntax when possible.
                        if not argv:
                            args = []
                        else:
                            argv2 = singleSpace(argv.replace("\n", " ").replace("\r", "").replace("\t", " "))
                            try:
                                args = shlex.split(argv2)
                            except ValueError:
                                args = argv2.split()
                        # Assign "guild" as an object that mimics the discord.py guild if there is none
                        if guild is None:
                            guild = bot.userGuild(
                                user=user,
                                channel=channel,
                            )
                            channel = guild.channel
                        # Automatically start typing if the command is time consuming
                        tc = getattr(command, "time_consuming", False)
                        if not loop and tc:
                            fut = create_task(channel.trigger_typing())
                        # Send bot event: user has executed command
                        await bot.event("_command_", user=user, command=command)
                        # Get maximum time allowed for command to process
                        timeout = getattr(command, "_timeout_", 1) * bot.timeout
                        if timeout >= inf:
                            timeout = None
                        # Create a future to run the command
                        future = create_task(asyncio.wait_for(forceCoro(
                            command,                # command is a callable object, may be async or not
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
                        ), timeout=timeout))
                        # Add a callback to typing in the channel if the command takes too long
                        if fut is None and not hasattr(command, "typing"):
                            create_task(delayed_callback(future, 2, typing, channel))
                        response = await future
                        # Process response to command if there is one
                        if response is not None:
                            if fut is not None:
                                await fut
                            # Raise exceptions returned by the command
                            if issubclass(type(response), Exception):
                                raise response
                            elif bool(response) is not False:
                                # If 2-tuple returned, send as message-react pair
                                if type(response) is tuple and len(response) == 2:
                                    response, react = response
                                    if react == 1:
                                        react = "❎"
                                else:
                                    react = False
                                sent = None
                                # Process list as a sequence of messages to send
                                if type(response) is list:
                                    for r in response:
                                        create_task(channel.send(r))
                                        await asyncio.sleep(0.5)
                                # Process dict as kwargs for a message send
                                elif issubclass(type(response), collections.abc.Mapping):
                                    if "file" in response:
                                        sent = await sendFile(channel, response.get("content", ""), **response)
                                    else:
                                        sent = await channel.send(**response)
                                else:
                                    if type(response) not in (str, bytes, bytearray):
                                        response = str(response)
                                    # Process everything else as a string
                                    if type(response) is str and len(response) <= 2000:
                                        sent = await channel.send(response)
                                    else:
                                        # Send a file if the message is too long
                                        if type(response) is not bytes:
                                            response = bytes(str(response), "utf-8")
                                            filemsg = "Response too long for message."
                                        else:
                                            filemsg = "Response data:"
                                        if len(response) <= guild.filesize_limit:
                                            b = io.BytesIO(response)
                                            f = discord.File(b, filename="message.txt")
                                            sent = await sendFile(channel, filemsg, f)
                                        else:
                                            raise OverflowError("Response too long for file upload.")
                                # Add targeted react if there is one
                                if react and sent:
                                    await sent.add_reaction(react)
                    # Represents any timeout error that occurs
                    except (T0, T1, T2):
                        if fut is not None:
                            await fut
                        print(msg)
                        raise TimeoutError("Request timed out.")
                    # Represents all other errors
                    except Exception as ex:
                        if fut is not None:
                            await fut
                        errmsg = limStr("```py\nError: " + repr(ex).replace("`", "") + "\n```", 2000)
                        print(traceback.format_exc())
                        create_task(sendReact(
                            channel,
                            errmsg,
                            reacts="❎",
                        ))
    # If message was not processed as a command, send a _nocommand_ event with the parsed message data.
    if not run and u_id != client.user.id:
        temp = to_alphanumeric(cpy).casefold()
        await bot.event("_nocommand_", text=temp, edit=edit, orig=orig, message=message, perm=u_perm)
    # Return the delay before the message can be called again. This is calculated by the rate limit of the command.
    return delay


# Heartbeat loop: Repeatedly deletes a file to inform the watchdog process that the bot's event loop is still running.
async def heartbeatLoop():
    print("Heartbeat Loop initiated.")
    try:
        while True:
            try:
                bot.client
            except NameError:
                sys.exit()
            d = await create_future(os.path.exists, bot.heartbeat, priority=True)
            if d:
                try:
                    await create_future(os.remove, bot.heartbeat, priority=True)
                except:
                    print(traceback.format_exc())
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        sys.exit(1)   

# The fast update loop that runs 24 times per second. Used for events where timing is important.
async def fastLoop():
    freq = 24
    sent = 0
    while True:
        try:
            sent = await bot.updateEmbeds()
        except:
            print(traceback.format_exc())
        x = freq if sent else 1
        for i in range(x):
            create_task(bot.event("_call_"))
            await asyncio.sleep(1 / freq)

# The lazy update loop that runs once every 2-4 seconds. Calls the bot database autosave event once every ~60 seconds.
async def slowLoop():
    autosave = 0
    while True:
        try:
            if utc() - autosave > 60:
                autosave = utc()
                create_future_ex(bot.update)
                create_future_ex(bot.updateClient, priority=True)
            while bot.blocked > 0:
                print("Update event blocked.")
                bot.blocked -= 1
                await asyncio.sleep(1)
            await bot.handleUpdate()
            await asyncio.sleep(frand(2) + 2)
        except:
            print(traceback.format_exc())


# The event called when the bot starts up.
@client.event
async def on_ready():
    bot.mention = (
        "<@" + str(client.user.id) + ">",
        "<@!" + str(client.user.id) + ">",
    )
    print("Successfully connected as " + str(client.user))
    try:
        futs = deque()
        futs.append(create_task(bot.getState()))
        print("Servers: ")
        for guild in client.guilds:
            if guild.unavailable:
                print("> " + str(guild.id) + " is not available.")
            else:
                print("> " + guild.name)
        await bot.handleUpdate()
        futs.append(create_future(bot.updateClient, priority=True))
        futs.append(create_future(bot.cacheFromGuilds, priority=True))
        create_task(bot.getIP())
        if not bot.started:
            bot.started = True
            # Wait until all modules have been loaded successfully, then shut down corresponding executor
            while bot.modload:
                await create_future(bot.modload.popleft().result, priority=True)
            # Assign all bot database events to their corresponding keys.
            for u in bot.database.values():
                for f in dir(u):
                    if f.startswith("_") and f[-1] == "_" and f[1] != "_":
                        func = getattr(u, f, None)
                        if callable(func):
                            bot.events.append(f, func)
            print(bot.events)
            for fut in futs:
                await fut
            await bot.fetch_user(bot.deleted_user)
            # Set bot avatar if none has been set.
            if not os.path.exists("misc/init.tmp"):
                print("Setting bot avatar...")
                f = await create_future(open, "misc/avatar.png", "rb", priority=True)
                b = await create_future(f.read, priority=True)
                create_future_ex(f.close)
                await client.user.edit(avatar=b)
                await seen(client.user, event="misc", raw="Editing their profile")
                f = await create_future(open, "misc/init.tmp", "wb", priority=True)
                create_future_ex(f.close)
            create_task(slowLoop())
            create_task(fastLoop())
            print("Update loops initiated.")
            # Load all webhooks from cached guilds.
            futs = [create_task(bot.load_webhooks(guild)) for guild in bot.cache.guilds.values()]
            print("Ready.")
            # Send ready event to all databases.
            await bot.event("_ready_", bot=bot)
            for fut in futs:
                await fut
            bot.ready = True
            print("Initialization complete.")
        else:
            for fut in futs:
                await fut
            print("Reinitialized.")
    except:
        print(traceback.format_exc())


# Server join message
@client.event
async def on_guild_join(guild):
    create_task(bot.load_webhooks(guild))
    print("New server: " + str(guild))
    g = await bot.fetch_guild(guild.id)
    m = guild.get_member(client.user.id)
    channel = await bot.get_first_sendable(g, m)
    emb = discord.Embed(colour=discord.Colour(8364031))
    url = strURL(client.user.avatar_url)
    emb.set_author(name=client.user.name, url=url, icon_url=url)
    emb.description = (
        "Hi there! I'm " + client.user.name
        + ", a multipurpose discord bot created by <@"
        + "201548633244565504" + ">. Thanks for adding me"
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
            + "That's completely understandable if intentional, but please note that without the required permissions, some features may not function well, or not at all."
        ))
    await channel.send(embed=emb)


# User seen event
seen = lambda user, delay=0, event=None, **kwargs: create_task(bot.event("_seen_", user=user, delay=delay, event=event, **kwargs))


# Deletes own messages if any of the "X" emojis are reacted by a user with delete message permission level, or if the message originally contained the corresponding reaction from the bot.
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


# Reaction add event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
@client.event
async def on_raw_reaction_add(payload):
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        user = await bot.fetch_user(payload.user_id)
        message = await bot.fetch_message(payload.message_id, channel=channel)
    except discord.NotFound:
        return
    await seen(user, event="reaction", raw="Adding a reaction")
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        await bot.reactCallback(message, reaction, user)
        create_task(checkDelete(message, reaction, user))


# Reaction remove event: uses raw payloads rather than discord.py message cache. calls _seen_ bot database event.
@client.event
async def on_raw_reaction_remove(payload):
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        user = await bot.fetch_user(payload.user_id)
        message = await bot.fetch_message(payload.message_id, channel=channel)
    except discord.NotFound:
        return
    await seen(user, event="reaction", raw="Removing a reaction")
    if user.id != client.user.id:
        reaction = str(payload.emoji)
        await bot.reactCallback(message, reaction, user)
        create_task(checkDelete(message, reaction, user))


# Voice state update event: automatically unmutes self if server muted, calls _seen_ bot database event.
@client.event
async def on_voice_state_update(member, before, after):
    if member.id == client.user.id:
        after = member.voice
        if after is not None:
            if after.mute or after.deaf:
                # print("Unmuted self in " + member.guild.name)
                await member.edit(mute=False, deafen=False)
            await bot.handleUpdate()
    # Check for users with a voice state.
    if after is not None and not after.afk:
        if before is None:
            await seen(member, event="misc", raw="Joining a voice channel")
        elif (before.self_mute, before.self_deaf, before.self_stream, before.self_video) != (after.self_mute, after.self_deaf, after.self_stream, after.self_video):
            await seen(member, event="misc", raw="Updating their voice settings")


# Handles a new sent message, calls processMessage and sends an error if an exception occurs.
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


# Typing event: calls _typing_ and _seen_ bot database events.
@client.event
async def on_typing(channel, user, when):
    await bot.event("_typing_", channel=channel, user=user)
    await seen(user, delay=10, event="typing", raw="Typing")


# Message send event: processes new message. calls _send_ and _seen_ bot database events.
@client.event
async def on_message(message):
    bot.cacheMessage(message)
    guild = message.guild
    if guild:
        create_task(bot.event("_send_", message=message))
    await seen(message.author, event="message", raw="Sending a message")
    await bot.reactCallback(message, None, message.author)
    await handleMessage(message, False)


# User update event: calls _user_update_ and _seen_ bot database events.
@client.event
async def on_user_update(before, after):
    await bot.event("_user_update_", before=before, after=after)
    await seen(after, event="misc", raw="Editing their profile")


# Member update event: calls _member_update_ and _seen_ bot database events.
@client.event
async def on_member_update(before, after):
    await bot.event("_member_update_", before=before, after=after)
    if str(before.status) != str(after.status) or str(before.activity) != str(after.activity):
        # A little bit of a trick to make sure this part is only called once per user event.
        # This is necessary because on_member_update is called once for every member object.
        # By fetching the first instance of a matching member object,
        # this ensures the event will not be called multiple times if the user shares multiple guilds with the bot.
        try:
            member = await bot.fetch_member(after.id, find_others=True)
        except LookupError:
            member = None
        if member is None or member.guild == after.guild:
            await seen(after, event="misc", raw="Changing their status")


# Member join event: calls _join_ and _seen_ bot database events.
@client.event
async def on_member_join(member):
    await bot.event("_join_", user=member, guild=member.guild)
    await seen(member, event="misc", raw="Joining a server")


# Member leave event: calls _leave_ bot database event.
@client.event
async def on_member_remove(member):
    await bot.event("_leave_", user=member, guild=member.guild)


# Message delete event: uses raw payloads rather than discord.py message cache. calls _delete_ bot database event.
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
            # If message was not in cache, create a ghost message object to represent old message.
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
        await bot.event("_delete_", message=message)
    bot.deleteMessage(message)


# Message bulk delete event: uses raw payloads rather than discord.py message cache. calls _bulk_delete_ and _delete_ bot database events.
@client.event
async def on_raw_bulk_message_delete(payload):
    try:
        messages = payload.cached_messages
        if messages is None or len(messages) < len(payload.message_ids):
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
                # If message was not in cache, create a ghost message object to represent old message.
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
    messages = sorted(messages, key=lambda m: m.id)
    await bot.event("_bulk_delete_", messages=messages)
    for message in messages:
        guild = getattr(message, "guild", None)
        if guild:
            await bot.event("_delete_", message=message, bulk=True)
        bot.deleteMessage(message)


# Channel create event: calls _channel_create_ bot database event.
@client.event
async def on_guild_channel_create(channel):
    bot.cache.channels[channel.id] = channel
    guild = channel.guild
    if guild:
        await bot.event("_channel_create_", channel=channel, guild=guild)


# Channel delete event: calls _channel_delete_ bot database event.
@client.event
async def on_guild_channel_delete(channel):
    print(channel, "was deleted from", channel.guild)
    guild = channel.guild
    if guild:
        await bot.event("_channel_delete_", channel=channel, guild=guild)


# Webhook update event: updates the bot's webhook cache if there are new webhooks.
@client.event
async def on_webhooks_update(channel):
    webhooks = await channel.webhooks()
    for w in tuple(bot.cw_cache.get(channel.id, {}).values()):
        if w not in webhooks:
            bot.cw_cache[channel.id].pop(w.id)
            bot.cache.users.pop(w.id)
    for w in webhooks:
        bot.add_webhook(w)


# User ban event: calls _ban_ bot database event.
@client.event
async def on_member_ban(guild, user):
    print(user, "was banned from", guild)
    if guild:
        await bot.event("_ban_", user=user, guild=guild)


# Guild destroy event: Remove guild from bot cache.
@client.event
async def on_guild_remove(guild):
    bot.cache.guilds.pop(guild.id, None)
    print(guild, "removed.")


# Message edit event: processes edited message, uses raw payloads rather than discord.py message cache. calls _edit_ and _seen_ bot database events.
@client.event
async def on_raw_message_edit(payload):
    data = payload.data
    m_id = int(data["id"])
    raw = False
    if payload.cached_message:
        before = payload.cached_message
        after = await bot.fetch_message(m_id, payload.message_id)
    else:
        try:
            before = messages[m_id]
        except LookupError:
            # If message was not in cache, create a ghost message object to represent old message.
            c_id = data.get("channel_id")
            if not c_id:
                return
            before = bot.ghostMessage()
            before.channel = channel = await bot.fetch_channel(c_id)
            before.guild = guild = getattr(channel, "guild", None)
            before.id = payload.message_id
            before.created_at = snowflake_time(before.id)
            try:
                u_id = data["author"]["id"]
            except KeyError:
                u_id = None
                before.author = None
            else:
                if guild is not None:
                    user = guild.get_member(u_id)
                else:
                    user = None
                if not user:
                    user = await bot.fetch_user(u_id)
                before.author = user
            try:
                after = await channel.fetch_message(before.id)
            except LookupError:
                after = copy.copy(before)
                after._update(data)
            else:
                before.author = after.author
            raw = True
        else:
            after = copy.copy(before)
            after._update(data)
    bot.cacheMessage(after)
    if raw or before.content != after.content:
        await handleMessage(after)
        if getattr(after, "guild", None):
            create_task(bot.event("_edit_", before=before, after=after))
        await seen(after.author, event="message", raw="Editing a message")


# If this is the module being run and not imported, create a new Bot instance and run it.
if __name__ == "__main__":
    bot = Bot()
    bot.run()