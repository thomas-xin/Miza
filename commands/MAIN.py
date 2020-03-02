import discord, os, sys, datetime
from smath import *


default_commands = ["main", "string", "admin"]
standard_commands = default_commands + ["voice", "nsfw", "image", "game"]


class Help:
    is_command = True

    def __init__(self):
        self.name = ["?"]
        self.min_level = -inf
        self.description = "Shows a list of usable commands, or gives a detailed description of a command."
        self.usage = "<command{all}> <category{all}> <verbose(?v)>"

    async def __call__(self, args, user, channel, guild, flags, perm, **void):
        _vars = self._vars
        enabled = _vars.data["enabled"]
        g_id = guild.id
        prefix = _vars.getPrefix(g_id)
        enabled = enabled.get(channel.id, list(default_commands))
        c_name = getattr(channel, "name", "DM")
        admin = (not inf > perm, perm is nan)[c_name == "DM"]
        categories = _vars.categories
        commands = hlist()
        for catg in categories:
            if catg in enabled or admin:
                commands.extend(categories[catg])
        verb = "v" in flags
        argv = " ".join(args).lower().replace(prefix, "")
        show = []
        for a in args:
            if (a in categories and (a in enabled or admin)):
                show.append(
                    "\nCommands for **" + user.name
                    + "** in <#" + str(channel.id)
                    + "> in category **" + a
                    + "**:\n"
                )
                for com in categories[a]:
                    name = com.__name__
                    min_level = com.min_level
                    description = com.description
                    usage = com.usage
                    if min_level > perm or (perm is not nan and min_level is nan):
                        continue
                    if c_name == "DM" and getattr(com, "server_only", False):
                        continue
                    newstr = (
                        "```xml\n" + prefix + name
                        + "\nAliases: " + str(com.name)
                        + "\nEffect: " + description
                        + "\nUsage: " + prefix + name + " " + usage
                        + "\nLevel: " + uniStr(min_level)
                        + "```"
                    )
                    show.append(newstr)
        if not show:
            for c in categories:
                catg = categories[c]
                if not (c in enabled or admin):
                    continue
                for com in catg:
                    name = com.__name__
                    min_level = com.min_level
                    description = com.description
                    usage = com.usage
                    if min_level > perm or (perm is not nan and min_level is nan):
                        continue
                    if c_name == "DM" and getattr(com, "server_only", False):
                        continue
                    found = False
                    for n in com.name:
                        n = n.lower()
                        if n in args:
                            found = True
                    if found:
                        newstr = (
                            "```xml\n" + prefix + name
                            + "\nCategory: " + c
                            + "\nAliases: " + str(com.name)
                            + "\nEffect: " + description
                            + "\nUsage: " + prefix + name + " " + usage
                            + "\nLevel: " + uniStr(min_level)
                            + "```"
                        )
                        if (not len(show)) or len(show[-1]) < len(newstr):
                            show = [newstr]
        if not show:
            for com in commands:
                name = com.__name__
                min_level = com.min_level
                description = com.description
                usage = com.usage
                if min_level > perm or (perm is not nan and min_level is nan):
                    continue
                if c_name == "DM" and getattr(com, "server_only", False):
                        continue
                if description != "":
                    if not verb:
                        show.append(prefix + name + " " + usage)
                    else:
                        show.append(
                            "\n" + prefix + com.__name__
                            + "\nEffect: " + com.description
                            + "\nUsage: " + prefix + name + " " + usage
                        )
            return (
                "Commands for **" + user.name + "** in <#" + str(channel.id)
                + ">:\n```xml\n" + "\n".join(show) + "```", 1
            )
        return "\n".join(show), 1


class Perms:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["ChangePerms", "Perm", "ChangePerm", "Permissions"]
        self.min_level = -inf
        self.description = "Shows or changes a user's permission level."
        self.usage = "<0:user{self}> <1:level[]> <hide(?h)>"

    async def __call__(self, _vars, args, user, perm, guild, flags, **void):
        if len(args) < 2:
            if len(args) < 1:
                t_user = user
            else:
                if "@e" in args[0] or "everyone" in args[0] or "here" in args[0]:
                    return (
                        "Current user permissions for **" + guild.name + "**:\n```json\n"
                        + str(_vars.data["perms"].get(guild.id, {})).replace("'", '"') + "```"
                    )
                else:
                    t_user = await _vars.fetch_user(_vars.verifyID(args[0]))
            print(t_user)
            t_perm = _vars.getPerms(t_user.id, guild)
        else:
            c_perm = await _vars.evalMath(" ".join(args[1:]), guild.id)
            s_user = user
            s_perm = perm
            check = args[0].lower()
            if "everyone" in check or "here" in check:
                t_user = None
                t_perm = inf
                name = "everyone"
            else:
                t_user = await _vars.fetch_user(_vars.verifyID(args[0]))
                t_perm = _vars.getPerms(t_user.id, guild)
                name = t_user.name
            if t_perm is nan or c_perm is nan:
                m_perm = nan
            else:
                m_perm = max(t_perm, c_perm, 1) + 1
            if not s_perm <= m_perm and m_perm is not nan:
                if t_user is None:
                    if not "f" in flags:
                        response = uniStr(
                            "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                            + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                        )
                        return ("```asciidoc\n[" + response + "]```")
                    for u in guild.members:
                        _vars.setPerms(u.id, guild, c_perm)
                else:
                    _vars.setPerms(t_user.id, guild, c_perm)
                if "h" in flags:
                    return
                return (
                    "```css\nChanged permissions for "+ uniStr(name)
                    + " in " + uniStr(guild.name)
                    + " from " + uniStr(t_perm)
                    + " to " + uniStr(c_perm) + ".```"
                )
            else:
                raise PermissionError(
                    "Insufficient priviliges to change permissions for " + uniStr(name)
                    + " in " + uniStr(guild.name)
                    + " from " + uniStr(t_perm)
                    + " to " + uniStr(c_perm)
                    + ".\nRequired level: " + uniStr(m_perm)
                    + ", Current level: " + uniStr(s_perm) + "."
                )
        return (
            "```css\nCurrent permissions for " + uniStr(t_user.name)
            + " in " + uniStr(guild.name)
            + ": " + uniStr(t_perm) + ".```"
        )


class EnableCommand:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["EC", "Enable"]
        self.min_level = 0
        self.description = "Shows, enables, or disables a command category in the current channel."
        self.usage = "<command{all}> <enable(?e)> <disable(?d)> <list(?l)> <hide(?h)>"

    async def __call__(self, argv, flags, user, channel, perm, **void):
        update = self.data["enabled"].update
        _vars = self._vars
        enabled = _vars.data["enabled"]
        if "e" in flags or "d" in flags:
            req = 3
            if perm < req:
                raise PermissionError(
                    "Insufficient priviliges to change command list for "
                    + uniStr(channel.name)
                    + ".\nRequred level: " + uniStr(req)
                    + ", Current level: " + uniStr(perm) + "."
                )
        catg = argv.lower()
        if not catg:
            if "l" in flags:
                return (
                    "```css\nStandard command categories:\n"
                    + str(standard_commands) + "```"
                )
            if "e" in flags:
                categories = list(standard_commands) #list(_vars.categories)
                enabled[channel.id] = categories
                update()
                if "h" in flags:
                    return
                return (
                    "```css\nEnabled standard command categories in "
                    + uniStr(channel.name) + ".```"
                )
            if "d" in flags:
                enabled[channel.id] = []
                update()
                if "h" in flags:
                    return
                return (
                    "```css\nDisabled all command categories in "
                    + uniStr(channel.name) + ".```"
                )
            return (
                "Currently enabled command categories in <#" + str(channel.id)
                + ">:\n```css\n"
                + str(enabled.get(channel.id, default_commands)) + "```"
            )
        else:
            if not catg in _vars.categories:
                raise KeyError("Unknown command category " + uniStr(argv) + ".")
            else:
                enabled = enabled.setdefault(channel.id, {})
                if "e" in flags:
                    if catg in enabled:
                        raise OverflowError(
                            "Command category " + uniStr(catg)
                            + " is already enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.append(catg)
                    update()
                    if "h" in flags:
                        return
                    return (
                        "```css\nEnabled command category " + uniStr(catg)
                        + " in " + uniStr(channel.name) + ".```"
                    )
                if "d" in flags:
                    if catg not in enabled:
                        raise OverflowError(
                            "Command category " + uniStr(catg)
                            + " is not currently enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.remove(catg)
                    update()
                    if "h" in flags:
                        return
                    return (
                        "```css\nDisabled command category " + uniStr(catg)
                        + " in " + uniStr(channel.name) + ".```"
                    )
                return (
                    "```css\nCommand category " + uniStr(catg)
                    + " is currently" + uniStr(" not" * (catg not in enabled))
                    + " enabled in " + uniStr(channel.name) + ".```"
                )


class Restart:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Shutdown"]
        self.min_level = inf
        self.description = "Restarts or shuts down the bot."
        self.usage = ""

    async def __call__(self, client, channel, user, guild, name, _vars, perm, **void):
        if name.lower() == "shutdown":
            if perm is not nan:
                raise PermissionError("Insufficient priviliges to request shutdown.")
            f = open(_vars.shutdown, "wb")
            f.close()
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        if perm is nan or frand() > 0.75:
            for i in range(64):
                try:
                    if _vars.suspected in os.listdir():
                        os.remove(_vars.suspected)
                    break
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
        _vars.update()
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        for i in range(5):
            try:
                f = open(_vars.restart, "wb")
                f.close()
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        if perm is nan:
            for i in range(8):
                try:
                    if "log.txt" in os.listdir():
                        os.remove("log.txt")
                    break
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
        try:
            await client.close()
        except:
            del client
        del _vars
        sys.exit()


class Suspend:
    is_command = True

    def __init__(self):
        self.name = ["Block", "Blacklist"]
        self.min_level = nan
        self.description = "Prevents a user from accessing the bot's commands. Overrides <perms>."
        self.usage = "<0:user> <1:value[]>"

    async def __call__(self, _vars, user, guild, args, **void):
        update = self.data["users"].update
        susp = _vars.data["users"]
        if len(args) < 2:
            if len(args) >= 1:
                user = await _vars.fetch_user(_vars.verifyID(args[0]))
            return (
                "```css\nCurrent suspension status of " + uniStr(user.name) + ": "
                + uniStr(susp.get(user.id, None)) + ".```"
            )
        else:
            user = await _vars.fetch_user(_vars.verifyID(args[0]))
            change = await _vars.evalMath(args[1], guild.id)
            susp[user.id] = change
            update()
            return (
                "```css\nChanged suspension status of " + uniStr(user.name) + " to "
                + uniStr(change) + ".```"
            )


class Prefix:
    is_command = True

    def __init__(self):
        self.name = ["ChangePrefix"]
        self.min_level = 0
        self.description = "Shows or changes the prefix for commands for this server."
        self.usage = "<prefix[]>"

    async def __call__(self, argv, guild, perm, _vars, **void):
        pref = _vars.data["prefixes"]
        update = self.data["prefixes"].update
        if not argv:
            return (
                "```css\nCurrent command prefix for " + uniStr(guild.name)
                + ": " + _vars.getPrefix(guild) + "```"
            )
        req = inf
        if perm < req:
            raise PermissionError(
                "Insufficient priviliges to change command prefix for "
                + uniStr(channel.name)
                + ".\nRequred level: " + uniStr(req)
                + ", Current level: " + uniStr(perm) + "."
            )
        pref[guild.id] = argv.strip(" ")
        update()
        return (
            "```css\nSuccessfully changed command prefix for " + uniStr(guild.name)
            + " to " + argv + "```"
        )


class Loop:
    is_command = True
    time_consuming = 3

    def __init__(self):
        self.name = ["For", "Rep", "Repeat", "While"]
        self.min_level = 1
        self.description = "Loops a command."
        self.usage = "<0:iterations> <1:command> <hide(?h)>"

    async def __call__(self, args, argv, message, callback, _vars, flags, perm, guild, **void):
        num = await _vars.evalMath(args[0], guild.id)
        iters = round(num)
        scale = 3
        limit = perm * scale
        if iters > limit:
            raise PermissionError(
                "insufficient priviliges to execute loop of " + uniStr(iters)
                + " iterations. Required level: " + uniStr(ceil(iters / scale))
                + ", Current level: " + uniStr(perm) + "."
            )
        func = func2 = " ".join(args[1:])
        if flags:
            func += " ?" + "?".join(flags)
        if func:
            while func[0] == " ":
                func = func[1:]
        for n in self.name:
            if func.startswith("~" + n):
                if isValid(perm):
                    raise PermissionError("Insufficient priviliges to execute nested loop.")
        func2 = " ".join(func2.split(" ")[1:])
        for i in range(iters):
            loop = i < iters - 1
            asyncio.create_task(callback(
                message, func, cb_argv=func2, cb_flags=flags, loop=loop,
            ))
        if not "h" in flags:
            return "```css\nLooping [" + func + "] " + uniStr(iters) + " times...```"


class Info:
    is_command = True

    def __init__(self):
        self.name = ["UserInfo", "ServerInfo"]
        self.min_level = 0
        self.description = "Shows information about the target user or server."
        self.usage = "<user> <verbose(?v)>"

    async def getGuildData(self, g, flags={}):
        _vars = self._vars
        url = str(g.icon_url)
        name = g.name
        try:
            u = g.owner
        except AttributeError:
            u = None
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        if u is not None:
            d = "Owner: <@" + str(u.id) + ">"
        else:
            d = ""
        if g.description is not None:
            d += "```\n" + str(g.description) + "```"
        emb.description = d
        top = None
        try:
            g.region
            pcount = await _vars.updaters["counts"].getUserMessages(None, g)
        except AttributeError:
            pcount = 0
        try:
            if "v" in flags:
                pavg = await _vars.updaters["counts"].getUserAverage(None, g)
                users = deque()
                us = await _vars.updaters["counts"].getGuildMessages(g)
                if type(us) is str:
                    top = us
                else:
                    ul = sorted(
                        us,
                        key=lambda k: us[k],
                        reverse=True,
                    )
                    for i in range(min(32, flags.get("v", 0) * 5, len(us))):
                        u_id = ul[i]
                        users.append(
                            "<@" + str(u_id) + ">: "
                            + str(us[u_id])
                        )
                    top = "\n".join(users)
        except AttributeError:
            pass
        emb.add_field(name="Server ID", value=str(g.id), inline=0)
        emb.add_field(name="Creation time", value=str(g.created_at), inline=1)
        if "v" in flags:
            try:
                emb.add_field(name="Region", value=str(g.region), inline=1)
                emb.add_field(name="Nitro boosts", value=str(g.premium_subscription_count), inline=1)
            except AttributeError:
                pass
        emb.add_field(name="User count", value=str(g.member_count), inline=1)
        if pcount:
            emb.add_field(name="Post count", value=str(pcount), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(pavg), inline=1)
        if top is not None:
            emb.add_field(name="Top users", value=top, inline=0)
        print(emb.to_dict())
        return {
            "embed": emb,
        }

    async def __call__(self, argv, guild, _vars, client, user, flags, **void):
        member = True
        if argv:
            u_id = _vars.verifyID(argv)
            try:
                u = guild.get_member(u_id)
                if u is None:
                    raise LookupError("Unable to find user or server from ID.")
            except:
                try:
                    u = await guild.fetch_member(u_id)
                except:
                    try:
                        u = await _vars.fetch_user(u_id)
                        member = False
                    except:
                        try:
                            guild = await _vars.fetch_guild(u_id)
                        except discord.Forbidden:
                            raise
                        except:
                            try:
                                channel = await _vars.fetch_channel(u_id)
                            except discord.Forbidden:
                                raise
                            except:
                                raise LookupError("Unable to find user or server from ID.")
                            try:
                                guild = channel.guild
                            except AttributeError:
                                guild = None
                                u = channel.recipient
                        if guild is not None:
                            return await self.getGuildData(guild, flags)                        
        else:
            u = user
        name = u.name
        dname = u.display_name
        disc = u.discriminator
        url = u.avatar_url
        try:
            is_sys = u.system
        except AttributeError:
            is_sys = False
        is_bot = u.bot
        is_self = u.id == client.user.id
        is_self_owner = u.id == _vars.owner_id
        is_guild_owner = u.id == guild.owner_id
        if member:
            joined = getattr(u, "joined_at", guild.created_at)
        else:
            joined = None
        created = u.created_at
        activity = "\n".join(str(i) for i in getattr(u, "activities", []))
        role = ", ".join(str(i) for i in getattr(u, "roles", []) if not i.is_default())
        coms = msgs = avgs = 0
        pos = None
        if "v" in flags:
            try:
                coms = _vars.data["users"][u.id]["commands"]
            except LookupError:
                pass
            try:
                msgs = await _vars.updaters["counts"].getUserMessages(u, guild)
                avgs = await _vars.updaters["counts"].getUserAverage(u, guild)
                if joined and guild.owner.id != client.user.id:
                    us = await _vars.updaters["counts"].getGuildMessages(guild)
                    if type(us) is str:
                        pos = us
                    else:
                        ul = sorted(
                            us,
                            key=lambda k: us[k],
                            reverse=True,
                        )
                        try:
                            pos = ul.index(u.id)
                        except ValueError:
                            pos = len(ul)
                        pos += 1
            except LookupError:
                pass
        if is_self and _vars.website is not None:
            url2 = _vars.website
        else:
            url2 = url
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name + "#" + disc, icon_url=url, url=url2)
        d = "<@" + str(u.id) + ">"
        if activity:
            d += "```\n" + activity + "```"
        if any((is_sys, is_bot, is_self, is_self_owner, is_guild_owner)):
            d += "```css\n"
            d += "[Discord staff]\n" * is_sys
            d += "[Bot]\n" * is_bot
            d += "[Myself :3]\n" * is_self
            d += "[My owner ❤️]\n" * is_self_owner
            d += "[Server owner]\n" * (is_guild_owner and not hasattr(guild, "isDM"))
            d = d.strip("\n")
            d += "```"
        emb.description = d
        emb.add_field(name="User ID", value=str(u.id), inline=0)
        emb.add_field(name="Creation time", value=str(created), inline=1)
        if joined is not None:
            emb.add_field(name="Join time", value=str(joined), inline=1)
        if coms:
            emb.add_field(name="Commands used", value=str(coms), inline=1)
        if dname and dname != name:
            emb.add_field(name="Nickname", value=dname, inline=1)
        if msgs:
            emb.add_field(name="Post count", value=str(msgs), inline=1)
        if avgs:
            emb.add_field(name="Average post length", value=str(avgs), inline=1)
        if pos:
            emb.add_field(name="Server rank", value=str(pos), inline=1)
        if role:
            emb.add_field(name="Roles", value=role, inline=0)
        print(emb.to_dict())
        return {
            "embed": emb,
        }


class Status:
    is_command = True

    def __init__(self):
        self.name = ["State"]
        self.min_level = 0
        self.description = "Shows the bot's current internal program state."
        self.usage = ""

    async def __call__(self, flags, client, _vars, **void):
        active = _vars.getActive()
        latency = sec2Time(client.latency)
        size = _vars.codeSize
        stats = _vars.currState
        return (
            "```css"
            + "\nActive users: " + uniStr(len(client.users))
            + ", Active servers: " + uniStr(_vars.guilds)
            + ", Active shards: " + uniStr(len(client.latencies))
            
            + ".\nActive processes: " + uniStr(active[0])
            + ", Active threads: " + uniStr(active[1])
            + ", Active coroutines: " + uniStr(active[2])
            
            + ".\nPing latency: " + uniStr(latency)
            
            + ".\nConnected voice channels: " + uniStr(len(client.voice_clients))
            
            + ".\nCached files: " + uniStr(len(os.listdir("cache/")))
            
            + ".\nCode size: " + uniStr(size[0]) + " bytes"
            + ", " + uniStr(size[1]) + " lines"
            
            + ".\nCPU usage: " + uniStr(round(stats[0], 3)) + "%"
            + ", RAM usage: " + uniStr(round(stats[1] / 1048576, 3)) + " MB"
            + ".```"
        )


class Execute:
    is_command = True

    def __init__(self):
        self.name = ["Exec", "Eval"]
        self.min_level = nan
        self.description = (
            "Causes all messages in the current channel to be executed as python code on the bot."
            + " WARNING: DO NOT ALLOW UNTRUSTED USERS TO POST IN CHANNEL."
        )
        self.usage = "<enable(?e)> <disable(?d)>"

    async def __call__(self, _vars, flags, channel, **void):
        data = _vars.data["eval"]
        if "e" in flags:
            _vars.updaters["eval"].channel = channel
            return (
                "```css\nSuccessfully changed code channel to "
                + uniStr(channel.id) + ".```"
            )
        elif "d" in flags:
            _vars.updaters["eval"].channel = freeClass(id=None)
            return (
                "```css\nSuccessfully removed code channel.```"
            )
        return (
            "```css\ncode channel is currently set to "
            + uniStr(_vars.updaters["eval"].channel.id) + ".```"
        )


class updateEval:
    is_update = True
    name = "eval"

    def __init__(self):
        self.prev = hlist()
        self.channel = self.ch = freeClass(id=None)

    async def __call__(self):
        pass

    def verifyChannel(self, channel):
        if channel is None:
            return None
        try:
            if channel.guild is None:
                raise TypeError
        except:
            channel = self._vars.userGuild(channel.recipient, channel).channel
        return channel

    async def _nocommand_(self, message, **void):
        _vars = self._vars
        if message.author.id == _vars.client.user.id:
            return
        if message.guild is None:
            emb = discord.Embed()
            emb.add_field(
                name=str(message.author),
                value=_vars.strMessage(message),
            )
            await self.channel.send(embed=emb)
            return
        if message.channel.id == self.channel.id:
            output = None
            try:
                proc = message.content
                if proc[0] == "#":
                    proc = proc[1:]
                    if not proc:
                        self.prev.append(self.ch)
                        self.ch = freeClass(id=None)
                        await self.channel.send("Successfully cleared target channel.")
                        return
                    elif proc[0] == "#":
                        self.ch = self.verifyChannel(self.prev.popright())
                        await self.channel.send(
                            "Successfully changed target channel to "
                            + str(self.ch.id) + "."
                        )
                        return
                    self.prev.append(self.ch)
                    try:
                        channel = await _vars.fetch_channel(proc)
                        self.ch = self.verifyChannel(channel)
                    except:
                        user = await _vars.fetch_user(proc)
                        temp = await _vars.getDM(user)
                        self.ch = _vars.userGuild(user, temp).channel
                    hist = await self.ch.history(limit=25).flatten()
                    hist = hlist(hist)
                    emb = discord.Embed()
                    for m in reversed(hist):
                        emb.add_field(
                            name=str(m.author),
                            value=_vars.strMessage(m),
                            inline=0,
                        )
                    await self.channel.send(embed=emb)
                elif proc[0] == "&":
                    proc = proc[1:]
                    hist = await self.ch.history(limit=1).flatten()
                    message = hist[0]
                    await message.add_reaction(proc)
                else:
                    print(proc)
                    if proc.startswith(">"):
                        await self.ch.send(proc[1:])
                    else:
                        try:
                            output = await eval(proc, _vars._globals)
                            await self.channel.send(limStr("```py\n" + str(output) + "```", 2000))
                        except (SyntaxError, TypeError):
                            try:
                                output = eval(proc, _vars._globals)
                                await self.channel.send(limStr("```py\n" + str(output) + "```", 2000))
                            except:
                                try:
                                    exec(proc, _vars._globals)
                                    await self.channel.send("```py\nNone```")
                                except:
                                    await self.channel.send(limStr(
                                        "```py\n" + traceback.format_exc() + "```",
                                        2000,
                                    ))
            except:
                await self.channel.send(limStr(
                    "```py\n" + traceback.format_exc() + "```",
                    2000,
                ))
            if output is not None:
                _vars._globals["output"] = output
                _vars._globals["_"] = output


class updateMessageCount:
    is_update = True
    name = "counts"
    no_file = True

    def getMessageLength(self, message):
        return len(message.system_content) + sum(len(e) for e in message.embeds)

    async def getUserMessages(self, user, guild):
        if self.scanned == -1:
            c_id = self._vars.client.user.id
            if guild is None or hasattr(guild, "isDM"):
                channel = user.dm_channel
                if channel is None:
                    return 0
                messages = await channel.history(limit=None).flatten()
                count = sum(1 for m in messages if m.author.id != c_id)
                return count
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return d
                d = d["counts"]
                if user is None:
                    return sum(d.values())
                return d.get(user.id, 0)
            self.data[guild.id] = "Calculating..."
            asyncio.create_task(self.getUserMessageCount(guild))
        return "Calculating..."

    async def getUserAverage(self, user, guild):
        if self.scanned == -1:
            c_id = self._vars.client.user.id
            if guild is None or hasattr(guild, "isDM"):
                channel = user.dm_channel
                if channel is None:
                    return 0
                messages = await channel.history(limit=None).flatten()
                gen = tuple(m for m in messages if m.author.id != c_id)
                avg = sum(self.getMessageLength(m) for m in gen) / len(gen)
                return avg
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return d
                t = d["totals"]
                c = d["counts"]
                if user is None:
                    return sum(t.values()) / sum(c.values())
                return t.get(user.id, 0) / c.get(user.id, 1)
        return "Calculating..."            

    async def getGuildMessages(self, guild):
        if self.scanned == -1:
            c_id = self._vars.client.user.id
            if guild is None or hasattr(guild, "isDM"):
                channel = user.dm_channel
                if channel is None:
                    return 0
                messages = await channel.history(limit=None).flatten()
                return len(messages)
            if guild.id in self.data:
                try:
                    return self.data[guild.id]["counts"]
                except:
                    return self.data[guild.id]
            self.data[guild.id] = "Calculating..."
            asyncio.create_task(self.getUserMessageCount(guild))
        return "Calculating..."

    async def getUserMessageCount(self, guild):

        async def getChannelHistory(channel, returns):
            try:
                history = channel.history(
                    limit=None,
                )
                print(history)
                for i in range(8):
                    try:
                        messages = await history.flatten()
                        break
                    except discord.Forbidden:
                        raise
                    except:
                        if i >= 7:
                            raise
                        print(traceback.format_exc())
                    await asyncio.sleep(30 * (i ** 2 + 1))
                print(len(messages))
                returns[0] = messages
            except:
                print(traceback.format_exc())
                returns[0] = []

        year = datetime.timedelta(seconds=31556925.216)
        oneyear = datetime.datetime.utcnow() - guild.created_at < year
        if guild.member_count > 512 and not oneyear:
            self.data["guild.id"] = "ERROR: Server is too large to estimate post counts."
            return
        print(guild)
        data = {}
        avgs = {}
        histories = deque()
        i = 1
        for channel in reversed(guild.text_channels):
            returns = [None]
            histories.append(returns)
            if not i % 5:
                await asyncio.sleep(5 + random.random() * 10)
            asyncio.create_task(getChannelHistory(
                channel,
                histories[-1],
            ))
            i += 1
        while [None] in histories:
            await asyncio.sleep(2)
        print("Counting...")
        for messages in histories:
            i = 1
            for message in messages[0]:
                u = message.author.id
                length = self.getMessageLength(message)
                if u in data:
                    data[u] += 1
                    avgs[u] += length
                else:
                    data[u] = 1
                    avgs[u] = length
                if not i & 8191:
                    await asyncio.sleep(0.5)
                if random.randint(0, self._vars.cachelim) > len(self._vars.cache["messages"]):
                    self._vars.cache["messages"][message.id] = message
                i += 1
        self.data[guild.id] = {"counts": data, "totals": avgs}
        print(guild)
        print(self.data[guild.id])

    def __init__(self):
        self.scanned = False

    async def __call__(self):
        if self.scanned:
            return
        self.scanned = True
        year = datetime.timedelta(seconds=31556925.216)
        guilds = self._vars.client.guilds
        i = 1
        for guild in sorted(guilds, key=lambda g: g.member_count, reverse=True):
            oneyear = datetime.datetime.utcnow() - guild.created_at < year
            if guild.member_count < 512 or oneyear:
                self.data[guild.id] = "Calculating..."
                asyncio.create_task(self.getUserMessageCount(guild))
            if not i & 63:
                await asyncio.sleep(20)
            i += 1
        self.scanned = -1

    async def _send_(self, message, **void):
        if self.scanned == -1:
            user = message.author
            guild = message.guild
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return
                count = d["counts"].get(user.id, 0) + 1
                total = d["totals"].get(user.id, 0) + self.getMessageLength(message)
                d["totals"][user.id] = total
                d["counts"][user.id] = count
            else:
                asyncio.create_task(self.getUserMessageCount(guild))

    async def _edit_(self, before, after, **void):
        if hasattr(before, "ghost"):
            return
        if self.scanned == -1:
            user = after.author
            guild = after.guild
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return
                total = (
                    d["totals"].get(user.id, 0)
                    - self.getMessageLength(before)
                    + self.getMessageLength(after)
                )
                d["totals"][user.id] = total
            else:
                asyncio.create_task(self.getUserMessageCount(guild))

    async def _delete_(self, message, **void):
        if self.scanned == -1:
            user = message.author
            guild = message.guild
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return
                count = d["counts"].get(user.id, 0) - 1
                total = d["totals"].get(user.id, 0) - self.getMessageLength(message)
                d["totals"][user.id] = total
                d["counts"][user.id] = count
            else:
                asyncio.create_task(self.getUserMessageCount(guild))


class updatePrefix:
    is_update = True
    name = "prefixes"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateEnabled:
    is_update = True
    name = "enabled"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateUsers:
    is_update = True
    name = "users"
    suspected = "users.json"
    user = True

    def __init__(self):
        self.suspclear = inf
        try:
            self.lastsusp = None
            f = open(self.suspected, "r")
            susp = f.read()
            print(susp)
            f.close()
            os.remove(self.suspected)
            if susp:
                u_id = int(susp)
                udata = self.data[u_id]
                days = max(0, (udata["suspended"] - time.time()) / 86400)
                days **= 4
                days += 1.125
                udata["suspended"] = time.time() + days * 86400
                if days >= self._vars.min_suspend - 1:
                    self.lastsusp = susp
                self.update()
                self.update(True)
        except FileNotFoundError:
            pass

    async def _command_(self, user, command, **void):
        udata = self.data.setdefault(user.id, {"commands": 0, "suspended": 0})
        udata["commands"] += 1
        self.update()
        tc = getattr(command, "time_consuming", False)
        self.suspclear = time.time() + 10 + (tc * 2) ** 2
        f = open(self.suspected, "w")
        f.write(str(user.id))
        f.close()

    async def __call__(self, **void):
        if time.time() - self.suspclear:
            self.suspclear = inf
            try:
                if self.suspected in os.listdir():
                    os.remove(self.suspected)
            except:
                print(traceback.format_exc())
        _vars = self._vars
        if self.lastsusp is not None:
            u_susp = await _vars.fetch_user(self.lastsusp)
            self.lastsusp = None
            channel = await _vars.getDM(u_susp)
            secs = self.data.get(u_susp.id, 0) - time.time()
            msg = (
                "Apologies for the inconvenience, but your account has been "
                + "flagged as having attempted a denial-of-service attack.\n"
                + "This will expire in `" + sec2Time(secs) + "`.\n"
                + "If you believe this is an error, please notify <@"
                + str(_vars.owner_id) + "> as soon as possible."
            )
            print(
                u_susp.name + " may be attempting a DDOS attack. Expires in "
                + sec2Time(secs) + "."
            )
            await channel.send(msg)
