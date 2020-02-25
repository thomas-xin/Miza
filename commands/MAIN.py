import asyncio, os, sys
from smath import *


default_commands = ["string", "admin"]


class help:
    is_command = True

    def __init__(self):
        self.name = ["?"]
        self.min_level = -inf
        self.description = "Shows a list of usable commands, or gives a detailed description of a command."
        self.usage = "<command{all}> <category{all}> <verbose(?v)>"

    async def __call__(self, args, user, channel, guild, flags, **void):
        _vars = self._vars
        enabled = _vars.data["enabled"]
        g_id = guild.id
        prefix = _vars.getPrefix(g_id)
        enabled = enabled.get(channel.id, list(default_commands))
        categories = _vars.categories
        commands = hlist()
        for catg in categories:
            if catg in enabled or catg == "main":
                commands.extend(categories[catg])
        c_name = getattr(channel, "name", "DM")
        u_perm = _vars.getPerms(user, guild)
        verb = "v" in flags
        argv = " ".join(args).lower().replace(prefix, "")
        show = []
        for a in args:
            if (a in categories and a in enabled) or a == "main":
                show.append(
                    "\nCommands for **" + user.name
                    + "** in **" + channel.name
                    + "** in category **" + a
                    + "**:\n"
                )
                for com in categories[a]:
                    name = com.__name__
                    min_level = com.min_level
                    description = com.description
                    usage = com.usage
                    if min_level > u_perm or (u_perm is not nan and min_level is nan):
                        continue
                    if c_name == "DM" and getattr(com, "server_only", False):
                        continue
                    newstr = (
                        "```xml\n" + prefix + name
                        + "\nAliases: " + str(com.name)
                        + "\nEffect: " + description
                        + "\nUsage: ~" + name + " " + usage
                        + "\nRequired permission level: " + uniStr(min_level)
                        + "```"
                    )
                    show.append(newstr)
        if not show:
            for c in categories:
                catg = categories[c]
                if not (c in enabled or c == "main"):
                    continue
                for com in catg:
                    name = com.__name__
                    min_level = com.min_level
                    description = com.description
                    usage = com.usage
                    if min_level > u_perm or (u_perm is not nan and min_level is nan):
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
                            + "\nUsage: ~" + name + " " + usage
                            + "\nRequired permission level: " + uniStr(min_level)
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
                if min_level > u_perm or (u_perm is not nan and min_level is nan):
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
                            + "\nUsage: ~" + name + " " + usage
                        )
            return (
                "Commands for **" + user.name + "** in **" + c_name
                + "**:\n```xml\n" + "\n".join(show) + "```", 1
            )
        return "\n".join(show), 1


class perms:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["changePerms", "perm", "changePerm"]
        self.min_level = -inf
        self.description = "Shows or changes a user's permission level."
        self.usage = "<0:user{self}> <1:level{curr}> <hide(?h)>"

    async def __call__(self, _vars, args, user, perm, guild, flags, **void):
        if len(args) < 2:
            if len(args) < 1:
                t_user = user
            else:
                if "everyone" in args[0] or "here" in args[0]:
                    return (
                        "Current user permissions for **" + guild.name + "**:\n```json\n"
                        + str(_vars.perms[guild.id]).replace("'", '"') + "```"
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
                    if not "c" in flags:
                        response = uniStr(
                            "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                            + "REPEAT COMMAND WITH \"?C\" FLAG TO CONFIRM."
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


class enableCommand:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["ec", "enable"]
        self.min_level = 0
        self.description = "Shows, enables, or disables a command category in the current channel."
        self.usage = "<command{all}> <enable(?e)> <disable(?d)> <hide(?h)>"

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
        print(catg)
        if not catg:
            if "e" in flags:
                categories = list(_vars.categories)
                categories.remove("main")
                enabled[channel.id] = categories
                update()
                if "h" in flags:
                    return
                return (
                    "```css\nEnabled all command categories in "
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
                "Currently enabled command categories in **" + channel.name
                + "**:\n```css\n"
                + str(["main"] + enabled.get(channel.id, default_commands)) + "```"
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


class restart:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["shutdown"]
        self.min_level = inf
        self.description = "Restarts or shuts down the bot."
        self.usage = ""

    async def __call__(self, client, channel, user, guild, name, _vars, perm, **void):
        if name == "shutdown":
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
                f = open(_vars.heartbeat, "wb")
                f.close()
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        try:
            await client.close()
        except:
            del client
        del _vars
        if perm is nan:
            for i in range(8):
                try:
                    if "log.txt" in os.listdir():
                        os.remove("log.txt")
                    break
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
        sys.exit()


class suspend:
    is_command = True

    def __init__(self):
        self.name = ["block", "blacklist"]
        self.min_level = nan
        self.description = "Prevents a user from accessing the bot's commands. Overrides ~perms."
        self.usage = "<0:user> <1:value[]>"

    async def __call__(self, _vars, user, guild, args, **void):
        update = self.data["suspended"].update
        susp = _vars.data["suspended"]
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


class prefix:
    is_command = True

    def __init__(self):
        self.name = ["changePrefix"]
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


class loop:
    is_command = True
    time_consuming = 3

    def __init__(self):
        self.name = ["for", "rep", "repeat", "while"]
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


class info:
    is_command = True

    def __init__(self):
        self.name = ["userInfo"]
        self.min_level = 0
        self.description = "Shows information about the target user."
        self.usage = "<user>"

    async def __call__(self, argv, guild, _vars, user, **void):
        if argv:
            u_id = _vars.verifyID(argv)
            try:
                u = guild.get_member(u_id)
                if u is None:
                    raise EOFError
            except:
                u = await guild.fetch_member(u_id)
        else:
            u = user
        name = u.name
        dname = u.display_name
        disc = u.discriminator
        url = u.avatar_url
        is_sys = u.system
        is_bot = u.bot
        is_self_owner = u.id == _vars.owner_id
        is_guild_owner = u.id == guild.owner_id
        joined = getattr(u, "joined_at", guild.created_at)
        created = u.created_at
        activity = "\n".join(str(i) for i in getattr(u, "activities", []))
        role = ", ".join(str(i) for i in getattr(u, "roles", []) if not i.is_default())
        coms = _vars.data["users"][u.id]["suspended"]
        emb = discord.Embed()
        emb.set_thumbnail(url=url)
        emb.set_author(name=name + "#" + disc)
        if any(activity, is_sys, is_bot, is_self_owner, is_guild_owner):
            d = "```css\n"
            d += activity + "\n" * bool(activity)
            d += "[Discord staff]\n" * is_sys
            d += "[Bot]\n" * is_bot
            d += "[My owner ❤️]\n" * is_self_owner
            d += "[Server owner]\n" * is_guild_owner
            d = d.strip("\n")
            d += "```"
        else:
            d = ""
        emb.description = d
        emb.add_field(name="ID", value=str(u.id), inline=0)
        emb.add_field(name="Creation date", value=str(created), inline=0)
        emb.add_field(name="Join date", value=str(joined), inline=0)
        emb.add_field(name="Commands used", value=str(coms), inline=0)
        if dname and dname != name:
            emb.add_field(name="Nickname", value=dname, inline=0)
        if role:
            emb.add_field(name="Roles", value=role, inline=0)
        print(embed.to_dict())
        return {
            "embed": embed,
        }


class state:
    is_command = True

    def __init__(self):
        self.name = ["status"]
        self.min_level = 0
        self.description = "Shows the bot's current program state."
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
            + ", Active shards: " + uniStr(1)
            
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
