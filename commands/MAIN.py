import asyncio, os, sys
from smath import *


default_commands = ["string", "admin"]


class help:
    is_command = True

    def __init__(self):
        self.name = ["?"]
        self.min_level = -inf
        self.description = "Shows a list of usable commands."
        self.usage = "<command{all}> <category{all}> <verbose(?v)>"

    async def __call__(self, _vars, args, user, channel, guild, flags, **void):
        if guild:
            g_id = guild.id
        else:
            g_id = 0
        if g_id:
            enabled = _vars.enabled.setdefault(channel.id, list(default_commands))
        else:
            enabled = default_commands
        categories = _vars.categories
        commands = []
        for catg in categories:
            if catg in enabled or catg == "main":
                commands += categories[catg]
        c_name = getattr(channel, "name", "DM")
        u_perm = _vars.getPerms(user, guild)
        verb = "v" in flags
        argv = " ".join(args).lower()
        show = []
        for a in args:
            if (a in categories and a in enabled) or a == "main":
                show.append(
                    "\nCommands for **" + user.name
                    + "** in **" + channel.name
                    + "** in category **" + a
                    + "**:\n")
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
                        "```xml\n~" + name
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
                            "```xml\n~" + name
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
                        show.append("~" + name + " " + usage)
                    else:
                        show.append(
                            "\n~" + com.__name__
                            + "\nEffect: " + com.description
                            + "\nUsage: ~" + name + " " + usage)
            return (
                "Commands for **" + user.name + "** in **" + c_name
                + "**:\n```xml\n" + "\n".join(show) + "```", 1
                )
        return "\n".join(show), 1


class clearCache:
    is_command = True

    def __init__(self):
        self.name = ["cc"]
        self.min_level = 1
        self.description = "Clears all cached data."
        self.usage = ""

    async def __call__(self, _vars, **void):
        _vars.resetGlobals()
        _vars.loadSave()
        return "```css\nCache cleared!```"


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
            c_perm = _vars.evalMath(" ".join(args[1:]))
            s_user = user
            s_perm = perm
            if "everyone" in args[0] or "here" in args[0]:
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

    async def __call__(self, _vars, argv, flags, user, channel, perm, **void):
        if "e" in flags or "d" in flags:
            if perm < 3:
                raise PermissionError(
                    "Insufficient priviliges to change command list for " + uniStr(channel.name)
                    + ".\nRequred level: " + uniStr(3)
                    + ", Current level: " + uniStr(perm) + "."
                    )
        catg = argv.lower()
        print(catg)
        if not catg:
            if "e" in flags:
                categories = list(_vars.categories)
                categories.remove("main")
                _vars.enabled[channel.id] = categories
                doParallel(_vars.update)
                if "h" in flags:
                    return
                return "```css\nEnabled all command categories in " + uniStr(channel.name) + ".```"
            if "d" in flags:
                _vars.enabled[channel.id] = []
                doParallel(_vars.update)
                if "h" in flags:
                    return
                return "```css\nDisabled all command categories in " + uniStr(channel.name) + ".```"
            return (
                "Currently enabled command categories in **" + channel.name
                + "**:\n```css\n"
                + str(["main"] + _vars.enabled.get(channel.id, default_commands)) + "```"
            )
        else:
            if not catg in _vars.categories:
                raise KeyError("Unknown command category " + uniStr(argv) + ".")
            else:
                enabled = _vars.enabled.setdefault(channel.id, {})
                if "e" in flags:
                    if catg in enabled:
                        raise IndexError(
                            "Command category " + uniStr(catg)
                            + " is already enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.append(catg)
                    doParallel(_vars.update)
                    if "h" in flags:
                        return
                    return (
                        "```css\nEnabled command category " + uniStr(catg)
                        + " in " + uniStr(channel.name) + ".```"
                        )
                if "d" in flags:
                    if catg not in enabled:
                        raise IndexError(
                            "Command category " + uniStr(catg)
                            + " is not currently enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.remove(catg)
                    doParallel(_vars.update)
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
    time_consuming = True

    def __init__(self):
        self.name = ["shutdown"]
        self.min_level = inf
        self.description = "Restarts or shuts down the bot."
        self.usage = ""

    async def __call__(self, client, channel, user, guild, name, _vars, **void):
        if name == "shutdown":
            s_perm = _vars.getPerms(user, guild)
            if s_perm is not nan:
                raise PermissionError("Insufficient priviliges to request shutdown.")
            await channel.send("Shutting down... :wave:")
            f = open("shutdown", "wb")
            f.close()
        else:
            await channel.send("Restarting... :wave:")
        _vars.update(True)
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        try:
            await client.close()
        except:
            del client
        del _vars
        sys.exit()
        raise BaseException("Shutting down...")


class suspend:
    is_command = True

    def __init__(self):
        self.name = ["block"]
        self.min_level = nan
        self.description = "Prevents a user from accessing the bot's commands. Overrides ~perms."
        self.usage = "<0:user> <1:value[]>"

    async def __call__(self, _vars, user, guild, args, **void):
        if len(args) < 2:
            if len(args) >= 1:
                user = await _vars.fetch_user(_vars.verifyID(args[0]))
            return (
                "```css\nCurrent suspension status of " + uniStr(user.name) + ": "
                + uniStr(_vars.bans[0].get(user.id, None)) + ".```"
                )
        else:
            user = await _vars.fetch_user(_vars.verifyID(args[0]))
            change = _vars.evalMath(args[1])
            _vars.bans[0][user.id] = change
            doParallel(_vars.update)
            return (
                "```css\nChanged suspension status of " + uniStr(user.name) + " to "
                + uniStr(change) + ".```"
                )


class loop:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["for", "rep", "repeat", "while"]
        self.min_level = 1
        self.description = "Loops a command."
        self.usage = "<0:iterations> <1:command> <hide(?h)>"

    async def __call__(self, args, argv, message, callback, _vars, flags, perm, **void):
        iters = round(float(_vars.evalMath(args[0])))
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
            #print(loop)
            asyncio.create_task(callback(
                message, func, cb_argv=func2, cb_flags=flags, loop=loop,
                ))
        if not "h" in flags:
            return "```css\nLooping [" + func + "] " + uniStr(iters) + " times...```"
