import asyncio
from smath import *

default_commands = ["string", "admin"]


class help:
    is_command = True

    def __init__(self):
        self.name = ["?"]
        self.min_level = -inf
        self.description = "Shows a list of usable commands."
        self.usage = "<command:[]> <verbose:(?v)>"

    async def __call__(self, _vars, client, args, user, channel, guild, flags, **void):
        if guild:
            g_id = guild.id
        else:
            g_id = 0
        if g_id:
            try:
                enabled = _vars.enabled[channel.id]
            except KeyError:
                enabled = _vars.enabled[channel.id] = default_commands
                _vars.update()
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
                        "```\n" + name
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
                        if n in argv:
                            found = True
                    if found:
                        newstr = (
                            "```\n" + name
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
                        show.append(name + " " + usage)
                    else:
                        show.append(
                            "\n" + com.__name__
                            + "\nEffect: " + com.description
                            + "\nUsage: ~" + name + " " + usage)
            return "Commands for **" + user.name + "** in **" + c_name + "**:\n```\n" + "\n".join(show) + "```"
        return "\n".join(show)


class clearCache:
    is_command = True

    def __init__(self):
        self.name = ["cc"]
        self.min_level = 1
        self.description = "Clears all cached data."
        self.usage = ""

    async def __call__(self, client, _vars, **void):
        _vars.resetGlobals()
        _vars.loadSave()
        return "```\nCache cleared!```"


class perms:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["changePerms", "perm", "changePerm"]
        self.min_level = -inf
        self.description = "Shows or changes a user's permission level."
        self.usage = "<0:user:{self}> <1:level{curr}> <hide:(?h)>"

    async def __call__(self, client, _vars, args, user, guild, flags, **void):
        if len(args) < 2:
            if len(args) < 1:
                t_user = user
            else:
                t_user = await client.fetch_user(_vars.verifyID(args[0]))
            print(t_user)
            t_perm = _vars.getPerms(t_user.id, guild)
        else:
            c_perm = _vars.evalMath(" ".join(args[1:]))
            s_user = user
            s_perm = _vars.getPerms(s_user.id, guild)
            if "everyone" in args[0] or "here" in args[0]:
                t_user = None
                t_perm = inf
                name = "everyone"
            else:
                t_user = await client.fetch_user(_vars.verifyID(args[0]))
                t_perm = _vars.getPerms(t_user.id, guild)
                name = t_user.name
            if t_perm is nan or c_perm is nan:
                m_perm = nan
            else:
                m_perm = max(t_perm, c_perm, 1) + 1
            if not s_perm <= m_perm and m_perm is not nan:
                if t_user is None:
                    for u in guild.members:
                        _vars.setPerms(u.id, guild, c_perm)
                else:
                    _vars.setPerms(t_user.id, guild, c_perm)
                if "h" in flags:
                    return
                return (
                    "```\nChanged permissions for "+ uniStr(name)
                    + " in " + uniStr(guild.name)
                    + " from " + uniStr(t_perm)
                    + " to " + uniStr(c_perm) + ".```"
                )
            else:
                return (
                    "```\nError: Insufficient priviliges to change permissions for " + uniStr(name)
                    + " in " + uniStr(guild.name)
                    + " from " + uniStr(t_perm)
                    + " to " + uniStr(c_perm)
                    + ".\nRequired level: " + uniStr(m_perm)
                    + ", Current level: " + uniStr(s_perm) + ".```"
                )
        return (
            "```\nCurrent permissions for " + uniStr(t_user.name)
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
        self.usage = "<command:{all}> <enable:(?e)> <disable:(?d)> <hide:(?h)>"

    async def __call__(self, client, _vars, argv, flags, user, channel, guild, **void):
        if "e" in flags or "d" in flags:
            s_perm = _vars.getPerms(user, guild)
            if s_perm < 3:
                raise PermissionError(
                    "Insufficient priviliges to modify enabled commands in "
                    + uniStr(channel.name) + "."
                    )
        catg = argv.lower()
        print(catg)
        if not catg:
            if "e" in flags:
                categories = list(_vars.categories)
                categories.remove("main")
                _vars.enabled[channel.id] = categories
                _vars.update()
                if "h" in flags:
                    return
                return "```\nEnabled all command categories in " + uniStr(channel.name) + ".```"
            if "d" in flags:
                _vars.enabled[channel.id] = []
                _vars.update()
                if "h" in flags:
                    return
                return "```\nDisabled all command categories in " + uniStr(channel.name) + ".```"
            return (
                "Currently enabled command categories in **" + channel.name
                + "**:\n```\n"
                + str(["main"] + _vars.enabled.get(channel.id, default_commands)) + "```"
            )
        else:
            if not catg in _vars.categories:
                raise EOFError("Error: Unknown command category " + uniStr(argv) + ".")
            else:
                try:
                    enabled = _vars.enabled[channel.id]
                except KeyError:
                    enabled = {}
                    _vars.enabled[channel.id] = enabled
                if "e" in flags:
                    if catg in enabled:
                        raise IndexError(
                            "Command category " + uniStr(catg)
                            + " is already enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.append(catg)
                    _vars.update()
                    if "h" in flags:
                        return
                    return "```\nEnabled command category " + uniStr(catg) + " in " + uniStr(channel.name) + ".```"
                if "d" in flags:
                    if catg not in enabled:
                        raise IndexError(
                            "Command category " + uniStr(catg)
                            + " is not currently enabled in " + uniStr(channel.name) + "."
                        )
                    enabled.remove(catg)
                    _vars.update()
                    if "h" in flags:
                        return
                    return "```\nDisabled command category " + uniStr(catg) + " in " + uniStr(channel.name) + ".```"
                return (
                    "```\nCommand category " + uniStr(catg)
                    + " is currently" + uniStr(" not" * (catg not in enabled))
                    + " enabled in " + uniStr(channel.name) + ".```"
                )


class shutdown:
    is_command = True

    def __init__(self):
        self.name = ["gtfo"]
        self.min_level = inf
        self.description = "Shuts down the bot."
        self.usage = ""

    async def __call__(self, client, channel, **void):
        await channel.send("Shutting down... :wave:")
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        await client.close()
        sys.exit()
        quit()


class suspend:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = nan
        self.description = "Prevents a user from accessing the bot's commands. Overrides ~perms."
        self.usage = "<0:user> <1:value:[]>"

    async def __call__(self, _vars, client, user, guild, args, **void):
        if len(args) < 2:
            if len(args) >= 1:
                user = await client.fetch_user(_vars.verifyID(args[0]))
            return (
                "```\nCurrent suspension status of " + uniStr(user.name) + ": "
                + uniStr(_vars.bans[0].get(user.id, None)) + ".```"
                )
        else:
            user = await client.fetch_user(_vars.verifyID(args[0]))
            change = _vars.evalMath(args[1])
            _vars.bans[0][user.id] = change
            _vars.update()
            return (
                "```\nChanged suspension status of " + uniStr(user.name) + " to "
                + uniStr(change) + ".```"
                )


class loop:
    is_command = True

    def __init__(self):
        self.name = ["for", "rep", "repeat", "while"]
        self.min_level = 2
        self.description = "Loops a command."
        self.usage = "<0:iterations> <1:command> <hide:(?h)>"

    async def __call__(self, args, argv, message, callback, _vars, flags, **void):
        iters = _vars.evalMath(args[0])
        func = " ".join(args[1:])
        if flags:
            func += " ?" + "?".join(flags)
        for i in range(iters):
            asyncio.create_task(callback(message, func, cb_argv=argv, cb_flags=flags, loop=i != iters - 1))
        if not "h" in flags:
            return "```\nLooping <" + func + "> " + uniStr(iters) + " times...```"
