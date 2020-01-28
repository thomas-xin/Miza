import asyncio
from smath import *

default_commands = ["string", "admin"]


class help:
    is_command = True

    def __init__(self):
        self.name = ["?"]
        self.minm = -inf
        self.desc = "Shows a list of usable commands."
        self.usag = "<command:[]> <verbose:(?v)>"

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
                    + "**:")
                for com in categories[a]:
                    name = com.__name__
                    minm = com.minm
                    desc = com.desc
                    usag = com.usag
                    if minm > u_perm or (u_perm is not nan and minm is nan):
                        continue
                    newstr = (
                        "\n`" + name
                        + "`\nAliases: " + str(com.name)
                        + "\nEffect: " + desc
                        + "\nUsage: " + usag
                        + "\nRequired permission level: **__" + str(minm)
                        + "__**"
                    )
                    show.append(newstr)
        if not show:
            for c in categories:
                catg = categories[c]
                if not (c in enabled or c == "main"):
                    continue
                for com in catg:
                    name = com.__name__
                    minm = com.minm
                    desc = com.desc
                    usag = com.usag
                    if minm > u_perm or (u_perm is not nan and minm is nan):
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
                            + "\nEffect: " + desc
                            + "\nUsage: " + usag
                            + "\nRequired permission level: " + str(minm)
                            + "```"
                        )
                        if (not len(show)) or len(show[-1]) < len(newstr):
                            show = [newstr]
        if not show:
            for com in commands:
                name = com.__name__
                minm = com.minm
                desc = com.desc
                usag = com.usag
                if minm > u_perm or (u_perm is not nan and minm is nan):
                    continue
                if desc != "":
                    if not verb:
                        show.append("`" + name + " " + usag + "`")
                    else:
                        show.append(
                            "\n`" + com.__name__
                            + "`\nEffect: " + com.desc
                            + "\nUsage: " + name + " " + usag)
            return "Commands for **" + user.name + "** in **" + channel.name + "**:\n" + "\n".join(show)
        return "\n".join(show)


class clearCache:
    is_command = True

    def __init__(self):
        self.name = ["cc"]
        self.minm = 1
        self.desc = "Clears all cached data."
        self.usag = ""

    async def __call__(self, client, _vars, **void):
        _vars.resetGlobals()
        _vars.loadSave()
        return "```\nCache cleared!```"


class perms:
    is_command = True

    def __init__(self):
        self.name = ["changePerms", "perm", "changePerm"]
        self.minm = -inf
        self.desc = "Shows or changes a user's permission level."
        self.usag = "<0:user:{self}> <1:level{curr}> <hide:(?h)>"

    async def __call__(self, client, _vars, args, user, guild, flags, **void):
        if guild is None:
            raise ReferenceError("This command is only available in servers.")
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
            t_user = await client.fetch_user(_vars.verifyID(args[0]))
            t_perm = _vars.getPerms(t_user.id, guild)
            if t_perm is nan or c_perm is nan:
                m_perm = nan
            else:
                m_perm = max(t_perm, c_perm, 1) + 1
            if not s_perm <= m_perm and m_perm is not nan:
                _vars.setPerms(t_user.id, guild, c_perm)
                if "h" in flags:
                    return
                return (
                    "Changed permissions for **"+ t_user.name
                    + "** in **" + guild.name
                    + "** from **__" + str(t_perm)
                    + "__** to **__" + str(c_perm) + "__**."
                )
            else:
                return (
                    "Error: Insufficient priviliges to change permissions for **" + t_user.name
                    + "** in **" + guild.name
                    + "** from **__" + str(t_perm)
                    + "__** to **__" + str(c_perm)
                    + "__**.\nRequired level: **__" + str(m_perm)
                    + "__**, Current level: **__" + str(s_perm) + "__**"
                )
        return (
            "Current permissions for **" + t_user.name
            + "** in **" + guild.name
            + "**: **__" + str(t_perm) + "__**"
            )


class enableCommand:
    is_command = True

    def __init__(self):
        self.name = ["ec", "enable"]
        self.minm = 3
        self.desc = "Shows, enables, or disables a command category in the current channel."
        self.usag = "<command:{all}> <enable:(?e)> <disable:(?d)> <hide:(?h)>"

    async def __call__(self, client, _vars, argv, flags, channel, **void):
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
                return "Enabled all command categories in **" + channel.name + "**."
            if "d" in flags:
                _vars.enabled[channel.id] = []
                _vars.update()
                if "h" in flags:
                    return
                return "Disabled all command categories in **" + channel.name + "**."
            return (
                "Currently enabled command categories in **" + channel.name
                + "**:\n```\n"
                + str(["main"] + _vars.enabled.get(channel.id, default_commands)) + "```"
            )
        else:
            if not catg in _vars.categories:
                return "Error: Unknown command category **" + argv + "**."
            else:
                try:
                    enabled = _vars.enabled[channel.id]
                except KeyError:
                    enabled = {}
                    _vars.enabled[channel.id] = enabled
                if "e" in flags:
                    if catg in enabled:
                        return (
                            "Error: command category **" + catg
                            + "** is already enabled in **" + channel.name + "**."
                        )
                    enabled.append(catg)
                    _vars.update()
                    if "h" in flags:
                        return
                    return "Enabled command category **" + catg + "** in **" + channel.name + "**."
                if "d" in flags:
                    if catg not in enabled:
                        return (
                            "Error: command category **" + catg
                            + "** is not currently enabled in **" + channel.name + "**."
                        )
                    enabled.remove(catg)
                    _vars.update()
                    if "h" in flags:
                        return
                    return "Disabled command category **" + catg + "** in **" + channel.name + "**."
                return (
                    "Command category **" + catg
                    + "** is currently" + " not" * (catg not in enabled)
                    + " enabled in **" + channel.name + "**."
                )


class shutdown:
    is_command = True

    def __init__(self):
        self.name = ["gtfo"]
        self.minm = inf
        self.desc = "Shuts down the bot."
        self.usag = ""

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
        self.minm = nan
        self.desc = "Prevents a user from accessing the bot's commands. Overrides ~perms."
        self.usag = "<0:user> <1:value:[]>"

    async def __call__(self, _vars, client, user, guild, args, **void):
        if len(args) < 2:
            if len(args) >= 1:
                user = await client.fetch_user(_vars.verifyID(args[0]))
            return "Current suspension status of **" + user.name + "**: **__" + str(_vars.bans[0].get(user.id, None)) + "__**."
        else:
            user = await client.fetch_user(_vars.verifyID(args[0]))
            change = _vars.evalMath(args[1])
            _vars.bans[0][user.id] = change
            _vars.update()
            return "Changed suspension status of **" + user.name + "** to **__" + str(change) + "__**."


class loop:
    is_command = True

    def __init__(self):
        self.name = ["for", "rep", "repeat", "while"]
        self.minm = 2
        self.desc = "Loops a command."
        self.usag = "<0:iterations> <1:command> <hide:(?h)>"

    async def __call__(self, args, argv, message, callback, _vars, flags, **void):
        iters = _vars.evalMath(args[0])
        func = " ".join(args[1:])
        if flags:
            func += " ?" + "?".join(flags)
        for i in range(iters):
            asyncio.create_task(callback(message, func + " ?h" * (i != iters - 1)))
        if not "h" in flags:
            return "```\nLooping <" + func + "> " + str(iters) + " times...```"
