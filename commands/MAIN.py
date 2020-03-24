import discord
try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
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
        self.flags = "v"

    async def __call__(self, args, user, channel, guild, flags, perm, **void):
        _vars = self._vars
        enabled = _vars.data["enabled"]
        g_id = guild.id
        prefix = _vars.getPrefix(g_id)
        enabled = enabled.get(channel.id, list(default_commands))
        c_name = getattr(channel, "name", "DM")
        admin = (not inf > perm, perm is nan)[c_name == "DM"]
        categories = _vars.categories
        verb = "v" in flags
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
                        ("```xml\n", "```ini\n")["v" in flags] + prefix + name
                        + "\nAliases: " + str(com.name)
                        + "\nEffect: " + description
                        + (
                            "\nUsage: " + prefix + name + " " + usage
                            + "\nLevel: " + uniStr(com.min_display)
                        ) * ("v" in flags)
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
                            + (
                                "\nUsage: " + prefix + name + " " + usage
                                + "\nLevel: " + uniStr(com.min_display)
                            )
                            + "```"
                        )
                        if (not len(show)) or len(show[-1]) < len(newstr):
                            show = [newstr]
        if not show:
            commands = hlist()
            for catg in categories:
                if catg in enabled or admin:
                    commands.extend(categories[catg])
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
                            "\nUsage: " + prefix + name + " " + usage
                            + "\nEffect: " + com.description
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
        self.flags = "fh"

    async def __call__(self, _vars, args, user, perm, guild, flags, **void):
        if len(args) < 2:
            if len(args) < 1:
                t_user = user
            else:
                if "@e" in args[0] or "everyone" in args[0] or "here" in args[0]:
                    return (
                        "Current user permissions for **" + guild.name + "**:\n```ini\n"
                        + strIter(_vars.data["perms"].get(guild.id, {})) + "```"
                    )
                else:
                    u_id = _vars.verifyID(args[0])
                    try:
                        t_user = await _vars.fetch_user(u_id)
                    except (TypeError, discord.NotFound):
                        try:
                            t_user = await _vars.fetch_member(u_id, guild)
                        except LookupError:
                            t_user = await _vars.fetch_whuser(u_id, guild)
            print(t_user)
            t_perm = _vars.getPerms(t_user.id, guild)
        else:
            c_perm = await _vars.evalMath(" ".join(args[1:]), guild.id)
            check = args[0].lower()
            if "everyone" in check or "here" in check:
                t_user = None
                t_perm = inf
                name = "everyone"
            else:
                u_id = _vars.verifyID(args[0])
                try:
                    t_user = await _vars.fetch_user(u_id)
                except (TypeError, discord.NotFound):
                    try:
                        t_user = await _vars.fetch_member(u_id, guild)
                    except LookupError:
                        t_user = await _vars.fetch_whuser(u_id, guild)
                t_perm = _vars.getPerms(t_user.id, guild)
                name = t_user.name
            if t_perm is nan or c_perm is nan:
                m_perm = nan
            else:
                m_perm = max(t_perm, c_perm, 1) + 1
            if not perm <= m_perm and m_perm is not nan:
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
                reason = (
                    "to change permissions for " + uniStr(name)
                    + " in " + uniStr(guild.name)
                    + " from " + uniStr(t_perm)
                    + " to " + uniStr(c_perm)
                )
                self.permError(perm, m_perm, reason)
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
        self.min_display = "0~3"
        self.description = "Shows, enables, or disables a command category in the current channel."
        self.usage = "<command{all}> <enable(?e)> <disable(?d)> <list(?l)> <hide(?h)>"
        self.flags = "edlh"

    async def __call__(self, argv, flags, user, channel, perm, **void):
        update = self.data["enabled"].update
        _vars = self._vars
        enabled = _vars.data["enabled"]
        if "e" in flags or "d" in flags:
            req = 3
            if perm < req:
                reason = (
                    "to change command list for "
                    + uniStr(channel.name)
                )
                self.permError(perm, req, reason)
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
                raise LookupError("Unknown command category " + uniStr(argv) + ".")
            else:
                enabled = enabled.setdefault(channel.id, {})
                if "e" in flags:
                    if catg in enabled:
                        raise ValueError(
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
                        raise ValueError(
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

    def __init__(self):
        self.name = ["Shutdown"]
        self.min_level = nan
        self.description = "Restarts or shuts down the bot."
        self.usage = ""

    async def __call__(self, client, channel, user, guild, name, _vars, **void):
        if name.lower() == "shutdown":
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        _vars.update()
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        for _ in loop(5):
            try:
                f = open(_vars.restart, "wb")
                f.close()
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        for _ in loop(8):
            try:
                if "log.txt" in os.listdir():
                    os.remove("log.txt")
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        if name.lower() == "shutdown":
            f = open(_vars.shutdown, "wb")
            f.close()
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
        susp = _vars.data["users"].get(user.id, {"commands": 0, "suspended": 0})
        if len(args) < 2:
            if len(args) >= 1:
                user = await _vars.fetch_user(_vars.verifyID(args[0]))
            return (
                "```css\nCurrent suspension status of " + uniStr(user.name) + ": "
                + uniStr(susp) + ".```"
            )
        else:
            user = await _vars.fetch_user(_vars.verifyID(args[0]))
            change = await _vars.evalMath(args[1], guild.id)
            susp["suspended"] = change
            _vars.data["users"][user.id] = susp
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
        self.min_display = "0~3"
        self.description = "Shows or changes the prefix for commands for this server."
        self.usage = "<prefix[]>"
        self.flags = "h"

    async def __call__(self, argv, guild, perm, _vars, flags, **void):
        pref = _vars.data["prefixes"]
        update = self.data["prefixes"].update
        if not argv:
            return (
                "```css\nCurrent command prefix for " + uniStr(guild.name)
                + ": " + _vars.getPrefix(guild) + "```"
            )
        req = 3
        if perm < req:
            reason = (
                "to change command prefix for "
                + uniStr(guild.name)
            )
            self.permError(perm, req, reason)
        prefix = argv
        if prefix.startswith("\\"):
            raise TypeError("Prefix must not begin with backslash.")
        pref[guild.id] = prefix
        update()
        if "h" not in flags:
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
        self.min_display = "1+"
        self.description = "Loops a command."
        self.usage = "<0:iterations> <1:command>"

    async def __call__(self, args, argv, message, channel, callback, _vars, perm, guild, **void):
        num = await _vars.evalMath(args[0], guild.id)
        iters = round(num)
        scale = 3
        limit = perm * scale
        if iters > limit:
            reason = (
                "to execute loop of " + uniStr(iters)
                + " iterations"
            )
            self.permError(perm, ceil(iters / scale), reason)
        func = func2 = " ".join(args[1:])
        if func:
            while func[0] == " ":
                func = func[1:]
        if isValid(perm):
            for n in self.name:
                if (_vars.getPrefix(guild) + n).upper() in func.replace(" ", "").upper():
                    raise PermissionError("Must be server owner to execute nested loop.")
        func2 = " ".join(func2.split(" ")[1:])
        create_task(_vars.sendReact(
            channel,
            (
                "```css\nLooping [" + func + "] " + uniStr(iters)
                + " time" + "s" * (iters != 1) + "...```"
            ),
            reacts=["❎"],
        ))
        for i in range(iters):
            loop = i < iters - 1
            create_task(callback(
                message, func, cb_argv=func2, loop=loop,
            ))
            await asyncio.sleep(0.5)


class Avatar:
    is_command = True

    def __init__(self):
        self.name = ["PFP", "Icon"]
        self.min_level = 0
        self.description = "Sends a link to the avatar of a user or server."
        self.usage = "<user>"

    async def getGuildData(self, g):
        _vars = self._vars
        url = _vars.strURL(g.icon_url)
        name = g.name
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + name + "](" + url + ")"
        print(emb.to_dict())
        return {
            "embed": emb,
        }

    def getMimicData(self, p):
        _vars = self._vars
        url = _vars.strURL(p.url)
        name = p.name
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + name + "](" + url + ")"
        print(emb.to_dict())
        return {
            "embed": emb,
        }

    async def __call__(self, argv, guild, _vars, client, user, **void):
        g, guild = guild, None
        if argv:
            try:
                u_id = _vars.verifyID(argv)
            except:
                u_id = argv
            try:
                p = _vars.get_mimic(u_id)
                return self.getMimicData(p)
            except:
                try:
                    u = await _vars.fetch_member(u_id, g)
                except:
                    try:
                        u = await _vars.fetch_user(u_id)
                    except:
                        if type(u_id) is str and ("everyone" in u_id or "here" in u_id):
                            guild = g
                        else:
                            try:
                                guild = await _vars.fetch_guild(u_id)
                            except:
                                try:
                                    channel = await _vars.fetch_channel(u_id)
                                except:
                                    try:
                                        u = await _vars.fetch_whuser(u_id, g)
                                    except EOFError:
                                        u = None
                                        if g.id in _vars.data["counts"]:
                                            if u_id in _vars.data["counts"][g.id]["counts"]:
                                                u = _vars.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("Unable to find user or server from ID.")
                                try:
                                    guild = channel.guild
                                except NameError:
                                    pass
                                except AttributeError:
                                    guild = None
                                    u = channel.recipient
                        if guild is not None:
                            return await self.getGuildData(guild)                        
        else:
            u = user
        guild = g
        name = str(u)
        url = _vars.strURL(u.avatar_url)
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + name + "](" + url + ")"
        print(emb.to_dict())
        return {
            "embed": emb,
        }


class Info:
    is_command = True

    def __init__(self):
        self.name = ["UserInfo", "ServerInfo"]
        self.min_level = 0
        self.description = "Shows information about the target user or server."
        self.usage = "<user> <verbose(?v)>"
        self.flags = "v"

    async def getGuildData(self, g, flags={}):
        _vars = self._vars
        url = _vars.strURL(g.icon_url)
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
        if g.description:
            d += "```\n" + str(g.description) + "```"
        emb.description = d
        top = None
        try:
            g.region
            pcount = await _vars.database["counts"].getUserMessages(None, g)
        except AttributeError:
            pcount = 0
        try:
            if "v" in flags:
                pavg = await _vars.database["counts"].getUserAverage(None, g)
                users = deque()
                us = await _vars.database["counts"].getGuildMessages(g)
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
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        if top is not None:
            emb.add_field(name="Top users", value=top, inline=0)
        print(emb.to_dict())
        return {
            "embed": emb,
        }

    def getMimicData(self, p, flags={}):
        _vars = self._vars
        url = _vars.strURL(p.url)
        name = p.name
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        if p.description:
            d = "```fix\n" + p.id + "``````\n" + str(p.description) + "```"
        else:
            d = ""
        emb.description = d
        pcnt = 0
        try:
            if "v" in flags:
                ptot = p.total
                pcnt = p.count
                pavg = ptot / pcnt
        except (AttributeError, KeyError):
            pass
        emb.add_field(name="Mimic ID", value=str(p.id), inline=0)
        emb.add_field(name="Name", value=str(p.name), inline=0)
        emb.add_field(name="Prefix", value=str(p.prefix), inline=1)
        emb.add_field(name="Creation time", value=str(p.created_at), inline=1)
        if "v" in flags:
            emb.add_field(name="Gender", value=str(p.gender), inline=1)
            ctime = p.birthday
            age = (datetime.datetime.utcnow() - ctime).total_seconds() / TIMEUNITS["year"]
            emb.add_field(name="Birthday", value=str(ctime), inline=1)
            emb.add_field(name="Age", value=str(roundMin(round(age, 1))), inline=1)
        if pcnt:
            emb.add_field(name="Post count", value=str(pcnt), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        print(emb.to_dict())
        return {
            "embed": emb,
        }

    async def __call__(self, argv, guild, _vars, client, user, flags, **void):
        member = True
        g, guild = guild, None
        if argv:
            try:
                u_id = _vars.verifyID(argv)
            except:
                u_id = argv
            try:
                p = _vars.get_mimic(u_id)
                return self.getMimicData(p, flags)
            except:
                try:
                    u = await _vars.fetch_member(u_id, g)
                except:
                    try:
                        u = await _vars.fetch_user(u_id)
                        member = False
                    except:
                        if type(u_id) is str and ("everyone" in u_id or "here" in u_id):
                            guild = g
                        else:
                            try:
                                guild = await _vars.fetch_guild(u_id)
                            except:
                                try:
                                    channel = await _vars.fetch_channel(u_id)
                                except:
                                    try:
                                        u = await _vars.fetch_whuser(u_id, g)
                                    except EOFError:
                                        u = None
                                        if g.id in _vars.data["counts"]:
                                            if u_id in _vars.data["counts"][g.id]["counts"]:
                                                u = _vars.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("Unable to find user or server from ID.")
                                try:
                                    guild = channel.guild
                                except NameError:
                                    pass
                                except AttributeError:
                                    guild = None
                                    u = channel.recipient
                        if guild is not None:
                            return await self.getGuildData(guild, flags)                        
        else:
            u = user
        guild = g
        name = str(u)
        dname = u.display_name * (u.display_name != u.name)
        url = _vars.strURL(u.avatar_url)
        try:
            is_sys = u.system
        except AttributeError:
            is_sys = False
        is_bot = u.bot
        is_self = u.id == client.user.id
        is_self_owner = u.id == _vars.owner_id
        is_guild_owner = u.id == guild.owner_id
        if member:
            joined = getattr(u, "joined_at", None)
        else:
            joined = None
        created = u.created_at
        activity = "\n".join(_vars.strActivity(i) for i in getattr(u, "activities", []))
        role = ", ".join(str(i) for i in getattr(u, "roles", []) if not i.is_default())
        coms = msgs = avgs = 0
        pos = None
        if "v" in flags:
            try:
                coms = _vars.data["users"][u.id]["commands"]
            except LookupError:
                pass
            try:
                msgs = await _vars.database["counts"].getUserMessages(u, guild)
                avgs = await _vars.database["counts"].getUserAverage(u, guild)
                if guild.owner.id != client.user.id:
                    us = await _vars.database["counts"].getGuildMessages(guild)
                    if type(us) is str:
                        pos = us
                    else:
                        ul = sorted(
                            us,
                            key=lambda k: us[k],
                            reverse=True,
                        )
                        try:
                            i = ul.index(u.id)
                            while i >= 1 and us[ul[i - 1]] == us[ul[i]]:
                                i -= 1
                            pos = i + 1
                        except ValueError:
                            if joined:
                                pos = len(ul) + 1
            except LookupError:
                pass
        if is_self and _vars.website is not None:
            url2 = _vars.website
        else:
            url2 = url
        emb = discord.Embed(colour=_vars.randColour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url2)
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
        if dname:
            emb.add_field(name="Nickname", value=dname, inline=1)
        if msgs:
            emb.add_field(name="Post count", value=str(msgs), inline=1)
        if avgs:
            emb.add_field(name="Average post length", value=str(round(avgs, 9)), inline=1)
        if pos:
            emb.add_field(name="Server rank", value=str(pos), inline=1)
        if role:
            emb.add_field(name="Roles", value=role, inline=0)
        print(emb.to_dict())
        return {
            "embed": emb,
        }
        

class Hello:
    is_command = True

    def __init__(self):
        self.name = ["Hi", "Ping", "👋", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
        self.min_level = 0
        self.description = "Sends a waving emoji. Useful for checking whether the bot is online."
        self.usage = ""
    
    async def __call__(self, channel, _vars, **void):
        return "👋"



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
        try:
            shards = len(client.latencies)
        except AttributeError:
            shards = 1
        size = _vars.codeSize
        stats = _vars.currState
        return (
            "```css"
            + "\nActive users: " + uniStr(len(client.users))
            + ", Active servers: " + uniStr(_vars.guilds)
            + ", Active shards: " + uniStr(shards)
            
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


class Reminder:
    is_command = True

    def __init__(self):
        self.name = ["RemindMe", "Reminders"]
        self.min_level = 0
        self.description = "Sets a reminder for a certain date and time."
        self.usage = "<1:message> <0:time> <disable(?d)>"
        self.flags = "ed"

    async def __call__(self, argv, args, flags, _vars, user, guild, **void):
        rems = _vars.data["reminders"].get(user.id, hlist())
        update = _vars.database["reminders"].update
        if "d" in flags:
            if not argv:
                i = 0
            else:
                i = await _vars.evalMath(argv, guild)
            x = rems.pop(i)
            update()
            return (
                "```css\nSuccessfully removed "
                + uniStr(x.msg) + " from reminders list for "
                + uniStr(user) + ".```"
            )
        if not argv:
            if not len(rems):
                return (
                    "```css\nNo reminders currently set for "
                    + uniStr(user) + ".```"
                )
            d = datetime.datetime.utcnow()
            s = strIter(rems, key=lambda x: limStr(x.msg, 64) + "➡️" + sec2Time((x.t - d).total_seconds()))
            return (
                "Current reminders set for **" + str(user)
                + "**:```ini" + s + "```"
            )
        if len(rems) >= 32:
            raise OverflowError("You have reached the maximum of 32 reminders. Please remove one to add another.")
        while True:
            spl = None
            if "in" in argv:
                if " in " in argv:
                    spl = argv.split(" in ")
                elif argv.startswith("in "):
                    spl = argv[3:]
                if spl is not None:
                    msg = " in ".join(spl[:-1])
                    t = await _vars.evalTime(args[-1], guild)
                    break
            if "at" in argv:
                if " at " in argv:
                    spl = argv.split(" at ")
                elif argv.startswith("at "):
                    spl = argv[3:]
                if spl is not None:
                    msg = " at ".join(spl[:-1])
                    t = tparser.parse(spl[-1]).timestamp() - datetime.datetime.utcnow().timestamp()
                    break
            msg = " ".join(args[:-1])
            t = await _vars.evalTime(args[-1], guild)
            break
        msg = msg.strip("< >")
        if not msg:
            msg = "[SAMPLE REMINDER]"
        elif len(msg) > 256:
            raise OverflowError("Reminder message too long (" + str(len(msg)) + "> 256).")
        rems.append(freeClass(
            msg=msg,
            t=datetime.timedelta(seconds=t) + datetime.datetime.utcnow(),
        ))
        _vars.data["reminders"][user.id] = sort(rems, key=lambda x: x.t)
        update()
        return (
            "```css\nSuccessfully set reminder for "
            + uniStr(user) + " in " + uniStr(sec2Time(t)) + ":\n"
            + msg + "```"
        )


class updateReminders:
    is_database = True
    name = "reminders"
    user = True

    def __init__(self):
        pass

    async def __call__(self):
        if self.busy:
            return
        t = datetime.datetime.utcnow()
        i = 1
        changed = False
        for u_id in tuple(self.data):
            temp = self.data[u_id]
            if not len(temp):
                self.data.pop(u_id)
                changed = True
                continue
            x = temp[0]
            if t >= x.t:
                temp.popleft()
                changed = True
                ch = await self._vars.getDM(u_id)
                await ch.send("```asciidoc\n" + x.msg + "```")
            if not i & 16383:
                await asyncio.sleep(0.4)
            i += 1
        if changed:
            self.update()
        self.busy = False


class updateMessageCount:
    is_database = True
    name = "counts"
    no_file = True

    def getMessageLength(self, message):
        return len(message.system_content) + sum(len(e) for e in message.embeds)

    def startCalculate(self, guild):
        self.data[guild.id] = {"counts": {}, "totals": {}}
        create_task(self.getUserMessageCount(guild))

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
                elif 0 not in d:
                    return "Calculating..."
                c = d["counts"]
                if user is None:
                    return sum(c.values())
                return c.get(user.id, 0)
            self.startCalculate(guild)
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
                elif 0 not in d:
                    return "Calculating..."
                t = d["totals"]
                c = d["counts"]
                if user is None:
                    return sum(t.values()) / sum(c.values())
                try:
                    return t.get(user.id, 0) / c.get(user.id, 1)
                except ZeroDivisionError:
                    c.pop(user.id)
                    try:
                        t.pop(user.id)
                    except KeyError:
                        pass
                    return 0
        return "Calculating..."            

    async def getGuildMessages(self, guild):
        if self.scanned == -1:
            c_id = self._vars.client.user.id
            if guild is None or hasattr(guild, "isDM"):
                channel = guild.channel
                if channel is None:
                    return 0
                messages = await channel.history(limit=None).flatten()
                return len(messages)
            if guild.id in self.data:
                try:
                    return self.data[guild.id]["counts"]
                except:
                    return self.data[guild.id]
            self.startCalculate(guild)
            create_task(self.getUserMessageCount(guild))
        return "Calculating..."

    async def getUserMessageCount(self, guild):

        async def getChannelHistory(channel, returns):
            try:
                messages = []
                for i in range(16):
                    history = channel.history(
                        limit=None,
                    )
                    print(history)
                    try:
                        messages = await history.flatten()
                        break
                    except discord.Forbidden:
                        raise
                    except:
                        print(traceback.format_exc())
                    await asyncio.sleep(20 * (i ** 2 + 1))
                print(len(messages))
                returns[0] = messages
            except:
                print(channel.name)
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
            create_task(getChannelHistory(
                channel,
                histories[-1],
            ))
            i += 1
        while [None] in histories:
            await asyncio.sleep(2)
        print("Counting...")
        while [] in histories:
            histories.remove([])
        mmax = 65536 / len(histories)
        caches = hlist()
        for messages in histories:
            temp = hlist()
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
                if len(temp) < mmax:
                    temp.append(message)
                i += 1
            caches.append(temp)
        addDict(self.data[guild.id], {"counts": data, "totals": avgs, 0: True})
        for temp in caches:
            print("[" + str(len(temp)) + "]")
            for message in temp:
                self._vars.cacheMessage(message)
                if not i & 8191:
                    await asyncio.sleep(0.5)
                i += 1
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
                self.startCalculate(guild)
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
                self.startCalculate(guild)

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
                self.startCalculate(guild)

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
                self.startCalculate(guild)


class updatePrefix:
    is_database = True
    name = "prefixes"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateEnabled:
    is_database = True
    name = "enabled"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateUsers:
    is_database = True
    name = "users"
    suspected = "users.json"
    user = True

    def __init__(self):
        self.suspclear = inf
        try:
            self.lastsusp = None
            f = open(self.suspected, "r")
            susp = f.read()
            f.close()
            os.remove(self.suspected)
            if susp:
                u_id = int(susp)
                udata = self.data[u_id]
                days = max(0, (udata["suspended"] - time.time()) / 86400)
                try:
                    days **= 4
                except (OverflowError, ValueError, TypeError):
                    days = inf
                days += 1.125
                udata["suspended"] = time.time() + days * 86400
                if days >= self._vars.min_suspend - 1:
                    self.lastsusp = u_id
                self.update()
                self.update(True)
            print(self.lastsusp)
        except FileNotFoundError:
            pass

    async def _command_(self, user, command, **void):
        udata = self.data.setdefault(user.id, {"commands": 0, "suspended": 0})
        udata["commands"] += 1
        self.update()
        if user.id not in (self._vars.client.user.id, self._vars.owner_id):
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
            secs = self.data.get(u_susp.id, {"suspended":0})["suspended"] - time.time()
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
