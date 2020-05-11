try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


default_commands = ["main", "string", "admin"]
standard_commands = default_commands + ["voice", "image", "game"]


class Help(Command):
    name = ["?"]
    description = "Shows a list of usable commands, or gives a detailed description of a command."
    usage = "<command{all}> <category{all}> <verbose(?v)>"
    flags = "v"

    async def __call__(self, args, user, channel, guild, flags, perm, **void):
        bot = self.bot
        enabled = bot.data.enabled
        g_id = guild.id
        prefix = bot.getPrefix(g_id)
        enabled = enabled.get(channel.id, list(default_commands))
        v = "v" in flags
        emb = discord.Embed(colour=randColour())
        emb.set_author(name="❓ Help ❓")
        found = {}
        for a in args:
            a = a.lower()
            if a in bot.categories:
                coms = bot.categories[a]
            elif a in bot.commands:
                coms = bot.commands[a]
            else:
                continue
            for com in coms:
                if com.__name__ in found:
                    found[com.__name__].append(com)
                else:
                    found[com.__name__] = hlist([com])
        if found:
            i = 0
            for k in found:
                if i >= 25:
                    break
                coms = found[k]
                for com in coms:
                    a = ", ".join(com.name)
                    if not a:
                        a = "[none]"
                    s = "```ini\n[Aliases] " + a
                    s += "\n[Effect] " + com.description.replace("⟨MIZA⟩", bot.client.user.name)
                    if v or len(found) <= 1:
                        s += (
                            "\n[Usage] " + prefix + com.__name__ + " " + com.usage
                            + "\n[Level] " + str(com.min_display)
                        )
                    s += "```"
                    emb.add_field(
                        name=prefix + com.__name__,
                        value=s,
                        inline=False
                    )
                i += 1
        else:
            emb.description = (
                "Please enter a command category to display usable commands,\nor see "
                + "[Commands](https://github.com/thomas-xin/Miza/wiki/Commands) for full command list."
            )
            if bot.categories:
                s = "```ini\n" + " ".join((sbHighlight(c) for c in standard_commands)) + "```"
                emb.add_field(name="Command category list", value=s)
        return freeClass(embed=emb), 1


class Perms(Command):
    server_only = True
    name = ["ChangePerms", "Perm", "ChangePerm", "Permissions"]
    description = "Shows or changes a user's permission level."
    usage = "<0:user{self}> <1:level[]> <hide(?h)>"
    flags = "fh"

    async def __call__(self, bot, args, user, perm, guild, flags, **void):
        if len(args) < 1:
            t_user = user
        else:
            check = args[0].lower()
            if "@" in args[0] and ("everyone" in check or "here" in check):
                args[0] = guild.id
            u_id = verifyID(args[0])
            try:
                t_user = await bot.fetch_user(u_id)
            except (TypeError, discord.NotFound):
                try:
                    t_user = await bot.fetch_member(u_id, guild)
                except LookupError:
                    try:
                        t_user = guild.get_role(u_id)
                        if t_user is None:
                            raise LookupError
                    except LookupError:
                        t_user = await bot.fetch_whuser(u_id, guild)
        print(t_user)
        t_perm = bot.getPerms(t_user.id, guild)
        if len(args) > 1:
            name = str(t_user)
            orig = t_perm
            expr = " ".join(args[1:])
            _op = None
            for operator in ("+=", "-=", "*=", "/=", "%="):
                if expr.startswith(operator):
                    expr = expr[2:].strip(" ")
                    _op = operator[0]
            num = await bot.evalMath(expr, guild)
            if _op is not None:
                num = eval(str(orig) + _op + str(num), {}, infinum)
            c_perm = num
            if t_perm is nan or isnan(c_perm):
                m_perm = nan
            else:
                m_perm = max(t_perm, abs(c_perm), 1) + 1
            if not perm < m_perm and not isnan(m_perm):
                if not m_perm < inf and guild.owner_id != user.id and not isnan(perm):
                    raise PermissionError("Must be server owner to assign non-finite permission level.")
                if t_user is None:
                    for u in guild.members:
                        bot.setPerms(u.id, guild, c_perm)
                else:
                    bot.setPerms(t_user.id, guild, c_perm)
                if "h" in flags:
                    return
                return (
                    "```css\nChanged permissions for [" + noHighlight(name)
                    + "] in [" + noHighlight(guild.name)
                    + "] from [" + noHighlight(t_perm)
                    + "] to [" + noHighlight(c_perm) + "].```"
                )
            else:
                reason = (
                    "to change permissions for " + str(name)
                    + " in " + str(guild.name)
                    + " from " + str(t_perm)
                    + " to " + str(c_perm)
                )
                self.permError(perm, m_perm, reason)
        return (
            "```css\nCurrent permissions for [" + noHighlight(t_user.name)
            + "] in [" + noHighlight(guild.name)
            + "]: [" + noHighlight(t_perm) + "].```"
        )


class EnabledCommands(Command):
    server_only = True
    name = ["EC", "Enable"]
    min_level = 0
    min_display = "0~3"
    description = "Shows, enables, or disables a command category in the current channel."
    usage = "<command{all}> <add(?e)> <remove(?d)> <list(?l)> <hide(?h)>"
    flags = "aedlh"

    async def __call__(self, argv, flags, user, channel, perm, **void):
        update = self.data.enabled.update
        bot = self.bot
        enabled = bot.data.enabled
        if "a" in flags or "e" in flags or "d" in flags:
            req = 3
            if perm < req:
                reason = (
                    "to change command list for "
                    + channel.name
                )
                self.permError(perm, req, reason)
        catg = argv.lower()
        if not catg:
            if "l" in flags:
                return (
                    "```css\nStandard command categories:\n"
                    + str(standard_commands) + "```"
                )
            if "e" in flags or "a" in flags:
                categories = list(standard_commands) #list(bot.categories)
                enabled[channel.id] = categories
                update()
                if "h" in flags:
                    return
                return (
                    "```css\nEnabled all standard command categories in ["
                    + noHighlight(channel.name) + "].```"
                )
            if "d" in flags:
                enabled[channel.id] = []
                update()
                if "h" in flags:
                    return
                return (
                    "```css\nDisabled all command categories in ["
                    + noHighlight(channel.name) + "].```"
                )
            return (
                "Currently enabled command categories in <#" + str(channel.id)
                + ">:\n```ini\n"
                + strIter(enabled.get(channel.id, default_commands)) + "```"
            )
        else:
            if not catg in bot.categories:
                raise LookupError("Unknown command category " + argv + ".")
            else:
                enabled = enabled.setdefault(channel.id, {})
                if "e" in flags or "a" in flags:
                    if catg in enabled:
                        raise ValueError(
                            "Command category " + catg
                            + " is already enabled in " + channel.name + "."
                        )
                    enabled.append(catg)
                    update()
                    if "h" in flags:
                        return
                    return (
                        "```css\nEnabled command category [" + noHighlight(catg)
                        + "] in [" + noHighlight(channel.name) + "].```"
                    )
                if "d" in flags:
                    if catg not in enabled:
                        raise ValueError(
                            "Command category " + catg
                            + " is not currently enabled in " + channel.name + "."
                        )
                    enabled.remove(catg)
                    update()
                    if "h" in flags:
                        return
                    return (
                        "```css\nDisabled command category [" + noHighlight(catg)
                        + "] in [" + noHighlight(channel.name) + "].```"
                    )
                return (
                    "```css\nCommand category [" + noHighlight(catg)
                    + "] is currently" + " not" * (catg not in enabled)
                    + "] enabled in [" + noHighlight(channel.name) + "].```"
                )


class Prefix(Command):
    name = ["ChangePrefix"]
    min_level = 0
    min_display = "0~3"
    description = "Shows or changes the prefix for ⟨MIZA⟩'s commands for this server."
    usage = "<prefix[]> <default(?d)>"
    flags = "hd"

    async def __call__(self, argv, guild, perm, bot, flags, **void):
        pref = bot.data.prefixes
        update = self.data.prefixes.update
        if "d" in flags:
            if guild.id in pref:
                pref.pop(guild.id)
                update()
            return (
                "```css\nSuccessfully reset command prefix for ["
                + noHighlight(guild.name) + "].```"
            )
        if not argv:
            return (
                "```css\nCurrent command prefix for [" + noHighlight(guild.name)
                + "]: [" + noHighlight(bot.getPrefix(guild)) + "].```"
            )
        req = 3
        if perm < req:
            reason = (
                "to change command prefix for "
                + guild.name
            )
            self.permError(perm, req, reason)
        prefix = argv
        if prefix.startswith("\\"):
            raise TypeError("Prefix must not begin with backslash.")
        pref[guild.id] = prefix
        update()
        if "h" not in flags:
            return (
                "```css\nSuccessfully changed command prefix for [" + noHighlight(guild.name)
                + "] to [" + noHighlight(argv) + "].```"
            )


class Loop(Command):
    time_consuming = 3
    name = ["For", "Rep", "Repeat", "While"]
    min_level = 1
    min_display = "1+"
    description = "Loops a command."
    usage = "<0:iterations> <1:command>"

    async def __call__(self, args, argv, message, channel, callback, bot, perm, guild, **void):
        num = await bot.evalMath(args[0], guild.id)
        iters = round(num)
        scale = 3
        limit = perm * scale
        if iters > limit:
            reason = (
                "to execute loop of " + str(iters)
                + " iterations"
            )
            self.permError(perm, ceil(iters / scale), reason)
        elif not isnan(perm) and iters > 256:
            raise PermissionError("Must be owner to execute loop of more than 256 iterations.")
        func = func2 = " ".join(args[1:])
        if func:
            while func[0] == " ":
                func = func[1:]
        if not isnan(perm):
            for n in self.name:
                if (
                    (bot.getPrefix(guild) + n).upper() in func.replace(" ", "").upper()
                ) or (
                    (str(bot.client.user.id) + ">" + n).upper() in func.replace(" ", "").upper()
                ):
                    raise PermissionError("Must be owner to execute nested loop.")
        func2 = " ".join(func2.split(" ")[1:])
        create_task(sendReact(
            channel,
            (
                "```css\nLooping [" + func + "] " + str(iters)
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


class Avatar(Command):
    name = ["PFP", "Icon"]
    min_level = 0
    description = "Sends a link to the avatar of a user or server."
    usage = "<user>"

    async def getGuildData(self, g):
        url = strURL(g.icon_url)
        for size in ("?size=1024", "?size=2048"):
            if url.endswith(size):
                url = url[:-len(size)] + "?size=4096"
        name = g.name
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        # print(emb.to_dict())
        return {
            "embed": emb,
        }

    def getMimicData(self, p):
        url = strURL(p.url)
        name = p.name
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        # print(emb.to_dict())
        return {
            "embed": emb,
        }

    async def __call__(self, argv, guild, bot, client, user, **void):
        g, guild = guild, None
        if argv:
            try:
                u_id = verifyID(argv)
            except:
                u_id = argv
            try:
                p = bot.get_mimic(u_id, user)
                return self.getMimicData(p)
            except:
                try:
                    u = await bot.fetch_member(u_id, g)
                except:
                    try:
                        u = await bot.fetch_user(u_id)
                    except:
                        if type(u_id) is str and ("everyone" in u_id or "here" in u_id):
                            guild = g
                        else:
                            try:
                                guild = await bot.fetch_guild(u_id)
                            except:
                                try:
                                    channel = await bot.fetch_channel(u_id)
                                except:
                                    try:
                                        u = await bot.fetch_whuser(u_id, g)
                                    except EOFError:
                                        u = None
                                        if g.id in bot.data.counts:
                                            if u_id in bot.data.counts[g.id]["counts"]:
                                                u = bot.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("Unable to find user or server from ID.")
                                try:
                                    guild = channel.guild
                                except NameError:
                                    pass
                                except (AttributeError, KeyError):
                                    guild = None
                                    u = channel.recipient
                        if guild is not None:
                            return await self.getGuildData(guild)                        
        else:
            u = user
        guild = g
        name = str(u)
        url = strURL(u.avatar_url)
        for size in ("?size=1024", "?size=2048"):
            if url.endswith(size):
                url = url[:-len(size)] + "?size=4096"
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        # print(emb.to_dict())
        return {
            "embed": emb,
        }


class Info(Command):
    name = ["UserInfo", "ServerInfo"]
    min_level = 0
    description = "Shows information about the target user or server."
    usage = "<user> <verbose(?v)>"
    flags = "v"

    async def getGuildData(self, g, flags={}):
        bot = self.bot
        url = strURL(g.icon_url)
        name = g.name
        try:
            u = g.owner
        except (AttributeError, KeyError):
            u = None
        emb = discord.Embed(colour=randColour())
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
            pcount = await bot.database.counts.getUserMessages(None, g)
        except (AttributeError, KeyError):
            pcount = 0
        try:
            if "v" in flags:
                pavg = await bot.database.counts.getUserAverage(None, g)
                users = deque()
                us = await bot.database.counts.getGuildMessages(g)
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
        except (AttributeError, KeyError):
            pass
        emb.add_field(name="Server ID", value=str(g.id), inline=0)
        emb.add_field(name="Creation time", value=str(g.created_at), inline=1)
        if "v" in flags:
            try:
                emb.add_field(name="Region", value=str(g.region), inline=1)
                emb.add_field(name="Nitro boosts", value=str(g.premium_subscription_count), inline=1)
            except (AttributeError, KeyError):
                pass
        emb.add_field(name="User count", value=str(g.member_count), inline=1)
        if pcount:
            emb.add_field(name="Post count", value=str(pcount), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        if top is not None:
            emb.add_field(name="Top users", value=top, inline=0)
        # print(emb.to_dict())
        return {
            "embed": emb,
        }

    def getMimicData(self, p, flags={}):
        bot = self.bot
        url = strURL(p.url)
        name = p.name
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        d = "<@" + str(p.u_id) + ">```fix\n" + p.id + "```"
        if p.description:
            d += "```\n" + str(p.description) + "```"
        emb.description = d
        pcnt = 0
        try:
            if "v" in flags:
                ptot = p.total
                pcnt = p.count
                if not pcnt:
                    pavg = 0
                else:
                    pavg = ptot / pcnt
        except (AttributeError, KeyError):
            pass
        emb.add_field(name="Mimic ID", value=str(p.id), inline=0)
        emb.add_field(name="Name", value=str(p.name), inline=0)
        emb.add_field(name="Prefix", value=str(p.prefix), inline=1)
        emb.add_field(name="Creation time", value=str(datetime.datetime.fromtimestamp(p.created_at)), inline=1)
        if "v" in flags:
            emb.add_field(name="Gender", value=str(p.gender), inline=1)
            ctime = datetime.datetime.fromtimestamp(p.birthday)
            age = (datetime.datetime.utcnow() - ctime).total_seconds() / TIMEUNITS["year"]
            emb.add_field(name="Birthday", value=str(ctime), inline=1)
            emb.add_field(name="Age", value=str(roundMin(round(age, 1))), inline=1)
        if pcnt:
            emb.add_field(name="Post count", value=str(pcnt), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        # print(emb.to_dict())
        return {
            "embed": emb,
        }

    async def __call__(self, argv, guild, bot, client, user, flags, **void):
        member = True
        g, guild = guild, None
        if argv:
            try:
                u_id = verifyID(argv)
            except:
                u_id = argv
            try:
                p = bot.get_mimic(u_id, user)
                return self.getMimicData(p, flags)
            except:
                try:
                    u = await bot.fetch_member(u_id, g)
                except:
                    try:
                        u = await bot.fetch_user(u_id)
                        member = False
                    except:
                        if type(u_id) is str and ("everyone" in u_id or "here" in u_id):
                            guild = g
                        else:
                            try:
                                guild = await bot.fetch_guild(u_id)
                            except:
                                try:
                                    channel = await bot.fetch_channel(u_id)
                                except:
                                    try:
                                        u = await bot.fetch_whuser(u_id, g)
                                    except EOFError:
                                        u = None
                                        if g.id in bot.data.counts:
                                            if u_id in bot.data.counts[g.id]["counts"]:
                                                u = bot.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("Unable to find user or server from ID.")
                                try:
                                    guild = channel.guild
                                except NameError:
                                    pass
                                except (AttributeError, KeyError):
                                    guild = None
                                    u = channel.recipient
                        if guild is not None:
                            return await self.getGuildData(guild, flags)                        
        else:
            u = user
        guild = g
        name = str(u)
        dname = u.display_name * (u.display_name != u.name)
        url = strURL(u.avatar_url)
        try:
            is_sys = u.system
        except (AttributeError, KeyError):
            is_sys = False
        is_bot = u.bot
        is_self = u.id == client.user.id
        is_self_owner = u.id == bot.owner_id
        is_guild_owner = u.id == guild.owner_id
        if member:
            joined = getattr(u, "joined_at", None)
        else:
            joined = None
        created = u.created_at
        activity = "\n".join(strActivity(i) for i in getattr(u, "activities", []))
        role = ", ".join(str(i) for i in getattr(u, "roles", []) if not i.is_default())
        coms = seen = msgs = avgs = gmsg = 0
        pos = None
        if "v" in flags:
            try:
                coms = bot.data.users[u.id]["commands"]
            except LookupError:
                pass
            try:
                ts = datetime.datetime.utcnow().timestamp()
                seen = sec2Time(max(0, ts - bot.data.users[u.id]["last_seen"])) + " ago"
            except LookupError:
                pass
            try:
                gmsg = bot.database.counts.getUserGlobalMessageCount(u)
                msgs = await bot.database.counts.getUserMessages(u, guild)
                avgs = await bot.database.counts.getUserAverage(u, guild)
                if guild.owner.id != client.user.id:
                    us = await bot.database.counts.getGuildMessages(guild)
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
        if is_self and bot.website is not None:
            url2 = bot.website
        else:
            url2 = url
        emb = discord.Embed(colour=randColour())
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
        if seen:
            emb.add_field(name="Last seen", value=str(seen), inline=1)
        if coms:
            emb.add_field(name="Commands used", value=str(coms), inline=1)
        if dname:
            emb.add_field(name="Nickname", value=dname, inline=1)
        if gmsg:
            emb.add_field(name="Global post count", value=str(gmsg), inline=1)
        if msgs:
            emb.add_field(name="Post count", value=str(msgs), inline=1)
        if avgs:
            emb.add_field(name="Average post length", value=str(round(avgs, 9)), inline=1)
        if pos:
            emb.add_field(name="Server rank", value=str(pos), inline=1)
        if role:
            emb.add_field(name="Roles", value=role, inline=0)
        # print(emb.to_dict())
        return {
            "embed": emb,
        }
        

class Hello(Command):
    name = ["Hi", "Ping", "👋", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
    min_level = 0
    description = "Sends a waving emoji. Useful for checking whether the bot is online."
    
    async def __call__(self, channel, bot, **void):
        return "👋"



class Status(Command):
    name = ["State"]
    min_level = 0
    description = "Shows the bot's current internal program state."

    async def __call__(self, flags, client, bot, **void):
        active = bot.getActive()
        latency = sec2Time(client.latency)
        try:
            shards = len(client.latencies)
        except AttributeError:
            shards = 1
        size = bot.codeSize
        stats = bot.currState
        return (
            "```ini"
            + "\nActive users: " + sbHighlight(len(client.users))
            + ", Active servers: " + sbHighlight(bot.guilds)
            + ", Active shards: " + sbHighlight(shards)
            
            + ".\nActive processes: " + sbHighlight(active[0])
            + ", Active threads: " + sbHighlight(active[1])
            + ", Active coroutines: " + sbHighlight(active[2])
            
            + ".\nConnected voice channels: " + sbHighlight(len(client.voice_clients))
            
            + ".\nCached files: " + sbHighlight(len(os.listdir("cache/")))
            
            + ".\nCode size: " + sbHighlight(size[0]) + " bytes"
            + ", " + sbHighlight(size[1]) + " lines"

            + ".\nSystem time: " + sbHighlight(datetime.datetime.now())
            + ".\nPing latency: " + sbHighlight(latency)
            
            + ".\nCPU usage: " + sbHighlight(round(stats[0], 3)) + "%"
            + ", RAM usage: " + sbHighlight(round(stats[1] / 1048576, 3)) + " MB"
            + ".```"
        )


class Reminder(Command):
    name = ["RemindMe", "Reminders", "Remind"]
    min_level = 0
    description = "Sets a reminder for a certain date and time."
    usage = "<1:message> <0:time> <disable(?d)>"
    flags = "aed"

    async def __call__(self, argv, name, message, flags, bot, user, guild, **void):
        msg = message.content
        argv2 = argv
        argv = msg[msg.lower().index(name) + len(name):].strip(" ").strip("\n")
        try:
            args = shlex.split(argv)
        except ValueError:
            args = argv.split(" ")
        rems = bot.data.reminders.get(user.id, [])
        update = bot.database.reminders.update
        if "d" in flags:
            if not argv:
                i = 0
            else:
                print(argv)
                i = await bot.evalMath(argv2, guild)
            x = rems.pop(i)
            update()
            return (
                "```ini\nSuccessfully removed ["
                + limStr(noHighlight(x["msg"]), 64) + "] from reminders list for ["
                + noHighlight(user) + "].```"
            )
        if not argv:
            if not len(rems):
                return (
                    "```ini\nNo reminders currently set for ["
                    + noHighlight(user) + "].```"
                )
            d = datetime.datetime.utcnow().timestamp()
            s = strIter(rems, key=lambda x: limStr(noHighlight(x["msg"]), 64) + " ➡️ " + sec2Time(x["t"] - d))
            return (
                "Current reminders set for **" + discord.utils.escape_markdown(str(user))
                + "**:```ini" + s + "```"
            )
        if len(rems) >= 64:
            raise OverflowError("You have reached the maximum of 64 reminders. Please remove one to add another.")
        while True:
            if name == "remind" and argv.startswith("me "):
                argv = argv[3:]
            if argv.startswith("to "):
                argv = argv[3:]
            elif argv.startswith("that "):
                argv = argv[3:]
            spl = None
            if "in" in argv:
                if " in " in argv:
                    spl = argv.split(" in ")
                elif argv.startswith("in "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " in ".join(spl[:-1])
                    t = await bot.evalTime(spl[-1], guild)
                    break
            if "at" in argv:
                if " at " in argv:
                    spl = argv.split(" at ")
                elif argv.startswith("at "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " at ".join(spl[:-1])
                    t = tparser.parse(spl[-1]).timestamp() - datetime.datetime.utcnow().timestamp()
                    break
            if "on" in argv:
                if " on " in argv:
                    spl = argv.split(" on ")
                elif argv.startswith("on "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " on ".join(spl[:-1])
                    t = tparser.parse(spl[-1]).timestamp() - datetime.datetime.utcnow().timestamp()
                    break
            msg = " ".join(args[:-1])
            t = await bot.evalTime(args[-1], guild)
            break
        msg = msg.strip(" ")
        if not msg:
            msg = "[SAMPLE REMINDER]"
        elif len(msg) > 512:
            raise OverflowError("Reminder message too long (" + str(len(msg)) + "> 512).")
        name = str(user)
        url = strURL(user.avatar_url)
        ts = datetime.datetime.utcnow().timestamp()
        rems.append(freeClass(
            name=name,
            url=url,
            msg=msg,
            t=t + ts,
            u=1
        ))
        bot.data.reminders[user.id] = sort(rems, key=lambda x: x["t"])
        try:
            bot.database.reminders.keyed.remove((0, user.id), key=lambda x: x[-1])
        except IndexError:
            pass
        bot.database.reminders.keyed.insort((bot.data.reminders[user.id][0]["t"], user.id), key=lambda x: x[0])
        update()
        emb = discord.Embed(description=msg)
        emb.set_author(name=name, url=url, icon_url=url)
        return {
            "content": ("```css\nSuccessfully set reminder for ["
                + noHighlight(user) + "] in [" + noHighlight(sec2Time(t)) + "]:```"
            ),
            "embed": emb,
        }


class Announcement(Command):
    name = ["Announce", "Announcements"]
    min_level = 2
    description = "Sets an announcement in the current channel for a certain date and time."
    usage = "<1:message> <0:time> <disable(?d)>"
    flags = "aed"

    async def __call__(self, name, message, flags, bot, user, channel, guild, **void):
        msg = message.content
        argv = msg[msg.lower().index(name) + len(name):].strip(" ").strip("\n")
        try:
            args = shlex.split(argv)
        except ValueError:
            args = argv.split(" ")
        rems = bot.data.reminders.get(channel.id, [])
        update = bot.database.reminders.update
        if "d" in flags:
            if not argv:
                i = 0
            else:
                i = await bot.evalMath(argv, guild)
            x = rems.pop(i)
            update()
            return (
                "```ini\nSuccessfully removed ["
                + limStr(noHighlight(x["name"] + ": " + x["msg"]), 128) + "] from announcements list for [#"
                + noHighlight(channel) + "].```"
            )
        if not argv:
            if not len(rems):
                return (
                    "```ini\nNo announcements currently set for [#"
                    + noHighlight(channel) + "].```"
                )
            d = datetime.datetime.utcnow().timestamp()
            s = strIter(rems, key=lambda x: limStr(noHighlight(x["name"] + ": " + x["msg"]), 128) + " ➡️ " + sec2Time(x["t"] - d))
            return (
                "Current announcements set for <#" + str(channel.id)
                + ">:```ini" + s + "```"
            )
        if len(rems) >= 32:
            raise OverflowError("Channel has reached the maximum of 32 announcements. Please remove one to add another.")
        while True:
            spl = None
            if "in" in argv:
                if " in " in argv:
                    spl = argv.split(" in ")
                elif argv.startswith("in "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " in ".join(spl[:-1])
                    t = await bot.evalTime(spl[-1], guild)
                    break
            if "at" in argv:
                if " at " in argv:
                    spl = argv.split(" at ")
                elif argv.startswith("at "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " at ".join(spl[:-1])
                    t = tparser.parse(spl[-1]).timestamp() - datetime.datetime.utcnow().timestamp()
                    break
            if "on" in argv:
                if " on " in argv:
                    spl = argv.split(" on ")
                elif argv.startswith("on "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " on ".join(spl[:-1])
                    t = tparser.parse(spl[-1]).timestamp() - datetime.datetime.utcnow().timestamp()
                    break
            msg = " ".join(args[:-1])
            t = await bot.evalTime(args[-1], guild)
            break
        msg = msg.strip(" ")
        if not msg:
            msg = "[SAMPLE ANNOUNCEMENT]"
        elif len(msg) > 512:
            raise OverflowError("Announcement message too long (" + str(len(msg)) + "> 512).")
        name = str(user)
        url = strURL(user.avatar_url)
        ts = datetime.datetime.utcnow().timestamp()
        rems.append(freeClass(
            name=name,
            url=url,
            msg=msg,
            t=t + ts,
            u=0
        ))
        bot.data.reminders[channel.id] = sort(rems, key=lambda x: x["t"])
        try:
            bot.database.reminders.keyed.remove((0, channel.id), key=lambda x: x[-1])
        except IndexError:
            pass
        bot.database.reminders.keyed.insort((bot.data.reminders[channel.id][0]["t"], channel.id), key=lambda x: x[0])
        update()
        emb = discord.Embed(description=msg)
        emb.set_author(name=name, url=url, icon_url=url)
        return {
            "content": ("```css\nSuccessfully set announcement for [#"
                + noHighlight(channel) + "] in [" + noHighlight(sec2Time(t)) + "]:```"
            ),
            "embed": emb,
        }


class UpdateReminders(Database):
    name = "reminders"
    no_delete = True

    def __init__(self, *args):
        super().__init__(*args)
        d = self.data
        self.keyed = hlist(sorted(((d[i][0]["t"], i) for i in d), key=lambda x: x[0]))

    async def __call__(self):
        t = datetime.datetime.utcnow().timestamp()
        while self.keyed:
            p = self.keyed[0]
            if t < p[0]:
                break
            print(self.keyed)
            self.keyed.popleft()
            u_id = p[1]
            temp = self.data[u_id]
            x = freeClass(temp.pop(0))
            if not temp:
                self.data.pop(u_id)
            else:
                z = temp[0]["t"]
                self.keyed.insort((z, u_id), key=lambda x: x[0])
            if x.u:
                ch = await self.bot.fetch_user(u_id)
            else:
                ch = await self.bot.fetch_channel(u_id)
            emb = discord.Embed(description=x.msg)
            emb.set_author(name=x.name, url=x.url, icon_url=x.url)
            try:
                await ch.send(embed=emb)
            except discord.Forbidden:
                pass
            self.update()


class UpdateMessageCount(Database):
    name = "counts"

    def getMessageLength(self, message):
        return len(message.system_content) + sum(len(e) for e in message.embeds) + sum(len(a.url) for a in message.attachments)

    def startCalculate(self, guild):
        self.data[guild.id] = {"counts": {}, "totals": {}}
        create_task(self.getUserMessageCount(guild))

    async def getUserMessages(self, user, guild):
        if self.scanned == -1:
            c_id = self.bot.client.user.id
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
    
    def getUserGlobalMessageCount(self, user):
        count = 0
        for g_id in self.data:
            try:
                c = self.data[g_id]["counts"]
                if user.id in c:
                    count += c[user.id]
            except TypeError:
                pass
        return count

    async def getUserAverage(self, user, guild):
        if self.scanned == -1:
            c_id = self.bot.client.user.id
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
        if guild is None or hasattr(guild, "ghost"):
            return "Invalid server."
        if self.scanned == -1:
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
            self.data["guild.id"] = {"counts": {}, "totals": {}, 0: True}
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
            histories = histories.remove([])
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
                i += 1
        addDict(self.data[guild.id], {"counts": data, "totals": avgs, 0: True})
        self.update()
        print(guild)
        print(self.data[guild.id])

    def __init__(self, *args):
        self.scanned = False
        super().__init__(*args)

    async def __call__(self):
        if self.scanned:
            return
        self.scanned = True
        year = datetime.timedelta(seconds=31556925.216)
        guilds = self.bot.client.guilds
        i = 1
        for guild in sorted(guilds, key=lambda g: g.member_count, reverse=True):
            if guild.id not in self.data:
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
                self.update()
            else:
                self.startCalculate(guild)


class UpdatePrefix(Database):
    name = "prefixes"

    async def __call__(self):
        for g in tuple(self.data):
            if self.data[g] == "~":
                self.data.pop(g)


class UpdateEnabled(Database):
    name = "enabled"


class updateUsers(Database):
    name = "users"
    suspected = "users.json"
    user = True
    bcheck = eval(bytes(x ^ 137 for x in hex2Bytes(
        "E5 E8 E4 EB ED E8 A9 FA A5 A9 FD B3 A9 E8 E7 F0 A1 A1 AB F0 E6 FC A9 E6"
        + "E2 AB A9 E0 E7 A9 FD A5 A9 AB EF FC EA E2 AB A9 E0 E7 A9 FD A0 A0"
    )), {}, {})

    async def _seen_(self, user, delay, **void):
        addDict(self.data, {user.id: {"last_seen": 0}})
        self.data[user.id]["last_seen"] = datetime.datetime.utcnow().timestamp() + delay

    async def _command_(self, user, command, **void):
        addDict(self.data, {user.id: {"commands": 1}})
        self.update()

    async def _nocommand_(self, text, message, **void):
        if not message.mentions:
            name = self.__dict__.setdefault("name", reconstitute(self.bot.client.user.name).lower())
            if name not in text:
                return
        else:
            ids = (u.id for u in message.mentions)
            if self.bot.client.user.id not in ids:
                return
        if self.bcheck(text):
            await message.channel.send("😢")