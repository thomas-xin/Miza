try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


with open("auth.json") as f:
    auth = ast.literal_eval(f.read())
try:
    discord_id = auth["discord_id"]
    if not discord_id:
        raise
except:
    discord_id = None
    print("WARNING: discord_id not found. Unable to automatically generate bot invites.")


# Default and standard command categories to enable.
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
        v = "v" in flags
        emb = discord.Embed(colour=randColour())
        emb.set_author(name="❓ Help ❓")
        found = {}
        # Get help on categories, then commands
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
            # Display list of found commands in an embed
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
            # Display main help page in an embed
            emb.description = (
                "Please enter a command category to display usable commands,\nor see "
                + "[Commands](https://github.com/thomas-xin/Miza/wiki/Commands) for full command list."
            )
            if bot.categories:
                s = "```ini\n" + " ".join((sbHighlight(c) for c in standard_commands)) + "```"
                emb.add_field(name="Command category list", value=s)
        return dict(embed=emb), 1


class Hello(Command):
    name = ["Hi", "👋", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
    min_level = 0
    description = "Sends a waving emoji. Useful for checking whether the bot is online."
    
    async def __call__(self, **void):
        # yay
        return "👋"


class Perms(Command):
    server_only = True
    name = ["ChangePerms", "Perm", "ChangePerm", "Permissions"]
    description = "Shows or changes a user's permission level."
    usage = "<0:user{self}> <1:level[]> <hide(?h)>"
    flags = "fh"

    async def __call__(self, bot, args, user, perm, guild, flags, **void):
        # Get target user from first argument
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
                    t_user = await bot.fetch_member_ex(u_id, guild)
                except LookupError:
                    t_user = guild.get_role(u_id)
                    if t_user is None:
                        raise LookupError("No results for " + str(u_id))
        print(t_user)
        t_perm = roundMin(bot.getPerms(t_user.id, guild))
        # If permission level is given, attempt to change permission level, otherwise get current permission level
        if len(args) > 1:
            name = str(t_user)
            orig = t_perm
            expr = " ".join(args[1:])
            num = await bot.evalMath(expr, user, orig)
            c_perm = roundMin(num)
            if t_perm is nan or isnan(c_perm):
                m_perm = nan
            else:
                # Required permission to change is absolute level + 1, with a minimum of 3
                m_perm = max(abs(t_perm), abs(c_perm), 2) + 1
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
                raise self.permError(perm, m_perm, reason)
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

    async def __call__(self, argv, args, flags, user, channel, perm, **void):
        update = self.data.enabled.update
        bot = self.bot
        enabled = bot.data.enabled
        # Flags to change enabled commands list
        if "a" in flags or "e" in flags or "d" in flags:
            req = 3
            if perm < req:
                reason = (
                    "to change command list for "
                    + channel.name
                )
                raise self.permError(perm, req, reason)
        if not args:
            if "l" in flags:
                return (
                    "```css\nStandard command categories:\n"
                    + str(standard_commands) + "```"
                )
            if "e" in flags or "a" in flags:
                categories = list(standard_commands)
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
            temp = enabled.get(channel.id, default_commands)
            if not temp:
                return (
                    "```ini\nNo currently enabled commands in [#" + str(channel) + "].```"
                )
            return (
                "Currently enabled command categories in <#" + str(channel.id)
                + ">:\n```ini\n"
                + strIter(temp) + "```"
            )
        else:
            if "e" not in flags and "a" not in flags and "d" not in flags:
                catg = argv.lower()
                if not catg in bot.categories:
                    raise LookupError("Unknown command category " + argv + ".")
                return (
                    "```css\nCommand category [" + noHighlight(catg)
                    + "] is currently" + " not" * (catg not in enabled)
                    + "] enabled in [" + noHighlight(channel.name) + "].```"
                )
            args = [i.lower() for i in args]
            for catg in args:
                if not catg in bot.categories:
                    raise LookupError("Unknown command category " + catg + ".")
            curr = setDict(enabled, channel.id, list(default_commands))
            for catg in args:
                if "d" not in flags:
                    if catg not in enabled:
                        curr.append(catg)
                        update()
                if "d" in flags:
                    if catg in enabled:
                        curr.remove(catg)
                        update()
            if curr == default_commands:
                enabled.pop(channel.id)
                update()
            if "h" in flags:
                return
            if "e" in flags:
                return (
                    "```css\nEnabled command category [" + noHighlight(", ".join(args))
                    + "] in [" + noHighlight(channel.name) + "].```"
                )
            return (
                "```css\nDisabled command category [" + noHighlight(", ".join(args))
                + "] in [" + noHighlight(channel.name) + "].```"
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
            raise self.permError(perm, req, reason)
        prefix = argv
        # Backslash is not allowed, it is used to escape commands normally
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
    _timeout_ = 8
    name = ["For", "Rep", "While"]
    min_level = 1
    min_display = "1+"
    description = "Loops a command."
    usage = "<0:iterations> <1:command>"
    rate_limit = 3

    async def __call__(self, args, argv, message, channel, callback, bot, perm, user, guild, **void):
        if not args:
            # Ah yes, I made this error specifically for people trying to use this command to loop songs 🙃
            raise ArgumentError("Please input loop iterations and target command. For looping songs in voice, consider using the aliases LoopQueue and Repeat under the AudioSettings command.")
        num = await bot.evalMath(args[0], user)
        iters = round(num)
        # Bot owner bypasses restrictions
        if not isnan(perm):
            if iters > 32 and not bot.isTrusted(guild.id):
                raise PermissionError("Must be in a trusted server to execute loop of more than 32 iterations.")
            # Required level is 1/3 the amount of loops required, rounded up
            scale = 3
            limit = perm * scale
            if iters > limit:
                reason = (
                    "to execute loop of " + str(iters)
                    + " iterations"
                )
                raise self.permError(perm, ceil(iters / scale), reason)
            elif iters > 256:
                raise PermissionError("Must be owner to execute loop of more than 256 iterations.")
        func = func2 = " ".join(args[1:])
        if func:
            while func[0] == " ":
                func = func[1:]
        if not isnan(perm):
            # Detects when an attempt is made to loop the loop command
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
            t = utc()
            # Calls processMessage with the argument containing the looped command.
            delay = await callback(message, func, cb_argv=func2, loop=loop)
            # Must abide by command rate limit rules
            delay = delay + t - utc()
            if delay > 0:
                await asyncio.sleep(delay)


class Avatar(Command):
    name = ["PFP", "Icon"]
    min_level = 0
    description = "Sends a link to the avatar of a user or server."
    usage = "<user>"

    async def getGuildData(self, g):
        # Gets icon display of a server and returns as an embed.
        url = strURL(g.icon_url)
        name = g.name
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        return dict(embed=emb)

    def getMimicData(self, p):
        # Gets icon display of a mimic and returns as an embed.
        url = strURL(p.url)
        name = p.name
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        return dict(embed=emb)

    async def __call__(self, argv, guild, bot, client, user, **void):
        g, guild = guild, None
        # This is a mess 🙃
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
                    u = await bot.fetch_member_ex(u_id, g)
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
                                        u = None
                                        if g.id in bot.data.counts:
                                            if u_id in bot.data.counts[g.id]["counts"]:
                                                u = bot.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("No results for " + argv)
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
        url = bestURL(u)
        emb = discord.Embed(colour=randColour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = "[" + discord.utils.escape_markdown(name) + "](" + url + ")"
        return dict(embed=emb)


class Info(Command):
    name = ["UserInfo", "ServerInfo", "WhoIs", "Profile"]
    min_level = 0
    description = "Shows information about the target user or server."
    usage = "<user> <verbose(?v)>"
    flags = "v"
    rate_limit = 1

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
                # Top users by message counts
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
        return dict(embed=emb)

    def getMimicData(self, p, flags={}):
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
            age = (utc_dt() - ctime).total_seconds() / TIMEUNITS["year"]
            emb.add_field(name="Birthday", value=str(ctime), inline=1)
            emb.add_field(name="Age", value=str(roundMin(round(age, 1))), inline=1)
        if pcnt:
            emb.add_field(name="Post count", value=str(pcnt), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        return dict(embed=emb)

    async def __call__(self, argv, name, guild, channel, bot, client, user, flags, **void):
        member = True
        g, guild = guild, None
        # This is a mess 🙃
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
                    u = await bot.fetch_member_ex(u_id, g)
                except:
                    try:
                        u = await bot.fetch_member(u_id, g, find_others=True)
                        member = False
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
                                        u = None
                                        if g.id in bot.data.counts:
                                            if u_id in bot.data.counts[g.id]["counts"]:
                                                u = bot.ghostUser()
                                                u.id = u_id
                                        if u is None:
                                            raise LookupError("No results for " + argv)
                                    try:
                                        guild = channel.guild
                                    except NameError:
                                        pass
                                    except (AttributeError, KeyError):
                                        guild = None
                                        u = channel.recipient
                            if guild is not None:
                                return await self.getGuildData(guild, flags)                        
        elif not name.startswith("server"):
            u = user
        else:
            return await self.getGuildData(g, flags)
        guild = g
        name = str(u)
        dname = u.display_name * (u.display_name != u.name) if member else None
        url = bestURL(u)
        try:
            is_sys = u.system
        except (AttributeError, KeyError):
            is_sys = False
        is_bot = u.bot
        is_self = u.id == client.user.id
        is_self_owner = u.id in bot.owners
        is_guild_owner = u.id == guild.owner_id
        if member:
            joined = getattr(u, "joined_at", None)
        else:
            joined = None
        created = u.created_at
        activity = "\n".join(strActivity(i) for i in getattr(u, "activities", []))
        status = None
        if hasattr(u, "status"):
            s = u.status
            if s == discord.Status.online:
                status = "Online 🟢"
            elif s == discord.Status.idle:
                status = "Idle 🟡"
            elif s == discord.Status.dnd:
                status = "DND 🔴"
            elif s in (discord.Status.offline, discord.Status.invisible):
                status = "Offline ⚫"
        if member:
            role = ", ".join("<@&" + str(i.id) + ">" for i in reversed(getattr(u, "roles", ())) if not i.is_default())
        else:
            role = None
        coms = seen = msgs = avgs = gmsg = old = 0
        fav = None
        pos = None
        if "v" in flags:
            try:
                if is_self:
                    # Count total commands used by all users
                    c = {}
                    for i, v in enumerate(tuple(bot.data.users.values()), 1):
                        try:
                            addDict(c, v["commands"])
                        except KeyError:
                            pass
                        if not i & 8191:
                            await asyncio.sleep(0.2)
                else:
                    c = bot.data.users[u.id]["commands"]
                coms = iterSum(c)
                if type(c) is dict:
                    try:
                        comfreq = deque(sort(c, reverse=True).keys())
                        while fav is None:
                            fav = comfreq.popleft()
                    except IndexError:
                        pass
            except LookupError:
                pass
            try:
                ts = utc()
                ls = bot.data.users[u.id]["last_seen"]
                la = bot.data.users[u.id].get("last_action")
                if type(ls) is str:
                    seen = ls
                else:
                    seen = sec2Time(max(0, ts - ls)) + " ago"
                if la:
                    seen = la + ", " + seen
            except LookupError:
                pass
            try:
                gmsg = bot.database.counts.getUserGlobalMessageCount(u)
                msgs = await bot.database.counts.getUserMessages(u, guild)
                avgs = await bot.database.counts.getUserAverage(u, guild)
                old = bot.data.counts.get(guild.id, {})["oldest"].get(u.id)
                if old:
                    old = snowflake_time(old)
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
        if joined:
            emb.add_field(name="Join time", value=str(joined), inline=1)
        if old:
            emb.add_field(name="Oldest post time", value=str(old), inline=1)
        if status:
            emb.add_field(name="Status", value=str(status), inline=1)
        if seen:
            emb.add_field(name="Last seen", value=str(seen), inline=1)
        if coms:
            emb.add_field(name="Commands used", value=str(coms), inline=1)
        if fav:
            emb.add_field(name="Favourite command", value=str(fav), inline=1)
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
        # Double verbose option sends an activity graph
        if flags.get("v", 0) > 1:
            fut = create_task(channel.trigger_typing())
            fut2 = create_task(channel.send(embed=emb))
            data = await create_future(bot.database.users.fetch_events, user.id, interval=3600)
            resp = await bot.solveMath("eval(\"plt_special(" + repr(data).replace('"', "'") + ", user='" + str(user) + "')\")", guild, 0, 1, authorize=True)
            fn = resp["file"]
            f = discord.File(fn)
            await fut2
            await fut
            return dict(file=f, filename=fn, best=True)
        return dict(embed=emb)


class Activity(Command):
    name = ["Recent", "Log"]
    min_level = 0
    description = "Shows recent Discord activity for the targeted user."
    usage = "<user> <verbose(?v)>"
    flags="v"
    rate_limit = 1
    typing = True

    async def __call__(self, guild, user, argv, flags, channel, bot, **void):
        if argv:
            u_id = verifyID(argv)
            try:
                user = await bot.fetch_user(u_id)
            except (TypeError, discord.NotFound):
                user = await bot.fetch_member_ex(u_id, guild)
        data = await create_future(bot.database.users.fetch_events, user.id, interval=max(900, 3600 >> flags.get("v", 0)))
        fut = create_task(channel.trigger_typing())
        try:
            resp = await bot.solveMath("eval(\"plt_special(" + repr(data).replace('"', "'") + ", user='" + str(user) + "')\")", guild, 0, 1, authorize=True)
        except:
            await fut
            raise
        fn = resp["file"]
        f = discord.File(fn)
        await fut
        return dict(file=f, filename=fn, best=True)


class Status(Command):
    name = ["State", "Ping"]
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
        cache = await create_future(os.listdir, "cache/")
        return (
            "```ini"
            + "\nLoaded users: " + sbHighlight(max(len(client.users), len(bot.cache.users)))
            + ", Loaded servers: " + sbHighlight(bot.guilds)
            + ", Loaded shards: " + sbHighlight(shards)
            
            + ".\nActive processes: " + sbHighlight(active[0])
            + ", Active threads: " + sbHighlight(active[1])
            + ", Active coroutines: " + sbHighlight(active[2])
            
            + ".\nConnected voice channels: " + sbHighlight(len(client.voice_clients))
            
            + ".\nCached files: " + sbHighlight(len(cache))
            
            + ".\nCode size: " + sbHighlight(size[0]) + " bytes"
            + ", " + sbHighlight(size[1]) + " lines"

            + ".\nSystem time: " + sbHighlight(datetime.datetime.now())
            + ".\nPing latency: " + sbHighlight(latency)
            + ".\nPublic IP address: " + sbHighlight(bot.ip)
            
            + ".\nCPU usage: " + sbHighlight(round(stats[0], 3)) + "%"
            + ", RAM usage: " + sbHighlight(round(stats[1] / 1048576, 3)) + " MB"
            + ", Disk usage: " + sbHighlight(round(stats[2] / 1048576, 3)) + " MB"
            + ".```"
        )


class Invite(Command):
    name = ["OAuth", "InviteBot", "InviteLink"]
    min_level = 0
    description = "Sends a link to ⟨MIZA⟩'s homepage and invite code."
    
    async def __call__(self, **void):
        if discord_id is None:
            raise FileNotFoundError("Unable to locate bot's Client ID.")
        emb = discord.Embed(colour=randColour())
        user = self.bot.client.user
        url = bestURL(user)
        emb.set_author(name=str(user), icon_url=url, url=url)
        emb.description = "[Homepage](" + self.bot.website + ")\n[Invite](https://discordapp.com/oauth2/authorize?permissions=8&client_id=" + str(discord_id) + "&scope=bot)"
        return dict(embed=emb)


class Reminder(Command):
    name = ["Announcement", "Announcements", "Announce", "RemindMe", "Reminders", "Remind"]
    min_level = 0
    description = "Sets a reminder for a certain date and time."
    usage = "<1:message> <0:time> <disable(?d)>"
    flags = "aed"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 1 / 3
    keywords = ["on", "at", "in", "when", "event"]
    keydict = {re.compile("(^|[^a-z0-9])" + i[::-1] + "([^a-z0-9]|$)", re.I): None for i in keywords}

    async def __call__(self, argv, name, message, flags, bot, user, guild, perm, **void):
        msg = message.content
        argv2 = argv
        argv = msg[msg.lower().index(name) + len(name):].strip(" ").strip("\n")
        try:
            args = shlex.split(argv)
        except ValueError:
            args = argv.split(" ")
        if "announce" in name:
            req = 2
            if req > perm:
                raise self.permError(perm, req, "for command " + name)
            sendable = message.channel
            word = "announcements"
        else:
            sendable = user
            word = "reminders"
        rems = bot.data.reminders.get(sendable.id, [])
        update = bot.database.reminders.update
        if "d" in flags:
            if not len(rems):
                return (
                    "```ini\nNo " + word + " currently set for ["
                    + noHighlight(sendable) + "].```"
                )
            if not argv:
                i = 0
            else:
                i = await bot.evalMath(argv2, user)
            i %= len(rems)
            x = rems.pop(i)
            if i == 0:
                try:
                    bot.database.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
                except IndexError:
                    pass
                if rems:
                    bot.database.reminders.listed.insort((rems[0]["t"], sendable.id), key=lambda x: x[0])
            update()
            return (
                "```ini\nSuccessfully removed ["
                + limStr(noHighlight(x["msg"]), 128) + "] from " + word + " list for ["
                + noHighlight(sendable) + "].```"
            )
        if not argv:
            # Set callback message for scrollable list
            return (
                "```" + "\n" * ("z" in flags) + "callback-main-reminder-"
                + str(user.id) + "_0_" + str(sendable.id)
                + "-\nLoading Reminder database...```"
            )
        if len(rems) >= 64:
            raise OverflowError("You have reached the maximum of 64 " + word + ". Please remove one to add another.")
        # This parser is so unnecessarily long for what it does...
        keyed = False
        while True:
            temp = argv.lower()
            if name == "remind" and temp.startswith("me "):
                argv = argv[3:]
                temp = argv.lower()
            if temp.startswith("to "):
                argv = argv[3:]
                temp = argv.lower()
            elif temp.startswith("that "):
                argv = argv[5:]
                temp = argv.lower()
            spl = None
            keywords = dict(self.keydict)
            # Reversed regex search
            temp2 = temp[::-1]
            for k in tuple(keywords):
                try:
                    i = re.search(k, temp2).end()
                    if not i:
                        raise ValueError
                except (ValueError, AttributeError):
                    keywords.pop(k)
                else:
                    keywords[k] = i
            # Sort found keywords by position
            indices = sorted(keywords, key=lambda k: keywords[k])
            if indices:
                foundkey = {self.keywords[tuple(self.keydict).index(indices[0])]: True}
            else:
                foundkey = cdict(get=lambda *void: None)
            if foundkey.get("event"):
                if " event " in argv:
                    spl = argv.split(" event ")
                elif temp.startswith("event "):
                    spl = [argv[6:]]
                    msg = ""
                if spl is not None:
                    msg = " event ".join(spl[:-1])
                    t = verifyID(spl[-1])
                    keyed = True
                    break
            if foundkey.get("when"):
                if temp.endswith("is online"):
                    argv = argv[:-9]
                if " when " in argv:
                    spl = argv.split(" when ")
                elif temp.startswith("when "):
                    spl = [argv[5:]]
                    msg = ""
                if spl is not None:
                    msg = " when ".join(spl[:-1])
                    t = verifyID(spl[-1])
                    keyed = True
                    break
            if foundkey.get("in"):
                if " in " in argv:
                    spl = argv.split(" in ")
                elif temp.startswith("in "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " in ".join(spl[:-1])
                    t = await bot.evalTime(spl[-1], guild)
                    break
            if foundkey.get("at"):
                if " at " in argv:
                    spl = argv.split(" at ")
                elif temp.startswith("at "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " at ".join(spl[:-1])
                    t = utc_ts(tzparse(spl[-1])) - utc()
                    break
            if foundkey.get("on"):
                if " on " in argv:
                    spl = argv.split(" on ")
                elif temp.startswith("on "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " on ".join(spl[:-1])
                    t = utc_ts(tzparse(spl[-1])) - utc()
                    break
            msg = " ".join(args[:-1])
            t = await bot.evalTime(args[-1], guild)
            break
        if keyed:
            try:
                u = await bot.fetch_user(t)
            except (TypeError, discord.NotFound):
                member = await bot.fetch_member_ex(t, guild)
                t = member.id
        msg = msg.strip(" ")
        if not msg:
            if "announce" in name:
                msg = "[SAMPLE ANNOUNCEMENT]"
            else:
                msg = "[SAMPLE REMINDER]"
            msg = "```asciidoc\n" + msg + "```"
        elif len(msg) > 1024:
            raise OverflowError("Input message too long (" + str(len(msg)) + "> 1024).")
        username = str(user)
        url = bestURL(user)
        ts = utc()
        if keyed:
            # Schedule for an event from a user
            rems.append(cdict(
                user=user.id,
                msg=msg,
                u_id=t,
                t=inf,
            ))
            s = "$" + str(t)
            seq = setDict(bot.data.reminders, s, deque())
            seq.append(sendable.id)
        else:
            # Schedule for an event at a certain time
            rems.append(cdict(
                user=user.id,
                msg=msg,
                t=t + ts,
            ))
        # Sort list of reminders
        bot.data.reminders[sendable.id] = sort(rems, key=lambda x: x["t"])
        try:
            # Remove existing schedule
            bot.database.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
        except IndexError:
            pass
        # Insert back into bot schedule
        bot.database.reminders.listed.insort((bot.data.reminders[sendable.id][0]["t"], sendable.id), key=lambda x: x[0])
        update()
        emb = discord.Embed(description=msg)
        emb.set_author(name=username, url=url, icon_url=url)
        if "announce" in name:
            out = "```css\nSuccessfully set announcement for " + sbHighlight(sendable)
        else:
            out = "```css\nSuccessfully set reminder for " + sbHighlight(sendable)
        if keyed:
            out += " upon next event from " + sbHighlight("<@" + str(t) + ">")
        else:
            out += " in " + sbHighlight(sec2Time(t))
        out += ":```"
        return dict(content=out, embed=emb)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos, s_id = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        rems = bot.data.reminders.get(s_id, [])
        sendable = await bot.fetch_sendable(s_id)
        page = 16
        last = max(0, len(rems) - page)
        if reaction is not None:
            i = self.directions.index(reaction)
            if i == 0:
                new = 0
            elif i == 1:
                new = max(0, pos - page)
            elif i == 2:
                new = min(last, pos + page)
            elif i == 3:
                new = last
            else:
                new = pos
            pos = new
        content = message.content
        if not content:
            content = message.embeds[0].description
        i = content.index("callback")
        content = content[:i] + (
            "callback-image-img-"
            + str(u_id) + "_" + str(pos) + "_" + str(s_id)
            + "-\n"
        )
        if not rems:
            content += "Schedule for " + str(sendable).replace("`", "") + " is currently empty.```"
            msg = ""
        else:
            t = utc()
            content += str(len(rems)) + " messages currently scheduled for " + str(sendable).replace("`", "") + ":```"
            msg = strIter(
                rems[pos:pos + page],
                key=lambda x: limStr(bot.get_user(x.get("user", -1), replace=True).mention + ": `" + noHighlight(x["msg"]), 96) + "` ➡️ " + ("<@" + str(x["u_id"]) + ">" if "u_id" in x else sec2Time(x["t"] - t)),
                left="`[",
                right="]`",
            )
        emb = discord.Embed(
            description=content + msg,
            colour=randColour(),
        )
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        more = len(rems) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


# This database is such a hassle to manage, it has to be able to persist between bot restarts, and has to be able to update with O(1) time complexity when idle
class UpdateReminders(Database):
    name = "reminders"
    no_delete = True
    rate_limit = 1

    def __load__(self):
        d = self.data
        # This exists so that checking next scheduled item is O(1)
        self.listed = hlist(sorted(((d[i][0]["t"], i) for i in d if type(i) is not str), key=lambda x: x[0]))

    # Fast call: runs 24 times per second
    async def _call_(self):
        t = utc()
        while self.listed:
            p = self.listed[0]
            # Only check first item in the schedule
            if t < p[0]:
                break
            # Grab expired item
            self.listed.popleft()
            u_id = p[1]
            temp = self.data[u_id]
            if not temp:
                self.data.pop(u_id)
                continue
            # Check next item in schedule
            x = temp[0]
            if t < x["t"]:
                # Insert back into schedule if not expired
                self.listed.insort((x["t"], u_id), key=lambda x: x[0])
                print(self.listed)
                continue
            # Grab target from database
            x = cdict(temp.pop(0))
            if not temp:
                self.data.pop(u_id)
            else:
                # Insert next listed item into schedule
                self.listed.insort((temp[0]["t"], u_id), key=lambda x: x[0])
            # print(self.listed)
            # Send reminder to target user/channel
            ch = await self.bot.fetch_sendable(u_id)
            emb = discord.Embed(description=x.msg)
            try:
                u = self.bot.get_user(x["user"], replace=True)
            except KeyError:
                u = x
            emb.set_author(name=u.name, url=bestURL(u), icon_url=bestURL(u))
            self.bot.embedSender(ch, emb)
            self.update()

    # Seen event: runs when users perform discord actions
    async def _seen_(self, user, **void):
        s = "$" + str(user.id)
        if s in self.data:
            assigned = self.data[s]
            # Ignore user events without assigned triggers
            if not assigned:
                self.data.pop(s)
                return
            try:
                for u_id in assigned:
                    # Send reminder to all targeted users/channels
                    ch = await self.bot.fetch_sendable(u_id)
                    rems = setDict(self.data, u_id, [])
                    pops = {}
                    for i, x in enumerate(reversed(rems), 1):
                        if x.get("u_id", None) == user.id:
                            emb = discord.Embed(description=x["msg"])
                            try:
                                u = self.bot.get_user(x["user"], replace=True)
                            except KeyError:
                                u = cdict(x)
                            emb.set_author(name=u.name, url=bestURL(u), icon_url=bestURL(u))
                            self.bot.embedSender(ch, emb)
                            pops[len(rems) - i] = True
                        elif isValid(x["t"]):
                            break
                    it = [rems[i] for i in range(len(rems)) if i not in pops]
                    rems.clear()
                    rems.extend(it)
                    if not rems:
                        self.data.pop(u_id)
            except:
                print(traceback.format_exc())
            try:
                self.data.pop(s)
                self.update()
            except KeyError:
                pass


# This database has caused so many rate limit issues
class UpdateMessageCount(Database):
    name = "counts"

    def getMessageLength(self, message):
        return len(message.system_content) + sum(len(e) for e in message.embeds) + sum(len(a.url) for a in message.attachments)

    def startCalculate(self, guild):
        self.data[guild.id] = {"counts": {}, "totals": {}, "oldest": {}}
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

    # What are the rate limits for the message history calls?
    async def getChannelHistory(self, channel, limit=None, callback=None):
        # Semaphore of 32, idk if this is enough
        while self.req > 32:
            await asyncio.sleep(4)
        self.req += 1
        try:
            messages = []
            # 16 attempts to download channel
            for i in range(16):
                history = channel.history(limit=limit, oldest_first=(limit is None))
                try:
                    messages = await history.flatten()
                except discord.Forbidden:
                    # Don't attempt any more if the response was forbidden
                    break
                except discord.HTTPException as ex:
                    # Wait longer between attempts if the error was a rate limit
                    if "429" in str(ex):
                        await asyncio.sleep(20 * (i ** 2 + 1))
                    else:
                        await asyncio.sleep(10)
                except:
                    print(traceback.format_exc())
                    await asyncio.sleep(5)
                else:
                    break
        except:
            self.req -= 1
            raise
        self.req -= 1
        if callback:
            fut = callback(channel=channel, messages=messages)
            if awaitable(fut):
                return await fut
            return fut
        return messages

    async def getGuildHistory(self, guild, limit=None, callback=None):
        lim = None if limit is None else ceil(limit / min(128, len(guild.text_channels)))
        output = cdict({channel.id: create_task(self.getChannelHistory(channel, limit=lim, callback=callback)) for channel in guild.text_channels})
        for k, v in output.items():
            output[k] = await v
        return output

    async def getUserMessageCount(self, guild):
        year = datetime.timedelta(seconds=31556925.216)
        oneyear = utc_dt() - guild.created_at < year
        if guild.member_count > 512 and not oneyear:
            addDict(self.data[guild.id], {0: True})
            return
        print(guild)
        data = {}
        avgs = {}
        oldest = self.data[guild.id]["oldest"]
        histories = await self.getGuildHistory(guild)
        for messages in histories.values():
            for i, message in enumerate(messages, 1):
                u = message.author.id
                orig_id = oldest.get(u)
                if not orig_id or message.id < orig_id:
                    oldest[u] = message.id
                length = self.getMessageLength(message)
                try:
                    data[u] += 1
                    avgs[u] += length
                except KeyError:
                    data[u] = 1
                    avgs[u] = length
                if not i & 8191:
                    await asyncio.sleep(0.5)
        addDict(self.data[guild.id], {"counts": data, "totals": avgs, 0: True})
        self.update()
        print(guild)
        print(self.data[guild.id])

    def __load__(self):
        self.scanned = False
        self.req = 0

    async def __call__(self):
        if self.scanned:
            return
        self.scanned = True
        year = datetime.timedelta(seconds=31556925.216)
        guilds = self.bot.client.guilds
        i = 1
        for guild in sorted(guilds, key=lambda g: g.member_count, reverse=True):
            if guild.id not in self.data:
                oneyear = utc_dt() - guild.created_at < year
                if guild.member_count < 512 or oneyear:
                    self.startCalculate(guild)
                    while self.req > 32:
                        await asyncio.sleep(5)
                    if not i & 7:
                        await asyncio.sleep(60)
                    i += 1
        self.scanned = -1

    # Add new messages to the post count database
    def _send_(self, message, **void):
        if self.scanned == -1:
            user = message.author
            guild = message.guild
            if guild.id in self.data:
                d = self.data[guild.id]
                if type(d) is str:
                    return
                if user.id not in setDict(d, "oldest", {}):
                    d["oldest"][user.id] = message.id
                count = d["counts"].get(user.id, 0) + 1
                total = d["totals"].get(user.id, 0) + self.getMessageLength(message)
                d["totals"][user.id] = total
                d["counts"][user.id] = count
                self.update()
            else:
                self.startCalculate(guild)


class UpdatePrefix(Database):
    name = "prefixes"

    # This is O(n) so it's on the lazy loop
    async def __call__(self):
        for g in tuple(self.data):
            if self.data[g] == self.bot.prefix:
                self.data.pop(g)
                self.update()


class UpdateEnabled(Database):
    name = "enabled"


EMPTY = {}

# This database takes up a lot of space, storing so many events from users
class UpdateUsers(Database):
    name = "users"
    suspected = "users.json"
    user = True
    hours = 168
    interval = 900
    scale = 3600 // interval

    def clear_events(self, data, minimum):
        for hour in tuple(data):
            if hour > minimum:
                return
            data.pop(hour)

    def send_event(self, u_id, event, count=1):
        # print(self.bot.cache.users.get(u_id), event, count)
        data = setDict(setDict(self.data, u_id, {}), "recent", {})
        hour = roundMin(round(utc() // self.interval) / self.scale)
        if data:
            self.clear_events(data, hour - self.hours)
        try:
            data[hour][event] += count
        except KeyError:
            try:
                data[hour][event] = count
            except KeyError:
                data[hour] = {event: count}

    fetch_events = lambda self, u_id, interval=3600: {i: self.get_events(u_id, interval=interval, event=i) for i in ("message", "typing", "command", "reaction", "misc")}

    # Get all events of a certain type from a certain user, with specified intervals.
    def get_events(self, u_id, interval=3600, event=None):
        data = self.data.get(u_id, EMPTY).get("recent")
        if not data:
            return list(repeat(0, round(self.hours / self.interval * interval)))
        hour = roundMin(round(utc() // self.interval) / self.scale)
        # print(hour)
        self.clear_events(data, hour - self.hours)
        start = hour - self.hours
        if event is None:
            out = [sum(data.get(i / self.scale + start, EMPTY).values()) for i in range(self.hours * self.scale)]
        else:
            out = [data.get(i / self.scale + start, EMPTY).get(event, 0) for i in range(self.hours * self.scale)]
        if interval != self.interval:
            factor = ceil(interval / self.interval)
            out = [sum(out[i:i + factor]) for i in range(0, len(out), factor)]
        return out

    # User seen, add event to activity database
    def _seen_(self, user, delay, event, count=1, raw=None, **void):
        self.send_event(user.id, event, count=count)
        addDict(self.data, {user.id: {"last_seen": 0}})
        self.data[user.id]["last_seen"] = utc() + delay
        self.data[user.id]["last_action"] = raw

    # User executed command, add to activity database
    def _command_(self, user, command, **void):
        self.send_event(user.id, "command")
        addDict(self.data, {user.id: {"commands": {str(command): 1}}})
        self.update()