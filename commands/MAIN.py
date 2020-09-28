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
default_commands = frozenset(("main", "string", "admin"))
standard_commands = default_commands.union(("voice", "image", "fun"))


class Help(Command):
    name = ["?"]
    description = "Shows a list of usable commands, or gives a detailed description of a command."
    usage = "<command{all}> <category{all}> <verbose(?v)>"
    flags = "v"

    def __call__(self, args, user, channel, guild, flags, perm, **void):
        bot = self.bot
        enabled = bot.data.enabled
        g_id = guild.id
        prefix = bot.get_prefix(g_id)
        v = "v" in flags
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(name="❓ Help ❓")
        found = {}
        # Get help on categories, then commands
        for a in args:
            a = a.casefold()
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
                    found[com.__name__] = alist((com,))
        if found:
            # Display list of found commands in an embed
            i = 0
            for k in found:
                if i >= 25:
                    break
                coms = found[k]
                for com in coms:
                    a = ", ".join(n.strip("_") for n in com.name)
                    if not a:
                        a = "[none]"
                    s = f"```ini\n[Aliases] {a}"
                    s += f"\n[Effect] {com.description.replace('⟨MIZA⟩', bot.user.name)}"
                    if v or len(found) <= 1:
                        s += f"\n[Usage] {prefix}{com.__name__} {com.usage}\n[Level] {com.min_display}"
                    s += "```"
                    emb.add_field(
                        name=prefix + com.__name__.strip("_"),
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
                s = bold(ini_md(' '.join((sqr_md(c) for c in standard_commands if c in bot.categories))))
                emb.add_field(name="Command category list", value=s)
        emb = emb.to_dict()
        emb["reacts"] = "❎"
        bot.send_embeds(channel, embed=emb)


class Hello(Command):
    name = ["Hi", "👋", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
    min_level = 0
    description = "Sends a waving emoji. Useful for checking whether the bot is online."
    
    def __call__(self, **void):
        return "👋"


class Perms(Command):
    server_only = True
    name = ["ChangePerms", "Perm", "ChangePerm", "Permissions"]
    description = "Shows or changes a user's permission level."
    usage = "<0:*users{self}> <1:level[]> <hide(?h)>"
    flags = "fh"
    multi = True

    async def __call__(self, bot, args, argl, user, perm, channel, guild, flags, **void):
        # Get target user from first argument
        users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        for t_user in users:
            t_perm = round_min(bot.get_perms(t_user.id, guild))
            # If permission level is given, attempt to change permission level, otherwise get current permission level
            if args:
                name = str(t_user)
                orig = t_perm
                expr = " ".join(args)
                num = await bot.eval_math(expr, user, orig)
                c_perm = round_min(num)
                if t_perm is nan or isnan(c_perm):
                    m_perm = nan
                else:
                    # Required permission to change is absolute level + 1, with a minimum of 3
                    m_perm = max(abs(t_perm), abs(c_perm), 2) + 1
                if not perm < m_perm and not isnan(m_perm):
                    if not m_perm < inf and guild.owner_id != user.id and not isnan(perm):
                        raise PermissionError("Must be server owner to assign non-finite permission level.")
                    bot.set_perms(t_user.id, guild, c_perm)
                    if "h" in flags:
                        return
                    create_task(channel.send(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(t_perm)} to {sqr_md(c_perm)}.")))
                    continue
                reason = f"to change permissions for {name} in {guild} from {t_perm} to {c_perm}"
                raise self.perm_error(perm, m_perm, reason)
            create_task(channel.send(css_md(f"Current permissions for {sqr_md(t_user)} in {sqr_md(guild)}: {sqr_md(t_perm)}.")))


class EnabledCommands(Command):
    server_only = True
    name = ["EC", "Enable"]
    min_level = 0
    min_display = "0~3"
    description = "Shows, enables, or disables a command category in the current channel."
    usage = "<command{all}> <add(?e)> <remove(?d)> <list(?l)> <hide(?h)>"
    flags = "aedlh"

    def __call__(self, argv, args, flags, user, channel, perm, **void):
        update = self.data.enabled.update
        bot = self.bot
        enabled = bot.data.enabled
        # Flags to change enabled commands list
        if "a" in flags or "e" in flags or "d" in flags:
            req = 3
            if perm < req:
                reason = f"to change enabled command list for {channel}"
                raise self.perm_error(perm, req, reason)
        if not args:
            if "l" in flags:
                return css_md(f"Standard command categories:\n[{', '.join(standard_commands)}]")
            if "e" in flags or "a" in flags:
                categories = set(standard_commands)
                if channel.id in enabled:
                    enabled[channel.id] = categories.union(enabled[channel.id])
                else:
                    enabled[channel.id] = categories
                update()
                if "h" in flags:
                    return
                return css_md(f"Enabled all standard command categories in {sqr_md(channel)}.")
            if "d" in flags:
                enabled[channel.id] = set()
                update()
                if "h" in flags:
                    return
                return css_md(f"Disabled all standard command categories in {sqr_md(channel)}.")
            temp = enabled.get(channel.id, default_commands)
            if not temp:
                return ini_md(f"No currently enabled commands in {sqr_md(channel)}.")
            return f"Currently enabled command categories in {channel_mention(channel.id)}:\n{ini_md(iter2str(temp))}"
        if "e" not in flags and "a" not in flags and "d" not in flags:
            catg = argv.casefold()
            if not catg in bot.categories:
                raise LookupError(f"Unknown command category {argv}.")
            if catg in enabled:
                return css_md(f"Command category {sqr_md(catg)} is currently enabled in {sqr_md(channel)}.")
            return css_md(f"Command category {sqr_md(catg)} is currently disabled in {sqr_md(channel)}. Use ?e to enable.")
        args = [i.casefold() for i in args]
        for catg in args:
            if not catg in bot.categories:
                raise LookupError(f"Unknown command category {catg}.")
        curr = set_dict(enabled, channel.id, set(default_commands))
        for catg in args:
            if "d" not in flags:
                if catg not in curr:
                    if type(curr) is set:
                        curr.add(catg)
                    else:
                        curr.append(catg)
                    update()
            else:
                if catg in curr:
                    curr.remove(catg)
                    update()
        check = curr if type(curr) is set else frozenset(curr)
        if check == default_commands:
            enabled.pop(channel.id)
            update()
        if "h" in flags:
            return
        category = "category" if len(args) == 1 else "categories"
        action = "Enabled" if "d" not in flags else "Disabled"
        return css_md(f"{action} command {category} {sqr_md(', '.join(args))} in {sqr_md(channel)}.")


class Prefix(Command):
    name = ["ChangePrefix"]
    min_level = 0
    min_display = "0~3"
    description = "Shows or changes the prefix for ⟨MIZA⟩'s commands for this server."
    usage = "<prefix[]> <default(?d)>"
    flags = "hd"

    def __call__(self, argv, guild, perm, bot, flags, **void):
        pref = bot.data.prefixes
        update = self.data.prefixes.update
        if "d" in flags:
            if guild.id in pref:
                pref.pop(guild.id)
                update()
            return css_md(f"Successfully reset command prefix for {sqr_md(guild)}.")
        if not argv:
            return css_md(f"Current command prefix for {sqr_md(guild)}: {sqr_md(bot.get_prefix(guild))}.")
        req = 3
        if perm < req:
            reason = f"to change command prefix for {guild}"
            raise self.perm_error(perm, req, reason)
        prefix = argv
        # Backslash is not allowed, it is used to escape commands normally
        if prefix.startswith("\\"):
            raise TypeError("Prefix must not begin with backslash.")
        pref[guild.id] = prefix
        update()
        if "h" not in flags:
            return css_md(f"Successfully changed command prefix for {sqr_md(guild)} to {sqr_md(argv)}.")


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
        num = await bot.eval_math(args[0], user)
        iters = round(num)
        # Bot owner bypasses restrictions
        if not isnan(perm):
            if iters > 32 and not bot.is_trusted(guild.id):
                raise PermissionError("Must be in a trusted server to execute loop of more than 32 iterations.")
            # Required level is 1/3 the amount of loops required, rounded up
            scale = 3
            limit = perm * scale
            if iters > limit:
                reason = (
                    "to execute loop of " + str(iters)
                    + " iterations"
                )
                raise self.perm_error(perm, ceil(iters / scale), reason)
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
                    (bot.get_prefix(guild) + n).upper() in func.replace(" ", "").upper()
                ) or (
                    (str(bot.id) + ">" + n).upper() in func.replace(" ", "").upper()
                ):
                    raise PermissionError("Must be owner to execute nested loop.")
        func2 = " ".join(func2.split(" ")[1:])
        create_task(send_with_react(
            channel,
            italics(css_md(f"Looping {sqr_md(func)} {iters} time{'s' if iters != 1 else ''}...")),
            reacts=["❎"],
        ))
        for i in range(iters):
            loop = i < iters - 1
            t = utc()
            # Calls process_message with the argument containing the looped command.
            delay = await callback(message, func, cb_argv=func2, loop=loop)
            # Must abide by command rate limit rules
            delay = delay + t - utc()
            if delay > 0:
                await asyncio.sleep(delay)


class Avatar(Command):
    name = ["PFP", "Icon"]
    min_level = 0
    description = "Sends a link to the avatar of a user or server."
    usage = "<*objects>"
    multi = True

    async def getGuildData(self, g):
        # Gets icon display of a server and returns as an embed.
        url = to_png(g.icon_url)
        name = g.name
        emb = discord.Embed(colour=rand_colour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = f"{sqr_md(name)}({url})"
        return emb

    def getMimicData(self, p):
        # Gets icon display of a mimic and returns as an embed.
        url = to_png(p.url)
        name = p.name
        emb = discord.Embed(colour=rand_colour())
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = f"{sqr_md(name)}({url})"
        return emb

    async def __call__(self, argv, argl, channel, guild, bot, user, **void):
        iterator = argl if argl else (argv,)
        embs = set()
        for argv in iterator:
            async with ExceptionSender(channel):
                with suppress(StopIteration):
                    if argv:
                        u_id = argv
                        with suppress():
                            u_id = verify_id(u_id)
                        u = guild.get_member(u_id)
                        g = None
                        while u is None and g is None:
                            with suppress():
                                u = bot.get_member(u_id, guild)
                                break
                            with suppress():
                                try:
                                    u = bot.get_user(u_id)
                                except:
                                    if not bot.in_cache(u_id):
                                        u = await bot.fetch_user(u_id)
                                    else:
                                        raise
                                break
                            if type(u_id) is str and "@" in u_id and ("everyone" in u_id or "here" in u_id):
                                g = guild
                                break
                            try:
                                p = bot.get_mimic(u_id, user)
                                embs.add(self.getMimicData(p, flags))
                            except:
                                pass
                            else:
                                raise StopIteration
                            with suppress():
                                g = bot.cache.guilds[u_id]
                                break
                            with suppress():
                                g = bot.cache.roles[u_id].guild
                                break
                            with suppress():
                                g = bot.cache.channels[u_id].guild
                            with suppress():
                                u = await bot.fetch_member_ex(u_id, guild)
                                break
                            raise LookupError(f"No results for {argv}.")     
                        if g:
                            emb = await self.getGuildData(g, flags)    
                            embs.add(emb)   
                            raise StopIteration         
                    else:
                        u = user
                    name = str(u)
                    url = best_url(u)
                    emb = discord.Embed(colour=rand_colour())
                    emb.set_thumbnail(url=url)
                    emb.set_image(url=url)
                    emb.set_author(name=name, icon_url=url, url=url)
                    emb.description = f"{sqr_md(name)}({url})"
                    embs.add(emb)
        bot.send_embeds(channel, embeds=embs)


class Info(Command):
    name = ["UserInfo", "ServerInfo", "WhoIs", "Profile"]
    min_level = 0
    description = "Shows information about the target user or server."
    usage = "<*objects> <verbose(?v)>"
    flags = "v"
    rate_limit = 1
    multi = True

    async def getGuildData(self, g, flags={}):
        bot = self.bot
        url = to_png(g.icon_url)
        name = g.name
        try:
            u = g.owner
        except (AttributeError, KeyError):
            u = None
        emb = discord.Embed(colour=rand_colour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        if u is not None:
            d = user_mention(u.id)
        else:
            d = ""
        if g.description:
            d += code_md(g.description)
        emb.description = d
        top = None
        try:
            g.region
            if not hasattr(g, "ghost"):
                pcount = await bot.database.counts.getUserMessages(None, g)
        except (AttributeError, KeyError):
            pcount = 0
        if not hasattr(g, "ghost"):
            with suppress(AttributeError, KeyError):
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
                    for i in range(min(32, flags.get("v", 0) * 5 + 5, len(us))):
                        u_id = ul[i]
                        users.append(f"{user_mention(u_id)}: {us[u_id]}")
                    top = "\n".join(users)
        emb.add_field(name="Server ID", value=str(g.id), inline=0)
        emb.add_field(name="Creation time", value=str(g.created_at), inline=1)
        if "v" in flags:
            with suppress(AttributeError, KeyError):
                emb.add_field(name="Region", value=str(g.region), inline=1)
                emb.add_field(name="Nitro boosts", value=str(g.premium_subscription_count), inline=1)
        emb.add_field(name="Text channels", value=str(len(g.text_channels)), inline=1)
        emb.add_field(name="Voice channels", value=str(len(g.voice_channels)), inline=1)
        emb.add_field(name="Member count", value=str(g.member_count), inline=1)
        if pcount:
            emb.add_field(name="Post count", value=str(pcount), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        if top is not None:
            emb.add_field(name="Top users", value=top, inline=0)
        return emb

    def getMimicData(self, p, flags={}):
        url = to_png(p.url)
        name = p.name
        emb = discord.Embed(colour=rand_colour())
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        d = f"{user_mention(p.u_id)}{fix_md(p.id)}"
        if p.description:
            d += code_md(p.description)
        emb.description = d
        pcnt = 0
        with suppress(AttributeError, KeyError):
            if "v" in flags:
                ptot = p.total
                pcnt = p.count
                if not pcnt:
                    pavg = 0
                else:
                    pavg = ptot / pcnt
        emb.add_field(name="Mimic ID", value=str(p.id), inline=0)
        emb.add_field(name="Name", value=str(p.name), inline=0)
        emb.add_field(name="Prefix", value=str(p.prefix), inline=1)
        emb.add_field(name="Creation time", value=str(datetime.datetime.fromtimestamp(p.created_at)), inline=1)
        if "v" in flags:
            emb.add_field(name="Gender", value=str(p.gender), inline=1)
            ctime = datetime.datetime.fromtimestamp(p.birthday)
            age = (utc_dt() - ctime).total_seconds() / TIMEUNITS["year"]
            emb.add_field(name="Birthday", value=str(ctime), inline=1)
            emb.add_field(name="Age", value=str(round_min(round(age, 1))), inline=1)
        if pcnt:
            emb.add_field(name="Post count", value=str(pcnt), inline=1)
            if "v" in flags:
                emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
        return emb

    async def __call__(self, argv, argl, name, guild, channel, bot, user, flags, **void):
        iterator = argl if argl else (argv,)
        embs = set()
        for argv in iterator:
            with ExceptionSender(channel):
                with suppress(StopIteration):
                    if argv:
                        u_id = argv
                        with suppress():
                            u_id = verify_id(u_id)
                        u = guild.get_member(u_id) if type(u_id) is int else None
                        g = None
                        while u is None and g is None:
                            with suppress():
                                u = bot.get_member(u_id, guild)
                                break
                            with suppress():
                                try:
                                    u = bot.get_user(u_id)
                                except:
                                    if not bot.in_cache(u_id):
                                        u = await bot.fetch_user(u_id)
                                    else:
                                        raise
                                break
                            if type(u_id) is str and "@" in u_id and ("everyone" in u_id or "here" in u_id):
                                g = guild
                                break
                            if "server" in name:
                                with suppress():
                                    g = await bot.fetch_guild(u_id)
                                    break
                                with suppress():
                                    role = await bot.fetch_role(u_id, g)
                                    g = role.guild
                                    break
                                with suppress():
                                    channel = await bot.fetch_channel(u_id)
                                    g = channel.guild
                                    break
                            try:
                                p = bot.get_mimic(u_id, user)
                                embs.add(self.getMimicData(p, flags))
                            except:
                                pass
                            else:
                                raise StopIteration
                            with suppress():
                                g = bot.cache.guilds[u_id]
                                break
                            with suppress():
                                g = bot.cache.roles[u_id].guild
                                break
                            with suppress():
                                g = bot.cache.channels[u_id].guild
                            with suppress():
                                u = await bot.fetch_member_ex(u_id, guild)
                                break
                            raise LookupError(f"No results for {argv}.")
                        if g:
                            emb = await self.getGuildData(g, flags)
                            embs.add(emb)
                            raise StopIteration
                    elif "server" not in name:
                        u = user
                    else:
                        if not hasattr(guild, "ghost"):
                            emb = await self.getGuildData(guild, flags)
                            embs.add(emb)
                            raise StopIteration
                        else:
                            u = bot.user
                    u = await bot.fetch_user_member(u.id, guild)
                    member = guild.get_member(u.id)
                    name = str(u)
                    url = best_url(u)
                    st = deque()
                    if u.id == bot.id:
                        st.append("Myself 🙃")
                        is_self = True
                    else:
                        is_self = False
                    if bot.is_owner(u.id):
                        st.append("My owner ❤️")
                    is_sys = False
                    if getattr(u, "system", None):
                        st.append("Discord System ⚙️")
                        is_sys = True
                    uf = getattr(u, "public_flags", None)
                    is_bot = False
                    if uf:
                        if uf.system and not is_sys:
                            st.append("Discord System ⚙️")
                        if uf.staff:
                            st.append("Discord Staff ⚠️")
                        if uf.partner:
                            st.append("Discord Partner 🎀:")
                        if uf.bug_hunter_level_2:
                            st.append("Bug Hunter Lv.2 🕷️")
                        elif uf.bug_hunter:
                            st.append("Bug Hunter 🐛")
                        is_hype = False
                        if uf.hypesquad_bravery:
                            st.append("Discord HypeSquad Bravery 🛡️")
                            is_hype = True
                        if uf.hypesquad_brilliance:
                            st.append("Discord HypeSquad Brilliance 🌟")
                            is_hype = True
                        if uf.hypesquad_balance:
                            st.append("Discord HypeSquad Balance 💠")
                            is_hype = True
                        if uf.hypesquad and not is_hype:
                            st.append("Discord HypeSquad 👀")
                        if uf.early_supporter:
                            st.append("Discord Early Supporter 🌄")
                        if uf.team_user:
                            st.append("Discord Team User 🧑‍🤝‍🧑")
                        if uf.verified_bot:
                            st.append("Verified Bot 👾")
                            is_bot = True
                        if uf.verified_bot_developer:
                            st.append("Verified Bot Developer 🏆")
                    if u.bot and not is_bot:
                        st.append("Bot 🤖")
                    if u.id == guild.owner_id and not hasattr(guild, "ghost"):
                        st.append("Server owner 👑")
                    dname = getattr(member, "nick", None)
                    if member:
                        joined = getattr(u, "joined_at", None)
                    else:
                        joined = None
                    created = u.created_at
                    activity = "\n".join(activity_repr(i) for i in getattr(u, "activities", ()))
                    status = None
                    if getattr(u, "status", None):
                        # Show up to 3 different statuses based on the user's desktop, web and mobile status.
                        if not is_self:
                            status_items = [(u.desktop_status, "🖥️"), (u.web_status, "🕸️"), (u.mobile_status, "📱")]
                        else:
                            status_items = [(bot.statuses[(i + bot.status_iter) % 3], x) for i, x in enumerate(("🖥️", "🕸️", "📱"))]
                        ordered = sorted(status_items, key=lambda x: status_order.index(x[0]))
                        for s, i in ordered:
                            if s == discord.Status.offline:
                                ls = bot.data.users.get(u.id, {}).get("last_seen", 0)
                                if utc() - ls < 300 and ls > bot.data.users.get(u.id, {}).get("last_offline", 0):
                                    s = discord.Status.invisible
                            icon = status_icon[s]
                            if not status:
                                s_ = u.status
                                if s != s_ and s == discord.Status.offline:
                                    status = status_text[s_]  + " `" + status_icon[s_] + "❓"
                                    if s not in (discord.Status.offline, discord.Status.invisible):
                                        status += icon
                                else:
                                    status = status_text[s] + " `" + icon
                            if s not in (discord.Status.offline, discord.Status.invisible):
                                if icon not in status:
                                    status += icon
                                status += i
                        if status:
                            status += "`"
                            if len(status) >= 16:
                                status = status.replace(" ", "\n")
                    if member:
                        rolelist = [role_mention(i.id) for i in reversed(getattr(u, "roles", ())) if not i.is_default()]
                        role = ", ".join(rolelist)
                    else:
                        role = None
                    coms = seen = msgs = avgs = gmsg = old = 0
                    fav = None
                    pos = None
                    zone = None
                    with suppress(LookupError):
                        ts = utc()
                        ls = bot.data.users[u.id]["last_seen"]
                        la = bot.data.users[u.id].get("last_action")
                        if type(ls) is str:
                            seen = ls
                        else:
                            seen = f"{sec2time(max(0, ts - ls))} ago"
                        if la:
                            seen = f"{la}, {seen}"
                        if "v" in flags:
                            tz = bot.database.users.estimate_timezone(u.id)
                            if tz >= 0:
                                zone = f"GMT+{tz}"
                            else:
                                zone = f"GMT{tz}"
                    with suppress(LookupError):
                        old = bot.data.counts.get(guild.id, {})["oldest"][u.id]
                        old = snowflake_time(old)
                    if "v" in flags:
                        with suppress(LookupError):
                            if is_self:
                                # Count total commands used by all users
                                c = {}
                                for i, v in enumerate(tuple(bot.data.users.values()), 1):
                                    with suppress(KeyError):
                                        add_dict(c, v["commands"])
                                    if not i & 8191:
                                        await asyncio.sleep(0.2)
                            else:
                                c = bot.data.users[u.id]["commands"]
                            coms = iter_sum(c)
                            if type(c) is dict:
                                with suppress(IndexError):
                                    comfreq = deque(sort(c, reverse=True).keys())
                                    while fav is None:
                                        fav = comfreq.popleft()
                        with suppress(LookupError):
                            if not hasattr(guild, "ghost"):
                                gmsg = bot.database.counts.getUserGlobalMessageCount(u)
                                msgs = await bot.database.counts.getUserMessages(u, guild)
                                avgs = await bot.database.counts.getUserAverage(u, guild)
                    with suppress(LookupError):
                        if not hasattr(guild, "ghost"):
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
                    if is_self and bot.website is not None:
                        url2 = bot.website
                    else:
                        url2 = url
                    emb = discord.Embed(colour=rand_colour())
                    emb.set_thumbnail(url=url)
                    emb.set_author(name=name, icon_url=url, url=url2)
                    d = user_mention(u.id)
                    if activity:
                        d += "\n" + italics(code_md(activity))
                    if st:
                        if d[-1] == "*":
                            d += " "
                        d += " **```css\n"
                        if st:
                            d += "\n".join(st)
                        d += "```**"
                    emb.description = d
                    emb.add_field(name="User ID", value="`" + str(u.id) + "`", inline=0)
                    emb.add_field(name="Creation time", value=str(created), inline=1)
                    if joined:
                        emb.add_field(name="Join time", value=str(joined), inline=1)
                    if old:
                        emb.add_field(name="Oldest post time", value=str(old), inline=1)
                    if status:
                        emb.add_field(name="Status", value=str(status), inline=1)
                    if zone:
                        emb.add_field(name="Estimated timezone", value=str(zone), inline=1)
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
                        emb.add_field(name=f"Roles ({len(rolelist)})", value=role, inline=0)
                    embs.add(emb)
        bot.send_embeds(channel, embeds=embs)


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
            u_id = verify_id(argv)
            try:
                user = await bot.fetch_user(u_id)
            except (TypeError, discord.NotFound):
                user = await bot.fetch_member_ex(u_id, guild)
        data = await create_future(bot.database.users.fetch_events, user.id, interval=max(900, 3600 >> flags.get("v", 0)), timeout=12)
        with discord.context_managers.Typing(channel):
            resp = await process_image("plt_special", "$", (data, str(user)), guild)
            fn = resp[0]
            f = discord.File(fn, filename=f"{user.id}.png")
        return dict(file=f, filename=fn, best=True)


class Status(Command):
    name = ["State", "Ping"]
    min_level = 0
    description = "Shows the bot's current internal program state."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, perm, flags, channel, bot, **void):
        if "d" in flags:
            if perm < 2:
                raise PermissionError("Permission level 2 or higher required to unset auto-updating status.")
            bot.data.messages.pop(channel.id)
            bot.database.messages.update()
            return fix_md("Successfully disabled status updates.")
        elif "a" not in flags and "e" not in flags:
            return await self._callback2_(channel)
        if perm < 2:
            raise PermissionError("Permission level 2 or higher required to set auto-updating status.")
        message = await channel.send(italics(code_md("Loading bot status...")))
        set_dict(bot.data.messages, channel.id, {})[message.id] = cdict(t=0, command="bot.commands.status[0]")
        bot.database.messages.update()
    
    async def _callback2_(self, channel, m_id=None, msg=None, **void):
        bot = self.bot
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(name="Status", url=bot.website, icon_url=best_url(bot.user))
        emb.timestamp = utc_dt()
        if msg is None:
            active = bot.get_active()
            try:
                shards = len(bot.latencies)
            except AttributeError:
                shards = 1
            size = sum(bot.size.values()) + sum(bot.size2.values())
            stats = bot.curr_state
            cache = await create_future(os.listdir, "cache/", timeout=12)

            bot_info = (
                f"Process count\n`{active[0]}`\nThread count\n`{active[1]}`\nCoroutine count\n`{active[2]}`\n"
                + f"CPU usage\n`{round(stats[0], 3)}%`\nRAM usage\n`{byte_scale(stats[1])}B`\nDisk usage\n`{byte_scale(stats[2])}B`\nNetwork usage\n`{byte_scale(bot.bitrate)}bps`"
            )
            emb.add_field(name="Bot info", value=bot_info)

            discord_info = (
                f"Shard count\n`{shards}`\nServer count\n`{len(bot.guilds)}`\nUser count\n`{len(bot.cache.users)}`\n"
                + f"Channel count\n`{len(bot.cache.channels)}`\nRole count\n`{len(bot.cache.roles)}`\nEmoji count\n`{len(bot.cache.emojis)}`"
            )
            emb.add_field(name="Discord info", value=discord_info)

            misc_info = (
                f"Cached messages\n`{len(bot.cache.messages)}`\nCached files\n`{len(cache)}`\nConnected voice channels\n`{len(bot.voice_clients)}`\nTotal data sent/received\n`{byte_scale(bot.total_bytes)}B`\n"
                + f"System time\n`{datetime.datetime.now()}`\nPing latency\n`{sec2time(bot.latency)}`\nPublic IP address\n`{bot.ip}`"
            )
            emb.add_field(name="Misc info", value=misc_info)
            emb.add_field(name="Code info", value=f"[`{byte_scale(size[0])}B, {size[1]} lines`]({bot.website})")
        else:
            emb.description = msg
        func = channel.send
        if m_id is not None:
            with tracebacksuppressor:
                message = bot.cache.messages.get(m_id)
                if message is None:
                    message = await aretry(channel.fetch_message, m_id, attempts=6, delay=2, exc=(discord.NotFound, discord.Forbidden))
                if message.id != channel.last_message_id:
                    hist = await channel.history(limit=1).flatten()
                    channel.last_message_id = hist[0].id
                    if message.id != hist[0].id:
                        create_task(bot.silent_delete(message))
                        raise StopIteration
                func = lambda *args, **kwargs: message.edit(*args, content=None, **kwargs)
        message = await func(embed=emb)
        if m_id is not None and message is not None:
            bot.data.messages[channel.id] = {message.id: cdict(t=utc(), command="bot.commands.status[0]")}
            bot.database.messages.update()


class Invite(Command):
    name = ["OAuth", "InviteBot", "InviteLink"]
    min_level = 0
    description = "Sends a link to ⟨MIZA⟩'s homepage and invite code."
    
    def __call__(self, channel, **void):
        if discord_id is None:
            raise FileNotFoundError("Unable to locate bot's Client ID.")
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(**get_author(self.bot.user))
        emb.description = f"[**`Homepage`**]({self.bot.website}) [**`Invite`**](https://discordapp.com/oauth2/authorize?permissions=8&client_id={discord_id}&scope=bot)"
        self.bot.send_embeds(channel, embed=emb)


class Reminder(Command):
    name = ["Announcement", "Announcements", "Announce", "RemindMe", "Reminders", "Remind"]
    min_level = 0
    description = "Sets a reminder for a certain date and time."
    usage = "<1:message> <0:time> <disable(?d)>"
    flags = "aed"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 1 / 3
    keywords = ["on", "at", "in", "when", "event"]
    keydict = {re.compile(f"(^|[^a-z0-9]){i[::-1]}([^a-z0-9]|$)", re.I): None for i in keywords}
    timefind = None

    def __load__(self):
        self.timefind = re.compile("(?:(?:(?:[0-9]+:)+[0-9.]+\\s*(?:am|pm)?|" + self.bot.num_words + "|[\\s\-+*\\/^%.,0-9]+\\s*(?:am|pm|s|m|h|d|w|y|century|centuries|millenium|millenia|(?:second|sec|minute|min|hour|hr|day|week|wk|month|mo|year|yr|decade|galactic[\\s\\-_]year)s?))\\s*)+$", re.I)

    async def __call__(self, argv, name, message, flags, bot, user, guild, perm, **void):
        msg = message.content
        argv2 = argv
        argv = msg[msg.casefold().index(name) + len(name):].strip(" ").strip("\n")
        try:
            args = shlex.split(argv)
        except ValueError:
            args = argv.split(" ")
        if "announce" in name:
            sendable = message.channel
            word = "announcements"
        else:
            sendable = user
            word = "reminders"
        rems = bot.data.reminders.get(sendable.id, [])
        update = bot.database.reminders.update
        if "d" in flags:
            if not len(rems):
                return ini_md(f"No {word} currently set for {sqr_md(sendable)}.")
            if not argv:
                i = 0
            else:
                i = await bot.eval_math(argv2, user)
            i %= len(rems)
            x = rems.pop(i)
            if i == 0:
                with suppress(IndexError):
                    bot.database.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
                if rems:
                    bot.database.reminders.listed.insort((rems[0]["t"], sendable.id), key=lambda x: x[0])
            update()
            return ini_md(f"Successfully removed {sqr_md(lim_str(x['msg'], 128))} from {word} list for {sqr_md(sendable)}.")
        if not argv:
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-main-reminder-"
                + str(user.id) + "_0_" + str(sendable.id)
                + "-\nLoading Reminder database...```*"
            )
        if len(rems) >= 64:
            raise OverflowError(f"You have reached the maximum of 64 {word}. Please remove one to add another.")
        # This parser is so unnecessarily long for what it does...
        keyed = False
        while True:
            argv = argv.replace("tomorrow at", "at tomorrow").replace("today at", "at today").replace("yesterday at", "at yesterday")
            temp = argv.casefold()
            if name == "remind" and temp.startswith("me "):
                argv = argv[3:]
                temp = argv.casefold()
            if temp.startswith("to "):
                argv = argv[3:]
                temp = argv.casefold()
            elif temp.startswith("that "):
                argv = argv[5:]
                temp = argv.casefold()
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
                    t = verify_id(spl[-1])
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
                    t = verify_id(spl[-1])
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
                    t = await bot.eval_time(spl[-1], user)
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
            if "today" in argv or "tomorrow" in argv or "yesterday" in argv:
                t = 0
                if " " in argv:
                    try:
                        args = shlex.split(argv)
                    except ValueError:
                        args = argv.split()
                    for arg in (args[0], args[-1]):
                        a = arg
                        h = 0
                        for op in "+-":
                            try:
                                i = arg.index(op)
                            except ValueError:
                                continue
                            a = arg[:i]
                            h += float(arg[i:])
                        tz = a.casefold()
                        if tz in TIMEZONES:
                            t = get_timezone(tz)
                            argv = argv.replace(arg, "")
                            break
                        h = 0
                    t += h * 3600
                match = re.search(self.timefind, argv)
                if match:
                    i = match.start()
                    spl = [argv[:i], argv[i:]]
                    msg = spl[0]
                    t += utc_ts(tzparse(spl[1])) - utc()
                    break
                msg = " ".join(args[:-1])
                t = utc_ts(tzparse(args[-1])) - utc()
                break
            t = 0
            if " " in argv:
                try:
                    args = shlex.split(argv)
                except ValueError:
                    args = argv.split()
                for arg in (args[0], args[-1]):
                    a = arg
                    h = 0
                    for op in "+-":
                        try:
                            i = arg.index(op)
                        except ValueError:
                            continue
                        a = arg[:i]
                        h += float(arg[i:])
                    tz = a.casefold()
                    if tz in TIMEZONES:
                        t = get_timezone(tz)
                        argv = argv.replace(arg, "")
                        break
                    h = 0
                t += h * 3600
            match = re.search(self.timefind, argv)
            if match:
                i = match.start()
                spl = [argv[:i], argv[i:]]
                msg = spl[0]
                t += await bot.eval_time(spl[1], user)
                break
            msg = " ".join(args[:-1])
            t = await bot.eval_time(args[-1], user)
            break
        if keyed:
            u = await bot.fetch_user_member(t, guild)
            t = u.id
        msg = msg.strip(" ")
        if not msg:
            if "announce" in name:
                msg = "[SAMPLE ANNOUNCEMENT]"
            else:
                msg = "[SAMPLE REMINDER]"
            msg = bold(css_md(msg))
        elif len(msg) > 1024:
            raise OverflowError(f"Input message too long ({len(msg)} > 1024).")
        username = str(user)
        url = best_url(user)
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
            seq = set_dict(bot.data.reminders, s, deque())
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
        with suppress(IndexError):
            # Remove existing schedule
            bot.database.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
        # Insert back into bot schedule
        bot.database.reminders.listed.insort((bot.data.reminders[sendable.id][0]["t"], sendable.id), key=lambda x: x[0])
        update()
        emb = discord.Embed(description=msg)
        emb.set_author(name=username, url=url, icon_url=url)
        if "announce" in name:
            out = f"```css\nSuccessfully set announcement for {sqr_md(sendable)}"
        else:
            out = f"```css\nSuccessfully set reminder for {sqr_md(sendable)}"
        if keyed:
            out += f" upon next event from {sqr_md(user_mention(t))}"
        else:
            out += f" in {sqr_md(sec2time(t))}"
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
        sendable = await bot.fetch_messageable(s_id)
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
        content = "*```" + "\n" * ("\n" in content[:i]) + (
            "callback-main-reminder-"
            + str(u_id) + "_" + str(pos) + "_" + str(s_id)
            + "-\n"
        )
        if not rems:
            content += f"Schedule for {str(sendable).replace('`', '')} is currently empty.```*"
            msg = ""
        else:
            t = utc()
            content += f"{len(rems)} messages currently scheduled for {str(sendable).replace('`', '')}:```*"
            msg = iter2str(
                rems[pos:pos + page],
                key=lambda x: lim_str(bot.get_user(x.get("user", -1), replace=True).mention + ": `" + no_md(x["msg"]), 96) + "` ➡️ " + (user_mention(x["u_id"]) if "u_id" in x else sec2time(x["t"] - t)),
                left="`[",
                right="]`",
            )
        emb = discord.Embed(
            description=content + msg,
            colour=rand_colour(),
        ).set_author(**get_author(user))
        more = len(rems) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                async with delay(0.5):
                    create_task(message.add_reaction(react.decode("utf-8")))


# This database is such a hassle to manage, it has to be able to persist between bot restarts, and has to be able to update with O(1) time complexity when idle
class UpdateReminders(Database):
    name = "reminders"
    no_delete = True
    rate_limit = 1

    def __load__(self):
        d = self.data
        # This exists so that checking next scheduled item is O(1)
        self.listed = alist(sorted(((d[i][0]["t"], i) for i in d if type(i) is not str and d[i]), key=lambda x: x[0]))

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
            ch = await self.bot.fetch_messageable(u_id)
            emb = discord.Embed(description=x.msg)
            try:
                u = self.bot.get_user(x["user"], replace=True)
            except KeyError:
                u = x
            emb.set_author(**get_author(u))
            self.bot.send_embeds(ch, emb)
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
            with tracebacksuppressor:
                for u_id in assigned:
                    # Send reminder to all targeted users/channels
                    ch = await self.bot.fetch_messageable(u_id)
                    rems = set_dict(self.data, u_id, [])
                    pops = set()
                    for i, x in enumerate(reversed(rems), 1):
                        if x.get("u_id", None) == user.id:
                            emb = discord.Embed(description=x["msg"])
                            try:
                                u = self.bot.get_user(x["user"], replace=True)
                            except KeyError:
                                u = cdict(x)
                            emb.set_author(**get_author(u))
                            self.bot.send_embeds(ch, emb)
                            pops.add(len(rems) - i)
                        elif is_finite(x["t"]):
                            break
                    it = [rems[i] for i in range(len(rems)) if i not in pops]
                    rems.clear()
                    rems.extend(it)
                    if not rems:
                        self.data.pop(u_id)
            with suppress(KeyError):
                self.data.pop(s)
                self.update()


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
            c_id = self.bot.id
            if guild is None or hasattr(guild, "ghost"):
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
            with suppress(TypeError):
                c = self.data[g_id]["counts"]
                if user.id in c:
                    count += c[user.id]
        return count

    async def getUserAverage(self, user, guild):
        if self.scanned == -1:
            c_id = self.bot.id
            if guild is None or hasattr(guild, "ghost"):
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
                with suppress(ZeroDivisionError):
                    return t.get(user.id, 0) / c.get(user.id, 1)
                c.pop(user.id, None)
                t.pop(user.id, None)
                return 0
        return "Calculating..."            

    async def getGuildMessages(self, guild):
        if guild is None or hasattr(guild, "ghost"):
            return "Invalid server."
        if self.scanned == -1:
            if guild.id in self.data:
                with suppress(LookupError):
                    return self.data[guild.id]["counts"]
                return self.data[guild.id]
            self.startCalculate(guild)
            create_task(self.getUserMessageCount(guild))
        return "Calculating..."

    # What are the rate limits for the message history calls?
    async def getChannelHistory(self, channel, limit=None, callback=None):

        async def delay_if_exc(channel):
            t = utc()
            history = channel.history(limit=limit, oldest_first=(limit is None))
            try:
                return await history.flatten()
            except discord.Forbidden:
                return []
            except:
                remaining = 20 - utc() + t
                if remaining > 0:
                    await asyncio.sleep(remaining)
                raise

        while True:
            with tracebacksuppressor(SemaphoreOverflowError):
                async with self.semaphore:
                    messages = []
                    # 16 attempts to download channel
                    messages = await aretry(delay_if_exc, channel, attempts=16, delay=20)
                    break
            await asyncio.sleep(30)
        if callback:
            return await create_future(callback, channel=channel, messages=messages)
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
            add_dict(self.data[guild.id], {0: True})
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
        add_dict(self.data[guild.id], {"counts": data, "totals": avgs, 0: True})
        self.update()
        print(guild)
        print(self.data[guild.id])

    def __load__(self):
        self.scanned = False
        self.semaphore = Semaphore(12, 256, delay=5)

    async def __call__(self):
        if self.scanned:
            return
        self.scanned = True
        year = datetime.timedelta(seconds=31556925.216)
        guilds = self.bot.guilds
        i = 1
        for guild in sorted(guilds, key=lambda g: g.member_count, reverse=True):
            if guild.id not in self.data:
                oneyear = utc_dt() - guild.created_at < year
                if guild.member_count < 512 or oneyear:
                    async with self.semaphore:
                        self.startCalculate(guild)
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
                if user.id not in set_dict(d, "oldest", {}):
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


class UpdateMessages(Database):
    name = "messages"
    semaphore = Semaphore(80, 1, delay=1, rate_limit=16)
    closed = False

    async def wrap_semaphore(self, func, *args, **kwargs):
        with tracebacksuppressor(SemaphoreOverflowError):
            async with self.semaphore:
                return await func(*args, **kwargs)

    async def __call__(self, **void):
        if not self.closed:
            t = utc()
            for c_id, data in tuple(self.data.items()):
                with tracebacksuppressor():
                    try:
                        channel = await self.bot.fetch_channel(c_id)
                        if hasattr(channel, "guild") and channel.guild not in self.bot.guilds:
                            raise
                    except:
                        self.data.pop(c_id)
                    else:
                        for m_id, v in data.items():
                            if t - v.t >= 1:
                                v.t = t
                                create_task(self.wrap_semaphore(eval(v.command, self.bot._globals)._callback2_, channel=channel, m_id=m_id))
    
    async def _destroy_(self, **void):
        self.closed = True
        msg = "Offline 😔"
        for c_id, data in self.data.items():
            with tracebacksuppressor(SemaphoreOverflowError):
                channel = await self.bot.fetch_channel(c_id)
                for m_id, v in data.items():
                    async with self.semaphore:
                        await eval(v.command, self.bot._globals)._callback2_(channel=channel, m_id=m_id, msg=msg)


EMPTY = {}

# This database takes up a lot of space, storing so many events from users
class UpdateUsers(Database):
    name = "users"
    suspected = "users.json"
    user = True
    hours = 168
    interval = 900
    scale = 3600 // interval
    mentionspam = re.compile("<@[!&]?[0-9]+>")

    def __load__(self):
        self.semaphore = Semaphore(3, 2, delay=0.5)
        self.facts = None
        self.flavour_buffer = deque()
        self.flavour_set = set()
        self.flavour = ()
        self.useless = ()
        with open("misc/facts.txt", "r", encoding="utf-8") as f:
            self.facts = f.read().splitlines()

    async def _bot_ready_(self, **void):
        data = {"Command": Command}
        exec(
            f"class {self.bot.name.replace(' ', '')}(Command):"
            +"\n\tmin_level = 0"
            +"\n\tno_parse = True"

            +"\n\tasync def __call__(self, message, argv, **void):"
            +"\n\t\tawait self.bot.database.users._nocommand_(message, self.bot.user.mention + ' ' + argv, force=True)",
            data,
        )
        mod = __name__
        for v in data.values():
            with suppress(TypeError):
                if issubclass(v, Command) and v != Command:
                    obj = v(self.bot, mod)
                    self.bot.categories[mod].append(obj)
                    print(f"Successfully loaded command {repr(obj)}.")
        return await self()

    def clear_events(self, data, minimum):
        for hour in tuple(data):
            if hour > minimum:
                return
            data.pop(hour, None)

    def send_event(self, u_id, event, count=1):
        # print(self.bot.cache.users.get(u_id), event, count)
        data = set_dict(set_dict(self.data, u_id, {}), "recent", {})
        hour = round_min(round(utc() // self.interval) / self.scale)
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
        hour = round_min(round(utc() // self.interval) / self.scale)
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

    def estimate_timezone(self, u_id):
        data = self.data.get(u_id, EMPTY).get("recent")
        if not data:
            return 0
        hour = round_min(round(utc() // self.interval) / self.scale)
        self.clear_events(data, hour - self.hours)
        start = hour - self.hours
        out = [sum(data.get(i / self.scale + start, EMPTY).values()) for i in range(self.hours * self.scale)]
        factor = ceil(3600 / self.interval)
        activity = [sum(out[i:i + factor]) for i in range(0, len(out), factor)]
        inactive = alist()
        def register(curr):
            if inactive:
                last = inactive[-1]
            if not inactive or curr[0] - last[0] >= 24:
                curr[1] += 1
                inactive.append(curr[:2])
                curr[2] = curr[0]
            elif curr[0] - last[0] - last[1] < 2:
                last[1] += curr[0] + curr[1] - last[0] - last[1]
                curr[2] = curr[0]
            elif last[1] <= curr[1] * 1.5:
                curr[1] += 1
                if curr[0] - curr[2] >= 18:
                    inactive.append(curr[:2])
                    curr[2] = curr[0]
                else:
                    inactive[-1] = curr[:2]
            curr[0] = None
            curr[1] = 0
        m = min(activity) * 4
        curr = [None, 0, 0]
        for i, x in enumerate(activity):
            if x <= m:
                if curr[0] is None:
                    curr[0] = i
                curr[1] += 1
            else:
                if curr[0] is not None:
                    register(curr)
        if curr[0] is not None:
            register(curr)
        total = 0
        if inactive:
            for i, curr in enumerate(inactive):
                t = (curr[0] + curr[1] / 2) % 24
                if i:
                    if total / i - t > 12:
                        total += 24
                    elif total / i - t < -12:
                        total -= 24
                total += t
            estimated = round(2.5 - utc_dt().hour - total / len(inactive)) % 24
            if estimated > 12:
                estimated -= 24
        else:
            estimated = 0
        # print(estimated, inactive, activity)
        return estimated

    async def __call__(self):
        with suppress(SemaphoreOverflowError):
            async with self.semaphore:
                changed = False
                while len(self.flavour_buffer) < 10:
                    out = None
                    i = xrand(7)
                    if i <= 1 and self.facts:
                        with tracebacksuppressor:
                            text = choice(self.facts)
                            fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
                            out = f"\n{fact} `{text}`"
                    elif i == 2:
                        with tracebacksuppressor:
                            data = await Request("https://www.affirmations.dev/", aio=True)
                            text = eval_json(data)["affirmation"].replace("`", "")
                            out = f"\nAffirmation: `{text}`"
                    elif i == 3:
                        with tracebacksuppressor:
                            data = await Request("https://geek-jokes.sameerkumar.website/api", aio=True)
                            text = eval_json(data).replace("`", "")
                            out = f"\nGeek joke: `{text}`"
                    else:
                        with tracebacksuppressor:
                            if len(self.useless) < 128 and (not self.useless or random.random() > 0.75):
                                data = await Request("https://www.uselessfacts.net/api/posts?d=" + str(datetime.datetime.fromtimestamp(xrand(1462456800, utc())).date()), aio=True)
                                factlist = [fact["title"].replace("`", "") for fact in eval_json(data) if "title" in fact]
                                random.shuffle(factlist)
                                self.useless = deque()
                                for text in factlist:
                                    fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
                                    out = f"\n{fact} `{text}`"
                                    self.useless.append(out)
                            out = self.useless.popleft()
                    if out:
                        self.flavour_buffer.append(out)
                        self.flavour_set.add(out)
                        changed = True
                amount = len(self.flavour_set)
                if changed and (not amount & amount - 1):
                    self.flavour = tuple(self.flavour_set)

    def _offline_(self, user, **void):
        set_dict(self.data, user.id, {})["last_offline"] = utc()

    # User seen, add event to activity database
    def _seen_(self, user, delay, event, count=1, raw=None, **void):
        self.send_event(user.id, event, count=count)
        add_dict(self.data, {user.id: {"last_seen": 0}})
        self.data[user.id]["last_seen"] = utc() + delay
        self.data[user.id]["last_action"] = raw

    # User executed command, add to activity database
    def _command_(self, user, command, **void):
        self.send_event(user.id, "command")
        add_dict(self.data, {user.id: {"commands": {command.__name__: 1}}})
        self.data[user.id]["last_used"] = utc()
        self.data.get(user.id, EMPTY).pop("last_mention", None)
        self.update()

    async def _mention_(self, user, message, msg, **void):
        bot = self.bot
        mentions = self.mentionspam.findall(msg)
        t = utc()
        out = None
        if len(mentions) >= xrand(8, 12) and self.data.get(user.id, EMPTY).get("last_mention", 0) > 3:
            out = f"{choice('🥴😣😪😢')} please calm down a second, I'm only here to help..."
        elif len(mentions) >= 3 and (self.data.get(user.id, EMPTY).get("last_mention", 0) > 2 or random.random() >= 2 / 3):
            out = f"{choice('😟😦😓')} oh, that's a lot of mentions, is everything okay?"
        elif len(mentions) >= 2 and self.data.get(user.id, EMPTY).get("last_mention", 0) > 0 and random.random() >= 0.75:
            out = "One mention is enough, but I appreciate your enthusiasm 🙂"
        if out:
            create_task(send_with_react(message.channel, out, reacts="❎"))
            await bot.seen(user, event="misc", raw="Being naughty")
            add_dict(self.data, {user.id: {"last_mention": 1}})
            self.data[user.id]["last_used"] = t
            raise CommandCancelledError

    async def _nocommand_(self, message, msg, force=False, **void):
        bot = self.bot
        user = message.author
        if bot.get_perms(user.id, message.guild) <= -inf:
            return
        if force or bot.is_mentioned(message, bot, message.guild):
            send = message.channel.send
            out = None
            count = self.data.get(user.id, EMPTY).get("last_talk", 0)
            # Simulates a randomized conversation
            if count < 5:
                create_task(message.add_reaction("👀"))
            if "?" in msg and "ask" in bot.commands and random.random() > math.atan(count / 16) / 4:
                argv = self.mentionspam.sub("", msg).strip()
                for ask in bot.commands.ask:
                    await ask(message.channel, user, argv)
                return
            if count:
                if count < 2 or count == 2 and xrand(2):
                    # Starts conversations
                    out = choice(
                        f"So, {user.display_name}, how's your day been?",
                        f"How do you do, {user.name}?",
                        f"How are you today, {user.name}?",
                        "What's up?",
                        "Can I entertain you with a little something today?",
                    )
                elif count < 16 or random.random() > math.atan(max(0, count / 8 - 3)) / 4:
                    # General messages
                    if (count < 6 or self.mentionspam.sub("", msg).strip()) and random.random() < 0.5:
                        out = choice((f"'sup, {user.display_name}?", f"There you are, {user.name}!", "Oh yeah!", "👋", f"Hey, {user.display_name}!"))
                    else:
                        out = ""
                elif count < 24:
                    # Occasional late message
                    if random.random() < 0.4:
                        out = choice(
                            "You seem rather bored... I may only be as good as my programming allows me to be, but I'll try my best to fix that! 🎆",
                            "You must be bored, allow me to entertain you! 🍿",
                        )
                    else:
                        out = ""
                else:
                    # Late conversation messages
                    out = choice(
                        "It's been a fun conversation, but don't you have anything better to do? 🌞",
                        "This is what I was made for, I can do it forever, but you're only a human, take a break! 😅",
                        f"Woah, have you checked the time? We've been talking for {count + 1} messages! 😮"
                    )
            elif utc() - self.data.get(user.id, EMPTY).get("last_used", inf) >= 259200:
                # Triggers for users not seen in 3 days or longer
                out = choice((f"Long time no see, {user.name}!", f"Great to see you again, {user.display_name}!", f"It's been a while, {user.name}!"))
            if out is not None:
                guild = message.guild
                # Add randomized flavour text if in conversation
                if xrand(4) or guild is None:
                    if self.flavour_buffer:
                        out += self.flavour_buffer.popleft()
                    else:
                        out += choice(self.flavour)
                else:
                    i = xrand(4)
                    member = choice(guild.members)
                    if i == 0:
                        count = await bot.database.counts.getUserMessages(member, guild)
                        out += f"\nServer insights: `{member} has posted {count} message{'s' if count != 1 else ''} in total!`"
                    elif i == 1:
                        curr = member.joined_at
                        old = None
                        with suppress(LookupError):
                            old = bot.data.counts.get(guild.id, {})["oldest"][member.id]
                            old = snowflake_time(old)
                        if old is not None and old < curr:
                            curr = old
                        out += f"\nServer insights: `{member} has been active here for {sec2time(utc() - curr.timestamp())}!`"
                    elif i == 2:
                        events = bot.database.users.get_events(member.id, interval=900)
                        out += f"\nServer insights: `{member} has performed {sum(events)} discord actions in the past 7 days!`"
                    else:
                        i = xrand(4)
                        if i == 0:
                            users = set(bot.data.counts.get(guild.id, {})["totals"])
                            users.update(guild._members)
                            out += f"\nServer insights: `{len(users)} have set their footprint into this server!`"
                        elif i == 1:
                            totals = bot.data.counts.get(guild.id, {})["totals"]
                            u_id = iter_max(totals)
                            try:
                                u = await bot.fetch_user(u_id)
                            except:
                                u = await bot.get_user(u_id, replace=True)
                            total = totals[u_id]
                            out += f"\nServer insights: `{u} has posted the most text, with {total} total characters!`"
                        elif i == 2:
                            counts = bot.data.counts.get(guild.id, {})["counts"]
                            u_id = iter_max(counts)
                            try:
                                u = await bot.fetch_user(u_id)
                            except:
                                u = await bot.get_user(u_id, replace=True)
                            count = counts[u_id]
                            out += f"\nServer insights: `{u} has posted the most messages, with {count} in total!`"
                        elif i == 3:
                            found = member
                            highest = 0
                            for u_id in guild._members:
                                if u_id != bot.id:
                                    mutual = 0
                                    for g in bot.guilds:
                                        if u_id in g._members:
                                            mutual += 1
                                    if mutual > highest:
                                        highest = mutual
                                        found = guild.get_member(u_id)
                            out += f"\nServer insights: `{found} shares the highest number of mutual servers with me, with {highest}!`"
            else:
                # Help message greetings
                i = xrand(7)
                if i == 0:
                    out = "I have been summoned!"
                elif i == 1:
                    out = f"Hey there! Name's {bot.name}!"
                elif i == 2:
                    out = f"Hello {user.name}, nice to see you! Can I help you?"
                elif i == 3:
                    out = f"Howdy, {user.display_name}!"
                elif i == 4:
                    out = f"Greetings, {user.name}! May I be of service?"
                elif i == 5:
                    out = f"Hi, {user.name}! What can I do for you today?"
                else:
                    out = f"Yo, what's good, {user.display_name}? Need me for anything?"
                prefix = bot.get_prefix(message.guild)
                out += f" Use `{prefix}?` or `{prefix}help` for help!"
                send = lambda *args, **kwargs: send_with_react(message.channel, *args, **kwargs, reacts="❎")
            add_dict(self.data, {user.id: {"last_talk": 1, "last_mention": 1}})
            self.data[user.id]["last_used"] = utc()
            await send(out)
            await bot.seen(user, event="misc", raw="Talking to me")
        else:
            if not self.data.get(user.id, EMPTY).get("last_mention") and random.random() > 0.6:
                self.data.get(user.id, EMPTY).pop("last_talk", None)
            self.data.get(user.id, EMPTY).pop("last_mention", None)