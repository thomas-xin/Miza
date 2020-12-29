try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

print = PRINT


# Default and standard command categories to enable.
default_commands = frozenset(("main", "string", "admin"))
standard_commands = default_commands.union(("voice", "image", "fun"))

help_colours = fcdict({
    None: 0xfffffe,
    "main": 0xff0000,
    "string": 0x00ff00,
    "admin": 0x0000ff,
    "voice": 0xff00ff,
    "image": 0xffff00,
    "fun": 0x00ffff,
    "owner": 0xbf7fff,
    "nsfw": 0xff9f9f,
    "misc": 0x007f00,
})


class Help(Command):
    name = ["❓", "❔", "?"]
    description = "Shows a list of usable commands, or gives a detailed description of a command."
    usage = "<(command|category)>? <verbose{?v}>?"
    flags = "v"
    slash = True

    async def __call__(self, args, user, channel, guild, message, flags, perm, **void):
        bot = self.bot
        enabled = bot.data.enabled
        g_id = guild.id
        prefix = bot.get_prefix(g_id)
        v = "v" in flags
        author=dict(name="❓ Help ❓")
        fields = deque()
        found = {}
        description = ""
        colour = None
        # Get help on categories, then commands
        for a in args:
            a = a.casefold()
            if a in bot.categories:
                coms = bot.categories[a]
                colour = discord.Colour(help_colours[a])
            elif a in bot.commands:
                coms = bot.commands[a]
                colour = discord.Colour(help_colours[coms[-1].catg])
            else:
                continue
            for com in coms:
                if com.parse_name() in found:
                    found[com.parse_name()].append(com)
                else:
                    found[com.parse_name()] = alist((com,))
        if found:
            # Display list of found commands in an embed
            for k in found:
                coms = found[k]
                for com in coms:
                    a = ", ".join(n.strip("_") for n in com.name)
                    if not a:
                        a = "[none]"
                    s = f"```ini\n[Aliases] {a}"
                    s += f"\n[Effect] {com.parse_description()}"
                    if v or len(found) <= 1:
                        s += f"\n[Usage] {prefix}{com.parse_name()} {com.usage}\n[Level] {com.min_display}"
                    s += "```"
                    fields.append(dict(
                        name=prefix + com.parse_name(),
                        value=s,
                        inline=False
                    ))
        else:
            # Display main help page in an embed
            colour = discord.Colour(help_colours[None])
            description = (
                "Please enter a command category to display usable commands,\nor see "
                + "[Commands](https://github.com/thomas-xin/Miza/wiki/Commands) for full command list."
            )
            if bot.categories:
                s = bold(ini_md(' '.join((sqr_md(c) for c in help_colours if c in bot.categories))))
                fields.append(dict(name="Command category list", value=s))
        bot.send_as_embeds(channel, description, author=author, fields=fields, colour=colour, reacts="❎", reference=message)


class Hello(Command):
    name = ["👋", "Hi", "Hi!", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
    description = "Sends a greeting message. Useful for checking whether the bot is online."
    usage = "<user>?"
    slash = True
    
    async def __call__(self, bot, user, name, argv, guild, **void):
        if "dailies" in bot.data:
            bot.data.dailies.progress_quests(user, "talk")
        if argv:
            user = await bot.fetch_user_member(argv, guild)
        elif bot.is_owner(user):
            return "👋"
        if name in ("bye", "cya", "goodbye"):
            start = choice("Bye", "Cya", "Goodbye")
        else:
            start = choice("Hi", "Hello", "Hey", "'sup")
        middle = choice(user.name, user.display_name)
        if name in ("bye", "cya", "goodbye"):
            end = choice("", "See you soon!", "Have a good one!", "Later!", "Talk to you again sometime!", "Was nice talking to you!")
        else:
            end = choice("", "How are you?", "Can I help you?", "What can I do for you today?", "Nice to see you!", "Great to see you!", "Always good to see you!")
        out = "👋 " + start + ", `" + middle + "`!"
        if end:
            out += " " + end
        return out


class Perms(Command):
    server_only = True
    name = ["ChangePerms", "Perm", "ChangePerm", "Permissions"]
    description = "Shows or changes a user's permission level."
    usage = "<0:users>* <1:new_level>? <default{?d}>? <hide{?h}>?"
    flags = "fhd"
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, user, perm, channel, guild, flags, **void):
        # Get target user from first argument
        users = await bot.find_users(argl, args, user, guild, roles=True)
        if not users:
            raise LookupError("No results found.")
        for t_user in users:
            t_perm = round_min(bot.get_perms(t_user.id, guild))
            # If permission level is given, attempt to change permission level, otherwise get current permission level
            if args or "d" in flags:
                name = str(t_user)
                if "d" in flags:
                    o_perm = round_min(bot.get_perms(t_user.id, guild))
                    bot.remove_perms(t_user.id, guild)
                    c_perm = round_min(bot.get_perms(t_user.id, guild))
                    m_perm = max(abs(t_perm), abs(c_perm), 2) + 1
                    if not perm < m_perm and not isnan(m_perm):
                        create_task(channel.send(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(t_perm)} to the default value of {sqr_md(c_perm)}.")))
                        continue
                    reason = f"to change permissions for {name} in {guild} from {t_perm} to {c_perm}"
                    bot.set_perms(t_user.id, guild, o_perm)
                    raise self.perm_error(perm, m_perm, reason)
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
    min_display = "0~3"
    description = "Shows, enables, or disables a command category in the current channel."
    usage = "(enable|disable)? <category>? <list{?l}>? <hide{?h}>?"
    flags = "aedlh"
    slash = True

    def __call__(self, argv, args, flags, user, channel, guild, perm, **void):
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
                if "h" in flags:
                    return
                return css_md(f"Enabled all standard command categories in {sqr_md(channel)}.")
            if "d" in flags:
                enabled[channel.id] = set()
                if "h" in flags:
                    return
                return css_md(f"Disabled all standard command categories in {sqr_md(channel)}.")
            temp = enabled.get(channel.id, default_commands)
            if not temp:
                return ini_md(f"No currently enabled commands in {sqr_md(channel)}.")
            return f"Currently enabled command categories in {channel_mention(channel.id)}:\n{ini_md(iter2str(temp))}"
        if "e" not in flags and "a" not in flags and "d" not in flags:
            catg = argv.casefold()
            if not bot.is_trusted(guild) and catg not in standard_commands:
                raise PermissionError(f"Elevated server priviliges required for specified command category.")
            if catg not in bot.categories:
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
                    update(channel.id)
            else:
                if catg in curr:
                    curr.remove(catg)
                    update(channel.id)
        check = curr if type(curr) is set else frozenset(curr)
        if check == default_commands:
            enabled.pop(channel.id)
        if "h" in flags:
            return
        category = "category" if len(args) == 1 else "categories"
        action = "Enabled" if "d" not in flags else "Disabled"
        return css_md(f"{action} command {category} {sqr_md(', '.join(args))} in {sqr_md(channel)}.")


class Prefix(Command):
    name = ["ChangePrefix"]
    min_display = "0~3"
    description = "Shows or changes the prefix for ⟨MIZA⟩'s commands for this server."
    usage = "<new_prefix>? <default{?d}>?"
    flags = "hd"
    umap = {c: "" for c in ZeroEnc}
    umap["\u200a"] = ""
    utrans = "".maketrans(umap)
    slash = True

    def __call__(self, argv, guild, perm, bot, flags, **void):
        pref = bot.data.prefixes
        update = self.data.prefixes.update
        if "d" in flags:
            if guild.id in pref:
                pref.pop(guild.id)
            return css_md(f"Successfully reset command prefix for {sqr_md(guild)}.")
        if not argv:
            return css_md(f"Current command prefix for {sqr_md(guild)}: {sqr_md(bot.get_prefix(guild))}.")
        req = 3
        if perm < req:
            reason = f"to change command prefix for {guild}"
            raise self.perm_error(perm, req, reason)
        prefix = argv
        if not prefix.isalnum():
            prefix = prefix.translate(self.utrans)
        # Backslash is not allowed, it is used to escape commands normally
        if prefix.startswith("\\"):
            raise TypeError("Prefix must not begin with backslash.")
        pref[guild.id] = prefix
        if "h" not in flags:
            return css_md(f"Successfully changed command prefix for {sqr_md(guild)} to {sqr_md(argv)}.")


class Loop(Command):
    time_consuming = 3
    _timeout_ = 8
    name = ["For", "Rep", "While"]
    min_level = 1
    min_display = "1+"
    description = "Loops a command."
    usage = "<0:iterations> <1:command>+"
    rate_limit = (3, 7)

    async def __call__(self, args, argv, message, channel, bot, perm, user, guild, **void):
        if not args:
            # Ah yes, I made this error specifically for people trying to use this command to loop songs 🙃
            raise ArgumentError("Please input loop iterations and target command. For looping songs in voice, consider using the aliases LoopQueue and Repeat under the AudioSettings command.")
        num = await bot.eval_math(args[0], user)
        iters = round(num)
        # Bot owner bypasses restrictions
        if not isnan(perm):
            if iters > 32 and not bot.is_trusted(guild.id):
                raise PermissionError(f"Elevated server priviliges required to execute loop of greater than 32 iterations.")
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
        func = func.lstrip()
        if not isnan(perm):
            # Detects when an attempt is made to loop the loop command
            for n in self.name:
                if (
                    (bot.get_prefix(guild) + n).upper() in func.replace(" ", "").upper()
                ) or (
                    (str(bot.id) + ">" + n).upper() in func.replace(" ", "").upper()
                ):
                    raise PermissionError("Must be owner to execute nested loop.")
        func2 = func2.split(None, 1)[-1]
        create_task(send_with_react(
            channel,
            italics(css_md(f"Looping {sqr_md(func)} {iters} time{'s' if iters != 1 else ''}...")),
            reacts=["❎"],
            reference=message,
        ))
        fake_message = copy.copy(message)
        fake_message.content = func2
        for i in range(iters):
            loop = i < iters - 1
            t = utc()
            # Calls process_message with the argument containing the looped command.
            delay = await bot.process_message(fake_message, func, loop=loop)
            # Must abide by command rate limit rules
            delay = delay + t - utc()
            if delay > 0:
                await asyncio.sleep(delay)


class Avatar(Command):
    name = ["PFP", "Icon"]
    description = "Sends a link to the avatar of a user or server."
    usage = "<objects>*"
    multi = True
    slash = True

    async def getGuildData(self, g):
        # Gets icon display of a server and returns as an embed.
        url = to_png(g.icon_url)
        name = g.name
        colour = await self.bot.data.colours.get(to_png_ex(g.icon_url))
        emb = discord.Embed(colour=colour)
        emb.set_thumbnail(url=url)
        emb.set_image(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        emb.description = f"{sqr_md(name)}({url})"
        return emb

    async def getMimicData(self, p):
        # Gets icon display of a mimic and returns as an embed.
        url = to_png(p.url)
        name = p.name
        colour = await self.bot.data.colours.get(to_png_ex(p.url))
        emb = discord.Embed(colour=colour)
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
                        if is_url(argv) or argv.startswith("discord.gg/"):
                            g = await bot.fetch_guild(argv)
                            emb = await self.getGuildData(g, flags)
                            embs.add(emb)
                            raise StopIteration
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
                                emb = await self.getMimicData(p)
                                embs.add(emb)
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
                    colour = await self.bot.get_colour(u)
                    emb = discord.Embed(colour=colour)
                    emb.set_thumbnail(url=url)
                    emb.set_image(url=url)
                    emb.set_author(name=name, icon_url=url, url=url)
                    emb.description = f"{sqr_md(name)}({url})"
                    embs.add(emb)
        bot.send_embeds(channel, embeds=embs)


class Info(Command):
    name = ["🔍", "🔎", "UserInfo", "ServerInfo", "WhoIs"]
    description = "Shows information about the target user or server."
    usage = "<objects>* <verbose{?v}>?"
    flags = "v"
    rate_limit = 1
    multi = True
    slash = True

    async def getGuildData(self, g, flags={}):
        bot = self.bot
        url = to_png(g.icon_url)
        name = g.name
        try:
            u = g.owner
        except (AttributeError, KeyError):
            u = None
        colour = await self.bot.data.colours.get(to_png_ex(g.icon_url))
        emb = discord.Embed(colour=colour)
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
        if not hasattr(g, "ghost"):
            pcount = await bot.data.counts.getUserMessages(None, g)
        else:
            pcount = None
        if not hasattr(g, "ghost"):
            with suppress(AttributeError, KeyError):
                # Top users by message counts
                pavg = await bot.data.counts.getUserAverage(None, g)
                users = deque()
                us = await bot.data.counts.getGuildMessages(g)
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
        emb.add_field(name="Creation time", value=str(g.created_at) + "\n" + dyn_time_diff(utc_dt().timestamp(), g.created_at.timestamp()) + " ago", inline=1)
        if "v" in flags:
            with suppress(AttributeError, KeyError):
                emb.add_field(name="Region", value=str(g.region), inline=1)
                emb.add_field(name="Nitro boosts", value=str(g.premium_subscription_count), inline=1)
        with suppress(AttributeError):
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

    async def getMimicData(self, p, flags={}):
        url = to_png(p.url)
        name = p.name
        colour = await self.bot.data.colours.get(to_png_ex(p.url))
        emb = discord.Embed(colour=url)
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
        emb.add_field(name="Creation time", value=str(datetime.datetime.fromtimestamp(p.created_at)) + "\n" + dyn_time_diff(utc_dt().timestamp(), p.created_at) + " ago", inline=1)
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
                        if is_url(argv) or argv.startswith("discord.gg/"):
                            g = await bot.fetch_guild(argv)
                            emb = await self.getGuildData(g, flags)
                            embs.add(emb)
                            raise StopIteration
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
                                emb = await self.getMimicData(p, flags)
                                embs.add(emb)
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
                    if member:
                        dname = getattr(member, "nick", None)
                        joined = getattr(u, "joined_at", None)
                    else:
                        dname = getattr(u, "nick", None)
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
                            seen = f"{dyn_time_diff(ts, min(ts, ls))} ago"
                        if la:
                            seen = f"{la}, {seen}"
                        if "v" in flags:
                            tz = bot.data.users.estimate_timezone(u.id)
                            if tz >= 0:
                                zone = f"GMT+{tz}"
                            else:
                                zone = f"GMT{tz}"
                    with suppress(LookupError):
                        old = bot.data.counts.get(guild.id, {})["oldest"][u.id]
                        old = snowflake_time(old)
                    if "v" in flags:
                        with suppress(LookupError):
                            if not hasattr(guild, "ghost"):
                                gmsg = bot.data.counts.getUserGlobalMessageCount(u)
                                msgs = await bot.data.counts.getUserMessages(u, guild)
                    with suppress(LookupError):
                        if not hasattr(guild, "ghost"):
                            us = await bot.data.counts.getGuildMessages(guild)
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
                    if is_self and bot.github is not None:
                        url2 = bot.github
                    else:
                        url2 = url
                    colour = await self.bot.get_colour(u)
                    emb = discord.Embed(colour=colour)
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
                    emb.add_field(name="Creation time", value=str(created) + "\n" + dyn_time_diff(utc_dt().timestamp(), created.timestamp()) + " ago", inline=1)
                    if joined:
                        emb.add_field(name="Join time", value=str(joined) + "\n" + dyn_time_diff(utc_dt().timestamp(), joined.timestamp()) + " ago", inline=1)
                    if old:
                        emb.add_field(name="Oldest post time", value=str(old) + "\n" + dyn_time_diff(utc_dt().timestamp(), old.timestamp()) + " ago", inline=1)
                    if status:
                        emb.add_field(name="Status", value=str(status), inline=1)
                    if zone:
                        emb.add_field(name="Estimated timezone", value=str(zone), inline=1)
                    if seen:
                        emb.add_field(name="Last seen", value=str(seen), inline=1)
                    if dname:
                        emb.add_field(name="Nickname", value=dname, inline=1)
                    if msgs:
                        emb.add_field(name="Post count", value=str(msgs), inline=1)
                    if pos:
                        emb.add_field(name="Server rank", value=str(pos), inline=1)
                    if role:
                        emb.add_field(name=f"Roles ({len(rolelist)})", value=role, inline=0)
                    embs.add(emb)
        bot.send_embeds(channel, embeds=embs)


class Insights(Command):
    name = ["📊", "UserInsights", "ServerInsights"]
    description = "Shows statistics relevant to the target user or server."
    usage = "<objects>*"
    rate_limit = 1
    multi = True
    slash = True

    async def getGuildData(self, g, flags={}):
        bot = self.bot
        url = to_png(g.icon_url)
        name = g.name
        try:
            u = g.owner
        except (AttributeError, KeyError):
            u = None
        colour = await self.bot.data.colours.get(to_png_ex(g.icon_url))
        emb = discord.Embed(colour=colour)
        emb.set_thumbnail(url=url)
        emb.set_author(name=name, icon_url=url, url=url)
        if u is not None:
            d = user_mention(u.id)
        else:
            d = ""
        if g.description:
            d += code_md(g.description)
        emb.description = d
        pcount = actv = topc = top = None
        if not hasattr(g, "ghost"):
            pcount = await bot.data.counts.getUserMessages(None, g)
            pavg = await bot.data.counts.getUserAverage(None, g)
            ptotal = await bot.data.counts.getUserTotal(None, g)
            pword = await bot.data.counts.getUserWords(None, g)
            activity = await create_future(bot.data.users.get_events, g.id, interval=3600)
            actv = int(np.sum(activity))
            actc = cdict()
            for c in guild.text_channels:
                a = await create_future(bot.data.users.get_events, f"#{c.id}", interval=3600)
                actc[c.id] = int(np.sum(a))
            m = max(actc.values())
            mag = (c for c, v in actc.values() if v >= m)
            topc = f"<#{next(mag)}>"
            with suppress(AttributeError, KeyError):
                # Top users by message counts
                users = deque()
                us = await bot.data.counts.getGuildMessages(g)
                if type(us) is str:
                    top = us
                else:
                    ul = sorted(
                        us,
                        key=lambda k: us[k],
                        reverse=True,
                    )
                    for i in range(min(32, flags.get("v", 0) * 5 + 10, len(us))):
                        u_id = ul[i]
                        users.append(f"{user_mention(u_id)}: {us[u_id]}")
                    top = "\n".join(users)
        emb.add_field(name="Server ID", value=str(g.id), inline=0)
        emb.add_field(name="Creation time", value=str(g.created_at) + "\n" + dyn_time_diff(utc_dt().timestamp(), g.created_at.timestamp()) + " ago", inline=1)
        emb.add_field(name="Member count", value=str(g.member_count), inline=1)
        if pcount:
            emb.add_field(name="Post count", value=str(pcount), inline=1)
            emb.add_field(name="Average post length", value=str(round(pavg, 9)), inline=1)
            emb.add_field(name="Total posted text", value=str(ptotal), inline=1)
            emb.add_field(name="Average word count", value=str(pword), inline=1)
        if actv:
            emb.add_field(name="Activity (last 2 weeks)", value=str(actv), inline=1)
        if topc:
            emb.add_field(name="Most active channel (last 2 weeks)", value=topc, inline=1)
        if top is not None:
            emb.add_field(name="Top users", value=top, inline=0)
        return emb

    async def __call__(self, argv, argl, name, guild, channel, bot, user, flags, **void):
        iterator = argl if argl else (argv,)
        embs = set()
        for argv in iterator:
            with ExceptionSender(channel):
                with suppress(StopIteration):
                    if argv:
                        if is_url(argv) or argv.startswith("discord.gg/"):
                            g = await bot.fetch_guild(argv)
                            emb = await self.getGuildData(g, flags)
                            embs.add(emb)
                            raise StopIteration
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
                    raise NotImplementedError
                    # member = guild.get_member(u.id)
                    # name = str(u)
                    # url = best_url(u)
                    # if u.id == bot.id:
                    #     is_self = True
                    # else:
                    #     is_self = False
                    # dname = getattr(member, "nick", None)
                    # if member:
                    #     joined = getattr(u, "joined_at", None)
                    # else:
                    #     joined = None
                    # created = u.created_at
                    # status = None
                    # if getattr(u, "status", None):
                    #     # Show up to 3 different statuses based on the user's desktop, web and mobile status.
                    #     if not is_self:
                    #         status_items = [(u.desktop_status, "🖥️"), (u.web_status, "🕸️"), (u.mobile_status, "📱")]
                    #     else:
                    #         status_items = [(bot.statuses[(i + bot.status_iter) % 3], x) for i, x in enumerate(("🖥️", "🕸️", "📱"))]
                    #     ordered = sorted(status_items, key=lambda x: status_order.index(x[0]))
                    #     for s, i in ordered:
                    #         if s == discord.Status.offline:
                    #             ls = bot.data.users.get(u.id, {}).get("last_seen", 0)
                    #             if utc() - ls < 300 and ls > bot.data.users.get(u.id, {}).get("last_offline", 0):
                    #                 s = discord.Status.invisible
                    #         icon = status_icon[s]
                    #         if not status:
                    #             s_ = u.status
                    #             if s != s_ and s == discord.Status.offline:
                    #                 status = status_text[s_]  + " `" + status_icon[s_] + "❓"
                    #                 if s not in (discord.Status.offline, discord.Status.invisible):
                    #                     status += icon
                    #             else:
                    #                 status = status_text[s] + " `" + icon
                    #         if s not in (discord.Status.offline, discord.Status.invisible):
                    #             if icon not in status:
                    #                 status += icon
                    #             status += i
                    #     if status:
                    #         status += "`"
                    #         if len(status) >= 16:
                    #             status = status.replace(" ", "\n")
                    # if member:
                    #     rolelist = [role_mention(i.id) for i in reversed(getattr(u, "roles", ())) if not i.is_default()]
                    #     role = ", ".join(rolelist)
                    # else:
                    #     role = None
                    # coms = seen = msgs = avgs = gmsg = old = 0
                    # fav = None
                    # pos = None
                    # zone = None
                    # with suppress(LookupError):
                    #     ts = utc()
                    #     ls = bot.data.users[u.id]["last_seen"]
                    #     la = bot.data.users[u.id].get("last_action")
                    #     if type(ls) is str:
                    #         seen = ls
                    #     else:
                    #         seen = f"{dyn_time_diff(ts, min(ts, ls))} ago"
                    #     if la:
                    #         seen = f"{la}, {seen}"
                    #     tz = bot.data.users.estimate_timezone(u.id)
                    #     if tz >= 0:
                    #         zone = f"GMT+{tz}"
                    #     else:
                    #         zone = f"GMT{tz}"
                    # with suppress(LookupError):
                    #     old = bot.data.counts.get(guild.id, {})["oldest"][u.id]
                    #     old = snowflake_time(old)
                    # with suppress(LookupError):
                    #     if not hasattr(guild, "ghost"):
                    #         gmsg = bot.data.counts.getUserGlobalMessageCount(u)
                    #         msgs = await bot.data.counts.getUserMessages(u, guild)
                    # with suppress(LookupError):
                    #     if not hasattr(guild, "ghost"):
                    #         us = await bot.data.counts.getGuildMessages(guild)
                    #         if type(us) is str:
                    #             pos = us
                    #         else:
                    #             ul = sorted(
                    #                 us,
                    #                 key=lambda k: us[k],
                    #                 reverse=True,
                    #             )
                    #             try:
                    #                 i = ul.index(u.id)
                    #                 while i >= 1 and us[ul[i - 1]] == us[ul[i]]:
                    #                     i -= 1
                    #                 pos = i + 1
                    #             except ValueError:
                    #                 if joined:
                    #                     pos = len(ul) + 1
                    # if is_self and bot.github is not None:
                    #     url2 = bot.github
                    # else:
                    #     url2 = url
                    # colour = await self.bot.get_colour(u)
                    # emb = discord.Embed(colour=colour)
                    # emb.set_thumbnail(url=url)
                    # emb.set_author(name=name, icon_url=url, url=url2)
                    # d = user_mention(u.id)
                    # if activity:
                    #     d += "\n" + italics(code_md(activity))
                    # if st:
                    #     if d[-1] == "*":
                    #         d += " "
                    #     d += " **```css\n"
                    #     if st:
                    #         d += "\n".join(st)
                    #     d += "```**"
                    # emb.description = d
                    # emb.add_field(name="User ID", value="`" + str(u.id) + "`", inline=0)
                    # emb.add_field(name="Creation time", value=str(created) + "\n" + dyn_time_diff(utc_dt().timestamp(), created.timestamp()) + " ago", inline=1)
                    # if joined:
                    #     emb.add_field(name="Join time", value=str(joined) + "\n" + dyn_time_diff(utc_dt().timestamp(), joined.timestamp()) + " ago", inline=1)
                    # if old:
                    #     emb.add_field(name="Oldest post time", value=str(old) + "\n" + dyn_time_diff(utc_dt().timestamp(), old.timestamp()) + " ago", inline=1)
                    # if status:
                    #     emb.add_field(name="Status", value=str(status), inline=1)
                    # if zone:
                    #     emb.add_field(name="Estimated timezone", value=str(zone), inline=1)
                    # if seen:
                    #     emb.add_field(name="Last seen", value=str(seen), inline=1)
                    # if dname:
                    #     emb.add_field(name="Nickname", value=dname, inline=1)
                    # if msgs:
                    #     emb.add_field(name="Post count", value=str(msgs), inline=1)
                    # if pos:
                    #     emb.add_field(name="Server rank", value=str(pos), inline=1)
                    # if role:
                    #     emb.add_field(name=f"Roles ({len(rolelist)})", value=role, inline=0)
                    # embs.add(emb)
        bot.send_embeds(channel, embeds=embs)


class Profile(Command):
    name = ["User", "UserProfile"]
    description = "Shows or edits a user profile on ⟨MIZA⟩."
    usage = "(user|description|timezone|birthday)? <value>? <delete{?d}>?"
    flags = "d"
    rate_limit = 1
    no_parse = True
    slash = True
    
    async def __call__(self, user, args, argv, flags, channel, guild, bot, **void):
        setting = None
        if not args:
            target = user
        elif args[0] in ("description", "timezone", "time", "birthday"):
            target = user
            setting = args.pop(0)
            if not args:
                value = None
            else:
                value = argv[len(setting) + 1:]
        else:
            target = await bot.fetch_user_member(" ".join(args), guild)
        if setting is None:
            profile = bot.data.users.get(target.id, EMPTY)
            description = profile.get("description", "")
            birthday = profile.get("birthday")
            timezone = profile.get("timezone")
            if timezone:
                td = datetime.timedelta(seconds=as_timezone(timezone))
                description += ini_md(f"Current time: {sqr_md(utc_dt() + td)}")
            if birthday:
                if type(birthday) is not DynamicDT:
                    birthday = profile["birthday"] = DynamicDT.fromdatetime(birthday)
                    bot.data.users.update(target.id)
                t = utc_dt()
                if timezone:
                    birthday -= td
                description += ini_md(f"Age: {sqr_md(time_diff(t, birthday))}\nBirthday in: {sqr_md(time_diff(next_date(birthday), t))}")
            fields = set()
            for field in ("timezone", "birthday"):
                value = profile.get(field)
                if type(value) is DynamicDT:
                    value = value.as_date()
                elif field == "timezone" and value is not None:
                    value = timezone_repr(value)
                fields.add((field, value, False))
            return bot.send_as_embeds(channel, description, fields=fields, author=get_author(target))
        if value is None:
            return ini_md(f"Currently set {setting} for {sqr_md(user)}: {sqr_md(bot.data.users.get(user.id, EMPTY).get(setting))}.")
        if setting != "description" and value.casefold() in ("undefined", "remove", "rem", "reset", "unset", "delete", "clear", "null", "none") or "d" in flags:
            profile.pop(setting, None)
            bot.data.users.update(user.id)
            return css_md(f"Successfully removed {setting} for {sqr_md(user)}.")
        if setting == "description":
            if len(value) > 1024:
                raise OverflowError("Description must be 1024 or fewer in length.")
        elif setting.startswith("time"):
            value = value.casefold()
            try:
                as_timezone(value)
            except KeyError:
                raise ArgumentError(f"Entered value could not be recognized as a timezone location or abbreviation. Use {bot.get_prefix(guild)}timezone list for list.")
        else:
            dt = tzparse(value)
            offs, year = divmod(dt.year, 400)
            value = DynamicDT(year + 2000, dt.month, dt.day).set_offset(offs * 400 - 2000)
        bot.data.users.setdefault(user.id, {})[setting] = value
        bot.data.users.update(user.id)
        if type(value) is DynamicDT:
            value = value.as_date()
        elif setting.startswith("time") and value is not None:
            value = timezone_repr(value)
        return css_md(f"Successfully changed {setting} for {sqr_md(user)} to {sqr_md(value)}.")


class Activity(Command):
    name = ["Recent", "Log"]
    description = "Shows recent Discord activity for the targeted user, server, or channel."
    usage = "<user>? <verbose{?v}>?"
    flags="v"
    rate_limit = (2, 9)
    typing = True
    slash = True

    async def __call__(self, guild, user, argv, flags, channel, bot, _timeout, **void):
        u_id = None
        if argv:
            user = None
            if not regexp("^<#[0-9]+>$").match(argv):
                with suppress():
                    user = bot.cache.guilds[int(argv)]
            if user is None:
                try:
                    user = bot.cache.channels[verify_id(argv)]
                except:
                    user = await bot.fetch_user_member(argv, guild)
                else:
                    u_id = f"#{user.id}"
        if not u_id:
            u_id = user.id
        data = await create_future(bot.data.users.fetch_events, u_id, interval=max(900, 3600 >> flags.get("v", 0)), timeout=_timeout)
        with discord.context_managers.Typing(channel):
            resp = await process_image("plt_special", "$", (data, str(user)), guild)
            fn = resp[0]
            f = CompatFile(fn, filename=f"{user.id}.png")
        return dict(file=f, filename=fn, best=True)


class Status(Command):
    name = ["State", "Ping"]
    description = "Shows the bot's current internal program state."
    usage = "(enable|disable)?"
    flags = "aed"
    slash = True

    async def __call__(self, perm, flags, channel, bot, **void):
        if "d" in flags:
            if perm < 2:
                raise PermissionError("Permission level 2 or higher required to unset auto-updating status.")
            bot.data.messages.pop(channel.id)
            bot.data.messages.update(channel.id)
            return fix_md("Successfully disabled status updates.")
        elif "a" not in flags and "e" not in flags:
            return await self._callback2_(channel)
        if perm < 2:
            raise PermissionError("Permission level 2 or higher required to set auto-updating status.")
        message = await channel.send(italics(code_md("Loading bot status...")))
        set_dict(bot.data.messages, channel.id, {})[message.id] = cdict(t=0, command="bot.commands.status[0]")
        bot.data.messages.update(channel.id)
    
    async def _callback2_(self, channel, m_id=None, msg=None, **void):
        bot = self.bot
        if not hasattr(bot, "bitrate"):
            return
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(name="Status", url=bot.webserver, icon_url=best_url(bot.user))
        emb.timestamp = utc_dt()
        if msg is None:
            active = bot.get_active()
            try:
                shards = len(bot.latencies)
            except AttributeError:
                shards = 1
            size = sum(bot.size.values()) + sum(bot.size2.values())
            stats = bot.curr_state

            bot_info = (
                f"Process count\n`{active[0]}`\nThread count\n`{active[1]}`\nCoroutine count\n`{active[2]}`\n"
                + f"CPU usage\n`{round(stats[0], 3)}%`\nRAM usage\n`{byte_scale(stats[1])}B`\nDisk usage\n`{byte_scale(stats[2])}B`\nNetwork usage\n`{byte_scale(bot.bitrate)}bps`"
            )
            emb.add_field(name="Bot info", value=bot_info)

            discord_info = (
                f"Shard count\n`{shards}`\nServer count\n`{len(bot.guilds)}`\nUser count\n`{len(bot.cache.users)}`\n"
                + f"Channel count\n`{len(bot.cache.channels)}`\nRole count\n`{len(bot.cache.roles)}`\nEmoji count\n`{len(bot.cache.emojis)}`\nCached messages\n`{len(bot.cache.messages)}`"
            )
            emb.add_field(name="Discord info", value=discord_info)

            misc_info = (
                f"Cached files\n`{bot.file_count}`\nConnected voice channels\n`{len(bot.voice_clients)}`\nTotal data sent/received\n`{byte_scale(bot.total_bytes)}B`\n"
                + f"System time\n`{datetime.datetime.now()}`\nAPI latency\n`{sec2time(bot.api_latency)}`\nCurrent uptime\n`{dyn_time_diff(utc(), bot.start_time)}`\nPublic IP address\n`{bot.ip}`"
            )
            emb.add_field(name="Misc info", value=misc_info)
            commands = set()
            for command in bot.commands.values():
                commands.update(command)
            emb.add_field(name="Code info", value=f"Code size\n[`{byte_scale(size[0])}B, {size[1]} lines`]({bot.github})\nCommand count\n[`{len(commands)}`](https://github.com/thomas-xin/Miza/wiki/Commands)")
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


class Invite(Command):
    name = ["Website", "BotInfo", "InviteLink"]
    description = "Sends a link to ⟨MIZA⟩'s homepage, github and invite code, as well as an invite link to the current server if applicable."
    slash = True
    
    async def __call__(self, channel, message, **void):
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(**get_author(self.bot.user))
        emb.description = f"[**`My Github`**]({self.bot.github}) [**`My Website`**]({self.bot.webserver}) [**`My Invite`**]({self.bot.invite})"
        if message.guild:
            with tracebacksuppressor:
                member = message.guild.get_member(self.bot.id)
                if member.guild_permissions.create_instant_invite:
                    invites = await member.guild.invites()
                    if not invites:
                        channel = await self.bot.get_first_sendable(member.guild, member)
                        invite = await channel.create_invite(reason="Invite command")
                    else:
                        invite = sorted(invites, key=lambda invite: (invite.max_age == 0, invite.max_uses - invite.uses != 0, len(invite.url)))[0]
                    emb.description += f" [**`Server Invite`**]({invite.url})"
        self.bot.send_embeds(channel, embed=emb, reference=message)


class Reminder(Command):
    name = ["Announcement", "Announcements", "Announce", "RemindMe", "Reminders", "Remind"]
    description = "Sets a reminder for a certain date and time."
    usage = "<1:message>? <0:time>? <urgent{?u}>? <delete{?d}>?"
    flags = "aedu"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1 / 3, 4)
    keywords = ["on", "at", "in", "when", "event"]
    keydict = {re.compile(f"(^|[^a-z0-9]){i[::-1]}([^a-z0-9]|$)", re.I): None for i in keywords}
    timefind = None
    slash = True

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
        update = bot.data.reminders.update
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
                    bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
                if rems:
                    bot.data.reminders.listed.insort((rems[0]["t"], sendable.id), key=lambda x: x[0])
            update(sendable.id)
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
        for f in "aeu":
            if f in flags:
                for c in (f, f.upper()):
                    for q in "?-+":
                        argv = argv.replace(q + c + " ", "").replace(" " + q + c, "")
        urgent = "u" in flags
        remind_as = user
        # This parser is so unnecessarily long for what it does...
        keyed = False
        while True:
            argv = argv.replace("tomorrow at", "at tomorrow").replace("today at", "at today").replace("yesterday at", "at yesterday")
            temp = argv.casefold()
            if name == "remind" and temp.startswith("me "):
                argv = argv[3:]
                temp = argv.casefold()
            if temp.startswith("urgently ") or temp.startswith("urgent "):
                argv = argv.split(" ", 1)[1]
                temp = argv.casefold()
                urgent = True
            if temp.startswith("as ") and " " in argv[3:]:
                query, argv = argv[3:].split(" ", 1)
                remind_as = await self.bot.fetch_user_member(query, guild)
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
                    spl = argv.rsplit(" event ", 1)
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
                    spl = argv.rsplit(" when ", 1)
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
                    spl = argv.rsplit(" in ", 1)
                elif temp.startswith("in "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " in ".join(spl[:-1])
                    t = await bot.eval_time(spl[-1], user)
                    break
            if foundkey.get("at"):
                if " at " in argv:
                    spl = argv.rsplit(" at ", 1)
                elif temp.startswith("at "):
                    spl = [argv[3:]]
                    msg = ""
                if spl is not None:
                    msg = " at ".join(spl[:-1])
                    t = utc_ts(tzparse(spl[-1])) - utc()
                    break
            if foundkey.get("on"):
                if " on " in argv:
                    spl = argv.rsplit(" on ", 1)
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
                    for i in (0, -1):
                        arg = args[i]
                        with suppress(KeyError):
                            t = as_timezone(arg)
                            args.pop(i)
                            expr = " ".join(args)
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
                for i in (0, -1):
                    arg = args[i]
                    with suppress(KeyError):
                        t = as_timezone(arg)
                        args.pop(i)
                        expr = " ".join(args)
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
            if urgent:
                msg = bold(css_md(msg))
            else:
                msg = bold(ini_md(msg))
        elif len(msg) > 2048:
            raise OverflowError(f"Input message too long ({len(msg)} > 2048).")
        username = str(remind_as)
        url = best_url(remind_as)
        ts = utc()
        if keyed:
            # Schedule for an event from a user
            rem = cdict(
                user=remind_as.id,
                msg=msg,
                u_id=t,
                t=inf,
            )
            rems.append(rem)
            s = "$" + str(t)
            seq = set_dict(bot.data.reminders, s, deque())
            seq.append(sendable.id)
        else:
            # Schedule for an event at a certain time
            rem = cdict(
                user=remind_as.id,
                msg=msg,
                t=t + ts,
            )
            rems.append(rem)
        if urgent:
            rem.urgent = True
        # Sort list of reminders
        bot.data.reminders[sendable.id] = sort(rems, key=lambda x: x["t"])
        with suppress(IndexError):
            # Remove existing schedule
            bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
        # Insert back into bot schedule
        bot.data.reminders.listed.insort((bot.data.reminders[sendable.id][0]["t"], sendable.id), key=lambda x: x[0])
        update(sendable.id)
        emb = discord.Embed(description=msg)
        emb.set_author(name=username, url=url, icon_url=url)
        out = "```css\nSuccessfully set "
        if urgent:
            out += "urgent "
        if "announce" in name:
            out += f"announcement for {sqr_md(sendable)}"
        else:
            out += f"reminder for {sqr_md(sendable)}"
        if keyed:
            out += f" upon next event from {sqr_md(user_mention(t))}"
        else:
            out += f" in {sqr_md(time_until(t + utc()))}"
        out += ":```"
        return dict(content=out, embed=emb)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos, s_id = [int(i) for i in vals.split("_", 2)]
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
            content += f"{len(rems)} message{'s' if len(rems) != 1 else ''} currently scheduled for {str(sendable).replace('`', '')}:```*"
            msg = iter2str(
                rems[pos:pos + page],
                key=lambda x: lim_str(bot.get_user(x.get("user", -1), replace=True).mention + ": `" + no_md(x["msg"]), 96) + "` ➡️ " + (user_mention(x["u_id"]) if "u_id" in x else time_until(x["t"])),
                left="`[",
                right="]`",
            )
        colour = await self.bot.get_colour(user)
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        ).set_author(**get_author(user))
        more = len(rems) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                async with delay(0.5):
                    create_task(message.add_reaction(react.decode("utf-8")))


class UpdateUrgentReminders(Database):
    name = "urgentreminders"
    no_delete = True

    async def _bot_ready_(self, **void):
        if "listed" not in self.data:
            self.data["listed"] = alist()
        create_task(self.update_urgents())

    async def update_urgents(self):
        while True:
            with tracebacksuppressor:
                t = utc()
                listed = self.data["listed"]
                while listed:
                    p = listed[0]
                    if t < p[0]:
                        break
                    with suppress(StopIteration):
                        listed.popleft()
                        self.update("listed")
                        c_id = p[1]
                        m_id = p[2]
                        emb = p[3]
                        p[0] = utc() + 60
                        channel = await self.bot.fetch_messageable(c_id)
                        message = await self.bot.fetch_message(m_id, channel)
                        for react in message.reactions:
                            if str(react) == "✅":
                                if react.count > 1:
                                    raise StopIteration
                                async for u in react.users():
                                    if u.id != self.bot.id:
                                        raise StopIteration
                        fut = create_task(channel.send(embed=emb))
                        await self.bot.silent_delete(message)
                        message = await fut
                        await message.add_reaction("✅")
                        p[2] = message.id
                        listed.insort(p, key=lambda x: x)
                        self.update("listed")
            await asyncio.sleep(1)


# This database is such a hassle to manage, it has to be able to persist between bot restarts, and has to be able to update with O(1) time complexity when idle
class UpdateReminders(Database):
    name = "reminders"
    no_delete = True

    def __load__(self):
        d = self.data
        # This exists so that checking next scheduled item is O(1)
        self.listed = alist(sorted(((d[i][0]["t"], i) for i in d if type(i) is not str and d[i]), key=lambda x: x[0]))

    async def urgent_message(self, channel, embed):
        t = utc()
        message = await channel.send(embed=embed)
        await message.add_reaction("✅")
        self.bot.data.urgentreminders.data["listed"].insort([t + 60, channel.id, message.id, embed], key=lambda x: x)
        self.bot.data.urgentreminders.update()
        # await asyncio.sleep(t - utc() + 60)
        # with suppress(StopIteration):
        #     while True:
        #         async with delay(60):
        #             for react in self.bot.cache.messages[message.id].reactions:
        #                 if str(react) == "✅":
        #                     if react.count > 1:
        #                         raise StopIteration
        #                     async for u in react.users():
        #                         if u.id != self.bot.id:
        #                             raise StopIteration
        #             fut = create_task(channel.send(embed=embed))
        #             await self.bot.silent_delete(message)
        #             message = await fut
        #             await message.add_reaction("✅")

    # Fast call: runs many times per second
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
            self.update(u_id)
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
            if not x.get("urgent"):
                self.bot.send_embeds(ch, emb)
            else:
                create_task(self.urgent_message(ch, emb))

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
                            if not x.get("urgent"):
                                self.bot.send_embeds(ch, emb)
                            else:
                                create_task(self.urgent_message(ch, emb))
                            pops.add(len(rems) - i)
                        elif is_finite(x["t"]):
                            break
                    it = [rems[i] for i in range(len(rems)) if i not in pops]
                    rems.clear()
                    rems.extend(it)
                    self.update(u_id)
                    if not rems:
                        self.data.pop(u_id)
            with suppress(KeyError):
                self.data.pop(s)


# This database has caused so many rate limit issues
class UpdateMessageCount(Database):
    name = "counts"

    def startCalculate(self, guild):
        self.data[guild.id] = {"counts": {}, "totals": {}, "words": {}, "oldest": {}}
        create_task(self.getUserMessageCount(guild))

    async def getUserMessages(self, user, guild):
        if self.scanned == -1:
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
        if self.scanned == -1:
            if guild.id in self.data:
                with suppress(LookupError):
                    return self.data[guild.id]["counts"]
                return self.data[guild.id]
        return "Calculating..."

    # What are the rate limits for the message history calls?
    async def getChannelHistory(self, channel, limit=None, after=None, callback=None):

        async def delay_if_exc(channel):
            t = utc()
            history = channel.history(limit=limit, after=after, oldest_first=(limit is None))
            try:
                add_message = self.bot.add_message
                out = deque()
                async for m in history:
                    add_message(m)
                    out.append(m)
                return list(out)
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

    async def getGuildHistory(self, guild, limit=None, after=None, callback=None):
        lim = None if limit is None else ceil(limit / min(128, len(guild.text_channels)))
        output = cdict({channel.id: create_task(self.getChannelHistory(channel, limit=lim, after=after, callback=callback)) for channel in guild.text_channels})
        for k, v in output.items():
            output[k] = await v
        return output

    async def getUserMessageCount(self, guild):
        print("Probing", guild)
        data = {}
        avgs = {}
        word = {}
        oldest = self.data[guild.id]["oldest"]
        histories = await self.getGuildHistory(guild)
        for messages in histories.values():
            for i, message in enumerate(messages, 1):
                u = message.author.id
                orig_id = oldest.get(u)
                if not orig_id or message.id < orig_id:
                    oldest[u] = message.id
                length = get_message_length(message)
                words = get_message_words(message)
                try:
                    data[u] += 1
                    avgs[u] += length
                    word[u] += words
                except KeyError:
                    data[u] = 1
                    avgs[u] = length
                    word[u] = words
                if not i & 8191:
                    await asyncio.sleep(0.5)
        add_dict(self.data[guild.id], {"counts": data, "totals": avgs, "words": word, 0: True})
        self.update(guild.id)
        print("Completed", guild)
        print(self.data[guild.id])

    def __load__(self):
        self.scanned = False
        self.semaphore = Semaphore(20, 256, delay=5)

    async def __call__(self):
        if self.scanned:
            return
        self.scanned = True
        guilds = self.bot.guilds
        for guild in sorted(guilds, key=lambda g: g.member_count, reverse=True):
            if guild.id not in self.data or 0 not in self.data[guild.id]:
                self.startCalculate(guild)
        self.scanned = -1

    # Add new messages to the post count database
    def _send_(self, message, **void):
        user = message.author
        guild = message.guild
        if self.scanned and guild.id not in self.data:
            self.startCalculate(guild)
        d = self.data[guild.id]
        if type(d) is str:
            return
        if user.id not in set_dict(d, "oldest", {}):
            d["oldest"][user.id] = message.id
        count = d["counts"].get(user.id, 0) + 1
        total = d["totals"].get(user.id, 0) + get_message_length(message)
        d["totals"][user.id] = total
        d["counts"][user.id] = count
        self.update(guild.id)                


class UpdatePrefix(Database):
    name = "prefixes"


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
        if self.bot.bot_ready and not self.closed:
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



class UpdateFlavour(Database):
    name = "flavour"
    no_delete = True

    async def get(self):
        out = x = None
        i = xrand(7)
        facts = self.bot.data.users.facts
        useless = self.bot.data.users.useless
        if i <= 1 and facts:
            with tracebacksuppressor:
                text = choice(facts)
                fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
                out = f"\n{fact} `{text}`"
        elif i == 2:
            x = "affirmations"
            with tracebacksuppressor:
                if self.data.get(x) and len(self.data[x]) > 64 and xrand(2):
                    return choice(self.data[x])
                data = await Request("https://www.affirmations.dev/", aio=True)
                text = eval_json(data)["affirmation"].replace("`", "")
                out = f"\nAffirmation: `{text}`"
        elif i == 3:
            x = "geek_jokes"
            with tracebacksuppressor:
                if self.data.get(x) and len(self.data[x]) > 64 and xrand(2):
                    return choice(self.data[x])
                data = await Request("https://geek-jokes.sameerkumar.website/api", aio=True)
                text = eval_json(data).replace("`", "")
                out = f"\nGeek joke: `{text}`"
        else:
            x = "useless_facts"
            with tracebacksuppressor:
                if self.data.get(x) and len(self.data[x]) > 256 and xrand(2):
                    return choice(self.data[x])
                if len(useless) < 128 and (not useless or random.random() > 0.75):
                    data = await Request("https://www.uselessfacts.net/api/posts?d=" + str(datetime.datetime.fromtimestamp(xrand(1462456800, utc())).date()), aio=True)
                    factlist = [fact["title"].replace("`", "") for fact in eval_json(data) if "title" in fact]
                    random.shuffle(factlist)
                    useless.clear()
                    for text in factlist:
                        fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
                        out = f"\n{fact} `{text}`"
                        useless.append(out)
                out = useless.popleft()
        if x and out:
            if x in self.data:
                if out not in self.data[x]:
                    self.data[x].add(out)
                    self.update(x)
            else:
                self.data[x] = alist((out,))
        return out


EMPTY = {}

# This database takes up a lot of space, storing so many events from users
class UpdateUsers(Database):
    name = "users"
    no_delete = True
    hours = 168
    interval = 900
    scale = 3600 // interval
    mentionspam = re.compile("<@[!&]?[0-9]+>")

    def __load__(self):
        self.semaphore = Semaphore(1, 2, delay=0.5)
        self.facts = None
        self.flavour_buffer = deque()
        self.flavour_set = set()
        self.flavour = ()
        self.useless = deque()
        with open("misc/facts.txt", "r", encoding="utf-8") as f:
            self.facts = f.read().splitlines()

    async def _bot_ready_(self, **void):
        data = {"Command": Command}
        exec(
            f"class {self.bot.name.replace(' ', '')}(Command):"
            + "\n\tdescription = 'Serves as an alias for mentioning the bot.'"
            + "\n\tno_parse = True"
            + "\n\tasync def __call__(self, message, argv, flags, **void):"
            + "\n\t\tawait self.bot.data.users._nocommand_(message, self.bot.user.mention + ' ' + argv, flags=flags, force=True)",
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
        hour = round_min(int(utc() // self.interval) / self.scale)
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
            return list(repeat(0, int(self.hours / self.interval * interval)))
        hour = round_min(int(utc() // self.interval) / self.scale)
        self.clear_events(data, hour - self.hours)
        start = hour - self.hours
        if event is None:
            out = [np.sum(data.get(i / self.scale + start, EMPTY).values()) for i in range(self.hours * self.scale)]
        else:
            out = [data.get(i / self.scale + start, EMPTY).get(event, 0) for i in range(self.hours * self.scale)]
        if interval != self.interval:
            factor = ceil(interval / self.interval)
            out = [np.sum(out[i:i + factor]) for i in range(0, len(out), factor)]
        return out

    def get_timezone(self, u_id):
        timezone = self.data.get(u_id, EMPTY).get("timezone")
        if timezone is not None:
            return round_min(as_timezone(timezone) / 3600)

    def estimate_timezone(self, u_id):
        data = self.data.get(u_id, EMPTY).get("recent")
        if not data:
            return 0
        hour = round_min(int(utc() // self.interval) / self.scale)
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
                while len(self.flavour_buffer) < 32:
                    out = await self.bot.data.flavour.get()
                    if out:
                        self.flavour_buffer.append(out)
                        self.flavour_set.add(out)
                        changed = True
                amount = len(self.flavour_set)
                if changed and (not amount & amount - 1):
                    self.flavour = tuple(self.flavour_set)

    def _offline_(self, user, **void):
        set_dict(self.data, user.id, {})["last_offline"] = utc()
        self.update(user.id)

    # User seen, add event to activity database
    def _seen_(self, user, delay, event, count=1, raw=None, **void):
        if issubclass(type(user), discord.abc.GuildChannel) or type(user) == discord.DMChannel:
            u_id = "#" + str(user.id)
        else:
            u_id = user.id
        self.send_event(u_id, event, count=count)
        if type(user) in (discord.User, discord.Member):
            add_dict(self.data, {u_id: {"last_seen": 0}})
            self.data[u_id]["last_seen"] = utc() + delay
            self.data[u_id]["last_action"] = raw
        self.update(u_id)

    # User executed command, add to activity database
    def _command_(self, user, loop, command, **void):
        self.send_event(user.id, "command")
        add_dict(self.data, {user.id: {"commands": {command.parse_name(): 1}}})
        self.data[user.id]["last_used"] = utc()
        self.data.get(user.id, EMPTY).pop("last_mention", None)
        if not loop:
            self.add_xp(user, getattr(command, "xp", xrand(6, 14)))
    
    def _send_(self, message, **void):
        user = message.author
        if user.id == self.bot.id or self.bot.get_perms(user, message.guild) <= -inf:
            return
        size = get_message_length(message)
        points = math.sqrt(size) + len(message.content.split())
        if points >= 8:
            typing = self.data.get(user.id, EMPTY).get("last_typing", None)
            if typing is None:
                set_dict(self.data, user.id, {})["last_typing"] = inf
            elif typing >= inf:
                return
            else:
                self.data.get(user.id, EMPTY).pop("last_typing", None)
        else:
            self.data.get(user.id, EMPTY).pop("last_typing", None)
        if not xrand(1000):
            self.add_diamonds(user, points)
            points *= 1000
            create_task(message.add_reaction("✨"))
            print(f"{user} has triggered the rare message bonus in {message.guild}!")
        else:
            self.add_gold(user, points)
        self.add_xp(user, points)
        if "dailies" in self.bot.data:
            self.bot.data.dailies.valid_message(message)

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
            create_task(send_with_react(message.channel, out, reacts="❎", reference=message))
            await bot.seen(user, event="misc", raw="Being naughty")
            add_dict(self.data, {user.id: {"last_mention": 1}})
            self.data[user.id]["last_used"] = t
            self.update(user.id)
            raise CommandCancelledError

    def get_xp(self, user):
        if self.bot.is_blacklisted(user.id):
            return -inf
        if user.id == self.bot.id:
            if self.data.get(self.bot.id, EMPTY).get("xp", 0) != inf:
                set_dict(self.data, self.bot.id, {})["xp"] = inf
                self.data[self.bot.id]["gold"] = inf
                self.data[self.bot.id]["diamonds"] = inf
                self.update(self.bot.id)
            return inf
        return self.data.get(user.id, EMPTY).get("xp", 0)

    def xp_to_level(self, xp):
        if is_finite(xp):
            return int((xp * 3 / 2000) ** (2 / 3)) + 1
        return xp
    
    def xp_to_next(self, level):
        if is_finite(level):
            return ceil(math.sqrt(level - 1) * 1000)
        return level
    
    def xp_required(self, level):
        if is_finite(level):
            return ceil((level - 1) ** 1.5 * 2000 / 3)
        return level

    async def get_balance(self, user):
        data = self.data.get(user.id, EMPTY)
        return await self.bot.as_rewards(data.get("diamonds"), data.get("gold"))

    def add_xp(self, user, amount):
        if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
            add_dict(set_dict(self.data, user.id, {}), {"xp": amount})
            if "dailies" in self.bot.data:
                self.bot.data.dailies.progress_quests(user, "xp", amount)
            self.update(user.id)
    
    def add_gold(self, user, amount):
        if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
            add_dict(set_dict(self.data, user.id, {}), {"gold": amount})
            self.update(user.id)

    def add_diamonds(self, user, amount):
        if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
            add_dict(set_dict(self.data, user.id, {}), {"diamonds": amount})
            if "dailies" in self.bot.data:
                self.bot.data.dailies.progress_quests(user, "diamond", amount)
            self.update(user.id)

    async def _typing_(self, user, **void):
        set_dict(self.data, user.id, {})["last_typing"] = utc()
        self.update(user.id)

    async def _nocommand_(self, message, msg, force=False, flags=(), **void):
        bot = self.bot
        user = message.author
        if force or bot.is_mentioned(message, bot, message.guild):
            if user.bot:
                async for m in self.bot.data.channel_cache.get(message.channel):
                    user = m.author
                    if bot.get_perms(user.id, message.guild) <= -inf:
                        return
                    if not user.bot:
                        break
            send = lambda *args, **kwargs: send_with_reply(message.channel, not flags and message, *args, **kwargs)
            out = None
            count = self.data.get(user.id, EMPTY).get("last_talk", 0)
            # Simulates a randomized conversation
            if count < 5:
                create_task(message.add_reaction("👀"))
            if "?" in msg and "ask" in bot.commands and random.random() > math.atan(count / 16) / 4:
                argv = self.mentionspam.sub("", msg).strip()
                for ask in bot.commands.ask:
                    await ask(message, message.channel, user, argv, flags=flags)
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
                if self.flavour_buffer:
                    out += self.flavour_buffer.popleft()
                else:
                    out += choice(self.flavour)
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
                send = lambda *args, **kwargs: send_with_react(message.channel, *args, reacts="❎", reference=not flags and message, **kwargs)
            add_dict(self.data, {user.id: {"last_talk": 1, "last_mention": 1}})
            self.data[user.id]["last_used"] = utc()
            await send(out)
            await bot.seen(user, event="misc", raw="Talking to me")
            self.add_xp(user, xrand(12, 20))
            if "dailies" in bot.data:
                bot.data.dailies.progress_quests(user, "talk")
        else:
            if not self.data.get(user.id, EMPTY).get("last_mention") and random.random() > 0.6:
                self.data.get(user.id, EMPTY).pop("last_talk", None)
            self.data.get(user.id, EMPTY).pop("last_mention", None)
        self.update(user.id)