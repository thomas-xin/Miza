print = PRINT


class Purge(Command):
    time_consuming = True
    _timeout_ = 16
    name = ["🗑", "Del", "Delete", "Purge_Range"]
    min_level = 3
    description = "Deletes a number of messages from a certain user in current channel."
    usage = "<1:users(bots)|?a>? <0:count(1)>? <ignore{?i}|range{?r}|hide{?h}>*"
    flags = "fiaehr"
    rate_limit = (2, 4)
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, user, channel, name, flags, perm, guild, **void):
        # print(self, bot, args, argl, user, channel, name, flags, perm, guild, void)
        end = None
        if args:
            count = await bot.eval_math(args.pop(-1))
            if args and "r" in flags or "range" in name:
                start = safe_eval(args.pop(-1))
                end = count
                if end < count:
                    start, end = end, start
                start -= 1
                end += 1
        else:
            count = 1
        if "a" in flags:
            uset = universal_set
        elif not argl and not args:
            uset = None
        else:
            users = await bot.find_users(argl, args, user, guild)
            uset = {u.id for u in users}
        if end is None and count <= 0:
            raise ValueError("Please enter a valid amount of messages to delete.")
        if end is None:
            print(count)
        else:
            print(start, end)
        delD = deque()
        if end is None:
            dt = None
            after = utc_dt() - datetime.timedelta(days=14) if "i" not in flags else None
            found = False
            if dt is None or after is None or dt > after:
                async with bot.guild_semaphore:
                    async for m in bot.history(channel, limit=None, before=dt, after=after):
                        found = True
                        dt = m.created_at
                        if uset is None and m.author.bot or uset and m.author.id in uset:
                            delD.append(m)
                            count -= 1
                            if count <= 0:
                                break
        else:
            async with bot.guild_semaphore:
                async for m in bot.history(channel, limit=None, before=end, after=start):
                    if uset is None and m.author.bot or uset and m.author.id in uset:
                        delD.append(m)
        if len(delD) >= 64 and "f" not in flags:
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(delD))} MESSAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True)
        # attempt to bulk delete up to 100 at a time, otherwise delete 1 at a time
        deleted = 0
        delM = alist(delD)
        while len(delM):
            try:
                if hasattr(channel, "delete_messages") and channel.permissions_for(channel.guild.me).manage_messages:
                    dels = delM[:100]
                    # bot.logDelete(dels[-1], -1)
                    await channel.delete_messages(dels)
                    deleted += len(dels)
                    for _ in loop(len(dels)):
                        delM.popleft()
                else:
                    await bot.silent_delete(delM[0], no_log=-1, exc=True)
                    deleted += 1
                    delM.popleft()
            except:
                print_exc()
                for _ in loop(min(5, len(delM))):
                    m = delM.popleft()
                    await bot.silent_delete(m, no_log=-1, exc=True)
                    deleted += 1
        s = italics(css_md(f"Deleted {sqr_md(deleted)} message{'s' if deleted != 1 else ''}!", force=True))
        if getattr(message, "slash", None):
            return s
        if not "h" in flags:
            create_task(send_with_react(
                channel,
                s,
                reacts="❎",
            ))


class Mute(Command):
    server_only = True
    _timeout_ = 16
    name = ["🔇", "Revoke", "Silence", "UnMute"]
    min_level = 3
    min_display = "3+"
    description = "Mutes a user for a certain amount of time, with an optional reason."
    usage = "<0:users>* <1:time>? (reason)? <2:reason>? <hide{?h}>?"
    flags = "fhz"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
    rate_limit = (2, 5)
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, message, channel, guild, flags, perm, user, name, **void):
        if not args and not argl:
            # Set callback message for scrollable list
            buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
            await send_with_reply(
                None,
                message,
                "*```" + "\n" * ("z" in flags) + "callback-admin-mute-"
                + str(user.id) + "_0"
                + "-\nLoading mute list...```*",
                buttons=buttons,
            )
            return
        update = self.bot.data.mutes.update
        ts = utc()
        mutelist = bot.data.mutes.get(guild.id, alist())
        if type(mutelist) is not alist:
            mutelist = bot.data.mutes[guild.id] = alist(mutelist)
        async with discord.context_managers.Typing(channel):
            mutes, glob = await self.getMutes(guild)
            users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        if len(users) > 1 and "f" not in flags:
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True)
        if not args or name == "unmute":
            for user in users:
                try:
                    mute = mutes[user.id]
                except LookupError:
                    create_task(channel.send(ini_md(f"{sqr_md(user)} is currently not muted in {sqr_md(guild)}.")))
                    continue
                if name == "unmute":
                    await self.unmute(guild, user)
                    try:
                        ind = mutelist.search(user.id, key=lambda b: b["u"])
                    except LookupError:
                        pass
                    else:
                        mutelist.pops(ind)["u"]
                        if 0 in ind:
                            with suppress(LookupError):
                                bot.data.mutes.listed.remove(guild.id, key=lambda x: x[-1])
                            if mutelist:
                                bot.data.mutes.listed.insort((mutelist[0]["t"], guild.id), key=lambda x: x[0])
                        update(guild.id)
                    create_task(channel.send(css_md(f"Successfully unmuted {sqr_md(user)} in {sqr_md(guild)}.")))
                    continue
                create_task(channel.send(italics(ini_md(f"Current mute for {sqr_md(user)} in {sqr_md(guild)}: {sqr_md(time_until(mute['t']))}."))))
            return
        # This parser is a mess too
        mutetype = " ".join(args)
        if mutetype.startswith("for "):
            mutetype = mutetype[4:]
        if "for " in mutetype:
            i = mutetype.index("for ")
            expr = mutetype[:i].strip()
            msg = mutetype[i + 4:].strip()
        if "with reason " in mutetype:
            i = mutetype.index("with reason ")
            expr = mutetype[:i].strip()
            msg = mutetype[i + 12:].strip()
        elif "reason " in mutetype:
            i = mutetype.index("reason ")
            expr = mutetype[:i].strip()
            msg = mutetype[i + 7:].strip()
        else:
            expr = mutetype
            msg = None
        _op = None
        for op, at in bot.op.items():
            if expr.startswith(op):
                expr = expr[len(op):].strip()
                _op = at
        num = await bot.eval_time(expr, op=False)
        create_task(message.add_reaction("❗"))
        for user in users:
            p = bot.get_perms(user, guild)
            if not p < 0 and not is_finite(p):
                ex = PermissionError(f"{user} has infinite permission level, and cannot be muted in this server.")
                bot.send_exception(channel, ex)
                continue
            elif not p + 1 <= perm and not isnan(perm):
                reason = "to mute " + str(user) + " in " + guild.name
                ex = self.perm_error(perm, p + 1, reason)
                bot.send_exception(channel, ex)
                continue
            if _op is not None:
                try:
                    mute = mutes[user.id]
                    orig = mute["t"] - ts
                except LookupError:
                    orig = 0
                new = getattr(float(orig), _op)(num)
            else:
                new = num
            create_task(self.createMute(guild, user, reason=msg, length=new, channel=channel, mutes=mutes, glob=glob))

    async def getMutes(self, guild):
        loc = self.bot.data.mutes.get(guild.id)
        # This API call could potentially be replaced with a single init call and a well maintained cache of muted users
        role = await self.bot.data.muteroles.get(guild)
        glob = [cdict(user=member) for member in role.members]
        mutes = {mute.user.id: {"u": mute.user.id, "r": None, "t": inf} for mute in glob}
        if loc:
            for b in tuple(loc):
                if b["u"] not in mutes:
                    loc.pop(b["u"])
                    continue
                mutes[b["u"]]["t"] = b["t"]
                mutes[b["u"]]["c"] = b["c"]
        return mutes, glob

    async def createMute(self, guild, user, reason, length, channel, mutes, glob):
        ts = utc()
        bot = self.bot
        mutelist = set_dict(bot.data.mutes, guild.id, alist())
        update = bot.data.mutes.update
        for m in glob:
            u = m.user
            if user.id == u.id:
                with bot.ExceptionSender(channel):
                    mute = mutes[u.id]
                    # Remove from global schedule, then sort and re-add
                    with suppress(LookupError):
                        mutelist.remove(user.id, key=lambda x: x["u"])
                    with suppress(LookupError):
                        bot.data.mutes.listed.remove(guild.id, key=lambda x: x[-1])
                    mutelist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": mute.get("r"), "x": mute.get("x", ())}, key=lambda x: x["t"])
                    bot.data.mutes.listed.insort((mutelist[0]["t"], guild.id), key=lambda x: x[0])
                    print(mutelist)
                    print(bot.data.mutes.listed)
                    update(guild.id)
                    msg = css_md(f"Updated mute for {sqr_md(user)} from {sqr_md(time_until(mute['t']))} to {sqr_md(time_until(ts + length))}.")
                    await channel.send(msg)
                return
        with bot.ExceptionSender(channel):
            role = await self.bot.data.muteroles.get(guild)
            member = guild.get_member(user.id)
            if member is None:
                roles = ()
            else:
                roles = member.roles[1:]
                await member.edit(roles=[role], reason=reason)
            with suppress(LookupError):
                mutelist.remove(user.id, key=lambda x: x["u"])
            with suppress(LookupError):
                bot.data.mutes.listed.remove(guild.id, key=lambda x: x[-1])
            mutelist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": reason, "x": {r.id for r in roles if not r.managed}}, key=lambda x: x["t"])
            bot.data.mutes.listed.insort((mutelist[0]["t"], guild.id), key=lambda x: x[0])
            print(mutelist)
            print(bot.data.mutes.listed)
            update(guild.id)
            msg = css_md(f"{sqr_md(user)} has been muted in {sqr_md(guild)} for {sqr_md(time_until(ts + length))}. Reason: {sqr_md(reason)}")
            await channel.send(msg)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = list(map(int, vals.split("_", 1)))
        if reaction not in (None, self.directions[-1]) and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        update = self.bot.data.mutes.update
        ts = utc()
        mutelist = bot.data.mutes.get(guild.id, [])
        mutes, glob = await self.getMutes(guild)
        page = 25
        last = max(0, len(mutes) - page)
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
            "callback-admin-mute-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not mutes:
            content += f"Mute list for {str(guild).replace('`', '')} is currently empty.```*"
        else:
            content += f"{len(mutes)} users currently muted in {str(guild).replace('`', '')}:```*"
        emb = discord.Embed(colour=discord.Colour(1))
        emb.description = content
        emb.set_author(**get_author(user))
        for i, mute in enumerate(sorted(mutes.values(), key=lambda x: x["t"])[pos:pos + page]):
            with tracebacksuppressor:
                user = await bot.fetch_user(mute["u"])
                emb.add_field(
                    name=f"{user} ({user.id})",
                    value=f"Duration {italics(single_md(time_until(mute['t'])))}\nReason: {italics(single_md(escape_markdown(str(mute['r']))))}"
                )
        more = len(mutes) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if hasattr(message, "int_token"):
            await bot.ignore_interaction(message)


class Ban(Command):
    server_only = True
    _timeout_ = 16
    name = ["🔨", "Bans", "Unban"]
    min_level = 3
    min_display = "3+"
    description = "Bans a user for a certain amount of time, with an optional reason."
    usage = "<0:users>* <1:time>? (reason)? <2:reason>? <hide{?h}>?"
    flags = "fhz"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
    rate_limit = (2, 5)
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, message, channel, guild, flags, perm, user, name, **void):
        if not args and not argl:
            # Set callback message for scrollable list
            buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
            await send_with_reply(
                None,
                message,
                "*```" + "\n" * ("z" in flags) + "callback-admin-ban-"
                + str(user.id) + "_0"
                + "-\nLoading ban list...```*",
                buttons=buttons,
            )
            return
        update = self.bot.data.bans.update
        ts = utc()
        banlist = bot.data.bans.get(guild.id, alist())
        if type(banlist) is not alist:
            banlist = bot.data.bans[guild.id] = alist(banlist)
        async with discord.context_managers.Typing(channel):
            bans, glob = await self.getBans(guild)
            users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        if len(users) > 1 and "f" not in flags:
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True)
        if not args or name == "unban":
            for user in users:
                try:
                    ban = bans[user.id]
                except LookupError:
                    create_task(channel.send(ini_md(f"{sqr_md(user)} is currently not banned from {sqr_md(guild)}.")))
                    continue
                if name == "unban":
                    await guild.unban(user)
                    try:
                        ind = banlist.search(user.id, key=lambda b: b["u"])
                    except LookupError:
                        pass
                    else:
                        banlist.pops(ind)["u"]
                        if 0 in ind:
                            with suppress(LookupError):
                                bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
                            if banlist:
                                bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
                        update(guild.id)
                    create_task(channel.send(css_md(f"Successfully unbanned {sqr_md(user)} from {sqr_md(guild)}.")))
                    continue
                create_task(channel.send(italics(ini_md(f"Current ban for {sqr_md(user)} from {sqr_md(guild)}: {sqr_md(time_until(ban['t']))}."))))
            return
        # This parser is a mess too
        bantype = " ".join(args)
        if bantype.startswith("for "):
            bantype = bantype[4:]
        if "for " in bantype:
            i = bantype.index("for ")
            expr = bantype[:i].strip()
            msg = bantype[i + 4:].strip()
        if "with reason " in bantype:
            i = bantype.index("with reason ")
            expr = bantype[:i].strip()
            msg = bantype[i + 12:].strip()
        elif "reason " in bantype:
            i = bantype.index("reason ")
            expr = bantype[:i].strip()
            msg = bantype[i + 7:].strip()
        else:
            expr = bantype
            msg = None
        _op = None
        for op, at in bot.op.items():
            if expr.startswith(op):
                expr = expr[len(op):].strip()
                _op = at
        num = await bot.eval_time(expr, op=False)
        create_task(message.add_reaction("❗"))
        for user in users:
            p = bot.get_perms(user, guild)
            if not p < 0 and not is_finite(p):
                ex = PermissionError(f"{user} has infinite permission level, and cannot be banned from this server.")
                bot.send_exception(channel, ex)
                continue
            elif not p + 1 <= perm and not isnan(perm):
                reason = "to ban " + str(user) + " from " + guild.name
                ex = self.perm_error(perm, p + 1, reason)
                bot.send_exception(channel, ex)
                continue
            if _op is not None:
                try:
                    ban = bans[user.id]
                    orig = ban["t"] - ts
                except LookupError:
                    orig = 0
                new = getattr(float(orig), _op)(num)
            else:
                new = num
            create_task(self.createBan(guild, user, reason=msg, length=new, channel=channel, bans=bans, glob=glob))

    async def getBans(self, guild):
        loc = self.bot.data.bans.get(guild.id)
        # This API call could potentially be replaced with a single init call and a well maintained cache of banned users
        glob = await guild.bans()
        bans = {ban.user.id: {"u": ban.user.id, "r": ban.reason, "t": inf} for ban in glob}
        if loc:
            for b in tuple(loc):
                if b["u"] not in bans:
                    loc.pop(b["u"])
                    continue
                bans[b["u"]]["t"] = b["t"]
                bans[b["u"]]["c"] = b["c"]
        return bans, glob

    async def createBan(self, guild, user, reason, length, channel, bans, glob):
        ts = utc()
        bot = self.bot
        banlist = set_dict(bot.data.bans, guild.id, alist())
        update = bot.data.bans.update
        for b in glob:
            u = b.user
            if user.id == u.id:
                with bot.ExceptionSender(channel):
                    ban = bans[u.id]
                    # Remove from global schedule, then sort and re-add
                    with suppress(LookupError):
                        banlist.remove(user.id, key=lambda x: x["u"])
                    with suppress(LookupError):
                        bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
                    if length < inf:
                        banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": ban.get("r")}, key=lambda x: x["t"])
                        bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
                    print(banlist)
                    print(bot.data.bans.listed)
                    update(guild.id)
                    msg = css_md(f"Updated ban for {sqr_md(user)} from {sqr_md(time_until(ban['t']))} to {sqr_md(time_until(ts + length))}.")
                    await channel.send(msg)
                return
        with bot.ExceptionSender(channel):
            await bot.verified_ban(user, guild, reason)
            with suppress(LookupError):
                banlist.remove(user.id, key=lambda x: x["u"])
            with suppress(LookupError):
                bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
            if length < inf:
                banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": reason}, key=lambda x: x["t"])
                bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
            print(banlist)
            print(bot.data.bans.listed)
            update(guild.id)
            msg = css_md(f"{sqr_md(user)} has been banned from {sqr_md(guild)} for {sqr_md(time_until(ts + length))}. Reason: {sqr_md(reason)}")
            await channel.send(msg)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = list(map(int, vals.split("_", 1)))
        if reaction not in (None, self.directions[-1]) and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        update = self.bot.data.bans.update
        ts = utc()
        banlist = bot.data.bans.get(guild.id, [])
        bans, glob = await self.getBans(guild)
        page = 25
        last = max(0, len(bans) - page)
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
            "callback-admin-ban-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not bans:
            content += f"Ban list for {str(guild).replace('`', '')} is currently empty.```*"
        else:
            content += f"{len(bans)} users currently banned from {str(guild).replace('`', '')}:```*"
        emb = discord.Embed(colour=discord.Colour(1))
        emb.description = content
        emb.set_author(**get_author(user))
        for i, ban in enumerate(sorted(bans.values(), key=lambda x: x["t"])[pos:pos + page]):
            with tracebacksuppressor:
                user = await bot.fetch_user(ban["u"])
                emb.add_field(
                    name=f"{user} ({user.id})",
                    value=f"Duration {italics(single_md(time_until(ban['t'])))}\nReason: {italics(single_md(escape_markdown(str(ban['r']))))}"
                )
        more = len(bans) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if hasattr(message, "int_token"):
            await bot.ignore_interaction(message)


class RoleSelect(Command):
    server_only = True
    name = ["ReactionRoles", "RoleButtons", "RoleSelection", "RoleSelector"]
    min_level = 3
    min_display = "3+"
    description = "Creates a message that allows users to self-assign roles from a specified list."
    usage = "<roles>+"
    flags = "ae"
    no_parse = True
    rate_limit = (1, 2)

    async def __call__(self, args, message, guild, user, perm, **void):
        if not args:
            raise ArgumentError("Please input one or more roles by name or ID.")
        roles = deque()
        for arg in args:
            role = None
            r = verify_id(unicode_prune(arg))
            if len(guild.roles) <= 1:
                guild.roles = await guild.fetch_roles()
                guild.roles.sort()
                guild._roles.update((r.id, r) for r in guild.roles)
            if type(r) is int:
                role = guild.get_role(r)
            if not role:
                role = await str_lookup(
                    guild.roles,
                    r,
                    qkey=lambda x: [str(x), full_prune(str(x).replace(" ", ""))],
                    fuzzy=0.125,
                )
            # Must ensure that the user does not assign roles higher than their own
            if inf > perm:
                memb = await self.bot.fetch_member_ex(user.id, guild)
                if memb is None:
                    raise LookupError("Member data not found for this server.")
                if memb.top_role <= role:
                    raise PermissionError("Target role is higher than your highest role.")
            roles.append(role)

        def get_ecolour(colour):
            h, s, l = rgb_to_hsl([c / 255 for c in colour])
            if l >= 0.875 or l >= 0.625 and s < 0.375 or l >= 0.5 and s < 0.875:
                return "⚪"
            if l <= 0.125 or l <= 0.375 and s < 0.375 or l <= 0.5 and s < 0.125:
                return "⚫"
            if h < 1 / 16 or h >= 41 / 48:
                return "🔴"
            if 1 / 16 <= h < 5 / 48:
                return "🟠"
            if 5 / 48 <= h < 1 / 4:
                return "🟡"
            if 1 / 4 <= h < 1 / 2:
                return "🟢"
            if 1 / 2 <= h < 35 / 48:
                return "🔵"
            if 35 / 48 <= h < 41 / 48:
                return "🟣"
            return "🟤"

        buttons = [cdict(name=role.name, emoji=get_ecolour(role.colour.to_rgb()), id="%04d" % (ihash(role.name) % 10000) + str(role.id)) for role in roles]
        colour = await self.bot.get_colour(self.bot.user)
        rolestr = "_".join(str(role.id) for role in roles)
        description = f"```callback-admin-roleselect-{rolestr}-\n{len(buttons)} roles available```Click a button to add or remove a role from yourself!"
        embed = discord.Embed(colour=colour, title="🗂 Role Selection 🗂", description=description)
        embed.set_author(**get_author(self.bot.user))
        await send_with_reply(None, message, embed=embed, buttons=buttons)

    async def _callback_(self, bot, message, reaction, user, vals, **void):
        if not reaction:
            return
        reaction = as_str(reaction)
        if not reaction.isnumeric():
            return
        roles = vals.split("_")
        h1, r_id = reaction[:4], reaction[4:]
        if r_id not in roles:
            raise LookupError(f"Role <@&{r_id}> no longer exists.")
        role = await bot.fetch_role(r_id)
        h2 = "%04d" % (ihash(role.name) % 10000)
        if h1 != h2:
            raise NameError(f"Incorrect hash: {h1} != {h2}")
        if role in user.roles:
            await user.remove_roles(role, reason="Role Select")
            await interaction_response(bot, message, user.mention + ": Successfully removed " + role.mention)
        else:
            await user.add_roles(role, reason="Role Select")
            await interaction_response(bot, message, user.mention + ": Successfully added " + role.mention)


class RoleGiver(Command):
    server_only = True
    name = ["Verifier"]
    min_level = 3
    min_display = "3+"
    description = "Adds an automated role giver to the current channel. Triggered by a keyword in messages, only applicable to users with permission level >= 0."
    usage = "<0:react_to>? <1:role>? <delete_messages{?x}>? <disable{?d}>?"
    flags = "aedx"
    no_parse = True
    rate_limit = (2, 4)
    slash = True

    async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
        update = self.bot.data.rolegivers.update
        bot = self.bot
        data = bot.data.rolegivers
        if "d" in flags:
            if argv:
                react = args[0].casefold()
                assigned = data.get(channel.id, {})
                if react not in assigned:
                    raise LookupError(f"Rolegiver {react} not currently assigned for #{channel}.")
                assigned.pop(react)
                return italics(css_md(f"Removed {sqr_md(react)} from the rolegiver list for {sqr_md(channel)}."))
            if channel.id in data:
                data.pop(channel.id)
            return italics(css_md(f"Removed all automated rolegivers from {sqr_md(channel)}."))
        assigned = set_dict(data, channel.id, {})
        if not argv:
            key = lambda alist: f"⟨{', '.join(str(r) for r in alist[0])}⟩, delete: {alist[1]}"
            if not assigned:
                return ini_md(f"No currently active rolegivers for {sqr_md(channel)}.")
            return f"Currently active rolegivers in {channel_mention(channel.id)}:\n{ini_md(iter2str(assigned, key=key))}"
        if sum(len(alist[0]) for alist in assigned) >= 8:
            raise OverflowError(f"Rolegiver list for #{channel} has reached the maximum of 8 items. Please remove an item to add another.")
        react = args[0].casefold()
        if len(react) > 256:
            raise OverflowError(f"Search substring too long ({len(react)} > 256).")
        r = verify_id(unicode_prune(" ".join(args[1:])))
        if len(guild.roles) <= 1:
            guild.roles = await guild.fetch_roles()
            guild.roles.sort()
            guild._roles.update((r.id, r) for r in guild.roles)
        if type(r) is int:
            role = guild.get_role(i)
        else:
            role = await str_lookup(
                guild.roles,
                r,
                qkey=lambda x: [str(x), full_prune(x.replace(" ", ""))],
                fuzzy=0.125,
            )
        # Must ensure that the user does not assign roles higher than their own
        if inf > perm:
            memb = await self.bot.fetch_member_ex(user.id, guild)
            if memb is None:
                raise LookupError("Member data not found for this server.")
            if memb.top_role <= role:
                raise PermissionError("Target role is higher than your highest role.")
        alist = set_dict(assigned, react, [[], False])
        alist[1] |= "x" in flags
        alist[0].append(role.id) 
        update(channel.id)
        return italics(css_md(f"Added {sqr_md(react)} ➡️ {sqr_md(role)} to {sqr_md(channel.name)}."))


class AutoRole(Command):
    server_only = True
    name = ["InstaRole"]
    min_level = 3
    min_display = "3+"
    _timeout_ = 12
    description = "Causes any new user joining the server to automatically gain the targeted role. Input multiple roles to create a randomized role giver."
    usage = "<role>? <update_all{?x}>? <disable{?d}>?"
    flags = "aedx"
    rate_limit = 1
    slash = True

    async def __call__(self, argv, args, name, user, channel, guild, perm, flags, **void):
        update = self.bot.data.autoroles.update
        bot = self.bot
        data = bot.data.autoroles
        if "d" in flags:
            assigned = data.get(guild.id, None)
            if argv and assigned:
                i = await bot.eval_math(argv)
                roles = assigned.pop(i)
                removed = deque()
                for r in roles:
                    try:
                        role = await bot.fetch_role(r, guild)
                    except:
                        print_exc()
                        continue
                    removed.append(role)
                # Update all users by removing roles
                if "x" in flags:
                    i = 1
                    for member in guild.members:
                        for role in removed:
                            if role in member.roles:
                                new = {role: True for role in member.roles}
                                for role in tuple(new):
                                    if role in removed:
                                        new.pop(role)
                                create_task(member.edit(roles=list(new), reason="InstaRole"))
                                break
                        if not i % 5:
                            await asyncio.sleep(5)
                        i += 1
                update(guild.id)
                rolestr = sqr_md(", ".join(str(role) for role in removed))
                return italics(css_md(f"Removed {rolestr} from the autorole list for {sqr_md(guild)}."))
            if guild.id in data:
                data.pop(guild.id)
            return italics(css_md(f"Removed all items from the autorole list for {sqr_md(guild)}."))
        assigned = set_dict(data, guild.id, alist())
        if not argv:
            rlist = alist()
            for roles in assigned:
                new = alist()
                for r in roles:
                    role = await bot.fetch_role(r, guild)
                    new.append(role)
                rlist.append(new)
            if not assigned:
                return ini_md(f"No currently active autoroles for {sqr_md(guild)}.")
            return f"Currently active autoroles for {bold(escape_markdown(guild.name))}:\n{ini_md(iter2str(rlist))}"
        if sum(len(alist) for alist in assigned) >= 8:
            raise OverflowError(f"Autorole list for #{channel} has reached the maximum of 8 items. Please remove an item to add another.")
        roles = alist()
        rolenames = (verify_id(i) for i in args)
        if len(guild.roles) <= 1:
            guild.roles = await guild.fetch_roles()
            guild.roles.sort()
        rolelist = guild.roles
        for r in rolenames:
            if type(r) is int:
                for i in rolelist:
                    if i.id == r:
                        role = i
                        break
            else:
                role = await str_lookup(
                    rolelist,
                    r,
                    qkey=lambda x: [str(x), full_prune(x.replace(" ", ""))],
                    fuzzy=0.125,
                )
            # Must ensure that the user does not assign roles higher than their own
            if not inf > perm:
                memb = await self.bot.fetch_member_ex(user.id, guild)
                if memb is None:
                    raise LookupError("Member data not found for this server.")
                if memb.top_role <= role:
                    raise PermissionError("Target role is higher than your highest role.")
            roles.append(role)
        new = alist(role.id for role in roles)
        if new not in assigned:
            assigned.append(new)
            update(guild.id)
        # Update all users by adding roles
        if "x" in flags or name == "instarole":
            if roles:
                async with discord.context_managers.Typing(channel):
                    i = 1
                    for member in guild.members:
                        role = roles.next()
                        if role not in member.roles:
                            create_task(member.add_roles(role, reason="InstaRole", atomic=True))
                            if not i % 5:
                                await asyncio.sleep(5)
                            i += 1
        rolestr = sqr_md(", ".join(str(role) for role in roles))
        return italics(css_md(f"Added {rolestr} to the autorole list for {sqr_md(guild)}."))


class RolePreserver(Command):
    server_only = True
    name = ["🕵️", "StickyRoles"]
    min_level = 3
    min_display = "3+"
    description = "Causes ⟨MIZA⟩ to save roles for all users, and re-add them when they leave and rejoin."
    usage = "(enable|disable)?"
    flags = "aed"
    slash = True

    def __call__(self, flags, guild, name, **void):
        bot = self.bot
        following = bot.data.rolepreservers
        update = following.update
        # Empty dictionary is enough to represent an active role preserver here
        curr = following.get(guild.id)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
            return italics(css_md(f"Disabled role preservation for {sqr_md(guild)}."))
        elif "e" in flags or "a" in flags:
            if guild.id not in following:
                following[guild.id] = {}
            return italics(css_md(f"Enabled role preservation for {sqr_md(guild)}."))
        if curr is None:
            return ini_md(f'Role preservation is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
        return ini_md(f"Role preservation is currently enabled in {sqr_md(guild)}.")


class NickPreserver(Command):
    server_only = True
    name = ["StickyNicks", "NicknamePreserver"]
    min_level = 3
    min_display = "3+"
    description = "Causes ⟨MIZA⟩ to save nicknames for all users, and re-add them when they leave and rejoin."
    usage = "(enable|disable)?"
    flags = "aed"

    def __call__(self, flags, guild, name, **void):
        bot = self.bot
        following = bot.data.nickpreservers
        update = following.update
        # Empty dictionary is enough to represent an active nick preserver here
        curr = following.get(guild.id)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
            return italics(css_md(f"Disabled nickname preservation for {sqr_md(guild)}."))
        elif "e" in flags or "a" in flags:
            if guild.id not in following:
                following[guild.id] = {}
            return italics(css_md(f"Enabled nickname preservation for {sqr_md(guild)}."))
        if curr is None:
            return ini_md(f'Nickname preservation is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
        return ini_md(f"Nickname preservation is currently enabled in {sqr_md(guild)}.")


class Lockdown(Command):
    server_only = True
    _timeout_ = 16
    name = ["🔒", "☣️"]
    min_level = inf
    description = "Completely locks down the server by removing send message permissions for all users and revoking all invites."
    flags = "f"
    rate_limit = 30

    async def roleLock(self, role, channel):
        perm = role.permissions
        perm.administrator = False
        perm.send_messages = False
        with self.bot.ExceptionSender(channel):
            await role.edit(permissions=perm, reason="Server Lockdown.")

    async def invLock(self, inv, channel):
        with self.bot.ExceptionSender(channel):
            await inv.delete(reason="Server Lockdown.")

    async def __call__(self, guild, channel, flags, **void):
        if "f" not in flags:
            return self.bot.dangerous_command
        u_id = self.bot.id
        for role in guild.roles:
            if len(role.members) != 1 or role.members[-1].id not in (u_id, guild.owner_id):
                create_task(self.roleLock(role, channel))
        invites = await guild.invites()
        for inv in invites:
            create_task(self.invLock(inv, channel))
        return bold(css_md(sqr_md(uni_str("LOCKDOWN REQUESTED."))))


class SaveChannel(Command):
    time_consuming = 1
    _timeout_ = 16
    name = ["BackupChannel", "DownloadChannel"]
    min_level = 3
    description = "Saves a number of messages in a channel, as well as their contents, to a .txt file."
    usage = "<0:channel>? <1:message_limit(4096)>?"

    async def __call__(self, guild, channel, args, **void):
        num = 4096
        ch = channel
        if args:
            if len(args) >= 2:
                num = await self.bot.eval_math(" ".join(args[1:]))
                if not num <= 65536:
                    raise OverflowError("Maximum number of messages allowed is 65536.")
                if num <= 0:
                    raise ValueError("Please input a valid message limit.")
            ch = await self.bot.fetch_channel(verify_id(args[0]))
            if guild is None or hasattr(guild, "ghost"):
                if guild.id != ch.id:
                    raise PermissionError("Target channel is not in this server.")
            elif ch.id not in (c.id for c in guild.channels):
                raise PermissionError("Target channel is not in this server.")
        h = await ch.history(limit=num).flatten()
        h = h[::-1]
        s = ""
        while h:
            async with Delay(0.32):
                if s:
                    s += "\n\n"
                s += "\n\n".join(message_repr(m, limit=4096, username=True) for m in h[:4096])
                h = h[4096:]
        return bytes(s, "utf-8")


class UserLog(Command):
    server_only = True
    name = ["MemberLog"]
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log user and member events from the server, in the current channel."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, name, **void):
        data = bot.data.logU
        update = bot.data.logU.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            return italics(css_md(f"Enabled user event logging in {sqr_md(channel)} for {sqr_md(guild)}."))
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
            return italics(css_md(f"Disabled user event logging for {sqr_md(guild)}."))
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return ini_md(f"User event logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
        return ini_md(f'User event logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class MessageLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log message events from the server, in the current channel."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, name, **void):
        data = bot.data.logM
        update = bot.data.logM.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            return italics(css_md(f"Enabled message event logging in {sqr_md(channel)} for {sqr_md(guild)}."))
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
            return italics(css_md(f"Disabled message event logging for {sqr_md(guild)}."))
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return ini_md(f"Message event logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
        return ini_md(f'Message event logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class FileLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log deleted files from the server, in the current channel."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, name, **void):
        data = bot.data.logF
        update = bot.data.logF.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            return italics(css_md(f"Enabled file deletion logging in {sqr_md(channel)} for {sqr_md(guild)}."))
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
            return italics(css_md(f"Disabled file deletion logging for {sqr_md(guild)}."))
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return ini_md(f"File deletion logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
        return ini_md(f'File deletion logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class StarBoard(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to repost popular messages with a certain number of a specified reaction anywhere from the server, into the current channel."
    usage = "<0:reaction> <1:react_count(1)>? <disable{?d}>?"
    flags = "d"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
    rate_limit = 1

    async def __call__(self, bot, args, user, message, channel, guild, flags, **void):
        data = bot.data.starboards
        if "d" in flags:
            if args:
                e_data = " ".join(args)
                try:
                    e_id = int(e_data)
                except:
                    emoji = e_data
                else:
                    emoji = await bot.fetch_emoji(e_id)
                emoji = str(emoji)
                try:
                    data[guild.id].pop(emoji)
                except KeyError:
                    pass
                else:
                    if any(v for k, v in data[guild.id].items() if k != None):
                        data.update(guild.id)
                    else:
                        data.pop(guild.id)
                return italics(css_md(f"Disabled starboard trigger {sqr_md(emoji)} for {sqr_md(guild)}."))
            for c_id, v in data.items():
                data.pop(guild.id, None)
            return italics(css_md(f"Disabled all starboard reposting for {sqr_md(guild)}."))
        if not args:
            buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
            await send_with_reply(
                None,
                message,
                "*```" + "\n" * ("z" in flags) + "callback-admin-starboard-"
                + str(user.id) + "_0"
                + "-\nLoading Starboard database...```*",
                buttons=buttons,
            )
            return
        if not args:
            try:
                e_id, count = data[channel.id]
            except KeyError:
                return ini_md(f"Starboard reposting is currently disabled in {sqr_md(channel)}.")
        e_data = args.pop(0)
        try:
            e_id = int(e_data)
        except:
            emoji = e_data
        else:
            emoji = await bot.fetch_emoji(e_id)
        emoji = str(emoji)
        if args:
            count = await bot.eval_math(" ".join(args))
        else:
            count = 1
        boards = data.setdefault(guild.id, {})
        boards[emoji] = (count, channel.id)
        data.update(guild.id)
        return ini_md(f"Successfully added starboard to {sqr_md(channel)}, with trigger {sqr_md(emoji)}: {sqr_md(count)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = list(map(int, vals.split("_", 1)))
        if reaction not in (None, self.directions[-1]) and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        data = bot.data.starboards
        curr = data.setdefault(guild.id, {})
        page = 16
        last = max(0, len(curr) - page)
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
            "callback-admin-starboard-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += f"No currently assigned starboard triggers for {str(guild).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(curr)} starboard triggers currently assigned for {str(guild).replace('`', '')}:```*"
            msg = ini_md(iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}, key=lambda t: f"×{t[0]} {sqr_md(bot.get_channel(t[1]))}"))
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if hasattr(message, "int_token"):
            await bot.ignore_interaction(message)


class Crosspost(Command):
    server_only = True
    name = ["Repost", "Subscribe"]
    min_level = 3
    description = "Causes ⟨MIZA⟩ to automatically crosspost all messages from the target channel, into the current channel."
    usage = "<channel> <disable{?d}>?"
    flags = "aed"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
    rate_limit = 1

    async def __call__(self, bot, argv, flags, user, message, channel, guild, **void):
        data = bot.data.crossposts
        if "d" in flags:
            if argv:
                target = await bot.fetch_channel(argv)
                if target.id in data:
                    data[target.id].discard(channel.id)
                data.update(target.id)
                return italics(css_md(f"Disabled message crossposting from {sqr_md(target)} to {sqr_md(channel)}."))
            for c_id, v in data.items():
                try:
                    v.remove(channel.id)
                except KeyError:
                    pass
                else:
                    data.update(c_id)
            return italics(css_md(f"Disabled all message crossposting for {sqr_md(channel)}."))
        if not argv:
            buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
            await send_with_reply(
                None,
                message,
                "*```" + "\n" * ("z" in flags) + "callback-admin-crosspost-"
                + str(user.id) + "_0"
                + "-\nLoading Crosspost database...```*",
                buttons=buttons,
            )
            return
        target = await bot.fetch_channel(argv)
        if not target.guild.get_member(user.id) or not target.permissions_for(target.guild.me).read_messages or not target.permissions_for(target.guild.get_member(user.id)).read_messages:
            raise PermissionError("Cannot follow channels without read message permissions.")
        channels = data.setdefault(target.id, set())
        channels.add(channel.id)
        data.update(target.id)
        return ini_md(f"Now crossposting all messages from {sqr_md(target)} to {sqr_md(channel)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = list(map(int, vals.split("_", 1)))
        if reaction not in (None, self.directions[-1]) and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        data = bot.data.crossposts
        curr = {k: sqr_md(bot.get_channel(k)) for k in sorted(c_id for c_id, v in data.items() if message.channel.id in v)}
        page = 16
        last = max(0, len(curr) - page)
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
            "callback-admin-crosspost-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += f"No currently assigned crosspost subscriptions for #{str(message.channel).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(curr)} crosspost subscriptions currently assigned for #{str(message.channel).replace('`', '')}:```*"
            msg = ini_md(iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}))
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if hasattr(message, "int_token"):
            await bot.ignore_interaction(message)


class Publish(Command):
    server_only = True
    name = ["News", "AutoPublish"]
    min_level = 3
    description = "Causes ⟨MIZA⟩ to automatically publish all posted messages in the current channel."
    usage = "(enable|disable)? <force{?x}>?"
    flags = "aedx"
    rate_limit = 1

    async def __call__(self, bot, flags, message, channel, guild, name, **void):
        data = bot.data.publishers
        if "e" in flags or "a" in flags:
            if channel.type != discord.ChannelType.news:
                raise TypeError("This feature can only be used in announcement channels.")
            if not channel.permissions_for(guild.me).manage_messages:
                raise PermissionError("Manage messages permission required to publish messages in channel.")
            data[channel.id] = 0 if "x" in flags else message.id
            return italics(css_md(f"Enabled automatic message publishing in {sqr_md(channel)} for {sqr_md(guild)}."))
        elif "d" in flags:
            data.pop(channel.id, None)
            return italics(css_md(f"Disabled automatic message publishing for {sqr_md(guild)}."))
        if channel.id in data:
            return ini_md(f"Automatic message publishing is currently enabled in {sqr_md(channel)}.")
        return ini_md(f'Automatic message publishing is currently disabled in {sqr_md(channel)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class AutoEmoji(Command):
    server_only = True
    name = ["NQN", "Emojis"]
    min_level = 0
    description = "Causes all failed emojis starting and ending with : to be deleted and reposted with a webhook, when possible."
    usage = "(enable|disable)?"
    flags = "aed"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
    rate_limit = 1

    async def __call__(self, bot, flags, guild, message, user, name, perm, **void):
        data = bot.data.autoemojis
        if flags and perm < 3:
            reason = "to modify autoemoji for " + guild.name
            raise self.perm_error(perm, 3, reason)
        if "e" in flags or "a" in flags:
            data[guild.id] = True
            return italics(css_md(f"Enabled automatic emoji substitution for {sqr_md(guild)}."))
        elif "d" in flags:
            data.pop(guild.id, None)
            return italics(css_md(f"Disabled automatic emoji substitution for {sqr_md(guild)}."))
        buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
        await send_with_reply(
            None,
            message,
            "*```" + "\n" * ("z" in flags) + "callback-admin-autoemoji-"
            + str(user.id) + "_0"
            + "-\nLoading AutoEmoji database...```*",
            buttons=buttons,
        )
    
    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = list(map(int, vals.split("_", 1)))
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        data = bot.data.autoemojis
        curr = {f":{e.name}:": f"({e.id})` {min_emoji(e)}" for e in sorted(guild.emojis, key=lambda e: full_prune(e.name))}
        page = 16
        last = max(0, len(curr) - page)
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
        content = "```" + "\n" * ("\n" in content[:i]) + (
            "callback-admin-autoemoji-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if guild.id in data:
            content += f"Automatic emoji substitution is currently enabled in {sqr_md(guild)}.```"
        else:
            content += f'Automatic emoji substitution is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}autoemoji enable" to enable.```'
        if not curr:
            msg = italics(code_md(f"No custom emojis found for {str(message.guild).replace('`', '')}."))
        else:
            msg = italics(code_md(f"{len(curr)} custom emojis currently assigned for {str(message.guild).replace('`', '')}:")) + "\n" + iter2str({k + " " * (32 - len(k)): curr[k] for k in tuple(curr)[pos:pos + page]}, left="`", right="")
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=content, embed=emb))
        if hasattr(message, "int_token"):
            await bot.ignore_interaction(message)


class UpdateAutoEmojis(Database):
    name = "autoemojis"

    def guild_emoji_map(self, guild, emojis={}):
        for e in sorted(guild.emojis, key=lambda e: e.id):
            n = e.name
            while n in emojis:
                if emojis[n] == e.id:
                    break
                t = n.rsplit("-", 1)
                if t[-1].isnumeric():
                    n = t[0] + "-" + str(int(t[-1]) + 1)
                else:
                    n = t[0] + "-1"
            emojis[n] = e
        return emojis

    async def _nocommand_(self, message, recursive=True, **void):
        if not message.content or getattr(message, "webhook_id", None) or message.content.count(":") < 2 or message.content.count("```") > 1:
            return
        emojis = find_emojis(message.content)
        for e in emojis:
            name, e_id = e.split(":")[1:]
            e_id = int("".join(regexp("[0-9]+").findall(e_id)))
            animated = self.bot.cache.emojis.get(name)
            if not animated:
                animated = await create_future(self.bot.is_animated, e_id, verify=True)
            else:
                name = animated.name
            if animated is not None and not message.webhook_id:
                orig = self.bot.data.emojilists.setdefault(message.author.id, {})
                orig[name] = e_id
                self.bot.data.emojilists.update(message.author.id)
        if not message.guild or message.guild.id not in self.data:
            return
        msg = message.content
        guild = message.guild
        orig = self.bot.data.emojilists.get(message.author.id, {})
        emojis = None
        if msg.startswith("+"):
            emi = msg[1:].strip()
            spl = emi.rsplit(None, 1)
            if len(spl) > 1:
                ems, m_id = spl
                if not m_id.isnumeric():
                    spl = [emi]
            if len(spl) == 1:
                ems = spl[0]
                m2 = await self.bot.history(message.channel, limit=1, before=message.id).__anext__()
            else:
                m2 = None
                m_id = int(m_id)
            if not m2 and m_id:
                try:
                    m2 = await self.bot.fetch_message(m_id)
                except LookupError:
                    m2 = None
            if m2:
                found = False
                ems = regexp("<a?:[A-Za-z0-9\\-~_]{1,32}").sub("", ems.replace(" ", "").replace("\\", "")).replace(">", ":")
                for name in (n for n in ems.strip(":").split(":") if n):
                    emoji = None
                    if emojis is None:
                        emojis = self.guild_emoji_map(guild, dict(orig))
                    emoji = emojis.get(name)
                    if not emoji:
                        r1 = regexp("^[A-Za-z0-9\\-~_]{1,32}$")
                        if r1.fullmatch(name):
                            if name.isnumeric():
                                emoji = int(name)
                            else:
                                t = name[::-1].replace("~", "-", 1)[::-1].rsplit("-", 1)
                                if t[-1].isnumeric():
                                    i = int(t[-1])
                                    if i < 1000:
                                        if not emoji:
                                            name = t[0]
                                            emoji = emojis.get(name)
                                        while i > 1 and not emoji:
                                            i -= 1
                                            name = t[0] + "-" + str(i)
                                            emoji = emojis.get(name)
                    if emoji:
                        if type(emoji) is int:
                            e_id = emoji
                            emoji = self.bot.cache.emojis.get(e_id)
                        found = True
                        create_task(m2.add_reaction(emoji))
                if found:
                    create_task(self.bot.silent_delete(message))
                    return
        regex = regexp("(?:^|^[^<\\\\`]|[^<][^\\\\`]|.[^a\\\\`])(:[A-Za-z0-9\\-~_]{1,32}:)(?:(?![^0-9]).)*(?:$|[^0-9>`])")
        pops = set()
        offs = 0
        while offs < len(msg):
            matched = regex.search(msg[offs:])
            if not matched:
                break
            substitutes = None
            s = matched.group()
            start = matched.start()
            while s and not regexp(":[A-Za-z0-9\\-~_]").fullmatch(s[:2]):
                s = s[1:]
                start += 1
            while s and not regexp("[A-Za-z0-9\\-~_]:").fullmatch(s[-2:]):
                s = s[:-1]
            offs = start = offs + start
            offs += len(s)
            if not s:
                continue
            name = s[1:-1]
            if emojis is None:
                emojis = self.guild_emoji_map(guild, dict(orig))
            emoji = emojis.get(name)
            if not emoji:
                if name.isnumeric():
                    emoji = int(name)
                else:
                    t = name[::-1].replace("~", "-", 1)[::-1].rsplit("-", 1)
                    if t[-1].isnumeric():
                        i = int(t[-1])
                        if i < 1000:
                            if not emoji:
                                name = t[0]
                                emoji = emojis.get(name)
                            while i > 1 and not emoji:
                                i -= 1
                                name = t[0] + "-" + str(i)
                                emoji = emojis.get(name)
            if type(emoji) is int:
                e_id = emoji
                emoji = self.bot.cache.emojis.get(e_id)
                if not emoji:
                    animated = await create_future(self.bot.is_animated, e_id, verify=True)
                    if animated is not None:
                        emoji = cdict(id=e_id, animated=animated)
                if not emoji and not message.webhook_id:
                    self.bot.data.emojilists.get(message.author.id, {}).pop(name, None)
                    self.bot.data.emojilists.update(message.author.id)
            if emoji:
                pops.add((str(name), emoji.id))
                if len(msg) < 1936:
                    sub = "<"
                    if emoji.animated:
                        sub += "a"
                    name = getattr(emoji, "name", None) or "_"
                    sub += f":{name}:{emoji.id}>"
                else:
                    sub = min_emoji(emoji)
                substitutes = (start, sub, start + len(s))
                if getattr(emoji, "name", None):
                    if not message.webhook_id:
                        orig = self.bot.data.emojilists.setdefault(message.author.id, {})
                        orig.setdefault(name, emoji.id)
                        self.bot.data.emojilists.update(message.author.id)
            if substitutes:
                msg = msg[:substitutes[0]] + substitutes[1] + msg[substitutes[2]:]
        if not msg or msg == message.content:
            return
        msg = escape_everyone(msg).strip("\u200b")
        if not msg or msg == message.content or len(msg) > 2000:
            return
        if not recursive:
            return msg
        create_task(self.bot.silent_delete(message))
        url = await self.bot.get_proxy_url(message.author)
        m = await self.bot.send_as_webhook(message.channel, msg, username=message.author.display_name, avatar_url=url)
        if recursive and regex.search(m.content):
            for k in tuple(pops):
                if str(k[1]) not in m.content:
                    orig.pop(k[0], None)
                else:
                    pops.discard(k)
            if pops:
                print("Removed emojis:", pops)
                msg = await self._nocommand_(message, recursive=False)
                if msg and msg != m.content:
                    create_task(self.bot.silent_delete(m))
                    await self.bot.send_as_webhook(message.channel, msg, username=message.author.display_name, avatar_url=url)


# TODO: Stop being lazy and finish this damn command
# class Welcomer(Command):
#     server_only = True
#     name = ["Welcome", "JoinMessage"]
#     min_level = 2
#     description = "Sets up or modifies a welcome message generator for new users in the server."
#     usage = "<0:option(image_url)(image_position{centre})(shape[{square},circle,rounded_square,hexagon])> <-1:value> <disable(?d)>"
#     flags = "h"
#     rate_limit = (2, 4)

#     async def __call__(self, bot, args, message, channel, guild, flags, perm, name, **void):
#         pass


# Just like the reminders database, the mute and ban databases need to be this way to keep O(1) time complexity when idle.

class UpdateMutes(Database):
    name = "mutes"

    def __load__(self):
        d = self.data
        self.listed = alist(sorted(((d[i][0]["t"], i) for i in d), key=lambda x: x[0]))

    async def _call_(self):
        t = utc()
        while self.listed:
            p = self.listed[0]
            if t < p[0]:
                break
            self.listed.popleft()
            g_id = p[1]
            temp = self.data[g_id]
            if not temp:
                self.data.pop(g_id)
                continue
            x = temp[0]
            if t < x["t"]:
                self.listed.insort((x["t"], g_id), key=lambda x: x[0])
                print(self.listed)
                continue
            x = cdict(temp.pop(0))
            self.update(g_id)
            if not temp:
                self.data.pop(g_id)
            else:
                z = temp[0]["t"]
                self.listed.insort((z, g_id), key=lambda x: x[0])
            print(self.listed)
            with tracebacksuppressor:
                guild = await self.bot.fetch_guild(g_id)
                user = await self.bot.fetch_user(x.u)
                m = guild.me
                try:
                    channel = await self.bot.fetch_channel(x.c)
                    if not channel.permissions_for(m).send_messages:
                        raise LookupError
                except (LookupError, discord.Forbidden, discord.NotFound):
                    channel = self.bot.get_first_sendable(guild, m)
                try:
                    role = await self.bot.data.muteroles.get(guild)
                    member = guild.get_member(user.id)
                    if member is None or role not in member.roles:
                        raise LookupError
                    roles = [r for r in member.roles[1:] if r.id != role.id]
                    r_ids = x.get("x", ())
                    attempt = set()
                    for r_id in r_ids:
                        with tracebacksuppressor:
                            r = await self.bot.fetch_role(r_id, guild)
                            attempt.add(r)
                    try:
                        await member.edit(roles=list(set(roles + list(attempt))), reason="Temporary mute expired.")
                    except discord.Forbidden:
                        await member.edit(roles=roles, reason="Temporary mute expired.")
                        if attempt:
                            await member.add_roles(*attempt, reason="Temporary mute expired.")
                    text = italics(css_md(f"{sqr_md(user)} has been unmuted in {sqr_md(guild)}."))
                except:
                    text = italics(css_md(f"Unable to unmute {sqr_md(user)} in {sqr_md(guild)}."))
                    print_exc()
                await channel.send(text)

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            for x in self.data[guild.id]:
                if x["u"] == user.id:
                    if not x.get("x"):
                        with suppress(KeyError):
                            x["x"] = self.bot.data.rolepreservers[guild.id][user.id]
                            self.bot.data.rolepreservers[guild.id].pop(user.id)
                            bot.data.rolepreservers.update(guild.id)
                    role = await self.bot.data.muteroles.get(guild)
                    return await user.add_roles(role, reason="Sticky mute")


class UpdateMuteRoles(Database):
    name = "muteroles"

    mute = discord.PermissionOverwrite(
        create_instant_invite=False,
        kick_members=False,
        ban_members=False,
        administrator=False,
        manage_channels=False,
        manage_guild=False,
        add_reactions=False,
        view_audit_log=False,
        priority_speaker=False,
        stream=False,
        read_messages=None,
        send_messages=False,
        send_tts_messages=False,
        manage_messages=False,
        embed_links=False,
        attach_files=False,
        read_message_history=None,
        mention_everyone=False,
        external_emojis=False,
        view_guild_insights=False,
        connect=False,
        speak=False,
        mute_members=False,
        deafen_members=False,
        move_members=False,
        use_voice_activation=False,
        change_nickname=False,
        manage_nicknames=False,
        manage_roles=False,
        manage_webhooks=False,
        manage_emojis=False,
    )

    failed = set()

    async def get(self, guild):
        with suppress(KeyError):
            return self.bot.cache.roles[self.data[guild.id]]
        role = await guild.create_role(name="Muted", colour=discord.Colour(1), reason="Mute role setup.")
        self.bot.cache.roles[role.id] = role
        self.data[guild.id] = role.id
        self.update(guild.id)
        for channel in guild.channels:
            if channel.permissions_for(guild.me).manage_channels and channel.permissions_for(guild.me).manage_roles and not channel.permissions_synced and channel.id not in self.failed:
                try:
                    await channel.set_permissions(target=role, overwrite=self.mute)
                except discord.Forbidden:
                    self.failed.add(channel.id)
                except:
                    print_exc()
        return role

    async def __call__(self):
        for g_id in deque(self.data):
            guild = self.bot.cache.guilds.get(g_id)
            try:
                role = self.bot.cache.roles[self.data[g_id]]
            except KeyError:
                self.data.pop(g_id, None)
                continue
            if guild:
                if role not in guild.roles:
                    self.data.pop(g_id)
                role = await self.get(guild)
                for channel in guild.channels:
                    if channel.permissions_for(guild.me).manage_channels and channel.permissions_for(guild.me).manage_roles and not channel.permissions_synced and channel.id not in self.failed:
                        if role not in channel.overwrites:
                            try:
                                await channel.set_permissions(target=role, overwrite=self.mute)
                            except discord.Forbidden:
                                self.failed.add(channel.id)
                            except:
                                print_exc()

    def _day_(self):
        self.failed.clear()


class UpdateBans(Database):
    name = "bans"

    def __load__(self):
        d = self.data
        self.listed = alist(sorted(((d[i][0]["t"], i) for i in d if d[i]), key=lambda x: x[0]))

    async def _call_(self):
        t = utc()
        while self.listed:
            p = self.listed[0]
            if t < p[0]:
                break
            self.listed.popleft()
            g_id = p[1]
            temp = self.data[g_id]
            if not temp:
                self.data.pop(g_id)
                continue
            x = temp[0]
            if t < x["t"]:
                self.listed.insort((x["t"], g_id), key=lambda x: x[0])
                print(self.listed)
                continue
            x = cdict(temp.pop(0))
            self.update(g_id)
            if not temp:
                self.data.pop(g_id)
            else:
                z = temp[0]["t"]
                self.listed.insort((z, g_id), key=lambda x: x[0])
            print(self.listed)
            with tracebacksuppressor:
                guild = await self.bot.fetch_guild(g_id)
                user = await self.bot.fetch_user(x.u)
                m = guild.me
                try:
                    channel = await self.bot.fetch_channel(x.c)
                    if not channel.permissions_for(m).send_messages:
                        raise LookupError
                except (LookupError, discord.Forbidden, discord.NotFound):
                    channel = self.bot.get_first_sendable(guild, m)
                try:
                    await guild.unban(user, reason="Temporary ban expired.")
                    text = italics(css_md(f"{sqr_md(user)} has been unbanned from {sqr_md(guild)}."))
                except:
                    text = italics(css_md(f"Unable to unban {sqr_md(user)} from {sqr_md(guild)}."))
                    print_exc()
                await channel.send(text)

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            for x in self.data[guild.id]:
                if x["u"] == user.id:
                    return await guild.ban(user, reason="Sticky ban")


# Triggers upon 3 channel deletions in 2 minutes or 6 bans in 10 seconds
class ServerProtector(Database):
    name = "prot"
    no_file = True

    async def kickWarn(self, u_id, guild, owner, msg):
        user = await self.bot.fetch_user(u_id)
        try:
            await guild.kick(user, reason="Triggered automated server protection response for excessive " + msg + ".")
            await owner.send(
                f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
                + f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
                + "and has been removed from the server to prevent any potential further attacks."
            )
        except discord.Forbidden:
            await owner.send(
                f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
                + f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
                + "and were unable to be automatically removed from the server; please watch them carefully to prevent any potential further attacks."
            )

    async def targetWarn(self, u_id, guild, msg):
        print(f"Channel Deletion warning by {user_mention(u_id)} in {guild}.")
        user = self.bot.user
        owner = guild.owner
        if owner.id == user.id:
            owner = await self.bot.fetch_user(next(iter(self.bot.owners)))
        if u_id == guild.owner.id:
            if u_id == user.id:
                return
            user = guild.owner
            await owner.send(
                f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
                + f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
                + "If this was intentional, please ignore this message."
            )
        elif u_id == user.id:
            create_task(guild.leave())
            await owner.send(
                f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
                + f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
                + "and will promptly leave the server to prevent any potential further attacks."
            )
        else:
            await self.kickWarn(u_id, guild, owner, msg)

    async def _channel_delete_(self, channel, guild, **void):
        if channel.id in self.bot.cache.deleted:
            return
        if self.bot.is_trusted(guild.id):
            audits = guild.audit_logs(limit=5, action=discord.AuditLogAction.channel_delete)
            ts = utc()
            cnt = {}
            async for log in audits:
                if ts - utc_ts(log.created_at) < 120:
                    add_dict(cnt, {log.user.id: 1})
            for u_id in cnt:
                if cnt[u_id] > 2:
                    create_task(self.targetWarn(u_id, guild, f"channel deletions `({cnt[u_id]})`"))

    async def _ban_(self, user, guild, **void):
        if not self.bot.recently_banned(user, guild):
            if self.bot.is_trusted(guild.id):
                audits = guild.audit_logs(limit=13, action=discord.AuditLogAction.ban)
                ts = utc()
                cnt = {}
                async for log in audits:
                    if ts - utc_ts(log.created_at) < 10:
                        add_dict(cnt, {log.user.id: 1})
                for u_id in cnt:
                    if cnt[u_id] > 5:
                        create_task(self.targetWarn(u_id, guild, f"banning `({cnt[u_id]})`"))


class CreateEmoji(Command):
    server_only = True
    name = ["EmojiCreate", "EmojiCopy", "CopyEmoji", "Emoji"]
    min_level = 2
    description = "Creates a custom emoji from a URL or attached file."
    usage = "<1:name>+ <0:url>"
    flags = "aed"
    no_parse = True
    rate_limit = (3, 6)
    _timeout_ = 3
    typing = True
    slash = ("Emoji",)
    msgcmd = ("Create Emoji",)

    async def __call__(self, bot, user, guild, channel, message, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args.extend(best_url(a) for a in message.attachments)
            argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
        if not args:
            raise ArgumentError("Please enter URL, emoji, or attached file to add.")
        with discord.context_managers.Typing(channel):
            try:
                if len(args) > 1 and is_url(args[0]):
                    args.append(args.pop(0))
                url = args.pop(-1)
                urls = await bot.follow_url(url, best=True, allow=True, limit=1)
                if not urls:
                    urls = await bot.follow_to_image(argv)
                    if not urls:
                        urls = await bot.follow_to_image(url)
                        if not urls:
                            raise ArgumentError
                url = urls[0]
            except ArgumentError:
                if not argv:
                    url = None
                    try:
                        url = await bot.get_last_image(message.channel)
                    except FileNotFoundError:
                        raise ArgumentError("Please input an image by URL or attachment.")
                else:
                    raise ArgumentError("Please input an image by URL or attachment.")
            name = " ".join(args).strip()
            if not name:
                name = "emoji_" + str(len(guild.emojis))
            # print(name, url)
            image = resp = await bot.get_request(url)
            if len(image) > 67108864:
                raise OverflowError("Max file size to load is 64MB.")
            if len(image) > 262144 or not is_image(url):
                ts = ts_us()
                path = "cache/" + str(ts)
                f = await create_future(open, path, "wb", timeout=18)
                await create_future(f.write, image, timeout=18)
                await create_future(f.close, timeout=18)
                try:
                    resp = await process_image(path, "resize_max", [128], timeout=_timeout)
                except:
                    with suppress():
                        os.remove(path)
                    raise
                else:
                    fn = resp[0]
                    f = await create_future(open, fn, "rb", timeout=18)
                    image = await create_future(f.read, timeout=18)
                    create_future_ex(f.close, timeout=18)
                    with suppress():
                        os.remove(fn)
                with suppress():
                    os.remove(path)
            emoji = await guild.create_custom_emoji(image=image, name=name, reason="CreateEmoji command")
            # This reaction indicates the emoji was created successfully
            with suppress(discord.Forbidden):
                await message.add_reaction(emoji)
        return css_md(f"Successfully created emoji {sqr_md(emoji)} for {sqr_md(guild)}.")


class ScanEmoji(Command):
    name = ["EmojiScan", "ScanEmojis"]
    min_level = 1
    description = "Scans all the emojis in the current server for potential issues."
    usage = "<count(inf)>"
    no_parse = True
    rate_limit = (4, 7)
    _timeout_ = 4
    typing = True

    ffprobe_start = (
        "ffprobe",
        "-v",
        "error",
        "-hide_banner",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
    )

    async def __call__(self, bot, guild, channel, message, argv, **void):
        # fut = create_task(send_with_reply(channel, message, "Emoji scan initiated. Delete the original message at any point in time to cancel."))
        p = bot.get_prefix(guild)
        if argv:
            count = await bot.eval_math(argv)
        else:
            count = inf
        found = 0
        with discord.context_managers.Typing(channel):
            for emoji in sorted(guild.emojis, key=lambda e: e.id):
                url = str(emoji.url)
                resp = await create_future(subprocess.run, self.ffprobe_start + (url,), stdout=subprocess.PIPE)
                width, height = map(int, resp.stdout.splitlines())
                if width < 128 or height < 128:
                    found += 1
                    w, h = width, height
                    while w < 128 or h < 128:
                        w, h = w << 1, h << 1
                    colour = await create_future(bot.get_colour, url)
                    bot.send_as_embeds(
                        channel,
                        description=f"{emoji} is {width}×{height}, which is below the recommended discord emoji size, and may appear blurry when scaled by Discord. Scaling the image using {p}resize, with filters `nearest`, `scale2x` or `lanczos` is advised.",
                        fields=(("Example", f"{p}resize {emoji} {w}×{h} scale2x"),),
                        title=f"⚠ Issue {found} ⚠",
                        colour=discord.Colour(colour),
                    )
                    if found >= count:
                        break
        if not found:
            return css_md(f"No emoji issues found for {guild}.")


class UpdateUserLogs(Database):
    name = "logU"

    # Send a member update globally for all user updates
    async def _user_update_(self, before, after, **void):
        found = False
        for g_id in self.data:
            try:
                guild = self.bot.cache.guilds[g_id]
            except KeyError:
                continue
            if guild.get_member(after.id):
                found = True
                break
        if found:
            b_url = best_url(before)
            a_url = best_url(after)
            if b_url != a_url:
                with tracebacksuppressor:
                    urls = await self.bot.data.exec.uproxy(b_url, a_url)
                    # print(after, after.id, *urls)
            for g_id in self.data:
                guild = self.bot.cache.guilds.get(g_id)
                if guild:
                    create_task(self._member_update_(before, after, guild))

    async def _member_update_(self, before, after, guild=None):
        if guild is None:
            guild = after.guild
        # Make sure user is in guild
        try:
            memb = guild.get_member(after.id)
            if memb is None:
                raise LookupError
        except LookupError:
            return
        except:
            print_exc()
            return
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            emb = discord.Embed()
            emb.description = (
                "<@" + str(after.id)
                + "> has been updated:"
            )
            colour = [0] * 3
            # Add fields for every update to the member data
            change = False
            if str(before) != str(after):
                emb.add_field(
                    name="Username",
                    value=escape_markdown(str(before)) + " ➡️ " + escape_markdown(str(after)),
                )
                change = True
                colour[0] += 255
            if hasattr(before, "guild"):
                if before.display_name != after.display_name:
                    emb.add_field(
                        name="Nickname",
                        value=escape_markdown(before.display_name) + " ➡️ " + escape_markdown(after.display_name),
                    )
                    change = True
                    colour[0] += 255
                if hash(tuple(r.id for r in before.roles)) != hash(tuple(r.id for r in after.roles)):
                    sub = alist()
                    add = alist()
                    for r in before.roles:
                        if r not in after.roles:
                            sub.append(r)
                    for r in after.roles:
                        if r not in before.roles:
                            add.append(r)
                    rchange = ""
                    if sub:
                        rchange = "❌ " + escape_markdown(", ".join(role_mention(r.id) for r in sub))
                    if add:
                        rchange += (
                            "\n" * bool(rchange) + "✅ " 
                            + escape_markdown(", ".join(role_mention(r.id) for r in add))
                        )
                    if rchange:
                        emb.add_field(name="Roles", value=rchange)
                        change = True
                        colour[1] += 255
            if before.avatar != after.avatar:
                b_url = best_url(before)
                a_url = best_url(after)
                if "exec" in self.bot.data:
                    urls = ()
                    with tracebacksuppressor:
                        urls = await self.bot.data.exec.uproxy(b_url, a_url)
                    for i, url in enumerate(urls):
                        if url:
                            if i:
                                a_url = url
                            else:
                                b_url = url
                emb.add_field(
                    name="Avatar",
                    value=f"[Before]({b_url}) ➡️ [After]({a_url})",
                )
                emb.set_thumbnail(url=a_url)
                change = True
                colour[2] += 255
            if change:
                b_url = await self.bot.get_proxy_url(before)
                a_url = await self.bot.get_proxy_url(after)
                emb.set_author(name=str(after), icon_url=a_url, url=a_url)
                emb.colour = colour2raw(colour)
                self.bot.send_embeds(channel, emb)

    async def _join_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            # Colour: White
            emb = discord.Embed(colour=16777214)
            emb.set_author(**get_author(user))
            emb.description = f"{user_mention(user.id)} has joined the server."
            age = utc() - utc_ts(user.created_at)
            if age < 86400 * 7:
                emb.description += f"\n⚠️ Account is {time_diff(utc_dt(), user.created_at)} old. ⚠️"
            self.bot.send_embeds(channel, emb)

    async def _leave_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            # Colour: Black
            emb = discord.Embed(colour=1)
            emb.set_author(**get_author(user))
            # Check audit log to find whether user left or was kicked/banned
            prune = None
            kick = None
            ban = None
            with tracebacksuppressor(StopIteration):
                ts = utc()
                futs = [create_task(guild.audit_logs(limit=4, action=getattr(discord.AuditLogAction, action)).flatten()) for action in ("ban", "kick", "member_prune")]
                bans = await futs[0]
                kicks = await futs[1]
                prunes = await futs[2]
                for log in bans:
                    if ts - utc_ts(log.created_at) < 3:
                        if log.target.id == user.id:
                            ban = cdict(id=log.user.id, reason=log.reason)
                            raise StopIteration
                for log in kicks:
                    if ts - utc_ts(log.created_at) < 3:
                        if log.target.id == user.id:
                            kick = cdict(id=log.user.id, reason=log.reason)
                            raise StopIteration
                for log in prunes:
                    if ts - utc_ts(log.created_at) < 3:
                        try:
                            reason = f"{log.extra.delete_member_days} days of inactivity"
                        except AttributeError:
                            reason = None
                        prune = cdict(id=log.user.id, reason=reason)
                        raise StopIteration
            if ban is not None:
                emb.description = f"{user_mention(user.id)} has been banned by {user_mention(ban.id)}."
                if ban.reason:
                    emb.description += f"\nReason: *`{no_md(ban.reason)}`*"
            elif kick is not None:
                emb.description = f"{user_mention(user.id)} has been kicked by {user_mention(kick.id)}."
                if kick.reason:
                    emb.description += f"\nReason: *`{no_md(kick.reason)}`*"
            elif prune is not None:
                emb.description = f"{user_mention(user.id)} has been pruned by {user_mention(prune.id)}."
                if prune.reason:
                    emb.description += f"\nReason: *`{no_md(prune.reason)}`*"
            else:
                emb.description = f"{user_mention(user.id)} has left the server."
            self.bot.send_embeds(channel, emb)


class UpdateMessageCache(Database):
    name = "message_cache"
    no_file = True
    files = "saves/message_cache"
    raws = {}
    loaded = {}
    saving = {}
    save_sem = Semaphore(1, 512, 5, 30)
    search_sem = Semaphore(20, 512, rate_limit=5)

    def __init__(self, *args):
        super().__init__(*args)
        if not os.path.exists(self.files):
            os.mkdir(self.files)

    def get_fn(self, m_id):
        return  m_id // 10 ** 13

    def load_file(self, fn, raw=False):
        if not raw:
            with suppress(KeyError):
                return self.loaded[fn]
            found = {}
            self.loaded[fn] = found
        try:
            data = self.raws[fn]
        except KeyError:
            path = self.files + "/" + str(fn)
            if not os.path.exists(path):
                path += "\x7f"
                if not os.path.exists(path):
                    return
            try:
                with open(path, "rb") as f:
                    out = zipped = decrypt(f.read())
                with tracebacksuppressor(zipfile.BadZipFile):
                    out = zip2bytes(zipped)
                data = pickle.loads(out)
            except:
                print_exc()
                data = {}
            if type(data) is not dict:
                data = {m["id"]: m for m in data}
            self.raws[fn] = data
            # if raw:
                # print(f"{len(data)} message{'s' if len(data) != 1 else ''} temporarily read from {fn}")
        if not raw:
            bot = self.bot
            i = 0
            for k, m in deque(data.items()):
                if "channel" in m:
                    m["channel_id"] = m.pop("channel")
                try:
                    message = bot.CachedMessage(m)
                except:
                    print(m)
                    print_exc()
                bot.cache.messages[k] = found[k] = message
                i += 1
                if not i & 2047:
                    time.sleep(0.1)
            # print(f"{len(data)} message{'s' if len(data) != 1 else ''} successfully loaded from {fn}")
            return found

    def load_message(self, m_id):
        fn = self.get_fn(m_id)
        with suppress(KeyError):
            return self.saving[fn][m_id]
        if fn in self.loaded:
            return self.loaded[fn][m_id]
        found = self.load_file(fn)
        if not found:
            raise KeyError(m_id)
        return found[m_id]

    def save_message(self, message):
        fn = self.get_fn(message.id)
        saving = self.saving.setdefault(fn, {})
        saving[message.id] = message
        return message

    def saves(self, fn, messages):
        self.load_file(fn, raw=True)
        bot = self.bot
        if fn in self.loaded:
            self.loaded[fn].update(messages)
        saved = self.raws.setdefault(fn, {})
        for i, message in enumerate(tuple(messages.values()), 1):
            m = getattr(message, "_data", None)
            if m:
                m = message._data
                if "author" not in m:
                    author = message.author
                    m["author"] = dict(id=author.id, s=str(author), avatar=author.avatar)
                if "channel_id" not in m:
                    try:
                        m["channel_id"] = message.channel.id
                    except AttributeError:
                        continue
            else:
                if message.channel is None:
                    continue
                reactions = []
                attachments = [dict(id=a.id, size=a.size, filename=a.filename, url=a.url, proxy_url=a.proxy_url) for a in message.attachments]
                embeds = [e.to_dict() for e in message.embeds]
                author = message.author
                m = dict(
                    id=message.id,
                    author=dict(id=author.id, s=str(author), avatar=author.avatar),
                    webhook_id=message.webhook_id,
                    reactions=reactions,
                    attachments=attachments,
                    embeds=embeds,
                    edited_timestamp=str(message._edited_timestamp) if getattr(message, "_edited_timestamp", None) else "",
                    type=getattr(message.type, "value", message.type),
                    pinned=message.pinned,
                    flags=message.flags.value if message.flags else 0,
                    mention_everyone=message.mention_everyone,
                    content=message.content,
                    channel_id=message.channel.id,
                )
                for reaction in message.reactions:
                    if not reaction.custom_emoji:
                        r = dict(emoji=dict(id=None, name=str(reaction)))
                        if reaction.count != 1:
                            r["count"] = reaction.count
                        if reaction.me:
                            r["me"] = reaction.me
                        reactions.append(r)
            saved[m["id"]] = m
            if not i & 1023:
                time.sleep(0.1)
        path = self.files + "/" + str(fn)
        if not saved:
            if os.path.exists(path):
                return os.remove(path)
        out = data = pickle.dumps(saved)
        if len(data) > 32768:
            out = bytes2zip(data)
        out = encrypt(out)
        safe_save(path, out)
        return len(saved)

    async def _save_(self, **void):
        if self.save_sem.is_busy():
            return
        async with self.save_sem:
            saving = deque(self.saving.items())
            self.saving.clear()
            i = 0
            for fn, messages in saving:
                await create_future(self.saves, fn, messages)
                i += 1
                if not i & 3 or len(messages) > 65536:
                    await asyncio.sleep(0.3)
            while len(self.loaded) > 512:
                with suppress(RuntimeError):
                    self.loaded.pop(next(iter(self.loaded)))
                i += 1
                if not i % 6:
                    await asyncio.sleep(0.2)
            if not self.save_sem.is_busy():
                while len(self.raws) > 512:
                    with suppress(RuntimeError):
                        self.raws.pop(next(iter(self.raws)))
                    i += 1
                    if not i % 6:
                        await asyncio.sleep(0.2)
            if len(saving) >= 8:
                print(f"Message Database: {len(saving)} files updated.")
            deleted = 0
            limit = str(self.get_fn(time_snowflake(utc_dt() - datetime.timedelta(days=28))))
            for f in os.listdir(self.files):
                if f.isnumeric() and f < limit or f.endswith("\x7f"):
                    with tracebacksuppressor(FileNotFoundError):
                        os.remove(self.files + "/" + f)
                        deleted += 1
            if deleted >= 8:
                print(f"Message Database: {deleted} files deleted.")
            if os.path.exists(self.files + "/-1"):
                self.setmtime()

    getmtime = lambda self: os.path.getmtime(self.files + "/-1")
    setmtime = lambda self: open(self.files + "/-1", "wb").close()

    async def _minute_loop_(self):
        await self._save_()


class UpdateMessageLogs(Database):
    name = "logM"
    searched = False
    dc = {}

    async def __call__(self):
        for h in tuple(self.dc):
            if utc_dt() - h > datetime.timedelta(seconds=3600):
                self.dc.pop(h)

    async def _bot_ready_(self, **void):
        if not self.bot.ready and not self.searched and len(self.bot.cache.messages) <= 65536:
            self.searched = True
            t = None
            with tracebacksuppressor(FileNotFoundError):
                t = utc_ft(self.bot.data.message_cache.getmtime())
            if t is None:
                t = utc_dt() - datetime.timedelta(days=7)
            create_task(self.load_new_messages(t))

    async def save_channel(self, channel, t=None):
        async with self.bot.data.message_cache.search_sem:
            async for message in channel.history(limit=32768, after=t, oldest_first=False):
                self.bot.add_message(message, files=False, force=True)

    async def load_new_messages(self, t):
        print(f"Probing new messages from {len(self.bot.guilds)} guild{'s' if len(self.bot.guilds) != 1 else ''}...")
        with tracebacksuppressor:
            for guild in self.bot.guilds:
                futs = deque()
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).read_message_history:
                        futs.append(create_task(self.save_channel(channel, t)))
                for fut in futs:
                    await fut
        self.bot.data.message_cache.finished = True
        self.bot.data.message_cache.setmtime()
        print("Loading new messages completed.")

    async def _command_(self, message, **void):
        if getattr(message, "slash", None):
            guild = message.guild
            if guild and guild.id in self.data:
                c_id = self.data[guild.id]
                try:
                    channel = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    return
                emb = as_embed(message, link=True)
                emb.colour = discord.Colour(0x00FFFF)
                action = f"**Slash command executed in** {channel_mention(message.channel.id)}:\n"
                emb.description = lim_str(action + (emb.description or ""), 4096)
                self.bot.send_embeds(channel, emb)

    # Edit events are rather straightforward to log
    async def _edit_(self, before, after, **void):
        if not after.author.bot:
            guild = before.guild
            if guild and guild.id in self.data:
                c_id = self.data[guild.id]
                try:
                    channel = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    return
                emb = as_embed(after)
                emb2 = await self.bot.as_embed(before)
                emb.colour = discord.Colour(0x0000FF)
                action = f"**Message edited in** {channel_mention(after.channel.id)}:\n[View Message](https://discord.com/channels/{guild.id}/{after.channel.id}/{after.id})"
                emb.add_field(name="Before", value=lim_str(emb2.description, 1024))
                emb.add_field(name="After", value=lim_str(emb.description, 1024))
                emb.description = action
                emb.timestamp = before.edited_at or after.created_at
                self.bot.send_embeds(channel, emb)

    # Delete events must attempt to find the user who deleted the message
    async def _delete_(self, message, bulk=False, **void):
        cu_id = self.bot.id
        if bulk:
            return
        guild = message.guild
        if guild and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            now = utc_dt()
            u = message.author
            name_id = str(u)
            url = await self.bot.get_proxy_url(u)
            action = discord.AuditLogAction.message_delete
            try:
                t = u
                init = user_mention(t.id)
                d_level = self.bot.is_deleted(message)
                if d_level:
                    if d_level > 1:
                        if d_level < 3:
                            pass
                            # self.logDeleted(message)
                        return
                    t = self.bot.user
                    init = user_mention(t.id)
                else:
                    # Attempt to find who deleted the message
                    if not guild.get_member(cu_id).guild_permissions.view_audit_log:
                        raise PermissionError
                    al = await guild.audit_logs(
                        limit=5,
                        action=action,
                    ).flatten()
                    for e in reversed(al):
                        # print(e, e.target, now - e.created_at)
                        # This is because message delete events stack
                        try:
                            cnt = e.extra.count
                        except AttributeError:
                            cnt = int(e.extra.get("count", 1))
                        h = e.created_at
                        cs = set_dict(self.dc, h, 0)
                        c = cnt - cs
                        if c >= 1:
                            if self.dc[h] == 0:
                                self.dc[h] = cnt
                            else:
                                self.dc[h] += cnt
                        s = (3, 3600)[c >= 1]
                        cid = e.extra.channel.id
                        if now - h < datetime.timedelta(seconds=s):
                            if (not e.target or e.target.id == u.id or u.id == self.bot.deleted_user) and cid == message.channel.id:
                                t = e.user
                                init = user_mention(t.id)
                                message.author = e.target
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = await self.bot.as_embed(message, link=True)
            emb.colour = discord.Colour(0xFF0000)
            action = f"{init} **deleted message from** {channel_mention(message.channel.id)}:\n"
            emb.description = lim_str(action + (emb.description or ""), 4096)
            self.bot.send_embeds(channel, emb)

    # Thanks to the embed sender feature, which allows this feature to send up to 10 logs in one message
    async def _bulk_delete_(self, messages, **void):
        cu = self.bot.user
        cu_id = cu.id
        guild = messages[0].guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            now = utc_dt()
            action = discord.AuditLogAction.message_bulk_delete
            try:
                init = "[UNKNOWN USER]"
                if self.bot.is_deleted(messages[-1]):
                    t = self.bot.user
                    init = user_mention(t.id)
                else:
                    # Attempt to find who deleted the messages
                    if not guild.get_member(cu_id).guild_permissions.view_audit_log:
                        raise PermissionError
                    al = await guild.audit_logs(
                        limit=5,
                        action=action,
                    ).flatten()
                    for e in reversed(al):
                        # print(e, e.target, now - e.created_at)
                        # For some reason bulk message delete events stack too
                        try:
                            cnt = e.extra.count
                        except AttributeError:
                            cnt = int(e.extra.get("count", 1))
                        h = e.created_at
                        cs = set_dict(self.dc, h, 0)
                        c = cnt - cs
                        if c >= len(messages):
                            if self.dc[h] == 0:
                                self.dc[h] = cnt
                            else:
                                self.dc[h] += cnt
                        s = (5, 3600)[c >= len(messages)]
                        if now - h < datetime.timedelta(seconds=s):
                            if (not e.target or e.target.id == messages[-1].channel.id):
                                t = e.user
                                init = user_mention(t.id)
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=0xFF00FF)
            emb.description = f"{init} **deleted {len(messages)} message{'s' if len(messages) != 1 else ''} from** {channel_mention(messages[-1].channel.id)}:\n"
            # for message in messages:
            #     nextline = f"\nhttps://discord.com/channels/{guild.id}/{message.channel.id}/{message.id}"
            #     if len(emb.description) + len(nextline) > 2048:
            #         break
            #     emb.description += nextline
            embs = deque((emb,))
            for message in messages:
                emb = await self.bot.as_embed(message, link=True)
                emb.colour = discord.Colour(0x7F007F)
                embs.append(emb)
            self.bot.send_embeds(channel, embs)


class UpdateFileLogs(Database):
    name = "logF"

    async def _delete_(self, message, **void):
        if self.bot.is_deleted(message) > 1:
            return
        guild = message.guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            if message.attachments:
                try:
                    channel = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    return
                # Attempt to recover files from their proxy URLs, otherwise send the original URLs
                msg = deque()
                fils = []
                for a in message.attachments:
                    try:
                        try:
                            b = self.bot.cache.attachments[a.id]
                        except KeyError:
                            b = await a.read(use_cached=True)
                        else:
                            for i in range(30):
                                if b:
                                    break
                                with Delay(1):
                                    b = self.bot.cache.attachments[a.id]
                        fil = CompatFile(io.BytesIO(b), filename=str(a).rsplit("/", 1)[-1])
                        fils.append(fil)
                    except:
                        msg.append(proxy_url(a))
                colour = await self.bot.get_colour(message.author)
                emb = discord.Embed(colour=colour)
                emb.description = f"File{'s' if len(fils) + len(msg) != 1 else ''} deleted from {user_mention(message.author.id)}"
                if not msg:
                    msg = None
                else:
                    msg = "\n".join(msg)
                if len(fils) == 1:
                    return await self.bot.send_with_file(channel, msg, embed=emb, file=fils[0])
                await channel.send(msg, embed=emb, files=fils)


class UpdatePublishers(Database):
    name = "publishers"

    async def _nocommand_(self, message, **void):
        if message.channel.id in self.data and not message.flags.crossposted and not message.flags.is_crossposted and not message.reference and "\u2009\u2009" not in message.author.name:
            try:
                if not message.channel.permissions_for(message.guild).manage_messages:
                    raise PermissionError("Manage messages permission missing from channel.")
                await message.publish()
            except Exception as ex:
                if "invalid message type" not in repr(ex).lower():
                    self.data.pop(message.channel.id, None)
                    print_exc()
                    bot.send_exception(message.channel, ex)


class UpdateCrossposts(Database):
    name = "crossposts"
    stack = {}
    sem = Semaphore(1, 0, rate_limit=1)

    async def _call_(self):
        if self.sem.is_busy():
            return
        if not self.stack:
            return
        with tracebacksuppressor:
            async with self.sem:
                async with Delay(1):
                    for c, s in self.stack.items():
                        channel = self.bot.get_channel(c)
                        for k, v in s.items():
                            embs = deque()
                            for emb in v:
                                if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
                                    create_task(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
                                    embs.clear()
                                embs.append(emb)
                                reacts = None
                            if embs:
                                create_task(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
                    self.stack.clear()

    async def _send_(self, message, **void):
        if message.channel.id in self.data and not message.flags.is_crossposted and "\u2009\u2009" not in message.author.name:
            with tracebacksuppressor:
                embed = await self.bot.as_embed(message, link=True, colour=True)
                for c_id in tuple(self.data[message.channel.id]):
                    try:
                        channel = await self.bot.fetch_channel(c_id)
                    except:
                        print_exc()
                        self.data[message.channel.id].discard(c_id)
                    data = (message.guild.name + "\u2009\u2009#" + str(message.channel), to_png(message.guild.icon_url))
                    self.stack.setdefault(channel.id, {}).setdefault(data, []).append(embed)


class UpdateStarboards(Database):
    name = "starboards"

    async def _reaction_add_(self, message, react, **void):
        if message.guild and message.guild.id in self.data and message.channel.id != self.data[message.guild.id].get(react, (message.channel.id,))[-1]:
            table = self.data[message.guild.id]
            req = table[react][0]
            if req < inf:
                count = sum(r.count for r in message.reactions if str(r.emoji) == react)
                if count <= 1:
                    message = await message.channel.fetch_message(message.id)
                    self.bot.add_message(message, files=False, force=True)
                    count = sum(r.count for r in message.reactions if str(r.emoji) == react)
                if message.id not in table.setdefault(None, {}):
                    if count >= req and count < req + 2:
                        embed = await self.bot.as_embed(message, link=True, colour=True)
                        text, link = embed.description.rsplit("\n\n", 1)
                        description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table) + "   " + link
                        embed.description = lim_str(description, 4096)
                        # data = ("#" + str(message.channel), to_png(message.guild.icon_url))
                        try:
                            channel = await self.bot.fetch_channel(table[react][1])
                            m = await channel.send(embed=embed)
                        except (discord.NotFound, discord.Forbidden):
                            table.pop(react)
                        else:
                            table[None][message.id] = m.id
                            with tracebacksuppressor(RuntimeError, KeyError):
                                while len(table[None]) > 16384:
                                    table[None].pop(next(iter(table[None])))
                            self.update(message.guild.id)
                else:
                    try:
                        channel = await self.bot.fetch_channel(table[react][1])
                        m = await self.bot.fetch_message(table[None][message.id], channel)
                        embed = await self.bot.as_embed(message, link=True, colour=True)
                        text, link = embed.description.rsplit("\n\n", 1)
                        description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table) + "   " + link
                        embed.description = lim_str(description, 4096)
                        await m.edit(content=None, embed=embed)
                    except (discord.NotFound, discord.Forbidden):
                        table[None].pop(message.id, None)
                    else:
                        table[None][message.id] = m.id
                        with tracebacksuppressor(RuntimeError, KeyError):
                            while len(table[None]) > 16384:
                                table[None].pop(next(iter(table[None])))
                        self.update(message.guild.id)

    async def _edit_(self, after, **void):
        message = after
        try:
            table = self.data[message.guild.id]
        except KeyError:
            return
        if message.id in table.setdefault(None, {}):
            try:
                reacts = sorted(map(str, message.reactions), key=lambda r: -r.count)
                if not reacts:
                    return
                react = reacts[0]
                channel = await self.bot.fetch_channel(table[react][1])
                m = await self.bot.fetch_message(table[None][message.id], channel)
                embed = await self.bot.as_embed(message, link=True, colour=True)
                text, link = embed.description.rsplit("\n\n", 1)
                description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in reacts if str(r.emoji) in table) + "   " + link
                embed.description = lim_str(description, 4096)
                await m.edit(content=None, embed=embed)
            except (discord.NotFound, discord.Forbidden):
                table[None].pop(message.id, None)
            else:
                table[None][message.id] = m.id
                with tracebacksuppressor(RuntimeError, KeyError):
                    while len(table[None]) > 16384:
                        table[None].pop(next(iter(table[None])))
                self.update(message.guild.id)


class UpdateRolegivers(Database):
    name = "rolegivers"

    async def _nocommand_(self, text, message, orig, **void):
        if not message.guild or not orig:
            return
        user = message.author
        guild = message.guild
        bot = self.bot
        if bot.get_perms(user, message.guild) < 0:
            return
        assigned = self.data.get(message.channel.id, ())
        for k in assigned:
            if ((k in text) if is_alphanumeric(k) else (k in message.content.casefold())):
                al = assigned[k]
                for r in al[0]:
                    try:
                        role = await bot.fetch_role(r, guild)
                        if role is None:
                            raise LookupError
                    except LookupError:
                        al[0].remove(r)
                        bot.data.rolegivers.update(message.channel.id)
                        continue
                    if role in user.roles:
                        continue
                    with bot.ExceptionSender(message.channel):
                        await user.add_roles(
                            role,
                            reason=f'Keyword "{k}" found in message "{message.content}".',
                            atomic=True,
                        )
                        print(f"RoleGiver: Granted {role} to {user} in {guild}.")
                if alist[1]:
                    await bot.silent_delete(message)


class UpdateAutoRoles(Database):
    name = "autoroles"

    async def _join_(self, user, guild, **void):
        if guild.id in self.data and guild.me.guild_permissions.manage_roles:
            # Do not apply autorole to users who have roles from role preservers
            with suppress(KeyError):
                return self.bot.data.rolepreservers[guild.id][user.id]
            roles = deque()
            assigned = self.data[guild.id]
            for rolelist in assigned:
                with tracebacksuppressor:
                    role = await self.bot.fetch_role(rolelist.next(), guild)
                    roles.append(role)
            # Attempt to add all roles in one API call
            try:
                await user.add_roles(*roles, reason="AutoRole", atomic=False)
            except discord.Forbidden:
                await user.add_roles(*roles, reason="AutoRole", atomic=True)
            print(f"AutoRole: Granted {roles} to {user} in {guild}.")


class UpdateRolePreservers(Database):
    name = "rolepreservers"
    no_delete = True

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            if user.id in self.data[guild.id] and guild.me.guild_permissions.manage_roles:
                if guild.id not in self.bot.data.mutes or user.id not in (x["u"] for x in self.bot.data.mutes[guild.id]):
                    roles = deque()
                    assigned = self.data[guild.id][user.id]
                    for r_id in assigned:
                        with tracebacksuppressor:
                            role = await self.bot.fetch_role(r_id, guild)
                            roles.append(role)
                    roles = [role for role in roles if role < guild.me.top_role]
                    # Attempt to add all roles in one API call
                    try:
                        await user.edit(roles=roles, reason="RolePreserver")
                    except discord.Forbidden:
                        try:
                            await user.add_roles(*roles, reason="RolePreserver", atomic=False)
                        except discord.Forbidden:
                            await user.add_roles(*roles, reason="RolePreserver", atomic=True)
                    self.data[guild.id].pop(user.id, None)
                    print(f"RolePreserver: Granted {roles} to {user} in {guild}.")

    async def _leave_(self, user, guild, **void):
        if guild.id in self.data:
            # roles[0] is always @everyone
            roles = user.roles[1:]
            if roles:
                assigned = [role.id for role in roles]
                print("_leave_", guild, user, assigned)
                self.data[guild.id][user.id] = assigned
            else:
                print("_leave_", guild, user, None)
                self.data[guild.id].pop(user.id, None)
            self.update(guild.id)


class UpdateNickPreservers(Database):
    name = "nickpreservers"
    no_delete = True

    async def _join_(self, user, guild, **void):
        try:
            nick = self.data[guild.id][user.id]
        except KeyError:
            pass
        else:
            if guild.me.guild_permissions.manage_nicknames:
                await user.edit(nick=nick, reason="NickPreserver")
                self.data[guild.id].pop(user.id, None)
                print(f"NickPreserver: Granted {nick} to {user} in {guild}.")

    async def _leave_(self, user, guild, **void):
        if guild.id in self.data:
            if getattr(user, "nick", None):
                self.data[guild.id][user.id] = user.nick
                self.update(guild.id)
            else:
                if self.data[guild.id].pop(user.id, None):
                    self.update(guild.id)


class UpdatePerms(Database):
    name = "perms"