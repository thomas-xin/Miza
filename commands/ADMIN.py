try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

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
        delD = {}
        if end is None:
            dt = None
            # Keep going until finding required amount of messages or reaching the end of the channel
            while count > 0:
                lim = count * 2 + 16 if count < inf else None
                after = utc_dt() - datetime.timedelta(days=14) if "i" not in flags else None
                found = False
                if dt is None or after is None or dt > after:
                    async with bot.guild_semaphore:
                        async for m in channel.history(limit=lim, before=dt, after=after, oldest_first=False):
                            bot.add_message(m, force=True)
                            found = True
                            dt = m.created_at
                            if uset is None and m.author.bot or uset and m.author.id in uset:
                                delD[m.id] = m
                                count -= 1
                                if count <= 0:
                                    break
                if lim is None or not found:
                    break
        else:
            async with bot.guild_semaphore:
                async for m in channel.history(limit=None, before=cdict(id=end), after=cdict(id=start), oldest_first=False):
                    bot.add_message(m, force=True)
                    if uset is None and m.author.bot or uset and m.author.id in uset:
                        delD[m.id] = m
        if len(delD) >= 64 and "f" not in flags:
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(delD))} MESSAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0))
        # attempt to bulk delete up to 100 at a time, otherwise delete 1 at a time
        deleted = 0
        delM = alist(delD.values())
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
        if not "h" in flags:
            return italics(css_md(f"Deleted {sqr_md(deleted)} message{'s' if deleted != 1 else ''}!"))


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
    rate_limit = (2, 5)
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, message, channel, guild, flags, perm, user, name, **void):
        if not args and not argl:
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-admin-mute-"
                + str(user.id) + "_0"
                + "-\nLoading mute list...```*"
            )
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
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0))
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
                await send_exception(channel, ex)
                continue
            elif not p + 1 <= perm and not isnan(perm):
                reason = "to mute " + str(user) + " in " + guild.name
                ex = self.perm_error(perm, p + 1, reason)
                await send_exception(channel, ex)
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
                async with ExceptionSender(channel):
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
        async with ExceptionSender(channel):
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
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                async with delay(0.5):
                    create_task(message.add_reaction(as_str(react)))


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
    rate_limit = (2, 5)
    multi = True
    slash = True

    async def __call__(self, bot, args, argl, message, channel, guild, flags, perm, user, name, **void):
        if not args and not argl:
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-admin-ban-"
                + str(user.id) + "_0"
                + "-\nLoading ban list...```*"
            )
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
            return css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0))
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
                await send_exception(channel, ex)
                continue
            elif not p + 1 <= perm and not isnan(perm):
                reason = "to ban " + str(user) + " from " + guild.name
                ex = self.perm_error(perm, p + 1, reason)
                await send_exception(channel, ex)
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
                async with ExceptionSender(channel):
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
        async with ExceptionSender(channel):
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
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                async with delay(0.5):
                    create_task(message.add_reaction(as_str(react)))


class RoleGiver(Command):
    server_only = True
    name = ["Verifier"]
    min_level = 3
    min_display = "3+"
    description = "Adds an automated role giver to the current channel."
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
        rolelist = guild.roles
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
                        role = choice(roles)
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

    def __call__(self, flags, guild, **void):
        update = self.bot.data.rolepreservers.update
        bot = self.bot
        following = bot.data.rolepreservers
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
            return ini_md(f"Role preservation is currently disabled in {sqr_md(guild)}. Use ?e to enable.")
        return ini_md(f"Role preservation is currently enabled in {sqr_md(guild)}.")


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
        async with ExceptionSender(channel):
            await role.edit(permissions=perm, reason="Server Lockdown.")

    async def invLock(self, inv, channel):
        async with ExceptionSender(channel):
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
            async with delay(0.32):
                if s:
                    s += "\n\n"
                s += "\n\n".join(message_repr(m, limit=2048, username=True) for m in h[:4096])
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

    async def __call__(self, bot, flags, channel, guild, **void):
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
        return ini_md(f"User event logging is currently disabled in {sqr_md(guild)}. Use ?e to enable.")


class MessageLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log message events from the server, in the current channel."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, **void):
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
        return ini_md(f"Message event logging is currently disabled in {sqr_md(guild)}. Use ?e to enable.")


class FileLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log deleted files from the server, in the current channel."
    usage = "(enable|disable)?"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, **void):
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
        return ini_md(f"File deletion logging is currently disabled in {sqr_md(guild)}. Use ?e to enable.")


class StarBoard(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to repost popular messages with a certain number of a specified reaction anywhere from the server, into the current channel."
    usage = "<0:reaction> <1:react_count(1)>? <disable{?d}>?"
    flags = "d"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 1

    async def __call__(self, bot, args, user, channel, guild, flags, **void):
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
                    data.update(guild.id)
                return italics(css_md(f"Disabled starboard trigger {sqr_md(emoji)} for {sqr_md(guild)}."))
            for c_id, v in data.items():
                data.pop(guild.id, None)
            return italics(css_md(f"Disabled all starboard reposting for {sqr_md(guild)}."))
        if not args:
            return (
                "*```" + "\n" * ("z" in flags) + "callback-admin-starboard-"
                + str(user.id) + "_0"
                + "-\nLoading Starboard database...```*"
            )
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
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
            msg = "```ini\n" + iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}) + "```"
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.5)


class Crosspost(Command):
    server_only = True
    name = ["Repost", "Subscribe"]
    min_level = 3
    description = "Causes ⟨MIZA⟩ to automatically crosspost all messages from the target channel, into the current channel."
    usage = "<channel> <disable{?d}>?"
    flags = "aed"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 1

    async def __call__(self, bot, argv, flags, user, channel, guild, **void):
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
            return (
                "*```" + "\n" * ("z" in flags) + "callback-admin-crosspost-"
                + str(user.id) + "_0"
                + "-\nLoading Crosspost database...```*"
            )
        target = await bot.fetch_channel(argv)
        if not target.guild.get_member(user.id) or not target.permissions_for(target.guild.me).read_messages or not target.permissions_for(target.guild.get_member(user.id)).read_messages:
            raise PermissionError("Cannot follow channels without read message permissions.")
        channels = data.setdefault(target.id, set())
        channels.add(channel.id)
        data.update(target.id)
        return ini_md(f"Now crossposting all messages from {sqr_md(target)} to {sqr_md(channel)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
            msg = "```ini\n" + iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}) + "```"
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.5)


class Publish(Command):
    server_only = True
    name = ["News", "AutoPublish"]
    min_level = 3
    description = "Causes ⟨MIZA⟩ to automatically publish all posted messages in the current channel."
    usage = "(enable|disable)? <force{?x}>?"
    flags = "aedx"
    rate_limit = 1

    async def __call__(self, bot, flags, message, channel, guild, **void):
        data = bot.data.publishers
        if "e" in flags or "a" in flags:
            if channel.type != discord.ChannelType.news:
                raise TypeError("This feature can only be used in announcement channels.")
            if not channel.permissions_for(guild.me).manage_messages:
                raise PermissionError("Manage messages permission required to publish messages in channel.")
            data[channel.id] = 0 if "x" in flags else message.id
            return italics(css_md(f"Enabled automatic message publishing in {sqr_md(channel)} for {sqr_md(guild)}."))
        elif "d" in flags:
            if channel.id in data:
                data.pop(channel.id)
            return italics(css_md(f"Disabled automatic message publishing for {sqr_md(guild)}."))
        if channel.id in data:
            return ini_md(f"Automatic message publishing is currently enabled in {sqr_md(channel)}.")
        return ini_md(f"Automatic message publishing is currently disabled in {sqr_md(channel)}. Use ?e to enable.")


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

    async def get(self, guild):
        with suppress(KeyError):
            return self.bot.cache.roles[self.data[guild.id]]
        role = await guild.create_role(name="Muted", colour=discord.Colour(1), reason="Mute role setup.")
        self.bot.cache.roles[role.id] = role
        self.data[guild.id] = role.id
        self.update(guild.id)
        for channel in guild.channels:
            if channel.permissions_for(guild.me).manage_channels and not channel.permissions_synced:
                with tracebacksuppressor:
                    await channel.set_permissions(target=role, overwrite=self.mute)
        return role

    async def __call__(self):
        for g_id in tuple(self.data):
            guild = self.bot.cache.guilds.get(g_id)
            role = self.bot.cache.roles[self.data[g_id]]
            if guild:
                if role not in guild.roles:
                    self.data.pop(g_id)
                role = await self.get(guild)
                for channel in guild.channels:
                    if channel.permissions_for(guild.me).manage_channels and not channel.permissions_synced:
                        if role not in channel.overwrites:
                            with tracebacksuppressor:
                                await channel.set_permissions(target=role, overwrite=self.mute)


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


class UpdateUserLogs(Database):
    name = "logU"

    # Send a member update globally for all user updates
    async def _user_update_(self, before, after, **void):
        for guild in self.bot.guilds:
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
            b_url = best_url(before)
            a_url = best_url(after)
            emb.set_author(name=str(after), icon_url=a_url, url=a_url)
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
            if b_url != a_url:
                emb.add_field(
                    name="Avatar",
                    value=f"[Before]({b_url}) ➡️ [After]({a_url})",
                )
                emb.set_thumbnail(url=a_url)
                change = True
                colour[2] += 255
            if change:
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
        return  m_id // 10 ** 14

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
            with open(path, "rb") as f:
                out = zipped = f.read()
            with tracebacksuppressor(zipfile.BadZipFile):
                out = zip2bytes(zipped)
            data = pickle.loads(out)
            if type(data) is not dict:
                data = {m["id"]: m for m in data}
            self.raws[fn] = data
            if raw:
                print(f"{len(data)} message{'s' if len(data) != 1 else ''} temporarily read from {fn}")
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
            print(f"{len(data)} message{'s' if len(data) != 1 else ''} successfully loaded from {fn}")
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
            if type(message) is bot.CachedMessage:
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
        safe_save(path, out)
        return len(saved)

    async def _save_(self, **void):
        if self.save_sem.is_busy():
            return
        async with self.save_sem:
            # with suppress(AttributeError):
            #     fut = create_task(self.bot.data.channel_cache.saves())
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
            limit = str(self.get_fn(time_snowflake(utc_dt() - datetime.timedelta(days=14))))
            for f in os.listdir(self.files):
                if f.isnumeric() and f < limit or f.endswith("\x7f"):
                    with tracebacksuppressor(FileNotFoundError):
                        os.remove(self.files + "/" + f)
                        deleted += 1
            if deleted >= 8:
                print(f"Message Database: {deleted} files deleted.")
            if os.path.exists(self.files + "/-1"):
                self.setmtime()
            # await fut

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
        with tracebacksuppressor:
            for guild in self.bot.guilds:
                print("Probing", guild)
                futs = deque()
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).read_message_history:
                        futs.append(create_task(self.save_channel(channel, t)))
                for fut in futs:
                    await fut
                self.callback(None)
        self.bot.data.message_cache.finished = True
        self.bot.data.message_cache.setmtime()
        print("Loading new messages completed.")

    def callback(self, messages, **void):
        create_future_ex(self.bot.update_from_client, priority=True)
        return messages

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
                emb = as_embed(message)
                emb.colour = discord.Colour(0x00FFFF)
                action = f"**Slash command executed in** {channel_mention(message.channel.id)}:\nhttps://discord.com/channels/{guild.id}/{message.channel.id}/{message.id}\n"
                emb.description = lim_str(action + emb.description, 2048)
                emb.timestamp = message.created_at
                self.bot.send_embeds(channel, emb)

    # Edit events are rather straightforward to log
    async def _edit_(self, before, after, **void):
        if not after.author.bot:
            guild = before.guild
            if guild.id in self.data:
                c_id = self.data[guild.id]
                try:
                    channel = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    return
                emb = as_embed(after)
                emb2 = as_embed(before)
                emb.colour = discord.Colour(0x0000FF)
                action = f"**Message edited in** {channel_mention(after.channel.id)}:\nhttps://discord.com/channels/{guild.id}/{after.channel.id}/{after.id}"
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
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                return
            now = utc_dt()
            u = message.author
            name_id = str(u)
            url = best_url(u)
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
                        s = (5, 3600)[c >= 1]
                        cid = e.extra.channel.id
                        targ = e.target.id
                        if now - h < datetime.timedelta(seconds=s):
                            if targ == u.id and cid == message.channel.id:
                                t = e.user
                                init = user_mention(t.id)
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = as_embed(message)
            emb.colour = discord.Colour(0xFF0000)
            action = f"{init} **deleted message from** {channel_mention(message.channel.id)}:\nhttps://discord.com/channels/{guild.id}/{message.channel.id}/{message.id}\n"
            emb.description = lim_str(action + emb.description, 2048)
            emb.timestamp = message.created_at
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
                            if e.target is None or e.target.id == messages[-1].channel.id:
                                t = e.user
                                init = user_mention(t.id)
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=0xFF00FF)
            emb.description = f"{init} **deleted {len(messages)} message{'s' if len(messages) != 1 else ''} from** {channel_mention(messages[-1].channel.id)}:\n"
            for message in messages:
                nextline = f"\nhttps://discord.com/channels/{guild.id}/{message.channel.id}/{message.id}"
                if len(emb.description) + len(nextline) > 2048:
                    break
                emb.description += nextline
            embs = deque([emb])
            for message in messages:
                emb = as_embed(message)
                emb.colour = discord.Colour(0x7F007F)
                emb.timestamp = message.created_at
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
                                with delay(1):
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
                    await send_exception(message.channel, ex)


class UpdateCrossposts(Database):
    name = "crossposts"
    stack = {}
    sem = Semaphore(1, 0, rate_limit=1)

    async def _call_(self):
        if self.sem.is_busy():
            return
        if self.stack:
            with tracebacksuppressor:
                async with self.sem:
                    async with delay(1):
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
                embed = as_embed(message)
                col = await self.bot.get_colour(message.author)
                embed.colour = discord.Colour(col)
                for c_id in tuple(self.data[message.channel.id]):
                    try:
                        channel = await self.bot.fetch_channel(c_id)
                    except:
                        print_exc()
                        self.data[message.channel.id].discard(c_id)
                    data = (message.guild.name + "\u2009\u2009#" + str(message.channel), to_png(message.guild.icon_url))
                    self.stack.setdefault(channel.id, {}).setdefault(data, alist()).append(embed)


class UpdateStarboards(Database):
    name = "starboards"

    def _bot_ready_(self, **void):
        if "triggered" not in self.data:
            self.data["triggered"] = set()

    async def _reaction_add_(self, message, react, count, **void):
        if message.guild and message.guild.id in self.data:
            req = self.data[message.guild.id].get(react, (inf,))[0]
            # print(react, count, req)
            if count >= req and count < req + 2:
                if message.id not in self.data["triggered"]:
                    self.data["triggered"].add(message.id)
                    with tracebacksuppressor(RuntimeError, KeyError):
                        while len(self.data["triggered"]) > 4096:
                            self.data["triggered"].discard(next(iter(self.data["triggered"])))
                    with tracebacksuppressor:
                        embed = as_embed(message)
                        col = await self.bot.get_colour(message.author)
                        embed.colour = discord.Colour(col)
                        data = ("#" + str(message.channel), to_png(message.guild.icon_url))
                        self.bot.data.crossposts.stack.setdefault(self.data[message.guild.id][react][1], {}).setdefault(data, alist()).append(embed)


class UpdateRolegivers(Database):
    name = "rolegivers"

    async def _nocommand_(self, text, message, orig, **void):
        if message.guild is None or not orig:
            return
        user = message.author
        guild = message.guild
        bot = self.bot
        assigned = self.data.get(message.channel.id, ())
        for k in assigned:
            if ((k in text) if is_alphanumeric(k) else (k in message.content.casefold())):
                alist = assigned[k]
                for r in alist[0]:
                    try:
                        role = await bot.fetch_role(r, guild)
                        if role is None:
                            raise LookupError
                    except LookupError:
                        alist[0].remove(r)
                        continue
                    if role in user.roles:
                        continue
                    async with ExceptionSender(message.channel):
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
        if guild.id in self.data:
            # Do not apply autorole to users who have roles from role preservers
            with suppress(KeyError):
                return self.bot.data.rolepreservers[guild.id][user.id]
            roles = deque()
            assigned = self.data[guild.id]
            for rolelist in assigned:
                with tracebacksuppressor:
                    role = await self.bot.fetch_role(choice(rolelist), guild)
                    roles.append(role)
            print(f"AutoRole: Granted {roles} to {user} in {guild}.")
            # Attempt to add all roles in one API call
            try:
                await user.add_roles(*roles, reason="AutoRole", atomic=False)
            except discord.Forbidden:
                await user.add_roles(*roles, reason="AutoRole", atomic=True)


class UpdateRolePreservers(Database):
    name = "rolepreservers"
    no_delete = True

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            if user.id in self.data[guild.id]:
                if guild.id not in self.bot.data.mutes or user.id not in (x["u"] for x in self.bot.data.mutes[guild.id]):
                    roles = deque()
                    assigned = self.data[guild.id][user.id]
                    for r_id in assigned:
                        with tracebacksuppressor:
                            role = await self.bot.fetch_role(r_id, guild)
                            roles.append(role)
                    print(f"RolePreserver: Granted {roles} to {user} in {guild}.")
                    # Attempt to add all roles in one API call
                    try:
                        await user.edit(roles=roles, reason="RolePreserver")
                    except discord.Forbidden:
                        try:
                            await user.add_roles(*roles, reason="RolePreserver", atomic=False)
                        except discord.Forbidden:
                            await user.add_roles(*roles, reason="RolePreserver", atomic=True)
                    self.data[guild.id].pop(user.id)

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


class UpdatePerms(Database):
    name = "perms"