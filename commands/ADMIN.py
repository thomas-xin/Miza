try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Purge(Command):
    time_consuming = True
    _timeout_ = 16
    name = ["Del", "Delete"]
    min_level = 3
    description = "Deletes a number of messages from a certain user in current channel."
    usage = "<1:user{bot}> <everyone(?a)> <0:count[1]> <hide(?h)>"
    flags = "aeh"
    rate_limit = 2

    async def __call__(self, client, bot, argv, args, channel, name, flags, perm, guild, **void):
        t_user = -1
        if "a" in flags or "e" in flags or "@" in argv and ("everyone" in argv or "here" in argv):
            t_user = None
        if len(args) < 2:
            if t_user == -1:
                t_user = client.user
            if len(args) < 1:
                count = 1
            else:
                num = await bot.evalMath(args[0], guild.id)
                count = round(num)
        else:
            a1 = args[0]
            a2 = " ".join(args[1:])
            num = await bot.evalMath(a2, guild.id)
            count = round(num)
            if t_user == -1:
                u_id = verifyID(a1)
                try:
                    t_user = await bot.fetch_user(u_id)
                except (TypeError, discord.NotFound):
                    try:
                        t_user = await bot.fetch_member_ex(u_id, guild)
                    except LookupError:
                        t_user = cdict(id=u_id)
        if count <= 0:
            raise ValueError("Please enter a valid amount of messages to delete.")
        dt = None
        delD = {}
        deleted = 0
        # Keep going until finding required amount of messages or reaching the end of the channel
        while count > 0:
            lim = count * 2 + 16
            if not lim < inf:
                lim = None
            hist = await channel.history(limit=lim, before=dt).flatten()
            isbot = t_user is not None and t_user.id == client.user.id
            for i in range(len(hist)):
                m = hist[i]
                if t_user is None or isbot and m.author.bot or m.author.id == t_user.id:
                    delD[m.id] = m
                    count -= 1
                    if count <= 0:
                        break
                if i == len(hist) - 1:
                    dt = m.created_at
            if lim is None or not hist:
                break
        # attempt to bulk delete up to 100 at a time, otherwise delete 1 at a time
        delM = hlist(delD.values())
        while len(delM):
            try:
                if hasattr(channel, "delete_messages"):
                    dels = delM[:100]
                    # bot.logDelete(dels[-1], -1)
                    await channel.delete_messages(dels)
                    deleted += len(dels)
                    for _ in loop(len(dels)):
                        delM.popleft()
                else:
                    await bot.silentDelete(delM[0], no_log=-1, exc=True)
                    deleted += 1
                    delM.popleft()
            except:
                print(traceback.format_exc())
                for _ in loop(min(5, len(delM))):
                    m = delM.popleft()
                    await bot.silentDelete(m, no_log=-1, exc=True)
                    deleted += 1
        if not "h" in flags:
            return (
                "```css\nDeleted [" + noHighlight(deleted)
                + "] message" + "s" * (deleted != 1) + "!```"
            )


class Ban(Command):
    server_only = True
    name = ["Bans", "Unban"]
    min_level = 3
    min_display = "3+"
    description = "Bans a user for a certain amount of time, with an optional reason."
    usage = "<0:user> <1:time[]> <2:reason[]> <hide(?h)> <debug(?z)>"
    flags = "hz"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 2

    async def __call__(self, bot, args, message, channel, guild, flags, perm, user, name, **void):
        if not args:
            # Set callback message for scrollable list
            return (
                "```" + "\n" * ("z" in flags) + "callback-admin-ban-"
                + str(user.id) + "_0"
                + "-\nLoading ban list...```"
            )
        update = self.bot.database.bans.update
        ts = utc()
        banlist = bot.data.bans.get(guild.id, [])
        fut = create_task(channel.trigger_typing())
        try:
            bans, glob = await self.getBans(guild)
        except:
            await fut
            raise
        await fut
        u_id = verifyID(args.pop(0))
        try:
            user = await bot.fetch_user(u_id)
            users = [user]
        except (TypeError, discord.NotFound):
            try:
                member = await bot.fetch_member_ex(u_id, guild)
                users = [member]
            except LookupError:
                role = await bot.fetch_role(u_id, guild)
                users = [role.members]
        if not args or name == "unban":
            user = users[0]
            try:
                ban = bans[user.id]
            except LookupError:
                return (
                    "```ini\n[" + noHighlight(user)
                    + "] is currently not banned from [" + noHighlight(guild) + "].```"
                )
            if name == "unban":
                await guild.unban(user)
                try:
                    ind = banlist.search(user.id, key=lambda b: b["u"])
                except LookupError:
                    pass
                else:
                    banlist.pops(ind)["u"]
                    if 0 in ind:
                        try:
                            bot.database.bans.listed.remove(guild.id, key=lambda x: x[-1])
                        except LookupError:
                            pass
                        if banlist:
                            bot.database.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
                    update()
                return (
                    "```ini\nSuccessfully unbanned [" + noHighlight(user)
                    + "] from [" + noHighlight(guild) + "].```"
                )
            return (
                "```ini\nCurrent ban for [" + noHighlight(user)
                + "] from [" + noHighlight(guild) + "]: ["
                + sec2Time(ban["t"] - ts) + "].```"
            )
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
        num = await bot.evalTime(expr, guild, op=False)
        create_task(message.add_reaction("❗"))
        for user in users:
            p = bot.getPerms(user, guild)
            if not p < 0 and not isValid(p):
                await channel.send("```py\nError: " + repr(PermissionError(
                    str(user) + " has infinite permission level, "
                    + "and cannot be banned from this server.```"
                )))
                continue
            elif not p + 1 <= perm and not isnan(perm):
                reason = "to ban " + str(user) + " from " + guild.name
                await channel.send("```py\nError: " + repr(self.permError(perm, p + 1, reason)) + "```")
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
        banlist = setDict(bot.data.bans, guild.id, hlist())
        update = bot.database.bans.update
        for b in glob:
            u = b.user
            if user.id == u.id:
                try:
                    ban = bans[u.id]
                    # Remove from global schedule, then sort and re-add
                    try:
                        banlist.remove(user.id, key=lambda x: x["u"])
                    except IndexError:
                        pass
                    try:
                        bot.database.bans.listed.remove(guild.id, key=lambda x: x[-1])
                    except LookupError:
                        pass
                    if length < inf:
                        banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": ban.get("r", None)}, key=lambda x: x["t"])
                        bot.database.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
                    print(banlist)
                    print(bot.database.bans.listed)
                    update()
                    await channel.send(
                        "```css\nUpdated ban for " + sbHighlight(user)
                        + " from [" + sec2Time(ban["t"] - ts)
                        + "] to [" + sec2Time(length) + "].```"
                    )
                except Exception as ex:
                    print(traceback.format_exc())
                    await channel.send("```py\nError: " + repr(ex) + "```")
                return
        try:
            await guild.ban(user, reason=reason, delete_message_days=0)
            try:
                banlist.remove(user.id, key=lambda x: x["u"])
            except IndexError:
                pass
            try:
                bot.database.bans.listed.remove(guild.id, key=lambda x: x[-1])
            except LookupError:
                pass
            if length < inf:
                banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": reason}, key=lambda x: x["t"])
                bot.database.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
            print(banlist)
            print(bot.database.bans.listed)
            update()
            await channel.send(
                "```css\n" + sbHighlight(user)
                + " has been banned from " + sbHighlight(guild)
                + " for [" + sec2Time(length) + "]. Reason: "
                + sbHighlight(reason) + "```"
            )
        except Exception as ex:
            print(traceback.format_exc())
            await channel.send("```py\nError: " + repr(ex) + "```")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        update = self.bot.database.bans.update
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
        content = content[:i] + (
            "callback-admin-ban-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not bans:
            content += "Ban list for " + str(guild).replace("`", "") + " is currently empty.```"
        else:
            content += str(len(bans)) + " users currently banned from " + str(guild).replace("`", "") + ":```"
        emb = discord.Embed(colour=discord.Colour(1))
        emb.description = content
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        for i, ban in enumerate(sorted(bans.values(), key=lambda x: x["t"])[pos:pos + page]):
            try:
                user = await bot.fetch_user(ban["u"])
                emb.add_field(
                    name=str(user) + " (" + str(user.id) + ")",
                    value=(
                        "Duration: " + sec2Time(ban["t"] - ts) + "\n"
                        + "Reason: " + escape_markdown(str(ban["r"]))
                    )
                )
            except:
                print(traceback.format_exc())
        more = len(bans) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class RoleGiver(Command):
    server_only = True
    name = ["Verifier"]
    min_level = 3
    min_display = "3+"
    description = "Adds an automated role giver to the current channel."
    usage = "<0:react_to[]> <1:role[]> <delete_messages(?x)> <disable(?d)>"
    flags = "aedx"
    no_parse = True
    rate_limit = 1

    async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
        update = self.bot.database.rolegivers.update
        bot = self.bot
        data = bot.data.rolegivers
        if "d" in flags:
            if argv:
                react = args[0].lower()
                assigned = data.get(channel.id, {})
                if react not in assigned:
                    raise LookupError("Rolegiver " + react + " not currently assigned for #" + channel.name + ".")
                assigned.pop(react)
                return "```css\nRemoved [" + react + "] from the rolegiver list for [#" + noHighlight(channel.name) + "].```"
            if channel.id in data:
                data.pop(channel.id)
                update()
            return "```css\nRemoved all automated rolegivers from [#" + noHighlight(channel.name) + "].```"
        assigned = setDict(data, channel.id, {})
        if not argv:
            key = lambda alist: "⟨" + ", ".join(str(r) for r in alist[0]) + "⟩, delete: " + str(alist[1])
            if not assigned:
                return (
                    "```ini\nNo currently active rolegivers for [#"
                    + noHighlight(channel) + "].```"
                )
            return (
                "Currently active rolegivers in <#" + str(channel.id)
                + ">:\n```ini\n" + strIter(assigned, key=key) + "```"
            )
        if sum(len(alist[0]) for alist in assigned) >= 8:
            raise OverflowError(
                "Rolegiver list for #" + channel.name
                + " has reached the maximum of 8 items. "
                + "Please remove an item to add another."
            )
        react = args[0].lower()
        if len(react) > 64:
            raise OverflowError("Search substring too long.")
        r = verifyID(reconstitute(" ".join(args[1:])))
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
            role = await strLookup(
                rolelist,
                r,
                qkey=lambda x: [str(x), reconstitute(x).replace(" ", "").lower()],
            )
        # Must ensure that the user does not assign roles higher than their own
        if inf > perm:
            memb = await self.bot.fetch_member_ex(user.id, guild)
            if memb is None:
                raise LookupError("Member data not found for this server.")
            if memb.top_role <= role:
                raise PermissionError("Target role is higher than your highest role.")
        alist = setDict(assigned, react, [[], False])
        alist[1] |= "x" in flags
        alist[0].append(role.id) 
        update()
        return (
            "```css\nAdded [" + noHighlight(react)
            + "] ➡️ [" + noHighlight(role)
            + "] to channel [#" + noHighlight(channel.name) + "].```"
        )


class AutoRole(Command):
    server_only = True
    name = ["InstaRole"]
    min_level = 3
    min_display = "3+"
    _timeout_ = 7
    description = (
        "Causes any new user joining the server to automatically gain the targeted role.\n"
        + "Input multiple roles to create a randomized role giver."
    )
    usage = "<role[]> <disable(?d)> <update_all(?x)>"
    flags = "aedx"
    rate_limit = 1

    async def __call__(self, argv, args, name, user, channel, guild, perm, flags, **void):
        update = self.bot.database.autoroles.update
        bot = self.bot
        data = bot.data.autoroles
        if "d" in flags:
            assigned = data.get(guild.id, None)
            if argv and assigned:
                i = await bot.evalMath(argv, guild)
                roles = assigned.pop(i)
                removed = deque()
                for r in roles:
                    try:
                        role = await bot.fetch_role(r, guild)
                    except:
                        print(traceback.format_exc())
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
                update()
                return "```css\nRemoved " + sbHighlight(", ".join(str(role) for role in removed)) + " from the autorole list for " + sbHighlight(guild) + ".```"
            if guild.id in data:
                data.pop(guild.id)
                update()
            return "```css\nRemoved all items from the autorole list for " + sbHighlight(guild) + ".```"
        assigned = setDict(data, guild.id, hlist())
        if not argv:
            rlist = hlist()
            for roles in assigned:
                new = hlist()
                for r in roles:
                    role = await bot.fetch_role(r, guild)
                    new.append(role)
                rlist.append(new)
            if not assigned:
                return (
                    "```ini\nNo currently active autoroles for " + sbHighlight(guild) + ".```"
                )
            return (
                "Currently active autoroles for **" + escape_markdown(guild.name)
                + "**:\n```ini\n" + strIter(rlist) + "```"
            )
        if sum(len(alist) for alist in assigned) >= 8:
            raise OverflowError(
                "Autorole list for " + channel.name
                + " has reached the maximum of 8 items. "
                + "Please remove an item to add another."
            )
        roles = hlist()
        rolenames = (verifyID(i) for i in args)
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
                role = await strLookup(
                    rolelist,
                    r,
                    qkey=lambda x: [str(x), reconstitute(x).replace(" ", "").lower()],
                )
            # Must ensure that the user does not assign roles higher than their own
            if not inf > perm:
                memb = await self.bot.fetch_member_ex(user.id, guild)
                if memb is None:
                    raise LookupError("Member data not found for this server.")
                if memb.top_role <= role:
                    raise PermissionError("Target role is higher than your highest role.")
            roles.append(role)
        new = hlist(role.id for role in roles)
        if new not in assigned:
            assigned.append(new)
            update()
        # Update all users by adding roles
        if "x" in flags or name == "instarole":
            if roles:
                fut = create_task(channel.trigger_typing())
                i = 1
                for member in guild.members:
                    role = random.choice(roles)
                    if role not in member.roles:
                        create_task(member.add_roles(role, reason="InstaRole", atomic=True))
                        if not i % 5:
                            await asyncio.sleep(5)
                        i += 1
                await fut
        return (
            "```css\nAdded [" + noHighlight(", ".join(str(role) for role in roles))
            + "] to the autorole list for [" + noHighlight(guild) + "].```"
        )


class RolePreserver(Command):
    server_only = True
    name = ["StickyRoles"]
    min_level = 3
    min_display = "3+"
    description = "Causes ⟨MIZA⟩ to save roles for all users, and re-add them when they leave and rejoin."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, flags, guild, **void):
        update = self.bot.database.rolepreservers.update
        bot = self.bot
        following = bot.data.rolepreservers
        # Empty dictionary is enough to represent an active role preserver here
        curr = following.get(guild.id)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
                update()
            return "```css\nDisabled role preservation for [" + noHighlight(guild) + "].```"
        elif "e" in flags or "a" in flags:
            following[guild.id] = {}
            update()
            return "```css\nEnabled role preservation for [" + noHighlight(guild) + "].```"
        else:
            return (
                "```ini\nRole preservation is currently " + "not " * (curr is None)
                + "enabled in [" + noHighlight(guild) + "].```"
            )


class Lockdown(Command):
    server_only = True
    min_level = inf
    description = "Completely locks down the server by removing send message permissions for all users and revoking all invites."
    flags = "f"
    rate_limit = 30

    async def roleLock(self, role, channel):
        perm = role.permissions
        perm.administrator = False
        perm.send_messages = False
        try:
            await role.edit(permissions=perm, reason="Server Lockdown.")
        except Exception as ex:
            await channel.send(limStr("```py\n" + noHighlight(repr(ex)) + "```", 2000))
    
    async def invLock(self, inv, channel):
        try:
            await inv.delete(reason="Server Lockdown.")
        except Exception as ex:
            await channel.send(limStr("```py\n" + noHighlight(repr(ex)) + "```", 2000))
    
    async def __call__(self, guild, channel, flags, **void):
        if "f" not in flags:
            response = uniStr(
                "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
            )
            return ("```asciidoc\n[" + response + "]```")
        u_id = self.bot.client.user.id
        for role in guild.roles:
            if len(role.members) != 1 or role.members[-1].id not in (u_id, guild.owner_id):
                create_task(self.roleLock(role, channel))
        invites = await guild.invites()
        for inv in invites:
            create_task(self.invLock(inv, channel))
        response = uniStr("LOCKDOWN REQUESTED.")
        return ("```asciidoc\n[" + response + "]```")


class SaveChannel(Command):
    time_consuming = 1
    _timeout_ = 10
    name = ["BackupChannel", "DownloadChannel"]
    min_level = 3
    description = "Saves a number of messages in a channel, as well as their contents, to a .txt file."
    usage = "<0:channel{current}> <1:message_limit[4096]>"

    async def __call__(self, guild, channel, args, **void):
        num = 4096
        ch = channel
        if args:
            if len(args) >= 2:
                num = await self.bot.evalMath(" ".join(args[1:]), guild)
                if not num <= 65536:
                    raise OverflowError("Maximum number of messages allowed is 65536.")
                if num <= 0:
                    raise ValueError("Please input a valid message limit.")
            ch = await self.bot.fetch_channel(verifyID(args[0]))
            if guild is None or hasattr(guild, "ghost"):
                if guild.id != ch.id:
                    raise PermissionError("Target channel is not in this server.")
            elif ch.id not in (c.id for c in guild.channels):
                raise PermissionError("Target channel is not in this server.")
        h = await ch.history(limit=num).flatten()
        h = h[::-1]
        s = ""
        while h:
            if s:
                s += "\n\n"
            s += "\n\n".join(strMessage(m, limit=2048, username=True) for m in h[:4096])
            h = h[4096:]
            await asyncio.sleep(0.32)
        return bytes(s, "utf-8")


class UserLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log user events from the server, in the current channel."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, **void):
        data = bot.data.logU
        update = bot.database.logU.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            update()
            return (
                "```css\nEnabled user logging in [" + noHighlight(channel.name)
                + "] for [" + noHighlight(guild.name) + "].```"
            )
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
                update()
            return (
                "```css\nDisabled user logging for [" + noHighlight(guild.name) + "].```"
            )
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return (
                "```css\nUser logging for [" + noHighlight(guild.name)
                + "] is currently enabled in [" + noHighlight(channel.name)
                + "].```"
            )
        return (
            "```css\nUser logging is currently disabled in ["
            + noHighlight(guild.name) + "].```"
        )


class MessageLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log message events from the server, in the current channel."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, **void):
        data = bot.data.logM
        update = bot.database.logM.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            update()
            return (
                "```css\nEnabled message logging in [" + noHighlight(channel.name)
                + "] for [" + noHighlight(guild.name) + "].```"
            )
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
                update()
            return (
                "```css\nDisabled message logging for [" + noHighlight(guild.name) + "].```"
            )
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return (
                "```css\nMessage logging for [" + noHighlight(guild.name)
                + "] is currently enabled in [" + noHighlight(channel.name)
                + "].```"
            )
        return (
            "```css\nMessage logging is currently disabled in ["
            + noHighlight(guild.name) + "].```"
        )


class FileLog(Command):
    server_only = True
    min_level = 3
    description = "Causes ⟨MIZA⟩ to log deleted files from the server, in the current channel."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"
    rate_limit = 1

    async def __call__(self, bot, flags, channel, guild, **void):
        if not bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server to log deleted files.")
        data = bot.data.logF
        update = bot.database.logF.update
        if "e" in flags or "a" in flags:
            data[guild.id] = channel.id
            update()
            return (
                "```css\nEnabled file logging in [" + noHighlight(channel.name)
                + "] for [" + noHighlight(guild.name) + "].```"
            )
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
                update()
            return (
                "```css\nDisabled file logging for [" + noHighlight(guild.name) + "].```"
            )
        if guild.id in data:
            c_id = data[guild.id]
            channel = await bot.fetch_channel(c_id)
            return (
                "```css\nFile logging for [" + noHighlight(guild.name)
                + "] is currently enabled in [" + noHighlight(channel.name)
                + "].```"
            )
        return (
            "```css\nFile logging is currently disabled in ["
            + noHighlight(guild.name) + "].```"
        )


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


# Just like the reminders database, the ban database needs to be this way to keep O(1) time complexity when idle.
class UpdateBans(Database):
    name = "bans"

    def __load__(self):
        d = self.data
        self.listed = hlist(sorted(((d[i][0]["t"], i) for i in d), key=lambda x: x[0]))

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
            if not temp:
                self.data.pop(g_id)
            else:
                z = temp[0]["t"]
                self.listed.insort((z, g_id), key=lambda x: x[0])
            print(self.listed)
            try:
                guild = await self.bot.fetch_guild(g_id)
                user = await self.bot.fetch_user(x.u)
                m = guild.get_member(self.bot.client.user.id)
                try:
                    channel = await self.bot.fetch_channel(x.c)
                    if not channel.permissions_for(m).send_messages:
                        raise LookupError
                except (LookupError, discord.Forbidden, discord.NotFound):
                    channel = await self.bot.get_first_sendable(guild, m)
                try:
                    await guild.unban(user, reason="Temporary ban expired.")
                    text = (
                        "```css\n[" + noHighlight(user)
                        + "] has been unbanned from [" + noHighlight(guild) + "].```"
                    )
                except:
                    text = (
                        "```css\nUnable to unban [" + noHighlight(user)
                        + "] from [" + noHighlight(guild) + "].```"
                    )
                await channel.send(text)
            except:
                print(traceback.format_exc())
            self.update()


# Triggers upon 3 channel deletions in 2 minutes or 6 bans in 10 seconds
class ServerProtector(Database):
    name = "prot"
    no_file = True

    async def kickWarn(self, u_id, guild, owner, msg):
        user = await self.bot.fetch_user(u_id)
        try:
            await guild.kick(user, reason="Triggered automated server protection response for excessive " + msg + ".")
            await owner.send(
                "Apologies for the inconvenience, but <@" + str(user.id) + "> `(" + str(user.id) + ")` has triggered an "
                + "automated server protection response due to exessive " + msg + " in `" + noHighlight(guild) + "` `(" + str(guild.id) + ")`, "
                + "and has been removed from the server to prevent any potential further attacks."
            )
        except discord.Forbidden:
            await owner.send(
                "Apologies for the inconvenience, but <@" + str(user.id) + "> `(" + str(user.id) + ")` has triggered an "
                + "automated server protection response due to exessive " + msg + " in `" + noHighlight(guild) + "` `(" + str(guild.id) + ")`, "
                + "and were unable to be automatically removed from the server; please watch them carefully to prevent any potential further attacks."
            )

    async def targetWarn(self, u_id, guild, msg):
        print("Channel Deletion warning by <@" + str(u_id) + "> in " + str(guild) + ".")
        user = self.bot.client.user
        owner = guild.owner
        if owner.id == user.id:
            owner = await self.bot.fetch_user(tuple(self.owners)[0])
        if u_id == guild.owner.id:
            if u_id == user.id:
                return
            user = guild.owner
            await owner.send(
                "Apologies for the inconvenience, but your account <@" + str(user.id) + "> `(" + str(user.id) + ")` has triggered an "
                + "automated server protection response due to exessive " + msg + " in `" + noHighlight(guild) + "` `(" + str(guild.id) + ")`. "
                + "If this was intentional, please ignore this message."
            )
        elif u_id == user.id:
            create_task(guild.leave())
            await owner.send(
                "Apologies for the inconvenience, but <@" + str(user.id) + "> `(" + str(user.id)+ ")` has triggered an "
                + "automated server protection response due to exessive " + msg + " in `" + noHighlight(guild) + "` `(" + str(guild.id) + ")`, "
                + "and will promptly leave the server to prevent any potential further attacks."
            )
        else:
            await self.kickWarn(u_id, guild, owner, msg)

    async def _channel_delete_(self, channel, guild, **void):
        if channel.id in self.bot.cache.deleted:
            return
        if self.bot.isTrusted(guild.id):
            audits = await guild.audit_logs(limit=5, action=discord.AuditLogAction.channel_delete).flatten()
            ts = utc()
            cnt = {}
            for log in audits:
                if ts - utc_ts(log.created_at) < 120:
                    addDict(cnt, {log.user.id: 1})
            for u_id in cnt:
                if cnt[u_id] > 2:
                    create_task(self.targetWarn(u_id, guild, "channel deletions `(" + str(cnt[u_id]) + ")`"))

    async def _ban_(self, user, guild, **void):
        if self.bot.isTrusted(guild.id):
            audits = await guild.audit_logs(limit=13, action=discord.AuditLogAction.ban).flatten()
            ts = utc()
            cnt = {}
            for log in audits:
                if ts - utc_ts(log.created_at) < 10:
                    addDict(cnt, {log.user.id: 1})
            for u_id in cnt:
                if cnt[u_id] > 5:
                    create_task(self.targetWarn(u_id, guild, "banning `(" + str(cnt[u_id]) + ")`"))


class UpdateUserLogs(Database):
    name = "logU"

    # Send a member update globally for all user updates
    async def _user_update_(self, before, after, **void):
        for guild in self.bot.client.guilds:
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
            print(traceback.format_exc())
            return
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            emb = discord.Embed()
            b_url = bestURL(before)
            a_url = bestURL(after)
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
                    sub = hlist()
                    add = hlist()
                    for r in before.roles:
                        if r not in after.roles:
                            sub.append(r)
                    for r in after.roles:
                        if r not in before.roles:
                            add.append(r)
                    rchange = ""
                    if sub:
                        rchange = "❌ " + escape_markdown(", ".join("<@&" + str(r.id) + ">" for r in sub))
                    if add:
                        rchange += (
                            "\n" * bool(rchange) + "✅ " 
                            + escape_markdown(", ".join("<@&" + str(r.id) + ">" for r in add))
                        )
                    if rchange:
                        emb.add_field(name="Roles", value=rchange)
                        change = True
                        colour[1] += 255
            if b_url != a_url:
                emb.add_field(
                    name="Avatar",
                    value=(
                        "[Before](" + str(b_url) 
                        + ") ➡️ [After](" 
                        + str(a_url) + ")"
                    ),
                )
                emb.set_thumbnail(url=a_url)
                change = True
                colour[2] += 255
            if change:
                emb.colour = colour2Raw(colour)
                self.bot.embedSender(channel, emb)

    async def _join_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            # Colour: White
            emb = discord.Embed(colour=16777214)
            url = bestURL(user)
            emb.set_author(name=str(user), icon_url=url, url=url)
            emb.description = (
                "<@" + str(user.id)
                + "> has joined the server."
            )
            self.bot.embedSender(channel, emb)
    
    async def _leave_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            # Colour: Black
            emb = discord.Embed(colour=1)
            url = bestURL(user)
            emb.set_author(name=str(user), icon_url=url, url=url)
            # Check audit log to find whether user left or was kicked/banned
            kick = None
            ban = None
            try:
                ts = utc()
                futs = [create_task(guild.audit_logs(limit=4, action=getattr(discord.AuditLogAction, action)).flatten()) for action in ("ban", "kick")]
                bans = await futs[0]
                kicks = await futs[1]
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
            except StopIteration:
                pass
            except:
                print(traceback.format_exc())
            if ban is not None:
                emb.description = (
                    "<@" + str(user.id)
                    + "> has been banned by <@"
                    + str(ban.id) + ">."
                )
                if ban.reason:
                    emb.description += "\nReason: `" + noHighlight(ban.reason) + "`"
            elif kick is not None:
                emb.description = (
                    "<@" + str(user.id)
                    + "> has been kicked by <@"
                    + str(kick.id) + ">."
                )
                if kick.reason:
                    emb.description += "\nReason: `" + noHighlight(kick.reason) + "`"
            else:
                emb.description = (
                    "<@" + str(user.id)
                    + "> has left the server."
                )
            self.bot.embedSender(channel, emb)


class UpdateMessageLogs(Database):
    name = "logM"

    def __load__(self):
        self.searched = False
        self.dc = {}

    async def __call__(self):
        for h in tuple(self.dc):
            if utc_dt() - h > datetime.timedelta(seconds=3600):
                self.dc.pop(h)
        if not self.searched:
            self.searched = True
            lim = floor(2097152 / len(self.bot.client.guilds))
            return [create_task(self.bot.database.counts.getGuildHistory(guild, lim, callback=self.callback)) for guild in self.bot.cache.guilds.values()]
    
    def callback(self, messages, **void):
        messages = [self.bot.cacheMessage(message) for message in messages]
        create_future_ex(self.bot.updateClient)
        return messages

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
                    self.update()
                    return
                u = before.author
                name = u.name
                name_id = name + bool(u.display_name) * ("#" + u.discriminator)
                url = bestURL(u)
                emb = discord.Embed(colour=colour2Raw(0, 0, 255))
                emb.set_author(name=name_id, icon_url=url, url=url)
                emb.description = (
                    "**Message edited in** <#"
                    + str(before.channel.id) + ">:"
                )
                emb.add_field(name="Before", value=strMessage(before))
                emb.add_field(name="After", value=strMessage(after))
                self.bot.embedSender(channel, emb)

    # Delete events must attempt to find the user who deleted the message
    async def _delete_(self, message, bulk=False, **void):
        cu_id = self.bot.client.user.id
        if bulk:
            return
        guild = message.guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            now = utc_dt()
            u = message.author
            name = u.name
            name_id = name + bool(u.display_name) * ("#" + u.discriminator)
            url = bestURL(u)
            action = discord.AuditLogAction.message_delete
            try:
                t = u
                init = "<@" + str(t.id) + ">"
                d_level = self.bot.isDeleted(message)
                if d_level:
                    if d_level > 1:
                        if d_level < 3:
                            pass
                            # self.logDeleted(message)
                        return
                    t = self.bot.client.user
                    init = "<@" + str(t.id) + ">"
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
                        cs = setDict(self.dc, h, 0)
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
                                init = "<@" + str(t.id) + ">"
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=colour2Raw(255, 0, 0))
            emb.set_author(name=name_id, icon_url=url, url=url)
            emb.description = (
                init + " **deleted message from** <#"
                + str(message.channel.id) + ">:\n"
            )
            emb.description += strMessage(message, limit=2048 - len(emb.description))
            self.bot.embedSender(channel, emb)

    # Thanks to the embed sender feature, which allows this feature to send up to 10 logs in one message
    async def _bulk_delete_(self, messages, **void):
        cu = self.bot.client.user
        cu_id = cu.id
        guild = messages[0].guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self.bot.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            now = utc_dt()
            action = discord.AuditLogAction.message_bulk_delete
            try:
                init = "[UNKNOWN USER]"
                if self.bot.isDeleted(messages[-1]):
                    t = self.bot.client.user
                    init = "<@" + str(t.id) + ">"
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
                        cs = setDict(self.dc, h, 0)
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
                                init = "<@" + str(t.id) + ">"
            except (PermissionError, discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=colour2Raw(255, 0, 255))
            emb.description = (
                init + " **deleted " + str(len(messages)) + " message" + "s" * (len(messages) != 1) + " from** <#"
                + str(messages[-1].channel.id) + ">:"
            )
            embs = deque([emb])
            for message in messages:
                u = message.author
                name = u.name
                name_id = name + bool(u.display_name) * ("#" + u.discriminator)
                url = bestURL(u)
                emb = discord.Embed(colour=colour2Raw(127, 0, 127))
                emb.set_author(name=name_id, icon_url=url, url=url)
                emb.description = strMessage(message, limit=2048)
                embs.append(emb)
            self.bot.embedSender(channel, embs)


class UpdateFileLogs(Database):
    name = "logF"

    async def _delete_(self, message, **void):
        if self.bot.isDeleted(message) > 1:
            return
        guild = message.guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            if message.attachments:
                try:
                    if not self.bot.isTrusted(guild.id):
                        raise EOFError
                    channel = await self.bot.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    self.update()
                    return
                # Attempt to recover files from their proxy URLs, otherwise send the proxy URLs
                msg = ""
                fils = []
                for a in message.attachments:
                    try:
                        try:
                            b = await a.read(use_cached=True)
                        except (discord.HTTPException, discord.NotFound):
                            b = await a.read(use_cached=False)
                        fil = discord.File(io.BytesIO(b), filename=str(a).split("/")[-1])
                        fils.append(fil)
                    except:
                        msg += bestURL(a) + "\n"
                emb = discord.Embed(colour=randColour())
                emb.description = "File" + "s" * (len(fils) + len(msg) != 1) + " deleted from <@" + str(message.author.id) + ">"
                if not msg:
                    msg = None
                await channel.send(msg, embed=emb, files=fils)


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
            if ((k in text) if is_alphanumeric(k) else (k in message.content.lower())):
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
                    await user.add_roles(
                        role,
                        reason="Keyword \"" + k + "\" found in message \"" + message.content + "\".",
                        atomic=True,
                    )
                    print("Granted role " + str(role) + " to " + str(user) + ".")
                if alist[1]:
                    await bot.silentDelete(message)


class UpdateAutoRoles(Database):
    name = "autoroles"

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            # Do not apply autorole to users who have roles from role preservers
            try:
                if user.id in self.bot.data.rolepreservers[guild.id]:
                    return
            except KeyError:
                pass
            roles = deque()
            assigned = self.data[guild.id]
            for rolelist in assigned:
                try:
                    role = await self.bot.fetch_role(random.choice(rolelist), guild)
                    roles.append(role)
                except:
                    print(traceback.format_exc())
            print("AutoRole", user, roles)
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
                roles = deque()
                assigned = self.data[guild.id][user.id]
                for r_id in assigned:
                    try:
                        role = await self.bot.fetch_role(r_id, guild)
                        roles.append(role)
                    except:
                        print(traceback.format_exc())
                print("RolePreserver", user, roles)
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
                try:
                    print("_leave_", guild, user, None)
                    self.data[guild.id].pop(user.id)
                except LookupError:
                    pass
            self.update()


class UpdatePerms(Database):
    name = "perms"