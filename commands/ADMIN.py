try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Purge(Command):
    time_consuming = True
    name = ["Del", "Delete"]
    min_level = 3
    description = "Deletes a number of messages from a certain user in current channel."
    usage = "<1:user{bot}(?a)> <0:count[1]> <hide(?h)>"
    flags = "ah"

    async def __call__(self, client, _vars, argv, args, channel, name, flags, perm, guild, **void):
        t_user = -1
        if "a" in flags or "everyone" in argv or "here" in argv:
            t_user = None
        if len(args) < 2:
            if t_user == -1:
                t_user = client.user
            if len(args) < 1:
                count = 1
            else:
                num = await _vars.evalMath(args[0], guild.id)
                count = round(num)
        else:
            a1 = args[0]
            a2 = " ".join(args[1:])
            num = await _vars.evalMath(a2, guild.id)
            count = round(num)
            if t_user == -1:
                u_id = verifyID(a1)
                try:
                    t_user = await _vars.fetch_user(u_id)
                except (TypeError, discord.NotFound):
                    try:
                        t_user = await _vars.fetch_member(u_id, guild)
                    except LookupError:
                        t_user = freeClass(id=u_id)
        lim = count * 2 + 16
        if lim < 0:
            lim = 0
        if not isValid(lim):
            lim = None
        hist = await channel.history(limit=lim).flatten()
        delM = hlist()
        isbot = t_user is not None and t_user.id == client.user.id
        deleted = 0
        for m in hist:
            if count <= 0:
                break
            if t_user is None or isbot and m.author.bot or m.author.id == t_user.id:
                delM.append(m)
                count -= 1
        while len(delM):
            try:
                if hasattr(channel, "delete_messages"):
                    await channel.delete_messages(delM[:100])
                    deleted += min(len(delM), 100)
                    for _ in loop(min(len(delM), 100)):
                        delM.popleft()
                else:
                    await _vars.silentDelete(delM[0], exc=True)
                    deleted += 1
                    delM.popleft()
            except:
                print(traceback.format_exc())
                for _ in loop(min(5, len(delM))):
                    m = delM.popleft()
                    await _vars.silentDelete(m, exc=True)
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
    usage = "<0:user> <1:time[]> <2:reason[]> <hide(?h)> <verbose(?v)>"
    flags = "hvf"

    async def __call__(self, _vars, args, user, channel, guild, flags, perm, name, **void):
        update = self._vars.database.bans.update
        dtime = datetime.datetime.utcnow().timestamp()
        if args:
            check = args[0].lower()
        else:
            check = ""
        if not args or "everyone" in check or "here" in check:
            t_user = None
            t_perm = inf
        else:
            u_id = verifyID(args[0])
            try:
                t_user = await _vars.fetch_user(u_id)
            except (TypeError, discord.NotFound):
                try:
                    t_user = await _vars.fetch_member(u_id, guild)
                except LookupError:
                    t_user = await _vars.fetch_whuser(u_id, guild)
            t_perm = _vars.getPerms(t_user, guild)
        if t_perm + 1 > perm or isnan(t_perm):
            if len(args) > 1:
                reason = (
                    "to ban " + t_user.name
                    + " from " + guild.name
                )
                self.permError(perm, t_perm + 1, reason)
        g_bans = await getBans(_vars, guild)
        msg = None
        if name.lower() == "unban":
            tm = -1
            args = ["", ""]
        elif len(args) < 2:
            if t_user is None:
                if not g_bans:
                    return (
                        "```css\nNo currently banned users for ["
                        + noHighlight(guild.name) + "].```"
                    )
                output = ""
                for u_id in g_bans:
                    try:
                        user = await _vars.fetch_user(u_id)
                        output += (
                            "[" + str(user) + "] "
                            + noHighlight(sec2Time(g_bans[u_id]["unban"] - dtime))
                        )
                        if "v" in flags:
                            output += " .ID: " + str(user.id)
                        output += " .Reason: " + noHighlight(g_bans[u_id]["reason"]) + "\n"
                    except:
                        print(traceback.format_exc())
                return (
                    "Currently banned users from **" 
                    + escape_markdown(guild.name) + "**:\n```css\n"
                    + output.strip("\n") + "```"
                )
            tm = 0
        else:
            if t_user is None:
                orig = 0
            else:
                orig = g_bans.get(t_user.id, 0)
            bantype = " ".join(args[1:])
            if "for " in bantype:
                i = bantype.index("for ")
                expr = bantype[:i].strip()
                msg = bantype[i + 3:].strip()
            else:
                expr = bantype
            _op = None
            for operator in ("+=", "-=", "*=", "/=", "%="):
                if expr.startswith(operator):
                    expr = expr[2:].strip()
                    _op = operator[0]
            num = await _vars.evalTime(expr, guild)
            if _op is not None:
                num = eval(str(orig) + _op + str(num), {}, infinum)
            tm = num
        await channel.trigger_typing()
        if t_user is None:
            if "f" not in flags:
                response = uniStr(
                    "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                    + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                )
                
                return ("```asciidoc\n[" + response + "]```")
            if tm >= 0:
                raise PermissionError(
                    "Banning every user in a server is no longer allowed "
                    + "in order to prevent misuse and/or security issues."
                )
                # it = guild.fetch_members(limit=None)
                # users = await it.flatten()
            else:
                users = []
                create_task(channel.send(
                    "```css\nUnbanning all users from ["
                    + noHighlight(guild.name) + "]...```"
                ))
            for u_id in g_bans:
                users.append(await _vars.fetch_user(u_id))
            is_banned = None
        else:
            users = [t_user]
            is_banned = g_bans.get(t_user.id, None)
            if is_banned is not None:
                is_banned = is_banned["unban"] - dtime
                if len(args) < 2:
                    return (
                        "```css\nCurrent ban for [" + noHighlight(t_user.name)
                        + "] from [" + noHighlight(guild.name) + "]: ["
                        + noHighlight(sec2Time(is_banned)) + "].```"
                    )
            elif len(args) < 2:
                return (
                    "```css\n[" + noHighlight(t_user.name)
                    + "] is currently not banned from [" + noHighlight(guild.name) + "].```"
                )
        response = "```css"
        for t_user in users:
            if tm >= 0:
                try:
                    if hasattr(t_user, "webhook"):
                        coro = t_user.webhook.delete()
                    else:
                        coro = guild.ban(t_user, reason=msg, delete_message_days=0)
                    if len(users) > 3:
                        create_task(coro)
                    else:
                        await coro
                    await asyncio.sleep(0.3)
                except Exception as ex:
                    response += "\nError: " + repr(ex)
                    continue
            if not hasattr(t_user, "webhook"):
                g_bans[t_user.id] = {
                    "unban": tm + dtime,
                    "reason": msg,
                    "channel": channel.id,
                }
                update()
            else:
                tm = inf
            if is_banned:
                response += (
                    "\nUpdated ban for [" + noHighlight(t_user.name)
                    + "] from [" + noHighlight(sec2Time(is_banned))
                    + "] to [" + noHighlight(sec2Time(tm)) + "]."
                )
            elif tm >= 0:
                response += (
                    "\n[" + noHighlight(t_user.name)
                    + "] has been banned from [" + noHighlight(guild.name)
                    + "] for [" + noHighlight(sec2Time(tm)) + "]."
                )
            if msg is not None and tm >= 0:
                response += " Reason: [" + noHighlight(msg) + "]."
        if len(response) > 6 and "h" not in flags:
            return response + "```"


class RoleGiver(Command):
    server_only = True
    name = ["Verifier"]
    min_level = 3
    min_display = "3+"
    description = "Adds an automated role giver to the current channel."
    usage = "<0:react_to[]> <1:role[]> <delete_messages(?x)> <disable(?d)>"
    flags = "aedx"

    async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
        update = self._vars.database.rolegivers.update
        _vars = self._vars
        data = _vars.data.rolegivers
        if "d" in flags:
            if argv:
                react = args[0].lower()
                assigned = data.get(channel.id, {})
                if react not in assigned:
                    raise LookupError("Rolegiver " + react + " not currently assigned for " + channel.name + ".")
                assigned.pop(react)
                return "```css\nRemoved [" + react + "] from the rolegiver list for [#" + noHighlight(channel.name) + "].```"
            if channel.id in data:
                data.pop(channel.id)
                update()
            return "```css\nRemoved all automated rolegivers from [#" + noHighlight(channel.name) + "].```"
        assigned = data.setdefault(channel.id, {})
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
        if sum(len(alist[0]) for alist in assigned) >= 16:
            raise OverflowError(
                "Rolegiver list for " + channel.name
                + " has reached the maximum of 16 items. "
                + "Please remove an item to add another."
            )
        react = args[0].lower()
        if len(react) > 64:
            raise OverflowError("Search substring too long.")
        r = verifyID(" ".join(args[1:]))
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
        if not inf > perm:
            memb = guild.get_member(user.id)
            if memb is None:
                memb = await guild.fetch_member(user.id)
            if memb.top_role <= role:
                raise PermissionError("Target role is higher than your highest role.")
        alist = assigned.setdefault(react, [[], False])
        alist[1] |= "x" in flags
        alist[0].append(role.id) 
        update()
        return (
            "```css\nAdded [" + noHighlight(react)
            + "] ➡️ [" + noHighlight(role)
            + "] to channel [" + noHighlight(channel.name) + "].```"
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

    async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
        update = self._vars.database.autoroles.update
        _vars = self._vars
        data = _vars.data.autoroles
        if "d" in flags:
            assigned = data.get(guild.id, None)
            if argv and assigned:
                i = await _vars.evalMath(argv, guild)
                roles = assigned.pop(i)
                removed = []
                for r in roles:
                    try:
                        role = await _vars.fetch_role(r, guild)
                    except:
                        print(traceback.format_exc())
                        continue
                    removed.append(role)
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
                data.pop(channel.id)
                update()
            return "```css\nRemoved all items from the autorole list for " + sbHighlight(guild) + ".```"
        assigned = data.setdefault(guild.id, hlist())
        if not argv:
            rlist = hlist()
            for roles in assigned:
                new = hlist()
                for r in roles:
                    role = await _vars.fetch_role(r, guild)
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
        rolenames = args
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
            if not inf > perm:
                memb = guild.get_member(user.id)
                if memb is None:
                    memb = await guild.fetch_member(user.id)
                if memb.top_role <= role:
                    raise PermissionError("Target role is higher than your highest role.")
            roles.append(role)
        assigned.append(hlist(role.id for role in roles))
        update()
        if "x" in flags:
            i = 1
            for member in guild.members:
                role = random.choice(roles)
                if role not in member.roles:
                    create_task(member.add_roles(role, reason="InstaRole", atomic=True))
                    if not i % 5:
                        await asyncio.sleep(5)
                    i += 1
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
        update = self._vars.database.rolepreservers.update
        _vars = self._vars
        following = _vars.data.rolepreservers
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
        u_id = self._vars.client.user.id
        for role in guild.roles:
            if len(role.members) != 1 or role.members[-1].id != u_id:
                create_task(self.roleLock(role, channel))
        for inv in guild.invites:
            create_task(self.invLock(inv, channel))
        response = uniStr(
            "LOCKDOWN REQUESTED."
        )
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
                num = await self._vars.evalMath(" ".join(args[1:]), guild)
                if not num <= 65536:
                    raise OverflowError("Maximum number of messages allowed is 65536.")
                if num <= 0:
                    raise ValueError("Please input a valid message limit.")
            ch = await self._vars.fetch_channel(verifyID(args[0]))
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

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data.logU
        update = _vars.database.logU.update
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
            channel = await _vars.fetch_channel(c_id)
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

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data.logM
        update = _vars.database.logM.update
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
            channel = await _vars.fetch_channel(c_id)
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

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data.logF
        update = _vars.database.logF.update
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
            channel = await _vars.fetch_channel(c_id)
            return (
                "```css\nFile logging for [" + noHighlight(guild.name)
                + "] is currently enabled in [" + noHighlight(channel.name)
                + "].```"
            )
        return (
            "```css\nFile logging is currently disabled in ["
            + noHighlight(guild.name) + "].```"
        )


class ServerProtector(Database):
    name = "prot"
    no_file = True

    async def kickWarn(self, u_id, guild, owner, msg):
        user = await self._vars.fetch_user(u_id)
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
        user = self._vars.client.user
        owner = guild.owner
        if owner.id == user.id:
            owner = await self._vars.fetch_user(self._vars.owner_id)
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
        audits = await guild.audit_logs(limit=5, action=discord.AuditLogAction.channel_delete).flatten()
        ts = datetime.datetime.utcnow().timestamp()
        cnt = {}
        for log in audits:
            if ts - log.created_at.timestamp() < 120:
                addDict(cnt, {log.user.id: 1})
        for u_id in cnt:
            if cnt[u_id] > 2:
                create_task(self.targetWarn(u_id, guild, "channel deletions `(" + str(cnt[u_id]) + ")`"))

    async def _ban_(self, user, guild, **void):
        audits = await guild.audit_logs(limit=11, action=discord.AuditLogAction.ban).flatten()
        ts = datetime.datetime.utcnow().timestamp()
        cnt = {}
        for log in audits:
            if ts - log.created_at.timestamp() < 10:
                addDict(cnt, {log.user.id: 1})
        for u_id in cnt:
            if cnt[u_id] > 5:
                create_task(self.targetWarn(u_id, guild, "banning `(" + str(cnt[u_id]) + ")`"))


class UpdateUserLogs(Database):
    name = "logU"

    async def _user_update_(self, before, after, **void):
        for guild in self._vars.client.guilds:
            create_task(self._member_update_(before, after, guild))

    async def _member_update_(self, before, after, guild=None):
        if guild is None:
            guild = after.guild
        elif guild.get_member(after.id) is None:
            try:
                memb = await guild.fetch_member(after.id)
                if memb is None:
                    raise EOFError
            except:
                return
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self._vars.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            emb = discord.Embed()
            b_url = strURL(before.avatar_url)
            a_url = strURL(after.avatar_url)
            emb.set_author(name=str(after), icon_url=a_url, url=a_url)
            emb.description = (
                "<@" + str(after.id)
                + "> has been updated:"
            )
            colour = [0] * 3
            change = False
            if str(before) != str(after):
                emb.add_field(
                    name="Username",
                    value=escape_markdown(str(before)) + " <:arrow:688320024586223620> " + escape_markdown(str(after)),
                )
                change = True
                colour[0] += 255
            if hasattr(before, "guild"):
                if before.display_name != after.display_name:
                    emb.add_field(
                        name="Nickname",
                        value=escape_markdown(before.display_name) + " <:arrow:688320024586223620> " + escape_markdown(after.display_name),
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
                        rchange = "<:minus:688316020359823364> " + escape_markdown(", ".join(str(r) for r in sub))
                    if add:
                        rchange += (
                            "\n" * bool(rchange) + "<:plus:688316007093370910> " 
                            + escape_markdown(", ".join(str(r) for r in add))
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
                        + ") <:arrow:688320024586223620> [After](" 
                        + str(a_url) + ")"
                    ),
                )
                emb.set_thumbnail(url=a_url)
                change = True
                colour[2] += 255
            if change:
                emb.colour = colour2Raw(colour)
                await channel.send(embed=emb)

    async def _join_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self._vars.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            emb = discord.Embed(colour=16777214)
            url = strURL(user.avatar_url)
            emb.set_author(name=str(user), icon_url=url, url=url)
            emb.description = (
                "<@" + str(user.id)
                + "> has joined the server."
            )
            await channel.send(embed=emb)
    
    async def _leave_(self, user, **void):
        guild = getattr(user, "guild", None)
        if guild is not None and guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self._vars.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            emb = discord.Embed(colour=1)
            url = strURL(user.avatar_url)
            emb.set_author(name=str(user), icon_url=url, url=url)
            kick = None
            ban = None
            try:
                ts = datetime.datetime.utcnow().timestamp()
                bans = await guild.audit_logs(limit=4, action=discord.AuditLogAction.ban).flatten()
                kicks = await guild.audit_logs(limit=4, action=discord.AuditLogAction.kick).flatten()
                for log in bans:
                    if ts - log.created_at.timestamp() < 3:
                        if log.target.id == user.id:
                            ban = freeClass(id=log.user.id, reason=log.reason)
                            raise StopIteration
                for log in kicks:
                    if ts - log.created_at.timestamp() < 3:
                        if log.target.id == user.id:
                            kick = freeClass(id=log.user.id, reason=log.reason)
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
            await channel.send(embed=emb)


class UpdateMessageLogs(Database):
    name = "logM"

    def __init__(self, *args):
        self.searched = False
        self.dc = {}
        super().__init__(*args)

    async def cacheGuild(self, guild, lim=65536):

        async def getChannelHistory(channel, returns, lim=2048):
            try:
                messages = []
                for i in range(16):
                    history = channel.history(limit=lim)
                    try:
                        messages = await history.flatten()
                        break
                    except discord.Forbidden:
                        raise
                    except:
                        print(traceback.format_exc())
                    await asyncio.sleep(20 * (i ** 2 + 1))
                returns[0] = messages
            except:
                print(channel.name)
                print(traceback.format_exc())
                returns[0] = []

        print(guild, "loading...")
        limit = ceil(lim / len(guild.text_channels))
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
                lim=limit,
            ))
            i += 1
        while [None] in histories:
            await asyncio.sleep(2)
        while [] in histories:
            histories = histories.remove([])
        i = 1
        for h in histories:
            temp = h[0]
            # print("[" + str(len(temp)) + "]")
            for message in temp:
                self._vars.cacheMessage(message)
                if not i & 8191:
                    await asyncio.sleep(0.5)
                i += 1
        print(guild, "finished.")

    async def __call__(self):
        for h in tuple(self.dc):
            if datetime.datetime.utcnow() - h > datetime.timedelta(seconds=3600):
                self.dc.pop(h)
        if not self.searched:
            self.searched = True
            lim = floor(1048576 / len(self._vars.client.guilds))
            for g in self._vars.client.guilds:
                create_task(self.cacheGuild(g, lim=lim))

    async def _edit_(self, before, after, **void):
        if not after.author.bot:
            guild = before.guild
            if guild.id in self.data:
                c_id = self.data[guild.id]
                try:
                    channel = await self._vars.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    self.update()
                    return
                u = before.author
                name = u.name
                name_id = name + bool(u.display_name) * ("#" + u.discriminator)
                url = strURL(u.avatar_url)
                emb = discord.Embed(colour=colour2Raw([0, 0, 255]))
                emb.set_author(name=name_id, icon_url=url, url=url)
                emb.description = (
                    "**Message edited in** <#"
                    + str(before.channel.id) + ">:"
                )
                emb.add_field(name="Before", value=strMessage(before))
                emb.add_field(name="After", value=strMessage(after))
                await channel.send(embed=emb)

    def logDeleted(self, message):
        if message.author.bot and message.author.id != self._vars.client.user.id:
            return
        if self._vars.isDeleted(message) < 2:
            s = strMessage(message, username=True)
            print(s, file="deleted.txt")

    async def _delete_(self, message, bulk=False, **void):
        cu_id = self._vars.client.user.id
        if bulk:
            self.logDeleted(message)
            return
        guild = message.guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self._vars.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            now = datetime.datetime.utcnow()
            u = message.author
            name = u.name
            name_id = name + bool(u.display_name) * ("#" + u.discriminator)
            url = strURL(u.avatar_url)
            action = (
                discord.AuditLogAction.message_delete,
                discord.AuditLogAction.message_bulk_delete,
            )[bulk]
            try:
                t = u
                init = "<@" + str(t.id) + ">"
                if self._vars.isDeleted(message):
                    t = self._vars.client.user
                else:
                    al = await guild.audit_logs(
                        limit=5,
                        action=action,
                    ).flatten()
                    for e in reversed(al):
                        # print(e, e.target, now - e.created_at)
                        try:
                            cnt = e.extra.count - 1
                        except AttributeError:
                            cnt = int(e.extra.get("count", 1)) - 1
                        h = e.created_at
                        cs = self.dc.setdefault(h, 0)
                        c = cnt - cs
                        if c > 0:
                            self.dc[h] += 1
                        s = (5, 3600)[c > 0]
                        if not bulk:
                            cid = e.extra.channel.id
                            targ = e.target.id
                        else:
                            try:
                                cid = e.target.id
                            except AttributeError:
                                cid = e._target_id
                            targ = u.id
                        if now - h < datetime.timedelta(seconds=s):
                            if targ == u.id and cid == message.channel.id:
                                t = e.user
                                init = "<@" + str(t.id) + ">"
                                # print(t, e.target)
                if t.bot or u.id == t.id == cu_id:
                    self.logDeleted(message)
                    return
            except (discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=colour2Raw([255, 0, 0]))
            emb.set_author(name=name_id, icon_url=url, url=url)
            emb.description = (
                init + " **deleted message from** <#"
                + str(message.channel.id) + ">:\n"
            )
            emb.description += strMessage(message, limit=2048 - len(emb.description))
            await channel.send(embed=emb)


class UpdateFileLogs(Database):
    name = "logF"

    # async def _user_update_(self, before, after, **void):
    #     sending = {}
    #     for guild in self._vars.client.guilds:
    #         if guild.get_member(after.id) is None:
    #             try:
    #                 memb = await guild.fetch_member(after.id)
    #                 if memb is None:
    #                     raise EOFError
    #             except:
    #                 continue
    #         sending[guild.id] = True
    #     if not sending:
    #         return
    #     b_url = strURL(before.avatar_url)
    #     a_url = strURL(after.avatar_url)
    #     if b_url != a_url:
    #         try:
    #             obj = before.avatar_url_as(format="gif", static_format="png", size=4096)
    #         except discord.InvalidArgument:
    #             obj = before.avatar_url_as(format="png", static_format="png", size=4096)
    #         if ".gif" in str(obj):
    #             fmt = ".gif"
    #         else:
    #             fmt = ".png"
    #         msg = None
    #         try:
    #             b = await obj.read()
    #             fil = discord.File(io.BytesIO(b), filename=str(before.id) + fmt)
    #         except:
    #             msg = str(obj)
    #             fil=None
    #         emb = discord.Embed(colour=randColour())
    #         emb.description = "File deleted from <@" + str(before.id) + ">"
    #         for g_id in sending:
    #             guild = self._vars.cache["guilds"].get(g_id, None)
    #             create_task(self.send_avatars(msg, fil, emb, guild))

    # async def send_avatars(self, msg, fil, emb, guild=None):
    #     if guild is None:
    #         return
    #     if guild.id in self.data:
    #         c_id = self.data[guild.id]
    #         try:
    #             channel = await self._vars.fetch_channel(c_id)
    #         except (EOFError, discord.NotFound):
    #             self.data.pop(guild.id)
    #             self.update()
    #             return
    #         await channel.send(msg, embed=emb, file=fil)

    async def _delete_(self, message, bulk=False, **void):
        guild = message.guild
        if guild.id in self.data:
            c_id = self.data[guild.id]
            if message.attachments:
                try:
                    channel = await self._vars.fetch_channel(c_id)
                except (EOFError, discord.NotFound):
                    self.data.pop(guild.id)
                    self.update()
                    return
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
                        msg += a.url + "\n"
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
        _vars = self._vars
        assigned = self.data.get(message.channel.id, ())
        for k in assigned:
            if ((k in text) if hasSymbol(k) else (k in message.content.lower())):
                alist = assigned[k]
                for r in alist[0]:
                    role = guild.get_role(r)
                    if role is None:
                        roles = await guild.fetch_roles()
                        for i in roles:
                            if i.id == r:
                                role = i
                    if role is None:
                        alist[0].remove(r)
                        continue
                    await user.add_roles(
                        role,
                        reason="Keyword \"" + k + "\" found in message \"" + message.content + "\".",
                        atomic=True,
                    )
                    print("Granted role " + str(role) + " to " + str(user) + ".")
                if alist[1]:
                    await _vars.silentDelete(message)


class UpdateAutoRoles(Database):
    name = "autoroles"

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            roles = []
            assigned = self.data[guild.id]
            for rolelist in assigned:
                try:
                    role = await self._vars.fetch_role(random.choice(rolelist), guild)
                    roles.append(role)
                except:
                    print(traceback.format_exc())
            print(roles)
            await user.add_roles(*roles, reason="AutoRole", atomic=False)


class UpdateRolePreservers(Database):
    name = "rolepreservers"

    async def _join_(self, user, guild, **void):
        if guild.id in self.data:
            if user.id in self.data[guild.id]:
                roles = []
                assigned = self.data[guild.id][user.id]
                for r_id in assigned:
                    try:
                        role = await self._vars.fetch_role(r_id, guild)
                        roles.append(role)
                    except:
                        print(traceback.format_exc())
                print(user, roles)
                await user.edit(roles=roles, reason="RolePreserver")
                self.data[guild.id].pop(user.id)

    async def _leave_(self, user, guild, **void):
        if guild.id in self.data:
            roles = user.roles[1:]
            assigned = [role.id for role in roles]
            print(user, assigned)
            self.data[guild.id][user.id] = assigned
            self.update()


class UpdatePerms(Database):
    name = "perms"


async def getBans(_vars, guild):
    bans = _vars.data["bans"].setdefault(guild.id, {})
    try:
        banlist = await guild.bans()
    except discord.Forbidden:
        print(traceback.format_exc())
        print("Unable to retrieve ban list for " + guild.name + ".")
        return []
    for ban in banlist:
        if ban.user.id not in bans:
            bans[ban.user.id] = {
                "unban": inf,
                "reason": ban.reason,
                "channel": None,
            }
    return bans


class UpdateBans(Database):
    name = "bans"

    def __init__(self, *args):
        self.synced = False
        super().__init__(*args)

    async def __call__(self, **void):
        while self.busy:
            await asyncio.sleep(0.5)
        self.busy = True
        try:
            _vars = self._vars
            dtime = datetime.datetime.utcnow().timestamp()
            bans = self.data
            changed = False
            if not self.synced:
                self.synced = True
                for guild in _vars.client.guilds:
                    create_task(getBans(_vars, guild))
                changed = True
            for g in list(bans):
                for b in list(bans[g]):
                    utime = bans[g][b]["unban"]
                    if dtime >= utime:
                        try:
                            u_target = await _vars.fetch_user(b)
                            g_target = await _vars.fetch_guild(g)
                            c_id = bans[g][b]["channel"]
                            if c_id is not None:
                                c_target = await _vars.fetch_channel(c_id)
                            try:
                                await g_target.unban(u_target)
                                if c_id is not None:
                                    await c_target.send(
                                        "```css\n[" + noHighlight(u_target.name)
                                        + "] has been unbanned from [" + noHighlight(g_target.name) + "].```"
                                    )
                            except:
                                if c_id is not None:
                                    await c_target.send(
                                        "```css\nUnable to unban [" + noHighlight(u_target.name)
                                        + "] from [" + noHighlight(g_target.name) + "].```"
                                    )
                                print(traceback.format_exc())
                            bans[g].pop(b)
                            changed = True
                        except:
                            print(traceback.format_exc())
                if not len(bans[g]):
                    bans.pop(g)
            if changed and len(bans):
                self.update()
        except:
            print(traceback.format_exc())
        self.busy = False
