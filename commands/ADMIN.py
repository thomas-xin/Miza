import discord
try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from smath import *


class Purge:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["Del", "Delete"]
        self.min_level = 3
        self.description = "Deletes a number of messages from a certain user in current channel."
        self.usage = "<1:user{bot}(?a)> <0:count[1]> <hide(?h)>"
        self.flags = "ah"

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
                u_id = _vars.verifyID(a1)
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


class Ban:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Bans", "Unban"]
        self.min_level = 3
        self.min_display = "3+"
        self.description = "Bans a user for a certain amount of time, with an optional reason."
        self.usage = "<0:user> <1:time[]> <2:reason[]> <hide(?h)> <verbose(?v)>"
        self.flags = "hvf"

    async def __call__(self, _vars, args, user, channel, guild, flags, perm, name, **void):
        update = self.data["bans"].update
        dtime = datetime.datetime.utcnow().timestamp()
        if args:
            check = args[0].lower()
        else:
            check = ""
        if not args or "everyone" in check or "here" in check:
            t_user = None
            t_perm = inf
        else:
            u_id = _vars.verifyID(args[0])
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
                    + discord.utils.escape_markdown(guild.name) + "**:\n```css\n"
                    + output.strip("\n") + "```"
                )
            tm = 0
        else:
            if t_user is None:
                orig = 0
            else:
                orig = g_bans.get(t_user.id, 0)
            if len(args) >= 3:
                expr = " ".join(args[1:])
            else:
                expr = " ".join(args[1:])
            _op = None
            for operator in ("+=", "-=", "*=", "/=", "%="):
                if expr.startswith(operator):
                    expr = expr[2:].strip(" ")
                    _op = operator[0]
            num = await _vars.evalTime(expr, guild)
            if _op is not None:
                num = eval(str(orig) + _op + str(num), {}, infinum)
            tm = num
        await channel.trigger_typing()
        if len(args) >= 3:
            msg = args[-1]
        else:
            msg = None
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


class RoleGiver:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Verifier"]
        self.min_level = 3
        self.min_display = "3+"
        self.description = "Adds an automated role giver to the current channel."
        self.usage = "<0:react_to[]> <1:role[]> <delete_messages(?x)> <disable(?d)>"
        self.flags = "aedx"

    async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
        update = self.data["rolegivers"].update
        _vars = self._vars
        data = _vars.data["rolegivers"]
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
            return "```css\nRemoved all automated role givers from [#" + noHighlight(channel.name) + "].```"
        assigned = data.setdefault(channel.id, {})
        if not argv:
            key = lambda alist: "⟨" + ", ".join([str(r) for r in alist[0]]) + "⟩, delete: " + str(alist[1])
            return (
                "Currently active permission givers in <#" + str(channel.id)
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
        roles = hlist()
        r = _vars.verifyID(" ".join(args[1:]))
        if len(guild.roles) <= 1:
            rolelist = await guild.fetch_roles()
        else:
            rolelist = guild.roles
        if type(r) is int:
            for i in rolelist:
                if i.id == r:
                    role = i
                    roles.append(role)
        else:
            r = r.replace(" ", "").lower()
            for i in rolelist:
                if reconstitute(i).replace(" ", "").lower() == r:
                    role = i
                    roles.append(role)
        alist = assigned.setdefault(react, [hlist(), False])
        alist[1] |= "x" in flags
        alist[0].append(role.id) 
        update()
        return (
            "```css\nAdded [" + noHighlight(react)
            + "] ➡️ [" + noHighlight(role)
            + "] to channel [" + noHighlight(channel.name) + "].```"
        )


class RainbowRole:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["DynamicRole"]
        self.min_level = 3
        self.description = "Causes target role to randomly change colour."
        self.usage = "<0:role[]> <mim_delay[16]> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, _vars, flags, args, argv, guild, **void):
        update = self.data["rolecolours"].update
        _vars = self._vars
        colours = _vars.data["rolecolours"]
        guild_special = colours.setdefault(guild.id, {})
        if not argv:
            if "d" in flags:
                colours.pop(guild.id)
                update()
                return (
                    "```css\nRemoved all active dynamic role colours in ["
                    + noHighlight(guild.name) + "].```"
                )
            return (
                "Currently active dynamic role colours in **" + guild.name
                + "**:\n```ini\n" + strIter(guild_special) + "```"
            )
        if len(curr["guild_special"]) >= 7:
            raise OverflowError(
                "Rainbow role list for " + guild.name
                + " has reached the maximum of 7 items. "
                + "Please remove an item to add another."
            )
        role = args[0].lower()
        if len(args) < 2:
            delay = 16
        else:
            delay = await _vars.evalMath(" ".join(args[1:]), guild.id)
        for r in guild.roles:
            if role in r.name.lower():
                if "d" in flags:
                    try:
                        guild_special.pop(r.id)
                    except KeyError:
                        pass
                else:
                    guild_special[r.id] = delay
        update()
        return (
            "Changed dynamic role colours for **" + guild.name
            + "** to:\n```ini\n" + strIter(guild_special) + "```"
        )


class Lockdown:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = inf
        self.description = "Completely locks down the server by removing send message permissions for all users and revoking all invites."
        self.usage = ""
        self.flags = "f"

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


class SaveChannel:
    is_command = True
    time_consuming = 1
    _timeout_ = 10

    def __init__(self):
        self.name = ["BackupChannel", "DownloadChannel"]
        self.min_level = 3
        self.description = "Saves a number of messages in a channel, as well as their contents, to a .txt file."
        self.usage = "<0:channel{current}> <1:message_limit[4096]>"

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
            ch = await self._vars.fetch_channel(self._vars.verifyID(args[0]))
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
            s += "\n\n".join([self._vars.strMessage(m, limit=2048, username=True) for m in h[:4096]])
            h = h[4096:]
            await asyncio.sleep(0.32)
        return bytes(s, "utf-8")
        

follow_default = {
    "follow": False,
    "reacts": {},
}

                  
class Dogpile:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 2
        self.description = "Causes Miza to automatically imitate users when 3+ of the same messages are posted in a row."
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, flags, guild, **void):
        update = self.data["follows"].update
        _vars = self._vars
        following = _vars.data["follows"]
        curr = following.setdefault(guild.id, copy.deepcopy(follow_default))
        if type(curr) is not dict:
            curr = copy.deepcopy(follow_default)
        if "d" in flags:
            curr["follow"] = False
            update()
            return "```css\nDisabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        elif "e" in flags or "a" in flags:
            curr["follow"] = True
            update()
            return "```css\nEnabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        else:
            return (
                "```css\nCurrently " + "not " * (not curr["follow"])
                + "dogpile imitating in [" + noHighlight(guild.name) + "].```"
            )


class React:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["AutoReact"]
        self.min_level = 2
        self.description = "Causes Miza to automatically assign a reaction to messages containing the substring."
        self.usage = "<0:react_to[]> <1:react_data[]> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, _vars, flags, guild, argv, args, **void):
        update = self.data["follows"].update
        _vars = self._vars
        following = _vars.data["follows"]
        curr = following.setdefault(guild.id, copy.deepcopy(follow_default))
        if type(curr) is not dict:
            curr = copy.deepcopy(follow_default)
        if not argv:
            if "d" in flags:
                curr["reacts"] = {}
                update()
                return "```css\nRemoved all auto reacts for [" + noHighlight(guild.name) + "].```"
            else:
                return (
                    "Currently active auto reacts for **" + discord.utils.escape_markdown(guild.name)
                    + "**:\n```ini\n" + strIter(curr.get("reacts", {})) + "```"
                )
        a = args[0].lower()[:64]
        if "d" in flags:
            if a in curr["reacts"]:
                curr["reacts"].pop(a)
                update()
                return (
                    "```css\nRemoved [" + noHighlight(a) + "] from the auto react list for ["
                    + noHighlight(guild.name) + "].```"
                )
            else:
                raise LookupError(str(a) + " is not in the auto react list.")
        if len(curr["reacts"]) >= 256:
            raise OverflowError(
                "React list for " + guild.name
                + " has reached the maximum of 256 items. "
                + "Please remove an item to add another."
            )
        curr["reacts"][a] = args[1]
        update()
        return (
            "```css\nAdded [" + noHighlight(a) + "] ➡️ [" + noHighlight(args[1]) + "] to the auto react list for ["
            + noHighlight(guild.name) + "].```"
        )


class UserLog:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Causes Miza to log user events from the server, in the current channel."
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data["logU"]
        update = _vars.database["logU"].update
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


class MessageLog:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Causes Miza to log message events from the server, in the current channel."
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data["logM"]
        update = _vars.database["logM"].update
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


class FileLog:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Causes Miza to log deleted files from the server, in the current channel."
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "aed"

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data["logF"]
        update = _vars.database["logF"].update
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


class ServerProtector:
    is_database = True
    name = "prot"
    no_file = True

    def __init__(self):
        pass

    async def __call__(self):
        pass

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


class updateUserLogs:
    is_database = True
    name = "logU"

    def __init__(self):
        pass

    async def __call__(self):
        pass

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
            b_url = self._vars.strURL(before.avatar_url)
            a_url = self._vars.strURL(after.avatar_url)
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
                    value=discord.utils.escape_markdown(str(before)) + " <:arrow:688320024586223620> " + discord.utils.escape_markdown(str(after)),
                )
                change = True
                colour[0] += 255
            if hasattr(before, "guild"):
                if before.display_name != after.display_name:
                    emb.add_field(
                        name="Nickname",
                        value=discord.utils.escape_markdown(before.display_name) + " <:arrow:688320024586223620> " + discord.utils.escape_markdown(after.display_name),
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
                        rchange = "<:minus:688316020359823364> " + discord.utils.escape_markdown(", ".join(str(r) for r in sub))
                    if add:
                        rchange += (
                            "\n" * bool(rchange) + "<:plus:688316007093370910> " 
                            + discord.utils.escape_markdown(", ".join(str(r) for r in add))
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
            url = self._vars.strURL(user.avatar_url)
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
            url = self._vars.strURL(user.avatar_url)
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


class updateMessageLogs:
    is_database = True
    name = "logM"

    def __init__(self):
        self.dc = {}

    async def __call__(self):
        for h in tuple(self.dc):
            if datetime.datetime.utcnow() - h > datetime.timedelta(seconds=3600):
                self.dc.pop(h)

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
                url = self._vars.strURL(u.avatar_url)
                emb = discord.Embed(colour=colour2Raw([0, 0, 255]))
                emb.set_author(name=name_id, icon_url=url, url=url)
                emb.description = (
                    "**Message edited in** <#"
                    + str(before.channel.id) + ">:"
                )
                emb.add_field(name="Before", value=self._vars.strMessage(before))
                emb.add_field(name="After", value=self._vars.strMessage(after))
                await channel.send(embed=emb)

    async def _delete_(self, message, bulk=False, **void):
        if bulk:
            if self._vars.isDeleted(message) < 2:
                print(self._vars.strMessage(message, username=True))
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
            url = self._vars.strURL(u.avatar_url)
            action = (
                discord.AuditLogAction.message_delete,
                discord.AuditLogAction.message_bulk_delete,
            )[bulk]
            try:
                cu_id = self._vars.client.user.id
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
                    if self._vars.isDeleted(message) < 2:
                        print(self._vars.strMessage(message, username=True))
                    return
            except (discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            emb = discord.Embed(colour=colour2Raw([255, 0, 0]))
            emb.set_author(name=name_id, icon_url=url, url=url)
            emb.description = (
                init + " **deleted message from** <#"
                + str(message.channel.id) + ">:\n"
            )
            emb.description += self._vars.strMessage(message, limit=2048 - len(emb.description))
            await channel.send(embed=emb)


class updateFileLogs:
    is_database = True
    name = "logF"

    def __init__(self):
        pass

    async def __call__(self):
        pass

    async def _user_update_(self, before, after, **void):
        return
        sending = {}
        for guild in self._vars.client.guilds:
            if guild.get_member(after.id) is None:
                try:
                    memb = await guild.fetch_member(after.id)
                    if memb is None:
                        raise EOFError
                except:
                    continue
            sending[guild.id] = True
        if not sending:
            return
        b_url = self._vars.strURL(before.avatar_url)
        a_url = self._vars.strURL(after.avatar_url)
        if b_url != a_url:
            try:
                obj = before.avatar_url_as(format="gif", static_format="png", size=4096)
            except discord.InvalidArgument:
                obj = before.avatar_url_as(format="png", static_format="png", size=4096)
            if ".gif" in str(obj):
                fmt = ".gif"
            else:
                fmt = ".png"
            msg = None
            try:
                b = await obj.read()
                fil = discord.File(io.BytesIO(b), filename=str(before.id) + fmt)
            except:
                msg = str(obj)
                fil=None
            emb = discord.Embed(colour=self._vars.randColour())
            emb.description = "File deleted from <@" + str(before.id) + ">"
            for g_id in sending:
                guild = self._vars.cache["guilds"].get(g_id, None)
                create_task(self.send_avatars(msg, fil, emb, guild))

    async def send_avatars(self, msg, fil, emb, guild=None):
        if guild is None:
            return
        if guild.id in self.data:
            c_id = self.data[guild.id]
            try:
                channel = await self._vars.fetch_channel(c_id)
            except (EOFError, discord.NotFound):
                self.data.pop(guild.id)
                self.update()
                return
            await channel.send(msg, embed=emb, file=fil)

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
                emb = discord.Embed(colour=self._vars.randColour())
                emb.description = "File" + "s" * (len(fils) + len(msg) != 1) + " deleted from <@" + str(message.author.id) + ">"
                if not msg:
                    msg = None
                await channel.send(msg, embed=emb, files=fils)


class updateFollows:
    is_database = True
    name = "follows"

    def __init__(self):
        self.msgFollow = {}

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None:
            return
        g_id = message.guild.id
        u_id = message.author.id
        following = self.data
        words = text.split(" ")
        if g_id in following:
            if not edit:
                if following[g_id]["follow"]:
                    checker = orig
                    curr = self.msgFollow.get(g_id)
                    if curr is None:
                        curr = [checker, 1, 0]
                        self.msgFollow[g_id] = curr
                    elif checker == curr[0] and u_id != curr[2]:
                        curr[1] += 1
                        if curr[1] >= 3:
                            curr[1] = xrand(-3) + 1
                            if len(checker):
                                create_task(message.channel.send(checker))
                    else:
                        if len(checker) > 100:
                            checker = ""
                        curr[0] = checker
                        curr[1] = xrand(-1, 2)
                    curr[2] = u_id
                    #print(curr)
            try:
                for k in following[g_id]["reacts"]:
                    if ((k in words) if self._vars.hasSymbol(k) else (k in message.content)):
                        await message.add_reaction(following[g_id]["reacts"][k])
            except discord.Forbidden:
                print(traceback.format_exc())

    async def __call__(self):
        pass


class updateRolegiver:
    is_database = True
    name = "rolegivers"

    def __init__(self):
        pass

    async def _nocommand_(self, text, message, **void):
        if message.guild is None:
            return
        user = message.author
        guild = message.guild
        _vars = self._vars
        assigned = self.data.get(message.channel.id, {})
        for k in assigned:
            if ((k in text) if self._vars.hasSymbol(k) else (k in message.content)):
                alist = assigned[k]
                d = alist[1]
                for r in alist[0]:
                    role = guild.get_role(r)
                    if role is None:
                        roles = await guild.fetch_roles()
                        for i in roles:
                            if i.id == r:
                                role = i
                    if role is None:
                        alist[0].remove(curr)
                        continue
                    await user.add_roles(
                        role,
                        reason="Keyword found in message.",
                        atomic=True,
                    )
                    print("Granted role " + str(role) + " to " + str(user) + ".")
                if alist[1]:
                    await _vars.silentDelete(message)

    async def __call__(self):
        pass


class updatePerms:
    is_database = True
    name = "perms"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateColours:
    is_database = True
    name = "rolecolours"

    def __init__(self):
        self.counter = 0
        self.count = 0
        self.delay = 0
        self.busy_guilds = {}

    async def changeColour(self, g_id, roles):
        if self.busy_guilds.get(g_id, 0) > 1:
            return
        addDict(self.busy_guilds, {g_id: 1})
        try:
            guild = await self._vars.fetch_guild(g_id)
            l = list(roles)
            for r in l:
                try:
                    role = guild.get_role(r)
                    delay = roles[r]
                    if not random.randint(0, ceil(delay)):
                        col = self._vars.randColour()
                        try:
                            await role.edit(colour=discord.Colour(col))
                        except KeyError:
                            self.count += 15
                            self._vars.blocked += 1
                            break
                        self.count += 1
                        #print("Edited role " + role.name)
                    await asyncio.sleep(frand(2))
                except discord.Forbidden:
                    print(traceback.format_exc())
                except discord.HTTPException:
                    print(traceback.format_exc())
                    self._vars.blocked += 60
                    break
        except:
            print(traceback.format_exc())
        addDict(self.busy_guilds, {g_id: -1})

    async def __call__(self):
        self.counter = self.counter + 1 & 65535
        if time.time() > self.delay:
            self.delay = time.time() + 60
            self.count = 0
        for g in self.data:
            if self.count < 48 and self._vars.blocked <= 0:
                create_task(self.changeColour(g, self.data[g]))
            else:
                break


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


class updateBans:
    is_database = True
    name = "bans"

    def __init__(self):
        self.synced = False

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
