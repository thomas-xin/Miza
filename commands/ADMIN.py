import datetime, discord
from smath import *


class Purge:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["Del", "Delete"]
        self.min_level = 1
        self.description = "Deletes a number of messages from a certain user in current channel."
        self.usage = "<1:user{bot}(?a)> <0:count[1]> <hide(?h)>"

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
                t_user = await _vars.fetch_user(_vars.verifyID(a1))
        if t_user != client.user:
            if perm < 3:
                raise PermissionError (
                    "Insufficient priviliges for command " + uniStr(name)
                    + " for target user.\nRequred level: " + uniStr(3)
                    + ", Current level: " + uniStr(perm) + "."
                )
        lim = count * 2 + 16
        if lim < 0:
            lim = 0
        if not isValid(lim):
            lim = None
        hist = await channel.history(limit=lim).flatten()
        delM = hlist()
        deleted = 0
        for m in hist:
            if count <= 0:
                break
            if t_user is None or m.author.id == t_user.id:
                delM.append(m)
                count -= 1
        while len(delM):
            try:
                try:
                    await channel.delete_messages(delM[:100])
                    deleted += min(len(delM), 100)
                    for i in range(min(len(delM), 100)):
                        delM.popleft()
                except AttributeError:
                    d_id = delM[0].id
                    await delM[0].delete()
                    _vars.logDelete(d_id)
                    deleted += 1
                    delM.popleft()
            except:
                print(traceback.format_exc())
                m = delM.popleft()
                await m.delete()
                _vars.logDelete(m.id)
                deleted += 1
        if not "h" in flags:
            return (
                "```css\nDeleted " + uniStr(deleted)
                + " message" + "s" * (deleted != 1) + "!```"
            )


class Ban:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Bans", "Unban"]
        self.min_level = 3
        self.description = "Bans a user for a certain amount of hours, with an optional reason."
        self.usage = "<0:user> <1:hours[]> <2:reason[]> <hide(?h)> <verbose(?v)>"

    async def __call__(self, _vars, args, user, channel, guild, flags, perm, name, **void):
        update = self.data["bans"].update
        bans = _vars.data["bans"]
        dtime = datetime.datetime.utcnow().timestamp()
        if args:
            check = args[0].lower()
        else:
            check = ""
        if not args or "everyone" in check or "here" in check:
            t_user = None
            t_perm = inf
        else:
            t_user = await _vars.fetch_user(_vars.verifyID(args[0]))
            t_perm = _vars.getPerms(t_user, guild)
        s_perm = perm
        if t_perm + 1 > s_perm or t_perm is nan:
            if len(args) > 1:
                raise PermissionError (
                    "Insufficient priviliges to ban " + uniStr(t_user.name)
                    + " from " + uniStr(guild.name)
                    + ".\nRequired level: " + uniStr(t_perm + 1)
                    + ", Current level: " + uniStr(s_perm) + "."
                )
        if name == "Unban":
            tm = -1
            args = ["", ""]
        elif len(args) < 2:
            if t_user is None:
                g_bans = await getBans(_vars, guild)
                if not g_bans:
                    return (
                        "```css\nNo currently banned users for "
                        + uniStr(guild.name) + ".```"
                    )
                output = ""
                for u_id in g_bans:
                    user = await _vars.fetch_user(u_id)
                    output += (
                        "[" + user.name + "] "
                        + uniStr(sec2Time(g_bans[u_id]["unban"] - dtime))
                    )
                    if "v" in flags:
                        output += " .ID: " + str(user.id)
                    output += " .Reason: " + str(g_bans[u_id]["reason"]) + "\n"
                return (
                    "Currently banned users from **" + guild.name + "**:\n```css\n"
                    + output.strip("\n") + "```"
                )
            tm = 0
        else:
            tm = await _vars.evalMath(args[1], guild.id)
        await channel.trigger_typing()
        if len(args) >= 3:
            msg = args[2]
        else:
            msg = None
        g_id = guild.id
        g_bans = await getBans(_vars, guild)
        if t_user is None:
            if not "f" in flags:
                response = uniStr(
                    "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                    + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                )
                return ("```asciidoc\n[" + response + "]```")
            if tm >= 0:
                it = guild.fetch_members(limit=None)
                users = await it.flatten()
            else:
                users = []
                asyncio.create_task(channel.send(
                    "```css\nUnbanning all users from "
                    + uniStr(guild.name) + "...```"
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
                        "```css\nCurrent ban for " + uniStr(t_user.name)
                        + " from " + uniStr(guild.name) + ": "
                        + uniStr(sec2Time(is_banned)) + ".```"
                    )
            elif len(args) < 2:
                return (
                    "```css\n" + uniStr(t_user.name)
                    + " is currently not banned from " + uniStr(guild.name) + ".```"
                )
        response = "```css"
        for t_user in users:
            secs = tm * 3600
            if tm >= 0:
                try:
                    if len(users) > 3:
                        asyncio.create_task(guild.ban(t_user, reason=msg, delete_message_days=0))
                    else:
                        await guild.ban(t_user, reason=msg, delete_message_days=0)
                    await asyncio.sleep(0.1)
                except Exception as ex:
                    response += "\nError: " + repr(ex)
                    continue
            g_bans[t_user.id] = {
                "unban": secs + dtime,
                "reason": msg,
                "channel": channel.id,
            }
            update()
            if is_banned:
                response += (
                    "\nUpdated ban for " + uniStr(t_user.name)
                    + " from " + uniStr(sec2Time(is_banned))
                    + " to " + uniStr(sec2Time(secs)) + "."
                )
            elif tm >= 0:
                response += (
                    "\n" + uniStr(t_user.name)
                    + " has been banned from " + uniStr(guild.name)
                    + " for " + uniStr(sec2Time(secs)) + "."
                )
            if msg is not None and tm >= 0:
                response += " Reason: " + uniStr(msg) + "."
        if len(response) > 6 and "h" not in flags:
            return response + "```"


class RoleGiver:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Verifier"]
        self.min_level = 3
        self.description = "Adds an automated role giver to the current channel."
        self.usage = "<0:react_to[]> <1:role[]> <1:perm[]> <disable(?d)> <remover(?r)>"

    async def __call__(self, argv, args, user, channel, guild, flags, **void):
        update = self.data["rolegivers"].update
        _vars = self._vars
        scheduled = _vars.data["rolegivers"]
        if "d" in flags:
            scheduled[channel.id] = {}
            update()
            return "```css\nRemoved all automated role givers from channel " + uniStr(channel.name) + ".```"
        currentSchedule = scheduled.setdefault(channel.id, {})
        if not argv:
            return (
                "Currently active permission givers in channel **" + channel.name
                + "**:\n```css\n" + repr(currentSchedule) + "```"
            )
        react = args[0].lower()
        try:
            role = float(args[1])
            s_perm = _vars.getPerms(user, guild)
            if s_perm < role + 1 or role is nan:
                raise PermissionError(
                    "Insufficient priviliges to assign permission giver to " + uniStr(guild.name)
                    + " with value " + uniStr(role)
                    + ".\nRequred level: " + uniStr(role + 1)
                    + ", Current level: " + uniStr(perm) + "."
                )
            r_type = "perm"
        except ValueError:
            role = args[1].lower()
            r_type = "role"
        currentSchedule[react] = {"role": role, "deleter": "r" in flags}
        update()
        return (
            "```css\nAdded role giver with reaction to " + uniStr(react)
            + " and " + r_type + " " + uniStr(role)
            + " to channel " + uniStr(channel.name) + ".```"
        )

        
class DefaultPerms:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["DefaultPerm"]
        self.min_level = 3
        self.description = "Sets the default bot permission levels for all users in current server."
        self.usage = "<level[]>"

    async def __call__(self, _vars, argv, user, guild, **void):
        update = self.data["perms"].update
        perms = _vars.data["perms"]
        currPerm = perms.setdefault("defaults", {}).get(guild.id, 0)
        if not argv:
            return (
                "```css\nCurrent default permission level for " + uniStr(guild.name)
                + ": " + uniStr(currPerm) + ".```"
            )
        s_perm = _vars.getPerms(user, guild.id)
        c_perm = await _vars.evalMath(argv, guild.id)
        if s_perm < c_perm + 1 or c_perm is nan:
            raise PermissionError(
                "Insufficient priviliges to change default permission level for " + uniStr(guild.name)
                + " to " + uniStr(c_perm)
                + ".\nRequred level: " + uniStr(c_perm + 1)
                + ", Current level: " + uniStr(perm) + "."
            )
        perms["defaults"][guild.id] = c_perm
        update()
        return (
            "```css\nChanged default permission level of " + uniStr(guild.name)
            + " to " + uniStr(c_perm) + ".```"
        )


class RainbowRole:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["DynamicRole"]
        self.min_level = 3
        self.description = "Causes target role to randomly change colour."
        self.usage = "<0:role[]> <mim_delay[12]> <disable(?d)>"

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
                    "```css\nRemoved all active dynamic role colours in "
                    + uniStr(guild.name) + ".```"
                )
            return (
                "Currently active dynamic role colours in **" + guild.name
                + "**:\n```css\n" + str(guild_special) + "```"
            )
        role = args[0].lower()
        if len(args) < 2:
            delay = 12
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
            + "** to:\n```css\n" + str(guild_special) + "```"
        )
        

follow_default = {
    "follow": False,
    "reacts": {},
}

                  
class Dogpile:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Causes Miza to automatically imitate users when 3+ of the same messages are posted in a row."
        self.usage = "<enable(?e)> <disable(?d)>"

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
            return "```css\nDisabled dogpile imitating for " + uniStr(guild.name) + ".```"
        elif "e" in flags:
            curr["follow"] = True
            update()
            return "```css\nEnabled dogpile imitating for " + uniStr(guild.name) + ".```"
        else:
            return (
                "```css\nCurrently " + uniStr("not " * (not curr["follow"]))
                + "dogpile imitating in " + uniStr(guild.name) + ".```"
            )


class React:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["AutoReact"]
        self.min_level = 3
        self.description = "Causes Miza to automatically assign a reaction to messages containing the substring."
        self.usage = "<0:react_to[]> <1:react_data[]> <disable(?d)>"

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
                return "```css\nRemoved all auto reacts for " + uniStr(guild.name) + ".```"
            else:
                return (
                    "Currently active auto reacts for " + uniStr(guild.name) + ":\n```json\n"
                    + str(curr) + "```"
                )
        a = args[0].lower()
        if "d" in flags:
            if a in curr["reacts"]:
                curr["reacts"].pop(a)
                update()
                return (
                    "```css\nRemoved " + uniStr(a) + " from the auto react list for "
                    + uniStr(guild.name) + ".```"
                )
            else:
                raise LookupError(uniStr(a) + " is not in the auto react list.")
        curr["reacts"][a] = args[1]
        update()
        return (
            "```css\nAdded " + uniStr(a) + ": " + uniStr(args[1]) + " to the auto react list for "
            + uniStr(guild.name) + ".```"
        )


class MessageLog:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Causes Miza to log message events into the current channel."
        self.usage = "<enable(?e)> <disable(?d)>"

    async def __call__(self, _vars, flags, channel, guild, **void):
        data = _vars.data["logs"]
        update = _vars.updaters["logs"].update
        if "e" in flags:
            data[guild.id] = channel.id
            update()
            return (
                "```css\nEnabled message logging in " + uniStr(channel.name)
                + " for " + uniStr(guild.name) + ".```"
            )
        elif "d" in flags:
            if guild.id in data:
                data.pop(guild.id)
                update()
            return (
                "```css\nDisabled message logging for " + uniStr(guild.name) + ".```"
            )
        if guild.id in data:
            c_id = data[guild.id]
            channel = await _vars.fetch_channel(c_id)
            return (
                "```css\nMessage logging for " + uniStr(guild.name)
                + " is currently enabled in " + uniStr(channel.name)
                + ".```"
            )
        return (
            "```css\nMessage logging is currently disabled in "
            + uniStr(guild.name) + ".```"
        )


class updateLogs:
    is_update = True
    name = "logs"

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
                u = before.author
                name = u.name
                name_id = name + bool(u.display_name) * ("#" + u.discriminator)
                url = u.avatar_url
                c_id = self.data[guild.id]
                channel = await self._vars.fetch_channel(c_id)
                emb = discord.Embed(colour=self._vars.randColour())
                emb.set_author(name=name_id, icon_url=url, url=url)
                emb.description = (
                    "Message edited in <#"
                    + str(before.channel.id) + ">:"
                )
                emb.add_field(name="Before", value=self._vars.strMessage(before))
                emb.add_field(name="After", value=self._vars.strMessage(after))
                await channel.send(embed=emb)

    async def _delete_(self, message, bulk=False, **void):
        if bulk:
            return
        guild = message.guild
        if guild.id in self.data:
            now = datetime.datetime.utcnow()
            u = message.author
            name = u.name
            name_id = name + bool(u.display_name) * ("#" + u.discriminator)
            url = u.avatar_url
            c_id = self.data[guild.id]
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
                        #print(e, e.target, now - e.created_at)
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
                                #print(t, e.target)
                if t.bot or u.id == t.id == cu_id:
                    return
            except (discord.Forbidden, discord.HTTPException):
                init = "[UNKNOWN USER]"
            channel = await self._vars.fetch_channel(c_id)
            emb = discord.Embed(colour=self._vars.randColour())
            emb.set_author(name=name_id, icon_url=url, url=url)
            emb.description = (
                init + " deleted message from <#"
                + str(message.channel.id) + ">:"
            )
            emb.add_field(name="Content", value=self._vars.strMessage(message))
            await channel.send(embed=emb)


class updateFollows:
    is_update = True
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
                                asyncio.create_task(message.channel.send(checker))
                    else:
                        if len(checker) > 100:
                            checker = ""
                        curr[0] = checker
                        curr[1] = xrand(-1, 2)
                    curr[2] = u_id
                    #print(curr)
            try:
                for r in following[g_id]["reacts"]:
                    if r in words:
                        await message.add_reaction(following[g_id]["reacts"][r])
            except discord.Forbidden:
                print(traceback.format_exc())

    async def __call__(self):
        pass


class updateRolegiver:
    is_update = True
    name = "rolegivers"

    def __init__(self):
        pass

    async def _nocommand_(self, text, message, **void):
        if message.guild is None:
            return
        user = message.author
        guild = message.guild
        _vars = self._vars
        currentSchedule = self.data.get(message.channel.id, {})
        for k in currentSchedule:
            if k in text:
                curr = currentSchedule[k]
                role = curr["role"]
                deleter = curr["deleter"]
                try:
                    perm = float(role)
                    currPerm = _vars.getPerms(user, guild)
                    if perm > currPerm:
                        _vars.setPerms(user, guild, perm)
                    print("Granted perm " + str(perm) + " to " + user.name + ".")
                except ValueError:
                    for r in guild.roles:
                        if r.name.lower() == role:
                            await user.add_roles(
                                r,
                                reason="Keyword found in message.",
                                atomic=True,
                            )
                            print("Granted role " + r.name + " to " + user.name + ".")
                if deleter:
                    try:
                        d_id = message.id
                        await message.delete()
                        _vars.logDelete(d_id)
                    except discord.NotFound:
                        pass

    async def __call__(self):
        pass


class updatePerms:
    is_update = True
    name = "perms"

    def __init__(self):
        pass

    async def __call__(self):
        pass


class updateColours:
    is_update = True
    name = "rolecolours"

    def __init__(self):
        self.counter = 0
        self.count = 0
        self.delay = 0

    async def changeColour(self, g_id, roles):
        guild = await self._vars.fetch_guild(g_id)
        colTime = 12
        l = list(roles)
        for r in l:
            try:
                role = guild.get_role(r)
                delay = roles[r]
                if not (self.counter + r) % delay:
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
            except discord.HTTPException as ex:
                print(traceback.format_exc())
                self._vars.blocked += 60
                break

    async def __call__(self):
        self.counter = self.counter + 1 & 65535
        if time.time() > self.delay:
            self.delay = time.time() + 60
            self.count = 0
        for g in self.data:
            if self.count < 48 and self._vars.blocked <= 0:
                asyncio.create_task(self.changeColour(g, self.data[g]))
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
    is_update = True
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
                    asyncio.create_task(getBans(_vars, guild))
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
                                        "```css\n" + uniStr(u_target.name)
                                        + " has been unbanned from " + uniStr(g_target.name) + ".```"
                                    )
                            except:
                                if c_id is not None:
                                    await c_target.send(
                                        "```css\nUnable to unban " + uniStr(u_target.name)
                                        + " from " + uniStr(g_target.name) + ".```"
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
