import datetime, traceback, copy
from smath import *


class purge:
    is_command = True

    def __init__(self):
        self.name = ["del", "delete"]
        self.min_level = 1
        self.description = "Deletes a number of messages from a certain user in current channel."
        self.usage = "<1:user{bot}(?a)> <0:count[1]> <hide(?h)>"

    async def __call__(self, client, _vars, argv, args, channel, name, flags, perm, **void):
        t_user = -1
        if "a" in flags or "@everyone" in argv or "@here" in argv:
            t_user = None
        if len(args) < 2:
            if t_user == -1:
                t_user = client.user
            if len(args) < 1:
                count = 1
            else:
                count = round(_vars.evalMath(args[0]))
        else:
            a1 = args[0]
            a2 = " ".join(args[1:])
            count = round(_vars.evalMath(a2))
            if t_user == -1:
                t_user = await client.fetch_user(_vars.verifyID(a1))
        if t_user != client.user:
            if perm < 3:
                raise PermissionError (
                    "Insufficient priviliges for command " + uniStr(name)
                    + " for target user.\nRequred level: " + uniStr(3)
                    + ", Current level: " + uniStr(perm) + "."
                )
        lim = count*2+16
        if not isValid(lim):
            lim = None
        hist = await channel.history(limit=lim).flatten()
        delM = []
        deleted = 0
        for m in hist:
            if count <= 0:
                break
            if t_user is None or m.author.id == t_user.id:
                delM.append(m)
                count -= 1
        try:
            while len(delM):
                await channel.delete_messages(delM[:100])
                deleted += min(len(delM), 100)
                delM = delM[100:]
        except:
            print(traceback.format_exc())
            for m in delM:
                try:
                    await m.delete()
                    deleted += 1
                except Exception as ex:
                    print(repr(ex))
        if not "h" in flags:
            return (
                "```css\nDeleted " + uniStr(deleted)
                + " message" + "s" * (deleted != 1) + "!```"
                )


class ban:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 3
        self.description = "Bans a user for a certain amount of hours, with an optional reason."
        self.usage = "<0:user> <1:hours[]> <2:reason[]> <hide(?h)>"

    async def __call__(self, client, _vars, args, user, channel, guild, flags, perm, **void):
        dtime = datetime.datetime.utcnow().timestamp()
        a1 = args[0]
        t_user = await client.fetch_user(_vars.verifyID(a1))
        s_perm = perm
        t_perm = _vars.getPerms(t_user, guild)
        if t_perm + 1 >= s_perm or not isValid(t_perm):
            if len(args) > 1:
                raise PermissionError (
                    "Insufficient priviliges to ban " + uniStr(t_user.name)
                    + " from " + uniStr(guild.name)
                    + ".\nRequired level: " + uniStr(t_perm + 1)
                    + ", Current level: " + uniStr(s_perm) + "."
                )
        if len(args) < 2:
            tm = 0
        else:
            tm = _vars.evalMath(args[1])
        if len(args) >= 3:
            msg = args[2]
        else:
            msg = None
        g_id = guild.id
        g_bans = _vars.bans.get(g_id, {})
        is_banned = g_bans.get(t_user.id, None)
        if is_banned is not None:
            is_banned = is_banned[0] - dtime
            if len(args) < 2:
                return (
                    "```css\nCurrent ban for " + uniStr(t_user.name)
                    + " from " + uniStr(guild.name) + ": "
                    + uniStr(" ".join(timeConv(is_banned))) +  ".```"
                )
        elif len(args) < 2:
            return (
                "```css\n" + uniStr(t_user.name)
                + " is currently not banned from " + uniStr(guild.name) + ".```"
                )
        secs = tm * 3600
        g_bans[t_user.id] = [secs + dtime, channel.id]
        _vars.bans[g_id] = g_bans
        doParallel(_vars.update)
        if tm >= 0:
            await guild.ban(t_user, reason=msg, delete_message_days=0)
        response = None
        if is_banned:
            response = (
                "```css\nUpdated ban for " + uniStr(t_user.name)
                + " from " + uniStr(" ".join(timeConv(is_banned)))
                + " to " + uniStr(" ".join(timeConv(secs)))
                + "."
            )
        elif tm >= 0:
            response = (
                "```css\n" + uniStr(t_user.name)
                + " has been banned from " + uniStr(guild.name)
                + " for " + uniStr(" ".join(timeConv(secs))) + "."
            )
        if msg:
            response += " Reason: " + uniStr(msg) + "."
        if response and "h" not in flags:
            return response + "```"


class roleGiver:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["verifier"]
        self.min_level = 3
        self.description = "Adds an automated role giver to the current channel."
        self.usage = "<0:react_to[]> <1:role[]> <1:perm[]> <disable(?d)> <remover(?r)>"

    async def __call__(self, _vars, argv, args, user, channel, guild, flags, **void):
        if "d" in flags:
            _vars.scheduled[channel.id] = {}
            doParallel(_vars.update)
            return "```css\nRemoved all automated role givers from channel " + uniStr(channel.name) + ".```"
        currentSchedule = _vars.scheduled.get(channel.id, {})
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
        _vars.scheduled[channel.id] = currentSchedule
        doParallel(_vars.update)
        return (
            "```css\nAdded role giver with reaction to " + uniStr(react)
            + " and " + r_type + " " + uniStr(role)
            + " to channel " + uniStr(channel.name) + ".```"
            )

        
class defaultPerms:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["defaultPerm"]
        self.min_level = 3
        self.description = "Sets the default bot permission levels for all users in current server."
        self.usage = "<level[]>"

    async def __call__(self, _vars, argv, user, guild, **void):
        currPerm = _vars.perms.get("defaults", {}).get(guild.id, 0)
        if not argv:
            return (
                "```css\nCurrent default permission level for " + uniStr(guild.name)
                + ": " + uniStr(currPerm) + ".```"
                )
        s_perm = _vars.getPerms(user, guild)
        c_perm = _vars.evalMath(argv)
        if s_perm < c_perm + 1 or c_perm is nan:
            raise PermissionError(
                "Insufficient priviliges to change default permission level for " + uniStr(guild.name)
                + " to " + uniStr(c_perm)
                + ".\nRequred level: " + uniStr(c_perm + 1)
                + ", Current level: " + uniStr(perm) + "."
                )
        if not "defaults" in _vars.perms:
            _vars.perms["defaults"] = {}
        _vars.perms["defaults"][guild.id] = c_perm
        doParallel(_vars.update)
        return (
            "```css\nChanged default permission level of " + uniStr(guild.name)
            + " to " + uniStr(c_perm) + ".```"
            )


class rainbowRole:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["colourRole"]
        self.min_level = 3
        self.description = "Causes target role to randomly change colour."
        self.usage = "<0:role[]> <mim_delay[6]> <disable(?d)>"

    async def __call__(self, _vars, flags, args, argv, guild, **void):
        guild_special = _vars.special.get(guild.id, {})
        if not argv:
            return (
                "Currently active dynamic role colours in **" + guild.name
                + "**:\n```css\n" + str(guild_special) + "```"
                )
        role = args[0].lower()
        if len(args) < 2:
            delay = 6
        else:
            delay = _vars.evalMath(" ".join(args[1:]))
        for r in guild.roles:
            if role in r.name.lower():
                if "d" in flags:
                    try:
                        guild_special.pop(r.id)
                    except KeyError:
                        pass
                else:
                    guild_special[r.id] = delay
        _vars.special[guild.id] = guild_special
        doParallel(_vars.update)
        return (
            "Changed dynamic role colours for **" + guild.name
            + "** to:\n```css\n" + str(guild_special) + "```"
            )
        

follow_default = {
    "follow": False,
    "reacts": {},
    }

                  
class follow:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["dogpile"]
        self.min_level = 3
        self.description = "Causes Miza to automatically imitate users when 3 of the same message is posted in a row."
        self.usage = "<enable(?e)> <disable(?d)>"

    async def __call__(self, _vars, flags, guild, **void):
        curr = _vars.following.get(guild.id, copy.deepcopy(follow_default))
        if type(curr) is not dict:
            curr = copy.deepcopy(follow_default)
        if "d" in flags:
            curr["follow"] = False
            _vars.following[guild.id] = curr
            doParallel(_vars.update)
            return "```css\nDisabled follow imitating for " + uniStr(guild.name) + ".```"
        elif "e" in flags:
            curr["follow"] = True
            _vars.following[guild.id] = curr
            doParallel(_vars.update)
            return "```css\nEnabled follow imitating for " + uniStr(guild.name) + ".```"
        else:
            return (
                "```css\nCurrently " + uniStr("not " * (not curr["follow"]))
                + "follow imitating in " + uniStr(guild.name) + ".```"
                )


class react:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["autoreact"]
        self.min_level = 3
        self.description = "Causes Miza to automatically assign a reaction to messages containing the substring."
        self.usage = "<react_to[]> <react_data[]> <disable(?d)>"

    async def __call__(self, _vars, flags, guild, argv, args, **void):
        curr = _vars.following.get(guild.id, copy.deepcopy(follow_default))
        if type(curr) is not dict:
            curr = copy.deepcopy(follow_default)
        if not argv:
            if "d" in flags:
                curr["reacts"] = {}
                _vars.following[guild.id] = curr
                doParallel(_vars.update)
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
                _vars.following[guild.id] = curr
                doParallel(_vars.update)
                return "```css\nRemoved " + uniStr(a) + " from the auto react list for " + uniStr(guild.name) + ".```"
            else:
                raise LookupError(uniStr(a) + " is not in the auto react list.")
        curr["reacts"][a] = args[1]
        _vars.following[guild.id] = curr
        doParallel(_vars.update)
        return (
            "```css\nAdded " + uniStr(a) + ": " + uniStr(args[1]) + " to the auto react list for "
            + uniStr(guild.name) + ".```"
            )
