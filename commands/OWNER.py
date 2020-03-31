import discord
try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from smath import *


class Execute:
    is_command = True

    def __init__(self):
        self.name = ["Exec", "Eval"]
        self.min_level = nan
        self.description = (
            "Causes all messages in the current channel to be executed as python code on the bot."
            + " WARNING: DO NOT ALLOW UNTRUSTED USERS TO POST IN CHANNEL."
        )
        self.usage = "<enable(?e)> <disable(?d)>"
        self.flags = "ed"

    async def __call__(self, _vars, flags, channel, **void):
        if "e" in flags:
            _vars.database["exec"].channel = channel
            return (
                "```css\nSuccessfully changed code channel to "
                + uniStr(channel.id) + ".```"
            )
        elif "d" in flags:
            _vars.database["exec"].channel = freeClass(id=None)
            return (
                "```css\nSuccessfully removed code channel.```"
            )
        return (
            "```css\ncode channel is currently set to "
            + uniStr(_vars.database["exec"].channel.id) + ".```"
        )
        

class updateExec:
    is_database = True
    name = "exec"
    no_file = True

    def __init__(self):
        self.channel = freeClass(id=None)

    async def __call__(self):
        pass

    async def _nocommand_(self, message, **void):
        _vars = self._vars
        if message.author.id == _vars.client.user.id:
            return
        if message.channel.id == self.channel.id:
            proc = message.content
            while proc[0] == " ":
                proc = proc[1:]
            if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\\\") or proc.startswith("#"):
                return
            if proc.startswith("`") and proc.endswith("`"):
                proc = proc.strip("`")
            if not proc:
                return
            output = None
            try:
                print(proc)
                try:
                    output = eval(proc, _vars._globals)
                except:
                    try:
                        exec(proc, _vars._globals)
                        output = str(proc) + " Successfully executed!"
                    except:
                        output = traceback.format_exc()
                try:
                    if type(output) in (str, bytes, dict) or isinstance(output, freeClass):
                        raise TypeError
                    output = tuple(output)
                except TypeError:
                    pass
                if type(output) is tuple:
                    output = await _vars.recursiveCoro(output)
                elif asyncio.iscoroutine(output):
                    output = await output
                await self.channel.send(limStr("```py\n" + str(output) + "```", 2000))
            except:
                await self.channel.send(limStr(
                    "```py\n" + traceback.format_exc().replace("```", "") + "```",
                    2000,
                ))
            if output is not None:
                _vars._globals["output"] = output
                _vars._globals["_"] = output
        elif message.guild is None:
            emb = discord.Embed()
            emb.add_field(
                name=str(message.author) + " (" + str(message.author.id) + ")",
                value=_vars.strMessage(message),
            )
            await self.channel.send(embed=emb)


class Suspend:
    is_command = True

    def __init__(self):
        self.name = ["Block", "Blacklist"]
        self.min_level = nan
        self.description = "Prevents a user from accessing the bot's commands. Overrides <perms>."
        self.usage = "<0:user> <1:value[]>"

    async def __call__(self, _vars, user, guild, args, **void):
        update = self.data["blacklist"].update
        if len(args) < 2:
            if len(args) >= 1:
                user = await _vars.fetch_user(_vars.verifyID(args[0]))
            susp = _vars.data["blacklist"].get(user.id, None)
            return (
                "```css\nCurrent suspension status of " + uniStr(user.name) + ": "
                + uniStr(susp) + ".```"
            )
        else:
            user = await _vars.fetch_user(_vars.verifyID(args[0]))
            change = await _vars.evalMath(args[1], guild.id)
            _vars.data["blacklist"][user.id] = change
            update()
            return (
                "```css\nChanged suspension status of " + uniStr(user.name) + " to "
                + uniStr(change) + ".```"
            )


class updateBlacklist:
    is_database = True
    name = "blacklist"
    suspected = "blacklist.json"
    user = True

    def __init__(self):
        self.suspclear = inf
        try:
            self.lastsusp = None
            f = open(self.suspected, "r")
            susp = f.read()
            f.close()
            os.remove(self.suspected)
            if susp:
                u_id = int(susp)
                udata = self.data[u_id]
                days = max(0, (udata - time.time()) / 86400)
                try:
                    days **= 4
                except (OverflowError, ValueError, TypeError):
                    days = inf
                days += 1.125
                udata = time.time() + days * 86400
                if days >= self._vars.min_suspend - 1:
                    self.lastsusp = u_id
                self.update()
                self.update(True)
            print(self.lastsusp)
        except FileNotFoundError:
            pass

    async def _command_(self, user, command, **void):
        if user.id not in (self._vars.client.user.id, self._vars.owner_id):
            tc = getattr(command, "time_consuming", 0)
            self.suspclear = time.time() + 10 + (tc * 2) ** 2
            f = open(self.suspected, "w")
            f.write(str(user.id))
            f.close()
            self.update()

    async def __call__(self, **void):
        if time.time() - self.suspclear:
            self.suspclear = inf
            try:
                if self.suspected in os.listdir():
                    os.remove(self.suspected)
            except:
                print(traceback.format_exc())
        _vars = self._vars
        if self.lastsusp is not None:
            u_susp = await _vars.fetch_user(self.lastsusp)
            self.lastsusp = None
            channel = await _vars.getDM(u_susp)
            secs = self.data.get(u_susp.id, 0) - time.time()
            msg = (
                "Apologies for the inconvenience, but your account has been "
                + "flagged as having attempted a denial-of-service attack.\n"
                + "This will expire in `" + sec2Time(secs) + "`.\n"
                + "If you believe this is an error, please notify <@"
                + str(_vars.owner_id) + "> as soon as possible."
            )
            print(
                u_susp.name + " may be attempting a DDOS attack. Expires in "
                + sec2Time(secs) + "."
            )
            await channel.send(msg)