try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Restart(Command):
    name = ["Shutdown"]
    min_level = nan
    description = "Restarts or shuts down ⟨MIZA⟩."

    async def __call__(self, channel, name, **void):
        _vars = self._vars
        client = _vars.client
        if name.lower() == "shutdown":
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        _vars.update()
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        for _ in loop(5):
            try:
                f = open(_vars.restart, "wb")
                f.close()
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        for _ in loop(8):
            try:
                if "log.txt" in os.listdir():
                    os.remove("log.txt")
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
        if name.lower() == "shutdown":
            f = open(_vars.shutdown, "wb")
            f.close()
        try:
            await client.close()
        except:
            del client
        del _vars
        sys.exit()


class Execute(Command):
    name = ["Exec", "Eval"]
    min_level = nan
    description = (
        "Causes all messages in the current channel to be executed as python code on ⟨MIZA⟩."
        + " WARNING: DO NOT ALLOW UNTRUSTED USERS TO POST IN CHANNEL."
    )
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, _vars, flags, channel, **void):
        if "e" in flags or "a" in flags:
            _vars.database.exec.channel = channel
            return (
                "```css\nSuccessfully changed code execution channel to ["
                + noHighlight(channel.name) + "].```"
            )
        elif "d" in flags:
            _vars.database.exec.channel = freeClass(id=None)
            return (
                "```fix\nSuccessfully removed code execution channel.```"
            )
        return (
            "```css\ncode channel is currently set to ["
            + noHighlight(_vars.database.exec.channel.name) + "].```"
        )


class Suspend(Command):
    name = ["Block", "Blacklist"]
    min_level = nan
    description = "Prevents a user from accessing ⟨MIZA⟩'s commands. Overrides <perms>."
    usage = "<0:user> <1:value[]>"

    async def __call__(self, _vars, user, guild, args, **void):
        update = self.data.blacklist.update
        if len(args) < 2:
            if len(args) >= 1:
                user = await _vars.fetch_user(verifyID(args[0]))
            susp = _vars.data.blacklist.get(user.id, None)
            return (
                "```css\nCurrent suspension status of [" + noHighlight(user.name) + "]: ["
                + noHighlight(susp) + "].```"
            )
        else:
            user = await _vars.fetch_user(verifyID(args[0]))
            change = await _vars.evalMath(args[1], guild.id)
            _vars.data.blacklist[user.id] = change
            update()
            return (
                "```css\nChanged suspension status of [" + noHighlight(user.name) + "] to ["
                + noHighlight(change) + "].```"
            )


class UpdateExec(Database):
    name = "exec"
    no_file = True

    def __init__(self, *args):
        self.channel = freeClass(id=None)
        super().__init__(*args)

    def procFunc(self, proc, _vars):
        print(proc)
        try:
            output = eval(proc, _vars._globals)
        except:
            exec(proc, _vars._globals)
            output = str(proc) + " Successfully executed!"
        try:
            if type(output) in (str, bytes, dict) or isinstance(output, freeClass):
                raise TypeError
            output = tuple(output)
        except TypeError:
            pass
        return output

    async def _typing_(self, user, channel, **void):
        _vars = self._vars
        if user.id == _vars.client.user.id:
            return
        if not hasattr(channel, "guild") or channel.guild is None:
            emb = discord.Embed()
            emb.add_field(
                name=str(user) + " (" + str(user.id) + ")",
                value="```ini\n[typing...]```",
            )
            await self.channel.send(embed=emb, delete_after=20)

    async def _nocommand_(self, message, **void):
        _vars = self._vars
        if message.author.id != self._vars.owner_id:
            return
        if message.channel.id == self.channel.id:
            proc = message.content
            while proc[0] == " ":
                proc = proc[1:]
            if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\") or proc.startswith("#"):
                return
            if proc.startswith("`") and proc.endswith("`"):
                proc = proc.strip("`")
            if not proc:
                return
            output = None
            try:
                output = await create_future(self.procFunc, proc, _vars)
                if type(output) is tuple:
                    output = await recursiveCoro(output)
                elif asyncio.iscoroutine(output):
                    output = await output
                await self.channel.send(limStr("```py\n" + str(output) + "```", 2000))
            except:
                print(traceback.format_exc())
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
                value=strMessage(message),
            )
            await self.channel.send(embed=emb)


class UpdateBlacklist(Database):
    name = "blacklist"
    suspected = "blacklist.json"
    user = True

    def __init__(self, *args):
        self.suspclear = inf
        try:
            self.lastsusp = None
            f = open(self.suspected, "r")
            susp = f.read()
            f.close()
            os.remove(self.suspected)
            if susp:
                u_id = int(susp)
                udata = self.data.get(u_id, 0)
                days = max(0, (udata - time.time()) / 86400)
                try:
                    days **= 4
                except (OverflowError, ValueError, TypeError):
                    days = inf
                days += 1.125
                udata = time.time() + days * 86400
                self.data[u_id] = udata
                if days >= self._vars.min_suspend - 1:
                    self.lastsusp = u_id
                self.update()
                self.update(True)
            print(self.lastsusp)
        except FileNotFoundError:
            pass
        super().__init__(*args)

    async def _command_(self, user, command, **void):
        if user.id not in (self._vars.client.user.id, self._vars.owner_id):
            tc = getattr(command, "time_consuming", 0)
            self.suspclear = time.time() + 10 + (tc * 2) ** 2
            f = open(self.suspected, "w")
            f.write(str(user.id))
            f.close()

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