try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Restart(Command):
    name = ["Shutdown", "Reload"]
    min_level = nan
    description = "Restarts, reloads, or shuts down ⟨MIZA⟩, with an optional delay."
    _timeout_ = inf

    async def __call__(self, message, channel, guild, argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        if argv:
            delay = await bot.evalTime(argv, guild)
            await channel.send("Preparing to " + name + " in " + sec2Time(delay) + "...")
            if delay > 0:
                await asyncio.sleep(delay)
        if name == "reload":
            create_future_ex(bot.getModules)
            return "Reloading... :wave:"
        elif name == "shutdown":
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        print.close()
        bot.update()
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        for _ in loop(5):
            try:
                f = open(bot.restart, "wb")
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
            f = open(bot.shutdown, "wb")
            f.close()
        try:
            await client.close()
        except:
            del client
        del bot
        sys.exit()


class Execute(Command):
    name = ["Exec", "Eval"]
    min_level = nan
    description = "Causes all messages by the bot owner in the current channel to be executed as python code on ⟨MIZA⟩."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, bot, flags, message, channel, **void):
        update = bot.database.exec.update
        if "e" in flags or "a" in flags:
            create_task(message.add_reaction("❗"))
            bot.data.exec[channel.id] = True
            update()
            return (
                "```css\nSuccessfully enabled code execution in ["
                + noHighlight(channel.name) + "].```"
            )
        elif "d" in flags:
            bot.data.exec.pop(channel.id)
            update()
            return (
                "```fix\nSuccessfully removed code execution channel.```"
            )
        return (
            "```css\ncode channel is currently set to "
            + noHighlight(list(bot.data.exec)) + ".```"
        )


class Suspend(Command):
    name = ["Block", "Blacklist"]
    min_level = nan
    description = "Prevents a user from accessing ⟨MIZA⟩'s commands. Overrides <perms>."
    usage = "<0:user> <1:value[]>"

    async def __call__(self, bot, user, guild, args, **void):
        update = self.data.blacklist.update
        if len(args) < 2:
            if len(args) >= 1:
                user = await bot.fetch_user(verifyID(args[0]))
            susp = bot.data.blacklist.get(user.id, None)
            return (
                "```css\nCurrent suspension status of [" + noHighlight(user.name) + "]: ["
                + noHighlight(susp) + "].```"
            )
        else:
            user = await bot.fetch_user(verifyID(args[0]))
            change = await bot.evalTime(" ".join(args[1]), guild.id, bot.data.blacklist.get(user.id, time.time()))
            bot.data.blacklist[user.id] = change
            update()
            return (
                "```css\nChanged suspension status of [" + noHighlight(user.name) + "] to ["
                + noHighlight(change) + "].```"
            )


class UpdateExec(Database):
    name = "exec"
    no_delete = True

    def procFunc(self, proc, bot):
        print(proc)
        try:
            output = eval(proc, bot._globals)
        except:
            exec(proc, bot._globals)
            output = None
        try:
            if awaitable(output):
                raise TypeError
            if type(output) in (str, bytes):
                raise TypeError
            if issubclass(type(output), collections.Mapping):
                raise TypeError
            output = tuple(output)
        except TypeError:
            pass
        return output

    async def sendDeleteID(self, c_id, delete_after=20, **kwargs):
        channel = await self.bot.fetch_channel(c_id)
        message = await channel.send(**kwargs)
        if isValid(delete_after):
            create_task(self.bot.silentDelete(message, no_log=True, delay=delete_after))

    async def _typing_(self, user, channel, **void):
        bot = self.bot
        if user.id == bot.client.user.id:
            return
        if not hasattr(channel, "guild") or channel.guild is None:
            emb = discord.Embed()
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=strURL(user.avatar_url))
            emb.description = "```ini\n[typing...]```"
            for c_id in self.data:
                create_task(self.sendDeleteID(c_id, embed=emb))

    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if message.author.id == self.bot.owner_id and channel.id in self.data:
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
                output = await create_future(self.procFunc, proc, bot, priority=True)
                if type(output) is tuple:
                    output = await recursiveCoro(output)
                elif awaitable(output):
                    output = await output
                await channel.send(limStr("```py\n" + str(output) + "```", 2000))
            except:
                print(traceback.format_exc())
                await sendReact(channel, limStr(
                    "```py\n" + traceback.format_exc().replace("```", "") + "```",
                    2000,
                ), reacts="❎")
            if output is not None:
                bot._globals["output"] = output
                bot._globals["_"] = output
        elif message.guild is None:
            user = message.author
            emb = discord.Embed()
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=strURL(user.avatar_url))
            emb.description = strMessage(message)
            for c_id in self.data:
                create_task(self.sendDeleteID(c_id, delete_after=inf, embed=emb))


class Trust(Command):
    name = ["Untrust"]
    min_level = nan
    description = "Adds or removes a server from the bot's trusted server list."
    usage = "<server_id(curr)(?a)> <enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, bot, flags, message, guild, argv, **void):
        update = bot.database.trusted.update
        if "a" in flags:
            guilds = bot.client.guilds
        else:
            if argv:
                g_id = verifyID(argv)
                guild = await bot.fetch_guild(g_id)
            guilds = [guild]
        if "e" in flags:
            create_task(message.add_reaction("❗"))
            for guild in guilds:
                bot.data.trusted[guild.id] = True
            update()
            return (
                "```css\nSuccessfully added ["
                + ", ".join(noHighlight(guild) for guild in guilds) + "] to trusted list.```"
            )
        elif "d" in flags:
            create_task(message.add_reaction("❗"))
            for guild in guilds:
                try:
                    bot.data.trusted.pop(guild.id)
                except KeyError:
                    pass
            update()
            return (
                "```fix\nSuccessfully removed trusted server.```"
            )
        return (
            "```css\nTrusted server list "
            + str(list(noHighlight(bot.cache.guilds[g]) for g in bot.data.trusted)) + ".```"
        )


class UpdateTrusted(Database):
    name = "trusted"


class UpdateBlacklist(Database):
    name = "blacklist"
    suspected = "blacklist.json"
    user = True

    def __load__(self):
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
                if days >= self.bot.min_suspend - 1:
                    self.lastsusp = u_id
                self.update()
                self.update(True)
            print(self.lastsusp)
        except FileNotFoundError:
            pass

    async def _command_(self, user, command, **void):
        if user.id not in (self.bot.client.user.id, self.bot.owner_id):
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
        bot = self.bot
        if self.lastsusp is not None:
            u_susp = await bot.fetch_user(self.lastsusp)
            self.lastsusp = None
            channel = await bot.getDM(u_susp)
            secs = self.data.get(u_susp.id, 0) - time.time()
            msg = (
                "Apologies for the inconvenience, but your account has been "
                + "flagged as having attempted a denial-of-service attack.\n"
                + "This will expire in `" + sec2Time(secs) + "`.\n"
                + "If you believe this is an error, please notify <@"
                + str(bot.owner_id) + "> as soon as possible."
            )
            print(
                u_susp.name + " may be attempting a DDOS attack. Expires in "
                + sec2Time(secs) + "."
            )
            await channel.send(msg)