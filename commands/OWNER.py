try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Restart(Command):
    name = ["Shutdown", "Reload", "Reboot"]
    min_level = nan
    description = "Restarts, reloads, or shuts down ⟨MIZA⟩, with an optional delay."
    _timeout_ = inf

    async def __call__(self, message, channel, guild, argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        if argv:
            if "in" in argv:
                argv = argv[argv.rindex("in") + 2:]
            delay = await bot.evalTime(argv, guild)
            await channel.send("Preparing to " + name + " in " + sec2Time(delay) + "...")
            emb = discord.Embed(colour=discord.Colour(1))
            emb.set_author(name=str(client.user), url=bot.website, icon_url=bestURL(client.user))
            emb.description = "```ini\n[I will be " + ("restarting", "shutting down")[name == "shutdown"] + " in " + sec2Time(delay) + ", apologies for any inconvenience.]```"
            await bot.event("_announce_", embed=emb)
            if delay > 0:
                await asyncio.sleep(delay)
        if name == "reload":
            create_future_ex(bot.getModules)
            return "Reloading... :wave:"
        elif name == "shutdown":
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        fut = create_task(channel.trigger_typing())
        bot.closed = True
        print.close()
        t = time.time()
        await bot.event("_destroy_")
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
        if time.time() - t < 1:
            await asyncio.sleep(1)
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
    usage = "<type> <enable(?e)> <disable(?d)>"
    flags = "aed"
    terminal_types = demap({
        "null": 0,
        "main": 1,
        "relay": 2,
        "virtual": 4,
        "log": 8,
    })

    async def __call__(self, bot, flags, argv, message, channel, guild, **void):
        update = bot.database.exec.update
        if not argv:
            argv = 0
        try:
            num = int(argv)
        except (TypeError, ValueError):
            out = argv.lower()
            num = self.terminal_types[out]
        else:
            out = self.terminal_types[num]
        if "e" in flags or "a" in flags:
            if num == 0:
                num = 4
            try:
                bot.data.exec[channel.id] |= num
            except KeyError:
                bot.data.exec[channel.id] = num
            update()
            out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec[channel.id]))
            create_task(message.add_reaction("❗"))
            return (
                "```css\n[" + out + "] terminal now enabled in [#"
                + noHighlight(channel.name) + "].```"
            )
        elif "d" in flags:
            try:
                if num == 0:
                    out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec.pop(channel.id)))
                else:
                    bot.data.exec[channel.id] &= -num - 1
                    if not bot.data.exec[channel.id]:
                        bot.data.exec.pop(channel.id)
            except KeyError:
                pass
            update()
            return (
                "```css\nSuccessfully removed [" + out + "] terminal.```"
            )
        return (
            "```css\nTerminals currently set to "
            + noHighlight(bot.data.exec) + ".```"
        )


class UpdateExec(Database):
    name = "exec"
    no_delete = True
    virtuals = {}
    listeners = {}

    qmap = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "〝": '"',
        "〞": '"',
        "⸌": "'",
        "⸍": "'",
        "⸢": "'",
        "⸣": "'",
        "⸤": "'",
        "⸥": "'",
    }
    qtrans = "".maketrans(qmap)

    _print = lambda self, *args, sep=" ", end="\n", prefix="", channel=None, **void: create_task(channel.send(limStr("```\n" + str(sep).join((i if type(i) is str else str(i)) for i in args) + str(end) + str(prefix) + "```", 2000)))
    def _input(self, *args, channel=None, **kwargs):
        self._print(*args, channel=channel, **kwargs)
        self.listeners.__setitem__(channel.id, None)
        while self.listeners[channel.id] is None:
            time.sleep(0.5)
        return self.listeners.pop(channel.id)

    async def procFunc(self, proc, channel, bot, term=0):
        try:
            if self.listeners[channel.id] is None:
                self.listeners[channel.id] = proc
                return
        except KeyError:
            pass
        if not proc:
            return
        if term & 1:
            glob = bot._globals
        else:
            try:
                glob = self.virtuals[channel.id]
            except KeyError:
                glob = self.virtuals[channel.id] = dict(bot._globals)
                glob.update(dict(
                    print=lambda *args, **kwargs: self._print(*args, channel=channel, **kwargs),
                    input=lambda *args, **kwargs: self._input(*args, channel=channel, **kwargs),
                ))
        succ = False
        if "\n" not in proc:
            if proc.startswith("await "):
                proc = proc[6:]
        try:
            output = await create_future(eval, proc, glob, priority=True)
        except SyntaxError:
            pass
        else:
            succ = True
        if not succ:
            output = await create_future(exec, proc, glob, priority=True)
        if awaitable(output):
            output = await output
        if output is not None:
            glob["_"] = output 
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
            emb = discord.Embed(colour=randColour())
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=bestURL(user))
            emb.description = "```ini\n[typing...]```"
            for c_id, flag in self.data.items():
                if flag & 2:
                    create_task(self.sendDeleteID(c_id, embed=emb))

    def prepare_string(self, s, lim=2000, fmt="py"):
        if type(s) is not str:
            s = str(s)
        if s:
            return limStr("```" + fmt + "\n" + s + "```", lim)
        return "``` ```"

    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if message.author.id in self.bot.owners and channel.id in self.data:
            flag = self.data[channel.id]
            for f in (flag & 1, flag & 4):
                if f:
                    proc = message.content
                    if proc:
                        while proc[0] == " ":
                            proc = proc[1:]
                        if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\") or proc.startswith("#"):
                            return
                        if proc.startswith("`") and proc.endswith("`"):
                            proc = proc.strip("`")
                        if not proc:
                            return
                        proc = proc.translate(self.qtrans)
                        output = None
                        try:
                            output = await self.procFunc(proc, channel, bot, term=f)
                            await channel.send(self.prepare_string(output, fmt=""))
                        except:
                            # print(traceback.format_exc())
                            await sendReact(channel, self.prepare_string(traceback.format_exc()), reacts="❎")
        elif message.guild is None:
            user = message.author
            emb = discord.Embed(colour=discord.Colour(16777214))
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=bestURL(user))
            emb.description = strMessage(message)
            for c_id, flag in self.data.items():
                if flag & 2:
                    channel = await self.bot.fetch_channel(c_id)
                    self.bot.embedSender(channel, embed=emb)

    async def _log_(self, msg, **void):
        if msg:
            msg = limStr(msg, 6000)
            if len(msg) > 2000:
                msgs = [msg[:2000], msg[2000:4000], msg[4000:]]
            else:
                msgs = [msg]
            embs = deque()
            for msg in msgs:
                if msg:
                    embs.append(discord.Embed(colour=discord.Colour(1), description=self.prepare_string(msg, lim=2048, fmt="")))
            for c_id, flag in self.data.items():
                if flag & 8:
                    channel = await self.bot.fetch_channel(c_id)
                    self.bot.embedSender(channel, embeds=embs)

    def __load__(self):
        print.funcs.append(lambda *args: create_task(self._log_(*args)))


class DownloadServer(Command):
    name = ["SaveServer", "ServerDownload"]
    min_level = nan
    description = "Downloads all posted messages in the target server into a sequence of .txt files."
    usage = "<server_id(curr)>"
    flags = "f"
    _timeout_ = 512
    
    async def __call__(self, bot, argv, flags, channel, guild, **void):
        if "f" not in flags:
            response = uniStr(
                "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
            )
            return ("```asciidoc\n[" + response + "]```")
        if argv:
            g_id = verifyID(argv)
            guild = await bot.fetch_guild(g_id)
        async with channel.typing():
            send = channel.send

            async def callback(channel, messages, **void):
                b = bytes()
                fn = str(channel) + " (" + str(channel.id) + ")"
                for i, message in enumerate(messages, 1):
                    temp = ("\n\n" + strMessage(message, username=True)).encode("utf-8")
                    if len(temp) + len(b) > 8388608:
                        await send(file=discord.File(io.BytesIO(b), filename=fn))
                        fn += "_"
                        b = temp[2:]
                    else:
                        if b:
                            b += temp
                        else:
                            b += temp[2:]
                    if not i & 8191:
                        await asyncio.sleep(0.2)
                if b:
                    await send(file=discord.File(io.BytesIO(b), filename=fn + ".txt"))

            await self.bot.database.counts.getGuildHistory(guild, callback=callback)
        response = uniStr("Download Complete.")
        return ("```ini\n[" + response + "]```")


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
                "```css\nCurrent blacklist status of [" + noHighlight(user.name) + "]: ["
                + noHighlight(susp) + "].```"
            )
        else:
            user = await bot.fetch_user(verifyID(args[0]))
            change = await bot.evalTime(" ".join(args[1]), guild.id, bot.data.blacklist.get(user.id, utc()))
            bot.data.blacklist[user.id] = change
            update()
            return (
                "```css\nChanged blacklist status of [" + noHighlight(user.name) + "] to ["
                + noHighlight(change) + "].```"
            )


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
                days = max(0, (udata - utc()) / 86400)
                try:
                    days **= 4
                except (OverflowError, ValueError, TypeError):
                    days = inf
                days += 1.125
                udata = utc() + days * 86400
                self.data[u_id] = udata
                if days >= self.bot.min_suspend - 1:
                    self.lastsusp = u_id
                self.update()
                self.update(True)
            print(self.lastsusp)
        except FileNotFoundError:
            pass

    async def _command_(self, user, command, **void):
        pass
        # if user.id not in (self.bot.client.user.id, self.bot.owner_id):
        #     tc = getattr(command, "time_consuming", 0)
        #     self.suspclear = utc() + 10 + (tc * 2) ** 2
        #     f = open(self.suspected, "w")
        #     f.write(str(user.id))
        #     f.close()

    async def __call__(self, **void):
        if utc() - self.suspclear:
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
            secs = self.data.get(u_susp.id, 0) - utc()
            msg = (
                "Apologies for the inconvenience, but your account has been "
                + "flagged as having attempted a denial-of-service attack.\n"
                + "This will expire in `" + sec2Time(secs) + "`.\n"
                + "If you believe this is an error, please notify <@"
                + str(tuple(self.owners)[0]) + "> as soon as possible."
            )
            print(
                u_susp.name + " may be attempting a DDOS attack. Expires in "
                + sec2Time(secs) + "."
            )
            await channel.send(msg)