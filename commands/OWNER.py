try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Restart(Command):
    name = ["Shutdown", "Reload", "Unload", "Reboot"]
    min_level = nan
    description = "Restarts, reloads, or shuts down ⟨MIZA⟩, with an optional delay."
    _timeout_ = inf

    async def __call__(self, message, channel, guild, argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        if name == "reload":
            argv = argv.upper()
            s = " " + argv if argv else ""
            await channel.send("Reloading" + s.lower() + "...")
            await create_future(bot.reload, argv, priority=True)
            return "Successfully reloaded" + s.lower() + "."
        if name == "unload":
            argv = argv.upper()
            s = " " + argv if argv else ""
            await channel.send("Unloading" + s.lower() + "...")
            await create_future(bot.unload, argv, priority=True)
            return "Successfully unloaded" + s.lower() + "."
        if argv:
            # Restart announcements for when a time input is specified
            if "in" in argv:
                argv = argv[argv.rindex("in") + 2:]
            delay = await bot.evalTime(argv, user)
            await channel.send("*Preparing to " + name + " in " + sec2Time(delay) + "...*")
            emb = discord.Embed(colour=discord.Colour(1))
            emb.set_author(name=str(client.user), url=bot.website, icon_url=bestURL(client.user))
            emb.description = "```ini\n[I will be " + ("restarting", "shutting down")[name == "shutdown"] + " in " + sec2Time(delay) + ", apologies for any inconvenience.]```"
            await bot.event("_announce_", embed=emb)
            if delay > 0:
                await asyncio.sleep(delay)
        elif name == "shutdown":
            await channel.send("Shutting down... :wave:")
        else:
            await channel.send("Restarting... :wave:")
        with discord.context_managers.Typing(channel):
            with suppress(AttributeError):
                PRINT.close()
            t = time.time()
            # Call _destroy_ bot event to indicate to all databases the imminent shutdown
            await bot.event("_destroy_")
            # Save any database that has not already been autosaved
            await create_future(bot.update, priority=True)
            # Disconnect as many voice clients as possible
            for vc in client.voice_clients:
                await vc.disconnect(force=True)
        with suppress():
            await client.close()
        if name.casefold() == "shutdown":
            with open(bot.shutdown, "wb"):
                pass
        else:
            for _ in loop(5):
                try:
                    with open(bot.restart, "wb"):
                        pass
                    break
                except:
                    print_exc()
                    time.sleep(0.1)
        for _ in loop(8):
            try:
                if os.path.exists("log.txt"):
                    os.remove("log.txt")
                break
            except:
                print_exc()
                time.sleep(0.1)
        if time.time() - t < 1:
            await asyncio.sleep(1)
        bot.close()
        del client
        del bot
        sys.exit()


class Execute(Command):
    name = ["Exec", "Eval"]
    min_level = nan
    description = "Causes all messages by the bot owner in the current channel to be executed as python code on ⟨MIZA⟩."
    usage = "<type> <enable(?e)> <disable(?d)>"
    flags = "aed"
    # Different types of terminals for different purposes
    terminal_types = demap({
        "null": 0,
        "main": 1,
        "relay": 2,
        "virtual": 4,
        "log": 8,
    })

    def __call__(self, bot, flags, argv, message, channel, guild, **void):
        update = bot.database.exec.update
        if not argv:
            argv = 0
        try:
            num = int(argv)
        except (TypeError, ValueError):
            out = argv.casefold()
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
            # Test bitwise flags for enabled terminals
            out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec[channel.id]))
            create_task(message.add_reaction("❗"))
            return (
                "```css\n[" + out + "] terminal now enabled in [#"
                + noHighlight(channel.name) + "].```"
            )
        elif "d" in flags:
            with suppress(KeyError):
                if num == 0:
                    # Test bitwise flags for enabled terminals
                    out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec.pop(channel.id)))
                else:
                    bot.data.exec[channel.id] &= -num - 1
                    if not bot.data.exec[channel.id]:
                        bot.data.exec.pop(channel.id)
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
    virtuals = cdict()
    listeners = cdict()

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

    # Custom print function to send a message instead
    _print = lambda self, *args, sep=" ", end="\n", prefix="", channel=None, **void: self.bot.embedSender(channel, embed=discord.Embed(colour=discord.Colour(1), description=limStr("```\n" + str(sep).join((i if type(i) is str else str(i)) for i in args) + str(end) + str(prefix) + "```", 2048)))
    def _input(self, *args, channel=None, **kwargs):
        self._print(*args, channel=channel, **kwargs)
        self.listeners.__setitem__(channel.id, None)
        t = utc()
        while self.listeners[channel.id] is None and utc() - t < 86400:
            time.sleep(0.2)
        return self.listeners.pop(channel.id, None)

    # Asynchronously evaluates Python code
    async def procFunc(self, proc, channel, bot, term=0):
        # Main terminal uses bot's global variables, virtual one uses a shallow copy per channel
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
        if "\n" not in proc:
            if proc.startswith("await "):
                proc = proc[6:]
        # Run concurrently to avoid blocking bot itself
        # Attempt eval first, then exec
        try:
            code = await create_future(compile, proc, "<terminal>", "eval", optimize=2, priority=True)
        except SyntaxError:
            code = await create_future(compile, proc, "<terminal>", "exec", optimize=2, priority=True)
        output = await create_future(eval, code, glob, priority=True)
        # Output sent to "_" variable if used
        if output is not None:
            glob["_"] = output 
        return output

    async def sendDeleteID(self, c_id, delete_after=20, **kwargs):
        # Autodeletes after a delay
        channel = await self.bot.fetch_channel(c_id)
        message = await channel.send(**kwargs)
        if isValid(delete_after):
            create_task(self.bot.silentDelete(message, no_log=True, delay=delete_after))

    async def _typing_(self, user, channel, **void):
        # Typing indicator for DM channels
        bot = self.bot
        if user.id == bot.client.user.id:
            return
        if not hasattr(channel, "guild") or channel.guild is None:
            emb = discord.Embed(colour=randColour())
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=bestURL(user))
            emb.description = "*```ini\n[typing...]```*"
            for c_id, flag in self.data.items():
                if flag & 2:
                    create_task(self.sendDeleteID(c_id, embed=emb))

    def prepare_string(self, s, lim=2000, fmt="py"):
        if type(s) is not str:
            s = str(s)
        if s:
            return limStr("```" + fmt + "\n" + s + "```", lim)
        return "``` ```"

    # Only process messages that were not treated as commands
    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if message.author.id in self.bot.owners and channel.id in self.data:
            flag = self.data[channel.id]
            # Both main and virtual terminals may be active simultaneously
            for f in (flag & 1, flag & 4):
                if f:
                    proc = message.content.strip()
                    if proc:
                        # Ignore commented messages
                        if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\") or proc.startswith("#"):
                            return
                        if proc.startswith("`") and proc.endswith("`"):
                            proc = proc.strip("`")
                        if not proc:
                            return
                        with suppress(KeyError):
                            # Write to input() listener if required
                            if self.listeners[channel.id] is None:
                                create_task(message.add_reaction("👀"))
                                self.listeners[channel.id] = proc
                                return
                        if not proc:
                            return
                        proc = proc.translate(self.qtrans)
                        output = None
                        try:
                            create_task(message.add_reaction("❗"))
                            output = await self.procFunc(proc, channel, bot, term=f)
                            await channel.send(self.prepare_string(output, fmt=""))
                        except:
                            # print_exc()
                            await sendReact(channel, self.prepare_string(traceback.format_exc()), reacts="❎")
        # Relay DM messages
        elif message.guild is None:
            if bot.isBlacklisted(message.author.id):
                return await sendReact(channel,
                    "Your message could not be delivered because you don't share a server with the recipient or you disabled direct messages on your shared server, "
                    + "recipient is only accepting direct messages from friends, or you were blocked by the recipient.",
                    reacts="❎"
                )
            user = message.author
            emb = discord.Embed(colour=discord.Colour(16777214))
            emb.set_author(name=str(user) + " (" + str(user.id) + ")", icon_url=bestURL(user))
            emb.description = strMessage(message)
            invalid = deque()
            for c_id, flag in self.data.items():
                if flag & 2:
                    channel = self.bot.cache.channels.get(c_id)
                    if channel is not None:
                        self.bot.embedSender(channel, embed=emb)

    # All logs that normally print to stdout/stderr now send to the assigned log channels
    def _log_(self, msg, **void):
        while not self.bot.ready:
            time.sleep(2)
        if msg:
            if len(msg) > 2041:
                if "\n" in msg:
                    msgs = deque()
                    new = ""
                    for line in msg.split("\n"):
                        if new:
                            line = "\n" + line
                        if len(new) + len(line) > 2048:
                            msgs.append(new)
                            new = line
                        else:
                            new += line
                    if new:
                        msgs.append(new)
                else:
                    msg = limStr(msg, 6000)
                    msgs = [msg[:2000], msg[2000:4000], msg[4000:]]
            else:
                msgs = [msg]
            embs = deque()
            for msg in msgs:
                if msg:
                    embs.append(discord.Embed(colour=discord.Colour(16711680), description=self.prepare_string(msg, lim=2048, fmt="")))
            invalid = deque()
            for c_id, flag in self.data.items():
                if flag & 8:
                    channel = self.bot.cache.channels.get(c_id)
                    if channel is None:
                        invalid.append(c_id)
                    else:
                        self.bot.embedSender(channel, embeds=embs)
            [self.data.pop(i) for i in invalid]                        

    def __load__(self):
        with suppress(AttributeError):
            PRINT.funcs.append(lambda *args: self._log_(*args))


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
            return ("**```asciidoc\n[" + response + "]```**")
        if argv:
            g_id = verifyID(argv)
            guild = await bot.fetch_guild(g_id)
        with discord.context_managers.Typing(channel):
            send = channel.send

            # Create callback function to send all results of the guild download.
            async def callback(channel, messages, **void):
                b = bytes()
                fn = str(channel) + " (" + str(channel.id) + ")"
                for i, message in enumerate(messages, 1):
                    temp = ("\n\n" + strMessage(message, username=True)).encode("utf-8")
                    if len(temp) + len(b) > 8388608:
                        await send(file=discord.File(io.BytesIO(b), filename=fn + ".txt"))
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
        return ("**```ini\n[" + response + "]```**")


class Trust(Command):
    name = ["Untrust"]
    min_level = nan
    description = "Adds or removes a server from the bot's trusted server list."
    usage = "<server_id(curr)(?a)> <enable(?e)> <disable(?d)>"
    flags = "aed"

    def __call__(self, bot, flags, message, guild, argv, **void):
        update = bot.database.trusted.update
        if "a" in flags:
            guilds = bot.client.guilds
        else:
            if argv:
                g_id = verifyID(argv)
                guild = cdict(id=g_id)
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
                bot.data.trusted.pop(guild.id, None)
            update()
            return (
                "```fix\nSuccessfully removed server from trusted list.```"
            )
        return (
            "```css\nTrusted server list "
            + str(list(noHighlight(bot.cache.guilds.get(g, g)) for g in bot.data.trusted)) + ".```"
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
            user = await bot.fetch_user(verifyID(args.pop(0)))
            new = await bot.evalMath(" ".join(args), user.id, bot.data.blacklist.get(user.id, 0))
            bot.data.blacklist[user.id] = new
            update()
            return (
                "```css\nChanged blacklist status of [" + noHighlight(user.name) + "] to ["
                + noHighlight(new) + "].```"
            )
        return (
            "```css\nUser blacklist "
            + str(list(noHighlight(bot.cache.users.get(u, u)) for u in bot.data.blacklist)) + ".```"
        )


class UpdateBlacklist(Database):
    name = "blacklist"
    user = True