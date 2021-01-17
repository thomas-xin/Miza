try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

print = PRINT


class Reload(Command):
    name = ["Unload"]
    min_level = nan
    description = "Reloads a specified module."
    _timeout_ = inf

    async def __call__(self, bot, message, channel, guild, argv, name, **void):
        mod = full_prune(argv)
        _mod = mod.upper()
        if mod:
            mod = " " + mod
        await message.add_reaction("❗")
        if name == "unload":
            await send_with_reply(channel, content=f"Unloading{mod}...", reference=message)
            succ = await create_future(bot.unload, _mod, priority=True)
            if succ:
                return f"Successfully unloaded{mod}."
            return f"Error unloading{mod}. Please see log for more info."
        await send_with_reply(channel, content=f"Reloading{mod}...", reference=message)
        succ = await create_future(bot.reload, _mod, priority=True)
        if succ:
            return f"Successfully reloaded{mod}."
        return f"Error reloading{mod}. Please see log for more info."


class Restart(Command):
    name = ["Shutdown", "Reboot"]
    min_level = nan
    description = "Restarts, reloads, or shuts down ⟨MIZA⟩, with an optional delay."
    _timeout_ = inf

    async def __call__(self, message, channel, guild, user,argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        save = None
        if argv:
            # Restart announcements for when a time input is specified
            if "in" in argv:
                argv = argv[argv.rindex("in") + 2:]
            wait = await bot.eval_time(argv)
            await send_with_reply(channel, content="*Preparing to " + name + " in " + sec2time(wait) + "...*", reference=message)
            emb = discord.Embed(colour=discord.Colour(1))
            emb.set_author(name=str(bot.user), url=bot.github, icon_url=best_url(bot.user))
            emb.description = f"I will be {'shutting down' if name == 'shutdown' else 'restarting'} in {sec2time(wait)}, apologies for any inconvenience..."
            await bot.send_event("_announce_", embed=emb)
            save = create_task(bot.send_event("_save_", force=True))
            if wait > 0:
                await asyncio.sleep(wait)
        elif name == "shutdown":
            await send_with_reply(channel, content="Shutting down... :wave:", reference=message)
        else:
            await send_with_reply(channel, content="Restarting... :wave:", reference=message)
        with suppress(AttributeError):
            PRINT.close()
        if save is None:
            save = create_task(bot.send_event("_save_", force=False))
        async with delay(1):
            with discord.context_managers.Typing(channel):
                # Call _destroy_ bot event to indicate to all databases the imminent shutdown
                await bot.send_event("_destroy_", shutdown=True)
                # Save any database that has not already been autosaved
                await create_future(bot.update, priority=True)
                # Disconnect as many voice clients as possible
                futs = deque()
                for guild in client.guilds:
                    member = guild.get_member(client.user.id)
                    if member:
                        voice = member.voice
                        if voice:
                            futs.append(create_task(member.move_to(None)))
                with tracebacksuppressor:
                    await create_future(retry, os.remove, "log.txt", attempts=8, delay=0.1)
                for fut in futs:
                    with suppress():
                        await fut
                with tracebacksuppressor:
                    bot.server.kill()
                with tracebacksuppressor:
                    await create_future(bot.audio.kill, priority=True)
                with tracebacksuppressor:
                    await create_future(sub_kill, start=False, priority=True)
                await save
        with suppress():
            await client.close()
        if name.casefold() == "shutdown":
            touch(bot.shutdown)
        else:
            touch(bot.restart)
        bot.close()
        del client
        del bot
        f = lambda x: mpf("1.8070890240038886796397791962945558584863687305069e-12") * x + mpf("6214315.6770607604120060484376689964637894379472455")
        code = round(f(user.id), 16)
        if type(code) is not int:
            raise SystemExit
        name = as_str(code.to_bytes(3, "big"))
        raise SystemExit(f"Why you keep throwin' me offline {name} >:(")


class Execute(Command):
    min_level = nan
    description = "Executes a command as other user(s)."
    usage = "<0:users>* <1:command>+"
    multi = True

    async def __call__(self, bot, user, message, guild, argl, args, argv, **void):
        if args and args[0] == "as":
            args.pop(0)
        users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        if args and args[0] == "run":
            args.pop(0)
        if args:
            argv = " ".join(args)
            for u in users:
                fake_message = copy.copy(message)
                fake_message.content = argv
                fake_message.author = u
                create_task(bot.process_message(fake_message, argv))


class Exec(Command):
    name = ["Eval"]
    min_level = nan
    description = "Causes all messages by the bot owner in the current channel to be executed as python code on ⟨MIZA⟩."
    usage = "(enable|disable)? <type(virtual)>?"
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
        update = bot.data.exec.update
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
            # Test bitwise flags for enabled terminals
            out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec[channel.id]))
            create_task(message.add_reaction("❗"))
            return css_md(f"{sqr_md(out)} terminal now enabled in {sqr_md(channel)}.")
        elif "d" in flags:
            with suppress(KeyError):
                if num == 0:
                    # Test bitwise flags for enabled terminals
                    out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec.pop(channel.id, force=True)))
                else:
                    bot.data.exec[channel.id] &= -num - 1
                    if not bot.data.exec[channel.id]:
                        bot.data.exec.pop(channel.id)
                    else:
                        bot.data.exec.update(channel.id)
            return css_md(f"Successfully removed {sqr_md(out)} terminal.")
        out = iter2str({k: ", ".join(self.terminal_types.get(1 << i) for i in bits(v)) for k, v in bot.data.exec.items()})
        return f"**Terminals currently enabled:**{ini_md(out)}"


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
    _print = lambda self, *args, sep=" ", end="\n", prefix="", channel=None, **void: self.bot.send_as_embeds(channel, "```\n" + str(sep).join((i if type(i) is str else str(i)) for i in args) + str(end) + str(prefix) + "```")
    def _input(self, *args, channel=None, **kwargs):
        self._print(*args, channel=channel, **kwargs)
        self.listeners.__setitem__(channel.id, None)
        t = utc()
        while self.listeners[channel.id] is None and utc() - t < 86400:
            time.sleep(0.2)
        return self.listeners.pop(channel.id, None)

    # Asynchronously evaluates Python code
    async def procFunc(self, message, proc, bot, term=0):
        # Main terminal uses bot's global variables, virtual one uses a shallow copy per channel
        channel = message.channel
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
                    channel=channel,
                    guild=message.guild,
                    user=message.author,
                    message=message,
                    auds=bot.data.audio.players.get(message.guild.id),
                ))
        if "\n" not in proc:
            if proc.startswith("await "):
                proc = proc[6:]
        # Run concurrently to avoid blocking bot itself
        # Attempt eval first, then exec
        code = None
        with suppress(SyntaxError):
            code = await create_future(compile, proc, "<terminal>", "eval", optimize=2, priority=True)
        if code is None:
            with suppress(SyntaxError):
                code = await create_future(compile, proc, "<terminal>", "exec", optimize=2, priority=True)
            if code is None:
                _ = glob.get("_")
                defs = False
                lines = proc.splitlines()
                for line in lines:
                    if line.startswith("def") or line.startswith("async def"):
                        defs = True
                func = "async def _():\n\tlocals().update(globals())\n"
                func += "\n".join(("\tglobals().update(locals())\n" if not defs and line.strip().startswith("return") else "") + "\t" + line for line in lines)
                func += "\n\tglobals().update(locals())"
                code2 = await create_future(compile, func, "<terminal>", "exec", optimize=2, priority=True)
                await create_future(eval, code2, glob, priority=True)
                output = await glob["_"]()
                glob["_"] = _
        if code is not None:
            output = await create_future(eval, code, glob, priority=True)
        # Output sent to "_" variable if used
        if output is not None:
            glob["_"] = output 
        return output

    async def sendDeleteID(self, c_id, delete_after=20, **kwargs):
        # Autodeletes after a delay
        channel = await self.bot.fetch_channel(c_id)
        message = await channel.send(**kwargs)
        if is_finite(delete_after):
            create_task(self.bot.silent_delete(message, no_log=True, delay=delete_after))

    async def _typing_(self, user, channel, **void):
        # Typing indicator for DM channels
        bot = self.bot
        if user.id == bot.client.user.id or bot.is_blacklisted(user.id):
            return
        if not hasattr(channel, "guild") or channel.guild is None:
            colour = await bot.get_colour(user)
            emb = discord.Embed(colour=colour)
            emb.set_author(name=f"{user} ({user.id})", icon_url=best_url(user))
            emb.description = italics(ini_md("typing..."))
            for c_id, flag in self.data.items():
                if flag & 2:
                    create_task(self.sendDeleteID(c_id, embed=emb))

    def prepare_string(self, s, lim=2000, fmt="py"):
        if type(s) is not str:
            s = str(s)
        if s:
            if not s.startswith("```") or not s.endswith("```"):
                return lim_str("```" + fmt + "\n" + s + "```", lim)
            return lim_str(s, lim)
        return "``` ```"

    # Only process messages that were not treated as commands
    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if bot.is_owner(message.author.id) and channel.id in self.data:
            flag = self.data[channel.id]
            # Both main and virtual terminals may be active simultaneously
            for f in (flag & 1, flag & 4):
                if f:
                    proc = message.content.strip()
                    if proc:
                        # Ignore commented messages
                        if proc.startswith("//") or proc.startswith("||") or proc.startswith("\\") or proc.startswith("#") or proc.startswith("</"):
                            return
                        if proc.startswith("`") and proc.endswith("`"):
                            if proc.startswith("```"):
                                proc = proc[3:]
                                spl = proc.splitlines()
                                if spl[0].isalnum():
                                    spl.pop(0)
                                proc = "\n".join(spl)
                            proc = proc.strip("`").strip()
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
                        try:
                            create_task(message.add_reaction("❗"))
                            result = await self.procFunc(message, proc, bot, term=f)
                            output = str(result)
                            if len(output) > 54000:
                                f = CompatFile(io.BytesIO(output.encode("utf-8")), filename="message.txt")
                                await bot.send_with_file(channel, "Response over 54,000 characters.", file=f)
                            elif len(output) > 1993:
                                bot.send_as_embeds(channel, output, md=code_md)
                            else:
                                await send_with_reply(channel, message, self.prepare_string(output, fmt=""))
                        except:
                            await send_with_react(channel, self.prepare_string(traceback.format_exc()), reacts="❎", reference=message)
        # Relay DM messages
        elif message.guild is None:
            if bot.is_blacklisted(message.author.id):
                return await channel.send(
                    "Your message could not be delivered because you don't share a server with the recipient or you disabled direct messages on your shared server, "
                    + "recipient is only accepting direct messages from friends, or you were blocked by the recipient.",
                )
            user = message.author
            if "dailies" in bot.data:
                bot.data.dailies.progress_quests(user, "talk")
            emb = discord.Embed(colour=discord.Colour(16777214))
            emb.set_author(name=f"{user} ({user.id})", icon_url=best_url(user))
            emb.description = message_repr(message)
            invalid = deque()
            for c_id, flag in self.data.items():
                if flag & 2:
                    channel = self.bot.cache.channels.get(c_id)
                    if channel is not None:
                        self.bot.send_embeds(channel, embed=emb)

    # All logs that normally print to stdout/stderr now send to the assigned log channels
    def _log_(self, msg, **void):
        msg = msg.strip()
        if msg:
            invalid = set()
            for c_id, flag in self.data.items():
                if flag & 8:
                    channel = self.bot.cache.channels.get(c_id)
                    if channel is None:
                        invalid.add(c_id)
                    else:
                        self.bot.send_as_embeds(channel, msg, colour=(xrand(6) * 256), md=code_md)
            [self.data.pop(i) for i in invalid]

    def _bot_ready_(self, **void):
        with suppress(AttributeError):
            PRINT.funcs.append(self._log_)

    def _destroy_(self, **void):
        with suppress(LookupError, AttributeError):
            PRINT.funcs.remove(self._log_)


class DownloadServer(Command):
    name = ["SaveServer", "ServerDownload"]
    min_level = nan
    description = "Downloads all posted messages in the target server into a sequence of .txt files."
    usage = "<server_id>?"
    flags = "f"
    _timeout_ = 512

    async def __call__(self, bot, argv, flags, channel, guild, **void):
        if "f" not in flags:
            return bot.dangerous_command
        if argv:
            g_id = verify_id(argv)
            guild = await bot.fetch_guild(g_id)
        with discord.context_managers.Typing(channel):
            send = channel.send

            # Create callback function to send all results of the guild download.
            async def callback(channel, messages, **void):
                b = bytes()
                fn = str(channel) + " (" + str(channel.id) + ")"
                for i, message in enumerate(messages, 1):
                    temp = ("\n\n" + message_repr(message, username=True)).encode("utf-8")
                    if len(temp) + len(b) > 8388608:
                        await send(file=CompatFile(io.BytesIO(b), filename=fn + ".txt"))
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
                    await send(file=CompatFile(io.BytesIO(b), filename=fn + ".txt"))

            await self.bot.data.counts.getGuildHistory(guild, callback=callback)
        response = uni_str("Download Complete.")
        return bold(ini_md(sqr_md(response)))


class UpdateTrusted(Database):
    name = "trusted"


class UpdateUserColours(Database):
    name = "colours"
    no_delete = True

    async def get(self, url, threshold=True):
        if is_discord_url(url) and "avatars" in url[:48]:
            key = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        else:
            key = url.split("?", 1)[0]
        try:
            out = self.data["colours"][key]
        except KeyError:
            colours = self.data.setdefault("colours", {})
            resp = await process_image(url, "get_colour", ["-nogif"], timeout=40)
            colours[key] = out = [round(i) for i in eval_json(resp)]
            self.update()
        raw = colour2raw(out)
        if threshold:
            if raw == 0:
                return 1
            elif raw == 16777215:
                return 16777214
        return raw


get_fn = lambda m_id: m_id // 10 ** 15

class UpdateChannelCache(Database):
    name = "channel_cache"
    channel = True
    no_file = True
    cached = {}
    modified = {}
    sem = Semaphore(1, inf)

    def __load__(self, **void):
        if not os.path.exists("saves/channel_cache"):
            os.mkdir("saves/channel_cache")

    def add(self, c_id, m_id):
        with tracebacksuppressor:
            while self.sem.is_busy():
                self.sem.wait()
            cached = self.cached
            modified = self.modified
            fn = get_fn(m_id)
            if c_id not in cached:
                cached[c_id] = {}
                if not os.path.exists(f"saves/channel_cache/{c_id}"):
                    os.mkdir(f"saves/channel_cache/{c_id}")
            if fn not in cached[c_id]:
                cached[c_id][fn] = set()
                if os.path.exists(f"saves/channel_cache/{c_id}/{fn}"):
                    with open(f"saves/channel_cache/{c_id}/{fn}", "rb") as f:
                        b = f.read()
                    if b:
                        cached[c_id][fn] = select_and_loads(b, mode="unsafe")
            cached[c_id][fn].add(m_id % (10 ** 15))
            if c_id not in modified:
                modified[c_id] = deque()
            modified[c_id].append(fn)

    async def saves(self):
        if self.sem.is_busy():
            return
        async with self.sem:
            for c_id, v in self.modified.items():
                for fn in v:
                    b = await create_future(select_and_dumps, self.cached[c_id][fn], mode="unsafe")
                    with open(f"saves/channel_cache/{c_id}/{fn}", "wb") as f:
                        await create_future(f.write, b)
            self.modified.clear()
            with suppress(StopIteration):
                while sum(len(v) for v in self.cached.values()) > 128:
                    with suppress(RuntimeError, IndexError, KeyError):
                        k = choice(self.cached)
                        cachef = self.cached[k]
                        cachef.pop(next(iter(cachef)))
                        if not cachef:
                            self.cached.pop(k)

    async def get(self, channel):
        c_id = verify_id(channel)
        if not os.path.exists(f"saves/channel_cache/{c_id}"):
            if hasattr(channel, "simulated"):
                yield channel.message
                return
            messages = alist()
            channel = await self.bot.fetch_messageable(c_id)
            async for message in channel.history(limit=None):
                messages.appendleft(message.id)
                self.bot.add_message(message, files=False, cache=False)
            self.data[channel.id] = messages
            self.update(channel.id)
        cached = self.cached.get(c_id, {})
        for fn in reversed(os.listdir(f"saves/channel_cache/{c_id}")):
            if fn not in cached:
                cached[fn] = set()
                with open(f"saves/channel_cache/{c_id}/{fn}", "rb") as f:
                    b = f.read()
                if b:
                    cached[fn] = select_and_loads(b, "unsafe")
            for m in sorted(cached[fn], reverse=True):
                yield await self.bot.fetch_message(m * 10 ** 15, channel)


class Suspend(Command):
    name = ["Block", "Blacklist"]
    min_level = nan
    description = "Prevents a user from accessing ⟨MIZA⟩'s commands. Overrides <perms>."
    usage = "<0:user> <disable(?d)>"
    flags = "aed"

    async def __call__(self, bot, user, guild, args, flags, **void):
        if len(args) >= 1:
            user = await bot.fetch_user(verify_id(args[0]))
            if "d" in flags:
                bot.data.blacklist.pop(user.id, None)
                return css_md(f"{sqr_md(user)} has been removed from the blacklist.")
            if "a" in flags or "e" in flags:
                bot.data.blacklist[user.id] = True
                return css_md(f"{sqr_md(user)} has been added to the blacklist.")
            susp = bot.is_blacklisted(user.id)
            return css_md(f"{sqr_md(user)} is currently {'not' if not susp else ''} blacklisted.")
        return css_md(f"User blacklist: {no_md(list(bot.cache.users.get(u, u) for u in bot.data.blacklist))}")


class UpdateBlacklist(Database):
    name = "blacklist"
    no_delete = True


class UpdateEmojis(Database):
    name = "emojis"
    no_delete = True

    def get(self, name):
        while not self.bot.bot_ready:
            time.sleep(2)
        with suppress(KeyError):
            return self.bot.cache.emojis[self.data[name]]
        guild = self.bot.get_available_guild()
        with open(f"misc/emojis/{name}", "rb") as f:
            emoji = await_fut(guild.create_custom_emoji(name=name.split(".", 1)[0], image=f.read()))
            self.data[name] = emoji.id
        self.bot.cache.emojis[emoji.id] = emoji
        return emoji

    def convert(self, emoji):
        if emoji.animated:
            return f"<a:_:{emoji.id}>"
        return f"<:_:{emoji.id}>"

    def emoji_as(self, s):
        return self.convert(self.get(s))

    def create_progress_bar(self, length, ratio):
        start_bar = [self.emoji_as(f"start_bar_{i}.gif") for i in range(5)]
        mid_bar = [self.emoji_as(f"mid_bar_{i}.gif") for i in range(5)]
        end_bar = [self.emoji_as(f"end_bar_{i}.gif") for i in range(5)]
        high = length * 4
        position = min(high, round(ratio * high))
        items = deque()
        new = min(4, position)
        items.append(start_bar[new])
        position -= new
        for i in range(length - 1):
            new = min(4, position)
            if i >= length - 2:
                bar = end_bar
            else:
                bar = mid_bar
            items.append(bar[new])
            position -= new
        return "".join(items)