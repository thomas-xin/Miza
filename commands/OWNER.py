print = PRINT


class Reload(Command):
    name = ["Unload"]
    min_level = nan
    description = "Reloads a specified module."
    example = ("reload admin", "unload string")
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
    name = ["Shutdown", "Reboot", "Update"]
    min_level = nan
    description = "Restarts, reloads, or shuts down ⟨MIZA⟩, with an optional delay."
    example = ("shutdown", "update", "restart")
    usage = "<delay>?"
    _timeout_ = inf

    async def __call__(self, message, channel, guild, user, argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        save = None
        if name == "update":
            resp = await create_future(subprocess.run, ["git", "pull"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(resp.stdout)
            print(resp.stderr)
        if argv == "when free":
            busy = True
            while busy:
                busy = False
                for vc in bot.audio.clients.values():
                    try:
                        if vc.is_playing():
                            busy = True
                    except:
                        print_exc()
                await asyncio.sleep(1)
        elif argv:
            # Restart announcements for when a time input is specified
            if argv.startswith("in"):
                argv = argv[2:].lstrip()
                wait = await bot.eval_time(argv)
                await send_with_reply(channel, content="*Preparing to " + name + " in " + sec2time(wait) + "...*", reference=message)
                emb = discord.Embed(colour=discord.Colour(1))
                url = await bot.get_proxy_url(bot.user)
                emb.set_author(name=str(bot.user), url=bot.github, icon_url=url)
                emb.description = f"I will be {'shutting down' if name == 'shutdown' else 'restarting'} in {sec2time(wait)}, apologies for any inconvenience..."
                await bot.send_event("_announce_", embed=emb)
                save = create_task(bot.send_event("_save_"))
                if wait > 0:
                    await asyncio.sleep(wait)
        if name == "shutdown":
            await send_with_reply(channel, content="Shutting down... :wave:", reference=message)
        else:
            await send_with_reply(channel, content="Restarting... :wave:", reference=message)
        if save is None:
            print("Saving message cache...")
            save = create_task(bot.send_event("_save_"))
        bot.closed = True
        async with Delay(1):
            with discord.context_managers.Typing(channel):
                # Call _destroy_ bot event to indicate to all databases the imminent shutdown
                print("Destroying database memory...")
                await bot.send_event("_destroy_", shutdown=True)
                # Kill the audio player client
                print("Shutting down audio client...")
                kill = create_future(bot.audio.kill, timeout=16, priority=True)
                # Save any database that has not already been autosaved
                print("Saving all databases...")
                await create_future(bot.update, force=True, priority=True)
                # Send the bot "offline"
                print("Going offline...")
                with tracebacksuppressor:
                    await asyncio.wait_for(bot.change_presence(status=discord.Status.invisible), timeout=3)
                # Kill math and image subprocesses
                print("Killing math and image subprocesses...")
                with tracebacksuppressor:
                    await create_future(sub_kill, start=False, timeout=8, priority=True)
                # Kill the webserver
                print("Killing webserver...")
                with tracebacksuppressor:
                    await create_future(force_kill, bot.server, priority=True)
                # Disconnect as many voice clients as possible
                print("Disconnecting remaining voice clients...")
                futs = deque()
                for guild in client.guilds:
                    member = guild.get_member(client.user.id)
                    if member:
                        voice = member.voice
                        if voice:
                            futs.append(create_task(member.move_to(None)))
                print("Goodbye.")
                with suppress(NameError, AttributeError):
                    PRINT.flush()
                    PRINT.close(force=True)
                with tracebacksuppressor:
                    await create_future(retry, os.remove, "log.txt", attempts=8, delay=0.1)
                for fut in futs:
                    with suppress():
                        await fut
                await kill
                await save
        with suppress():
            await bot.close()
        if name.casefold() == "shutdown":
            touch(bot.shutdown)
        else:
            touch(bot.restart)
        # bot.close()
        del client
        del bot
        f = lambda x: mpf("1.8070890240038886796397791962945558584863687305069e-12") * x + mpf("6214315.6770607604120060484376689964637894379472455")
        code = round(f(user.id), 16)
        if type(code) is not int:
            raise SystemExit
        name = as_str(code.to_bytes(3, "little"))
        raise SystemExit(f"Why you keep throwin' me offline {name} >:(")


class Execute(Command):
    min_level = nan
    description = "Executes a command as other user(s), similar to the command's function in Minecraft."
    usage = "as <0:users>* run <1:command>+"
    example = ("execute as @Miza run ~info",)
    multi = True

    async def __call__(self, bot, user, message, channel, guild, argl, args, argv, **void):
        env = (user, channel)
        envs = [env]
        while args:
            if args[0] == "as":
                args.pop(0)
                al = args.pop(0).split()
                users = await bot.find_users(al, al, user, guild)
                if not users:
                    raise LookupError("No results found.")
                temp = []
                for env in envs:
                    temp.extend((u, env[1]) for u in users)
                envs = temp
            elif args[0] == "at":
                args.pop(0)
                al = args.pop(0).split()
                users = await bot.find_users(al, al, user, guild)
                if not users:
                    raise LookupError("No results found.")
                channels = []
                for u in users:
                    cid = bot.data.users.get(u.id, {}).get("last_channel")
                    try:
                        if not cid:
                            raise
                        c = await bot.fetch_channel(cid)
                    except:
                        m = bot.get_member(u.id, guild, find_others=True)
                        if hasattr(m, "guild"):
                            c = bot.get_first_sendable(m.guild, m)
                            channels.append(c)
                    else:
                        channels.append(c)
                temp = []
                for env in envs:
                    temp.extend((env[0], c) for c in channels)
                envs = temp
            elif args[0] == "in":
                args.pop(0)
                al = args.pop(0).split()
                channels = []
                for i in al:
                    c = await bot.fetch_channel(verify_id(i))
                    channels.append(c)
                temp = []
                for env in envs:
                    temp.extend((env[0], c) for c in channels)
                envs = temp
            else:
                break
        if not args:
            return
        try:
            argv = message.content.split("run ", 1)[1]
        except IndexError:
            pass
            # raise ArgumentError('"run" must be specified as a separator.')
        print(envs, argv)
        futs = deque()
        for u, c in envs:
            fake_message = copy.copy(message)
            fake_message.content = argv
            fake_message.channel = c
            g = getattr(c, "guild", None)
            fake_message.guild = g
            if g:
                fake_message.author = g.get_member(u.id) or u
            else:
                fake_message.author = u
            futs.append(create_task(bot.process_message(fake_message, argv)))
        for fut in futs:
            await fut


class Exec(Command):
    name = ["Eval"]
    min_level = nan
    description = "Causes all messages by the bot owner(s) in the current channel to be executed as python code on ⟨MIZA⟩."
    usage = "(enable|disable)? <type(virtual)>?"
    example = ("exec enable", "exec ?d")
    flags = "aed"
    # Different types of terminals for different purposes
    terminal_types = demap(dict(
        null=0,
        main=1,
        relay=2,
        virtual=4,
        log=8,
        proxy=16,
        shell=32,
        chat=64,
    ))

    def __call__(self, bot, flags, argv, message, channel, guild, **void):
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
                    out = ", ".join(self.terminal_types.get(1 << i) for i in bits(bot.data.exec.pop(channel.id, 0, force=True)))
                else:
                    bot.data.exec[channel.id] &= -num - 1
                    if not bot.data.exec[channel.id]:
                        bot.data.exec.pop(channel.id)
            return css_md(f"Successfully removed {sqr_md(out)} terminal.")
        out = iter2str({k: ", ".join(self.terminal_types.get(1 << i) for i in bits(v)) for k, v in bot.data.exec.items()})
        return f"**Terminals currently enabled:**{ini_md(out)}"


class UpdateExec(Database):
    name = "exec"
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
    temp = {}

    # Custom print function to send a message instead
    _print = lambda self, *args, sep=" ", end="\n", prefix="", channel=None, **void: self.bot.send_as_embeds(channel, "```\n" + str(sep).join((i if type(i) is str else str(i)) for i in args) + str(end) + str(prefix) + "```")
    def _input(self, *args, channel=None, **kwargs):
        self._print(*args, channel=channel, **kwargs)
        self.listeners[channel.id] = fut = concurent.futures.Future()
        return fut.result(timeout=86400)

    # Asynchronously evaluates Python code
    async def procFunc(self, message, proc, bot, term=0):
        proc = as_str(proc)
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
                ))
            glob.update(dict(
                user=message.author,
                member=message.author,
                message=message,
            ))
            with suppress():
                glob["auds"] = bot.data.audio.players[message.guild.id]
            if term & 32:
                proc = await asyncio.create_subprocess_shell(proc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, limit=65536)
                out = await proc.stdout.read()
                err = await proc.stderr.read()
                output = (as_str(out) + "\n" + as_str(err)).strip()
                if output:
                    glob["_"] = output
                return output
        if "\n" not in proc:
            if proc.startswith("await "):
                proc = proc[6:]
        # Run concurrently to avoid blocking bot itself
        # Attempt eval first, then exec
        code = None
        with suppress(SyntaxError):
            code = compile(proc, "<terminal>", "eval", optimize=2)
        if code is None:
            with suppress(SyntaxError):
                code = compile(proc, "<terminal>", "exec", optimize=2)
            if code is None:
                _ = glob.get("_")
                defs = False
                lines = proc.splitlines()
                for line in lines:
                    if line.startswith("def ") or line.startswith("async def "):
                        defs = True
                func = "async def _():\n\tlocals().update(globals())\n"
                func += "\n".join(("\tglobals().update(locals())\n" if not defs and line.strip().startswith("return") else "") + "\t" + line for line in lines)
                func += "\n\tglobals().update(locals())"
                code2 = compile(func, "<terminal>", "exec", optimize=2)
                eval(code2, glob)
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

    # async def _typing_(self, user, channel, **void):
    #     # Typing indicator for DM channels
    #     bot = self.bot
    #     if user.id == bot.client.user.id or bot.is_blacklisted(user.id):
    #         return
    #     if not hasattr(channel, "guild") or channel.guild is None:
    #         colour = await bot.get_colour(user)
    #         emb = discord.Embed(colour=colour)
    #         url = await bot.get_proxy_url(user)
    #         emb.set_author(name=f"{user} ({user.id})", icon_url=url)
    #         emb.description = italics(ini_md("typing..."))
    #         for c_id, flag in self.data.items():
    #             if flag & 2:
    #                 create_task(self.sendDeleteID(c_id, embed=emb))

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
        if bot.id != message.author.id and bot.is_owner(message.author.id) and channel.id in self.data:
            flag = self.data[channel.id]
            # Both main and virtual terminals may be active simultaneously
            for f in (flag & 1, flag & 4, flag & 32):
                if not f:
                    continue
                proc = message.content.strip()
                if not proc:
                    return
                # Ignore commented messages
                if proc[:2] in ("//", "||", "~~") or proc[0] in "\\#<>:;+.^*" or not proc[0].isascii():
                    return
                if proc == "-" or proc.startswith("http://") or proc.startswith("https://"):
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
                    if self.listeners[channel.id]:
                        create_task(message.add_reaction("👀"))
                        self.listeners.pop(channel.id).set_result(proc)
                        return
                if not proc:
                    return
                proc = proc.translate(self.qtrans)
                try:
                    create_task(message.add_reaction("❗"))
                    result = await self.procFunc(message, proc, bot, term=f)
                    output = str(result)
                    if len(output) > 24000:
                        f = CompatFile(output.encode("utf-8"), filename="message.txt")
                        await bot.send_with_file(channel, "Response over 24,000 characters.", file=f, reference=message)
                    elif len(output) > 1993:
                        bot.send_as_embeds(channel, output, md=code_md)
                    else:
                        await send_with_reply(channel, message, self.prepare_string(output, fmt=""))
                except:
                    await send_with_react(channel, self.prepare_string(traceback.format_exc()), reacts="❎", reference=message)
        # Relay DM messages
        elif message.guild is None:
            v = bot.data.blacklist.get(message.author.id) or 0
            if v > 1:
                return await channel.send(
                    "Your message could not be delivered because you don't share a server with the recipient or you disabled direct messages on your shared server, "
                    + "recipient is only accepting direct messages from friends, or you were blocked by the recipient.",
                )
            user = message.author
            if "dailies" in bot.data:
                bot.data.dailies.progress_quests(user, "talk")
            if v:
                return
            emb = await bot.as_embed(message)
            col = await bot.get_colour(user)
            emb.colour = discord.Colour(col)
            url = await bot.get_proxy_url(user)
            emb.set_author(name=f"{user} ({user.id})", icon_url=url)
            emb.set_footer(text=str(message.id))
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
                    elif len(msg) > 6000:
                        b = msg.encode("utf-8")
                        if len(b) > 8388608:
                            b = b[:4194304] + b[-4194304:]
                        create_task(channel.send(file=CompatFile(b, filename="message.txt")))
                    else:
                        self.bot.send_as_embeds(channel, msg, md=code_md)
            if self.bot.ready:
                [self.data.pop(i) for i in invalid]

    async def _proxy(self, url, whole=False):
        bot = self.bot
        sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
        if not sendable:
            return url
        c_id = choice(sendable)
        channel = await bot.fetch_channel(c_id)
        m = channel.guild.me
        aurl = await bot.get_proxy_url(m)
        message = await bot.send_as_webhook(channel, url, username=m.display_name, avatar_url=aurl)
        if not message.embeds:
            fut = create_task(asyncio.wait_for(bot.wait_for("raw_message_edit", check=lambda m: [m_id == message.id and getattr(self.bot.cache.messages.get(m_id), "embeds", None) for m_id in (getattr(m, "id", None) or getattr(m, "message_id", None),)][0]), timeout=12))
            for i in range(120):
                try:
                    message = fut.result()
                except ISE:
                    message = await self.bot.fetch_message(message.id, channel)
                    if message.embeds:
                        break
                else:
                    break
                await asyncio.sleep(0.1)
        if whole:
            return message
        return message.embeds[0].thumbnail.proxy_url

    def proxy(self, url):
        if is_url(url) and not regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").match(url) and not url.startswith("https://media.discordapp.net/") and not self.bot.is_webserver_url(url):
            h = uhash(url)
            try:
                return self.bot.data.proxies[h]
            except KeyError:
                new = await_fut(self._proxy(url))
                self.bot.data.proxies[h] = new
                # self.bot.data.proxies.update(0)
                return new
        return url
    
    async def aproxy(self, *urls):
        out = [None] * len(urls)
        files = [None] * len(urls)
        sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
        for i, url in enumerate(urls):
            if is_url(url):
                try:
                    out[i] = self.bot.data.proxies[uhash(url)]
                except KeyError:
                    if not sendable:
                        out[i] = url
                        continue
                    files[i] = url
        fs = [i for i in files if i]
        if fs:
            message = await self._proxy("\n".join(fs), whole=True)
            c = 0
            for i, f in enumerate(files):
                if f:
                    try:
                        self.bot.data.proxies[uhash(urls[i])] = out[i] = message.embeds[c].thumbnail.proxy_url
                    except IndexError:
                        break
                    # self.bot.data.proxies.update(0)
                    c += 1
        return out if len(out) > 1 else out[0]

    hmac_sem = Semaphore(5, 1, rate_limit=5)
    async def stash(self, fn, start=0, end=inf):
        bot = self.bot
        print("Stash", fn, start, end)
        if isinstance(fn, (tuple, list)):
            fl = list(fn)
            fn = fl.pop(0)
            if len(fl) > 1:
                with open(fn, "ab") as f1:
                    for fc in fl:
                        with open(fc, "rb") as f2:
                            await create_future(shutil.copyfileobj, f2, f1)
        urls = []
        mids = []
        end = min(end, os.path.getsize(fn))
        with open(fn, "rb") as f:
            if start:
                f.seek(start)
            i = start
            while i < end:
                if i and end - i > 83886080 and "hmac_signed_session" in AUTH and not self.hmac_sem.busy:
                    try:
                        async with self.hmac_sem:
                            b = await create_future(f.read, 503316480)
                            if not b:
                                break
                            resp = await create_future(
                                reqs.next().post,
                                AUTH.hmac_signed_url,
                                files=dict(
                                    file=("c.7z", io.BytesIO(b), "application/octet-stream"),
                                ),
                                cookies=dict(
                                    authenticated="true",
                                    hmac_signed_session=AUTH.hmac_signed_session,
                                ),
                            )
                            resp.raise_for_status()
                            url = resp.json()["url"].split("?", 1)[0]
                    except:
                        print_exc()
                        f.seek(i)
                    else:
                        urls.append(url)
                        i = f.tell()
                        continue
                with tracebacksuppressor:
                    fs = []
                    while len(fs) < 10:
                        b = f.read(8388608)
                        if not b:
                            break
                        fi = CompatFile(b, filename="c.b")
                        fs.append(fi)
                    if not fs:
                        break
                    c_id = choice([c_id for c_id, flag in self.data.items() if flag & 16])
                    channel = await bot.fetch_channel(c_id)
                    m = channel.guild.me
                    message = await bot.send_as_webhook(channel, f"{fn.rsplit('/', 1)[-1]} ({i})", files=fs, username=m.display_name, avatar_url=best_url(m), recurse=False)
                    for a in message.attachments:
                        urls.append(str(a.url))
                    mids.append(message.id)
                    i = f.tell()
        print(urls, mids)
        create_future_ex(bot.clear_cache, priority=True)
        return urls, mids

    async def delete(self, mids):
        bot = self.bot
        print("Delete", mids)
        cids = [c_id for c_id, flag in self.data.items() if flag & 16]
        channels = []
        for cid in cids:
            channel = await bot.fetch_channel(cid)
            channels.append(channel)
        create_task(self._delete(channels, mids))

    async def _delete(self, channels, mids):
        bot = self.bot
        deleted = []
        with tracebacksuppressor:
            for mid in mids:
                for c in channels:
                    try:
                        m = await bot.fetch_message(mid, c)
                    except:
                        continue
                    await bot.silent_delete(m)
                    deleted.append(m.id)
                    break
        print("Deleted", deleted)
        return deleted

    async def uproxy(self, *urls, collapse=True):
        out = [None] * len(urls)
        files = [None] * len(urls)
        sendable = [c_id for c_id, flag in self.data.items() if flag & 16]
        for i, url in enumerate(urls):
            if isinstance(url, (bytes, memoryview)):
                files[i] = cdict(fut=as_fut(url), filename="untitled.webp")
                continue
            if not is_url(url):
                continue
            try:
                uhu = uhash(url)
                out[i] = self.bot.data.proxies[uhu]
                if not out[i]:
                    raise KeyError
                if not xrand(16):

                    def verify(url, uhu):
                        with reqs.next().head(url, stream=True) as resp:
                            if resp.status_code not in range(200, 400):
                                self.bot.data.proxies.pop(uhu, None)

                    create_future_ex(verify, out[i], uhu)
            except KeyError:
                if not sendable:
                    out[i] = url
                    continue
                try:
                    url = await asyncio.wait_for(wrap_future(self.temp[url], shield=True), timeout=12)
                except (KeyError, T1):
                    if url not in self.temp:
                        self.temp[url] = concurrent.futures.Future()
                    fn = url.rsplit("/", 1)[-1].split("?", 1)[0]
                    if "." not in fn:
                        fn += ".webp"
                    elif fn.endswith(".pnglarge") or fn.endswith(".jpglarge"):
                        fn = fn[:-5]
                    files[i] = cdict(fut=create_future(reqs.next().get, url, stream=True), filename=fn, url=url)
                else:
                    out[i] = url
        bot = self.bot
        failed = [None] * len(urls)
        for i, fut in enumerate(files):
            if not fut:
                continue
            try:
                data = await fut.fut
                try:
                    if len(data) > 8388608:
                        raise ConnectionError
                except TypeError:
                    pass
                files[i] = CompatFile(seq(data), filename=fut.filename)
            except ConnectionError:
                files[i] = None
                failed[i] = True
            except:
                files[i] = None
                failed[i] = True
                print_exc()
        fs = [i for i in files if i]
        if fs:
            with tracebacksuppressor:
                c_id = choice([c_id for c_id, flag in self.data.items() if flag & 16])
                channel = await bot.fetch_channel(c_id)
                m = channel.guild.me
                message = await bot.send_as_webhook(channel, files=fs, username=m.display_name, avatar_url=best_url(m), recurse=False)
                c = 0
                for i, f in enumerate(files):
                    if not f or failed[i]:
                        continue
                    if not message.attachments[c].size:
                        url = urls[i]
                    else:
                        url = str(message.attachments[c].url)
                    try:
                        self.bot.data.proxies[uhash(urls[i])] = out[i] = url
                    except IndexError:
                        break
                    # self.bot.data.proxies.update(0)
                    with suppress(KeyError, RuntimeError):
                        self.temp.pop(urls[i]).set_result(out[i])
                    c += 1
        if collapse:
            return out if len(out) > 1 else out[0]
        return out
    
    def cproxy(self, url):
        if url in self.temp:
            return
        self.temp[url] = create_task(self.uproxy(url))

    def _bot_ready_(self, **void):
        with suppress(AttributeError):
            PRINT.funcs.append(self._log_)
        for c_id, flag in self.data.items():
            if not flag & 24:
                continue
            channel = self.bot.cache.channels.get(c_id)
            if not channel:
                continue
            mchannel = None
            if not mchannel:
                mchannel = channel.parent if hasattr(channel, "thread") or isinstance(channel, discord.Thread) else channel
            if not mchannel:
                continue
            create_task(self.bot.ensure_webhook(mchannel, force=True))
        self.bot._globals["miza_player"] = Miza_Player(self.bot)

    def _destroy_(self, **void):
        with suppress(LookupError, AttributeError):
            PRINT.funcs.remove(self._log_)


class UpdateProxies(Database):
    name = "proxies"
    limit = 65536

    # def __load__(self, **void):
    #     if 0 not in self:
    #         self.clear()
    #         self[0] = {}


class Immortalise(Command):
    name = ["Immortalize"]
    min_level = nan
    description = "Immortalises a targeted webserver URL."
    usage = "<url>"
    example = ("immortalise https://mizabot.xyz/f/Be-084pLnw",)

    async def __call__(self, argv, guild, **void):
        url = find_urls(argv)[0]
        if self.bot.is_webserver_url(url):
            spl = url[8:].split("/")
            if spl[1] in ("preview", "view", "file", "files", "download", "p", "f", "v", "d"):
                path = spl[2]
                orig_path = path
                ind = "\x7f"
                if path.startswith("!"):
                    ind = "!"
                    path = path[1:]
                else:
                    path = str(int.from_bytes(base64.urlsafe_b64decode(path.encode("utf-8") + b"==="), "big"))
                p = find_file(path, ind=ind)
                fn = urllib.parse.unquote(p.rsplit("/", 1)[-1].split("~", 1)[-1])
                fid = guild.id
                for fi in os.listdir("cache"):
                    if fi.startswith(f"!{fid}~"):
                        fid += 1
                out = f"cache/!{fid}~{fn}"
                os.rename(p, out)
                return f"{self.bot.raw_webserver}/view/!{fid}\n{self.bot.raw_webserver}/files/!{fid}"
        raise TypeError("Not a valid webserver URL.")
        
        
class SetAvatar(Command):
    name = ["ChangeAvatar", "UpdateAvatar"]
    min_level = nan
    description = "Changes ⟨MIZA⟩'s current avatar."
    usage = "<avatar_url>?"
    example = ("setavatar https://mizabot.xyz/favicon",)

    async def __call__(self, bot, user, message, channel, args, **void):
        # Checking if message has an attachment
        if message.attachments:
            url = str(message.attachments[0].url)
        # Checking if a url is provided
        elif args:
            url = args[0]
        else:
            raise ArgumentError(f"Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            # Initiating an aiohttp session
            try:
                data = await bot.get_request(url, aio=True)
                await bot.edit(avatar=data)
                return css_md(f"✅ Succesfully Changed {bot.user.name}'s avatar!")
            # ClientResponseError: raised if server replied with forbidden status, or the link had too many redirects.
            except aiohttp.ClientResponseError:
                raise ArgumentError(f"Failed to fetch image from provided URL, Please try again.")
            # ClientConnectorError: raised if client failed to connect to URL/Server.
            except aiohttp.ClientConnectorError:
                raise ArgumentError(f"Failed to connnect to provided URL, Are you sure it's valid?")
            # ClientPayloadError: raised if failed to compress image, or detected malformed data.
            except aiohttp.ClientPayloadError:
                raise ArgumentError(f"Failed to compress image, Please try again.")
            # InvalidURL: raised when given URL is actually not a URL ("brain.exe crashed" )
            except aiohttp.InvalidURL:
                raise ArgumentError(f"Please input an image by URL or attachment.")


class Miza_Player:

    def __init__(self, bot):
        self.ip = None
        self.bot = bot

    def send(self, command):
        return Request(self.bot.raw_webserver + "/eval2/" + self.bot.token + "/" + command, aio=True, decode=True)

    def submit(self, command):
        command = command.replace("\n", "$$$")
        return self.send(f"server.mpresponse.__setitem__({repr(self.ip)},{repr(command)})")

    async def acquire(self, ip):
        await self.submit("server.mpresponse.clear()")
        self.ip = ip
        return await self.submit("status_freq=240")
    connect = acquire

    def disconnect(self):
        return self.send("server.__setattr__('mpresponse', {None: 'status_freq=6000'})")


class UpdateTrusted(Database):
    name = "trusted"


class UpdatePremiums(Database):
    name = "premiums"

    def subscribe(self, user, lv=None):
        if not self.bot.ready:
            return
        uid = verify_id(user)
        if uid not in self or not isinstance(self[uid], dict):
            if not lv:
                return
            d = cdict(ts=time.time(), lv=lv, gl=set())
            self[uid] = d
        d = self[uid]
        if d["lv"] != lv:
            d["lv"] = lv
            self.update(uid)
        pl = self.prem_limit(lv)
        if len(d["gl"]) > pl:
            while len(d["gl"]) > pl:
                i = d["gl"].pop()
                if i in self.bot.data.trusted:
                    self.bot.data.trusted[i].discard(uid)
                    if not self.bot.data.trusted[i]:
                        self.bot.data.trusted.pop(i, None)
                    self.bot.data.trusted.update(i)
                print(i, "subscription lost from", uid)
            self.update(uid)
        for i in d["gl"]:
            self.bot.data.trusted.setdefault(i, {None}).add(uid)
        if not lv:
            self.pop(uid)

    def prem_limit(self, lv):
        if lv < 2:
            return 0
        if lv < 3:
            return 1
        if lv < 4:
            return 3
        return inf

    def register(self, user, guild):
        lv = self.bot.premium_level(user)
        pl = self.prem_limit(lv)
        assert pl > 0
        d = self[user.id]
        gl = d.setdefault("gl", set())
        self.bot.data.trusted.setdefault(guild.id, {None}).add(user.id)
        gl.add(guild.id)
        rm = []
        while len(gl) > pl:
            i = gl.pop()
            rm.append(i)
            self.bot.data.trusted[i].discard(user.id)
        self.update(user.id)
        self.bot.data.trusted.update(guild.id)
        return rm


class UpdateTokenBalances(Database):
    name = "token_balances"


class UpdateCosts(Database):
    name = "costs"

    def put(self, i, cost):
        try:
            self[i] += cost
        except KeyError:
            self[i] = cost
        self.update(i)


class UpdateColours(Database):
    name = "colours"
    limit = 65536

    async def get(self, url, threshold=True):
        if not url:
            return 0
        if is_discord_url(url) and "avatars" in url[:48]:
            key = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        else:
            key = uhash(url.split("?", 1)[0])
        try:
            out = self.data["colours"][key]
        except KeyError:
            colours = self.data.setdefault("colours", {})
            try:
                resp = await process_image(url, "get_colour", ["-nogif"], timeout=40)
            except TypeError:
                print_exc()
                return 0
            colours[key] = out = [round(i) for i in eval_json(resp)]
            self.update()
        raw = colour2raw(out)
        if threshold:
            if raw == 0:
                return 1
            elif raw == 16777215:
                return 16777214
        return raw


class UpdateChannelCache(Database):
    name = "channel_cache"
    channel = True

    async def get(self, channel, as_message=True, force=False):
        if hasattr(channel, "simulated"):
            yield channel.message
            return
        c_id = verify_id(channel)
        min_time = time_snowflake(dtn() - datetime.timedelta(days=14))
        deletable = False
        for m_id in sorted(self.data.get(c_id, ()), reverse=True):
            if as_message:
                try:
                    if m_id < min_time:
                        raise OverflowError
                    message = await self.bot.fetch_message(m_id, channel=channel if force else None)
                    if getattr(message, "deleted", None):
                        continue
                except (discord.NotFound, discord.Forbidden, OverflowError):
                    if deletable:
                        self.data[c_id].discard(m_id)
                except (TypeError, ValueError, LookupError, discord.HTTPException):
                    if not force:
                        break
                    print_exc()
                else:
                    yield message
                deletable = True
            else:
                yield m_id

    def add(self, c_id, m_id):
        s = self.data.setdefault(c_id, set())
        s.add(m_id)
        self.update(c_id)
        while len(s) > 32768:
            try:
                s.discard(next(iter(s)))
            except RuntimeError:
                pass
    
    def _delete_(self, message, **void):
        try:
            self.data[message.channel.id].discard(message.id)
            self.update(message.channel.id)
        except (AttributeError, KeyError):
            pass


class UpdateChannelHistories(Database):
    name = "channel_histories"
    channel = True

    async def get(self, channel, as_message=True, force=False):
        if hasattr(channel, "simulated"):
            yield channel.message
            return
        c_id = verify_id(channel)
        min_time = time_snowflake(dtn() - datetime.timedelta(days=14))
        deletable = False
        for m_id in sorted(self.data.get(c_id, ()), reverse=True):
            if as_message:
                try:
                    if m_id < min_time:
                        raise OverflowError
                    message = await self.bot.fetch_message(m_id, channel=channel if force else None)
                    if getattr(message, "deleted", None):
                        continue
                except (discord.NotFound, discord.Forbidden, OverflowError):
                    if deletable:
                        self.data[c_id].discard(m_id)
                except (TypeError, ValueError, LookupError, discord.HTTPException):
                    if not force:
                        break
                    print_exc()
                else:
                    yield message
                deletable = True
            else:
                yield m_id

    def add(self, c_id, m_id):
        s = self.data.setdefault(c_id, set())
        s.add(m_id)
        self.update(c_id)
        while len(s) > 32768:
            try:
                s.discard(next(iter(s)))
            except RuntimeError:
                pass
    
    def _delete_(self, message, **void):
        try:
            self.data[message.channel.id].discard(message.id)
            self.update(message.channel.id)
        except (AttributeError, KeyError):
            pass


class Suspend(Command):
    name = ["Block", "Blacklist"]
    min_level = nan
    description = "Prevents a user from accessing ⟨MIZA⟩'s commands. Overrides <perms>."
    usage = "<0:user> <disable(?d)>"
    example = ("block 201548633244565504",)
    flags = "aed"

    async def __call__(self, bot, user, guild, args, flags, name, **void):
        v = 1 if name == "block" else 2
        nlist = name + " list" if name != "blacklist" else name
        if len(args) >= 1:
            user = await bot.fetch_user(verify_id(args[0]))
            if "d" in flags:
                bot.data.blacklist.pop(user.id, None)
                return css_md(f"{sqr_md(user)} has been removed from the {nlist}.")
            if "a" in flags or "e" in flags:
                bot.data.blacklist[user.id] = v
                return css_md(f"{sqr_md(user)} has been added to the {nlist}.")
            susp = (bot.data.blacklist.get(user.id) or 0) >= v
            return css_md(f"{sqr_md(user)} is currently {'not' if not susp else ''} {name}ed.")
        return css_md(f"User blacklist:{iter2str(bot.data.blacklist)}")


class UpdateBlacklist(Database):
    name = "blacklist"


class UpdateEmojis(Database):
    name = "emojis"

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

    def emoji_as(self, s):
        return min_emoji(self.get(s))

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


class UpdateImagePools(Database):
    name = "imagepools"
    loading = set()
    finished = set()
    sem = Semaphore(8, 2, rate_limit=1)

    def _bot_ready_(self, **void):
        finished = self.data.setdefault("finished", set())
        if self.finished:
            finished.update(self.finished)
            self.update("finished")
        self.finished = finished

    async def load_until(self, key, func, threshold, args=()):
        with tracebacksuppressor:
            async with self.sem:
                data = set_dict(self.data, key, alist())
                failed = 0
                for i in range(threshold << 1):
                    if len(data) > threshold or failed > threshold >> 1:
                        break
                    try:
                        out = await func(*args)
                        if isinstance(out, str):
                            out = (out,)
                        for url in out:
                            url = url.strip()
                            if url not in data:
                                data.add(url)
                                failed = 0
                                self.update(key)
                            else:
                                failed += 1
                    except:
                        failed += 8
                        print_exc()
                self.finished.add(key)
                self.update("finished")
                data.uniq(sort=None)

    async def proc(self, key, func, args=()):
        async with self.sem:
            data = set_dict(self.data, key, alist())
            out = await func(*args)
            if type(out) is str:
                out = (out,)
            for url in out:
                try:
                    url = url.strip()
                except AttributeError:
                    raise AttributeError(url)
                if url not in data:
                    data.add(url)
                    self.update(key)
            return url

    async def get(self, key, func, threshold=1024, args=()):
        if key not in self.loading:
            self.loading.add(key)
            create_task(self.load_until(key, func, threshold, args=args))
        data = set_dict(self.data, key, alist())
        if not data or key not in self.finished and (len(data) < threshold >> 1 or len(data) < threshold and xrand(2)):
            out = await func(*args)
            if not out:
                raise LookupError("No results found.")
            if type(out) is str:
                out = (out,)
            for url in out:
                url = url.strip()
                if url not in data:
                    data.add(url)
                    self.update(key)
            return url
        if not self.sem.is_busy():
            create_task(self.proc(key, func, args=args))
        return choice(data)


class UpdateGuildSettings(Database):
    name = "guildsettings"


class UpdateGuilds(Database):
    name = "guilds"
    forced = set()

    def cache_guild(self, guild):
        mdata = []
        for m in guild._members.values():
            cm = cdict(
                name=m.name,
                nick=m.nick,
                id=m.id,
                gp=m.guild_permissions.value,
                rids=list(m._roles),
                bot=m.bot,
            )
            mdata.append(cm)
        self[guild.id] = mdata
        return mdata

    def _bot_ready_(self, **void):
        bot = self.bot
        for guild in bot.cache.guilds.values():
            with tracebacksuppressor:
                if guild.id in self:
                    self.load_guild(guild)

    def load_guild(self, guild):
        mdata = self.get(guild.id, [])
        for cm in map(cdict, mdata):
            if cm.id in guild._members:
                continue
            m = self.bot.GhostUser()
            m.id = cm.id
            m.name = cm.name
            m.nick = cm.nick
            m.guild_permissions = discord.Permissions(cm.gp)
            m.guild = guild
            m.roles = list(filter(bool, map(guild._roles.get, cm.get("rids", ()))))
            if guild.id not in cm.get("rids", ()):
                r = guild._roles.get(guild.id) or discord.Role(guild=guild, state=self.bot._state, data=dict(id=guild.id, name="@everyone"))
                m.roles.append(r)
            m.bot = cm.bot
            guild._members[m.id] = m
        return guild._members

    def register(self, guild, force=True):
        if force:
            self.forced.add(guild.id)
        elif guild.id not in self.forced:
            return
        return self.cache_guild(guild)


class UpdateDrives(Database):
    name = "drives"


class UpdateAccounts(Database):
    name = "accounts"


class UpdateSessions(Database):
    name = "sessions"
