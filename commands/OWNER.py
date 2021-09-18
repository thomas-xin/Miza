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
    usage = "<delay>?"
    _timeout_ = inf

    async def __call__(self, message, channel, guild, user, argv, name, **void):
        bot = self.bot
        client = bot.client
        await message.add_reaction("❗")
        save = None
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
                save = create_task(bot.send_event("_save_", force=True))
                if wait > 0:
                    await asyncio.sleep(wait)
        if name == "shutdown":
            await send_with_reply(channel, content="Shutting down... :wave:", reference=message)
        else:
            await send_with_reply(channel, content="Restarting... :wave:", reference=message)
        if save is None:
            print("Saving message cache...")
            save = create_task(bot.send_event("_save_", force=False))
        async with Delay(1):
            with discord.context_managers.Typing(channel):
                # Call _destroy_ bot event to indicate to all databases the imminent shutdown
                print("Destroying database memory...")
                await bot.send_event("_destroy_", shutdown=True)
                # Kill the audio player client
                print("Shutting down audio client...")
                kill = create_future(bot.audio.kill, priority=True)
                # Save any database that has not already been autosaved
                print("Saving all databases...")
                await create_future(bot.update, force=True, priority=True)
                # Send the bot "offline"
                print("Going offline...")
                await bot.change_presence(status=discord.Status.invisible)
                # Kill math and image subprocesses
                print("Killing math and image subprocesses...")
                with tracebacksuppressor:
                    await create_future(sub_kill, start=False, priority=True)
                # Kill the webserver
                print("Killing webserver...")
                with tracebacksuppressor:
                    await create_future(bot.server.kill, priority=True)
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
        bot.close()
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
    multi = True

    async def __call__(self, bot, user, message, guild, argl, args, argv, **void):
        if args and args[0] == "as":
            args.pop(0)
        users = await bot.find_users(argl, args, user, guild)
        if not users:
            raise LookupError("No results found.")
        if args:
            try:
                argv = message.content.split("run ", 1)[1]
            except IndexError:
                pass
                # raise ArgumentError('"run" must be specified as a separator.')
            futs = deque()
            for u in users:
                fake_message = copy.copy(message)
                fake_message.content = argv
                fake_message.author = u
                futs.append(create_task(bot.process_message(fake_message, argv)))
            for fut in futs:
                await fut


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
        "proxy": 16,
    })

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
    temp = {}

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
            url = await bot.get_proxy_url(user)
            emb.set_author(name=f"{user} ({user.id})", icon_url=url)
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
                                await bot.send_with_file(channel, "Response over 54,000 characters.", file=f, reference=message)
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
            emb = as_embed(message)
            col = await bot.get_colour(user)
            emb.colour = discord.Colour(col)
            url = await bot.get_proxy_url(user)
            emb.set_author(name=f"{user} ({user.id})", icon_url=url)
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
            h = shash(url)
            try:
                return self.bot.data.proxies[0][h]
            except KeyError:
                new = await_fut(self._proxy(url))
                self.bot.data.proxies[0][h] = new
                self.bot.data.proxies.update(0)
                return new
        return url
    
    async def aproxy(self, *urls):
        out = [None] * len(urls)
        files = [None] * len(urls)
        sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
        for i, url in enumerate(urls):
            if is_url(url):
                try:
                    out[i] = self.bot.data.proxies[0][shash(url)]
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
                        self.bot.data.proxies[0][shash(urls[i])] = out[i] = message.embeds[c].thumbnail.proxy_url
                    except IndexError:
                        break
                    self.bot.data.proxies.update(0)
                    c += 1
        return out if len(out) > 1 else out[0]
    
    async def uproxy(self, *urls, collapse=True):
        out = [None] * len(urls)
        files = [None] * len(urls)
        sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
        for i, url in enumerate(urls):
            if is_url(url):
                try:
                    out[i] = self.bot.data.proxies[0][shash(url)]
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
                            fn += ".png"
                        files[i] = cdict(fut=create_task(Request(url, aio=True)), filename="SPOILER_" + fn, url=url)
                    else:
                        out[i] = url
        bot = self.bot
        failed = [None] * len(urls)
        for i, fut in enumerate(files):
            if fut:
                try:
                    data = await fut.fut
                    if len(data) > 8388608:
                        raise ConnectionError
                    files[i] = CompatFile(data, filename=fut.filename)
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
                c_id = choice(list(c_id for c_id, flag in self.data.items() if flag & 16))
                channel = await bot.fetch_channel(c_id)
                m = channel.guild.me
                message = await bot.send_as_webhook(channel, files=fs, username=m.display_name, avatar_url=best_url(m), recurse=False)
                c = 0
                for i, f in enumerate(files):
                    if f and not failed[i]:
                        try:
                            self.bot.data.proxies[0][shash(urls[i])] = out[i] = str(message.attachments[c].url)
                        except IndexError:
                            break
                        self.bot.data.proxies.update(0)
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
            if flag & 24:
                channel = self.bot.cache.channels.get(c_id)
                if channel:
                    create_task(self.bot.ensure_webhook(channel, force=True))
        self.bot._globals["miza_player"] = Miza_Player(self.bot)

    def _destroy_(self, **void):
        with suppress(LookupError, AttributeError):
            PRINT.funcs.remove(self._log_)


class UpdateProxies(Database):
    name = "proxies"
    no_delete = True
    limit = 65536

    def __load__(self, **void):
        if 0 not in self:
            self.clear()
            self[0] = {}


class Immortalise(Command):
    name = ["Immortalize"]
    min_level = nan
    description = "Immortalises a targeted webserver URL."
    usage = "<url>"

    async def __call__(self, argv, guild, **void):
        url = find_urls(argv)[0]
        if self.bot.is_webserver_url(url):
            spl = url[8:].split("/")
            if spl[1] in ("preview", "view", "file", "files", "download"):
                path = spl[2]
                orig_path = path
                ind = "\x7f"
                if path.startswith("~"):
                    path = str(int.from_bytes(base64.urlsafe_b64decode(path.encode("utf-8") + b"==="), "big"))
                elif path.startswith("!"):
                    ind = "!"
                    path = path[1:]
                p = find_file(path, ind=ind)
                fn = urllib.parse.unquote(p.rsplit("/", 1)[-1].split("~", 1)[-1])
                fid = guild.id
                for fi in os.listdir("cache"):
                    if fi.startswith(f"!{fid}~"):
                        fid += 1
                out = f"cache/!{fid}~{fn}"
                await create_future(os.rename, p, out, priority=True)
                return f"{self.bot.webserver}/view/!{fid}\n{self.bot.webserver}/download/!{fid}"
        raise TypeError("Not a valid webserver URL.")
        
        
class SetAvatar(Command):
    name = ["ChangeAvatar", "UpdateAvatar"]
    min_level = nan
    description = "Changes ⟨MIZA⟩'s current avatar."
    usage = "<avatar_url>?"

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


# class DownloadServer(Command):
#     name = ["SaveServer", "ServerDownload"]
#     min_level = nan
#     description = "Downloads all posted messages in the target server into a sequence of .txt files."
#     usage = "<server_id>?"
#     flags = "f"
#     _timeout_ = 512

#     async def __call__(self, bot, argv, flags, channel, guild, **void):
#         if "f" not in flags:
#             return bot.dangerous_command
#         if argv:
#             g_id = verify_id(argv)
#             guild = await bot.fetch_guild(g_id)
#         with discord.context_managers.Typing(channel):
#             send = channel.send

#             # Create callback function to send all results of the guild download.
#             async def callback(channel, messages, **void):
#                 b = bytes()
#                 fn = str(channel) + " (" + str(channel.id) + ")"
#                 for i, message in enumerate(messages, 1):
#                     temp = ("\n\n" + message_repr(message, username=True)).encode("utf-8")
#                     if len(temp) + len(b) > 8388608:
#                         await send(file=CompatFile(io.BytesIO(b), filename=fn + ".txt"))
#                         fn += "_"
#                         b = temp[2:]
#                     else:
#                         if b:
#                             b += temp
#                         else:
#                             b += temp[2:]
#                     if not i & 8191:
#                         await asyncio.sleep(0.2)
#                 if b:
#                     await send(file=CompatFile(io.BytesIO(b), filename=fn + ".txt"))

#             await self.bot.data.counts.getGuildHistory(guild, callback=callback)
#         response = uni_str("Download Complete.")
#         return bold(ini_md(sqr_md(response)))


class UpdateTrusted(Database):
    name = "trusted"


class UpdateColours(Database):
    name = "colours"
    no_delete = True
    limit = 65536

    async def get(self, url, threshold=True):
        if not url:
            return 0
        if is_discord_url(url) and "avatars" in url[:48]:
            key = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        else:
            key = shash(url.split("?", 1)[0])
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


class UpdateWebhooks(Database):
    name = "webhooks"
    channel = True
    CID = collections.namedtuple("id", ["id"])
    temp = {}

    def from_dict(self, d, c_id):
        d = copy.copy(d)
        d.url = f"https://discord.com/api/webhooks/{d.id}/{d.token}"
        w = discord.Webhook.from_url(d.url, adapter=discord.AsyncWebhookAdapter(Request.session))
        d.send = w.send
        d.avatar_url = d.avatar and f"https://cdn.discordapp.com/avatars/{d.id}/{d.avatar}.png?size=1024"
        d.channel = self.CID(id=c_id)
        return self.add(d)

    def to_dict(self, user):
        return cdict(
            id=user.id,
            name=user.name,
            created_at=user.created_at,
            avatar=user.avatar,
            token=user.token,
        )

    def add(self, w):
        user = self.bot.GhostUser()
        with suppress(AttributeError):
            user.channel = w.channel
        user.id = w.id
        user.name = w.name
        user.display_name = w.name
        user.joined_at = w.created_at
        user.avatar = w.avatar
        user.avatar_url = w.avatar_url
        user.bot = True
        user.send = w.send
        user.dm_channel = getattr(w, "channel", None)
        user.webhook = w
        try:
            sem = self.bot.cache.users[w.id].semaphore
        except (AttributeError, KeyError):
            sem = None
        self.bot.cache.users[w.id] = user
        if w.token:
            webhooks = set_dict(self.data, w.channel.id, cdict())
            webhooks[w.id] = self.to_dict(w)
            if sem is None:
                sem = Semaphore(5, 256, rate_limit=5)
            user.semaphore = sem
        return user

    async def get(self, channel, force=False, bypass=False):
        guild = getattr(channel, "guild", None)
        if not guild:
            raise TypeError("DM channels cannot have webhooks.")
        if not force:
            with suppress(KeyError):
                return self.temp[channel.id]
            if channel.id in self.data:
                self.temp[channel.id] = temp = alist(self.from_dict(w, channel.id) for w in self.data[channel.id].values())
                return temp
        async with self.bot.guild_semaphore if not bypass else emptyctx:
            self.data.pop(channel.id, None)
            if not channel.permissions_for(channel.guild.me).manage_webhooks:
                raise PermissionError("Not permitted to create webhooks in channel.")
            webhooks = None
            if guild.me.guild_permissions.manage_webhooks:
                with suppress(discord.Forbidden):
                    webhooks = await guild.webhooks()
            if webhooks is None:
                webhooks = await aretry(channel.webhooks, attempts=5, delay=15, exc=(discord.Forbidden, discord.NotFound))
        self.temp[channel.id] = temp = alist(w for w in [self.add(w) for w in webhooks if w.token] if w.channel.id == channel.id)
        return temp


class UpdateChannelCache(Database):
    name = "channel_cache"
    channel = True

    async def get(self, channel, as_message=True):
        if hasattr(channel, "simulated"):
            yield channel.message
            return
        c_id = verify_id(channel)
        min_time = time_snowflake(utc_dt() - datetime.timedelta(days=14))
        deletable = False
        for m_id in sorted(self.data.get(c_id, ()), reverse=True):
            if as_message:
                try:
                    if m_id < min_time:
                        raise OverflowError
                    message = await self.bot.fetch_message(m_id, channel)
                    if getattr(message, "deleted", None):
                        continue
                except (discord.NotFound, discord.Forbidden, OverflowError):
                    if deletable:
                        self.data[c_id].discard(m_id)
                except (TypeError, ValueError, discord.HTTPException):
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
    no_delete = True

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
                        if type(out) is str:
                            out = (out,)
                        for url in out:
                            url = url.strip()
                            if url not in data:
                                if i & 1:
                                    data.appendleft(url)
                                else:
                                    data.append(url)
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
                url = url.strip()
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