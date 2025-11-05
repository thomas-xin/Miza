# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


class Reload(Command):
	min_level = nan
	description = "Reloads a specified module."
	schema = cdict(
		reload=cdict(
			type="word",
		),
		unload=cdict(
			type="word",
		),
	)
	_timeout_ = inf

	async def __call__(self, bot, _channel, _message, unload, reload, **void):
		await _message.add_reaction("❗")
		if unload:
			unload = unload.upper()
			message = await send_with_reply(_channel, content=f"Unloading {unload}...", reference=_message)
			succ = await asubmit(bot.unload, unload, priority=1)
			if succ:
				await message.edit(content=f"Successfully unloaded {unload}.")
			else:
				await message.edit(content=f"Error unloading {unload}. Please see log for more info.")
		if reload:
			reload = reload.upper()
			message = await send_with_reply(_channel, content=f"Reloading {reload}...", reference=_message)
			succ = await asubmit(bot.reload, reload, priority=True)
			if succ:
				await message.edit(f"Successfully reloaded {reload}.")
			else:
				await message.edit(f"Error reloading {reload}. Please see log for more info.")


class Restart(Command):
	name = ["Reboot"]
	min_level = nan
	description = "Restarts, reloads, or shuts down ⟨BOT⟩, with an optional delay."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("workers", "audio", "server", "maintain", "reboot", "shutdown", "update"),
				accepts=dict(restart="reboot", wait="maintain"),
			),
			description="Image supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			default="maintain",
		),
		delay=cdict(
			type="timedelta",
			description="Time to wait",
			aliases=["d"],
			default=0,
		),
	)
	macros = cdict(
		Shutdown=cdict(
			mode="shutdown",
		),
		Maintain=cdict(
			mode="maintain",
		),
		Update=cdict(
			mode="update",
		),
	)
	typing = True
	_timeout_ = inf

	async def __call__(self, bot, _message, _channel, _user, mode, delay, **void):
		t = utc()
		await _message.add_reaction("❗")
		if mode == "workers":
			m = await send_with_reply(_channel, content="Restarting compute workers...", reference=_message)
			await restart_workers()
			await m.edit("Compute workers restarted successfully.")
			return
		if mode == "audio":
			m = await send_with_reply(_channel, content="Restarting audio client...", reference=_message)
			await asubmit(bot.start_audio_client)
			await m.edit("Audio client restarted successfully.")
			return
		if mode == "server":
			m = await send_with_reply(_channel, content="Restarting webserver...", reference=_message)
			await asubmit(bot.start_webserver)
			await m.edit("Webserver restarted successfully.")
			return
		save = None
		if mode == "update":
			resp = await asubmit(subprocess.run, ["git", "pull"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			print(resp.stdout)
			print(resp.stderr)
		if mode == "maintain":
			m = None
			busy = -inf
			with tracebacksuppressor:
				async with discord.context_managers.Typing(_channel):
					while busy:
						lb = busy
						busy = bot.command_semaphore.active
						if busy:
							if busy != lb:
								content = f"*Waiting until idle ({busy})...*"
								if not m:
									m = await send_with_reply(_channel, content=content, reference=_message)
								else:
									m = await bot.edit_message(m, content=content)
							await asyncio.sleep(1)
			if m:
				csubmit(bot.edit_message(m, content=f"*Waiting until idle (complete)*"))
		delay = float(delay) - (utc() - t)
		if delay > 0:
			# Restart announcements for when a time input is specified
			with tracebacksuppressor:
				async with discord.context_managers.Typing(_channel):
					async with Delay(delay):
						await send_with_reply(_channel, content="*Preparing to " + mode + " in " + sec2time(delay) + "...*", reference=_message)
						emb = discord.Embed(colour=discord.Colour(1))
						url = await bot.get_proxy_url(bot.user)
						emb.set_author(name=str(bot.user), url=bot.github, icon_url=url)
						emb.description = f"I will be {'shutting down' if mode == 'shutdown' else 'restarting'} in {sec2time(delay)}, apologies for any inconvenience..."
						await bot.send_event("_announce_", embed=emb)
						save = csubmit(bot.send_event("_save_"))
		with tracebacksuppressor:
			if mode == "shutdown":
				await send_with_reply(_channel, content="Shutting down... :wave:", reference=_message)
			else:
				await send_with_reply(_channel, content="Restarting... :wave:", reference=_message)
		return await self.confirm_shutdown(_channel, _user, save=save, mode=mode)

	async def confirm_shutdown(self, _channel=None, _user=None, save=None, mode="restart"):
		bot = self.bot
		client = bot.client
		bot.closing = True
		if save is None:
			print("Saving message cache...")
			save = csubmit(bot.send_event("_save_"))
		ctx = discord.context_managers.Typing(_channel) if _channel else emptyctx
		async with Delay(1):
			async with ctx:
				# Call _destroy_ bot event to indicate to all databases the imminent shutdown
				print("Destroying database memory...")
				with tracebacksuppressor:
					await bot.send_event("_destroy_", shutdown=mode == "shutdown")
				# Save any database that has not already been autosaved
				print("Saving all databases...")
				with tracebacksuppressor:
					await asubmit(bot.handle_update, force=True, priority=True)
					await asubmit(bot.update, force=True, priority=True)
				# Send the bot "offline"
				bot.closed = True
				print("Going offline...")
				with tracebacksuppressor:
					async with asyncio.timeout(3):
						await bot.change_presence(status=discord.Status.invisible)
				print("Waiting on save...")
				with tracebacksuppressor:
					await save
				print("Goodbye.")
				with suppress(NameError, AttributeError):
					PRINT.flush()
					PRINT.close(force=True)
				# await gather(*futs, return_exceptions=True)
		import pathlib
		if mode and mode.casefold() == "shutdown":
			pathlib.Path.touch(bot.shutdown)
		else:
			pathlib.Path.touch(bot.restart)
		with suppress():
			os.remove(bot.heartbeat_file)
		with suppress():
			await bot.close()
		del client
		del bot
		raise SystemExit


class Execute(Command):
	min_level = nan
	description = "Executes a command as other user(s), similar to the command's function in Minecraft."
	usage = "as <0:user>* run <1:command>+ <inherit_perms(-i)>?"
	example = ("execute as @Miza run ~info",)
	flags = "i"
	multi = True

	async def __call__(self, bot, user, message, channel, guild, argl, args, argv, flags, perm, **void):
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
			g = T(c).get("guild")
			fake_message.guild = g
			if g:
				fake_message.author = g.get_member(u.id) or u
			else:
				fake_message.author = u
			futs.append(bot.process_message(fake_message, argv, min_perm=-inf if "i" in flags else perm))
		await gather(*futs)


class Exec(Command):
	name = ["Aexec", "Aeval", "Eval"]
	min_level = nan
	description = "Causes all messages by the bot owner(s) in the current channel to be executed as python code on ⟨BOT⟩."
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
		lfproxy=128,
	))
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("view", "enable", "disable"),
			),
			description="Action to perform",
			example="enable",
			default="view"
		),
		type=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(terminal_types.a),
			),
			description="Terminal mode",
			example="virtual",
			default="null",
		),
		code=cdict(
			type="string",
			description="Code to evaluate directly",
			example="1 + 1",
		),
	)

	async def __call__(self, bot, _channel, _message, mode, type, code, **void):
		num = self.terminal_types[type]
		execd = bot.data.exec
		if mode == "enable":
			num = num or 4
			try:
				execd[_channel.id] |= num
			except KeyError:
				execd[_channel.id] = num
			# Test bitwise flags for enabled terminals
			out = ", ".join(self.terminal_types.get(1 << i) for i in bits(execd[_channel.id]))
			csubmit(_message.add_reaction("❗"))
			attachment_cache["@channels"] = [k for k, v in execd.items() if v & 16]
			return css_md(f"{sqr_md(out)} terminal now enabled in {sqr_md(_channel)}.")
		if mode == "disable":
			if not num:
				# Test bitwise flags for enabled terminals
				out = ", ".join(self.terminal_types.get(1 << i) for i in bits(execd.pop(_channel.id, 0, force=True)))
			else:
				out = ", ".join(self.terminal_types.get(1 << i) for i in bits(num))
				with suppress(KeyError):
					execd[_channel.id] &= -num - 1
					if not execd[_channel.id]:
						execd.pop(_channel.id)
			attachment_cache["@channels"] = [k for k, v in execd.items() if v & 16]
			return css_md(f"Successfully removed {sqr_md(out)} terminal.")
		if code:
			proc = code.translate(execd.qtrans)
			try:
				csubmit(_message.add_reaction("❗"))
				result = await execd.procFunc(_message, proc, bot, term=4)
				output = str(result)
				return cdict(content=output, prefix="```\n", suffix="```")
			except BaseException as ex:
				tb = T(ex).get("original_traceback") or traceback.format_exc()
				return cdict(
					content=execd.prepare_string(tb),
					reacts="❎",
				)
		out = iter2str({k: ", ".join(self.terminal_types.get(1 << i) for i in bits(v)) for k, v in sorted(execd.items())})
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
		self.listeners[channel.id] = fut = Future()
		return fut.result(timeout=86400)

	psem = Semaphore(1, 64)

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
			try:
				glob["auds"] = bot.data.audio.players[message.guild.id]
				glob["ytdl"] = bot.ytdl = ytdl = bot._globals["VOICE"].ytdl
				ytdl.bot = bot
			except (AttributeError, KeyError):
				pass
		if term & 32:
			proc = await asyncio.create_subprocess_shell(proc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, limit=65536)
			out = await proc.stdout.read()
			err = await proc.stderr.read()
			output = (as_str(out) + "\n" + as_str(err)).strip()
			if output:
				glob["_"] = output
			return output
		if self.psem.busy:
			output = await asubmit(aeval, proc, glob, priority=0)
		else:
			async with self.psem:
				output = await asubmit(aeval, proc, glob, priority=3)
		# Output sent to "_" variable if used
		if output is not None:
			glob["_"] = output
		return output

	async def sendDeleteID(self, c_id, delete_after=20, **kwargs):
		# Autodeletes after a delay
		channel = await self.bot.fetch_channel(c_id)
		message = await channel.send(**kwargs)
		if isfinite(delete_after):
			csubmit(self.bot.silent_delete(message, no_log=True, delay=delete_after))

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
	#                 csubmit(self.sendDeleteID(c_id, embed=emb))

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
			if bot.is_mentioned(message, bot, message.guild):
				return
			flag = self.data[channel.id]
			# Both main and virtual terminals may be active simultaneously
			for f in (flag & 1, flag & 4, flag & 32):
				if not f:
					continue
				c = proc = message.content.strip()
				if not proc:
					return
				# Ignore commented messages
				if c[0] in COMM or c[:2] in ("//", "/*"):
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
						csubmit(message.add_reaction("👀"))
						self.listeners.pop(channel.id).set_result(proc)
						return
				if not proc:
					return
				proc = proc.translate(self.qtrans)
				try:
					csubmit(message.add_reaction("❗"))
					result = await self.procFunc(message, proc, bot, term=f)
					output = str(result)
					await bot.respond_with(cdict(content=output, prefix="```\n", suffix="```"), message=message)
				except BaseException as ex:
					tb = T(ex).get("original_traceback") or traceback.format_exc()
					await send_with_react(channel, self.prepare_string(tb), reacts="❎", reference=message)
		# Relay DM messages
		elif message.guild is None and T(message.channel).get("recipient"):
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

	def __load__(self, **void):
		sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
		AUTH["proxy_channels"] = sendable
		save_auth(AUTH)
		misc.caches.attachment_cache.init()

	# All logs that normally print to stdout/stderr now send to the assigned log channels
	def _log_(self, msg, **void):
		if not self.bot or self.bot.api_latency > 6 or self.backoff > utc():
			return
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
						csubmit(self.logto(channel, CompatFile(b, filename="message.txt")))
					else:
						self.bot.send_as_embeds(channel, msg, md=ansi_md, bottleneck=True)
			if self.bot.ready:
				[self.data.pop(i) for i in invalid]

	backoff = 0
	skip_until = 0
	async def logto(self, channel, file):
		try:
			await channel.send(file=file)
		except aiohttp.client_exceptions.ClientConnectorDNSError:
			self.backoff = max(self.backoff * 2, 1)
			self.skip_until = utc() + self.backoff
		else:
			self.backoff = 0
			self.skip_until = 0

	async def _proxy(self, url, whole=False):
		bot = self.bot
		sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
		if not sendable:
			return url
		ext = url2ext(url)
		if not IMAGE_FORMS.get(ext):
			return await self.uproxy(url)
		c_id = choice(sendable)
		channel = await bot.fetch_channel(c_id)
		m = channel.guild.me
		embed = discord.Embed(colour=rand_colour())
		embed.set_thumbnail(url=url)
		message = await bot.send_as_webhook(channel, url, embed=embed, username=m.display_name)
		if not message.embeds:
			return await self.uproxy(url)
		if whole:
			return message
		return message.embeds[0].thumbnail.proxy_url

	def proxy(self, url):
		if is_url(url) and not regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").match(url) and not url.startswith("https://media.discordapp.net/") and not self.bot.is_webserver_url(url):
			h = uuhash(url)
			try:
				return self.bot.data.proxies[h]
			except KeyError:
				new = await_fut(self._proxy(url))
				self.bot.data.proxies[h] = new
				return new
		return url
	
	async def aproxy(self, *urls):
		out = [None] * len(urls)
		files = [None] * len(urls)
		sendable = list(c_id for c_id, flag in self.data.items() if flag & 16)
		for i, url in enumerate(urls):
			if is_url(url):
				try:
					out[i] = self.bot.data.proxies[uuhash(url)]
					if discord_expired(out[i]):
						raise KeyError(out[i])
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
						self.bot.data.proxies[uuhash(urls[i])] = out[i] = message.embeds[c].thumbnail.proxy_url
					except IndexError:
						break
					c += 1
		return out if len(out) > 1 else out[0]

	hmac_sem = Semaphore(5, 1, rate_limit=5)
	async def stash(self, fn, start=0, end=inf, filename=None, dont=False, raw=True):
		bot = self.bot
		fns = [fn] if isinstance(fn, (str, bytes, memoryview, io.BytesIO, discord.File)) else fn
		print("Stash", lim_str(str(fns), 256), start, end, filename)
		urls = []
		with FileStreamer(*fns) as f:
			fn = f.filename or "untitled"
			i = start
			f.seek(i)
			while i < end:
				b = None
				if 0 and end - i > 83886080 and "hmac_signed_session" in AUTH and not self.hmac_sem.full:
					try:
						async with self.hmac_sem:
							b = await asubmit(f.read, 503316480)
							if not b:
								break
							if len(b) <= 100663296 or dont:
								raise StopIteration(f"Skipping small chunk {len(b)}")
							fn2 = filename or "c.7z"
							resp = await asubmit(
								reqs.next().post,
								AUTH.hmac_signed_url,
								files=dict(
									file=(fn2, io.BytesIO(b), "application/octet-stream"),
								),
								cookies=dict(
									authenticated="true",
									hmac_signed_session=AUTH.hmac_signed_session,
								),
							)
							resp.raise_for_status()
							url = resp.json()["url"].split("?", 1)[0]
					except StopIteration as ex:
						print("Stash error:", repr(ex))
						if not b:
							print_exc()
							f.seek(i)
					except Exception:
						print_exc()
						f.seek(i)
					else:
						if raw:
							u = url
						else:
							u = self.bot.preserve_attachment(url) + "?S=" + str(len(b))
						urls.append(u)
						i = f.tell()
						continue
				with tracebacksuppressor:
					chunkf = []
					fs = []
					sizes = []
					fn2 = filename or "c.b"
					if b:
						bm = memoryview(b)
						if len(bm) > CACHE_FILESIZE * 8:
							bm = bm[:CACHE_FILESIZE * 8]
						for n in range(0, len(bm), CACHE_FILESIZE):
							bi = bm[n:n + CACHE_FILESIZE]
							chunkf.append(bi)
							fi = CompatFile(bi, filename=fn2)
							fs.append(fi)
							sizes.append(len(bi))
						f.seek(i + len(bm))
					while len(fs) < 8:
						b = await asubmit(f.read, CACHE_FILESIZE)
						if not b:
							break
						chunkf.append(b)
						fi = CompatFile(b, filename=fn2)
						fs.append(fi)
						sizes.append(len(b))
					if not fs:
						break
					c_id = choice([c_id for c_id, flag in self.data.items() if flag & 16])
					channel = await bot.fetch_channel(c_id)
					fstr = f"{fn.rsplit('/', 1)[-1]} ({i})"
					try:
						message = await channel.send(fstr, files=fs)
					except Exception:
						print(channel, c_id)
						print(sizes)
						print_exc()
						await asyncio.sleep(10)
						try:
							message = await channel.send(fstr, files=[CompatFile(b, filename=fn2) for b in chunkf])
						except:
							print_exc()
							f.seek(i)
							await asyncio.sleep(20)
							continue
					# message = await bot.send_as_webhook(channel, fstr, files=fs, username=m.display_name, avatar_url=best_url(m), recurse=False)
					for a, bs in zip(message.attachments, sizes):
						if raw:
							u = self.bot.preserve_as_long(channel.id, message.id, a.id, fn=a.url) + "?S=" + str(bs)
						else:
							u = self.bot.preserve_into(channel.id, message.id, a.id, fn=a.url) + "?S=" + str(bs)
						urls.append(u)
						# u = str(a.url).rstrip("&")
						# u += "?" if "?" not in u else "&"
						# u += "size=" + str(bs) + "&mid=" + str(message.id)
						# urls.append(u)
					# mids.append(message.id)
					i = f.tell()
				await asyncio.sleep(0.25)
		print(urls)
		esubmit(bot.clear_cache, priority=True)
		return urls

	async def delete(self, mids):
		bot = self.bot
		print("Delete:", mids)
		cids = [c_id for c_id, flag in self.data.items() if flag & 16]
		channels = []
		for cid in cids:
			channel = await bot.fetch_channel(cid)
			channels.append(channel)
		csubmit(self._delete(channels, mids))

	@tracebacksuppressor
	async def _delete(self, channels, mids):
		bot = self.bot
		deleted = []
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

	DEFAULT_LIMIT = 48 * 1048576
	async def get_lfs_channel(self, size=DEFAULT_LIMIT):
		bot = self.bot
		log_channels = set(getattr(c, "parent", c) for c in list(filter(bool, (bot.get_channel(cid) for cid in bot.data.logM.values()))) + list(filter(bool, (bot.get_channel(cid) for cid in bot.data.logU.values()))) if c.guild.filesize_limit >= size and bot.permissions_in(c).create_private_threads and bot.permissions_in(c).embed_links)
		log_channels2 = [c for c in log_channels if bot.owners[0] in c.guild._members and c.permissions_for(c.guild.get_member(bot.owners[0])).read_messages]
		log_channels = log_channels2 or log_channels
		if log_channels:
			channel = choice(log_channels)
			for thread in channel.threads:
				if thread.owner_id == bot.id:
					return thread
			async for thread in channel.archived_threads(private=True):
				if thread.owner_id == bot.id:
					return thread
			return await channel.create_thread(name="backup")
		raise NotImplementedError(size)

	async def mproxy(self, b, fn=None, channel=None, minimise=False):
		bot = self.bot
		b = MemoryBytes(b)
		groups = []
		total_size = len(b)
		if total_size > attachment_cache.max_size * 100:
			chunksize = self.DEFAULT_LIMIT
		elif attachment_cache.max_size < total_size <= self.DEFAULT_LIMIT:
			chunksize = self.DEFAULT_LIMIT
		else:
			chunksize = attachment_cache.max_size
		if chunksize <= attachment_cache.max_size:
			if not channel:
				channel = await bot.fetch_channel(choice(attachment_cache.channels))
			private = True
		else:
			channel = await self.get_lfs_channel(self.DEFAULT_LIMIT)
			if bot.owners.intersection(channel.guild._members) and none(m.id in bot.owners for m in channel.members):
				with tracebacksuppressor:
					await channel.add_user(bot.get_user(bot.owners[0]))
			private = False
		groupsize = min(10, 2 + int(total_size / chunksize))
		start = 0
		while start < total_size:
			if not groups or len(groups[-1]) >= groupsize:
				groups.append([])
			if start == 0:
				semi = ((total_size % chunksize) or chunksize)
				if semi >= 2097152:
					semi = semi + 1 >> 1
				chunk = b[start:start + semi]
				start += semi
			else:
				chunk = b[start:start + chunksize]
				start += chunksize
			groups[-1].append(chunk)
		ofn = fn
		n = 0
		mids = []
		futs = []
		for group in groups:
			files = []
			embeds = []
			for chunk in group:
				b = bytes(chunk)
				file = CompatFile(b, filename=fn or "c.b")
				if not private:
					member = choice(channel.guild.members)
					try:
						thumb = member.avatar.url
					except Exception:
						thumb = bot.discord_icon
				ext = magic.from_buffer(b) if not n else ""
				if private:
					embed = None
				elif ext.startswith("image/"):
					embed = discord.Embed(colour=rand_colour()).set_author(name=member.name, icon_url=f"attachment://{fn}").set_thumbnail(url=thumb)
				else:
					embed = discord.Embed(colour=rand_colour()).set_image(url=f"attachment://{fn}")
				files.append(file)
				if embed:
					embeds.append(embed)
				fn = str(n)
				n += 1
			futs.append(csubmit(channel.send(files=files, embeds=embeds)))
		for fut in futs:
			message = await fut
			assert not len(message.embeds) or len(message.embeds) == len(group), message.id
			mids.append(message.id)
		return shorten_chunks(chunksize // 1048576, channel.id, mids, ofn, mode="c", base="https://mizabot.xyz", minimise=minimise)

	async def lproxy(self, url, filename=None, channel=None, minimise=False):
		bot = self.bot
		if isinstance(url, byte_like):
			fn = filetransd(filename or "c.b")
			b = url
		elif is_url(url):
			fn = filetransd(filename or url2fn(url))
			b = await attachment_cache.download(url, read=True)
		elif isinstance(url, str):
			fn = filetransd((filename or url).replace("\\", "/").rsplit("/", 1)[-1])
			b = open(url, "rb")
		else:
			fn = filetransd(filename or getattr(url, "name", None) or "c.b")
			b = url
		if getsize(b) <= attachment_cache.max_size:
			return await attachment_cache.create(b, filename=fn, channel=channel, minimise=minimise)
		try:
			channel = await self.get_lfs_channel(getsize(b))
		except NotImplementedError:
			return await self.mproxy(b, fn, channel=channel, minimise=minimise)
		if bot.owners.intersection(channel.guild._members) and none(m.id in bot.owners for m in channel.members):
			with tracebacksuppressor:
				await channel.add_user(bot.get_user(bot.owners[0]))
		file = CompatFile(b, filename=fn)
		member = choice(channel.guild.members)
		try:
			thumb = member.avatar.url
		except Exception:
			thumb = bot.discord_icon
		ext = magic.from_buffer(b)
		if ext.startswith("image/"):
			embed = discord.Embed(colour=rand_colour()).set_author(name=member.name, icon_url=f"attachment://{fn}").set_thumbnail(url=thumb)
		else:
			embed = discord.Embed(colour=rand_colour()).set_image(url=f"attachment://{fn}")
		message = await channel.send(file=file, embed=embed)
		assert message.embeds, message.id
		out = shorten_attachment(message.embeds[0].author.icon_url if ext.startswith("image/") else message.embeds[0].image.url, message.id, minimise=minimise)
		print("LPROXY:", url, out)
		return out

	async def uproxy(self, *urls, collapse=True, mode="upload", filename=None, channel=None, optimise=False, **kwargs):
		bot = self.bot
		async def proxy_url(url):
			nonlocal filename
			uhu = None
			data = None
			if isinstance(url, byte_like):
				data = url
			elif isinstance(url, CompatFile):
				data = await asubmit(url.fp.read)
				filename = filename or url.filename
			elif not is_url(url):
				raise TypeError(url)
			elif url.startswith(bot.webserver + "/u/") or url.startswith(bot.raw_webserver + "/u/") or url.startswith("https://cdn.discordapp.com/embed/avatars/"):
				return url
			else:
				uhu = uuhash(url)
				try:
					url2 = bot.data.proxies[uhu]
				except KeyError:
					pass
				else:
					if is_discord_attachment(url2):
						url2 = bot.data.proxies[uhu] = shorten_attachment(url2, 0)
					return url2
				if mode == "raise":
					raise FileNotFoundError(url)
				elif mode == "download":
					return await attachment_cache.download(url)
				elif mode == "upload":
					pass
				else:
					return
			if optimise:
				if not data and is_url(url):
					data = await attachment_cache.download(url, filename=True)
				if data and getsize(data) > 1048576 and magic.from_file(data).split("/", 1)[0] in ("image", "video"):
					data = await bot.optimise_image(data, fsize=1048576, fmt="avif")
					filename = replace_ext(filename, "avif")
			url2 = await self.lproxy(data or url, filename=filename, channel=channel)
			if uhu:
				bot.data.proxies[uhu] = url2
			return url2
		if collapse and len(urls) == 1:
			return await proxy_url(urls[0])
		return await gather(*map(proxy_url, urls))

	def uregister(self, k, url, m_id=0):
		bot = self.bot
		uhu = uuhash(k)
		url = shorten_attachment(url, m_id)
		bot.data.proxies[uhu] = url
		return url
	
	def cproxy(self, url):
		if url in self.temp:
			return
		self.temp[url] = csubmit(self.uproxy(url))

	async def _bot_ready_(self, **void):
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
			await self.bot.ensure_webhook(mchannel, force=True)

	def _destroy_(self, **void):
		with suppress(LookupError, AttributeError):
			PRINT.funcs.discard(self._log_)


class UpdateProxies(Database):
	name = "proxies"
	limit = 65536


class SetAvatar(Command):
	name = ["ChangeAvatar", "UpdateAvatar"]
	min_level = nan
	description = "Changes ⟨BOT⟩'s current avatar."
	schema = cdict(
		url=cdict(
			type="visual",
			description="The new image to use as avatar",
			required=True,
		),
	)

	async def __call__(self, bot, url, **void):
		data = await bot.optimise_image(url, msize=10485760, fmt="gif")
		await bot.edit(avatar=data)
		return cdict(
			content=f"✅ Succesfully Changed {bot.user.name}'s avatar!",
			prefix="```css\n",
			suffix="```",
		)


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
		pl = self.prem_limit(lv)
		if len(d["gl"]) > pl:
			while len(d["gl"]) > pl:
				i = d["gl"].pop()
				if i in self.bot.data.trusted:
					self.bot.data.trusted[i].discard(uid)
					if not self.bot.data.trusted[i]:
						self.bot.data.trusted.pop(i, None)
				print(i, "subscription lost from", uid)
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
			return 1
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
		return rm


class UpdateCosts(Database):
	name = "costs"


class UpdateUsage(Database):
	name = "usage"

	def add(self, command):
		s = str(utc_dt().date())
		usage = self.setdefault(s, {})
		name = command if isinstance(command, str) else command.parse_name()
		add_dict(usage, {name: 1})


class UpdateColours(Database):
	name = "colours"
	no_file = True

	async def _get(self, url, threshold=True):
		resp = await asubmit(colour_cache.obtain, url)
		out = [round(i) for i in resp]
		try:
			raw = colour2raw(out)
		except Exception:
			print_exc()
			return 0
		if threshold:
			if raw == 0:
				return 1
			elif raw == 16777215:
				return 16777214
		return raw

	async def get(self, url, threshold=True):
		if not url:
			return 0
		resp = await self._get(url, threshold)
		if isinstance(resp, (discord.Colour, int)):
			return resp
		return 0


class UpdateChannelCache(Database):
	name = "channel_cache"
	channel = True

	async def grab(self, channel, as_message=True, force=False):
		if hasattr(channel, "simulated"):
			yield channel.message
			return
		c_id = verify_id(channel)
		min_time = time_snowflake(dtn() - datetime.timedelta(days=14))
		deletable = False
		s = self.get(c_id, ())
		if isinstance(s, set):
			s = self[c_id] = sorted(s, reverse=True)
		for m_id in s:
			if m_id in self.bot.data.deleted.cache:
				continue
			if as_message:
				try:
					if m_id < min_time:
						raise OverflowError
					message = await self.bot.fetch_message(m_id, channel=channel if force else None)
					if T(message).get("deleted"):
						continue
				except (discord.NotFound, discord.Forbidden, OverflowError):
					if deletable:
						self.data[c_id].remove(m_id)
				except (TypeError, ValueError, LookupError, discord.HTTPException):
					if not force:
						break
					print_exc()
				else:
					yield message
				deletable = True
			else:
				yield m_id

	async def splice(self, channel, messages):
		if not messages or hasattr(channel, "simulated"):
			return []
		c_id = verify_id(channel)
		iids = sorted(m.id for m in messages)
		mids = sorted(self.get(c_id, ()))
		i = bisect.bisect_left(mids, iids[0])
		j = bisect.bisect_right(mids, iids[-1])
		min_id = time_snowflake(cdict(timestamp=lambda: utc() - 14 * 86400))
		midl = mids[:i]
		if mids and mids[0] < min_id:
			k = bisect.bisect_right(mids, min_id)
			midl = midl[k:]
		mids = list(chain(midl, iids, mids[j:]))
		mids.reverse()
		self[c_id] = mids
		return mids

	def add(self, c_id, m_id):
		s = self.data.get(c_id, ())
		if not isinstance(s, set):
			s = set(s)
		s.add(m_id)
		while len(s) > 32768:
			try:
				s.discard(next(iter(s)))
			except RuntimeError:
				pass
		self[c_id] = s
	
	def _delete_(self, message, **void):
		try:
			self.data[message.channel.id].remove(message.id)
		except (AttributeError, KeyError, ValueError):
			pass


class UpdateDeleted(Database):
	name = "deleted"
	cache = AutoCache(stale=0, timeout=86400 * 7)
	no_file = True


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
		s = self.get(c_id, ())
		if isinstance(s, set):
			s = self[c_id] = sorted(s, reverse=True)
		for m_id in s:
			if as_message:
				try:
					if m_id < min_time:
						raise OverflowError
					message = await self.bot.fetch_message(m_id, channel=channel if force else None)
					if T(message).get("deleted"):
						continue
				except (discord.NotFound, discord.Forbidden, OverflowError):
					if deletable:
						self.data[c_id].remove(m_id)
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
		if not isinstance(s, set):
			s = set(s)
		s.add(m_id)
		while len(s) > 32768:
			try:
				s.discard(next(iter(s)))
			except RuntimeError:
				pass
		self[c_id] = s
	
	def _delete_(self, message, **void):
		try:
			self.data[message.channel.id].remove(message.id)
		except (AttributeError, KeyError, ValueError):
			pass


class Maintenance(Command):
	min_level = nan
	description = "Toggles Maintenance mode, which will block all use of commands for all servers except the current one while active."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("enable", "disable"),
			),
			description="New state to apply",
			example="disable",
		),
	)

	async def __call__(self, bot, _guild, mode, **void):
		if mode == "disable":
			bot.data.blacklist.pop(0)
			await bot.update_status(force=True)
			return css_md(f"Maintenance mode deactivated.")
		if mode == "enable":
			bot.data.blacklist[0] = _guild.id
			await bot.update_status(force=True)
			return css_md(f"Maintenance mode activated. No longer serving commands outside of {sqr_md(_guild)}.")
		maintenance = bot.data.blacklist.get(0)
		return css_md(f"Maintenance mode: {sqr_md(maintenance)}")


class Suspend(Command):
	name = ["Block", "Blacklist"]
	min_level = nan
	description = "Prevents a user from accessing ⟨BOT⟩'s commands. Overrides <perms>."
	usage = "<0:user> <disable(-d)>"
	example = ("block 201548633244565504",)
	flags = "aed"

	async def __call__(self, bot, user, args, flags, name, **void):
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
	sem2 = Semaphore(1, inf, rate_limit=1)
	sem3 = Semaphore(1, inf, rate_limit=1)
	emojidata = None

	async def load_own(self):
		while self.sem2.busy:
			await asyncio.sleep(1)
		if self.emojidata:
			return self.emojidata
		bot = self.bot
		async with self.sem2:
			emojidata = self.emojidata = await Request(
				f"https://discord.com/api/{api}/applications/{bot.id}/emojis",
				authorise=True,
				json=True,
				aio=True,
				timeout=32,
			)
			fns = set(chain.from_iterable(t[2] for t in os.walk("misc/emojis")))
			for edata in emojidata["items"]:
				name = edata["name"]
				for fn in fns:
					if fn.rsplit(".", 1)[0] == name:
						name = fn
						break
				emoji = discord.Emoji(guild=bot.user, state=bot._state, data=edata)
				# emoji.application_id = bot.id
				self.data[name] = emoji.id
				bot.cache.emojis[emoji.id] = emoji
		return self.emojidata

	def is_available(self, emoji):
		bot = self.bot
		if not emoji:
			return
		if bot.ready and getattr(emoji, "guild", None):
			if not getattr(emoji, "available", True) or emoji.id not in (e.id for e in emoji.guild.emojis):
				return False
		return True

	async def get_colour(self, rgb):
		bot = self.bot
		while not bot.bot_ready:
			await asyncio.sleep(2)

		if isinstance(rgb, int):
			rgb = raw2colour(rgb)
		c = 256 / 6
		rgb = [min(255, round(round(x / c) * c)) for x in rgb[:3]]
		name = "C_" + "".join(("0" + hex(x)[2:].upper())[-2:] for x in rgb)
		emojidata = await self.load_own()
		try:
			emoji = bot.cache.emojis[self.data[name]]
			if not self.is_available(emoji):
				raise KeyError
		except KeyError:
			pass
		else:
			return emoji
		if emojidata is not None and len(emojidata["items"]) < 2000:
			with open("misc/emojis/heart.svg", "rb") as f:
				b = await asubmit(f.read)
			b = await process_image(b, "replace_colour", [rgb, "-f", "webp"], timeout=60)
			b2 = await bot.to_data_url(b)
			async with self.sem3:
				edata = await Request(
					f"https://discord.com/api/{api}/applications/{bot.id}/emojis",
					method="POST",
					data=orjson.dumps(dict(
						name=name,
						image=b2,
					)),
					headers={
						"Content-Type": "application/json",
					},
					authorise=True,
					json=True,
					aio=True,
					timeout=32,
				)
			emoji = discord.Emoji(guild=bot.user, state=bot._state, data=edata)
			# emoji.application_id = bot.id
			self.data[name] = emoji.id
			bot.cache.emojis[emoji.id] = emoji
			self.emojidata = None
			return emoji
		raise LookupError("Unable to find space for the required emoji.")

	async def grab(self, name):
		bot = self.bot
		while not bot.bot_ready:
			await asyncio.sleep(2)

		ename = name.rsplit(".", 1)[0]
		animated = name.endswith(".gif")
		emojidata = await self.load_own()
		try:
			emoji = bot.cache.emojis[self.data[name]]
			if not self.is_available(emoji):
				raise KeyError
		except KeyError:
			pass
		else:
			return emoji
		guilds, limits = bot.get_available_guild(animated=animated, return_all=True)
		for guild in guilds:
			for emoji in guild.emojis:
				if emoji.name == ename and emoji.animated == animated:
					if not self.is_available(emoji):
						continue
					self.data[name] = emoji.id
					bot.cache.emojis[emoji.id] = emoji
					return emoji
		if emojidata is not None:
			for edata in emojidata["items"]:
				if edata["name"] == ename and edata["animated"] == animated:
					emoji = discord.Emoji(guild=bot.user, state=bot._state, data=edata)
					# emoji.application_id = bot.id
					self.data[name] = emoji.id
					bot.cache.emojis[emoji.id] = emoji
					return emoji
		if emojidata is not None and len(emojidata["items"]) < 2000:
			with open(f"misc/emojis/{name}", "rb") as f:
				b = await asubmit(f.read)
			b2 = await bot.to_data_url(b)
			async with self.sem3:
				edata = await Request(
					f"https://discord.com/api/{api}/applications/{bot.id}/emojis",
					method="POST",
					data=orjson.dumps(dict(
						name=ename,
						image=b2,
					)),
					headers={
						"Content-Type": "application/json",
					},
					authorise=True,
					json=True,
					aio=True,
					timeout=32,
				)
			emoji = discord.Emoji(guild=bot.user, state=bot._state, data=edata)
			# emoji.application_id = bot.id
			self.data[name] = emoji.id
			bot.cache.emojis[emoji.id] = emoji
			self.emojidata = None
			return emoji
		if not sum(limits):
			raise LookupError("Unable to find suitable guild for the required emoji.")
		with open(f"misc/emojis/{name}", "rb") as f:
			b = await asubmit(f.read)
		emoji = await guilds[0].create_custom_emoji(name=ename, image=b)
		self.data[name] = emoji.id
		bot.cache.emojis[emoji.id] = emoji
		return emoji

	async def emoji_as(self, s, full=False):
		e = await self.grab(s)
		return min_emoji(e, full=full)

	async def create_progress_bar(self, length, ratio):
		start_bar = await gather(*[self.emoji_as(f"start_bar_{i}.gif") for i in range(5)])
		mid_bar = await gather(*[self.emoji_as(f"mid_bar_{i}.gif") for i in range(5)])
		end_bar = await gather(*[self.emoji_as(f"end_bar_{i}.gif") for i in range(5)])
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
		self.finished = finished

	@tracebacksuppressor
	async def load_until(self, key, func, threshold, args=()):
		async with self.sem:
			data = self.data.coercedefault(key, alist, alist())
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
						else:
							failed += 1
				except:
					failed += 8
					print_exc()
			self.finished.add(key)
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
			return url

	async def get(self, key, func, threshold=1024, args=()):
		if key not in self.loading:
			self.loading.add(key)
			csubmit(self.load_until(key, func, threshold, args=args))
		data = self.coercedefault(key, alist, alist())
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
			return url
		if not self.sem.is_busy():
			csubmit(self.proc(key, func, args=args))
		return choice(data)


class UpdateAttachments(Database):
	name = "attachments"


class UpdateAnalysed(Database):
	name = "analysed"


class UpdateInsights(Database):
	name = "insights"


class UpdateInfo(Database):
	name = "info"


class UpdateOnceoffs(Database):
	name = "onceoffs"

	def use(self, key, timeout=60):
		t = utc()
		if t - self.get(key, 0) > timeout:
			self[key] = t
			return True
		return False


class UpdateEmojiStats(Database):
	name = "emojistats"

	def __load__(self, **void):
		self.bot.emoji_stuff = self


class UpdateGuildSettings(Database):
	name = "guildsettings"


class UpdateGuilds(Database):
	name = "guilds"
	forced = set()

	def cache_guild(self, guild):
		if T(guild).get("unavailable", False) or T(guild).get("ghost") or T(guild).get("simulated"):
			return guild._members.values()
		mdata = []
		ts = utc()
		for m in guild._members.values():
			m.guild = guild
			cm = cdict(
				name=m.name,
				nick=m.nick,
				global_name=T(m).get("global_name"),
				id=m.id,
				gp=m.guild_permissions.value,
				rids=list(m._roles),
			)
			if m.bot:
				cm.bot = True
			if m._avatar:
				cm._a = m._avatar
			tou = T(m).get("timed_out_until")
			if tou and ts - tou.timestamp() > 0:
				cm.tou = tou.timestamp()
			mdata.append(cm)
		gdata = cdict(
			id=guild.id,
			name=guild.name,
			icon=guild.icon and str(guild.icon),
			description=guild.description,
			_member_count=len(mdata),
			features=guild.features,
			banner=guild.banner and str(guild.banner),
			owner_id=guild.owner_id,
			filesize_limit=guild.filesize_limit,
			members=mdata,
		)
		self[guild.id] = gdata
		self.bot.cache.guilds[guild.id] = guild
		return mdata

	def __bot_ready__(self, **void):
		bot = self.bot
		for k, v in self.items():
			with tracebacksuppressor:
				g = self.get(k)
				if not isinstance(g, dict):
					continue
				if k not in bot.cache.guilds:
					guild = bot.UserGuild()
					guild.channel = bot.user
					guild._members = {}
					guild._roles = {}
					guild._channels = {}
					guild._threads = {}
					guild.channels = []
					guild.text_channels = []
					guild.voice_channels = []
					guild.categories = []
					guild.me = bot.user
					guild.__dict__.update(g)
				self.load_guild(guild)
				bot.cache.guilds[guild.id] = guild

	def load_guild(self, guild):
		bot = self.bot
		gdata = self.get(guild.id, {})
		if not isinstance(gdata, dict):
			gdata = dict(members=gdata)
		mdata = gdata["members"]
		for cm in map(cdict, mdata):
			if cm.id in guild._members and isinstance(guild._members[cm.id], discord.Member):
				continue
			m = bot.GhostUser()
			m.id = cm.id
			m.name = cm.name
			m.nick = cm.get("nick")
			m.global_name = cm.get("global_name")
			m.guild_permissions = discord.Permissions(cm.gp)
			m.guild = guild
			m.roles = list(filter(bool, map(guild._roles.get, cm.get("rids", ()))))
			if guild.id not in cm.get("rids", ()):
				r = guild._roles.get(guild.id) or discord.Role(guild=guild, state=T(bot).get("_state"), data=dict(id=guild.id, name="@everyone"))
				m.roles.append(r)
			m._roles = discord.utils.SnowflakeList([r.id for r in m.roles if r.id != guild.id])
			m.bot = cm.get("bot", False)
			m._avatar = T(cm).get("_a")
			if T(cm).get("tou"):
				m.timed_out_until = datetime.datetime.utcfromtimestamp(cm.tou).replace(tzinfo=datetime.timezone.utc)
			guild._members[m.id] = m
		if isinstance(guild, dict):
			guild["owner"] = guild._members.get(guild.owner_id)
		return guild._members

	def update_guild(self, guild):
		bot = self.bot
		if guild.id in bot.cache.guilds:
			members = bot.cache.guilds[guild.id]._members
			guild._members.update(members)
			guild._member_count = len(guild._members)
		bot.cache.guilds[guild.id] = guild
		return guild

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
