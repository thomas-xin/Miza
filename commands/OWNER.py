# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

try:
	import compression.zstd
except ImportError:
	compression = None


class Reload(Command):
	min_level = nan
	description = "Reloads a specified module."
	schema = cdict(
		reload=cdict(
			type="word",
			excludes=("unload",),
		),
		unload=cdict(
			type="word",
			excludes=("reload",),
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
						# save = csubmit(bot.send_event("_save_"))
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
		# if save is None:
		# 	print("Saving message cache...")
		# 	save = csubmit(bot.send_event("_save_"))
		ctx = discord.context_managers.Typing(_channel) if _channel else emptyctx
		async with Delay(1):
			async with ctx:
				# Call _destroy_ bot event to indicate to all databases the imminent shutdown
				print("Destroying database memory...")
				with tracebacksuppressor:
					await bot.send_event("_destroy_", shutdown=mode == "shutdown")
				with tracebacksuppressor:
					await asubmit(bot.handle_update, force=True, priority=True)
				# Save any database that has not already been autosaved
				print("Saving all databases...")
				with tracebacksuppressor:
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
					cid = bot.get_userbase(u.id, "last_channel")
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
			default="view",
			excludes=("code", "audio", "server"),
		),
		type=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(terminal_types.a),
			),
			description="Terminal mode",
			example="virtual",
			default="null",
			excludes=("code", "audio", "server"),
		),
		code=cdict(
			type="string",
			description="Code to evaluate directly in the main process",
			example="1 + 1",
			excludes=("mode", "type", "audio", "server"),
		),
		audio=cdict(
			type="string",
			description="Code to evaluate directly in the audio process",
			example="1 + 2",
			excludes=("mode", "type", "code", "server"),
		),
		server=cdict(
			type="string",
			description="Code to evaluate directly in the server process",
			example="2 + 2",
			excludes=("mode", "type", "audio", "code"),
		),
	)

	async def __call__(self, bot, _channel, _message, mode, type, code, audio, server, **void):
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
		if server:
			code = f"await bot.server.asubmit({repr(server)})"
		elif audio:
			code = f"await bot.audio.asubmit({repr(audio)})"
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
	virtuals = None
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
			glob = self.virtuals
			if glob is None:
				glob = self.virtuals = dict(bot._globals)
			glob.update(dict(
				user=message.author,
				member=message.author,
				message=message,
				channel=channel,
				guild=message.guild,
				print=lambda *args, **kwargs: self._print(*args, channel=channel, **kwargs),
				input=lambda *args, **kwargs: self._input(*args, channel=channel, **kwargs),
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

	async def aproxy(self, *urls):
		out = [self.bot.data.proxies.get(uuhash(url)) or url for url in urls]
		out = [attachment_cache.preserve(url, 0) for url in out]
		return out if len(out) > 1 else out[0]

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
				deleted.append(m)
				break
		await bot.autodelete(*deleted)
		deli = [m.id for m in deleted]
		print("Deleted:", deli)
		return deli

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
			try:
				async for thread in channel.archived_threads(private=True):
					if thread.owner_id == bot.id:
						return thread
			except discord.Forbidden:
				pass
			return await channel.create_thread(name="backup")
		raise NotImplementedError(size)

	async def mproxy(self, b, fn=None, channel=None, minimise=False):
		bot = self.bot
		b = MemoryBytes(b)
		groups = []
		total_size = len(b)
		if total_size > attachment_cache.max_size * 100:
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

	async def lproxy(self, url, filename=None, channel=None, minimise=False, allow_empty=True):
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
		if not allow_empty and not getsize(b):
			raise EOFError(b)
		if getsize(b) <= attachment_cache.max_size:
			return await attachment_cache.create(b, filename=fn, channel=channel, minimise=minimise)
		return await self.mproxy(b, fn, channel=channel, minimise=minimise)

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
					try:
						await attachment_cache.scan_headers(url2)
					except ConnectionError as ex:
						print(repr(ex))
					else:
						if is_discord_attachment(url2):
							url2 = bot.data.proxies[uhu] = attachment_cache.preserve(url2, 0)
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
					filename = replace_ext(filename or "Untitled", "avif")
			url2 = await self.lproxy(data or url, filename=filename, channel=channel, allow_empty=False)
			# print("UPROXY:", urls, filename, url2)
			if uhu:
				bot.data.proxies[uhu] = url2
			return url2
		if collapse and len(urls) == 1:
			return await proxy_url(urls[0])
		return await gather(*map(proxy_url, urls))

	def uregister(self, k, url, m_id=0):
		bot = self.bot
		uhu = uuhash(k)
		url = attachment_cache.preserve(url, m_id)
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
		resp = await colour_cache.obtain(url)
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
	no_file = True

	def remove(self, c_id, removed: list):
		if not removed:
			return
		bot = self.bot
		bot.channel_cache[c_id] = set(bot.channel_cache[c_id]).difference(removed)

	async def grab(self, channel, as_message=True, force=False):
		if hasattr(channel, "simulated"):
			yield channel.message
			return
		bot = self.bot
		c_id = verify_id(channel)
		min_time = time_snowflake(dtn() - datetime.timedelta(days=14))
		deletable = False
		s = bot.channel_cache.get(c_id, ())
		if isinstance(s, set):
			s = bot.channel_cache[c_id] = sorted(s, reverse=True)
		removed = []
		for m_id in s:
			if as_message:
				try:
					if m_id < min_time:
						self.remove(c_id, removed)
						raise OverflowError
					message = await bot.fetch_message(m_id, channel=channel if force else None)
					if T(message).get("deleted"):
						continue
				except (discord.NotFound, discord.Forbidden, OverflowError):
					removed.append(m_id)
				except (TypeError, ValueError, LookupError, discord.HTTPException):
					if not force:
						break
					print_exc()
				else:
					yield message
			else:
				yield m_id
		self.remove(c_id, removed)

	async def splice(self, channel, messages):
		if not messages or hasattr(channel, "simulated"):
			return []
		bot = self.bot
		c_id = verify_id(channel)
		iids = sorted(m.id for m in messages)
		mids = sorted(bot.channel_cache.get(c_id, ()))
		i = bisect.bisect_left(mids, iids[0])
		j = bisect.bisect_right(mids, iids[-1])
		min_id = time_snowflake(cdict(timestamp=lambda: utc() - 14 * 86400))
		midl = mids[:i]
		if mids and mids[0] < min_id:
			k = bisect.bisect_right(mids, min_id)
			midl = midl[k:]
		mids = list(chain(midl, iids, mids[j:]))
		mids.reverse()
		bot.channel_cache[c_id] = mids
		return mids

	def add(self, c_id, m_id):
		bot = self.bot
		s = bot.channel_cache.get(c_id, ())
		if isinstance(s, tuple):
			s = [m_id]
		elif isinstance(s, list):
			if m_id > s[0]:
				s.insert(0, m_id)
			elif m_id == s[0]:
				return
			else:
				s = set(s)
		if isinstance(s, set):
			if m_id in s:
				return
			s.add(m_id)
		bot.channel_cache[c_id] = s

	def _delete_(self, message, **void):
		bot = self.bot
		try:
			bot.channel_cache[message.channel.id].remove(message.id)
		except (AttributeError, KeyError, ValueError):
			pass


class UpdateMessageCache(Database):
	name = "message_cache"
	no_file = True
	checked = set()
	loader = FileHashDict(path=f"{CACHE_PATH}/message_cache_loader")

	async def load_messages(self, channel):
		if channel.id in self.checked:
			return
		bot = self.bot
		perms = bot.permissions_in(channel)
		if perms.read_messages and perms.read_message_history:
			pass
		else:
			return
		self.checked.add(channel.id)
		m_id = self.loader.get(channel.id, 0)
		last = max(
			time_snowflake(datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=14)),
			m_id,
		)
		async with bot.guild_semaphore:
			messages = []
			async for message in channel.history(after=last, limit=None, oldest_first=True):
				messages.append(message)
		if messages:
			await asubmit(self.store_messages, messages)
			self.loader[channel.id] = message.id

	def store_messages(self, messages):
		return [self.store_message(message) for message in messages]

	async def _send_(self, message, **void):
		return await self.load_messages(message.channel)

	def store_message(self, message):
		bot = self.bot
		m = getattr(message, "_data", None)
		if m:
			if "author" not in m:
				author = message.author
				m["author"] = dict(id=author.id, username=author.name, avatar=author.avatar if not author.avatar or isinstance(author.avatar, str) else author.avatar.key)
			elif not message.webhook_id:
				m["author"] = dict(id=m["author"]["id"], username=m["author"]["username"], avatar=m["author"]["avatar"])
			if "channel_id" not in m:
				try:
					m["channel_id"] = message.channel.id
				except AttributeError:
					return
		else:
			if message.channel is None:
				return
			author = message.author
			m = dict(
				author=dict(id=author.id, username=author.name, avatar=author.avatar if not author.avatar or isinstance(author.avatar, str) else author.avatar.key),
				channel_id=message.channel.id,
			)
			if message.content:
				m["content"] = readstring(message.content)
			mtype = T(message.type).get("value", message.type)
			if mtype:
				m["type"] = mtype
			flags = message.flags.value if message.flags else 0
			if flags:
				m["flags"] = flags
			for k in ("tts", "pinned", "mention_everyone", "webhook_id"):
				v = T(message).get(k)
				if v:
					m[k] = v
			edited_timestamp = as_str(T(message).get("_edited_timestamp") or "")
			if edited_timestamp:
				m["edited_timestamp"] = edited_timestamp
			reactions = []
			for reaction in message.reactions:
				if not reaction.is_custom_emoji():
					r = dict(emoji=dict(id=None, name=str(reaction.emoji)))
				else:
					ename, eid = str(reaction.emoji).rsplit(":", 1)
					eid = int(eid.removesuffix(">"))
					ename = ename.split(":", 1)[-1]
					r = dict(emoji=dict(id=eid, name=ename))
				if reaction.count != 1:
					r["count"] = reaction.count
				if reaction.me:
					r["me"] = reaction.me
				reactions.append(r)
			if reactions:
				m["reactions"] = reactions
			try:
				attachments = [dict(id=a.id, size=a.size, filename=a.filename, url=a.url, proxy_url=a.proxy_url) for a in message.attachments if T(a).get("size")]
			except AttributeError:
				print(message.id)
				raise
			if attachments:
				m["attachments"] = attachments
			embeds = [e.to_dict() for e in message.embeds]
			if embeds:
				m["embeds"] = embeds
		try:
			m["id"] = int(m["id"])
		except (KeyError, ValueError):
			m["id"] = message.id
		if not message.webhook_id:
			m.pop("member", None)
		m.pop("nonce", None)
		m.pop("timestamp", None)
		m.pop("referenced_message", None)
		for k in ("type", "attachments", "embeds", "components", "reactions", "channel_type", "edited_timestamp", "flags", "pinned", "mentions", "mention_roles", "mention_everyone", "tts", "deaf"):
			if not m.get(k):
				m.pop(k, None)
		data = orjson.dumps(m)
		if compression and len(data) > 1024:
			data2 = compression.zstd.compress(data, 10)
			if len(data2) < len(data) - 1:
				data = b"\x80" + data2
		bot.message_cache[m["id"]] = encrypt(data)
		bot.data.channel_cache.add(message.channel.id, message.id)
		return m

	def load_message(self, m_id):
		bot = self.bot
		data = decrypt(bot.message_cache[m_id])
		if data.startswith(b"\x80"):
			data = compression.zstd.decompress(data[1:])
		return bot.CachedMessage(orjson.loads(data))


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
			emojidata = self.emojidata = await Request.aio(
				f"https://discord.com/api/{api}/applications/{bot.id}/emojis",
				authorise=True,
				json=True,
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
			b = await read_file_a("misc/emojis/heart.svg")
			b = await process_image(b, "replace_colour", [rgb, "-f", "webp"], timeout=60)
			b2 = await bot.to_data_url(b)
			async with self.sem3:
				edata = await Request.aio(
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
			b = await read_file_a(f"misc/emojis/{name}")
			b2 = await bot.to_data_url(b)
			async with self.sem3:
				edata = await Request.aio(
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
		b = await read_file_a(f"misc/emojis/{name}")
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


class UpdateAnalysed(Database):
	name = "analysed"


class UpdateGuildSettings(Database):
	name = "guildsettings"


class UpdateDrives(Database):
	name = "drives"


class UpdateAccounts(Database):
	name = "accounts"


class UpdateSessions(Database):
	name = "sessions"
