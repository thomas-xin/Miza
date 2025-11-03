# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


help_colours = fcdict({
	None: 0xfffffe,
	"main": 0xff0000,
	"string": 0x00ff00,
	"admin": 0x0000ff,
	"voice": 0xff00ff,
	"image": 0xffff00,
	"webhook": 0xff007f,
	"fun": 0x00ffff,
	"ai": 0x007fff,
	"owner": 0xbf7fff,
	"nsfw": 0xff9f9f,
	"misc": 0x007f00,
})
help_emojis = fcdict((
	(None, "♾"),
	("main", "🌐"),
	("string", "🔢"),
	("admin", "🕵️‍♀️"),
	("voice", "🎼"),
	("image", "🖼"),
	("webhook", "🕸️"),
	("fun", "🙃"),
	("ai", "🤖"),
	("owner", "💜"),
	("nsfw", "🔞"),
	("misc", "🌌"),
))
help_descriptions = fcdict((
	(None, "N/A"),
	("main", "General commands, mostly involves individual users"),
	("string", "Text-based commands, usually helper tools"),
	("admin", "Moderation-based, used to help staffing servers"),
	("voice", "Play, convert, or download songs through VC"),
	("image", "Create or edit images, animations, and videos"),
	("webhook", "Webhook-related commands and management"),
	("fun", "Text-based games and other fun features"),
	("ai", "Commands mostly focused on generative artificial intelligence"),
	("owner", "Restricted owner-only commands; highly volatile"),
	("nsfw", "Not Safe For Work; only usable in 18+ channels"),
	("misc", "Miscellaneous, mostly not relevant to Discord and restricted to trusted servers"),
))

class Help(PaginationCommand):
	name = ["❓", "❔", "?", "Halp"]
	description = "Shows a list of usable commands, or gives a detailed description of a command."
	cats = [c for c in sorted(standard_commands)]
	schema = cdict(
		category=cdict(
			type="enum",
			validation=cdict(
				enum=cats,
				accepts=dict(owner="owner", misc="misc", nsfw="nsfw"),
			),
			description="Command category to search",
			example="image",
		),
		command=cdict(
			type="string",
			description="Help on a specific command",
			example="download",
			greedy=False,
		),
	)
	usage = f"<category({cats})>? <command>?"
	rate_limit = (0.25, 1)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _message, _channel, _user, category, command, **void):
		if command and command not in bot.commands:
			raise KeyError(f'Command "{command}" does not exist.')
		# Set callback message for scrollable list
		return await self.display(_user.id, bool(getattr(_message, "slash", None)), _channel.id, category, command)

	async def display(self, uid, is_slash, cid, category, command):
		bot = self.bot
		user = await bot.fetch_user(uid)
		channel = await bot.fetch_channel(cid)
		guild = channel.guild

		prefix = "/" if is_slash else bot.get_prefix(guild)
		if " " in prefix:
			prefix += " "
		embed = discord.Embed(
			title=f"{user} has asked for help!",
		).set_author(name="❓ Help ❓", icon_url=best_url(user), url=bot.webserver)

		content = None
		enabled = bot.get_enabled(channel)
		nsfw = bot.is_nsfw(channel, user)

		def category_repr(catg):
			return (catg.capitalize() if len(catg) > 2 else catg.upper()) + (" [DISABLED]" if catg.lower() not in enabled else "")

		if command:
			com = bot.commands[command][0]
			a = ", ".join(n.strip("_") for n in com.name) or "[none]"
			content = (
				f"[[Category]] {colourise(category_repr(com.category), fg='white')}\n"
				+ f"[[Aliases]] {a}\n"
				+ f"[[Description]] {colourise(com.parse_description(), fg='white')}\n"
				+ f"[[Usage]] {colourise(prefix, fg='white')}{com.parse_usage()}\n"
				+ f"[[Level]] {colourise(com.min_display)}"
			)
			rl = com.rate_limit
			if rl:
				rl = rl[bool(bot.is_trusted(guild))] if isinstance(rl, (tuple, list)) else rl
				pm = bot.premium_multiplier(bot.premium_level(user))
				rl /= pm
				content += f"\n[[Rate Limit]] {colourise(sec2time(rl), fg='yellow')}"
			content += f"\n[[Example]] {colourise(prefix, fg='white')}{com.parse_example()}"
			if not com.schema or getattr(com, "maintenance", False):
				content += colourise("\nNote: This command may currently be under maintenance. Full functionality is not guaranteed!", fg='red')
			content = ansi_md(content)
		else:
			content = (
				"Yo! Use the menu below to select from my command list!\n"
				+ f"Alternatively, visit [`mizatlas`]({bot.webserver}/mizatlas) for a full command list and tester.\n\n"
				+ f"If you're an admin and wish to disable me in a particular channel, check out `{prefix}ec`!\n"
				+ f"Want to try the premium features, unsure about anything, or have a bug to report? check out the [`support server`]({bot.rcc_invite})!\n"
				+ f"Finally, donate to me or purchase a subscription [`here`]({bot.kofi_url})! Any support is greatly appreciated!"
			)
		embed.description = content
		embed.colour = discord.Colour(help_colours[category])
		categories = visible_commands if nsfw else standard_commands
		if not category:
			coms = chain.from_iterable(v for k, v in bot.categories.items() if k in categories)
		else:
			coms = bot.categories[category]
		coms = sorted(coms, key=lambda c: c.parse_name())
		catsel = [cdict(
			emoji=cdict(id=None, name=help_emojis[c]),
			label=category_repr(c),
			value=c,
			description=help_descriptions[c],
			default=category == c,
		) for c in categories if c in bot.categories]
		comsel = [cdict(
			emoji=cdict(id=None, name=c.emoji) if getattr(c, "emoji", None) else None,
			label=lim_str(prefix + " " * (" " in prefix) + c.parse_name(), 25, mode=None),
			value=c.parse_name().casefold(),
			description=lim_str(c.parse_description(), 50, mode=None),
			default=command and com == c,
		) for i, c in enumerate(coms) if i < 25]
		catmenu = cdict(
			type=3,
			custom_id="\x7f0",
			options=catsel,
			min_values=0,
			placeholder=category_repr(category) if category else "Choose a category...",
		)
		commenu = cdict(
			type=3,
			custom_id="\x7f1",
			options=comsel,
			min_values=0,
			placeholder=com.parse_name() if command else "Choose a command...",
		)
		buttons = [[catmenu], [commenu]]
		return self.construct(user.id, bytes([is_slash]) + leb128(cid) + (category or "").encode("utf-8") + b"\x00" + (command or "").encode("utf-8"), embed=embed, buttons=buttons)

	async def _callback_(self, _message, _user, data, reaction, **void):
		is_slash, more = data[0], data[1:]
		cid, more = decode_leb128(more)
		bot = self.bot
		category = as_str(reaction)
		if not category:
			category, command = map(as_str, more.split(b"\x00", 1))
		if category in bot.commands:
			command = category
			category = bot.commands[command][0].category
		elif category in bot.categories:
			command = None
		else:
			return await bot.ignore_interaction(_message)

		return await self.display(_user.id, is_slash, cid, category, command)


class Hello(Command):
	name = ["Hi", "👋", "Bye"]
	description = "👋"
	schema = cdict()
	rate_limit = (0, 0)
	ephemeral = True

	def __call__(self, **void):
		return "👋"


class Loop(Command):
	time_consuming = 3
	_timeout_ = 12
	min_level = 1
	min_display = "1+"
	description = "Loops a command. Delete the original message to terminate the loop if necessary. Subject to regular command restrictions and rate limits."
	schema = cdict(
		iterations=cdict(
			type="integer",
			validation="[1, 256]",
			description="Amount of times to perform target command",
			example="4",
		),
		command=cdict(
			type="string",
			description="Command and arguments to run",
			example="~cat",
		),
	)
	rate_limit = (10, 15)
	recursive = True

	async def __call__(self, bot, _channel, _message, _perm, iterations, command, **void):
		if not iterations:
			# Ah yes, I made this error specifically for people trying to use this command to loop songs 🙃
			raise ArgumentError("Please input loop iterations using the -i flag, followed by target command. For looping songs in voice, consider using the aliases LoopQueue and Repeat under the AudioState command.")
		# Bot owner bypasses restrictions
		if not isnan(_perm):
			if _channel.id in self.active:
				raise PermissionError("Only one loop may be active in a channel at any time.")
		content = f"Running `{no_md(command)}` {iterations} time{'s' if iterations != 1 else ''}..."
		message = await send_with_reply(_channel, _message, content)
		manager = await bot.StreamedMessage.attach(message)
		await bot.require_integrity(_message)
		fake_message = copy.copy(_message)
		fake_message.content = command
		done = deque()
		futs = deque()
		for i in range(iterations):
			async for command, command_check, argv, from_mention in bot.parse_command(fake_message):
				while len(futs) >= 3:
					fut = futs.popleft()
					await fut
					done.append(fut)
				fut = csubmit(bot.run_command(command, message=fake_message, argv=argv, command_check=command_check, respond=False, allow_recursion=False))
				futs.append(fut)
		response = None
		t = 0
		for fut in chain(done, futs):
			await bot.require_integrity(_message)
			try:
				response = await fut
			# Represents any timeout error that occurs
			except (T0, T1, T2):
				print(command, argv)
				raise TimeoutError("Request timed out.")
			if fut:
				await fut
			if not response:
				continue
			done = i >= iterations - 1
			t2 = utc()
			if done or t2 - t > 1:
				fut = csubmit(bot.respond_with(response, message=fake_message, command=command, manager=manager, done=done))
			continue
		assert response, "No response was captured. (Make sure you inputted the command correctly!)"
		if fut:
			await fut
		await bot.edit_message(message, content=message.content + " (done!)")


class Edit(Command):
	time_consuming = 3
	_timeout_ = 12
	min_level = 1
	min_display = "1+"
	description = "Edits an existing bot message using new output from a command. Subject to regular command restrictions and rate limits."
	schema = cdict(
		target=cdict(
			type="message",
			description="Target message; must be visible to the bot, and must use full message links if older than 2 weeks.",
			example="https://discord.com/channels/247184721262411776/247184721262411776/803501099633737758",
		),
		command=cdict(
			type="string",
			description="Command and arguments to run",
			example="~cat",
		),
	)
	rate_limit = (10, 15)
	recursive = True

	async def __call__(self, bot, _channel, _message, _user, _perm, target, command, **void):
		if target.author.id != bot.id:
			raise PermissionError("Target message must belong to me.")
		if _perm < 3:
			if not getattr(target, "reference", None):
				raise self.perm_error(_perm, 3, "to edit commands not associated to your user")
			reference = await bot.fetch_reference(target)
			if reference.author.id != _user.id:
				raise self.perm_error(_perm, 3, "to edit commands not associated to your user")
		content = f"Running `{no_md(command)}` on {target.jump_url}..."
		message = await send_with_reply(_channel, _message, content)
		manager = await bot.StreamedMessage.attach(target, replace=True)
		fake_message = copy.copy(_message)
		fake_message.content = command
		response = None
		async for command, command_check, argv, from_mention in bot.parse_command(fake_message):
			await bot.require_integrity(_message)
			try:
				response = await bot.run_command(command, message=fake_message, argv=argv, command_check=command_check, respond=False, allow_recursion=True)
			# Represents any timeout error that occurs
			except (T0, T1, T2):
				print(command, argv)
				raise TimeoutError("Request timed out.")
			await bot.respond_with(response, message=fake_message, command=command, manager=manager)
		assert response, "No response was captured. (Make sure you inputted the command correctly!)"
		await bot.edit_message(message, content=message.content + " (done!)")


class Pipe(Command):
	time_consuming = 3
	_timeout_ = 12
	min_level = 1
	min_display = "1+"
	description = "Redirects the output of one command into another. Subject to regular command restrictions and rate limits."
	schema = cdict(
		pipeline=cdict(
			type="string",
			description="Pipeline syntax, specified as `first` | `second` | `third` etc",
			example="~gradient spiral (0,255,255) (255,255,255) --repetitions 5 --space hsv | ~rainbow",
		),
	)
	rate_limit = (10, 15)
	recursive = True
	maintenance = True

	async def __call__(self, bot, _channel, _message, _user, _perm, pipeline, **void):
		if not pipeline:
			raise ArgumentError("Please input at least one target command.")
		pipe = [s.strip() for s in pipeline.split("|")]
		run = " | ".join(f"`{cmd}`" for cmd in pipe)
		content = f"Running {run}..."
		message = await send_with_reply(_channel, _message, content)
		fake_message = copy.copy(_message)
		fake_message.content = pipe.pop(0)
		response = None
		async for command, command_check, argv, from_mention in bot.parse_command(fake_message):
			await bot.require_integrity(_message)
			try:
				response = await bot.run_command(command, message=fake_message, argv=argv, command_check=command_check, respond=False, allow_recursion=True)
			# Represents any timeout error that occurs
			except (T0, T1, T2):
				print(command, argv)
				raise TimeoutError("Request timed out.")
		while pipe:
			second = pipe.pop(0)
			if response:
				inter = await bot.respond_with(response, message=fake_message, command=command)
				print("RW:", inter)
				original = inter.content
				fake_message = copy.copy(inter)
				fake_message.content = second + "\n" + json.dumps(original)
				manager = await bot.StreamedMessage.attach(fake_message)
			else:
				original = ""
				fake_message.content = second
				manager = None
			if second:
				async for command, command_check, argv, from_mention in bot.parse_command(fake_message):
					await bot.require_integrity(_message)
					try:
						response = await bot.run_command(command, message=fake_message, argv=argv, command_check=command_check, respond=False, allow_recursion=True)
					# Represents any timeout error that occurs
					except (T0, T1, T2):
						print(command, argv)
						raise TimeoutError("Request timed out.")
				if response:
					fake_message.content = original
					await bot.respond_with(response, message=fake_message, command=command, manager=manager)
		assert response, "No response was captured. (Make sure you inputted the command correctly!)"
		await bot.edit_message(message, content=message.content + " (done!)")


class Avatar(Command):
	name = ["PFP", "Icon"]
	description = "Sends a link to the avatar of a user or server."
	usage = "<objects>*"
	example = ("icon 247184721262411776", "avatar bob", "pfp")
	rate_limit = (5, 7)
	multi = True
	slash = True
	ephemeral = True
	exact = False

	async def getGuildData(self, g):
		# Gets icon display of a server and returns as an embed.
		url = best_url(g)
		name = g.name
		colour = await self.bot.get_colour(g)
		emb = discord.Embed(
			description=f"{sqr_md(name)}({url})",
			colour=colour,
		).set_thumbnail(url=url).set_image(url=url).set_author(name=name, icon_url=url, url=url)
		return emb

	async def getMimicData(self, p):
		# Gets icon display of a mimic and returns as an embed.
		url = best_url(p)
		name = p.name
		colour = await self.bot.get_colour(p)
		emb = discord.Embed(
			description=f"{sqr_md(name)}({url})",
			colour=colour,
		).set_thumbnail(url=url).set_image(url=url).set_author(name=name, icon_url=url, url=url)
		return emb

	async def __call__(self, argv, argl, channel, guild, bot, user, message, **void):
		iterator = argl if argl else (argv,)
		embs = set()
		for argv in iterator:
			with self.bot.ExceptionSender(channel):
				with suppress(StopIteration):
					if argv:
						if is_url(argv) or argv.startswith("discord.gg/"):
							g = await bot.fetch_guild(argv)
							emb = await self.getGuildData(g)
							embs.add(emb)
							raise StopIteration
						u_id = argv
						with suppress():
							u_id = verify_id(u_id)
						u = guild.get_member(u_id)
						g = None
						while u is None and g is None:
							with suppress():
								u = bot.get_member(u_id, guild)
								break
							with suppress():
								try:
									u = bot.get_user(u_id)
								except:
									if not bot.in_cache(u_id):
										u = await bot.fetch_user(u_id)
									else:
										raise
								break
							if type(u_id) is str and "@" in u_id and ("everyone" in u_id or "here" in u_id):
								g = guild
								break
							try:
								p = bot.get_mimic(u_id, user)
								emb = await self.getMimicData(p)
								embs.add(emb)
							except:
								pass
							else:
								raise StopIteration
							with suppress():
								g = bot.cache.guilds[u_id]
								break
							with suppress():
								g = bot.cache.roles[u_id].guild
								break
							with suppress():
								g = bot.cache.channels[u_id].guild
							with suppress():
								u = await bot.fetch_member_ex(u_id, guild)
								break
							raise LookupError(f"No results for {argv}.")     
						if g:
							emb = await self.getGuildData(g)    
							embs.add(emb)   
							raise StopIteration         
					else:
						u = user
					name = getattr(u, "name", None) or str(u)
					url = await self.bot.get_proxy_url(u, force=True)
					colour = await self.bot.get_colour(u)
					url2 = await self.bot.get_proxy_url(str(u.avatar), force=True)
					emb = discord.Embed(colour=colour)
					emb.set_thumbnail(url=url2)
					emb.set_image(url=url)
					emb.set_author(name=name, icon_url=url, url=url)
					emb.description = f"{sqr_md(getattr(u, 'display_name', None) or name)}({url})"
					embs.add(emb)
		bot.send_embeds(channel, embeds=embs, reference=message)


class Info(Command):
	name = ["🔍", "🔎", "UserInfo", "ServerInfo", "WhoIs"]
	description = "Shows information about the target user or server."
	usage = "<user>* <verbose(-v)>?"
	example = ("info 201548633244565504", "info")
	flags = "v"
	rate_limit = (6, 9)
	multi = True
	slash = True
	ephemeral = True
	usercmd = True
	exact = False

	async def getGuildData(self, g, flags={}, is_current=False):
		bot = self.bot
		url = await bot.get_proxy_url(g, force=True)
		name = g.name
		try:
			u = g.owner
		except (AttributeError, KeyError):
			u = None
		colour = await self.bot.get_colour(g)
		emb = discord.Embed(colour=colour).set_thumbnail(url=url).set_author(name=name, icon_url=url, url=url)
		if u is not None:
			d = user_mention(u.id)
		else:
			d = ""
		if g.description:
			d += code_md(g.description)
		lv = bot.is_trusted(g)
		if lv > 0:
			d += f"\n{bot.name} Premium Upgraded Lv{lv} " + "💎" * lv
			if lv < 2:
				d += f"; Visit {bot.kofi_url} to upgrade!"
		elif is_current:
			d += f"\nNo {bot.name} Premium Upgrades! Visit {bot.kofi_url} for more info!"
		emb.description = d
		emb.add_field(name="Server ID", value=str(g.id), inline=0)
		emb.add_field(name="Creation time", value=time_repr(g.created_at), inline=1)
		if "v" in flags:
			with suppress(AttributeError, KeyError):
				emb.add_field(name="Region", value=str(g.region), inline=1)
				emb.add_field(name="Nitro boosts", value=str(g.premium_subscription_count), inline=1)
		with suppress(AttributeError):
			x = len(g.channels)
			t = len(g.text_channels)
			t2 = len(g._threads)
			v = len(voice_channels(g))
			c = len(g.categories)
			channelinfo = f"Text: {t}\nThread: {t2}\nVoice: {v}\nCategory: {c}"
			if x > t + v + c:
				channelinfo += f"\nOther: {x - (t + v + c)}"
			emb.add_field(name=f"Channels ({x + t2})", value=channelinfo, inline=1)
		try:
			a = r = 0
			m = len(g._members)
			for member in g.members:
				if member.guild_permissions.administrator:
					a += 1
				else:
					r += bool(standard_roles(member))
			memberinfo = f"Admins: {a}\nOther roles: {r}\nNo roles: {m - a - r}"
			emb.add_field(name=f"Member count ({m})", value=memberinfo, inline=1)
		except AttributeError:
			if getattr(g, "member_count", None):
				m = g.member_count
				emb.add_field(name=f"Member count ({m})", value="N/A", inline=1)
		with suppress(AttributeError):
			r = len(g._roles)
			a = sum(1 for r in g._roles.values() if r.permissions.administrator and not r.is_default())
			roleinfo = f"Admins: {a}\nOther: {r - a}"
			emb.add_field(name=f"Role count ({r})", value=roleinfo, inline=1)
		with suppress(AttributeError):
			c = len(g.emojis)
			a = sum(getattr(e, "animated", False) for e in g.emojis)
			emojiinfo = f"Animated: {a}\nRegular: {c - a}"
			emb.add_field(name=f"Emoji count ({c})", value=emojiinfo, inline=1)
		return emb

	async def getMimicData(self, p, flags={}):
		url = to_webp(p.url)
		name = p.name
		colour = await self.bot.get_colour(p)
		emb = discord.Embed(colour=colour)
		emb.set_thumbnail(url=url)
		emb.set_author(name=name, icon_url=url, url=url)
		d = f"{user_mention(p.u_id)}{fix_md(p.id)}"
		if p.description:
			d += code_md(p.description)
		emb.description = d
		emb.add_field(name="Mimic ID", value=str(p.id), inline=0)
		emb.add_field(name="Name", value=str(p.name), inline=0)
		emb.add_field(name="Prefix", value=str(p.prefix), inline=1)
		emb.add_field(name="Creation time", value=time_repr(p.created_at), inline=1)
		if "v" in flags:
			emb.add_field(name="Gender", value=str(p.gender), inline=1)
			ctime = DynamicDT.utcfromtimestamp(p.birthday)
			age = (DynamicDT.utcnow() - ctime).total_seconds() / TIMEUNITS["year"]
			emb.add_field(name="Birthday", value=str(ctime), inline=1)
			emb.add_field(name="Age", value=str(round_min(round(age, 1))), inline=1)
		return emb

	async def __call__(self, argv, argl, name, guild, channel, bot, user, message, flags, **void):
		iterator = argl if argl else (argv,)
		embs = set()
		for argv in iterator:
			if argv.startswith("<") and argv[-1] == ">":
				argv = argv[1:-1]
			with self.bot.ExceptionSender(channel):
				with suppress(StopIteration):
					if argv:
						if is_url(argv) or argv.startswith("discord.gg/"):
							g = await bot.fetch_guild(argv)
							emb = await self.getGuildData(g, flags, is_current=guild.id == g.id)
							embs.add(emb)
							raise StopIteration
						u_id = argv
						with suppress():
							u_id = verify_id(u_id)
						u = guild.get_member(u_id) if type(u_id) is int else None
						g = None
						while u is None and g is None:
							with suppress():
								u = bot.get_member(u_id, guild)
								break
							with suppress():
								try:
									u = bot.get_user(u_id)
								except:
									if not bot.in_cache(u_id):
										u = await bot.fetch_user(u_id)
									else:
										raise
								break
							if type(u_id) is str and "@" in u_id and ("everyone" in u_id or "here" in u_id):
								g = guild
								break
							if "server" in name:
								with suppress():
									g = await bot.fetch_guild(u_id)
									break
								with suppress():
									role = await bot.fetch_role(u_id, g)
									g = role.guild
									break
								with suppress():
									channel = await bot.fetch_channel(u_id)
									g = channel.guild
									break
							try:
								p = bot.get_mimic(u_id, user)
								emb = await self.getMimicData(p, flags)
								embs.add(emb)
							except:
								pass
							else:
								raise StopIteration
							with suppress():
								g = bot.cache.guilds[u_id]
								break
							with suppress():
								g = bot.cache.roles[u_id].guild
								break
							with suppress():
								g = bot.cache.channels[u_id].guild
							with suppress():
								u = await bot.fetch_member_ex(u_id, guild)
								break
							raise LookupError(f"No results for {argv}.")
						if g:
							emb = await self.getGuildData(g, flags)
							embs.add(emb)
							raise StopIteration
					elif "server" not in name:
						u = user
					else:
						if not hasattr(guild, "ghost"):
							emb = await self.getGuildData(guild, flags)
							embs.add(emb)
							raise StopIteration
						else:
							u = bot.user
					u = await bot.fetch_user_member(u.id, guild)
					member = guild.get_member(u.id)
					name = getattr(u, "name", None) or str(u)
					url = await bot.get_proxy_url(u, force=True)
					st = deque()
					if u.id == bot.id:
						st.append("Myself 🙃")
						is_self = True
					else:
						is_self = False
					if bot.is_owner(u.id):
						st.append("My owner ❤️")
					deleted = False
					with suppress(LookupError):
						deleted = bot.data.users[u.id]["deleted"]
					if deleted:
						st.append("Deleted User ⚠️")
					is_sys = False
					if getattr(u, "system", None):
						st.append("Discord System ⚙️")
						is_sys = True
					lv = bot.premium_level(u)
					lv2 = bot.premium_level(u, absolute=True)
					if not isfinite(lv2):
						pass
					elif lv2 > 0:
						st.append(f"{bot.name} Premium Supporter Lv{lv2} " + "💎" * lv2)
					elif lv > 0:
						st.append(f"{bot.name} Trial Supporter Lv{lv} " + "💎" * lv)
					uf = getattr(u, "public_flags", None)
					is_bot = False
					if uf:
						if uf.system and not is_sys:
							st.append("Discord System ⚙️")
						if uf.staff:
							st.append("Discord Staff 👮")
						if uf.partner:
							st.append("Discord Partner 🎀:")
						if uf.bug_hunter_level_2:
							st.append("Bug Hunter Lv.2 🕷️")
						elif uf.bug_hunter:
							st.append("Bug Hunter 🐛")
						is_hype = False
						if uf.hypesquad_bravery:
							st.append("HypeSquad Bravery 🛡️")
							is_hype = True
						if uf.hypesquad_brilliance:
							st.append("HypeSquad Brilliance 🌟")
							is_hype = True
						if uf.hypesquad_balance:
							st.append("HypeSquad Balance 💠")
							is_hype = True
						if uf.hypesquad and not is_hype:
							st.append("HypeSquad 👀")
						if uf.early_supporter:
							st.append("Discord Early Supporter 🌄")
						if uf.team_user:
							st.append("Discord Team User 🧑‍🤝‍🧑")
						if uf.verified_bot:
							st.append("Verified Bot 👾")
							is_bot = True
						if uf.verified_bot_developer:
							st.append("Verified Bot Developer 🏆")
					if u.bot and not is_bot:
						st.append("Bot 🤖")
					if u.id == guild.owner_id and not hasattr(guild, "ghost"):
						st.append("Server owner 👑")
					if member:
						dname = getattr(member, "nick", None)
						joined = getattr(u, "joined_at", None)
					else:
						dname = getattr(u, "simulated", None) and getattr(u, "nick", None)
						joined = None
					created = u.created_at
					if member:
						rolelist = [role_mention(i.id) for i in reversed(getattr(u, "roles", ())) if not i.is_default()]
						role = ", ".join(rolelist)
					else:
						role = None
					seen = None
					zone = None
					with suppress(LookupError):
						ls = bot.data.users[u.id]["last_seen"]
						la = bot.data.users[u.id].get("last_action")
						if type(ls) is str:
							seen = ls
						else:
							seen = time_repr(ls, mode="R")
						if la:
							seen = f"{la}, {seen}"
						if "v" in flags:
							tz = self.bot.data.users.get_timezone(u.id)
							if tz is None:
								tz, c = self.bot.data.users.estimate_timezone(u.id)
								estimated = True
							else:
								estimated = False
							if tz >= 0:
								zone = f"GMT+{tz}"
							else:
								zone = f"GMT{tz}"
					if is_self and bot.webserver:
						url2 = bot.webserver
					else:
						url2 = url
					colour = await self.bot.get_colour(u)
					emb = discord.Embed(colour=colour)
					emb.set_thumbnail(url=url)
					emb.set_author(name=name, icon_url=url, url=url2)
					d = user_mention(u.id)
					if st:
						if d[-1] == "*":
							d += " "
						d += " **```css\n"
						if st:
							d += "\n".join(st)
						d += "```**"
					emb.description = d
					emb.add_field(name="User ID", value="`" + str(u.id) + "`", inline=0)
					emb.add_field(name="Creation time", value=time_repr(created), inline=1)
					if joined:
						emb.add_field(name="Join time", value=time_repr(joined), inline=1)
					if zone:
						tn = "Estimated timezone" if estimated else "Timezone"
						emb.add_field(name=tn, value=str(zone), inline=1)
					if seen:
						emb.add_field(name="Last seen", value=str(seen), inline=1)
					if dname:
						emb.add_field(name="Nickname", value=dname, inline=1)
					tag = getattr_chain(u, "primary_guild.tag", None)
					if tag:
						emb.add_field(name="Server Tag", value=tag, inline=1)
					if role:
						emb.add_field(name=f"Roles ({len(rolelist)})", value=role, inline=0)
					embs.add(emb)
		bot.send_embeds(channel, embeds=embs, reference=message)


class Profile(Command):
	name = ["User", "UserProfile"]
	description = "Shows or edits a user profile on ⟨BOT⟩."
	schema = cdict(
		user=cdict(
			type="user",
			description="Target user to retrieve profile from",
			example="201548633244565504",
		),
		description=cdict(
			type="text",
			description="Profile description. Will show up in embeds",
			example="You should play Cave Story, it's a good game",
		),
		pronouns=cdict(
			type="text",
			description="Optional field containing any text; will show above description when set",
			example="he/she/they",
		),
		website=cdict(
			type="url",
			description="Optional field denoting a redirect when the profile is clicked on",
			example="https://mizabot.xyz",
		),
		icon=cdict(
			type="visual",
			description="Thumbnail to display on corner of embeds",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
		),
		timezone=cdict(
			type="text",
			description='Timezone. Used in datetime calculations where applicable; supports both city/state names (e.g. "London") and abbreviations (e.g. "AET"). Daylight savings is automatically calculated for non-fixed timezones',
			example="US/Eastern",
		),
		birthday=cdict(
			type="datetime",
			description="Birthday date; affected by the timezone parameter",
			example="february 29th",
		),
	)
	rate_limit = (4, 6)
	slash = True
	ephemeral = True
	usercmd = True

	async def __call__(self, bot, _user, user, description, pronouns, website, icon, timezone, birthday, **void):
		target = user or _user
		is_owner = target.id == _user.id
		updated = False
		profile = T(bot.data.users).coercedefault(target.id, cdict, cdict())

		def raise_unless_owner():
			nonlocal updated
			if not is_owner:
				raise PermissionError("Modification of other users' profiles is not allowed.")
			updated = True

		if description is not None:
			raise_unless_owner()
			profile.description = description
		if pronouns is not None:
			raise_unless_owner()
			profile.pronouns = pronouns
		if website is not None:
			raise_unless_owner()
			profile.website = website
		if icon is not None:
			raise_unless_owner()
			profile.icon = icon
		if timezone is not None:
			raise_unless_owner()
			assert get_timezone(timezone)
			profile.timezone = timezone
		if birthday is not None:
			raise_unless_owner()
			profile.birthday = birthday

		colour = await bot.get_colour(target)
		emb = discord.Embed(colour=colour)
		emb.set_author(**get_author(target))
		emb.description = profile.get("description", "")
		emb.title = "Updated Profile" if updated else "Profile"
		url = profile.get("website")
		if url:
			emb.url = emb.author.url = url
		icon = profile.get("icon") or profile.get("thumbnail")
		if icon:
			emb.set_thumbnail(url=icon)
		c = 1
		tzinfo = self.bot.data.users.get_timezone(target.id)
		if tzinfo is None:
			tzinfo, c = self.bot.data.users.estimate_timezone(target.id)
			estimated = True
		else:
			estimated = False
		t = DynamicDT.unix()
		if tzinfo:
			ts = f"`{DynamicDT.fromtimestamp(t, tz=tzinfo)}`"
			if estimated:
				ts += f"\nEstimated from Discord activity ({round(c * 100)}% confidence)"
			emb.add_field(name="Time", value=ts)
		birthday = profile.get("birthday")
		if birthday:
			if not isinstance(birthday, DynamicDT):
				birthday = profile["birthday"] = DynamicDT.fromdatetime(birthday)
			birthday = birthday.replace(tzinfo=tzinfo)
			temp = birthday.replace(time=0)
			now = DynamicDT.fromtimestamp(t, tz=tzinfo)
			temp = temp.add_years(now.year - temp.year)
			while now > temp:
				temp = temp + TimeDelta(years=1)
			ts = temp.as_rel_discord()
			bi = birthday.as_discord()
			emb.add_field(name="Birthday", value=bi + "\n" + ts)
		return cdict(embed=emb)


class Activity(Command):
	name = ["Recent", "Log"]
	description = "Shows your recent Discord activity"
	schema = cdict()
	rate_limit = (8, 11)
	slash = True

	async def __call__(self, bot, _user, _timeout, **void):
		data = await asubmit(bot.data.users.fetch_events, _user.id, interval=900, timeout=_timeout)
		resp = await process_image("plt_special", "&", (data, str(_user)), cap="math")
		fn = resp
		f = CompatFile(fn, filename=f"{_user}.png")
		return dict(file=f, filename=fn, best=True)


class Status(Command):
	name = ["Ping"]
	description = "Shows the bot's current internal program state."
	usage = "<mode(enable|disable)>?"
	example = ("status", "status enable")
	flags = "aed"
	slash = True
	ephemeral = True
	rate_limit = (9, 13)

	async def __call__(self, perm, flags, channel, bot, **void):
		if "d" in flags:
			if perm < 2:
				raise PermissionError("Permission level 2 or higher required to unset auto-updating status.")
			bot.data.messages.pop(channel.id)
			return fix_md("Successfully disabled status updates.")
		elif "a" not in flags and "e" not in flags:
			return await self._callback2_(channel)
		if perm < 2:
			raise PermissionError("Permission level 2 or higher required to set auto-updating status.")
		message = await channel.send(italics(code_md("Loading bot status...")))
		set_dict(bot.data.messages, channel.id, {})[message.id] = cdict(t=0, command="bot.commands.status[0]")

	async def _callback2_(self, channel, m_id=None, msg=None, colour=None, **void):
		bot = self.bot
		if not hasattr(bot, "bitrate"):
			return
		emb = discord.Embed(colour=colour or rand_colour())
		url = best_url(bot.user)
		emb.set_author(name="Status", url=bot.webserver, icon_url=url)
		emb.timestamp = DynamicDT.utcnow()
		if msg is None:
			def subs(n, x):
				if n == "Code size":
					return f"{n}\n[`{byte_scale(x[0])}B, {x[1]} lines`]({bot.github})"
				if n == "Command count":
					return f"{n}\n[`{x}`](https://github.com/thomas-xin/Miza/wiki/Commands)"
				if n == "Website URL":
					x2 = x.replace("https://", "").replace("http://", "")
					return f"{n}\n[`{x2}`]({x})"
				return f"{n}\n`{x}`"
			s = await bot.status(simplified=True)
			g_id = getattr_chain(channel, "guild.id")
			s["Discord info"]["Current shard"] = bot.guild_shard(g_id)
			for k, v in s.items():
				emb.add_field(
					name=k,
					value="\n".join(subs(n, x) for n, x in v.items()),
				)
		else:
			emb.description = msg
		func = channel.send
		message = None
		if m_id is not None:
			with tracebacksuppressor(StopIteration, discord.NotFound, discord.Forbidden):
				message = bot.cache.messages.get(m_id)
				if message is None:
					message = await aretry(channel.fetch_message, m_id, attempts=6, delay=2, exc=(discord.NotFound, discord.Forbidden))
					if utc() - snowflake_time_2(message.id).timestamp() > 86400 * 14 - 60:
						csubmit(bot.silent_delete(message))
						raise StopIteration
				if message.id != channel.last_message_id or getattr(message, "rated", False):
					async for m in bot.data.channel_cache.grab(channel):
						if message.id != m.id or utc() - snowflake_time_2(m.id).timestamp() > 86400 * 14 - 60 or getattr(m, "rated", False):
							csubmit(bot.silent_delete(m))
							raise StopIteration
						break
				func = lambda *args, **kwargs: bot.edit_message(message, *args, content=None, **kwargs)
		try:
			message = await func(embed=emb)
		except discord.HTTPException as ex:
			if message and "429" in repr(ex):
				message.rated = True
			raise
		if m_id is not None and message is not None:
			bot.data.messages[channel.id] = {message.id: cdict(t=utc(), command="bot.commands.status[0]")}


class Invite(Command):
	name = ["Website", "BotInfo", "InviteLink"]
	description = "Sends a link to ⟨BOT⟩'s homepage, github and invite code, as well as an invite link to the current server if applicable."
	example = ("invite",)
	rate_limit = (9, 13)
	slash = True
	ephemeral = True

	async def __call__(self, channel, message, **void):
		emb = discord.Embed(colour=rand_colour()).set_author(**get_author(self.bot.user))
		emb.description = f"[**`My Github`**]({self.bot.github}) [**`My Website`**]({self.bot.webserver}) [**`My Invite`**]({self.bot.invite})"
		if message.guild:
			with tracebacksuppressor:
				member = message.guild.get_member(self.bot.id)
				if member.guild_permissions.create_instant_invite:
					invites = await member.guild.invites()
					invites = sorted(invites, key=lambda invite: (invite.max_age == 0, -abs(invite.max_uses - invite.uses), len(invite.url)))
					if not invites:
						c = self.bot.get_first_sendable(member.guild, member)
						invite = await c.create_invite(reason="Invite command")
					else:
						invite = invites[0]
					emb.description += f" [**`Server Invite`**]({invite.url})"
		self.bot.send_embeds(channel, embed=emb, reference=message)


class Preserve(Command):
	name = ["PreserveAttachmentLinks"]
	description = "Sends a reverse proxy link to preserve a Discord attachment URL, or sends a link to ⟨BOT⟩'s webserver's upload page: ⟨WEBSERVER⟩/files"
	schema = cdict(
		minimise=cdict(
			type="bool",
			description="Whether to produce the shortest possible alias",
			default=False,
		),
		urls=cdict(
			type="url",
			description="URL or attachment to preserve",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			multiple=True,
			required=True,
		),
	)
	macros = cdict(
		Shorten=cdict(
			minimise=True,
		),
		Minimise=cdict(
			minimise=True,
		),
		Minimize=cdict(
			minimise=True,
		),
	)
	rate_limit = (12, 17)
	_timeout_ = 50
	slash = ("Preserve",)
	msgcmd = ("Preserve Attachment Links",)
	ephemeral = True

	async def __call__(self, bot, _channel, _message, minimise, urls, **void):
		futs = deque()
		for url in urls:
			if is_miza_attachment(url):
				args = expand_attachment(url)
				futs.append(as_fut(shorten_attachment(*args, minimise=minimise)))
				continue
			if is_discord_attachment(url):
				a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
				found = None
				for attachment in _message.attachments:
					if attachment.id == a_id:
						found = attachment
						break
				if found:
					futs.append(as_fut(shorten_attachment(url, _message.id, minimise=minimise)))
					continue
				futs.append(as_fut(shorten_attachment(url, 0, minimise=minimise)))
				continue
			futs.append(bot.data.exec.lproxy(url, channel=_channel, minimise=minimise))
			await asyncio.sleep(0.1)
		out = await gather(*futs)
		return "\n".join("<" + u + ">" for u in out)


class Reminder(PaginationCommand):
	name = ["RemindMe", "Reminders", "Remind"]
	description = "Sets a reminder for a certain date and time in the future."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("reminder", "announcement", "urgent"),
				accepts={"remind": "reminder", "announce": "announcement", "urgently": "urgent"},
			),
			description="Reminders are sent as direct messages, announcements are sent to the origin channel, urgent reminders will continuously update a direct message until the acknowledged by a reaction",
			default="reminder",
		),
		message=cdict(
			type="string",
			description="Message to receive. Will show up as an embed",
			example="Doctor's appointment",
			validation="[0, 4096]",
		),
		icon=cdict(
			type="visual",
			description="Thumbnail to display on corner of embed",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
		),
		time=cdict(
			type="datetime",
			description="Time input to parse",
			example="35 mins and 6.25 secs before 3am next tuesday, EDT",
		),
		every=cdict(
			type="timedelta",
			description="Turn the reminder into a recurring one that will resend itself after a given amount of time. Note: Must be removed manually!",
			example="One hundred thousand thirty seconds",
		),
		edit=cdict(
			type="index",
			description="Index of reminder(s) to edit",
			example="3..7",
		),
		delete=cdict(
			type="index",
			description="Index of reminder(s) to delete",
			example="3..7",
		),
	)
	macros = cdict(
		Announce=cdict(
			mode="announcement",
		),
		Announcement=cdict(
			mode="announcement",
		),
		Announcements=cdict(
			mode="announcements",
		),
	)
	rate_limit = (8, 13)
	slash = True

	async def __call__(self, bot, _message, _comment, _channel, _user, mode, message, icon, time, every, edit, delete, **void):
		sendable = _channel if mode == "announce" else _user
		rems = bot.data.reminders.get(sendable.id, [])
		for r in rems:
			if isinstance(r.t, number):
				r.t = DynamicDT.utcfromtimestamp(r.t)
		if all_none(message, icon, time, edit, delete):
			# Set callback message for scrollable list
			return await self.display(_user.id, 0, sendable.id)

		if edit is not None:
			if not len(rems):
				return ini_md(f"No {mode}s currently set for {sqr_md(sendable)}.")
			if all_none(message, icon, time, every):
				raise ValueError("Please input at least one valid field to edit")
			targets = RangeSet.parse([edit], len(rems))
			assert targets, "No reminders at the specified index."
			for i in targets:
				if message is not None:
					rems[i].msg = message
				if icon is not None:
					rems[i].icon = icon
				if time is not None:
					rems[i].t = time
				if every is not None:
					rems[i].e = every
			rems = astype(rems, alist)
			if time is not None:
				rems.sort(rems, key=lambda x: x.t.timestamp_exact())
			bot.data.reminders[sendable.id] = rems
			if 0 in targets:
				with suppress(ValueError):
					bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
				if rems:
					bot.data.reminders.listed.insort((rems[0].t.timestamp_exact(), sendable.id), key=lambda x: x[0])
			out = _comment + f"\n```css\nSuccessfully edited {sqr_md(len(targets))} "
			s = "s" if len(targets) != 1 else ""
			if mode == "announcement":
				out += f"announcement{s} for {sqr_md(sendable)}"
			else:
				out += f"reminder{s} for {sqr_md(sendable)}"
			if time is not None:
				out += f" at {sqr_md(time)}"
			if every:
				out += f", every {sqr_md(sec2time(every.total_seconds()))}"
			out += ".```"
			if time is not None:
				out += time.as_discord() + " (" + time.as_rel_discord() + ")"
			return cdict(
				content=out,
			)

		if delete is not None:
			if not len(rems):
				return ini_md(f"No {mode}s currently set for {sqr_md(sendable)}.")
			targets = RangeSet.parse([delete], len(rems))
			assert targets, "No reminders at the specified index."
			removed = (r := rems[targets[0]]).msg + "; " + r.t.as_full()
			if len(targets) > 1:
				removed += f" (+{len(targets) - 1})"
			rems = astype(rems, alist)
			rems.pops(targets)
			bot.data.reminders[sendable.id] = rems
			if 0 in targets:
				with suppress(ValueError):
					bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
				if rems:
					bot.data.reminders.listed.insort((rems[0].t.timestamp_exact(), sendable.id), key=lambda x: x[0])
			return ini_md(f"Successfully removed {sqr_md(removed)} from {mode} list for {sqr_md(sendable)}.")

		if time is None:
			raise ValueError("Please input a valid time.")
		if not message:
			msg = "[SAMPLE ANNOUNCEMENT]" if mode == "announcement" else "[SAMPLE REMINDER]"
			if mode == "urgent":
				message = bold(css_md(msg, force=True))
			else:
				message = bold(ini_md(msg))
		rem = cdict(
			user=_user.id,
			msg=message,
			t=time,
			e=every,
			ref=(_message.channel.id, _message.id),
		)
		recur = 60 if mode == "urgent" else None
		if recur:
			rem.recur = recur
		if icon:
			rem.icon = icon
		rems.append(rem)
		# Sort list of reminders
		bot.data.reminders[sendable.id] = sort(rems, key=lambda x: x.t.timestamp_exact())
		with suppress(ValueError):
			# Remove existing schedule
			bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
		# Insert back into bot schedule
		tup = (bot.data.reminders[sendable.id][0].t.timestamp_exact(), sendable.id)
		if isfinite(tup[0]):
			bot.data.reminders.listed.insort(tup, key=lambda x: x[0])
		emb = discord.Embed(description=message)
		if icon:
			emb.set_thumbnail(url=icon)
		emb.colour = await bot.get_colour(_user)
		emb.set_author(**get_author(_user))
		out = _comment + "\n```css\nSuccessfully set "
		if mode == "urgent":
			out += "urgent "
		if mode == "announcement":
			out += f"announcement for {sqr_md(sendable)}"
		else:
			out += f"reminder for {sqr_md(sendable)}"
		out += f" at {sqr_md(time)}"
		if every:
			out += f", every {sqr_md(sec2time(every.total_seconds()))}"
		out += ":```"
		out += time.as_discord() + " (" + time.as_rel_discord() + ")"
		return cdict(
			content=out,
			embed=emb,
		)

	async def display(self, uid, pos, sid, diridx=-1):
		bot = self.bot
		user = await bot.fetch_user(uid)
		curr = bot.data.reminders.get(sid, [])
		for r in curr:
			if isinstance(r.t, number):
				r.t = DynamicDT.utcfromtimestamp(r.t)
		sendable = await bot.fetch_messageable(sid)
		page = 16
		pos = self.paginate(pos, len(curr), page, diridx)
		colour = await self.bot.get_colour(user)
		emb = discord.Embed(
			colour=colour,
		).set_author(**get_author(user))
		if not curr:
			emb.title = f"Schedule for {sendable} is currently empty."
		else:
			s = "s" if len(curr) != 1 else ""
			emb.title = f"{len(curr)} message{s} currently scheduled for {sendable}:"
			t = DynamicDT.utcnow()

			def format_reminder(x):
				delta = x.t.as_rel_discord()
				if delta.startswith("`"):
					delta = "`" + (x.t - t).to_short() + "`"
				s = lim_str(bot.get_user(x.get("user", -1), replace=True).mention + ": `" + no_md(x.msg), 96) + f"` ➡️ {delta}"
				if x.get("e"):
					every = x.e.to_short()
					s += f", every `{every}`"
				return s

			emb.description = iter2str(
				curr[pos:pos + page],
				key=format_reminder,
				left="`【",
				right="】`",
				offset=pos,
			).strip()
		more = len(curr) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		return self.construct(uid, leb128(pos) + leb128(sendable.id), embed=emb)

	async def _callback_(self, _user, index, data, **void):
		pos, more = decode_leb128(data)
		sid, _ = decode_leb128(more)
		return await self.display(_user.id, pos, sid, index)


class Note(PaginationCommand):
	name = ["Notes"]
	description = "Takes note of a given string and allows you to view and edit a to-do list!"
	schema = cdict(
		message=cdict(
			type="string",
			description="Message to receive. Will show up as an embed",
			example="Doctor's appointment",
		),
		edit=cdict(
			type="index",
			description="Index of reminder(s) to edit",
			example="3..7",
		),
		delete=cdict(
			type="index",
			description="Index of reminder(s) to delete",
			example="3..7",
		),
	)
	rate_limit = (6, 10)

	async def __call__(self, bot, _user, message, edit, delete, **void):
		notes = bot.data.notes.get(_user.id, [])
		if all_none(message, edit, delete):
			# Set callback message for scrollable list
			return await self.display(_user.id, 0)

		if edit:
			targets = RangeSet.parse([edit], len(notes))
			assert targets, "No reminders at the specified index."
			for i in targets:
				notes[i] = message
			bot.data.notes[_user.id] = notes
			return cdict(
				f"Successfully updated {sqr_md(len(targets))} notes for {sqr_md(_user)}.",
				prefix="```css\n",
				suffix="```",
			)
		if delete:
			targets = RangeSet.parse([delete], len(notes))
			assert targets, "No reminders at the specified index."
			notes = [note for i, note in enumerate(notes) if i not in targets]
			bot.data.notes[_user.id] = notes
			return cdict(
				f"Successfully removed {sqr_md(len(targets))} notes for {sqr_md(_user)}.",
				prefix="```css\n",
				suffix="```",
			)

		nth = rank_format(len(notes))
		notes.append(message)
		bot.data.notes[_user.id] = notes
		return cdict(
			f"Successfully added {sqr(nth)} note for {sqr_md(_user)}!",
			prefix="```css\n",
			suffix="```",
		)

	async def display(self, uid, pos, diridx=-1):
		bot = self.bot
		return await self.default_display("note", uid, pos, bot.data.notes.get(uid, ()), diridx)

	async def _callback_(self, _user, index, data, **void):
		pos, _ = decode_leb128(data)
		return await self.display(_user.id, pos, index)


class UpdateUrgentReminders(Database):
	name = "urgentreminders"

	async def _ready_(self, **void):
		try:
			self.data["listed"]
		except KeyError:
			self.data["listed"] = alist()
		# csubmit(self.update_urgents())

	async def __call__(self, **void):
		if self.db is None:
			return
		t = utc()
		listed = self.data["listed"]
		with tracebacksuppressor:
			while listed:
				p = listed[0]
				if t < p[0]:
					break
				with suppress(StopIteration):
					listed.popleft()
					c_id = p[1]
					m_id = p[2]
					emb = p[3]
					if len(p) < 4:
						p.append(60)
					i = 0
					while p[0] < utc() + 1 and i < 4096:
						p[0] += p[4]
						i += 1
					p[0] = max(utc() + 1, p[0])
					channel = await self.bot.fetch_messageable(c_id)
					message = await bot.fetch_message(m_id, channel)
					message = await bot.ensure_reactions(message)
					for react in message.reactions:
						if str(react) == "✅":
							if react.count > 1:
								raise StopIteration
							async for u in react.users():
								if u.id != self.bot.id:
									raise StopIteration
					reference = None
					content = None
					if len(p) > 5 and p[5]:
						content = p[5]
					if len(p) > 6 and p[6]:
						reference = await self.fetch_message(p[6], channel)
					fut = csubmit(channel.send(content, embed=emb, reference=reference))
					await self.bot.silent_delete(message)
					message = await fut
					await message.add_reaction("✅")
					p[2] = message.id
					listed.insort(p, key=lambda x: x[:3])
		self.data["listed"] = listed


# This database is such a hassle to manage, it has to be able to persist between bot restarts, and has to be able to update with O(1) time complexity when idle
class UpdateReminders(Database):
	name = "reminders"
	listed = alist()

	def __load__(self, **void):
		with tracebacksuppressor:
			d = self.data
			# This exists so that checking next scheduled item is O(1)
			for i in tuple(d):
				try:
					assert d[i][0].t is not None
				except Exception:
					print_exc()
					d.pop(i, None)
			gen = (((block[0].t if isinstance(block[0].t, number) else block[0].t.timestamp_exact()), i) for i, block in d.items() if block and isinstance(block[0], dict) and block[0].get("t") is not None)
			self.listed.extend(sorted(gen, key=lambda x: x[0]))

	async def recurrent_message(self, channel, content, embed, t=0, wait=60, reference=None):
		message = await channel.send(content, embed=embed, reference=reference)
		await message.add_reaction("✅")
		self.bot.data.urgentreminders.coercedefault("listed", alist, alist()).insort([t + wait, channel.id, message.id, embed, wait, content, reference and reference.id], key=lambda x: x[:3])

	async def _call_(self):
		t = utc()
		while self.listed:
			p = self.listed[0]
			# Only check first item in the schedule
			if t < p[0]:
				break
			# Grab expired item
			self.listed.popleft()
			u_id = p[1]
			temp = self.data[u_id]
			if not temp:
				self.data.pop(u_id)
				continue
			# Check next item in schedule
			x = temp[0]
			xt = x.t if isinstance(x.t, number) else x.t.timestamp_exact()
			if t < xt:
				# Insert back into schedule if not expired
				self.listed.insort((xt, u_id), key=lambda x: x[0])
				print(self.listed)
				continue
			# Grab target from database
			x = cdict(temp.pop(0))
			every = x.get("e")
			if every:
				x.t += every
				temp.insert(0, x)
				temp.sort(key=lambda x: x.t)
				self.listed.insort((temp[0].t.timestamp_exact(), u_id), key=lambda x: x[0])
				self[u_id] = temp
			elif not temp:
				self.data.pop(u_id)
			else:
				# Insert next listed item into schedule
				self.listed.insort((temp[0].t if isinstance(temp[0].t, number) else temp[0].t.timestamp_exact(), u_id), key=lambda x: x[0])
				self[u_id] = temp
			# print(self.listed)
			# Send reminder to target user/channel
			ch = await self.bot.fetch_messageable(u_id)
			if not self.bot.permissions_in(ch).send_messages:
				continue
			try:
				u = self.bot.get_user(x["user"], replace=True)
			except KeyError:
				u = x
			emb = discord.Embed(description=x.msg).set_author(**get_author(u))
			if x.get("icon"):
				emb.set_thumbnail(url=x["icon"])
			reference = None
			content = None
			if x.get("ref"):
				try:
					channel = await self.bot.fetch_channel(x["ref"][0])
					reference = await self.bot.fetch_message(x["ref"][1], channel)
				except Exception:
					pass
				else:
					if channel.id != ch.id:
						content = fake_reply(reference)
						reference = None
			if not x.get("recur"):
				csubmit(ch.send(content, embed=emb, reference=reference))
			else:
				csubmit(self.recurrent_message(ch, content, emb, t, x.get("recur", 60), reference=reference))


class UpdateNotes(Database):
	name = "notes"


class UpdateMessages(Database):
	name = "messages"
	semaphore = Semaphore(8, 1, rate_limit=3)
	closed = False
	hue = 0

	async def wrap_semaphore(self, func, *args, **kwargs):
		if not self.semaphore.busy:
			async with self.semaphore:
				return await func(*args, **kwargs)

	async def __call__(self, **void):
		if self.bot.ready and not self.closed:
			self.hue += 128
			col = colour2raw(hue2colour(self.hue))
			t = utc()
			for c_id, data in tuple(self.data.items()):
				with tracebacksuppressor():
					try:
						channel = await self.bot.fetch_channel(c_id)
						if hasattr(channel, "guild") and channel.guild not in self.bot.guilds:
							raise
					except:
						self.data.pop(c_id)
					else:
						for m_id, v in data.items():
							if t - v.t >= 4:
								v.t = t
								csubmit(self.wrap_semaphore(eval(v.command, self.bot._globals)._callback2_, channel=channel, m_id=m_id, colour=col))

	async def _destroy_(self, **void):
		self.closed = True



class UpdateFlavour(Database):
	name = "flavour"

	@tracebacksuppressor
	async def get(self, p=True, q=True):
		out = None
		i = xrand(7)
		facts = self.bot.data.users.facts
		questions = self.bot.data.users.questions
		useless = self.bot.data.useless.setdefault(0, ())
		if not isinstance(useless, alist):
			useless = self.bot.data.useless[0] = alist(useless)
		if i < 3 - q and facts:
			text = choice(facts)
			if not p:
				return text
			fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
			out = f"\n{fact} `{text}`"
		elif i < 4 and questions and q:
			text = choice(questions)
			if not p:
				return text
			out = f"\nRandom question: `{text}`"
		else:
			try:
				if not len(useless) or not xrand(len(useless) / 8):
					raise KeyError
			except KeyError:
				s = str(datetime.datetime.fromtimestamp(xrand(1462456800, utc())).date())
				data = await Request("https://www.uselessfacts.net/api/posts?d=" + s, timeout=24, json=True, aio=True)
				factlist = [fact["title"].replace("`", "") for fact in data if "title" in fact]
				useless.extend(factlist)
				useless.uniq()
			text = choice(useless)
			if not p:
				return text
			fact = choice(("Fun fact:", "Did you know?", "Useless fact:", "Random fact:"))
			out = f"\n{fact} `{text}`"
		return out


class UpdateUseless(Database):
	name = "useless"


EMPTY = {}

# This database takes up a lot of space, storing so many events from users
class UpdateUsers(Database):
	name = "users"
	hours = 336
	interval = 900
	scale = 3600 // interval
	mentionspam = re.compile("<@[!&]?[0-9]+>")

	async def garbage_collect(self):
		bot = self.bot
		data = self.data
		for key in tuple(data):
			if type(key) is str:
				if key.startswith("#"):
					c_id = int(key[1:].rstrip("\x7f"))
					if c_id not in bot.cache.channels and c_id not in bot.cache.guilds:
						print(f"Deleting {key} from {self}...")
						data.pop(key, None)
						await asyncio.sleep(0.1)
				continue
			try:
				if not data[key]:
					raise LookupError
				d = await bot.fetch_user(key)
				if d is not None:
					continue
			except (LookupError, discord.NotFound):
				pass
			except:
				print_exc()
				continue
			print(f"Deleting {key} from {self}...")
			data.pop(key, None)

	def __load__(self):
		self.semaphore = Semaphore(1, 0, rate_limit=120)
		self.facts = None
		self.flavour_buffer = deque()
		self.flavour_set = set()
		self.flavour = ()
		self.useless = deque()
		with open("misc/facts.txt", "r", encoding="utf-8") as f:
			self.facts = f.read().splitlines()
		with open("misc/questions.txt", "r", encoding="utf-8") as f:
			self.questions = f.read().splitlines()
		with open("misc/r-questions.txt", "r", encoding="utf-8") as f:
			self.rquestions = f.read().splitlines()
		with open("misc/pickup_lines.txt", "r", encoding="utf-8") as f:
			self.pickup_lines = f.read().splitlines()
		with open("misc/nsfw_pickup_lines.txt", "r", encoding="utf-8") as f:
			self.nsfw_pickup_lines = f.read().splitlines()

	async def _bot_ready_(self, **void):
		data = {"Command": Command}
		name = "".join(regexp("[A-Za-z_]+").findall(self.bot.name.translate("".maketrans({
			" ": "_",
			"-": "_",
			".": "_",
		}))))
		exec(
			f"class {name}(Command):"
			+ "\n\tdescription = 'Serves as an alias for mentioning the bot.'"
			+ "\n\tno_parse = True"
			+ "\n\tasync def __call__(self, message, argv, flags, **void):"
			+ "\n\t\tawait self.bot.data.users._nocommand_(message, self.bot.user.mention + ' ' + argv, flags=flags, force=True)",
			data,
		)
		mod = "MAIN"
		for v in data.values():
			with suppress(TypeError):
				if issubclass(v, Command) and v != Command:
					obj = v(self.bot, mod)
					self.bot.categories[mod].append(obj)
					# print(f"Successfully loaded command {repr(obj)}.")
		return await self()

	def clear_events(self, data, minimum):
		curr = round_min(int(utc() // self.interval) / self.scale)
		for hour in tuple(data):
			if minimum < hour < curr + 1:
				continue
			data.pop(hour, None)

	def send_event(self, u_id, event, count=1):
		# print(self.bot.cache.users.get(u_id), event, count)
		data = set_dict(set_dict(self.data, u_id, {}), "recent", {})
		hour = round_min(int(utc() // self.interval) / self.scale)
		if data and not xrand(12):
			self.clear_events(data, hour - self.hours)
		try:
			data[hour][event] += count
		except KeyError:
			try:
				data[hour][event] = count
			except KeyError:
				data[hour] = {event: count}

	def fetch_events(self, u_id, interval=3600):
		return {i: self.get_events(u_id, interval=interval, event=i) for i in ("message", "typing", "command", "reaction", "misc")}

	# Get all events of a certain type from a certain user, with specified intervals.
	def get_events(self, u_id, interval=3600, event=None):
		data = self.data.get(u_id, EMPTY).get("recent")
		if not data:
			return list(repeat(0, int(self.hours / self.interval * interval)))
		hour = round_min(int(utc() // self.interval) / self.scale)
		self.clear_events(data, hour - self.hours)
		start = hour - self.hours
		if event is None:
			out = [np.sum(data.get(i / self.scale + start, EMPTY).values()) for i in range(self.hours * self.scale)]
		else:
			out = [data.get(i / self.scale + start, EMPTY).get(event, 0) for i in range(self.hours * self.scale)]
		if interval != self.interval:
			factor = ceil(interval / self.interval)
			out = [np.sum(out[i:i + factor]) for i in range(0, len(out), factor)]
		return out

	def any_timezone(self, u_id):
		return self.get_timezone(u_id) or self.estimate_timezone(u_id)[0]

	def get_timezone(self, u_id):
		timezone = self.data.get(u_id, EMPTY).get("timezone")
		if timezone is not None:
			return get_timezone(timezone)

	def estimate_timezone(self, u_id):
		data = self.data.get(u_id, EMPTY).get("recent")
		if not data:
			return datetime.timezone.utc, 0
		hour = round_min(int(utc() // self.interval) / self.scale)
		self.clear_events(data, hour - self.hours)
		start = hour - self.hours
		out = [sum(data.get(i / self.scale + start, EMPTY).values()) for i in range(self.hours * self.scale)]
		factor = ceil(3600 / self.interval)
		activity = [sum(out[i:i + factor]) for i in range(0, len(out), factor)]
		inactive = alist()
		def register(curr):
			if inactive:
				last = inactive[-1]
			if not inactive or curr[0] - last[0] >= 24:
				curr[1] += 1
				inactive.append(curr[:2])
				curr[2] = curr[0]
			elif curr[0] - last[0] - last[1] < 2:
				last[1] += curr[0] + curr[1] - last[0] - last[1]
				curr[2] = curr[0]
			elif last[1] <= curr[1] * 1.5:
				curr[1] += 1
				if curr[0] - curr[2] >= 18:
					inactive.append(curr[:2])
					curr[2] = curr[0]
				else:
					inactive[-1] = curr[:2]
			curr[0] = None
			curr[1] = 0
		m = min(activity) * 4
		curr = [None, 0, 0]
		for i, x in enumerate(activity):
			if x <= m:
				if curr[0] is None:
					curr[0] = i
				curr[1] += 1
			else:
				if curr[0] is not None:
					register(curr)
		if curr[0] is not None:
			register(curr)
		total = 0
		if inactive:
			for i, curr in enumerate(inactive):
				t = (curr[0] + curr[1] / 2) % 24
				if i:
					if total / i - t > 12:
						total += 24
					elif total / i - t < -12:
						total -= 24
				total += t
			estimated = round(2.5 - utc_dt().hour - total / len(inactive)) % 24
			if estimated > 12:
				estimated -= 24
		else:
			estimated = 0
		return get_timezone(estimated), min(1, len(data) / self.hours)

	async def __call__(self):
		changed = False
		if not self.semaphore.busy:
			async with self.semaphore:
				for i in range(64):
					out = await self.bot.data.flavour.get()
					if out:
						self.flavour_buffer.append(out)
						self.flavour_set.add(out)
						changed = True
						if len(self.flavour_buffer) >= 32:
							break
		amount = len(self.flavour_set)
		if changed and (not amount & amount - 1):
			self.flavour = tuple(self.flavour_set)

	def _offline_(self, user, **void):
		set_dict(self.data, user.id, {})["last_offline"] = utc()

	# User seen, add event to activity database
	def _seen_(self, user, delay, event, count=1, raw=None, **void):
		if is_channel(user):
			u_id = "#" + str(user.id)
		else:
			u_id = user.id
		self.send_event(u_id, event, count=count)
		if type(user) in (discord.User, discord.Member):
			add_dict(self.data, {u_id: {"last_seen": 0}})
			self.data[u_id]["last_seen"] = utc() + delay
			self.data[u_id]["last_action"] = raw

	# User executed command, add to activity database
	def _command_(self, user, loop, command, **void):
		self.send_event(user.id, "command")
		add_dict(self.data, {user.id: {"commands": {command.parse_name(): 1}}})
		self.data[user.id]["last_used"] = utc()
		self.data.get(user.id, EMPTY).pop("last_mention", None)
		if not loop:
			self.add_xp(user, getattr(command, "xp", xrand(6, 14)))

	def _send_(self, message, **void):
		user = message.author
		bot = self.bot
		if user.id == bot.id or bot.get_perms(user, message.guild) <= -inf:
			return
		if not bot.get_enabled(message.channel):
			return
		size = get_message_length(message)
		points = sqrt(size) + sum(1 for w in message.content.split() if len(w) > 1)
		if points >= 32 and not message.attachments:
			typing = self.data.get(user.id, EMPTY).get("last_typing", None)
			if typing is None:
				set_dict(self.data, user.id, {})["last_typing"] = inf
			elif typing >= inf:
				return
			else:
				self.data.get(user.id, EMPTY).pop("last_typing", None)
		else:
			self.data.get(user.id, EMPTY).pop("last_typing", None)
		mid = message.id
		if mid % 1000 == 0:
			if bot.data.enabled.get(message.channel.id, True) and (fun := bot._globals.get("FUN")) is not None:
				for k, v in fun.sparkle_odds.items():
					if mid % k == 0:
						self.add_diamonds(user, points * k / 1000)
						add_dict(self.data.setdefault(user.id, {}).setdefault("sparkles", {}), {v: 1})
						sparkle_name = fun.sparkle_values[v]
						csubmit(self.bot.react_with(message, sparkle_name + ".gif"))
						print(f"{user} has obtained {sparkle_name} in {message.guild}!")
						break
			points *= 1000
		else:
			self.add_gold(user, points)
		self.add_xp(user, points)
		if "dailies" in bot.data:
			csubmit(bot.data.dailies.valid_message(message))

	def get_xp(self, user):
		if self.bot.is_blacklisted(user.id):
			return -inf
		if user.id == self.bot.id:
			if self.data.get(self.bot.id, EMPTY).get("xp", 0) != inf:
				set_dict(self.data, self.bot.id, {})["xp"] = inf
				self.data[self.bot.id]["gold"] = inf
				self.data[self.bot.id]["diamonds"] = inf
			return inf
		return self.data.get(user.id, EMPTY).get("xp", 0)

	def xp_to_level(self, xp):
		if isfinite(xp):
			return int((xp * 3 / 2000) ** (2 / 3)) + 1
		return xp

	def xp_to_next(self, level):
		if isfinite(level):
			return ceil(sqrt(level - 1) * 1000)
		return level

	def xp_required(self, level):
		if isfinite(level):
			return ceil((level - 1) ** 1.5 * 2000 / 3)
		return level

	async def get_balance(self, user):
		data = self.data.get(user.id, EMPTY)
		return await self.bot.as_rewards(data.get("diamonds"), data.get("gold"))

	def add_xp(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user)
			amount *= min(5, self.bot.premium_multiplier(pl))
			add_dict(set_dict(self.data, user.id, {}), {"xp": amount})
			if "dailies" in self.bot.data:
				self.bot.data.dailies.progress_quests(user, "xp", amount)

	def add_gold(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user, absolute=True)
			amount *= min(5, self.bot.premium_multiplier(pl))
			add_dict(set_dict(self.data, user.id, {}), {"gold": amount})
			if amount < 0 and self[user.id]["gold"] < 0:
				self[user.id]["gold"] = 0

	def add_diamonds(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user, absolute=True)
			amount *= min(5, self.bot.premium_multiplier(pl))
			add_dict(set_dict(self.data, user.id, {}), {"diamonds": amount})
			if amount > 0 and "dailies" in self.bot.data:
				self.bot.data.dailies.progress_quests(user, "diamond", amount)
			if amount < 0 and self[user.id]["diamonds"] < 0:
				self[user.id]["diamonds"] = 0

	async def _typing_(self, user, **void):
		self.data.setdefault(user.id, {})["last_typing"] = utc()

	async def _nocommand_(self, message, msg, force=False, flags=(), before=None, truemention=True, perm=0, **void):
		if getattr(message, "noresponse", False):
			return
		bot = self.bot
		if isinstance(before, bot.GhostMessage):
			return
		user = message.author
		channel = message.channel
		guild = message.guild
		if not getattr(message, "simulated", None):
			self.data.setdefault(user.id, {})["last_channel"] = channel.id
			stored = self.data[user.id].setdefault("stored", {})
			if channel.id in stored and len(stored) < 5:
				m_id = stored[channel.id]
				try:
					await bot.fetch_message(m_id, channel)
				except (discord.NotFound, LookupError):
					stored[channel.id] = message.id
				except:
					print_exc()
					stored[channel.id] = message.id
			elif len(stored) >= 5:
				m_id, c_id = choice(stored.items())
				if c_id not in bot.cache.channels:
					stored.pop(c_id, None)
				else:
					try:
						m = await bot.fetch_message(m_id, c)
					except (discord.NotFound, LookupError):
						stored.pop(c_id, None)
					except:
						print_exc()
						stored.pop(c_id, None)
			if channel.id in bot.cache.channels:
				stored[channel.id] = message.id
		if force or bot.is_mentioned(message, bot, guild):
			if user.bot:
				with suppress(AttributeError):
					async for m in bot.data.channel_cache.grab(channel):
						user = m.author
						if bot.get_perms(user.id, guild) <= -inf:
							return
						if not user.bot:
							break
			try:
				reference = await bot.fetch_reference(message)
			except (LookupError, discord.NotFound):
				pass
			else:
				if reference.author.id == bot.id and reference.content.startswith("*```callback-admin-relay-"):
					return
			out = None
			count = self.data.get(user.id, EMPTY).get("last_talk", 0)
			if count < 5:
				csubmit(message.add_reaction("👀"))
			argv = message.clean_content.strip()
			me = getattr(guild, "me", bot.user)
			argv = argv.removeprefix(f"@{me.display_name}")
			argv = argv.removesuffix(f"@{me.display_name}")
			argv = argv.strip()
			if "ask" in bot.commands and ("ai" in bot.get_enabled(channel) or not perm < inf):
				with bot.ExceptionSender(message.channel, reference=message):
					await bot.run_command(bot.commands.ask[0], dict(prompt=argv), message=message, respond=True)
				return
			# Help message greetings
			i = xrand(7)
			if i == 0:
				out = "I have been summoned!"
			elif i == 1:
				out = f"Hey there! Name's {bot.name}!"
			elif i == 2:
				out = f"Hello {user.name}, nice to see you! Can I help you?"
			elif i == 3:
				out = f"Howdy, {user.display_name}!"
			elif i == 4:
				out = f"Greetings, {user.name}! May I be of service?"
			elif i == 5:
				out = f"Hi, {user.name}! What can I do for you today?"
			else:
				out = f"Yo, what's good, {user.display_name}? Need me for anything?"
			prefix = bot.get_prefix(message.guild)
			out += f"\n> Use `{prefix}help` or `/help` for help!"
			if argv:
				out += f"\n-# If your intention was to chat with me, my AI is not currently enabled in this channel! If you are a moderator and wish to enable it, use `{prefix}ec --enable AI`."
			send = lambda *args, **kwargs: send_with_react(channel, *args, reacts="❎", reference=not flags and message, **kwargs)
			add_dict(self.data, {user.id: {"last_talk": 1, "last_mention": 1}})
			self.data[user.id]["last_used"] = utc()
			await send(out)
			await bot.seen(user, event="misc", raw="Talking to me")
			self.add_xp(user, xrand(12, 20))
			if "dailies" in bot.data:
				bot.data.dailies.progress_quests(user, "talk")
		else:
			if not self.data.get(user.id, EMPTY).get("last_mention") and random.random() > 0.6:
				self.data.get(user.id, EMPTY).pop("last_talk", None)
			self.data.get(user.id, EMPTY).pop("last_mention", None)
