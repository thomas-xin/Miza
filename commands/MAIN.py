print = PRINT


# Default and standard command categories to enable.
basic_commands = frozenset(("main", "string", "admin"))
standard_commands = default_commands = basic_commands.union(("voice", "image", "webhook", "fun"))

help_colours = fcdict({
	None: 0xfffffe,
	"main": 0xff0000,
	"string": 0x00ff00,
	"admin": 0x0000ff,
	"voice": 0xff00ff,
	"image": 0xffff00,
	"webhook": 0xff007f,
	"fun": 0x00ffff,
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
	("owner", "Restricted owner-only commands; highly volatile"),
	("nsfw", "Not Safe For Work; only usable in 18+ channels"),
	("misc", "Miscellaneous, mostly not relevant to Discord and restricted to trusted servers"),
))


class Help(Command):
	name = ["❓", "❔", "?", "Halp"]
	description = "Shows a list of usable commands, or gives a detailed description of a command."
	usage = "<(command|category)>?"
	example = ("help string", "help waifu2x")
	flags = "v"
	rate_limit = (3, 5)
	no_parse = True
	slash = True

	async def __call__(self, bot, argv, user, message, original=None, **void):
		bot = self.bot
		guild = message.guild
		channel = message.channel
		prefix = "/" if getattr(message, "slash", None) else bot.get_prefix(guild)
		if " " in prefix:
			prefix += " "
		embed = discord.Embed()
		embed.set_author(name="❓ Help ❓", icon_url=best_url(user), url=bot.webserver)
		argv = full_prune(argv).replace("*", "").replace("_", "").replace("||", "")
		comm = None
		if argv in bot.categories:
			catg = argv
		elif argv in bot.commands:
			comm = argv
			catg = bot.commands[argv][0].catg.casefold()
		elif argv.startswith(prefix):
			argv = argv[len(prefix):].lstrip()
			if argv in bot.categories:
				catg = argv
			elif argv in bot.commands:
				comm = argv
				catg = bot.commands[argv][0].catg.casefold()
		else:
			catg = None
		if catg not in bot.categories:
			catg = None
		content = None
		enabled = bot.get_enabled(channel)
		category_repr = lambda catg: catg.capitalize() + (" [DISABLED]" if catg.lower() not in enabled else "")
		if comm:
			com = bot.commands[comm][0]
			a = ", ".join(n.strip("_") for n in com.name) or "[none]"
			content = (
				f"[Category] {category_repr(com.catg)}\n"
				+ f"[Usage] {prefix}{com.parse_name()} {com.usage}\n"
				+ f"[Aliases] {a}\n"
				+ f"[Effect] {com.parse_description()}\n"
				+ f"[Level] {com.min_display}"
			)
			x = com.rate_limit
			if x:
				if user.id in bot.owners:
					x = 0
				elif isinstance(x, collections.abc.Sequence):
					x = x[not bot.is_trusted(getattr(guild, "id", 0))]
					x /= 2 ** bot.premium_level(user)
				content += f"\n[Rate Limit] {sec2time(x)}"
			if getattr(com, "example", None):
				example = com.example
				if isinstance(example, str):
					example = [example]
				exs = []
				for ex in example:
					exs.append(prefix + ex.replace("~", prefix))
				content += "\n[Examples]\n" + "\n".join(exs)
			content = ini_md(content)
		else:
			content = (
				f"Yo! Use the menu below to select from my command list!\n"
				+ f"Alternatively, visit [`mizatlas`]({bot.raw_webserver}/mizatlas) for a full command list and tester.\n\n"
				+ f"If you're an admin and wish to disable me in a particular channel, check out `{prefix}ec`!\n"
				+ f"Want to try the premium features, unsure about anything, or have a bug to report? check out the [`support server`]({bot.rcc_invite})!\n"
				+ f"Finally, donate to me or purchase a subscription [`here`]({bot.kofi_url})! Any support is greatly appreciated!"
			)
		embed.colour = discord.Colour(help_colours[catg])
		if not catg:
			coms = chain.from_iterable(v for k, v in bot.categories.items() if k in standard_commands)
		else:
			coms = bot.categories[catg]
		coms = sorted(coms, key=lambda c: c.parse_name())
		catsel = [cdict(
			emoji=cdict(id=None, name=help_emojis[c]),
			label=category_repr(c),
			value=c,
			description=help_descriptions[c],
			default=catg == c,
		) for c in standard_commands if c in bot.categories]
		comsel = [cdict(
			emoji=cdict(id=None, name=c.emoji) if getattr(c, "emoji", None) else None,
			label=lim_str(prefix + " " * (" " in prefix) + c.parse_name(), 25, mode=None),
			value=c.parse_name().casefold(),
			description=lim_str(c.parse_description(), 50, mode=None),
			default=comm and com == c,
		) for i, c in enumerate(coms) if i < 25]
		catmenu = cdict(
			type=3,
			custom_id="\x7f0",
			options=catsel,
			min_values=0,
			placeholder=category_repr(catg) if catg else "Choose a category...",
		)
		commenu = cdict(
			type=3,
			custom_id="\x7f1",
			options=comsel,
			min_values=0,
			placeholder=com.parse_name() if comm else "Choose a command...",
		)
		buttons = [[catmenu], [commenu]]
		if content:
			embed.description = f"```callback-main-help-{user.id}-\n{user.display_name} has asked for help!```" + content
		if original:
			if getattr(message, "int_id", None):
				await interaction_patch(bot, original, embed=embed, buttons=buttons)
			else:
				if getattr(message, "slash", None):
					create_task(bot.ignore_interaction, message)
				await Request(
					f"https://discord.com/api/{api}/channels/{original.channel.id}/messages/{original.id}",
					data=dict(
						embeds=[embed.to_dict()],
						components=restructure_buttons(buttons),
					),
					method="PATCH",
					authorise=True,
					aio=True,
				)
			return
		# elif getattr(message, "slash", None):
		#     await interaction_response(bot, message, embed=embed, buttons=buttons, ephemeral=True)
		else:
			await send_with_reply(channel, message, embed=embed, buttons=buttons, ephemeral=True)
		return

	async def _callback_(self, message, reaction, user, vals, perm, **void):
		u_id = int(vals)
		if reaction is None or u_id != user.id and perm < 3:
			return
		await self.__call__(self.bot, as_str(reaction), user, message=message, original=message)


# class Hello(Command):
#     name = ["👋", "Hi", "Hi!", "Hewwo", "Herro", "'sup", "Hey", "Greetings", "Welcome", "Bye", "Cya", "Goodbye"]
#     description = "Sends a greeting message. Useful for checking whether the bot is online."
#     usage = "<user>?"
#     example = ("hello",)
#     rate_limit = (1, 2)
#     slash = True

#     async def __call__(self, bot, user, name, argv, guild, **void):
#         if "dailies" in bot.data:
#             bot.data.dailies.progress_quests(user, "talk")
#         if argv:
#             try:
#                 u = await bot.fetch_user_member(argv, guild)
#             except:
#                 u = None
#             if u and u.id != bot.id:
#                 user = u
#         elif bot.is_owner(user):
#             return "👋"
#         if name in ("bye", "cya", "goodbye"):
#             start = choice("Bye", "Cya", "Goodbye")
#         else:
#             if not argv and not xrand(5):
#                 return choice(
#                     f"Hi, {user}. I'm feeling a little lonely, so I appreciate you talking to me. 😊",
#                     f"Hello? {get_random_emoji()}",
#                     "Hi! " + choice(HEARTS),
#                 )
#             start = choice("Hi", "Hello", "Hey", "'sup")
#         middle = choice(user.name, user.display_name)
#         if name in ("bye", "cya", "goodbye"):
#             end = choice(
#                 "",
#                 "See you soon!",
#                 "See you around!",
#                 "Have a good one!",
#                 "Later!",
#                 "Talk to you again sometime!",
#                 "Was nice talking to you!",
#                 "Peace!",
#             )
#         else:
#             end = choice(
#                 "",
#                 "How are you?",
#                 "Can I help you?",
#                 "What can I do for you today?",
#                 "Nice to see you!",
#                 "Great to see you!",
#                 "Always good to see you!",
#                 "Do you need something?",
#             )
#         out = "👋 " + start + ", `" + middle + "`!"
#         if end:
#             out += " " + end
#         return out


class Perms(Command):
	server_only = True
	name = ["DefaultPerms", "ChangePerms", "Perm", "ChangePerm", "Permissions"]
	description = "Shows or changes a user's permission level."
	usage = "<0:user>* <1:new_level>? <default{?d}>? <hide{?h}>?"
	example = ("perms steven 2", "perms 201548633244565504 ?d")
	flags = "fhd"
	rate_limit = (5, 8)
	multi = True
	slash = True

	async def __call__(self, bot, args, argl, user, name, perm, channel, guild, flags, **void):
		if name == "defaultperms":
			users = (guild.get_role(guild.id),)
		else:
			# Get target user from first argument
			users = await bot.find_users(argl, args, user, guild, roles=True)
		if not users:
			raise LookupError("No results found.")
		msgs = []
		try:
			for t_user in users:
				t_perm = round_min(bot.get_perms(t_user, guild))
				# If permission level is given, attempt to change permission level, otherwise get current permission level
				if args or "d" in flags:
					name = str(t_user)
					if "d" in flags:
						o_perm = round_min(bot.get_perms(t_user, guild))
						bot.remove_perms(t_user, guild)
						c_perm = round_min(bot.get_perms(t_user, guild))
						m_perm = max(abs(t_perm), abs(c_perm), 2) + 1
						if not perm < m_perm and not isnan(m_perm):
							msgs.append(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(t_perm)} to the default value of {sqr_md(c_perm)}."))
							continue
						reason = f"to change permissions for {name} in {guild} from {t_perm} to {c_perm}"
						bot.set_perms(t_user, guild, o_perm)
						raise self.perm_error(perm, m_perm, reason)
					orig = t_perm
					expr = " ".join(args)
					num = await bot.eval_math(expr, orig)
					c_perm = round_min(num)
					if t_perm is nan or isnan(c_perm):
						m_perm = nan
					else:
						# Required permission to change is absolute level + 1, with a minimum of 3
						m_perm = max(abs(t_perm), abs(c_perm), 2) + 1
					if not perm < m_perm and not isnan(m_perm):
						if not m_perm < inf and guild.owner_id != user.id and not isnan(perm):
							raise PermissionError("Must be server owner to assign non-finite permission level.")
						bot.set_perms(t_user, guild, c_perm)
						if "h" in flags:
							continue
						msgs.append(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(t_perm)} to {sqr_md(c_perm)}."))
						continue
					reason = f"to change permissions for {name} in {guild} from {t_perm} to {c_perm}"
					raise self.perm_error(perm, m_perm, reason)
				msgs.append(css_md(f"Current permissions for {sqr_md(t_user)} in {sqr_md(guild)}: {sqr_md(t_perm)}."))
		finally:
			return "".join(msgs)


class EnabledCommands(Command):
	server_only = True
	name = ["EC", "Enable", "Disable"]
	min_display = "0~3"
	description = "Shows, enables, or disables a command category in the current channel."
	usage = "(enable|disable|clear)? <category>? <server-wide(?s)> <list{?l}>? <hide{?h}>?"
	example = ("enable fun ", "ec disable main", "ec -l")
	flags = "aedlhrs"
	rate_limit = (5, 8)
	slash = True

	def __call__(self, argv, args, flags, user, channel, guild, perm, name, **void):
		bot = self.bot
		update = bot.data.enabled.update
		enabled = bot.data.enabled
		if name == "enable":
			flags["e"] = 1
		elif name == "disable":
			flags["d"] = 1
		if "s" in flags:
			target = guild
			mention = lambda *args: str(guild)
		else:
			target = channel
			mention = channel_mention
		# Flags to change enabled commands list
		if any(k in flags for k in "acder"):
			req = 3
			if perm < req:
				reason = f"to change enabled command list for {channel_repr(target)}"
				raise self.perm_error(perm, req, reason)
		else:
			req = 0
		if not args or argv.casefold() == "all" or "r" in flags:
			if "l" in flags:
				return css_md(f"Standard command categories:\n[{', '.join(standard_commands)}]")
			if "e" in flags or "a" in flags:
				categories = set(standard_commands)
				if target.id in enabled:
					enabled[target.id] = categories.union(enabled[target.id])
				else:
					enabled[target.id] = categories
				if "h" in flags:
					return
				return css_md(f"Enabled all standard command categories in {sqr_md(target)}.")
			if "r" in flags:
				enabled.pop(target.id, None)
				if "h" in flags:
					return
				return css_md(f"Reset enabled status of all commands in {sqr_md(target)}.")
			if "d" in flags:
				enabled[target.id] = set()
				if "h" in flags:
					return
				return css_md(f"Disabled all commands in {sqr_md(target)}.")
			temp = bot.get_enabled(target)
			if not temp:
				return ini_md(f"No currently enabled commands in {sqr_md(target)}.")
			return f"Currently enabled command categories in {mention(target.id)}:\n{ini_md(iter2str(temp))}"
		if not req:
			catg = argv.casefold()
			# if not bot.is_trusted(guild) and catg not in standard_commands:
				# raise PermissionError(f"Elevated server priviliges required for specified command category.")
			if catg not in bot.categories:
				raise LookupError(f"Unknown command category {argv}.")
			if catg in bot.get_enabled(target):
				return css_md(f"Command category {sqr_md(catg)} is currently enabled in {sqr_md(target)}.")
			return css_md(f'Command category {sqr_md(catg)} is currently disabled in {sqr_md(target)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
		args = [i.casefold() for i in args]
		for catg in args:
			# if not bot.is_trusted(guild) and catg not in standard_commands:
				# raise PermissionError(f"Elevated server priviliges required for specified command category.")
			if not catg in bot.categories:
				raise LookupError(f"Unknown command category {catg}.")
		curr = set(bot.get_enabled(target))
		for catg in args:
			if "d" not in flags:
				if catg not in curr:
					if isinstance(curr, set):
						curr.add(catg)
					else:
						curr.append(catg)
			else:
				if catg in curr:
					curr.remove(catg)
		enabled[target.id] = astype(curr, set)
		check = astype(curr, (frozenset, set))
		if check == default_commands:
			enabled.pop(target.id)
		if "h" in flags:
			return
		category = "category" if len(args) == 1 else "categories"
		action = "Enabled" if "d" not in flags else "Disabled"
		return css_md(f"{action} command {category} {sqr_md(', '.join(args))} in {sqr_md(target)}.")


class Prefix(Command):
	name = ["ChangePrefix"]
	min_display = "0~3"
	description = "Shows or changes the prefix for ⟨MIZA⟩'s commands for this server."
	usage = "<new_prefix>? <default{?d}>?"
	example = ("prefix !", "change_prefix >", "prefix -d")
	flags = "hd"
	rate_limit = (5, 8)
	umap = {c: "" for c in ZeroEnc}
	umap["\u200a"] = ""
	utrans = "".maketrans(umap)
	slash = True

	def __call__(self, argv, guild, perm, bot, flags, **void):
		pref = bot.data.prefixes
		update = self.data.prefixes.update
		if "d" in flags:
			if guild.id in pref:
				pref.pop(guild.id)
			return css_md(f"Successfully reset command prefix for {sqr_md(guild)}.")
		if not argv:
			return css_md(f"Current command prefix for {sqr_md(guild)}: {sqr_md(bot.get_prefix(guild))}.")
		req = 3
		if perm < req:
			reason = f"to change command prefix for {guild}"
			raise self.perm_error(perm, req, reason)
		prefix = argv
		if not prefix.isalnum():
			prefix = prefix.translate(self.utrans)
		# Backslash is not allowed, it is used to escape commands normally
		if prefix.startswith("\\"):
			raise TypeError("Prefix must not begin with backslash.")
		pref[guild.id] = prefix
		if "h" not in flags:
			return css_md(f"Successfully changed command prefix for {sqr_md(guild)} to {sqr_md(argv)}.")


class Loop(Command):
	time_consuming = 3
	_timeout_ = 12
	name = ["For", "Rep", "While"]
	min_level = 1
	min_display = "1+"
	description = "Loops a command. Delete the original message to terminate the loop if necessary."
	usage = "<0:iterations> <1:command>+"
	example = ("loop 3 ~cat", "loop 8 ~sharpen")
	rate_limit = (10, 15)
	active = set()

	async def __call__(self, args, argv, message, channel, bot, perm, user, guild, **void):
		if not args:
			# Ah yes, I made this error specifically for people trying to use this command to loop songs 🙃
			raise ArgumentError("Please input loop iterations and target command. For looping songs in voice, consider using the aliases LoopQueue and Repeat under the AudioSettings command.")
		num = await bot.eval_math(args[0])
		iters = round(num)
		# Bot owner bypasses restrictions
		if not isnan(perm):
			if channel.id in self.active:
				raise PermissionError("Only one loop may be active in a channel at any time.")
			elif iters > 64 and not bot.is_trusted(guild.id):
				raise PermissionError("Server premium level 1 or higher required to execute loop of greater than 64 iterations; please see {bot.kofi_url} for more info!")
			elif iters > 256:
				raise PermissionError("Loops cannot be more than 256 iterations.")
		func = func2 = " ".join(args[1:])
		func = func.lstrip()
		if not isnan(perm):
			# Detects when an attempt is made to loop the loop command
			for n in self.name:
				if (
					(bot.get_prefix(guild) + n).upper() in func.replace(" ", "").upper()
				) or (
					(str(bot.id) + ">" + n).upper() in func.replace(" ", "").upper()
				):
					raise PermissionError("Loops must not be nested.")
		func2 = func2.split(None, 1)[-1]
		create_task(send_with_react(
			channel,
			italics(css_md(f"Looping {sqr_md(func)} {iters} time{'s' if iters != 1 else ''}...")),
			reacts=["❎"],
			reference=message,
		))
		fake_message = copy.copy(message)
		fake_message.content = func2
		self.active.add(channel.id)
		try:
			for i in range(iters):
				if hasattr(message, "simulated"):
					curr_message = message
				else:
					curr_message = await bot.fetch_message(message.id, channel)
				if getattr(message, "deleted", None) or getattr(curr_message, "deleted", None):
					break
				loop = i < iters - 1
				t = utc()
				# Calls process_message with the argument containing the looped command.
				delay = await bot.process_message(fake_message, func, loop=loop)
				# Must abide by command rate limit rules
				delay = delay + t - utc()
				if delay > 0:
					await asyncio.sleep(delay)
		finally:
			self.active.discard(channel.id)


class Avatar(Command):
	name = ["PFP", "Icon"]
	description = "Sends a link to the avatar of a user or server."
	usage = "<objects>*"
	example = ("icon 247184721262411776", "avatar bob", "pfp")
	rate_limit = (5, 7)
	multi = True
	slash = True

	async def getGuildData(self, g):
		# Gets icon display of a server and returns as an embed.
		url = best_url(g)
		name = g.name
		colour = await self.bot.get_colour(g)
		emb = discord.Embed(colour=colour)
		emb.set_thumbnail(url=url)
		emb.set_image(url=url)
		emb.set_author(name=name, icon_url=url, url=url)
		emb.description = f"{sqr_md(name)}({url})"
		return emb

	async def getMimicData(self, p):
		# Gets icon display of a mimic and returns as an embed.
		url = best_url(p)
		name = p.name
		colour = await self.bot.get_colour(p)
		emb = discord.Embed(colour=colour)
		emb.set_thumbnail(url=url)
		emb.set_image(url=url)
		emb.set_author(name=name, icon_url=url, url=url)
		emb.description = f"{sqr_md(name)}({url})"
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
					name = str(u)
					url = await self.bot.get_proxy_url(u, force=True)
					colour = await self.bot.get_colour(u)
					emb = discord.Embed(colour=colour)
					emb.set_thumbnail(url=url)
					emb.set_image(url=url)
					emb.set_author(name=name, icon_url=url, url=url)
					emb.description = f"{sqr_md(name)}({url})"
					embs.add(emb)
		bot.send_embeds(channel, embeds=embs, reference=message)


class Info(Command):
	name = ["🔍", "🔎", "UserInfo", "ServerInfo", "WhoIs"]
	description = "Shows information about the target user or server."
	usage = "<user>* <verbose{?v}>?"
	example = ("info 201548633244565504", "info")
	flags = "v"
	rate_limit = (6, 9)
	multi = True
	slash = True
	usercmd = True

	async def getGuildData(self, g, flags={}):
		bot = self.bot
		url = await bot.get_proxy_url(g, force=True)
		name = g.name
		try:
			u = g.owner
		except (AttributeError, KeyError):
			u = None
		colour = await self.bot.get_colour(g)
		emb = discord.Embed(colour=colour)
		emb.set_thumbnail(url=url)
		emb.set_author(name=name, icon_url=url, url=url)
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
		else:
			d += f"\nNo {bot.name} Premium Upgrades! Visit {bot.kofi_url} to purchase one!"
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
		with suppress(AttributeError):
			a = r = 0
			m = len(g._members)
			for member in g.members:
				if member.guild_permissions.administrator:
					a += 1
				else:
					r += len(member.roles) > 1
			memberinfo = f"Admins: {a}\nOther roles: {r}\nNo roles: {m - a - r}"
			emb.add_field(name=f"Member count ({m})", value=memberinfo, inline=1)
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
							emb = await self.getGuildData(g, flags)
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
					name = str(u)
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
					if lv2 > 0:
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
							tz = bot.data.users.estimate_timezone(u.id)
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
						emb.add_field(name="Estimated timezone", value=str(zone), inline=1)
					if seen:
						emb.add_field(name="Last seen", value=str(seen), inline=1)
					if dname:
						emb.add_field(name="Nickname", value=dname, inline=1)
					if role:
						emb.add_field(name=f"Roles ({len(rolelist)})", value=role, inline=0)
					embs.add(emb)
		bot.send_embeds(channel, embeds=embs, reference=message)


class Profile(Command):
	name = ["User", "UserProfile"]
	description = "Shows or edits a user profile on ⟨MIZA⟩."
	usage = "<user>? (description|thumbnail|timezone|birthday)? <value>? <delete{?d}>?"
	example = ("profile 201548633244565504", "profile timezone singapore", "profile thumbnail https://cdn.discordapp.com/emojis/879989027711877130.gif", "user")
	flags = "d"
	rate_limit = (4, 6)
	no_parse = True
	slash = True
	usercmd = True

	async def __call__(self, user, args, argv, flags, channel, guild, bot, message, **void):
		if message.attachments:
			args += [best_url(a) for a in message.attachments]
			argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
		setting = None
		if not args:
			target = user
		elif args[0] in ("description", "thumbnail", "timezone", "time", "birthday"):
			target = user
			setting = args.pop(0)
			if not args:
				value = None
			else:
				value = argv[len(setting) + 1:]
		else:
			target = await bot.fetch_user_member(" ".join(args), guild)
		profile = bot.data.users.get(target.id, EMPTY)
		if setting is None:
			description = profile.get("description", "")
			if description and is_url(description.rsplit(None, 1)[-1]):
				description += "\n"
			thumbnail = profile.get("thumbnail")
			birthday = profile.get("birthday")
			timezone = profile.get("timezone")
			t = utc()
			if timezone:
				td = as_timezone(timezone)
				description += ini_md(f"Current time: {sqr_md(utc_ft(t + td))}")
			if birthday:
				if not isinstance(birthday, DynamicDT):
					birthday = profile["birthday"] = DynamicDT.fromdatetime(birthday)
					bot.data.users.update(target.id)
				now = DynamicDT.utcfromtimestamp(t)
				birthday_in = next_date(birthday)
				if timezone:
					birthday -= td
					birthday_in -= datetime.timedelta(seconds=td)
				description += ini_md(f"Age: {sqr_md(time_diff(now, birthday))}\nBirthday in: {sqr_md(time_diff(birthday_in, now))}")
			fields = set()
			for field in ("timezone", "birthday"):
				value = profile.get(field)
				if isinstance(value, DynamicDT):
					value = value.as_date()
				elif field == "timezone" and value is not None:
					value = timezone_repr(value)
				fields.add((field, value, False))
			bot.send_as_embeds(channel, description, thumbnail=thumbnail, fields=fields, author=get_author(target), reference=message)
			return
		if setting != "description" and value.casefold() in ("undefined", "remove", "rem", "reset", "unset", "delete", "clear", "null", "none") or "d" in flags:
			profile.pop(setting, None)
			bot.data.users.update(user.id)
			return css_md(f"Successfully removed {setting} for {sqr_md(user)}.")
		if value is None:
			return ini_md(f"Currently set {setting} for {sqr_md(user)}: {sqr_md(bot.data.users.get(user.id, EMPTY).get(setting))}.")
		if setting == "description":
			if len(value) > 1024:
				raise OverflowError("Description must be 1024 or fewer in length.")
		elif setting == "thumbnail":
			urls = await bot.follow_url(value)
			if not urls:
				raise ValueError("Thumbnail must be an attachment or URL.")
			value = urls[0]
		elif setting.startswith("time"):
			value = value.casefold()
			try:
				as_timezone(value)
			except KeyError:
				raise ArgumentError(f"Entered value could not be recognized as a timezone location or abbreviation. Use {bot.get_prefix(guild)}timezone list for list.")
		else:
			dt = tzparse(value)
			offs, year = divmod(dt.year, 400)
			value = DynamicDT(year + 2000, dt.month, dt.day, tzinfo=datetime.timezone.utc).set_offset(offs * 400 - 2000)
		bot.data.users.setdefault(user.id, {})[setting] = value
		bot.data.users.update(user.id)
		if type(value) is DynamicDT:
			value = value.as_date()
		elif setting.startswith("time") and value is not None:
			value = timezone_repr(value)
		return css_md(f"Successfully changed {setting} for {sqr_md(user)} to {sqr_md(value)}.")


class Activity(Command):
	name = ["Recent", "Log"]
	description = "Shows recent Discord activity for the targeted user, server, or channel."
	usage = "<user>? <verbose{?v}>?"
	example = ("recent 201548633244565504", "log")
	flags = "v"
	rate_limit = (8, 11)
	typing = True
	slash = True
	# usercmd = True

	async def __call__(self, guild, user, argv="", flags="", channel=None, _timeout=90, **void):
		bot = self.bot
		u_id = None
		if argv:
			user = None
			if "#" not in argv:
				with suppress():
					user = bot.cache.guilds[int(argv)]
			if user is None:
				try:
					user = bot.cache.channels[verify_id(argv)]
				except:
					user = await bot.fetch_user_member(argv, guild)
				else:
					u_id = f"#{user.id}"
		if not u_id:
			u_id = user.id
		data = await create_future(bot.data.users.fetch_events, u_id, interval=max(900, 3600 >> flags.get("v", 0)), timeout=_timeout)
		ctx = discord.context_managers.Typing(channel) if channel else emptyctx
		async with ctx:
			resp = await process_image("plt_special", "&", (data, str(user)), cap="math")
			fn = resp
			f = CompatFile(fn, filename=f"{user.id}.png")
		return dict(file=f, filename=fn, best=True)


class Status(Command):
	name = ["State", "Ping"]
	description = "Shows the bot's current internal program state."
	usage = "(enable|disable)?"
	example = ("status", "status enable")
	flags = "aed"
	slash = True
	rate_limit = (9, 13)

	async def __call__(self, perm, flags, channel, bot, **void):
		if "d" in flags:
			if perm < 2:
				raise PermissionError("Permission level 2 or higher required to unset auto-updating status.")
			bot.data.messages.pop(channel.id)
			bot.data.messages.update(channel.id)
			return fix_md("Successfully disabled status updates.")
		elif "a" not in flags and "e" not in flags:
			return await self._callback2_(channel)
		if perm < 2:
			raise PermissionError("Permission level 2 or higher required to set auto-updating status.")
		message = await channel.send(italics(code_md("Loading bot status...")))
		set_dict(bot.data.messages, channel.id, {})[message.id] = cdict(t=0, command="bot.commands.status[0]")
		bot.data.messages.update(channel.id)

	async def _callback2_(self, channel, m_id=None, msg=None, colour=None, **void):
		bot = self.bot
		if not hasattr(bot, "bitrate"):
			return
		emb = discord.Embed(colour=colour or rand_colour())
		url = await self.bot.get_proxy_url(self.bot.user)
		emb.set_author(name="Status", url=bot.webserver, icon_url=url)
		emb.timestamp = utc_dt()
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
			for k, v in s.items():
				emb.add_field(
					name=k,
					value="\n".join(subs(n, x) for n, x in v.items()),
				)
		else:
			emb.description = msg
		func = channel.send
		if m_id is not None:
			with tracebacksuppressor(StopIteration, discord.NotFound, discord.Forbidden):
				message = bot.cache.messages.get(m_id)
				if message is None:
					message = await aretry(channel.fetch_message, m_id, attempts=6, delay=2, exc=(discord.NotFound, discord.Forbidden))
				if message.id != channel.last_message_id:
					async for m in bot.data.channel_cache.get(channel):
						if message.id != m.id:
							create_task(bot.silent_delete(message))
							raise StopIteration
						break
				func = lambda *args, **kwargs: message.edit(*args, content=None, **kwargs)
		message = await func(embed=emb)
		if m_id is not None and message is not None:
			bot.data.messages[channel.id] = {message.id: cdict(t=utc(), command="bot.commands.status[0]")}


class Invite(Command):
	name = ["Website", "BotInfo", "InviteLink"]
	description = "Sends a link to ⟨MIZA⟩'s homepage, github and invite code, as well as an invite link to the current server if applicable."
	example = ("invite",)
	rate_limit = (9, 13)
	slash = True

	async def __call__(self, channel, message, **void):
		emb = discord.Embed(colour=rand_colour())
		emb.set_author(**get_author(self.bot.user))
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


class Upload(Command):
	name = ["Filehost", "Files", "Preserve"]
	description = "Sends a reverse proxy link to preserve a Discord attachment URL, or sends a link to ⟨MIZA⟩'s webserver's upload page: ⟨WEBSERVER⟩/files"
	example = ("upload https://cdn.discordapp.com/attachments/911168940246442006/1026474858705588224/6e74595fa98e9c52e2fab6ece4639604.png", "files")
	rate_limit = (12, 17)
	msgcmd = True
	_timeout_ = 50
	slash = "preserve"

	async def __call__(self, name, message, argv, **void):
		if message.attachments:
			argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
		args = await self.bot.follow_url(argv)
		if not args:
			return self.bot.raw_webserver + "/files"
		futs = deque()
		for url in args:
			if name in ("files", "preserve") and is_discord_attachment(url):
				a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
				if a_id in self.bot.data.attachments:
					futs.append(as_fut("<" + self.bot.preserve_attachment(a_id) + ">"))
					continue
			futs.append(Request(self.bot.raw_webserver + "/upload_url?url=" + url, decode=True, aio=True, timeout=1200))
			await asyncio.sleep(0.1)
		out = deque()
		for fut in futs:
			url = await fut
			out.append(url)
		return "\n".join(out)


class Reminder(Command):
	name = ["Announcement", "Announcements", "Announce", "RemindMe", "Reminders", "Remind"]
	description = "Sets a reminder for a certain date and time."
	usage = "<1:message>? <0:time>? <urgent{?u}>? <delete{?d}>?"
	flags = "aedurf"
	example = ("remindme test in 3 hours 27 mins", "remind urgently submit ticket on 3 june 2023", "announce look at me! in 10 minutes", "remind every 8h take meds in 2d")
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (8, 13)
	keywords = ["on", "at", "in", "when", "event"]
	keydict = {re.compile(f"(^|[^a-z0-9]){i[::-1]}([^a-z0-9]|$)", re.I): None for i in keywords}
	no_parse = True
	timefind = None
	slash = True

	def __load__(self):
		self.timefind = re.compile("(?:(?:(?:[0-9]+:)+[0-9.]+\\s*(?:am|pm)?|" + self.bot.num_words + "|[\\s\-+*\\/^%.,0-9]+\\s*(?:am|pm|s|m|h|d|w|y|century|centuries|millenium|millenia|(?:second|sec|minute|min|hour|hr|day|week|wk|month|mo|year|yr|decade|galactic[\\s\\-_]year)s?))\\s*)+$", re.I)

	async def __call__(self, name, message, flags, bot, user, guild, perm, argv, args, comment="", **void):
		if getattr(message, "slash", None) and args:
			msg = "in " + args.pop(-1)
			if args:
				msg = " ".join(args) + " " + msg
		else:
			msg = message.content
		try:
			msg = msg[msg.casefold().index(name) + len(name):]
		except ValueError:
			print_exc(msg)
			msg = msg.casefold().split(None, 1)[-1]
		orig = argv
		argv = msg.strip()
		args = argv.split()
		if "announce" in name:
			sendable = message.channel
			word = "announcements"
		else:
			sendable = user
			word = "reminders"
		rems = bot.data.reminders.get(sendable.id, [])
		update = bot.data.reminders.update
		if "d" in flags:
			if not len(rems):
				return ini_md(f"No {word} currently set for {sqr_md(sendable)}.")
			if not orig:
				i = 0
			else:
				i = await bot.eval_math(orig)
			i %= len(rems)
			x = rems.pop(i)
			if i == 0:
				with suppress(IndexError):
					bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
				if rems:
					bot.data.reminders.listed.insort((rems[0]["t"], sendable.id), key=lambda x: x[0])
			update(sendable.id)
			return ini_md(f"Successfully removed {sqr_md(lim_str(x['msg'], 128))} from {word} list for {sqr_md(sendable)}.")
		elif "r" in flags:
			if not len(rems):
				return ini_md(f"No {word} currently set for {sqr_md(sendable)}.")
			if "f" not in flags:
				raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(rems))} ITEMS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
			rems.clear()
			bot.data.reminders.pop(sendable.id, None)
			with suppress(IndexError):
				bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
			update(sendable.id)
			return italics(css_md(f"Successfully cleared all {word} for {sqr_md(sendable)}."))
		if not argv:
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-main-reminder-"
				+ str(user.id) + "_0_" + str(sendable.id)
				+ "-\nLoading Reminder database...```*",
				buttons=buttons,
			)
			return
		if len(rems) >= 64:
			raise OverflowError(f"You have reached the maximum of 64 {word}. Please remove one to add another.")
		for f in "aeu":
			if f in flags:
				for c in (f, f.upper()):
					for q in "?-+":
						argv = argv.replace(q + c + " ", "").replace(" " + q + c, "")
		urgent = "u" in flags
		recur = 60 if urgent else None
		remind_as = user
		# This parser is so unnecessarily long for what it does...
		keyed = False
		for i in range(256):
			temp = argv.casefold()
			if name == "remind" and temp.startswith("me "):
				argv = argv[3:]
				temp = argv.casefold()
			if temp.startswith("every ") and " " in argv[6:]:
				duration, argv = argv[6:].split(None, 1)
				recur = await bot.eval_time(duration)
				temp = argv.casefold()
			elif temp.startswith("urgently ") or temp.startswith("urgent "):
				argv = argv.split(None, 1)[1]
				temp = argv.casefold()
				recur = 60
				urgent = True
			if temp.startswith("as ") and " " in argv[3:]:
				query, argv = argv[3:].split(None, 1)
				remind_as = await self.bot.fetch_user_member(query, guild)
				temp = argv.casefold()
			if temp.startswith("to "):
				argv = argv[3:]
				temp = argv.casefold()
			elif temp.startswith("that "):
				argv = argv[5:]
				temp = argv.casefold()
			spl = None
			keywords = dict(self.keydict)
			# Reversed regex search
			temp2 = temp[::-1]
			for k in tuple(keywords):
				try:
					i = re.search(k, temp2).end()
					if not i:
						raise ValueError
				except (ValueError, AttributeError):
					keywords.pop(k)
				else:
					keywords[k] = i
			# Sort found keywords by position
			indices = sorted(keywords, key=lambda k: keywords[k])
			if indices:
				foundkey = {self.keywords[tuple(self.keydict).index(indices[0])]: True}
			else:
				foundkey = cdict(get=lambda *void: None)
			if foundkey.get("event"):
				if " event " in argv:
					spl = argv.rsplit(" event ", 1)
				elif temp.startswith("event "):
					spl = [argv[6:]]
					msg = ""
				if spl is not None:
					msg = " event ".join(spl[:-1])
					t = verify_id(spl[-1])
					keyed = True
					break
			if foundkey.get("when"):
				if temp.endswith("is online"):
					argv = argv[:-9]
				if " when " in argv:
					spl = argv.rsplit(" when ", 1)
				elif temp.startswith("when "):
					spl = [argv[5:]]
					msg = ""
				if spl is not None:
					msg = " when ".join(spl[:-1])
					t = verify_id(spl[-1])
					keyed = True
					break
			if foundkey.get("in"):
				if " in " in argv:
					spl = argv.rsplit(" in ", 1)
				elif temp.startswith("in "):
					spl = [argv[3:]]
					msg = ""
				if spl is not None:
					msg = " in ".join(spl[:-1])
					t = await bot.eval_time(spl[-1])
					break
			if foundkey.get("at"):
				if " at " in argv:
					spl = argv.rsplit(" at ", 1)
				elif temp.startswith("at "):
					spl = [argv[3:]]
					msg = ""
				if spl is not None:
					if len(spl) > 1:
						spl2 = spl[0].rsplit(None, 1)
						if spl2[-1] in ("today", "tomorrow", "yesterday"):
							spl[0] = "" if len(spl2) <= 1 else spl2[0]
							spl[-1] = "tomorrow " + spl[-1]
					msg = " at ".join(spl[:-1])
					t = utc_ts(tzparse(spl[-1])) - utc()
					break
			if foundkey.get("on"):
				if " on " in argv:
					spl = argv.rsplit(" on ", 1)
				elif temp.startswith("on "):
					spl = [argv[3:]]
					msg = ""
				if spl is not None:
					msg = " on ".join(spl[:-1])
					t = utc_ts(tzparse(spl[-1])) - utc()
					break
			if "today" in argv or "tomorrow" in argv or "yesterday" in argv:
				t = 0
				if " " in argv:
					args = argv.split()
					for i in (0, -1):
						arg = args[i]
						with suppress(KeyError):
							t = as_timezone(arg)
							args.pop(i)
							expr = " ".join(args)
							break
					#     h = 0
					# t += h * 3600
				match = re.search(self.timefind, argv)
				if match:
					i = match.start()
					spl = [argv[:i], argv[i:]]
					msg = spl[0]
					t += utc_ts(tzparse(spl[1])) - utc()
					break
				msg = " ".join(args[:-1])
				t = utc_ts(tzparse(args[-1])) - utc()
				break
			t = 0
			if " " in argv:
				args = argv.split()
				for i in (0, -1):
					arg = args[i]
					with suppress(KeyError):
						t = as_timezone(arg)
						args.pop(i)
						expr = " ".join(args)
						break
				#     h = 0
				# t += h * 3600
			match = re.search(self.timefind, argv)
			if match:
				i = match.start()
				j = match.end()
				spl = [argv[:i], argv[i:j], argv[j:]]
				msg = spl[0] + spl[-1]
				t += await bot.eval_time(spl[1])
				break
			msg = " ".join(args[:-1])
			t = await bot.eval_time(args[-1])
			break
		if keyed:
			u = await bot.fetch_user_member(t, guild)
			t = u.id
		msg = msg.strip()
		if not msg:
			if "announce" in name:
				msg = "[SAMPLE ANNOUNCEMENT]"
			else:
				msg = "[SAMPLE REMINDER]"
			if urgent:
				msg = bold(css_md(msg, force=True))
			else:
				msg = bold(ini_md(msg))
		elif len(msg) > 4096:
			raise OverflowError(f"Input message too long ({len(msg)} > 4096).")
		username = str(remind_as)
		url = await bot.get_proxy_url(remind_as)
		ts = utc()
		if keyed:
			# Schedule for an event from a user
			rem = cdict(
				user=remind_as.id,
				msg=msg,
				u_id=t,
				t=inf,
			)
			rems.append(rem)
			s = "$" + str(t)
			seq = set_dict(bot.data.reminders, s, deque())
			seq.append(sendable.id)
			update(s)
		else:
			# Schedule for an event at a certain time
			rem = cdict(
				user=remind_as.id,
				msg=msg,
				t=t + ts,
			)
			rems.append(rem)
		if recur:
			rem.recur = recur
		# Sort list of reminders
		bot.data.reminders[sendable.id] = sort(rems, key=lambda x: x["t"])
		with suppress(IndexError):
			# Remove existing schedule
			bot.data.reminders.listed.remove(sendable.id, key=lambda x: x[-1])
		# Insert back into bot schedule
		tup = (bot.data.reminders[sendable.id][0]["t"], sendable.id)
		if is_finite(tup[0]):
			bot.data.reminders.listed.insort(tup, key=lambda x: x[0])
		update(sendable.id)
		emb = discord.Embed(description=msg)
		emb.colour = await bot.get_colour(remind_as)
		emb.set_author(name=username, url=url, icon_url=url)
		out = comment + "\n```css\nSuccessfully set "
		if urgent:
			out += "urgent "
		if "announce" in name:
			out += f"announcement for {sqr_md(sendable)}"
		else:
			out += f"reminder for {sqr_md(sendable)}"
		if not urgent and recur:
			out += f" every {sqr_md(sec2time(recur))},"
		if keyed:
			out += f" upon next event from {sqr_md(user_mention(t))}"
		else:
			out += f" in {sqr_md(time_until(t + utc()))}"
		out += ":```"
		return await send_with_reply(message.channel, message, content=out, embed=emb)
		# return dict(content=out, embed=emb)

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos, s_id = list(map(int, vals.split("_", 2)))
		if reaction not in (None, self.directions[-1]) and u_id != user.id:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		rems = bot.data.reminders.get(s_id, [])
		sendable = await bot.fetch_messageable(s_id)
		page = 16
		last = max(0, len(rems) - page)
		if reaction is not None:
			i = self.directions.index(reaction)
			if i == 0:
				new = 0
			elif i == 1:
				new = max(0, pos - page)
			elif i == 2:
				new = min(last, pos + page)
			elif i == 3:
				new = last
			else:
				new = pos
			pos = new
		content = message.content
		if not content:
			content = message.embeds[0].description
		i = content.index("callback")
		content = "*```" + "\n" * ("\n" in content[:i]) + (
			"callback-main-reminder-"
			+ str(u_id) + "_" + str(pos) + "_" + str(s_id)
			+ "-\n"
		)
		if not rems:
			content += f"Schedule for {str(sendable).replace('`', '')} is currently empty.```*"
			msg = ""
		else:
			t = utc()
			content += f"{len(rems)} message{'s' if len(rems) != 1 else ''} currently scheduled for {str(sendable).replace('`', '')}:```*"
			msg = iter2str(
				rems[pos:pos + page],
				key=lambda x: lim_str(bot.get_user(x.get("user", -1), replace=True).mention + ": `" + no_md(x["msg"]), 96) + "` ➡️ " + (user_mention(x["u_id"]) if "u_id" in x else time_until(x["t"])),
				left="`【",
				right="】`",
			)
		colour = await self.bot.get_colour(user)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		).set_author(**get_author(user))
		more = len(rems) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Note(Command):
	name = ["Trash", "Notes"]
	description = "Takes note of a given string and allows you to view and edit a to-do list!"
	usage = "(edit|delete)? <id|note>?"
	example = ("note test", "trash 1", "note edit 0 do the laundry")
	rate_limit = (6, 10)
	flags = "aed"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]

	async def __call__(self, name, message, channel, flags, bot, user, argv, **void):
		note_userbase = bot.data.notes
		if argv.startswith("edit "):
			if argv:
				argv = argv.split(None, 1)[-1]
			add_dict(flags, dict(e=1))
		elif argv.startswith("delete ") or argv.startswith("remove ") or name == "trash":
			if argv:
				argv = argv.split(None, 1)[-1]
			add_dict(flags, dict(d=1))
		if "d" in flags:
			if not argv:
				argv = 0
			try:
				n = note_userbase[user.id].pop(int(argv))
			except (KeyError, IndexError):
				argv = rank_format(int(argv))
				raise LookupError(f"You don't have a {argv} note!")
			argv = rank_format(int(argv))
			if not note_userbase.get(user.id):
				note_userbase.discard(user.id)
			else:
				note_userbase.update(user.id)
			return ini_md(f"Successfully removed {argv} note: {sqr_md(n)}")
		elif not argv:
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-main-note-"
				+ str(user.id) + "_0"
				+ "-\nLoading Notes database...```*",
				buttons=buttons,
			)
			return
		elif "e" in flags:
			pass
			# Kind of want to implement buttons for this one, so Miza will ask if the user wants to append below or to the side of an existing note in a less clunky way. Leaving this one to Txin. XD
		try:
			note_userbase[user.id].append(argv)
		except KeyError:
			note_userbase[user.id] = [argv]
		else:
			note_userbase.update(user.id)
		notecount = rank_format(len(note_userbase[user.id]) - 1)
		return ini_md(f"Successfully added {notecount} note for [{user}]!")

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		user = await bot.fetch_user(u_id)
		data = bot.data.notes
		curr = data.get(user.id, ())
		page = 16
		last = max(0, len(curr) - page)
		if reaction is not None:
			i = self.directions.index(reaction)
			if i == 0:
				new = 0
			elif i == 1:
				new = max(0, pos - page)
			elif i == 2:
				new = min(last, pos + page)
			elif i == 3:
				new = last
			else:
				new = pos
			pos = new
		content = message.content
		if not content:
			content = message.embeds[0].description
		i = content.index("callback")
		content = "*```" + "\n" * ("\n" in content[:i]) + (
			"callback-main-note-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not curr:
			content += f"No currently assigned notes for {str(user).replace('`', '')}.```*"
			msg = ""
		else:
			content += f"{len(curr)} note(s) currently assigned for {str(user).replace('`', '')}:```*"
			msg = iter2str(tuple(curr)[pos:pos + page], left="`【", right="】`")
		colour = await self.bot.get_colour(user)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		)
		emb.set_author(**get_author(user))
		more = len(curr) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class UpdateUrgentReminders(Database):
	name = "urgentreminders"

	async def _ready_(self, **void):
		try:
			self.data["listed"]
		except KeyError:
			self.data["listed"] = alist()
		create_task(self.update_urgents())

	async def update_urgents(self):
		while True:
			with tracebacksuppressor:
				t = utc()
				listed = self.data["listed"]
				while listed:
					p = listed[0]
					if t < p[0]:
						break
					with suppress(StopIteration):
						listed.popleft()
						self.update("listed")
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
						try:
							message = self.bot.cache.messages[m_id]
							if not message.reactions:
								raise KeyError
						except KeyError:
							channel = await self.bot.fetch_messageable(c_id)
							message = await discord.abc.Messageable.fetch_message(channel, m_id)
						else:
							channel = message.channel
						for react in message.reactions:
							if str(react) == "✅":
								if react.count > 1:
									raise StopIteration
								async for u in react.users():
									if u.id != self.bot.id:
										raise StopIteration
						fut = create_task(channel.send(embed=emb))
						await self.bot.silent_delete(message)
						message = await fut
						await message.add_reaction("✅")
						p[2] = message.id
						listed.insort(p, key=lambda x: x)
						self.update("listed")
			await asyncio.sleep(1)


# This database is such a hassle to manage, it has to be able to persist between bot restarts, and has to be able to update with O(1) time complexity when idle
class UpdateReminders(Database):
	name = "reminders"

	def __load__(self):
		d = self.data
		# This exists so that checking next scheduled item is O(1)
		for i in tuple(d):
			try:
				assert d[i][0]["t"]
			except:
				print_exc()
				d.pop(i, None)
		gen = ((block[0]["t"], i) for i, block in d.items() if block and isinstance(block[0], dict) and block[0].get("t") is not None)
		self.listed = alist(sorted(gen, key=lambda x: x[0]))
		self.t = utc()

	async def recurrent_message(self, channel, embed, wait=60):
		t = utc()
		message = await channel.send(embed=embed)
		await message.add_reaction("✅")
		self.bot.data.urgentreminders.data["listed"].insort([t + wait, channel.id, message.id, embed, wait], key=lambda x: x)
		self.bot.data.urgentreminders.update()

	async def __call__(self):
		if utc() - self.t >= 4800:
			create_future_ex(self.__load__)
			self.t = utc()

	# Fast call: runs many times per second
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
			if t < x["t"]:
				# Insert back into schedule if not expired
				self.listed.insort((x["t"], u_id), key=lambda x: x[0])
				print(self.listed)
				continue
			# Grab target from database
			x = cdict(temp.pop(0))
			self.update(u_id)
			if not temp:
				self.data.pop(u_id)
			else:
				# Insert next listed item into schedule
				self.listed.insort((temp[0]["t"], u_id), key=lambda x: x[0])
			# print(self.listed)
			# Send reminder to target user/channel
			ch = await self.bot.fetch_messageable(u_id)
			emb = discord.Embed(description=x.msg)
			try:
				u = self.bot.get_user(x["user"], replace=True)
			except KeyError:
				u = x
			emb.set_author(**get_author(u))
			if not x.get("recur"):
				self.bot.send_embeds(ch, emb)
			else:
				create_task(self.recurrent_message(ch, emb, x.get("recur", 60)))

	# Seen event: runs when users perform discord actions
	async def _seen_(self, user, **void):
		s = "$" + str(user.id)
		if s in self.data:
			assigned = self.data[s]
			# Ignore user events without assigned triggers
			if not assigned:
				self.data.pop(s)
				return
			with tracebacksuppressor:
				for u_id in assigned:
					# Send reminder to all targeted users/channels
					ch = await self.bot.fetch_messageable(u_id)
					rems = set_dict(self.data, u_id, [])
					pops = set()
					for i, x in enumerate(reversed(rems), 1):
						if x.get("u_id", None) == user.id:
							emb = discord.Embed(description=x["msg"])
							try:
								u = self.bot.get_user(x["user"], replace=True)
							except KeyError:
								u = cdict(x)
							emb.set_author(**get_author(u))
							if not x.get("recur"):
								self.bot.send_embeds(ch, emb)
							else:
								create_task(self.recurrent_message(ch, emb, x.get("recur", 60)))
							pops.add(len(rems) - i)
						elif is_finite(x["t"]):
							break
					it = [rems[i] for i in range(len(rems)) if i not in pops]
					rems.clear()
					rems.extend(it)
					self.update(u_id)
					if not rems:
						self.data.pop(u_id)
			with suppress(KeyError):
				self.data.pop(s)


class UpdateNotes(Database):
	name = "notes"


class UpdatePrefix(Database):
	name = "prefixes"


class UpdateEnabled(Database):
	name = "enabled"


class UpdateMessages(Database):
	name = "messages"
	semaphore = Semaphore(8, 1, delay=1, rate_limit=8)
	closed = False
	hue = 0

	@tracebacksuppressor(SemaphoreOverflowError)
	async def wrap_semaphore(self, func, *args, **kwargs):
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
							if t - v.t >= 8:
								v.t = t
								create_task(self.wrap_semaphore(eval(v.command, self.bot._globals)._callback2_, channel=channel, m_id=m_id, colour=col))

	async def _destroy_(self, **void):
		self.closed = True
		self.hue += 128
		col = colour2raw(hue2colour(self.hue))
		msg = "Offline 😔"
		for c_id, data in self.data.items():
			with tracebacksuppressor(SemaphoreOverflowError):
				channel = await self.bot.fetch_channel(c_id)
				for m_id, v in data.items():
					async with self.semaphore:
						await eval(v.command, self.bot._globals)._callback2_(channel=channel, m_id=m_id, msg=msg, colour=col)



class UpdateFlavour(Database):
	name = "flavour"

	@tracebacksuppressor
	async def get(self, p=True, q=True):
		out = None
		i = xrand(7)
		facts = self.bot.data.users.facts
		questions = self.bot.data.users.questions
		useless = self.bot.data.useless.setdefault(0, alist())
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
				if not useless or not xrand(len(useless) / 8):
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
					if c_id not in bot.cache.channels:
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
		self.semaphore = Semaphore(1, 1, rate_limit=120)
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

	fetch_events = lambda self, u_id, interval=3600: {i: self.get_events(u_id, interval=interval, event=i) for i in ("message", "typing", "command", "reaction", "misc")}

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

	def get_timezone(self, u_id):
		timezone = self.data.get(u_id, EMPTY).get("timezone")
		if timezone is not None:
			return round_min(as_timezone(timezone) / 3600)

	def estimate_timezone(self, u_id):
		data = self.data.get(u_id, EMPTY).get("recent")
		if not data:
			return 0
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
		# print(estimated, inactive, activity)
		return estimated

	async def __call__(self):
		changed = False
		with tracebacksuppressor(SemaphoreOverflowError):
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
		self.update(user.id)

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
		self.update(u_id)

	# User executed command, add to activity database
	def _command_(self, user, loop, command, **void):
		self.send_event(user.id, "command")
		add_dict(self.data, {user.id: {"commands": {command.parse_name(): 1}}})
		self.data[user.id]["last_used"] = utc()
		self.data.get(user.id, EMPTY).pop("last_mention", None)
		if not loop:
			self.add_xp(user, getattr(command, "xp", xrand(6, 14)))

	async def react_sparkle(self, message):
		bot = self.bot
		react = await create_future(bot.data.emojis.get, "sparkles.gif")
		return await message.add_reaction(react)

	def _send_(self, message, **void):
		user = message.author
		if user.id == self.bot.id or self.bot.get_perms(user, message.guild) <= -inf:
			return
		if not self.bot.get_enabled(message.channel):
			return
		size = get_message_length(message)
		points = math.sqrt(size) + sum(1 for w in message.content.split() if len(w) > 1)
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
		if message.id % 1000 == 0:
			self.add_diamonds(user, points)
			points *= 1000
			# create_task(message.add_reaction("✨"))
			if self.bot.data.enabled.get(message.channel.id, True):
				create_task(self.react_sparkle(message))
			print(f"{user} has triggered the rare message bonus in {message.guild}!")
		else:
			self.add_gold(user, points)
		self.add_xp(user, points)
		if "dailies" in self.bot.data:
			create_task(self.bot.data.dailies.valid_message(message))

	async def _mention_(self, user, message, msg, **void):
		bot = self.bot
		mentions = self.mentionspam.findall(msg)
		t = utc()
		out = None
		if len(mentions) >= xrand(8, 12) and self.data.get(user.id, EMPTY).get("last_mention", 0) > 3:
			out = f"{choice('🥴😣😪😢')} Please calm down a second, I'm only here to help..."
		elif len(mentions) >= 3 and (self.data.get(user.id, EMPTY).get("last_mention", 0) > 2 or random.random() >= 2 / 3):
			out = f"{choice('😟😦😓')} Oh, that's a lot of mentions, is everything okay?"
		elif len(mentions) >= 2 and self.data.get(user.id, EMPTY).get("last_mention", 0) > 0 and random.random() >= 0.75:
			out = "One mention is enough, but I appreciate your enthusiasm 🙂"
		if out:
			create_task(send_with_react(message.channel, out, reacts="❎", reference=message))
			await bot.seen(user, event="misc", raw="Being naughty")
			add_dict(self.data, {user.id: {"last_mention": 1}})
			self.data[user.id]["last_used"] = t
			self.update(user.id)
			with suppress():
				message.noresponse = True
			raise CommandCancelledError

	def get_xp(self, user):
		if self.bot.is_blacklisted(user.id):
			return -inf
		if user.id == self.bot.id:
			if self.data.get(self.bot.id, EMPTY).get("xp", 0) != inf:
				set_dict(self.data, self.bot.id, {})["xp"] = inf
				self.data[self.bot.id]["gold"] = inf
				self.data[self.bot.id]["diamonds"] = inf
				self.update(self.bot.id)
			return inf
		return self.data.get(user.id, EMPTY).get("xp", 0)

	def xp_to_level(self, xp):
		if is_finite(xp):
			return int((xp * 3 / 2000) ** (2 / 3)) + 1
		return xp

	def xp_to_next(self, level):
		if is_finite(level):
			return ceil(math.sqrt(level - 1) * 1000)
		return level

	def xp_required(self, level):
		if is_finite(level):
			return ceil((level - 1) ** 1.5 * 2000 / 3)
		return level

	async def get_balance(self, user):
		data = self.data.get(user.id, EMPTY)
		return await self.bot.as_rewards(data.get("diamonds"), data.get("gold"))

	def add_xp(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user)
			if pl:
				amount *= 2 ** pl
			add_dict(set_dict(self.data, user.id, {}), {"xp": amount})
			if "dailies" in self.bot.data:
				self.bot.data.dailies.progress_quests(user, "xp", amount)
			self.update(user.id)

	def add_gold(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user, absolute=True)
			if pl:
				amount *= 2 ** pl
			add_dict(set_dict(self.data, user.id, {}), {"gold": amount})
			if amount < 0 and self[user.id]["gold"] < 0:
				self[user.id]["gold"] = 0
			self.update(user.id)

	def add_diamonds(self, user, amount, multiplier=True):
		if user.id != self.bot.id and amount and not self.bot.is_blacklisted(user.id):
			pl = self.bot.premium_level(user, absolute=True)
			if pl:
				amount *= 2 ** pl
			add_dict(set_dict(self.data, user.id, {}), {"diamonds": amount})
			if amount > 0 and "dailies" in self.bot.data:
				self.bot.data.dailies.progress_quests(user, "diamond", amount)
			if amount < 0 and self[user.id]["diamonds"] < 0:
				self[user.id]["diamonds"] = 0
			self.update(user.id)

	async def _typing_(self, user, **void):
		set_dict(self.data, user.id, {})["last_typing"] = utc()
		self.update(user.id)

	async def _nocommand_(self, message, msg, force=False, flags=(), truemention=True, perm=0, **void):
		if getattr(message, "noresponse", False):
			return
		bot = self.bot
		user = message.author
		channel = message.channel
		guild = message.guild
		if force or bot.is_mentioned(message, bot, guild):
			if user.bot:
				with suppress(AttributeError):
					async for m in self.bot.data.channel_cache.get(channel):
						user = m.author
						if bot.get_perms(user.id, guild) <= -inf:
							return
						if not user.bot:
							break
			if not isnan(perm) and "blacklist" in self.bot.data:
				gid = self.bot.data.blacklist.get(0)
				if gid and gid != getattr(guild, "id", None):
					create_task(send_with_react(
						channel,
						"I am currently under maintenance, please stay tuned!",
						reacts="❎",
						reference=message,
					))
					return
			send = lambda *args, **kwargs: send_with_reply(channel, not flags and message, *args, **kwargs)
			out = None
			count = self.data.get(user.id, EMPTY).get("last_talk", 0)
			# Simulates a randomized conversation
			if count < 5:
				create_task(message.add_reaction("👀"))
			if "ask" in bot.commands and ("string" in bot.get_enabled(channel) or not perm < inf):
				argv = message.clean_content.strip()
				me = guild.me if guild else bot
				argv = argv.removeprefix(f"@{me.display_name}")
				argv = argv.removesuffix(f"@{me.display_name}")
				argv = argv.strip()
				with bot.ExceptionSender(channel, reference=message):
					u_perm = bot.get_perms(user.id, guild)
					u_id = user.id
					for ask in bot.commands.ask:
						command = ask
						req = command.min_level
						if not isnan(u_perm):
							if not u_perm >= req:
								raise command.perm_error(u_perm, req, "for command ask")
							x = command.rate_limit
							if x:
								x2 = x
								if user.id in bot.owners:
									x = x2 = 0
								elif isinstance(x, collections.abc.Sequence):
									x = x2 = x[not bot.is_trusted(getattr(guild, "id", 0))]
									x /= 2 ** bot.premium_level(user)
									x2 /= 2 ** bot.premium_level(user, absolute=True)
								# remaining += x
								d = command.used
								t = d.get(u_id, -inf)
								wait = utc() - t - x
								if wait > min(1 - x, -1):
									if x < x2 and (utc() - t - x2) < min(1 - x2, -1):
										bot.data.users.add_diamonds(user, (x - x2) / 100)
									if wait < 0:
										w = -wait
										d[u_id] = max(t, utc()) + w
										await asyncio.sleep(w)
									if len(d) >= 4096:
										with suppress(RuntimeError):
											d.pop(next(iter(d)))
									d[u_id] = max(t, utc())
								else:
									raise TooManyRequests(f"Command has a rate limit of {sec2time(x)}; please wait {sec2time(-wait)}.")
						m = await ask(message, guild, channel, user, argv, name="ask", flags=flags)
						if m and "exec" in bot.data and not message.guild and ("blacklist" not in bot.data or (bot.data.blacklist.get(user.id) or 0) < 1):
							await bot.data.exec._nocommand_(message=m)
				return
			if count:
				if count < 2 or count == 2 and xrand(2):
					# Starts conversations
					out = choice(
						f"So, {user.display_name}, how's your day been?",
						f"How do you do, {user.name}?",
						f"How are you today, {user.name}?",
						"What's up?",
						"Can I entertain you with a little something today?",
					)
				elif count < 16 or random.random() > math.atan(max(0, count / 8 - 3)) / 4:
					# General messages
					if (count < 6 or self.mentionspam.sub("", msg).strip()) and random.random() < 0.5:
						out = choice((f"'sup, {user.display_name}?", f"There you are, {user.name}!", "Oh yeah!", "👋", f"Hey, {user.display_name}!"))
					else:
						out = ""
				elif count < 24:
					# Occasional late message
					if random.random() < 0.4:
						out = choice(
							"You seem rather bored... I may only be as good as my programming allows me to be, but I'll try my best to fix that! 🎆",
							"You must be bored, allow me to entertain you! 🍿",
						)
					else:
						out = ""
				else:
					# Late conversation messages
					out = choice(
						"It's been a fun conversation, but don't you have anything better to do? 🌞",
						"This is what I was made for, I can do it forever, but you're only a human, take a break! 😅",
						f"Woah, have you checked the time? We've been talking for {count + 1} messages! 😮"
					)
			elif utc() - self.data.get(user.id, EMPTY).get("last_used", inf) >= 259200:
				# Triggers for users not seen in 3 days or longer
				out = choice((f"Long time no see, {user.name}!", f"Great to see you again, {user.display_name}!", f"It's been a while, {user.name}!"))
			if out is not None:
				guild = message.guild
				# Add randomized flavour text if in conversation
				if not xrand(4):
					front = choice(
						"Random question",
						"Question for you",
						"Conversation starter",
					)
					out += f"\n{front}: `{choice(self.questions)}`"
				elif self.flavour_buffer:
					out += self.flavour_buffer.popleft()
				else:
					out += choice(self.flavour)
			else:
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
				out += f" Use `{prefix}help` or `/help` for help!"
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
						stored[channel.id] = message.id
					except:
						print_exc()
						stored.pop(c_id, None)
						stored[channel.id] = message.id
			if channel.id in bot.cache.channels:
				stored[channel.id] = message.id
		self.update(user.id)
