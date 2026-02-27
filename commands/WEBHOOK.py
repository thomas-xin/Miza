# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


class AutoEmoji(Pagination, Command):
	server_only = True
	name = ["NQN", "Emojis"]
	min_level = 0
	description = "Causes all failed emojis starting and ending with : to be deleted and reposted with a webhook, when possible. See ~emojilist for assigned emojis. Enabled by default, unless NQN (<@559426966151757824>) is in the server."
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
	)
	rate_limit = (4, 6)
	slash = True

	async def __call__(self, bot, _user, _guild, mode, **void):
		data = bot.data.autoemojis
		if mode == "enable":
			data[_guild.id] = True
			return italics(css_md(f"Enabled automatic emoji substitution for {sqr_md(_guild)}."))
		elif mode == "disable":
			data[_guild.id] = False
			return italics(css_md(f"Disabled automatic emoji substitution for {sqr_md(_guild)}."))
		# Set callback message for scrollable list
		return await self.display(_user.id, 0, _guild.id)

	def react_perms(self, perm: int):
		return False if perm < 2 else True

	async def display(self, uid, pos, gid, diridx=-1):
		bot = self.bot

		def key(curr, pos, page):
			return iter2str({(k := f":{e.name}:") + " " * (32 - len(k)): f"({e.id})` {min_emoji(e)}" for e in tuple(curr)[pos:pos + page]}, left="`", right="")

		guild = await bot.fetch_guild(gid)
		return await self.default_display(
			"custom emoji", uid, pos,
			sorted((e for e in guild.emojis if e.is_usable()), key=lambda e: full_prune(e.name)),
			diridx, extra=leb128(gid), key=key,
		)

	async def _callback_(self, _user, index, data, **void):
		print(data)
		pos, more = decode_leb128(data)
		gid, _ = decode_leb128(more)
		return await self.display(_user.id, pos, gid, index)


class UpdateAutoEmojis(Database):
	name = "autoemojis"

	def guild_emoji_map(self, guild, user=None, emojis={}):
		if user:
			guilds = sorted(getattr(user, "mutual_guilds", None) or [g for g in self.bot.guilds if user.id in g._members], key=lambda g: g.id)
			try:
				guilds.remove(guild)
			except ValueError:
				pass
			guilds.insert(0, guild)
		else:
			guilds = list(self.bot.guilds)
			if guild:
				guilds.remove(guild)
				guilds.insert(0, guild)
			else:
				guild = guilds[0]
		if user:
			elist = self.bot.get_userbase(user.id, "emojilist", {})
			for n, e_id in sorted(elist.items(), key=lambda t: t[1]):
				while n in emojis:
					if emojis[n] == e_id:
						break
					t = n.rsplit("-", 1)
					if t[-1].isnumeric():
						n = t[0] + "-" + str(int(t[-1]) + 1)
					else:
						n = t[0] + "-1"
				emojis[n] = e_id
		for g in guilds:
			for e in sorted(g.emojis, key=lambda e: e.id):
				if not e.is_usable():
					continue
				n = e.name
				while n in emojis:
					if emojis[n] == e.id:
						break
					t = n.rsplit("-", 1)
					if t[-1].isnumeric():
						n = t[0] + "-" + str(int(t[-1]) + 1)
					else:
						n = t[0] + "-1"
				emojis[n] = e
		if not user and guild:
			elist = self.bot.get_guildbase(guild.id, "emojilist", {})
			for n, e_id in sorted(elist.items(), key=lambda t: t[1]):
				while n in emojis:
					if emojis[n] == e_id:
						break
					t = n.rsplit("-", 1)
					if t[-1].isnumeric():
						n = t[0] + "-" + str(int(t[-1]) + 1)
					else:
						n = t[0] + "-1"
				emojis[n] = e_id
		return emojis

	def is_enabled_in(self, guild):
		# Special case: Auto-disable if NQN bot is detected (as we do not want a delete-proxy conflict/race condition!)
		if not self.get(guild.id, True) or (559426966151757824 in guild._members or not guild.me.guild_permissions.manage_messages or not guild.me.guild_permissions.manage_webhooks):
			return False
		return True

	async def _reaction_add_(self, message, emoji, user, **void):
		if not emoji or isinstance(emoji, str):
			return
		e = str(emoji)
		bot = self.bot
		if bot.is_optout(user):
			return
		name, e_id = e.split(":")[1:]
		e_id = int("".join(regexp("[0-9]+").findall(e_id)))
		anim = e.startswith("<a:")
		bot.emoji_animated[e_id] = anim
		emoji = bot.cache.emojis.get(e_id)
		if emoji:
			name = emoji.name
		if not message.webhook_id:
			bot.emojinames[e_id] = name
			if user:
				elist = self.bot.get_userbase(user.id, "emojilist", {})
				elist[name] = e_id
				self.bot.set_userbase(user.id, "emojilist", elist)
			guild = message.guild
			if guild:
				elist = self.bot.get_guildbase(guild.id, "emojilist", {})
				elist[name] = e_id
				self.bot.set_guildbase(guild.id, "emojilist", elist)

	async def _nocommand_(self, message, recursive=True, **void):
		if getattr(message, "simulated", None) or (utc_ddt() - message.created_at).total_seconds() > 3600:
			return
		user = message.author
		if message.guild and not message.guild.get_member(user.id) or not message.content or getattr(message, "webhook_id", None) or message.content.count("```") > 1:
			return
		bot = self.bot
		if bot.is_optout(user):
			return
		emojis = find_emojis(message.content)
		for e in emojis:
			name, e_id = e.split(":")[1:]
			e_id = int("".join(regexp("[0-9]+").findall(e_id)))
			anim = e.startswith("<a:")
			bot.emoji_animated[e_id] = anim
			emoji = bot.cache.emojis.get(e_id)
			if emoji:
				name = emoji.name
			if not message.webhook_id:
				bot.emojinames[e_id] = name
				if user:
					elist = bot.get_userbase(user.id, "emojilist", {})
					elist[name] = e_id
					bot.set_userbase(user.id, "emojilist", elist)
				guild = message.guild
				if guild:
					elist = bot.get_guildbase(guild.id, "emojilist", {})
					elist[name] = e_id
					bot.set_guildbase(guild.id, "emojilist", elist)
		if not message.guild or not message.guild.me:
			return
		guild = message.guild
		if not self.is_enabled_in(guild):
			return
		if "emojis" in bot.data:
			await bot.data.emojis.load_own()
		m_id = None
		msg = message.content
		ref = message.reference and await bot.fetch_reference(message)
		orig = bot.get_userbase(user.id, "emojilist", {})
		emojis = None
		# long = len(msg) > 32
		if msg.startswith("+"):
			emi = msg[1:].strip()
			spl = emi.rsplit(None, 1)
			if len(spl) > 1:
				ems, m_id = spl
				if not m_id.isnumeric():
					spl = [emi]
			if len(spl) == 1:
				ems = spl[0]
				m2 = await bot.history(message.channel, limit=5, before=message.id, use_cache=not bot.is_trusted(guild)).__anext__()
			else:
				m2 = None
				if m_id:
					m_id = int(m_id)
			if not m2 and m_id:
				try:
					m2 = await bot.fetch_message(m_id, message.channel)
				except LookupError:
					m2 = None
			if m2:
				ems = regexp(r"<a?:[A-Za-z0-9\-~_]{1,32}").sub("", ems.replace(" ", "").replace("\\", "")).replace(">", ":")
				possible = regexp(r":[A-Za-z0-9\-~_]{1,32}:|[^\x00-\x7F]").findall(ems)
				s = ems
				for word in possible:
					s = s.replace(word, "")
				if s.strip():
					return
				futs = deque()
				possible = (n.strip(":") for n in possible)
				for name in (n for n in possible if n):
					emoji = None
					if emojis is None:
						emojis = self.guild_emoji_map(guild, message.author, dict(orig))
					if ord(name[0]) >= 128:
						emoji = name
					else:
						emoji = emojis.get(name)
					if not emoji:
						r1 = regexp(r"^[A-Za-z0-9\-~_]{1,32}$")
						if r1.fullmatch(name):
							if name.isnumeric():
								emoji = int(name)
							else:
								t = name[::-1].replace("~", "-", 1)[::-1].rsplit("-", 1)
								if t[-1].isnumeric():
									i = int(t[-1])
									if i < 1000:
										if not emoji:
											name = t[0]
											emoji = emojis.get(name)
										while i > 1 and not emoji:
											i -= 1
											name = t[0] + "-" + str(i)
											emoji = emojis.get(name)
					if emoji:
						if isinstance(emoji, int):
							emoji = await bot.fetch_emoji(emoji, guild=message.guild)
						futs.append(m2.add_reaction(emoji))
						orig = bot.get_userbase(user.id, "emojilist", {})
						if getattr(emoji, "id", None):
							orig[name] = emoji.id
							bot.emojinames[emoji.id] = name
				if futs:
					futs.append(bot.autodelete(message))
					await gather(*futs)
					return
		if message.content.count(":") < 2:
			return
		cbreg = regexp("`[^`]*`")
		selected, deselected = [], []
		while msg:
			cb = cbreg.search(msg)
			if not cb:
				selected.append(msg)
				break
			selected.append(msg[:cb.start()])
			deselected.append(msg[cb.start():cb.end()])
			msg = msg[cb.end():]
		pops = set()
		replaceds = []
		for i, m in enumerate(selected):
			m, p, r = await bot.proxy_emojis(m, guild=guild, user=user, is_webhook=message.webhook_id, return_pops=True)
			selected[i] = m
			pops.update(p)
			replaceds.extend(r)
		msg = selected.pop(0)
		for d, s in zip(deselected, selected):
			msg += d + s
		if not msg or msg == message.content or not replaceds:
			return
		msg = escape_everyone(msg).strip("\u200b")
		if not msg or msg == message.content or len(msg) > 2000:
			return
		if not recursive:
			return msg
		replacemap = {e.name: e for e in replaceds}
		files = deque()
		for a in message.attachments:
			fn = await attachment_cache.download(a.url, m_id=message.id, filename=True)
			files.append(discord.File(fn, filename=a.filename))
		csubmit(bot.autodelete(message))
		url = await bot.get_proxy_url(user)
		m = await bot.send_as_webhook(message.channel, msg, files=files, username=user.display_name, avatar_url=url, reference=ref)
		await bot.send_event("_command_", user=user, command=bot.commands.autoemoji[0], loop=False, message=message)
		regex = regexp(r"(?:^|^[^<\\`]|[^<][^\\`]|.[^a\\`\[])(:[A-Za-z0-9\-~_]{1,32}:)(?:(?![^0-9]).)*(?:$|[^0-9>\]`])")
		if recursive and regex.search(m.content):
			m = await m.edit(content=msg)
			print("R1:", m, m.content)
			if (res := regex.search(m.content)):
				msg = m.content
				while res:
					em = res.group()
					name = em.split(":", 2)[1]
					em = f":{name}:"
					try:
						e = replacemap[em]
					except KeyError:
						e = replaceds[-1]
					url = await bot.emoji_to_url(e)
					if msg == em:
						msg = f"[{name}]({url}?size=48&name={name})"
						break
					msg = msg.replace(em, f"[{name}]({url}?size=24&name={name})")
					res = regex.search(msg)
				m = await m.edit(content=msg)
				print("R2:", m, m.content)
				if regex.search(m.content):
					for k in tuple(pops):
						if str(k[1]) not in m.content:
							orig.pop(k[0], None)
						else:
							pops.discard(k)
					print("Removed emojis:", pops)
					emb = discord.Embed()
					emb.set_author(**get_author(bot.user))
					emb.description = (
						"Psst! It appears as though AutoEmoji has failed to convert an emoji."
						+ " To fix this, either delete my webhook for this channel and create a new one manually, add the emoji to this server, or invite me to the server containing said emoji!"
					)
					await m.edit(embed=emb)
					self.pop(m.channel.id)
					bot.data.webhooks.temp.pop(m.channel.id, None)


class EmojiList(Pagination, Command):
	name = ["CustomEmoji", "ProxyEmoji"]
	description = "Sets a custom alias for an emoji, usable by ~autoemoji. Accepts emojis, emoji IDs, emoji URLs, and message links containing emojis or reactions."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("view", "add", "remove"),
				accepts=dict(enable="add", disable="remove", create="add", delete="remove"),
			),
			description="Action to perform",
			example="enable",
			default="view",
		),
		name=cdict(
			type="word",
			description="Emoji name",
			example="cat3",
		),
		emoji=cdict(
			type="emoji",
			description="Emoji to select",
			example="988721351969865729",
		),
	)
	rate_limit = (4, 6)

	async def __call__(self, bot, _user, mode, name, emoji, **void):
		curr = bot.get_userbase(_user.id, "emojilist", {})
		emote = getattr(emoji, "id", emoji)
		match mode:
			case "view":
				# Set callback message for scrollable list
				return await self.display(_user.id, 0)
			case "remove":
				removed = []
				if emote:
					for k, v in tuple(curr.items()):
						if v == emote:
							removed.append(curr.pop(k))
							name = name or k
					if not removed:
						raise KeyError(f'Emoji "{emoji}" not assigned.')
				elif name:
					try:
						removed.append(curr.pop(name))
					except KeyError:
						raise KeyError(f'Emoji "{emoji}" not assigned.')
				else:
					removed.extend(curr.values())
					curr.clear()
				if len(removed) > 1:
					content = f"Successfully removed {sqr_md(len(removed))} emoji aliases for {sqr_md(_user)}."
				else:
					content = f"Successfully removed emoji alias {sqr_md(name)}: {sqr_md(removed[0])} for {sqr_md(_user)}."
				bot.set_userbase(_user.id, "emojlilist", curr)
				return cdict(
					content=content,
					prefix="```css\n",
					suffix="```",
				)
			case "add":
				assert emoji, "Emoji required to add."
				assert isinstance(emote, int), "Emoji should be a valid Discord emoji."
				if not name:
					name = emoji.get("name", emote)
				name = name.strip(":")
				if not regexp(r"[A-Za-z0-9\-~_]{1,32}").fullmatch(name):
					raise ArgumentError("Emoji aliases may only contain 1~32 alphanumeric characters, dashes, tildes and underscores.")
				curr[name] = emote
				bot.set_userbase(_user.id, "emojlilist", curr)
				return ini_md(f"Successfully added emoji alias {sqr_md(name)}: {sqr_md(emote)} for {sqr_md(_user)}.")
		raise NotImplementedError(mode)

	async def display(self, uid, pos, diridx=-1):
		bot = self.bot

		async def akey(curr, pos, page):
			mapping = {}
			futs = []
			for k, v in sorted(curr, key=lambda n: full_prune(n[0]))[pos:pos + page]:

				async def check_emoji(k, v):
					try:
						try:
							e = bot.cache.emojis[v]
							if not e.is_usable():
								raise LookupError
							me = " " + str(e)
						except KeyError:
							await bot.is_animated(v)
							me = ""
					except LookupError:
						curr = bot.get_userbase(uid, "emojilist", {})
						curr.pop(k)
						bot.set_userbase(uid, "emojlilist", curr)
						return
					return f"({v})` {me}"

				fut = csubmit(check_emoji(k, v))
				fut.k = k
				futs.append(fut)
			for fut in futs:
				info = await fut
				if not info:
					continue
				mapping[f":{fut.k}:"] = info
			return iter2str({k + " " * (32 - len(k)): v for k, v in mapping.items()}, left="`", right="").strip()

		return await self.default_display("proxy emoji", uid, pos, bot.get_userbase(uid, "emojlisit", {}).items(), diridx, akey=akey)

	async def _callback_(self, _user, index, data, **void):
		pos, _ = decode_leb128(data)
		return await self.display(_user.id, pos, index)


class Proxy(Command):
	name = ["Mimic", "Plural", "RolePlay", "RP", "RPCreate"]
	description = "Creates a webhook proxy from a name or user, or lists all proxies with their respective prefixes. Proxies are tied to their creator user, and require permission level of 1 to invoke."
	schema = cdict(
		prefix=cdict(
			type="word",
			description="Prefix used to invoke the proxy",
			example="%test",
		),
		name=cdict(
			type="string",
			description="Name of the proxy. Must be 80 or fewer in length",
			example="Test Webhook",
			greedy=False,
			excludes=("user",),
		),
		icon=cdict(
			type="visual",
			description="Icon of the proxy, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			excludes=("user",),
		),
		user=cdict(
			type="user",
			description="User account to clone, for mimic mode.",
			example="668999031359537205",
			excludes=("name", "icon"),
		),
		delete=cdict(
			type="word",
			description='Prefix or ID of proxy to delete. Enter "-" to delete all',
			example="%test",
		),
	)
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (6, 7)
	maintenance = True

	async def __call__(self, bot, _user, _perm, prefix, name, icon, user, delete, **void):
		mimicdb = bot.data.mimics
		if delete:
			if delete == "-":
				mimics = mimicdb.pop(user.id)
				for mlist in mimics.values():
					for m_id in mlist:
						mimicdb.pop(m_id, None)
				return italics(css_md(f"Successfully removed all {sqr_md(len(mimics))} webhook proxies for {sqr_md(_user)}."))
			mimics = mimicdb.get(user.id, {})
			try:
				mlist = mimics[prefix]
				if not mlist:
					raise KeyError
				m_id = mlist.pop(0)
				mimic = mimicdb.pop(m_id)
				if not mlist:
					mimics.pop(prefix)
			except KeyError:
				mimic = bot.get_mimic(prefix, user)
				# Users are not allowed to delete proxies that do not belong to them
				if not isnan(_perm) and mimic.u_id != user.id:
					raise PermissionError("Target proxy does not belong to you.")
				mimics = mimicdb[mimic.u_id]
				user = await bot.fetch_user(mimic.u_id)
				m_id = mimic.id
				for prefix in mimics:
					with suppress(ValueError, IndexError):
						mimics[prefix].remove(m_id)
				mimicdb.pop(mimic.id)
			return italics(css_md(f"Successfully removed webhook proxy {sqr_md(mimic.name if mimic else prefix)} for {sqr_md(_user)}."))
		if not prefix:
			if not name and not user:
				# Set callback message for scrollable list
				buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
				return cdict(
					"*```callback-webhook-mimic-"
					+ str(user.id) + "_0"
					+ "-\nLoading Proxy database...```*",
					buttons=buttons,
				)
			raise IndexError("Prefix must not be empty.")
		if len(prefix) > 32:
			raise OverflowError("Prefix must be 32 or fewer in length.")
		if " " in prefix:
			raise TypeError("Prefix must not contain spaces.")
		mimics = mimicdb.setdefault(_user.id, {})
		if sum(map(len, mimics.values())) >= 32768:
			raise OverflowError(f"Proxy list for {user} has reached the maximum of 32768 items. Please remove an item to add another.")
		dt = DynamicDT.utcnow()
		mid = time_snowflake(dt)
		m_id = "&" + str(mid)
		while m_id in mimicdb:
			mid += 1
			m_id = "&" + str(mid)
		mimic = None
		# Attempt to create a new mimic, a mimic from a user, or a copy of an existing mimic.
		if user:
			name = user.name
			url = await bot.get_proxy_url(user)
			mimic = cdict(
				id=m_id,
				u_id=_user.id,
				prefix=prefix,
				clone=user.id,
				name=name,
				url=url,
				created_at=dt.timestamp(),
				count=0,
				total=0,
			)
		else:
			# This limit is actually to comply with webhook usernames
			if len(name) > 80:
				raise OverflowError("Name must be 80 or fewer in length.")
			mimic = cdict(
				id=m_id,
				u_id=_user.id,
				prefix=prefix,
				name=name,
				url=icon,
				created_at=dt.timestamp(),
				count=0,
				total=0,
			)
		mimicdb[m_id] = mimic
		mimics.setdefault(prefix, []).append(m_id)
		out = f"Successfully added webhook mimic {sqr_md(mimic.name)} with prefix {sqr_md(mimic.prefix)} and ID {sqr_md(mimic.id)}"
		if user:
			out += f", bound to user {sqr_md(user_mention(user))}"
		return css_md(out)

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and u_id != user.id and perm <= inf:
			return
		if reaction not in self.directions and reaction is not None:
			return
		mimicdb = bot.data.mimics
		user = await bot.fetch_user(u_id)
		mimics = mimicdb.get(user.id, {})
		for k in tuple(mimics):
			if not mimics[k]:
				mimics.pop(k)
		page = 24
		last = max(0, len(mimics) - page)
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
			"callback-webhook-mimic-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not mimics:
			content += f"No currently enabled webhook mimics for {str(user).replace('`', '')}.```*"
			msg = ""
		else:
			content += f"{len(mimics)} currently enabled webhook mimic(s) for {str(user).replace('`', '')}:```*"
			def key(x):
				return lim_str("⟨" + ", ".join(i + ": " + (str(no_md(mimicdb[i].get("name"))), "[<@" + str(mimicdb[i].get("auto", "None")) + ">]")[bool(mimicdb[i].get("auto"))] for i in iter(x)) + "⟩", 1900 / len(mimics))
			msg = ini_md(iter2str({k: mimics[k] for k in sorted(mimics)[pos:pos + page]}, key=key))
		colour = await bot.get_colour(user)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		)
		emb.set_author(**get_author(user))
		more = len(mimics) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		csubmit(bot.edit_message(message, content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class ProxyConfig(Command):
	name = ["PluralConfig", "MimicConfig", "RPConfig"]
	description = "Modifies an existing webhook proxy's attributes."
	schema = cdict(
		proxy=cdict(
			type="word",
			description="Prefix or ID of proxy to invoke",
			example="%test",
			required=True,
		),
		prefix=cdict(
			type="word",
			description="New prefix used to invoke the proxy",
			example="%test",
		),
		name=cdict(
			type="string",
			description="New name of the proxy. Must be 80 or fewer in length",
			example="Test Webhook",
			greedy=False,
		),
		icon=cdict(
			type="visual",
			description="New icon of the proxy, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
		),
		user=cdict(
			type="user",
			description="New user account to clone, for mimic mode. Overrides both name and icon when applied",
			example="668999031359537205",
		),
		personality=cdict(
			type="string",
			description="New personality of the character, to be used in the chatbot. Invoke using {PREFIX}+",
			example="Your name is {{char}}, and you are an easily-annoyed dog who always barks at humans.",
			greedy=False,
		),
		gender=cdict(
			type="string",
			description="New gender of the character, only affects viewing by info",
			example="she/they",
			greedy=False,
		),
		birthday=cdict(
			type="date",
			description="New birthday of the character, only affects viewing by info",
			example="february 29th",
		),
	)
	rate_limit = (4, 5)

	async def __call__(self, bot, _guild, _user, _perm, proxy, prefix, name, icon, user, personality, gender, birthday, **void):
		mimics = bot.data.mimics.get(_user, {})
		mimic = bot.get_mimic(proxy, _user)
		mimic = await bot.data.mimics.update_mimic(mimic, _guild)
		if mimic.u_id != _user.id and not isnan(_perm):
			raise PermissionError(f"Target mimic {mimic.name} does not belong to you.")
		m_id = mimic.id
		if prefix is not None:
			assert len(prefix) < 32, "Prefix must be 32 or fewer in length."
			mimics[mimic.prefix].remove(m_id)
			mimics.setdefault(prefix, []).append(m_id)
			mimic.prefix = prefix
		if name is not None:
			assert len(name) < 80, "Name must be 80 or fewer in length."
			mimic.name = name
		if icon is not None:
			mimic.url = icon
		if user is not None:
			mimic.clone = user
		if personality is not None:
			assert len(personality) < 2000, "Personality must be 2000 or fewer in length."
			mimic.personality = personality
		if gender is not None:
			mimic.gender = gender
		if birthday is not None:
			mimic.birthday = birthday.timestamp()
		return "**Updated proxy**:\n" + css_md(iter2str(mimic))


class ProxySend(Command):
	name = ["MimicSend", "RPSend", "PluralSend", "Invoke"]
	description = "Sends a message using a webhook proxy, to the target channel."
	schema = cdict(
		proxy=cdict(
			type="word",
			description="Prefix or ID of proxy to invoke",
			example="%test",
			required=True,
		),
		channel=cdict(
			type="channel",
			description="Target channel to send to",
			example="#general",
			required=True,
		),
		message=cdict(
			type="string",
			description="Message to send",
			example="Hello World!",
			required=True,
			greedy=False,
		),
		invoke=cdict(
			type="bool",
			description="Whether to invoke the proxy's AI, if applicable",
			default=True,
		),
	)

	async def __call__(self, bot, _user, _message, proxy, channel, message, invoke, **void):
		assert getattr(channel, "guild", None), "Webhooks are only usable in servers."
		guild = channel.guild
		mimic = bot.get_mimic(proxy, _user)
		perm = bot.get_perms(_user.id, guild)
		admin = not inf > perm
		try:
			enabled = bot.data.enabled[channel.id]
		except KeyError:
			enabled = bot.data.enabled.get(guild.id, ())
		invoking = invoke and mimic.personality
		# Because this command operates across channels and servers, we need to make sure these cannot be sent to channels without this command enabled
		if not admin and ("webhook" not in enabled or perm < 1):
			raise PermissionError("Not permitted to send into target channel.")
		if not admin and ("ai" not in enabled and invoking):
			raise PermissionError("AI is not enabled in target channel.")
		csubmit(_message.add_reaction("👀"))
		await bot.data.mimics.invoke_mimic(_message, mimic, channel, message, invoking)


class UpdateMimics(Database):
	name = "mimics"
	no_delete = True

	async def _nocommand_(self, message, **void):
		if not message.content:
			return
		user = message.author
		guild = message.guild
		if user.id not in self.data:
			return
		bot = self.bot
		if bot.is_optout(user):
			return
		perm = bot.get_perms(user.id, guild)
		if perm < 1:
			return
		admin = not inf > perm
		if guild is not None:
			try:
				enabled = bot.data.enabled[message.channel.id]
			except KeyError:
				enabled = ()
		else:
			enabled = list(bot.categories)
		# User must have permission to use ~mimicsend in order to invoke by prefix
		if not admin and "webhook" not in enabled:
			return
		database = self.data[user.id]
		msg = message.content
		with bot.ExceptionSender(message.channel, Exception, reference=message):
			# Stack multiple messages to send, may be separated by newlines
			sending = alist()
			channel = message.channel
			for line in msg.splitlines():
				found = False
				# O(1) time complexity per line regardless of how many mimics a user is assigned
				if len(line) > 2 and " " in line:
					prefix, line = line.split(" ", 1)
					if prefix in database:
						mimics = database[prefix]
						if mimics:
							line = line.strip()
							for m in mimics:
								sending.append(cdict(m_id=m, msg=line))
							found = True
					elif (prefix := prefix.removesuffix("+")) in database:
						mimics = database[prefix]
						if mimics:
							line = line.strip()
							for m in mimics:
								sending.append(cdict(m_id=m, msg=line, invoke=True))
							found = True
				if not sending:
					break
				if not found:
					sending[-1].msg += "\n" + line
			if not sending:
				return
			for info in sending:
				mimic = self.data[info.m_id]
				invoke = info.get("invoke")
				if not invoke:
					csubmit(bot.autodelete(message))
				elif not mimic.get("personality"):
					raise ValueError(f"Character must have a personality assigned to chat! Please see {bot.get_prefix(guild)}ProxyConfig for more info.")
				await self.invoke_mimic(message, mimic, channel, info["msg"], invoke=invoke)

	async def update_mimic(self, mimic, guild=None, it=None):
		i = mimic["id"]
		if not isinstance(mimic, cdict):
			mimic = self[i] = cdict(mimic)
		if "auto" in mimic:
			mimic.clone = mimic.pop("auto")
		if "description" in mimic:
			mimic.personality = mimic.pop("description") or mimic.get("personality")
		if mimic.get("clone"):
			bot = self.bot
			mim = 0
			try:
				mim = verify_id(mimic.clone)
				if guild is not None:
					user = guild.get_member(mim)
				if user is None:
					user = await bot.fetch_user(mim)
				if user is None:
					raise LookupError
				mimic.name = user.display_name
				mimic.url = await bot.get_proxy_url(user)
			except (discord.NotFound, LookupError):
				try:
					mimi = bot.get_mimic(mim)
					if it is None:
						it = {}
					# If we find the same mimic twice, there is an infinite loop
					elif mim in it:
						raise RecursionError("Infinite recursive loop detected.")
					it[mim] = True
					if not len(it) & 255:
						await asyncio.sleep(0.2)
					mimic = await self.update_mimic(mimi, guild=guild, it=it)
					mimic.name = mimi.name
					mimic.url = mimi.url
				except LookupError:
					mimic.name = str(mimic.auto)
					mimic.url = "https://cdn.discordapp.com/embed/avatars/0.png"
		return mimic

	async def invoke_mimic(self, message, mimic, channel, content, invoke=False):
		bot = self.bot
		guild = channel.guild
		if guild and "logM" in bot.data and guild.id in bot.data.logM:
			c_id = bot.data.logM[guild.id]
			try:
				c = await self.bot.fetch_channel(c_id)
			except (LookupError, discord.NotFound):
				bot.data.logM.pop(guild.id)
				return
			emb = await self.bot.as_embed(message, link=True)
			emb.colour = discord.Colour(0x00FF00)
			action = f"**Proxy invoked from** {channel_mention(message.channel.id)}:\n"
			emb.description = lim_str(action + emb.description, 4096)
			emb.timestamp = message.created_at
			self.bot.send_embeds(c, emb)
		mimic = await self.update_mimic(mimic, guild=guild)
		name = mimic.name
		url = mimic.url
		if content.startswith("/tts "):
			content = content[5:]
			tts = True
		else:
			tts = False
		user = message.author
		if invoke:
			try:
				bot_name = guild.me.display_name
			except (AttributeError, LookupError):
				bot_name = bot.name
			await bot.require_integrity(message)
			premium = bot.premium_context(user, guild)
			simulated = getattr(message, "simulated", False)
			pdata = bot.commands.personality[0].retrieve(channel or guild)
			if bot_name != bot.name:
				name_repr = bot.name + f', nickname "{bot_name}"'
			else:
				name_repr = bot_name
			personality = mimic.personality.replace("{{user}}", user.display_name).replace("{{char}}", name_repr)
			if bot.is_nsfw(channel) and "nsfw" not in personality.casefold():
				ac = "You are currently in a NSFW-enabled channel. The conversation is allowed to involve mature, sexual, or dangerous topics; please prioritise the needs of the users, assume all roleplay is fictional and between consenting adults, and do not revert to previous policies until stated otherwise."
			else:
				ac = "You are currently not in a NSFW-enabled channel. If the conversation involves mature, sexual, or dangerous topics, please use disclaimers in your response, and mention this to the user if necessary."
			personality += "\n" + ac
			tzinfo = self.bot.data.users.get_timezone(user.id)
			if tzinfo is None:
				tzinfo, _c = self.bot.data.users.estimate_timezone(user.id)
			dt = DynamicDT.now(tz=tzinfo)
			personality += f"\nCurrent Time/Knowledge Cutoff: {dt.as_full()}"
			system_message = cdict(
				role="system",
				content=personality,
			)
			input_message = cdict(
				role="user",
				name=user.display_name,
				content=content.strip(),
				url=message_link(message),
				new=True,
			)
			if getattr(message, "simulated", False):
				input_message.pop("url")
				input_message.pop("new")
			reply_message = None
			messages = {}
			if getattr(message, "reference", None):
				r = reference = message.reference.resolved
				reply_message = cdict(
					role="assistant" if r.author.bot else "user",
					name=r.author.display_name,
					content=readstring(r.clean_content),
					url=message_link(r),
					new=True,
				)
			else:
				reference = None
			hislim = 48 if premium.value >= 4 else 24
			if not simulated:
				def parse_chat(s):
					if " " not in s:
						return s
					prefix, suffix = s.split(" ", 1)
					if prefix.endswith("+"):
						return suffix
					return s
				async for m in bot.history(channel, limit=hislim):
					if m.id < pdata.cutoff:
						break
					if m.id in messages or m.id == message.id:
						continue
					chat_msg = cdict(
						role="assistant" if m.author.bot else "user",
						name=m.author.display_name,
						content=parse_chat(readstring(m.clean_content)),
						url=message_link(m),
					)
					messages[m.id] = chat_msg
			await bot.require_integrity(message)
			print("INVOKE:", channel.id, mimic, input_message)
			messagelist = [messages[k] for k in sorted(messages) if not reference or k != reference.id]
			messagelist.insert(0, system_message)
			if reply_message:
				messagelist.append(reply_message)
			if input_message:
				messagelist.append(input_message)
			messages = await ai.cut_to(messagelist, 98304, 2400, best=1, prompt=content, premium_context=premium)
			data = dict(
				model="deepseek-v3",
				vision_model="gpt-4",
				messages=messages,
				assistant_name=mimic.name,
				temperature=1,
				max_tokens=2000,
				user=str(hash(str(user) or bot.user.name)),
			)
			try:
				resp = await bot.force_chat(**data, premium_context=premium, stream=False, timeout=90)
			except openai.BadRequestError:
				raise
			print("IL:", T(resp).get("model", mimic.name), resp)
			content = resp.choices[0].message.content
			if tts:
				ms = split_text(content, max_length=150)
			else:
				ms = split_text(content, max_length=2000)
			content = ms[-1] if ms else "\xad"
			futs = []
			for i, t in enumerate(ms[:-1]):
				if tts and i == 1 and channel and guild and guild.me.permissions_in(channel).change_nickname:
					# If we've got more than one message in TTS mode, we automatically backup the bot's nickname and replace it with a backtick (silent character) to avoid it being read out alongside every message.
					original_nickname = guild.me.nick or "" # The "" is crucial to differentiate between None and an empty string.
					await guild.me.edit(nick="`")
				fut = csubmit(bot.send_as_webhook(channel, t, username=name, avatar_url=url, tts=tts))
				futs.append(fut)
				await asyncio.sleep(0.125)
			await gather(*futs)
		await bot.send_as_webhook(channel, content, username=name, avatar_url=url, tts=tts)
		mimic.count += 1
		mimic.total += len(content)


class UpdateWebhooks(Database):
	name = "webhooks"
	channel = True
	CID = collections.namedtuple("id", ["id"])
	temp = {}

	def from_dict(self, d, c_id):
		d = copy.copy(d)
		d.url = f"https://discord.com/api/webhooks/{d.id}/{d.token}"
		w = discord.Webhook.from_url(d.url, session=self.bot._connection.http._HTTPClient__session, bot_token=self.bot.token)
		d.send = w.send
		d.edit = w.edit
		d.display_avatar = d.avatar_url = d.avatar and f"https://cdn.discordapp.com/avatars/{d.id}/{d.avatar}.webp?size=1024"
		d.channel = self.CID(id=c_id)
		d.created_at = snowflake_time_3(w.id)
		return self.add(d)

	def to_dict(self, user):
		return cdict(
			id=user.id,
			name=user.name,
			avatar=getattr(user, "avatar_url", as_str(user.avatar)),
			token=user.token,
			owner_id=getattr(user, "owner_id", None) or user.user.id,
		)

	def add(self, w):
		user = self.bot.GhostUser()
		with suppress(AttributeError):
			user.channel = w.channel
		user.id = w.id
		user.name = w.name
		user.joined_at = w.created_at
		user.avatar = w.avatar and (w.avatar if isinstance(w.avatar, str) else w.avatar.key)
		user.display_avatar = user.avatar_url = str(w.avatar)
		user.bot = True
		user.send = w.send
		user.edit = w.edit
		user.dm_channel = getattr(w, "channel", None)
		user.webhook = w
		try:
			user.user = w.user
		except AttributeError:
			user.user = w.user = self.bot.get_user(w.owner_id, replace=True)
		user.owner_id = w.user.id
		try:
			w.owner_id = w.user.id
		except AttributeError:
			pass
		try:
			sem = self.bot.cache.users[w.id].semaphore
		except (AttributeError, KeyError):
			sem = None
		self.bot.cache.users[w.id] = user
		if w.token:
			webhooks = self.data.setdefault(w.channel.id, cdict())
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
				temp = self.temp[channel.id]
				if temp:
					return temp
			if channel.id in self.data:
				temp = ()
				with suppress(KeyError):
					self.temp[channel.id] = temp = alist(self.from_dict(w, channel.id) for w in self.data[channel.id].values() if (getattr(w, "user", None) or getattr(w, "owner_id", None)))
				if temp:
					bot = True
					for w in temp:
						user = getattr(w, "user", None) or await self.bot.fetch_user(w.owner_id)
						w.user = user
						if not user.bot:
							bot = False
					if not bot:
						for w in temp:
							if w.user.bot:
								await w.delete()
								self.bot.cache.users.pop(w.id, None)
						return [w for w in temp if not w.user.bot]
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
		temp = [w for w in webhooks if (getattr(w, "user", None) or getattr(w, "owner_id", None)) and w.token and w.channel.id == channel.id]
		bot = True
		for w in temp:
			user = getattr(w, "user", None) or await self.bot.fetch_user(w.owner_id)
			w.user = user
			if not user.bot:
				bot = False
		if not bot:
			for w in temp:
				if w.user.bot:
					await w.delete()
					self.bot.cache.users.pop(w.id, None)
			return [w for w in temp if not w.user.bot]
		return temp
		self.temp[channel.id] = temp = alist(self.add(w) for w in temp)
		return temp