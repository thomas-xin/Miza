# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


class Perms(Command):
	server_only = True
	name = ["DefaultPerms", "ChangePerms", "Perm", "ChangePerm", "Permissions"]
	description = "Shows or changes a user's permission level. Server Administrators bypass this."
	usage = "<0:user>* <1:new_level|default(-d)>?"
	example = ("perms steven 2", "perms 201548633244565504 ?d")
	flags = "fhd"
	rate_limit = (5, 8)
	multi = True
	slash = True
	ephemeral = True

	def perm_display(self, value):
		if isinstance(value, str):
			pass
		elif not value <= inf:
			return "nan (Bot Owner)"
		elif value >= inf:
			return "inf (Administrator)"
		elif value >= 3:
			return f"{value} (Ban Members or Manage Channels/Server)"
		elif value >= 2:
			return f"{value} (Manage Messages/Threads/Nicknames/Roles/Webhooks/Emojis/Events)"
		elif value >= 1:
			return f"{value} (View Audit Log/Server Insights or Move/Mute/Deafen Members or Mention Everyone)"
		elif value >= 0:
			return f"{value} (Member)"
		return f"{value} (Guest)"

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
							msgs.append(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(self.perm_display(t_perm))} to the default value of {sqr_md(self.perm_display(c_perm))}."))
							continue
						reason = f"to change permissions for {name} in {guild} from {self.perm_display(t_perm)} to {self.perm_display(c_perm)}"
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
						msgs.append(css_md(f"Changed permissions for {sqr_md(name)} in {sqr_md(guild)} from {sqr_md(self.perm_display(t_perm))} to {sqr_md(self.perm_display(c_perm))}."))
						continue
					reason = f"to change permissions for {name} in {guild} from {self.perm_display(t_perm)} to {self.perm_display(c_perm)}"
					raise self.perm_error(perm, m_perm, reason)
				msgs.append(css_md(f"Current permissions for {sqr_md(t_user)} in {sqr_md(guild)}: {sqr_md(self.perm_display(t_perm))}."))
		finally:
			return "".join(msgs)


class Timeout(Command):
	name = ["Mute", "Silence"]
	min_level = 3
	description = "Times out one or more users for a certain amount of time."
	schema = cdict(
		users=cdict(
			type="user",
			description="User(s) to time out",
			example="668999031359537205",
			multiple=True,
			required=True,
		),
		duration=cdict(
			type="datetime",
			description="Time input to parse",
			example="35 mins and 6.25 secs before 3am next tuesday, EDT",
			default="now",
		),
		reason=cdict(
			type="string",
			description="Timeout reason (will show up in Audit Log)",
			example="Being naughty >:(",
			greedy=False,
		),
	)
	rate_limit = (5, 10)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _guild, users, duration, reason, **void):
		if duration < DynamicDT.now():
			duration = None
		futs = [csubmit(user.timeout(duration, reason=reason)) for user in users]
		for fut in futs:
			await fut
		userstr = ", ".join(map(str, users))
		resp = f"Successfully timed out [{userstr}] until {sqr_md(duration)}" if duration else f"Successfully removed all timeouts from [{userstr}]"
		if reason:
			resp += f", with reason {sqr_md(reason)}"
		else:
			resp += "."
		return cdict(
			content=resp,
			prefix="```css\n",
			suffix="```",
		)


class Purge(Command):
	time_consuming = True
	_timeout_ = 16
	name = ["🗑", "Del", "Delete", "Purge_Range"]
	min_level = 3
	description = "Deletes a number or range of messages from a certain user in current channel."
	schema = cdict(
		user=cdict(
			type="user",
			description="User to delete messages from",
			example="668999031359537205",
		),
		count=cdict(
			type="integer",
			description="Maximum amount of messages to delete. Defaults to infinite if range is specified, else 1. Does NOT include the message invoking the command.",
			example="123",
		),
		range=cdict(
			type="index",
			description="Range of message IDs to delete",
			example="1162630678890950746:1215499763089408030",
		),
	)
	no_cancel = True
	rate_limit = (7, 12)
	multi = True
	slash = True
	ephemeral = True

	async def __call__(self, bot, _message, _channel, _guild, user=None, count=None, range=None, **void):
		if not count and not range:
			raise ValueError("Please enter a valid amount of messages to delete.")
		if count is None:
			count = inf
		if not range:
			left = 0
			right = inf
		elif len(range) == 1:
			if not count:
				left = 0
				right = inf
				count = range[0]
			else:
				left = range[0] - 1
				right = range[0] + 1
		else:
			left = range[0] or 0
			right = range[1] or inf
			left, right = sorted((left, right))
			left -= 1
			right += 1
		deleting = alist()
		found = set()
		if left <= _message.id <= right and not getattr(_message, "simulated", None) and not getattr(_message, "slash", None):
			deleting.append(_message)
			found.add(_message.id)
			count += 1
		delcount = 0
		async def use_delete():
			nonlocal delcount, deleting
			try:
				if hasattr(_channel, "delete_messages") and _channel.permissions_for(_guild.me).manage_messages:
					dels = deleting[:100]
					dels = []
					t = utc()
					for i, m in enumerate(deleting):
						if t - discord.utils.snowflake_time(m.id).timestamp() > 14 * 86400 - 60:
							break
						m._state = bot._connection
						dels.append(m)
					if not dels:
						raise StopIteration
					await _channel.delete_messages(dels)
					for m in dels:
						bot.data.deleted.cache[m.id] = 2
					deleting = deleting[len(dels):]
					delcount += len(dels)
				else:
					m = deleting[0]
					m.channel = _channel
					await bot.silent_delete(m, keep_log=True, exc=True)
					deleting.popleft()
					delcount += 1
			except Exception:
				for _ in loop(min(5, len(deleting))):
					m = deleting.popleft()
					m.channel = _channel
					with tracebacksuppressor:
						await bot.silent_delete(m, keep_log=True, exc=True)
					delcount += 1
		async with bot.guild_semaphore:
			async for m in bot.history(_channel, limit=count, after=left, before=right, full=False):
				if len(found) >= count:
					break
				if user and m.author.id != user.id:
					continue
				if m.id in found:
					continue
				deleting.append(m)
				found.add(m.id)
				if len(deleting) >= 100:
					await use_delete()
			while deleting:
				await use_delete()
		s = italics(css_md(f"Deleted {sqr_md(delcount)} message{'s' if delcount != 1 else ''}!", force=True))
		return cdict(content=s, reacts="❎")


class Ban(Command):
	server_only = True
	_timeout_ = 16
	name = ["🔨", "Bans", "Unban"]
	min_level = 3
	min_display = "3+"
	description = "Bans a user for a certain amount of time, with an optional reason."
	usage = "<0:user>* <1:time>? <2:reason>?"
	example = ("ban @Miza 30m for being naughty",)
	flags = "fhz"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (9, 16)
	multi = True
	slash = True
	maintenance = True

	async def __call__(self, bot, argv, args, argl, message, channel, guild, flags, perm, user, name, **void):
		if not args and not argl:
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-admin-ban-"
				+ str(user.id) + "_0"
				+ "-\nLoading ban list...```*",
				buttons=buttons,
			)
			return
		ts = utc()
		banlist = bot.data.bans.get(guild.id, alist())
		if type(banlist) is not alist:
			banlist = bot.data.bans[guild.id] = alist(banlist)
		async with discord.context_managers.Typing(channel):
			bans, glob = await self.getBans(guild)
			users = await bot.find_users(argl, args, user, guild)
		if not users:
			raise LookupError(f"No results found for {argv}.")
		if len(users) > 1 and "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		if not args or name == "unban":
			for user in users:
				try:
					ban = bans[user.id]
				except LookupError:
					csubmit(channel.send(ini_md(f"{sqr_md(user)} is currently not banned from {sqr_md(guild)}. Specify a duration for temporary bans, or `inf` for permanent bans.")))
					continue
				if name == "unban":
					await guild.unban(user)
					try:
						ind = banlist.search(user.id, key=lambda b: b["u"])
					except LookupError:
						pass
					else:
						banlist.pops(ind)["u"]
						if 0 in ind:
							with suppress(ValueError):
								bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
							if banlist:
								bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
					csubmit(channel.send(css_md(f"Successfully unbanned {sqr_md(user)} from {sqr_md(guild)}.")))
					continue
				csubmit(channel.send(italics(ini_md(f"Current ban for {sqr_md(user)} from {sqr_md(guild)}: {sqr_md(time_until(ban['t']))}."))))
			return
		# This parser is a mess too
		bantype = " ".join(args)
		if bantype.startswith("for "):
			bantype = bantype[4:]
		if "for " in bantype:
			i = bantype.index("for ")
			expr = bantype[:i].strip()
			msg = bantype[i + 4:].strip()
		if "with reason " in bantype:
			i = bantype.index("with reason ")
			expr = bantype[:i].strip()
			msg = bantype[i + 12:].strip()
		elif "reason " in bantype:
			i = bantype.index("reason ")
			expr = bantype[:i].strip()
			msg = bantype[i + 7:].strip()
		elif '"' in argv and len(args) == 2:
			expr, msg = args
		else:
			expr = bantype
			msg = None
		msg = msg or None
		_op = None
		for op, at in bot.op.items():
			if expr.startswith(op):
				expr = expr[len(op):].strip()
				_op = at
		num = await bot.eval_time(expr, op=False)
		csubmit(message.add_reaction("❗"))
		for user in users:
			p = bot.get_perms(user, guild)
			if not p < 0 and not isfinite(p):
				ex = PermissionError(f"{user} has administrator permission level, and cannot be banned from this server.")
				bot.send_exception(channel, ex)
				continue
			elif not p + 1 <= perm and not isnan(perm):
				reason = "to ban " + str(user) + " from " + guild.name
				ex = self.perm_error(perm, p + 1, reason)
				bot.send_exception(channel, ex)
				continue
			if _op is not None:
				try:
					ban = bans[user.id]
					orig = ban["t"] - ts
				except LookupError:
					orig = 0
				new = getattr(float(orig), _op)(num)
			else:
				new = num
			csubmit(self.createBan(guild, user, reason=msg, length=new, channel=channel, bans=bans, glob=glob))

	async def getBans(self, guild):
		loc = self.bot.data.bans.get(guild.id)
		# This API call could potentially be replaced with a single init call and a well maintained cache of banned users
		glob = []
		async for ban in guild.bans():
			glob.append(ban)
		bans = {ban.user.id: {"u": ban.user.id, "r": ban.reason, "t": inf} for ban in glob}
		if loc:
			for b in tuple(loc):
				if b["u"] not in bans:
					loc.pop(b["u"])
					continue
				bans[b["u"]]["t"] = b["t"]
				bans[b["u"]]["c"] = b["c"]
		return bans, glob

	async def createBan(self, guild, user, reason, length, channel, bans, glob):
		ts = utc()
		bot = self.bot
		banlist = set_dict(bot.data.bans, guild.id, alist())
		update = bot.data.bans.update
		for b in glob:
			u = b.user
			if user.id == u.id:
				with bot.ExceptionSender(channel):
					ban = bans[u.id]
					# Remove from global schedule, then sort and re-add
					with suppress(ValueError):
						banlist.remove(user.id, key=lambda x: x["u"])
					with suppress(ValueError):
						bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
					if length < inf:
						banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": ban.get("r")}, key=lambda x: x["t"])
						bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
					print(banlist)
					print(bot.data.bans.listed)
					msg = css_md(f"Updated ban for {sqr_md(user)} from {sqr_md(time_until(ban['t']))} to {sqr_md(time_until(ts + length))}.")
					await channel.send(msg)
				return
		with bot.ExceptionSender(channel):
			await bot.verified_ban(user, guild, reason)
			with suppress(ValueError):
				banlist.remove(user.id, key=lambda x: x["u"])
			with suppress(ValueError):
				bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
			if length < inf:
				banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": reason}, key=lambda x: x["t"])
				bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
			print(banlist)
			print(bot.data.bans.listed)
			msg = css_md(f"{sqr_md(user)} has been banned from {sqr_md(guild)} for {sqr_md(time_until(ts + length))}. Reason: {sqr_md(reason)}")
			await channel.send(msg)

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		update = self.bot.data.bans.update
		ts = utc()
		banlist = bot.data.bans.get(guild.id, [])
		bans, glob = await self.getBans(guild)
		page = 25
		last = max(0, len(bans) - page)
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
			"callback-admin-ban-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not bans:
			content += f"Ban list for {str(guild).replace('`', '')} is currently empty.```*"
		else:
			content += f"{len(bans)} user(s) currently banned from {str(guild).replace('`', '')}:```*"
		emb = discord.Embed(colour=discord.Colour(1))
		emb.description = content
		emb.set_author(**get_author(guild))
		for i, ban in enumerate(sorted(bans.values(), key=lambda x: x["t"])[pos:pos + page]):
			with tracebacksuppressor:
				user = await bot.fetch_user(ban["u"])
				emb.add_field(
					name=f"{user} ({user.id})",
					value=f"Duration {italics(single_md(time_until(ban['t'])))}\nReason: {italics(single_md(escape_markdown(str(ban['r']))))}"
				)
		more = len(bans) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		csubmit(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class RoleSelect(Command):
	server_only = True
	name = ["ReactionRoles", "RoleButtons", "RoleSelection", "RoleSelector"]
	min_level = 3
	min_display = "3+"
	description = "Creates a message that allows users to self-assign roles from a specified list."
	schema = cdict(
		roles=cdict(
			type="role",
			description="Role choices",
			example="@Members",
			multiple=True,
		),
		limit=cdict(
			type="integer",
			description="Amount of roles each user can assign",
			example="2",
		),
		title=cdict(
			type="text",
			description="Title to display in embed",
			example="🌟 Pronoun Roles 🌟",
			default="🗂 Role Selection 🗂",
		),
		description=cdict(
			type="text",
			description="Description to display in embed",
			example="*The buttons are self-explanatory!*",
			default="Click a button to add or remove a role from yourself!",
		),
		custom=cdict(
			type="text",
			description="Newline-separated emoji-role pairs",
			example="- 🐱 @Cat Lovers",
		),
	)
	rate_limit = (9, 12)

	async def get_role_emoji(self, role):
		colour = role.colour.to_rgb()
		if "emojis" in self.bot.data:
			return await self.bot.data.emojis.get_colour(colour)
		return get_closest_heart(colour)

	async def __call__(self, bot, _guild, _user, _perm, roles, limit, title, description, custom, **void):
		rolelist = []
		if roles:
			for role in roles:
				e = await self.get_role_emoji(role)
				rolelist.append((e, role))
		if custom:
			role_re = regexp(r"(?:<@&[0-9]+>|[0-9]+)$")
			for line in custom.splitlines():
				if not line.strip():
					continue
				found = role_re.search(line)
				if found:
					r_id = verify_id(found.group())
					role = await bot.fetch_role(r_id)
					ems = find_emojis_ex(line[:found.start()], cast_urls=False)
					if not ems:
						e = await self.get_role_emoji(role)
					else:
						e = ems[0]
					rolelist.append((e, role))
					continue
				if "@" in line:
					em, r = line.split("@", 1)
					role = await str_lookup(
						_guild.roles,
						r.strip(),
						qkey=userQuery1,
						ikey=userIter1,
						fuzzy=2 / 3,
					)
					if em:
						ems = find_emojis_ex(em, cast_urls=False)
					else:
						ems = ()
					if not ems:
						e = await self.get_role_emoji(role)
					else:
						e = ems[0]
					rolelist.append((e, role))
					continue
				ems = find_emojis_ex(line, cast_urls=False)
				if not ems:
					role = await str_lookup(
						_guild.roles,
						line.strip(),
						qkey=userQuery1,
						ikey=userIter1,
						fuzzy=2 / 3,
					)
					e = await self.get_role_emoji(role)
					rolelist.append((e, role))
					continue
				e = ems[0]
				role = await str_lookup(
					_guild.roles,
					line.split(e, 1)[-1],
					qkey=userQuery1,
					ikey=userIter1,
					fuzzy=2 / 3,
				)
				rolelist.append((e, role))
		if not rolelist:
			raise ArgumentError("Please input one or more roles by name or ID.")
		for e, role in rolelist:
			if inf > _perm:
				memb = await self.bot.fetch_member_ex(_user.id, _guild)
				if memb is None:
					raise LookupError("Member data not found for this server.")
				if memb.top_role <= role:
					raise PermissionError("Target role is higher than your highest role.")
		buttons = [cdict(name=role.name, emoji=e, id="%04d" % (ihash(role.name) % 10000) + str(role.id)) for e, role in rolelist]
		colour = await self.bot.get_colour(_guild)
		emb = discord.Embed(colour=colour)
		emb.set_author(**get_author(_guild))
		emb.title = title
		if limit and limit < len(buttons):
			available = f"{limit}/{len(buttons)}"
		else:
			available = f"{len(buttons)}"
		emb.description = italics(f"{available} roles assignable") + "\n" + description
		return self.construct(_user.id, leb128(limit) + serialise_nums([role.id for _, role in rolelist]), embed=emb, buttons=buttons)

	def react_perms(self, perm):
		return True

	async def _callback_(self, bot, message, reaction, user, vals, **void):
		if not reaction:
			return
		reaction = as_str(reaction)
		if not reaction.isnumeric():
			return
		if "__" in vals:
			limit, vals = vals.split("__", 1)
			limit = int(limit)
		else:
			limit = inf
		role_ids = vals.split("_")
		h1, r_id = reaction[:4], reaction[4:]
		if r_id not in role_ids:
			raise LookupError(f"Role <@&{r_id}> is not self-assignable.")
		try:
			role = await bot.fetch_role(r_id)
		except Exception:
			raise LookupError(f"Role <@&{r_id}> no longer exists.")
		h2 = "%04d" % (ihash(role.name) % 10000)
		if h1 != h2:
			raise NameError(f"Role hash mismatch ({h1} != {h2}). Please check if the role name was modified!")
		role_ids = set(map(int, role_ids))
		if role in user.roles:
			removals = [role]
			if isfinite(limit):
				rids = {r.id for r in standard_roles(user)}
				rids.remove(role.id)
				has_after = rids.intersection(role_ids)
				while len(has_after) > limit:
					rid = choice(has_after)
					has_after.remove(rid)
					r = await bot.fetch_role(rid)
					removals.append(r)
			await user.remove_roles(*removals, reason="Role Select", atomic=False)
			text = user.mention + ": Successfully removed " + ", ".join(role.mention for role in removals)
			await interaction_response(bot, message, text, ephemeral=True)
		else:
			rolelist = set(standard_roles(user))
			rolelist.add(role)
			removals = []
			if isfinite(limit):
				rids = {r.id for r in standard_roles(user)}
				has_after = rids.intersection(role_ids)
				while len(has_after) + 1 > limit:
					rid = choice(has_after)
					has_after.remove(rid)
					r = await bot.fetch_role(rid)
					removals.append(r)
			text = ""
			if removals:
				rolelist.difference_update(removals)
				await user.edit(roles=rolelist, reason="Role Select")
				text = user.mention + ": Successfully removed " + ", ".join(role.mention for role in removals) + "\n"
			else:
				await user.add_roles(role, reason="Role Select")
			text += user.mention + ": Successfully added " + role.mention
			await interaction_response(bot, message, text, ephemeral=True)


class RoleGiver(Command):
	server_only = True
	name = ["Verifier"]
	min_level = 3
	min_display = "3+"
	description = "Adds an automated role giver to the current channel. Triggered by a keyword in messages, only applicable to users with permission level >= 0 and account age >= 7d. Searches for word if only word characters, any substring if non-word characters are included, or regex if trigger begins and ends with a slash (/)."
	usage = "<0:react_to>? <1:role>? <delete_messages(-x)>?"
	flags = "aedx"
	rate_limit = (9, 12)

	async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
		bot = self.bot
		data = bot.data.rolegivers
		if "d" in flags:
			if argv:
				react = args[0].casefold()
				assigned = data.get(channel.id, {})
				if react not in assigned:
					raise LookupError(f"Rolegiver {react} not currently assigned for #{channel}.")
				assigned.pop(react)
				return italics(css_md(f"Removed {sqr_md(react)} from the rolegiver list for {sqr_md(channel)}."))
			if channel.id in data:
				data.pop(channel.id)
			return italics(css_md(f"Removed all automated rolegivers from {sqr_md(channel)}."))
		assigned = set_dict(data, channel.id, {})
		if not argv:
			key = lambda alist: f"⟨{', '.join(str(r) for r in alist[0])}⟩, delete: {alist[1]}"
			if not assigned:
				return ini_md(f"No currently active rolegivers for {sqr_md(channel)}.")
			return f"Currently active rolegivers in {channel_mention(channel.id)}:\n{ini_md(iter2str(assigned, key=key))}"
		if sum(len(alist[0]) for alist in assigned) >= 8:
			raise OverflowError(f"Rolegiver list for #{channel} has reached the maximum of 8 items. Please remove an item to add another.")
		# Limit substring length to 256
		a = args[0].strip()
		if len(a) > 256:
			raise OverflowError(f"Search substring too long ({len(a)} > 256).")
		if not a:
			raise ValueError("Input string must not be empty.")
		if len(a) > 2 and a[0] == a[-1] == "/":
			re.compile(a[1:-1])
		else:
			a = full_prune(a)
		r = verify_id(unicode_prune(" ".join(args[1:])))
		if len(guild.roles) <= 1:
			guild.roles = await guild.fetch_roles()
			guild.roles.sort()
			guild._roles.update((r.id, r) for r in guild.roles)
		if type(r) is int:
			role = guild.get_role(i)
		else:
			role = await str_lookup(
				standard_roles(guild),
				r,
				qkey=lambda x: [str(x), full_prune(x.replace(" ", ""))],
				fuzzy=0.125,
			)
		# Must ensure that the user does not assign roles higher than their own
		if inf > perm:
			memb = await self.bot.fetch_member_ex(user.id, guild)
			if memb is None:
				raise LookupError("Member data not found for this server.")
			if memb.top_role <= role:
				raise PermissionError("Target role is higher than your highest role.")
		alist = set_dict(assigned, a, [[], False])
		alist[1] |= "x" in flags
		alist[0].append(role.id)
		return italics(css_md(f"Added {sqr_md(a)} ➡️ {sqr_md(role)} to {sqr_md(channel.name)}."))


class AutoRole(Command):
	server_only = True
	name = ["InstaRole"]
	min_level = 3
	min_display = "3+"
	_timeout_ = 12
	description = "Causes any new user joining the server to automatically gain the targeted role. Input multiple roles to create a randomized role giver."
	usage = "<role>? <update_all(-x)>? <disable(-d)>?"
	example = ("autorole welcome", 'autorole -x "lovely people"')
	flags = "aedx"
	rate_limit = (9, 12)
	slash = True
	ephemeral = True

	async def __call__(self, argv, args, name, user, channel, guild, perm, flags, **void):
		bot = self.bot
		data = bot.data.autoroles
		if "d" in flags:
			assigned = data.get(guild.id, None)
			if argv and assigned:
				i = await bot.eval_math(argv)
				roles = assigned.pop(i)
				removed = deque()
				for r in roles:
					try:
						role = await bot.fetch_role(r, guild)
					except:
						print_exc()
						continue
					removed.append(role)
				# Update all users by removing roles
				if "x" in flags:
					i = 1
					for member in guild.members:
						for role in removed:
							if role in member.roles:
								new = {role: True for role in member.roles}
								for role in tuple(new):
									if role in removed:
										new.pop(role)
								csubmit(member.edit(roles=list(new), reason="InstaRole"))
								break
						if not i % 5:
							await asyncio.sleep(5)
						i += 1
				rolestr = sqr_md(", ".join(str(role) for role in removed))
				return italics(css_md(f"Removed {rolestr} from the autorole list for {sqr_md(guild)}."))
			if guild.id in data:
				data.pop(guild.id)
			return italics(css_md(f"Removed all items from the autorole list for {sqr_md(guild)}."))
		assigned = set_dict(data, guild.id, alist())
		if not argv:
			rlist = alist()
			for roles in assigned:
				new = alist()
				for r in roles:
					role = await bot.fetch_role(r, guild)
					new.append(role)
				rlist.append(new)
			if not assigned:
				return ini_md(f"No currently active autoroles for {sqr_md(guild)}.")
			return f"Currently active autoroles for {bold(escape_markdown(guild.name))}:\n{ini_md(iter2str(rlist))}"
		if sum(len(alist) for alist in assigned) >= 12:
			raise OverflowError(f"Autorole list for #{channel} has reached the maximum of 12 items. Please remove an item to add another.")
		roles = alist()
		rolenames = (verify_id(i) for i in args)
		if len(guild.roles) <= 1:
			guild.roles = await guild.fetch_roles()
			guild.roles.sort()
		rolelist = standard_roles(guild)
		for r in rolenames:
			if type(r) is int:
				for i in rolelist:
					if i.id == r:
						role = i
						break
			else:
				role = await str_lookup(
					rolelist,
					r,
					qkey=lambda x: [str(x), full_prune(x.replace(" ", ""))],
					fuzzy=0.125,
				)
			# Must ensure that the user does not assign roles higher than their own
			if not inf > perm:
				memb = await self.bot.fetch_member_ex(user.id, guild)
				if memb is None:
					raise LookupError("Member data not found for this server.")
				if memb.top_role <= role:
					raise PermissionError("Target role is higher than your highest role.")
			roles.append(role)
		new = alist(frozenset(role.id for role in roles))
		if new not in assigned:
			assigned.append(new)
		# Update all users by adding roles
		if "x" in flags or name == "instarole":
			if roles:
				async with discord.context_managers.Typing(channel):
					i = 1
					for member in guild.members:
						role = roles.next()
						if role not in member.roles:
							csubmit(member.add_roles(role, reason="InstaRole", atomic=True))
							if not i % 5:
								await asyncio.sleep(5)
							i += 1
		rolestr = sqr_md(", ".join(str(role) for role in roles))
		return italics(css_md(f"Added {rolestr} to the autorole list for {sqr_md(guild)}."))


class RolePreserver(Command):
	server_only = True
	name = ["🕵️", "StickyRoles"]
	min_level = 3
	min_display = "3+"
	description = "Causes ⟨BOT⟩ to save roles for all users, and re-add them when they leave and rejoin."
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
	rate_limit = (9, 12)
	slash = True
	ephemeral = True

	def __call__(self, _guild, _name, mode, **void):
		bot = self.bot
		following = bot.data.rolepreservers
		# Empty dictionary is enough to represent an active role preserver here
		curr = following.get(_guild.id)
		match mode:
			case "disable":
				if _guild.id in following:
					following.pop(_guild.id)
				return italics(css_md(f"Disabled role preservation for {sqr_md(_guild)}."))
			case "enable":
				if _guild.id not in following:
					following[_guild.id] = {}
				return italics(css_md(f"Enabled role preservation for {sqr_md(_guild)}."))
			case _:
				if curr is None:
					return ini_md(f'Role preservation is currently disabled in {sqr_md(_guild)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')
				return ini_md(f"Role preservation is currently enabled in {sqr_md(_guild)}.")


class NickPreserver(Command):
	server_only = True
	name = ["StickyNicks", "NicknamePreserver"]
	min_level = 3
	min_display = "3+"
	description = "Causes ⟨BOT⟩ to save nicknames for all users, and re-add them when they leave and rejoin."
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
	rate_limit = (9, 12)

	def __call__(self, flags, guild, name, **void):
		bot = self.bot
		following = bot.data.nickpreservers
		# Empty dictionary is enough to represent an active nickname preserver here
		curr = following.get(_guild.id)
		match mode:
			case "disable":
				if _guild.id in following:
					following.pop(_guild.id)
				return italics(css_md(f"Disabled nickname preservation for {sqr_md(_guild)}."))
			case "enable":
				if _guild.id not in following:
					following[_guild.id] = {}
				return italics(css_md(f"Enabled nickname preservation for {sqr_md(_guild)}."))
			case _:
				if curr is None:
					return ini_md(f'Nickname preservation is currently disabled in {sqr_md(_guild)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')
				return ini_md(f"Nickname preservation is currently enabled in {sqr_md(_guild)}.")


class ThreadPreserver(Command):
	server_only = True
	name = ["KeepAlive", "ThreadBump", "AutoBump", "UnArchive"]
	min_level = 3
	min_display = "3+"
	description = 'Causes ⟨BOT⟩ to "bump" (revive) the current thread when auto-archived.'
	usage = "<mode(enable|disable)>?"
	example = ("keepalive enable", "threadpreserver disable")
	rate_limit = (9, 12)
	flags = "aed"

	async def __call__(self, bot, guild, channel, name, args, flags, **void):
		if args:
			thr = await bot.fetch_channel(verify_id(args[0]))
		else:
			thr = channel
		if not isinstance(thr, discord.Thread) and not hasattr(thr, "archived"):
			raise TypeError("This command can only be used in threads.")
		if "d" in flags:
			bot.data.threadpreservers.pop(thr.id, None)
			return italics(css_md(f"Disabled thread preservation for {sqr_md(thr)}."))
		elif "e" in flags or "a" in flags:
			bot.data.threadpreservers[thr.id] = True
			return italics(css_md(f"Enabled thread preservation for {sqr_md(thr)}."))
		curr = bot.data.threadpreservers.get(thr.id)
		if curr is None:
			return ini_md(f'Thread preservation is currently disabled for {sqr_md(thr)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
		return ini_md(f"Thread preservation is currently enabled for {sqr_md(thr)}.")


class Lockdown(Command):
	server_only = True
	_timeout_ = 16
	name = ["🔒", "☣️"]
	min_level = inf
	description = "Completely locks down the server by removing send message permissions for all users, revoking all invites, and archiving all threads."
	flags = "f"
	rate_limit = (30, 40)

	async def roleLock(self, role, channel):
		perm = role.permissions
		perm.administrator = False
		perm.send_messages = False
		with self.bot.ExceptionSender(channel):
			await role.edit(permissions=perm, reason="Server Lockdown.")

	async def invLock(self, inv, channel):
		with self.bot.ExceptionSender(channel):
			await inv.delete(reason="Server Lockdown.")

	async def threadLock(self, thread, channel):
		with self.bot.ExceptionSender(channel):
			await thread.edit(archived=True, locked=True)

	async def __call__(self, guild, channel, flags, **void):
		if "f" not in flags:
			raise InterruptedError(self.bot.dangerous_command)
		u_id = self.bot.id
		for role in guild.roles:
			if len(role.members) != 1 or role.members[-1].id not in (u_id, guild.owner_id):
				csubmit(self.roleLock(role, channel))
		for thread in guild.threads:
			csubmit(self.threadLock(thread, channel))
		invites = await guild.invites()
		for inv in invites:
			csubmit(self.invLock(inv, channel))
		return bold(css_md(sqr_md(uni_str("LOCKDOWN REQUESTED.")), force=True))


class Archive(Command):
	time_consuming = 1
	_timeout_ = 3600
	name = ["ArchiveServer", "DownloadServer"]
	min_level = 3
	description = "Archives all messages, attachments and users into a .zip folder. Defaults to entire server if no channel specified, requires server permission level 3 as well as a Lv2 or above ⟨BOT⟩ subscription to perform, and may take a significant amount of time."
	usage = "<server|channel>? <start-id>? <end-id>? <token>?"
	flags = "f"
	rate_limit = 172800

	async def __call__(self, bot, message, guild, user, channel, args, flags, **void):
		target = guild.id
		tarname = guild.name
		token = bot.token
		if args:
			if len(args) > 1:
				token = regexp(r"[\w.-]{22,}").fullmatch(args[-1])
				if token:
					token = args.pop(-1)
					if token.startswith("Bot "):
						resp = None
					else:
						headers = {"Authorization": "Bot " + token, "Content-Type": "application/json"}
						resp = await create_future(
							requests.get,
							"https://discord.com/api/v10/users/@me",
							headers=headers,
						)
					if not resp or resp.status_code == 401:
						headers = {"Authorization": token, "Content-Type": "application/json"}
						resp = await create_future(
							requests.get,
							"https://discord.com/api/v10/users/@me",
							headers=headers,
						)
					else:
						token = "Bot " + token
					resp.raise_for_status()
			oid = verify_id(args.pop(0))
			if oid in bot.cache.channels:
				c = await bot.fetch_channel(oid)
				tarname = c.name
				target = str(c.id) + ","
			elif oid in bot.cache.guilds:
				g = await bot.fetch_guild(oid)
				tarname = g.name
				target = str(g.id)
			else:
				headers = {"Authorization": token, "Content-Type": "application/json"}
				try:
					data = await create_future(
						Request,
						f"https://discord.com/api/v10/channels/{oid}",
						headers=headers,
						aio=True,
						json=True,
						bypass=False,
					)
					c = cdict(data)
					tarname = c.name
					target = str(c.id) + ","
				except ConnectionError:
					data = await create_future(
						Request,
						f"https://discord.com/api/v10/guilds/{oid}",
						headers=headers,
						aio=True,
						json=True,
						bypass=False,
					)
					target = cdict(data)
					tarname = g.name
					target = str(g.id)
		if bot.get_perms(user, guild) < 3:
			raise PermissionError("You must be in the target server and have a permission level of minimum 3.")
		if max(bot.is_trusted(guild), bot.premium_level(user) * 2) < 2:
			raise PermissionError(f"Sorry, unfortunately this feature is for premium users only. Please make sure you have a subscription level of minimum 1 from {bot.kofi_url}!")
		if "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: SERVER DOWNLOAD REQUESTED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		fn = temporary_file("zip")
		args = [
			sys.executable,
			"misc/server-dump.py",
			"~" + token,
			target,
			fn,
		]
		info = ini_md("Archive Started!")
		m = await send_with_reply(channel, message, info)
		async with discord.context_managers.Typing(channel):
			print(args)
			proc = psutil.Popen(args, stdout=subprocess.PIPE)
			t = utc()
			while proc.is_running():
				line = bytearray()
				while not line or line[-1] != 10:
					b = await asubmit(proc.stdout.read, 1)
					if not b:
						break
					line.append(b[0])
				line = line.decode("utf-8").strip()
				if utc() - t >= 2:
					t = utc()
					if ": " in line:
						p, q = line.split(": ", 1)
					else:
						p = "Progress"
						q = line
					info = ini_md(f"Archive {p}: {sqr_md(q)}")
					await bot.edit_message(m, content=info)
		if not fn:
			raise FileNotFoundError("The requested file was not found. If this issue persists, please report it in the support server.")
		if not tarname.endswith(".zip"):
			tarname += ".zip"
		await bot.send_with_file(channel, file=CompatFile(fn, filename=tarname), reference=message)
		info = ini_md("Archive Complete!")
		await bot.edit_message(m, content=info)


class UserLog(Command):
	server_only = True
	name = ["MemberLog"]
	min_level = 3
	description = "Causes ⟨BOT⟩ to log user and member events from the server, in the current channel."
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
	rate_limit = 1

	async def __call__(self, bot, _channel, _guild, _name, mode, **void):
		data = bot.data.logU
		if mode == "enable":
			data[_guild.id] = _channel.id
			return italics(css_md(f"Enabled user event logging in {sqr_md(_channel)} for {sqr_md(_guild)}."))
		elif mode == "disable":
			if _guild.id in data:
				data.pop(_guild.id)
			return italics(css_md(f"Disabled user event logging for {sqr_md(_guild)}."))
		if _guild.id in data:
			c_id = data[_guild.id]
			channel = await bot.fetch_channel(c_id)
			return ini_md(f"User event logging for {sqr_md(_guild)} is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'User event logging is currently disabled in {sqr_md(_guild)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')


class MessageLog(Command):
	server_only = True
	name = ["MemberLog"]
	min_level = 3
	description = "Causes ⟨BOT⟩ to log message and file events from the server, in the current channel."
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
	rate_limit = 1

	async def __call__(self, bot, _channel, _guild, _name, mode, **void):
		data = bot.data.logM
		if mode == "enable":
			data[_guild.id] = _channel.id
			return italics(css_md(f"Enabled message event logging in {sqr_md(_channel)} for {sqr_md(_guild)}."))
		elif mode == "disable":
			if _guild.id in data:
				data.pop(_guild.id)
			return italics(css_md(f"Disabled message event logging for {sqr_md(_guild)}."))
		if _guild.id in data:
			c_id = data[_guild.id]
			channel = await bot.fetch_channel(c_id)
			return ini_md(f"Message event logging for {sqr_md(_guild)} is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'Message event logging is currently disabled in {sqr_md(_guild)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')


class StarBoard(PaginationCommand):
	server_only = True
	min_level = 2
	description = "Causes ⟨BOT⟩ to repost popular messages with a certain number of a specified reaction anywhere from the server, into the current channel."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("view", "add", "remove"),
				accepts=dict(enable="add", disable="remove", create="add", delete="remove"),
			),
			description="Action to perform",
			example="enable",
		),
		emoji=cdict(
			type="emoji",
			description="Emoji to treat as starboard trigger.",
			example="🐱",
		),
		special=cdict(
			type="enum",
			validation=cdict(
				enum=("SPARKLES",),
				strict_case=True,
			),
			description="Overrides emoji; special categories of reactions",
			example="SPARKLES",
		),
		count=cdict(
			type="integer",
			description="Amount of reactions required to trigger. Counts both regular and super reactions",
			default=1,
			example="3",
		),
		channel=cdict(
			type="channel",
			description="Channel to add or remove from ignore list; will not forward messages from channels within this list",
			example="#Starboard",
		)
	)
	rate_limit = 1

	async def __call__(self, bot, _message, _channel, _guild, _user, mode, emoji, special, count, channel, **void):
		data = bot.data.starboards
		if mode in ("add", "remove"):
			selected = []
			if data.get(_guild.id):
				for k, t in data[_guild.id].items():
					if k and (not channel or t[1] == channel.id):
						selected.append(k)
			if not selected:
				d = dict(data[_guild.id])
				d.pop(None, None)
				if not d:
					data.pop(_guild.id, None)
				return ini_md(f"Starboard reposting is currently disabled in {sqr_md(channel or _channel)}.")
			emojis = []
			for e_data, (count, c_id, *extra) in zip(selected, map(data[_guild.id].get, selected)):
				emoji = await bot.resolve_emoji(e_data, guild=_guild)
				emojis.append(str(emoji))
			if len(selected) > 1:
				triggers = "triggers "
			else:
				triggers = "trigger "
			triggers += sqr_md(", ".join(emojis))
			if not channel:
				if mode == "remove":
					for k in selected:
						data[_guild.id].pop(k, None)
					d = dict(data[_guild.id])
					d.pop(None, None)
					if not d:
						data.pop(_guild.id, None)
					return italics(css_md(f"Disabled starboard {triggers} for {sqr_md(_guild)}."))
				for k in selected:
					data[_guild.id][k] = data[_guild.id][k][:2]
				return italics(css_md(f"No longer exluding channels for starboard {triggers}."))
			if channel.guild.id != _guild.id:
				raise IndexError("Channel is not part of this server.")
			for k in selected:
				count, c_id2, *extra = data[_guild.id][k]
				if not extra:
					extra = [set()]
				disabled = extra[0]
				c_id = channel.id
				if mode == "remove":
					disabled.add(c_id)
				else:
					disabled.discard(c_id)
				data[_guild.id][k] = (count, c_id2, disabled)
			now = "Now" if mode == "remove" else "No longer"
			return italics(css_md(f"{now} excluding {sqr_md(channel)} from starboard {triggers}."))
		if not emoji and not special:
			# Set callback message for scrollable list
			return await self.display(_user.id, 0, _guild.id)
		channel = channel or _channel
		emote = special if special else str(emoji)
		boards = data.setdefault(_guild.id, {})
		boards[emote] = (count, channel.id, set([channel.id]))
		return cdict(
			content=f"Successfully added starboard to {sqr_md(channel)}, with trigger {sqr_md(emote)}: {sqr_md(count)}.",
			prefix="```css\n",
			suffix="```",
		)

	def react_perms(self, perm: int):
		return False if perm < 2 else True

	async def display(self, uid, pos, gid, diridx=-1):
		bot = self.bot

		def key(curr, pos, page):
			def disp(t):
				disabled = ",".join(map(channel_mention, sorted(t[2])))
				return f"×{t[0]} -> {channel_mention(t[1])}, *excludes {disabled}*"
			return iter2str({k: curr[k] for k in tuple(filter(bool, curr))[pos:pos + page]}, key=disp)

		return await self.default_display("starboard trigger", uid, pos, bot.data.starboards.get(gid, {}), diridx, extra=leb128(gid), key=key)

	async def _callback_(self, _user, index, data, **void):
		pos, more = decode_leb128(data)
		gid, _ = decode_leb128(more)
		return await self.display(_user.id, pos, gid, index)


class Crosspost(Command):
	server_only = True
	name = ["Repost", "Subscribe"]
	min_level = 3
	description = "Causes ⟨BOT⟩ to automatically crosspost all messages from the target channel, into the current channel."
	usage = "<channel> <disable(-d)>?"
	example = ("crosspost 683634093464092672", "crosspost -d #general")
	flags = "aed"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (7, 11)

	async def __call__(self, bot, argv, flags, user, message, channel, guild, **void):
		data = bot.data.crossposts
		if "d" in flags:
			if argv:
				argv = verify_id(argv)
				target = await bot.fetch_channel(argv)
				if target.id in data:
					data[target.id].discard(channel.id)
				return italics(css_md(f"Disabled message crossposting from {sqr_md(target)} to {sqr_md(channel)}."))
			for c_id, v in data.items():
				try:
					v.remove(channel.id)
				except KeyError:
					pass
			return italics(css_md(f"Disabled all message crossposting for {sqr_md(channel)}."))
		if not argv:
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-admin-crosspost-"
				+ str(user.id) + "_0"
				+ "-\nLoading Crosspost database...```*",
				buttons=buttons,
			)
			return
		target = await bot.fetch_channel(argv)
		if not bot.is_owner(user) and (not target.guild.get_member(user.id) or not target.permissions_for(target.guild.me).read_messages or not target.permissions_for(target.guild.get_member(user.id)).read_messages):
			raise PermissionError("Cannot follow channels without read message permissions.")
		channels = data.setdefault(target.id, set())
		channels.add(channel.id)
		return ini_md(f"Now crossposting all messages from {sqr_md(target)} to {sqr_md(channel)}.")

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		data = bot.data.crossposts
		curr = {k: sqr_md(bot.get_channel(k)) for k in sorted(c_id for c_id, v in data.items() if message.channel.id in v)}
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
			"callback-admin-crosspost-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not curr:
			content += f"No currently assigned crosspost subscriptions for #{str(message.channel).replace('`', '')}.```*"
			msg = ""
		else:
			content += f"{len(curr)} crosspost subscription(s) currently assigned for #{str(message.channel).replace('`', '')}:```*"
			msg = ini_md(iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}))
		colour = await self.bot.get_colour(guild)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		)
		emb.set_author(**get_author(guild))
		more = len(curr) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		csubmit(bot.edit_message(message, content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Publish(Command):
	server_only = True
	name = ["News", "AutoPublish"]
	min_level = 3
	description = "Causes ⟨BOT⟩ to automatically publish all posted messages in the current channel."
	usage = "<mode(enable|disable)>? <force(-x)>?"
	example = ("publish enable", "news disable")
	flags = "aedx"
	rate_limit = (16, 24)

	async def __call__(self, bot, flags, message, channel, guild, name, **void):
		data = bot.data.publishers
		if "e" in flags or "a" in flags:
			if channel.type != discord.ChannelType.news:
				raise TypeError("This feature can only be used in announcement channels.")
			if not channel.permissions_for(guild.me).manage_messages:
				raise PermissionError("Manage messages permission required to publish messages in channel.")
			data[channel.id] = 0 if "x" in flags else message.id
			return italics(css_md(f"Enabled automatic message publishing in {sqr_md(channel)} for {sqr_md(guild)}."))
		elif "d" in flags:
			data.pop(channel.id, None)
			return italics(css_md(f"Disabled automatic message publishing for {sqr_md(guild)}."))
		if channel.id in data:
			return ini_md(f"Automatic message publishing is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'Automatic message publishing is currently disabled in {sqr_md(channel)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


# TODO: Stop being lazy and finish this damn command
# class Welcomer(Command):
#     server_only = True
#     name = ["Welcome", "JoinMessage"]
#     min_level = 2
#     description = "Sets up or modifies a welcome message generator for new users in the server."
#     usage = "<0:option(image_url)(image_position{centre})(shape[{square},circle,rounded_square,hexagon])> <-1:value> <disable(-d)>"
#     flags = "h"
#     rate_limit = (2, 4)

#     async def __call__(self, bot, args, message, channel, guild, flags, perm, name, **void):
#         pass


class Relay(Command):
	server_only = True
	name = ["Forward"]
	min_level = 4
	description = "Causes ⟨BOT⟩ to send the target user(s) a DM that enables them to communicate through the current channel."
	usage = "<user>* <disable(-d)>?"
	example = ("relay @Miza Sorry, you have been banned from this server. Reply to this message to appeal!", "relay -d 201548633244565504")
	flags = "aedf"
	rate_limit = (16, 24)

	async def __call__(self, bot, argv, args, argl, message, channel, guild, flags, user, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		users = await bot.find_users(argl, args, user, guild)
		if not users:
			raise LookupError(f"No results found for {argv}.")
		if len(users) > 1 and "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		msg = " ".join(args)
		if not msg:
			msg = "[SAMPLE MESSAGE]"
			msg = bold(ini_md(msg))
		csubmit(message.add_reaction("📧"))
		fut = csubmit(send_with_reply(channel, message, "*```\nLoading DM relay...```*"))
		colour = await bot.get_colour(user)
		emb = discord.Embed(colour=colour, description=msg)
		url = await bot.get_proxy_url(user)
		emb.set_author(name=f"{user} ({user.id})", icon_url=url)
		emb.timestamp = message.created_at
		# emb.set_footer(text=str(message.id))
		m = await fut
		futs = deque()
		for u in users:
			fut = csubmit(u.send(f"*```callback-admin-relay-{m.channel.id}_{m.id}-\nThis is a relayed message. Use Discord reply to send a response.```*", embed=emb))
			futs.append(fut)
		mids = []
		uids = []
		for fut, u in zip(futs, users):
			with bot.ExceptionSender(channel):
				mes = await fut
				mids.append(mes.id)
				uids.append(u.id)
		if not mids or not uids:
			return
		uidf = "x".join(map(str, uids))
		midf = "x".join(map(str, mids))
		unames = ", ".join(map(user_mention, uids))
		await bot.edit_message(m, content=f"*```callback-admin-relay-{uidf}_{midf}-\nMessage successfully forwarded.```*\n> Received by {unames}. Use Discord reply to send additional messages.", embed=emb)

	_callback_ = _react_callback_ = async_nop


class UpdateRelays(Database):
	name = "relays"

	async def _nocommand_(self, message, **void):
		bot = self.bot
		tup = None
		try:
			reference = await bot.fetch_reference(message)
		except (LookupError, discord.NotFound):
			pass
		else:
			if reference.author.id == bot.id and reference.content.startswith("*```callback-admin-relay-"):
				tup = reference.content.removeprefix("*```callback-admin-relay-").split("\n", 1)[0].rstrip("-").split("_")
		if not tup:
			return
		if len(tup) != 2:
			print(reference.content)
			raise ValueError(tup)
		csubmit(message.add_reaction("📧"))
		emb = await bot.as_embed(message)
		user = message.author
		channel = message.channel
		col = await bot.get_colour(user)
		emb.colour = discord.Colour(col)
		url = await bot.get_proxy_url(user)
		emb.set_author(name=f"{user} ({user.id})", icon_url=url)
		emb.timestamp = message.created_at
		# emb.set_footer(text=str(message.id))
		sidf, midf = tup
		sids, mids = sidf.split("x"), midf.split("x")
		futs = deque()
		for si, mi in zip(sids, mids):
			with bot.ExceptionSender(channel):
				sendable = await bot.fetch_messageable(si)
				m = await bot.fetch_message(mi, sendable)
				fut = csubmit(send_with_reply(sendable, m, f"*```callback-admin-relay-{message.channel.id}_{message.id}-\nThis is a relayed message. Use Discord reply to send a response.```*", embed=emb))
				futs.append(fut)
		msent = []
		for fut in futs:
			with bot.ExceptionSender(channel):
				m = await fut
				msent.append(m)
		print(msent)


class UpdateBans(Database):
	name = "bans"

	def __load__(self):
		d = self.data
		for i in tuple(d):
			try:
				assert d[i][0]["t"]
			except:
				print_exc()
				d.pop(i, None)
		gen = ((block[0]["t"], i) for i, block in d.items() if block and isinstance(block[0], dict) and block[0].get("t") is not None)
		self.listed = alist(sorted(gen, key=lambda x: x[0]))

	async def _call_(self):
		t = utc()
		while self.listed:
			p = self.listed[0]
			if t < p[0]:
				break
			self.listed.popleft()
			g_id = p[1]
			temp = self.data[g_id]
			if not temp:
				self.data.pop(g_id)
				continue
			x = temp[0]
			if t < x["t"]:
				self.listed.insort((x["t"], g_id), key=lambda x: x[0])
				print(self.listed)
				continue
			x = cdict(temp.pop(0))
			if not temp:
				self.data.pop(g_id)
			else:
				z = temp[0]["t"]
				self.listed.insort((z, g_id), key=lambda x: x[0])
			print(self.listed)
			with tracebacksuppressor:
				guild = await self.bot.fetch_guild(g_id)
				user = await self.bot.fetch_user(x.u)
				m = guild.me
				try:
					channel = await self.bot.fetch_channel(x.c)
					if not channel.permissions_for(m).send_messages:
						raise LookupError
				except (LookupError, discord.Forbidden, discord.NotFound):
					channel = self.bot.get_first_sendable(guild, m)
				try:
					await guild.unban(user, reason="Temporary ban expired.")
					text = italics(css_md(f"{sqr_md(user)} has been unbanned from {sqr_md(guild)}."))
				except:
					text = italics(css_md(f"Unable to unban {sqr_md(user)} from {sqr_md(guild)}."))
					print_exc()
				await channel.send(text)

	async def _join_(self, user, guild, **void):
		if guild.id in self.data:
			for x in self.data[guild.id]:
				if x["u"] == user.id:
					return await guild.ban(user, reason="Sticky ban")


# Triggers upon 3 channel deletions in 2 minutes or 6 bans in 10 seconds
class ServerProtector(Database):
	name = "prot"

	async def kickWarn(self, u_id, guild, owner, msg):
		user = await self.bot.fetch_user(u_id)
		try:
			await guild.kick(user, reason="Triggered automated server protection response for excessive " + msg + ".")
			await owner.send(
				f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
				+ f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
				+ "and has been removed from the server to prevent any potential further attacks."
			)
		except discord.Forbidden:
			await owner.send(
				f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
				+ f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
				+ "and were unable to be automatically removed from the server; please watch them carefully to prevent any potential further attacks."
			)

	async def targetWarn(self, u_id, guild, msg):
		print(f"Channel Deletion warning by {user_mention(u_id)} in {guild}.")
		user = self.bot.user
		owner = guild.owner
		if owner.id == user.id:
			owner = await self.bot.fetch_user(next(iter(self.bot.owners)))
		if u_id == guild.owner.id:
			if u_id == user.id:
				return
			user = guild.owner
			await owner.send(
				f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
				+ f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
				+ "If this was intentional, please ignore this message."
			)
		elif u_id == user.id:
			csubmit(guild.leave())
			await owner.send(
				f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
				+ f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
				+ "and will promptly leave the server to prevent any potential further attacks."
			)
		else:
			await self.kickWarn(u_id, guild, owner, msg)

	async def _channel_delete_(self, channel, guild, **void):
		if channel.id in self.bot.data.deleted.cache:
			return
		user = None
		if not isinstance(channel, discord.Thread) and channel.permissions_for(guild.me).view_audit_log:
			ts = utc()
			cnt = {}
			audits = guild.audit_logs(limit=5, action=discord.AuditLogAction.channel_delete)
			async for log in audits:
				if ts - utc_ts(log.created_at) < 120:
					add_dict(cnt, {log.user.id: 1})
					if user is None and log.target.id == channel.id:
						user = log.user
				else:
					break
			else:
				audits = guild.audit_logs(limit=5, action=discord.AuditLogAction.thread_delete)
				async for log in audits:
					if ts - utc_ts(log.created_at) < 120:
						add_dict(cnt, {log.user.id: 1})
						if user is None and log.target.id == channel.id:
							user = log.user
					else:
						break
			for u_id in cnt:
				if cnt[u_id] > 2:
					if self.bot.is_trusted(guild.id) or u_id == self.bot.user.id:
						csubmit(self.targetWarn(u_id, guild, f"channel deletions `({cnt[u_id]})`"))
		if guild.id in self.bot.data.logU:
			await self.bot.data.logU._channel_delete_2_(channel, guild, user)

	async def _ban_(self, user, guild, **void):
		if self.bot.recently_banned(user, guild):
			return
		if not self.bot.is_trusted(guild.id) or not guild.me.guild_permissions.view_audit_log:
			return
		audits = guild.audit_logs(limit=13, action=discord.AuditLogAction.ban)
		ts = utc()
		cnt = {}
		async for log in audits:
			if ts - utc_ts(log.created_at) < 10:
				add_dict(cnt, {log.user.id: 1})
			else:
				break
		for u_id in cnt:
			if cnt[u_id] > 5:
				csubmit(self.targetWarn(u_id, guild, f"banning `({cnt[u_id]})`"))

	async def scan(self, message, url, known=None, **void):
		self.data["scans"] = self.data.get("scans", 0) + 1
		if hasattr(url, "name") and os.path.exists(url.name):
			url = url.name
		elif hasattr(url, "read"):
			url = url.read()
		resp = known or await process_image("ectoplasm", "$", [url, b"", "-f", "webp"], cap="caption", priority=True, timeout=60)
		if not resp:
			return "", 0

		def analyse(resp):
			if not isinstance(resp, bytes):
				return 0
			if re.search(r'^{\s*"', as_str(resp[:64])):
				try:
					data = orjson.loads(resp)
				except Exception:
					print_exc()
				else:
					try:
						issuer = data["manifests"][data["active_manifest"]]["signature_info"]["issuer"]
					except KeyError:
						pass
					else:
						action = 1
						try:
							software_agent = data["manifests"][data["active_manifest"]]["assertions"][0]
							if all(action["action"] == "c2pa.edited" for action in software_agent["data"].get("actions", ())):
								action = 2
							while "softwareAgent" not in software_agent:
								curr = software_agent["data"]
								if isinstance(curr, list):
									software_agent = curr[0]
								else:
									software_agent = next(iter(curr.values()))
									if isinstance(software_agent, list):
										software_agent = software_agent[0]
							software_agent = software_agent["softwareAgent"]
						except LookupError:
							software_agent = None
						if issuer in ("Miza", "OpenAI", "StabilityAI") or software_agent in ("Adobe Firefly", "DALL·E", "Bing Image Creator"):
							return action
					if str(data.get("issuer_id") or data.get("copyright")) == str(self.bot.id):
						return 1 if data.get("type") != "AI_EDITED" else 2
			elif resp.startswith(b"{'prompt':"):
				return 1
			return 0

		is_ai = analyse(resp)
		print("META:", type(resp), is_ai, resp)
		if is_ai == 1:
			self.data["pos"] = self.data.get("pos", 0) + 1
			try:
				await self.bot.react_with(message, "ai_generated.gif")
			except Exception:
				print_exc()
				try:
					await message.reply("-# *This image was generated using AI.*")
				except Exception:
					await message.channel.send("-# " + user_mention(message.author) + ": " + "*This image was generated using AI.*")
			return resp, is_ai
		elif is_ai == 2:
			self.data["pos"] = self.data.get("pos", 0) + 1
			await self.bot.react_with(message, "ai_edited.png")
		return "", is_ai


class EnabledCommands(Command):
	server_only = True
	name = ["EC", "Enable", "Disable"]
	min_display = "0~3"
	description = "Shows, enables, or disables a command category in the current channel."
	cats = [c for c in sorted(standard_commands)]
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("view", "enable", "disable", "enable-all", "disable-all", "reset"),
			),
			description="Action to perform",
			example="enable",
		),
		category=cdict(
			type="enum",
			validation=cdict(
				enum=cats,
				accepts=dict(owner="owner", misc="misc", nsfw="nsfw", text="string", music="voice", audio="voice", all=None, default=None),
			),
			description="Target category",
			example="voice",
			multiple=True,
		),
		apply_all=cdict(
			type="bool",
			description="Whether to apply to server (will affect all non-assigned channels)",
			example="true",
			default=False,
		),
		target=cdict(
			type="mentionable",
			description="Target channel (specifying the server's ID will act the same way as --enable-all/--disable-all)",
			example="247184721262411776",
		),
	)
	rate_limit = (5, 8)
	slash = True
	ephemeral = True
	exact = False

	def __call__(self, bot, _channel, _user, _guild, _perm, mode, category, apply_all, target, **void):
		enabled = bot.data.enabled
		if category == [None]:
			category = None
		if target is None:
			if apply_all:
				target = _guild
			else:
				target = _channel
		elif hasattr(target, "avatar"):
			raise TypeError("Target must be a Discord channel or server.")
		if not category and not mode or mode == "view":
			temp = bot.get_enabled(target)
			if not temp:
				return ini_md(f"No currently enabled commands in {sqr_md(target)}.")
			return f"Currently enabled command categories in {auto_mention(target)}:\n{ini_md(iter2str(temp))}"
		req = 3
		_perm = bot.get_perms(_user, target)
		if _perm < req:
			reason = f"to change enabled command list for {channel_repr(target)}"
			raise self.perm_error(_perm, req, reason)
		categories = visible_commands if bot.is_nsfw(_channel) else standard_commands
		if not category:
			mode = "enable-all" if mode == "enable" else "disable-all" if mode == "disable" else mode
		if mode == "enable-all":
			categories = set(categories)
			if target.id in enabled:
				enabled[target.id] = categories.union(enabled[target.id])
			else:
				enabled[target.id] = categories
			return css_md(f"Enabled all standard command categories in {sqr_md(target)}.")
		if mode == "disable-all":
			if apply_all:
				for channel in _guild.channels:
					enabled.pop(channel.id, None)
			enabled[target.id] = set()
			return css_md(f"Disabled all commands in {sqr_md(target)}.")
		if mode == "reset":
			if apply_all:
				for channel in _guild.channels:
					enabled.pop(channel.id, None)
			enabled.pop(target.id, None)
			return css_md(f"Reset enabled status of all commands in {sqr_md(target)}.")
		args = [i.casefold() for i in category]
		for catg in args:
			if not catg in bot.categories:
				raise LookupError(f"Unknown command category {catg}.")
		curr = set(bot.get_enabled(target))
		for catg in args:
			if mode == "enable":
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
		if check == categories:
			enabled.pop(target.id)
		category = "category" if len(args) == 1 else "categories"
		action = "Enabled" if mode != "disable" else "Disabled"
		return css_md(f"{action} command {category} {sqr_md(', '.join(args))} in {sqr_md(target)}.")


class Prefix(Command):
	name = ["ChangePrefix"]
	min_display = "0~3"
	description = "Shows or changes the prefix for ⟨BOT⟩'s commands for this server."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "reset"),
				accepts=dict(default="reset"),
			),
			description="Action to perform",
		),
		prefix=cdict(
			type="word",
			validation="[0, 1024]",
			description="New prefix to assign"
		)
	)
	rate_limit = (5, 8)
	slash = True
	ephemeral = True
	exact = False

	def __call__(self, bot, _guild, _perm, mode, prefix, **void):
		pref = bot.data.prefixes
		req = 3
		match mode:
			case "reset":
				if _perm < req:
					reason = f"to change command prefix for {_guild}"
					raise self.perm_error(_perm, req, reason)
				if _guild.id in pref:
					pref.pop(_guild.id)
				return cdict(
					content=f"Successfully reset command prefix for {sqr_md(_guild)}.",
					prefix="```css\n",
					suffix="```",
				)
			case _ if prefix:
				if _perm < req:
					reason = f"to change command prefix for {_guild}"
					raise self.perm_error(_perm, req, reason)
				# Backslash is not allowed, it is used to escape commands normally
				if prefix.startswith("\\"):
					raise TypeError("Prefix must not begin with backslash.")
				pref[_guild.id] = prefix
				return cdict(
					content=f"Successfully changed command prefix for {sqr_md(_guild)} to {sqr_md(prefix)}.",
					prefix="```css\n",
					suffix="```",
				)
		return cdict(
			content=f"Current command prefix for {sqr_md(_guild)}: {sqr_md(bot.get_prefix(_guild))}.",
			prefix="```ini\n",
			suffix="```",
		)


class UpdateEnabled(Database):
	name = "enabled"


class UpdatePrefix(Database):
	name = "prefixes"


class CreateEmoji(Command):
	server_only = True
	name = ["EmojiCreate", "EmojiCopy", "CopyEmoji", "Emote", "Emoji"]
	min_level = 0
	description = "Creates a custom emoji from a URL or attached file."
	schema = cdict(
		name=cdict(
			type="word",
			description="The name of the emoji",
			example="Untitled",
		),
		url=cdict(
			type="visual",
			description="The image to use (will automatically be resized to <256kb if larger)",
			required=True,
		),
	)
	rate_limit = (8, 12)
	_timeout_ = 6
	typing = True
	slash = ("Emoji",)

	async def __call__(self, bot, _guild, _message, _perm, _name, name, url, **void):
		if _perm < 2:
			raise self.perm_error(_perm, 2, "for command " + _name)
		name = name or url2fn(url).rsplit(".", 1)[0]
		image = await bot.optimise_image(url, fsize=262144, csize=128, fmt="gif")
		emoji = await _guild.create_custom_emoji(image=image, name=name, reason="CreateEmoji command")
		# This reaction indicates the emoji was created successfully
		with suppress(discord.Forbidden):
			await _message.add_reaction(emoji)
		return cdict(
			content=f"Successfully created emoji {sqr_md(emoji)} for {sqr_md(_guild)}.",
			prefix="```css\n",
			suffix="```",
		)


class CreateSound(Command):
	server_only = True
	name = ["SoundCreate", "SoundBoard", "SFX"]
	min_level = 0
	description = "Creates a custom soundboard from a URL or attached file."
	schema = cdict(
		name=cdict(
			type="word",
			description="The name of the sound",
			example="Untitled",
		),
		emoji=cdict(
			type="emoji",
			description="The emoji icon to use",
		),
		url=cdict(
			type="audio",
			description="The audio to use (will automatically be cut to <10s if longer)",
			required=True,
		),
	)
	rate_limit = (8, 12)
	_timeout_ = 6
	typing = True
	slash = ("Soundboard",)

	async def __call__(self, bot, _guild, _perm, _name, name, emoji, url, **void):
		if _perm < 2:
			raise self.perm_error(_perm, 2, "for command " + _name)
		if emoji and emoji.isnumeric():
			emoji = await bot.fetch_emoji(emoji, _guild)
			assert emoji.guild.id == _guild.id, "Emoji must be from the current server."
		name = name or getattr(emoji, "name", None) or url2fn(url).rsplit(".", 1)[0]
		d = await asubmit(get_duration, url)
		i = ts_us()
		if d <= 5.5:
			fn = f"{TEMP_PATH}/{i}.ogg"
			args = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-vn", "-i", url, "-to", "5.1875", "-c:a", "libopus", "-b:a", "160k", fn]
			print(args)
			proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.DEVNULL)
			try:
				async with asyncio.timeout(3200):
					await proc.wait()
			except (T0, T1, T2):
				with tracebacksuppressor:
					force_kill(proc)
				raise
			with open(fn, "rb") as f:
				data = await asubmit(f.read)
		else:
			fn1 = f"{TEMP_PATH}/{i}~1.mp3"
			fn2 = f"{TEMP_PATH}/{i}~2.mp3"
			args1 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-vn", "-i", url, "-ar", "48000", "-to", "1.175", "-c:a", "libmp3lame", "-b:a", "144k", fn1]
			if d <= 11:
				args2 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-vn", "-i", url, "-i", "misc/10s-soundboard-template.ogg", "-filter_complex", "amix=inputs=2:duration=longest", "-ss", "1.175", "-to", "11", "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "144k", fn2]
			else:
				args2 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-vn", "-i", url, "-ss", "1.175", "-to", "11", "-c:a", "libmp3lame", "-b:a", "144k", fn2]
			print(args1)
			print(args2)
			proc1 = await asyncio.create_subprocess_exec(*args1, stdout=subprocess.DEVNULL)
			proc2 = await asyncio.create_subprocess_exec(*args2, stdout=subprocess.DEVNULL)
			try:
				async with asyncio.timeout(3200):
					await proc1.wait()
					await proc2.wait()
			except (T0, T1, T2):
				with tracebacksuppressor:
					force_kill(proc1)
				with tracebacksuppressor:
					force_kill(proc2)
				raise
			fn3 = f"{TEMP_PATH}/{i}~3.mp3"
			args = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-vn", "-i", fn1, "-c:a", "libmp3lame", "-b:a", "320k", fn3]
			print(args)
			proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.DEVNULL)
			try:
				async with asyncio.timeout(3200):
					await proc.wait()
			except (T0, T1, T2):
				with tracebacksuppressor:
					force_kill(proc)
				raise

			def write_to():
				with open(fn3, "rb") as f3:
					data = f3.read()
				with open(fn2, "rb") as f2:
					data += f2.read()
				with open(f"{TEMP_PATH}/{i}.mp3", "wb") as f:
					f.write(data)
				return data

			data = await asubmit(write_to)
		await _guild.create_soundboard_sound(name=name, emoji=emoji, sound=data)
		return cdict(
			content=f"Successfully created soundboard {sqr_md(name)} for {sqr_md(_guild)}.",
			prefix="```css\n",
			suffix="```",
		)


class CreateSticker(Command):
	server_only = True
	name = ["StickerCreate", "StickerCopy", "CopySticker", "Sticker"]
	min_level = 2
	description = "Creates a custom sticker from a URL or attached file."
	schema = cdict(
		name=cdict(
			type="word",
			description="The name of the sticker",
			example="Untitled",
		),
		emoji=cdict(
			type="emoji",
			description="The emoji icon to use",
		),
		url=cdict(
			type="visual",
			description="The image to use (will automatically be resized to <256kb if larger)",
			required=True,
		),
	)
	rate_limit = (8, 12)
	_timeout_ = 8
	typing = True
	slash = ("Sticker",)

	async def __call__(self, bot, _guild, name, emoji, url, **void):
		if not name:
			name = "sticker_" + str(len(guild.stickers))
		image = await bot.optimise_image(url, fsize=512000, csize=320, fmt="apng", duration=5, opt=False)
		if emoji and emoji.isnumeric():
			emoji = await bot.fetch_emoji(emoji, _guild)
			assert emoji.guild.id == _guild.id, "Emoji must be from the current server."
		if not emoji:
			emoji = await asubmit(colour_cache.obtain_heart, url)
		sticker = await _guild.create_sticker(name=name, emoji=emoji, description="", file=CompatFile(image))
		colour = await bot.get_colour(sticker.url)
		embed = discord.Embed(colour=colour)
		embed.set_image(url=sticker.url)
		return cdict(
			content=f"Successfully created sticker {sqr_md(sticker.name)} for {sqr_md(_guild)}.",
			embed=embed,
			prefix="```css\n",
			suffix="```",
		)


class ScanEmoji(Command):
	name = ["EmojiScan", "ScanEmojis"]
	min_level = 1
	description = "Scans all the emojis in the current server for potential issues."
	usage = "<count>?"
	example = ("scanemoji",)
	rate_limit = (24, 32)
	_timeout_ = 4
	typing = True

	ffprobe_start = (
		"ffprobe",
		"-v",
		"error",
		"-hide_banner",
		"-select_streams",
		"v:0",
		"-show_entries",
		"stream=width,height",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
	)

	async def __call__(self, bot, guild, channel, message, argv, **void):
		# fut = csubmit(send_with_reply(channel, message, "Emoji scan initiated. Delete the original message at any point in time to cancel."))
		p = bot.get_prefix(guild)
		if argv:
			count = await bot.eval_math(argv)
		else:
			count = inf
		found = 0
		async with discord.context_managers.Typing(channel):
			for emoji in sorted(guild.emojis, key=lambda e: e.id):
				url = str(emoji.url)
				resp = await asubmit(subprocess.run, self.ffprobe_start + (url,), stdout=subprocess.PIPE)
				width, height = map(int, resp.stdout.splitlines())
				if width < 128 or height < 128:
					found += 1
					w, h = width, height
					while w < 128 or h < 128:
						w, h = w << 1, h << 1
					colour = await asubmit(bot.get_colour, url)
					bot.send_as_embeds(
						channel,
						description=f"{emoji} is {width}×{height}, which is below the recommended discord emoji size, and may appear blurry when scaled by Discord. Scaling the image using {p}resize, with filters `nearest`, `scale2x` or `lanczos` is advised.",
						fields=(("Example", f"{p}resize {emoji} {w}×{h} scale2x"),),
						title=f"⚠ Issue {found} ⚠",
						colour=discord.Colour(colour),
					)
					if found >= count:
						break
		if not found:
			return css_md(f"No emoji issues found for {guild}.")


class UpdateUserLogs(Database):
	name = "logU"

	# Send a member update globally for all user updates
	async def _user_update_(self, before, after, **void):
		found = False
		for g_id in self.data:
			try:
				guild = self.bot.cache.guilds[g_id]
			except KeyError:
				continue
			if guild.get_member(after.id):
				found = True
				break
		if not found:
			return
		for g_id in self.data:
			guild = self.bot.cache.guilds.get(g_id)
			if guild:
				csubmit(self._member_update_(before, after, guild))

	async def _member_update_(self, before, after, guild=None):
		bot = self.bot
		if guild is None:
			guild = after.guild
		# Make sure user is in guild
		try:
			memb = guild.get_member(after.id)
			if memb is None:
				raise LookupError
		except (LookupError, AttributeError):
			return
		except:
			print_exc()
			return
		if guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		emb = discord.Embed()
		files = []
		emb.description = f"{user_mention(after.id)} has been updated:"
		colour = [0] * 3
		# Add fields for every update to the member data
		change = False
		if str(before) != str(after):
			emb.add_field(
				name="Username",
				value=escape_markdown(str(before)) + " ➡️ " + escape_markdown(str(after)),
			)
			change = True
			colour[0] += 255
		if hasattr(before, "guild") and hasattr(after, "guild"):
			if before.display_name != after.display_name:
				emb.add_field(
					name="Nickname",
					value=escape_markdown(before.display_name) + " ➡️ " + escape_markdown(after.display_name),
				)
				change = True
				colour[0] += 255
			if hash(tuple(r.id for r in before.roles)) != hash(tuple(r.id for r in after.roles)):
				sub = alist()
				add = alist()
				for r in before.roles:
					if r not in after.roles:
						sub.append(r)
				for r in after.roles:
					if r not in before.roles:
						add.append(r)
				rchange = ""
				if sub:
					rchange = "❌ " + escape_markdown(", ".join(role_mention(r.id) for r in sub))
				if add:
					rchange += (
						"\n" * bool(rchange) + "✅ " 
						+ escape_markdown(", ".join(role_mention(r.id) for r in add))
					)
				if rchange:
					emb.add_field(name="Roles", value=rchange)
					change = True
					colour[1] += 255
		bk, ak = before.avatar, after.avatar
		if hasattr(bk, "key"):
			bk = bk.key
		if hasattr(ak, "key"):
			ak = ak.key
		a_url = None
		b_url = best_url(before)
		if "exec" in bot.data:
			with tracebacksuppressor:
				bf = await bot.data.exec.uproxy(b_url, collapse=True, mode="download")
				if isinstance(bf, byte_like):
					fn = b_url.split("?", 1)[0].rsplit("/", 1)[-1]
					files.append(CompatFile(bf, filename=fn))
					b_url = "attachment://" + fn
				else:
					b_url = bf
		requires_edit = False
		if bk != ak:
			a_url = best_url(after)
			if "exec" in bot.data:
				with tracebacksuppressor:
					af = await bot.data.exec.uproxy(a_url, collapse=True, mode="download")
					if isinstance(af, byte_like):
						fn = a_url.split("?", 1)[0].rsplit("/", 1)[-1]
						files.append(CompatFile(af, filename=fn))
						a_url = "attachment://" + fn
					else:
						a_url = af
			emb.add_field(
				name="Avatar",
				value=f"[Before]({b_url}) ➡️ [After]({a_url})",
			)
			requires_edit = not is_url(b_url) or not is_url(a_url)
			emb.set_thumbnail(url=a_url)
			change = True
			colour[2] += 255
		if not change:
			return
		ua, ub = a_url, b_url
		emb.set_author(name=str(after), icon_url=b_url, url=b_url if is_url(b_url) else None)
		emb.colour = colour2raw(colour)
		message = await channel.send(embed=emb, files=files)
		if "exec" in bot.data:
			with tracebacksuppressor:
				ub2 = message.embeds[0].author.icon_url
				if is_discord_attachment(ub2):
					b_url = bot.data.exec.uregister(best_url(before), ub2, message.id)
				ua2 = message.embeds[0].thumbnail and message.embeds[0].thumbnail.url
				if is_discord_attachment(ua2):
					a_url = bot.data.exec.uregister(best_url(after), ua2, message.id)
				if not is_url(a_url) and is_url(ua2):
					a_url = ua2
				if requires_edit:
					emb._fields[-1]["value"] = f"[Before]({b_url}) ➡️ [After]({a_url})"
					emb.set_author(name=str(after), icon_url=ub, url=b_url)
					await message.edit(embed=emb)

	async def _channel_update_(self, before, after, guild, **void):
		if guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		emb = discord.Embed(colour=8323072)
		mlist = self.bot.data.channel_cache.get(after.id)
		count = f" ({len(mlist)}+)" if mlist else ""
		if before.name != after.name:
			emb.add_field(
				name="Name",
				value=escape_markdown(before.name) + " ➡️ " + escape_markdown(after.name),
			)
		elif before.position != after.position:
			emb.add_field(
				name="Position",
				value=str(before.position) + " ➡️ " + str(after.position),
			)
		emb.description = f"{channel_mention(after.id)}{count} has been updated:"	
		self.bot.send_embeds(channel, emb)

	async def _channel_delete_2_(self, ch, guild, user, **void):
		if guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		emb = discord.Embed(colour=8323072)
		mlist = self.bot.data.channel_cache.get(ch.id)
		count = f" ({len(mlist)}+)" if mlist else ""
		if user:
			emb.set_author(**get_author(user))
			emb.description = f"#{ch.name}{count}: {ch.id} was deleted by {user_mention(user.id)}."
		else:
			emb.description = f"#{ch.name}{count}: {ch.id} has been deleted."	
		self.bot.send_embeds(channel, emb)

	async def _guild_update_(self, before, after, **void):
		if after.id not in self.data:
			return
		c_id = self.data[after.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(after.id)
			return
		colour = await self.bot.get_colour(after)
		emb = discord.Embed(colour=colour)
		emb.description = f"{after} has been updated:"
		change = False
		if str(before) != str(after):
			emb.add_field(
				name="Name",
				value=escape_markdown(str(before)) + " ➡️ " + escape_markdown(str(after)),
			)
			change = True
		if before.icon != after.icon:
			b_url = best_url(before)
			a_url = best_url(after)
			if "exec" in self.bot.data:
				urls = ()
				with tracebacksuppressor:
					urls = await self.bot.data.exec.uproxy(b_url, a_url, collapse=False)
				for i, url in enumerate(urls):
					if url:
						if i:
							a_url = url
						else:
							b_url = url
			emb.add_field(
				name="Icon",
				value=f"[Before]({b_url}) ➡️ [After]({a_url})",
			)
			emb.set_thumbnail(url=a_url)
			change = True
		if before.owner_id != after.owner_id:
			emb.add_field(
				name="Owner",
				value=f"{user_mention(before.owner_id)} ➡️ {user_mention(after.owner_id)}",
			)
			change = True
		if not change:
			return
		b_url = await self.bot.get_proxy_url(before)
		a_url = await self.bot.get_proxy_url(after)
		emb.set_author(name=str(after), icon_url=a_url, url=a_url)
		self.bot.send_embeds(channel, emb)

	async def _join_(self, user, **void):
		guild = T(user).get("guild")
		if guild is None or guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		# Colour: White
		emb = discord.Embed(colour=16777214)
		emb.set_author(**get_author(user))
		emb.description = f"{user_mention(user.id)} has joined the server."
		age = DynamicDT.now() - user.created_at
		if age < 86400 * 7:
			emb.description += f"\n⚠️ Account is {age} old. ⚠️"
		self.bot.send_embeds(channel, emb)

	async def _leave_(self, user, **void):
		guild = T(user).get("guild")
		if guild is None or guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		await asyncio.sleep(1)
		deleted = None
		prune = None
		kick = None
		ban = None
		bot = self.bot
		if "users" in bot.data:
			try:
				stored = bot.data.users[user.id]["stored"]
			except LookupError:
				pass
			else:
				for c_id, m_id in tuple(stored.items()):
					try:
						c = bot.cache.channels[c_id]
					except KeyError:
						stored.pop(c_id)
						continue
					try:
						m = await c.fetch_message(m_id)
					except:
						print_exc()
						stored.pop(c_id, None)
						continue
					print("Remaining author:", m.author, m.author.id, guild)
					if m.author.id == bot.deleted_user:
						print(user, user.id, "deleted!!")
						bot.data.users[user.id]["deleted"] = True
					break
		# Colour: Black
		emb = discord.Embed(colour=1)
		emb.set_author(**get_author(user))
		if not bot.permissions_in(guild).view_audit_log:
			pass
		elif not bot.data.users.get(user.id, {}).get("deleted"):
			# Check audit log to find whether user left or was kicked/banned
			with tracebacksuppressor(StopIteration):
				ts = utc()
				futs = [csubmit(flatten(guild.audit_logs(limit=4, action=getattr(discord.AuditLogAction, action)))) for action in ("ban", "kick", "member_prune")]
				bans = kicks = prunes = ()
				with tracebacksuppressor:
					bans = await futs[0]
					kicks = await futs[1]
					prunes = await futs[2]
				for log in bans:
					if ts - utc_ts(log.created_at) < 3:
						if log.target.id == user.id:
							ban = cdict(id=log.user.id, reason=log.reason)
							raise StopIteration
				for log in kicks:
					if ts - utc_ts(log.created_at) < 3:
						if log.target.id == user.id:
							kick = cdict(id=log.user.id, reason=log.reason)
							raise StopIteration
				for log in prunes:
					if ts - utc_ts(log.created_at) < 3:
						try:
							reason = f"{log.extra.delete_member_days} days of inactivity"
						except AttributeError:
							reason = None
						prune = cdict(id=log.user.id, reason=reason)
						raise StopIteration
		else:
			deleted = True
		if deleted is not None:
			emb.description = f"{user_mention(user.id)} has been deleted."
		elif ban is not None:
			emb.description = f"{user_mention(user.id)} has been banned by {user_mention(ban.id)}."
			if ban.reason:
				emb.description += f"\nReason: *`{no_md(ban.reason)}`*"
		elif kick is not None:
			emb.description = f"{user_mention(user.id)} has been kicked by {user_mention(kick.id)}."
			if kick.reason:
				emb.description += f"\nReason: *`{no_md(kick.reason)}`*"
		elif prune is not None:
			emb.description = f"{user_mention(user.id)} has been pruned by {user_mention(prune.id)}."
			if prune.reason:
				emb.description += f"\nReason: *`{no_md(prune.reason)}`*"
		else:
			emb.description = f"{user_mention(user.id)} has left the server."
		rchange = escape_markdown(", ".join(role_mention(r.id) for r in standard_roles(user)))
		if rchange:
			emb.add_field(name="Roles", value=rchange)
		self.bot.send_embeds(channel, emb)


class UpdateMessageCache(Database):
	name = "message_cache"
	# no_file = True
	files = "saves/message_cache"
	raws = {}
	loaded = {}
	saving = {}
	save_sem = Semaphore(1, 512, 5, 30)
	search_sem = Semaphore(16, 4096, rate_limit=5)
	locks = {}

	def __load__(self, **void):
		self.data.encoder = [encrypt, decrypt]

	# def __init__(self, *args):
	#     super().__init__(*args)
	#     if not os.path.exists(self.files):
	#         os.mkdir(self.files)

	def get_fn(self, m_id):
		return  m_id // 10 ** 12

	def load_file(self, fn, raw=False):
		if not raw:
			with suppress(KeyError):
				return self.loaded[fn]
		try:
			data = self.raws[fn]
		except KeyError:
			data = self.get(fn, {})
			if type(data) is not dict:
				data = {as_str(m["id"]): m for m in data}
			self.raws[fn] = data
		if raw:
			return
		found = self.loaded.setdefault(fn, {})
		bot = self.bot
		i = 0
		for k, m in deque(data.items()):
			if "channel" in m:
				m["channel_id"] = m.pop("channel")
			try:
				message = bot.CachedMessage(m)
			except:
				print(m)
				print_exc()
			k = int(k)
			bot.cache.messages[k] = found[k] = message
			i += 1
			if not i & 2047:
				time.sleep(0.1)
		return found

	def load_message(self, m_id):
		m_id = int(m_id)
		fn = self.get_fn(m_id)
		if fn in self.saving:
			return self.saving[fn][m_id]
		if fn in self.loaded:
			return self.loaded[fn][m_id]
		lock = self.locks.get(fn)
		if lock is None:
			lock = self.locks[fn] = Semaphore(1, inf)
		with lock:
			found = self.load_file(fn)
		if not found:
			fn = self.get_fn(m_id // 10)
			if fn in self.saving:
				return self.saving[fn][m_id]
			if fn in self.loaded:
				return self.loaded[fn][m_id]
			with lock:
				found = self.load_file(fn)
			if not found:
				raise KeyError(m_id)
		return found[m_id]

	def save_message(self, message):
		fn = self.get_fn(message.id)
		saving = self.saving.setdefault(fn, {})
		saving[message.id] = message
		return message

	def saves(self, fn, messages):
		lock = self.locks.get(fn)
		if lock is None:
			lock = self.locks[fn] = Semaphore(1, inf)
		with lock:
			self.load_file(fn, raw=True)
		if fn in self.loaded:
			self.loaded[fn].update(messages)
		saved = self.raws.setdefault(fn, {})
		for m_id, message in messages.items():
			m = T(message).get("_data")
			if m:
				if "author" not in m:
					author = message.author
					m["author"] = dict(id=author.id, s=str(author), avatar=author.avatar if not author.avatar or isinstance(author.avatar, str) else author.avatar.key)
				if "channel_id" not in m:
					try:
						m["channel_id"] = message.channel.id
					except AttributeError:
						continue
			else:
				if message.channel is None:
					continue
				author = message.author
				m = dict(
					author=dict(id=author.id, s=str(author), avatar=author.avatar if not author.avatar or isinstance(author.avatar, str) else author.avatar.key),
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
			m["id"] = str(m_id)
			saved[m["id"]] = m
		self[fn] = saved
		return len(saved)

	async def _save_(self, **void):
		if self.save_sem.is_busy():
			return
		# print("MESSAGE DATABASE UPDATING...")
		async with self.save_sem:
			saving = deque(self.saving.items())
			self.saving.clear()
			i = 0
			for fn, messages in saving:
				await asubmit(self.saves, fn, messages)
				i += 1
				if not i & 15 or len(messages) > 65536:
					await asyncio.sleep(0.3)
			while len(self.loaded) > 64:
				with suppress(RuntimeError):
					self.loaded.pop(next(iter(self.loaded)))
				i += 1
				if not i % 24:
					await asyncio.sleep(0.2)
			if not self.save_sem.is_busy():
				while len(self.raws) > 64:
					with suppress(RuntimeError):
						self.raws.pop(next(iter(self.raws)))
					i += 1
					if not i % 24:
						await asyncio.sleep(0.2)
			if len(saving) >= 100:
				print(f"Message Database: {len(saving)} files updated.")
			deleted = 0
			limit = self.get_fn(time_snowflake(dtn() - datetime.timedelta(days=28)))
			for f in sorted(k for k in self.keys() if isinstance(k, int)):
				if f == -1:
					continue
				if f < limit:
					self.pop(f, None)
					deleted += 1
				else:
					break
			if deleted >= 3:
				print(f"Message Database: {deleted} files deleted.")
			if len(self) > 1:
				self.setmtime()
		# print("MESSAGE DATABASE COMPLETE.")

	def getmtime(self):
		try:
			return self["~~"]
		except FileNotFoundError:
			return utc() - 28 * 86400
	setmtime = lambda self: self.__setitem__("~~", utc())

	async def _minute_loop_(self):
		await self._save_()


class UpdateMessageLogs(Database):
	name = "logM"
	searched = False
	dc = {}

	async def __call__(self):
		for h in tuple(self.dc):
			if isinstance(h, datetime.datetime):
				x = h.timestamp()
			else:
				x = h
			if utc() - x > 3600:
				self.dc.pop(h)

	async def _bot_ready_(self, **void):
		if not self.bot.ready and not self.bot.maintenance and not self.searched and len(self.bot.cache.messages) <= 65536:
			self.searched = True
			t = None
			with tracebacksuppressor(FileNotFoundError):
				t = utc_ft(self.bot.data.message_cache.getmtime())
			if not t:
				t = utc_dt() - datetime.timedelta(days=7)
			csubmit(self.load_new_messages(t))

	async def save_channel(self, channel, t=None):
		i = T(channel).get("last_message_id")
		if i:
			if id2ts(i) < self.bot.data.message_cache.getmtime():
				return
			# async for m in self.bot.data.channel_cache.grab(channel, as_message=False):
			#     if m == i:
			#         return
			#     break
		async with self.bot.data.message_cache.search_sem:
			async for message in channel.history(limit=32768, after=t, oldest_first=False):
				self.bot.add_message(message, files=False, force=True)

	async def load_new_messages(self, t):
		while "channel_cache" not in self.bot.data:
			await asyncio.sleep(0.5)
		print(f"Probing new messages from {len(self.bot.guilds)} guild{'s' if len(self.bot.guilds) != 1 else ''}...")
		for guild in self.bot.guilds:
			with tracebacksuppressor:
				futs = deque()
				for channel in itertools.chain(guild.text_channels, guild.threads):
					try:
						perm = channel.permissions_for(guild.me).read_message_history
					except discord.errors.ClientException:
						pass
					except:
						print_exc()
						perm = True
					if perm:
						futs.append(csubmit(self.save_channel(channel, t)))
					if len(futs) >= 4:
						with tracebacksuppressor:
							await futs.popleft()
		for fut in futs:
			with tracebacksuppressor:
				await fut
		self.bot.data.message_cache.finished = True
		self.bot.data.message_cache.setmtime()
		print("Loading new messages completed.")

	# Edit events are rather straightforward to log
	async def _edit_(self, before, after, force=False, **void):
		if after.author.bot and not force:
			return
		guild = before.guild
		if not guild or guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		emb = await self.bot.as_embed(after, proxy_images=False)
		emb2 = await self.bot.as_embed(before, proxy_images=False, refresh=False)
		emb.colour = discord.Colour(0x0000FF)
		action = f"**Message edited in** {channel_mention(after.channel.id)}:\n[View Message]({after.jump_url})"
		emb.add_field(name="Before", value=lim_str(emb2.description, 1024))
		emb.add_field(name="After", value=lim_str(emb.description, 1024))
		emb.description = action
		emb.timestamp = before.edited_at or after.created_at
		self.bot.send_embeds(channel, emb)

	async def reattachments(self, channel, message):
		if not message.attachments:
			return
		msg = deque()
		files = []
		for a in message.attachments:
			try:
				fn = await attachment_cache.download(a.url, read=True)
				fil = CompatFile(fn, filename=a.filename.removeprefix("SPOILER_"))
				files.append(fil)
			except:
				msg.append(proxy_url(a))
		colour = await self.bot.get_colour(message.author)
		emb = discord.Embed(colour=colour)
		emb.description = f"File{'s' if len(files) + len(msg) != 1 else ''} deleted from {user_mention(message.author.id)}"
		msg = "\n".join(msg) if msg else None
		if len(files) == 1:
			m2 = await self.bot.send_with_file(channel, msg, embed=emb, file=files[0])
		else:
			m2 = await channel.send(msg, embed=emb, files=files)
		message.attachments = [cdict(name=a.filename, id=a.id, url=self.bot.preserve_as_long(channel.id, m2.id, a.id, fn=a.url)) for a in m2.attachments]

	# Delete events must attempt to find the user who deleted the message
	async def _raw_delete_(self, message, bulk=False, **void):
		cu_id = self.bot.id
		if bulk:
			return
		guild = message.guild
		if not guild or guild.id not in self.data:
			return
		if message.author.bot:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		now = utc()
		if not message.author.bot:
			await self.reattachments(channel, message)
		u = message.author
		name_id = str(u)
		url = await self.bot.get_proxy_url(u)
		action = discord.AuditLogAction.message_delete
		try:
			t = u
			init = user_mention(t.id)
			d_level = self.bot.is_deleted(message)
			if d_level > 1:
				if d_level > 2:
					return
				t = self.bot.user
				init = user_mention(t.id)
			else:
				# Attempt to find who deleted the message
				if not guild.get_member(cu_id).guild_permissions.view_audit_log:
					raise PermissionError
				al = await flatten(guild.audit_logs(
					limit=50,
					action=action,
				))
				al2 = self.bot.deletes.get(guild.id, [])
				amap = {a.id: a for a in al2}
				for e in reversed(al):
					try:
						if now - e.created_at.timestamp() > 3:
							if e.id not in amap and now - e.created_at.timestamp() > 3600 or amap.get(e.id) and T(amap.get(e.id).extra).get("count", 1) >= T(e.extra).get("count", 1):
								continue
						if e.target and e.target.id != message.author.id:
							continue
						init = user_mention(e.user.id)
					finally:
						amap[e.id] = e
				alid = sorted(amap.keys())
				al3 = deque(map(amap.__getitem__, alid), maxlen=256)
				self.bot.deletes[guild.id] = al3
		except (PermissionError, discord.Forbidden, discord.HTTPException):
			init = "[UNKNOWN USER]"
		emb = await self.bot.as_embed(message, link=True)
		emb.colour = discord.Colour(0xFF0000)
		action = f"{init} **deleted message from** {channel_mention(message.channel.id)}:\n"
		emb.description = lim_str(action + (emb.description or ""), 4096)
		self.bot.send_embeds(channel, emb)

	# Thanks to the embed sender feature, which allows this feature to send up to 10 logs in one message
	async def _bulk_delete_(self, messages, **void):
		cu = self.bot.user
		cu_id = cu.id
		guild = messages[0].guild
		if guild.id not in self.data:
			return
		messages = [m for m in messages if not m.author.bot]
		if not messages:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		message = messages[-1]
		now = utc()
		futs = [self.reattachments(channel, m) for m in messages if not m.author.bot]
		await gather(*futs)
		action = discord.AuditLogAction.message_bulk_delete
		try:
			init = "[UNKNOWN USER]"
			if self.bot.is_deleted(message):
				t = self.bot.user
				init = user_mention(t.id)
			else:
				# Attempt to find who deleted the messages
				if not guild.get_member(cu_id).guild_permissions.view_audit_log:
					raise PermissionError
				al = await flatten(guild.audit_logs(
					limit=50,
					action=action,
				))
				al2 = self.bot.bulk_deletes.get(guild.id, [])
				amap = {a.id: a for a in al2}
				for e in al:
					try:
						if now - e.created_at.timestamp() > 3:
							if e.id not in amap and now - e.created_at.timestamp() > 3600 or amap.get(e.id) and T(amap.get(e.id).extra).get("count", 1) >= T(e.extra).get("count", 1):
								continue
						if e.target and e.target.id != message.channel.id:
							continue
						init = user_mention(e.user.id)
						break
					finally:
						amap[e.id] = e
				alid = sorted(amap.keys())
				al3 = deque(map(amap.__getitem__, alid), maxlen=256)
				self.bot.bulk_deletes[guild.id] = al3
		except (PermissionError, discord.Forbidden, discord.HTTPException):
			init = "[UNKNOWN USER]"
		emb = discord.Embed(colour=0xFF00FF)
		emb.description = f"{init} **deleted {len(messages)} message{'s' if len(messages) != 1 else ''} from** {channel_mention(messages[-1].channel.id)}:\n"
		# for message in messages:
		#     nextline = f"\nhttps://discord.com/channels/{guild.id}/{message.channel.id}/{message.id}"
		#     if len(emb.description) + len(nextline) > 2048:
		#         break
		#     emb.description += nextline
		embs = deque((emb,))
		for message in messages:
			try:
				emb = await self.bot.as_embed(message, link=True)
			except Exception:
				print_exc()
				continue
			emb.colour = discord.Colour(0x7F007F)
			embs.append(emb)
		self.bot.send_embeds(channel, embs)


class UpdatePublishers(Database):
	name = "publishers"

	async def _nocommand_(self, message, **void):
		if message.channel.id not in self.data:
			return
		if message.flags.crossposted or message.flags.is_crossposted:
			return
		if message.reference:
			return
		if "\u2009\u2009" in message.author.name:
			return
		try:
			# if not message.channel.permissions_for(message.guild.me).manage_messages:
			#     raise PermissionError("Manage messages permission missing from channel.")
			await message.publish()
		except Exception as ex:
			if "invalid message type" in repr(ex).lower():
				return
			print_exc()
			self.bot.send_exception(message.channel, ex)


class UpdateCrossposts(Database):
	name = "crossposts"
	stack = {}
	sem = Semaphore(1, 0, rate_limit=1)

	@tracebacksuppressor
	async def __call__(self):
		if self.sem.is_busy():
			return
		if not self.stack:
			return
		async with self.sem:
			async with Delay(1):
				for c, s in self.stack.items():
					channel = self.bot.get_channel(c)
					for k, v in s.items():
						embs = deque()
						for emb in v:
							if len(embs) > 9 or len(emb) + sum(len(e) for e in embs) > 6000:
								csubmit(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
								embs.clear()
							embs.append(emb)
							reacts = None
						if embs:
							csubmit(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
				self.stack.clear()

	@tracebacksuppressor
	async def _send_(self, message, **void):
		if message.channel.id not in self.data:
			return
		# if message.flags.is_crossposted:
			# return
		if "\u2009\u2009" in message.author.name:
			return
		content = message.content or message.system_content
		embeds = deque()
		for emb in message.embeds:
			embed = discord.Embed(
				description=emb.description,
				colour=emb.colour,
			)
			if emb.title:
				embed.title = emb.title
			if emb.url:
				embed.url = emb.url
			if emb.author:
				author = emb.author
				embed.set_author(name=author.name, url=author.url, icon_url=author.icon_url)
			if emb.image:
				image = emb.image.url
				embed.set_image(url=image)
			if emb.thumbnail:
				thumbnail = emb.thumbnail.url
				embed.set_thumbnail(url=thumbnail)
			if emb.footer:
				footer = emb.footer
				footer = footer.to_dict() if T(footer).get("to_dict") else eval(repr(footer), dict(EmbedProxy=dict))
				footer.pop("proxy_icon_url", None)
				embed.set_footer(**footer)
			if emb.timestamp:
				embed.timestamp = emb.timestamp
			for f in T(emb).get("_fields", ()):
				if f:
					embed.add_field(name=f["name"], value=f["value"], inline=f.get("inline", True))
			embeds.append(embed)
		files = deque()
		for a in message.attachments:
			fn = await attachment_cache.download(a.url, read=True)
			files.append(CompatFile(fn, filename=a.filename.removeprefix("SPOILER_")))
		for c_id in tuple(self.data[message.channel.id]):
			try:
				channel = await self.bot.fetch_channel(c_id)
			except:
				print_exc()
				s = self.data[message.channel.id]
				s.discard(c_id)
				if not s:
					self.pop(message.channel.id)
				continue
			name = message.guild.name + "\u2009﹟" + str(message.channel)
			url = best_url(message.guild)
			csubmit(self.bot.send_as_webhook(channel, content, embeds=list(embeds), files=list(files), username=name, avatar_url=url))


class UpdateStarboards(Database):
	name = "starboards"
	sems = {}
	sparkle_ids = {}

	async def _ready_(self, bot, **void):
		if (fun := bot._globals.get("FUN")) is not None:
			sparkle_emojis = await gather(*(bot.data.emojis.grab(t[1] + ".gif") for t in fun.sparkle_values))
			self.sparkle_ids = {e.id: e.name for e in sparkle_emojis}
			print("Sparkle IDs:", self.sparkle_ids)

	async def _reaction_add_(self, message, react, **void):
		if not message.guild or message.guild.id not in self.data:
			return
		table = self.data[message.guild.id]
		if verify_id(react) in self.sparkle_ids:
			react = "SPARKLES"
		temp = table.get(react)
		if not temp:
			return
		e_id, count, *disabled = temp
		if disabled and message.channel.id in disabled[0]:
			return
		req = table[react][0]
		if not req < inf:
			return
		message = await self.bot.ensure_reactions(message)
		count = sum(r.count for r in message.reactions if str(r.emoji) == react) if react != "SPARKLES" else sum(r.count for r in message.reactions if getattr(r.emoji, "id", None) and r.emoji.id in self.sparkle_ids)
		sem = self.sems.setdefault(message.guild.id, Semaphore(1, inf))
		async with sem:
			if message.id in table.get(None, ()):
				channel = await self.bot.fetch_channel(table[react][1])
				try:
					m = await self.bot.fetch_message(table[None][message.id], channel)
					res = await self.bot.verify_integrity(m)
					if not res:
						table[None].pop(message.id)
				except:
					print_exc()
					table[None].pop(message.id)
			if message.id not in table.setdefault(None, {}):
				if count >= req:# and count < req * 2 + 2:
					embed = await self.bot.as_embed(message, link=True, colour=True)
					text, link = embed.description.rsplit("\n\n", 1)
					description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table or "SPARKLES" in table and getattr(r.emoji, "id", None) and r.emoji.id in self.sparkle_ids) + "   " + link
					embed.description = lim_str(description, 4096)
					try:
						channel = await self.bot.fetch_channel(table[react][1])
						m = await channel.send(embed=embed)
					except (discord.NotFound, discord.Forbidden):
						table.pop(react)
					else:
						table[None][message.id] = m.id
						with tracebacksuppressor(RuntimeError, KeyError):
							while len(table[None]) > 32768:
								table[None].pop(next(iter(table[None])))
			else:
				try:
					channel = await self.bot.fetch_channel(table[react][1])
					m = await self.bot.fetch_message(table[None][message.id], channel)
					embed = await self.bot.as_embed(message, link=True, colour=True)
					text, link = embed.description.rsplit("\n\n", 1)
					description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table or "SPARKLES" in table and getattr(r.emoji, "id", None) and r.emoji.id in self.sparkle_ids) + "   " + link
					embed.description = lim_str(description, 4096)
					await self.bot.edit_message(m, content=None, embed=embed)
				except (discord.NotFound, discord.Forbidden):
					table[None].pop(message.id, None)
				else:
					table[None][message.id] = m.id
					with tracebacksuppressor(RuntimeError, KeyError):
						while len(table[None]) > 32768:
							table[None].pop(next(iter(table[None])))

	async def _edit_(self, after, **void):
		message = after
		try:
			table = self.data[message.guild.id]
		except KeyError:
			return
		if message.id not in table.get(None, {}):
			return
		sem = self.sems.setdefault(message.guild.id, Semaphore(1, inf))
		async with sem:
			try:
				reacts = sorted(message.reactions, key=lambda r: -r.count)
				if not reacts:
					return
				react = str(reacts[0].emoji)
				if verify_id(react) in self.sparkle_ids:
					react = "SPARKLES"
				channel = await self.bot.fetch_channel(table[react][1])
				m = await self.bot.fetch_message(table[None][message.id], channel)
				embed = await self.bot.as_embed(message, link=True, colour=True)
				text, link = embed.description.rsplit("\n\n", 1)
				description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in reacts if str(r.emoji) in table or "SPARKLES" in table and getattr(r.emoji, "id", None) and r.emoji.id in self.sparkle_ids) + "   " + link
				embed.description = lim_str(description, 4096)
				await self.bot.edit_message(m, content=None, embed=embed)
			except (discord.NotFound, discord.Forbidden):
				table[None].pop(message.id, None)
			else:
				table[None][message.id] = m.id
				with tracebacksuppressor(RuntimeError, KeyError):
					while len(table[None]) > 16384:
						table[None].pop(next(iter(table[None])))


class UpdateRolegivers(Database):
	name = "rolegivers"

	async def _nocommand_(self, text, message, **void):
		if not message.guild:
			return
		user = message.author
		guild = message.guild
		bot = self.bot
		if bot.get_perms(user, message.guild) < 0 or utc() - user.created_at.timestamp() < 86400 * 7:
			return
		assigned = self.data.get(message.channel.id, ())
		for k in assigned:
			if len(k) > 3 and k[0] == k[-1] == "/":
				x = message.content
				rk = k[1:-1]
				if len(k) * len(x) > 4096:
					res = await process_image("lambda rk, x: bool(re.search(rk, x))", "$", [rk, x], timeout=4)
				else:
					res = bool(re.search(rk, x))
				if not res:
					continue
			elif not ((k in text) if is_alphanumeric(k) else (k in message.content.casefold())):
				continue
			al = assigned[k]
			for r in al[0]:
				try:
					role = await bot.fetch_role(r, guild)
					if role is None:
						raise LookupError
				except LookupError:
					al[0].remove(r)
					continue
				if role in user.roles:
					continue
				with bot.ExceptionSender(message.channel):
					await user.add_roles(
						role,
						reason=f'Keyword "{k}" found in message "{message.content}".',
						atomic=True,
					)
					print(f"RoleGiver: Granted {role} to {user} in {guild}.")
			if alist[1]:
				await bot.silent_delete(message)


class UpdateAutoRoles(Database):
	name = "autoroles"

	async def _join_(self, user, guild, **void):
		if guild.id not in self.data:
			return
		if not guild.me.guild_permissions.manage_roles:
			return
		# Do not apply autorole to users who have roles from role preservers
		with suppress(KeyError):
			return self.bot.data.rolepreservers[guild.id][user.id]
		roles = []
		assigned = self.data[guild.id]
		for rolelist in assigned:
			with tracebacksuppressor:
				r = choice(rolelist)
				role = await self.bot.fetch_role(r, guild)
				if role not in roles:
					roles.append(role)
		# Attempt to add all roles in one API call
		try:
			await user.add_roles(*roles, reason="AutoRole", atomic=False)
		except discord.Forbidden:
			print_exc()
			await user.add_roles(*roles, reason="AutoRole", atomic=True)
		print(f"AutoRole: Granted {roles} to {user} in {guild}.")


class UpdateRolePreservers(Database):
	name = "rolepreservers"

	async def _join_(self, user, guild, **void):
		try:
			assigned = self.data[guild.id][user.id]
		except KeyError:
			return
		if not guild.me.guild_permissions.manage_roles:
			return
		if guild.id in self.bot.data.mutes and user.id in (x["u"] for x in self.bot.data.mutes[guild.id]):
			return
		roles = deque()
		assigned = self.data[guild.id][user.id]
		for r_id in assigned:
			with tracebacksuppressor:
				role = await self.bot.fetch_role(r_id, guild)
				roles.append(role)
		roles = [role for role in roles if role < guild.me.top_role]
		# Attempt to add all roles in one API call
		try:
			nick = cdict(nick=self.bot.data.nickpreservers[guild.id][user.id])
		except KeyError:
			nick = {}
		if (not nick or nick == user.display_name) and (not roles or {r.id for r in roles} == {r.id for r in standard_roles(user)}):
			return
		granted = []
		try:
			await user.edit(roles=roles, reason="RolePreserver", **nick)
			granted.extend(roles)
		except discord.Forbidden:
			print_exc()
			if nick:
				csubmit(user.edit(nick=nick.nick, reason="NickPreserver"))
			try:
				await user.add_roles(*roles, reason="RolePreserver", atomic=False)
				granted.extend(roles)
			except discord.Forbidden:
				for role in roles:
					try:
						await user.add_roles(role, reason="RolePreserver", atomic=True)
					except Exception:
						print_exc()
						print(f"RolePreserver: Failed to grant role: {role.id}, {role}")
					granted.append(role)
		self.data[guild.id].pop(user.id, None)
		print(f"RolePreserver: Granted {granted} to {user} in {guild}.")

	async def _leave_(self, user, guild, **void):
		if guild.id not in self.data:
			return
		# roles[0] is always @everyone
		roles = standard_roles(user)
		if roles:
			assigned = [role.id for role in roles]
			print("_leave_", guild, user, assigned)
			self.data[guild.id][user.id] = assigned
		else:
			print("_leave_", guild, user, None)
			self.data[guild.id].pop(user.id, None)


class UpdateNickPreservers(Database):
	name = "nickpreservers"

	async def _join_(self, user, guild, **void):
		try:
			nick = self.data[guild.id][user.id]
		except KeyError:
			return
		if not guild.me.guild_permissions.manage_nicknames:
			return
		with suppress(KeyError):
			return self.bot.data.rolepreservers[guild.id][user.id]
		if not nick or nick == user.display_name:
			return
		await user.edit(nick=nick, reason="NickPreserver")
		self.data[guild.id].pop(user.id, None)
		print(f"NickPreserver: Granted {nick} to {user} in {guild}.")

	async def _leave_(self, user, guild, **void):
		if guild.id not in self.data:
			return
		if T(user).get("nick"):
			self.data[guild.id][user.id] = user.nick
		else:
			self.data[guild.id].pop(user.id, None)


class ThreadList(Command):
	name = ["ListThreads", "Threads", "ReviveThreads", "ReviveAll"]
	description = "Shows or revives all threads in the current server."
	flags = "r"
	time_consuming = True
	_timeout_ = 8
	rate_limit = (8, 30)

	async def __call__(self, bot, channel, flags, guild, user, name, perm, **void):
		revive = "r" in flags or "revive" in name
		if revive and perm < 3:
			raise self.perm_error(perm, 3, f"to revive all threads for {guild.name}")
		sem = Semaphore(3, 480, rate_limit=5)
		threads = guild.threads
		futs = deque()
		for c in sorted(guild.text_channels, key=lambda c: c.id):
			if c.type is not discord.ChannelType.text:
				continue
			for mode in ("public", "private"):
				async with sem:
					url = f"https://discord.com/api/{api}/channels/{c.id}/threads/archived/{mode}"
					futs.append(Request(
						url,
						method="GET",
						authorise=True,
						aio=True,
						json=True,
					))
		for fut in futs:
			await fut
		for data in itertools.chain.from_iterable(fut.result().get("threads", ()) for fut in futs):
			factory, ch_type = discord.client._threaded_channel_factory(data['type'])
			if factory is None:
				raise discord.InvalidData('Unknown channel type {type} for channel ID {id}.'.format_map(data))
			if ch_type in (discord.ChannelType.group, discord.ChannelType.private):
				c = factory(me=bot.user, data=data, state=bot._connection)
			else:
				guild_id = int(data["guild_id"])
				guild = bot.get_guild(guild_id) or discord.Object(id=guild_id)
				c = factory(guild=guild, state=bot._connection, data=data)
			threads.append(c)
		threads = {c.id: c for c in threads}
		guild._threads.update(threads)
		if threads:
			title = f"{len(threads)} threads found:"
		else:
			title = "No threads found."
		def chm(c):
			s = channel_mention(c.id)
			n = 9223372036854775807
			return f"[{s}](https://discord.com/channels/{c.guild.id}/{c.id}/{n})"
		description = "\n".join(chm(c) for c in threads.values()) or "\xad"
		bot.send_as_embeds(channel, author=get_author(user), title=title, description=description, thumbnail=best_url(guild))
		if revive:
			for thread in threads.values():
				if thread.permissions_for(guild.me).manage_channels:
					await thread.edit(archived=False, locked=False)
				else:
					m = await thread.send("\xad")
					csubmit(bot.silent_delete(m))


class UpdateThreadPreservers(Database):
	name = "threadpreservers"

	async def _ready_(self, **void):
		for k in self.keys():
			await self._thread_update_(k, k)

	async def _thread_update_(self, before, after):
		if type(after) is int:
			if after not in self.data:
				return
			try:
				after = await self.bot.fetch_channel(after)
			except:
				print_exc()
				self.pop(after)
				return
		if after.id not in self.data:
			return
		if after.archived:
			if after.permissions_for(after.guild.me).manage_channels:
				await after.edit(archived=False, locked=False)
			else:
				m = await after.send("\xad")
				await self.bot.silent_delete(m)

	async def _channel_delete_(self, channel, **void):
		if not isinstance(channel, discord.Thread):
			return
		with suppress():
			self.pop(channel.id)


class UpdatePerms(Database):
	name = "perms"
