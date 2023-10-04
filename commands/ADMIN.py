print = PRINT


class Purge(Command):
	time_consuming = True
	_timeout_ = 16
	name = ["🗑", "Del", "Delete", "Purge_Range"]
	min_level = 3
	description = "Deletes a number of messages from a certain user in current channel."
	usage = "<1:user>? <0:count(1)>? <ignore{?i}|range{?r}|hide{?h}>*"
	example = ("purge @Miza 3", "purge 50", "purge_range 1038565892185222287 1128125804136579235")
	flags = "fiaehr"
	rate_limit = (7, 12)
	multi = True
	slash = True

	async def __call__(self, bot, args, argl, user, message, channel, name, flags, perm, guild, **void):
		# print(self, bot, args, argl, user, channel, name, flags, perm, guild, void)
		end = None
		if args:
			count = await bot.eval_math(args.pop(-1))
			if args and "r" in flags or "range" in name:
				start = safe_eval(args.pop(-1))
				end = count
				if end < start:
					start, end = end, start
				start -= 1
				end += 1
		else:
			count = 1
		if not argl and not args:
			uset = universal_set if guild and not getattr(guild, "ghost", None) else None
		else:
			users = await bot.find_users(argl, args, user, guild)
			uset = {u.id for u in users}
		if end is None and count <= 0:
			raise ValueError("Please enter a valid amount of messages to delete.")
		if end is None:
			print(count)
		else:
			print(start, end)
		delD = deque()
		if end is None:
			dt = None
			after = utc_dt() - datetime.timedelta(days=14) if "i" not in flags else None
			found = False
			if dt is None or after is None or dt > after:
				async with bot.guild_semaphore:
					async for m in bot.history(channel, limit=None, before=dt, after=after, care=uset is not universal_set):
						found = True
						dt = m.id
						if uset is universal_set or uset is None and m.author.bot or uset and m.author.id in uset:
							delD.append(m)
							count -= 1
							if count <= 0:
								break
		else:
			async with bot.guild_semaphore:
				async for m in bot.history(channel, limit=None, before=end, after=start, care=uset is not universal_set):
					if uset is universal_set or uset is None and m.author.bot or uset and m.author.id in uset:
						delD.append(m)
		if len(delD) >= 64 and "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(delD))} MESSAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		# attempt to bulk delete up to 100 at a time, otherwise delete 1 at a time
		deleted = 0
		delM = alist(delD)
		while len(delM):
			try:
				if hasattr(channel, "delete_messages") and channel.permissions_for(channel.guild.me).manage_messages:
					dels = delM[:100]
					t = utc()
					if t - discord.utils.snowflake_time(dels[0].id).timestamp() > 14 * 86400:
						raise
					if t - discord.utils.snowflake_time(dels[-1].id).timestamp() > 14 * 86400:
						raise
					await channel.delete_messages(dels)
					deleted += len(dels)
					for _ in loop(len(dels)):
						delM.popleft()
				else:
					m = delM[0]
					m.channel = channel
					await bot.silent_delete(m, no_log=-1, exc=True)
					deleted += 1
					delM.popleft()
			except:
				for _ in loop(min(5, len(delM))):
					m = delM.popleft()
					m.channel = channel
					with tracebacksuppressor:
						await bot.silent_delete(m, no_log=-1, exc=True)
					deleted += 1
		if "h" in flags:
			return
		s = italics(css_md(f"Deleted {sqr_md(deleted)} message{'s' if deleted != 1 else ''}!", force=True))
		if getattr(message, "slash", None):
			return s
		create_task(send_with_react(
			channel,
			s,
			reacts="❎",
		))


class Mute(Command):
	server_only = True
	_timeout_ = 16
	name = ["🔇", "Revoke", "Silence", "UnMute", "Timeout", "Mutes"]
	min_level = 3
	min_display = "3+"
	description = "Mutes a user for a certain amount of time, with an optional reason."
	usage = "<0:user>* <1:time>? <2:reason>? <hide{?h}>?"
	example = ("mute @Miza 1h for being naughty",)
	flags = "fhz"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (9, 16)
	multi = True
	slash = True

	async def __call__(self, bot, argv, args, argl, message, channel, guild, flags, perm, user, name, **void):
		if not args and not argl:
			# Set callback message for scrollable list
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-admin-mute-"
				+ str(user.id) + "_0"
				+ "-\nLoading mute list...```*",
				buttons=buttons,
			)
			return
		update = bot.data.mutes.update
		ts = utc()
		dt = discord.utils.utcnow()
		omutes = bot.data.mutes.setdefault(guild.id, {})
		if not isinstance(omutes, dict):
			omutes = bot.data.mutes[guild.id] = {m["u"]: m["t"] for m in omutes}
		async with discord.context_managers.Typing(channel):
			mutes, glob = await self.getMutes(guild)
			users = await bot.find_users(argl, args, user, guild)
		if not users:
			raise LookupError("No results found.")
		if len(users) > 1 and "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		if not args or name == "unmute":
			out = deque()
			for user in users:
				try:
					mute = mutes[user.id]
				except LookupError:
					out.append(f"{sqr_md(user)} is currently not muted in {sqr_md(guild)}. Specify a duration for temporary mutes, or `inf` for permanent mutes.")
					continue
				if name == "unmute":
					omutes.pop(user.id, None)
					update(guild.id)
					await user.timeout(datetime.timedelta(0))
					with suppress(AttributeError):
						user.timed_out_until = None
					out.append(f"Successfully unmuted {sqr_md(user)} in {sqr_md(guild)}.")
					continue
				out.append(f"Current mute for {sqr_md(user)} in {sqr_md(guild)}: {sqr_md(time_until(mute))}.")
			if out:
				return italics(ini_md("\n".join(out)))
			return
		# This parser is a mess too
		mutetype = " ".join(args)
		if mutetype.startswith("for "):
			mutetype = mutetype[4:]
		if "for " in mutetype:
			i = mutetype.index("for ")
			expr = mutetype[:i].strip()
			msg = mutetype[i + 4:].strip()
		if "with reason " in mutetype:
			i = mutetype.index("with reason ")
			expr = mutetype[:i].strip()
			msg = mutetype[i + 12:].strip()
		elif "reason " in mutetype:
			i = mutetype.index("reason ")
			expr = mutetype[:i].strip()
			msg = mutetype[i + 7:].strip()
		elif '"' in argv and len(args) == 2:
			expr, msg = args
		else:
			expr = mutetype
			msg = None
		msg = msg or None
		_op = None
		for op, at in bot.op.items():
			if expr.startswith(op):
				expr = expr[len(op):].strip()
				_op = at
		num = await bot.eval_time(expr, op=False)
		create_task(message.add_reaction("❗"))
		out = deque()
		for user in users:
			p = bot.get_perms(user, guild)
			if not p < 0 and not is_finite(p):
				ex = PermissionError(f"{user} has administrator permission level, and cannot be muted in this server.")
				bot.send_exception(channel, ex)
				continue
			elif not p + 1 <= perm and not isnan(perm):
				reason = "to mute " + str(user) + " in " + guild.name
				ex = self.perm_error(perm, p + 1, reason)
				bot.send_exception(channel, ex)
				continue
			orig = 0
			try:
				mute = mutes[user.id]
				orig = mute - ts
			except LookupError:
				try:
					mute = user.timed_out_until.timestamp()
					orig = mute - ts
				except AttributeError:
					mute = 0
			orig = max(0, orig)
			if _op is not None:
				new = getattr(float(orig), _op)(num)
			else:
				new = num
			new_ts = ts + new
			if new <= 21 * 86400:
				omutes.pop(user.id, None)
				update(guild.id)
				await user.timeout(datetime.timedelta(seconds=new), reason=msg)
			else:
				omutes[user.id] = new_ts
				update(guild.id)
				await user.timeout(datetime.timedelta(days=21), reason=msg)
			if orig:
				orig_ts = ts + orig
				out.append(f"Updated mute for {sqr_md(user)} in {sqr_md(guild)} from {sqr_md(time_until(orig_ts))} to {sqr_md(time_until(new_ts))}.")
			else:
				out.append(f"{sqr_md(user)} has been muted in {sqr_md(guild)} for {sqr_md(time_until(new_ts))}. Reason: {sqr_md(msg)}")
		if out:
			return italics(ini_md("\n".join(out)))

	async def getMutes(self, guild):
		ts = utc()
		mutes = {user.id: user.timed_out_until.timestamp() for user in guild._members.values() if getattr(user, "timed_out_until", None)}
		mutes.update(self.bot.data.mutes.get(guild.id, {}))
		for k, v in tuple(mutes.items()):
			if v <= ts:
				mutes.pop(k, None)
		return mutes, list(mutes.keys())

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		update = self.bot.data.mutes.update
		ts = utc()
		mutes, glob = await self.getMutes(guild)
		page = 25
		last = max(0, len(mutes) - page)
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
			"callback-admin-mute-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not mutes:
			content += f"Mute list for {str(guild).replace('`', '')} is currently empty.```*"
			msg = ""
		else:
			content += f"{len(mutes)} mute(s) currently assigned for {str(guild).replace('`', '')}:```*"
			msg = iter2str({user_mention(k): time_until(mutes[k]) for k in tuple(mutes)[pos:pos + page]}, left="", right="")
		emb = discord.Embed(colour=discord.Colour(1))
		emb.description = content + "\n" + msg
		emb.set_author(**get_author(guild))
		more = len(mutes) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Ban(Command):
	server_only = True
	_timeout_ = 16
	name = ["🔨", "Bans", "Unban"]
	min_level = 3
	min_display = "3+"
	description = "Bans a user for a certain amount of time, with an optional reason."
	usage = "<0:user>* <1:time>? <2:reason>? <hide{?h}>?"
	example = ("ban @Miza 30m for being naughty",)
	flags = "fhz"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (9, 16)
	multi = True
	slash = True

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
		update = self.bot.data.bans.update
		ts = utc()
		banlist = bot.data.bans.get(guild.id, alist())
		if type(banlist) is not alist:
			banlist = bot.data.bans[guild.id] = alist(banlist)
		async with discord.context_managers.Typing(channel):
			bans, glob = await self.getBans(guild)
			users = await bot.find_users(argl, args, user, guild)
		if not users:
			raise LookupError("No results found.")
		if len(users) > 1 and "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: {sqr_md(len(users))} USERS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		if not args or name == "unban":
			for user in users:
				try:
					ban = bans[user.id]
				except LookupError:
					create_task(channel.send(ini_md(f"{sqr_md(user)} is currently not banned from {sqr_md(guild)}. Specify a duration for temporary bans, or `inf` for permanent bans.")))
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
							with suppress(LookupError):
								bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
							if banlist:
								bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
						update(guild.id)
					create_task(channel.send(css_md(f"Successfully unbanned {sqr_md(user)} from {sqr_md(guild)}.")))
					continue
				create_task(channel.send(italics(ini_md(f"Current ban for {sqr_md(user)} from {sqr_md(guild)}: {sqr_md(time_until(ban['t']))}."))))
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
		create_task(message.add_reaction("❗"))
		for user in users:
			p = bot.get_perms(user, guild)
			if not p < 0 and not is_finite(p):
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
			create_task(self.createBan(guild, user, reason=msg, length=new, channel=channel, bans=bans, glob=glob))

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
					with suppress(LookupError):
						banlist.remove(user.id, key=lambda x: x["u"])
					with suppress(LookupError):
						bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
					if length < inf:
						banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": ban.get("r")}, key=lambda x: x["t"])
						bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
					print(banlist)
					print(bot.data.bans.listed)
					update(guild.id)
					msg = css_md(f"Updated ban for {sqr_md(user)} from {sqr_md(time_until(ban['t']))} to {sqr_md(time_until(ts + length))}.")
					await channel.send(msg)
				return
		with bot.ExceptionSender(channel):
			await bot.verified_ban(user, guild, reason)
			with suppress(LookupError):
				banlist.remove(user.id, key=lambda x: x["u"])
			with suppress(LookupError):
				bot.data.bans.listed.remove(guild.id, key=lambda x: x[-1])
			if length < inf:
				banlist.insort({"u": user.id, "t": ts + length, "c": channel.id, "r": reason}, key=lambda x: x["t"])
				bot.data.bans.listed.insort((banlist[0]["t"], guild.id), key=lambda x: x[0])
			print(banlist)
			print(bot.data.bans.listed)
			update(guild.id)
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
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class RoleSelect(Command):
	server_only = True
	name = ["ReactionRoles", "RoleButtons", "RoleSelection", "RoleSelector"]
	min_level = 3
	min_display = "3+"
	description = "Creates a message that allows users to self-assign roles from a specified list."
	usage = "<roles>+"
	example = ("roleselect A B C", 'reactionroles "role 1", "role 2", "role 3"')
	flags = "ae"
	no_parse = True
	rate_limit = (9, 12)

	async def __call__(self, args, message, guild, user, perm, **void):
		if not args:
			raise ArgumentError("Please input one or more roles by name or ID.")
		roles = deque()
		for arg in args:
			role = None
			r = verify_id(unicode_prune(arg))
			if len(guild.roles) <= 1:
				guild.roles = await guild.fetch_roles()
				guild.roles.sort()
				guild._roles.update((r.id, r) for r in guild.roles)
			if type(r) is int:
				role = guild.get_role(r)
			if not role:
				role = await str_lookup(
					guild.roles[1:],
					r,
					qkey=lambda x: [str(x), full_prune(str(x).replace(" ", ""))],
					fuzzy=0.125,
				)
			# Must ensure that the user does not assign roles higher than their own
			if inf > perm:
				memb = await self.bot.fetch_member_ex(user.id, guild)
				if memb is None:
					raise LookupError("Member data not found for this server.")
				if memb.top_role <= role:
					raise PermissionError("Target role is higher than your highest role.")
			roles.append(role)

		def get_ecolour(colour):
			h, s, l = rgb_to_hsl([c / 255 for c in colour])
			if l >= 0.875 or l >= 0.625 and s < 0.375 or l >= 0.5 and s < 0.875:
				return "⚪"
			if l <= 0.125 or l <= 0.375 and s < 0.375 or l <= 0.5 and s < 0.125:
				return "⚫"
			if h < 1 / 16 or h >= 41 / 48:
				return "🔴"
			if 1 / 16 <= h < 5 / 48:
				return "🟠"
			if 5 / 48 <= h < 1 / 4:
				return "🟡"
			if 1 / 4 <= h < 1 / 2:
				return "🟢"
			if 1 / 2 <= h < 35 / 48:
				return "🔵"
			if 35 / 48 <= h < 41 / 48:
				return "🟣"
			return "🟤"

		buttons = [cdict(name=role.name, emoji=get_ecolour(role.colour.to_rgb()), id="%04d" % (ihash(role.name) % 10000) + str(role.id)) for role in roles]
		colour = await self.bot.get_colour(self.bot.user)
		rolestr = "_".join(str(role.id) for role in roles)
		description = f"```callback-admin-roleselect-{rolestr}-\n{len(buttons)} roles available```Click a button to add or remove a role from yourself!"
		embed = discord.Embed(colour=colour, title="🗂 Role Selection 🗂", description=description)
		embed.set_author(**get_author(self.bot.user))
		await send_with_reply(None, message, embed=embed, buttons=buttons)

	async def _callback_(self, bot, message, reaction, user, vals, **void):
		if not reaction:
			return
		reaction = as_str(reaction)
		if not reaction.isnumeric():
			return
		roles = vals.split("_")
		h1, r_id = reaction[:4], reaction[4:]
		if r_id not in roles:
			raise LookupError(f"Role <@&{r_id}> no longer exists.")
		role = await bot.fetch_role(r_id)
		h2 = "%04d" % (ihash(role.name) % 10000)
		if h1 != h2:
			raise NameError(f"Incorrect hash: {h1} != {h2}")
		if role in user.roles:
			await user.remove_roles(role, reason="Role Select")
			await interaction_response(bot, message, user.mention + ": Successfully removed " + role.mention, ephemeral=True)
		else:
			await user.add_roles(role, reason="Role Select")
			await interaction_response(bot, message, user.mention + ": Successfully added " + role.mention, ephemeral=True)


class RoleGiver(Command):
	server_only = True
	name = ["Verifier"]
	min_level = 3
	min_display = "3+"
	description = "Adds an automated role giver to the current channel. Triggered by a keyword in messages, only applicable to users with permission level >= 0 and account age >= 7d."
	usage = "<0:react_to>? <1:role>? <delete_messages{?x}>? <disable{?d}>?"
	example = ("rolegiver lol lol_role", "rolegiver n*gger muted")
	flags = "aedx"
	no_parse = True
	rate_limit = (9, 12)

	async def __call__(self, argv, args, user, channel, guild, perm, flags, **void):
		update = self.bot.data.rolegivers.update
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
		react = args[0].casefold()
		if len(react) > 256:
			raise OverflowError(f"Search substring too long ({len(react)} > 256).")
		r = verify_id(unicode_prune(" ".join(args[1:])))
		if len(guild.roles) <= 1:
			guild.roles = await guild.fetch_roles()
			guild.roles.sort()
			guild._roles.update((r.id, r) for r in guild.roles)
		if type(r) is int:
			role = guild.get_role(i)
		else:
			role = await str_lookup(
				guild.roles[1:],
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
		alist = set_dict(assigned, react, [[], False])
		alist[1] |= "x" in flags
		alist[0].append(role.id) 
		update(channel.id)
		return italics(css_md(f"Added {sqr_md(react)} ➡️ {sqr_md(role)} to {sqr_md(channel.name)}."))


class AutoRole(Command):
	server_only = True
	name = ["InstaRole"]
	min_level = 3
	min_display = "3+"
	_timeout_ = 12
	description = "Causes any new user joining the server to automatically gain the targeted role. Input multiple roles to create a randomized role giver."
	usage = "<role>? <update_all{?x}>? <disable{?d}>?"
	example = ("autorole welcome", 'autorole -x "lovely people"')
	flags = "aedx"
	rate_limit = (9, 12)
	slash = True

	async def __call__(self, argv, args, name, user, channel, guild, perm, flags, **void):
		update = self.bot.data.autoroles.update
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
								create_task(member.edit(roles=list(new), reason="InstaRole"))
								break
						if not i % 5:
							await asyncio.sleep(5)
						i += 1
				update(guild.id)
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
		rolelist = guild.roles[1:]
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
			update(guild.id)
		# Update all users by adding roles
		if "x" in flags or name == "instarole":
			if roles:
				async with discord.context_managers.Typing(channel):
					i = 1
					for member in guild.members:
						role = roles.next()
						if role not in member.roles:
							create_task(member.add_roles(role, reason="InstaRole", atomic=True))
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
	description = "Causes ⟨MIZA⟩ to save roles for all users, and re-add them when they leave and rejoin."
	usage = "(enable|disable)?"
	example = ("rolepreserver enable", "stickyroles disable")
	flags = "aed"
	rate_limit = (9, 12)
	slash = True

	def __call__(self, flags, guild, name, **void):
		bot = self.bot
		following = bot.data.rolepreservers
		update = following.update
		# Empty dictionary is enough to represent an active role preserver here
		curr = following.get(guild.id)
		if "d" in flags:
			if guild.id in following:
				following.pop(guild.id)
			return italics(css_md(f"Disabled role preservation for {sqr_md(guild)}."))
		elif "e" in flags or "a" in flags:
			if guild.id not in following:
				following[guild.id] = {}
			return italics(css_md(f"Enabled role preservation for {sqr_md(guild)}."))
		if curr is None:
			return ini_md(f'Role preservation is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
		return ini_md(f"Role preservation is currently enabled in {sqr_md(guild)}.")


class NickPreserver(Command):
	server_only = True
	name = ["StickyNicks", "NicknamePreserver"]
	min_level = 3
	min_display = "3+"
	description = "Causes ⟨MIZA⟩ to save nicknames for all users, and re-add them when they leave and rejoin."
	usage = "(enable|disable)?"
	example = ("nickpreserver enable", "stickynicks disable")
	rate_limit = (9, 12)
	flags = "aed"

	def __call__(self, flags, guild, name, **void):
		bot = self.bot
		following = bot.data.nickpreservers
		update = following.update
		# Empty dictionary is enough to represent an active nick preserver here
		curr = following.get(guild.id)
		if "d" in flags:
			if guild.id in following:
				following.pop(guild.id)
			return italics(css_md(f"Disabled nickname preservation for {sqr_md(guild)}."))
		elif "e" in flags or "a" in flags:
			if guild.id not in following:
				following[guild.id] = {}
			return italics(css_md(f"Enabled nickname preservation for {sqr_md(guild)}."))
		if curr is None:
			return ini_md(f'Nickname preservation is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')
		return ini_md(f"Nickname preservation is currently enabled in {sqr_md(guild)}.")


class ThreadPreserver(Command):
	server_only = True
	name = ["KeepAlive", "ThreadBump", "AutoBump", "UnArchive"]
	min_level = 3
	min_display = "3+"
	description = 'Causes ⟨MIZA⟩ to "bump" (revive) the current thread when auto-archived.'
	usage = "(enable|disable)?"
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
				create_task(self.roleLock(role, channel))
		for thread in guild.threads:
			create_task(self.threadLock(thread, channel))
		invites = await guild.invites()
		for inv in invites:
			create_task(self.invLock(inv, channel))
		return bold(css_md(sqr_md(uni_str("LOCKDOWN REQUESTED.")), force=True))


class Archive(Command):
	time_consuming = 1
	_timeout_ = 3600
	name = ["ArchiveServer", "DownloadServer"]
	min_level = 3
	description = "Archives all messages, attachments and users into a .zip folder. Requires server permission level 3 as well as a Lv2 or above ⟨MIZA⟩ subscription to perform, and may take a significant amount of time."
	usage = "<server>?"
	flags = "f"
	rate_limit = 172800

	async def __call__(self, bot, message, guild, user, channel, argv, flags, **void):
		if argv:
			guild = await bot.fetch_guild(argv)
		if bot.get_perms(user, guild) < 3:
			raise PermissionError("You must be in the target server and have a permission level of minimum 3.")
		if max(bot.is_trusted(guild), bot.premium_level(user) * 2) < 2:
			raise PermissionError(f"Sorry, unfortunately this feature is for premium users only. Please make sure you have a subscription level of minimum 1 from {bot.kofi_url}!")
		if "f" not in flags:
			raise InterruptedError(css_md(uni_str(sqr_md(f"WARNING: SERVER DOWNLOAD REQUESTED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), 0), force=True))
		fn = f"cache/{ts_us()}.zip"
		args = [
			sys.executable,
			"misc/server-dump.py",
			bot.token,
			str(guild.id),
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
					b = await create_future(proc.stdout.read, 1)
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
					await m.edit(content=info)
		if not fn:
			raise FileNotFoundError("The requested file was not found. If this issue persists, please report it in the support server.")
		await bot.send_with_file(channel, file=CompatFile(fn), reference=message)
		info = ini_md("Archive Complete!")
		await m.edit(content=info)


class UserLog(Command):
	server_only = True
	name = ["MemberLog"]
	min_level = 3
	description = "Causes ⟨MIZA⟩ to log user and member events from the server, in the current channel."
	usage = "(enable|disable)?"
	example = ("userlog enable",)
	flags = "aed"
	rate_limit = 1

	async def __call__(self, bot, flags, channel, guild, name, **void):
		data = bot.data.logU
		update = bot.data.logU.update
		if "e" in flags or "a" in flags:
			data[guild.id] = channel.id
			return italics(css_md(f"Enabled user event logging in {sqr_md(channel)} for {sqr_md(guild)}."))
		elif "d" in flags:
			if guild.id in data:
				data.pop(guild.id)
			return italics(css_md(f"Disabled user event logging for {sqr_md(guild)}."))
		if guild.id in data:
			c_id = data[guild.id]
			channel = await bot.fetch_channel(c_id)
			return ini_md(f"User event logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'User event logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class MessageLog(Command):
	server_only = True
	min_level = 3
	description = "Causes ⟨MIZA⟩ to log message events from the server, in the current channel."
	usage = "(enable|disable)?"
	example = ("messagelog enable",)
	flags = "aed"
	rate_limit = 1

	async def __call__(self, bot, flags, channel, guild, name, **void):
		data = bot.data.logM
		update = bot.data.logM.update
		if "e" in flags or "a" in flags:
			data[guild.id] = channel.id
			return italics(css_md(f"Enabled message event logging in {sqr_md(channel)} for {sqr_md(guild)}."))
		elif "d" in flags:
			if guild.id in data:
				data.pop(guild.id)
			return italics(css_md(f"Disabled message event logging for {sqr_md(guild)}."))
		if guild.id in data:
			c_id = data[guild.id]
			channel = await bot.fetch_channel(c_id)
			return ini_md(f"Message event logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'Message event logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class FileLog(Command):
	server_only = True
	min_level = 3
	description = "Causes ⟨MIZA⟩ to log deleted files from the server, in the current channel."
	usage = "(enable|disable)?"
	example = ("filelog enable",)
	flags = "aed"
	rate_limit = 1

	async def __call__(self, bot, flags, channel, guild, name, **void):
		data = bot.data.logF
		update = bot.data.logF.update
		if "e" in flags or "a" in flags:
			data[guild.id] = channel.id
			return italics(css_md(f"Enabled file deletion logging in {sqr_md(channel)} for {sqr_md(guild)}."))
		elif "d" in flags:
			if guild.id in data:
				data.pop(guild.id)
			return italics(css_md(f"Disabled file deletion logging for {sqr_md(guild)}."))
		if guild.id in data:
			c_id = data[guild.id]
			channel = await bot.fetch_channel(c_id)
			return ini_md(f"File deletion logging for {sqr_md(guild)} is currently enabled in {sqr_md(channel)}.")
		return ini_md(f'File deletion logging is currently disabled in {sqr_md(guild)}. Use "{bot.get_prefix(guild)}{name} enable" to enable.')


class StarBoard(Command):
	server_only = True
	min_level = 2
	description = "Causes ⟨MIZA⟩ to repost popular messages with a certain number of a specified reaction anywhere from the server, into the current channel."
	usage = "<0:reaction> <1:react_count(1)>? <enable_channel{?e}>? <disable_channel{?d}>? <channel_ids(-1)>*"
	example = ("starboard 🐱 6", "starboard disable")
	flags = "aed"
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = 1

	async def __call__(self, bot, args, user, message, channel, guild, flags, **void):
		data = bot.data.starboards
		if "e" in flags or "d" in flags:
			selected = []
			if data.get(guild.id):
				for k, t in data[guild.id].items():
					if k and t[1] == channel.id:
						selected.append(k)
			if not selected:
				d = dict(data[guild.id])
				d.pop(None, None)
				if not d:
					data.pop(guild.id, None)
					data.update(guild.id)
				return ini_md(f"Starboard reposting is currently disabled in {sqr_md(channel)}.")
			emojis = []
			for e_data, (count, c_id, *disabled) in zip(selected, map(data[guild.id].get, selected)):
				try:
					e_id = int(e_data)
				except:
					emoji = e_data
				else:
					emoji = await bot.fetch_emoji(e_id)
				emojis.append(str(emoji))
			if len(selected) > 1:
				triggers = "triggers "
			else:
				triggers = "trigger "
			triggers += sqr_md(", ".join(emojis))
			if not args:
				if "d" in flags:
					for k in selected:
						data[guild.id].pop(k, None)
					d = dict(data[guild.id])
					d.pop(None, None)
					if not d:
						data.pop(guild.id, None)
					data.update(guild.id)
					return italics(css_md(f"Disabled starboard {triggers} for {sqr_md(guild)}."))
				for k in selected:
					data[guild.id][k] = data[guild.id][k][:2]
				data.update(guild.id)
				return italics(css_md(f"No longer exluding channels for starboard {triggers}."))
			args = set(verify_id(a) for a in args)
			if guild.id in args:
				args.remove(guild.id)
				args = set(c.id for c in guild.text_channels).difference(args)
			channels = []
			for c_id in args:
				c = await bot.fetch_channel(c_id)
				if c.guild.id != guild.id:
					continue
				channels.append(c)
				for k in selected:
					count, c_id2, *disabled = data[guild.id][k]
					if not disabled:
						disabled = [set()]
					disabled = disabled[0]
					if "d" in flags:
						disabled.add(c_id)
					else:
						disabled.discard(c_id)
					data[guild.id][k] = (count, c_id2, disabled)
			data.update(guild.id)
			channels = sqr_md(", ".join(map(str, sorted(channels, key=lambda c: c.id))))
			now = "Now" if "d" in flags else "No longer"
			return italics(css_md(f"{now} excluding {channels} from starboard {triggers}."))
		if not args:
			buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
			await send_with_reply(
				None,
				message,
				"*```" + "\n" * ("z" in flags) + "callback-admin-starboard-"
				+ str(user.id) + "_0"
				+ "-\nLoading Starboard database...```*",
				buttons=buttons,
			)
			return
		e_data = args.pop(0)
		try:
			e_id = int(e_data)
		except:
			emoji = e_data
		else:
			emoji = await bot.fetch_emoji(e_id)
		emoji = str(emoji)
		if args:
			count = await bot.eval_math(" ".join(args))
		else:
			count = 1
		boards = data.setdefault(guild.id, {})
		boards[emoji] = (count, channel.id, set())
		data.update(guild.id)
		return ini_md(f"Successfully added starboard to {sqr_md(channel)}, with trigger {sqr_md(emoji)}: {sqr_md(count)}.")

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos = list(map(int, vals.split("_", 1)))
		if reaction not in (None, self.directions[-1]) and perm < 3:
			return
		if reaction not in self.directions and reaction is not None:
			return
		guild = message.guild
		user = await bot.fetch_user(u_id)
		data = bot.data.starboards
		curr = data.setdefault(guild.id, {}).copy()
		curr.pop(None, None)
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
			"callback-admin-starboard-"
			+ str(u_id) + "_" + str(pos)
			+ "-\n"
		)
		if not curr:
			content += f"No currently assigned starboard triggers for {str(guild).replace('`', '')}.```*"
			msg = ""
		else:
			content += f"{len(curr)} starboard triggers currently assigned for {str(guild).replace('`', '')}:```*"

			def disp(t):
				s = f"×{t[0]} -> {sqr_md(bot.get_channel(t[1]))}"
				if len(t) > 2:
					s += ", excludes " + ", ".join(sqr_md(bot.get_channel(i)) for i in t[2])
				return s

			msg = ini_md(iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}, key=disp))
		colour = await self.bot.get_colour(guild)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		)
		emb.set_author(**get_author(guild))
		more = len(curr) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Crosspost(Command):
	server_only = True
	name = ["Repost", "Subscribe"]
	min_level = 3
	description = "Causes ⟨MIZA⟩ to automatically crosspost all messages from the target channel, into the current channel."
	usage = "<channel> <disable{?d}>?"
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
				data.update(target.id)
				return italics(css_md(f"Disabled message crossposting from {sqr_md(target)} to {sqr_md(channel)}."))
			for c_id, v in data.items():
				try:
					v.remove(channel.id)
				except KeyError:
					pass
				else:
					data.update(c_id)
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
		data.update(target.id)
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
		create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)


class Publish(Command):
	server_only = True
	name = ["News", "AutoPublish"]
	min_level = 3
	description = "Causes ⟨MIZA⟩ to automatically publish all posted messages in the current channel."
	usage = "(enable|disable)? <force{?x}>?"
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
#     usage = "<0:option(image_url)(image_position{centre})(shape[{square},circle,rounded_square,hexagon])> <-1:value> <disable(?d)>"
#     flags = "h"
#     rate_limit = (2, 4)

#     async def __call__(self, bot, args, message, channel, guild, flags, perm, name, **void):
#         pass


class UpdateMutes(Database):
	name = "mutes"
	sem = Semaphore(1, 1, rate_limit=3600)

	async def _call_(self):
		if self.sem.active or self.sem.busy:
			return
		ts = utc()
		async with self.sem:
			for g_id, data in tuple(self.items()):
				try:
					guild = self.bot.cache.guilds[g_id]
				except LookupError:
					self.pop(g_id, None)
					continue
				for u_id, t in tuple(data.items()):
					user = guild._members.get(u_id)
					if not u_id:
						data.pop(u_id)
						self.update(g_id)
						continue
					tou = getattr(user, "timed_out_until", None)
					if not tou and not getattr(user, "ghost", None):
						data.pop(u_id)
						self.update(g_id)
						continue
					if tou and abs(tou.timestamp() - t) < 1:
						continue
					rem = ts - t
					reason = "Mute refreshed"
					if rem < 21 * 86400:
						if rem < 0:
							rem = 0
							reason = "Mute expired"
							if not tou:
								continue
						data.pop(u_id)
						self.update(g_id)
						await user.timeout(datetime.timedelta(seconds=rem), reason=reason)
						continue
					await user.timeout(datetime.timedelta(days=21), reason=reason)

	async def _join_(self, user, guild, **void):
		ts = utc()
		data = self.get(guild.id)
		if not data:
			return
		t = data.get(user.id)
		if not t:
			return
		rem = ts - t
		if rem <= 0:
			data.pop(t, None)
			self.update(guild.id)
			return
		if rem <= 21 * 86400:
			data.pop(t, None)
			self.update(guild.id)
		else:
			rem = 21 * 86400
		await user.timeout(datetime.timedelta(seconds=rem), reason="Mute restored")


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
			self.update(g_id)
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
	no_file = True

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
			create_task(guild.leave())
			await owner.send(
				f"Apologies for the inconvenience, but {user_mention(user.id)} `({user.id})` has triggered an "
				+ f"automated server protection response due to exessive {msg} in `{no_md(guild)}` `({guild.id})`, "
				+ "and will promptly leave the server to prevent any potential further attacks."
			)
		else:
			await self.kickWarn(u_id, guild, owner, msg)

	async def _channel_delete_(self, channel, guild, **void):
		if channel.id in self.bot.cache.deleted:
			return
		user = None
		if not isinstance(channel, discord.Thread) and channel.permissions_for(guild.me).view_audit_log:
			audits = guild.audit_logs(limit=5, action=discord.AuditLogAction.channel_delete)
			ts = utc()
			cnt = {}
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
						create_task(self.targetWarn(u_id, guild, f"channel deletions `({cnt[u_id]})`"))
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
				create_task(self.targetWarn(u_id, guild, f"banning `({cnt[u_id]})`"))

	async def call(self, message, fn, known=None):
		args = (
			sys.executable,
			"misc/steganography.py",
			fn,
		)
		# print(args)
		if not known:
			proc = psutil.Popen(args, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		try:
			if not known:
				await create_future(proc.wait, timeout=3200)
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		else:
			if known:
				text = known
			else:
				text = proc.stdout.read().decode("utf-8", "replace").strip().rsplit("\n", 1)[-1]
			if text.startswith("Copyright detected"):
				print(text)
				i = text.split(": ", 1)[-1]
				text = f"Copyright detected in cache: {i}"
				if i.isnumeric():
					i = int(i)
					print(i)
					try:
						u = await self.bot.fetch_user(i)
					except:
						pass
					else:
						print(u)
						if u.id == message.author.id:
							return text
						if str(u.id) in message.content:
							return text
						if u.bot:
							react = await create_future(self.bot.data.emojis.get, "ai_art.gif")
							await message.add_reaction(react)
							return text
						create_task(send_with_react(
							u,
							(
								f"```callback-image-steganography-{u.id}_{message.channel.id}_{message.id}-\n⚠️ Steganography alert ⚠️```"
								+ f"Hey there, {user_mention(message.author.id)} has posted an image belonging to you without mentioning you. "
								+ f"Check it by visiting <https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}>, "
								+ "or react with 🗑️ to take it down!"
							),
							reacts="🗑️",
						))
						await message.reply(
							(
								f"Woah, hey there, this image you posted belongs to {user_mention(u.id)}! "
								+ "Please make sure you have permission from the author before posting their art!"
							),
						)
		return text


class CreateEmoji(Command):
	server_only = True
	name = ["EmojiCreate", "EmojiCopy", "CopyEmoji", "Emote", "Emoticon", "Emoji"]
	min_level = 2
	description = "Creates a custom emoji from a URL or attached file."
	usage = "<1:name>+ <0:url>"
	example = ("emoji how https://cdn.discordapp.com/emojis/645188934267043840.gif?size=128",)
	flags = "aed"
	no_parse = True
	rate_limit = (8, 12)
	_timeout_ = 6
	typing = True
	slash = ("Emoji",)

	async def __call__(self, bot, user, guild, channel, message, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args.extend(best_url(a) for a in message.attachments)
			argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
		if not args:
			raise ArgumentError("Please enter URL, emoji, or attached file to add.")
		async with discord.context_managers.Typing(channel):
			try:
				if len(args) > 1 and is_url(args[0]):
					args.append(args.pop(0))
				url = args.pop(-1)
				urls = await bot.follow_url(url, best=True, allow=True, limit=1)
				if not urls:
					urls = await bot.follow_to_image(argv)
					if not urls:
						urls = await bot.follow_to_image(url)
						if not urls:
							raise ArgumentError
				url = urls[0]
			except ArgumentError:
				if not argv:
					url = None
					try:
						url = await bot.get_last_image(message.channel)
					except FileNotFoundError:
						raise ArgumentError("Please input an image by URL or attachment.")
				else:
					raise ArgumentError("Please input an image by URL or attachment.")
			name = " ".join(args).strip()
			if not name:
				name = "emoji_" + str(len(guild.emojis))
			# print(name, url)
			image = resp = await bot.get_request(url, timeout=60)
			if len(image) > 1073741824:
				raise OverflowError("Max file size to load is 1GB.")
			if len(image) > 262144 or not is_image(url):
				# ts = ts_us()
				# path = "cache/" + str(ts)
				# with open(path, "wb") as f:
					# await create_future(f.write, image, timeout=18)
				o_image = image
				verified = False
				width = 128
				while len(image) > 262144 or not verified:
					print("RESIZE:", width)
					try:
						resp = await process_image(o_image, "resize_max", [width, "-o"], timeout=_timeout)
					except:
						raise
					else:
						fn = resp
						if isinstance(fn, str):
							if not os.path.exists(fn) or not os.path.getsize(fn):
								break
							r = os.path.getsize(fn) / 262144
							if r > 1:
								width = min(width - 1, floor(width / sqrt(r)))
								continue
							with open(fn, "rb") as f:
								image = await create_future(f.read, timeout=18)
						else:
							image = fn
							if not image:
								break
							r = len(image) / 262144
							if r > 1:
								width = min(width - 1, floor(width / sqrt(r)))
								continue
						verified = True
					finally:
						with suppress():
							os.remove(fn)
			emoji = await guild.create_custom_emoji(image=image, name=name, reason="CreateEmoji command")
			# This reaction indicates the emoji was created successfully
			with suppress(discord.Forbidden):
				await message.add_reaction(emoji)
		s = css_md(f"Successfully created emoji {sqr_md(emoji)} for {sqr_md(guild)}.")
		await send_with_reply(channel, message, s, ephemeral=True)


class CreateSticker(Command):
	server_only = True
	name = ["StickerCreate", "StickerCopy", "CopySticker", "Sticker"]
	min_level = 2
	description = "Creates a custom sticker from a URL or attached file."
	usage = "<1:name>+ <0:url>"
	example = ("sticker HOW https://cdn.discordapp.com/stickers/974228511357284372.png",)
	flags = "aed"
	no_parse = True
	rate_limit = (8, 12)
	_timeout_ = 8
	typing = True
	slash = ("Sticker",)

	async def __call__(self, bot, user, guild, channel, message, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args.extend(best_url(a) for a in message.attachments)
			argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
		if not args:
			raise ArgumentError("Please enter URL, emoji, or attached file to add.")
		async with discord.context_managers.Typing(channel):
			try:
				if len(args) > 1 and is_url(args[0]):
					args.append(args.pop(0))
				url = args.pop(-1)
				urls = await bot.follow_url(url, best=True, allow=True, limit=1)
				if not urls:
					urls = await bot.follow_to_image(argv)
					if not urls:
						urls = await bot.follow_to_image(url)
						if not urls:
							raise ArgumentError
				url = urls[0]
			except ArgumentError:
				if not argv:
					url = None
					try:
						url = await bot.get_last_image(message.channel)
					except FileNotFoundError:
						raise ArgumentError("Please input an image by URL or attachment.")
				else:
					raise ArgumentError("Please input an image by URL or attachment.")
			name = " ".join(args).strip()
			if not name:
				name = "emoji_" + str(len(guild.emojis))
			# print(name, url)
			image = resp = await bot.get_request(url, timeout=60)
			if len(image) > 1073741824:
				raise OverflowError("Max file size to load is 1GB.")
			# if len(image) > 512000 or not is_image(url):
			# ts = ts_us()
			# path = "cache/" + str(ts)
			# with open(path, "wb") as f:
				# await create_future(f.write, image, timeout=18)
			o_image = image
			verified = False
			width = 320
			while len(image) > 512000 or not verified:
				print("RESIZE:", width)
				try:
					resp = await process_image(o_image, "resize_max", [width, "-o", "-d", 5, "-f", "apng"], timeout=_timeout)
				except:
					raise
				else:
					fn = resp
					if isinstance(fn, str):
						if not os.path.exists(fn) or not os.path.getsize(fn):
							break
						r = os.path.getsize(fn) / 512000
						if r > 1:
							width = min(width - 1, floor(width / sqrt(r)))
							continue
						with open(fn, "rb") as f:
							image = await create_future(f.read, timeout=18)
					else:
						image = fn
						if not image:
							break
						r = len(image) / 512000
						if r > 1:
							width = min(width - 1, floor(width / sqrt(r)))
							continue
					verified = True
				finally:
					with suppress():
						os.remove(fn)
			try:
				data = await create_future(
					Request,
					f"https://discord.com/api/{api}/guilds/{guild.id}/stickers",
					method="POST",
					files=dict(
						name=(None, name),
						tags=(None, "upside_down"),
						file=image,
					),
					authorise=True,
					json=True,
				)
			except Exception as ex:
				if isinstance(ex, ConnectionError):
					msg = f"Unable to create sticker (Error {ex.errno}). Please download and add manually."
				else:
					msg = "Unable to create sticker (Unknown error). Please download and add manually."
				print_exc()
				await bot.send_with_file(
					channel,
					msg,
					file=CompatFile(image, filename=name + ".png"),
				)
				return
			sticker = f"https://media.discordapp.net/stickers/{data['id']}"
			embed = discord.Embed(colour=colour)
			embed.set_image(url=sticker)
			content = css_md(f"Successfully created sticker {sqr_md(name)} for {sqr_md(guild)}.")
		await send_with_reply(channel, message, content, embed=embed)


class ScanEmoji(Command):
	name = ["EmojiScan", "ScanEmojis"]
	min_level = 1
	description = "Scans all the emojis in the current server for potential issues."
	usage = "<count(inf)>"
	example = ("scanemoji",)
	no_parse = True
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
		# fut = create_task(send_with_reply(channel, message, "Emoji scan initiated. Delete the original message at any point in time to cancel."))
		p = bot.get_prefix(guild)
		if argv:
			count = await bot.eval_math(argv)
		else:
			count = inf
		found = 0
		async with discord.context_managers.Typing(channel):
			for emoji in sorted(guild.emojis, key=lambda e: e.id):
				url = str(emoji.url)
				resp = await create_future(subprocess.run, self.ffprobe_start + (url,), stdout=subprocess.PIPE)
				width, height = map(int, resp.stdout.splitlines())
				if width < 128 or height < 128:
					found += 1
					w, h = width, height
					while w < 128 or h < 128:
						w, h = w << 1, h << 1
					colour = await create_future(bot.get_colour, url)
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
		b_url = best_url(before)
		a_url = best_url(after)
		if b_url != a_url:
			with tracebacksuppressor:
				urls = await self.bot.data.exec.uproxy(b_url, a_url)
		for g_id in self.data:
			guild = self.bot.cache.guilds.get(g_id)
			if guild:
				create_task(self._member_update_(before, after, guild))

	async def _member_update_(self, before, after, guild=None):
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
		if bk != ak:
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
				name="Avatar",
				value=f"[Before]({b_url}) ➡️ [After]({a_url})",
			)
			emb.set_thumbnail(url=a_url)
			change = True
			colour[2] += 255
		if not change:
			return
		b_url = await self.bot.get_proxy_url(before)
		a_url = await self.bot.get_proxy_url(after)
		emb.set_author(name=str(after), icon_url=a_url, url=a_url)
		emb.colour = colour2raw(colour)
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
		emb.set_author(**get_author(user))
		mlist = self.bot.data.channel_cache.data.get(ch.id, ())
		count = f" ({len(mlist)}+)" if mlist else ""
		emb.description = f"{channel_mention(ch.id)}{count} was deleted by {user_mention(user.id)}."
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
		guild = getattr(user, "guild", None)
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
		age = utc() - utc_ts(user.created_at)
		if age < 86400 * 7:
			emb.description += f"\n⚠️ Account is {time_diff(utc_dt(), user.created_at)} old. ⚠️"
		self.bot.send_embeds(channel, emb)

	async def _leave_(self, user, **void):
		guild = getattr(user, "guild", None)
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
					if m.author.id == bot.deleted_user:
						print(user, user.id, "deleted!!")
						bot.data.users[user.id]["deleted"] = True
					break
		# Colour: Black
		emb = discord.Embed(colour=1)
		emb.set_author(**get_author(user))
		if not bot.data.users.get(user.id, {}).get("deleted"):
			# Check audit log to find whether user left or was kicked/banned
			with tracebacksuppressor(StopIteration):
				ts = utc()
				futs = [create_task(bot.flatten(guild.audit_logs(limit=4, action=getattr(discord.AuditLogAction, action)))) for action in ("ban", "kick", "member_prune")]
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
		roles = getattr(user, "roles", None) or ()
		rchange = escape_markdown(", ".join(role_mention(r.id) for r in roles[1:]))
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
	search_sem = Semaphore(20, 512, rate_limit=5)

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
		fn = self.get_fn(m_id)
		with suppress(KeyError):
			return self.saving[fn][m_id]
		if fn in self.loaded:
			return self.loaded[fn][m_id]
		found = self.load_file(fn)
		if not found:
			fn = self.get_fn(m_id // 10)
			with suppress(KeyError):
				return self.saving[fn][m_id]
			if fn in self.loaded:
				return self.loaded[fn][m_id]
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
		self.load_file(fn, raw=True)
		bot = self.bot
		if fn in self.loaded:
			self.loaded[fn].update(messages)
		saved = self.raws.setdefault(fn, {})
		for m_id, message in messages.items():
			m = getattr(message, "_data", None)
			if m:
				if "author" not in m:
					author = message.author
					m["author"] = dict(id=author.id, s=str(author), avatar=author.avatar and author.avatar.key)
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
					author=dict(id=author.id, s=str(author), avatar=author.avatar and author.avatar.key),
					channel_id=message.channel.id,
				)
				if message.content:
					m["content"] = message.content
				mtype = getattr(message.type, "value", message.type)
				if mtype:
					m["type"] = mtype
				flags = message.flags.value if message.flags else 0
				if flags:
					m["flags"] = flags
				for k in ("tts", "pinned", "mention_everyone", "webhook_id"):
					v = getattr(message, k, None)
					if v:
						m[k] = v
				edited_timestamp = as_str(getattr(message, "_edited_timestamp", None) or "")
				if edited_timestamp:
					m["edited_timestamp"] = edited_timestamp
				reactions = []
				for reaction in message.reactions:
					if not reaction.is_custom_emoji():
						r = dict(emoji=dict(id=None, name=str(reaction)))
						if reaction.count != 1:
							r["count"] = reaction.count
						if reaction.me:
							r["me"] = reaction.me
						reactions.append(r)
				if reactions:
					m["reactions"] = reactions
				attachments = [dict(id=a.id, size=a.size, filename=a.filename, url=a.url, proxy_url=a.proxy_url) for a in message.attachments]
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
				await create_future(self.saves, fn, messages)
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
			if len(saving) >= 8:
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
			if deleted >= 8:
				print(f"Message Database: {deleted} files deleted.")
			if os.path.exists(self.files + "/~~~"):
				self.setmtime()
		# print("MESSAGE DATABASE COMPLETE.")

	def getmtime(self):
		try:
			return os.path.getmtime(self.files + "/~~~")
		except FileNotFoundError:
			return utc() - 28 * 86400
	setmtime = lambda self: open(self.files + "/~~~", "wb").close()

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
		if not self.bot.ready and not self.searched and len(self.bot.cache.messages) <= 65536:
			self.searched = True
			t = None
			with tracebacksuppressor(FileNotFoundError):
				t = utc_ft(self.bot.data.message_cache.getmtime())
			if not t:
				t = utc_dt() - datetime.timedelta(days=7)
			create_task(self.load_new_messages(t))

	async def save_channel(self, channel, t=None):
		i = getattr(channel, "last_message_id", None)
		if i:
			if id2ts(i) < self.bot.data.message_cache.getmtime():
				return
			# async for m in self.bot.data.channel_cache.get(channel, as_message=False):
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
						futs.append(create_task(self.save_channel(channel, t)))
					if len(futs) >= 4:
						with tracebacksuppressor:
							await futs.popleft()
		for fut in futs:
			with tracebacksuppressor:
				await fut
		self.bot.data.message_cache.finished = True
		self.bot.data.message_cache.setmtime()
		print("Loading new messages completed.")

	# async def _command_(self, message, **void):
	#     if not getattr(message, "slash", None):
	#         return
	#     guild = message.guild
	#     if not guild or guild.id not in self.data:
	#         return
	#     c_id = self.data[guild.id]
	#     try:
	#         channel = await self.bot.fetch_channel(c_id)
	#     except (EOFError, discord.NotFound):
	#         self.data.pop(guild.id)
	#         return
	#     emb = await self.bot.as_embed(message, link=True)
	#     emb.colour = discord.Colour(0x00FFFF)
	#     action = f"**Slash command executed in** {channel_mention(message.channel.id)}:\n"
	#     emb.description = lim_str(action + (emb.description or ""), 4096)
	#     self.bot.send_embeds(channel, emb)

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
		emb = await self.bot.as_embed(after)
		emb2 = await self.bot.as_embed(before)
		emb.colour = discord.Colour(0x0000FF)
		action = f"**Message edited in** {channel_mention(after.channel.id)}:\n[View Message](https://discord.com/channels/{guild.id}/{after.channel.id}/{after.id})"
		emb.add_field(name="Before", value=lim_str(emb2.description, 1024))
		emb.add_field(name="After", value=lim_str(emb.description, 1024))
		emb.description = action
		emb.timestamp = before.edited_at or after.created_at
		self.bot.send_embeds(channel, emb)

	# Delete events must attempt to find the user who deleted the message
	async def _delete_(self, message, bulk=False, **void):
		cu_id = self.bot.id
		if bulk:
			return
		guild = message.guild
		if not guild or guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		now = utc()
		u = message.author
		name_id = str(u)
		url = await self.bot.get_proxy_url(u)
		action = discord.AuditLogAction.message_delete
		try:
			t = u
			init = user_mention(t.id)
			d_level = self.bot.is_deleted(message)
			if d_level:
				if d_level > 1:
					if d_level < 3:
						pass
						# self.logDeleted(message)
					return
				t = self.bot.user
				init = user_mention(t.id)
			else:
				# Attempt to find who deleted the message
				if not guild.get_member(cu_id).guild_permissions.view_audit_log:
					raise PermissionError
				al = await self.bot.flatten(guild.audit_logs(
					limit=5,
					action=action,
				))
				for e in reversed(al):
					# This is because message delete events stack
					try:
						cnt = e.extra.count
					except AttributeError:
						cnt = int(e.extra.get("count", 1))
					h = e.created_at.timestamp()
					cs = set_dict(self.dc, h, 0)
					c = cnt - cs
					if c >= 1:
						if self.dc[h] == 0:
							self.dc[h] = cnt
						else:
							self.dc[h] += cnt
					s = (3, 3600)[c >= 1]
					cid = e.extra.channel.id
					if now - h < s:
						if (not e.target or e.target.id == u.id or u.id == self.bot.deleted_user) and cid == message.channel.id:
							t = e.user
							init = user_mention(t.id)
							message.author = e.target
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
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		now = utc()
		action = discord.AuditLogAction.message_bulk_delete
		try:
			init = "[UNKNOWN USER]"
			if self.bot.is_deleted(messages[-1]):
				t = self.bot.user
				init = user_mention(t.id)
			else:
				# Attempt to find who deleted the messages
				if not guild.get_member(cu_id).guild_permissions.view_audit_log:
					raise PermissionError
				al = await self.bot.flatten(guild.audit_logs(
					limit=5,
					action=action,
				))
				for e in reversed(al):
					# For some reason bulk message delete events stack too
					try:
						cnt = e.extra.count
					except AttributeError:
						cnt = int(e.extra.get("count", 1))
					h = e.created_at.timestamp()
					cs = set_dict(self.dc, h, 0)
					c = cnt - cs
					if c >= len(messages):
						if self.dc[h] == 0:
							self.dc[h] = cnt
						else:
							self.dc[h] += cnt
					s = (5, 3600)[c >= len(messages)]
					if now - h < s:
						if (not e.target or e.target.id == messages[-1].channel.id):
							t = e.user
							init = user_mention(t.id)
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
			emb = await self.bot.as_embed(message, link=True)
			emb.colour = discord.Colour(0x7F007F)
			embs.append(emb)
		self.bot.send_embeds(channel, embs)


class UpdateFileLogs(Database):
	name = "logF"

	async def _delete_(self, message, **void):
		if self.bot.is_deleted(message) > 1:
			return
		if not message.attachments:
			return
		guild = message.guild
		if guild.id not in self.data:
			return
		c_id = self.data[guild.id]
		try:
			channel = await self.bot.fetch_channel(c_id)
		except (EOFError, discord.NotFound):
			self.data.pop(guild.id)
			return
		# Attempt to recover files from their proxy URLs, otherwise send the original URLs
		msg = deque()
		fils = []
		for a in message.attachments:
			try:
				b = await self.bot.get_attachment(a.url, full=False, allow_proxy=True)
				fil = CompatFile(seq(b), filename=str(a).rsplit("/", 1)[-1])
				fils.append(fil)
			except:
				msg.append(proxy_url(a))
		colour = await self.bot.get_colour(message.author)
		emb = discord.Embed(colour=colour)
		emb.description = f"File{'s' if len(fils) + len(msg) != 1 else ''} deleted from {user_mention(message.author.id)}"
		msg = "\n".join(msg) if msg else None
		if len(fils) == 1:
			return await self.bot.send_with_file(channel, msg, embed=emb, file=fils[0])
		await channel.send(msg, embed=emb, files=fils)


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
	async def _call_(self):
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
								create_task(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
								embs.clear()
							embs.append(emb)
							reacts = None
						if embs:
							create_task(self.bot.send_as_webhook(channel, embeds=embs, username=k[0], avatar_url=k[1]))
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
				footer = eval(repr(emb.footer), dict(EmbedProxy=dict))
				footer.pop("proxy_icon_url", None)
				embed.set_footer(**footer)
			if emb.timestamp:
				embed.timestamp = emb.timestamp
			for f in emb.fields:
				if f:
					embed.add_field(name=f.name, value=f.value, inline=getattr(f, "inline", True))
			embeds.append(embed)
		files = deque()
		for a in message.attachments:
			b = await self.bot.get_attachment(a.url, full=False)
			files.append(CompatFile(seq(b), filename=getattr(a, "filename", "untitled")))
		for c_id in tuple(self.data[message.channel.id]):
			try:
				channel = await self.bot.fetch_channel(c_id)
			except:
				print_exc()
				s = self.data[message.channel.id]
				s.discard(c_id)
				if not s:
					self.pop(message.channel.id)
				self.update(message.channel.id)
				continue
			name = message.guild.name + "\u2009﹟" + str(message.channel)
			url = best_url(message.guild)
			create_task(self.bot.send_as_webhook(channel, content, embeds=list(embeds), files=list(files), username=name, avatar_url=url))


class UpdateStarboards(Database):
	name = "starboards"
	sems = {}

	async def _reaction_add_(self, message, react, **void):
		if not message.guild or message.guild.id not in self.data:
			return
		temp = self.data[message.guild.id].get(react)
		if not temp:
			return
		e_id, count, *disabled = temp
		if disabled and message.channel.id in disabled[0]:
			return
		table = self.data[message.guild.id]
		req = table[react][0]
		if not req < inf:
			return
		count = sum(r.count for r in message.reactions if str(r.emoji) == react)
		if count <= 1:
			message = await discord.abc.Messageable.fetch_message(message.channel, message.id)
			self.bot.add_message(message, files=False, force=True)
			count = sum(r.count for r in message.reactions if str(r.emoji) == react)
		sem = self.sems.setdefault(message.guild.id, Semaphore(1, inf))
		async with sem:
			if message.id not in table.setdefault(None, {}):
				if count >= req and count < req + 2:
					embed = await self.bot.as_embed(message, link=True, colour=True)
					text, link = embed.description.rsplit("\n\n", 1)
					description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table) + "   " + link
					embed.description = lim_str(description, 4096)
					try:
						channel = await self.bot.fetch_channel(table[react][1])
						m = await channel.send(embed=embed)
					except (discord.NotFound, discord.Forbidden):
						table.pop(react)
					else:
						table[None][message.id] = m.id
						with tracebacksuppressor(RuntimeError, KeyError):
							while len(table[None]) > 16384:
								table[None].pop(next(iter(table[None])))
						self.update(message.guild.id)
			else:
				try:
					channel = await self.bot.fetch_channel(table[react][1])
					m = await self.bot.fetch_message(table[None][message.id], channel)
					embed = await self.bot.as_embed(message, link=True, colour=True)
					text, link = embed.description.rsplit("\n\n", 1)
					description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in sorted(message.reactions, key=lambda r: -r.count) if str(r.emoji) in table) + "   " + link
					embed.description = lim_str(description, 4096)
					await m.edit(content=None, embed=embed)
				except (discord.NotFound, discord.Forbidden):
					table[None].pop(message.id, None)
				else:
					table[None][message.id] = m.id
					with tracebacksuppressor(RuntimeError, KeyError):
						while len(table[None]) > 16384:
							table[None].pop(next(iter(table[None])))
					self.update(message.guild.id)

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
				reacts = sorted(map(str, message.reactions), key=lambda r: -r.count)
				if not reacts:
					return
				react = reacts[0]
				channel = await self.bot.fetch_channel(table[react][1])
				m = await self.bot.fetch_message(table[None][message.id], channel)
				embed = await self.bot.as_embed(message, link=True, colour=True)
				text, link = embed.description.rsplit("\n\n", 1)
				description = text + "\n\n" + " ".join(f"{r.emoji} {r.count}" for r in reacts if str(r.emoji) in table) + "   " + link
				embed.description = lim_str(description, 4096)
				await m.edit(content=None, embed=embed)
			except (discord.NotFound, discord.Forbidden):
				table[None].pop(message.id, None)
			else:
				table[None][message.id] = m.id
				with tracebacksuppressor(RuntimeError, KeyError):
					while len(table[None]) > 16384:
						table[None].pop(next(iter(table[None])))
				self.update(message.guild.id)


class UpdateRolegivers(Database):
	name = "rolegivers"

	async def _nocommand_(self, text, message, orig, **void):
		if not message.guild or not orig:
			return
		user = message.author
		guild = message.guild
		bot = self.bot
		if bot.get_perms(user, message.guild) < 0 or utc() - user.created_at.timestamp() < 86400 * 7:
			return
		assigned = self.data.get(message.channel.id, ())
		for k in assigned:
			if not ((k in text) if is_alphanumeric(k) else (k in message.content.casefold())):
				continue
			al = assigned[k]
			for r in al[0]:
				try:
					role = await bot.fetch_role(r, guild)
					if role is None:
						raise LookupError
				except LookupError:
					al[0].remove(r)
					bot.data.rolegivers.update(message.channel.id)
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
				# if len(rolelist) > 1 and hasattr(rolelist, "next"):
				#     self.update(guild.id)
		# Attempt to add all roles in one API call
		try:
			await user.add_roles(*roles, reason="AutoRole", atomic=False)
		except discord.Forbidden:
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
		if (not nick or nick == user.display_name) and (not roles or {r.id for r in roles} == {r.id for r in user.roles[1:]}):
			return
		try:
			await user.edit(roles=roles, reason="RolePreserver", **nick)
		except discord.Forbidden:
			if nick:
				create_task(user.edit(nick=nick.nick, reason="NickPreserver"))
			try:
				await user.add_roles(*roles, reason="RolePreserver", atomic=False)
			except discord.Forbidden:
				await user.add_roles(*roles, reason="RolePreserver", atomic=True)
		self.data[guild.id].pop(user.id, None)
		print(f"RolePreserver: Granted {roles} to {user} in {guild}.")

	async def _leave_(self, user, guild, **void):
		if guild.id not in self.data:
			return
		# roles[0] is always @everyone
		roles = user.roles[1:]
		if roles:
			assigned = [role.id for role in roles]
			print("_leave_", guild, user, assigned)
			self.data[guild.id][user.id] = assigned
		else:
			print("_leave_", guild, user, None)
			self.data[guild.id].pop(user.id, None)
		self.update(guild.id)


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
		if getattr(user, "nick", None):
			self.data[guild.id][user.id] = user.nick
			self.update(guild.id)
		else:
			if self.data[guild.id].pop(user.id, None):
				self.update(guild.id)


class ThreadList(Command):
	name = ["ListThreads", "Threads", "ReviveThreads", "ReviveAll"]
	description = "Shows or revives all threads in the current server."
	flags = "r"
	no_parse = True
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
		for data in itertools.chain(*(fut.result().get("threads", ()) for fut in futs)):
			factory, ch_type = discord.client._threaded_channel_factory(data['type'])
			if factory is None:
				raise discord.InvalidData('Unknown channel type {type} for channel ID {id}.'.format_map(data))
			if ch_type in (discord.ChannelType.group, discord.ChannelType.private):
				c = factory(me=bot.user, data=data, state=bot._connection)
			else:
				guild_id = int(data["guild_id"])
				guild = bot.get_guild(guild_id) or Object(id=guild_id)
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
					create_task(bot.silent_delete(m))


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
