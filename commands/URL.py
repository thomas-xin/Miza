# Make linter shut up lol
from fontTools.voltLib.voltToFea import Lookup
from sympy.abc import m
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


class Invite(Command):
	description = "Sends a link to ⟨BOT⟩'s homepage, github and invite code, as well as an invite link to the current server if applicable."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("server", "website", "github", "bot", "all"),
			),
			example="server",
			default="all",
		),
	)
	macros = cdict(
		Server=cdict(
			mode="server",
		),
		Website=cdict(
			mode="website",
		),
		Github=cdict(
			mode="github",
		),
		InviteBot=cdict(
			mode="bot",
		),
	)
	rate_limit = (9, 13)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _guild, mode, **void):
		emb = discord.Embed(colour=rand_colour()).set_author(**get_author(bot.user))
		fields = cdict()
		if mode in ("server", "all"):
			try:
				assert _guild, "No Discord server found."
				member = _guild.me
				assert member.guild_permissions.create_instant_invite, "No permission to access server invites."
			except AssertionError:
				if mode == "server":
					raise
			else:
				invites = await member.guild.invites()
				invites = sorted(invites, key=lambda invite: (invite.max_age == 0, -abs(invite.max_uses - invite.uses), len(invite.url)))
				if not invites:
					c = bot.get_first_sendable(member.guild, member)
					invite = await c.create_invite(reason="Invite command")
				else:
					invite = invites[0]
				fields["Server Invite"] = str(invite.url)
		if mode in ("website", "all"):
			fields["My Website"] = bot.webserver
		if mode in ("github", "all"):
			fields["My Github"] = bot.github
		if mode in ("bot", "all"):
			fields["My Invite"] = bot.invite
		for k, v in fields.items():
			emb.add_field(name=k, value=v)
		return cdict(embed=emb)


class Preserve(Command):
	name = ["PreserveAttachmentLinks"]
	description = "Sends a reverse proxy link to preserve a Discord attachment URL, or sends a link to ⟨BOT⟩'s webserver's upload page: ⟨WEBSERVER⟩/files"
	schema = cdict(
		minimise=cdict(
			type="bool",
			description="Whether to produce the shortest possible alias",
		),
		preview=cdict(
			type="bool",
			description="Whether to produce the in-site media previews instead",
		),
		concatenate=cdict(
			type="bool",
			description="Whether to treat links as an attachment chain (messages are from same channel, all segments except head and tail have same size divisible by 1MB)",
			aliases=["concat"],
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
		Preview=cdict(
			preview=True,
		)
	)
	rate_limit = (12, 17)
	_timeout_ = 50
	slash = ("Preserve",)
	msgcmd = ("Preserve Attachment Links",)
	ephemeral = True

	async def __call__(self, bot, _channel, _message, minimise, preview, concatenate, urls, **void):
		targets = [] if concatenate else [_message]
		try:
			reference = await bot.fetch_reference(_message)
		except (LookupError, discord.NotFound):
			pass
		else:
			targets.append(reference)
		for url in find_urls(_message.content):
			if is_discord_message_link(url):
				with tracebacksuppressor:
					m = await bot.fetch_message(url)
					if m.attachments:
						targets.append(m)

		if concatenate and targets:
			cid = targets[0].channel.id
			Ms = 0
			mismatch = False
			is_head = True
			for m in targets:
				if mismatch:
					raise DomainError("Size mismatch! Only head and tail segments may have differing filesize")
				if m.channel.id != cid:
					raise DomainError(f"Channel mismatch! {m.channel.id} != {cid}")
				for a in m.attachments:
					if not is_head:
						if Ms and a.size != Ms:
							mismatch = True
						else:
							Ms = a.size
							if Ms % 1048576:
								raise ValueError(f"Middle segment filesize ({Ms}) must be divisible by 1048576 (1MB)")
					is_head = False
			mids = [m.id for m in targets]
			return shorten_chunks(Ms // 1048576, cid, mids, targets[0].attachments[0].filename, mode="c", base="https://mizabot.xyz", minimise=minimise)

		kvs = {}
		for m in targets:
			for a in m.attachments:
				kvs[a.id] = m.id
		print(kvs)
		futs = deque()
		for url in urls:
			try:
				futs.append(as_fut(minimise_url(url, kvs=kvs, minimise=minimise)))
			except Exception:
				print_exc()
				futs.append(bot.data.exec.lproxy(url, channel=_channel, minimise=minimise))
				await asyncio.sleep(0.1)
		out = await gather(*futs, max_concurrency=2)
		print(urls)
		print(out)
		if preview:
			return "\n".join(preview_url(u) for u in out)
		return "\n".join(f"<{u}>" for u in out)


class Inspect(Command):
	name = ["📂", "Magic", "Mime", "MimeType", "FileType", "FileInfo", "Identify", "Inspect", "InspectFiles"]
	description = "Detects the type, mime, and optionally details of an input file."
	schema = cdict(
		urls=cdict(
			type="url",
			description="URL or attachment to inspect",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			multiple=True,
			required=True,
		),
	)
	rate_limit = (12, 16)
	slash = True
	ephemeral = True
	msgcmd = ("Inspect Files",)

	async def identify(self, url):
		info = cdict(file=dict())
		heads = await attachment_cache.scan_headers(url, fc=True)
		info.file["Name"] = try_header_filename(heads, url)
		mimetype = heads.get("Content-Type", "application/octet-stream")
		fmt  = info.file["Format"] = mime_into(mimetype)
		info.file["Mimetype"] = mimetype
		size = heads.get("Content-Length")
		if not size or int(size) <= 10 * 1048576:
			path = await attachment_cache.download(url, filename=True)
			size = os.path.getsize(path)
		else:
			path = url
		info.file["Size"] = byte_scale(size) + "B" + f" ({size})"
		if fmt in AUDIO_FORMS or fmt in VIDEO_FORMS:
			try:
				meta = await _run_async(audio_meta, path)
			except Exception as ex:
				print(repr(ex))
			else:
				if meta.sample_rate:
					info.audio = dict()
					if meta.name:
						info.audio["Track Name"] = meta.name
					if meta.format:
						info.audio["Format"] = meta.format
					if meta.codec != "auto":
						info.audio["Codec"] = meta.codec
					if meta.duration:
						info.audio["Duration"] = round(meta.duration, 4)
					if meta.channels:
						info.audio["Channels"] = meta.channels
					if meta.bitrate:
						info.audio["Bitrate"] = byte_scale(round(meta.bitrate)) + "bps"
					if meta.sample_rate:
						info.audio["Sample Rate"] = meta.sample_rate
		if fmt in MEDIA_FORMS:
			try:
				meta = await _run_async(video_meta, path)
			except Exception as ex:
				print(repr(ex))
			else:
				if meta.format not in IMAGE_FORMS:
					info.video = dict()
					if meta.format:
						info.video["Format"] = meta.format
					if meta.codec != "auto":
						info.video["Codec"] = meta.codec
					if meta.duration:
						info.video["Duration"] = round(meta.duration, 4)
					if meta.fps:
						info.video["FPS"] = round(meta.fps, 4)
					if meta.bitrate:
						info.video["Bitrate"] = byte_scale(round(meta.bitrate)) + "bps"
					if meta.pixel_format:
						info.video["Pixel Format"] = meta.pixel_format
					if meta.frame_count:
						info.video["Frame Count"] = meta.frame_count
					if meta.width:
						info.video["Width"] = meta.width
					if meta.height:
						info.video["Height"] = meta.height
				else:
					info.image = dict()
					if meta.format:
						info.image["Format"] = meta.format
					if meta.codec != "auto":
						info.image["Codec"] = meta.codec
					if meta.duration:
						info.image["Duration"] = round(meta.duration, 4)
					if meta.fps:
						info.image["FPS"] = round(meta.fps, 4)
					if meta.bitrate:
						info.image["Bitrate"] = byte_scale(round(meta.bitrate)) + "bps"
					if meta.pixel_format:
						info.image["Pixel Format"] = meta.pixel_format
					if meta.frame_count:
						info.image["Frame Count"] = meta.frame_count
					if meta.width:
						info.image["Width"] = meta.width
					if meta.height:
						info.image["Height"] = meta.height
		return info

	async def __call__(self, bot, urls, **void):
		urls = await gather(*(bot.follow_url(url) for url in urls), max_concurrency=3)
		urls = list(itertools.chain(*urls))
		if not urls:
			raise FileNotFoundError("Please input a file by URL or attachment.")
		futs = []
		for url in urls:
			futs.append(create_task(self.identify(url)))
		colours = await gather(*(bot.get_colour(url) for url in urls), max_concurrency=3, return_exceptions=True)
		embeds = []
		for url, fut, c in zip(urls, futs, colours):
			info = await fut
			emb = discord.Embed(title=url2fn(url), url=url)
			if not isinstance(c, BaseException):
				emb.colour = c
			for k, v in info.items():
				v2 = iter2str(v).strip()
				emb.add_field(name=k.capitalize(), value=ini_md(v2), inline=False)
			embeds.append(emb)
		return cdict(embeds=embeds)


class Follow(Command):
	name = ["🚶"]
	description = "Follows a discord message link and/or finds URLs in a string."
	schema = cdict(
		urls=cdict(
			type="string",
			description="Text containing one or more URLs to search",
		),
	)
	rate_limit = (7, 10)
	slash = True
	ephemeral = True

	async def __call__(self, bot, urls, **void):
		out = await bot.follow_url(urls)
		if not out:
			raise FileNotFoundError("No valid URLs detected.")
		output = f"`Detected {len(out)} url{'s' if len(out) != 1 else ''}:`\n" + "\n".join(out)
		return output


class Urban(Command):
	name = ["📖", "UrbanDictionary"]
	description = "Searches Urban Dictionary for an item."
	schema = cdict(
		query=cdict(
			type="string",
			description="Search query",
			example="ur mom",
			required=True,
		),
	)
	rate_limit = (5, 8)
	slash = True
	ephemeral = True
	header = {
		"accept-encoding": "application/gzip",
		"x-rapidapi-host": "mashape-community-urban-dictionary.p.rapidapi.com",
		"x-rapidapi-key": AUTH.get("rapidapi_key", ""),
	}

	async def __call__(self, _channel, _message, _user, query, **void):
		url = f"https://mashape-community-urban-dictionary.p.rapidapi.com/define?term={quote_plus(query)}"
		d = await Request.aio(url, headers=self.header, timeout=12, json=True)
		resp = d["list"]
		if not resp:
			raise LookupError(f"No results for {query}.")
		resp.sort(
			key=lambda e: scale_ratio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
			reverse=True,
		)
		title = query
		fields = deque()
		for e in resp:
			fields.append(dict(
				name=e.get("word", query),
				value=ini_md(e.get("definition", "")),
				inline=False,
			))
		self.bot.send_as_embeds(_channel, title=title, fields=fields, author=get_author(_user), reference=_message)


class Browse(Pagination, Interactable, Command):
	name = ["🦆", "🌐", "Google", "Browser"]
	description = "Searches the web, and displays as text or image."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=["auto", "text"],
			),
			description="Controls how direct URLs are visited; produces an image by default",
			example="text",
			default="auto",
		),
		query=cdict(
			type="string",
			description="Search query; may be a string or URL",
			example="https://youtu.be/dQw4w9WgXcQ",
			required=True,
		),
	)
	rate_limit = (10, 16)
	slash = True
	ephemeral = True
	page_size = 7

	async def __call__(self, _user, mode, query, page, **void):
		m = 0 if mode == "auto" else 1
		# Set callback message for scrollable list
		return await self.display(_user.id, page * self.page_size, m, query)

	async def display(self, uid, pos, mode, query, diridx=-1):
		bot = self.bot

		ss = True if int(mode) == 0 else False
		urls = await bot.follow_url(query)
		argv = urls[0] if urls else query
		s = await bot.browse(argv, uid=uid)
		if isinstance(s, bytes):
			return cdict(
				file=CompatFile(s),
			)
		elif is_url(argv):
			return cdict(
				content=s,
				prefix="\xad",
			)
		return await self.default_display("search result", uid, pos, s.split("\n\n"), diridx, extra=leb128(mode) + as_bytes(query))

	async def _callback_(self, _user, index, data, **void):
		pos, more = decode_leb128(data)
		mode, more = decode_leb128(more)
		query = as_str(more)
		return await self.display(_user.id, pos, mode, query, index)