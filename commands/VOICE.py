# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

from bs4 import BeautifulSoup
from hyperchoron import encoder_mapping as hyperchoron_encoder_mapping
import pyradios

if BOT[0]:
	bot = BOT[0]


# Gets the best icon/thumbnail for a queue entry.
def get_best_icon(entry):
	try:
		return entry["thumbnail"]
	except KeyError:
		try:
			return entry["icon"]
		except KeyError:
			pass
	try:
		thumbnails = entry["thumbnails"]
		if not thumbnails:
			raise KeyError(thumbnails)
	except KeyError:
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		if not url:
			return ""
		if is_discord_attachment(url):
			if not is_image(url):
				return "https://cdn.discordapp.com/embed/avatars/0.png"
		if is_youtube_url(url):
			if "?v=" in url:
				vid = url.split("?v=", 1)[-1]
			else:
				vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
			entry["icon"] = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
			return entry["icon"]
		if is_miza_url(url):
			return "https://mizabot.xyz/static/mizaleaf.png"
		return ""
	return sorted(
		thumbnails,
		key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)),
		reverse=True,
	)[0]["url"]

async def disconnect_members(bot, guild, members, channel=None):
	futs = [member.move_to(None) for member in members]
	await gather(*futs)

async def force_dc(bot, guild):
	if bot.permissions_in(guild).move_members:
		await guild.me.move_to(None)
	return await guild.change_voice_state(channel=None)

# This messy regex helps identify and remove certain words in song titles
lyric_trans = re.compile(
	(
		"[([]+"
		"(((official|full|demo|original|extended) *)?"
		"((version|ver.?) *)?"
		"((w\\/)?"
		"(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq|english) *)?"
		"((album|video|audio|cover|remix) *)?"
		"(upload|reupload|version|ver.?)?"
		"|(feat|ft)"
		".+)"
		"[)\\]]+"
	),
	flags=re.I,
)

# Gets estimated duration from duration stored in queue entry
def e_dur(d):
	return float(d) if type(d) is str else d if d is not None else 300
def e_dur_2(e):
	return min(e_dur(e.get("duration")), e.get("end", inf)) - e.get("start", 0)
def e_dur_e(e):
	return min(e_dur(e.get("duration")), e.get("end", inf))
def e_dur_n(e):
	d = e.get("duration")
	s = time_disp(d) if d is not None else "N/A"
	start, end = e.get("start", 0), e.get("end", d)
	if start != 0 or end != d:
		start = time_disp(start or 0)
		end = time_disp(end) if end is not None else s
		return f"{start}-{end}/{s}"
	return s
def e_remainder(elapsed, length, e, reverse):
	if reverse:
		return elapsed - e.get("start", 0)
	return min(e.get("end", inf), length) - elapsed


# runs org2xm on a file, with an optional custom sample bank.
def org2xm(org):
	if not org or not isinstance(org, (bytes, memoryview)):
		if not is_url(org):
			raise TypeError("Invalid input URL.")
		org = verify_url(org)
		data = Request(org)
		if not data:
			raise FileNotFoundError("Error downloading file content.")
	else:
		if org[:4] != b"Org-":
			raise ValueError("Invalid file header.")
		data = org
	# Write org data to file.
	r_org = temporary_file("org")
	with open(r_org, "wb") as f:
		f.write(data)
	args = ["misc/OrgExport", r_org, "48000", "0"]
	print(args)
	subprocess.check_output(args, stdin=subprocess.DEVNULL)
	r_wav = temporary_file("wav")
	if not os.path.exists(r_wav):
		raise FileNotFoundError("Unable to locate converted file.")
	if not os.path.getsize(r_wav):
		raise RuntimeError("Converted file is empty.")
	with suppress():
		os.remove(r_org)
	return r_wav

def mid2mp3(mid):
	url = Request(
		"https://hostfast.onlineconverter.com/file/send",
		files={
			"class": (None, "audio"),
			"from": (None, "midi"),
			"to": (None, "mp3"),
			"source": (None, "file"),
			"file": mid,
			"audio_quality": (None, "192"),
		},
		method="post",
		decode=True,
	)
	fn = url.rsplit("/", 1)[-1].strip("\x00")
	for i in range(360):
		with Delay(1):
			test = Request(f"https://hostfast.onlineconverter.com/file/{fn}")
			if test == b"d":
				break
	r_mp3 = temporary_file("mp3")
	with open(r_mp3, "wb") as f:
		f.write(Request(f"https://hostfast.onlineconverter.com/file/{fn}/download"))
	return r_mp3

def png2wav(png):
	r_png = temporary_file("png")
	r_wav = temporary_file("wav")
	args = [sys.executable, "png2wav.py", r_png, r_wav]
	with open(r_png, "wb") as f:
		f.write(png)
	print(args)
	subprocess.run(args, cwd="misc", stderr=subprocess.PIPE)
	return r_wav

def ecdc_encode(ecdc, bitrate="24k", name=None, source=None, thumbnail=None):
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	if source and thumbnail and unyt(thumbnail) == unyt(source):
		thumbnail = None
	b = await_fut(process_image("ecdc_encode", "$", [ecdc, bitrate, name, source, thumbnail], cap="ecdc", timeout=300))
	out = temporary_file("ecdc")
	with open(out, "wb") as f:
		f.write(b)
	return out

def ecdc_decode(ecdc, out=None):
	fmt = out.rsplit(".", 1)[-1] if out else "opus"
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	b = await_fut(process_image("ecdc_decode", "$", [ecdc, fmt], cap="ecdc", timeout=300))
	out = out or temporary_file(fmt)
	with open(out, "wb") as f:
		f.write(b)
	return out

async def ecdc_encode_a(ecdc, bitrate="24k", name=None, source=None, thumbnail=None):
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	if source and thumbnail and unyt(thumbnail) == unyt(source):
		thumbnail = None
	b = await process_image("ecdc_encode", "$", [ecdc, bitrate, name, source, thumbnail], cap="ecdc", timeout=300)
	out = temporary_file("ecdc")
	with open(out, "wb") as f:
		f.write(b)
	return out

async def ecdc_decode_a(ecdc, out=None):
	fmt = out.rsplit(".", 1)[-1] if out else "opus"
	if isinstance(ecdc, str):
		with open(ecdc, "rb") as f:
			ecdc = f.read()
	b = await process_image("ecdc_decode", "$", [ecdc, fmt], cap="ecdc", timeout=300)
	out = out or temporary_file(fmt)
	with open(out, "wb") as f:
		f.write(b)
	return out

CONVERTERS = {
	b"MThd": mid2mp3,
	b"Org-": org2xm,
	b"ECDC": ecdc_decode,
}

def select_and_convert(stream):
	print("Selecting and converting", stream)
	resp = reqs.next().get(stream, headers=Request.header(), timeout=8, stream=True)
	b = seq(resp)
	try:
		convert = CONVERTERS[b[:4]]
	except KeyError:
		convert = png2wav
	b = b.read()
	return convert(b)


async def search_one(bot, query):
	# Perform search concurrently, may contain multiple URLs
	out = None
	urls = await bot.follow_url(query, priority_order=("text", "video", "audio"))
	if urls:
		if len(urls) == 1:
			query = urls[0]
		else:
			out = [csubmit(bot.audio.asubmit(f"ytdl.search({repr(url)})")) for url in urls]
	if out is None:
		resp = await bot.audio.asubmit(f"ytdl.search({repr(query)})")
	else:
		resp = deque()
		for fut in out:
			temp = await fut
			# Ignore errors when searching with multiple URLs
			if type(temp) not in (str, bytes):
				resp.extend(temp)
	return resp


class Queue(Pagination, Command):
	server_only = True
	name = ["▶️", "P", "Q", "Play", "PlayNow", "PlayNext", "Enqueue", "Search&Play"]
	alias = name + ["LS"]
	description = "Shows the music queue, or plays a song in voice."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("all", "first", "random", "last", "now", "next"),
				accepts=dict(force="now", budge="next"),
			),
			description="Determines which song(s) to add if the link resolves to a playlist",
			example="next",
			default="all",
		),
		query=cdict(
			type="string",
			description="Song by name or URL",
			example="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
		),
		index=cdict(
			type="index",
			description="Position to insert song(s)",
			example="4",
			default=[-1],
		),
		start=cdict(
			type="timedelta",
			description="start position; automatically seeks when played",
			greedy=False,
		),
		end=cdict(
			type="timedelta",
			description="end position; subsequent audio skipped automatically",
			greedy=False,
		),
	)
	macros = cdict(
		PlayNow=cdict(
			mode="now",
		),
		PlayNext=cdict(
			mode="next",
		),
		PN=cdict(
			mode="next",
		),
	)
	_timeout_ = 2
	rate_limit = (3.5, 5)
	typing = True
	slash = ("Play", "Queue")
	msgcmd = ("Search & Play",)
	exact = False

	async def __call__(self, bot, _user, _perm, _message, _channel, _guild, _name, _comment, mode="all", query=None, index=-1, start=None, end=None, **void):
		vc_ = await select_voice_channel(_user, _channel)
		if query and _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		# Start typing event asynchronously to avoid delays
		async with discord.context_managers.Typing(_channel):
			fut = csubmit(bot.audio.asubmit(f"AP.join({vc_.id},{_channel.id},{_user.id})"))
			if not query:
				await fut
				q, paused = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue,a.settings.pause")
				if len(q) and paused and ("▶️" in _name or _name.startswith("p")):
					# With no item specified, the "play" alias is used for resuming rather than listing the queue
					await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).settings.__setitem__('pause',False)\nreturn a.ensure_play()")
					return cdict(
						content=css_md(f"Successfully resumed audio playback in {sqr_md(_guild)}."),
						reacts="❎",
					)
				# Set callback message for scrollable list
				return await self.display(_user.id, 0, _guild.id)
			if mode == "now":
				index = [0]
			elif mode == "next":
				index = [1]
			if index[0] != -1:
				if _perm < 1 and {m.id for m in vc_.members}.difference([_user.id, bot.id]):
					raise self.perm_error(_perm, 1, "to force insert while other users are in voice")
			resp = await search_one(bot, query)
			# Wait for audio player to finish loading if necessary
			await fut
			try:
				q, settings, paused, reverse, (elapsed, length) = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue,a.settings,a.settings.pause,a.reverse,a.epos")
			except KeyError:
				raise KeyError("Unable to communicate with voice client! (Please verify that I have permission to join the voice channel?)")
		settings = astype(settings, cdict)
		# Raise exceptions returned by searches
		if type(resp) is str:
			print(resp)
			raise evalEX(resp)
		if not resp:
			raise LookupError(f"No results for {query}.")
		if len(resp) > 1:
			if mode == "first":
				resp = [resp[0]]
			elif mode == "last":
				resp = [resp[-1]]
			elif mode == "random":
				resp = [choice(resp)]
		index = index and list(index)
		if not index:
			index = [len(q) + 1]
		elif index[0] is None:
			index[0] = 0
		elif index[0] < 0:
			index[0] += len(q) + 1
		qstart = index[0]
		qstride = 1 if len(index) < 3 or index[2] is None else index[2]
		qend = index[1] if len(index) > 1 else None
		resp = resp[:(qend - qstart) // qstride] if qend is not None else resp
		if settings.shuffle:
			resp = shuffle(resp)
		# Assign search results to queue entries
		items = alist()
		total_dur = 0
		start = 0 if start is None else start.total_seconds()
		end = inf if end is None else end.total_seconds()
		for i, e in enumerate(resp, 1):
			if i > 262144:
				break
			temp = cdict(
				name=e["name"],
				url=e["url"],
				duration=e.get("duration"),
				u_id=_user.id,
			)
			if e.get("orig"):
				temp["orig"] = e["orig"]
			elif "stream" in e:
				temp.stream = e["stream"]
				temp.icon = e.get("icon")
			dur = e_dur(temp.duration) if i < len(resp) - 1 or temp.get("duration") else inf
			if dur + total_dur < start:
				print(f"Skipping {temp.name} due to start position ({dur + total_dur} < {start}).")
				total_dur += dur
				continue
			elif start:
				temp.start = start
			items.append(temp)
			if dur + total_dur > end:
				print(f"Ending queue at {temp.name} due to end position ({dur + total_dur} > {end}).")
				temp.end = end - total_dur
				total_dur = end
				break
			total_dur += dur
		estimated = sum(e_dur_2(e) for e in q[1:(qstart if qstart >= 0 else None)])
		if q and qstart > 0:
			estimated += e_remainder(elapsed, length, q[0], reverse)
		delay = 0
		assert len(items), "No valid items to add to queue."
		cache_level = bot.audio.run(f"ytdl.in_cache({repr(items[0].url)})")
		print(items[0].url, cache_level)
		if not cache_level:
			delay = 5
		elif cache_level == 1:
			delay = 1
		total_duration = max(delay, estimated / abs(settings.speed))
		qstride = 1
		if len(index) > 1:
			qend = index[1] - index[0] if index[1] is not None else len(items)
			if len(index) > 2:
				qstride = index[2] or qstride
			items = items[:qend // qstride]
		icon = get_best_icon(resp[0])
		if icon:
			colour = await bot.get_colour(icon)
		else:
			colour = 0
		emb = discord.Embed(colour=colour)
		if icon:
			emb.set_thumbnail(url=icon)
		title = no_links(resp[0]["name"])
		if len(items) > 1:
			title += f" (+{len(items) - 1})"
		emb.title = title
		emb.url = resp[0]["url"]
		if qstride != 1:
			positions = ":".join(str(i) if i is not None else "" for i in index)
			posstr = f"Positions {positions};"
		elif qstart < len(q):
			posstr = f"Position {qstart};"
		else:
			posstr = "Estimated"
		final_duration = sum(e_dur_2(e) for e in items) if len(items) != 1 else items[0].get("duration")
		durstr = "" if not final_duration else f" ({sec2time(final_duration)})"
		print(items)
		count = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).enqueue({json_dumpstr(items)},start={qstart},stride={qstride})")
		adding = f"{count}/{len(items)} items added!" if len(items) != count else f"{len(items)} items added!" if len(items) > 1 else "Added to the queue!"
		emb.description = f"🎶 {adding} 🎶{durstr}\n*{posstr} time to play: {(DynamicDT.now() + total_duration).as_rel_discord() if isfinite(total_duration) else total_duration}.*"
		if paused:
			emb.description += f"\nNote: Player is currently paused. Use {bot.get_prefix(_guild)}resume to resume!"
		return cdict(
			content=_comment,
			embed=emb,
			reacts="❎",
		)

	def react_perms(self, perm: int):
		return None if perm < 2 else True

	async def display(self, uid, pos, gid, diridx=-1):
		bot = self.bot
		user = await bot.fetch_user(uid)
		curr, settings, paused, reverse, (elapsed, length) = await bot.audio.asubmit(f"(a:=AP.from_guild({gid})).queue,a.settings,a.settings.pause,a.reverse,a.epos")
		settings = astype(settings, cdict)
		page = 10
		pos = self.paginate(pos, len(curr), page, diridx)
		start_time = 0
		if not curr:
			stime = "`0`"
		elif settings.loop:
			stime = "`indefinite (loop)`"
		elif settings.repeat:
			stime = "`indefinite (repeat)`"
		elif paused:
			stime = "`indefinite (paused)`"
		else:
			total_time = e_remainder(elapsed, length, curr[0], reverse)
			i = 0
			for e in curr[1:]:
				total_time += e_dur_2(e)
				if i < pos:
					start_time += e_dur_2(e)
				if not 1 + i & 262143:
					await asyncio.sleep(0.25)
				i += 1
			stime = (DynamicDT.now() + total_time / abs(settings.speed)).as_rel_discord() if isfinite(total_time) else "`indefinite (stream)`"
		s = "s" if len(curr) != 1 else ""
		emb = discord.Embed(
			title=f"Queue: {len(curr)} item{s}" if len(curr) else "Queue: Currently empty",
		).set_author(**get_author(user))
		bar = await bot.create_progress_bar(18, elapsed / max(0.0001, length if curr else 0))
		if not curr:
			emb.description = f"*Use {bot.get_prefix(gid)}play to add a song!* `"
		else:
			if not curr[0].get("duration"):
				resp = await bot.audio.asubmit(f"ytdl.search(AP.from_guild({gid}).queue[0].url)")
				curr[0] = resp[0]
			emb.description = (
				f"**Estimated finish time: {stime}**"
				+ f'\n{"[`" + no_links(curr[0]["name"]) + "`]"}({curr[0]["url"]})` '
			)
		emb.description += f"({uni_str(time_disp(elapsed))}/{uni_str(time_disp(length))})`\n{bar}\n"
		icon = ""
		if curr:
			if curr[0].get("has_storyboard") or diridx != -1:
				try:
					fut = csubmit(bot.audio.asubmit(f"ytdl.get_thumbnail({json_dumpstr(curr[0])},pos={elapsed})"))
					icon = await asyncio.wait_for(asyncio.shield(fut), timeout=0.5)
				except asyncio.TimeoutError:
					pass
				except Exception:
					print_exc()
				else:
					curr[0]["has_storyboard"] = True
			if not icon:
				icon = get_best_icon(curr[0])
		if icon:
			if isinstance(icon, str):
				emb.set_thumbnail(url=icon)
			else:
				emb.set_thumbnail(url="attachment://thumb.jpg")
			emb.colour = await bot.get_colour(icon)
		embstr = ""
		curr_time = start_time
		i = pos
		maxlen = 36 if icon else 48
		highest = min(pos + page, len(curr))
		l10 = int(math.log10(max(highest - 1, 1)))
		maxlen = maxlen - l10 if curr else maxlen
		while i < highest:
			try:
				e = cdict(curr[i])
			except ValueError:
				print(i, curr[i])
				i += 1
				continue
			space = l10 - int(math.log10(max(1, i)))
			temp = "`" + " " * space
			ename = no_links(e.name)
			temp += f'【{i}】`{"[`" + lim_str(ename + " " * (maxlen - len(ename)), maxlen) + "`]"}({e.url})` ({e_dur_n(e)})`\n'
			if len(embstr) + len(temp) > 4096 - len(emb.description):
				break
			embstr += temp
			if i <= 1 or not settings.shuffle:
				curr_time += e_dur_2(e)
			i += 1
		emb.description += embstr.strip()
		max_page = len(curr) // page + 1
		if max_page > 1:
			curr_page = min(max_page, ceil(pos / page)) + 1
			emb.set_footer(text=f"Page {curr_page}/{max_page}")
		file = CompatFile(icon, filename="thumb.jpg") if isinstance(icon, byte_like) else None
		filed = {"file": file} if file else {}
		return self.construct(uid, leb128(pos), embed=emb, **filed, attachments=[])

	async def _callback_(self, _user, _guild, index, data, **void):
		pos, _ = decode_leb128(data)
		return await self.display(_user.id, pos, _guild.id, index)


class Connect(Command):
	server_only = True
	name = ["📲", "🎤", "🎵", "🎶", "Summon", "J", "Join", "Move", "Reconnect"]
	description = "Summons the bot into a voice channel, or advises it to leave."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("connect", "disconnect", "listen", "deafen"),
				aliases=dict(join="connect", leave="disconnect"),
			),
			description="Action to perform",
			example="remove",
			default="connect",
		),
		channel=cdict(
			type="channel",
			description="Target channel to join (defaults to voice channel of closest proximity)",
			example="#Voice",
		),
	)
	macros = cdict(
		Leave=cdict(
			mode="disconnect",
		),
		Disconnect=cdict(
			mode="disconnect",
		),
		DC=cdict(
			mode="disconnect",
		),
		Yeet=cdict(
			mode="disconnect",
		),
		FuckOff=cdict(
			mode="disconnect",
		),
		Listen=cdict(
			mode="listen",
		),
		Deafen=cdict(
			mode="deafen",
		),
	)
	rate_limit = (3, 4)
	slash = ("Connect", "Leave")

	async def __call__(self, bot, _user, _channel, _message=None, _perm=0, channel=None, mode="connect", vc=None, **void):
		force = bool(channel)
		if mode == "disconnect":
			vc_ = None
		elif channel:
			vc_ = channel
		else:
			# If voice channel is already selected, use that
			if vc is not None:
				vc_ = vc
			else:
				vc_ = await select_voice_channel(_user, _channel)
		# target guild may be different from source guild
		if vc_ is None:
			guild = _channel.guild
		else:
			guild = vc_.guild
		if not guild.me:
			raise RuntimeError("Server not detected!")
		# Use permission level in target guild to make sure user is able to perform command
		if _perm < 0:
			raise self.perm_error(_perm, 0, f"for command {self.name} in {guild}")
		# If no voice channel is selected, perform disconnect
		if vc_ is None:
			# if not auds.is_alone(_user) and auds.queue and _perm < 1:
			# 	raise self.perm_error(_perm, 1, "to disconnect while other users are in voice")
			try:
				await bot.audio.asubmit(f"AP.disconnect({guild.id},announce=1,cid={_channel.id},clear=1)")
			except KeyError:
				raise LookupError("Not currently in a voice channel.")
			return
		if not vc_.permissions_for(guild.me).connect:
			raise ConnectionError("Insufficient permissions to connect to voice channel.")
		# Create audio source if none already exists
		await bot.audio.asubmit(f"AP.join({vc_.id},{_channel.id},{_user.id},announce=1,force={force})")


class Skip(Command):
	server_only = True
	name = ["⏭", "🚫", "S", "SK"]
	min_display = "0~1"
	description = "Removes an entry or range of entries from the voice channel queue."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "vote", "force"),
			),
			description="Skip mode; force skips voting and bypasses repeats, but requires trusted priviledge level for other users' songs",
			example="force",
			default="auto",
		),
		slices=cdict(
			type="index",
			description="Entry or sequence of entries to remove; accepts one or more slice indicators such as 1..100 or 6:73:4",
			example="0 1 2..4 5:8:1",
			default=[[0]],
			multiple=True,
		),
		after=cdict(
			type="timedelta",
			description="Skips the song(s) after the provided timestamp rather than immediately; requires trusted priviledge level for other users' songs",
			greedy=False,
		),
	)
	macros = cdict(
		VoteSkip=cdict(
			mode="vote",
		),
		ForceSkip=cdict(
			mode="force",
		),
		FS=cdict(
			mode="force",
		),
		Remove=cdict(
			mode="force",
		),
		Rem=cdict(
			mode="force",
		),
		ClearQueue=cdict(
			mode="force",
			slices=[[None, None]],
		),
		Clear=cdict(
			mode="force",
			slices=[[None, None]],
		),
		Stop=cdict(
			mode="force",
			slices=[[None, None]],
		),
		SkipAll=cdict(
			slices=[[None, None]],
		),
	)
	macros["⏹️"] = cdict(
		mode="force",
		slices=[[None, None]],
	),
	rate_limit = (3.5, 5)
	slash = True

	async def __call__(self, bot, _guild, _channel, _user, _perm, mode, slices, after, **void):
		try:
			q, cid, settings, (elapsed, _length) = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue, a.vcc.id, a.settings, a.epos")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		settings = cdict(settings)
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		count = len(q)
		if not count:
			raise IndexError("Queue is currently empty.")
		# Calculate required vote count based on amount of non-bot members in voice
		members = sum(1 for m in vc_.members if not m.bot)
		required = 1 + members >> 1
		qsize = len(q)
		targets = RangeSet.parse(slices, qsize)
		dups = []
		votes = []
		skips = []
		for i in targets:
			entry = q[i]
			if not entry:
				q[i] = dict(name="null")
				skips.append(i)
				continue
			vote = mode == "vote" or mode == "auto" and entry.get("u_id") not in (_user.id, bot.id)
			if vote:
				if _user.id in entry.get("skips", ()):
					if len(entry.get("skips", ())) >= required:
						skips.append(i)
					else:
						dups.append(i)
				elif len(entry.get("skips", ())) + 1 >= required:
					skips.append(i)
				else:
					votes.append(i)
			elif _perm < 1 and entry.get("u_id") not in (_user.id, bot.id) and {m.id for m in vc_.members}.difference([_user.id, bot.id]):
				raise self.perm_error(_perm, 1, "to force-skip other users' entries")
			else:
				skips.append(i)
		desc = []
		if dups:
			desc.append(f"Entry {dups[0]} (`{q[dups[0]]['name']}`) has already been voted for, `{len(q[dups[0]]['skips'])}/{required}`." if len(dups) == 1 else f"{len(dups)} entries have already been voted for.")
		if votes:
			await bot.audio.asubmit(f"[e.setdefault('skips',set()).add({_user.id}) for e in AP.from_guild({_guild.id}).queue]")
			desc.append(f"Voted to skip entry {votes[0]} (`{q[votes[0]]['name']}`), `{len(votes)}/{required}`." if len(votes) == 1 else f"Voted to skip {len(votes)} entries.")
		if skips:
			if after is None:
				if mode != "force":
					lost = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).skip({skips},loop={settings.loop},repeat={settings.repeat},shuffle={settings.shuffle})")
					desc.append(f"Skipped entry {skips[0]} (`{lost[0]['name']}`)." if len(skips) == 1 else f"Skipped all ({len(skips)}) entries." if not q else f"Skipped {len(skips)} entries.")
				else:
					lost = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).skip({skips})")
					desc.append(f"Removed entry {skips[0]} (`{lost[0]['name']}`)." if len(skips) == 1 else f"Removed all ({len(skips)}) entries." if not q else f"Removed {len(skips)} entries.")
			else:
				temp = skips.copy()
				if 0 in temp:
					await bot.audio.asubmit(f"AP.from_guild({_guild.id}).queue[0].__setitem__('end',{elapsed + after.total_seconds()})")
					temp.remove(0)
				await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})) and [a.queue[i].__setitem__('end',{after.total_seconds()}) for i in {temp}]")
				desc.append((f"Entry {skips[0]} (`{q[skips[0]]['name']}`)" if len(skips) == 1 else f"{len(skips)} entries") + f" will automatically skip after {after}.")
		if not desc:
			raise IndexError("No items were skipped (Please verify your query with the current queue).")
		colour = await bot.get_colour(_user)
		emb = discord.Embed(colour=colour)
		emb.description = "\n- ".join(desc)
		return cdict(
			embed=emb,
			reacts="❎",
		)


audio_states = ("pause", "loop", "repeat", "shuffle", "quiet", "stay")
audio_settings = ("volume", "speed", "pitch", "pan", "bassboost", "reverb", "compressor", "chorus", "resample", "bitrate")
percentage_settings = ("volume", "speed", "pitch", "pan", "bassboost", "reverb", "compressor", "chorus", "resample")
def audio_key(d):
	return {k: f"{v * 100}%" if k in percentage_settings else v for k, v in d.items()}

class AudioState(Command):
	server_only = True
	name = ["State"]
	min_display = "0~1"
	description = "Adjusts boolean states of the audio player"
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=audio_states,
				accepts=dict(reset=None),
			),
			description="Setting to modify",
			example="shuffle",
		),
		value=cdict(
			type="bool",
			description="Specify a state (toggles by default)",
			example="false",
		),
	)
	macros = {
		"⏯️": cdict(
			mode="pause",
		),
		"⏸️": cdict(
			mode="pause",
			value=True,
		),
		"Toggle": cdict(
			mode="pause",
		),
		"Pause": cdict(
			mode="pause",
			value=True,
		),
		"Resume": cdict(
			mode="pause",
			value=False,
		),
		"Unpause": cdict(
			mode="pause",
			value=False,
		),
		"LoopQueue": cdict(
			mode="loop",
		),
		"LQ": cdict(
			mode="loop",
		),
		"RepeatAll": cdict(
			mode="loop",
		),
		"Repeat": cdict(
			mode="repeat",
		),
		"RepeatOne": cdict(
			mode="repeat",
		),
		"Shuffler": cdict(
			mode="shuffle",
		),
		"AutoShuffle": cdict(
			mode="shuffle",
		),
		"Quiet": cdict(
			mode="quiet",
		),
		"Stay": cdict(
			mode="stay",
		),
		"24/7": cdict(
			mode="stay",
		),
		"A": cdict(),
		"Reset": cdict(
			mode=None,
			value=True,
		),
	}
	rate_limit = (2, 4)
	slash = True

	async def __call__(self, bot, _comment, _guild, _user, _perm, mode, value, **void):
		try:
			cid, settings = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).vcc.id,a.settings")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		if not mode:
			if value:
				await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).settings.update(AP.defaults)\nreturn a.ensure_play(1)")
				return italics(css_md(f"Successfully reset all audio settings for {sqr_md(_guild)}."))
			d = audio_key(settings) # {k: v for k, v in settings.items() if k in audio_states}
			return f"Current audio states for **{escape_markdown(_guild.name)}**:\n{ini_md(iter2str(d))}"
		if value is None:
			value = not settings.get(mode)
		await bot.audio.asubmit(f"AP.from_guild({_guild.id}).settings.{mode} = {value}")
		if mode == "pause":
			await bot.audio.asubmit(f"AP.from_guild({_guild.id}).ensure_play()")
			if value:
				content = css_md(f"Successfully paused audio playback in {sqr_md(_guild)}.")
			else:
				content = css_md(f"Successfully resumed audio playback in {sqr_md(_guild)}.")
		else:
			content = css_md(f"{sqr_md(mode.capitalize())} status for audio playback in {sqr_md(_guild)} has been updated to {sqr_md(value)}.")
		if _comment:
			content = _comment + "\n" + content
		return cdict(
			content=content,
			reacts="❎",
		)


class AudioSettings(Command):
	server_only = True
	name = ["Settings"]
	min_display = "0~1"
	description = "Adjusts variable settings of the audio player"
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=audio_settings,
				accepts=dict(nightcore="resample", reset=None),
			),
			description="Setting to modify",
			example="bassboost",
		),
		value=cdict(
			type="string",
			description='Specify a value; all settings except bitrate are represented using percentages; enter "DEFAULT" to reset',
			example="200",
			greedy=False,
		),
	)
	macros = {
		"Volume": cdict(
			mode="volume",
		),
		"Vol": cdict(
			mode="volume",
		),
		"V": cdict(
			mode="volume",
		),
		"🔉": cdict(
			mode="volume",
			value=25,
		),
		"🔊": cdict(
			mode="volume",
			value=100,
		),
		"📢": cdict(
			mode="volume",
			value=400,
		),
		"Speed": cdict(
			mode="speed",
		),
		"SP": cdict(
			mode="speed",
		),
		"⏩": cdict(
			mode="speed",
			value=200,
		),
		"Rewind": cdict(
			mode="speed",
			value=-100,
		),
		"Rew": cdict(
			mode="speed",
			value=-100,
		),
		"⏪": cdict(
			mode="speed",
			value=-100,
		),
		"Pitch": cdict(
			mode="pitch",
		),
		"Transpose": cdict(
			mode="pitch",
		),
		"↕️": cdict(
			mode="pitch",
		),
		"Pan": cdict(
			mode="pan",
		),
		"Bassboost": cdict(
			mode="bassboost",
		),
		"BB": cdict(
			mode="bassboost",
		),
		"🥁": cdict(
			mode="bassboost",
		),
		"Reverb": cdict(
			mode="reverb",
		),
		"RV": cdict(
			mode="reverb",
		),
		"📉": cdict(
			mode="reverb",
		),
		"Compressor": cdict(
			mode="compressor",
		),
		"CO": cdict(
			mode="compressor",
		),
		"🗜": cdict(
			mode="compressor",
		),
		"Chorus": cdict(
			mode="chorus",
		),
		"CH": cdict(
			mode="chorus",
		),
		"📊": cdict(
			mode="chorus",
		),
		"Resample": cdict(
			mode="resample",
		),
		"Nightcore": cdict(
			mode="resample",
		),
		"NC": cdict(
			mode="resample",
		),
		"Bitrate": cdict(
			mode="bitrate",
		),
		"BPS": cdict(
			mode="bitrate",
		),
		"BR": cdict(
			mode="bitrate",
		),
	}
	rate_limit = (4, 8)
	slash = True
	exact = False

	async def __call__(self, bot, _comment, _guild, _user, _perm, mode, value, **void):
		try:
			cid, settings = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).vcc.id,a.settings")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		if not mode or value is None:
			if value:
				await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).settings.update(AP.defaults);return a.ensure_play(1)")
				return italics(css_md(f"Successfully reset all audio settings for {sqr_md(_guild)}."))
			d = audio_key(settings) # {k: v for k, v in settings.items() if k in audio_settings}
			return f"Current audio settings for **{escape_markdown(_guild.name)}**:\n{ini_md(iter2str(d))}"
		if value == "DEFAULT":
			value = await bot.audio.asubmit(f"AP.defaults[{repr(mode)}]")
			if mode in percentage_settings:
				value *= 100
		else:
			value = await bot.eval_math(full_prune(value))
		if mode in percentage_settings:
			valstr = f"{value}%"
			value = round_min(value / 100)
		else:
			value = await bot.eval_math(full_prune(value))
			valstr = str(value)
		await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).settings.{mode}={value};return a.ensure_play(1)")
		content = css_md(f"{sqr_md(mode.capitalize())} setting for audio playback in {sqr_md(_guild)} has been updated to {sqr_md(valstr)}.")
		if _comment:
			content = _comment + "\n" + content
		return cdict(
			content=content,
			reacts="❎",
		)


class Dump(Command):
	server_only = True
	name = ["Dujmpö"]
	min_display = "0~1"
	description = "Saves or loads the currently playing audio, including queue and settings, as a re-usable checkpoint file."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("save", "load", "append"),
			),
			description="Whether to save or load queue, or append loaded data to current queue",
			example="load",
		),
		url=cdict(
			type="url",
			description="Queue data to load",
			example="https://cdn.discordapp.com/attachments/731709481863479436/1052210287303999528/dump.json",
		),
	)
	macros = cdict(
		Export=cdict(
			mode="save",
		),
		Save=cdict(
			mode="save",
		),
		Import=cdict(
			mode="append",
		),
		Load=cdict(
			mode="load",
		),
		Restore=cdict(
			mode="load",
		),
	)
	rate_limit = (1, 2)
	slash = True

	async def __call__(self, bot, _guild, _channel, _user, _perm, mode, url, **void):
		vc_ = await select_voice_channel(_user, _channel)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		await bot.audio.asubmit(f"AP.join({vc_.id},{_channel.id},{_user.id})")
		if url or mode == "load":
			if mode == "save":
				raise TypeError("Unexpected file input for saving.")
			link = None
			if not url:
				async for message in bot.history(_channel, limit=300):
					if message.author.id == bot.id and message.attachments and message.attachments[0].url.split("?", 1)[0].rsplit("/", 1)[-1] in ("dump.json", "dump.zip"):
						url = message.attachments[0].url
						link = message.jump_url
						break
				if not url:
					raise LookupError("No valid dump file provided or found.")
			b = await attachment_cache.download(url)
			queue = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).load_dump({maybe_json(b).decode('ascii')},{_user.id},append={mode == 'append'})")
			count = len(queue)
			return cdict(
				content=(link + "\n" if link else "") + italics(css_md(f"Successfully loaded audio data ({count} item{'s' if count != 1 else ''}) for {sqr_md(_guild)}.")),
				reacts="❎",
			)
		b = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).get_dump()")
		ext = "json" if get_ext(b) != "zip" else "zip"
		return cdict(file=CompatFile(b, filename=f"dump.{ext}"), reacts="❎")


class Seek(Command):
	server_only = True
	name = ["↔️", "Replay"]
	min_display = "0~1"
	description = "Seeks to a position in the current audio file."
	schema = cdict(
		position=cdict(
			type="timedelta",
			description="Seek position",
			example="3m59.2s",
			default="0",
		),
	)
	rate_limit = (0.5, 3)
	slash = True

	async def __call__(self, bot, _guild, _user, _perm, position, **void):
		try:
			cid = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).vcc.id")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		try:
			await bot.audio.asubmit(f"(e:=AP.from_guild({_guild.id}).queue[0]).pop('start',0),e.pop('end',0)")
		except LookupError:
			raise LookupError("Unable to perform seek (Am I currently playing a song?)")
		await bot.audio.asubmit(f"AP.from_guild({_guild.id}).seek({position.total_seconds()})")
		return cdict(
			content=italics(css_md(f"Successfully moved audio position to {sqr_md(position)}.")),
			reacts="❎",
		)


class Jump(Command):
	server_only = True
	name = ["🔄", "Roll", "Next", "RotateQueue"]
	min_display = "0~1"
	description = "Rotates the queue to the left by a certain amount of steps."
	schema = cdict(
		position=cdict(
			type="integer",
			description="Jump position",
			example="-3",
			default="0",
		),
	)
	rate_limit = (0.5, 3)
	slash = True

	async def __call__(self, bot, _guild, _user, _perm, position, **void):
		try:
			cid = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).vcc.id")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue.rotate({-position}),a.ensure_play(2)")
		return cdict(
			content=italics(css_md(f"Successfully rotated queue {sqr_md(position)} step{'s' if abs(position) != 1 else ''}.")),
			reacts="❎",
		)


class Shuffle(Command):
	server_only = True
	name = ["🔀", "Scramble"]
	min_display = "0~1"
	description = "Immediately shuffles the queue. See ~autoshuffle for an automatic, non-disruptive version"
	schema = cdict()
	rate_limit = (0.5, 3)
	slash = True

	async def __call__(self, bot, _guild, _user, _perm, **void):
		try:
			cid = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).vcc.id")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue.shuffle(),a.ensure_play(2)")
		return cdict(
			content=italics(css_md(f"Successfully shuffled queue for {sqr_md(_guild)}.")),
			reacts="❎",
		)


class Dedup(Command):
	server_only = True
	name = ["Unique", "Deduplicate", "RemoveDuplicates"]
	min_display = "0~1"
	description = "Removes all duplicate elements from the queue."
	schema = cdict()
	rate_limit = (0.5, 3)
	slash = True

	async def __call__(self, bot, _guild, _user, _perm, **void):
		try:
			cid = await bot.audio.asubmit(f"AP.from_guild({_guild.id}).vcc.id")
		except KeyError:
			raise LookupError("Currently not playing in a voice channel.")
		vc_ = await bot.fetch_channel(cid)
		if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
			raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
		lx, ly, *_ = await bot.audio.asubmit(f"len((a:=AP.from_guild({_guild.id})).queue),len(a.queue.dedup(key=lambda e: e.url)),a.ensure_play()")
		if lx <= ly:
			raise LookupError("No duplicate elements in queue.")
		n = lx - ly
		return cdict(
			content=italics(css_md(f"Successfully removed {sqr_md(n)} duplicate item{'s' if n != 1 else ''} from queue for {sqr_md(_guild)}.")),
			reacts="❎",
		)


class UnmuteAll(Command):
	server_only = True
	time_consuming = True
	min_level = 3
	description = "Disables server mute/deafen for all members."
	rate_limit = 10

	async def __call__(self, guild, **void):
		for vc in guild.voice_channels:
			for user in vc.members:
				if user.voice is not None:
					if user.voice.deaf or user.voice.mute or user.voice.afk:
						csubmit(user.edit(mute=False, deafen=False))
		return italics(css_md(f"Successfully unmuted all users in voice channels in {sqr_md(guild)}.")), 1


class VoiceNuke(Command):
	server_only = True
	min_level = 0
	min_display = "2?"
	name = ["☢️"]
	description = "Removes all users from voice channels in the current server."
	schema = cdict()
	rate_limit = 10
	ephemeral = True

	async def __call__(self, _guild, _user, _perm, **void):
		if _perm < 2:
			await _user.move_to(None)
			return italics(css_md(f"Successfully removed {_user} from voice channels in {sqr_md(_guild)}.")), 1
		connected = set()
		for vc in voice_channels(_guild):
			for user in vc.members:
				if user.id != self.bot.id:
					if user.voice is not None:
						connected.add(user)
		await disconnect_members(self.bot, _guild, connected)
		return italics(css_md(f"Successfully removed all users from voice channels in {sqr_md(_guild)}.")), 1


class RefreshRegion(Command):
	server_only = True
	min_level = 1
	description = "Changes the current voice channel's region, always forcing a refresh."
	schema = cdict(
		region=cdict(
			type="enum",
			validation=cdict(
				enum=('brazil', 'hongkong', 'india', 'japan', 'rotterdam', 'singapore', 'south-korea', 'southafrica', 'sydney', 'us-central', 'us-east', 'us-south', 'us-west'),
			),
		),
	)
	rate_limit = 20

	async def __call__(self, _user, region, **void):
		if not _user.voice:
			raise LookupError("This command currently requires that you are in a voice channel!")
		vc = _user.voice.channel
		if vc.rtc_region == region:
			await vc.edit(rtc_region="rotterdam")
			await asyncio.sleep(1)
		await vc.edit(rtc_region=region)
		return italics(css_md(f"Successfully refreshed voice region for {sqr_md(vc)} ({sqr_md(region)}).")), 1


RB = pyradios.RadioBrowser()

class RadioCache:
	cache = AutoCache(f"{CACHE_PATH}/radio_facets", shards=1, stale=86400, timeout=86400 * 30)
	def load_countries():
		countries = fcdict()
		try:
			rb_countries = RB.countries()
		except Exception:
			globals()["RB"] = pyradios.RadioBrowser()
			rb_countries = RB.countries()
		for info in rb_countries:
			if not info.get("iso_3166_1"):
				continue
			code = info["iso_3166_1"].upper()
			countries[info["name"]] = code
			countries[code] = code
		return countries
	countries = cache.retrieve("countries", load_countries)

	@staticmethod
	def filter_results(results):
		unique = {}
		for radio in results:
			radio["name"] = radio["name"].strip()
			radio["countrycode"] = radio["countrycode"].upper()
			k = no_links(radio["url"].split("#", 1)[0])
			if k in unique and len(unique[k]["url"]) >= len(radio["url"]):
				continue
			unique[k] = radio
		return list(unique.values())

	@classmethod
	async def search(cls, countrycode="", query=""):
		key = (countrycode, query)
		return await cls.cache.aretrieve(key, cls._search, countrycode, query)

	@classmethod
	async def _search(cls, countrycode, query):
		if not countrycode:
			if not query:
				rf = await asubmit(pyradios.RadioFacets, RB)
				results = cls.filter_results(rf.result)
				results.sort(key=lambda radio: radio["name"])
				return results
			results = await asubmit(RB.search, name=query, name_exact=False)
			if not results:
				facets = await cls.search()
				results = [str_lookup(
					facets,
					query,
					key=lambda radio: radio["name"],
					fuzzy=0,
				)]
			results = cls.filter_results(results)
			return results
		if not query:
			facets = await cls.search()
			return [r for r in facets if r["countrycode"] == countrycode]
		results = await asubmit(RB.search, name=query, name_exact=False, countrycode=countrycode)
		if not results:
			country_results = await cls.search(countrycode=countrycode)
			results = [str_lookup(
				country_results,
				query,
				key=lambda radio: radio["name"],
				fuzzy=0,
			)]
		results = cls.filter_results(results)
		return results

class Radio(Pagination, Command):
	name = ["Radios", "RadioBrowser"]
	description = "Searches for a radio station livestream on https://api.radio-browser.info that can be played in voice."
	schema = cdict(
		country=cdict(
			type="enum",
			validation=cdict(
				enum=sorted(set(RadioCache.countries.values())),
				accepts=RadioCache.countries,
			),
			description="Radio station by country",
		),
		query=cdict(
			type="string",
			description="Radio station by name",
			example="BBC Radio",
			greedy=False,
		),
	)
	rate_limit = (6, 8)
	slash = True
	ephemeral = True

	async def __call__(self, _user, country, query, **void):
		query = query or ""
		if country:
			try:
				countrycode = RadioCache.countries[country]
			except KeyError:
				country = str_lookup(
					RadioCache.countries,
					query,
					fuzzy=0.05,
				)
				countrycode = RadioCache.countries[country]
			# Set callback message for scrollable list
			return await self.display(_user.id, 0, countrycode, query)
		# Set callback message for scrollable list
		return await self.display(_user.id, 0, "", query)

	async def display(self, uid, pos, countrycode, query, diridx=-1):
		curr = await RadioCache.search(countrycode, query)

		def key(curr, pos, page):

			def radio_repr(radio):
				name = lim_str(no_links(radio["name"].strip() or radio["tags"].strip() or url2fn(radio["homepage"].strip())), 80)
				return f"[{name}]({radio['url_resolved']})"

			return "\n".join(radio_repr(radio) for radio in tuple(curr)[pos:pos + page])

		search = f" for ({countrycode.upper()}, {query})" if countrycode and query else f" for ({countrycode.upper() or query})" if countrycode or query else ""
		s = "s" if len(curr) != 1 else ""
		return await self.default_display(f"result{s}{search}", uid, pos, curr, diridx, extra=as_bytes(countrycode) + b"\xad" + as_bytes(query), key=key, page_size=32, plural=False)

	async def _callback_(self, _user, index, data, **void):
		pos, more = decode_leb128(data)
		countrycode, query = map(as_str, more.split(b"\xad"))
		return await self.display(_user.id, pos, countrycode, query, index)


class Player(Command):
	server_only = True
	buttons = demap({
		b'\xe2\x8f\xaf\xef\xb8\x8f': 0,
		b'\xf0\x9f\x94\x81': 1,
		b'\xf0\x9f\x94\x80': 2,
		b'\xe2\x8f\xae': 3,
		b'\xe2\x8f\xad': 4,
		b'\xf0\x9f\x94\x8a': 5,
		b'\xf0\x9f\xa5\x81': 6,
		b'\xf0\x9f\x93\x89': 7,
		b'\xf0\x9f\x93\x8a': 8,
		b'\xe2\x99\xbb': 9,
		# b'\xe2\x8f\xaa': 10,
		# b'\xe2\x8f\xa9': 11,
		# b'\xe2\x8f\xab': 12,
		# b'\xe2\x8f\xac': 13,
		b'\xe2\x8f\x8f': 14,
		b'\xe2\x9c\x96': 15,
	})
	barsize = 24
	name = ["NP", "NowPlaying", "Playing"]
	min_display = "0~3"
	description = "Creates an auto-updating virtual audio player for the current server."
	usage = "<mode(enable|disable)>?"
	example = ("player", "np -d")
	flags = "adez"
	rate_limit = (6, 9)
	maintenance = True

	async def show(self, auds):
		q = auds.queue
		if q:
			s = q[0].skips
			if s is not None:
				skips = len(s)
			else:
				skips = 0
			output = "Playing " + str(len(q)) + " item" + "s" * (len(q) != 1) + " "
			output += skips * "🚫"
		else:
			output = "Queue is currently empty. "
		if auds.settings.repeat:
			output += "🔂"
		else:
			if auds.settings.loop:
				output += "🔁"
			if auds.settings.shuffle:
				output += "🔀"
		if auds.settings.quiet:
			output += "🔕"
		if q:
			p = auds.epos
		else:
			p = [0, 1]
		output += "```"
		output += await self.bot.create_progress_bar(18, p[0] / p[1])
		if q:
			output += "\n[`" + no_links(q[0].name) + "`](" + ensure_url(q[0].url) + ")"
		output += "\n`"
		if auds.paused or not auds.settings.speed:
			output += "⏸️"
		elif auds.settings.speed > 0:
			output += "▶️"
		else:
			output += "◀️"
		if q:
			p = auds.epos
		else:
			p = [0, 0.25]
		output += uni_str(f" ({time_disp(p[0])}/{time_disp(p[1])})`\n")
		if auds.has_options():
			v = abs(auds.settings.volume)
			if v == 0:
				output += "🔇"
			if v <= 0.5:
				output += "🔉"
			elif v <= 1.5:
				output += "🔊"
			elif v <= 5:
				output += "📢"
			else:
				output += "🌪️"
			b = auds.settings.bassboost
			if abs(b) > 1 / 6:
				if abs(b) > 5:
					output += "💥"
				elif b > 0:
					output += "🥁"
				else:
					output += "🎻"
			r = auds.settings.reverb
			if r:
				if abs(r) >= 1:
					output += "📈"
				else:
					output += "📉"
			u = auds.settings.chorus
			if u:
				output += "📊"
			c = auds.settings.compressor
			if c:
				output += "🗜️"
			e = auds.settings.pan
			if abs(e - 1) > 0.25:
				output += "♒"
			s = auds.settings.speed * 2 ** (auds.settings.resample / 12)
			if s < 0:
				output += "⏪"
			elif s > 1:
				output += "⏩"
			elif s > 0 and s < 1:
				output += "🐌"
			p = auds.settings.pitch + auds.settings.resample
			if p > 0:
				output += "⏫"
			elif p < 0:
				output += "⏬"
		return output

	async def _callback_(self, message, guild, channel, reaction, bot, perm, **void):
		if not guild.id in bot.data.audio.players:
			return
		auds = bot.data.audio.players[guild.id]
		if reaction is None:
			return
		elif reaction == 0:
			auds.player.time = inf
		elif auds.player is None or auds.player.message.id != message.id:
			return
		if perm < 1:
			return
		if not message:
			content = "```callback-voice-player-\n"
		elif message.content:
			content = message.content
		else:
			content = message.embeds[0].description
		orig = content.split("\n", 1)[0] + "\n"
		if reaction:
			if type(reaction) is bytes:
				emoji = reaction
			else:
				try:
					emoji = reaction.emoji
				except:
					emoji = str(reaction)
			if type(emoji) is str:
				emoji = reaction.encode("utf-8")
			if emoji in self.buttons:
				if hasattr(message, "int_token"):
					csubmit(bot.ignore_interaction(message))
				i = self.buttons[emoji]
				if i == 0:
					await asubmit(auds.pause, unpause=True)
				elif i == 1:
					if auds.settings.loop:
						auds.settings.loop = False
						auds.settings.repeat = True
					elif auds.settings.repeat:
						auds.settings.loop = False
						auds.settings.repeat = False
					else:
						auds.settings.loop = True
						auds.settings.repeat = False
				elif i == 2:
					auds.settings.shuffle = bool(auds.settings.shuffle ^ 1)
				elif i == 3 or i == 4:
					if i == 3:
						auds.seek(0)
					else:
						auds.queue.pop(0)
						auds.clear_source()
						await asubmit(auds.reset)
					return
				elif i == 5:
					v = abs(auds.settings.volume)
					if v < 0.25 or v >= 2:
						v = 1 / 3
					elif v < 1:
						v = 1
					else:
						v = 2
					auds.settings.volume = v
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 6:
					b = auds.settings.bassboost
					if abs(b) < 1 / 3:
						b = 1
					elif b < 0:
						b = 0
					else:
						b = -1
					auds.settings.bassboost = b
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 7:
					r = auds.settings.reverb
					if r >= 1:
						r = 0
					elif r < 0.5:
						r = 0.5
					else:
						r = 1
					auds.settings.reverb = r
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 8:
					c = abs(auds.settings.chorus)
					if c:
						c = 0
					else:
						c = 1 / 3
					auds.settings.chorus = c
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 9:
					pos = auds.pos
					auds.settings = cdict(auds.defaults)
					auds.settings.quiet = True
					await asubmit(auds.play, auds.source, pos, timeout=18)
				elif i == 10 or i == 11:
					s = 0.25 if i == 11 else -0.25
					auds.settings.speed = round(auds.settings.speed + s, 5)
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 12 or i == 13:
					p = 1 if i == 13 else -1
					auds.settings.pitch -= p
					await asubmit(auds.play, auds.source, auds.pos, timeout=18)
				elif i == 14:
					await asubmit(auds.kill)
					await bot.autodelete(message)
					return
				else:
					auds.player = None
					await bot.autodelete(message)
					return
		other = await self.show(auds)
		text = lim_str(orig + other, 4096)
		last = await self.bot.get_last_message(channel)
		emb = discord.Embed(
			description=text,
			colour=rand_colour(),
			timestamp=utc_dt(),
		).set_author(**get_author(self.bot.user))
		if message and last and message.id == last.id:
			await bot.edit_message(
				message,
				embed=emb,
			)
		else:
			buttons = [[] for _ in loop(3)]
			for s, i in self.buttons.a.items():
				s = as_str(s)
				if i < 5:
					buttons[0].append(cdict(emoji=s, custom_id=s, style=3))
				elif i < 14:
					j = 1 if len(buttons[1]) < 5 else 2
					buttons[j].append(cdict(emoji=s, custom_id=s, style=1))
				else:
					buttons[-1].append(cdict(emoji=s, custom_id=s, style=4))
			auds.player.time = inf
			temp = message
			message = await send_with_reply(
				channel,
				reference=None,
				embed=emb,
				buttons=buttons,
			)
			auds.player.message = message
			await bot.autodelete(temp)
		if auds.queue and not auds.paused & 1:
			p = auds.epos
			maxdel = p[1] - p[0] + 2
			delay = min(maxdel, p[1] / self.barsize / 2 / auds.speed)
			if delay > 10:
				delay = 10
			elif delay < 5:
				delay = 5
		else:
			delay = inf
		auds.player.time = utc() + delay
		auds.settings.quiet = True

	async def __call__(self, guild, channel, user, bot, flags, perm, **void):
		auds = await auto_join(channel.guild, channel, user, bot)
		auds.player = cdict(time=0, message=None)
		esubmit(auds.update)


# Small helper function to fetch song lyrics from json data, because sometimes genius.com refuses to include it in the HTML
def extract_lyrics(s):
	s = s[s.index("JSON.parse(") + len("JSON.parse("):]
	s = s[:s.index("</script>")]
	if "window.__" in s:
		s = s[:s.index("window.__")]
	s = s[:s.rindex(");")]
	data = literal_eval(s)
	d = eval_json(data)
	lyrics = d["songPage"]["lyricsData"]["body"]["children"][0]["children"]
	newline = True
	output = ""
	while lyrics:
		line = lyrics.pop(0)
		if type(line) is str:
			if line:
				if line.startswith("["):
					output += "\n"
					newline = False
				if "]" in line:
					if line == "]":
						if output.endswith(" ") or output.endswith("\n"):
							output = output[:-1]
					newline = True
				output += line + ("\n" if newline else (" " if not line.endswith(" ") else ""))
		elif type(line) is dict:
			if "children" in line:
				# This is a mess, the children objects may or may not represent single lines
				lyrics = line["children"] + lyrics
	return output


# Main helper function to fetch song lyrics from genius.com searches
async def get_lyrics(item, url=None):
	name = None
	description = None
	if is_url(url):
		resp = await bot.audio.asubmit(f"ytdl.extract_info({repr(url)})")
		name = resp.get("title") or resp["webpage_url"].rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
		if "description" in resp:
			description = resp["description"]
			lyr = []
			spl = resp["description"].splitlines()
			for i, line in enumerate(spl):
				if to_alphanumeric(full_prune(line)).strip() == "lyrics":
					para = []
					for j, line in enumerate(spl[i + 1:]):
						line = line.strip()
						if line and not to_alphanumeric(line).strip():
							break
						if find_urls(line):
							if para and para[-1].endswith(":") or para[-1].startswith("#"):
								para.pop(-1)
							break
						para.append(line)
					if len(para) >= 3:
						lyr.extend(para)
			lyrics = "\n".join(lyr).strip()
			if lyrics:
				print("lyrics_raw", lyrics)
				return name, lyrics
		if resp.get("automatic_captions"):
			lang = "en"
			if "formats" in resp:
				lang = None
				for fmt in resp["formats"]:
					if fmt.get("language"):
						lang = fmt["language"]
						break
			if lang in resp["automatic_captions"]:
				for cap in shuffle(resp["automatic_captions"][lang]):
					if "json" in cap["ext"]:
						break
				with tracebacksuppressor:
					data = await Request.aio(cap["url"], json=True, timeout=18)
					lyr = []
					for event in data["events"]:
						para = "".join(seg.get("utf8", "") for seg in event.get("segs", ()))
						lyr.append(para)
					lyrics = "".join(lyr).strip()
					if lyrics:
						print("lyrics_captions", lyrics)
						return name, lyrics
	url = f"https://genius.com/api/search/multi?q={item}"
	for i in range(2):
		data = {"q": item}
		rdata = await Request.aio(url, data=data, json=True, timeout=18)
		hits = chain.from_iterable(sect["hits"] for sect in rdata["response"]["sections"])
		path = None
		for h in hits:
			with tracebacksuppressor:
				name = h["result"]["title"] or name
				path = h["result"]["api_path"]
				break
		if path:
			s = "https://genius.com" + path
			page = await Request.aio(s, decode=True)
			text = page
			html = await asubmit(BeautifulSoup, text, "html.parser", timeout=18)
			lyricobj = html.find('div', class_='lyrics')
			if lyricobj is not None:
				lyrics = lyricobj.get_text().strip()
				print("lyrics_html", s)
				return name, lyrics
			try:
				lyrics = extract_lyrics(text).strip()
				print("lyrics_json", s)
				return name, lyrics
			except Exception:
				if i:
					raise
				print_exc()
				print(s)
				print(text)
	if description:
		print("lyrics_description", description)
		return name, description
	raise LookupError(f"No results for {item}.")


class Lyrics(Command):
	time_consuming = True
	name = ["SongLyrics"]
	description = "Searches genius.com for lyrics of a song."
	schema = cdict(
		query=cdict(
			type="string",
			description="Song by name or URL",
			example="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
		),
	)
	rate_limit = (7, 12)
	typing = True
	slash = True
	maintenance = True

	async def __call__(self, bot, _guild, _channel, _message, query, **void):
		if not query:
			try:
				entry = await bot.audio.asubmit(f"(a:=AP.from_guild({_guild.id})).queue[0]")
				query = entry["url"]
			except LookupError:
				raise IndexError("Queue not found. Please input a search term, URL, or file.")
		async with discord.context_managers.Typing(_channel):
			# Extract song name if input is a URL, otherwise search song name directly
			url = None
			urls = await bot.follow_url(query, priority_order=("text", "video", "audio"))
			if urls:
				resp = await search_one(bot, query)
				search = resp[0]["name"]
			else:
				search = query
			search = search.translate(self.bot.mtrans)
			# Attempt to find best query based on the song name
			item = verify_search(to_alphanumeric(lyric_trans.sub("", search)))
			ic = item.casefold()
			if ic.endswith(" with lyrics"):
				item = item[:-len(" with lyrics")]
			elif ic.endswith(" lyrics"):
				item = item[:-len(" lyrics")]
			elif ic.endswith(" acoustic"):
				item = item[:-len(" acoustic")]
			item = item.rsplit(" ft ", 1)[0].strip()
			if not item:
				item = verify_search(to_alphanumeric(search))
				if not item:
					item = search
			try:
				name, lyrics = await get_lyrics(item, url=url)
			except KeyError:
				print_exc()
				raise KeyError(f"Invalid response from genius.com for {item}")
		# Escape colour markdown because that will interfere with the colours we want
		text = clr_md(lyrics.strip()).replace("#", "♯")
		title = f"Lyrics for {name}:"
		if len(text) > 54000:
			return (title + "\n\n" + text).strip()
		bot.send_as_embeds(_channel, text, author=dict(name=title), colour=(1024, 128), md=ini_md, reference=_message)


class Download(Command):
	time_consuming = True
	_timeout_ = 75
	name = ["📥", "Search", "YTDL", "Convert", "Trim", "ConvertOrg"]
	description = "Searches and/or downloads a song from a YouTube/SoundCloud query or stream link."
	schema = cdict(
		url=cdict(
			type="url",
			description="The URL of a song or playlist to download."
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=("h264", "h265", "h266", "av1", "mp4", "mkv", "webm", "avif", "webp", "gif", "ogg", "opus", "mp3", "flac", "wav"),
			),
			description="Output format of the downloaded file.",
			default="opus",
		),
		query=cdict(
			type="text",
			description="A search query to look for a song or video.",
		),
		start=cdict(
			type="timedelta",
			description="The start time to trim the file at.",
		),
		end=cdict(
			type="timedelta",
			description="The end time to trim the file at.",
		),
	)
	rate_limit = (30, 45)
	typing = True
	slash = True
	msgcmd = ("Download as mp3",)
	ephemeral = True
	exact = False

	async def download_single(self, channel, message, url, fmt, start, end, **void):
		downloader_url = f"https://api.mizabot.xyz/ytdl?d={quote_plus(url)}&fmt={fmt}"
		if start:
			downloader_url += f"&start={float(start)}"
		if end:
			downloader_url += f"&end={float(end)}"
		fut = csubmit(send_with_reply(
			channel,
			reference=message,
			content=italics(ini_md(f"Downloading and converting {sqr_md(url)}...")),
		))
		headers = dict(Accept="application/json")
		async with niquests.AsyncSession() as session:
			resp = await session.get(downloader_url, headers=headers, verify=False, timeout=14400)
		response = await fut
		print(resp.headers)
		try:
			resp.raise_for_status()
		except Exception as ex:
			raise RuntimeError([repr(ex), lim_str(as_str(resp.content), 1024)])
		file = CompatFile(resp.content, resp.headers["Content-Disposition"].split("=", 1)[-1].strip('"'))
		response = await self.bot.edit_message(
			response,
			content=italics(ini_md(f"Uploading {file.filename}...")),
		)
		await self.bot.send_with_file(channel, file=file, reference=message)
		await self.bot.autodelete(response)

	async def __call__(self, bot, _channel, _message, url, format, query, start, end, **void):
		if not query and not url:
			raise IndexError("Please input a search term or URL.")
		if query and not url:
			query = verify_search(query)
			res = await Request.aio(
				f"https://api.mizabot.xyz/ytdl?q={quote_plus(query)}",
				json=True,
			)
			if not res:
				raise LookupError(f"No results found for {query}.")
			if len(res) == 1:
				url = res[0].url
		if url:
			# If only one result is found and the input is a URL, directly download it
			# TODO: Implement direct download
			return await self.download_single(
				channel=_channel,
				message=_message,
				url=url,
				fmt=format or "opus",
				start=start,
				end=end,
			)
		# If multiple results are found, display them for selection
		# TODO: Implement selection
		raise NotImplementedError("Multiple results found, selection not yet implemented. Please use the web version at https://api.mizabot.xyz/static/downloader.html for now!")
		# Add reaction numbers corresponding to search results for selection
		for i in range(len(res)):
			await sent.add_reaction(str(i) + as_str(b"\xef\xb8\x8f\xe2\x83\xa3"))


class Hyperchoron(Command):
	_timeout_ = 15
	description = "Runs Hyperchoron on the input URL. See https://github.com/thomas-xin/hyperchoron for more info, or to run it yourself!"
	schema = cdict(
		url=cdict(
			type="audio",
			description="Audio supplied by URL or attachment",
			example="https://cocobeanzies.mizabot.xyz/music/rainbow-critter.webm",
			aliases=["i"],
			required=True,
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(sorted(set(hyperchoron_encoder_mapping).difference("_"))),
			),
			default="nbs",
		),
		kwargs=cdict(
			type="string",
			greedy=False,
		)
	)
	macros = cdict(
		Midi2Org=cdict(
			format="org",
		),
	)
	rate_limit = (10, 20)
	slash = True

	async def __call__(self, url, format, kwargs, **void):
		default_archive = "zip"
		fo = os.path.abspath(TEMP_PATH + "/" + replace_ext(url2fn(url), default_archive))
		args = ["hyperchoron", "-i", url, *unicode_prune(kwargs or "").split(), "-f", format.casefold(), "-o", fo]
		print(args)
		proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd() + "/misc", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
		try:
			async with asyncio.timeout(3200):
				stdout, stderr = await proc.communicate()
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		if not os.path.exists(fo) or not os.path.getsize(fo):
			if proc.returncode != 0:
				stderr = as_str(stderr)
				if "```" not in stderr:
					stderr = py_md(stderr)
				raise RuntimeError(stderr)
			raise FileNotFoundError("No valid output detected!")
		b = fo
		z = zipfile.ZipFile(fo, "r")
		if len(z.filelist) == 1:
			b = z.open(z.filelist[0])
		return cdict(
			file=CompatFile(b, filename=replace_ext(url2fn(url), format if isinstance(b, zipfile.ZipExtFile) else default_archive)),
		)


class UpdateAudio(Database):
	name = "audio"
	no_file = True
	timestamp = utc()

	# Updates all voice clients
	async def __call__(self, guild=None, **void):
		t = utc()
		td, self.timestamp = t - self.timestamp, t
		bot = self.bot
		try:
			player_states = await bot.audio.asubmit("[(k,v.vc and v.vc.is_playing(),{m.id for m in v.vcc.members if (mv:=m.voice) and not m.voice.deaf and not m.voice.self_deaf}) for k,v in AP.players.items()]")
		except AttributeError:
			player_states = ()
		for state in player_states:
			gid, playing, uids = state
			if gid not in bot.cache.guilds or not playing:
				continue
			guild = await bot.fetch_guild(gid)
			for uid in uids:
				if uid != bot.id:
					member = guild.get_member(uid) or bot.cache.users[uid]
					bot.data.users.add_gold(member, td / 10)
					bot.data.dailies.progress_quests(member, "music", td)

	async def _day_(self, **void):
		import importlib
		old_version = importlib.metadata.version("yt-dlp")
		args = [python, "-m", "pip", "install", "--upgrade", "--pre", "yt-dlp"]
		print(args)
		proc = await asyncio.create_subprocess_exec(*args)
		await proc.wait()
		new_version = importlib.metadata.version("yt-dlp")
		if old_version != new_version:
			print("yt-dlp updated! Reloading dependencies...")
			bot = self.bot
			await bot.audio.asubmit("ytdl and ytdl.restart()")
			await bot.server.asubmit("ytdl and ytdl.restart()")

	# Restores all audio players from temporary database when applicable
	async def _ready_(self, bot, **void):
		globals()["bot"] = bot
		try:
			args = ["ffmpeg"]
			proc = await asyncio.create_subprocess_exec(*args)
			await proc.wait()
		except FileNotFoundError:
			print("WARNING: FFmpeg not found. Unable to convert and play audio.")


class UpdateAudioSettings(Database):
	name = "audiosettings"