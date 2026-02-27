# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

import csv
from misc import shard
from tsc_utils.flags import address_to_flag, flag_to_address
from tsc_utils.numbers import tsc_value_to_num, num_to_tsc_value


class CS_mem2flag(Command):
	name = ["mem2flag", "m2f", "CS_m2f"]
	description = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
	usage = "<0:address> <1:value[1]>?"
	example = ("cs_m2f 49e6e8 123",)
	rate_limit = 1

	async def __call__(self, bot, args, user, **void):
		if len(args) < 2:
			num = 1
		else:
			num = await bot.eval_math(" ".join(args[1:]))
		return css_md("".join(address_to_flag(int(args[0], 16), num)))


class CS_flag2mem(Command):
	name = ["flag2mem", "f2m", "CS_f2m"]
	description = "Returns the memory offset and specific bit pointed to by a given flag number."
	usage = "<flag>"
	example = ("cs_f2m A036",)
	rate_limit = 1

	async def __call__(self, bot, args, user, **void):
		flag = args[0]
		if len(flag) > 4:
			raise ValueError("Flag number should be no more than 4 characters long.")
		flag = flag.zfill(4)
		return css_md(str(flag_to_address(flag)))


class CS_num2val(Command):
	name = ["num2val", "n2v", "CS_n2v"]
	description = "Returns a TSC value representing the desired number, within a certain number of characters."
	schema = cdict(
		number=cdict(
			type="integer",
			validation="[-17776, 86658]",
			description="The number to convert",
			example="12345",
			required=True,
		),
		length=cdict(
			type="integer",
			validation="[0, 1000)",
			description="Number of available ASCII characters",
			default=4,
		),
	)
	rate_limit = 1

	def __call__(self, number, length, **void):
		return css_md(as_str(num_to_tsc_value(number, length)))


class CS_val2num(Command):
	name = ["val2num", "v2n", "CS_v2n"]
	description = "Returns the number encoded by a given TSC value."
	usage = "<tsc_value>"
	example = ("cs_v2n CCCC",)
	rate_limit = 1

	async def __call__(self, bot, args, user, **void):
		return css_md(as_str(tsc_value_to_num(args[0])))


class CS_hex2xml(Command):
	time_consuming = True
	name = ["hex2xml", "h2x", "CS_h2x"]
	description = "Converts a given Cave Story hex patch to an xml file readable by Booster's Lab."
	usage = "<hex_data>"
	example = ("cs_h2x 0x481D27 C3",)
	rate_limit = (3, 5)

	async def __call__(self, bot, argv, channel, message, **void):
		hacks = {}
		hack = argv.replace(" ", "").replace("`", "").strip("\n")
		while len(hack):
			# hack XML parser
			try:
				i = hack.index("0x")
			except ValueError:
				break
			hack = hack[i:]
			i = hack.index("\n")
			offs = hack[:i]
			hack = hack[i + 1:]
			try:
				i = hack.index("0x")
				curr = hack[:i]
				hack = hack[i:]
			except ValueError:
				curr = hack
				hack = ""
			curr = curr.replace(" ", "").replace("\n", "").replace("\r", "")
			n = 2
			curr = " ".join([curr[i:i + n] for i in range(0, len(curr), n)])
			if offs in hacks:
				hacks[offs] = curr + hacks[offs][len(curr):]
			else:
				hacks[offs] = curr
		# Generate hack template
		output = (
			'<?xml version="1.0" encoding="UTF-8"?>\n<hack name="HEX PATCH">\n'
			+ '\t<panel>\n'
			+ '\t\t<panel title="Description">\n'
			+ '\t\t</panel>\n'
			+ '\t\t<field type="info">\n'
			+ '\t\t\tHex patch converted by ' + bot.user.name + '.\n'
			+ '\t\t</field>\n'
			+ '\t\t<panel title="Data">\n'
			+ '\t\t</panel>\n'
			+ '\t\t<panel>\n'
		)
		col = 0
		for hack in sorted(hacks):
			n = 63
			p = hacks[hack]
			p = '\n\t\t\t\t'.join([p[i:i + n] for i in range(0, len(p), n)])
			output += (
				'\t\t\t<field type="data" offset="' + hack + '" col="' + str(col) + '">\n'
				+ '\t\t\t\t' + p + '\n'
				+ '\t\t\t</field>\n'
			)
			col = 1 + col & 3
		output += (
			'\t\t</panel>\n'
			+ '\t</panel>\n'
			+ '</hack>'
		)
		b = output.encode("utf-8")
		f = CompatFile(b, filename="patch.xml")
		csubmit(bot.send_with_file(channel, "Patch successfully converted!", f, reference=message))


async def periwinkle_variables_txt():
	url = "https://github.com/periwinkle9/CSMemoryMap/raw/refs/heads/main/cs%20global%20and%20static%20variables.txt"
	b = await attachment_cache.download(url, force=True)
	s = as_str(b.strip())
	lines = [line.rstrip() for line in s.splitlines()]
	assert lines[7] == "Address (bytes)  Data Type               Description", "Header not found!"
	entries = []
	entry = cdict()
	for line in lines[9:]:
		addr = line[:6].strip()
		size = line[6:13].strip()
		dtype = line[13:41].strip()
		desc = line[41:].strip()
		if size.startswith("(TODO") or size.startswith("-") or dtype == ".":
			continue
		if addr:
			if entry:
				entries.append(entry)
			entry = cdict(addr=int(addr, 16))
		elif size:
			if entry:
				entries.append(entry.copy())
				if entry.get("size"):
					entry.addr += entry.size
		if size:
			entry.size = int(size)
		if dtype:
			entry.dtype = dtype
			if dtype == "(struct padding)":
				entry.desc = ""
		if desc:
			if addr or size or not entry.desc:
				entry.desc = desc
			elif desc[0].islower():
				entry.desc += " " + desc
			else:
				entry.desc += "\n" + desc
	if entry:
		entries.append(entry)
	results = []
	for entry in entries:
		res = {
			"Address": hex(entry.addr)[2:].upper(),
		}
		if entry.get("size"):
			res["Size (bytes)"] = entry.size
		if entry.get("dtype"):
			res["Data Type"] = entry.dtype
		if entry.get("desc"):
			res["Description"] = entry.desc
		results.append(res)
	return results

cs_sheets = cdict(
	entity=["https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryEntities.tsv", "https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/FreewareOutOfBoundsEntities.tsv", "https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/FreewareMisc.tsv"],
	bullet=["https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryBullets.tsv"],
	caret=["https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryCaretsEffects.tsv"],
	music=["https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryMusicOrganya.tsv"],
	variable=[periwinkle_variables_txt],
)
cs_sheets.all = list(itertools.chain(*cs_sheets.values()))
sheet_searchables = {
	"Id (dec)": False,
	"Id (hex)": False,
	"Id (dec) (CMU)": False,
	"Entity Name": True,
	"Function Address": False,
	"Type": False,
	"PXE/TSC Compatibility": False,
	"Other notes": True,
	"Function Pointer": False,
	"Default Sprite Sheet": False,
	"Bullet Name": True,
	"Effect Name": True,
	"Official Music Name": True,
	"Internal Name": True,
	"Memory Offset": False,
	"Description": True,
	"Address": False,
	"Size (bytes)": False,
	"Data Type": False,
}
sheet_keys = {
	"Entity Name",
	"Bullet Name",
	"Effect Name",
	"Official Music Name",
	"Address",
}
cs_cache = AutoCache(f"{CACHE_PATH}/cs", stale=43200, timeout=86400 * 7)

async def sheet_pull(*urls, delimiter="\t"):
	data = []
	for url in urls:
		if callable(url):
			sheet = await url()
			data.extend(sheet)
			continue
		assert isinstance(url, str) and is_url(url)
		b = await attachment_cache.download(url, force=True)
		reader = csv.reader(as_str(b).splitlines(), delimiter=delimiter)
		sheet = list(reader)
		header = [h.removeprefix("Default ") for h in sheet.pop(0)]
		for row in sheet:
			entry = cdict()
			for col, val in zip(header, row):
				if not val:
					continue
				entry[col] = val
			data.append(entry)
	return data

async def sheet_structure(sheet_id):
	data = await sheet_pull(*cs_sheets[sheet_id])
	sheet = cdict(
		data=data,
		ids={},
		fcids={},
		searchables=[],
	)
	for i, entry in enumerate(data):
		for k, v in entry.items():
			if not v or v == "-":
				continue
			if sheet_searchables.get(k):
				sheet.searchables.append((v, i))
			else:
				sheet.ids.setdefault(v, []).append(i)
				sheet.fcids.setdefault(full_prune(v), []).append(i)
	return sheet

async def sheet_search(sheet_id, query, n=25):
	result_ids = alist()
	sheet = await cs_cache.aretrieve(sheet_id, sheet_structure, sheet_id)
	result_ids.extend(sheet.ids.get(query, ()))
	result_ids.extend(sheet.fcids.get(query, ()))
	result_ids.dedup(sort=False)
	if len(result_ids) < n:
		other_results = [(string_similarity(query, k), i, v) for i, (k, v) in enumerate(sheet.searchables)]
		other_ids = [k for k in sorted(other_results, reverse=True) if k[0] >= 0.15]
		result_ids.extend(t[-1] for t in other_ids)
	if not result_ids:
		result_ids = alist([str_lookup(sheet.searchables, query, key=lambda t: t[0], fuzzy=0)[1]])
	result_ids.dedup(sort=False)
	embed = discord.Embed(colour=colour2raw(0, 127, 255), title=f'Search results for {json.dumps(query)}:')
	for entry in map(sheet.data.__getitem__, result_ids[:n]):
		name = []
		for k in sheet_keys:
			v = entry.pop(k, None)
			if v:
				name.append(v)
		if "Id (dec)" in entry or "Id (dec) (CMU)" in entry:
			try:
				dec = int(entry.pop("Id (dec)", 0) or entry.pop("Id (dec) (CMU)", 0))
			except ValueError:
				dec = 0
			entry.pop("Id (hex)", None)
			entry = cdict(
				ID=f"{dec} (0x{hex(dec)[2:].upper()})",
				**entry,
			)
		for k in ("Function Address", "Function Pointer", "Memory Offset", "Description"):
			if k in entry:
				entry[k] = entry.pop(k)
		if "Other notes" in entry:
			entry["Other Notes"] = entry.pop("Other notes")
		name = " ".join(name) or "N/A"
		value = ini_md(iter2str(entry).strip())
		embed.add_field(name=name, value=value, inline=len(result_ids) > 5)
		if len(embed) > 6000:
			embed.remove_field(index=-1)
			break
	return embed


class CSsearch(Command):
	description = "Searches the Cave Story NPC list for an NPC/bullet/effect/music/variable by ID, name or description."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("entity", "bullet", "effect", "music", "variable", "all"),
			),
			default="all",
		),
		query=cdict(
			type="string",
			description="Search query",
			required=True,
		),
	)
	macros = cdict(
		CSnpc=cdict(
			mode="entity",
		),
		CSbul=cdict(
			mode="bullet",
		),
		CSeff=cdict(
			mode="effect",
		),
		CSmus=cdict(
			mode="music",
		),
		CSvar=cdict(
			mode="variable",
		),
	)
	rate_limit = 2

	async def __call__(self, bot, mode, query, **void):
		return cdict(
			embed=await sheet_search(mode, query, n=25),
		)


class Wav2Png(Command):
	name = ["Image2Audio", "Audio2Image", "Webp2Flac", "Flac2Webp", "Png2Wav"]
	_timeout_ = 15
	description = "Runs wav2png on the input URL. See https://github.com/thomas-xin/audioptic for more info, or to run it yourself!"
	schema = cdict(
		url=cdict(
			type="media",
			description="Image, animation, video or audio, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		fmt=cdict(
			type="enum",
			validation=cdict(
				enum=("ogg", "opus", "mp3", "aac", "flac", "wav", "webp", "png"),
			),
			description="The file format or codec of the output",
		),
	)
	rate_limit = (15, 25)

	async def __call__(self, bot, url, fmt, **void):
		fn = url2fn(url)
		ext = url2ext(url)
		was_image = ext in IMAGE_FORMS
		if not fmt:
			fmt = "opus" if was_image else "webp"
		dest = temporary_file(fmt)
		args = ["audioptic", url, dest]
		print(args)
		proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.DEVNULL)
		try:
			async with asyncio.timeout(3200):
				await proc.wait()
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		name = replace_ext(fn, fmt)
		return cdict(file=CompatFile(dest, filename=name))


class SpectralPulse(Command):
	_timeout_ = 150
	description = "Runs SpectralPulse on the input URL. Operates on a global queue system. See https://github.com/thomas-xin/SpectralPulse for more info, or to run it yourself!"
	usage = "<0:search_links>"
	example = ("spectralpulse https://www.youtube.com/watch?v=IgOci6JXPIc", "spectralpulse -fps 60 https://www.youtube.com/watch?v=kJQP7kiw5Fk")
	rate_limit = (120, 180)
	typing = True
	spec_sem = Semaphore(1, 256, rate_limit=1)
	maintenance = True

	async def __call__(self, bot, channel, message, args, **void):
		for a in message.attachments:
			args.insert(0, a.url)
		if not args:
			raise ArgumentError("Input string is empty.")
		urls = await bot.follow_url(args.pop(0), allow=True, images=False)
		if not urls or not urls[0]:
			raise ArgumentError("Please input a valid URL.")
		url = urls[0]
		kwargs = {
			"-size": "[1280,720]",
			"-fps": "30",
			"-sample_rate": "48000",
			"-amplitude": "0.1",
			"-smudge_ratio": "0.875",
			"-speed": "2",
			"-lower_bound": "A0",
			"-higher_bound": "F#10",
			"-particles": "piano",
			"-skip": "true",
			"-display": "false",
			"-render": "true",
			"-play": "false",
			"-image": "true",
		}
		i = 0
		while i < len(args):
			arg = args[i]
			if arg.startswith("-"):
				if arg in {
					"-size", "-fps", "-sample_rate", "-amplitude",
					"-smudge_ratio", "-speed", "-lower_bound", "-higher_bound",
					"-particles", "-skip", "-display", "-render", "-play", "-image",
					"-dest",
					"-width", "-height",
				}:
					kwargs[arg] = args[i + 1].replace("\xad", "#")
					i += 1
			i += 1
		if "-width" in kwargs and "-height" in kwargs:
			kwargs["-size"] = f'({kwargs["-width"]},{kwargs["-height"]})'
		name = kwargs.get("-dest") or url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
		n1 = name + ".mp4"
		n2 = name + ".png"
		dest = temporary_file()
		fn1 = dest + ".mp4"
		fn2 = dest + ".png"
		args = [
			python, "misc/spectralpulse/main.py",
			*itertools.chain.from_iterable(kwargs.items()),
			"-dest", dest, url,
		]
		async with discord.context_managers.Typing(channel):
			if self.spec_sem.is_busy() and not getattr(message, "simulated", False):
				await send_with_react(channel, italics(ini_md(f"SpectralPulse: {sqr_md(url)} enqueued in position {sqr_md(self.spec_sem.passive + 1)}.")), reacts="❎", reference=message)
			async with self.spec_sem:
				print(args)
				proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd(), stdout=subprocess.DEVNULL)
				with suppress():
					message.__dict__.setdefault("inits", []).append(proc)
				try:
					async with asyncio.timeout(3200):
						await proc.wait()
				except (T0, T1, T2):
					with tracebacksuppressor:
						force_kill(proc)
					raise
				for ext in ("pcm", "riff"):
					await asubmit(os.remove, f"{dest}.{ext}")
		await bot.send_with_file(channel, "", fn1, filename=n1, reference=message)
		if kwargs.get("-image") in ("true", "True"):
			await bot.send_with_file(channel, "", fn2, filename=n2, reference=message)


class UMP(Command):
	description = "lol"
	schema = cdict()
	rate_limit = (0, 1)
	ephemeral = True

	def __call__(self, _channel, _message, **void):
		self.bot.send_as_embeds(_channel, image="https://mizabot.xyz/u/7KjPmqgCGJ_x2xI5OB__BwAGzRt0/image.png", reference=_message)


class BTD6Paragon(Command):
	name = ["Paragon", "GenerateParagon"]
	description = "Given a tower and provided parameters, generates a list of Bloons TD 6 optimised paragon sacrifices. Parameters are \"p\" for pops, \"g\" for cash generated, \"t\" for Geraldo totems, and \"l\" for additional tower limit."
	usage = "<tower> <sacrifices>* <parameters>*"
	example = ("paragon dartmonkey 520 050 2*025", "paragon boat 400000p 30l", "paragon monkey_ace 2x500 2x050 2x005")
	rate_limit = (4, 5)

	def __call__(self, args, **void):
		from misc import paragon_calc
		if not args:
			raise ArgumentError("Input string is empty.")
		return "\xad" + paragon_calc.parse(args)


class SkyShardReminder(Command):
	name = ["SkyShard", "SkyShards"]
	description = "Tracks and sends DM reminders for Shard Eruptions in the game Sky: Children of the Light. Referenced code from https://github.com/PlutoyDev/sky-shards. When active, reminders will be sent 12 hours prior to the first landing, and additionally 1 hour, 5 minutes, and at the start of all landings."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("none", "black", "red", "all"),
			),
			description="Whether to disable reminders, or to target a particular type of shard",
			default="all",
		),
		ping=cdict(
			type="bool",
			description="Whether to ping at all intervals by default",
			default=True,
		),
	)

	def __call__(self, bot, _user, mode, ping, **void):
		match mode:
			case "none":
				bot.data.skyshardreminders.pop(_user.id, None)
				return fix_md("Successfully unsubscribed from Sky Shard Reminders.")
			case "black":
				bot.data.skyshardreminders[_user.id] = cdict(subscription=1, ping=ping, reminded={})
				return fix_md("Successfully subscribed to Black Shards.")
			case "red":
				bot.data.skyshardreminders[_user.id] = cdict(subscription=2, ping=ping, reminded={})
				return fix_md("Successfully subscribed to Red Shards.")
			case "all":
				bot.data.skyshardreminders[_user.id] = cdict(subscription=3, ping=ping, reminded={})
				return fix_md("Successfully subscribed to all Shards.")


class UpdateSkyShardReminders(Database):
	name = "skyshardreminders"

	def parse_pings(self, text):
		nums = text.split(":", 1)[-1].strip()
		if nums == "none":
			return []
		return list(map(int, nums.split(",")))

	async def _reaction_add_(self, message, react, user, **void):
		if user.id not in self or message.channel.id != (await self.bot.get_dm(user)).id:
			return
		if not message.embeds:
			return
		if (r := str(react)) != "✅" and r not in number_emojis:
			print(r)
			return
		default = [1, 2, 3]
		embed = message.embeds[0]
		pinged_occurrences = self.parse_pings(embed.footer.text) if embed.footer.text else default
		try:
			n = number_emojis.index(r)
		except ValueError:
			pinged_occurrences = default if not pinged_occurrences else []
		else:
			try:
				pinged_occurrences.remove(n)
			except ValueError:
				pinged_occurrences.append(n)
				pinged_occurrences.sort()
		embed.set_footer(text=f"Pings for landings: {', '.join(map(str, pinged_occurrences)) or 'none'}", icon_url="https://cdn.discordapp.com/emojis/695800620682313740.webp" if pinged_occurrences else "https://cdn.discordapp.com/emojis/695800875150475294.webp")
		await message.edit(embed=embed)

	_reaction_remove_ = _reaction_add_

	async def __call__(self, **void):
		bot = self.bot
		t = utc()
		taken_shards = self.get(0, {})
		ct = DynamicDT.parse("now pacific")
		pt = DynamicDT.parse("today pacific")
		nt = pt + TimeDelta(hours=24)
		s1 = shard.find_next_shard(pt)
		s2 = shard.find_next_shard(nt)
		if s1.occurrences[0].start == s2.occurrences[0].start:
			# s1 = shard.find_next_shard(pt - datetime.timedelta(days=1))
			# if s1.occurrences[0].start == s2.occurrences[0].start:
			s1 = None
		shards = (s1, s2) if s1 is not None else [s2]

		def format_landing(o):
			land = DynamicDT.fromdatetime(o.land)
			end = DynamicDT.fromdatetime(o.end)
			return f"{land.as_rel_discord()} ~ {end.as_rel_discord()}"

		current_shard = None
		for n, s in enumerate(shards):
			shard_hash = int(s.occurrences[0].start.timestamp())
			taken = taken_shards.get(shard_hash, 0)
			ping2 = True
			occurrence_number = 0
			all_occurrences = list(n + 1 for n in range(len(s.occurrences)))
			try:
				for i, o in reversed(tuple(enumerate(s.occurrences))):
					ts = o.land.timestamp()
					reminders = [ts - 3600, ts - 300, ts]
					if not i:
						reminders.insert(0, ts - 43200)
					reminders.append(o.end.timestamp() - 900)
					reminders.append(o.end.timestamp() + 1)
					for r in reversed(reminders):
						if r > taken and t >= r:
							taken = taken_shards[shard_hash] = r
							self[0] = taken_shards
							if r == reminders[-1]:
								ping2 = False
							occurrence_number = i + 1
							raise StopIteration
			except StopIteration:
				current_shard = s
			else:
				if n == len(shards) - 1 and current_shard is None and (len(shards) == 1 or shards[0].occurrences[-1].end < ct):
					r = s.occurrences[0].land.timestamp() - 43200
					if r > taken:
						taken = taken_shards[shard_hash] = r
						self[0] = taken_shards
						occurrence_number = 1
					else:
						continue
				else:
					continue
			print("SkyShard:", occurrence_number, s)
			embed = discord.Embed().set_image(url=f"https://github.com/PlutoyDev/sky-shards/raw/refs/heads/production/public/infographics/data_gale/{s.map}.webp").set_thumbnail(url=f"https://github.com/PlutoyDev/sky-shards/raw/refs/heads/production/public/infographics/map_clement/{s.map}.webp")
			url = f"https://sky-shards.pages.dev/en/{s.date.year}/{'%02d' % s.date.month}/{'%02d' % s.date.day}"
			if s.is_red:
				shard_bits = 2
				embed.colour = discord.Colour(16711680)
				embed.set_author(name="Red Shard", url=url, icon_url="https://raw.githubusercontent.com/PlutoyDev/sky-shards/refs/heads/production/public/emojis/ShardRed.webp")
				emoji = await bot.data.emojis.grab("ascended_candle.png")
				reward = f"{s.reward_ac} ×{emoji}"
			else:
				shard_bits = 1
				embed.colour = discord.Colour(1)
				embed.set_author(name="Black Shard", url=url, icon_url="https://raw.githubusercontent.com/PlutoyDev/sky-shards/refs/heads/production/public/emojis/ShardBlack.webp")
				emoji = await bot.data.emojis.grab("piece_of_light.png")
				reward = f"200 ×{emoji}"
			if any(o.land < ct < o.end - datetime.timedelta(seconds=900) for o in s.occurrences):
				timing = "Active"
			elif any(o.land < ct < o.end for o in s.occurrences):
				timing = "Last Call"
			else:
				timing = next((DynamicDT.fromdatetime(o.land).as_rel_discord() for o in s.occurrences if ct < o.land), "Expired")
			location = " -> ".join(w.capitalize() for w in s.map.split("."))
			landings = "\n".join(f"- **{format_landing(o)}**" if o.land <= ct < o.end else f"- ~~{format_landing(o)}~~" if o.end <= ct else f"- {format_landing(o)}" for o in s.occurrences)
			embed.add_field(name="Status", value=f"**{timing}**", inline=True)
			embed.add_field(name="Location", value=location, inline=True)
			embed.add_field(name="Reward", value=reward, inline=True)
			embed.add_field(name="Landings", value=landings, inline=False)

			async def notify_user(k, v):
				try:
					user = await bot.fetch_user(k)
				except Exception as ex:
					print_exc()
					if isinstance(ex, (LookupError, discord.NotFound)):
						self.pop(k)
					return
				message = None
				pinged_occurrences = all_occurrences.copy() if v.get("ping", True) else []
				ping = ping2
				try:
					m_id = v.reminded[shard_hash]
				except KeyError:
					pass
				else:
					try:
						message = await bot.fetch_message(m_id, user)
						message = await bot.ensure_reactions(message)
					except Exception:
						print_exc()
						return
					else:
						if message.embeds and message.embeds[0].footer.text and occurrence_number not in (pinged_occurrences := self.parse_pings(message.embeds[0].footer.text)):
							ping = False
						if ping:
							csubmit(bot.autodelete(message))
				embed.set_footer(text=f"Pings for landings: {', '.join(map(str, pinged_occurrences)) or 'none'}", icon_url="https://cdn.discordapp.com/emojis/695800620682313740.webp" if pinged_occurrences else "https://cdn.discordapp.com/emojis/695800875150475294.webp")
				if ping or not message:
					message = await send_with_react(user, embed=embed, reacts=["✅"] + [number_emojis[n] for n in all_occurrences])
					print(f"SkyShard {occurrence_number}: Pinged @", user)
				else:
					await message.edit(embed=embed)
					print(f"SkyShard {occurrence_number}: Edited @", user)
				v.reminded[shard_hash] = message.id
				self[k] = v

			futs = []
			for k, v in tuple(self.items()):
				if not k:
					continue
				if not v.subscription & shard_bits:
					continue
				fut = csubmit(notify_user(k, v))
				futs.append(fut)
			await gather(*futs)