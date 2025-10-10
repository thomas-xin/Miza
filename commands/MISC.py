# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

import csv, knackpy
from misc import shard
from prettytable import PrettyTable as ptable
from tsc_utils.flags import address_to_flag, flag_to_address
from tsc_utils.numbers import tsc_value_to_num, num_to_tsc_value


class DouClub:

	def __init__(self, c_id, c_sec):
		self.id = c_id
		self.secret = c_sec
		self.time = utc()
		self.knack = knackpy.App(app_id=self.id, api_key=self.secret)
		esubmit(self.pull)

	@tracebacksuppressor
	def pull(self):
		# print("Pulling Doukutsu Club...")
		self.data = self.knack.get("object_1")
		self.time = utc()

	def update(self):
		if utc() - self.time > 720:
			esubmit(self.pull, timeout=60)
			self.time = utc()

	def search(self, query):
		# This string search algorithm could be better
		output = []
		query = query.casefold()
		qlist = set(query.split())
		for res in self.data:
			author = res["Author"][0]["identifier"]
			name = res["Title"]
			description = res["Description"]
			s = (name, description, author)
			if not all(any(q in v for v in s) for q in qlist):
				continue
			url = f"https://doukutsuclub.knack.com/database#search-database/mod-details/{res['id']}"
			output.append({
				"author": author,
				"name": name,
				"description": description,
				"url": url,
			})
		return output

douclub = None


async def searchForums(query):
	url = f"https://forum.cavestory.org/search/320966/?q={url_parse(query)}"
	s = await Request(url, aio=True, timeout=16, ssl=False, decode=True)
	output = []
	i = 0
	while i < len(s):
		# HTML is a mess
		try:
			search = '<li class="block-row block-row--separated  js-inlineModContainer" data-author="'
			s = s[s.index(search) + len(search):]
		except ValueError:
			break
		j = s.index('">')
		curr = {"author": s[:j]}
		s = s[s.index('<h3 class="contentRow-title">'):]
		search = '<a href="/'
		s = s[s.index(search) + len(search):]
		j = s.index('">')
		curr["url"] = 'https://www.cavestory.org/forums/' + s[:j].lstrip("/")
		s = s[j + 2:]
		j = s.index('</a>')
		curr["name"] = s[:j]
		search = '<div class="contentRow-snippet">'
		s = s[s.index(search) + len(search):]
		j = s.index('</div>')
		curr["description"] = s[:j]
		for elem in curr:
			temp = curr[elem].replace('<em class="textHighlight">', "**").replace("</em>", "**")
			temp = html_decode(temp)
			curr[elem] = temp
		output.append(curr)
	return output


class SheetPull:

	def __init__(self, *urls, mode="csv"):
		self.mode = mode
		self.urls = urls
		self.data = diskcache.Cache(directory=f"{CACHE_PATH}/cs", expiry=720)

	async def pull(self):
		data = {}
		for url in self.urls:
			s = await Request(url, aio=True, timeout=16, decode=True)
			reader = csv.reader(s.splitlines(), delimiter="\t" if self.mode == "tsv" else ",")
			data[url] = list(reader)[1:] # Skip header
		return data

	async def search(self, query, max_results=20):
		output = []
		data = await retrieve_from(self.data, None, self.pull)
		for file in data.values():
			for line in file:
				if not line:
					continue
				if query in line:
					output.append([inf, line])
				elif query.casefold() in " ".join(line).casefold():
					output.append([100, line])
				else:
					try:
						score = max(string_similarity(query, x) for x in line if x and not x.isnumeric())
					except ValueError:
						score = 0
					output.append([score, line])
		output.sort(reverse=True)
		return [x[1] for x in output[:max_results]]


# URLs of Google Sheets .csv download links
entity_list = SheetPull(
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryEntities.tsv",
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/FreewareOutOfBoundsEntities.tsv",
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/FreewareMisc.tsv",
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryBullets.tsv",
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryCaretsEffects.tsv",
	"https://github.com/CaveStoryModdingCommunity/CaveStoryLists/raw/main/CaveStoryMusicOrganya.tsv",
	mode="tsv",
)


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


class CS_npc(Command):
	time_consuming = True
	name = ["npc", "cs_bul", "cs_sfx", "cs_mus"]
	description = "Searches the Cave Story NPC list for an NPC/bullet/effect/music by name or ID."
	usage = "<query> <condensed(-c)>?"
	example = ("cs_npc misery",)
	flags = "c"
	rate_limit = 2

	async def __call__(self, bot, args, flags, **void):
		lim = ("c" not in flags) * 40 + 20
		argv = " ".join(args)
		data = await asubmit(entity_list.search, argv, lim, timeout=8)
		# Sends multiple messages up to 20000 characters total
		if len(data):
			head = entity_list.data[0][1]
			for i in range(len(head)):
				if head[i] == "":
					head[i] = i * " "
			table = ptable(head)
			for line in data:
				table.add_row(line)
			output = str(table)
			if len(output) < 20000 and len(output) > 1900:
				response = [f"Search results for `{argv}`:"]
				lines = output.splitlines()
				curr = "```\n"
				for line in lines:
					if len(curr) + len(line) > 1900:
						response.append(curr + "```")
						curr = "```\n"
					if len(line):
						curr += line + "\n"
				response.append(curr + "```")
				return response
			return f"Search results for `{argv}`:\n{code_md(output)}"
		raise LookupError(f"No results for {argv}.")


# class CS_flag(Command):
	# name = ["OOB", "CS_OOB", "CS_flags"]
	# description = "Searches the Cave Story OOB flags list for a memory variable."
	# usage = "<query> <condensed(-c)>?"
	# example = ("cs_oob key",)
	# flags = "c"
	# # rate_limit = 2

	# async def __call__(self, args, flags, **void):
		# lim = ("c" not in flags) * 40 + 20
		# argv = " ".join(args)
		# data = await asubmit(tsc_list.search, argv, lim, timeout=8)
		# Sends multiple messages up to 20000 characters total
		# if len(data):
			# head = tsc_list.data[0][0]
			# for i in range(len(head)):
				# if head[i] == "":
					# head[i] = i * " "
			# table = ptable(head)
			# for line in data:
				# table.add_row(line)
			# output = str(table)
			# if len(output) < 20000 and len(output) > 1900:
				# response = [f"Search results for `{argv}`:"]
				# lines = output.splitlines()
				# curr = "```\n"
				# for line in lines:
					# if len(curr) + len(line) > 1900:
						# response.append(curr + "```")
						# curr = "```\n"
					# if len(line):
						# curr += line + "\n"
				# response.append(curr + "```")
				# return response
			# return f"Search results for `{argv}`:\n{code_md(output)}"
		# raise LookupError(f"No results for {argv}.")


class CS_mod(Command):
	time_consuming = True
	name = ["CS_search"]
	description = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
	usage = "<query>"
	example = ("cs_mod critter",)
	rate_limit = (3, 7)

	async def __call__(self, channel, user, args, **void):
		argv = " ".join(args)
		fut = asubmit(douclub.search, argv, timeout=8)
		try:
			data = await searchForums(argv)
		except ConnectionError as ex:
			if ex.errno != 404:
				raise
			data = []
		data += await fut
		if not data:
			raise LookupError(f"No results for {argv}.")
		description = f"Search results for `{argv}`:\n"
		fields = deque()
		for res in data:
			fields.append(dict(
				name=res["name"],
				value=res["url"] + "\n" + lim_str(res["description"], 128).replace("\n", " ") + f"\n> {res['author']}",
				inline=False,
			))
		self.bot.send_as_embeds(channel, description=description, fields=fields, author=get_author(user))


class CS_Database(Database):
	name = "cs_database"
	no_file = True


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
		ext = fn.rsplit(".", 1)[-1]
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

	async def _reaction_add_(self, message, react, user):
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
							csubmit(bot.silent_delete(message))
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


def load_douclub():
	global douclub
	while True:
		try:
			douclub = DouClub(AUTH["knack_id"], AUTH["knack_secret"])
		except KeyError:
			break
		except:
			print_exc()
			time.sleep(30)
			continue
		break

douclub = cdict(
	search=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to search Doukutsu Club.")'),
	update=lambda: None,
	pull=lambda: None,
)
esubmit(load_douclub, priority=True)