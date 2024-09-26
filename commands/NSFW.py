# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

import nekos, rule34, pybooru

# This entire module is a mess, I honestly don't care for it much
e_loop = asyncio.new_event_loop()
asyncio.set_event_loop(e_loop)
rule34_sync = rule34.Sync()
booruSites = list(pybooru.resources.SITE_LIST.keys())
LOG = False


def pull_e621(argv, delay=5):
	try:
		v1, v2 = 1, 1
		items = argv.replace(" ", "%20").casefold()
		baseurl = "https://e621.net/posts?tags="
		url = baseurl + items
		s = Request(url, decode=True)
		with suppress(ValueError):
			i = s.index("</a></li><li class='arrow'><a rel=\"next\"")
			sx = s[:i]
			ri = sx.rindex(">")
			pages = int(sx[ri + 1:])
			v1 = xrand(1, pages + 1)
			s = Request(f"{url}&page={v1}", decode=True)

		with suppress(ValueError):
			limit = s.index('class="next_page" rel="next"')
			s = s[:limit]

		search = ' data-file-url="'
		sources = deque()
		with suppress(ValueError):
			while True:
				ind1 = s.index(search)
				s = s[ind1 + len(search):]
				ind2 = s.index('"')
				target = s[:ind2]
				with suppress(ValueError):
					sources.append(target)
		sources = list(sources)
		for _ in loop(8):
			v2 = xrand(len(sources))
			url = sources[v2]
			if is_image(url) is not None:
				break
		return [url, v1, v2 + 1]
	except:
		if LOG:
			print_exc()


def pull_booru(argv, delay=5):
	client = pybooru.Moebooru(choice(tuple(booruSites)))
	try:
		posts = client.post_list(tags=argv, random=True, limit=16)
		if not posts:
			raise EOFError
		i = xrand(len(posts))
		url = posts[i]["file_url"]
		return [url, 1, i + 1]
	except:
		if LOG:
			print_exc()


def pull_rule34_xxx(argv, delay=5):
	v1, v2 = 1, 1
	try:
		t = utc()
		while utc() - t < delay:
			with suppress(TimeoutError):
				sources = rule34_sync.getImages(
					tags=argv,
					fuzzy=True,
					randomPID=True,
					singlePage=True
				)
				break
		if sources:
			attempts = 0
			while attempts < 256:
				v2 = xrand(len(sources))
				url = sources[v2].file_url
				if is_image(url) is not None:
					break
				attempts += 1
			if attempts >= 256:
				raise TimeoutError
			v1 = 1
			return [url, v1, v2 + 1]
		else:
			raise EOFError
	except:
		if LOG:
			print_exc()


def pull_rule34_paheal(argv, delay=5):
	try:
		v1, v2 = 1, 1
		items = argv.casefold().split(" ")
		if not len(argv.replace(" ", "")):
			tagsearch = [chr(i + 65) for i in range(26)]
		else:
			tagsearch = []
			for i in items:
				if i[0] not in tagsearch:
					tagsearch.append(i[0])
		rx = xrand(len(tagsearch))
		baseurl = "https://rule34.paheal.net/tags/alphabetic?starts_with="
		url = baseurl + tagsearch[rx] + "&mincount=1"
		s = Request(url, decode=True)
		tags = s.split("href='/post/list/")[1:]
		valid = []
		for i in range(len(tags)):
			ind = tags[i].index("/")
			tags[i] = tags[i][:ind]
			t = tags[i].casefold()
			for a in items:
				if a in t or a[:-1] == t:
					valid.append(tags[i])
		rx = xrand(len(valid))
		items = valid[rx]

		baseurl = "https://rule34.paheal.net/post/list/"
		try:
			s = Request(baseurl + items + "/1", decode=True)
		except ConnectionError:
			s = Request(baseurl + items.upper() + "/1", decode=True)
		with suppress(ValueError):
			ind = s.index('">Last</a><br>')
			s = s[ind - 5:ind]
			v1 = xrand(1, int(s.rsplit("/", 1)[-1]))
			url = url[:-1] + str(v1)

			s = Request(url, decode=True)
		with suppress(ValueError):
			limit = s.index("class=''>Images</h3><div class='blockbody'>")
			s = s[limit:]
			limit = s.index("</div></div></section>")
			s = s[:limit]

		search = 'href="'
		sources = []
		with suppress(ValueError):
			while True:
				ind1 = s.index(search)
				s = s[ind1 + len(search):]
				ind2 = s.index('"')
				target = s[:ind2]
				if not "." in target:
					continue
				elif ".js" in target:
					continue
				found = is_image(target) is not None
				if target[0] == "h" and found:
					sources.append(target)
		v2 = xrand(len(sources))
		url = sources[v2]
		return [url, v1, v2 + 1]
	except:
		if LOG:
			print_exc()


async def searchRandomNSFW(argv, delay=10):
	funcs = [
		pull_booru,
		pull_rule34_paheal,
		pull_rule34_xxx,
		pull_e621,
	]
	data = [asubmit(f, argv, delay - 3) for f in funcs]
	out = deque()
	for fut in data:
		with tracebacksuppressor:
			temp = await fut
			if temp:
				out.append(temp)
	print(out)
	if not out:
		raise LookupError(f"No results for {argv}.")
	item = choice(out)
	return item


neko_tags = {
	"feet": True,
	"yuri": True,
	"trap": True,
	"futanari": True,
	"hololewd": True,
	"lewdkemo": True,
	"solog": True,
	"feetg": True,
	"cum": True,
	"erokemo": True,
	"les": True,
	"wallpaper": True,
	"lewdk": True,
	"ngif": True,
	"tickle": False,
	"lewd": True,
	"feed": False,
	"gecg": False,
	"eroyuri": True,
	"eron": True,
	"cum_jpg": True,
	"bj": True,
	"nsfw_neko_gif": True,
	"solo": True,
	"kemonomimi": True,
	"nsfw_avatar": True,
	"gasm": False,
	"poke": False,
	"anal": True,
	"slap": False,
	"hentai": True,
	"avatar": False,
	"erofeet": True,
	"holo": True,
	"keta": True,
	"blowjob": True,
	"pussy": True,
	"tits": True,
	"holoero": True,
	"lizard": False,
	"pussy_jpg": True,
	"pwankg": True,
	"classic": True,
	"kuni": True,
	"waifu": False,
	"pat": False,
	"8ball": False,
	"kiss": False,
	"femdom": True,
	"neko": False,
	"spank": True,
	"cuddle": False,
	"erok": True,
	"fox_girl": False,
	"boobs": True,
	"pussy": True,
	"random_hentai_gif": True,
	"hug": False,
	"ero": True,
	"smug": False,
	"goose": False,
	"baka": False,
	"cat": False,
	"dog": False,
	"gif": True,
	"meow": False,
	"woof": False,
	"404": False,
	"ass": True,
	"hmidriff": True,
	"hneko": True,
	"hkitsune": True,
	"kemonomimi": True,
	"kanna": False,
	"thigh": True,
	"gah": False,
	"coffee": False,
	"food": False,
	"paizuri": True,
	"tentacle": True,
	"yaoi": True,
}
nekobot_shared = {
	"hentai", "holo", "neko", "anal", "boobs",
}
nekobot_exclusive = {
	"ass", "hass", "hmidriff", "pgif", "4k", "hneko", "hkitsune", "kemonomimi", "hanal", "gonewild", "kanna",
	"thigh", "hthigh", "gah", "coffee", "food", "paizuri", "tentacle", "hboobs", "yaoi",
}
nekoslife_deprecated = {
	"meow", "lewd",
}


class Neko(Command):
	name = ["Nya"]
	description = "Pulls a random image from nekos.life and embeds it."
	usage = "<tags[neko]>? <verbose(-v)|random(-r)|list(-l)>?"
	example = ("neko poke", "neko lizard")
	flags = "lrv"
	rate_limit = (0.5, 3)
	threshold = 256
	moe_sem = Semaphore(1, 0, rate_limit=10)
	nekobot_sem = cdict()

	def img(self, tag=None):

		async def fetch(nekos, tag):
			if tag in (None, "neko"):
				tag = "neko"
				if not xrand(50) and not self.moe_sem.is_busy():
					async with self.moe_sem:
						resp = await Request(
							"https://nekos.moe/api/v1/images/search",
							data=json_dumps(dict(nsfw=False, limit=50, skip=xrand(10) * 50, sort="newest", artist="", uploader="")),
							headers={"Content-Type": "application/json"},
							method="POST",
							json=True,
							aio=True,
						)
					out = set("https://nekos.moe/image/" + e["id"] for e in resp["images"])
					if out:
						print("nekos.moe", len(out))
						return out
				if "neko" not in self.nekobot_sem:
					self.nekobot_sem.neko = Semaphore(56, 56, rate_limit=61, last=True)
				if xrand(2) and not self.nekobot_sem.neko.is_busy():
					nekobot_sem = self.nekobot_sem.neko
					async with nekobot_sem:
						data = await Request("https://nekobot.xyz/api/image?type=neko", aio=True, json=True)
					return data["message"]
			if (tag in nekobot_exclusive or tag in nekobot_shared and xrand(2)):
				if tag not in self.nekobot_sem:
					self.nekobot_sem[tag] = Semaphore(56, 56, rate_limit=61, last=True)
				nekobot_sem = self.nekobot_sem[tag]
				if tag in ("ass", "anal", "boobs", "thigh"):
					tag = "h" + tag
				with suppress(SemaphoreOverflowError):
					async with nekobot_sem:
						url = f"https://nekobot.xyz/api/image?type={tag}"
						# print(url, len(self.nekobot_sem.rate_bin))
						data = await Request(url, aio=True, json=True)
					return data["message"]
				return
			if tag in nekoslife_deprecated:
				data = await Request(f"https://nekos.life/api/v2/img/{tag}", aio=True, json=True)
				return data["url"]
			return await asubmit(nekos.img, tag)

		file = f"neko~{tag}" if tag else "neko"
		return self.bot.data.imagepools.get(file, fetch, self.threshold, args=(nekos, tag))

	async def __call__(self, bot, message, channel, flags, args, argv, **void):
		isNSFW = bot.is_nsfw(channel)
		if "l" in flags or argv == "list":
			text = "Available tags in **" + channel.name + "**:\n```ini\n"
			available = [k for k, v in neko_tags.items() if not v or isNSFW]
			text += str(sorted(available)) + "```"
			return text
		guild = getattr(channel, "guild", None)
		tagNSFW = False
		selected = []
		for tag in args:
			tag = tag.replace(",", "").casefold()
			if tag in neko_tags:
				if neko_tags.get(tag, 0) == True:
					tagNSFW = True
					if not isNSFW:
						if hasattr(channel, "recipient"):
							raise PermissionError(f"This tag is only available in {uni_str('NSFW')} channels. Please verify your age using ~verify within a NSFW channel to enable NSFW in DMs.")
						raise PermissionError(f"This tag is only available in {uni_str('NSFW')} channels.")
				selected.append(tag)
		if "r" in flags:
			if guild and guild.owner_id == bot.id and isNSFW and not xrand(100):
				selected = {}
			else:
				for _ in loop(flags["r"]):
					possible = [k for k, v in neko_tags.items() if not v or isNSFW]
					selected.append(choice(possible))
		if not selected:
			if isinstance(selected, dict):
				m = await bot.fetch_message(867429880596791327, bot.get_channel(849651458495610910))
				url = as_str(m.attachments[0].url)
			elif not argv:
				url = await self.img()
			else:
				raise LookupError(f"Search tag {argv} not found. Use {bot.get_prefix(guild)}neko list for list.")
		else:
			v = xrand(len(selected))
			get = selected[v]
			if get == "gif":
				if tagNSFW and xrand(2):
					url = await self.img("nsfw_neko_gif")
				else:
					url = await self.img("ngif")
			elif get == "cat":
				url = await self.img("meow")
			elif get == "dog":
				url = await self.img("woof")
			elif get == "404":
				url = "https://cdn.nekos.life/smallboobs/404.png"
			else:
				url = await self.img(get)
		if "v" in flags:
			return escape_roles(url)
		embed = discord.Embed(colour=await bot.get_colour(url))
		embed.set_image(url=url)
		await send_with_react(channel, embed=embed, reference=message, reacts="🔳")


class Lewd(Command):
	time_consuming = True
	_timeout_ = 2
	name = ["NSFW"]
	min_level = 1
	description = "Pulls a random image from a search on Rule34 and e621, and embeds it."
	usage = "<query> <verbose(-v)>?"
	example = ("lewd pokemon",)
	flags = "v"
	rate_limit = (1, 6)

	async def __call__(self, bot, args, flags, message, channel, **void):
		if not bot.is_nsfw(channel):
			if hasattr(channel, "recipient"):
				raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels. Please verify your age using ~verify within a NSFW channel to enable NSFW in DMs.")
			raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels.")
		objs = await searchRandomNSFW(" ".join(args), 12)
		url = verify_url(objs[0].strip().replace(" ", "%20"))
		if "v" in flags:
			text = (
				"Pulled from " + url
				+ "\nImage **__" + str(objs[2])
				+ "__** on page **__" + str(objs[1])
				+ "__**"
			)
			return escape_roles(text)
		embed = discord.Embed(colour=await self.bot.get_colour(url))
		embed.set_image(url=url)
		await send_with_react(channel, embed=embed, reference=message, reacts="🔳")


class Verify(Command):
	name = ["AgeVerify"]
	min_level = 0
	description = "Verifies your account age as 18+, allowing you to access NSFW-restricted commands within DM channels."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("enable", "disable", "view"),
			),
			description="Determines whether to enable, disable, or view verification status",
			example="enable",
			default="view",
		),
	)
	rate_limit = (1, 6)

	async def __call__(self, bot, _guild, _user, _nsfw, _name, mode, **void):
		following = bot.data.nsfw
		if not _nsfw:
			if _user.id not in following:
				raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels, or for users who have posted in at least one {uni_str('NSFW')} channel shared with {bot.name}.")
		curr = following.get(_user.id)
		if mode == "disable":
			following[_user.id] = False
			return italics(css_md(f"Disabled age-verified DMs for {sqr_md(_user)}."))
		elif mode == "enable":
			following[_user.id] = True
			return italics(css_md(f"Enabled age-verified DMs for {sqr_md(_user)}."))
		if not curr:
			return ini_md(f'Age-verified DMs are currently disabled for {sqr_md(_user)}. Use "{bot.get_prefix(_guild)}{_name} enable" to enable.')
		return ini_md(f"Age-verified DMs are currently enabled for {sqr_md(_user)}.")


class UpdateNSFW(Database):
	name = "nsfw"
	user = True

	def _send_(self, message, **void):
		channel = message.channel
		if is_nsfw(channel):
			user = message.author
			if user.id not in self:
				self[user.id] = False