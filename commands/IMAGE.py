print = PRINT

try:
	import yt_dlp as youtube_dl
except ModuleNotFoundError:
	import youtube_dl
import aiohttp, torch
import imagebot
from PIL import Image

getattr(youtube_dl, "__builtins__", {})["print"] = print


ydl_opts = {
	"quiet": 1,
	"format": "bestvideo/best",
	"nocheckcertificate": 1,
	"no_call_home": 1,
	"nooverwrites": 1,
	"noplaylist": 1,
	"logtostderr": 0,
	"ignoreerrors": 0,
	"default_search": "auto",
	"source_address": "0.0.0.0",
}
downloader = youtube_dl.YoutubeDL(ydl_opts)

def get_video(url, fps=None):
	try:
		entry = downloader.extract_info(url, download=False)
	except:
		print_exc()
		return url, None, None, None
	best = 0
	size = None
	dur = None
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ""
	for fmt in fmts:
		q = fmt.get("height", 0)
		if type(q) is not int:
			q = 0
		# Attempt to get as close to 720p as possible for download
		if abs(q - 720) < abs(best - 720):
			best = q
			url = fmt["url"]
			size = [fmt["width"], fmt["height"]]
			dur = fmt.get("duration", entry.get("duration"))
			fps = fmt.get("fps", entry.get("fps"))
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	return url, size, dur, fps

VIDEOS = ("gif", "webp", "apng", "mp4", "mkv", "webm", "mov", "wmv", "flv", "avi", "qt", "f4v", "zip")


# class IMG(Command):
#     min_display = "0~2"
#     description = "Sends an image in the current chat from a list."
#     usage = "(add|delete)? <0:tags>* <1:url>? <verbose{?v}|delete{?x}|hide{?h}>?"
#     example = ("img add how https://media.discordapp.net/attachments/500919580596764673/642515924578205696/HOW.gif", "img delete how")
#     flags = "vraedhzfx"
#     rate_limit = (4, 6)
#     no_parse = True
#     directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
#     dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
#     slash = True
# 
#     async def __call__(self, bot, flags, args, argv, user, message, channel, guild, perm, **void):
#         update = bot.data.images.update
#         imglists = bot.data.images
#         images = imglists.get(guild.id, {})
#         if "a" in flags or "e" in flags or "d" in flags:
#             if message.attachments:
#                 args.extend(best_url(a) for a in message.attachments)
#                 argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
#             req = 2
#             if perm < req:
#                 reason = "to change image list for " + guild.name
#                 raise self.perm_error(perm, req, reason)
#             if "a" in flags or "e" in flags:
#                 lim = 256 << bot.is_trusted(guild.id) * 2 + 1
#                 if len(images) > lim:
#                     raise OverflowError(f"Image list for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
#                 key = " ".join(args[:-1]).casefold()
#                 if len(key) > 2000:
#                     raise ArgumentError("Image tag too long.")
#                 elif not key:
#                     raise ArgumentError("Image tag must not be empty.")
#                 if is_url(args[0]):
#                     if len(args) > 1:
#                         args = (args[-1], args[0])
#                     else:
#                         args = (args[0].split("?", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[0], args[0])
#                 urls = await bot.follow_url(args[-1], best=True, allow=True, limit=1)
#                 url = urls[0]
#                 if len(url) > 2000:
#                     raise ArgumentError("Image url too long.")
#                 images[key] = url
#                 images = {i: images[i] for i in sorted(images)}
#                 imglists[guild.id] = images
#                 if not "h" in flags:
#                     return css_md(f"Successfully added {sqr_md(key)} to the image list for {sqr_md(guild)}.")
#             if not args:
#                 # This deletes all images for the current guild
#                 if "f" not in flags and len(images) > 1:
#                     raise InterruptedError(css_md(sqr_md(f"WARNING: {len(images)} IMAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True))
#                 imglists[guild.id] = {}
#                 return italics(css_md(f"Successfully removed all {sqr_md(len(images))} images from the image list for {sqr_md(guild)}."))
#             key = argv.casefold()
#             images.pop(key)
#             imglists[guild.id] = images
#             return italics(css_md(f"Successfully removed {sqr_md(key)} from the image list for {sqr_md(guild)}."))
#         if not argv and not "r" in flags:
#             # Set callback message for scrollable list
#             buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
#             await send_with_reply(
#                 None,
#                 message,
#                 "*```" + "\n" * ("z" in flags) + "callback-image-img-"
#                 + str(user.id) + "_0"
#                 + "-\nLoading Image database...```*",
#                 buttons=buttons,
#             )
#             return
#         sources = alist()
#         for tag in args:
#             t = tag.casefold()
#             if t in images:
#                 sources.append(images[t])
#         r = flags.get("r", 0)
#         for _ in loop(r):
#             sources.append(choice(images.values()))
#         if not len(sources):
#             raise LookupError(f"Target image {argv} not found. Use img for list.")
#         url = choice(sources)
#         if "x" in flags:
#             create_task(bot.silent_delete(message))
#         if "v" in flags:
#             msg = escape_roles(url)
#         else:
#             msg = None
#         url2 = await bot.get_proxy_url(message.author)
#         colour = await asubmit(bot.get_colour, message.author)
#         emb = discord.Embed(colour=colour, url=url).set_image(url=url)
#         await bot.send_as_webhook(channel, msg, embed=emb, username=message.author.display_name, avatar_url=url2)
# 
#     async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
#         u_id, pos = list(map(int, vals.split("_", 1)))
#         if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
#             return
#         if reaction not in self.directions and reaction is not None:
#             return
#         guild = message.guild
#         user = await bot.fetch_user(u_id)
#         imglists = bot.data.images
#         images = imglists.get(guild.id, {})
#         page = 12
#         last = max(0, len(images) - page)
#         if reaction is not None:
#             i = self.directions.index(reaction)
#             if i == 0:
#                 new = 0
#             elif i == 1:
#                 new = max(0, pos - page)
#             elif i == 2:
#                 new = min(last, pos + page)
#             elif i == 3:
#                 new = last
#             else:
#                 new = pos
#             pos = new
#         content = message.content
#         if not content:
#             content = message.embeds[0].description
#         i = content.index("callback")
#         content = "*```" + "\n" * ("\n" in content[:i]) + (
#             "callback-image-img-"
#             + str(u_id) + "_" + str(pos)
#             + "-\n"
#         )
#         if not images:
#             content += f"Image list for {str(guild).replace('`', '')} is currently empty.```*"
#             msg = ""
#         else:
#             content += f"{len(images)} image(s) currently assigned for {str(guild).replace('`', '')}:```*"
#             msg = ini_md(iter2str({k: "\n" + images[k] for k in tuple(images)[pos:pos + page]}))
#         colour = await self.bot.get_colour(guild)
#         emb = discord.Embed(
#             description=content + msg,
#             colour=colour,
#         )
#         emb.set_author(**get_author(user))
#         more = len(images) - pos - page
#         if more > 0:
#             emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
#         create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
#         if hasattr(message, "int_token"):
#             await bot.ignore_interaction(message)


async def get_image(bot, user, message, args, argv, default=2, raw=False, ext="webp", count=0):
	try:
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		if not args:
			raise ArgumentError
		url = args.pop(0)
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
	if args and args[-1] in VIDEOS:
		ext = args.pop(-1)
	extra = []
	while count > 0 and len(args) > 1:
		extra.append(args.pop(-1))
	value = " ".join(args).strip()
	if not value:
		value = default
	elif not raw:
		value = await bot.eval_math(value)
		if not abs(value) <= 256:
			raise OverflowError("Maximum multiplier input is 256.")
	# Try and find a good name for the output image
	try:
		name = url[url.rindex("/") + 1:]
		if not name:
			raise ValueError
		if "." in name:
			name = name[:name.rindex(".")]
	except ValueError:
		name = "unknown"
	if not name.endswith("." + ext):
		name += "." + ext
	return name, value, url, ext, extra


class ImageAdjust(Command):
	name = [
		"Saturation", "Saturate",
		"Contrast",
		"Brightness", "Brighten", "Lighten", "Lightness",
		"Luminance", "Luminosity",
		"Sharpness", "Sharpen",
		"HueShift", "Hue",
		"Blur", "Gaussian",
	]
	description = "Applies an adjustment filter to the supplied image."
	usage = "<0:url> <1:multiplier(2)>?"
	example = ("saturate https://mizabot.xyz/favicon", "hue https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 0.8")
	no_parse = True
	rate_limit = (5, 9)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, name, _timeout, **void):
		if name.startswith("hue"):
			default = 0.5
		elif name in ("blur", "gaussian"):
			default = 8
		else:
			default = 2
		name2, value, url, fmt, extra = await get_image(bot, user, message, args, argv, default=default)
		async with discord.context_managers.Typing(channel):
			if name.startswith("sat"):
				argi = ("Enhance", ["Color", value, "-f", fmt])
			elif name.startswith("con"):
				argi = ("Enhance", ["Contrast", value, "-f", fmt])
			elif name.startswith("bri") or name.startswith("lig"):
				argi = ("brightness", [value, "-f", fmt])
			elif name.startswith("lum"):
				argi = ("luminance", [value, "-f", fmt])
			elif name.startswith("sha"):
				argi = ("Enhance", ["Sharpness", value, "-f", fmt])
			elif name.startswith("hue"):
				argi = ("hue_shift", [value, "-f", fmt])
			elif name in ("blur", "gaussian"):
				argi = ("blur", ["gaussian", value, "-f", fmt])
			else:
				raise RuntimeError(name)
			resp = await process_image(url, *argi, timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name2, reference=message, reacts="🔳")


class ColourDeficiency(Command):
	name = ["ColorBlind", "ColourBlind", "ColorBlindness", "ColourBlindness", "ColorDeficiency"]
	alias = name + ["Protanopia", "Protanomaly", "Deuteranopia", "Deuteranomaly", "Tritanopia", "Tritanomaly", "Achromatopsia", "Achromatonomaly"]
	description = "Applies a colourblindness filter to the target image."
	usage = "<0:url> (protanopia|protanomaly|deuteranopia|deuteranomaly|tritanopia|tritanomaly|achromatopsia|achromatonomaly)? <1:ratio(0.9)>?"
	example = ("colourdeficiency tritanomaly https://mizabot.xyz/favicon", "colourblind protanopia https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png")
	no_parse = True
	rate_limit = (6, 10)
	_timeout_ = 3.5
	typing = True

	async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		try:
			if not args:
				raise ArgumentError
			url = args.pop(0)
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
		if "color" not in name and "colour" not in name:
			operation = name
		elif args:
			operation = args.pop(0).casefold()
		else:
			operation = "deuteranomaly"
		value = " ".join(args).strip()
		if not value:
			value = None
		else:
			value = await bot.eval_math(value)
			if not abs(value) <= 2:
				raise OverflowError("Maximum multiplier input is 2.")
		# Try and find a good name for the output image
		try:
			name = url[url.rindex("/") + 1:]
			if not name:
				raise ValueError
			if "." in name:
				name = name[:name.rindex(".")]
		except ValueError:
			name = "unknown"
		ext = "webp"
		if not name.endswith("." + ext):
			name += "." + ext
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "colour_deficiency", [operation, value], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


# class RemoveMatte(Command):
#     name = ["RemoveColor", "RemoveColour"]
#     description = "Removes a colour from the supplied image."
#     usage = "<0:url> <colour(#FFFFFF)>?"
#     no_parse = True
#     rate_limit = (4, 9)
#     _timeout_ = 4.5
#     typing = True

#     async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
#         # Take input from any attachments, or otherwise the message contents
#         if message.attachments:
#             args = [best_url(a) for a in message.attachments] + args
#             argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
#         try:
#             if not args:
#                 raise ArgumentError
#             url = args.pop(0)
#             urls = await bot.follow_url(url, best=True, allow=True, limit=1)
#             if not urls:
#                 urls = await bot.follow_to_image(argv)
#                 if not urls:
#                     urls = await bot.follow_to_image(url)
#                     if not urls:
#                         raise ArgumentError
#             url = urls[0]
#         except ArgumentError:
#             if not argv:
#                 url = None
#                 try:
#                     url = await bot.get_last_image(message.channel)
#                 except FileNotFoundError:
#                     raise ArgumentError("Please input an image by URL or attachment.")
#             else:
#                 raise ArgumentError("Please input an image by URL or attachment.")
#         colour = parse_colour(" ".join(args), default=(255,) * 3)
#         # Try and find a good name for the output image
#         try:
#             name = url[url.rindex("/") + 1:]
#             if not name:
#                 raise ValueError
#             if "." in name:
#                 name = name[:name.rindex(".")]
#         except ValueError:
#             name = "unknown"
#         ext = "png"
#         if not name.endswith("." + ext):
#             name += "." + ext
#         async with discord.context_managers.Typing(channel):
#             resp = await process_image(url, "remove_matte", [colour], timeout=_timeout)
#             fn = resp
#             if fn.endswith(".gif"):
#                 if not name.endswith(".gif"):
#                     if "." in name:
#                         name = name[:name.rindex(".")]
#                     name += ".gif"
#         await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Invert(Command):
	name = ["Negate"]
	description = "Inverts supplied image."
	usage = "<url>"
	example = ("invert https://mizabot.xyz/favicon",)
	no_parse = True
	rate_limit = (5, 7)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "invert", ["-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class GreyScale(Command):
	name = ["GrayScale"]
	description = "Greyscales supplied image."
	usage = "<url>"
	example = ("greyscale https://mizabot.xyz/favicon",)
	no_parse = True
	rate_limit = (5, 7)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "greyscale", ["-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Laplacian(Command):
	name = ["EdgeDetect", "Edges"]
	description = "Applies the Laplacian edge-detect algorithm to the image."
	usage = "<url>"
	example = ("laplacian https://mizabot.xyz/favicon",)
	no_parse = True
	rate_limit = (5, 7)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "laplacian", ["-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class ColourSpace(Command):
	name = ["ColorSpace"]
	description = "Changes the colour space of the supplied image."
	usage = "<0:url> <2:source(rgb)>? <1:dest(hsv)>?"
	example = ("colourspace https://mizabot.xyz/favicon rgb hsv", "colorspace https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png cmy hsi")
	no_parse = True
	rate_limit = (7, 11)
	_timeout_ = 4
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, raw=True, default="")
		spl = value.rsplit(None, 1)
		if not spl:
			source = "rgb"
			dest = "hsv"
		elif len(spl) == 1:
			source = "rgb"
			dest = spl[0].casefold()
		else:
			source, dest = (i.casefold() for i in spl)
		if source == dest:
			raise TypeError("Colour spaces must be different.")
		for i in (source, dest):
			if i not in ("rgb", "cmy", "xyz", "hsv", "hsl", "hsi", "hcl", "lab", "luv", "yiq", "yuv"):
				raise TypeError(f"Invalid colour space {i}.")
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "colourspace", [source, dest, "-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Magik(Command):
	name = ["Distort"]
	description = "Applies the Magik image filter to supplied image."
	usage = "<0:url> <cell_count(7)>?"
	example = ("magik https://mizabot.xyz/favicon", "magik https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 3")
	no_parse = True
	rate_limit = (8, 12)
	_timeout_ = 4
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, default=7)
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "magik", [value, "-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Colour(Command):
	name = ["RGB", "HSV", "HSL", "CMY", "LAB", "LUV", "XYZ", "Color"]
	description = "Creates a 128x128 image filled with the target colour."
	usage = "<colour>"
	example = ("colour bf7fff", "rgb (50, 150, 250)", "hsv 50 20 30", "color blue")
	no_parse = True
	rate_limit = (3, 5)
	flags = "v"
	trans = {
		"hsv": hsv_to_rgb,
		"hsl": hsl_to_rgb,
		"cmy": cmy_to_rgb,
		"lab": lab_to_rgb,
		"luv": luv_to_rgb,
		"xyz": xyz_to_rgb,
	}
	typing = True
	slash = True

	async def __call__(self, bot, user, message, channel, name, argv, **void):
		channels = parse_colour(argv)
		if name in self.trans:
			if name in "lab luv":
				adj = channels
			else:
				adj = [x / 255 for x in channels]
			channels = [round(x * 255) for x in self.trans[name](adj)]
		adj = [x / 255 for x in channels]
		# Any exceptions encountered during colour transformations will immediately terminate the command
		msg = ini_md(
			"HEX colour code: " + sqr_md(bytes(channels).hex().upper())
			+ "\nDEC colour code: " + sqr_md(colour2raw(channels))
			+ "\nRGB values: " + str(channels if type(channels) is list else list(channels))
			+ "\nHSV values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsv(adj)))
			+ "\nHSL values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsl(adj)))
			+ "\nCMY values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
			+ "\nLAB values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
			+ "\nLUV values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
			+ "\nXYZ values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
		)
		async with discord.context_managers.Typing(channel):
			resp = await process_image("from_colour", "$", [channels])
			fn = resp
			f = CompatFile(fn, filename="colour.png")
		await bot.send_with_file(channel, msg, f, filename=fn, best=True, reference=message)


class Gradient(Command):
	description = "Generates a gradient with a specific shape."
	usage = "(linear|radial|conical|spiral|polygon)? <0:count(1)>? <-1:colour(white)>?"
	example = ("gradient radial red", "gradient linear green")
	no_parse = True
	rate_limit = (4, 6)
	typing = True

	async def __call__(self, bot, user, message, channel, args, **void):
		if not args:
			shape = "linear"
		else:
			shape = args.pop(0)
		if shape not in "linear|radial|conical|spiral|polygon".split("|"):
			raise TypeError(f"Invalid gradient shape {args[0]}.")
		if args:
			colour = args.pop(-1)
			colour = parse_colour(colour)
		else:
			colour = (255,) * 3
		if args:
			count = await bot.eval_math(" ".join(args))
		else:
			count = 1
		async with discord.context_managers.Typing(channel):
			resp = await process_image("from_gradient", "$", [shape, count, colour])
			fn = resp
			f = CompatFile(fn, filename="gradient.png")
		await bot.send_with_file(channel, "", f, filename=fn, best=True, reference=message)


class Average(Command):
	name = ["AverageColour"]
	description = "Computes the average pixel colour in RGB for the supplied image."
	usage = "<url>"
	example = ("average https://mizabot.xyz/favicon",)
	no_parse = True
	rate_limit = (5, 7)
	_timeout_ = 2
	typing = True

	async def __call__(self, bot, channel, user, message, argv, args, **void):
		if message.attachments:
			args = [worst_url(a) for a in message.attachments] + args
			argv = " ".join(worst_url(a) for a in message.attachments) + " " * bool(argv) + argv
		try:
			if not args:
				raise ArgumentError
			url = args.pop(0)
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
		async with discord.context_managers.Typing(channel):
			colour = await bot.data.colours.get(url, threshold=False)
			channels = raw2colour(colour)
			adj = [x / 255 for x in channels]
			# Any exceptions encountered during colour transformations will immediately terminate the command
			msg = ini_md(
				"HEX colour code: " + sqr_md(bytes(channels).hex().upper())
				+ "\nDEC colour code: " + sqr_md(colour2raw(channels))
				+ "\nRGB values: " + str(channels)
				+ "\nHSV values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsv(adj)))
				+ "\nHSL values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsl(adj)))
				+ "\nCMY values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
				+ "\nLAB values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
				+ "\nLUV values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
				+ "\nXYZ values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
			)
			resp = await process_image("from_colour", "$", [channels])
			fn = resp
			f = CompatFile(fn, filename="average.png")
		await bot.send_with_file(channel, msg, f, filename=fn, best=True, reference=message)
		# return css_md("#" + bytes2hex(bytes(raw2colour(colour)), space=False))


class QR(Command):
	name = ["RainbowQR"]
	description = "Creates a QR code image from an input string, optionally adding a rainbow swirl effect."
	usage = "<string>"
	example = ("QR https://mizabot.xyz/favicon", "rainbow_qr you found the funny!")
	no_parse = True
	rate_limit = (8, 11)
	_timeout_ = 4
	typing = True

	async def __call__(self, bot, message, channel, argv, name, _timeout, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		async with discord.context_managers.Typing(channel):
			resp = await process_image("to_qr", "$", [argv, "rainbow" in name], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename="QR." + ("gif" if "rainbow" in name else "png"), reference=message)


class Rainbow(Command):
	name = ["RainbowGIF", "Gay"]
	description = "Creates a .gif image from repeatedly hueshifting supplied image."
	usage = "<0:url> <1:duration(2)>?"
	example = ("rainbow https://mizabot.xyz/favicon", "rainbow https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 6")
	no_parse = True
	rate_limit = (10, 13)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, ext="gif")
		async with discord.context_managers.Typing(channel):
			# -gif signals to image subprocess that the output is always a .gif image
			resp = await process_image(url, "rainbow_gif", [value, "-gif", "-f", fmt], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Scroll(Command):
	name = ["Parallax", "Offset", "ScrollGIF"]
	description = "Creates a .gif image from repeatedly shifting supplied image in a specified direction."
	usage = "<0:url> <1:direction(left)>? <2:duration(2)>? <3:fps(32)>?"
	example = ("scroll https://mizabot.xyz/favicon", "scroll https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png down 4")
	no_parse = True
	rate_limit = (10, 13)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		try:
			if message.attachments:
				args = [best_url(a) for a in message.attachments] + args
				argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
			if not args:
				raise ArgumentError
			url = args.pop(0)
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
		if args:
			direction = args.pop(0)
		else:
			direction = "LEFT"
		if args:
			duration = await bot.eval_math(args.pop(0))
		else:
			duration = 2
		if args:
			fps = await bot.eval_math(" ".join(args))
			fps = round(fps)
			if fps <= 0:
				raise ValueError("FPS value must be positive.")
			elif fps > 256:
				raise OverflowError("Maximum FPS value is 256.")
		else:
			fps = 32
		try:
			name = url[url.rindex("/") + 1:]
			if not name:
				raise ValueError
			if "." in name:
				name = name[:name.rindex(".")]
		except ValueError:
			name = "unknown"
		if not name.endswith(".gif"):
			name += ".gif"
		async with discord.context_managers.Typing(channel):
			# -gif signals to image subprocess that the output is always a .gif image
			resp = await process_image(url, "scroll_gif", [direction, duration, fps, "-gif"], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Spin(Command):
	name = ["SpinGIF"]
	description = "Creates a .gif image from repeatedly rotating supplied image."
	usage = "<0:url> <1:duration(2)>?"
	example = ("spin https://mizabot.xyz/favicon", "spin https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 3")
	no_parse = True
	rate_limit = (10, 13)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, ext="gif")
		async with discord.context_managers.Typing(channel):
			# -gif signals to image subprocess that the output is always a .gif image
			resp = await process_image(url, "spin_gif", [value, "-gif", "-f", fmt], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Orbit(Command):
	name = ["Orbital", "Orbitals"]
	description = "Renders a ring of orbiting sprites of the supplied image."
	usage = "<0:url>+ <1:orbital_count(5)>? <2:duration(2)>?"
	example = ("orbitals https://mizabot.xyz/favicon", "orbit https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 3 10")
	no_parse = True
	rate_limit = (16, 22)
	_timeout_ = 13
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, ext="gif", raw=True, default="")
		extras = deque()
		while value:
			spl = value.split(None, 1)
			urls = await bot.follow_url(spl[0], best=True, allow=True, limit=1)
			if not urls:
				break
			value = spl[-1] if len(spl) > 1 else ""
			extras.append(urls[0])
		# if extras:
		#     print(url, *extras)
		spl = value.rsplit(None, 1)
		if not spl:
			if not extras:
				count = 5
			else:
				count = len(extras) + 1
			duration = 2
		elif len(spl) == 1:
			if not extras:
				count = await bot.eval_math(spl[0])
				duration = 2
			else:
				count = len(extras) + 1
				duration = await bot.eval_math(spl[0])
		else:
			count = await bot.eval_math(spl[0])
			duration = await bot.eval_math(spl[1])
		if count > 256:
			raise OverflowError("Maximum multiplier input is 256.")
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "orbit_gif", [count, duration, list(extras), "-gif", "-f", fmt], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Pet(Command):
	name = ["PatGIF", "Pat", "PetGIF"]
	description = "Creates a .gif image from applying the Petpet generator to the supplied image."
	usage = "<0:url> <1:squish(0.1)>? <2:duration(0.25)>?"
	example = ("pet https://mizabot.xyz/favicon", "pet https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 3")
	no_parse = True
	rate_limit = (10, 13)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, ext="gif", raw=True, default="")
		extras = deque()
		while value:
			spl = value.split(None, 1)
			urls = await bot.follow_url(spl[0], best=True, allow=True, limit=1)
			if not urls:
				break
			value = spl[-1] if len(spl) > 1 else ""
			extras.append(urls[0])
		spl = value.rsplit(None, 1)
		if not spl:
			squish = 0.1
			duration = 0.25
		elif len(spl) == 1:
			squish = await bot.eval_math(spl[0])
			duration = 0.25
		else:
			squish = await bot.eval_math(spl[0])
			duration = await bot.eval_math(spl[1])
		if squish < -0.5 or squish > 0.5:
			raise OverflowError("Maximum multiplier input is 0.5.")
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "pet_gif", [squish, duration, "-gif", "-f", fmt], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class GMagik(Command):
	name = ["Liquefy", "MagikGIF"]
	description = "Repeatedly applies the Magik image filter to supplied image."
	usage = "<0:url> <cell_size(7)>? <iterations(64)>? <duration(2)>?"
	example = ("gmagik https://mizabot.xyz/favicon", "liquefy https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 36")
	no_parse = True
	rate_limit = (11, 14)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
		if name == "liquefy":
			default = 32
		else:
			default = 7
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, default=default, count=1, ext="gif")
		if name == "liquefy":
			arr = [abs(value), 2]
		else:
			arr = [abs(value), 1]
		if extra:
			arr.append(int(extra.pop(0)))
		if extra:
			arr.append(int(extra.pop(0)))
		arr.extend(("-gif", "-f", fmt))
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "magik_gif", arr, timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class CreateGIF(Command):
	name = ["Animate", "GIF", "Frames", "ImageSequence"]
	description = "Combines multiple supplied images, and/or optionally a video, into an animated image, image sequence, or video."
	usage = "<0:url>+ <-2:fps(20)>? <-1:format(gif)>?"
	example = ("gif https://www.youtube.com/watch?v=dQw4w9WgXcQ", "gif https://discord.com/assets/7c010dc6da25c012643ea22c1f002bb4.svg https://discord.com/assets/66f6c781fe86c346fbaf3390618668fc.svg https://discord.com/assets/626aaed496ac12bbdb68a86b46871a1f.svg -r 3")
	no_parse = True
	rate_limit = (12, 16)
	_timeout_ = 20
	flags = "r"
	typing = True

	async def __call__(self, bot, user, guild, channel, message, flags, name, args, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args += [best_url(a) for a in message.attachments]
		try:
			if not args:
				raise ArgumentError
		except ArgumentError:
			if not args:
				url = None
				try:
					url = await bot.get_last_image(message.channel)
				except FileNotFoundError:
					raise ArgumentError("Please input an image by URL or attachment.")
			else:
				raise ArgumentError("Please input an image by URL or attachment.")
		if name in ("frames", "imagesequence"):
			fmt = "zip"
		elif args[-1] in VIDEOS:
			fmt = args.pop(-1)
		else:
			fmt = "gif"
		if "r" in flags or args[-1].isnumeric():
			fr = args.pop(-1)
			rate = await bot.eval_math(fr)
		else:
			rate = None
		# Validate framerate values to prevent issues further down the line
		if rate and rate <= 0:
			args = args[:1]
			rate = 1
		delay = round(1000 / rate) if rate else None
		if delay and delay <= 0:
			args = args[-1:]
			delay = 1000
		elif delay and delay >= 16777216:
			raise OverflowError("GIF image framerate too low.")
		async with discord.context_managers.Typing(channel):
			found = []
			links = args.copy()
			while links:
				url = links.pop(0)
				if is_discord_url(url):
					urls = await bot.follow_url(url, best=True, allow=True, reactions=False, limit=None)
					found.extend(urls)
					continue
				urls = await bot.follow_url(url, best=True, allow=True, limit=1)
				url = urls[0]
				if "channels" not in url:
					with tracebacksuppressor:
						url, size, dur, fps = await asubmit(get_video, url, None, timeout=60)
						if size and dur and fps:
							# video = (url, size, dur, fps)
							delay = delay or 1000 / fps
				if not url:
					continue
					# raise ArgumentError(f'Invalid URL detected: "{urls[0]}".')
				found.append(url)
			filename = "unknown." + fmt
			# if video is None:
			#     video = args
			resp = await process_image("create_gif", "$", ["image", found, delay, "-f", fmt], cap="video", timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=filename, reference=message, reacts="🔳")


class Resize(Command):
	name = ["ImageScale", "Scale", "Rescale", "ImageResize", "Denoise", "Enhance", "Refine", "Copy", "Jumbo"]
	description = "Changes size of supplied image, using an optional scaling operation."
	usage = "<0:url> <1:resolution|multiplier>? <2:y_multiplier(x)>? (nearest|linear|hamming|bicubic|lanczos|scale2x|sdxl|crop|auto)?"
	example = ("scale https://mizabot.xyz/favicon 4", "resize https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 2048x2048 scale2x")
	no_parse = True
	rate_limit = (8, 13)
	flags = "l"
	_timeout_ = 4
	typing = True
	slash = True

	async def __call__(self, bot, user, guild, channel, message, flags, name, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		ops = {"nearest", "linear", "hamming", "bicubic", "lanczos", "scale2x", "crop", "auto"}
		if not args or argv == "list":
			if "l" in flags or argv == "list":
				return ini_md(f"Available scaling operations: [{', '.join(ops)}]")
			# raise ArgumentError("Please input an image by URL or attachment.")
		async with discord.context_managers.Typing(channel):
			try:
				url = args.pop(0)
				urls = await bot.follow_url(url, best=True, allow=True, limit=1)
				if not urls:
					urls = await bot.follow_to_image(argv)
					if not urls:
						urls = await bot.follow_to_image(url)
						if not urls:
							raise ArgumentError
				url = urls[0]
			except (LookupError, ArgumentError):
				if not argv:
					url = None
					try:
						url = await bot.get_last_image(message.channel)
					except FileNotFoundError:
						raise ArgumentError("Please input an image by URL or attachment.")
				else:
					raise ArgumentError("Please input an image by URL or attachment.")
			func = "resize_mult"
			fmt2 = url.split("?", 1)[0].rsplit(".", 1)[-1]
			if fmt2 not in ("mp4", "gif"):
				if is_url(url):
					resp = await asubmit(requests.head, url)
					fmt2 = resp.headers["Content-Type"].rsplit("/", 1)[-1]
					if fmt2 not in ("mp4", "gif"):
						fmt2 = "mp4"
				else:
					fmt2 = "mp4"
			if args and ("." + args[-1] in IMAGE_FORMS or "." + args[-1] in VIDEO_FORMS):
				fmt = args.pop(-1)
			else:
				fmt = fmt2
			if args and args[-1] in ops:
				op = args.pop(-1)
			else:
				if name in ("denoise", "enhance", "refine"):
					op = "sdxl"
				else:
					op = "auto"
			value = " ".join(args).strip()
			if not value:
				x = y = 1
			else:
				# Parse width and height multipliers
				if "x" in value or "X" in value or "*" in value or "×" in value or ":" in value:
					func = "resize_to"
					value = value.replace("x", "X", 1).replace("X", "*", 1).replace("*", "×", 1).replace("×", ":", 1).replace(":", " ", 1)
				try:
					spl = smart_split(value)
				except ValueError:
					spl = value.split()
				x = spl.pop(0)
				if x != "-":
					x = round_min(x)
				if spl:
					y = spl.pop(0)
					if y != "-":
						y = round_min(y)
				else:
					y = "-"
				if func == "resize_mult":
					if y == "-":
						y = x
					for value in (x, y):
						if not value >= -256 or not value <= 256:
							raise OverflowError("Maximum multiplier input is 256.")
			# Try and find a good name for the output image
			try:
				name = url[url.rindex("/") + 1:]
				if not name:
					raise ValueError
				if "." in name:
					name = name[:name.rindex(".")]
			except ValueError:
				name = "unknown"
			if not name.endswith("." + fmt):
				name += "." + fmt
			if op == "sdxl":
				fut = create_task(bot.caption(url, best=True))
			# print(url, func, x, y, op, fmt)
			resp = await process_image(url, func, [x, y, op, "-f", fmt], timeout=_timeout)
			if op == "sdxl":
				pt, p1, p2 = await fut
				prompt = "4k, " + p1 + "\n" + p2
				resp = await process_image("IBASR", "&", [prompt, resp], cap="sdxlr", timeout=240)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Rotate(Command):
	name = ["Orientate", "Orientation", "Transpose"]
	description = "Rotates an image."
	usage = "<0:url> <1:angle(90)>?"
	example = ("rotate https://mizabot.xyz/favicon 90", "rotate https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 123.456")
	no_parse = True
	rate_limit = (8, 13)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, default=90, raw=True)
		value = await bot.eval_math(value)
		async with discord.context_managers.Typing(channel):
			resp = await process_image(url, "rotate_to", [value, "-f", fmt], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Fill(Command):
	name = ["ImageFill", "FillChannel", "FillImage"]
	description = "Fills an optional amount of channels in the target image with an optional value."
	usage = "<0:url> [rgbcmyhsva]* <-1:value(0)>?"
	example = ("fill https://mizabot.xyz/favicon gb 255", "fill https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png r 0")
	no_parse = True
	rate_limit = (7, 10)
	flags = "l"
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		try:
			if not args:
				raise ArgumentError
			url = args.pop(0)
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
		async with discord.context_managers.Typing(channel):
			if is_numeric(args[-1]):
				value = await bot.eval_math(args.pop(-1))
				if type(value) is not int:
					if abs(value) <= 1:
						value = round(value * 255)
					else:
						raise ValueError("invalid non-integer input value.")
			else:
				value = 255
			if not args:
				args = "rgb"
			# Try and find a good name for the output image
			try:
				name = url[url.rindex("/") + 1:]
				if not name:
					raise ValueError
				if "." in name:
					name = name[:name.rindex(".")]
			except ValueError:
				name = "unknown"
			if not name.endswith(".png"):
				name += ".png"
			resp = await process_image(url, "fill_channels", [value, *args], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Blend(Command):
	name = ["ImageBlend", "ImageOP"]
	description = "Combines the two supplied images, using an optional blend operation."
	usage = "<0:url1> <1:url2> (normal|replace|add|sub|mul|div|mod|and|or|xor|nand|nor|xnor|difference|overlay|screen|soft|hard|lighten|darken|plusdarken|overflow|lighting|burn|linearburn|dodge|hue|sat|lum|colour|extract|merge)? <3:opacity(0.5|1)>?"
	example = ("blend https://mizabot.xyz/favicon https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png overflow",)
	no_parse = True
	rate_limit = (13, 17)
	flags = "l"
	_timeout_ = 7
	typing = True

	async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		if not args or argv == "list":
			if "l" in flags or argv == "list":
				return ini_md(
					"Available blend operations: ["
					+ "replace, add, sub, mul, div, mod, and, or, xor, nand, nor, xnor, "
					+ "difference, overlay, screen, soft, hard, lighten, darken, plusdarken, overflow, lighting, "
					+ "burn, linearburn, dodge, lineardodge, hue, sat, lum, colour, extract, merge]"
				)
			raise ArgumentError("Please input an image by URL or attachment.")
		async with discord.context_managers.Typing(channel):
			urls = await bot.follow_url(args.pop(0), best=True, allow=True, limit=1)
			if urls:
				url1 = urls[0]
			else:
				url1 = None
			if not args:
				raise ArgumentError("This command requires two image inputs as URL or attachment.")
			urls = await bot.follow_url(args.pop(0), best=True, allow=True, limit=1)
			if urls:
				url2 = urls[0]
			else:
				url1 = None
			fromA = False
			if not url1 or not url2:
				urls = await bot.follow_to_image(argv)
				if not urls:
					urls = await bot.follow_to_image(argv)
					if not urls:
						raise ArgumentError("Please input an image by URL or attachment.")
				if type(urls) not in (list, alist):
					urls = alist(urls)
				if not url1:
					url1 = urls.pop(0)
				if not url2:
					url2 = urls.pop(0)
			if fromA:
				value = argv
			else:
				value = " ".join(args).strip()
			if not value:
				opacity = 0.5
				operation = "replace"
			else:
				try:
					spl = smart_split(value)
				except ValueError:
					spl = value.split()
				operation = spl.pop(0)
				if spl:
					opacity = await bot.eval_math(spl.pop(-1))
				else:
					opacity = 1
				if not opacity >= -256 or not opacity <= 256:
					raise OverflowError("Maximum multiplier input is 256.")
				if spl:
					operation += " ".join(spl)
				if not operation:
					operation = "replace"
			# Try and find a good name for the output image
			try:
				name = url1[url1.rindex("/") + 1:]
				if not name:
					raise ValueError
				if "." in name:
					name = name[:name.rindex(".")]
			except ValueError:
				name = "unknown"
			if not name.endswith(".png"):
				name += ".png"
			resp = await process_image(url1, "blend_op", [url2, operation, opacity], timeout=_timeout)
			fn = resp
			if isinstance(fn, str) and "." in fn:
				fmt = "." + fn.rsplit(".", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
			elif isinstance(fn, (bytes, memoryview)):
				fmt = magic.from_buffer(fn).rsplit("/", 1)[-1]
				if not name.endswith(fmt):
					if "." in name:
						name = name[:name.rindex(".")]
					name += "." + fmt
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Steganography(Command):
	name = ["Watermark", "Copyright", "Ownership", "NFT", "C", "©"]
	description = "Tags an image with your discord user, or a message (input a user ID to tag another user). Raises an error if the image already has a tag."
	usage = "<0:url> <1:data>? <2:message>?"
	example = ("watermark https://mizabot.xyz/favicon", "nft https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 201548633244565504")
	no_parse = True
	rate_limit = (12, 15)
	_timeout_ = 6
	typing = True

	async def __call__(self, bot, user, message, channel, args, name, **void):
		for a in message.attachments:
			args.insert(0, a.url)
		if not args:
			raise ArgumentError("Please input an image by URL or attachment.")
		urls = await bot.follow_url(args.pop(0))
		if not urls:
			raise ArgumentError("Please input an image by URL or attachment.")
		url = urls[0]
		b = await bot.get_request(url)
		if name == "nft":
			await bot.silent_delete(message)
		if args:
			msg = args.pop(0)
			n = verify_id(msg)
			if isinstance(n, int):
				try:
					user = await bot.fetch_user(n)
				except:
					pass
				else:
					msg = str(user.id)
		else:
			msg = str(user.id)
		remsg = " ".join(args)
		fon = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
		async with discord.context_managers.Typing(channel):
			fn = await self.call(b, msg)
		fn = f"cache/{ts}~1.png"
		if name == "nft":
			f = CompatFile(fn, filename=f"{fon}.png")
			url = await self.bot.get_proxy_url(user)
			await self.bot.send_as_webhook(message.channel, remsg, files=[f], username=user.display_name, avatar_url=url)
		else:
			await bot.send_with_file(channel, f'Successfully created image with encoded message "{msg}".', fn, filename=f"{fon}.webp", reference=message, reacts="🔳")

	async def call(self, b, msg=""):
		ts = hash(b)
		args = (
			sys.executable,
			"misc/steganography.py",
			f"cache/{ts}.png",
			msg,
			"-o",
			f"cache/{ts}~1.png",
		)
		with open(f"cache/{ts}.png", "wb") as f:
			await asubmit(f.write, b)
		print(args)
		proc = psutil.Popen(args, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		try:
			await asubmit(proc.wait, timeout=3200)
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		else:
			text = proc.stdout.read().decode("utf-8", "replace").strip()
			if text.startswith("Copyright detected"):
				i = text.split(": ", 1)[-1]
				if i.isnumeric():
					i = int(i)
					try:
						u = await bot.fetch_user(i)
					except:
						pass
					else:
						pe = PermissionError(f"Copyright detected; image belongs to {user_mention(u.id)}")
						pe.no_react = True
						raise pe
				pe = PermissionError(text)
				pe.no_react = True
				raise pe
		return f"cache/{ts}~1.png"

	async def _callback_(self, bot, message, reaction, user, vals, **void):
		u_id, c_id, m_id = map(int, vals.split("_", 2))
		if user.id != u_id:
			return
		if reaction.decode("utf-8", "replace") != "🗑️":
			return
		m = message
		channel = await bot.fetch_channel(c_id)
		message = await bot.fetch_message(m_id, channel)
		await bot.silent_delete(message)
		guild = message.guild
		if guild and "logM" in bot.data and guild.id in bot.data.logM:
			c_id = bot.data.logM[guild.id]
			try:
				c = await self.bot.fetch_channel(c_id)
			except (EOFError, discord.NotFound):
				bot.data.logM.pop(guild.id)
				return
			emb = await bot.as_embed(message, link=True)
			emb.colour = discord.Colour(0x00FF00)
			action = f"{user_mention(u_id)} **deleted a copyrighted image deleted from** {channel_mention(channel.id)}:\n"
			emb.description = lim_str(action + emb.description, 4096)
			emb.timestamp = message.created_at
			self.bot.send_embeds(c, emb)
		await m.reply("Message has been successfully taken down.")


class OCR(Command):
	name = ["Tesseract", "Read", "Image2Text"]
	description = "Attempts to read text in an image using Optical Character Recognition AI."
	usage = "<url>"
	example = ("ocr https://opengraph.githubassets.com/c3922b6d44ff4a498c1607bec89b70a1c755e2d44d115bec93b5bb981aa1ad36/tesseract-ocr/tesseract",)
	rate_limit = (10, 15)
	slash = ("Read")

	async def __call__(self, bot, user, message, args, argv, **void):
		fut = asubmit(__import__, "pytesseract")
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		resp = await process_image(url, "resize_max", ["-nogif", 1024, "auto", "-f", "png"], timeout=60)
		if isinstance(resp, str):
			f = open(resp, "rb")
		else:
			f = io.BytesIO(resp)
		im = await asubmit(Image.open, f)
		pytesseract = await fut
		text = await asubmit(pytesseract.image_to_string, im, config="--psm 1", timeout=8)
		return css_md(f"[Detected text]\n{no_md(text)}.")


class Art(Command):
	_timeout_ = 150
	name = ["AIArt", "Inpaint", "StableDiffusion", "SDXL", "Dalle", "Dalle2", "Dream", "Imagine", "Inspire", "Openjourney", "Midjourney"]
	description = "Runs a Stable Diffusion AI art generator on the input prompt or image. Operates on a global queue system for image prompts. Configurable parameters are --strength, --guidance-scale, --aspect-ratio and --negative-prompt."
	usage = "<0:prompt> <inpaint{?i}>? <single{?s}> <raw{?r}>"
	example = ("art cute kitten", "art https://mizabot.xyz/favicon")
	rate_limit = (45, 60)
	flags = "irs"
	typing = True
	slash = ("Art", "Imagine")
	sdiff_sem = Semaphore(3, 256, rate_limit=7)
	fut = None
	try:
		imagebot = imagebot.Bot(token=AUTH.get("openai_key"))
	except:
		print_exc()
		imagebot = None
	has_py39 = subprocess.run("py -3.9 -m pip").returncode == 0

	async def __call__(self, bot, guild, user, channel, message, name, args, flags, comment="", **void):
		if not torch:
			raise NotImplementedError("AI features are currently disabled, sorry!")
		for a in reversed(message.attachments):
			args.insert(0, a.url)
		if not args:
			if getattr(message, "reference", None) and getattr(message.reference, "resolved", None):
				m = message.reference.resolved
				url = f"https://discord.com/channels/0/{channel.id}/{m.id}"
				urls = await bot.follow_url(url, allow=True, images=True)
				if urls and urls[0] != url:
					args = urls
		if not args:
			# raise ArgumentError("Input string is empty.")
			# s = await Request(
			#     "https://magatsu.net/generators/art/index.php",
			#     data=dict(selGenCount="1", selStyle="2", subGenerate="Generate Prompts"),
			#     method="POST",
			#     decode=True,
			#     aio=True,
			# )
			# s = s.split("<strong>Prompt:</strong>", 1)[-1].removeprefix("<BR>")
			# s = s.split("<BR>", 1)[0].replace("<br />", " ").replace("<br>", " ").replace("<i>", "*").replace("<b>", "**").replace("<u>", "__")
			# args = [s]
			# print(s)
			args = [("art " * xrand(1, 64)).rstrip()]
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		freelim = 50
		if premium < 2:
			data = bot.data.users.setdefault(user.id, {})
			freebies = [t for t in data.get("freebies", ()) if utc() - t < 86400]
			if len(freebies) < freelim:
				premium = 2
		else:
			freebies = None
		sdxl = premium >= 2
		req = " ".join(args)
		url = None
		url2 = None
		rems = deque()
		nprompt = ""
		kwargs = {
			"--device": "GPU",
			"--num-inference-steps": "48" if premium < 4 else "64",
			"--guidance-scale": "9.9",
			"--eta": "0.8",
			"--aspect-ratio": "0",
			"--negative-prompt": "blurry, distorted, disfigured, bad anatomy, poorly drawn, low quality, ugly",
		}
		inpaint = "i" in flags or name == "inpaint"
		specified = set()
		kwarg = ""
		for arg in args:
			if kwarg:
				# if kwarg == "--model":
				#     kwargs[kwarg] = arg
				if kwarg.startswith("--"):
					kwarg = kwarg.replace("_", "-")
				if kwarg == "--seed":
					kwargs[kwarg] = arg
				elif kwarg in ("--num-inference-steps", "--ddim-steps"):
					kwarg = "--num-inference-steps"
					kwargs[kwarg] = str(max(1, min(64, int(arg))))
				elif kwarg in ("--guidance-scale", "--guidance", "--scale"):
					kwarg = "--guidance-scale"
					kwargs[kwarg] = str(max(0, min(100, float(arg))))
				elif kwarg == "--eta":
					kwargs[kwarg] = str(max(0, min(1, float(arg))))
				# elif kwarg in ("--tokenizer", "--tokeniser"):
				#     kwargs["--tokenizer"] = arg
				elif kwarg == "--prompt":
					kwargs[kwarg] = arg
				elif kwarg in ("--negative-prompt", "--negative", "--neg"):
					kwarg = "--negative-prompt"
					kwargs[kwarg] = arg
				elif kwarg == "--strength":
					kwargs[kwarg] = str(max(0, min(1, float(arg))))
				elif kwarg in ("--aspect-ratio", "--aspect", "--ar"):
					arg = await bot.eval_math(arg.replace(":", "/").replace("x", "/").replace("×", "/"))
					arg = float(arg)
					if arg < 1 / 256 or arg > 256:
						raise OverflowError("Maximum permitted aspect ratio is 1:256.")
					kwarg = "--aspect-ratio"
					kwargs[kwarg] = str(arg)
				# elif kwargs == "--mask":
				#     kwargs[kwarg] = arg
				specified = kwarg
				kwarg = ""
				continue
			if arg.startswith("--"):
				kwarg = arg
				continue
			urls = None
			i = verify_id(arg)
			if isinstance(i, int):
				with suppress():
					u = await bot.fetch_user(i)
					if not rems:
						nprompt = u.display_name
					urls = [best_url(u)]
			if not urls:
				urls = await bot.follow_url(arg, allow=True, images=True)
				if not urls:
					rems.append(arg)
				else:
					urls = list(urls)
			if urls and not url:
				url = urls.pop(0)
			if urls and not url2:
				url2 = urls.pop(0)
		prompt = " ".join(rems).strip()
		if not prompt and not sdxl:
			prompt = nprompt
		if not prompt:
			if not url:
				raise ArgumentError("Please input a valid prompt.")
			pt, p1, p2 = await bot.caption(url, best=premium >= 4)
			prompt = "\n".join((p1, p2))
			if not prompt:
				prompt = "art"
			print(url, prompt)
			force = False
		else:
			force = True
		if sdxl and "r" not in flags and prompt.count(" ") < 48:
			oprompt = prompt
			uid = user.id
			temp = oprompt.replace('"""', "'''")
			prompt = f'"""\n{temp}\n"""\n\nImprove the above image caption as a description to send to Dall·E image generation. Be as detailed as possible in at least 2 sentences, and reword anything that may violate policy guidelines.'
			if bot.is_trusted(guild) >= 2:
				for uid in bot.data.trusted[guild.id]:
					if uid and bot.premium_level(uid, absolute=True) >= 2:
						break
				else:
					uid = next(iter(bot.data.trusted[guild.id]))
				u = await bot.fetch_user(uid)
			else:
				u = user
			data = bot.data.users.get(u.id, {})
			oai = data.get("trial") and data.get("openai_key")
			premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
			inputs = dict(
				user_id=user.id,
				channel_id=channel.id,
				prompt=prompt,
				key=AUTH.get("openai_key"),
				huggingface_token=AUTH.get("huggingface_key"),
				vis_session=AUTH.get("vis_session"),
				bals={k: v for k, v in bot.data.token_balances.items() if v < 0},
				oai=oai,
				nsfw=bot.is_nsfw(channel),
				premium=premium,
			)
			out = await process_image("CBAU", "$", [inputs], cap="agpt", timeout=20)
			if out and out[0] == out[-1] == '"' and not oprompt[0] == oprompt[-1] == '"':
				try:
					out = orjson.loads(out)
				except orjson.JSONDecodeError:
					pass
			out = out.replace("Dall·E", "art")
			prompt = (nprompt or oprompt) + ".\n\n" + out.strip()
		req = prompt
		print("PROMPT:", prompt)
		if url:
			if force:
				if req:
					req += " "
				req += url
				if url2:
					req += " " + url2
			else:
				req = url
		if specified:
			req += " ".join(f"{k} {v}" for k, v in kwargs.items() if k in specified)
		aspect = float(kwargs.get("--aspect-ratio", 1))
		negative = kwargs.get("--negative-prompt", "")
		nsfw = bot.is_nsfw(channel)
		# if not nsfw and prompt and AUTH.get("openai_key"):
		#     import openai
		#     openai.api_key = AUTH["openai_key"]
		#     resp = openai.Moderation.create(
		#         input=prompt,
		#     )
		#     results = resp.results[0].categories
		#     if results.hate or results["self-harm"] or results["sexual/minors"] or results["violence/graphic"]:
		#         raise PermissionError("NSFW filter detected in non-NSFW channel. If you believe this was a mistake, please try again.")
		emb = None
		fn = None
		futs = []
		if "s" in flags:
			amount = 1
		else:
			amount = 4 if premium >= 4 else 2 if premium >= 3 else 1
		if not sdxl:
			amount = 9 if premium >= 4 else 4
		amount2 = 0
		if bot.is_trusted(guild) >= 2:
			for uid in bot.data.trusted[guild.id]:
				if uid and bot.premium_level(uid, absolute=True) >= 2:
					break
			else:
				uid = next(iter(bot.data.trusted[guild.id]))
			u = await bot.fetch_user(uid)
		else:
			u = user
		data = bot.data.users.get(u.id, {})
		oai = data.get("trial") and data.get("openai_key")

		async def ibasl_r(p, k, n, f, c, s, a, np):
			if sdxl and (c > 1 or premium >= 4 or random.randint(0, 1)):
				try:
					await process_image("lambda: 1+1", "$", (), cap="sdxlr", timeout=2)
				except:
					print_exc()
				else:
					return await process_image("IBASLR", "&", [p, k, n, f, c, a, np], cap="sdxlr", timeout=420)
			resp = await process_image("IBASL", "&", [p, k, n, f, c, s, a, np], cap="sdxl" if s else "sd", timeout=420)
			if s:
				out = []
				for r1 in resp:
					r2 = await process_image("IBASR", "$", [p, r1, 64, np], cap="sdxlr", timeout=240)
					out.append(r2)
				return out
			return resp

		if amount2 < amount:
			dalle2 = name.startswith("dalle")
			if dalle2 and premium < 4:
				raise PermissionError("Premium subscription required to perform DALL·E 2 operations.")
			openjourney = "journey" in name
			async with discord.context_managers.Typing(channel):
				futt = []
				c = 0
				if not url and (aspect != 1 or negative or amount >= 1 and not dalle2 and not openjourney and not self.sdiff_sem.is_busy()):
					noprompt = not force and not kwargs.get("--mask")
					p = "" if noprompt and not sdxl else prompt
					c = min(amount, 9 if nsfw and not self.sdiff_sem.active else 5)
					c2 = c
					while c2 > 0:
						n = min(c2, xrand(floor(sqrt(c2) + 1)) + 1, 2 if sdxl else 9)
						if not n:
							n = c2
						fut = create_task(ibasl_r(p, kwargs, nsfw, False, n, sdxl, aspect, negative))
						futt.append(fut)
						c2 -= n
				self.imagebot.token = oai or AUTH.get("openai_key")
				ims = []
				try:
					if c > amount:
						raise PermissionError
					ims = await asubmit(self.imagebot.art, prompt, url, url2, kwargs, specified, dalle2, openjourney, sdxl, nsfw, amount - c, timeout=480)
				except PermissionError:
					async with self.sdiff_sem:
						for fut in futt:
							await fut
				# print(ims)
				async with self.sdiff_sem:
					for fut in futt:
						try:
							ims2 = await fut
						except PermissionError:
							if not ims:
								raise
						if ims2:
							ims2.extend(ims)
							ims = ims2
				futs.extend(ims)
				amount2 = len(futs)
		if amount2 < amount:
			if self.has_py39:
				with tracebacksuppressor:
					if not self.fut and not os.path.exists("misc/stable_diffusion.openvino"):
						self.fut = asubmit(subprocess.run(
							[
								"git",
								"clone",
								"https://github.com/bes-dev/stable_diffusion.openvino.git",
							],
							cwd="misc",
						))
					await self.fut
					if os.name == "nt":
						self.fut = asubmit(subprocess.run(
							[
								python,
								"-m",
								"pip",
								"install",
								"-r",
								"requirements.txt",
							],
							cwd="misc",
						))
					else:
						self.fut = asubmit(subprocess.run(
							[
								sys.executable,
								"-m",
								"pip",
								"install",
								"-r",
								"requirements.txt",
							],
							cwd="misc",
						))
					await self.fut
					self.fut = None
			if os.name == "nt":
				args = [
					"py",
					"-3.9",
					"demo.py",
				]
			else:
				args = [
					sys.executable,
					"demo.py",
				]
			if prompt and "--prompt" not in kwargs:
				args.extend((
					"--prompt",
					prompt,
				))
			async with discord.context_managers.Typing(channel):
				image_1 = image_2 = None
				image_1b = image_2b = None
				if url:
					resp = await process_image(url, "resize_max", ["-nogif", 1024 if sdxl else 768, "auto", "-f", "png"], timeout=60)
					image_1 = resp
					if inpaint and url2:
						image_2b = await bot.get_request(url2)
					if inpaint and not url2:
						resp = await process_image(image_1, "get_mask", ["-nogif", "-nodel", "-f", "png"], timeout=60)
						image_2 = resp
						resp = await process_image(image_1, "inpaint", [image_2, "-nodel", "-f", "png"], timeout=60)
						image_1 = resp
						resp = await process_image(image_2, "expand_mask", ["-nogif", 12, "-f", "png"], timeout=60)
						image_2 = resp
						print(image_1, image_2)
					if "--strength" not in kwargs:
						args.extend((
							"--strength",
							"0.8",
						))
						kwargs["--strength"] = 0.8
					# if premium >= 2 and not force and "--strength" not in kwargs and str(kwargs["--guidance-scale"]) == "7.5" and str(kwargs["--eta"]) == "0.8":
					#     if isinstance(image_1, bytes):
					#         image_1b = image_1
					#     else:
					#         with open(image_1, "rb") as f:
					#             image_1b = f.read()
					#     if image_2:
					#         if isinstance(image_2, bytes):
					#             image_2b = image_2
					#         else:
					#             with open(image_2, "rb") as f:
					#                 image_2b = f.read()
					#     with tracebacksuppressor:
					#         if bot.is_trusted(guild) >= 2:
					#             for uid in bot.data.trusted[guild.id]:
					#                 if uid and bot.premium_level(uid, absolute=True) >= 2:
					#                     break
					#             else:
					#                 uid = next(iter(bot.data.trusted[guild.id]))
					#             u = await bot.fetch_user(uid)
					#         else:
					#             u = user
					#         data = bot.data.users.get(u.id, {})
					#         oai = data.get("trial") and data.get("openai_key")
					#         self.imagebot.token = oai or AUTH.get("openai_key")
					#         ims = await asubmit(self.imagebot.dalle_i2i, prompt, image_1b, image_2b, False, amount, timeout=60)
					#         futs.extend(ims)
					#         amount2 = len(futs)
				oargs = args
				att = 0
				while amount2 < amount and att < 5:
					att += 1
					args = list(oargs)
					if self.sdiff_sem.is_busy() and not getattr(message, "simulated", False):
						await send_with_react(channel, italics(ini_md(f"StableDiffusion: {sqr_md(req)} enqueued in position {sqr_md(self.sdiff_sem.passive + 1)}.")), reacts="❎", reference=message)
					async with self.sdiff_sem:
						if url:
							fn = os.path.abspath(image_1) if isinstance(image_1, str) else image_1
							# fn = "misc/stable_diffusion.openvino/input.png"
							# if os.path.exists(fn):
							#     os.remove(fn)
							# os.rename(image_1, fn)
							args.extend((
								"--init-image",
								fn,
							))
							kwargs["--init-image"] = fn
							if image_2:
								fm = os.path.abspath(image_2) if isinstance(image_2, str) else image_2
								# fm = "misc/stable_diffusion.openvino/mask.png"
								# if os.path.exists(fm):
								#     os.remove(fm)
								# os.rename(image_2, fm)
								args.extend((
									"--mask",
									fm,
								))
								kwargs["--mask"] = fm
							elif image_2b:
								fm = os.path.abspath(f"cache/{ts_us()}.png")
								# fm = "misc/stable_diffusion.openvino/mask.png"
								with open(fm, "wb") as f:
									f.write(image_2b)
								args.extend((
									"--mask",
									fm,
								))
								kwargs["--mask"] = fm
						for k, v in kwargs.items():
							args.extend((k, v))
						print(args)
						if self.has_py39:
							proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd() + "/misc/stable_diffusion.openvino", stdout=subprocess.DEVNULL)
							try:
								await asyncio.wait_for(proc.wait(), timeout=3200)
							except (T0, T1, T2):
								with tracebacksuppressor:
									force_kill(proc)
								raise
							with open("misc/stable_diffusion.openvino/output.png", "rb") as f:
								futs.append(f.read())
							amount2 = len(futs)
						else:
							noprompt = not force and not kwargs.get("--mask")
							p = "" if noprompt and not sdxl else prompt
							futt = []
							c = amount - amount2
							c2 = c
							while c2 > 0:
								n = min(c2, xrand(floor(sqrt(c2) + 1)) + 1, 2 if sdxl else 9)
								if not n:
									n = c2
								fut = create_task(ibasl_r(p, kwargs, nsfw, False, n, sdxl, aspect, negative))
								futt.append(fut)
								c2 -= n
							for fut in futt:
								ims = await fut
								futs.extend(ims)
							amount2 = len(futs)
		ffuts = []
		exc = RuntimeError("Unknown error occured.")
		async with discord.context_managers.Typing(channel):
			for tup in futs:
				if len(ffuts) >= amount:
					break
				if not tup:
					continue
				if not isinstance(tup, tuple):
					tup = (tup,)
				fn = tup[0]
				if not fn:
					continue
				if fn and not oai and len(tup) > 1:
					cost = tup[1]
					if "costs" in bot.data:
						bot.data.costs.put(user.id, cost)
						if guild:
							bot.data.costs.put(guild.id, cost)
					if bot.is_trusted(guild) >= 2:
						for uid in bot.data.trusted[guild.id]:
							if uid and bot.premium_level(uid, absolute=True) >= 2:
								break
						else:
							uid = next(iter(bot.data.trusted[guild.id]))
						u = await bot.fetch_user(uid)
					else:
						u = user
					data = bot.data.users.get(u.id)
					if data and data.get("trial"):
						bot.data.users.add_diamonds(user, cost / -25000)
						if data.get("diamonds", 0) < 1:
							bot.premium_level(u)
							emb = discord.Embed(colour=rand_colour())
							emb.set_author(**get_author(bot.user))
							emb.description = (
								"Uh-oh, it appears your tokens have run out! Check ~wallet to view your balance, top up using a donation [here]({bot.kofi_url}), "
								+ "or purchase a subscription to gain temporary unlimited usage!"
							)
				if isinstance(fn, str):
					with open(fn, "rb") as f:
						fn = f.read()
				ffut = create_task(bot.commands.steganography[0].call(fn, str(bot.id)))
				ffut.name = (lim_str(prompt, 96).rstrip(".").replace(".", "_") or "untitled") + ".png"
				ffut.back = fn
				ffuts.append(ffut)
		if not ffuts:
			raise exc
		files = []
		for ffut in ffuts:
			name = ffut.name
			fn = ffut.back
			with tracebacksuppressor:
				fn = await ffut
			files.append(CompatFile(fn, filename=name))
		if premium >= 2 and freebies is not None:
			bot.data.users[user.id].setdefault("freebies", []).append(utc())
			rem = freelim - len(freebies)
			if not emb and rem in (27, 9, 3, 1):
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = f"{rem}/{freelim} premium commands remaining today (free commands will be used after).\nIf you're able to contribute towards [funding my API]({bot.kofi_url}) hosting costs it would mean the world to us, and ensure that I can continue providing up-to-date tools and entertainment.\nEvery little bit helps due to the size of my audience, and you will receive access to unlimited and various improved commands as thanks!"
		if len(files) == 2:
			fe = files.pop(1)
			urls, mids = await bot.data.exec.stash(fe, filename=fe.filename)
			if urls:
				comment = ("\n".join(url.split("?", 1)[0] for url in urls) + "\n" + comment).strip()
		return await send_with_react(channel, comment, files=files, reference=message, reacts="🔳", embed=emb)
		# await bot.send_with_file(channel, "", fn, filename=lim_str(prompt, 96) + ".png", reference=message, reacts="🔳", embed=emb)


class UpdateImages(Database):
	name = "images"

	def __call__(self, **void):
		if imagebot:
			imagebot.update()
