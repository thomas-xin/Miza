# Make linter shut up lol
if "common" not in globals():
	import common
	from common import *
print = PRINT

try:
	import yt_dlp as youtube_dl
except ModuleNotFoundError:
	try:
		youtube_dl = __import__("youtube_dl")
	except ModuleNotFoundError:
		youtube_dl = None
if AUTH["ai_features"]:
	import torch
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

VIDEOS = ("gif", "webp", "apng", "mp4", "mkv", "ts", "webm", "mov", "wmv", "flv", "avi", "qt", "f4v", "zip")


async def get_image(bot, user, message, args, argv, default=2, raw=False, ext="png", count=0):
	try:
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		if not args:
			raise ArgumentError
		url = args.pop(0)
		urls = await bot.follow_url(url, best=True, allow=not raw, limit=1)
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
	usage = "<0:url> <1:multiplier[2]>?"
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
	usage = "<0:url> <mode(protanopia|protanomaly|deuteranopia|deuteranomaly|tritanopia|tritanomaly|achromatopsia|achromatonomaly)>? <1:ratio[0.9]>?"
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
		ext = "png"
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


class EdgeDetect(Command):
	name = ["Laplacian", "Canny", "Depth", "DepthMap", "Edges", "Edge"]
	description = "Applies an edge or depth detection algorithm to the image."
	usage = "<url>"
	example = ("laplacian https://mizabot.xyz/favicon",)
	no_parse = True
	rate_limit = (5, 7)
	_timeout_ = 3
	typing = True

	async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
		fname = name
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		async with discord.context_managers.Typing(channel):
			func = "canny"
			cap = "caption"
			if fname in ("depth", "depthmap"):
				func = "depth"
				cap = "sdxl"
			elif fname == "laplacian":
				func = "laplacian"
				cap = "image"
			resp = await process_image(url, func, ["-f", fmt], cap=cap, timeout=_timeout)
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
	usage = "<0:url> <2:source(rgb|cmy|xyz|hsv|hsl|hsi|lab|luv|yiq|yuv)>? <1:dest(hsv|hsl|hsi|lab|luv|yiq|yuv|rgb|cmy|xyz)>?"
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
	usage = "<0:url> <cell_count[7]>?"
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
			fn = "colour.png"
			f = CompatFile(resp, filename=fn)
		await bot.send_with_file(channel, msg, f, filename=fn, best=True, reference=message)


class Gradient(Command):
	description = "Generates a gradient with a specific shape."
	usage = "<mode(linear|radial|conical|spiral|polygon)>? <0:count[1]>? <-1:colour[white]>?"
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
			fn = "gradient.png"
			f = CompatFile(resp, filename=fn)
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
			# fn = resp
			fn = "average.png"
			f = CompatFile(resp, filename=fn)
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
	name = ["RainbowGIF", "Gay", "Shiny"]
	description = "Creates a .gif image from repeatedly hueshifting supplied image."
	usage = "<0:url> <1:duration[2]>?"
	example = ("rainbow https://mizabot.xyz/favicon", "rainbow https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 6")
	no_parse = True
	rate_limit = (10, 13)
	_timeout_ = 8
	typing = True

	async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
		func = "shiny_gif" if name == "shiny" else "rainbow_gif"
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv, ext="gif")
		async with discord.context_managers.Typing(channel):
			# -gif signals to image subprocess that the output is always a .gif image
			resp = await process_image(url, func, [value, "-gif", "-f", fmt], timeout=_timeout)
			fn = resp
		await bot.send_with_file(channel, "", fn, filename=name, reference=message, reacts="🔳")


class Scroll(Command):
	name = ["Parallax", "Offset", "ScrollGIF"]
	description = "Creates a .gif image from repeatedly shifting supplied image in a specified direction."
	usage = "<0:url> <1:direction[left]>? <2:duration[2]>? <3:fps[32]>?"
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
	usage = "<0:url> <1:duration[2]>?"
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
	usage = "<0:url>+ <1:orbital_count[5]>? <2:duration[2]>?"
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
			urls = await bot.follow_url(spl[0], best=True, limit=1)
			if not urls:
				break
			value = spl[-1] if len(spl) > 1 else ""
			extras.append(urls[0])
		# if extras:
		#     print(url, *extras)
		ems = find_emojis_ex(value)
		for em in ems:
			if is_url(em):
				u = em
			else:
				u = await bot.emoji_to_url(em)
			if u:
				extras.append(u)
			else:
				ems.remove(u)
		if ems:
			value = translate_emojis(replace_emojis(value))
			for search in ems:
				value = value.replace(search, "").strip()
		spl = value.rsplit(None, 1)
		print("ORBIT:", name, value, url, fmt, extra, extras, sep=", ")
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
	name = ["PetPet", "Attack", "PatGIF", "Pat", "PetGIF"]
	description = "Creates a .gif image from applying the Petpet generator to the supplied image."
	usage = "<0:url> <1:squish[0.1]>? <2:duration[0.25]>?"
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
	name = ["Liquefy", "Crumple", "Crush", "MagikGIF"]
	description = "Repeatedly applies the Magik image filter to supplied image."
	usage = "<0:url> <cell_size[7]>? <iterations[64]>? <duration[2]>?"
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
	usage = "<0:url>+ <-2:fps[20]>? <-1:format[gif]>?"
	example = ("gif https://www.youtube.com/watch?v=dQw4w9WgXcQ", "gif https://discord.com/assets/7c010dc6da25c012643ea22c1f002bb4.svg https://discord.com/assets/66f6c781fe86c346fbaf3390618668fc.svg https://discord.com/assets/626aaed496ac12bbdb68a86b46871a1f.svg -r 3")
	no_parse = True
	rate_limit = (12, 16)
	_timeout_ = 20
	flags = "r"
	typing = True

	async def __call__(self, bot, user, guild, channel, message, flags, name, args, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + list(args)
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
				urls = await bot.follow_url(url, best=True, images=True, allow=True, limit=1)
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
	name = ["ImageScale", "Scale", "Rescale", "ImageResize", "Scale2x", "SwinIR", "Denoise", "Enhance", "Refine", "Copy", "Jumbo"]
	description = "Changes size of supplied image, using an optional scaling operation."
	usage = "<0:url> <1:size(?:resolution|multiplier|filesize)>* <-1:mode(nearest|linear|hamming|bicubic|lanczos|scale2x|swinir|crop|auto)>?"
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
		ops = {"nearest", "linear", "hamming", "bicubic", "lanczos", "scale2x", "swinir", "crop", "auto"}
		if not args and name not in ops or argv == "list":
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
					resp = await asubmit(requests.head, url, headers=Request.header(), stream=True)
					fmt2 = resp.headers.get("Content-Type", "").rsplit("/", 1)[-1]
					if fmt2 not in ("mp4", "gif"):
						fmt2 = "mp4"
				else:
					fmt2 = "mp4"
			if args and ("." + args[-1] in IMAGE_FORMS or "." + args[-1] in VIDEO_FORMS):
				fmt = args.pop(-1)
			else:
				fmt = fmt2
			if name in ops:
				op = name
			elif args and args[-1] in ops:
				op = args.pop(-1)
			else:
				if name in ("denoise", "enhance", "refine"):
					op = "swinir"
				else:
					op = "auto"
			fl = None
			x = y = 1
			value = " ".join(args).strip()
			if not value:
				pass
			elif value.endswith("b") or value.endswith("B"):
				fl = byte_unscale(value[:-1])
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
					x = await bot.eval_math(x)
				if spl:
					y = spl.pop(0)
					if y != "-":
						y = await bot.eval_math(y)
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
			cap = "image"
			if op == "swinir":
				func = "IBASU"
				cap = "sdxl"
			resp = url
			if not fl or not x == y == 1:
				resp = await process_image(resp, func, [x, y, op, "-f", fmt], cap=cap, timeout=_timeout)
			if fl:
				resp = await bot.optimise_image(resp, fsize=fl, fmt=fmt)
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


class Crop(Command):
	name = ["ImageCut", "Cut", "ImageCrop"]
	description = "Crops the supplied image to a given size, expanding if necessary. Measurements are in pixels; use \"-\" to omit a value."
	usage = "<0:url> <1:left> <2:top> <3:right> <4:bottom>"
	example = ("crop https://mizabot.xyz/favicon 10 - 20 10", "cut https://cdn.discordapp.com/attachments/911172125438660648/1026492110871990313/3d8860e07889ebddae42222a9793ab85.png 120 120 -120 -120 webp")
	no_parse = True
	rate_limit = (4, 9)
	_timeout_ = 4
	typing = True
	slash = True

	async def __call__(self, bot, user, guild, channel, message, flags, name, args, argv, _timeout, **void):
		# Take input from any attachments, or otherwise the message contents
		if message.attachments:
			args = [best_url(a) for a in message.attachments] + args
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
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
			fmt2 = url.split("?", 1)[0].rsplit(".", 1)[-1]
			if fmt2 not in ("mp4", "gif"):
				if is_url(url):
					resp = await asubmit(requests.head, url, headers=Request.header(), stream=True)
					fmt2 = resp.headers["Content-Type"].rsplit("/", 1)[-1]
					if fmt2 not in ("mp4", "gif"):
						fmt2 = "mp4"
				else:
					fmt2 = "mp4"
			if args and ("." + args[-1] in IMAGE_FORMS or "." + args[-1] in VIDEO_FORMS):
				fmt = args.pop(-1)
			else:
				fmt = fmt2
			if len(args) < 4:
				raise ArgumentError("All 4 positional values are currently required to use this command.")
			coords = args[:4]
			resp = await process_image(url, "crop_to", [*coords, "-f", fmt], timeout=_timeout)
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
	description = "Rotates an image. Defaults to counterclockwise rotation."
	usage = "<0:url> <1:angle[90]>?"
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
	usage = "<0:url> <channels(r|g|b|c|m|y|h|s|v|a)>* <-1:value[0]>?"
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
	usage = "<0:url1> <1:url2> <mode(normal|replace|add|sub|mul|div|mod|and|or|xor|nand|nor|xnor|difference|overlay|screen|soft|hard|lighten|darken|plusdarken|overflow|lighting|burn|linearburn|dodge|hue|sat|lum|colour|extract|merge)>? <3:opacity[0.5/1]>?"
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
				url2 = None
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
		# fn = f"cache/{ts}~1.png"
		if name == "nft":
			f = CompatFile(fn, filename=f"{fon}.png")
			url = await self.bot.get_proxy_url(user)
			await self.bot.send_as_webhook(message.channel, remsg, files=[f], username=user.display_name, avatar_url=url)
		else:
			await bot.send_with_file(channel, f'Successfully created image with encoded message "{msg}".', fn, filename=f"{fon}.png", reference=message, reacts="🔳")

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
						u = await self.bot.fetch_user(i)
					except:
						pass
					else:
						pe = PermissionError(f"Copyright detected; image belongs to {user_mention(u.id)}")
						pe.no_react = True
						raise pe
				pe = PermissionError(text.replace("Copyright detected", "Text detected", 1))
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
	ephemeral = True

	async def __call__(self, bot, user, message, args, argv, **void):
		fut = asubmit(__import__, "pytesseract")
		name, value, url, fmt, extra = await get_image(bot, user, message, args, argv)
		resp = await process_image(url, "resize_max", ["-nogif", 4096, 0, "auto", "-f", "png"], timeout=60)
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
	name = ["AIArt", "Inpaint", "Morph", "ControlNet", "StableDiffusion", "SDXL", "Dalle", "Dalle2", "Dalle3", "Dream", "Imagine", "Inspire"]
	description = "Runs a Stable Diffusion AI art generator on the input prompt or image. Operates on a global queue system for image prompts. Configurable parameters are --strength, --guidance-scale, --aspect-ratio and --negative-prompt."
	usage = "<0:prompt> <inpaint(-i)|morph(-m)>? <single(-s)>? <raw(-r)>?"
	example = ("art cute kitten", "art https://mizabot.xyz/favicon")
	rate_limit = (45, 60)
	flags = "imrsz"
	typing = True
	slash = ("Art", "Imagine")
	sdiff_sem = Semaphore(3, 256, rate_limit=7)
	fut = None

	async def __call__(self, bot, guild, user, channel, message, name, args, flags, comment="", **void):
		if not torch:
			raise NotImplementedError("AI features are currently disabled, sorry!")
		for a in reversed(message.attachments):
			args.insert(0, a.url)
		if not args:
			if getattr(message, "reference", None) and getattr(message.reference, "resolved", None):
				m = message.reference.resolved
				urls = await bot.follow_url(m, allow=True, images=True)
				if urls:
					args = urls
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		freelim = 50
		if premium < 2:
			data = bot.data.users.setdefault(user.id, {})
			freebies = [t for t in data.get("freebies", ()) if utc() - t < 86400]
			if len(freebies) < freelim:
				premium = 2
		else:
			freebies = None
		sdxl = premium >= 2 and name != "stablediffusion"
		req = " ".join(args)
		url = None
		url2 = None
		rems = deque()
		nprompt = ""
		kwargs = {
			"--device": "GPU",
			"--num-inference-steps": "38" if premium < 4 else "44",
			"--guidance-scale": "7",
			"--eta": "0.8",
			"--aspect-ratio": "0",
			"--negative-prompt": None,
		}
		inpaint = "i" in flags or name == "inpaint"
		specified = set()
		kwarg = ""
		if "--mask" in args or name in ("morph", "controlnet") or "m" in flags:
			morph = True
		else:
			morph = False
		count = inf
		for arg in args:
			if kwarg:
				# if kwarg == "--model":
				#     kwargs[kwarg] = arg
				if kwarg.startswith("--"):
					kwarg = kwarg.replace("_", "-")
				if kwarg == "--seed":
					kwargs[kwarg] = arg
				elif kwarg in ("--num-inference-steps", "--ddim-steps", "--steps", "--step"):
					kwarg = "--num-inference-steps"
					kwargs[kwarg] = str(max(1, min(32 * (premium + 2), int(arg))))
				elif kwarg in ("--guidance-scale", "--guidance", "--scale", "--gs"):
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
				elif kwarg == "--count":
					count = round_random(float(arg))
				elif kwarg in ("--aspect-ratio", "--aspect", "--ar"):
					arg = await bot.eval_math(arg.replace(":", "/").replace("x", "/").replace("×", "/"))
					arg = float(arg)
					if arg < 1 / 256 or arg > 256:
						raise OverflowError("Maximum permitted aspect ratio is 1:256.")
					kwarg = "--aspect-ratio"
					kwargs[kwarg] = str(arg)
				# elif kwarg == "--mask":
					# kwargs[kwarg] = arg
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
		nsfw = bot.is_nsfw(channel)
		prompt = " ".join(rems).strip()
		if not prompt and not sdxl:
			prompt = nprompt
		if not prompt:
			if url:
				# raise ArgumentError("Please input a valid prompt.")
				pt, *p1 = await bot.caption(url, best=3 if premium >= 4 else 0)
				prompt = "\n".join(filter(bool, p1))
			print(url, prompt)
			force = False
		else:
			force = True
		if not prompt:
			prompt = "art"
		else:
			resp = await bot.moderate("Create an image of: " + prompt)
			flagged = resp.flagged
			if not nsfw and flagged:
				raise PermissionError(
					"Apologies, my AI has detected that your input may be inappropriate.\n"
					+ "Please move to a NSFW channel, reword, or consider contacting the support server if you believe this is a mistake!"
				)
			if flagged:
				print("Flagged:", resp)
				kwargs["--nsfw"] = True
		kwargs["--negative-prompt"] = ", ".join(set(("watermark", "blurry", "distorted", "disfigured", "bad anatomy", "poorly drawn", "low quality", "ugly")).difference(prompt.split()))

		if not bot.verify_integrity(message):
			return

		if name == "dalle":
			dalle = 3 if premium >= 4 else 2
		else:
			dalle = name.startswith("dalle") and int(name.removeprefix("dalle"))
		if dalle:
			if dalle == 3 and premium < 4:
				raise PermissionError(f"Distributed premium level 2 or higher required; please see {bot.kofi_url} for more info!")
			if premium < 3:
				raise PermissionError(f"Distributed premium level 1 or higher required; please see {bot.kofi_url} for more info!")
		if "s" in flags:
			count = 1
		elif not isfinite(count) and url:
			count = 1
		if dalle:
			amount = 4 if premium >= 6 else 2 if premium >= 4 else 1
		elif sdxl:
			amount = 9 if premium >= 5 else 4 if premium >= 3 else 1
		else:
			amount = 9 if premium >= 4 else 4 if premium >= 2 else 2
		amount = min(count, amount)
		dups = ceil(amount / 2)
		eprompts = alist()
		if sdxl and not dalle and "r" not in flags and (not force or prompt.count(" ") < 32):
			oprompt = prompt
			uid = user.id
			temp = oprompt.replace('"""', "'''")
			prompt = f'### Instruction:\n"""\n{temp}\n"""\n\nImprove the above image caption as a description to send to txt2img image generation. Be as detailed as possible in at least 2 sentences, but stay concise!\n\n### Response:'
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
			resp = cdict(choices=[])
			if len(resp.choices) < dups:
				futs = []
				for i in range(min(3, max(1, dups // 2))):
					fut = csubmit(bot.instruct(
						dict(
							prompt=prompt,
							temperature=1,
							max_tokens=200,
							top_p=0.9,
							frequency_penalty=0.25,
							presence_penalty=0,
						),
						best=premium >= 3 and not (dups > 2 and not i),
						skip=not nsfw,
					))
					futs.append(fut)
				for fut in futs:
					try:
						s = await fut
						assert len(s.strip()) > 12
					except:
						print_exc()
						continue
					resp.choices.append(cdict(text=s))
			if not resp or len(resp.choices) < max(1, dups - 1):
				resp2 = await bot.llm(
					"chat.completions.create",
					model="gpt-3.5-turbo-0125",
					messages=[dict(role="user", content=prompt)],
					temperature=1,
					max_tokens=120,
					top_p=0.9,
					frequency_penalty=0.25,
					presence_penalty=0,
					user=str(user.id),
					n=max(1, dups - len(resp.choices) - 1),
				)
				if resp:
					resp.choices.extend(cdict(text=choice.message.content) for choice in reversed(resp2.choices))
				else:
					resp = resp2
			resp.choices.append(cdict(text=""))
			print("REWRITE:", resp)
			for choice in resp.choices:
				out = choice.text.strip()
				if bot.decensor.search(out):
					out = ""
				if out and out[0] == out[-1] == '"' and not oprompt[0] == oprompt[-1] == '"':
					try:
						out = str(literal_eval(out)).strip()
					except SyntaxError:
						pass
				out = regexp(r"^(?: [Pp]lease)?(?: [Gg]enerate| [Cc]reate)?(?: [Aa]n image (?:of|with|containing))? ").sub("", " " + regexp(r"[Tt]hank you[.!]$").sub("", out.strip().replace("txt2img", "art").removeprefix("art").removeprefix(":"))).strip(' \t\n,`"')
				if out:
					if not out[0].isupper() and " " in out:
						s, o = out.split(" ", 1)
						out = s.capitalize() + " " + o
					prompt = (nprompt or oprompt).removesuffix(".") + " BREAK " + out.strip()
				else:
					prompt = nprompt or oprompt
				eprompts.append(prompt)
		else:
			eprompts.extend([prompt] * dups)
		req = prompt
		print("PROMPT:", eprompts or prompt)
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
		emb = None
		fn = None
		futs = []
		pnames = []
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

		if not bot.verify_integrity(message):
			return

		async def ibasl_r(p, k, n, f, c, s, a, np):
			m = "--mask" in k and "--init-image" not in k
			if m or s and "z" not in flags:
				# try:
				# 	if not m and c <= 1:
				# 		await bot.lambdassert("sdcc")
				# except:
				# 	print_exc()
				# else:
				cap = "sdxl" if "--mask" in k or "--init-image" in k or "--nsfw" in k else "sdcc"
				return await process_image("IBASLR", "&", [p, k, n, f, c, a, np], cap=cap, timeout=420)
			resp = await process_image("IBASL", "&", [p, k, n, f, c, s, a, np, "z" in flags], cap="sdxl" if s else "sd", timeout=420)
			# if s and "z" not in flags:
			# 	out = []
			# 	for r1 in resp:
			# 		r2 = await process_image("IBASR", "$", [p, r1, 48, np], cap="sdcc", timeout=240)
			# 		out.append(r2)
			# 	return out
			return resp

		if amount2 < amount:
			if dalle:
				ar = float(kwargs["--aspect-ratio"]) or 1
				if url:
					raise NotImplementedError(f"Dall·E {dalle} interface currently does not support image prompts.")
				if dalle < 3:
					if max(ar, 1 / ar) < 1.1:
						size = "1024x1024"
					else:
						raise ValueError(f"Dall·E {dalle} interface currently only supports 1:1 aspect ratio.")
					async with discord.context_managers.Typing(channel):
						prompt = eprompts.next()
						prompt = lim_str(prompt, 1000)
						response = await bot.get_oai("images.generate")(
							model=f"dall-e-{dalle}",
							prompt=prompt,
							size=size,
							n=amount - amount2,
							user=str(user.id) if premium < 3 else str(hash(user.name)),
						)
						images = response.data
						pnames.extend([prompt] * len(images))
				else:
					if max(ar, 1 / ar) < 1.1:
						size = "1024x1024"
					elif max(ar / 1.75, 1.75 / ar) < 1.1:
						size = "1792x1024"
					elif max(ar * 1.75, 1 / 1.75 / ar) < 1.1:
						size = "1024x1792"
					else:
						raise ValueError(f"Dall·E {dalle} interface currently only supports 1:1, 7:4 and 4:7 aspect ratios.")
					q = "hd" if premium >= 6 and "s" in flags else "standard"
					async with discord.context_managers.Typing(channel):
						futn = []
						for i in range(amount - amount2):
							fut = csubmit(bot.get_oai("images.generate")(
								model=f"dall-e-{dalle}",
								prompt=lim_str(eprompts.next(), 4000),
								size=size,
								quality=q,
								n=1,
								style="natural" if i else "vivid",
								user=str(user.id) if premium < 3 else str(hash(user.name)),
							))
							futn.append(fut)
						images = []
						for fut in futn:
							try:
								response = await fut
							except openai.RateLimitError:
								print_exc()
								await asyncio.sleep(60)
								response = await bot.get_oai("images.generate")(
									model=f"dall-e-{dalle}",
									prompt=lim_str(eprompts.next(), 4000),
									size=size,
									quality="standard",
									n=1,
									style="natural",
									user=str(user.id) if premium < 3 else str(hash(user.name)),
								)
							except:
								print_exc()
								try:
									response = await bot.get_oai("images.generate")(
										model=f"dall-e-{dalle}",
										prompt=lim_str(eprompts.next(), 4000),
										size=size,
										quality="standard",
										n=1,
										style="natural",
										user=str(user.id) if premium < 3 else str(hash(user.name)),
									)
								except:
									if amount <= 1:
										raise
									print_exc()
									print("SKIPPED")
									continue
							images.append(response.data[0])
							pnames.append(response.data[0].revised_prompt)
				futs.extend(csubmit(Request(im.url, timeout=48, aio=True)) for im in images)
				amount2 += len(images)
		if amount2 < amount and not url:
			async with discord.context_managers.Typing(channel):
				futt = []
				c = 0
				async with self.sdiff_sem:
					noprompt = not force and not kwargs.get("--mask")
					c = min(amount, 9)
					c2 = c
					while c2 > 0:
						prompt = eprompts.next()
						p = "" if noprompt and not sdxl else prompt
						n = min(c2, round_random(amount / dups))
						if not n:
							n = c2
						fut = csubmit(ibasl_r(p, kwargs, nsfw, False, n, sdxl, aspect, negative))
						futt.append(fut)
						pnames.extend([prompt] * n)
						c2 -= n
						await asyncio.sleep(0.5)
					ims = []
					for fut in futt:
						try:
							ims2 = await fut
						except PermissionError:
							if not ims:
								raise
						if ims2:
							ims.extend(ims2)
				futs.extend(ims)
				amount2 = len(futs)
		if amount2 < amount:
			args = [
				python,
				"demo.py",
			]
			if not prompt and "--prompt" not in kwargs:
				prompt = eprompts.next()
				args.extend((
					"--prompt",
					prompt,
				))
			async with discord.context_managers.Typing(channel):
				image_1 = image_2 = None
				if url:
					resp = await process_image(url, "downsample", ["-nogif", 5, 1024 if sdxl else 768, "-bg", "-f", "png"], timeout=60)
					image_1 = resp
					if inpaint and url2:
						image_2 = await bot.get_request(url2)
					if inpaint and not url2:
						resp = await process_image(image_1, "get_mask", ["-nogif", "-nodel", "-f", "png"], timeout=60)
						image_2 = resp
						resp = await process_image(image_1, "inpaint", [image_2, "-nodel", "-f", "png"], timeout=60)
						image_1 = resp
						resp = await process_image(image_2, "expand_mask", ["-nogif", 12, "-f", "png"], timeout=60)
						image_2 = resp
						print(image_1, image_2)
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
							if morph:
								if "--strength" not in kwargs:
									args.extend((
										"--strength",
										"0.5",
									))
									kwargs["--strength"] = 0.5
								args.extend((
									"--mask",
									fn,
								))
								kwargs["--mask"] = fn
							else:
								if "--strength" not in kwargs:
									args.extend((
										"--strength",
										"0.6",
									))
									kwargs["--strength"] = 0.6
								args.extend((
									"--init-image",
									fn,
								))
								kwargs["--init-image"] = fn
								if image_2:
									fm = os.path.abspath(image_2) if isinstance(image_2, str) else image_2
									args.extend((
										"--mask",
										fm,
									))
									kwargs["--mask"] = fm
						for k, v in kwargs.items():
							args.extend((k, v))
						# print(args)
						noprompt = not force and not kwargs.get("--mask")
						futt = []
						c = amount - amount2
						c2 = c
						while c2 > 0:
							prompt = eprompts.next()
							p = "" if noprompt and not sdxl else prompt
							n = min(c2, round_random(amount / dups))
							if not n:
								n = c2
							fut = csubmit(ibasl_r(p, kwargs, nsfw, False, n, sdxl, aspect, negative))
							futt.append(fut)
							pnames.extend([prompt] * n)
							c2 -= n
							await asyncio.sleep(0.5)
						for fut in futt:
							ims = await fut
							futs.extend(ims)
							# print(dups, len(ims), len(futt), len(futs), amount, amount2)
						amount2 = len(futs)
		ffuts = []
		exc = RuntimeError("Unknown error occured.")
		async with discord.context_managers.Typing(channel):
			for tup, prompt in zip(futs, pnames):
				if len(ffuts) >= amount:
					break
				if not tup:
					continue
				if not isinstance(tup, tuple):
					if awaitable(tup):
						tup = await tup
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
					if is_url(fn):
						fn = await Request(fn, timeout=24, aio=True)
					else:
						with open(fn, "rb") as f:
							fn = f.read()
				ffut = csubmit(bot.commands.steganography[0].call(fn, str(bot.id)))
				ffut.prompt = prompt
				ffut.back = fn
				ffuts.append(ffut)
		if not ffuts:
			raise exc
		files = []
		for ffut in ffuts:
			prompt = ffut.prompt
			fn = ffut.back
			with tracebacksuppressor:
				fn = await ffut
			files.append(CompatFile(fn, filename=prompt + ".png", description=prompt))
		if premium >= 2 and freebies is not None:
			data = bot.data.users.setdefault(user.id, {})
			freebies = [t for t in data.get("freebies", ()) if utc() - t < 86400]
			freebies.extend((utc() - 1, utc()))
			data["freebies"] = freebies
			rem = freelim - len(freebies)
			print("REM:", user, rem)
			if not emb and rem in (27, 26, 9, 8, 3, 2, 1):
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = f"{rem}/{freelim} premium commands remaining today (free commands will be used after).\nIf you're able to contribute towards [funding my API]({bot.kofi_url}) hosting costs it would mean the world to us, and ensure that I can continue providing up-to-date tools and entertainment.\nEvery little bit helps due to the size of my audience, and you will receive access to unlimited and various improved commands as thanks!"
		if len(files) == 2:
			fe = files.pop(1)
			urls = await bot.data.exec.stash(fe, filename=fe.filename)
			if urls:
				comment = ("\n".join(url.split("?", 1)[0] for url in urls) + "\n" + comment).strip()
		return await send_with_react(channel, comment, files=files, reference=message, reacts="🔳", embed=emb)
		# await bot.send_with_file(channel, "", fn, filename=lim_str(prompt, 96) + ".png", reference=message, reacts="🔳", embed=emb)
