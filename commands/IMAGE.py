# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

from PIL import Image

spaces = ("rgb", "bgr", "cmy", "xyz", "hsv", "hsl", "hsi", "hcl", "hcy", "lab", "luv", "yuv")


class ColourDeficiency(Command):
	name = ["ColorBlind", "ColourBlind", "ColorBlindness", "ColourBlindness", "ColorDeficiency"]
	alias = name + ["Protanopia", "Protanomaly", "Deuteranopia", "Deuteranomaly", "Tritanopia", "Tritanomaly", "Achromatopsia", "Achromatonomaly"]
	description = "Applies a colourblindness filter to the target image."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("protanopia", "protanomaly", "deuteranopia", "deuteranomaly", "tritanopia", "tritanomaly", "achromatopsia", "achromatonomaly"),
				accepts=dict(protan="protanomaly", deutan="deuteranomaly", tritan="tritanomaly", achro="achromatonomaly"),
			),
			description="The colourblindness filter to apply",
			example="tritanopia",
			default="deuteranomaly",
		),
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		strength=cdict(
			type="number",
			validation="[0, 2]",
			description="The filter strength",
			example="0.9",
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (6, 10)
	_timeout_ = 3.5

	async def __call__(self, _timeout, mode, url, strength, filesize, format, **void):
		resp = await process_image(url, "colour_deficiency", [mode, strength, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class EdgeDetect(Command):
	description = "Applies an edge or depth detection algorithm to the image."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("canny", "laplacian", "depth"),
			),
			description="The algorithm to apply",
			example="depth",
			default="canny",
		),
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Laplacian=cdict(
			mode="laplacian",
		),
		Canny=cdict(
			mode="canny",
		),
		Depth=cdict(
			mode="depth",
		),
		DepthMap=cdict(
			mode="depth",
		),
	)
	rate_limit = (5, 7)
	_timeout_ = 3

	async def __call__(self, _timeout, mode, url, filesize, format, **void):
		func = "canny"
		cap = "caption"
		if mode in ("depth", "depthmap"):
			func = "depth"
			cap = "sdxl"
		elif mode == "laplacian":
			func = "laplacian"
			cap = "image"
		resp = await process_image(url, func, ["-fs", filesize, "-f", format], cap=cap, timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Blur(Command):
	description = "Applies a blur algorithm to the image."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("gaussian",),
			),
			description="The algorithm to apply",
			example="depth",
			default="gaussian",
		),
		strength=cdict(
			type="number",
			validation="[0, 1048576]",
			example="8",
			default=6,
		),
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Gaussian=cdict(
			mode="gaussian",
		),
	)
	rate_limit = (5, 7)
	_timeout_ = 3

	async def __call__(self, _timeout, mode, strength, url, filesize, format, **void):
		resp = await process_image(url, "blur_map", [[], None, None, mode, strength, "-fs", filesize, "-f", format], cap="image", timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class ColourSpace(Command):
	name = ["ColorSpace"]
	description = "Changes the colour space of the supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		source=cdict(
			type="enum",
			validation=cdict(
				enum=spaces,
			),
			description="The colourspace which should be treated as the image's original space",
			example="hsv",
			default="rgb",
		),
		dest=cdict(
			type="enum",
			validation=cdict(
				enum=spaces,
			),
			description="The colourspace which should be treated as the image's new space",
			example="rgb",
			default="hsv",
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (7, 11)
	_timeout_ = 4

	async def __call__(self, _timeout, url, source, dest, filesize, format, **void):
		resp = await process_image(url, "colourspace", [source, dest, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class GMagik(Command):
	description = "Applies the Magik image filter to supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			# aliases=["i"],
			required=True,
		),
		intensity=cdict(
			type="number",
			validation="[0, 256]",
			description="Distortion distance, relative to cell size",
			example="0.75",
			default=7,
		),
		cells=cdict(
			type="number",
			validation="(0, 256]",
			description="How many transform cells should tile each edge of the image",
			example="13.75",
			default=7,
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Magik=cdict(
			intensity=1,
			duration=0,
		),
		Liquefy=cdict(
			intensity=0.125,
			cells=256,
		),
	)
	rate_limit = (8, 12)
	_timeout_ = 7

	async def __call__(self, _timeout, url, intensity, cells, duration, speed, fps, filesize, format, **void):
		resp = await process_image(url, "magik_map", [[], float(duration) / speed, fps, intensity, cells], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Gradient(Command):
	description = "Generates a gradient with a specific shape."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("linear", "radial", "conical", "spiral", "polygon"),
			),
			description="Indicates the type of spiral to generate",
			example="radial",
			default="linear",
		),
		colour=cdict(
			type="colour",
			description="The colour of the gradient on one side",
			example="#BF7FFF",
			default=(255, 255, 255),
			aliases=["color"],
		),
		background=cdict(
			type="colour",
			description="The colour of the gradient on the other side",
			example="#000000",
			default=(0, 0, 0),
			aliases=["bg"],
		),
		repetitions=cdict(
			type="number",
			validation="[-256, 256]",
			description="How many times the gradient repeats across the image",
			example="4",
			default=1,
		),
		size=cdict(
			type="number",
			validation="[1, 65536]",
			description="Width of the output image",
			example="1536",
			default=1024,
		),
		colourspace=cdict(
			type="enum",
			validation=cdict(
				enum=spaces,
			),
			description="The colourspace used to produce the image",
			example="hsv",
			default="rgb",
			aliases=["space"],
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(IMAGE_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		RainbowGradient=cdict(
			colour=(0, 255, 255),
			background=(255, 255, 255),
			colourspace="hsv",
		),
	)
	rate_limit = (4, 6)
	slash = True

	async def __call__(self, _timeout, mode, colour, background, repetitions, size, colourspace, filesize, format, **void):
		resp = await process_image("from_gradient", "$", [mode, colour, background, repetitions, size, colourspace, "-fs", filesize, "-f", format], timeout=_timeout)
		name = "Gradient." + get_ext(resp)
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Average(Command):
	name = ["AverageColour"]
	description = "Computes the average pixel colour in RGB for the supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(IMAGE_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (5, 7)
	_timeout_ = 2

	async def __call__(self, bot, url, filesize, format, **void):
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
		resp = await process_image("from_colour", "$", [channels, "-fs", filesize, "-f", format])
		name = "Average." + get_ext(resp)
		return cdict(content=msg, file=CompatFile(resp, filename=name), reacts="🔳")


class QR(Command):
	description = "Creates a QR code image from an input string, optionally adding a rainbow swirl effect."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("normal", "rainbow"),
			),
			description="Indicates whether to apply an animated rainbow filter to the output",
			example="rainbow",
			default="normal",
		),
		text=cdict(
			type="string",
			description="Text to encode",
			example="You found the funny!",
			aliases=["i"],
			required=True,
		),
		arms=cdict(
			type="integer",
			description="Number of spiral arms, for rainbow mode",
			example="5",
			default=2,
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		RainbowQR=cdict(
			mode="rainbow",
		),
	)
	rate_limit = (8, 11)
	_timeout_ = 4
	slash = True

	async def __call__(self, _timeout, mode, text, arms, duration, speed, fps, filesize, format, **void):
		duration = duration if mode != "normal" else 0
		resp = await process_image("to_qr", "$", [text, arms, float(duration) / speed, fps, "-fs", filesize, "-f", format], timeout=_timeout)
		name = "QR." + get_ext(resp)
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Rainbow(Command):
	name = ["Gay"]
	description = "Creates an animation from repeatedly hueshifting supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("rgb", "hsv", "hsi", "hcl", "hcy"),
			),
			description="The colour space to perform hueshifts in. Changes the way chroma/saturation is treated",
			example="hcl",
			default="hsv",
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (10, 13)
	_timeout_ = 4
	slash = True

	async def __call__(self, _timeout, url, mode, duration, speed, fps, filesize, format, **void):
		resp = await process_image(url, "rainbow_map", [[], float(duration) / speed, fps, mode, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Scroll(Command):
	name = ["Parallax"]
	description = "Creates an animation from repeatedly shifting supplied image in a specified direction."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		direction=cdict(
			type="enum",
			validation=cdict(
				enum=("left", "up", "right", "down"),
				accepts=dict(l="left", u="up", r="right", d="down", v="up", h="left", vertical="up", horizontal="left"),
			),
			description="The direction of motion",
			example="down",
			default="left",
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (10, 13)
	_timeout_ = 4
	slash = True

	async def __call__(self, _timeout, url, direction, duration, speed, fps, filesize, format, **void):
		resp = await process_image(url, "scroll_map", [[], float(duration) / speed, fps, direction, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Spin(Command):
	description = "Creates an animation from repeatedly rotating supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		angle=cdict(
			type="number",
			validation="[0, 360)",
			description="Initial orientation of the image, in counterclockwise degrees",
			example="90",
			default=0,
		),
		circle=cdict(
			type="bool",
			description="Indicates whether to crop the image to a circle",
			example="False",
			default=True,
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Rotate=cdict(
			duration=0,
			circle=False,
		),
	)
	rate_limit = (10, 13)
	_timeout_ = 4
	slash = True

	async def __call__(self, _timeout, url, angle, circle, duration, speed, fps, filesize, format, **void):
		resp = await process_image(url, "spin_map", [[], float(duration) / speed, fps, angle, circle, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Orbit(Command):
	name = ["Orbital", "Orbitals"]
	description = "Renders a ring of orbiting sprites of the supplied image."
	schema = cdict(
		urls=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
			multiple=True,
		),
		count=cdict(
			type="integer",
			validation="[1, 256]",
			description="The amount of each image to include in the ring",
			example="5",
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=2,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (16, 22)
	_timeout_ = 13

	async def __call__(self, _timeout, urls, count, duration, speed, fps, filesize, format, **void):
		url = urls.pop(0)
		resp = await process_image(url, "orbit_map", [urls, float(duration) / speed, fps, count, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Pet(Command):
	name = ["PetPet", "Attack", "Pat"]
	description = "Creates an animation from applying the Petpet generator to the supplied image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		squish=cdict(
			type="number",
			validation="[-0.5, 0.5]",
			description="Maximum and minimum stretch factor",
			example="1/7",
			default=0.1,
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=0.25,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (10, 13)
	_timeout_ = 5
	slash = True

	async def __call__(self, _timeout, url, squish, duration, speed, fps, filesize, format, **void):
		resp = await process_image(url, "pet_map", [[], float(duration) / speed, fps, squish, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Tesseract(Command):
	name = ["4D", "Octachoron", "Hypercube"]
	description = "Creates an animation from applying the image as a tesseract texture and rotating."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		size=cdict(
			type="integer",
			validation="[64, 1024]",
			description="Maximum and minimum stretch factor",
			example="256",
			default=512,
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=0.25,
		),
		speed=cdict(
			type="number",
			validation="[-60, 60]",
			description="Inverse of duration; higher values reduce the duration",
			example="3.[3]",
			default=1,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=30,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (10, 13)
	_timeout_ = 5
	slash = True

	def tesseract(self, data, size=512, timeout=30):
		if isinstance(data, str):
			fi = data
		else:
			fi = f"cache/{ts_us()}.png"
			with open(fi, "wb") as f:
				f.write(data)
		fn = f"cache/{ts_us()}.tar"
		page = new_playwright_page()
		with page:
			page.goto(f"https://api.mizabot.xyz/static/tesseract.html?size={size}&texture=:")
			with page.expect_file_chooser() as fc_info:
				page.locator("#inp").click()
			file_chooser = fc_info.value
			file_chooser.set_files(fi)
			with page.expect_download(timeout=timeout * 1000) as download_info:
				page.locator("canvas").click()
			download = download_info.value
			download.save_as(fn)
		return fn

	async def __call__(self, _timeout, url, size, duration, speed, fps, filesize, format, **void):
		data = await process_image(url, "resize_max", ["-nogif", size, 0, "auto", "-f", "png"], timeout=60)
		resp = await asubmit(self.tesseract, data, size, timeout=_timeout)
		resp = await process_image(resp, "resize_map", [[], duration, fps, "mult", 1, 1, "nearest", None, "-fs", filesize, "-f", format], cap="image", timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Resize(Command):
	name = ["Scale", "Rescale"]
	description = "Changes size of supplied image, using an optional scaling operation."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "crop", "nearest", "linear", "area", "scale2x", "hamming", "gauss", "cubic", "spline", "sinc", "lanczos"),
				accepts=dict(bilinear="linear", bicubic="cubic")
			),
			description="Mode used to resize image",
			example="scale2x",
			default="auto",
		),
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		resolution=cdict(
			type="resolution",
			validation="[1, 65536]",
			description="Width and height of output, in pixels",
			example="1920x1080",
			aliases=["size"],
		),
		multiplier=cdict(
			type="number",
			validation="(0, 1920]",
			description="Multiplies width and height by a constant factor",
			example="3",
			default=1,
		),
		area=cdict(
			type="number",
			validation="(0, 4294967296]",
			description="Desired area in square pixels; will automatically scale current aspect ratio",
			example="2073600",
		),
		duration=cdict(
			type="timedelta",
			validation="[-3600, 3600]",
			description="The duration of the animation (auto-syncs if the input is animated, negative values reverse the animation)",
			example="1:26.3",
			default=None,
		),
		fps=cdict(
			type="number",
			validation="(0, 256]",
			description="The framerate of the animation (does not affect duration)",
			example="120/7",
			default=None,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Scale2x=cdict(
			mode="scale2x",
		),
		Jumbo=cdict(
			mode="nearest",
		),
		GIF=cdict(
			format="gif",
		),
		WebP=cdict(
			format="webp",
		),
		AVIF=cdict(
			format="avif",
		),
	)
	rate_limit = (8, 13)
	_timeout_ = 4
	slash = True

	async def __call__(self, bot, _timeout, mode, url, resolution, multiplier, area, duration, fps, filesize, format, **void):
		if resolution:
			if len(resolution) == 1:
				x = y = resolution[0]
				func = "rel"
			else:
				x, y = resolution
				func = "set"
			x *= multiplier
			y *= multiplier
		else:
			x = y = multiplier
			func = "mult"
		resp = url
		if func != "mult" or not x == y == 1 or area or duration is not None or fps is not None:
			resp = await process_image(resp, "resize_map", [[], duration, fps, func, x, y, mode, area, "-fs", filesize, "-f", format], cap="image", timeout=_timeout)
		else:
			resp = await bot.optimise_image(resp, fsize=filesize, fmt=format)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Crop(Command):
	name = ["Cut"]
	description = "Crops the supplied image to a given size, expanding if necessary. Use \"-\" to omit measurements."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		left=cdict(
			type="integer",
			validation="[-16384, 16384]",
			description="Left edge of crop in pixels, absolute",
			example="10",
			default="-",
		),
		top=cdict(
			type="integer",
			validation="[-16384, 16384]",
			description="Top edge of crop in pixels, absolute",
			example="-",
			default="-",
		),
		right=cdict(
			type="integer",
			validation="[0, 32768]",
			description="Right edge of crop in pixels, absolute",
			example="20",
			default="-",
		),
		bottom=cdict(
			type="integer",
			validation="[0, 32768]",
			description="Bottom edge of crop in pixels, absolute",
			example="15",
			default="-",
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (4, 9)
	_timeout_ = 4
	slash = True

	async def __call__(self, _timeout, url, left, top, right, bottom, filesize, format, **void):
		resp = await process_image(url, "crop_map", [[], None, None, left, top, right, bottom, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Adjust(Command):
	description = "Adjusts an optional amount of channels in the target image with a given operation and optional value."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		channels=cdict(
			type="enum",
			validation=cdict(
				enum=("r", "g", "b", "h", "s", "c", "v", "l", "i", "y", "a"),
				accepts=cdict(red="r", green="g", blue="b", hue="h", saturation="s", chroma="c", value="s", lightness="l", intensity="i", luma="y", alpha="a"),
			),
			description="Target colour channel(s)",
			example="gb",
			default="rgb",
			multiple=True,
		),
		operation=cdict(
			type="enum",
			validation=cdict(
				enum=("=","+", "-", "*", "/", "**", "%", "&", "|", "^"),
				accepts={"set":"=","replace":"=","add":"+","additive":"+","sub":"-","subtract":"-","subtractive":"-","mul":"*","multiply":"*","multiplicative":"*","div":"/","divide":"/","divisive":"/","pow":"**","power":"**","mod":"%","modulo":"%","and":"&","or":"|","xor":"^"},
			),
			description="Operation to perform",
			example="multiply",
			default="=",
		),
		value=cdict(
			type="number",
			validation="[0, 1]",
			description="Value to set, 32-bit float",
			example="0.75",
			default="0.5",
		),
		clip=cdict(
			type="bool",
			description="Clips out-of-bound values. Wraps values if not set",
			example="False",
			default=True,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	macros = cdict(
		Fill=cdict(
			operation="=",
		),
		HueShift=cdict(
			operation="+",
			channels="h",
			clip=False,
		),
		Invert=cdict(
			operation="^",
			channels="rgb",
			value="1",
		),
		GreyScale=cdict(
			operation="=",
			channels="c",
			value=0,
		),
		Lightness=cdict(
			operation="+",
			channels="l",
		),
		Lighten=cdict(
			operation="+",
			channels="l",
		),
		Luma=cdict(
			operation="+",
			channels="y",
		),
		Brightness=cdict(
			operation="+",
			channels="y",
		),
		Brighten=cdict(
			operation="+",
			channels="y",
		),
	)
	rate_limit = (7, 10)
	_timeout_ = 4

	async def __call__(self, _timeout, url, channels, operation, value, clip, filesize, format, **void):
		resp = await process_image(url, "adjust_map", [[], None, None, operation, value, channels, clip, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Blend(Command):
	description = "Combines the two supplied images, using an optional blend operation."
	schema = cdict(
		urls=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=2,
			multiple=True,
		),
		operation=cdict(
			type="enum",
			validation=cdict(
				enum=("blend", "replace", "add", "sub", "mul", "div", "mod", "and", "or", "xor", "nand", "nor", "xnor", "difference", "overlay", "screen", "soft", "hard", "lighten", "darken", "plusdarken", "overflow", "lighting", "burn", "linearburn", "dodge", "hue", "saturation", "lightness", "lum", "value", "colour", "extract", "merge", "alpha"),
				accepts={"normal": "blend", "blt": "blend", "blit": "blend", "+": "add", "addition": "add", "additive": "add", "-": "sub", "subtract": "sub", "subtraction": "sub", "subtractive": "sub", "*": "mul", "multiply": "mul", "mult": "mul", "multiplication": "mul", "multiplicative": "mul", "/": "div", "divide": "div", "division": "div", "divisive": "div", "modulo": "mod", "%": "mod", "&": "and", "|": "or", "^": "xor", "~&": "nand", "~|": "nor", "~^": "xnor", "diff": "difference", "softlight": "soft", "hardlight": "hard","lighter": "lighten", "darker": "darken", "plusdarker": "plusdarken", "grainextract": "extract", "grainmerge": "merge", "colorburn": "burn", "colourburn": "burn", "colordodge": "dodge", "colourdodge": "dodge", "lineardodge": "add", "sat": "saturation", "brightness": "lightness", "luminosity": "lum", "luminance": "lum", "val": "value", "color": "colour"},
			),
			description="Blend operation to perform",
			example="multiply",
			default="blend",
		),
		opacity=cdict(
			type="number",
			validation="[0, 1]",
			description="Opacity of the second image",
			example="0.75",
			default=0.5,
		),
		filesize=cdict(
			type="filesize",
			validation="[1024, 1073741824]",
			description="The maximum filesize in bytes",
			example="10kb",
			default=CACHE_FILESIZE,
			aliases=["fs"],
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(VISUAL_FORMS),
				accepts={k: v for k, v in CODECS.items() if v in VISUAL_FORMS},
			),
			description="The file format or codec of the output",
			example="mp4",
			default="auto",
		),
	)
	rate_limit = (13, 17)
	flags = "l"
	_timeout_ = 7
	maintenance = True

	async def __call__(self, _timeout, urls, operation, opacity, filesize, format, **void):
		url = urls.pop(0)
		resp = await process_image(url, "blend_map", [urls, None, None, operation, opacity, "-fs", filesize, "-f", format], timeout=_timeout)
		fn = url2fn(url)
		name = replace_ext(fn, get_ext(resp))
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class Steganography(Command):
	name = ["Ectoplasm", "Watermark", "Copyright", "Ownership", "©"]
	description = "Reads an image's tagged information, or embeds a message into an image (input a user ID to tag as a Discord user). Aborts if image already has a known tag."
	schema = cdict(
		url=cdict(
			type="image",
			description="Image supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
		data=cdict(
			type="string",
			description="Message to encode",
			example="This image was produced by X in collaboration with Y",
		),
	)
	rate_limit = (12, 15)
	_timeout_ = 6

	async def __call__(self, bot, _user, url, data, **void):
		resp = await process_image("ectoplasm", "$", ["-nogif", url, data, "-f", "png"], cap="caption", priority=True, timeout=60)
		if isinstance(resp, bytes):
			msg = resp[1:]
			emb = discord.Embed()
			emb.set_title("Detected Tag")
			emb.description = msg.decode("utf-8", "replace")
			return cdict(embed=emb)
		fn = url2fn(url)
		name = replace_ext(fn, "png")
		return cdict(file=CompatFile(resp, filename=name), reacts="🔳")


class OCR(Command):
	name = ["Read", "Image2Text"]
	description = "Attempts to read text in an image using Optical Character Recognition AI."
	schema = cdict(
		url=cdict(
			type="image",
			description="Image supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
	)
	rate_limit = (10, 15)
	slash = ("Read")
	ephemeral = True

	async def __call__(self, bot, _user, url, **void):
		s = await bot.ocr(url)
		return cdict(
			embed=discord.Embed(description=s, title="Detected text").set_author(**get_author(_user)),
		)