# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

from dynamic_dt import TIMEZONES, get_timezone
import pytz


class Math(Command):
	_timeout_ = 4
	name = ["🔢", "M", "PY", "Sympy", "Plot", "Calc"]
	description = "Evaluates a math formula."
	limit = 1073741824
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("default", "list_vars", "clear_vars", "plot"),
			),
			description="Alternative actions to perform",
			example="list_vars",
			default="default",
		),
		query=cdict(
			type="string",
			description="Input expression or formula, in Sympy syntax",
			example="lim(diff(-atan(x)), x=-sqrt(-1.02))",
			required=True,
			aliases=["expression"],
		),
		precision=cdict(
			type="integer",
			validation=f"(0, {limit}]",
			description="Floating point calculation precision. Also affects accuracy of scientific notation and recurring decimals",
			example="8000",
			default=1024,
		),
		rationalise=cdict(
			type="bool",
			description="Whether to simplify/rationalise results where possible.",
			default=False,
		)
	)
	rate_limit = (4.5, 6)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _user, _premium, mode, query, precision, rationalise, **void):
		env = bot.get_userbase(_user.id, "variables", {})
		if mode == "list_vars":
			if not env:
				output = f"No currently assigned variables for {sqr_md(_user)}."
				return cdict(content=output, prefix="```ini\n", suffix="```")
			output = f"Currently assigned variables for {_user}:{iter2str(env)}"
			return cdict(content=output, prefix="```ini\n", suffix="```")
		if mode == "clear_vars":
			bot.pop_userbase(_user.id, "variables")
			output = f"Successfully cleared all variables for {sqr_md(_user)}."
			return cdict(content=output, prefix="*```css\n", suffix="```*")
		var = None
		if mode == "plot":
			query = f"{mode}({query})"
		else:
			for equals in ("=", ":="):
				if equals in query:
					ii = query.index(equals)
					for i, c in enumerate(query):
						if i >= ii:
							temp = query[i + len(equals):]
							if temp.startswith("="):
								break
							check = query[:i].strip().replace(" ", "")
							if check.isnumeric():
								break
							var = check
							query = temp.strip()
							break
						elif not (c.isalnum() or c in " _"):
							break
					if var is not None:
						break
		timeout = 240 if _premium.value >= 3 else 30
		resp = await bot.solve_math(query, precision, rationalise, timeout=timeout, variables=env)
		# Determine whether output is a direct answer or a file
		if type(resp) is dict and "file" in resp:
			fn = resp["file"]
			f = CompatFile(fn, filename=query + ".png")
			return cdict(file=f)
		assert resp, "Response empty."
		answer = "\n".join(a for a in map(str, resp) if a != query.strip()) or str(resp[0])
		if var is not None:
			env[var] = resp[0]
			while len(env) > 64:
				env.pop(next(iter(env)))
			bot.set_userbase(_user.id, "variables", env)
			s = lim_str(f"Variable {sqr_md(var)} set to {sqr_md(resp[0])}.", self.limit, mode="left")
			return cdict(content=s, prefix="```css\n", suffix="```")
		s = lim_str(f"{query} = {answer}", self.limit, mode="left")
		return cdict(content=s, prefix="```py\n", suffix="```")


class Unicode(Command):
	name = [
		"Uni2Hex", "U2H", "HexEncode",
		"Hex2Uni", "H2U", "HexDecode",
		"Uni2Bin", "U2B", "BinEncode",
		"Bin2Uni", "B2U", "BinDecode",
		"Uni2B64", "U64", "B64Encode",
		"B642Uni", "64U", "B64Decode",
		"Uni2S64", "S64", "S64Encode",
		"S642Uni", "64S", "S64Decode",
		"Uni2B32", "U32", "B32Encode",
		"B322Uni", "32U", "B32Decode",
		"Uni2A85", "A85", "A85Encode",
		"A852Uni", "85A", "A85Decode",
		"Uni2B85", "B85", "B85Encode",
		"B852Uni", "85B", "B85Decode",
	]
	description = "Converts unicode text to hexadecimal or binary numbers."
	usage = "<string>"
	example = ("u2h test", "uni2bin this is a secret message", "32u NRXWY")
	rate_limit = (3.5, 5)
	ephemeral = True

	def __call__(self, argv, name, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		if name in ("uni2hex", "u2h", "hexencode"):
			b = bytes2hex(argv.encode("utf-8"))
			return fix_md(b)
		if name in ("hex2uni", "h2u", "hexdecode"):
			b = as_str(hex2bytes(regexp(r"[^0-9A-Fa-f]+").sub("", argv).replace("0x", "")))
			return fix_md(b)
		if name in ("uni2bin", "u2b", "binencode"):
			b = " ".join(f"{ord(c):08b}" for c in argv)
			return fix_md(b)
		if name in ("bin2uni", "b2u", "bindecode"):
			b = to_alphanumeric(argv).replace("0x", "").replace(" ", "").encode("ascii")
			b = (np.frombuffer(b, dtype=np.uint8) - 48).astype(bool)
			if len(b) & 7:
				a = np.zeros(8 - len(b) % 8, dtype=bool)
				if len(b) < 8:
					b = np.append(a, b)
				else:
					b = np.append(b, a)
			a = np.zeros(len(b) >> 3, dtype=np.uint8)
			for i in range(8):
				c = b[i::8]
				if i < 7:
					c = c.astype(np.uint8)
					c <<= 7 - i
				a += c
			b = as_str(a.tobytes())
			return fix_md(b)
		if name in ("uni2b64", "u64", "b64encode"):
			b = as_str(bytes2b64(argv).rstrip(b"="))
			return fix_md(b)
		if name in ("b642uni", "64u", "b64decode"):
			b = as_str(b642bytes(argv))
			return fix_md(b)
		if name in ("uni2s64", "s64", "s64encode"):
			b = as_str(bytes2b64(argv, alt_char_set=True).rstrip(b"="))
			return fix_md(b)
		if name in ("s642uni", "64s", "s64decode"):
			b = as_str(b642bytes(argv, alt_char_set=True))
			return fix_md(b)
		if name in ("uni2b32", "u32", "b32encode"):
			b = as_str(base64.b32encode(argv.encode("utf-8")).rstrip(b"="))
			return fix_md(b)
		if name in ("b322uni", "32u", "b32decode"):
			b = unicode_prune(argv).encode("utf-8")
			if len(b) & 7:
				b += b"=" * (8 - len(b) % 8)
			b = as_str(base64.b32decode(b))
			return fix_md(b)
		if name in ("uni2a85", "a85", "a85encode"):
			b = as_str(base64.a85encode(argv.encode("utf-8")))
			return fix_md(b)
		if name in ("a852uni", "85a", "a85decode"):
			b = unicode_prune(argv).encode("utf-8")
			try:
				b = base64.a85decode(b)
			except ValueError:
				ba = bytearray()
				for i in range(0, len(b), 5):
					try:
						bi = base64.a85decode(b[i:i + 5])
					except ValueError:
						pass
					else:
						ba.extend(bi)
				if not ba:
					raise
				print_exc()
				b = ba
			b = as_str(b)
			return fix_md(b)
		if name in ("uni2b85", "b85", "b85encode"):
			b = as_str(base64.b85encode(argv.encode("utf-8")))
			return fix_md(b)
		if name in ("b852uni", "85b", "b85decode"):
			b = unicode_prune(argv).encode("utf-8")
			try:
				b = base64.b85decode(b)
			except ValueError:
				ba = bytearray()
				for i in range(0, len(b), 5):
					try:
						bi = base64.b85decode(b[i:i + 5])
					except ValueError:
						pass
					else:
						ba.extend(bi)
				if not ba:
					raise
				print_exc()
				b = ba
			b = as_str(b)
			return fix_md(b)
		b = shash(argv)
		return fix_md(b)


class ID2Time(Command):
	name = ["I2T", "CreateTime", "Time2ID", "T2I"]
	description = "Converts a discord ID to its corresponding UTC time."
	schema = cdict(
		id=cdict(
			type="number",
			example="201548633244565504",
			excludes=("time",),
		),
		time=cdict(
			type="datetime",
			example="35 minutes and 6.25 seconds before 3am next tuesday, EDT",
			excludes=("id",),
		),
	)
	rate_limit = (3, 4)
	ephemeral = True

	def __call__(self, id, time, **void):
		if not id and not time:
			raise ArgumentError("Input string is empty.")
		if id:
			return fix_md(snowflake_time(id))
		return fix_md(time_snowflake(time))


class Fancy(Command):
	name = ["Chaos"]
	description = "Creates fun string translations using unicode fonts."
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("fancy", "zalgo", "format"),
			),
			default="fancy",
		),
		text=cdict(
			type="string",
			example="This is a cool message",
			required=True,
		)
	)
	macros = cdict(
		Zalgo=cdict(
			mode="zalgo",
		),
		Format=cdict(
			mode="format",
		),
	)
	rate_limit = (4, 5)
	slash = ("Fancy", "Zalgo", "Format")
	ephemeral = True

	chrs = [chr(n) for n in zalgo_map]
	randz = lambda self: choice(self.chrs)
	def zalgo(self, s, x):
		if unfont(s) == s:
			return "".join(c + self.randz() for c in s)
		return s[0] + "".join("".join(self.randz() + "\u200b" for i in range(x + 1 >> 1)) + c + "\u200a" + "".join(self.randz() + "\u200b" for i in range(x >> 1)) for c in s[1:])
	formats = "".join(chr(i) for i in (0x30a, 0x325, 0x303, 0x330, 0x30c, 0x32d, 0x33d, 0x353, 0x35b, 0x20f0))

	def __call__(self, mode, text, **void):
		fields = deque()
		match mode:
			case "fancy":
				for i in range(len(UNIFMTS) - 1):
					s = uni_str(text, i)
					if i == len(UNIFMTS) - 2:
						s = s[::-1]
					fields.append((f"Font {i + 1}", s + "\n"))
			case "format":
				for i, f in enumerate(self.formats):
					s = "".join(c + f for c in text)
					fields.append((f"Format {i}", s + "\n"))
				s = "".join("_" if c in " _" else c if c in "gjpqy" else c + chr(818) for c in text)
				fields.append((f"Format {i + 1}", s))
			case "zalgo":
				for i in range(1, 9):
					s = self.zalgo(text, i)
					fields.append((f"Level {i}", s + "\n"))
			case _:
				raise NotImplementedError(mode)
		return cdict(embed=add_embed_fields(discord.Embed().set_author(name=lim_str(text, 256)), fields=fields))


class UnFancy(Command):
	name = ["UnFormat", "UnZalgo", "Deobfuscate", "Unobscure"]
	description = "Removes unicode formatting and diacritic characters from inputted text."
	schema = cdict(
		text=cdict(
			type="string",
			example=zwremove("𝔱𝔥𝔦𝔰 𝔦𝔰 𝔞 𝔠𝔬𝔬𝔩 𝔪𝔢𝔰𝔰𝔞𝔤𝔢​"),
			required=True,
		)
	)
	rate_limit = (4, 5)
	slash = True
	ephemeral = True

	def __call__(self, text, **void):
		return fix_md(unicode_prune(text))


class OwOify(Command):
	omap = {
		"r": "w",
		"R": "W",
		"l": "w",
		"L": "W",
	}
	otrans = "".maketrans(omap)
	name = ["UwU", "OwO", "UwUify"]
	description = "Applies the owo/uwu text filter to a string."
	usage = "<string> <aggressive(-a)>? <basic(-b)>?"
	example = ("owoify hello, what's your name?", "owoify -a Greetings, this is your cat god speaking")
	rate_limit = (4, 5)
	flags = "ab"
	ephemeral = True

	def __call__(self, argv, flags, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		out = argv.translate(self.otrans)
		temp = None
		if "a" in flags:
			out = out.replace("v", "w").replace("V", "W")
		if "a" in flags or "b" not in flags:
			temp = list(out)
			for i, c in enumerate(out):
				if i > 0 and c in "yY" and out[i - 1].casefold() not in "aeioucdhvwy \n\t":
					if c.isupper():
						temp[i] = "W" + c.casefold()
					else:
						temp[i] = "w" + c
				if i < len(out) - 1 and c in "nN" and out[i + 1].casefold() in "aeiou":
					temp[i] = c + "y"
			if "a" in flags and "b" not in flags:
				out = "".join(temp)
				temp = list(out)
				for i, c in enumerate(out):
					if i > 0 and c.casefold() in "aeiou" and out[i - 1].casefold() not in "aeioucdhvwy \n\t":
						if c.isupper():
							temp[i] = "W" + c.casefold()
						else:
							temp[i] = "w" + c
		if temp is not None:
			out = "".join(temp)
			if "a" in flags:
				for c in " \n\t":
					if c in out:
						spl = out.split(c)
						for i, w in enumerate(spl):
							if w.casefold().startswith("th"):
								spl[i] = ("D" if w[0].isupper() else "d") + w[2:]
							elif "th" in w:
								spl[i] = w.replace("th", "ff")
						out = c.join(spl)
		return fix_md(out)


class AltCaps(Command):
	description = "Alternates the capitalization on characters in a string."
	schema = cdict(
		text=cdict(
			type="string",
			example="that's what she said",
			required=True,
		),
	)
	rate_limit = (1, 2)
	ephemeral = True

	def __call__(self, text, **void):
		text = unicode_prune(text)
		i = text[0].isupper()
		a = text[i::2].lower()
		b = text[1 - i::2].upper()
		if i:
			a, b = b, a
		if len(a) > len(b):
			c = a[-1]
			a = a[:-1]
		else:
			c = ""
		return fix_md("".join(i[0] + i[1] for i in zip(a, b)) + c)


class Obfuscate(Command):
	name = ["Obscure"]
	description = "Obfuscates English text by substituting identical look-alikes with a unicode table."
	schema = cdict(
		text=cdict(
			type="string",
			example="that's what she said",
			required=True,
		),
	)
	rate_limit = (1, 2)
	ephemeral = True

	def __call__(self, text, **void):
		return fix_md(obfuscate(unicode_prune(text)))


class Invisicode(Command):
	description = "Encodes arbitrary text or data in the invisicode format. See https://github.com/thomas-xin/invisicode for more info, or to run it yourself!"
	schema = cdict(
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("encode", "decode"),
			),
			default="encode",
		),
		url=cdict(
			type="url",
			description="File to encode or decode",
		),
		text=cdict(
			type="string",
			description="Text to encode or decode",
			example="Hello World! ❤️",
			autoresolve=True,
		),
	)
	macros = cdict(
		Invisencode=cdict(
			mode="encode",
		),
		Invisiencode=cdict(
			mode="encode",
		),
		Invisdecode=cdict(
			mode="decode",
		),
		Invisidecode=cdict(
			mode="decode",
		),
	)
	rate_limit = (1, 2)
	ephemeral = True

	async def __call__(self, mode, text, url, **void):
		if not text and not url:
			raise ArgumentError("Please input text or a URL to encode/decode.")
		if url:
			data = await attachment_cache.download(url, read=False)
		else:
			data = text

		def _invisicode(mode, data):
			match mode:
				case "encode":
					return [invisicode.encode(data)]
				case "decode":
					return invisicode.detect_and_decode(as_str(data))
			return []

		blocks = await _run_async(_invisicode, mode, data)
		blocks = list(filter(bool, (block.strip() for block in blocks)))
		if not blocks:
			raise EOFError("Output was empty.")
		texts = []
		files = []
		for block in blocks:
			if isinstance(block, byte_like):
				files.append(CompatFile(block))
			else:
				texts.append(block)
		return cdict(
			content="\n\n".join(texts),
			files=files,
		)


# Char2Emoj, a simple script to convert a string into a block of text
def _c2e(string, em1, em2):
	chars = {
		" ": [0, 0, 0, 0, 0],
		"_": [0, 0, 0, 0, 7],
		"!": [2, 2, 2, 0, 2],
		'"': [5, 5, 0, 0, 0],
		":": [0, 2, 0, 2, 0],
		";": [0, 2, 0, 2, 4],
		"~": [0, 5, 7, 2, 0],
		"#": [10, 31, 10, 31, 10],
		"$": [7, 10, 6, 5, 14],
		"?": [6, 1, 2, 0, 2],
		"%": [5, 1, 2, 4, 5],
		"&": [4, 10, 4, 10, 7],
		"'": [2, 2, 0, 0, 0],
		"(": [2, 4, 4, 4, 2],
		")": [2, 1, 1, 1, 2],
		"[": [6, 4, 4, 4, 6],
		"]": [3, 1, 1, 1, 3],
		"|": [2, 2, 2, 2, 2],
		"*": [21, 14, 4, 14, 21],
		"+": [0, 2, 7, 2, 0],
		"=": [0, 7, 0, 7, 0],
		",": [0, 0, 3, 3, 4],
		"-": [0, 0, 7, 0, 0],
		".": [0, 0, 3, 3, 0],
		"/": [1, 1, 2, 4, 4],
		"\\": [4, 4, 2, 1, 1],
		"@": [14, 17, 17, 17, 14],
		"0": [7, 5, 5, 5, 7],
		"1": [3, 1, 1, 1, 1],
		"2": [7, 1, 7, 4, 7],
		"3": [7, 1, 7, 1, 7],
		"4": [5, 5, 7, 1, 1],
		"5": [7, 4, 7, 1, 7],
		"6": [7, 4, 7, 5, 7],
		"7": [7, 5, 1, 1, 1],
		"8": [7, 5, 7, 5, 7],
		"9": [7, 5, 7, 1, 7],
		"A": [2, 5, 7, 5, 5],
		"B": [6, 5, 7, 5, 6],
		"C": [3, 4, 4, 4, 3],
		"D": [6, 5, 5, 5, 6],
		"E": [7, 4, 7, 4, 7],
		"F": [7, 4, 7, 4, 4],
		"G": [7, 4, 5, 5, 7],
		"H": [5, 5, 7, 5, 5],
		"I": [7, 2, 2, 2, 7],
		"J": [7, 1, 1, 5, 7],
		"K": [5, 5, 6, 5, 5],
		"L": [4, 4, 4, 4, 7],
		"M": [17, 27, 21, 17, 17],
		"N": [9, 13, 15, 11, 9],
		"O": [2, 5, 5, 5, 2],
		"P": [7, 5, 7, 4, 4],
		"Q": [4, 10, 10, 10, 5],
		"R": [6, 5, 7, 6, 5],
		"S": [3, 4, 7, 1, 6],
		"T": [7, 2, 2, 2, 2],
		"U": [5, 5, 5, 5, 7],
		"V": [5, 5, 5, 5, 2],
		"W": [17, 17, 21, 21, 10],
		"X": [5, 5, 2, 5, 5],
		"Y": [5, 5, 2, 2, 2],
		"Z": [7, 1, 2, 4, 7],
	}
	printed = [em2] * 7
	string = string.upper()
	for i, c in enumerate(string):
		data = chars.get(c, [15] * 5)
		size = max(1, max(data))
		lim = max(2, int(log2(size))) + 1
		printed[0] += em2 * lim
		printed[6] += em2 * lim
		if len(data) == 5:
			for y in range(5):
				for p in range(lim):
					if (data[y] >> (lim - 1 - p)) & 1:
						printed[y + 1] += em1
					else:
						printed[y + 1] += em2
		for x in range(len(printed)):
			printed[x] += em2
	return printed

class Char2Emoji(Command):
	name = ["C2E", "Char2Emoj"]
	description = "Makes emoji blocks using a string."
	schema = cdict(
		text=cdict(
			type="string",
			example="POOP",
			required=True,
			greedy=False,
		),
		emoji1=cdict(
			type="emoji",
			example="💩",
			default="⬜",
		),
		emoji2=cdict(
			type="emoji",
			example="🪰",
			default="⬛",
		),
	)
	rate_limit = (10, 14)
	slash = True

	def fits(self, text, ratio=1):
		lim = 200 * ratio
		if sum(not c.isalnum() for c in text) > lim:
			return False
		lim = 2000 * ratio
		if len(text) > lim:
			return False
		return True

	async def __call__(self, bot, _guild, _channel, _message, text, emoji1, emoji2, **void):
		use_webhook = not getattr(_guild, "ghost", None)
		e1 = min_emoji(emoji1)
		e2 = min_emoji(emoji2)
		resp = _c2e(text, e1, e2)
		temp = "\xad" + "\n".join(resp)
		if hasattr(_message, "simulated") or self.fits(temp):
			return temp
		out = ["\xad" + "\n".join(i) for i in (resp[:2], resp[2:5], resp[5:])]
		if not all(self.fits(line) for line in out):
			out = []
			for line in resp:
				if not out or not self.fits(out[-1] + "\n" + line):
					out.append("\xad" + line)
				else:
					out[-1] += "\n" + line
		if use_webhook:
			for line in out:
				await bot.send_as_webhook(_channel, line)
			return

		async def c2e_iterator():
			yield out[0]
			for line in out[1:]:
				yield "\n" + line

		return cdict(
			content=c2e_iterator(),
		)


class Emoticon(Command):
	description = "Sends an ASCII emoticon from a selection."
	em_map = {k: v for v, k in (line.rsplit(None, 1) for line in """
( ͡° ͜ʖ ͡°) Lenny
ಠ_ಠ Disapprove
ฅ(^•ﻌ•^ฅ) Cat
ʕ •ᴥ•ʔ Bear
(⁄ ⁄•⁄ω⁄•⁄ ⁄) Embarrassed
(≧∇≦)/ Hai
(/◕ヮ◕)/ Joy
(凸ಠ益ಠ)凸 FlipOff
Ｔ▽Ｔ Cry
(✿◠‿◠) Flower
(*´▽｀*) Infatuated
ᕕ( ᐛ )ᕗ HappyGary
ヽ(´ー｀)┌ Mellow
(´･ω･`) UwU
(*^3^)/~☆ Smooch
.....φ(・∀・＊) Studying
☆彡 Star
(｀-´)> Salute
(´；ω；`) Sad
（ ^_^）o自自o（^_^ ） Cheers
⊂二二二（＾ω＾）二⊃ Com'ere
ヽ(´ー`)人(´∇｀)人(`Д´)ノ Friends
（･∀･)つ⑩ Money
d(*⌒▽⌒*)b Happy
(≧ロ≦) Shout
""".splitlines() if line)}
	em_mapper = fcdict(em_map)
	schema = cdict(
		emoticon=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(sorted(em_mapper)),
			),
			description="Emoticon to insert",
			required=True,
		),
		position=cdict(
			type="enum",
			validation=cdict(
				enum=("start", "end"),
			),
			description="Position to insert emoticon",
			default="end",
		),
		text=cdict(
			type="string",
			description="Extra message content to send",
			greedy=False,
		),
	)
	no_cancel = True
	rate_limit = (1, 5)
	slash = True

	async def __call__(self, bot, _user, _slash, _channel, _message, emoticon, position, text, **void):
		insertion = self.em_mapper[emoticon]
		if text:
			msg = text.strip() + " " + insertion if position == "end" else insertion + " " + text.strip()
		else:
			msg = insertion
		url = await self.bot.get_proxy_url(_user)
		if _slash:
			return cdict(content=msg)
		fut = create_task(bot.autodelete(_message))
		await bot.send_as_webhook(_channel, msg, username=_user.display_name, avatar_url=url)
		await fut


class Time(Command):
	name = ["🕰️", "⏰", "⏲️", "UTC", "GMT", "T"]
	description = "Shows the current time at a certain GMT/UTC offset, the current time for a user, or the closest timezone(s) to a given current time."
	schema = cdict(
		mode=cdict(
			type="enum",
			description="Whether to retrieve all time formats, simple formats, or the closest timezones",
			validation=cdict(
				enum=("full", "relative", "absolute", "timezone"),
			),
			default="full",
		),
		input=cdict(
			type="string",
			description="Time input to parse",
			example="last sunday 3pm hawaii",
			default="",
		),
		user=cdict(
			type="user",
			description="Target user to retrieve timezone from. Will use automatically estimated timezone if not provided",
			example="201548633244565504",
		),
		estimate=cdict(
			type="bool",
			description="Force a timezone estimate, even if a timezone is assigned",
			default=False,
		),
		timezone=cdict(
			type="word",
			description="Target timezone to display input after conversion",
			example="Europe/Stockholm",
		),
	)
	macros = cdict(
		TS=cdict(
			mode="relative",
		),
		Timestamp=cdict(
			mode="relative",
		),
		TimeZone=cdict(
			mode="timezone",
		),
		EstimateTime=cdict(
			estimate=True,
		),
		EstimateTimezone=cdict(
			estimate=True,
		),
	)
	rate_limit = (3, 5)
	slash = True
	ephemeral = True
	tzcache = AutoCache(f"{CACHE_PATH}/timezones", shards=1, stale=86400, timeout=86400 * 30)
	tzlist = {str(tz) if isinstance(tz, pytz.BaseTzInfo) else k.upper(): tz for k, tz in TIMEZONES.items() if not k.startswith("etc/")}

	async def __call__(self, _user, mode, input, user, estimate, timezone, **void):
		target = user or _user
		c = 1
		tzinfo = self.bot.data.users.get_timezone(target.id) if not estimate else None
		if tzinfo is None:
			tzinfo, c = self.bot.data.users.estimate_timezone(target.id)
			estimated = True
		else:
			estimated = False
		dt2 = DynamicDT.now(tz=tzinfo)
		dt = DynamicDT.parse(input, timestamp=dt2.timestamp_exact(), timezone=get_name(tzinfo))
		match mode:
			case "timezone":

				def offset_repr(hours):
					if hours > 12:
						hours -= 24
					sym = "+"
					if hours < 0:
						sym = "-"
						hours = -hours
					seconds = hours * 3600
					disp = time_disp(seconds)
					while disp.endswith(":00"):
						disp = disp[:-3]
					return sym + disp

				central = dt2.cast(get_timezone("utc"))
				new = dt.replace(tzinfo=get_timezone("utc"))
				offset = (new - central).total_seconds() / 3600 % 24
				colour = await self.bot.get_colour(target)
				emb = discord.Embed(colour=colour)
				emb.add_field(name="Parsed As", value="`" + ", ".join(dt.parsed_as) + "`")
				emb.add_field(name="GMT/UTC Offset", value=f"`{offset_repr(offset)}`")

				def retrieval():
					data = mdict()
					now = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
					for k, tz in self.tzlist.items():
						data.append(round_min(tz.utcoffset(now).total_seconds() / 3600), (k, tz))
					return data

				data = await self.tzcache.aretrieve("offsets", retrieval)
				match_list = sorted((min((offset - delta) ** 2, (offset - delta - 24) ** 2), delta, k, tz) for delta, v in data.items() for k, tz in v)
				matches = []
				while len(matches) < 20 and match_list and (not matches or match_list[0][0] <= 0.25):
					hyp, delta, k, tz = match_list.pop(0)
					matches.append(f"*{k}* **({offset_repr(delta)})**: `{dt2.cast(tz).time()}`")
				emb.add_field(name="Closest Matches", value="\n".join(matches), inline=False)
				with tracebacksuppressor:
					emb.timestamp = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)
				return cdict(embed=emb)
			case "relative":
				return dt.as_rel_discord()
			case "absolute":
				return dt.as_discord()
		colour = await self.bot.get_colour(target)
		emb = discord.Embed(colour=colour)
		emb.add_field(name="Parsed As", value="`" + ", ".join(dt.parsed_as) + "`")
		if timezone:
			dt = dt.cast(get_timezone(timezone))
			tzstats = italics(get_name(tzinfo))
			emb.add_field(name="Local Timezone", value=tzstats)
		elif user:
			tzstats = italics(get_name(tzinfo))
			if estimated:
				tzstats += f" {user_mention(target.id)}, estimated ({round(c * 100)}% confidence)"
			else:
				tzstats += f" {user_mention(target.id)}, assigned"
			emb.add_field(name="Local Timezone", value=tzstats)
		emb.add_field(name="Displayed Time", value=str(dt))
		emb.add_field(name="Unix Timestamp", value=f"`{dt.timestamp()}`")
		emb.add_field(name="ISO Timestamp", value=f"`{dt.as_iso()}`")
		emb.add_field(name="Time Delta", value=f"`{dt - dt2}`")
		emb.add_field(name="Live Timestamp", value=f"`{dt.as_discord()}`\n{dt.as_discord()}")
		emb.add_field(name="Live Delta", value=f"`{dt.as_rel_discord()}`\n{dt.as_rel_discord()}")
		with tracebacksuppressor:
			emb.timestamp = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)
		return cdict(embed=emb)


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
		mimetype = info.file["Mimetype"] = heads.get("Content-Type", "application/octet-stream")
		fmt  = info.file["Format"] = mime_into(mimetype)
		info.file["Name"] = try_header_filename(heads, url)
		size = heads.get("Content-Length")
		if not size or int(size) <= 10 * 1048576:
			path = await attachment_cache.download(url, filename=True)
			size = os.path.getsize(path)
		else:
			path = url
		info.file["Size"] = byte_scale(size) + "B"
		if fmt in AUDIO_FORMS or fmt in VIDEO_FORMS:
			meta = await _run_async(audio_meta, path)
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
					info.audio["Bitrate"] = round(meta.bitrate)
				if meta.sample_rate:
					info.audio["Sample Rate"] = meta.sample_rate
		if fmt in MEDIA_FORMS:
			meta = await _run_async(video_meta, path)
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


class Match(Command):
	name = ["RE", "RegEx", "RexExp", "GREP"]
	description = "matches two strings using Linux-style RegExp, or computes the match ratio of two strings."
	usage = "<0:string1> <1:string2>?"
	example = ("match test test2", "regex t*e+s?t test")
	rate_limit = (4, 6)
	ephemeral = True

	async def __call__(self, args, name, **void):
		if len(args) < 2:
			raise ArgumentError("Please enter two or more strings to match.")
		if name == "match":
			regex = None
			for i in (1, -1):
				s = args[i]
				if len(s) >= 2 and s[0] == s[-1] == "/":
					if regex:
						raise ArgumentError("Cannot match two Regular Expressions.")
					regex = s[1:-1]
					args.pop(i)
		else:
			regex = args.pop(0)
		if regex:
			temp = await _run_async(re.findall, regex, " ".join(args))
			match = "\n".join(sqr_md(i) for i in temp)
		else:
			search = args.pop(0)
			s = " ".join(args)
			match = (
				sqr_md(round_min(round(string_similarity(search, s) * 100, 6))) + "% literal match,\n"
				+ sqr_md(round_min(round(string_similarity(search.casefold(), s.casefold()) * 100, 6))) + "% case-insensitive match,\n"
				+ sqr_md(round_min(round(string_similarity(full_prune(search), full_prune(s)) * 100, 6))) + "% unicode mapping match."
			)
		return cdict(content=match, prefix="```ini\n", suffix="```")


class Random(Command):
	name = ["choice", "choose"]
	description = "Randomly chooses from a list of words."
	schema = cdict(
		args=cdict(
			type="word",
			description="List of possible choices, separated by newline or space.",
			example="one two three",
			multiple=True,
		),
	)
	slash = True
	ephemeral = True

	def __call__(self, args, **void):
		if not args:
			raise ArgumentError("Input string is empty.")
		if "\n" in args:
			x = choice(args.splitlines())
		else:
			x = choice(args)
		return f"I choose `{x}`!"


class Rate(Command):
	name = ["Rating", "Rank", "Ranking"]
	description = "Rates a given object with a random value out of 10!"
	usage = "<string>"

	async def __call__(self, bot, guild, argv, **void):
		rate = random.randint(0, 10)
		pronoun = "that"
		lego = f"`{grammarly_2_point_1(argv)}`"
		try:
			user = await bot.fetch_member_ex(verify_id(argv), guild, allow_banned=False, fuzzy=0)
		except:
			if re.match("<a?:[A-Za-z0-9\\-~_]+:[0-9]+>", argv):
				lego = argv
				pronoun = "it"
		else:
			lego = f"`{user.display_name}`"
			rate = 10
			pronoun = "them"
		lego = lego.replace("?", "").replace("!", "")
		return f"{lego}? I rate {pronoun} a `{rate}/10`!"


class WordCount(Command):
	name = ["Lc", "Wc", "Cc", "Count"]
	description = "Simple command that returns the word and character count of a supplied message. message.txt files work too!"
	schema = cdict(
		string=cdict(
			type="string",
			description="Text to count words from",
			example="Lorem ipsum dolor sit amet.",
		),
	)
	slash = True
	ephemeral = True

	async def __call__(self, string, **void):
		if not string:
			raise ArgumentError("Input string is empty.")
		lc = string.count("\n") + 1
		wc = len(string.split())
		cc = len(string)
		return f"Line count: `{lc}`\nWord count: `{wc}`\nCharacter count: `{cc}`"


class Topic(Command):
	name = ["Question"]
	description = "Asks a random question."
	usage = "<relationship(-r)>? <pickup-line(-p)>? <nsfw-pickup-line(-n)>?"
	example = ("topic", "question -r")
	flags = "npr"

	def __call__(self, bot, user, flags, channel, **void):
		create_task(bot.seen(user, event="misc", raw="Talking to me"))
		if "r" in flags:
			return "\u200b" + choice(bot.data.users.rquestions)
		elif "p" in flags:
			return "\u200b" + choice(bot.data.users.pickup_lines)
		elif "n" in flags:
			if bot.is_nsfw(channel):
				return "\u200b" + choice(bot.data.users.nsfw_pickup_lines)
			if hasattr(channel, "recipient"):
				raise PermissionError(f"This tag is only available in {uni_str('NSFW')} channels. Please verify your age using ~verify within a NSFW channel to enable NSFW in DMs.")
			raise PermissionError(f"This tag is only available in {uni_str('NSFW')} channels.")
		return "\u200b" + choice(bot.data.users.questions)


class Fact(Command):
	name = ["DailyFact", "UselessFact"]
	description = "Provides a random fact."

	async def __call__(self, bot, user, **void):
		create_task(bot.seen(user, event="misc", raw="Talking to me"))
		fact = await bot.data.flavour.get(p=False, q=False)
		return "\u200b" + fact


class BubbleWrap(Command):
	name = ["Pop", "Bubble", "Bubbles"]
	description = "Creates a sheet of bubble wrap using spoilers."
	schema = cdict(
		size=cdict(
			type="integer",
			validation="(0, 1250]",
			description="Amount of bubbles to generate",
			example="1000",
			default=250,
		),
	)
	rate_limit = (3, 5)
	slash = True
	ephemeral = True

	async def __call__(self, size, **void):
		bubbles = np.array(["||pop!||"] * size)
		boos = np.random.randint(0, 1000, size=size)
		bubbles[boos == 0] = "||boo!||"
		return cdict(content="".join(bubbles))


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