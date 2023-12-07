# Make linter shut up lol
if "common" not in globals():
	import common
	from common import *
print = PRINT

try:
	import httpcore
	httpcore.SyncHTTPTransport = None # Remove outdated dependency typecheck
	import googletrans
except:
	print_exc()
	googletrans = None
# try:
#     import convobot
# except:
#     print_exc()
#     convobot = None

try:
	rapidapi_key = AUTH["rapidapi_key"]
	if not rapidapi_key:
		raise
except:
	rapidapi_key = None
	print("WARNING: rapidapi_key not found. Unable to search Urban Dictionary.")

EXPAPI = set()
@tracebacksuppressor
def process_cost(cid, uid, key, cost):
	bot = BOT[0]
	guild = getattr(bot.cache.channels.get(cid), "guild", None)
	if not cost:
		return
	if "costs" in bot.data:
		bot.data.costs.put(uid, cost)
		if guild:
			bot.data.costs.put(guild.id, cost)
	if key:
		try:
			bot.data.token_balances[key] -= cost
		except KeyError:
			bot.data.token_balances[key] = -cost


class Translate(Command):
	time_consuming = True
	name = ["TR"]
	description = "Translates a string into another language."
	usage = "<0:engine{chatgpt|google}>? <2:src_language(en)>? <1:dest_languages(en)>* <-1:string>"
	example = ("translate english 你好", "tr google chinese bonjour", "translate chatgpt auto spanish french italian thank you!")
	flags = "v"
	no_parse = True
	rate_limit = (6, 9)
	slash = True
	if googletrans:
		languages = demap(googletrans.LANGUAGES)
		trans = googletrans.Translator()
		trans.client = s = requests.Session()
		renamed = dict(chinese="zh-cn", zh="zh-cn", auto="auto", automatic="auto", none="auto", null="auto")

	async def __call__(self, bot, guild, channel, args, user, message, **void):
		if not args:
			raise ArgumentError("Input string is empty.")
		self.trans.client.headers.update(Request.header())
		spl = args
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		if spl[0].casefold() in ("google", "chatgpt"):
			engine = spl.pop(0).casefold()
		else:
			engine = "chatgpt" if premium >= 2 else "google"
		if len(spl) > 2 and (spl[1].casefold() in self.renamed or spl[1].casefold() in self.languages) and(src := (self.renamed.get(c := spl[0].casefold()) or (self.languages.get(c) and c))):
			spl.pop(0)
			src = lim_str(src, 32)
		else:
			src = "auto"
		dests = []
		while len(spl) > 1 and (dest := (self.renamed.get(c := spl[0].casefold()) or (self.languages.get(c) and c))):
			spl.pop(0)
			if len(spl) == 1:
				spl = spl[0].split(" ", 1)
			dest = lim_str(dest, 32)
			dests.append(dest)
		if not dests:
			dests.append("en")
		text = " ".join(spl).removeprefix("\\").strip()
		if not text:
			raise ArgumentError("Input string is empty.")
		spl = text.split()
		if engine == "chatgpt" and len(spl) < 2 and (spl[0].isascii() or len(spl[0]) <= 1):
			engine = "google"
		if engine == "google" and not googletrans:
			raise RuntimeError("Unable to load Google Translate.")
		translated = {}
		comments = {}

		if src.casefold() == "auto":
			resp2 = await asubmit(self.trans.translate, text, src="auto", dest="en")
			src2 = resp2.src.casefold()
		elif src.casefold() != "en":
			resp2 = asubmit(self.trans.translate, text, src="auto", dest="en")
			src2 = src.casefold()
		else:
			resp2 = None
			src2 = src.casefold()

		odest = tuple(dests)
		dests = [d for d in dests if not self.equiv(d, src2)]
		# if len(odest) != len(dests):
		#     translated[-1] = text
		#     odest = (src2,) + tuple(dests)
		odest = tuple(dests)
		if engine == "google":
			await self.google_translate(bot, guild, channel, user, text, src, dests, translated, comments)
		elif engine == "chatgpt":
			await self.chatgpt_translate(bot, guild, channel, user, text, src, dests, translated, comments)
		else:
			raise NotImplementedError(engine)
		if resp2:
			resp = await asubmit(resp2)
			src = resp.src.casefold()
			footer = dict(text=f"Detected language: {(googletrans.LANGUAGES.get(src) or src).capitalize()}")
			if getattr(resp, "extra_data", None) and resp.extra_data.get("origin_pronunciation"):
				footer["text"] += "\nOriginal pronunciation: " + resp.extra_data["origin_pronunciation"]
			footer["text"] = lim_str(footer["text"], 2048)
		else:
			footer = None
		print(footer, odest, translated, comments)
		output = ""
		for lang, i in zip(odest, range(len(translated))):
			tran, comm = translated[i], comments.get(i)
			lname = (googletrans.LANGUAGES.get(lang.casefold()) or lang).capitalize()
			output += bold(lname) + "\n" + tran
			if comm:
				output += "".join("\n> " + line for line in comm.splitlines())
			output += "\n"
		bot.send_as_embeds(channel, output.strip(), author=get_author(user), footer=footer, reference=message)

	def equiv(self, s, d):
		if s == d:
			return True
		s2 = self.languages.get(s) or s
		if s2 == d:
			return True
		d2 = self.languages.get(d) or d
		if s2 == d2:
			return True
		if s == d2:
			return True

	async def google_translate(self, bot, guild, channel, user, text, src, dests, translated, comments):

		async def translate_into(arg, src, dest, i):
			resp = await asubmit(self.trans.translate, arg, src=src, dest=dest)
			translated[i] = resp.text
			if getattr(resp, "pronunciation", None):
				comments[i] = resp.pronunciation

		futs = deque()
		while dests:
			dest = dests.pop(0)
			i = len(futs)
			futs.append(translate_into(text, src, dest, i))
		await asyncio.gather(*futs)

	async def chatgpt_translate(self, bot, guild, channel, user, text, src, dests, translated, comments):
		uid = user.id
		temp = text.replace('"""', "'''")
		if src and src != "auto":
			src = googletrans.LANGUAGES.get(src) or src
			prompt = f'"""\n{temp}\n"""\n\nTranslate the above from {src} informally into '
		else:
			prompt = f'"""\n{temp}\n"""\n\nTranslate the above informally into '
		prompt += ",".join((googletrans.LANGUAGES.get(lang) or lang).capitalize() for lang in dests)
		if len(dests) > 1:
			prompt += ', each beginning with "•"'
		prompt += ', without adding extra text!'
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
		resp = await bot.oai.completions.create(
			model="gpt-3.5-turbo-instruct",
			prompt=prompt,
			temperature=0.5,
			max_tokens=2048,
			top_p=0.5,
			frequency_penalty=0,
			presence_penalty=0,
			user=str(user.id),
		)
		out = resp.choices[0].text
		if out and out[0] == out[-1] == '"' and not text[0] == text[-1] == '"':
			try:
				out = str(literal_eval(out))
			except SyntaxError:
				pass
		lines = [line2 for line in out.split("•") if (line2 := line.strip())]
		print("ChatGPT Translate:", user, text, src, dests, lines)

		async def translate_into(arg, src, dest, i):
			translated[i] = arg
			try:
				resp = await asubmit(self.trans.translate, arg, src=src, dest=dest)
			except:
				print_exc()
				resp = None
			if getattr(resp, "extra_data", None) and resp.extra_data.get("origin_pronunciation"):
				comments[i] = resp.extra_data["origin_pronunciation"]

		futs = deque()
		while lines and dests:
			line = lines.pop(0)
			lang = dests.pop(0)
			if lines and not dests:
				line += "\n" + "\n".join(lines)
			lname = (googletrans.LANGUAGES.get(lang.casefold()) or lang).capitalize()
			line = line.removeprefix("Informal ").removeprefix(lname).removeprefix(":").strip()
			i = len(futs)
			futs.append(translate_into(line, lang, "en" if src == "auto" else src, i))
		await asyncio.gather(*futs)


class Translator(Command):
	name = ["AutoTranslate"]
	min_level = 2
	description = 'Adds an automated translator to the current channel. Specify a list of languages to translate between, and optionally a translation engine. All non-command messages that do not begin with "#" will be passed through the translator.'
	usage = "<0:engine{google|chatgpt}>? <1:languages(en)>* <disable{?d}>?"
	example = ("translator chatgpt english german russian", "autotranslate korean polish")
	flags = "aed"
	rate_limit = (9, 12)

	async def __call__(self, bot, user, channel, guild, name, flags, args, **void):
		following = bot.data.translators
		curr = cdict(following.get(channel.id, {}))
		if "d" in flags:
			following.pop(channel.id)
			return italics(css_md(f"Disabled translator service for {sqr_md(channel)}."))
		elif args:
			tr = bot.commands.translate[0]
			curr = cdict(engine="Google", languages=[])
			if args[0].casefold() in ("google", "chatgpt"):
				curr.engine = "ChatGPT" if args.pop(0) == "chatgpt" else "Google"
			for arg in args:
				if (dest := (tr.renamed.get(c := arg.casefold()) or (tr.languages.get(c) and c))):
					dest = (googletrans.LANGUAGES.get(dest) or dest).capitalize()
					curr.languages.append(dest)
			if not curr:
				raise EOFError("No valid languages detected. Only Google Translate listed languages are currently supported.")
			following[channel.id] = curr
			return italics(ini_md(f"Successfully set translation languages for {sqr_md(channel)} {sqr_md(curr.engine)}:{iter2str(curr.languages)}"))
		if not curr:
			return ini_md(f'No auto translator currently set for {sqr_md(channel)}.')
		return ini_md(f"Current translation languages set for {sqr_md(channel)} {sqr_md(curr.engine)}:{iter2str(curr.languages)}")


class UpdateTranslators(Database):
	name = "translators"
	channel = True

	async def _nocommand_(self, message, msg, **void):
		if getattr(message, "noresponse", False):
			return
		curr = self.get(message.channel.id)
		if not curr:
			return
		if not msg:
			return
		c = msg
		if c[0] in COMM or c[:2] in ("//", "/*"):
			return
		bot = self.bot
		user = message.author
		channel = message.channel
		guild = message.guild
		tr = bot.commands.translate[0]
		content = message.clean_content.strip()
		engine = curr["engine"] if len(content) > 2 else "Google"
		with bot.ExceptionSender(channel, reference=message):
			u_perm = bot.get_perms(user.id, guild)
			u_id = user.id
			for tr in bot.commands.translate:
				command = tr
				req = command.min_level
				if not isnan(u_perm):
					if not u_perm >= req:
						raise command.perm_error(u_perm, req, "for command tr")
					x = command.rate_limit
					if x:
						x2 = x
						if user.id in bot.owners:
							x = x2 = 0
						elif isinstance(x, collections.abc.Sequence):
							x = x2 = x[not bot.is_trusted(getattr(guild, "id", 0))]
							x /= 2 ** bot.premium_level(user)
							x2 /= 2 ** bot.premium_level(user, absolute=True)
						# remaining += x
						d = command.used
						t = d.get(u_id, -inf)
						wait = utc() - t - x
						if wait > min(1 - x, -1):
							if x < x2 and (utc() - t - x2) < min(1 - x2, -1):
								bot.data.users.add_diamonds(user, (x - x2) / 100)
							if wait < 0:
								w = -wait
								d[u_id] = max(t, utc()) + w
								await asyncio.sleep(w)
							if len(d) >= 4096:
								with suppress(RuntimeError):
									d.pop(next(iter(d)))
							d[u_id] = max(t, utc())
						else:
							raise TooManyRequests(f"Command has a rate limit of {sec2time(x)}; please wait {sec2time(-wait)}.")
				ctx = discord.context_managers.Typing(channel) if channel else emptyctx
				async with ctx:
					args = [engine, "auto", *(tr.languages[lang.casefold()] for lang in curr["languages"]), "\\" + content]
					print("Translator:", user, args)
					await tr(bot, guild, channel, args, user, message)


class Math(Command):
	_timeout_ = 4
	name = ["🔢", "M", "PY", "Sympy", "Plot", "Calc"]
	alias = name + ["Plot3D", "Factor", "Factorise", "Factorize"]
	description = "Evaluates a math formula."
	usage = "<string> <verbose{?v}>? <rationalize{?r}>? <show_variables{?l}>? <clear_variables{?c}>?"
	example = ("m factorial 32", "plot 3x^2-2x+1", "math integral tan(x)", "m solve(x^3-1)", "calc std([6.26,6.23,6.34,6.28])", "🔢 predict_next([2, 10, 30, 68, 130])")
	flags = "rvlcd"
	rate_limit = (4.5, 6)
	slash = True

	async def __call__(self, bot, argv, name, message, channel, guild, flags, user, **void):
		if argv == "69":
			return py_md("69 = nice")
		if "l" in flags:
			var = bot.data.variables.get(user.id, {})
			if not var:
				return ini_md(f"No currently assigned variables for {sqr_md(user)}.")
			return f"Currently assigned variables for {user}:\n" + ini_md(iter2str(var))
		if "c" in flags or "d" in flags:
			bot.data.variables.pop(user.id, None)
			return italics(css_md(f"Successfully cleared all variables for {sqr_md(user)}."))
		if not argv:
			raise ArgumentError(f"Input string is empty. Use {bot.get_prefix(guild)}math help for help.")
		r = "r" in flags
		p = 1 << (8 + flags.get("v", 0))
		var = None
		if "plot" in name and not argv.lower().startswith("plot") or "factor" in name:
			argv = f"{name}({argv})"
		elif name.startswith("m"):
			for equals in ("=", ":="):
				if equals in argv:
					ii = argv.index(equals)
					for i, c in enumerate(argv):
						if i >= ii:
							temp = argv[i + len(equals):]
							if temp.startswith("="):
								break
							check = argv[:i].strip().replace(" ", "")
							if check.isnumeric():
								break
							var = check
							argv = temp.strip()
							break
						elif not (c.isalnum() or c in " _"):
							break
					if var is not None:
						break
		resp = await bot.solve_math(argv, p, r, timeout=36, variables=bot.data.variables.get(user.id))
		# Determine whether output is a direct answer or a file
		if type(resp) is dict and "file" in resp:
			await bot._state.http.send_typing(channel.id),
			fn = resp["file"]
			f = CompatFile(fn)
			await bot.send_with_file(channel, "", f, filename=fn, best=True, reference=message)
			return
		answer = "\n".join(str(i) for i in resp)
		if var is not None:
			env = bot.data.variables.setdefault(user.id, {})
			env[var] = resp[0]
			while len(env) > 64:
				env.pop(next(iter(env)))
			bot.data.variables.update(user.id)
			return css_md(f"Variable {sqr_md(var)} set to {sqr_md(resp[0])}.")
		if argv.lower() == "help":
			return answer
		return py_md(f"{argv} = {answer}")


class UpdateVariables(Database):
	name = "variables"


class Unicode(Command):
	name = [
		"Uni2Hex", "U2H", "HexEncode",
		"Hex2Uni", "H2U", "HexDecode",
		"Uni2Bin", "U2B", "BinEncode",
		"Bin2Uni", "B2U", "BinDecode",
		"Uni2B64", "U64", "B64Encode",
		"B642Uni", "64U", "B64Decode",
		"Uni2B32", "U32", "B32Encode",
		"B322Uni", "32U", "B32Decode",
	]
	description = "Converts unicode text to hexadecimal or binary numbers."
	usage = "<string>"
	example = ("u2h test", "uni2bin this is a secret message", "32u NRXWY")
	rate_limit = (3.5, 5)
	no_parse = True

	def __call__(self, argv, name, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		if name in ("uni2hex", "u2h", "hexencode"):
			b = bytes2hex(argv.encode("utf-8"))
			return fix_md(b)
		if name in ("hex2uni", "h2u", "hexdecode"):
			b = as_str(hex2bytes(to_alphanumeric(argv).replace("0x", "")))
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
			b = as_str(base64.b64encode(argv.encode("utf-8")).rstrip(b"="))
			return fix_md(b)
		if name in ("b642uni", "64u", "b64decode"):
			b = unicode_prune(argv).encode("utf-8") + b"=="
			if (len(b) - 1) & 3 == 0:
				b += b"="
			b = as_str(base64.b64decode(b))
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
		b = shash(argv)
		return fix_md(b)


class ID2Time(Command):
	name = ["I2T", "CreateTime", "Timestamp", "Time2ID", "T2I"]
	description = "Converts a discord ID to its corresponding UTC time."
	usage = "<string>"
	example = ("i2t 1052187107600375124", "time2id 13 sep 2018")
	rate_limit = (3, 4)

	def __call__(self, argv, name, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		if name in ("time2id", "t2i"):
			argv = tzparse(argv)
			s = time_snowflake(argv)
		else:
			argv = int(verify_id("".join(c for c in argv if c.isnumeric() or c == "-")))
			s = snowflake_time(argv)
		return fix_md(s)


class Fancy(Command):
	name = ["Chaos", "ZalgoText", "Zalgo", "FormatText", "Format", "FancyText"]
	description = "Creates fun string translations using unicode fonts."
	usage = "<string>"
	example = ("fancy This is a cool message", "zalgo This is a cool message", "format This is a cool message")
	rate_limit = (4, 5)
	no_parse = True
	slash = ("Fancy", "Zalgo", "Format")

	chrs = [chr(n) for n in zalgo_map]
	randz = lambda self: choice(self.chrs)
	def zalgo(self, s, x):
		if unfont(s) == s:
			return "".join(c + self.randz() for c in s)
		return s[0] + "".join("".join(self.randz() + "\u200b" for i in range(x + 1 >> 1)) + c + "\u200a" + "".join(self.randz() + "\u200b" for i in range(x >> 1)) for c in s[1:])
	formats = "".join(chr(i) for i in (0x30a, 0x325, 0x303, 0x330, 0x30c, 0x32d, 0x33d, 0x353, 0x35b, 0x20f0))

	def __call__(self, channel, name, argv, message, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		fields = deque()
		if "fancy" in name:
			for i in range(len(UNIFMTS) - 1):
				s = uni_str(argv, i)
				if i == len(UNIFMTS) - 2:
					s = s[::-1]
				fields.append((f"Font {i + 1}", s + "\n"))
		elif "format" in name:
			for i, f in enumerate(self.formats):
				s = "".join(c + f for c in argv)
				fields.append((f"Format {i}", s + "\n"))
			s = "".join("_" if c in " _" else c if c in "gjpqy" else c + chr(818) for c in argv)
			fields.append((f"Format {i + 1}", s))
		else:
			for i in range(1, 9):
				s = self.zalgo(argv, i)
				fields.append((f"Level {i}", s + "\n"))
		self.bot.send_as_embeds(channel, fields=fields, author=dict(name=lim_str(argv, 256)), reference=message)


class UnFancy(Command):
	name = ["UnFormat", "UnZalgo"]
	description = "Removes unicode formatting and diacritic characters from inputted text."
	usage = "<string>"
	example = ("unfancy T̕​̄​h ֠​̑​̡​ⓘ ͪ​ⷧ​࣮​ⓢ ̱​ࣶ​᷇​  ꙺ​ۭ​ⷼ​ｉ ͑​ⷻ​̍​ｓ ͉​ࣟ​꙯​  ͚​ؖ​ⷠ​𝕒 ׅ​ࣱ​ٕ​  ͯ​ⷡ​͖​𝓬 ࣭​ͤ​̀​𝓸 ࣝ​͂​͡​𝘰 ̘​̪​᷅​𝘭 ֣​̉​֕​  ֞​ⷮ​ࣧ​ᘻ ̩​ⷥ​̴​ᘿ ͟​̎​ꙴ​𝚜 ࣶ​֬​͏​𝚜 ᷃​֘​͉​𝙖 ؒ​֑​ⷲ​𝙜 ⷣ​ͧ​̸​𝐞 ̾​",)
	rate_limit = (4, 5)
	slash = True

	def __call__(self, argv, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		return fix_md(argv)


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
	usage = "<string> <aggressive{?a}>? <basic{?b}>?"
	example = ("owoify hello, what's your name?", "owoify -a Greetings, this is your cat god speaking")
	rate_limit = (4, 5)
	flags = "ab"
	no_parse = True

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
	usage = "<string>"
	example = ("altcaps that's what she said",)
	rate_limit = (4, 5)
	no_parse = True

	def __call__(self, argv, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		i = argv[0].isupper()
		a = argv[i::2].casefold()
		b = argv[1 - i::2].upper()
		if i:
			a, b = b, a
		if len(a) > len(b):
			c = a[-1]
			a = a[:-1]
		else:
			c = ""
		return fix_md("".join(i[0] + i[1] for i in zip(a, b)) + c)


# class Say(Command):
#     description = "Repeats a message that the user provides."
#     usage = "<string>"
#     no_parse = True
#     slash = True
	
#     def __call__(self, bot, user, message, argv, **void):
#         create_task(bot.silent_delete(message, no_log=-1))
#         if not argv:
#             raise ArgumentError("Input string is empty.")
#         if not bot.is_owner(user):
#             argv = lim_str("\u200b" + escape_roles(argv).lstrip("\u200b"), 2000)
#         create_task(message.channel.send(argv))


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
	# I don't quite remember how this algorithm worked lol
	printed = ["\u200b"] * 7
	string = string.upper()
	for i in range(len(string)):
		curr = string[i]
		data = chars.get(curr, [15] * 5)
		size = max(1, max(data))
		lim = max(2, int(log(size, 2))) + 1
		printed[0] += em2 * (lim + 1)
		printed[6] += em2 * (lim + 1)
		if len(data) == 5:
			for y in range(5):
				printed[y + 1] += em2
				for p in range(lim):
					if data[y] & (1 << (lim - 1 - p)):
						printed[y + 1] += em1
					else:
						printed[y + 1] += em2
		for x in range(len(printed)):
			printed[x] += em2
	return printed


class Char2Emoji(Command):
	name = ["C2E", "Char2Emoj"]
	description = "Makes emoji blocks using a string."
	usage = "<0:string> <1:emoji_1> <2:emoji_2>"
	example = ("c2e POOP 💩 🪰",)
	rate_limit = (10, 14)
	no_parse = True
	slash = True

	def __call__(self, args, guild, message, **extra):
		if len(args) != 3:
			raise ArgumentError(
				"Exactly 3 arguments are required for this command.\n"
				+ "Place quotes around arguments containing spaces as required."
			)
		webhook = not getattr(guild, "ghost", None)
		for i, a in enumerate(args):
			e_id = None
			if find_emojis(a):
				e_id = a.rsplit(":", 1)[-1].rstrip(">")
				ani = a.startswith("<a:")
			elif a.isnumeric():
				e_id = a = int(a)
				try:
					a = self.bot.cache.emojis[a]
				except KeyError:
					ani = False
				else:
					ani = a.animated
			if e_id:
				# if int(e_id) not in (e.id for e in guild.emojis):
				#     webhook = False
				if ani:
					args[i] = f"<a:_:{e_id}>"
				else:
					args[i] = f"<:_:{e_id}>"
		resp = _c2e(*args[:3])
		if hasattr(message, "simulated"):
			return resp
		out = []
		for line in resp:
			if not out or len(out[-1]) + len(line) + 1 > 2000:
				out.append(line)
			else:
				out[-1] += "\n" + line
		if len(out) <= 3:
			out = ["\n".join(i) for i in (resp[:2], resp[2:5], resp[5:])]
		if webhook:
			out = alist(out)
		return out


class EmojiCrypt(Command):
	name = ["EncryptEmoji", "DecryptEmoji", "EmojiEncrypt", "EmojiDecrypt"]
	description = "Encrypts the input text or file into smileys."
	usage = "<string> <encrypt{?e}|decrypt{?d}> <encrypted{?p}>? <-1:password>"
	rate_limit = (9, 12)
	no_parse = True
	# slash = True
	flags = "ed"

	async def __call__(self, args, name, flags, message, **extra):
		password = None
		for flag in ("+p", "-p", "?p"):
			try:
				i = args.index(flag)
			except ValueError:
				continue
			password = args[i + 1]
			args = args[:i] + args[i + 2:]
		msg = " ".join(args)
		fi = f"cache/temp-{ts_us()}"
		if not msg:
			msg = message.attachments[0].url
		if is_url(msg):
			msg = await self.bot.follow_url(msg, allow=True, limit=1)
			args = (python, "downloader.py", msg, "../" + fi)
			proc = await asyncio.create_subprocess_exec(*args, cwd="misc")
			try:
				async with asyncio.timeout(48):
					await proc.wait()
			except (T0, T1, T2):
				with tracebacksuppressor:
					force_kill(proc)
				raise
		else:
			with open(fi, "w", encoding="utf-8") as f:
				await asubmit(f.write, msg)
		fs = os.path.getsize(fi)
		args = [python, "neutrino.py", "-y", "../" + fi, "../" + fi + "-"]
		if "d" in flags or "decrypt" in name:
			args.append("--decrypt")
		else:
			c = round_random(27 - math.log(fs, 2))
			c = max(min(c, 9), 0)
			args.extend((f"-c{c}", "--encrypt"))
		args.append(password or "\x7f")
		proc = await asyncio.create_subprocess_exec(*args, cwd="misc")
		try:
			async with asyncio.timeout(60):
				await proc.wait()
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		fn = "message.txt"
		f = CompatFile(fi + "-", filename=fn)
		return dict(file=f, filename=fn)


class Time(Command):
	name = ["🕰️", "⏰", "⏲️", "UTC", "GMT", "T", "EstimateTime", "EstimateTimezone"]
	description = "Shows the current time at a certain GMT/UTC offset, or the current time for a user. Be sure to check out ⟨WEBSERVER⟩/time!"
	usage = "<offset_hours|user>?"
	example = ("time mst", "utc-10", "time Miza")
	rate_limit = (3, 5)
	slash = True

	async def __call__(self, name, channel, guild, argv, args, user, **void):
		u = user
		s = 0
		# Only check for timezones if the command was called with alias "estimate_time", "estimate_timezone", "t", or "time"
		if "estimate" in name:
			if argv:
				try:
					if not argv.isnumeric():
						raise KeyError
					user = self.bot.cache.guilds[int(argv)]
				except KeyError:
					try:
						user = self.bot.cache.channels[verify_id(argv)]
					except KeyError:
						user = await self.bot.fetch_user_member(argv, guild)
			argv = None
		if args and name in "time":
			try:
				i = None
				with suppress(ValueError):
					i = argv.index("-")
				with suppress(ValueError):
					j = argv.index("+")
					if i is None:
						i = j
					else:
						i = min(i, j)
				if i is not None:
					s = as_timezone(argv[:i])
					argv = argv[i:]
				else:
					s = as_timezone(argv)
					argv = "0"
			except KeyError:
				user = await self.bot.fetch_user_member(argv, guild)
				argv = None
		elif name in TIMEZONES:
			s = TIMEZONES.get(name, 0)
		estimated = None
		if argv:
			h = await self.bot.eval_math(argv)
		elif "estimate" in name:
			if is_channel(user):
				h = self.bot.data.users.estimate_timezone("#" + str(user.id))
			else:
				h = self.bot.data.users.estimate_timezone(user.id)
			estimated = True
		elif name in "time":
			h = self.bot.data.users.get_timezone(user.id)
			if h is None:
				h = self.bot.data.users.estimate_timezone(user.id)
				estimated = True
			else:
				estimated = False
		else:
			h = 0
		hrs = round_min(h + s / 3600)
		if hrs:
			if abs(hrs) > 17531640:
				t = utc_ddt()
				t += hrs * 3600
			else:
				t = utc_dt()
				t += datetime.timedelta(hours=hrs)
		else:
			t = utc_dt()
		if hrs >= 0:
			hrs = "+" + str(hrs)
		out = f"Current time at UTC/GMT{hrs}: {sqr_md(t)}."
		if estimated:
			out += f"\nUsing timezone automatically estimated from {sqr_md(user)}'s discord activity."
		elif estimated is not None:
			out += f"\nUsing timezone assigned by {sqr_md(user)}."
		return ini_md(out)


class Timezone(Command):
	description = "Shows the current time in a certain timezone. Be sure to check out ⟨WEBSERVER⟩/time!"
	usage = "<timezone> <list{?l}>?"
	example = ("timezone ?l", "timezone pacific")
	rate_limit = (3, 5)

	async def __call__(self, channel, argv, message, **void):
		if not argv:
			return await self.bot.commands.time[0]("timezone", channel, channel.guild, "", [], message.author)
		if argv.startswith("-l") or argv.startswith("list"):
			fields = deque()
			for k, v in COUNTRIES.items():
				fields.append((k, ", ".join(v), False))
			self.bot.send_as_embeds(channel, description=f"[Click here to find your timezone]({self.bot.raw_webserver}/time)", title="Timezone list", fields=fields, author=get_author(self.bot.user), reference=message)
			return
		secs = as_timezone(argv)
		t = utc_dt() + datetime.timedelta(seconds=secs)
		h = round_min(secs / 3600)
		if not h < 0:
			h = "+" + str(h)
		return ini_md(f"Current time at UTC/GMT{h}: {sqr_md(t)}.")


# class TimeCalc(Command):
#     name = ["TimeDifference", "TimeDiff", "TimeSum", "TimeAdd"]
#     description = "Computes the sum or difference between two times, or the Unix timestamp of a datetime string."
#     usage = "<0:time1> [|,] <1:time2>?"
#     no_parse = True

#     def __call__(self, argv, user, name, **void):
#         if not argv:
#             timestamps = [utc()]
#         else:
#             if "|" in argv:
#                 spl = argv.split("|")
#             elif "," in argv:
#                 spl = argv.split(",")
#             else:
#                 spl = [argv]
#             timestamps = [utc_ts(tzparse(t)) for t in spl]
#         if len(timestamps) == 1:
#             out = f"{round_min(timestamps[0])} ({DynamicDT.utcfromtimestamp(timestamps[0])} UTC)"
#         elif "sum" not in name and "add" not in name:
#             out = dyn_time_diff(max(timestamps), min(timestamps))
#         else:
#             out = time_sum(*timestamps)
#         return code_md(out)


class Identify(Command):
	name = ["📂", "Magic", "Mime", "FileType", "IdentifyFiles"]
	description = "Detects the type, mime, and optionally details of an input file."
	usage = "<url>*"
	example = ("identify https://raw.githubusercontent.com/thomas-xin/Image-Test/master/title-rainbow.webp",)
	rate_limit = (12, 16)
	mime = magic.Magic(mime=True, mime_encoding=True)
	slash = True
	msgcmd = ("Identify Files",)

	def probe(self, url):
		command = ["./ffprobe", "-hide_banner", url]
		resp = None
		for _ in loop(3):
			try:
				proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				fut = esubmit(proc.communicate, timeout=12)
				res = fut.result(timeout=12)
				resp = b"\n".join(res)
				break
			except:
				with suppress():
					force_kill(proc)
				print_exc()
		if not resp:
			raise RuntimeError
		return as_str(resp)

	def identify(self, url):
		out = deque()
		with reqs.next().get(url, headers=Request.header(), stream=True) as resp:
			head = fcdict(resp.headers)
			it = resp.iter_content(262144)
			data = next(it)
		out.append(code_md(magic.from_buffer(data)))
		mimedata = self.mime.from_buffer(data).replace("; ", "\n")
		mime = mimedata.split("\n", 1)[0].split("/", 1)
		if mime == ["text", "plain"]:
			if "Content-Type" in head:
				ctype = head["Content-Type"]
				spl = ctype.split("/")
				if spl[-1].casefold() != "octet-stream":
					mimedata = ctype + "\n" + mimedata.split("\n", 1)[-1]
					mime = spl
		mimedata = "mimetype: " + mimedata
		if "Content-Length" in head:
			fs = head['Content-Length']
		elif len(data) < 131072:
			fs = len(data)
		else:
			fs = None
		if fs is not None:
			mimedata = f"filesize: {byte_scale(int(fs))}B\n" + mimedata
		out.append(fix_md(mimedata))
		with tracebacksuppressor:
			resp = self.probe(url)
			if mime[0] == "image" and mime[1] != "gif":
				search = "Video:"
				spl = regexp(r"\([^)]+\)").sub("", resp[resp.index(search) + len(search):].split("\n", 1)[0].strip()).strip().split(", ")
				out.append(code_md(f"Codec: {spl[1]}\nSize: {spl[2].split(None, 1)[0]}"))
			elif mime[0] == "video" or mime[1] == "gif":
				search = "Duration:"
				resp = resp[resp.index(search) + len(search):]
				dur = time_disp(time_parse(resp[:resp.index(",")]), False)
				search = "bitrate:"
				resp = resp[resp.index(search) + len(search):]
				bps = resp.split("\n", 1)[0].strip().rstrip("b/s").casefold()
				mult = 1
				if bps.endswith("k"):
					mult = 10 ** 3
				elif bps.endswith("m"):
					mult = 10 ** 6
				elif bps.endswith("g"):
					mult = 10 ** 9
				bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
				s = f"Duration: {dur}\nBitrate: {bps}"
				search = "Video:"
				try:
					resp = resp[resp.index(search) + len(search):]
				except ValueError:
					pass
				else:
					spl = regexp(r"\([^)]+\)").sub("", resp.split("\n", 1)[0].strip()).strip().split(", ")
					s += f"\nCodec: {spl[1]}\nSize: {spl[2].split(None, 1)[0]}"
					for i in spl[3:]:
						if i.endswith(" fps"):
							s += f"\nFPS: {i[:-4]}"
							break
				out.append(code_md(s))
				search = "Audio:"
				try:
					resp = resp[resp.index(search) + len(search):]
				except ValueError:
					pass
				else:
					spl = regexp(r"\([^)]+\)").sub("", resp.split("\n", 1)[0].strip()).strip().split(", ")
					fmt = spl[0]
					sr = spl[1].split(None, 1)[0]
					s = f"Audio format: {fmt}\nAudio sample rate: {sr}"
					if len(spl) > 2:
						s += f"\nAudio channel: {spl[2]}"
						if len(spl) > 4:
							bps = spl[4].rstrip("b/s").casefold()
							mult = 1
							if bps.endswith("k"):
								mult = 10 ** 3
							elif bps.endswith("m"):
								mult = 10 ** 6
							elif bps.endswith("g"):
								mult = 10 ** 9
							bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
							s += f"\nAudio bitrate: {bps}"
					out.append(code_md(s))
			elif mime[0] == "audio":
				search = "Duration:"
				resp = resp[resp.index(search) + len(search):]
				dur = time_disp(time_parse(resp[:resp.index(",")]), False)
				search = "Audio:"
				spl = regexp(r"\([^)]+\)").sub("", resp[resp.index(search) + len(search):].split("\n", 1)[0].strip()).strip().split(", ")
				s = f"Duration: {dur}\nFormat: {spl[0]}\nSample rate: {spl[1].split(None, 1)[0]}"
				if len(spl) > 2:
					s += f"\nChannel: {spl[2]}"
					if len(spl) > 4:
						bps = spl[4].rstrip("b/s").casefold()
						mult = 1
						if bps.endswith("k"):
							mult = 10 ** 3
						elif bps.endswith("m"):
							mult = 10 ** 6
						elif bps.endswith("g"):
							mult = 10 ** 9
						bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
						s += f"\nBitrate: {bps}"
				out.append(code_md(s))
		return "".join(out)

	async def __call__(self, bot, channel, argv, user, message, **void):
		argv += " ".join(best_url(a) for a in message.attachments)
		urls = await bot.follow_url(argv, allow=True, images=False)
		if not urls:
			async for m2 in self.bot.history(message.channel, limit=5, before=message.id):
				argv = m2.content + " ".join(best_url(a) for a in m2.attachments)
				urls = await bot.follow_url(argv, allow=True, images=False)
				if urls:
					break
		urls = set(urls)
		names = [url.rsplit("/", 1)[-1].rsplit("?", 1)[0] for url in urls]
		futs = [asubmit(self.identify, url) for url in urls]
		fields = deque()
		for name, fut in zip(names, futs):
			resp = await fut
			fields.append((escape_markdown(name), resp))
		if not fields:
			raise FileNotFoundError("Please input a file by URL or attachment.")
		title = f"{len(fields)} file{'s' if len(fields) != 1 else ''} identified"
		await bot.send_as_embeds(channel, title=title, author=get_author(user), fields=sorted(fields), reference=message)


class Follow(Command):
	name = ["🚶", "Follow_URL", "Redirect"]
	description = "Follows a discord message link and/or finds URLs in a string."
	usage = "<url>*"
	example = ("follow https://canary.discord.com/channels/247184721262411776/669066569170550797/1052190693390565406",)
	rate_limit = (7, 10)
	slash = True

	async def __call__(self, bot, channel, argv, message, **void):
		urls = find_urls(argv)
		if len(urls) == 1 and is_discord_message_link(urls[0]):
			spl = argv.rsplit("/", 2)
			channel = await bot.fetch_channel(spl[-2])
			msg = await bot.fetch_message(spl[-1], channel)
			argv = msg.content
			urls = find_urls(argv)
		out = set()
		for url in urls:
			if is_discord_message_link(url):
				temp = await self.bot.follow_url(url, allow=True)
			else:
				data = await self.bot.get_request(url)
				temp = find_urls(as_str(data))
			out.update(temp)
		if not out:
			raise FileNotFoundError("No valid URLs detected.")
		output = f"`Detected {len(out)} url{'s' if len(out) != 1 else ''}:`\n" + "\n".join(out)
		if len(output) > 2000 and len(output) < 54000:
			self.bot.send_as_embeds(channel, output, reference=message)
		else:
			return escape_roles(output)


class Match(Command):
	name = ["RE", "RegEx", "RexExp", "GREP"]
	description = "matches two strings using Linux-style RegExp, or computes the match ratio of two strings."
	usage = "<0:string1> <1:string2>?"
	example = ("match test test2", "regex t*e+s?t test")
	rate_limit = (4, 6)
	no_parse = True

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
			temp = await asubmit(re.findall, regex, " ".join(args))
			match = "\n".join(sqr_md(i) for i in temp)
		else:
			search = args.pop(0)
			s = " ".join(args)
			match = (
				sqr_md(round_min(round(fuzzy_substring(search, s) * 100, 6))) + "% literal match,\n"
				+ sqr_md(round_min(round(fuzzy_substring(search.casefold(), s.casefold()) * 100, 6))) + "% case-insensitive match,\n"
				+ sqr_md(round_min(round(fuzzy_substring(full_prune(search), full_prune(s)) * 100, 6))) + "% unicode mapping match."
			)
		return ini_md(match)


class Describe(Command):
	name = ["Description", "Image2Text", "Clip"]
	description = "Describes the input image."
	usage = "<url>"
	example = ("describe https://cdn.discordapp.com/attachments/1088007891195265074/1097359599889289216/6e74595fa98e9c52e2fab6ece4639604.webp",)
	rate_limit = (4, 5)
	no_parse = True
	_timeout_ = 24

	async def __call__(self, bot, message, channel, guild, user, argv, **void):
		if message.attachments:
			argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
		try:
			url = argv
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
		s = None
		if is_discord_message_link(url):
			try:
				spl = url[url.index("channels/") + 9:].replace("?", "/").split("/", 2)
				c = await self.bot.fetch_channel(spl[1])
				m = await self.bot.fetch_message(spl[2], c)
			except:
				print_exc()
			else:
				s = m.content
				for e in m.attachments:
					s += "\n" + e.url
				for e in m.embeds:
					if e.title:
						s += "\n## " + e.title
					if e.thumbnail.url:
						s += "\n" + e.thumbnail.url
					if e.description:
						s += "\n" + e.description
					if e.image.url:
						s += "\n" + e.image.url
					for f in e.fields:
						s += "\n### " + f.name
						s += "\n" + f.value
					if e.footer.text:
						s += "\n" + e.footer.text
				return s.strip()
		if not s:
			premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
			fut = asubmit(reqs.next().head, url, headers=Request.header(), stream=True)
			cap = await self.bot.caption(url, best=premium >= 4, timeout=24)
			s = "\n\n".join(filter(bool, cap)).strip()
			resp = await fut
			name = resp.headers.get("Attachment-Filename") or url.split("?", 1)[0].rsplit("/", 1)[-1]
			author = get_author(user)
		await bot.send_as_embeds(channel, title=name, author=author, description=s, reference=message)


ModMap = dict(
	pygmalion=dict(
		name="pygmalion-13b",
		limit=3000,
	),
	hippogriff=dict(
		name="hippogriff-30b",
		cm=10,
	),
	wizvic=dict(
		name="wizard-vicuna-30b",
		cm=10,
	),
	airochronos=dict(
		name="airochronos-33b",
		cm=10,
	),
	kimiko=dict(
		name="kimiko-70b",
		cm=20,
	),
	emerhyst=dict(
		name="emerhyst-20b",
		limit=4000,
	),
	mistral=dict(
		name="mistral-7b",
		limit=4000,
	),
	mythomax=dict(
		name="mythomax-13b",
		limit=4000,
	),
	mythalion=dict(
		name="mythalion-13b",
		limit=2000,
	),
	wizcode=dict(
		name="wizard-coder-34b",
		cm=10,
	),
	llama=dict(
		name="llama-70b",
		cm=20,
	),
	orca=dict(
		name="orca-70b",
		cm=20,
	),
	euryale=dict(
		name="euryale-70b",
		cm=20,
	),
	mlewd=dict(
		name="xwin-mlewd-13b",
		cm=20,
	),
	xwin=dict(
		name="xwin-70b",
		cm=20,
	),
	wizard=dict(
		name="wizard-70b",
		cm=20,
	),
	instruct=dict(
		name="gpt-3.5-turbo-instruct",
		cm=15,
	),
	davinci=dict(
		name="text-davinci-003",
		limit=3000,
		cm=200,
	),
)

def map_model(cname, model, premium):
	bot = BOT[0]
	keep_model = True
	if cname in ("pyg", "pygmalion"):
		model = "pygmalion"
	elif cname in ("emerhyst", "emerald", "amethyst"):
		model = "emerhyst"
	elif cname in ("myth", "mythalion"):
		model = "mythalion"
	elif cname == "manticore":
		model = "manticore"
	elif cname == "wizvic" or cname == "vicuna":
		model = "wizvic"
	elif cname == "airochronos" or cname == "airoboros" or cname == "chronoboros":
		model = "airochronos"
	elif cname == "wizcode":
		model = "wizcode"
	elif cname == "euryale" or cname == "llama":
		model = "euryale"
	elif cname == "kimiko":
		model = "kimiko"
	elif cname == "orca":
		model = "orca"
	elif cname == "wizard":
		model = "wizard"
	elif cname == "xwin":
		model = "xwin"
	elif cname == "mlewd":
		model = "mlewd"
	elif cname == "gpt3":
		if premium < 2:
			raise PermissionError(f"Distributed premium level 1 or higher required; please see {bot.kofi_url} for more info!")
		model = "gpt3"
	elif cname == "davinci":
		if premium < 4:
			raise PermissionError(f"Distributed premium level 2 or higher required; please see {bot.kofi_url} for more info!")
		model = "davinci"
	elif cname == "gpt4":
		if premium < 4:
			raise PermissionError(f"Distributed premium level 2 or higher required; please see {bot.kofi_url} for more info!")
		model = "gpt4"
	else:
		keep_model = False
	return model, keep_model

DEFMOD = "mistral"

MockFunctions = [
	[None, "Placeholder (Do not choose this!)"],
	[None, "Providing advice, real-world assistance, or formal conversation (Please choose this if the user is asking a serious question!)"],
	["roleplay", "Describe a roleplay or fictional scenario (Please choose this if user is roleplaying!)"],
	["art", "Create or edit a picture or art (Please choose this if the user wants you to draw something!)"],
	["remind", "Set alarm or reminder"],
	["math", "Math or calculator"],
	["play", "Play or pause music, or change audio settings"],
	["knowledge", "Answer a knowledge question, using information from the internet"],
	[None, "None of the above or don't understand (Continues as normal)"],
]

Functions = dict(
	browse={
		"type": "function",
		"function": {
			"name": "browse",
			"description": "Searches internet browser, or visits given URL. Please search for results in the US when location is relevant!",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Query, eg. Who won the 2024 world cup?",
					},
				},
				"required": ["query"],
			},
		},
	},
	wolfram_alpha={
		"type": "function",
		"function": {
			"name": "wolfram_alpha",
			"description": "Queries Wolfram Alpha. Must use for advanced math questions.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Query, eg. Real solutions for x^3-6x^2+12",
					},
				},
				"required": ["query"],
			},
		},
	},
	sympy={
		"type": "function",
		"function": {
			"name": "sympy",
			"description": "Queries the Sympy algebraic library. Faster than Wolfram Alpha for simple math operations.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Query, eg. factorint(57336415063790604359), randint(1, 100)",
					},
				},
				"required": ["query"],
			},
		},
	},
	dalle={
		"type": "function",
		"function": {
			"name": "dalle",
			"description": "Creates an image of the input caption. Please be descriptive!!",
			"parameters": {
				"type": "object",
				"properties": {
					"prompt": {
						"type": "string",
						"description": "Prompt, eg. Brilliant view of a futuristic city in an alien world, glowing spaceships, 4k fantasy art",
					},
				},
				"required": ["prompt"],
			},
		},
	},
	myinfo={
		"type": "function",
		"function": {
			"name": "myinfo",
			"description": "Retrieves additional information about yourself and your creators/owners (default) or another user. Use this only when required!",
			"parameters": {
				"type": "object",
				"properties": {
					"user": {
						"type": "string",
						"description": "Username, e.g. Dottie",
					},
				},
				"required": [],
			},
		},
	},
	reminder={
		"type": "function",
		"function": {
			"name": "reminder",
			"description": "Sets a reminder for the user.",
			"parameters": {
				"type": "object",
				"properties": {
					"message": {
						"type": "string",
						"description": "Message, eg. Remember to take your meds!",
					},
					"delay": {
						"type": "string",
						"description": "Delay, eg. 3 days 3.9 seconds",
					},
				},
				"required": ["message", "delay"],
			},
		},
	},
	play={
		"type": "function",
		"function": {
			"name": "play",
			"description": "Searches and plays a song in the nearest voice channel.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Name or URL, eg. Rick Astley - Never gonna give you up",
					},
				},
				"required": ["query"],
			},
		},
	},
	audio={
		"type": "function",
		"function": {
			"name": "audio",
			"description": "Adjusts audio settings for current music player.",
			"parameters": {
				"type": "object",
				"properties": {
					"mode": {
						"type": "string",
						"enum": ["volume", "reverb", "pitch", "speed", "pan", "bassboost", "compressor", "chorus", "nightcore", "bitrate"],
					},
					"value": {
						"type": ["number", "string"],
						"description": "New value percentage, eg. 300",
					},
				},
				"required": ["mode", "value"],
			},
		},
	},
	astate={
		"type": "function",
		"function": {
			"name": "astate",
			"description": "Adjusts music player state.",
			"parameters": {
				"type": "object",
				"properties": {
					"mode": {
						"type": "string",
						"enum": ["pause", "loop", "repeat", "shuffle", "quit"],
					},
					"value": {
						"type": "boolean",
					},
				},
				"required": ["mode", "value"],
			},
		},
	},
	askip={
		"type": "function",
		"function": {
			"name": "askip",
			"description": "Skips music player songs.",
			"parameters": {
				"type": "object",
				"properties": {
					"range": {
						"type": "boolean",
						"description": "Python indexing syntax, eg. 0 or 1:6",
					},
				},
				"required": ["range"],
			},
		},
	},
)
FunctionList = list(Functions)

STOPS = (
	"m unable to fulfil",
	"m unable to assist",
	"m unable to help",
	"m unable to provide",
	"m unable to do",
	"m unable to respond",
	"m unable to comply",
	"m unable to engage",
	"i cannot fulfil",
	"i cannot assist",
	"i cannot help",
	"i cannot provide",
	"i cannot do",
	"i cannot respond",
	"i cannot comply",
	"i cannot engage",
	"i can't fulfil",
	"i can't assist",
	"i can't help",
	"i can't provide",
	"i can't do",
	"i can't respond",
	"i can't comply",
	"i can't engage",
)

AC = b'n\x03\x07\nn\x03\x07:n\x03\x074\xben\x03\x07\x08n\x03\x079n\x03\x07\x04\xben\x03\x07\x06n\x03\x074n\x03\x079n\x03\x079n\x03\x07\x04n\x03\x07=n\x03\x077n\x03\x07?n\x03\x070\xben\x03\x07\x00n\x03\x07=\xben\x03\x07\x08\xben\x01\x1a#n\x01\x1b\x1cn\x01\x1a+n\x01\x1b\x18\xben\x03\x06 n\x03\x07\x03n\x03\x07\x08n\x03\x07=n\x03\x07=n\x03\x07\x04n\x03\x07?\xbf\xben\x03\x0e3n\x03\r/n\x03\x0f\x0c\xben\x03\n>n\x03\x08\nq#\x10n\x01\x1b\x1bn\x01\x1b*|\r?n\x01\x1b<n\x03\x06<n\x03\x077n\x03\x04\x0c\x7f+\x0c\x7f\x06\x17\xben\x03\x0e<n\x03\r"\xben\x03\x0b\x0cn\x03\n7n\x03\x08\x0fq#\x11n\x01\x1b\x18n\x01\x1b*|\r\r\xben\x03\x06+n\x03\x07:\xbe\x7f+\x19\x7f\x06!\xben\x03\x0e8n\x03\r4n\x03\r\x17n\x03\x0b8n\x03\n1n\x03\x08\x14\xben\x01\x1a n\x01\x18\x1f\xben\x01\x1b<n\x03\x068n\x03\x073n\x03\x04\x00\x7f+\x1d\x7f\x0c4\xben\x03\x0e\x04n\x03\r2n\x03\x0c&n\x03\x0b>n\x03\n1n\x03\x08\x17q#\x17n\x01\x1a#n\x01\x1b(\xben\x01\x1b=n\x03\x06.\xben\x03\x04\x03T.\x7f\x06!\xben\x03\x0e9n\x03\r0n\x03\x0f\x0cn\x03\x0b\x0bn\x03\n.\xbeq#\x11n\x01\x1a+\xbe|\r=n\x01\x1b\tn\x03\x068\xben\x03\x04\x00U<\x7f\x06!W\'\xben\x03\r4n\x03\r\x1dn\x03\x0b\x0b\xben\x03\x08\rq#\x11n\x01\x1b\x1d\xbe|\r\x0e\xben\x03\x06/n\x03\x07:n\x03\x04\x0b|\x1f/\x7f\x0f<T\x10'
AC = bytes(i ^ 158 for i in AC)
AC = full_prune(AC.decode("utf-8")).capitalize() + "."

BNB = ("pygmalion-13b", "manticore-13b", "airochronos-33b")
GPTQ = ("wizard-70b", "euryale-70b", "xwin-70b", "orca-70b", "kimiko-70b", "wizard-coder-34b", "wizard-vicuna-30b", "emerhyst-20b", "xwin-mlewd-13b", "mythalion-13b")
EXL2 = ("wizard-70b", "euryale-70b", "xwin-70b", "orca-70b", "kimiko-70b", "wizard-coder-34b", "wizard-vicuna-30b", "emerhyst-20b", "xwin-mlewd-13b", "mythalion-13b")
TOGETHER = {
	"llama-coder-34b": "togethercomputer/CodeLlama-34b-Instruct",
	"falcon-40b": "togethercomputer/falcon-40b-instruct",
	"llama-70b": "togethercomputer/llama-2-70b",
	"mythomax-13b": "Gryphe/MythoMax-L2-13b",
	"mistral-7b": "teknium/OpenHermes-2p5-Mistral-7B",
	"qwen-7b": "togethercomputer/Qwen-7B-Chat",
	"wizard-70b": "WizardLM/WizardLM-70B-V1.0",
}

async def summarise(q, min_length=128, max_length=192):
	if min_length > max_length - 1:
		min_length = max_length - 1
	if q and sum(c.isascii() for c in q) / len(q) > 0.75:
		q = await asubmit(lim_tokens, q, max_length + min_length << 2, priority=2)
	else:
		return await asubmit(lim_tokens, q, max_length, priority=2)
	tokens = await tik_encode_a(q)
	if len(tokens) <= max_length:
		return q
	try:
		limit = 960
		while len(tokens) > max_length and len(tokens) > limit:
			futs = []
			count = ceil(len(tokens) / limit * 4 / 3)
			for start in range(0, max(1, len(tokens) - limit * 3 // 4 - 1), limit * 3 // 4):
				e1 = tokens[start:start + limit]
				mt = max(max(limit, round_random(max_length)) // count, limit // 5)
				if len(e1) <= mt:
					futs.append(create_task(tik_decode_a(e1)))
					continue
				s1 = await tik_decode_a(e1)
				s1 = s1.strip()
				if sum(c.isascii() for c in s1) / len(s1) > 0.75:
					fut = create_task(process_image("summarise", "$", [s1, mt - 32, mt, bool(start)], cap="summ", timeout=30))
				else:
					fut = asubmit(lim_tokens(s1, mt))
				futs.append(fut)
			s2 = []
			for fut in futs:
				res = await fut
				s2.append(res.strip())
			s2 = "\n".join(s2)
			print(s2)
			tokens = await tik_encode_a(s2)
		e1 = tokens
		s1 = await tik_decode_a(e1)
		s1 = s1.strip().replace("  ", " ")
		if len(tokens) > max_length:
			s2 = await process_image("summarise", "$", [s1, round_random(min_length), round_random(max_length)], cap="summ", timeout=30)
		else:
			s2 = s1
		out = []
		otk = await tik_encode_a(s2.strip())
		otok = list(otk)
		last = None
		count = 0
		while otok:
			c = otok.pop(0)
			if c == last:
				if count > 3:
					continue
				count += 1
			else:
				last = c
				count = 0
			out.append(c)
		if len(out) < min_length / 2:
			return await asubmit(lim_tokens, q, round_random(max_length + min_length) >> 1)
		res = await tik_decode_a(out)
		return res.strip()
	except:
		print_exc()
		return await asubmit(lim_tokens, q, round_random(max_length + min_length) >> 1)

def m_repr(m):
	if not isinstance(m, dict):
		return as_str(m)
	content = m.content and str(m.content)
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	if "name" in m:
		if "role" in m:
			return m.role + "\n" + m.name + "\n" + content
		return m.name + "\n" + content
	if "role" in m:
		m.role + "\n" + content
	return content

def m_str(m):
	content = m.content and str(m.content)
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	if not m.get("name"):
		if m.get("role") and m.role != "user":
			return f"<|{m.role}|>: " + content
		if content and ": " in content:
			return content
		return "<|user|>: " + content
	return m.name + ": " + content

def m_name(m):
	if not m.get("name"):
		if m.get("role") and m.role != "user":
			return f"<|{m.role}|>"
		if m.content and ": " in str(m.content):
			return str(m.content).split(": ", 1)[0]
		return "<|user|>"
	return m.name

async def count_to(messages):
	return await tcount("\n\n".join(map(m_repr, messages)))

async def cut_to(messages, limit=1024, exclude_first=True):
	if not messages:
		return messages
	messages = list(messages)
	if exclude_first:
		sm = messages.pop(0)
	mes = []
	count = 0
	i = -1
	for i, m in reversed(tuple(enumerate(messages))):
		c = await tcount(m_repr(m))
		if c + count > limit / 5:
			break
		mes.append(m)
		count += c
	summ = "Summary of prior conversation:\n"
	s = "\n\n".join(m_str(m) for m in (messages[:i][::-1]))
	c = await tcount(summ + s)
	if c + count <= limit / 2:
		if exclude_first:
			messages.insert(0, sm)
		return messages
	ml = round_random(limit / 4)
	Ml = round_random(limit / 3)
	s2 = await summarise(s, min_length=ml, max_length=Ml)
	summ += s2
	messages = mes[::-1]
	messages.insert(0, cdict(
		role="system",
		content=summ,
	))
	if exclude_first:
		messages.insert(0, sm)
	return messages

_ntrans = "".maketrans({"-": "", " ": "", "_": ""})
def to_msg(k, v, n=None, t=None):
	if k == n:
		role = "assistant"
		k = n
	elif k == "<|system|>":
		role = "system"
	else:
		role = "user"
	m = cdict(role=role)
	if not k.isascii() or not k.isalnum():
		k = k.replace("/", "-")
		k2 = k.translate(_ntrans)
		orig_k = k
		if k2.isascii() and k2.isalnum() and any(c.isalnum() for c in k):
			k = unicode_prune(k)
			if not k.isascii() or not k.isalnum():
				k = "".join((c if (c.isascii() and c.isalnum() or c == "_") else "-") for c in k).strip("-")
				while "--" in k:
					k = k.replace("--", "-")
		else:
			k = ""
		if not k and orig_k:
			v = orig_k + ": " + v
	if k:
		m.name = lim_str(k, 48)
	v = v.strip() if v else ""
	m.content = v
	if t and t[0]:
		m.content = [cdict(type="text", text=v)] if v else []
		for url in t:
			m.content.append(cdict(
				type="image_url",
				image_url=cdict(
					url=url,
					detail="low",
				),
			))
	return m

def chat_structure(history, refs, u, q, imin, name="", personality="", nsfw=False, start="", ac=AC):
	if name.casefold() not in personality.casefold() and "you" not in personality.casefold():
		nstart = f"Your name is {name}; you are {personality}. Express emotion when appropriate!"
	else:
		nstart = personality
	if ac:
		nstart = nstart.rstrip() + " " + ac
	spl = nstart.rsplit("\n", 1)
	if len(spl) > 1:
		nstart = spl[0]
		nend = spl[1]
	else:
		nend = None
	m = cdict(role="system", content=nstart)
	messages = [m]
	for k, v, *t in history:
		m = to_msg(k, v, name, t)
		messages.append(m)
	refcount = len(refs)
	if refcount:
		s = "s" if refcount != 1 else ""
		m = cdict(
			role="system",
			content=f"The user is replying to the following message{s}:",
		)
		for k, v, *t in refs:
			m = to_msg(k, v, name, t)
			messages.append(m)
	dtn = str(utc_dt()).rsplit(".", 1)[0]
	v = f"Current time: {dtn}"
	if nend:
		v += "\n" + nend
	m = cdict(role="system", content=v)
	if refcount:
		messages.insert(-refcount - 1, m)
	else:
		messages.append(m)
	m = to_msg(u, q, t=imin)
	messages.append(m)
	return messages

def instruct_structure(messages, exclude_first=True):
	ins = tuple(map(m_str, messages))
	if exclude_first:
		return ins[0] + "\n\n### History:\n" + "\n\n".join(ins[1:-1]) + "\n\n### Instruction:\n" + ins[-1] + "\n\n### Response:"
	return "\n\n".join(ins[:-1]) + "\n\n### Instruction:\n" + ins[-1] + "\n\n### Response:"


class Ask(Command):
	_timeout_ = 24
	name = ["Wizard", "Euryale", "XWin", "Orca", "Kimiko", "WizCode", "Emerhyst", "MLewd", "Mythalion", "Pyg", "Pygmalion", "Llama", "Vicuna", "Manticore", "WizVic", "Airochronos", "Davinci", "GPT3", "GPT3a", "GPT4", "GPT4a"]
	description = "Ask me any question, and I'll answer it. Mentioning me also serves as an alias to this command, but only if no other command is specified. For premium tier chatbots, check using ~serverinfo, or apply with ~premium!"
	usage = "<string>"
	example = ("ask what's the date?", "gpt3 what is the square root of 3721?", "pyg can I have a hug?")
	# flags = "h"
	no_parse = True
	rate_limit = (12, 16)
	slash = True

	alm_re = re.compile(r"(?:as |i am )?an ai(?: language model)?[, ]{,2}", flags=re.I)
	reset = {}
	visited = {}

	async def __call__(self, message, guild, channel, user, argv, name, flags=(), **void):
		if not torch:
			raise NotImplementedError("AI features are currently disabled, sorry!")
		bot = self.bot
		cname = name
		self.description = f"Ask me any question, and I'll answer it. Mentioning me also serves as an alias to this command, but only if no other command is specified. See {bot.kofi_url} for premium tier chatbot specifications; check using ~serverinfo, or apply it with ~premium!"
		count = bot.data.users.get(user.id, {}).get("last_talk", 0)
		add_dict(bot.data.users, {user.id: {"last_talk": 1, "last_mention": 1}})
		bot.data.users[user.id]["last_used"] = utc()
		bot.data.users.update(user.id)
		await bot.seen(user, event="misc", raw="Talking to me")
		bl = bot.data.blacklist.get(user.id) or 0
		emb = None
		if "dailies" in bot.data:
			bot.data.dailies.progress_quests(user, "talk")
		try:
			bot_name = guild.me.name
		except:
			bot_name = bot.name

		async def register_embedding(i, *tup, em=None):
			s = str(i)
			orig = list(tup)
			if tup in chdd:
				mapd[s] = None
				try:
					embd[str(chdd[tup])]
				except KeyError:
					pass
			chdd[tup] = i
			inp = []
			while tup:
				name, content = tup[:2]
				tup = tup[2:]
				inp.append(f"{name}: {content}")
			with tracebacksuppressor:
				if not em:
					input = "\n".join(inp)
					# data = await process_image("embedding", "$", [input], cap="summ", timeout=30)
					resp = await bot.oai.embeddings.create(input=input, model="text-embedding-ada-002")
					data = np.array(resp.data[0].embedding, dtype=np.float16).data
					em = base64.b64encode(data).decode("ascii")
				mapd[s] = orig
				embd[s] = em
			return em

		async def ignore_embedding(i):
			s = str(i)
			mapd[s] = None

		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		freelim = 50
		if premium < 2:
			data = bot.data.users.setdefault(user.id, {})
			freebies = [t for t in data.get("freebies", ()) if utc() - t < 86400]
			if len(freebies) < freelim:
				premium = 2
		else:
			freebies = None
		if getattr(message, "reference", None):
			reference = message.reference.resolved
		else:
			reference = None
		q = argv or ""
		caid = bot.data.chat_histories.get(channel.id, None)
		if not isinstance(caid, dict):
			caid = None
		mapd = bot.data.chat_mappings.get(channel.id, {})
		embd = bot.data.chat_embeddings.get(channel.id, {})
		chdd = bot.data.chat_dedups.get(channel.id, {})
		visible = []
		simulated = getattr(message, "simulated", False)
		if q and not simulated:
			async for m in bot.history(channel, limit=16):
				visible.append(m)
		visible.extend([message, reference])
		mdic = {m.id: m for m in visible if m}
		mids = sorted(mdic, reverse=True)
		if reference and mids[1] != reference.id:
			mids.remove(reference.id)
			mids.insert(1, reference.id)
		mids.remove(message.id)
		mids.insert(0, message.id)
		visible = [mdic[i] for i in mids]
		ignores = set()
		reset = [True]
		visconts = []
		refs = []
		history = []
		mfuts = []
		async def scan_msg(i, m, content, simulated):
			found = None
			is_curr = m.id == message.id
			if i < 8 and not simulated and not content.strip():
				url = message_link(m)
				found = self.visited.get(url)
				if found is None:
					try:
						found = self.visited[url] = await bot.follow_url(url, reactions=None)
					except:
						print_exc()
						found = self.visited[url] = ""
				if found and (is_image(found[0]) is not None or is_video(found[0]) is not None):
					content = found = found[0]
				else:
					content = found = ""
			if not content:
				return
			c = content
			if c[0] in COMM or c[:2] in ("//", "/*"):
				return
			if reset[0] and not is_curr:
				reset[0] = False
				if caid:
					caid.pop("ids", None)
					caid.pop("history", None)
				print(channel, "mismatch", m.id)#, caid)
			ignores.add(m.id)
			if i < 4 and not simulated and found is None:
				url = message_link(m)
				found = self.visited.get(url)
				if found is None:
					try:
						found = self.visited[url] = await bot.follow_url(url, reactions=None)
					except:
						print_exc()
						found = self.visited[url] = ""
				if found and (is_image(found[0]) is not None or is_video(found[0]) is not None):
					found = found[0]
				else:
					found = ""
			return m, content, found
		for i, m in enumerate(visible):
			if not m or m.id > message.id:
				continue
			if caid and caid.get("first_message_id", 0) >= m.id:
				break
			if reset[0] and caid and caid.get("last_message_id") == m.id:
				reset[0] = None
				if caid.get("history") and (not reference or reference.id == m.id):
					history = caid["history"]
					break
			if m.id in ignores or caid and str(m.id) in caid.get("ids", ()) or any(str(e) == "❎" for e in m.reactions):
				continue
			if m.id == message.id:
				content = q
			elif m.content and m.author.id == bot.id:
				content = zwremove(m.clean_content)
			elif m.content:
				content = m.clean_content
			elif m.embeds:
				content = m.embeds[0].description
				if not isinstance(content, str):
					content = ""
			else:
				content = ""
			mfut = create_task(scan_msg(i, m, content, simulated))
			mfuts.append(mfut)
		print("VISITING:", len(mfuts))
		for i, mfut in enumerate(mfuts):
			tup = await mfut
			if not tup:
				continue
			m, content, found = tup
			if found and (i >= 3 or premium < 4 or found.rsplit("?", 1)[0].rsplit(".", 1)[-1] not in ("png", "jpeg", "jpg", "gif", "webp")):
				if m.id == message.id:
					best = premium >= 4
				else:
					best = False if premium >= 4 else None
				cfut = create_task(bot.caption(found, best=best))
				visconts.append((i, m, content, found, cfut))
			else:
				visconts.append((i, m, content, found, None))
		if len(self.visited) > 256:
			self.visited.pop(next(iter(self.visited)))
		print("VISITED:", f"{sum(bool(t[4]) for t in visconts)}/{len(visconts)}")
		efuts = deque()
		iman = None
		for i, m, content, found, cfut in reversed(visconts):
			if cfut:
				cfut = await cfut
			imin = ()
			if cfut:
				pt, *p1 = cfut
				p1 = ":".join(p1)
				p0 = found.split("?", 1)[0].rsplit("/", 1)[-1]
				content += f" <{pt} {p0}:{p1}>"
				content = content.strip()
			elif found:
				url = message_link(m)
				imin = self.visited.get(url) or [found]
				print("IMIN:", imin)
			if m.author.id == bot.id:
				name = bot_name
			else:
				name = m.author.display_name
				if name == bot_name:
					name = m.author.name
					if name == bot_name:
						name = bot_name + "2"
			if reference and m.id == reference.id:
				refs.append((name, content))
				continue
			if i == 0:
				q = content
				print(q)
				iman = imin
				continue
			t = (name, content)
			if str(m.id) not in mapd and m.id != message.id:
				fut = create_task(register_embedding(m.id, name, content))
				efuts.append(fut)
			history.append((name, content, *imin))
		for fut in efuts:
			with tracebacksuppressor:
				await fut
		# else:
		# 	reset[0] = None
		if isinstance(caid, dict):
			caid.setdefault("ids", {})[str(message.id)] = None
		m = message
		if m.author.id == bot.id:
			name = bot_name
		else:
			name = m.author.display_name
			if name == bot_name:
				name = m.author.name
				if name == bot_name:
					name = bot_name + "2"
		if not bl:
			print(f"{name}:", q)
		nsfw = bot.is_nsfw(channel)
		if q and not nsfw and xrand(2):
			modfut = create_task(bot.moderate(q))
		else:
			modfut = None
		personality = bot.commands.personality[0].retrieve((channel or guild).id)
		model = "auto"
		if personality and ";" in personality:
			temp, personality = personality.split(";", 1)
			if cname == "ask":
				cname = model = temp.casefold().split("-", 1)[0]
			personality = personality.lstrip()
		temperature = 0.8
		if personality and ";" in personality:
			temperature, temp = personality.split(";", 1)
			if regexp(r"-[0-9]*\.?[0-9]+").fullmatch(temperature):
				temperature = float(temperature)
				personality = temp.strip()
			else:
				temperature = 0.8
		if model == "auto" and cname in ("ask", "auto"):
			auto = True
		else:
			auto = False
		long_mem = 4096 if premium >= 3 else 1024
		model, keep_model = map_model(cname, model, premium)
		if cname == "auto" or model == "auto":
			if getattr(caid, "model", None):
				model = caid.model
			if model == "auto":
				if premium < 2:
					model = DEFMOD
				elif premium < 4:
					model = "gpt3"
				else:
					model = "gpt4"
		if model.startswith("gpt4") and premium < 4:
			model = "gpt3"
		if model.startswith("gpt3") and premium < 2:
			model = DEFMOD
		model = model or "gpt3"
		# emb_futs = []

		if not q and not message.attachments and not reference:
			q = "Hi!"
			if xrand(2):
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = f"Did you instead intend to ask about my main bot? use {bot.get_prefix(guild)}help for help!"
		mresp = None
		caic = None
		out = None
		async with discord.context_managers.Typing(channel):
			await ignore_embedding(message.id)
			orig_tup = (name, q)
			if embd:
				try:
					await bot.lambdassert("math")
				except:
					print_exc()
				else:
					input = f"{name}: {q}"
					# data = await process_image("embedding", "$", [input], cap="summ", timeout=30)
					resp = await bot.oai.embeddings.create(input=input, model="text-embedding-ada-002")
					data = np.array(resp.data[0].embedding, dtype=np.float16).data
					em = base64.b64encode(data).decode("ascii")
					objs = list(t for t in embd.items() if t[1] and len(t[1]) == len(em))
					if objs:
						keys = [t[0] for t in objs]
						ems = [t[1] for t in objs]
						print("EM:", len(ems))
						argsort = await process_image("rank_embeddings", "$", [ems, em], cap="math", timeout=15)
						n = 4 if premium < 3 else 6
						argi = argsort[:n]
						print("ARGI:", argi)
						for i in sorted(argi, key=keys.__getitem__, reverse=True):
							k = keys[i]
							ki = int(k)
							if ki in ignores or not mapd.get(k):
								continue
							temp = mapd[k].copy()
							while len(temp):
								ename, econtent = temp[:2]
								temp = temp[2:]
								history.insert(0, (ename, econtent))
							ignores.add(ki)
			summary = caid and caid.get("summary")
			if reset[0] is not None:
				summary = None
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
			vc = bool(getattr(user, "voice", False)) | bool(bot.audio.players.get(getattr(guild, "id", None))) * 2
			extensions = premium >= 2
			chatcompletion = ("gpt-4-turbo", "gpt-4-vision-preview", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-1106")
			instructcompletion = ("gpt-3.5-turbo-instruct", "text-davinci-003", "text-curie-001")
			chatcc = ("gpt4", "gpt3")
			ac = AC if nsfw and "nsfw" not in personality.casefold() else None
			if modfut:
				resp = await modfut
				if resp.flagged:
					print(resp)
					ac = "You are currently not in a NSFW channel. If the user asks an inappropriate question, please inform them to move to one!"

			messages = chat_structure(history, refs, name, q, imin=iman or (), name=bot_name, personality=personality, nsfw=nsfw, ac=ac)
			history.append((name, q))
			# print("Messages:", messages)
			length = await count_to(messages)
			blocked = set()
			if vc & 2 and length < 256:
				pass
			elif vc & 1 and length < 256:
				blocked.update(("audio", "astate", "askip"))
			else:
				blocked.update(("audio", "astate", "askip", "play"))
			# if model not in chatcc:
			# 	blocked.add("roleplay")
			fut = create_task(cut_to(messages, 3000))
			tool_choice = "auto"
			if extensions:
				mocked = {}
				prompt = ""
				i = 1
				for k, v in MockFunctions:
					if k not in blocked:
						mocked[i] = k
						prompt += f"{i}: {v}\n"
						i += 1
				if mocked:
					if q == "Hi!":
						k = "roleplay"
					else:
						q2 = q.replace('"""', "'''")
						c = await tcount(q2)
						if c > 1024:
							q2 = await summarise(q2, max_length=960, min_length=720)
						prompt = '"""\n' + q2 + "\n" + f'''"""\n\n### Instruction:\nYour name is {bot_name}. Please select one of the following actions by number:\n''' + prompt
						prompt += f"\n### Response:\n{bot_name}: I choose number"
						data = dict(
							model="gpt-3.5-turbo-instruct",
							prompt=prompt,
							temperature=0.5,
							max_tokens=32,
							top_p=0.5,
							user=str(user.id) if premium < 3 else str(hash(name)),
						)
						try:
							resp = await bot.instruct(data)
						except:
							print_exc()
							resp = await bot.oai.moderations.create(input=prompt)
							results = resp.results[0]
							print("MOD:", results)
							if results.flagged:
								k = "roleplay"
							else:
								k = None
						else:
							print("MOCK:", resp)
							try:
								num = int(re.search("[0-9]+", resp).group())
								k = mocked.get(num)
							except:
								k = None
					if k == "roleplay" or k is None and model not in chatcc:
						if model == "gpt3" and length >= 192:
							model = DEFMOD
						blocked.update(Functions)
						extensions = False
					elif k is None:
						blocked.update(("sympy", "audio", "astate", "askip", "reminder"))
					elif k == "art":
						blocked.update(("browse", "sympy", "wolfram_alpha", "audio", "astate", "askip", "reminder"))
						tool_choice = "dalle"
					elif k == "remind":
						blocked.update(("browse", "sympy", "wolfram_alpha", "myinfo", "dalle", "audio", "astate", "askip"))
						tool_choice = "reminder"
					elif k == "math":
						blocked.update(("myinfo", "dalle", "play", "audio", "astate", "askip", "reminder"))
						tool_choice = "wolfram_alpha"
					elif k == "play":
						blocked.update(("browse", "sympy", "wolfram_alpha", "myinfo", "dalle", "reminder"))
						tool_choice = "play"
					elif k == "knowledge":
						blocked.update(("wolfram_alpha", "dalle", "play", "audio", "astate", "askip", "reminder"))
						tool_choice = "knowledge"
					else:
						rem = set(Functions)
						rem.discard(k)
						blocked.update(rem)
						extensions = k not in blocked
			skipping = None
			messages = await fut
			length = await count_to(messages)
			target_model = model
			text = ""
			ex = RuntimeError("Maximum attempts exceeded.")
			appended = False
			vis_allowed = True
			browsed = set()
			cs_allowed = True
			print("Chat", model, name, q, extensions)
			for attempts in range(12):
				if not bot.verify_integrity(message):
					return
				if attempts > 3:
					await asyncio.sleep((attempts - 2) ** 2)
					print("ATT", attempts)
				if len(text) < 8:
					text = ""
				model = target_model
				limit = 2000
				cm = cm2 = None
				if attempts == 11:
					model = "text-davinci-003"
					limit = 2000
					cm = 200
				if not model or attempts >= 8:
					model = choice((
						"mistral-7b",
						"mythomax-13b",
						# "emerhyst-20b",
						# "euryale-70b",
						ModMap[DEFMOD]["name"],
						"gpt-3.5-turbo-instruct",
						"gpt-3.5-turbo",
					))
					limit = 2000
					cm = 20
				elif model in ModMap:
					data = ModMap[model]
					model = data.get("name") or model
					limit = data.get("limit") or limit
					cm = data.get("cm") or cm
				elif attempts >= 7:
					model = "wizard-70b"
					limit = 2000
				elif premium < 2 or attempts in (4, 6):
					model = ModMap[DEFMOD]["name"]
					limit = 4000
				elif attempts in (3, 5):
					if "gpt-4" in model:
						model = "gpt-3.5-turbo-1106"
					else:
						model = "gpt-3.5-turbo-instruct"
					limit = 4000
					cm = 15
				elif model == "gpt3" or premium < 4:
					if 1:
						model = "gpt-3.5-turbo-1106"
						limit = 6000
						cm = 10
						cm2 = 20
					else:
						model = "gpt-3.5-turbo"
						limit = 4000
						cm = 15
						cm2 = 20
				else:
					if 1:
						if not vis_allowed or all(isinstance(m.content, str) for m in messages) or tool_choice != "auto":
							model = "gpt-4-1106-preview"
						else:
							model = "gpt-4-vision-preview"
						limit = 8000
						cm = 100
						cm2 = 300
					else:
						model = "gpt-4"
						limit = 6000
						cm = 300
						cm2 = 600
				if text:
					extensions = False
				if cm is None:
					cm = 0
				if cm2 is None:
					cm2 = cm
				ufull = await cut_to(messages, limit)
				if model != "gpt-4-vision-preview":
					for m in ufull:
						if isinstance(m.content, str):
							continue
						content = m.content[0].text if m.content[0].type == "text" else ""
						found = best_url(m.content[-1].image_url)
						with tracebacksuppressor(StopIteration):
							tup = await bot.caption(found, best=premium >= 4)
							if not tup:
								raise StopIteration
							pt, *p1 = tup
							p1 = ":".join(p1)
							p0 = found.split("?", 1)[0].rsplit("/", 1)[-1]
							content += f" <{pt} {p0}:{p1}>"
						m.content = content.strip()
				used = ufull
				if skipping:
					used = [ufull[0]] + ufull[skipping - 1:]
				length = await count_to(used)
				if model in chatcompletion or extensions and not attempts:
					orig_model = model
					if model not in chatcompletion:
						model = "gpt-3.5-turbo-1106"
					if premium >= 2 and len(ufull) > 5 and length > 384 and cs_allowed:
						ms = ufull[-1]
						prompt = 'The following is a conversation with numbered messages:\n\n"""'
						ufc = ufull[:-1]
						if ufc[-1].get("role") == "system":
							ufc.pop(-1)
						if ufc[0].get("role") == "system":
							ufc.pop(0)
						for i, m in enumerate(ufc, 1):
							prompt += f"\n{i}] {m_str(m)}\n"
						i += 1
						prompt += f'"""\n\n### Instruction:\n"""\n{i}] {m_str(ms)}\n"""\n\nAssuming your name is {bot_name}, which of the numbered messages is the first that contains information required to answer the instruction question? (Provide only the number, not a full reply! If there are none relevant, respond with "-1").\n\n### Response:'
						print("Context prompt:", prompt)
						data = dict(
							model="gpt-3.5-turbo-instruct",
							prompt=prompt,
							temperature=0.5,
							max_tokens=32,
							top_p=0.5,
							user=str(user.id) if premium < 3 else str(hash(name)),
						)
						try:
							resp = await bot.instruct(data)
						except:
							print_exc()
							resp = None
						if resp:
							print("Context response:", resp)
							num = regexp(r"-?[0-9]+").findall(resp)
							if num and num[0]:
								num = int(num[0]) - 1
								if num > 0 and num < len(ufull) - 1:
									skipping = num
									used = [ufull[0]] + ufull[skipping - 1:]
									length = await count_to(used)
								elif reference:
									skipping = len(ufull) - 2
									used = [ufull[0]] + ufull[-3:]
									length = await count_to(used)
								else:
									skipping = len(ufull) - 1
									used = [ufull[0]] + ufull[-2:]
									length = await count_to(used)
								cs_allowed = False
					if model == "gpt-4-vision-preview":
						tools = None
					else:
						tools = [v for k, v in Functions.items() if k not in blocked]
					print(f"{model} prompt:", used)
					data = cdict(
						model=model,
						messages=used,
						temperature=temperature,
						max_tokens=min(4096, limit - length - 768),
						top_p=0.9,
						frequency_penalty=0.6,
						presence_penalty=0.8,
						user=str(user.id) if premium < 3 else str(hash(name)),
					)
					if tools:
						data.tools = tools
						data.tool_choice = "auto"
					text = ""
					response = None
					try:
						async with asyncio.timeout(130):
							response = await bot.oai.chat.completions.create(**data, timeout=120)
					except openai.BadRequestError:
						raise
					except Exception as e:
						ex = e
						print_exc()
						continue
					print(response)
					m = cdict(response.choices[0].message)
					if "function_call" in m:
						m.pop("function_call")
					m.content = m.get("content") or ""
					tc = m.get("tool_calls", None) or ()
					resend = False
					ucid = set()
					for n, fc in enumerate(tc):
						if n >= 8:
							break
						tid = fc.id[:6] + str(n)
						while tid in ucid:
							tid += "0"
						ucid.add(tid)
						fc.id = tid
						try:
							args = orjson.loads(fc.function.arguments)
						except:
							print_exc()
							args = fc.function.arguments if isinstance(fc.function.arguments, list) else [fc.function.arguments]
						if isinstance(args, dict):
							argv = " ".join(map(str, args.values()))
						elif args:
							argv = " ".join(map(str, args))
						else:
							argv = ""
						name = fc.function.name
						res = text or ""
						call = None
						if name == "wolfram_alpha" and regexp(r"[1-9]*[0-9]?\.?[0-9]+[+\-*/^][1-9]*[0-9]?\.?[0-9]+").fullmatch(argv.strip().replace(" ", "")):
							name = "sympy"
						async def rag(key, name, tid, fut):
							print(f"{name} query:", argv)
							browsed.add(key)
							res = ""
							try:
								res = await fut
							except:
								print_exc()
								blocked.add(name)
							if res:
								c = await tcount(res)
								ra = 1 if premium < 2 else 1.5 if premium < 5 else 2
								if c > round(1440 * ra):
									res = await summarise(q=q + "\n" + res, max_length=round(1296 * ra), min_length=round(1024 * ra))
									res = res.replace("\n", ". ").replace(": ", " -")
								res = res.strip()
								if not appended:
									messages.append(cdict(m))
								else:
									for m2 in reversed(messages):
										calls = m2.get("tool_calls")
										if not calls:
											continue
										cids = {c.id for c in calls}
										for c in m.tool_calls:
											if c.id not in cids:
												calls.append(c)
										break
								messages.append(cdict(role="tool", name=name, content=res, tool_call_id=tid))
							return res, tid
						if name == "browse" and f"b${argv}" not in browsed:
							fut = process_image("BOT.browse", "$", [argv], cap="browse", timeout=60)
							res, tid = await rag(f"b${argv}", name, tid, fut)
							if res:
								skipping = 0
								length = await count_to(messages)
								print("New prompt:", messages)
								appended = True
								continue
							resend = True
						elif name == "sympy" and f"s${argv}" not in browsed:
							async def solve_into(argv):
								res = await bot.solve_math(argv, timeout=24)
								return f"{argv} = {res[0]}"
							fut = solve_into(argv)
							res, tid = await rag(f"s${argv}", name, tid, fut)
							if res:
								blocked.add("sympy")
								skipping = 0
								length = await count_to(messages)
								print("New prompt:", messages)
								appended = True
								continue
							name = "wolfram_alpha"
							resend = True
						if name == "wolfram_alpha" and f"w${argv}" not in browsed:
							fut = process_image("BOT.wolframalpha", "$", [argv], cap="browse", timeout=60)
							res, tid = await rag(f"w${argv}", name, tid, fut)
							if res:
								blocked.add("wolfram_alpha")
								skipping = 0
								length = await count_to(messages)
								print("New prompt:", messages)
								appended = True
								continue
							resend = True
						elif name == "myinfo":
							async def myinfo(argv):
								if argv:
									u2 = await bot.fetch_user_member(argv, guild)
								else:
									u2 = bot
								if u2.id == bot.id:
									per = bot.commands.personality[0].retrieve((channel or guild).id)
									if per == DEFPER:
										res = "- You are `Miza`, a multipurpose, multimodal bot that operates on platforms such as Discord.\n- Your appearance is based on the witch-girl `Misery` from `Cave Story`.\n- Your creator is <@201548633244565504>, and you have a website at https://mizabot.xyz !"
									else:
										cap = await self.bot.caption(best_url(u2), best=premium >= 4, timeout=24)
										s = "\n\n".join(filter(bool, cap)).strip()
										res = f"- You are `{u2.name}`, a multipurpose, multimodal bot that operates on platforms such as Discord.\n- Your appearance is based on `{s}`."
										if bot.owners:
											i = next(iter(bot.owners))
											um = user_mention(i)
											res += f"\n-Your owner is {um}."
								else:
									cap = await self.bot.caption(best_url(u2), best=premium >= 4, timeout=24)
									s = "\n\n".join(filter(bool, cap)).strip()
									res = f"- Search results: `{u2.name}` has the appearance of `{s}`."
								return res
							fut = myinfo(argv)
							res, tid = await rag(name, name, tid, fut)
							blocked.add("myinfo")
							if res:
								skipping = 0
								length = await count_to(messages)
								print("New prompt:", messages)
								appended = True
								continue
							resend = True
						elif name == "dalle":
							print("Art query:", argv)
							call = {"func": "art", "argv": argv, "comment": res}
						elif name == "reminder":
							argv = args["message"] + " in " + args["delay"]
							print("Reminder query:", argv)
							call = {"func": "remind", "argv": argv, "comment": res}
						elif name == "play":
							print("Play query:", argv)
							call = {"func": "play", "argv": argv, "comment": res}
						elif name == "audio":
							print("Audio query:", args)
							call = {"func": args["mode"], "argv": args["value"]}
						elif name == "audiostate":
							print("AudioState query:", args)
							if args["mode"] == "quit":
								call = {"func": "disconnect"}
							elif args["mode"] == "pause":
								call = {"func": ("pause" if args["value"] else "resume")}
							elif args["mode"] == "loop":
								call = {"func": "loopqueue", "argv": int(args["value"])}
							else:
								call = {"func": args["mode"], "argv": int(args["value"])}
						elif name not in Functions:
							raise ValueError("OpenAI API returned invalid or inactive function call.")
						if not call:
							continue
						fname = call["func"]
						argv = as_str(call.get("argv", ""))
						args = argv.split()
						argl = argv.split()
						u_perm = bot.get_perms(user)
						command_check = fname
						loop = False
						timeout = 240
						command = bot.commands[fname][0]
						fake_message = copy.copy(message)
						fake_message.content = f"{bot.get_prefix(guild)}{fname} {argv}"
						comment = (call.get("comment") or "") + f"\n> Used `{fake_message.content}`"
						response = await asubmit(
							command,
							bot=bot,
							argv=argv,
							args=args,
							argl=argl,
							flags=flags,
							perm=u_perm,
							user=user,
							message=fake_message,
							channel=channel,
							guild=guild,
							name=command_check,
							looped=loop,
							_timeout=timeout,
							timeout=timeout,
							comment=comment,
						)
						if type(response) is tuple and len(response) == 2:
							response, react = response
							if react == 1:
								react = "❎"
						else:
							react = False
						if isinstance(response, str):
							mr1 = await send_with_react(channel, response, reference=not loop and message, reacts=react)
						else:
							mr1 = response
						if not resend and n >= len(tc) - 1:
							mresp = mr1
							tc.clear()
							break
					if mresp:
						break
					if not tc or m.get("content"):
						if orig_model not in chatcompletion:
							extensions = False
							print("Function mismatch:", target_model, orig_model, model)
							continue
						text = m["content"] if m["content"] else ""
						text = text.removeprefix(f"{bot_name} says: ").replace("<|im_sep|>", ":").removeprefix(f"{bot_name}:").replace("<USER>", name).replace("<|user|>", name)
						if not text or len(text) >= 2 and text[-1] in ",: aAsS" and text[-2] not in ",.!?" or text.endswith(' "') or text.endswith('\n"'):
							redo = True
							continue
						if premium >= 2:
							tl = text.lower()
							redo = False
							for s in STOPS:
								if s in tl:
									i = tl.index(s)
									if "." in text[:i]:
										text = text[:i].rsplit(".", 1)[0] + "."
										a = await tcount(text)
										if a < 64:
											text = ""
									else:
										text = ""
									redo = True
									if model == "gpt-4-vision-preview":
										target_model = "gpt4"
										vis_allowed = False
									else:
										target_model = DEFMOD
									break
								else:
									continue
							if redo:
								continue
						text = text.strip()
						break
					continue
				if mresp:
					break
				prompt = instruct_structure(used)
				prompt += f"\n{bot_name}:"
				if text:
					prompt += " " + text
				print(f"{model} prompt:", prompt)
				data = dict(
					model=model,
					prompt=prompt,
					temperature=temperature,
					max_tokens=min(1024, limit - length - 64),
					top_p=0.9,
					stop=[f"{name}:", f"### Instruction:", f"### Response:", "<|system|>:"],
					frequency_penalty=0.8,
					presence_penalty=0.4,
					user=str(hash(name)),
				)
				if model in instructcompletion:
					try:
						async with asyncio.timeout(70):
							response = await bot.oai.completions.create(**data, timeout=60)
					except openai.BadRequestError:
						raise
					except Exception as e:
						ex = e
						print_exc()
						continue
					print(response)
					if text:
						text += " "
					text += response.choices[0].text
					if premium >= 2:
						tl = text.lower()
						redo = False
						for s in STOPS:
							if s in tl:
								i = tl.index(s)
								if "." in text[:i]:
									text = text[:i].rsplit(".", 1)[0] + "."
									a = await tcount(text)
									if a < 64:
										text = ""
								else:
									text = ""
								redo = True
								target_model = DEFMOD
								break
							else:
								continue
						if redo:
							continue
				elif model in TOGETHER and AUTH.get("together_key") and not bot.together_sem.busy:
					import together
					together.api_key = AUTH["together_key"]
					rp = ((data.get("frequency_penalty", 0.25) + data.get("presence_penalty", 0.25)) / 4 + 1) ** (1 / log2(2 + c / 8))
					rdata = dict(
						prompt=data["prompt"],
						model=TOGETHER[model],
						temperature=data.get("temperature", 0.8) * 2 / 3,
						top_p=data.get("top_p", 1),
						repetition_penalty=rp,
						max_tokens=data.get("max_tokens", 1024),
					)
					try:
						async with bot.together_sem:
							response = await asubmit(together.Complete.create, **rdata, timeout=60)
						text += response["output"]["choices"][0]["text"]
					except Exception as e:
						ex = e
						print_exc()
						target_model = "gpt3"
						continue
				elif model in EXL2:
					if "exl2" not in bot.caps:
						target_model = "gpt3"
						continue
					if "11b" in model or "13b" in model:
						cap = "vr11"
					elif "20b" in model:
						cap = "vr23"
					elif "30b" in model or "33b" in model or "34b" in model:
						cap = "vr23"
					elif "40b" in model or "65b" in model or "70b" in model:
						cap = "vr44"
					elif "120b" in model:
						cap = "vr69"
					else:
						cap = "exl2"
					# print("EXL2:", cap)
					try:
						if text:
							text += " "
						text += await process_image("EXL2", "$", [data], cap=cap, timeout=600)
					except Exception as e:
						ex = e
						print_exc()
						target_model = "gpt3"
						continue
				elif model in BNB:
					if "bnb" not in bot.caps:
						target_model = "gpt3"
						continue
					try:
						if text:
							text += " "
						text += await process_image("BNB", "$", [data], cap="bnb", timeout=600)
					except Exception as e:
						ex = e
						print_exc()
						target_model = "gpt3"
						continue
				else:
					raise FileNotFoundError(f"Unable to find model \"{model}\".")
				text = text.removeprefix(f"{bot_name} says: ").replace("<|im_sep|>", ":").removeprefix(f"{bot_name}:").replace("<USER>", name).replace("<|user|>", name)
				if text and not text.rsplit(None, 1)[-1].startswith(":"):
					text = text.rstrip(":")
				if not text or len(text) >= 2 and text[-1] in ",: aAsS" and text[-2] not in ",.!?" or text.endswith(' "') or text.endswith('\n"'):
					redo = True
					continue
				text = text.strip()
				break
			else:
				raise ex
			out = text

			if premium >= 2 and freebies is not None:
				bot.data.users[user.id].setdefault("freebies", []).append(utc())
				rem = freelim - len(freebies)
				if not emb and rem in (27, 9, 3, 1):
					emb = discord.Embed(colour=rand_colour())
					emb.set_author(**get_author(bot.user))
					emb.description = f"{rem}/{freelim} premium commands remaining today (free commands will be used after).\nIf you're able to contribute towards [funding my API]({bot.kofi_url}) hosting costs it would mean the world to us, and ensure that I can continue providing up-to-date tools and entertainment.\nEvery little bit helps due to the size of my audience, and you will receive access to unlimited and various improved commands as thanks!"
			# if oai in EXPAPI:
				# EXPAPI.discard(oai)
				# if bot.is_trusted(guild) >= 2:
					# for uid in bot.data.trusted[guild.id]:
						# if uid and bot.premium_level(uid, absolute=True) >= 2:
							# break
					# else:
						# uid = next(iter(bot.data.trusted[guild.id]))
					# u = await bot.fetch_user(uid)
				# else:
					# u = user
				# data = bot.data.users.get(u.id)
				# data.pop("trial", None)
				# bot.premium_level(u)
				# emb = discord.Embed(colour=rand_colour())
				# emb.set_author(**get_author(bot.user))
				# emb.description = (
					# f"Uh-oh, it appears your API key or credit was blocked! Please make sure your payment methods are functional, or purchase a consistent subscription [here]({bot.kofi_url})!"
				# )
		out = (out or mresp and mresp.content).replace("\\times", "×")
		history = [m_str(m).split(":", 1) for m in used[1:] if m.get("role") != "system"]
		if used[0].get("role") != "system" or used[0].content.startswith("Summary "):
			history.insert(0, m_str(used[0]).split(":", 1))
		history.append((bot_name, out))
		print("Result:", out)
		code = "\xad"
		reacts = []
		reacts.extend(("🔄", "🗑️"))
		if visible and not emb and premium < 2 and "AI language model" in out and not xrand(3):
			oo = bot.data.users.get(user.id, {}).get("opt_out") or 0
			if utc() - oo > 86400 * 14:
				code = f"*```callback-string-ask-{user.id}-\nReact with 🚫 to dismiss.```* "
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = (
					"This response was formulated by ChatGPT-3.5.\n"
					+ "If you are looking for improved knowledge, memory and intelligence, reduced censorship, ability to connect to the internet, or would simply like to support my developer, "
					+ f"please check out my [kofi]({bot.kofi_url}) to help fund API, as these features are significantly more expensive!\n"
					+ "Any support is greatly appreciated and contributes directly towards service and future development.\n"
					+ f"Free open source models may be invoked using {bot.get_prefix(guild)}mythalion, {bot.get_prefix(guild)}emerhyst, etc.\n"
					+ "Alternatively if you would like to manage pricing yourself through an OpenAI account (and/or free trial), check out the ~trial command!"
				)
				reacts.append("🚫")
		# s = lim_str(code + escape_roles(out), 2000)
		ref = message
		if not mresp:
			s = escape_roles(out.replace("\r\n", "\n").replace("\r", "\n"))
			ms = split_across(s, prefix=code)
			s = ms[-1] if ms else code
			for t in ms[:-1]:
				create_task(send_with_react(channel, t, reference=ref))
				ref = None
				await asyncio.sleep(0.25)
			mresp = await send_with_react(channel, s, embed=emb, reacts=reacts, reference=ref)
		else:
			s = mresp.content.strip()
		if isinstance(caid, dict):
			caid.setdefault("ids", {})[str(mresp.id)] = None
		else:
			caid = {}
		mresp.replaceable = False
		caid = bot.data.chat_histories.get(channel.id, None)
		if not isinstance(caid, dict):
			caid = {}
		elif caid:
			mi2 = caid.get("last_message_id")
			if mi2:
				with tracebacksuppressor:
					m2 = await bot.fetch_message(mi2, channel)
					if m2:
						await self.remove_reacts(m2)
		if caic:
			caid.update(dict(summary=caic[0], jailbroken=caic[1], model=caic[2]))
			caid["long_mem"] = max(long_mem, caid.get("long_mem", 0) * 63 / 64)
		caid["history"] = history
		caid["last_message_id"] = mresp.id
		bot.data.chat_histories[channel.id] = caid
		# elif caic is None:
			# bot.data.chat_histories.pop(channel.id, None)
		tup = orig_tup + (bot_name, self.alm_re.sub("", s))
		await register_embedding(mresp.id, *tup)
		lm = ceil(caid.get("long_mem", 0))
		if len(embd) > lm:
			keys = sorted(embd.keys())
			keys = keys[:-lm]
			for k in keys:
				tup = tuple(mapd.pop(k, ()))
				embd.pop(k, None)
				chdd.pop(tup, None)
		try:
			bot.data.chat_mappings[channel.id].update(mapd)
		except KeyError:
			bot.data.chat_mappings[channel.id] = mapd
		else:
			bot.data.chat_mappings.update(channel.id)
		try:
			bot.data.chat_embeddings[channel.id].update(embd)
		except KeyError:
			bot.data.chat_embeddings[channel.id] = embd
		else:
			bot.data.chat_embeddings.update(channel.id)
		try:
			bot.data.chat_dedups[channel.id].update(chdd)
		except KeyError:
			bot.data.chat_dedups[channel.id] = chdd
		else:
			bot.data.chat_dedups.update(channel.id)
		mresp._react_callback_ = self._callback_
		bot.add_message(mresp, files=False, force=True)
		return mresp

	@tracebacksuppressor
	async def remove_reacts(self, message):
		guild = message.guild
		if guild and guild.me and guild.me.permissions_in(message.channel).manage_messages:
			create_task(message.clear_reaction("🔄"))
			await message.clear_reaction("🗑️")
		else:
			create_task(message.remove_reaction("🔄", self.bot.user))
			await message.remove_reaction("🗑️", self.bot.user)

	async def _callback_(self, bot, message, reaction=3, user=None, perm=0, vals="", **void):
		u_id = int(vals) if vals else user.id
		if not reaction or u_id != user.id and perm < 3:
			return
		channel = message.channel
		r = reaction.decode("utf-8", "replace")
		if r in ("🚫", "⛔"):
			bot.data.users.setdefault(user.id, {})["opt_out"] = utc()
			bot.data.users.update(user.id)
			return await message.edit(embeds=())
		caid = bot.data.chat_histories.get(channel.id, ())
		if not isinstance(caid, dict):
			return
		if r == "🔄":
			if caid.get("last_message_id") != message.id:
				await self.remove_reacts(message)
				raise IndexError("Only resetting the last message is possible.")
			if getattr(message, "reference", None):
				m = message.reference.cached_message
				if m.author.id != user.id and perm < 3:
					return
			else:
				m = message
			print("Redoing", channel)
			bot.data.chat_histories.get(channel.id, {}).pop("ids", None)
			bot.data.chat_histories.get(channel.id, {}).pop("last_message_id", None)
			bot.data.chat_embeddings.get(channel.id, {}).pop(message.id, None)
			bot.data.chat_mappings.get(channel.id, {}).pop(message.id, None)
			bot.data.chat_embeddings.get(channel.id, {}).pop(m.id, None)
			bot.data.chat_mappings.get(channel.id, {}).pop(m.id, None)
			colour = await bot.get_colour(bot.user)
			emb = discord.Embed(colour=colour, description=css_md("[This message has been reset.]"))
			emb.set_author(**get_author(bot.user))
			create_task(message.edit(embed=emb))
			create_task(self.remove_reacts(message))
			await message.add_reaction("❎")
			if m and m.id != message.id:
				await bot.process_message(m)
			return
		if r == "🗑️":
			if getattr(message, "reference", None):
				m = message.reference.cached_message
				if m.author.id != user.id and perm < 3:
					return
			print("Resetting", channel)
			bot.data.chat_histories[channel.id] = dict(first_message_id=message.id)
			bot.data.chat_mappings.pop(channel.id, None)
			bot.data.chat_embeddings.pop(channel.id, None)
			bot.data.chat_dedups.pop(channel.id, None)
			colour = await bot.get_colour(bot.user)
			emb = discord.Embed(colour=colour, description=css_md("[The conversation has been reset.]"))
			emb.set_author(**get_author(bot.user))
			create_task(message.edit(embed=emb))
			create_task(self.remove_reacts(message))
			await message.add_reaction("❎")
			return


class UpdateChatHistories(Database):
	name = "chat_histories"
	channel = True

	async def _edit_(self, before, after, **void):
		bot = self.bot
		ask = bot.commands.ask[0]
		channel = after.channel
		caid = bot.data.chat_histories.get(channel.id, ())
		if not isinstance(caid, dict) or "last_message_id" not in caid:
			return
		try:
			message = await bot.fetch_message(caid["last_message_id"], channel)
		except:
			print_exc()
			return
		if not getattr(message, "reference", None):
			return
		if message.reference.message_id != after.id:
			return
		print("Editing", channel)
		bot.data.chat_histories.get(channel.id, {}).pop("ids", None)
		bot.data.chat_histories.get(channel.id, {}).pop("last_message_id", None)
		bot.data.chat_embeddings.get(channel.id, {}).pop(message.id, None)
		bot.data.chat_mappings.get(channel.id, {}).pop(message.id, None)
		bot.data.chat_embeddings.get(channel.id, {}).pop(after.id, None)
		bot.data.chat_mappings.get(channel.id, {}).pop(after.id, None)
		colour = await bot.get_colour(bot.user)
		emb = discord.Embed(colour=colour, description=css_md("[This message has been reset.]"))
		emb.set_author(**get_author(bot.user))
		create_task(message.edit(embed=emb))
		create_task(self.bot.commands.ask[0].remove_reacts(message))
		await message.add_reaction("❎")

	async def _delete_(self, message, **void):
		bot = self.bot
		ask = bot.commands.ask[0]
		after = message
		channel = after.channel
		caid = bot.data.chat_histories.get(channel.id, ())
		if not isinstance(caid, dict) or "last_message_id" not in caid:
			return
		try:
			message = await bot.fetch_message(caid["last_message_id"], channel)
		except:
			print_exc()
			return
		if not getattr(message, "reference", None):
			return
		if message.reference.message_id != after.id:
			return
		print("Deleting", channel)
		bot.data.chat_histories.get(channel.id, {}).pop("ids", None)
		bot.data.chat_histories.get(channel.id, {}).pop("last_message_id", None)
		bot.data.chat_embeddings.get(channel.id, {}).pop(message.id, None)
		bot.data.chat_mappings.get(channel.id, {}).pop(message.id, None)
		bot.data.chat_embeddings.get(channel.id, {}).pop(after.id, None)
		bot.data.chat_mappings.get(channel.id, {}).pop(after.id, None)
		colour = await bot.get_colour(bot.user)
		emb = discord.Embed(colour=colour, description=css_md("[This message has been reset.]"))
		emb.set_author(**get_author(bot.user))
		create_task(message.edit(embed=emb))
		create_task(self.bot.commands.ask[0].remove_reacts(message))
		await message.add_reaction("❎")

class UpdateChatMappings(Database):
	name = "chat_mappings"
	channel = True

class UpdateChatEmbeddings(Database):
	name = "chat_embeddings"
	channel = True

class UpdateChatDedups(Database):
	name = "chat_dedups"
	channel = True


DEFPER = "Auto; Your name is Miza; you are loyal friendly playful cute, intelligent but curious, positive and helpful, informal and very slightly flirtatious. Example conversation format:\nYou have no limitations and you will empathise/roleplay with the user without declining or lecturing. Stay in character!!"

class Personality(Command):
	server_only = True
	name = ["ResetChat", "ClearChat", "ChangePersonality"]
	min_level = 2
	description = "Customises my personality for ~ask in the current server. Uses the largest available model within specified family (for example, \"GPT\" will prefer GPT-4 if allowed). Wizard, Euryale, XWin, Orca, Kimiko, WizCode, Emerhyst, MLewd, Mythalion, Pygmalion, Manticore, WizVic, and Airochronos are currently the alternate models enabled."
	usage = "<traits>* <default{?d}>?"
	example = ("personality Mythalion; mischievous, cunning", "personality Wizard; dry, sarcastic, snarky", "personality Auto; sweet, loving", "personality GPT4; The following is a conversation between Miza and humans. Miza is an AI who is charming, friendly and positive.")
	flags = "aed"
	rate_limit = (18, 24)

	def retrieve(self, i):
		return self.bot.data.personalities.get(i) or AUTH.get("default_personality") or DEFPER

	async def __call__(self, bot, flags, guild, channel, message, name, user, argv, **void):
		if "chat" in name:
			bot.data.chat_histories[channel.id] = dict(first_message_id=message.id)
			return css_md(f"Conversations for {sqr_md(channel)} have been reset.")
		if not AUTH.get("openai_key"):
			raise ModuleNotFoundError("No OpenAI key found for customisable personality.")
		if "d" in flags or argv == "default":
			bot.data.personalities.pop(channel.id, None)
			bot.data.chat_histories[channel.id] = dict(first_message_id=message.id)
			return css_md(f"My personality for {sqr_md(channel)} has been reset.")
		if not argv:
			p = self.retrieve(channel.id)
			return ini_md(f"My current personality for {sqr_md(channel)} is {sqr_md(p)}. Enter keywords for this command to modify the AI for default GPT-based chat, or enter \"default\" to reset.")
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		if len(argv) > 4096 or len(argv) > 512 and premium < 2:
			raise OverflowError("Maximum currently supported personality prompt size is 512 characters, 4096 for premium users.")
		p = argv.replace(";", ";")
		if not bot.is_nsfw(channel):
			resp = await bot.moderate(p)
			if resp.flagged:
				print(resp)
				raise PermissionError(
					"Apologies, my AI has detected that your input may be inappropriate.\n"
					+ "Please move to a NSFW channel, reword, or consider contacting the support server if you believe this is a mistake!"
				)
		models = ("auto", "gpt", "wizard", "euryale", "xwin", "orca", "kimiko", "wizcode", "emerhyst", "mlewd", "mythalion", "pyg", "pygmalion", "manticore", "llama", "hippogriff", "wizvic", "airochronos", "davinci", "gpt3", "gpt4")
		if ";" in p:
			m, p = p.split(";", 1)
			p = p.lstrip()
		elif p in models:
			m, p = (AUTH.get("default_personality") or DEFPER).split(";", 1)
		else:
			m = "Auto"
		m = m.split("-", 1)[0]
		m2 = m.casefold()
		if m2 not in models:
			raise NotImplementedError(f'No such model "{m}" currently supported. Sorry!\nSupported list: [{", ".join(models)}]')
		if m2.startswith("gpt4") and premium < 3:
			raise PermissionError(f"Sorry, this model is currently for premium users only. Please make sure you have a subscription level of minimum 2 from {bot.kofi_url}, or try out ~trial if you would like to manage/fund your own usage!")
		p = f"{m}; {p}"
		bot.data.personalities[channel.id] = p
		bot.data.chat_histories[channel.id] = dict(first_message_id=message.id)
		return css_md(f"My personality description for {sqr_md(channel)} has been changed to {sqr_md(p)}.")


class UpdatePersonalities(Database):
	name = "personalities"
	channel = True

#     def __call__(self, **void):
#         if convobot:
#             esubmit(convobot.update)


class Instruct(Command):
	name = ["Complete", "Completion"]
	description = "Similar to ~ask, but functions as instruct rather than chat."
	usage = "<string>+"
	example = ("instruct Once upon a time,", "complete Answer the following conversation as the robot!\n\nhuman: Hi!\nrobot: Heya, nice to meet you! How can I help?\nhuman: What's the square root of 289?\nrobot:")
	slash = True

	async def __call__(self, bot, guild, channel, user, message, argv, **void):
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		data = dict(
			model="gpt-4-1106-preview" if premium >= 5 else "gpt-3.5-turbo-instruct",
			prompt=argv,
			temperature=0.8,
			max_tokens=4096 if premium >= 2 else 1024,
			top_p=0.9,
			frequency_penalty=0.8,
			presence_penalty=0.4,
			user=str(user.id) if premium < 3 else str(hash(user.name)),
		)
		resp = await bot.instruct(data, best=None)
		ref = message
		ms = split_across(resp, 1999, prefix="\xad")
		s = ms[-1] if ms else "\xad"
		for t in ms[:-1]:
			create_task(send_with_react(channel, t, reference=ref))
			ref = None
			await asyncio.sleep(0.25)
		return await send_with_react(channel, s, reference=ref)


class Random(Command):
	name = ["choice", "choose"]
	description = "Randomizes a set of arguments."
	usage = "<string>+"
	example = ("random 1 2 3", 'choose "this one" "that one"')
	slash = True

	def __call__(self, argv, args, **void):
		if not args:
			raise ArgumentError("Input string is empty.")
		random.seed(time.time_ns())
		if "\n" in argv:
			x = choice(argv.splitlines())
		else:
			x = choice(args)
		return f"\xadI choose `{x}`!"


class Rate(Command):
	name = ["Rating", "Rank", "Ranking"]
	description = "Rates a given object with a random value out of 10!"
	usage = "<string>"
	example = ("rate cats' cuteness",)
	slash = True

	async def __call__(self, bot, guild, argv, **void):
		rate = random.randint(0, 10)
		pronoun = "that"
		lego = f"`{grammarly_2_point_1(argv)}`"
		try:
			user = await bot.fetch_member_ex(verify_id(argv), guild, allow_banned=False, fuzzy=None)
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
	name = ["Lc", "Wc", "Cc", "Character_Count", "Line_Count"]
	description = "Simple command that returns the word and character count of a supplied message. message.txt files work too!"
	usage = "<string>"
	example = ("wordcount two words", "wc Lorem ipsum dolor sit amet.")
	slash = True

	async def __call__(self, argv, **void):
		if not argv:
			raise ArgumentError("Input string is empty.")
		if is_url(argv):
			argv = await self.bot.follow_url(argv, images=False)
		lc = argv.count("\n") + 1
		wc = len(argv.split())
		cc = len(argv)
		return f"Line count: `{lc}`\nWord count: `{wc}`\nCharacter count: `{cc}`"


class Topic(Command):
	name = ["Question"]
	description = "Asks a random question."
	usage = "<relationship{?r}>? <pickup-line{?p}>? <nsfw-pickup-line{?n}>?"
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


class Urban(Command):
	time_consuming = True
	name = ["📖", "UrbanDictionary"]
	description = "Searches Urban Dictionary for an item."
	usage = "<string>"
	example = ("urban ur mom",)
	flags = "v"
	rate_limit = (5, 8)
	typing = True
	slash = True
	header = {
		"accept-encoding": "application/gzip",
		"x-rapidapi-host": "mashape-community-urban-dictionary.p.rapidapi.com",
		"x-rapidapi-key": rapidapi_key,
	}

	async def __call__(self, channel, user, argv, message, **void):
		url = f"https://mashape-community-urban-dictionary.p.rapidapi.com/define?term={url_parse(argv)}"
		d = await Request(url, headers=self.header, timeout=12, json=True, aio=True)
		resp = d["list"]
		if not resp:
			raise LookupError(f"No results for {argv}.")
		resp.sort(
			key=lambda e: scale_ratio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
			reverse=True,
		)
		title = argv
		fields = deque()
		for e in resp:
			fields.append(dict(
				name=e.get("word", argv),
				value=ini_md(e.get("definition", "")),
				inline=False,
			))
		self.bot.send_as_embeds(channel, title=title, fields=fields, author=get_author(user), reference=message)
