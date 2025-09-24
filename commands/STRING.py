# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

try:
	import httpcore
	httpcore.SyncHTTPTransport = None # Remove outdated dependency typecheck
	import googletrans
except Exception:
	print_exc()
	googletrans = None

try:
	rapidapi_key = AUTH["rapidapi_key"]
	if not rapidapi_key:
		raise
except:
	rapidapi_key = None
	print("WARNING: rapidapi_key not found. Unable to search Urban Dictionary.")


class Translate(Command):
	name = ["TR"]
	description = "Translates a string into another language."
	usage = "<0:engine(google|command-r-plus|gpt-3.5-turbo-instruct)>? <2:src_language>? <1:dest_languages>* <-1:string>"
	example = ("translate english 你好", "tr gpt-4.1-mini chinese bonjour, comment-t'appelles-tu?", "translate gpt-3.5-turbo-instruct auto spanish french italian thank you!")
	flags = "v"
	rate_limit = (6, 9)
	slash = True
	ephemeral = True
	LLMs = set(("google",) + tuple(ai.available))
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
		if spl[0].casefold() in self.LLMs:
			engine = spl.pop(0).casefold()
		elif spl[0].casefold() == "auto":
			engine = "gpt-5-mini"
			spl.pop(0)
		else:
			engine = "gpt-5-mini"
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
		text = " ".join(spl).removeprefix("\\")
		urls = find_urls(text)
		for url in urls:
			text = text.replace(url, " ")
		emojis = find_urls(text)
		for emoji in emojis:
			if emoji.startswith("<") and ":" in emoji:
				e = ":" + emoji.split(":", 1)[-1].split(":", 1)[0] + ":"
			else:
				continue
			text = text.replace(emoji, e)
		text = single_space(text).strip()
		if not text:
			raise ArgumentError("Input string is empty.")
		translated = {}
		comments = {}

		if src.casefold() == "auto":
			try:
				resp2 = await asubmit(self.trans.translate, text, src="auto", dest="en")
				src2 = resp2.src.casefold()
			except:
				print_exc()
				resp2 = await self.det(text)
				src2 = resp2.src.casefold() if resp2 else src.casefold()
		elif src.casefold() != "en":
			resp2 = asubmit(self.trans.translate, text, src="auto", dest="en")
			src2 = src.casefold()
		else:
			resp2 = None
			src2 = src.casefold()

		odest = tuple(dests)
		dests = [d for d in dests if not self.equiv(d, src2)] or [dests[0]]
		# if len(odest) != len(dests):
		#     translated[-1] = text
		#     odest = (src2,) + tuple(dests)
		odest = tuple(dests)
		spl = text.split()
		if engine != "google" and len(spl) < 2 and (spl[0].isascii() or len(spl[0]) <= 1):
			engine = "google"
		print("TEST:", engine, spl, dests)
		if engine == "google" and not googletrans:
			raise RuntimeError("Unable to load Google Translate.")

		if engine == "google":
			await self.google_translate(bot, guild, channel, user, text, src, dests, translated, comments, engine=engine)
		elif engine in self.LLMs:
			await self.llm_translate(bot, guild, channel, user, text, src, dests, translated, comments, engine=engine)
		else:
			raise NotImplementedError(engine)
		if resp2:
			if awaitable(resp2):
				try:
					resp = await asubmit(resp2)
				except:
					print_exc()
					resp = await self.det(text)
					if not resp:
						resp = cdict(src=src)
			else:
				resp = resp2
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

	async def det(self, s):
		try:
			ftlangdetect = await asubmit(__import__, "ftlangdetect.detect")
		except ImportError:
			return cdict(src="auto", score=0, dest="en")
		resp = await asubmit(ftlangdetect.detect, s, low_memory=psutil.virtual_memory().total < 14 * 1073741824)
		return cdict(src=resp["lang"], score=resp["score"], dest="en")

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

	async def google_translate(self, bot, guild, channel, user, text, src, dests, translated, comments, engine=None):

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
		await gather(*futs)

	async def llm_translate(self, bot, guild, channel, user, text, src, dests, translated, comments, engine="cohere"):
		uid = user.id
		temp = text.replace('"""', "'''")
		start = "### Input:\n"
		instruction = "### Instruction:\n"
		response = "\n\n### Response:"
		if src and src != "auto":
			src = googletrans.LANGUAGES.get(src) or src
			prompt = f'{start}"""\n{temp}\n"""\n\n{instruction}Translate the above from {src} informally into '
		else:
			prompt = f'{start}"""\n{temp}\n"""\n\n{instruction}Translate the above informally into '
		prompt += ",".join((googletrans.LANGUAGES.get(lang) or lang).capitalize() for lang in dests)
		if len(dests) > 1:
			prompt += ', each beginning with "•"'
		prompt += f', keeping formatting as accurate as possible, and do NOT add extra text!{response}'
		kwargs = {}
		if engine in ai.is_reasoning:
			kwargs = dict(reasoning_effort="low")
		try:
			out = await ai.instruct(
				data=dict(
					prompt=prompt,
					model=engine,
					temperature=0.5,
					max_tokens=2048,
					top_p=0.5,
					user=str(user.id),
					premium_context=bot.premium_context(user, guild),
					**kwargs,
				),
				skip=True,
			)
		except:
			print_exc()
			out = ""
		if ai.decensor.search(out):
			out = ""
		if not out:
			print("Instruct translate: Empty response, retrying...")
			out = await ai.instruct(
				data=dict(
					prompt=prompt,
					model="gemini-2.5-flash",
					temperature=0.5,
					max_tokens=2048,
					top_p=0.5,
					user=str(user.id),
					premium_context=bot.premium_context(user, guild),
				),
				skip=True,
			)
			# resp = await ai.llm(
			# 	"completions.create",
			# 	model="gpt-4.1-mini",
			# 	prompt=prompt,
			# 	temperature=0.5,
			# 	max_tokens=2048,
			# 	top_p=0.5,
			# 	frequency_penalty=0,
			# 	presence_penalty=0,
			# 	user=str(user.id),
			# )
			# out = resp.choices[0].text
		out = out.strip()
		if out and out[0] == out[-1] == '"' and not text[0] == text[-1] == '"':
			try:
				out = str(literal_eval(out)).strip()
			except SyntaxError:
				pass
		lines = [line2 for line in out.split("•") if (line2 := line.strip())]
		print(f"{engine} Translate:", user, text, src, dests, lines)

		async def translate_into(arg, src, dest, i):
			translated[i] = arg
			try:
				resp = await asubmit(self.trans.translate, arg, src=src, dest=dest)
			except Exception:
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
		await gather(*futs)


class Translator(Command):
	name = ["AutoTranslate"]
	min_level = 2
	description = 'Adds an automated translator to the current channel. Specify a list of languages to translate between, and optionally a translation engine. All non-command messages that do not begin with "#" will be passed through the translator.'
	usage = "<0:engine(google|command-r-plus|gpt-3.5-turbo-instruct)>? <1:languages>* <disable(-d)>?"
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
			curr = cdict(engine="Auto", languages=[])
			if args[0].casefold() in self.bot.commands.translate[0].LLMs:
				curr.engine = args[0]
				args.pop(0)
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
			aliases=["expression"]
		),
		precision=cdict(
			type="integer",
			validation=f"(0, {limit}]",
			description="Floating point calculation precision. Also affects notation of scientific notation and recurring decimals",
			example="1000",
			default=256,
		),
	)
	usage = "<string> <verbose(-v)|rationalize(-r)>? <show_variables(-l)|clear_variables(-c)>?"
	example = ("m factorial 32", "plot 3x^2-2x+1", "math integral tan(x)", "m solve(x^3-1)", "calc std([6.26,6.23,6.34,6.28])", "🔢 predict_next([2, 10, 30, 68, 130])")
	flags = "rvlcd"
	rate_limit = (4.5, 6)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _user, _premium, mode, query, precision, **void):
		if query == "69":
			return py_md("69 = nice")
		if mode == "list_vars":
			var = bot.data.variables.get(_user.id, {})
			if not var:
				return ini_md(f"No currently assigned variables for {sqr_md(_user)}.")
			return ini_md(f"Currently assigned variables for {_user}:{iter2str(var)}")
		if mode == "clear_vars":
			bot.data.variables.pop(_user.id, None)
			return italics(css_md(f"Successfully cleared all variables for {sqr_md(_user)}."))
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
		resp = await bot.solve_math(query, precision, False, timeout=timeout, variables=bot.data.variables.get(_user.id))
		# Determine whether output is a direct answer or a file
		if type(resp) is dict and "file" in resp:
			fn = resp["file"]
			f = CompatFile(fn, filename=query + ".png")
			return cdict(file=f)
		assert resp, "Response empty."
		answer = "\n".join(a for a in map(str, resp) if a != query.strip()) or str(resp[0])
		if var is not None:
			env = bot.data.variables.setdefault(_user.id, {})
			env[var] = resp[0]
			while len(env) > 64:
				env.pop(next(iter(env)))
			s = lim_str(f"Variable {sqr_md(var)} set to {sqr_md(resp[0])}.", self.limit, mode="left")
			return cdict(content=s, prefix="```css\n", suffix="```")
		s = lim_str(f"{query} = {answer}", self.limit, mode="left")
		return cdict(content=s, prefix="```py\n", suffix="```")


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
	name = ["I2T", "CreateTime", "Timestamp", "Time2ID", "T2I"]
	description = "Converts a discord ID to its corresponding UTC time."
	schema = cdict(
		id=cdict(
			type="number",
			example="201548633244565504",
		),
		time=cdict(
			type="datetime",
			example="35 minutes and 6.25 seconds before 3am next tuesday, EDT",
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
	name = ["UnFormat", "UnZalgo"]
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
	usage = "<0:string> <1:emoji_1> <2:emoji_2>?"
	example = ("c2e POOP 💩 🪰",)
	rate_limit = (10, 14)
	slash = True

	def __call__(self, args, guild, message, **void):
		if len(args) < 2:
			raise ArgumentError(
				"At least 2 arguments are required for this command.\n"
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
		if len(args) < 3:
			args.append("⬛")
		resp = _c2e(*args[:3])
		if hasattr(message, "simulated") or len(args[0]) <= 25:
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
		csubmit(bot.silent_delete(_message))
		await bot.send_as_webhook(_channel, msg, username=_user.display_name, avatar_url=url)


class EmojiCrypt(Command):
	name = ["EncryptEmoji", "DecryptEmoji", "EmojiEncrypt", "EmojiDecrypt"]
	description = "Encrypts the input text or file into smileys."
	usage = "<string> <mode(encrypt|decrypt)> <encrypted(-p)>? <-1:password>"
	rate_limit = (9, 12)
	# slash = True
	ephemeral = True
	flags = "ed"

	async def __call__(self, args, name, flags, message, **void):
		password = None
		for flag in ("+p", "-p", "?p"):
			try:
				i = args.index(flag)
			except ValueError:
				continue
			password = args[i + 1]
			args = args[:i] + args[i + 2:]
		msg = " ".join(args)
		fi = temporary_file()
		if not msg:
			msg = message.attachments[0].url
		if is_url(msg):
			msg = await self.bot.follow_url(msg, allow=True, limit=1)
			args = ("streamshatter", msg, "../" + fi)
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
	name = ["🕰️", "⏰", "⏲️", "UTC", "GMT", "T"]
	description = "Shows the current time at a certain GMT/UTC offset, or the current time for a user."
	schema = cdict(
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

	async def __call__(self, _user, input, user, estimate, timezone, **void):
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


class Identify(Command):
	name = ["📂", "Magic", "Mime", "MimeType", "FileType", "FileInfo", "IdentifyFiles"]
	description = "Detects the type, mime, and optionally details of an input file."
	usage = "<url>*"
	example = ("identify https://raw.githubusercontent.com/thomas-xin/Image-Test/master/title-rainbow.webp",)
	rate_limit = (12, 16)
	mime = magic.Magic(mime=True, mime_encoding=True)
	slash = True
	ephemeral = True
	msgcmd = ("Identify Files",)

	def probe(self, url):
		command = ["ffprobe", "-hide_banner", url]
		resp = None
		for _ in loop(2):
			try:
				proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				fut = esubmit(proc.communicate, timeout=12)
				res = fut.result(timeout=12)
				resp = b"\n".join(res)
				break
			except Exception:
				with suppress():
					force_kill(proc)
				print_exc()
		if not resp:
			raise RuntimeError(proc)
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
	ephemeral = True

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
			temp = await asubmit(re.findall, regex, " ".join(args))
			match = "\n".join(sqr_md(i) for i in temp)
		else:
			search = args.pop(0)
			s = " ".join(args)
			match = (
				sqr_md(round_min(round(string_similarity(search, s) * 100, 6))) + "% literal match,\n"
				+ sqr_md(round_min(round(string_similarity(search.casefold(), s.casefold()) * 100, 6))) + "% case-insensitive match,\n"
				+ sqr_md(round_min(round(string_similarity(full_prune(search), full_prune(s)) * 100, 6))) + "% unicode mapping match."
			)
		return ini_md(match)


class Random(Command):
	name = ["choice", "choose"]
	description = "Randomizes a set of arguments."
	usage = "<string>+"
	example = ("random 1 2 3", 'choose "this one" "that one"')
	slash = True
	ephemeral = True

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
	ephemeral = True

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
	usage = "<relationship(-r)>? <pickup-line(-p)>? <nsfw-pickup-line(-n)>?"
	example = ("topic", "question -r")
	flags = "npr"

	def __call__(self, bot, user, flags, channel, **void):
		csubmit(bot.seen(user, event="misc", raw="Talking to me"))
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
		csubmit(bot.seen(user, event="misc", raw="Talking to me"))
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
	usage = "<string>"
	example = ("urban ur mom",)
	flags = "v"
	rate_limit = (5, 8)
	typing = True
	slash = True
	ephemeral = True
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


class Browse(Command):
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
	directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
	dirnames = ["First", "Prev", "Next", "Last", "Refresh"]
	rate_limit = (10, 16)
	typing = True
	slash = True
	ephemeral = True
	no_parse = True

	async def __call__(self, _user, mode, query, **void):
		m = 0 if mode == "auto" else 1
		data = bytes2b64(query.encode("utf-8"), alt_char_set=True).decode("ascii")
		out = f'*```callback-string-browse-{_user.id}_0_{m}_{data}-\nBrowsing "{query}"...```*'
		buttons = [cdict(emoji=dirn, name=name, custom_id=dirn) for dirn, name in zip(map(as_str, self.directions), self.dirnames)]
		return cdict(content=out, buttons=buttons)

	async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
		u_id, pos, m, data = vals.split("_", 3)
		print("Browse CB:", vals)
		if reaction and u_id != user.id and perm < 1:
			return
		if reaction not in self.directions and reaction is not None:
			return
		user = await bot.fetch_user(u_id)
		pos = int(pos)
		argv = b642bytes(data.encode("ascii"), alt_char_set=True).decode("utf-8")
		ss = True if int(m) == 0 else False
		urls = await bot.follow_url(argv, ytd=False)
		argv = urls[0] if urls else argv
		s = await bot.browse(argv, uid=user.id, screenshot=ss, best=True, include_hrefs=True)
		ref = getattr(getattr(message, "reference", None), "cached_message", None)
		if isinstance(s, bytes):
			csubmit(bot.silent_delete(message))
			return await bot.respond_with(cdict(file=CompatFile(s)), message=ref)
		elif is_url(argv):
			csubmit(bot.silent_delete(message))
			return await bot.respond_with(cdict(content=s, prefix="\xad"), message=ref)
		rems = s.split("\n\n")
		page = 8
		last = max(0, len(rems) - page)
		if reaction is not None:
			i = self.directions.index(reaction)
			if i == 0:
				new = 0
			elif i == 1:
				new = max(0, pos - page)
			elif i == 2:
				new = min(last, pos + page)
			elif i == 3:
				new = last
			else:
				new = pos
			pos = new
		content = message.content
		if not content:
			content = message.embeds[0].description
		i = content.index("callback")
		content = "*```" + "\n" * ("\n" in content[:i]) + (
			"callback-string-search-"
			+ str(u_id) + "_" + str(pos) + "_" + str(data)
			+ f'-\nSearch results for "{argv}":```*'
		)
		if not rems:
			msg = ""
		else:
			t = utc()
			msg = iter2str(
				rems[pos:pos + page],
				left="`【",
				right="】`",
				offset=pos,
			)
		colour = await self.bot.get_colour(user)
		emb = discord.Embed(
			description=content + msg,
			colour=colour,
		).set_author(**get_author(user))
		more = len(rems) - pos - page
		if more > 0:
			emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
		csubmit(bot.edit_message(message, content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
		if hasattr(message, "int_token"):
			await bot.ignore_interaction(message)