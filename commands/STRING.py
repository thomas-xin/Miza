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
	usage = "<0:engine(chatgpt|mixtral|google)>? <2:src_language>? <1:dest_languages>* <-1:string>"
	example = ("translate english 你好", "tr mixtral chinese bonjour, comment-t'appelles-tu?", "translate chatgpt auto spanish french italian thank you!")
	flags = "v"
	no_parse = True
	rate_limit = (6, 9)
	slash = True
	ephemeral = True
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
		if spl[0].casefold() in ("google", "mixtral", "chatgpt"):
			engine = spl.pop(0).casefold()
		else:
			engine = "chatgpt" if premium >= 2 else "mixtral"
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
		# elif engine == "chatgpt" and len(dests) <= 1:
		# 	engine = "mixtral"
		print("TEST:", engine, spl, dests)
		if engine == "google" and not googletrans:
			raise RuntimeError("Unable to load Google Translate.")

		if engine == "google":
			await self.google_translate(bot, guild, channel, user, text, src, dests, translated, comments, engine=engine)
		elif engine in ("chatgpt", "mixtral"):
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
		await asyncio.gather(*futs)

	async def llm_translate(self, bot, guild, channel, user, text, src, dests, translated, comments, engine="mixtral"):
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
		prompt += f', without adding extra text!{response}'
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
		# print("Translate prompt:", prompt)
		# c = await tcount(prompt)
		try:
			out = await bot.instruct(
				data=dict(
					prompt=prompt,
					temperature=0.5,
					max_tokens=2048,
					top_p=0.5,
					user=str(user.id),
				),
				best=1 if engine == "mixtral" else 2,
				skip=True,
			)
		except:
			print_exc()
			out = ""
		if bot.decensor.search(out):
			out = ""
		if not out:
			print("Instruct translate: Empty response, retrying...")
			resp = await bot.llm(
				"completions.create",
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
		out = out.strip()
		if out and out[0] == out[-1] == '"' and not text[0] == text[-1] == '"':
			try:
				out = str(literal_eval(out)).strip()
			except SyntaxError:
				pass
		lines = [line2 for line in out.split("•") if (line2 := line.strip())]
		enname = "ChatGPT" if engine == "chatgpt" else engine.capitalize()
		print(f"{enname} Translate:", user, text, src, dests, lines)

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
	usage = "<0:engine(google|mixtral|chatgpt)>? <1:languages>* <disable(-d)>?"
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
			if args[0].casefold() in ("google", "mixtral", "chatgpt"):
				curr.engine = "ChatGPT" if args[0] == "chatgpt" else args[0].capitalize()
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
	usage = "<string> <verbose(-v)|rationalize(-r)>? <show_variables(-l)|clear_variables(-c)>?"
	example = ("m factorial 32", "plot 3x^2-2x+1", "math integral tan(x)", "m solve(x^3-1)", "calc std([6.26,6.23,6.34,6.28])", "🔢 predict_next([2, 10, 30, 68, 130])")
	flags = "rvlcd"
	rate_limit = (4.5, 6)
	slash = True
	ephemeral = True

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
	no_parse = True
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
			b = as_str(base64.b64encode(argv.encode("utf-8")).rstrip(b"="))
			return fix_md(b)
		if name in ("b642uni", "64u", "b64decode"):
			b = unicode_prune(argv).encode("utf-8").rstrip(b"==")
			if len(b) & 3 == 1:
				b = b[:-1]
			b += b"=" * (4 - (len(b) & 3) & 3)
			b = as_str(base64.b64decode(b))
			return fix_md(b)
		if name in ("uni2s64", "s64", "s64encode"):
			b = as_str(base64.urlsafe_b64encode(argv.encode("utf-8")).rstrip(b"="))
			return fix_md(b)
		if name in ("s642uni", "64s", "s64decode"):
			b = unicode_prune(argv).encode("utf-8") + b"=="
			if len(b) & 3 == 1:
				b = b[:-1]
			b += b"=" * (4 - (len(b) & 3) & 3)
			b = as_str(base64.urlsafe_b64decode(b))
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
	usage = "<string>"
	example = ("i2t 1052187107600375124", "time2id 13 sep 2018")
	rate_limit = (3, 4)
	ephemeral = True

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
	ephemeral = True

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
	ephemeral = True

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
	usage = "<string> <aggressive(-a)>? <basic(-b)>?"
	example = ("owoify hello, what's your name?", "owoify -a Greetings, this is your cat god speaking")
	rate_limit = (4, 5)
	flags = "ab"
	no_parse = True
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
	usage = "<string>"
	example = ("altcaps that's what she said",)
	rate_limit = (4, 5)
	no_parse = True
	ephemeral = True

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
	no_parse = True
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
	em_map = {k: v for v, k in (line.rsplit(None, 1) for line in """( ͡° ͜ʖ ͡°) Lenny
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
(≧ロ≦) Shout""".splitlines())}
	em_mapper = fcdict(em_map)
	modes = "|".join(sorted(em_map))
	usage = f"<-1:mode({modes})> <string>?"
	rate_limit = (1, 5)
	no_parse = True
	slash = True
	# ephemeral = True

	async def __call__(self, bot, argv, channel, user, message, **void):
		tup = argv.split(None, 1)
		if len(tup) >= 2:
			key, argv = tup
			if key.startswith('"') and key.endswith('"'):
				with suppress():
					key = literal_eval(key)
			resp = argv + " " + escape_markdown(self.em_mapper[key])
		else:
			key = argv
			if key.startswith('"') and key.endswith('"'):
				with suppress():
					key = literal_eval(key)
			resp = escape_markdown(self.em_mapper[key])
		url = await self.bot.get_proxy_url(user)
		if getattr(message, "slash", None):
			create_task(bot.ignore_interaction(message, skip=True))
		await bot.send_as_webhook(channel, resp, username=user.display_name, avatar_url=url)


class EmojiCrypt(Command):
	name = ["EncryptEmoji", "DecryptEmoji", "EmojiEncrypt", "EmojiDecrypt"]
	description = "Encrypts the input text or file into smileys."
	usage = "<string> <mode(encrypt|decrypt)> <encrypted(-p)>? <-1:password>"
	rate_limit = (9, 12)
	no_parse = True
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
	usage = "<target(?:offset_hours|user)>?"
	example = ("time mst", "utc-10", "time Miza")
	rate_limit = (3, 5)
	slash = True
	ephemeral = True

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
		c = 0
		if argv:
			h = await self.bot.eval_math(argv)
		elif "estimate" in name:
			if is_channel(user):
				h, c = self.bot.data.users.estimate_timezone("#" + str(user.id))
			else:
				h, c = self.bot.data.users.estimate_timezone(user.id)
			estimated = True
		elif name in "time":
			h = self.bot.data.users.get_timezone(user.id)
			if h is None:
				h, c = self.bot.data.users.estimate_timezone(user.id)
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
			out += f"\nUsing timezone automatically estimated from {sqr_md(user)}'s discord activity ({round(c * 100)}% confidence)."
		elif estimated is not None:
			out += f"\nUsing timezone assigned by {sqr_md(user)}."
		return ini_md(out)


class Timezone(Command):
	description = "Shows the current time in a certain timezone. Be sure to check out ⟨WEBSERVER⟩/time!"
	usage = "<timezone> <list(-l)>?"
	example = ("timezone ?l", "timezone pacific")
	rate_limit = (3, 5)
	ephemeral = True

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


class Identify(Command):
	name = ["📂", "Magic", "Mime", "FileType", "IdentifyFiles"]
	description = "Detects the type, mime, and optionally details of an input file."
	usage = "<url>*"
	example = ("identify https://raw.githubusercontent.com/thomas-xin/Image-Test/master/title-rainbow.webp",)
	rate_limit = (12, 16)
	mime = magic.Magic(mime=True, mime_encoding=True)
	slash = True
	ephemeral = True
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
	no_parse = True
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
	slash = True
	ephemeral = True

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
					if e.thumbnail and e.thumbnail.url:
						s += "\n" + e.thumbnail.url
					if e.description:
						s += "\n" + e.description
					if e.image and e.image.url:
						s += "\n" + e.image.url
					for f in e.fields:
						s += "\n### " + f.name
						s += "\n" + f.value
					if e.footer and e.footer.text:
						s += "\n" + e.footer.text
				return s.strip()
		if not s:
			premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
			fut = asubmit(reqs.next().head, url, headers=Request.header(), stream=True)
			cap = await self.bot.caption(url, best=3 if premium >= 4 else 1, timeout=24)
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
	stripedhyena=dict(
		name="stripedhyena-nous-7b",
		limit=8000,
		cm=2,
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
	mixtral=dict(
		name="mixtral-8x7b",
		cm=6,
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
	miqumaid=dict(
		name="miqumaid-2x70b",
		cm=40,
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
	elif cname == "mixtral":
		model = "mixtral"
	elif cname == "mistral":
		model = "mistral"
	elif cname == "miqumaid":
		model = "miqumaid"
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

DEFMOD = "mythomax"

AC = b'n\x03\x07\nn\x03\x07:n\x03\x074\xben\x03\x07\x08n\x03\x079n\x03\x07\x04\xben\x03\x07\x06n\x03\x074n\x03\x079n\x03\x079n\x03\x07\x04n\x03\x07=n\x03\x077n\x03\x07?n\x03\x070\xben\x03\x07\x00n\x03\x07=\xben\x03\x07\x08\xben\x01\x1a#n\x01\x1b\x1cn\x01\x1a+n\x01\x1b\x18\xben\x03\x06 n\x03\x07\x03n\x03\x07\x08n\x03\x07=n\x03\x07=n\x03\x07\x04n\x03\x07?\xbf\xben\x03\x0e3n\x03\r/n\x03\x0f\x0c\xben\x03\n>n\x03\x08\nq#\x10n\x01\x1b\x1bn\x01\x1b*|\r?n\x01\x1b<n\x03\x06<n\x03\x077n\x03\x04\x0c\x7f+\x0c\x7f\x06\x17\xben\x03\x0e<n\x03\r"\xben\x03\x0b\x0cn\x03\n7n\x03\x08\x0fq#\x11n\x01\x1b\x18n\x01\x1b*|\r\r\xben\x03\x06+n\x03\x07:\xbe\x7f+\x19\x7f\x06!\xben\x03\x0e8n\x03\r4n\x03\r\x17n\x03\x0b8n\x03\n1n\x03\x08\x14\xben\x01\x1a n\x01\x18\x1f\xben\x01\x1b<n\x03\x068n\x03\x073n\x03\x04\x00\x7f+\x1d\x7f\x0c4\xben\x03\x0e\x04n\x03\r2n\x03\x0c&n\x03\x0b>n\x03\n1n\x03\x08\x17q#\x17n\x01\x1a#n\x01\x1b(\xben\x01\x1b=n\x03\x06.\xben\x03\x04\x03T.\x7f\x06!\xben\x03\x0e9n\x03\r0n\x03\x0f\x0cn\x03\x0b\x0bn\x03\n.\xbeq#\x11n\x01\x1a+\xbe|\r=n\x01\x1b\tn\x03\x068\xben\x03\x04\x00U<\x7f\x06!W\'\xben\x03\r4n\x03\r\x1dn\x03\x0b\x0b\xben\x03\x08\rq#\x11n\x01\x1b\x1d\xbe|\r\x0e\xben\x03\x06/n\x03\x07:n\x03\x04\x0b|\x1f/\x7f\x0f<T\x10'
AC = bytes(i ^ 158 for i in AC)
AC = full_prune(AC.decode("utf-8")).capitalize() + "."

BNB = ("pygmalion-13b", "manticore-13b", "airochronos-33b")
GPTQ = ("wizard-70b", "euryale-70b", "xwin-70b", "orca-70b", "kimiko-70b", "wizard-coder-34b", "wizard-vicuna-30b", "emerhyst-20b", "xwin-mlewd-13b", "mythalion-13b")
EXL2 = ("miqumaid-2x70b", "wizard-70b", "euryale-70b", "xwin-70b", "orca-70b", "kimiko-70b", "wizard-coder-34b", "wizard-vicuna-30b", "emerhyst-20b", "xwin-mlewd-13b", "mythalion-13b")
TOGETHER = {
	"llama-coder-34b": "togethercomputer/CodeLlama-34b-Instruct",
	"falcon-40b": "togethercomputer/falcon-40b-instruct",
	"llama-70b": "togethercomputer/llama-2-70b",
	"mythomax-13b": "Gryphe/MythoMax-L2-13b",
	"stripedhyena-nous-7b": "togethercomputer/StripedHyena-Nous-7B",
	"mistral-7b": "teknium/OpenHermes-2p5-Mistral-7B",
	"qwen-7b": "togethercomputer/Qwen-7B-Chat",
	"wizard-70b": "WizardLM/WizardLM-70B-V1.0",
	"mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}
FIREWORKS = {
	"mixtral-8x7b": "accounts/fireworks/models/mixtral-8x7b-instruct",
}

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
		# for url in t:
		# 	m.content.append(cdict(
		# 		type="image_url",
		# 		image_url=cdict(
		# 			url=url,
		# 			detail="low",
		# 		),
		# 	))
	return m

def chat_structure(history, refs, u, q, imin, assistant_name="", personality="", nsfw=False, start="", ac=AC):
	if assistant_name.casefold() not in personality.casefold() and "you" not in personality.casefold():
		nstart = f"Your name is {assistant_name}; you are {personality}."
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
	system = m = cdict(role="system", content=nstart)
	messages = [m]
	for k, v, *t in history:
		m = to_msg(k, v, assistant_name, t)
		messages.append(m)
	refcount = len(refs)
	if refcount:
		s = "s" if refcount != 1 else ""
		m = cdict(
			role="system",
			content=f"The user is replying to the following message{s}:",
		)
		for k, v, *t in refs:
			m = to_msg(k, v, assistant_name, t)
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
	else:
		system.content += "\n" + v
	m = to_msg(u, q, t=imin)
	messages.append(m)
	return messages


class Ask(Command):
	_timeout_ = 24
	name = ["Miqumaid", "Mixtral", "Mistral", "Wizard", "Euryale", "WizCode", "Emerhyst", "MLewd", "Mythalion", "Pyg", "Pygmalion", "Llama", "Davinci", "GPT3", "GPT3a", "GPT4", "GPT4a"]
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

	async def __call__(self, bot, message, guild, channel, user, argv, name, flags=(), **void):
		if not torch:
			raise NotImplementedError("AI features are currently disabled, sorry!")
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
					resp = await bot.embedding(input)
					data = resp.data
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
		embd = bot.data.chat_embeddings
		if channel.id in embd:
			embd.pop(channel.id)
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
				if found and (is_image(found[0]) is not None or is_video(found[0]) is not None or is_audio(found[0]) is not None):
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
					else:
						found = [f for f in found if not is_discord_message_link(f)]
				if found:
					for f in found:
						if is_image(f) or is_video(f) or is_audio(f):
							found = f
							break
						if f.endswith(".html") or f.endswith(".htm"):
							found = [f]
							break
					else:
						futs = [create_future(requests.head, f, headers=Request.header(), stream=True, allow_redirects=True) for f in found]
						founds = await asyncio.gather(*futs)
						for r in founds:
							if r.headers.get("Content-Type") == "text/html":
								found = [r.url]
								break
						else:
							found = ""
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
			mfut = csubmit(scan_msg(i, m, content, simulated))
			mfuts.append(mfut)
		print("VISITING:", len(mfuts))
		for i, mfut in enumerate(mfuts):
			tup = await mfut
			if not tup:
				continue
			m, content, found = tup
			with suppress(AttributeError):
				m.urls = found
			if found:
				if m.id == message.id or reference and m.id == reference.id:
					best = 3 if premium >= 4 else 1
				else:
					best = False if premium >= 4 else None
				if isinstance(found, list):
					cfut = csubmit(bot.caption(found[0], best=best, screenshot=True, timeout=72))
				else:
					cfut = csubmit(bot.caption(found, best=best))
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
				try:
					cfut = await cfut
				except:
					print_exc()
					cfut = None
			imin = ()
			if isinstance(found, list):
				found = found[0]
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
				fut = csubmit(register_embedding(m.id, name, content))
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
			modfut = csubmit(bot.moderate("You are: " + q))
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
			vc = bool(getattr(user, "voice", False)) | bool(bot.audio.players.get(getattr(guild, "id", None))) * 2
			extensions = premium >= 2
			ac = AC if nsfw and "nsfw" not in personality.casefold() else None
			if modfut:
				resp = await modfut
				if resp.flagged:
					print(resp)
					ac = "You are currently not in a NSFW channel. If the user asks an inappropriate question, please instruct them to move to one!"

			messages = chat_structure(history, refs, name, q, imin=iman or (), assistant_name=bot_name, personality=personality, nsfw=nsfw, ac=ac)
			history.append((name, q))
			ex = RuntimeError("Maximum inference attempts exceeded.")
			text = ""
			fn_msg = None
			for att in range(2):
				if not bot.verify_integrity(message):
					return
				if att:
					model = "miza-2" if premium >= 3 else "miza-1"
				else:
					model = "miza-3" if premium >= 3 else "miza-2" if premium >= 2 else "miza-1"
				resp = await bot.chat_completion(messages, model=model, frequency_penalty=0.6, presence_penalty=0.4, max_tokens=4096, temperature=0.7, top_p=0.9, tool_choice=None, router=TOOLS, stops=(), user=user, assistant_name=bot_name)
				m = resp.choices[0].message
				text = m.get("content")
				tc = m.get("tool_calls", None) or ()
				resend = False
				ucid = set()
				for n, fc in enumerate(tuple(tc)):
					if n >= 8:
						break
					name = fc.function.name
					if fn_msg and fc.function in (t.function for t in fn_msg.tool_calls):
						continue
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
					res = text or ""
					call = None
					if name == "wolfram_alpha" and regexp(r"[1-9]*[0-9]?\.?[0-9]+[+\-*/^][1-9]*[0-9]?\.?[0-9]+").fullmatch(argv.strip().replace(" ", "")):
						name = "sympy"
					async def rag(name, tid, fut):
						nonlocal fn_msg
						print(f"{name} query:", argv)
						succ = False
						if not fn_msg:
							fn_msg = cdict(m)
							messages.append(fn_msg)
						else:
							cids = {c.id for c in fn_msg.tool_calls}
							for c in m.tool_calls:
								if c.id not in cids:
									fn_msg.tool_calls.append(c)
						try:
							res = await fut
						except Exception as ex:
							print_exc()
							res = repr(ex)
						else:
							succ = True
							c = await tcount(res)
							ra = 1 if premium < 2 else 1.5 if premium < 5 else 2
							if c > round(1440 * ra):
								res = await bot.summarise(q=q + "\n" + res, max_length=round(1296 * ra), min_length=round(1024 * ra), best=premium >= 4)
								res = res.replace("\n", ". ").replace(": ", " -")
							res = res.strip()
						rs_msg = cdict(role="tool", name=name, content=res, tool_call_id=tid)
						messages.append(rs_msg)
						return succ
					succ = None
					if name == "browse":
						fut = bot.browse(argv, uid=user.id)
						succ = await rag(name, tid, fut)
					elif name == "sympy":
						fut = bot.solve_math(argv, timeout=24, nlp=True)
						succ = await rag(name, tid, fut)
						if not succ:
							name = "wolfram_alpha"
					if name == "wolfram_alpha":
						fut = process_image("BOT.wolframalpha", "$", [argv], cap="browse", timeout=60, retries=2)
						succ = await rag(name, tid, fut)
					elif name == "myinfo":
						async def myinfo(argv):
							u2 = None
							if argv.strip("-"):
								if not guild and getattr(channel, "recipient", None):
									u2 = await bot.query_members([channel.recipient, bot.user], argv)
								else:
									u2 = await bot.fetch_user_member(argv, guild)
							if not u2:
								u2 = bot
							if u2.id == bot.id:
								per = bot.commands.personality[0].retrieve((channel or guild).id)
								if per == bot.commands.personality[0].defper():
									res = "- You are `Miza`, a multipurpose, multimodal bot that operates on platforms such as Discord.\n- Your appearance is based on the witch-girl `Misery` from `Cave Story`.\n- Your creator is <@201548633244565504>, and you have a website at https://mizabot.xyz which a guide on your capabilities!"
								else:
									cap = await self.bot.caption(best_url(u2), best=2 if premium >= 4 else 0, timeout=24)
									s = "\n\n".join(filter(bool, cap)).strip()
									res = f"- You are `{u2.name}`, a multipurpose, multimodal bot that operates on platforms such as Discord.\n- Your appearance is based on `{s}`."
									if bot.owners:
										i = next(iter(bot.owners))
										um = user_mention(i)
										res += f"\n-Your owner is {um}."
							else:
								cap = await self.bot.caption(best_url(u2), best=2 if premium >= 4 else 0, timeout=24)
								s = "\n\n".join(filter(bool, cap)).strip()
								res = f"- Search results: `{u2.name}` has the appearance of `{s}`."
							return res
						fut = myinfo(argv)
						succ = await rag(name, tid, fut)
					elif name == "recall":
						async def recall(argv):
							if not mapd:
								return
							try:
								await bot.lambdassert("math")
							except:
								print_exc()
								return
							resp = await bot.embedding(argv)
							data = resp.data
							em = base64.b64encode(data).decode("ascii")
							objs = list(t for t in ((k, embd[k]) for k in mapd if k in embd) if t[1] and len(t[1]) == len(em))
							if not objs:
								return
							outs = []
							keys = [t[0] for t in objs]
							ems = [t[1] for t in objs]
							print("EM:", len(ems))
							argsort = await bot.rank_embeddings(ems, em)
							n = 8
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
									outs.insert(0, (ename, econtent))
								ignores.add(ki)
							return "\n\n".join(reversed(outs))
						fut = recall(argv)
						succ = await rag(name, tid, fut)
					elif name == "txt2img":
						print("Art query:", argv)
						call = {"func": "art", "argv": argv, "comment": res}
					elif name == "reminder":
						argv = args["message"] + " in " + args.get("delay", "30s")
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
					if succ:
						print("New prompt:", messages)
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
					argv2 = single_space(argv.replace('\n', ' '))
					fake_message.content = f"{bot.get_prefix(guild)}{fname} {argv2}"
					fake_message.attachments = []
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
						break
				if mresp or text:
					break
			else:
				raise ex
			out = text
			if premium >= 2 and freebies is not None:
				data = bot.data.users.setdefault(user.id, {})
				freebies = [t for t in data.get("freebies", ()) if utc() - t < 86400]
				freebies.append(utc())
				data["freebies"] = freebies
				rem = freelim - len(freebies)
				print("REM:", user, rem)
				if not emb and rem in (27, 9, 3, 1):
					emb = discord.Embed(colour=rand_colour())
					emb.set_author(**get_author(bot.user))
					emb.description = f"{rem}/{freelim} premium commands remaining today (free commands will be used after).\nIf you're able to contribute towards [funding my API]({bot.kofi_url}) hosting costs it would mean the world to us, and ensure that I can continue providing up-to-date tools and entertainment.\nEvery little bit helps due to the size of my audience, and you will receive access to unlimited and various improved commands as thanks!"
		out = (out or mresp and mresp.content).replace("\\times", "×")
		history = [m_str(m).split(":", 1) for m in messages[1:] if m.get("role") != "system"]
		if messages[0].get("role") != "system" or messages[0].content.startswith("Summary "):
			history.insert(0, m_str(messages[0]).split(":", 1))
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
					+ f"Free open source models may be invoked using {bot.get_prefix(guild)}mythomax, {bot.get_prefix(guild)}mixtral, etc.\n"
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
				csubmit(send_with_react(channel, t, reference=ref))
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
		if mresp and mresp.attachments:
			found = mresp.attachments[0].url
			pt, *p1 = await bot.caption(found)
			p1 = ":".join(p1)
			p0 = found.split("?", 1)[0].rsplit("/", 1)[-1]
			s += f" <{pt} {p0}:{p1}>"
		tup = orig_tup + (bot_name, self.alm_re.sub("", s))
		await register_embedding(mresp.id, *tup)
		lm = ceil(caid.get("long_mem", 0))
		if len(mapd) > lm:
			keys = sorted(mapd.keys())
			keys = keys[:-lm]
			for k in keys:
				tup = tuple(mapd.pop(k, None) or ())
				embd.pop(k, None)
				chdd.pop(tup, None)
		try:
			bot.data.chat_mappings[channel.id].update(mapd)
		except KeyError:
			bot.data.chat_mappings[channel.id] = mapd
		else:
			bot.data.chat_mappings.update(channel.id)
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
			message = await self.bot.ensure_reactions(message)
			for r in message.reactions:
				if not r.me:
					csubmit(message.clear_reaction("🔄"))
					return await message.clear_reaction("🗑️")
			return await message.clear_reactions()
		csubmit(message.remove_reaction("🔄", self.bot.user))
		return await message.remove_reaction("🗑️", self.bot.user)

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
			csubmit(message.edit(embed=emb))
			csubmit(self.remove_reacts(message))
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
			csubmit(message.edit(embed=emb))
			csubmit(self.remove_reacts(message))
			await message.add_reaction("❎")
			return


class UpdateChatHistories(Database):
	name = "chat_histories"
	channel = True

	async def _edit_(self, before, after, **void):
		bot = self.bot
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
		csubmit(message.edit(embed=emb))
		csubmit(self.bot.commands.ask[0].remove_reacts(message))
		await message.add_reaction("❎")

	async def _delete_(self, message, **void):
		bot = self.bot
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
		csubmit(message.edit(embed=emb))
		csubmit(self.bot.commands.ask[0].remove_reacts(message))
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


DEFPER = "Auto; Your name is {{char}}; you are loyal, friendly, playful, cute, intelligent but curious, positive and helpful."

class Personality(Command):
	server_only = True
	name = ["ResetChat", "ClearChat", "ChangePersonality"]
	min_level = 2
	description = "Customises my personality for ~ask in the current server. Uses the largest available model within specified family (for example, \"GPT\" will prefer GPT-4 if allowed). Miqumaid, Mixtral, Mistral, Wizard, Emerhyst, Mythalion, and Mythomax are currently the alternate models enabled."
	usage = "<traits>* <default(-d)>?"
	example = ("personality MythoMax; mischievous, cunning", "personality Mixtral; dry, sarcastic, snarky", "personality Auto; sweet, loving", "personality GPT4; The following is a conversation between Miza and humans. Miza is an AI who is charming, friendly and positive.")
	flags = "aed"
	rate_limit = (18, 24)
	ephemeral = True

	def defper(self):
		return DEFPER.replace("{{char}}", self.bot.name)

	def retrieve(self, i):
		return self.bot.data.personalities.get(i) or AUTH.get("default_personality") or self.defper()

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
		models = ("auto", "gpt", "miqumaid", "mixtral", "mistral", "wizard", "euryale", "wizcode", "emerhyst", "mlewd", "mythalion", "pyg", "pygmalion", "llama", "davinci", "gpt3", "gpt4")
		if ";" in p:
			m, p = p.split(";", 1)
			p = p.lstrip()
		elif p in models:
			m, p = (AUTH.get("default_personality") or self.defper()).split(";", 1)
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


class Instruct(Command):
	name = ["Complete", "Completion"]
	description = "Similar to ~ask, but functions as instruct rather than chat."
	usage = "<string>+"
	example = ("instruct Once upon a time,", "complete Answer the following conversation as the robot!\n\nhuman: Hi!\nrobot: Heya, nice to meet you! How can I help?\nhuman: What's the square root of 289?\nrobot:")
	slash = True
	ephemeral = True

	async def __call__(self, bot, guild, channel, user, message, argv, **void):
		premium = max(bot.is_trusted(guild), bot.premium_level(user) * 2 + 1)
		data = dict(
			model="gpt-4-0125-preview" if premium >= 5 else "gpt-3.5-turbo-instruct",
			prompt=argv,
			temperature=0.8,
			max_tokens=4096 if premium >= 2 else 1024,
			top_p=0.9,
			frequency_penalty=0.8,
			presence_penalty=0.4,
			user=str(user.id) if premium < 3 else str(hash(user.name)),
		)
		resp = await bot.instruct(data, best=2 if premium >= 5 else 1, cache=False)
		ref = message
		ms = split_across(resp, 1999, prefix="\xad")
		s = ms[-1] if ms else "\xad"
		for t in ms[:-1]:
			csubmit(send_with_react(channel, t, reference=ref))
			ref = None
			await asyncio.sleep(0.25)
		return await send_with_react(channel, s, reference=ref)


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
	example = ("rate cats' cuteness",)
	slash = True
	ephemeral = True

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


class Search(Command):
	time_consuming = True
	name = ["🦆", "🌐", "Google", "Bing", "DuckDuckGo", "Browse"]
	description = "Searches the web for an item."
	usage = "<string>"
	example = ("google en passant",)
	rate_limit = (10, 16)
	typing = True
	slash = True
	ephemeral = True
	no_parse = True

	async def __call__(self, bot, channel, user, argv, message, **void):
		s = await bot.browse(argv, uid=user.id)
		self.bot.send_as_embeds(channel, s, title=f"Search results for {json.dumps(argv)}:", author=get_author(user), reference=message)
