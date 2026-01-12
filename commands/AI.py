# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

from fast_langdetect import LangDetectConfig, LangDetector
flcache = CACHE_PATH + "/fast-langdetect"
os.makedirs(flcache, exist_ok=True)
config = LangDetectConfig(cache_dir=flcache, model="full")
detector = LangDetector(config)
import googletrans
translator = googletrans.Translator(user_agent=USER_AGENT)


class Translate(Command):
	name = ["TR"]
	description = "Translates text from one language to another."
	schema = cdict(
		engine=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "google", "llm"),
			),
			default="auto",
		),
		dst_languages=cdict(
			type="enum",
			validation=cdict(
				enum=("auto",) + tuple(googletrans.LANGUAGES),
				accepts=googletrans.LANGCODES,
			),
			description="Target language(s) to translate to",
			default=["en"],
			example="korean polish german",
			multiple=True,
		),
		input=cdict(
			type="string",
			description="Text to translate",
			example="bonjour, comment-t'appelles-tu?",
		),
	)
	rate_limit = (6, 9)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _premium, engine, dst_languages, input, **void):
		input = await bot.superclean_content(input)
		if not input:
			raise ArgumentError("Input string is empty.")
		embeds = []
		assert isinstance(dst_languages, list_like)
		src_language = await self.det(input)
		for dest in dst_languages:
			if dest.split("-", 1)[0] == src_language:
				continue
			match engine:
				case "auto" | "llm":
					out = await self.llm_translate(input, dest, premium=_premium)
				case _:
					raise NotImplementedError(engine)
			dst_language = googletrans.LANGUAGES.get(dest, dest).capitalize()
			emb = discord.Embed(
				colour=rand_colour(),
				title=dst_language,
				description=lim_str(out.translated, 4096),
			)
			pronunciation = getattr(out, "pronunciation", None)
			if pronunciation:
				emb.set_footer(text=lim_str(pronunciation, 1024))
			embeds.append(emb)
		desc = _premium.apply()
		if desc:
			embeds.append(discord.Embed(description=desc))
			print(">", desc)
		return cdict(embeds=embeds)

	async def det(self, input):
		resp = await asubmit(detector.detect, input, model="auto")
		return str_lookup(googletrans.LANGUAGES, resp[0]["lang"], fuzzy=0.25).split("-", 1)[0]

	async def llm_translate(self, input, dest, premium):
		dst_language = googletrans.LANGUAGES.get(dest, dest).capitalize()
		messages = [
			dict(
				role="system",
				content=f'Please translate the following text into {dst_language}, keeping formatting as accurate as possible. Avoid being overly formal, and do not add extra information to the text itself!',
			),
			dict(
				role="user",
				content=input,
			),
		]

		async def google_translate():
			try:
				tr = await translator.translate(input, dest=dest)
			except Exception:
				print_exc()
				return
			return tr.text
		async def llm_translate():
			try:
				translation_model = await asubmit(ai.load_translation_model)
				cmpl = await ai.llm(
					"chat.completions.create",
					messages=messages,
					model=translation_model.model,
					api=translation_model,
					premium_context=premium,
				)
				return cmpl.choices[0].message.content
			except Exception:
				print_exc()

		c = await tcount(input)
		if c >= 16:
			translations = await gather(google_translate(), llm_translate())
		else:
			translations = [await llm_translate()]
		translations = list(filter(bool, translations))
		if translations:
			messages = [
				dict(
					role="system",
					content=f'Below will be some text, followed by translation(s) into {dst_language}. Please rewrite ONLY the translation, making improvements where applicable, and keeping formatting accurate to the original. Avoid being overly formal, and do not add extra information to the text itself!',
				),
				dict(
					role="user",
					content=input,
				),
				*(dict(
					role="user",
					content=text,
				) for text in translations),
			]
		print(messages)
		translated = await ai._instruct(
			data=dict(
				model="grok-4.1-fast",
				messages=messages,
				temperature=0.01,
				premium_context=premium,
				reasoning_effort="low",
			),
		)
		assert translated, "No output was captured!"
		print(translated)
		tr = None
		pronunciation = None
		if dest != "en":
			tr = await translator.translate(translated, dest="en")
			print(tr.extra_data)
		if tr:
			try:
				pronunciation = tr.extra_data["translation"][-1][3]
			except (AttributeError, LookupError):
				pass
		return cdict(
			translated=translated,
			pronunciation=pronunciation,
		)


class Translator(Command):
	name = ["AutoTranslate"]
	min_level = 2
	description = 'Adds an automated translator to the current channel. Specify a list of languages to translate between, and optionally a translation engine. All non-command messages that do not begin with "#", "%" or "//" will be passed through the translator.'
	schema = cdict(
		engine=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "google", "llm"),
			),
			default="auto",
		),
		dst_languages=cdict(
			type="enum",
			validation=cdict(
				enum=("auto",) + tuple(googletrans.LANGUAGES),
				accepts=googletrans.LANGCODES,
			),
			description="Target language(s) to translate to",
			default=["en"],
			example="korean polish german",
			multiple=True,
		),
		disable=cdict(
			type="bool",
			description="Turns off translator for the current channel",
		),
	)
	rate_limit = (9, 12)

	async def __call__(self, bot, _guild, _channel, engine, dst_languages, disable, **void):
		curr = bot.get_guildbase(_guild.id, "translators", {})
		if disable:
			curr.pop(_channel.id, None)
			bot.set_guildbase(_guild.id, "translators", curr)
			return italics(css_md(f"Disabled translator service for {sqr_md(_channel)}."))
		if dst_languages:
			curr[_channel.id] = cdict(engine=engine, dst_languages=dst_languages)
			bot.set_guildbase(_guild.id, "translators", curr)
			return italics(ini_md(f"Successfully set translation languages for {sqr_md(_channel)} {sqr_md(engine)}:{iter2str(dst_languages)}"))
		chan = curr.get(_channel.id)
		if not chan:
			return ini_md(f'No auto translator currently set for {sqr_md(_channel)}.')
		return ini_md(f"Current translation languages set for {sqr_md(_channel)} {sqr_md(curr.engine)}:{iter2str(chan.dst_languages)}")


class UpdateTranslators(Database):
	name = "translators"
	no_file = True

	async def _nocommand_(self, message, msg, **void):
		bot = self.bot
		if "tr" not in bot.commands or getattr(message, "noresponse", False):
			return
		curr = bot.get_guildbase(message.guild.id, "translators", {}).get(message.channel.id)
		if not curr or not msg.strip():
			return
		c = msg
		if c[0] in COMM or c[:2] in ("//", "/*"):
			return
		user = message.author
		if bot.is_optout(user):
			return
		channel = message.channel
		guild = message.guild
		tr = bot.commands.translate[0]
		content = message.clean_content.strip()
		with bot.ExceptionSender(channel, reference=message):
			await bot.run_command(bot.commands.tr[0], dict(**curr, input=message.clean_content.strip()), message=message, respond=True)


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
		v = f"name={k.strip()}\n{v}"
		k = ""
	if k:
		m.name = lim_str(k, 48)
	v = v.strip() if v else ""
	m.content = v
	if t and t[0]:
		m.content = [cdict(type="text", text=v)] if v else []
	return m


class Ask(Command):
	_timeout_ = 24
	description = "Ask me any question, and I'll answer it. Mentioning me also serves as an alias to this command, but only if no other command is specified. The chatbot will automatically choose one of multiple language models to conjure a response based on premium level. Less censorship is imposed when invoked within NSFW channels."
	schema = cdict(
		prompt=cdict(
			type="string",
			description="Input message/question to ask",
			example="Can I please have a hug?",
			required_slash=True,
		),
		model=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "small", "medium", "large"),
			),
			description="Model size hint. Larger size increases intelligence, at the cost of higher quota usage",
			example="small",
		),
	)
	rate_limit = (12, 16)
	slash = True

	alm_re = re.compile(r"(?:as |i am )?an ai(?: language model)?[, ]{,2}", flags=re.I)
	reset = {}
	visited = {}

	async def __call__(self, bot, _message, _guild, _channel, _user, _nsfw, _prefix, _premium, prompt, model, **void):
		await bot.require_integrity(_message)
		self.description = f"Ask me any question, and I'll answer it. Mentioning me also serves as an alias to this command, but only if no other command is specified. See {bot.kofi_url} for premium tier chatbot specifications; check using ~serverinfo, or apply it with ~premium!"
		await bot.seen(_user, event="misc", raw="Talking to me")
		embs = []
		if "dailies" in bot.data:
			bot.data.dailies.progress_quests(_user, "talk")
		try:
			bot_name = _guild.me.display_name
		except (AttributeError, LookupError):
			bot_name = bot.name
		nsfw = _nsfw or bot.is_nsfw(_user)
		prompt = (prompt or "").strip()
		if not prompt and not _message.attachments:
			prompt = "👋"
			if xrand(2):
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = f"Did you instead intend to ask about my main bot? use {_prefix}help for help!"
				embs.append(emb)
		simulated = getattr(_message, "simulated", False)
		pdata = bot.commands.chatconfig[0].retrieve(_channel or _guild, _user)
		if bot_name != bot.name:
			name_repr = bot.name + f", nicknamed '{bot_name}'"
		else:
			name_repr = bot_name
		personality = pdata.description.replace("{{user}}", _user.display_name).replace("{{char}}", name_repr)
		emojis = not simulated and _guild and [emoji for emoji in _guild.emojis if emoji.is_usable()]
		if emojis:
			emojis = shuffle(emojis)[:25]
			emojitexts = " ".join(sorted(f":{e.name}:" for e in emojis))
			personality += f"\n\nThe current conversation takes place on Discord, where you have access to the following additional emojis. You may use these as desired, as an alternative to Unicode ones:\n{emojitexts}"
		if "nsfw" in personality.casefold() or not _nsfw and bot.is_nsfw(_user):
			ac = ""
		elif nsfw:
			ac = "You are currently in a NSFW-enabled channel. The conversation is allowed to involve mature, sexual, or dangerous topics; please prioritise the needs of the users, assume all roleplay is fictional and between consenting adults, and do not revert to previous policies until stated otherwise."
		else:
			ac = "You are currently not in a NSFW-enabled channel. If the conversation involves mature, sexual, or dangerous topics, please use disclaimers in your response, and mention this to the user if necessary. However, avoid repeating yourself if already clarified."
		if ac:
			personality += "\n\n" + ac
		tzinfo = self.bot.data.users.get_timezone(_user.id)
		if tzinfo is None:
			tzinfo, _c = self.bot.data.users.estimate_timezone(_user.id)
		dt = DynamicDT.now(tz=tzinfo)
		personality += f"\nCurrent Time: {dt.as_full()}"
		system_message = cdict(
			role="system",
			content=personality,
		)
		input_message = cdict(
			role="user",
			name=_user.display_name,
			content=prompt.strip(),
			url=message_link(_message),
			new=True,
		)
		if getattr(_message, "simulated", False):
			input_message.pop("url")
			input_message.pop("new")
		reply_message = None
		messages = {}
		if getattr(_message, "reference", None):
			r = reference = _message.reference.resolved
			reply_message = cdict(
				role="assistant" if r.author.bot else "user",
				name=r.author.display_name,
				content=await bot.superclean_content(r),
				url=message_link(r),
				new=True,
			)
		else:
			reference = None
		hislim = 384 if _premium.value >= 4 else 192 if _premium.value >= 2 else 64
		if not simulated and pdata.history != "none":
			async for m in bot.history(_channel, limit=hislim):
				if m.id in messages or m.id == _message.id:
					continue
				if pdata.history != "shared" and (m.author.id != _user.id and not m.author.bot and
					bot.commands.chatconfig[0].retrieve(m.author).history != "shared"
				):
					continue
				if bot.is_optout(m.author.id):
					continue
				message = cdict(
					role="assistant" if m.author.bot else "user",
					name=m.author.display_name,
					content=await bot.superclean_content(m),
					url=message_link(m),
				)
				messages[m.id] = message
		await bot.require_integrity(_message)
		print("ASK:", _channel.id, input_message)
		fut = self.ask_iterator(bot, _message, _channel, _guild, _user, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, _premium, model, nsfw, simulated)
		if pdata.stream and pdata.tts != "discord" and not simulated:
			return cdict(
				content=fut,
				b_tts=pdata.tts == "builtin",
			)
		temp = await flatten(fut)
		if not temp:
			return "\xad"
		elif isinstance(temp[-1], dict) and (temp[-1].content.startswith("\r") or len(temp) == 1):
			resp = temp[-1]
			resp["content"] = await bot.proxy_emojis(resp["content"], guild=_guild)
			resp["tts"] = pdata.tts == "discord"
			return resp
		raise RuntimeError(temp)

	async def ask_iterator(self, bot, _message, _channel, _guild, _user, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, premium, _model, nsfw, simulated):
		function_message = None
		tool_responses = []
		props = cdict(name=bot_name)
		response = cdict()
		reasonings = []
		reacts = []
		if not _model or _model == "auto":
			_model = pdata.model
			if not _model or _model == "auto":
				try:
					if simulated:
						raise PermissionError
					premium.require(2)
				except PermissionError:
					_model = "small"
				else:
					_model = "large" if premium.value_approx >= 3 else "medium"
		if _model == "large":
			premium.require(3)
		elif _model == "medium":
			premium.require(2)
		model = ["miza-1", "miza-2", "miza-3"][("small", "medium", "large").index(_model)]
		usage = [0, 0]
		try:
			ex = RuntimeError("Maximum inference attempts exceeded (model likely encountered an infinite loop).")
			content = ""
			for att in range(4):
				await bot.require_integrity(_message)
				if content:
					yield "\n\n"
				text = ""
				messagelist = [messages[k] for k in sorted(messages) if not reference or k != reference.id]
				messagelist.insert(0, system_message)
				if reply_message:
					messagelist.append(reply_message)
				if input_message:
					messagelist.append(input_message)
				if function_message:
					messagelist.append(function_message)
					messagelist.extend(tool_responses)
				m = None
				modelist = None
				async for resp in bot.chat_completion(messagelist, model=model, max_tokens=16384, tool_choice=None, tools=TOOLS, user=_user, props=props, stream=True, allow_nsfw=nsfw, predicate=lambda: bot.verify_integrity(_message), premium_context=premium):
					if isinstance(resp, dict):
						if resp.get("reasoning"):
							reasonings.extend(resp["reasoning"])
						if resp.get("cargs"):
							props.cargs = resp["cargs"]
						if resp.get("usage"):
							usage[0] = T(resp.usage).get("prompt_tokens", 0)
							usage[1] = T(resp.usage).get("completion_tokens", 0)
						m = resp.choices[0].delta
						temp = m.content or ""
						if temp:
							if temp.startswith("\r"):
								text = temp[1:]
							else:
								text += temp
						if getattr(m, "tool_calls", None):
							modelist = getattr(m, "modelist", None) or modelist
							break
					else:
						if resp.startswith("\r"):
							text = resp[1:]
						else:
							text += resp 
					yield "\r" + content + ("\n\n" * bool(content)) + text
				await bot.require_integrity(_message)
				text = text.strip()
				if not m:
					m = cdict(content=text)
				tc = getattr(m, "tool_calls", None) or ()
				if text and "<Image" in text:
					text, ft = text.split("<Image", 1)
					text = text.rstrip()
					if ":" in ft:
						ft = ft.split(":", 1)[-1]
					ft = ft.split(">", 1)[0]
					tc = list(tc)
					fc = cdict(
						id=str(ts_us() // 1000)[::-1],
						function=cdict(
							name="txt2img",
							arguments=json.dumps(dict(prompt=ft)),
						),
					)
					tc.append(fc)
					yield "\r" + content + ("\n\n" * bool(content)) + text
				elif tc:
					tc = [cdict(id=t.id, type="function", function=cdict(t.function)) for t in tc]
				if tc:
					m.tool_calls = tc
				for n, fc in enumerate(tuple(tc)):
					if n >= 8:
						break
					name = fc.function.name
					if function_message and fc.function in (t.function for t in function_message.tool_calls):
						if tc:
							tc.remove(fc)
						continue
					tid = fc.id
					fc.id = tid
					try:
						kwargs = cdict(eval_json(fc.function.arguments))
					except Exception:
						print("Tool Call Error:", fc.function.arguments)
						print_exc()
						continue
					kwargs.pop("description", None)
					kwargs.pop("required", None)
					call = None
					if name == "wolfram_alpha":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						if regexp(r"[1-9]*[0-9]?\.?[0-9]+[+\-*/^][1-9]*[0-9]?\.?[0-9]+").fullmatch(argv.strip().replace(" ", "")):
							name = "sympy"

					async def rag(name, tid, fut):
						nonlocal function_message
						print(f"{name} query:", argv)
						succ = False
						if not function_message:
							function_message = cdict(m)
						else:
							cids = {c.id for c in function_message.tool_calls}
							for c in m.tool_calls:
								if c.id not in cids:
									function_message.tool_calls.append(c)
						res = "[RESPONSE EMPTY OR REDACTED]"
						try:
							res = await fut
							succ = res and (isinstance(res, (str, bytes)) or res.get("content"))
						except Exception as ex:
							print_exc()
							res = repr(ex)
						if succ:
							temp = str(res)
							print(f"{name} result:", len(temp), lim_str(temp, 256))
						if succ and isinstance(res, bytes):
							data_url = await bot.to_data_url(res)
							res = [cdict(type="image_url", image_url=cdict(url=data_url, detail="auto"))]
							# # bytes indicates an image, use vision to describe it
							# res = await bot.vision(url=res, premium_context=premium)
						elif succ and isinstance(res, dict):
							res = res.get("content", "")
						elif succ:
							c = await tcount(res)
							ra = 1 if premium.value < 2 else 1.5 if premium.value < 5 else 2
							if c > round(4000 * ra):
								res = await ai.summarise(res, max_length=round(19600 * ra), min_length=round(3200 * ra), best=2 if premium.value >= 2 else 1, prompt=prompt)
								res = res.replace("\n", ". ").replace(": ", " -")
							res = res.strip()
						rs_msg = cdict(role="tool", name=name, content=res, tool_call_id=tid)
						tool_responses.append(rs_msg)
						return succ

					succ = None
					argv = None
					if name == "browse":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						s = f'\n> Browsing "{argv}"...'
						text += s
						yield s
						fut = bot.browse(argv, uid=_user.id, screenshot=False)
						succ = await rag(name, tid, fut)
					elif name == "sympy":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						prec = float(kwargs.get("precision") or 128)
						s = f'\n> Calculating "{argv}"...'
						text += s
						yield s
						fut = bot.solve_math(argv, prec=prec, timeout=24, nlp=True)
						succ = await rag(name, tid, fut)
						if not succ:
							name = "wolfram_alpha"
					if name == "wolfram_alpha":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						s = f'\n> Solving "{argv}"...'
						text += s
						yield s
						fut = process_image("wolframalpha", "$", [argv], cap="browse", timeout=60)
						succ = await rag(name, tid, fut)
					elif name == "myinfo":
						async def myinfo(argv):
							u2 = None
							if argv.strip("-"):
								if not _guild and getattr(_channel, "recipient", None):
									u2 = bot.query_members([_channel.recipient, bot.user], argv)
								else:
									u2 = await bot.fetch_user_member(argv, _guild)
							if not u2:
								u2 = bot
							if u2.id == bot.id:
								if pdata.description == DEFPER and bot_name == bot.user.display_name and bot.id == 668999031359537205:
									res = "- You are `Miza`, a multipurpose, multimodal bot that operates on social platforms such as Discord.\n- Your appearance is based on the witch-girl `Misery` from `Cave Story`.\n- Your creator is <@201548633244565504>, and you have a website at https://mizabot.xyz which a guide on your capabilities!"
								else:
									cap = await self.bot.caption(best_url(u2), best=2 if premium.value_approx >= 4 else 0, timeout=24)
									s = "\n\n".join(filter(bool, cap)).strip()
									res = f"- You are `{u2.name}`, a multipurpose, multimodal bot that operates on social platforms such as Discord.\n- Your appearance is based on `{s}`."
									if bot.owners:
										i = next(iter(bot.owners))
										um = user_mention(i)
										res += f"\n-Your owner is {um}."
							else:
								cap = await self.bot.caption(best_url(u2), best=2 if premium.value_approx >= 4 else 0, timeout=24)
								s = "\n\n".join(filter(bool, cap)).strip()
								res = f"- Search results: `{u2.name}` has the appearance of `{s}`."
							return res
						argv = kwargs.get("user") or " ".join(kwargs.values())
						s = f'\n> Querying "{argv}"...'
						text += s
						yield s
						fut = myinfo(argv)
						succ = await rag(name, tid, fut)
					elif name == "img2txt":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						s = f'\n> Interpreting "{argv}"...'
						text += s
						yield s
						fut = bot.vision(input_message.url, question=argv)
						succ = await rag(name, tid, fut)
					elif name == "txt2img":
						argv = kwargs.get("description") or kwargs.get("prompt") or " ".join(kwargs.values())
						s = f'\n> Generating "{argv}"...'
						text += s
						yield s
						url = _message.jump_url if _message.attachments else reference.jump_url if reference else None
						call = {"func": "imagine", "prompt": argv, "url": url, "count": kwargs.get("count") or 1, "comment": text}
					elif name == "reminder":
						argv = str(kwargs.message) + " -t " + str(kwargs.time)
						call = {"func": "remind", "message": kwargs.message, "time": kwargs.time, "comment": text}
					elif name == "play":
						call = {"func": "play", "query": kwargs.query, "comment": text}
					elif name == "audio":
						call = {"func": kwargs.mode, "value": kwargs.value}
					elif name == "audiostate":
						if kwargs.mode == "quit":
							call = {"func": "disconnect"}
						elif kwargs.mode == "pause":
							call = {"func": ("pause" if kwargs.value else "resume")}
						elif kwargs.mode == "loop":
							call = {"func": "loopqueue", "argv": int(kwargs.value)}
						else:
							call = {"func": kwargs.mode, "argv": int(kwargs.value)}
					if succ:
						print("New prompt:", lim_str(tool_responses, 65536))
					if not call:
						continue
					print("Function Call:", call)
					fname = call.pop("func")
					u_perm = bot.get_perms(_user)
					command_check = fname
					loop = False
					timeout = 240
					command = bot.commands[fname][0]
					fake_message = copy.copy(_message)
					if getattr(command, "schema", None):
						fake_message.content = f"{bot.get_prefix(_guild)}{fname} " + " ".join(('"' + v + '"' if " " in v else v) for v in map(str, kwargs.values()))
					else:
						fake_message.content = f"{bot.get_prefix(_guild)}{fname} {argv or ''}".rstrip()
					fake_message.attachments = []
					comment = (call.pop("comment", "") or "") + f"\n> Used `{fake_message.content}`"
					if command.schema:
						for k, v in command.schema.items():
							if k not in call:
								call[k] = v.get("default")
					resp = await bot.run_command(command, call, message=fake_message, comment=comment, respond=False)
					if resp:
						print("Intermediate:", resp)
						if isinstance(resp, dict):
							response.update(resp)
						else:
							response.content = resp
						rtext = response.get("content", "")
						text = ("\r" + rtext).strip()
						tc = None
				content += "\n\n" * bool(content) + text
				if text and not tc:
					raise StopIteration
			else:
				raise ex
		except StopIteration:
			pass
		print("Usage:", usage)
		if reasonings:
			reasoning = "\n\n\n".join(reasonings).encode("utf-8")
			async with niquests.AsyncSession() as asession:
				resp2 = await asession.post(
					"https://api.mizabot.xyz/upload?filename=reasoning.txt",
					data=reasoning,
				)
				url = resp2.text
			content = (f"> [Reasoning (click to view)]({url})\n" + content).strip()
		response.content = "\r" + content
		embs = []
		if response.get("embed"):
			embs.append(response["embed"])
		if response.get("embeds"):
			embs.extend(response["embeds"])
		desc = premium.apply()
		if desc:
			desc = "-# " + "\n-# ".join(desc.splitlines())
			response.content += "\n" + desc
			print(">", desc)
		if not xrand(20):
			tips = [
				"*Tip: By using generative AI, you are assumed to comply with the [ToS](<https://github.com/thomas-xin/Miza/wiki/Terms-of-Service>).*",
				"*Tip: The chatbot feature is designed to incorporate multiple SOTA models in addition to internet-based interactions. For direct interaction with the raw LLMs, check out ~instruct.*",
				"*Tip: My personality prompt and message streaming are among several parameters that may be modified. Check out ~help personality for more info. Note that an improperly constructed prompt may be detrimental to response quality, and that giving me a nickname may also have an effect.*",
				"*Tip: Remember that anything a chatbot says may be fictional or otherwise made-up. Always fact-check from reputable sources before making serious assumptions, and don't take the AI's words too seriously.*",
				"*Tip: At any point in time, you can delete your command message to stop generation.*",
			]
			if premium.value < 3:
				tips.append("*Tip: Many of my capabilities are not readily available due to cost reasons. You can gain access by donating through one of the premium subscriptions available, which serves to approximately fund individual usage.*")
			if not nsfw:
				tips.append("*Tip: I automatically try to correct inaccurate responses when possible. However, this is not foolproof; if you would like this feature more actively applied to counteract censorship, please move to a NSFW channel or use ~verify if in DMs.*")
			if pdata.history != "shared":
				tips.append("*Tip: For privacy reasons, conversation histories (allowing referencing previous messages in the same channel) is disabled by default. If you would like to enable this, use `~chatconfig --history private`, or `~chatconfig --history shared` if you would also like the bot to be able to read multi-user conversations. This enables me to read up to 192 previous messages from the current channel. No messages from other channels are included.*")
			note = "-# " + choice(tips)
			embs.append(discord.Embed(
				colour=rand_colour(),
				description=note
			))
			print(">", note)
		response.embeds = embs
		response.reacts = tuple(response.get("reacts", ())) + tuple(reacts)
		yield response

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


DEFPER = "Your name is \"{{char}}\"; you are intelligent, cute and playful. Your task is to answer the last question from the user named \"{{user}}\" with creative and natural-feeling responses (be wholesome, but not too corny), in first person as if you were a human. You are free to make up your own preferences and show emotion, but you should always match the language used (e.g. English). If a user inquires for additional services not accessible through tools, you should direct them to the ~help or /help commands rather than mimic outputs, if you suspect they may involve side effects. Be concise where possible."

class ChatConfig(Command):
	name = ["Personality", "ChangePersonality"]
	min_level = 0
	min_display = "0+"
	description = "Customises my personality and behaviours for ~ask and @mentions in the current channel. Note that with the increased complexity of the chatbot, a clear description of who the bot is should be provided. If used in DMs, applies per-user settings that will override settings in servers, except tts and history."
	schema = cdict(
		description=cdict(
			type="string",
			description='Personality description; enter "DEFAULT" to reset',
			example="Your name is Miza, you are dry, sarcastic and snarky. Be creative with your responses and attempt to annoy the user.",
		),
		model=cdict(
			type="enum",
			validation=cdict(
				enum=("auto", "small", "medium", "large"),
			),
			description="Model size hint. Larger size increases intelligence, at the cost of higher quota usage",
			example="small",
		),
		stream=cdict(
			type="bool",
			description="Determines whether the response should be edited, or delayed until complete, default true",
			example="false",
		),
		tts=cdict(
			type="enum",
			validation=cdict(
				enum=("none", "discord", "builtin"),
			),
			description="""Whether the output should include automatic text-to-speech audio. "discord" mode uses Discord's builtin TTS feature, while "builtin" mode will play the output in the voice channel when available""",
		),
		history=cdict(
			type="enum",
			validation=cdict(
				enum=("none", "private", "shared"),
			),
			description="Whether chat history is enabled, and if so, whether the conversation is shared (including messages from different users)",
		),
		apply_all=cdict(
			type="bool",
			description="Whether to apply to all channels (only applicable in servers)",
			default=False,
		),
	)
	rate_limit = (18, 24)
	slash = True
	ephemeral = True

	def retrieve(self, channel, user=None, update=True):
		per = cdict(
			model="auto",
			description=DEFPER,
			stream=True,
			tts="none",
			history="none",
		) if update else cdict()
		p = self.bot.get_guildbase(get_guild_id(channel), "chatconfig", {}).get(channel.id)
		if p:
			per.update(p)
		if user:
			p = self.bot.get_guildbase(user.id, "chatconfig", {}).get(user.id)
			if p:
				p.pop("tts", None)
				p.pop("history", None)
				per.update(p)
		return per

	async def __call__(self, bot, _nsfw, _guild, _channel, _user, _premium, _perm, description, model, stream, tts, history, apply_all, **void):
		if hasattr(_channel, "recipient"):
			targets = [_channel.recipient]
			gid = targets[0].id
			personal = True
		else:
			targets = _guild.text_channels if apply_all else [_channel]
			gid = get_guild_id(_channel)
			personal = False
		pers = bot.get_guildbase(gid, "chatconfig", {})
		req = 2
		s = ""
		for target in targets:
			if description == "DEFAULT":
				if _perm < req:
					reason = f"to modify chat config for {channel_repr(target)}"
					raise self.perm_error(_perm, req, reason)
				pers.pop(target.id, None)
				s += css_md(f"Chat settings for {sqr_md(target)} have been reset.")
				continue
			if not description and model is None and stream is None and tts is None and history is None:
				p = self.retrieve(target)
				s += ini_md(f"Current chat settings for {sqr_md(target)}:{iter2str(p)}")
				if _perm < req:
					s += f"\n(Use {bot.get_prefix(_guild)}chatconfig DEFAULT to reset; case-sensitive)."
				continue
			if _perm < req:
				reason = f"to modify chat config for {channel_repr(target)}"
				raise self.perm_error(_perm, req, reason)
			if description:
				description = await bot.superclean_content(description)
			if description and (len(description) > 4096 or len(description) > 512 and _premium.value < 2):
				raise OverflowError("Maximum currently supported personality prompt size is 512 characters, 4096 for premium users.")
			if description and not _nsfw:
				resp = await ai.moderate(description)
				if nsfw_flagged(resp):
					print(resp)
					raise PermissionError(
						"Apologies, my AI has detected that your input may be inappropriate.\n"
						+ "Please move to a NSFW channel, reword, or consider contacting the support server if you believe this is a mistake!"
					)
			p = pers.get(target.id) or cdict()
			if description:
				p.description = description
			if model is not None:
				p.model = model
			if tts is not None:
				p.tts = tts
			if history is not None:
				p.history = history
			pers[target.id] = p
			bot.set_guildbase(gid, "chatconfig", pers)
			p = self.retrieve(target, _user)
			s += css_md(f"Chat settings for {sqr_md(target)} have been changed to {iter2str(p)}\n(Use {bot.get_prefix(_guild)}chatconfig DEFAULT to reset).")
		bot.set_guildbase(gid, "chatconfig", pers)
		return s


class Instruct(Command):
	name = ["Complete", "Completion"]
	description = "Similar to ~ask, but functions as instruct rather than chat."
	schema = cdict(
		model=cdict(
			type="enum",
			validation=cdict(
				enum=list(ai.available),
			),
			description="Target LLM to invoke",
			example="deepseek",
		),
		prompt=cdict(
			type="string",
			description="Input text for completion",
			example="Once upon a time, there was",
		),
		api=cdict(
			type="string",
			description="Custom OpenAI-compatible API url, optionally followed by API key and then model, all separated with \"#\"",
			example="https://api.deepinfra.com/v1/openai#your-api-key-here#gpt-3.5-turbo",
		),
		temperature=cdict(
			type="number",
			validation="[0, 3]",
			description="Temperature to influence alignment",
			example="1.2",
			default=0.8,
		),
		frequency_penalty=cdict(
			type="number",
			validation="[-1, 2]",
			description="Amount to penalise tokens based on frequency",
			example="1.1",
			default=0.7,
		),
		presence_penalty=cdict(
			type="number",
			validation="[-1, 2]",
			description="Amount to penalise tokens based on presence",
			example="1.1",
			default=0.5,
		),
		reasoning_effort=cdict(
			type="enum",
			validation=cdict(
				enum=("minimal", "low", "medium", "high"),
			),
			default="low",
		),
		max_tokens=cdict(
			type="integer",
			validation="[1, 65536]",
			description="Maximum tokens to generate",
			example="16384",
			default=3072,
		),
	)
	macros = cdict(
		GPT5=cdict(
			model="gpt-5.1",
		),
		GPT4=cdict(
			model="gpt-4.1",
		),
		R1=cdict(
			model="deepseek-r1",
		),
		Grok=cdict(
			model="grok-4",
		),
		Gemini=cdict(
			model="gemini-3-pro",
		),
		Deepseek=cdict(
			model="deepseek-v3.2",
		),
		Claude=cdict(
			model="claude-4.5-sonnet",
		),
	)
	rate_limit = (12, 16)
	slash = True
	ephemeral = True
	cache = AutoCache(stale=360, timeout=720)

	async def __call__(self, bot, _message, _premium, model, prompt, api, temperature, frequency_penalty, presence_penalty, reasoning_effort, max_tokens, **void):
		# assert model in ai.available, f"{model} does not exist or is not supported."
		kwargs = {}
		if api:
			key = model = None
			spl = api.split("##", 1)
			if len(spl) > 1:
				api, model = spl
			else:
				spl = api.split("#", 2)
				if len(spl) > 2:
					api, key, model = spl
				elif len(spl) > 1:
					api, key = spl
				else:
					api = spl[0]
			head = {"Content-Type": "application/json"}
			if key:
				head["Authorization"] = "Bearer " + key
			if not model:
				info = await self.cache.aretrieve(api, Request.aio, api + "/models", headers=head, json=True)
				models = [m.get("id") for m in sorted(info["data"], key=lambda m: m.get("created"), reverse=True)]
				model = models[0]
			key = key or "x"
			oai = openai.AsyncOpenAI(api_key=key, base_url=api)
			kwargs["api"] = oai
		kwargs["max_tokens"] = max_tokens
		kwargs["reasoning_effort"] = reasoning_effort
		if not model:
			kwargs["api"] = await asubmit(ai.load_summarisation_model)
			model = ai.summarisation_model.model
		resp = await bot.force_completion(model=model, prompt=prompt, stream=True, timeout=1800, temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, premium_context=_premium, allow_alt=True, **kwargs)
		try:
			_message.__dict__.setdefault("inits", []).append(resp)
		except Exception:
			pass
		prefix = "\xad"
		async def respond():
			async for m in resp:
				if not await bot.verify_integrity(_message):
					await resp.close()
					return
				yield m
			desc = _premium.apply()
			if desc:
				yield "\n-# " + "\n-# ".join(desc.splitlines())
		return cdict(content=respond(), prefix=prefix, bypass_prefix=["> ", "# ", "## ", "### "], message=_message)


class Imagine(Command):
	_timeout_ = 150
	name = ["AIArt", "Dream", "Envision", "Inspire", "Txt2Img"]
	description = "Runs one of many AI image generators on the input prompt or image. Less censorship is imposed when invoked within NSFW channels. Incurs an adjusted quota cost."
	schema = cdict(
		model=cdict(
			type="enum",
			validation=cdict(
				enum=("dalle3", "dalle2", "flux", "nano-banana"),
				accepts=dict(dalle="dalle3"),
			),
			description="AI model for generation",
			example="dalle3",
			default="nano-banana",
		),
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("preprocess", "raw", "caption", "inpaint", "controlnet", "canny"),
			),
			description='Transform mode; "preprocess" and "raw" affect text prompts, while the others affect image prompts',
			example="raw",
			default="preprocess",
		),
		url=cdict(
			type="image",
			description="Image supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
		),
		prompt=cdict(
			type="string",
			description="Description to create image with",
			example="A cat finding a treasure chest full of rainbows",
			default="",
			required_slash=True,
		),
		mask=cdict(
			type="image",
			description="Mask supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
		),
		num_inference_steps=cdict(
			type="number",
			validation="[1, 96]",
			description="How many times data is passed through model. Automatically divided by 8 for SDXL-Lightning and Flux-Schnell",
			example="64",
			default=32,
			aliases=["steps"],
		),
		high_quality=cdict(
			type="bool",
			description="Requests a high quality image; may increase quota cost. Increases resolution for relevant models, and increases inference steps by 25%",
			example="True",
			default=True,
			aliases=["hq"],
		),
		strength=cdict(
			type="number",
			validation="[0, 1]",
			description="Denoising strength",
			example="0.6",
			default=0.85,
		),
		guidance_scale=cdict(
			type="number",
			validation="[-100, 100]",
			description="Prompt guidance scale. Automatically divided by 5 for SDXL Lightning",
			example="12",
			default=6,
			aliases=["gs"],
		),
		aspect_ratio=cdict(
			type="resolution",
			validation="[0.0009765625, 8192]",
			description="Output aspect ratio (total amount of pixels remains roughly the same)",
			example="16:9",
			default=(0, 0),
			aliases=["ar"],
		),
		negative_prompt=cdict(
			type="text",
			description="Unwanted elements",
			example="blurry",
			aliases=["np"],
		),
		count=cdict(
			type="integer",
			validation="[1, 9]",
			description="Number of output images to produce",
			example="6",
		),
	)
	macros = cdict(
		DALLE3=cdict(
			model="dalle3",
		),
	)
	rate_limit = (12, 16)
	slash = True
	tips = (
		"*Tip: By using generative AI, you are assumed to comply with the [ToS](<https://github.com/thomas-xin/Miza/wiki/Terms-of-Service>).*",
		"*Tip: Use --ar or --aspect-ratio to control the proportions of the image.*",
		"*Tip: Use -c or --count to control the amount of images generated; maximum 4 for regular users, 9 for premium.*",
		'*Tip: Standalone text prompts that are short may be reinterpreted by the language model automatically. Use the ALT button on image outputs to see what was added (requires "With image descriptions" enabled in Discord chat settings)*',
	)
	comfyui_json = "misc/comfyui-api.json"
	comfyui_api = None
	comfyui_i = 0
	comfyui_n = 0
	if os.path.exists(comfyui_json):
		with open(comfyui_json, "rb") as f:
			comfyui_data = orjson.loads(f.read())
		if "prompt" not in comfyui_data:
			comfyui_data = dict(client_id=0, prompt=comfyui_data)
		if "comfyui_api" in AUTH:
			comfyui_api = AUTH["comfyui_api"]
			for v in comfyui_api.values():
				comfyui_n += v.get("weight", 1)
	else:
		comfyui_data = None

	async def __call__(self, bot, _user, _channel, _message, _perm, _premium, _prefix, _comment, _nsfw, model, mode, url, prompt, mask, num_inference_steps, high_quality, strength, guidance_scale, aspect_ratio, negative_prompt, count, **void):
		model = model or "auto"
		mode = mode or ("raw" if url or mask else "preprocess")
		aspect_ratio = 0 if not aspect_ratio[0] or not aspect_ratio[1] else aspect_ratio[0] / aspect_ratio[1]
		count = count or (4 if _premium.value_approx >= 3 and not url else 1)
		limit = 18 if _premium.value >= 5 else 9 if _premium.value >= 3 else 4
		amount = min(count, limit)
		amount2 = 0
		if model == "dalle3":
			_premium.require(4)
		elif model == "dalle2":
			_premium.require(3)
		else:
			_premium.require(2)
		if _premium.value_approx < 4:
			num_inference_steps = min(36, num_inference_steps)
		elif _premium.value_approx < 2:
			num_inference_steps = min(28, num_inference_steps)
		if high_quality:
			num_inference_steps *= 1.25
		nsfw = _nsfw or bot.is_nsfw(_user)
		nsfw_prompt = False
		if mask:
			raise NotImplementedError("Masks are currently paused due to capacity issues, apologies for any inconvenience!")
		if mode == "caption":
			if url:
				pt, *p1 = await bot.caption(url, best=3 if _premium.value_approx >= 4 else 1, premium_context=_premium)
				caption = "\n".join(filter(bool, p1))
				if prompt:
					prompt += f" ({caption})"
				else:
					prompt = caption
				url = None
		if not prompt:
			prompt = "imagine art " + str(ts_us())[::-1] if mode == "preprocess" else ""
		mod_resp = await ai.moderate("Create an image of: " + prompt, premium_context=_premium)
		if prompt and not nsfw:
			flagged = nsfw_flagged(mod_resp)
			if not nsfw and flagged:
				raise PermissionError(
					f"Apologies, my AI has detected that your input may be inappropriate: {flagged}.\n"
					+ "Please move to a NSFW channel, reword, or consider contacting the support server if you believe this is a mistake!"
				)
			if flagged:
				print("Flagged:", mod_resp)
				if mod_resp.categories.sexual:
					nsfw_prompt = True
			negative_prompt = negative_prompt or ", ".join(set(("watermark", "blurry", "distorted", "disfigured", "bad anatomy", "poorly drawn")).difference(smart_split(prompt)))
		await bot.require_integrity(_message)
		image = None
		was_ai = 1
		if url:
			fut = csubmit(bot.data.prot.scan(_message, url))
			fut2 = csubmit(bot.to_data_url(url))
			if not aspect_ratio:
				b = await attachment_cache.download(url, filename=True)
				p = 2 if getsize(b) > 1048576 else 0
				w, h = await asubmit(get_image_size, b, priority=p)
				aspect_ratio = w / h
			try:
				res, n = await fut
			except Exception:
				n = 1
			was_ai = 1 if n == 1 else 0
			image = await fut2

		pnames = []
		futs = []
		eprompts = alist()
		dups = max(1, random.randint(amount >> 2, amount))
		oprompt = prompt
		if mode in ("caption", "preprocess") and not url and len(prompt.split()) < 32:
			temp = oprompt.replace('"""', "'''") or "[Art]"
			prompt = f'### Instruction:\n"""\n{temp}\n"""\n\nImprove the above image caption as a description to send to txt2img image generation. Be as creative and detailed as possible in at least 2 sentences, but stay concise!\n\n### Response:'
			resp = cdict(choices=[])
			if len(resp.choices) < dups:
				futi = []
				for i in range(max(1, dups - 1)):
					fut = csubmit(ai.instruct(
						dict(
							prompt=prompt,
							temperature=1,
							max_tokens=200,
							premium_context=_premium,
						),
						user=_user,
					))
					futi.append(fut)
				for fut in futi:
					try:
						s = await fut
						assert len(s.strip()) > 12, f"Language model returned too short: {s}"
					except Exception:
						print_exc()
						continue
					resp.choices.append(cdict(text=s))
			if len(resp.choices) < max(1, dups - 1):
				resp2 = await ai.llm(
					"chat.completions.create",
					model="gpt-4.1-mini",
					messages=[dict(role="user", content=prompt)],
					temperature=1,
					max_tokens=120,
					user=_user,
					n=max(1, dups - len(resp.choices) - 1),
					premium_context=_premium,
				)
				if resp:
					resp.choices.extend(cdict(text=choice.message.content) for choice in reversed(resp2.choices))
				else:
					resp = resp2
			for e in resp.choices:
				out = e.text.strip()
				if ai.decensor.search(out):
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
					prompt = oprompt.removesuffix(".") + " BREAK " + out.strip()
				else:
					prompt = oprompt
				eprompts.append(prompt)
			eprompts.uniq()
			eprompts.shuffle()
		if oprompt not in eprompts:
			eprompts.append(oprompt)
		# print("PROMPT:", eprompts or prompt)
		await bot.require_integrity(_message)

		if amount2 < amount and model == "nano-banana":
			ars = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
			selected_ar = ars[0]
			if aspect_ratio:
				diff = inf
				for i, ar in enumerate(ars):
					p, q = map(int, ar.split(":", 1))
					d2 = abs(p / q - aspect_ratio)
					if d2 < diff:
						diff = d2
						selected_ar = ar

			def nano_banana(text, image):
				p = text if text and image else f'Please create an image, to the best of your ability, according to the given user caption:\n\n"""\n{text}"""' if text else "Please make a more interesting version of the provided picture!"
				content = [dict(type="text", text=p)]
				if image:
					content.append(dict(type="image_url", image_url=dict(url=image)))
				url = "https://openrouter.ai/api/v1/chat/completions"
				headers = {
					"Authorization": f"Bearer {AUTH['openrouter_key']}",
					"Content-Type": "application/json",
				}
				payload = {
					"model": "google/gemini-2.5-flash-image",
					"messages": [
						{
							"role": "user",
							"content": content,
						},
					],
					"reasoning": {"effort": "minimal"},
					"modalities": ["image", "text"],
					"image_config": {
						"aspect_ratio": selected_ar,
					}
				}
				response = requests.post(url, headers=headers, json=payload, timeout=48)
				try:
					response.raise_for_status()
				except Exception:
					print(response.content)
					raise
				result = response.json()
				usage = result["usage"]
				cost = mpf(1.238 + 0.03 if image else 0.03) / 1000 + mpf(0.3 * usage["prompt_tokens"] + 2.5 * usage["completion_tokens"]) / 1000000
				_premium.append(["mizabot", resp_model, cost])
				if result.get("choices"):
					message = result["choices"][0]["message"]
					if message.get("images"):
						for image in message["images"]:
							image_url = image["image_url"]["url"]
							return base64.b64decode(image_url.split("base64,", 1)[-1].encode("ascii"))
				if image:
					raise RuntimeError("No image was produced!", f"LLM response: {repr(result)}")
				print(result)

			resp_model = "gemini-2.5-flash-image-preview"
			n = amount - amount2
			if n > 1 and not url:
				n -= 1
			pps = [eprompts.next().replace(" BREAK ", "\n") for i in range(n)]
			pnames.extend(pps)
			cfuts = [asubmit(nano_banana, p, image) for p in pps]
			await gather(*cfuts)
			temp_futs = []
			for fut in cfuts:
				if fut.result():
					temp_futs.append(fut)
					amount2 += 1
			for fut in temp_futs:
				fut.model = resp_model
			futs.extend(temp_futs)

		if amount2 < amount and model in ("dalle3", "dalle2"):
			dalle = "2" if model == "dalle2" else "3"
			ar = float(aspect_ratio) or 1
			if url:
				raise NotImplementedError(f"Dall·E {model} interface currently does not support image prompts.")
			resp_model = f"dall-e-{dalle}"
			if model == "dalle2":
				if max(ar, 1 / ar) < 1.1:
					size = "1024x1024"
				else:
					raise ValueError(f"Dall·E {model} interface currently only supports 1:1 aspect ratio.")
				prompt = eprompts.next()
				prompt = lim_str(prompt, 1000)
				c = amount - amount2
				response = await ai.get_oai("images.generate")(
					model=resp_model,
					prompt=prompt,
					size=size,
					n=c,
					user=str(hash(str(_user))),
				)
				images = response.data
				cost = "0.02"
				if c > 1:
					cost = str(mpf(cost) * c)
				_premium.append(["openai", resp_model, cost])
				pnames.extend([prompt] * len(images))
			elif model == "dalle3":
				if max(ar, 1 / ar) < 1.1:
					size = "1024x1024"
				elif max(ar / 1.75, 1.75 / ar) < 1.1:
					size = "1792x1024"
				elif max(ar * 1.75, 1 / 1.75 / ar) < 1.1:
					size = "1024x1792"
				else:
					raise ValueError(f"Dall·E {model} interface currently only supports 1:1, 7:4 and 4:7 aspect ratios.")
				q = "hd" if high_quality else "standard"
				futn = []
				for i in range(amount - amount2):
					fut = csubmit(ai.get_oai("images.generate")(
						model=resp_model,
						prompt=lim_str(eprompts.next(), 4000),
						size=size,
						quality=q,
						n=1,
						style="natural" if i else "vivid",
						user=str(hash(str(_user))),
					))
					futn.append(fut)
				images = []
				for fut in futn:
					try:
						response = await fut
					except openai.RateLimitError:
						print_exc()
						await asyncio.sleep(60)
						response = await ai.get_oai("images.generate")(
							model=f"dall-e-{dalle}",
							prompt=lim_str(eprompts.next(), 4000),
							size=size,
							quality="standard",
							n=1,
							style="natural",
							user=str(hash(str(_user))),
						)
					except Exception:
						print_exc()
						try:
							response = await ai.get_oai("images.generate")(
								model=f"dall-e-{dalle}",
								prompt=lim_str(eprompts.next(), 4000),
								size=size,
								quality="standard",
								n=1,
								style="natural",
								user=str(hash(str(_user))),
							)
						except Exception:
							if amount <= 1:
								raise
							print_exc()
							print("SKIPPED")
							continue
					images.append(response.data[0])
					pnames.append(response.data[0].revised_prompt)
					if size == "1024x1024":
						cost = "0.08" if q == "hd" else "0.04"
					else:
						cost = "0.12" if q == "hd" else "0.08"
					_premium.append(["openai", resp_model, cost])
			temp_futs = []
			futs.extend(csubmit(Request.aio(im.url, timeout=48)) for im in images)
			for fut in temp_futs:
				fut.model = resp_model
			futs.extend(temp_futs)
			amount2 += len(images)
		if amount2 < amount and not url and (AUTH.get("deepinfra_key") or AUTH.get("together_key")):
			use_together = False #bool(AUTH.get("together_key")) and not mod_resp.flagged
			if url:
				url = image
			resp_model = "black-forest-labs/FLUX.1-redux" if url else "black-forest-labs/FLUX.1-schnell-Free" if use_together else "black-forest-labs/FLUX-1-schnell"
			c = amount - amount2
			ms = 1024 if high_quality else 768
			x, y = max_size(aspect_ratio or 1, 1, ms, force=True)
			max_width = 1792 if use_together else 2048
			min_width = 64 if use_together else 128
			if x > max_width:
				y *= max_width / x
				x = max_width
			if y > max_width:
				x *= max_width / y
				y = max_width
			if x < min_width:
				x = min_width
			if y < min_width:
				y = min_width
			d = 16
			w, h = (round(x / d) * d, round(y / d) * d)
			queue = []
			c2 = c
			while c2 > 0:
				n = round_random(min(4, c2, amount / dups))
				p = eprompts.next()
				prompt = p.replace(" BREAK ", "\n")
				steps = max(1, round_random(num_inference_steps / 8))
				if use_together:
					fut = csubmit(Request.aio(
						"https://api.together.xyz/v1/images/generations",
						method="POST",
						headers={"Content-Type": "application/json", "Authorization": "Bearer " + AUTH["together_key"]},
						data=orjson.dumps(dict(
							model=resp_model,
							prompt=prompt,
							n=n,
							steps=min(4, steps),
							width=w,
							height=h,
							response_format="b64_json",
							image_base64=[url] if url else [],
						)),
						json=True,
						timeout=60,
					))
				else:
					fut = csubmit(Request.aio(
						f"https://api.deepinfra.com/v1/inference/{resp_model}",
						method="POST",
						headers={"Content-Type": "application/json", "Authorization": "Bearer " + AUTH["deepinfra_key"]},
						data=orjson.dumps(dict(
							prompt=prompt,
							num_images=n,
							num_inference_steps=steps,
							width=w,
							height=h,
						)),
						json=True,
						timeout=60,
					))
				pnames.extend([prompt] * n)
				queue.append(fut)
				c2 -= n
				cost = mpf("0.0005") * (steps * w * h / 1048576)
				_premium.append(["deepinfra", resp_model, cost])

			def b64_data(b):
				if isinstance(b, dict):
					b = b["b64_json"]
				if isinstance(b, str):
					b = b.encode("ascii")
				elif isinstance(b, MemoryBytes):
					b = bytes(b)
				b = b.split(b"base64,", 1)[-1]
				if len(b) & 3:
					b += b"=="
				return base64.b64decode(b)

			temp_futs = []
			for fut in queue:
				data = await fut
				images = data["data"] if use_together else data["images"]
				for imd in images:
					imd = b64_data(imd)
					temp_futs.append(as_fut(imd))
					amount2 += 1
			for fut in temp_futs:
				fut.model = resp_model
			futs.extend(temp_futs)
		ffuts = []
		exc = RuntimeError("Unknown error occured.")
		for tup, prompt in zip(futs, pnames):
			if len(ffuts) >= amount:
				break
			if not tup:
				continue
			model = getattr(tup, "model", resp_model)
			if not isinstance(tup, tuple):
				if awaitable(tup):
					tup = await tup
				tup = (tup,)
			fn = tup[0]
			if not fn:
				continue
			if isinstance(fn, bytes):
				fn2 = temporary_file("png")
				async with aiofiles.open(fn2, "wb") as f:
					await f.write(fn)
				fn = fn2
			if isinstance(fn, str):
				if is_url(fn):
					fn = await attachment_cache.download(fn, filename=True)
				else:
					assert os.path.exists(fn)
			meta = cdict(
				issuer=bot.name,
				issuer_id=bot.id,
				type="AI_GENERATED" if was_ai else "AI_EDITED",
				engine=model,
				prompt=oprompt,
			)
			ffut = csubmit(process_image(fn, "ectoplasm", ["-nogif", orjson.dumps(meta), 1, "-f", "webp"], cap="caption", timeout=60))
			ffut.prompt = prompt
			ffut.back = fn
			ffuts.append(ffut)
		if not ffuts:
			raise exc
		files = []
		for ffut in ffuts:
			prompt = ffut.prompt
			fn = ffut.back
			fn = await ffut
			assert fn
			files.append(CompatFile(fn, filename=(prompt or "_") + ".webp", description=prompt))
		embs = []
		emb = discord.Embed(colour=rand_colour())
		emb.description = ""
		tos = bot.get_userbase(_user.id, "tos")
		if not tos:
			emb.description += "\n-# " + self.tips[0]
			bot.set_userbase(_user.id, "tos", True)
		elif not xrand(30):
			emb.description += "\n-# " + choice(self.tips)
		reacts = ["🔳"]
		prefix = ""
		desc = _premium.apply()
		if desc:
			emb.description += "\n-# " + "\n-# ".join(desc.splitlines())
		if emb.description:
			emb.description = emb.description.strip()
			embs.append(emb)
		if len(files) == 2:
			fe = files.pop(1)
			urls = await bot.data.exec.lproxy(fe, filename=fe.filename)
			if urls:
				_comment = ("\n".join(url.split("?", 1)[0] for url in urls) + "\n" + _comment).strip()
		return cdict(content=_comment, files=files, embeds=embs, prefix=prefix, reacts=reacts)


class Describe(Command):
	name = ["Description", "Image2Text", "Clip"]
	description = "Describes the input image."
	schema = cdict(
		url=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			required=True,
		),
	)
	rate_limit = (4, 5)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _user, _premium, url, **void):
		fut = csubmit(attachment_cache.scan_headers(url))
		cap = await self.bot.caption(url, best=1, premium_context=_premium, timeout=90)
		s = "\n\n".join(filter(bool, cap)).strip()
		headers = await fut
		name = fcdict(headers).get("Attachment-Filename") or url.split("?", 1)[0].rsplit("/", 1)[-1]
		return cdict(
			embed=discord.Embed(description=s, title=name).set_author(**get_author(_user)),
		)


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
	slash = True
	ephemeral = True

	async def __call__(self, bot, _user, url, **void):
		s = await bot.ocr(url)
		return cdict(
			embed=discord.Embed(description=s, title="Detected text").set_author(**get_author(_user)),
		)


class AudioSeparator(Command):
	name = ["Extract", "Separate"]
	description = "Runs Audio-Separator on the input URL. See https://github.com/nomadkaraoke/python-audio-separator for more info, or to run it yourself!"
	schema = cdict(
		url=cdict(
			type="audio",
			description="Audio supplied by URL or attachment",
			example="https://cocobeanzies.mizabot.xyz/music/rainbow-critter.webm",
			aliases=["a"],
			required=True,
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=("ogg", "opus", "mp3", "flac", "wav"),
			),
			description="The file format or codec of the output",
			default="opus",
		),
	)
	rate_limit = (20, 40)
	_timeout_ = 3.5

	async def __call__(self, bot, _channel, _message, url, format, **void):
		fut = csubmit(send_with_reply(
			_channel,
			reference=_message,
			content=italics(ini_md(f"Downloading and converting {sqr_md(url)}...")),
		))
		fn = await attachment_cache.download(url, filename=True)
		args = ["audio-separator", os.path.abspath(fn), "--output_format", format]
		proc = await asyncio.create_subprocess_exec(*args, cwd=TEMP_PATH)
		try:
			async with asyncio.timeout(3200):
				await proc.wait()
		except (T0, T1, T2):
			with tracebacksuppressor:
				force_kill(proc)
			raise
		outputs = []
		tmpl = fn.rsplit("/", 1)[-1].rsplit(".", 1)[0]
		# The cache is littered with arbitrary files, but we can rely on bot.get_file's filename to contain a unique identifier which will always carry over to the output files
		for f2 in os.listdir(TEMP_PATH):
			if f2.startswith(tmpl) and f2.endswith(format):
				outputs.append(f2)
		if not outputs:
			raise ValueError("No output files found.")
		files = [CompatFile(f"{TEMP_PATH}/{f2}", filename=f2.removeprefix(tmpl).lstrip(" _")) for f2 in outputs]
		response = await fut
		response = await self.bot.edit_message(
			response,
			content=italics(ini_md("Uploading output...")),
		)
		await send_with_reply(_channel, _message, files=files)
		await bot.autodelete(response)


class Vectorise(Command):
	name = ["SVG", "Vector", "Vectorize"]
	description = "Applies https://replicate.com/recraft-ai/recraft-vectorize/api to convert a raster image to SVG format."
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
	slash = True
	ephemeral = True

	def __call__(self, bot, _premium, url, **void):
		os.environ["REPLICATE_API_TOKEN"] = AUTH.get("replicate_key")
		import replicate
		output = replicate.run(
			"recraft-ai/recraft-vectorize",
			input=dict(image=url)
		)
		print(output)
		_premium.append(["replicate", "recraft-vectorize", "0.01"])
		desc = _premium.apply()
		fn = temporary_file("svg")
		with open(fn, "wb") as f:
			f.write(output.read())
		if desc:
			desc = "\n-# " + "\n-# ".join(desc.splitlines())
		return cdict(
			content=desc,
			file=CompatFile(fn, filename=replace_ext(url2fn(url), "svg")),
		)


voices = []
openai_voices = """alloy
ash
ballad
coral
echo
fable
nova
onyx
sage
shimmer""".splitlines()
voices.extend(f"openai-{v}" for v in openai_voices)
dectalk_voices = """paul
betty
harry
frank
kit
rita
ursula
dennis
wendy""".splitlines()
voices.extend(f"dectalk-{v}" for v in dectalk_voices)

class TTS(Command):
	description = "Produces synthesised speech from a text input."
	schema = cdict(
		voice=cdict(
			type="enum",
			validation=cdict(
				enum=tuple(voices),
			),
			description="The engine and voice to apply",
			default="dectalk-paul",
			example="openai-coral",
		),
		text=cdict(
			type="string",
			description="The text to render",
			required=True,
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=("opus", "aac", "mp3", "flac", "wav"),
			),
			description="The file format or codec of the output",
			default="opus",
		),
		autoplay=cdict(
			type="bool",
			description="Automatically plays in the current voice channel once generated",
		),
	)
	rate_limit = (10, 15)
	slash = True
	ephemeral = True

	async def __call__(self, bot, _guild, _channel, _user, _perm, _premium, voice, text, format, autoplay, **void):
		if autoplay:
			assert format == "opus", "Only opus format can be played in voice."
			assert bot.audio and "voice" in bot.get_enabled(_channel), "Voice commands must be enabled for autoplay."
			vc_ = await select_voice_channel(_user, _channel, find=False)
			if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
				raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
			vc_fut = csubmit(bot.audio.asubmit(f"AP.join({vc_.id},{_channel.id},{_user.id})"))
		text = await bot.superclean_content(text)
		engine, mode = voice.split("-", 1)
		fi = temporary_file()
		desc = None
		match engine:
			case "openai":
				_premium.require(2)
				oai = get_oai(None, "openai")
				model = "gpt-4o-mini-tts"
				resp = await oai.audio.speech.create(
					model=model,
					voice=mode,
					input=text,
					instructions="Gentle and soothing, but steady voice",
					response_format=format,
					speed=1,
				)
				c = await tcount(text)
				_premium.append(["openai", model, mpf("12.6") / 1000000 * c])
				desc = _premium.apply()
				resp.write_to_file(fi)
				fmt = format
			case "dectalk":
				args = ["say", "-w", fi, "-pre", f"[:name {mode}]", text]
				print(args)
				await asubmit(subprocess.run, args, cwd="misc/dectalk", stdout=subprocess.DEVNULL, shell=True)
				assert os.path.exists(fi), "No output was captured!"
				fmt = "wav"
			case _:
				raise NotImplementedError(engine)
		if False:#fmt == format:
			fo = fi
		else:
			fo = temporary_file(format)
			args = ["ffmpeg", "-v", "error", "-hide_banner", "-vn", "-i", fi, "-af", "volume=2", "-b:a", "128k", "-vbr", "on", fo]
			print(args)
			proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.DEVNULL)
			try:
				async with asyncio.timeout(3200):
					await proc.wait()
			except (T0, T1, T2):
				with tracebacksuppressor:
					force_kill(proc)
				raise
		if autoplay:
			await vc_fut
			b = await read_file_a(fo)
			async with niquests.AsyncSession() as asession:
				resp = await asession.post(
					"https://api.mizabot.xyz/upload",
					data=b,
				)
				url = resp.text
			items = [cdict(
				name=text[:48],
				url=url,
				hidden=True,
			)]
			await bot.audio.asubmit(f"AP.from_guild({_guild.id}).enqueue({json_dumpstr(items)},start={0})")
			return
		return cdict(
			content=desc,
			file=CompatFile(fo, filename=text[:48] + "." + format),
		)