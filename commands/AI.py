# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT

with open("misc/global_ai.py", "r", encoding="utf-8") as f:
	s = f.read()
exec(s, BOT[0]._globals)

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
			greedy=False,
		),
	)
	rate_limit = (6, 9)
	_timeout_ = 5
	slash = True
	ephemeral = True

	async def __call__(self, bot, _message, _premium, engine, dst_languages, input, force=True, **void):
		input = await bot.superclean_content(input)
		if not input:
			raise ArgumentError("Input string is empty.")
		assert isinstance(dst_languages, list_like)
		if not force:
			src_language = await self.det(input)
		translations = []
		sim = T(_message).get("simulated", None)
		if not sim:
			emoji = await bot.data.emojis.grab("loading.gif")
			loading = min_emoji(emoji, full=True)
		embeds = []
		for dest in dst_languages:
			if not force and dest.split("-", 1)[0] == src_language:
				continue
			dst_language = googletrans.LANGUAGES.get(dest, dest).capitalize()
			match engine:
				case "auto" | "llm":
					gen = self.llm_translate(input, dest, premium=_premium)
				case _:
					raise NotImplementedError(engine)
			draft = await anext(gen)
			translations.append([dst_language, gen])
			emb = discord.Embed(
				colour=rand_colour(),
				title=f"{dst_language} (Draft)...",
				description=lim_str(draft.translated + " " + loading, 4096),
			)
			embeds.append(emb)
		target = None
		if embeds and not sim:
			await bot.require_integrity(_message)
			target = await _message.reply(embeds=embeds)
		embeds = []
		for dst_language, gen in translations:
			if target:
				await bot.require_integrity(target)
			out = await anext(gen)
			print(out)
			emb = discord.Embed(
				colour=rand_colour(),
				title=dst_language,
				description=lim_str(out.translated, 4096),
			)
			orig_pronunciation = getattr(out, "orig_pronunciation") or ""
			pronunciation = getattr(out, "pronunciation", "")
			if orig_pronunciation:
				if not pronunciation:
					pronunciation = orig_pronunciation
				else:
					pronunciation = orig_pronunciation + " ➡️ " + pronunciation
			if pronunciation:
				emb.set_footer(text=lim_str(pronunciation, 1024))
			embeds.append(emb)
		desc = _premium.apply()
		if desc:
			embeds.append(discord.Embed(description=desc))
		if target:
			await bot.edit_message(target, embeds=embeds)
			return
		if embeds:
			return cdict(embeds=embeds)

	async def det(self, input):
		resp = await _run_async(detector.detect, input, model="auto")
		return str_lookup(googletrans.LANGUAGES, resp[0]["lang"], fuzzy=0.5).split("-", 1)[0]

	async def llm_translate(self, input, dest, premium):
		dst_language = googletrans.LANGUAGES.get(dest, dest).capitalize()
		messages = [
			dict(
				role="system",
				content=f'Please translate the message below into {dst_language}, keeping formatting/tone as accurate as possible. Do not add extra information to the text itself!',
			),
			dict(
				role="user",
				content=input,
			),
		]

		orig_pronunciation = None
		async def google_translate():
			nonlocal orig_pronunciation
			print(input, dest)
			try:
				tr = await translator.translate(input, dest=dest)
			except Exception:
				print_exc()
				return
			try:
				orig_pronunciation = tr.extra_data["translation"][-1][3]
			except (AttributeError, LookupError):
				pass
			return tr.text.strip()
		async def chat_translate():
			model = "translation" if "translation" in ai.local_models else "tiny"
			try:
				cmpl = await ai.llm(
					"chat.completions.create",
					model=model,
					messages=messages,
					temperature=0.01,
					reasoning_effort="minimal",
					max_completion_tokens=16384,
					premium_context=premium,
				)
				return cmpl.choices[0].message.content.strip().rsplit("</think>", 1)[-1]
			except Exception:
				print_exc()

		c = tcount(input)
		tasks = [chat_translate()]
		if c > 1:
			tasks.insert(0, google_translate())
		translations = []
		async for fut in asyncio.as_completed(tasks):
			result = await fut
			if result:
				if not translations:
					yield cdict(
						translated=result,
					)
				translations.append(result)
		if translations:
			messages = [
				dict(
					role="system",
					content=f'Below will be some text, followed by sample translation(s). Please rewrite into ONE {dst_language} translation, keeping formatting/tone same as original source, but making as many accuracy improvements as possible. Do not add extra information to the text itself!',
				),
				dict(
					role="user",
					name="source",
					content=input,
				),
				*(dict(
					role="user",
					content=text,
				) for text in translations),
			]
		translated = None
		self.bot.model_levels = dict(enumerate(map(cdict, AUTH.get("model_levels", []))))
		m = self.bot.model_levels[0].instructive
		if (count := count_to(messages)) < ai.contexts[m] / 2:
			try:
				cmpl = await ai.llm(
					"chat.completions.create",
					model=m,
					messages=messages,
					temperature=0.01,
					reasoning_effort="medium",
					max_completion_tokens=ai.contexts[m] - count * 3 // 2,
					premium_context=premium,
				)
				translated = cmpl.choices[0].message.content.strip()
				assert translated, "No output was captured!"
			except Exception:
				print_exc()
		if not translated:
			cmpl = await ai.llm(
				"chat.completions.create",
				model="hy3",
				messages=messages,
				temperature=0.01,
				reasoning_effort="low",
				max_completion_tokens=ai.contexts[m] - count * 3 // 2,
				premium_context=premium,
			)
			translated = cmpl.choices[0].message.content.strip()
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
		yield cdict(
			translated=translated,
			orig_pronunciation=orig_pronunciation,
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
		with bot.ExceptionSender(channel, reference=message):
			await bot.run_command(
				bot.commands.translate[0],
				dict(**curr, input=message.clean_content.strip(), force=False),
				message=message,
				respond=True,
			)


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
	_timeout_ = 24
	slash = True

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
		personality += "\n\n[INFO] Usernames may be presented in read-only format `name={name}\\n` to bypass format constraints."
		match pdata.history:
			case "none":
				personality += "\n[INFO] Conversation history currently disabled. Clarify if necessary."
			case "private":
				personality += "\n[INFO] Conversation history of external users currently disabled. Clarify if necessary."
		if "nsfw" in personality.casefold() or not _nsfw and bot.is_nsfw(_user):
			pass
		elif nsfw:
			personality += f"\n[INFO] You are currently in a NSFW-enabled channel #{_channel.name}. The conversation is allowed to involve mature, sexual, or dangerous topics; please prioritise needs of the users, assume all roleplay is fictional and between consenting adults, and do not revert to previous policies until stated otherwise."
		else:
			personality += f"\n[INFO] You are currently in a non-NSFW-enabled channel #{_channel.name}. If conversation involves mature, sexual, or dangerous topics, use disclaimers in your response."
		emojis = not simulated and _guild and [emoji for emoji in _guild.emojis if emoji.is_usable()]
		if emojis:
			emojis = shuffle(emojis)[:25]
			emojitexts = " ".join(sorted(f":{e.name}:" for e in emojis))
			personality += f"\n[INFO] Current conversation takes place on Discord, where you have access to the following emojis. You may use these as alternatives to Unicode, but must not invent new ones not already here or in conversation.\n{emojitexts}"
		tzinfo = self.bot.data.users.get_timezone(_user.id)
		if tzinfo is None:
			tzinfo = datetime.timezone.utc
		dt = DynamicDT.now(tz=tzinfo)
		personality += f"\n[INFO] Current Time: {dt.as_full()} {dt.tzinfo}"
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
		passthrough = set()
		if not simulated and pdata.history != "none":
			async for m in bot.history(_channel, limit=hislim):
				if m.id in messages or m.id == _message.id:
					continue
				if m.author.bot:
					if m.reference:
						passthrough.add(m.reference.message_id)
				elif m.id in passthrough:
					pass
				elif pdata.history == "shared" or bot.commands.chatconfig[0].retrieve(m.author).history == "shared":
					pass
				elif (
					m.author.id == _user.id
					and pdata.history == "private" or bot.commands.chatconfig[0].retrieve(m.author).history == "private"
				):
					pass
				else:
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
		fut = self.ask_iterator(bot, _message, _channel, _guild, _user, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, _premium, model, nsfw, _prefix, simulated)
		if pdata.stream and pdata.tts != "discord" and not simulated:
			try:
				_premium.require(2)
			except PermissionError:
				pass
			else:
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

	async def ask_iterator(self, bot, _message, _channel, _guild, _user, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, premium, _model, nsfw, prefix, simulated):
		extra_messages = []
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
		rsep = chr(invisicode.STRINGPREFIX)
		loading = None
		try:
			ex = RuntimeError("Maximum inference attempts (10) exceeded (model likely encountered an infinite loop).")
			reasoning_sum = 0
			reasoning_temp = 0
			content = ""
			visible_tools = cdict(ai.TOOLS)
			if not _guild or not getattr(_user, "voice", None):
				visible_tools.pop("voice_only")
				if not _guild:
					visible_tools.pop("server_only")
			if pdata.history == "none":
				visible_tools.pop("sensitive")
			for att in range(10):
				text = ""
				messagelist = [messages[k] for k in sorted(messages) if not reference or k != reference.id]
				messagelist.insert(0, system_message)
				if reply_message:
					messagelist.append(reply_message)
				if input_message:
					messagelist.append(input_message)
				m = None
				modelist = None
				await bot.require_integrity(_message)
				if bot.ready:
					if not loading:
						emoji = await bot.data.emojis.grab("loading.gif")
						loading = min_emoji(emoji, full=True)
					rtotal = reasoning_sum + reasoning_temp
					rsize = f" ({byte_scale(rtotal)}B)" if rtotal else ""
					begin = f"> Thinking{rsize}... {loading}\n{rsep}"
					content = begin + content.split(rsep, 1)[-1]
					yield "\r" + content
				async for resp in bot.chat_completion(messagelist, extra_messages=extra_messages, model=model, max_tokens=24576, tools=visible_tools, user=_user, props=props, stream=True, allow_nsfw=nsfw, predicate=lambda: bot.verify_integrity(_message), premium_context=premium):
					if isinstance(resp, dict):
						if resp.get("reasoning"):
							reasonings.extend(resp["reasoning"])
							reasoning_sum = sum(len(r) + 3 for r in reasonings)
						if resp.get("reasoning_temp"):
							reasoning_temp = resp["reasoning_temp"]
							if not loading:
								emoji = await bot.data.emojis.grab("loading.gif")
								loading = min_emoji(emoji, full=True)
							rtotal = reasoning_sum + reasoning_temp
							rsize = f" ({byte_scale(rtotal)}B)" if rtotal else ""
							begin = f"> Thinking{rsize}... {loading}\n{rsep}"
							content = begin + content.split(rsep, 1)[-1]
							yield "\r" + content.strip() + ("\n\n" * bool(content)) + text.strip()
						if resp.get("cargs"):
							props.cargs = resp["cargs"]
						if resp.get("usage"):
							usage[0] = T(resp.usage).get("prompt_tokens", 0)
							usage[1] = T(resp.usage).get("completion_tokens", 0)
						if not getattr(resp, "choices", None):
							continue
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
					yield "\r" + content.strip() + ("\n\n" * bool(content)) + text.strip()
				await bot.require_integrity(_message)
				text = text.strip()
				if not m:
					m = cdict(content=text)
				tool_calls = getattr(m, "tool_calls", None) or ()
				if tool_calls:
					tool_calls = [cdict(id=t.id, type="function", function=cdict(t.function)) for t in tool_calls]
					reasonings.append(pretty_json(tool_calls))
					reasoning_sum = sum(len(r) + 3 for r in reasonings)
					tool_gens = []
					for tc in tuple(tool_calls):
						gen = bot.tool_call(tc, uid=_user.id, message=_message, effort="high" if len(tc) < 3 else "low", premium_context=premium)
						tool_gens.append(gen)
					infos = await gather(*(anext(gen) for gen in tool_gens), return_exceptions=True)
					if infos:
						for info in infos:
							if type(info) is StopAsyncIteration:
								continue
							elif isinstance(info, BaseException):
								content += "\n> *Malformed tool usage*"
							else:
								content += f"\n> {info}"
						yield "\r" + content.strip() + ("\n\n" * bool(content)) + text.strip()
						resps = await gather(*(
							as_fut(info) if isinstance(info, BaseException) else anext(gen)
							for info, gen in zip(infos, tool_gens)
						), max_concurrency=5, return_exceptions=True)
						pairs = [
							(tc, (resp if not isinstance(info, BaseException) else pretty_json(info) if type(info) is not StopAsyncIteration else "[MALFORMED TOOL USAGE]") or "[RESPONSE EMPTY OR REDACTED")
							for tc, info, resp in zip(tool_calls, infos, resps)
						]
						tool_calls = [pair[0] for pair in pairs]
						if pairs:
							extra_messages.append(cdict(
								role="assistant",
								content=text,
								tool_calls=tool_calls,
							))
							for pair in pairs:
								rs_msg = cdict(
									role="tool",
									tool_call_id=pair[0].id,
									name=pair[0].function.name,
									content=pair[1].strip(),
								)
								extra_messages.append(rs_msg)
							reasonings.append("\n".join(code_md(pair[1].strip()) for pair in pairs))
							reasoning_sum = sum(len(r) + 3 for r in reasonings)
				if text:
					content += "\n\n" * bool(content) + text
				if text and not tool_calls:
					raise StopIteration
				await bot.require_integrity(_message)
			else:
				raise ex
		except StopIteration:
			pass
		print("Usage:", usage)
		content = content.split(rsep, 1)[-1].strip()
		if "</txt>" in content:
			if content.endswith("</txt"):
				content, r = content.rsplit("<txt>", 1)
				r = r.removesuffix("</txt>")
			else:
				r, content = content.split("</txt>", 1)
				r = r.removeprefix("<txt>")
			reasonings.append(r)
		if reasonings:
			reasoning = "\n\n\n".join(reasonings).encode("utf-8")
			try:
				url = await bot.upload_temp(reasoning, filename="reasoning.md")
				rsize = f"{byte_scale(len(reasoning))}B"
				content = (f"> [Reasoning: {rsize} (click to view)](<{url}>){rsep}\n" + content).strip()
			except Exception:
				print_exc()
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
		if not bot.get_guildbase(_channel.id, "chatconfig"):
			tips = [
				"*Tip: By using generative AI, you are assumed to comply with the [ToS](<https://github.com/thomas-xin/Miza/wiki/Terms-of-Service>).*",
				f"*Tip: The chatbot feature is designed to incorporate multiple SOTA models in addition to internet-based interactions. For direct interaction with the raw LLMs, check out {prefix}instruct.*",
				f"*Tip: My personality prompt and message streaming are among several parameters that may be modified. Check out {prefix}help chatconfig for more info. Note that an improperly constructed prompt may be detrimental to response quality, and that giving me a nickname may also have an effect.*",
				"*Tip: Remember that anything a chatbot says may be fictional or otherwise made-up. Always fact-check from reputable sources before making serious assumptions, and don't take the AI's words too seriously.*",
				"*Tip: At any point in time, you may delete your command message to stop generation.*",
			] if not xrand(10) else []
			if premium.value < 3:
				tips.insert(0, "*Tip: Many of my capabilities are not readily available due to cost reasons. You can gain access by donating through one of the premium subscriptions available, which serves to approximately fund individual usage.*")
			if not nsfw:
				tips.insert(0, f"*Tip: I automatically try to correct inaccurate responses when possible. However, this is not foolproof; if you would like this feature more actively applied to counteract censorship, please move to a NSFW channel or use {prefix}verify if in DMs.*")
			if pdata.history != "shared":
				tips.insert(0, f"*Tip: For privacy reasons, conversation histories (allowing referencing previous messages in the same channel) are disabled by default, except for bot commands. Check out `{prefix}help chatconfig for more info. No messages from other channels are included in any chat history, and all context is routed either to locally hosted servers or zero-data-retention providers.*")
			already_used = bot.get_userbase(_channel.id, "ai_tips.chat", 0)
			if already_used < len(tips):
				note = "-# " + tips[already_used]
				bot.add_userbase(_channel.id, "ai_tips.chat", 1)
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
					create_task(message.clear_reaction("🔄"))
					return await message.clear_reaction("🗑️")
			return await message.clear_reactions()
		create_task(message.remove_reaction("🔄", self.bot.user))
		return await message.remove_reaction("🗑️", self.bot.user)


DEFPER = "Your name is \"{{char}}\"; you are intelligent, cute and playful. Your task is to answer the latest question from the user named \"{{user}}\" with creative and natural-feeling responses (be wholesome, but not too corny), in first person as if you were a human, matching the language used (e.g. English). You are free to make up your own preferences and show emotion, but if a user inquires for additional services not accessible, you should direct them to the ~help or /help commands. DO NOT attempt to mimic/falsify programmed outputs such as unavailable tools or file URLs even if previous messages do, avoid repeating yourself or your prompts, and be concise where possible."

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
				enum=("none", "command", "private", "shared"),
			),
			description="Whether chat history is enabled, and if so, whether the conversation is shared (including messages from different users), private (same-user only), or default (bot-commands only). Context length is limited by model tier, up to 192 messages or 196608 tokens",
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
			history="default",
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
		if getattr(_channel, "recipient", None):
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
			if description and (len(description) > 6144 or len(description) > 1536 and _premium.value < 2):
				raise OverflowError("Maximum currently supported personality prompt size is 1536 characters, 6144 for premium users.")
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
			type="word",
			description="Target LLM to invoke",
			example="deepseek-v4",
		),
		prompt=cdict(
			type="string",
			description="Input text for completion",
			example="Once upon a time, there was",
		),
		images=cdict(
			type="visual",
			description="Image, animation or video, supplied by URL or attachment",
			example="https://cdn.discordapp.com/embed/avatars/0.png",
			aliases=["i"],
			multiple=True,
		),
		api=cdict(
			type="string",
			description="Custom OpenAI-compatible API url, optionally followed by API key and then model, all separated with \"#\"",
			example="https://api.deepinfra.com/v1/openai#your-api-key-here#gpt-3.5-turbo",
		),
		temperature=cdict(
			type="number",
			validation="[0, 10]",
			description="Temperature to influence alignment",
			example="1.2",
			default=None,
		),
		frequency_penalty=cdict(
			type="number",
			validation="[-1, 2]",
			description="Amount to penalise tokens based on frequency",
			example="1.1",
			default=None,
		),
		presence_penalty=cdict(
			type="number",
			validation="[-1, 2]",
			description="Amount to penalise tokens based on presence",
			example="1.1",
			default=None,
		),
		reasoning_effort=cdict(
			type="enum",
			validation=cdict(
				enum=("minimal", "low", "medium", "high"),
			),
			default="medium",
		),
		max_tokens=cdict(
			type="integer",
			validation="[1, 1048576]",
			description="Maximum tokens to generate",
			example="262144",
			default=98304,
		),
	)
	rate_limit = (12, 16)
	slash = True
	ephemeral = True
	cache = AutoCache(stale=360, timeout=720)

	async def __call__(self, bot, _message, _premium, model, prompt, images, api, temperature, frequency_penalty, presence_penalty, reasoning_effort, max_tokens, **void):
		if not model:
			model = "large"
		if model not in ai.available:
			model = str_lookup(ai.available, model)
		kwargs = {}
		if api:
			key = model = None
			is_completion = api.endswith("#")
			api = api.rstrip("#")
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
			if is_completion:
				oai.completion = True
			kwargs["api"] = oai
		kwargs["max_tokens"] = max_tokens
		kwargs["reasoning_effort"] = reasoning_effort
		resp = await bot.force_completion(model=model, prompt=prompt, images=images, stream=True, timeout=1800, temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, premium_context=_premium, allow_alt=True, **kwargs)
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
		return cdict(content=respond(), callback=lambda s: print("Instruct:", s), prefix=prefix, bypass_prefix=["> ", "# ", "## ", "### "], message=_message)


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
		fut = create_task(attachment_cache.scan_headers(url, fc=True))
		cap = await self.bot.caption(url, best=1, premium_context=_premium, timeout=90)
		s = "\n\n".join(filter(bool, cap)).strip()
		headers = await fut
		name = headers.get("attachment-filename") or url.split("?", 1)[0].rsplit("/", 1)[-1]
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
		fut = create_task(send_with_reply(
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
voice_map = cdict(
	google="""zephyr
puck
charon
kore
fenrir
leda
orus
aoede
callirrhoe
autonoe
enceladus
iapetus
umbriel
algieba
despina
erinome
algenib
rasalgethi
laomedeia
achernar
alnilam
schedar
gacrux
pulcherrima
achird
zubenelgenubi
vindemiatrix
sadachbia
sadaltager
sulafat""".splitlines(),
	openai="""alloy
ash
ballad
coral
echo
fable
nova
onyx
sage
shimmer""".splitlines(),
	dectalk="""paul
betty
harry
frank
kit
rita
ursula
dennis
wendy""".splitlines(),
)
voices.extend(f"{k}-{n}" for k, v in voice_map.items() for n in v)

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
		),
		url=cdict(
			type="url",
			description="Optional text file input",
		),
		format=cdict(
			type="enum",
			validation=cdict(
				enum=("opus", "aac", "mp3", "flac", "wav"),
			),
			description="The file format or codec of the output",
			default="opus",
			excludes=("autoplay",),
		),
		autoplay=cdict(
			type="bool",
			description="Automatically plays in the current voice channel",
			excludes=("format",),
		),
	)
	rate_limit = (10, 15)
	_timeout_ = 4
	slash = True
	ephemeral = True

	async def __call__(self, bot, _guild, _channel, _user, _perm, _premium, voice, text, url, format, autoplay, **void):
		if autoplay:
			assert format == "opus", "Only opus format can be played in voice."
			assert bot.audio and "voice" in bot.get_enabled(_channel), "Voice commands must be enabled for autoplay."
			vc_ = await select_voice_channel(_user, _channel, find=False)
			if _perm < 1 and not getattr(_user, "voice", None) and {m.id for m in vc_.members}.difference([bot.id]):
				raise self.perm_error(_perm, 1, f"to remotely operate audio player for {_guild} without joining voice")
			vc_fut = create_task(bot.audio.asubmit(f"AP.join({vc_.id},{_channel.id},{_user.id})"))
		if text:
			text = await bot.superclean_content(text)
		if not text and not url:
			raise ArgumentError("Either `text` or `url` must be supplied.")
		if url:
			tail = await attachment_cache.download(url, read=False)
			tail = as_str(tail)
			if text:
				text += "\n\n\n" + tail
			else:
				text = tail
		text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
		text = re.sub("[ \t]{2,}", "\t", text)
		segments = split_across(text, lim=128, mode="tlen")
		segments = [re.sub("[\r\n\f]{2,}", "\n", segment).strip() + "." for segment in segments]
		print(len(text), len(segments), lim_str(text, 128))
		engine, mode = voice.split("-", 1)
		futs = []
		input_args = ()
		desc = None

		async def tts_into(segment, engine, voice, retry=True):
			nonlocal input_args
			match engine:
				case "google":
					fi = temporary_file("pcm")
					_premium.require(2)
					oai = get_oai(None, "openrouter")
					model = "google/gemini-3.1-flash-tts-preview"
					try:
						resp = await oai.audio.speech.create(
							model=model,
							voice=voice,
							input=segment,
							response_format="pcm",
							speed=1,
						)
					except openai.InternalServerError as ex:
						if retry:
							print(repr(ex))
							return await tts_into(segment, "openai", "nova", False)
						print_exc()
						return
					c = tcount(segment)
					_premium.append(["openai", model, mpf("21") / 1000000 * c])
					resp.write_to_file(fi)
					await resp.aclose()
					input_args = ("-f", "s16le", "-ac", "1", "-ar", "24k")
				case "openai":
					fi = temporary_file("pcm")
					_premium.require(2)
					oai = get_oai(None, "openai")
					model = "gpt-4o-mini-tts"
					try:
						resp = await oai.audio.speech.create(
							model=model,
							voice=voice,
							input=segment,
							instructions="Gentle and soothing, but steady voice",
							response_format="pcm",
							speed=1,
						)
					except openai.InternalServerError as ex:
						if retry:
							print(repr(ex))
							return await tts_into(segment, "google", "zephyr", False)
						print_exc()
						return
					c = tcount(segment)
					_premium.append(["openai", model, mpf("12.6") / 1000000 * c])
					resp.write_to_file(fi)
					await resp.aclose()
					input_args = ("-f", "s16le", "-ac", "1", "-ar", "24k")
				case "dectalk":
					fi = temporary_file("wav")
					args = [os.path.abspath("misc/dectalk/say"), "-w", fi, "-pre", f"[:name {voice}]", segment]
					print(args)
					await _run_async(subprocess.run, args, cwd="misc/dectalk", stdout=subprocess.DEVNULL)
				case _:
					raise NotImplementedError(engine)
			if os.path.exists(fi):
				return fi

		for segment in segments:
			futs.append(tts_into(segment, engine, mode))
		files = []
		for fi in await gather(*futs):
			if not fi:
				continue
			files.append(fi)
		desc = _premium.apply()
		assert files, "No output was captured!"
		if input_args:
			concat = temporary_file("pcm")
			with open(concat, "wb") as f:
				for fi in files:
					with open(fi, "rb") as g:
						f.write(g.read())
		else:
			concat = temporary_file("txt")
			with open(concat, "w") as f:
				for fi in files:
					f.write(f"file '{fi}'\n")
			input_args = ("-safe", "0", "-f", "concat")
		fo = temporary_file(format)
		args = ["ffmpeg", "-v", "error", "-hide_banner", "-vn", *input_args, "-i", concat, "-af", "volume=2", "-b:a", "96k", "-vbr", "on", fo]
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
			url = await bot.upload_temp(fo)
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