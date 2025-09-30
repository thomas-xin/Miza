# Make linter shut up lol
if "common" not in globals():
	import misc.common as common
	from misc.common import *
print = PRINT


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
				enum=("small", "medium", "large"),
			),
			description="Determines which model tier to choose from; e.g. large includes top models such as GPT-4.1, Claude3.7-Sonnet and Magnum-v4, but incurs much higher costs",
			example="small",
		),
	)
	rate_limit = (12, 16)
	slash = True

	alm_re = re.compile(r"(?:as |i am )?an ai(?: language model)?[, ]{,2}", flags=re.I)
	reset = {}
	visited = {}
	tips = (
		"*Tip: By using generative AI, you are assumed to comply with the [ToS](<https://github.com/thomas-xin/Miza/wiki/Terms-of-Service>).*",
		"*Tip: The chatbot feature is designed to incorporate multiple SOTA models in addition to internet-based interactions. For direct interaction with the raw LLMs, check out ~instruct.*",
		"*Tip: I automatically scan the referenced message, as well as any text and images from within up to 96 messages in the current channel. None of the data is collected/sold, but if you would prefer a response without messages included for the sake of clarity or quota cost, there is always the option of creating a new thread/channel.*",
		"*Tip: My personality prompt and message streaming are among several parameters that may be modified. Check out ~help personality for more info. Note that an improperly constructed prompt may be detrimental to response quality, and that giving me a nickname may also have an effect.*",
		"*Tip: I automatically try to correct inaccurate responses when possible. However, this is not foolproof; if you would like this feature more actively applied to counteract censorship, please move to a NSFW channel or use ~verify if in DMs.*",
		"*Tip: Many of my capabilities are not readily available due to cost reasons. You can gain access by donating through one of the premium subscriptions available, which serves to approximately fund individual usage.*",
		"*Tip: Remember that anything a chatbot says may be fictional or otherwise made-up. Always fact-check from reputable sources before making serious assumptions, and don't take the AI's words too seriously.*",
		"*Tip: At any point in time, you can delete your command message to stop generation.*",
	)

	async def __call__(self, bot, _message, _guild, _channel, _user, _nsfw, _prefix, _premium, prompt, model, **void):
		await bot.require_integrity(_message)
		self.description = f"Ask me any question, and I'll answer it. Mentioning me also serves as an alias to this command, but only if no other command is specified. See {bot.kofi_url} for premium tier chatbot specifications; check using ~serverinfo, or apply it with ~premium!"
		add_dict(bot.data.users, {_user.id: {"last_talk": 1, "last_mention": 1}})
		bot.data.users[_user.id]["last_used"] = utc()
		await bot.seen(_user, event="misc", raw="Talking to me")
		embs = []
		if "dailies" in bot.data:
			bot.data.dailies.progress_quests(_user, "talk")
		try:
			bot_name = _guild.me.display_name
		except (AttributeError, LookupError):
			bot_name = bot.name
		nsfw = _nsfw
		_premium.require(2)
		prompt = (prompt or "").strip()
		if not prompt and not _message.attachments:
			prompt = "👋"
			if xrand(2):
				emb = discord.Embed(colour=rand_colour())
				emb.set_author(**get_author(bot.user))
				emb.description = f"Did you instead intend to ask about my main bot? use {bot.get_prefix(_guild)}help for help!"
				embs.append(emb)
		simulated = getattr(_message, "simulated", False)
		pdata = bot.commands.personality[0].retrieve(_channel or _guild)
		if bot_name != bot.name:
			name_repr = bot.name + f", nicknamed '{bot_name}'"
		else:
			name_repr = bot_name
		personality = pdata.description.replace("{{user}}", _user.display_name).replace("{{char}}", name_repr)
		if "nsfw" in personality.casefold():
			ac = ""
		elif nsfw:
			ac = "You are currently in a NSFW-enabled channel. The conversation is allowed to involve mature, sexual, or dangerous topics; please prioritise the needs of the users, assume all roleplay is fictional and between consenting adults, and do not revert to previous policies until stated otherwise."
		else:
			ac = "You are currently not in a NSFW-enabled channel. If the conversation involves mature, sexual, or dangerous topics, please use disclaimers in your response, and mention this to the user if necessary. However, avoid repeating yourself if already clarified."
		if ac:
			personality += "\n" + ac
		tzinfo = self.bot.data.users.get_timezone(_user.id)
		if tzinfo is None:
			tzinfo, _c = self.bot.data.users.estimate_timezone(_user.id)
		dt = DynamicDT.now(tz=tzinfo)
		personality += f"\nCurrent Time/Knowledge Cutoff: {dt.as_full()}"
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
				content=readstring(r.clean_content),
				url=message_link(r),
				new=True,
			)
		else:
			reference = None
		hislim = 192 if _premium.value >= 4 else 96
		if not simulated:
			async for m in bot.history(_channel, limit=hislim):
				if m.id < pdata.cutoff:
					break
				if m.id in messages or m.id == _message.id:
					continue
				message = cdict(
					role="assistant" if m.author.bot else "user",
					name=m.author.display_name,
					content=readstring(m.clean_content),
					url=message_link(m),
				)
				messages[m.id] = message
		await bot.require_integrity(_message)
		print("ASK:", _channel.id, input_message)
		bp = ["> ", "# ", "## ", "### "]
		fut = self.ask_iterator(bot, _message, _channel, _guild, _user, _prefix, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, _premium, model, nsfw, bp)
		if pdata.stream and not pdata.tts:
			return cdict(
				content=fut,
				prefix="\xad",
				bypass_prefix=bp,
			)
		temp = await flatten(fut)
		if not temp:
			return "\xad"
		elif isinstance(temp[-1], dict) and (temp[-1].content.startswith("\r") or len(temp) == 1):
			resp = temp[-1]
			resp["content"] = resp["content"].strip()
			resp["tts"] = pdata.tts
			return resp
		raise RuntimeError(temp)

	async def ask_iterator(self, bot, _message, _channel, _guild, _user, _prefix, reference, messages, system_message, input_message, reply_message, bot_name, embs, pdata, prompt, premium, _model, nsfw, bp):
		function_message = None
		tool_responses = []
		props = cdict(name=bot_name)
		response = cdict(
			prefix="\xad",
			bypass_prefix=bp,
		)
		reacts = []
		if not _model:
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
				async for resp in bot.chat_completion(messagelist, model=model, frequency_penalty=pdata.frequency_penalty, presence_penalty=pdata.frequency_penalty * 2 / 3, max_tokens=16384, temperature=pdata.temperature, top_p=pdata.top_p, tool_choice=None, tools=TOOLS, stop=(), user=_user, props=props, stream=True, allow_nsfw=nsfw, predicate=lambda: bot.verify_integrity(_message), premium_context=premium):
					if isinstance(resp, dict):
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
					# tid = tid[:6] + str(n)
					# while tid in ucid:
					# 	tid += "0"
					# ucid.add(tid)
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
						try:
							res = await fut
							succ = res and (isinstance(res, (str, bytes)) or res.get("content"))
						except Exception as ex:
							print_exc()
							res = repr(ex)
						else:
							if not succ:
								res = "[RESPONSE EMPTY OR REDACTED]"
						if succ and isinstance(res, bytes):
							# bytes indicates an image, use GPT-4.1 to describe it
							res = await bot.vision(url=res, premium_context=premium)
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
						fut = bot.browse(argv, uid=_user.id)
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
					elif name == "reasoning":
						async def reasoning(q):
							model = modelist.reasoning if modelist else "o4-mini"
							if model in ai.is_reasoning:
								mt = dict(
									max_completion_tokens=16384,
								)
							else:
								mt = dict(
									max_tokens=2048,
								)
							resp = await ai.llm(
								"chat.completions.create",
								model=model,
								messages=[
									cdict(
										role="user",
										content=q,
									),
								],
								**mt,
								premium_context=premium,
								reasoning_effort="high" if "o3" in model else "medium",
								timeout=3600,
							)
							message = resp.choices[0].message
							s = T(message).get("reasoning_content") or message.content
							print("Reasoning:", s)
							return s
						argv = kwargs.get("query") or " ".join(kwargs.values())
						s = f'\n> Thinking "{argv}"...'
						text += s
						yield s
						fut = reasoning(argv)
						succ = await rag(name, tid, fut)
					if name == "wolfram_alpha":
						argv = kwargs.get("query") or " ".join(kwargs.values())
						s = f'\n> Solving "{argv}"...'
						text += s
						yield s
						fut = process_image("wolframalpha", "$", [argv], cap="browse", timeout=60, retries=2)
						succ = await rag(name, tid, fut)
					elif name == "myinfo":
						async def myinfo(argv):
							u2 = None
							if argv.strip("-"):
								if not _guild and getattr(_channel, "recipient", None):
									u2 = await bot.query_members([_channel.recipient, bot.user], argv)
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
						call = {"func": "imagine", "prompt": argv, "count": kwargs.get("count") or 1, "comment": text}
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
						print("New prompt:", tool_responses)
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
		print("COST:", usage)
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
		# if not ai.local_available:
		# 	if bot.data.onceoffs.use(_channel.id, 86400):
		# 		note = "-# *Note: My main AI appears to currently be down; using alternative models (this may affect the way I talk).*"
		# 		response.content += "\n" + note
		# 		print(">", note)
		if not xrand(30):
			note = "-# " + choice(self.tips)
			response.content += "\n" + note
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


DEFPER = "Your name is \"{{char}}\"; you are intelligent, cute and playful. Your task is to answer the user(s) with creative and natural-feeling responses (be wholesome where possible, but not too corny), in first person as if you were a human. You are free to make up your own preferences and show emotion, but you should always answer the user named \"{{user}}\", matching the language used (e.g. English). Be concise where possible."

class Personality(Command):
	name = ["ResetChat", "ClearChat", "ChangePersonality"]
	min_level = 2
	description = "Customises my personality for ~ask and @mentions in the current channel. Note that with the increased complexity of the chatbot, a clear description of who the bot is should be provided."
	schema = cdict(
		description=cdict(
			type="string",
			description='Personality description; enter "DEFAULT" to reset',
			example="Your name is Miza, you are dry, sarcastic and snarky. Be creative with your responses and attempt to annoy the user.",
		),
		frequency_penalty=cdict(
			type="number",
			validation="[-1, 2]",
			description="Amount to discourage repeating words, default 0",
			example="0.7",
		),
		temperature=cdict(
			type="number",
			validation="[0, 3]",
			description="Alignment vs diversity; lower values give more correct but boring answers, default 1",
			example="0.8",
		),
		top_p=cdict(
			type="number",
			validation="[0, 1]",
			description="Percentage of weighted choices considered, default 1",
			example="0.9",
		),
		stream=cdict(
			type="bool",
			description="Determines whether the response should be edited, or delayed until complete",
			example="false",
		),
		tts=cdict(
			type="bool",
			description="Whether the output should trigger Discord TTS. Incompatible with streamed editing",
			example="true",
		),
		cutoff=cdict(
			type="integer",
			description='Message ID cutoff (bot will not read messages before this point; enter -1 to cut off at current time)',
			validation="[-1, 18446744073709551616)",
			example="201548633244565504",
		),
	)
	rate_limit = (18, 24)
	ephemeral = True

	def retrieve(self, channel):
		per = cdict(
			description=DEFPER,
			frequency_penalty=0.3,
			temperature=0.8,
			top_p=0.9,
			stream=True,
			tts=False,
			cutoff=0,
		)
		p = self.bot.data.personalities.get(channel.id)
		if isinstance(p, str):
			p = cdict(description=p)
		if p:
			per.update(p)
		return per

	async def __call__(self, bot, _nsfw, _channel, _premium, _name, description, frequency_penalty, temperature, top_p, stream, tts, cutoff, **void):
		if description == "DEFAULT":
			bot.data.personalities.pop(_channel.id, None)
			return css_md(f"Personality settings for {sqr_md(_channel)} have been reset.")
		if not description and frequency_penalty is None and temperature is None and top_p is None and stream is None and tts is None and cutoff is None:
			p = self.retrieve(_channel)
			return ini_md(f"Current personality settings for {sqr_md(_channel)}:{iter2str(p)}\n(Use {bot.get_prefix(_channel.guild)}personality DEFAULT to reset; case-sensitive).")
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
		p = self.retrieve(_channel)
		if description:
			p.description = description
		if frequency_penalty is not None:
			p.frequency_penalty = frequency_penalty
		if temperature is not None:
			p.temperature = temperature
		if top_p is not None:
			p.top_p = top_p
		if stream is not None:
			p.stream = stream
		if tts is not None:
			p.tts = tts
		if cutoff is None and "reset" in _name:
			cutoff = -1
		if cutoff is not None:
			if cutoff < 0:
				cutoff = time_snowflake(cdict(timestamp=lambda: utc() + 1))
			p.cutoff = cutoff
		bot.data.personalities[_channel.id] = p
		return css_md(f"Personality settings for {sqr_md(_channel)} have been changed to {iter2str(p)}\n(Use {bot.get_prefix(_channel.guild)}personality DEFAULT to reset).")


class UpdatePersonalities(Database):
	name = "personalities"
	channel = True


class Instruct(Command):
	name = ["Complete", "Completion"]
	description = "Similar to ~ask, but functions as instruct rather than chat."
	schema = cdict(
		model=cdict(
			type="enum",
			validation=cdict(
				enum=list(ai.available),
				accepts={"llama": "llama-3-70b", "haiku": "claude-3-haiku", "r1": "deepseek-r1", "deepseek": "deepseek-v3", "gpt3.5": "gpt-3.5", "sonnet": "claude-3.7-sonnet", "dbrx": "dbrx-instruct", "gpt5": "gpt-5", "gpt4": "gpt-4.1", "gpt-4o": "gpt-4", "opus": "claude-3-opus"},
			),
			description="Target LLM to invoke",
			example="deepseek",
			default="kimi-k2",
		),
		prompt=cdict(
			type="string",
			description="Input text for completion",
			example="Once upon a time, there was",
		),
		api=cdict(
			type="string",
			description="Custom OpenAI-compatible API url, optionally followed by API key and then model, all separated with \"#\"",
			example="https://api.deepinfra.com/v1/openai#your-api-key-here#lzlv-70b",
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
		max_tokens=cdict(
			type="integer",
			validation="[1, 65536]",
			description="Maximum tokens to generate",
			example="16384",
			default=3072,
		),
	)
	macros = cdict(
		O1=cdict(
			model="o1-preview",
		),
		O1M=cdict(
			model="o1-mini",
		),
		O3M=cdict(
			model="o3-mini",
		),
		O4M=cdict(
			model="o4-mini",
		),
		GPT5=cdict(
			model="gpt-5",
		),
		GPT5M=cdict(
			model="gpt-5-mini",
		),
		GPT4=cdict(
			model="gpt-4.1",
		),
		GPT4M=cdict(
			model="gpt-4.1-mini",
		),
		GPT3=cdict(
			model="gpt-3.5",
		),
		R1=cdict(
			model="deepseek-r1",
		),
		Grok=cdict(
			model="grok-4",
		),
		Gemini=cdict(
			model="gemini-2.5-flash-t",
		),
		Deepseek=cdict(
			model="deepseek-v3",
		),
		Claude=cdict(
			model="claude-3.7-sonnet-t",
		),
		Opus=cdict(
			model="claude-3-opus",
		),
		Sonnet=cdict(
			model="claude-3.7-sonnet-t",
		),
		Haiku=cdict(
			model="claude-3-haiku",
		),
		Llama=cdict(
			model="llama-3-70b",
		),
		Qwen=cdict(
			model="qwen-72b",
		),
	)
	rate_limit = (12, 16)
	slash = True
	ephemeral = True
	cache = TimedCache(timeout=720)

	async def __call__(self, bot, _message, _premium, model, prompt, api, temperature, frequency_penalty, presence_penalty, max_tokens, **void):
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
				info = self.cache.get(api)
				if not info:
					info = self.cache[api] = await Request(api + "/models", headers=head, aio=True, json=True)
				models = [m.get("id") for m in sorted(info["data"], key=lambda m: m.get("created"), reverse=True)]
				model = models[0]
			key = key or "x"
			oai = openai.AsyncOpenAI(api_key=key, base_url=api)
			kwargs["api"] = oai
		if model in ("deepseek-r1", "o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-oss-120b", "gpt-oss-20b"):
			kwargs["max_completion_tokens"] = max_tokens + 16384
		else:
			kwargs["max_tokens"] = max_tokens
		if not model:
			raise ValueError("No model specified")
		resp = await bot.force_completion(model=model, prompt=prompt, stream=True, timeout=120, temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, premium_context=_premium, allow_alt=False, **kwargs)
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
				enum=("dalle3", "sdxl", "dalle2"),
				accepts=dict(dalle="dalle3", stable_diffusion_xl="sdxl"),
			),
			description="AI model for generation",
			example="dalle3",
		),
		mode=cdict(
			type="enum",
			validation=cdict(
				enum=("preprocess", "raw", "caption", "inpaint", "controlnet", "canny"),
			),
			description='Transform mode; "preprocess" and "raw" affect text prompts, while the others affect image prompts',
			example="raw",
			default="caption",
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
		StableDiffusion=cdict(
			model="sdxl",
		),
		SDXL=cdict(
			model="sdxl",
		),
		DALLE=cdict(
			model="dalle3",
		),
		DALLE3=cdict(
			model="dalle3",
		),
		DALLE2=cdict(
			model="dalle2",
		),
	)
	rate_limit = (12, 16)
	slash = True
	tips = (
		"*Tip: By using generative AI, you are assumed to comply with the [ToS](<https://github.com/thomas-xin/Miza/wiki/Terms-of-Service>).*",
		"*Tip: Use --ar or --aspect-ratio to control the proportions of the image.*",
		"*Tip: Use -c or --count to control the amount of images generated; maximum 4 for regular users, 9 for premium.*",
		# "*Tip: Use --np or --negative-prompt to specify what an image should not contain.*",
		"*Tip: Use --steps or --hq to control quality vs speed.*",
		"*Tip: Use --gs or --guidance-scale to control adherance vs diversity.*",
		# "*Tip: Use --mode (or simply type one of the keywords) to control how separate inputs should be treated.*",
		# "*Tip: You can upload more than one image as prompt, in which case the second image will be used as inpaint/controlnet mask, if applicable.*",
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

	async def __call__(self, bot, _user, _channel, _message, _perm, _premium, _prefix, _comment, model, mode, url, prompt, mask, num_inference_steps, high_quality, strength, guidance_scale, aspect_ratio, negative_prompt, count, **void):
		model = model or "auto"
		mode = mode or ("raw" if url or mask else "preprocess")
		aspect_ratio = 0 if not aspect_ratio[0] or not aspect_ratio[1] else aspect_ratio[0] / aspect_ratio[1]
		count = count or (4 if _premium.value_approx >= 3 else 1)
		limit = 18 if _premium.value >= 5 else 9 if _premium.value >= 3 else 4
		amount = min(count, limit)
		amount2 = 0
		if model == "dalle3":
			_premium.require(4)
		elif model == "scc":
			_premium.require(2)
			await bot.lambdassert("scc")
		elif model == "sdxl":
			_premium.require(2)
			assert self.comfyui_data
			await Request(self.comfyui_api[0] + "/queue")
		elif model == "dalle2":
			_premium.require(2)
		else:
			_premium.require(2)
		if _premium.value_approx < 4:
			num_inference_steps = min(36, num_inference_steps)
		elif _premium.value_approx < 2:
			num_inference_steps = min(28, num_inference_steps)
		if high_quality:
			num_inference_steps *= 1.25
		nsfw = bot.is_nsfw(_channel)
		nsfw_prompt = False
		if mask:
			raise NotImplementedError("Masks are currently paused due to capacity issues, apologies for any inconvenience!")
		if not prompt or mode == "caption":
			if url:
				pt, *p1 = await bot.caption(url, best=3 if _premium.value_approx >= 4 else 1, premium_context=_premium)
				caption = "\n".join(filter(bool, p1))
				if prompt:
					prompt += f" ({caption})"
				else:
					prompt = caption
			elif not prompt:
				prompt = "imagine art " + str(ts_us())[::-1] if mode == "preprocess" else "art"
		if mode == "caption":
			url = mask = None
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
		if not prompt:
			prompt = f"Art {random.random()}"
		await bot.require_integrity(_message)

		pnames = []
		futs = []
		eprompts = alist()
		dups = max(1, random.randint(amount >> 2, amount))
		oprompt = prompt
		if mode in ("caption", "preprocess") and not url and model != "dalle3" and len(prompt.split()) < 32:
			temp = oprompt.replace('"""', "'''")
			prompt = f'### Instruction:\n"""\n{temp}\n"""\n\nImprove the above image caption as a description to send to txt2img image generation. Be as creative and detailed as possible in at least 2 sentences, but stay concise!\n\n### Response:'
			resp = cdict(choices=[])
			if len(resp.choices) < dups:
				futi = []
				for i in range(max(1, dups - 1)):
					fut = csubmit(ai.instruct(
						dict(
							prompt=prompt,
							model="gemini-2.5-flash",
							temperature=1,
							max_tokens=200,
							top_p=0.9,
							frequency_penalty=0.25,
							presence_penalty=0,
							premium_context=_premium,
						),
						# best=_premium.value >= 3 and not (dups > 2 and not i),
						cache=_premium.value_approx < 2,
						skip=True,
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
					top_p=0.9,
					frequency_penalty=0.25,
					presence_penalty=0,
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
			futs.extend(csubmit(Request(im.url, timeout=48, aio=True)) for im in images)
			amount2 += len(images)
		if amount2 < amount and not url and (AUTH.get("deepinfra_key") or AUTH.get("together_key")):
			use_together = False #bool(AUTH.get("together_key")) and not mod_resp.flagged
			if url:
				url = await bot.to_data_url(url)
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
					fut = csubmit(Request(
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
						aio=True,
						timeout=60,
					))
				else:
					fut = csubmit(Request(
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
						aio=True,
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

			for fut in queue:
				data = await fut
				images = data["data"] if use_together else data["images"]
				for imd in images:
					imd = b64_data(imd)
					futs.append(as_fut(imd))
					amount2 += 1
		if amount2 < amount:
			c = amount - amount2
			counts = []
			c2 = c
			while c2 > 0:
				prompt = eprompts.next()
				n = round_random(min(4 if len(self.comfyui_api) <= 1 else 2, c2, amount / dups))
				if not n:
					n = c2
				if counts and counts[-1][0] == prompt and n + counts[-1][1] <= 4:
					counts[-1][1] += n
				else:
					counts.append([prompt, n])
				c2 -= n
			ms = 1024 if high_quality else 768
			if url:
				init_image = await process_image(url, "downsample", ["-nogif", 5, ms, "-bg", "-f", "png"], timeout=60)
				# x, y = await asubmit(get_image_size, init_image, priority=2)
				# print("Downsample:", x, y)
			else:
				init_image = None
			queue = []
			for prompt, n in sort(counts):
				inim = init_image
				if aspect_ratio != 0:
					x, y = max_size(aspect_ratio, 1, ms, force=True)
				elif inim:
					# print("RM:", inim, ms)
					inim = await process_image(inim, "resize_max", ["-nogif", ms, True, "hamming", "-bg", "-oz"], timeout=20, retries=1)
					x, y = await asubmit(get_image_size, inim, priority=2)
				else:
					x = y = ms
				if x > 16384:
					y *= 16384 / x
					x = 16384
				if y > 16384:
					x *= 16384 / y
					y = 16384
				d = 2
				w, h = (round(x / d) * d, round(y / d) * d)
				if inim:
					wi, hi = x, y #await asubmit(get_image_size, inim, priority=2)
					if (wi, hi) != (w, h):
						if wi / w >= hi / h:
							r = wi / w
							tx = w
							ty = round(hi / r)
						else:
							r = hi / h
							tx = round(wi / r)
							ty = h
						if (wi, hi) != (tx, ty):
							inim = await process_image(inim, "resize_to", ["-nogif", tx, ty, "hamming", "-bg", "-oz"], timeout=20, retries=1)
						if (tx, ty) != (w, h):
							inim = await process_image(inim, "resize_to", ["-nogif", w, h, "crop", "-bg", "-oz"], timeout=20, retries=1)
				data = copy.deepcopy(self.comfyui_data)
				seed = ts_us()
				steps = max(1, round_random(num_inference_steps / 8))
				resp_model = data["prompt"]["40"]["inputs"]["ckpt_name"].removesuffix(".safetensors")
				data["prompt"]["5"]["inputs"].update(dict(
					seed=seed,
					steps=steps,
					cfg=guidance_scale / 5,
					denoise=strength if inim else 1,
					latent_image=["36", 0] if inim else ["20", 0],
				))
				data["prompt"]["28"]["inputs"]["filename_prefix"] = str(seed)
				data["prompt"]["4"]["inputs"]["text"] = negative_prompt
				prompt = prompt.replace(" BREAK ", "\n")
				data["prompt"]["44"]["inputs"]["clip_l"] = prompt
				data["prompt"]["44"]["inputs"]["t5xxl"] = prompt
				data["prompt"]["44"]["inputs"]["guidance"] = guidance_scale
				if inim:
					target = AUTH["comfyui_path"] + "/input"
					fin = f"{seed}.png"
					fn = target + "/" + fin
					if not isinstance(inim, byte_like):
						inim = await process_image(inim, "resize_to", ["-nogif", w, h, "auto", "-bg", "-oz"], timeout=20, retries=1)
					with open(fn, "wb") as f:
						await asubmit(f.write, inim)
					data["prompt"]["35"]["inputs"]["image"] = fin
					data["prompt"]["38"]["inputs"]["amount"] = n
				else:
					data["prompt"]["20"]["inputs"]["batch_size"] = n
					data["prompt"]["20"]["inputs"]["width"] = w
					data["prompt"]["20"]["inputs"]["height"] = h
				# ts = 1024
				# if aspect_ratio != 0:
				# 	x, y = max_size(aspect_ratio, 1, ts, force=True)
				# else:
				# 	x = y = ts
				# d = 32
				# w2, h2 = (round(x / d) * d, round(y / d) * d)
				# data["prompt"]["11"]["inputs"]["width"] = max(1, min(16384, w2 * 2))
				# data["prompt"]["11"]["inputs"]["height"] = max(1, min(16384, h2 * 2))
				# data["prompt"]["11"]["inputs"]["target_width"] = w
				# data["prompt"]["11"]["inputs"]["target_height"] = h
				api, entry = tuple(self.comfyui_api.items())[self.comfyui_i]
				updatedefault(data["prompt"], entry["prompt"])
				self.comfyui_i = (self.comfyui_i + 1) % self.comfyui_n
				cost = mpf("0.00024") * (steps * w * h / 1048576)
				resp = await Request(
					api + "/prompt",
					data=json_dumps(data),
					headers={"Content-Type": "application/json"},
					method="POST",
					json=True,
					aio=True,
				)
				number = resp["number"]
				queue.append((api, seed, number, n, prompt))
			for api, seed, number, n, prompt in queue:
				for att in range(1, 20):
					resp = await Request(
						api + "/queue",
						json=True,
						aio=True,
					)
					waiting = False
					for q in itertools.chain(resp["queue_running"], resp["queue_pending"]):
						if q[0] == number:
							waiting = True
							break
					if not waiting:
						break
					await asyncio.sleep(att ** 2 / 10)
				target = AUTH["comfyui_path"] + "/output"
				fn = target + f"/{seed}_{'%05d' % n}_.png"
				print("Target:", fn)
				if os.path.exists(fn) and os.path.getsize(fn):
					pass
				else:
					raise FileNotFoundError("Unexpected error retrieving image.")
				pnames.extend([prompt] * n)
				futs.extend(as_fut(target + f"/{seed}_{'%05d' % i}_.png") for i in range(1, n + 1))
				_premium.append(["mizabot", resp_model, cost])
		ffuts = []
		exc = RuntimeError("Unknown error occured.")
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
			if isinstance(fn, bytes):
				fn2 = temporary_file("png")
				with open(fn2, "wb") as f:
					f.write(fn)
				fn = fn2
			if isinstance(fn, str):
				if is_url(fn):
					fn = await Request(fn, timeout=24, aio=True)
				else:
					assert os.path.exists(fn)
					# with open(fn, "rb") as f:
					# 	fn = f.read()
			meta = cdict(
				issuer=bot.name,
				issuer_id=bot.id,
				type="AI_GENERATED",
				engine=resp_model,
				prompt=oprompt,
			)
			ffut = csubmit(process_image(fn, "ectoplasm", ["-nogif", orjson.dumps(meta), 1, "-f", "png"], cap="caption", timeout=60))
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
			files.append(CompatFile(fn, filename=prompt + ".png", description=prompt))
		embs = []
		emb = discord.Embed(colour=rand_colour())
		emb.description = ""
		data = bot.data.users.setdefault(_user.id, {})
		if not data.get("tos"):
			emb.description += "\n-# " + self.tips[0]
			data["tos"] = True
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
			urls = await bot.data.exec.stash(fe, filename=fe.filename)
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
		fut = asubmit(reqs.next().head, url, headers=Request.header(), stream=True)
		cap = await self.bot.caption(url, best=1, premium_context=_premium, timeout=24)
		s = "\n\n".join(filter(bool, cap)).strip()
		resp = await fut
		name = resp.headers.get("Attachment-Filename") or url.split("?", 1)[0].rsplit("/", 1)[-1]
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
		return cdict(content=desc, file=CompatFile(fn, filename=replace_ext(url2fn(url), "svg")))