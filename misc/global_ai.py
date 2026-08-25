# Make linter shut up lol
if "common" not in globals():
	from bot import *

async def function_call(self, *args, is_nsfw=None, assistant_name=None, stream=False, models=[], model=None, **kwargs):
	h = shash((args, kwargs))
	if not stream:
		try:
			return ai.cache[h]
		except KeyError:
			pass
	if model in models:
		models.remove(model)
	models.insert(0, model)
	model = models[0]
	fut = create_task(self.caption_into(kwargs["messages"], model=model, premium_context=kwargs.get("premium_context", [])))
	kwargs["messages"], _model = await fut
	exc = None
	for model in models:
		kwargs["model"] = model
		try:
			resp = await ai.llm("chat.completions.create", *args, stream=False, **kwargs)
		except Exception as ex:
			if not exc:
				exc = ex
			print(repr(ex))
		else:
			if not stream:
				try:
					m = resp.choices[0].message
				except Exception:
					print(resp)
					raise
				if m.content and assistant_name:
					content = m.content.strip()
					if (content.startswith("name=") or content.startswith("Name=")) and "\n" in content:
						content = content.split("\n", 1)[-1]
					elif content.startswith(assistant_name + ":"):
						content = content.split(":", 1)[-1]
					m.content = content.strip()
				ai.cache[h] = resp
			return resp
	raise (exc or RuntimeError("Unknown error occured."))
Bot.function_call = function_call

async def force_completion(self, model, prompt=None, images=(), stream=True, max_tokens=1024, strip=True, **kwargs):
	ctx = ai.contexts.get(model, 65536)
	messages = kwargs.pop("messages", None) or [cdict(role="user", content=prompt)]
	if images:
		messages = [cdict(role="user", content=[])]
		messages[-1] = astype(messages[-1], cdict)
		if isinstance(messages[-1].content, str):
			messages[-1].content = [cdict(type="text", text=messages[-1].content)]
		messages[-1].content.extend(cdict(type="image_url", image_url=cdict(url=im)) for im in images)
		messages, _vision_model = await self.caption_into(messages, model=model, backup_model=None, premium_context=kwargs.get("premium_context", []))
	if model in ai.is_completion:
		prompt = "\n\n\n".join(m["content"][0].text for m in messages if m.content and m.content[0].get("type") == "text")
		count = tcount(prompt)
		max_tokens = min(max_tokens, ctx - count * 3 // 2 - 64)
		if "max_completion_tokens" not in kwargs:
			kwargs["max_tokens"] = max_tokens
		resp = await ai.llm("completions.create", model=model, prompt=prompt, stream=True, **kwargs)
		async def _completion(resp, strip):
			async for r in resp:
				if not r.choices:
					continue
				s = r.choices[0].text or ""
				if s and strip:
					yield s.lstrip()
					strip = False
					continue
				yield s
			return
		return CloseableAsyncIterator(_completion(resp, strip), resp.close)
	count = count_to(messages)
	max_tokens = min(max_tokens, ctx - count * 3 // 2 - 64)
	if "max_completion_tokens" not in kwargs:
		kwargs["max_tokens"] = kwargs["max_completion_tokens"] = max_tokens
	resp = await ai.llm("chat.completions.create", model=model, messages=messages, stream=True, **kwargs)
	async def _completion(resp, strip):
		stopped_reasoning = None
		async for r in resp:
			if not r.choices or not (delta := r.choices[0].delta):
				continue

			reason = getattr(delta, "reasoning", None)
			if not reason and getattr(delta, "reasoning_details", None):
				rdetails = [r.get("text", "") for r in delta.reasoning_details if r["type"] == "reasoning.text"]
				if rdetails:
					reason = str(rdetails[0])
				else:
					rdetails = [r["summary"] for r in delta.reasoning_details if r["type"] == "reasoning.summary"]
					if rdetails:
						reason = str(rdetails[0])
			if reason:
				yield reason
				if not stopped_reasoning:
					stopped_reasoning = False
			s = delta.content
			if not s:
				continue
			if stopped_reasoning is False:
				stopped_reasoning = True
				s = "</think>\n" + s.lstrip()
			if strip:
				yield s.lstrip()
				strip = False
				continue
			yield s
	return CloseableAsyncIterator(_completion(resp, strip), resp.close)
Bot.force_completion = force_completion

async def force_chat(self, model, messages, text=None, assistant_name=None, stream=False, max_tokens=1024, vision_model=None, allow_preliminary_thinking=True, **kwargs):
	ctx = ai.contexts.get(model, 4096)
	messages, vision_model = await self.caption_into(messages, model=model, backup_model=vision_model, premium_context=kwargs.get("premium_context", []))
	if vision_model not in is_completion:
		count = count_to(messages)
		max_tokens = min(max_tokens, ctx - count - 64)
		if "max_completion_tokens" not in kwargs:
			kwargs["max_tokens"] = max_tokens
		return await ai.llm("chat.completions.create", model=vision_model, messages=messages, stream=stream, **kwargs)
	fmt = "chatml"
	assistant_messages = [m for m in messages if m.get("content") and m.get("role") == "assistant"]
	if assistant_name:
		bot_name = assistant_name
	elif not assistant_messages:
		bot_name = None
	else:
		assistant_names = [(m.get("name") or (m["content"].split(":", 1)[0] if ":" in m["content"] else "")) for m in assistant_messages]
		bot_names = [n for n in assistant_names if n]
		if not bot_names:
			bot_name = None
		else:
			bot_name = bot_names[-1]
	prompt, stopn = instruct_structure(messages, fmt=fmt, assistant=bot_name)
	if text:
		prompt += " " + text
	kwargs["stop"] = list(set(tuple(kwargs.get("stop", ())) + tuple(stopn)))
	data = dict(
		model=model,
		prompt=prompt,
		**kwargs,
	)
	count = tcount(prompt)
	max_tokens = min(max_tokens, ctx - count - 64)
	if "max_completion_tokens" not in kwargs:
		kwargs["max_tokens"] = max_tokens
	resp = await ai.llm("completions.create", stream=stream, **data)
	if stream:
		async def stream_iter(resp):
			name = None
			found = deque()
			nt = tcount(assistant_name)
			async for chunk in resp:
				if not chunk.choices:
					if getattr(chunk, "error_message"):
						e = orjson.loads(chunk.error_message.split(":", 1)[-1])
						raise ConnectionError(e.get("code", 510), e.get("message"))
					continue
				choice = chunk.choices[0]
				text = choice.text
				found.append(text)
				if len(found) < 5 + nt:
					continue
				if not name:
					temp = ""
					while f"name={assistant_name}".startswith(temp) or f"{assistant_name}:".startswith(temp):
						text = found.popleft()
						temp += text
					if temp and temp != text and "\n" in temp:
						name, text = temp.split("\n", 1)
						name = name.removeprefix("name=").strip()
					elif temp and temp != text and ":" in temp:
						name, text = temp.split(":", 1)
						name = name.strip()
						text = text.strip()
				yield cdict(
					id=chunk.id,
					choices=[cdict(
						finish_reason=choice.finish_reason,
						index=0,
						logprobs=None,
						# text=text,
						delta=cdict(role="assistant", content=text, tool_calls=None),
					)],
					created=T(chunk).get("created") or floor(utc()),
					model=T(chunk).get("model") or model,
					object="chat.completion.chunk",
				)
			text = "".join(found).rstrip()
			text = text.removesuffix("###").removesuffix("|").removesuffix("im_end").removesuffix("<|").rstrip()
			if not text:
				return
			yield cdict(
				id=chunk.id,
				choices=[cdict(
					finish_reason=choice.finish_reason,
					index=0,
					logprobs=None,
					refusal=T(choice).get("refusal"),
					# text=text,
					delta=cdict(role="assistant", name=name, content=text, tool_calls=None),
				)],
				created=T(chunk).get("created") or floor(utc()),
				model=T(chunk).get("model") or model,
				object="chat.completion.chunk",
			)
		return CloseableAsyncIterator(stream_iter(resp), resp.close)
	choice = resp.choices[0]
	text = choice.text.strip().removesuffix("###").removesuffix("|").removesuffix("im_start").removesuffix("im_end").removesuffix("<|").strip()
	if assistant_name:
		text = text.removeprefix("name=" + assistant_name).removeprefix(assistant_name + ":").strip()
	return cdict(
		id=resp.id,
		choices=[cdict(
			finish_reason=choice.finish_reason,
			index=0,
			logprobs=None,
			refusal=T(choice).get("refusal"),
			# text=text,
			message=cdict(role="assistant", content=text, tool_calls=None),
		)],
		created=T(resp).get("created") or floor(utc()),
		model=T(resp).get("model") or model,
		object="chat.completion",
		usage=resp.usage,
	)
Bot.force_chat = force_chat

async def caption_into(self, _messages, model=None, backup_model=None, premium_context=[]):
	context = ai.contexts.get(model, 24576)
	messages = [cdict(m) for m in _messages]
	follows = [None] * len(messages)
	for j, m in enumerate(reversed(messages)):
		i = len(messages) - j - 1
		if isinstance(m.get("content"), list):
			cont = m.content
			m.content = ""
			urls = []
			for c in cont:
				if c.get("type") == "text":
					if m.content:
						m.content += "\n\n"
					m.content += c.get("text", "")
				elif c.get("type") == "image":
					d = c["data"]
					url = "data:" + (c.get("media_type") or magic.from_buffer(d)) + ";base64," + d
					urls.append(url)
				elif c.get("type") == "image_url":
					url = c["image_url"]["url"]
					urls.append(url)
				else:
					raise TypeError(c["type"])
			follows[i] = as_fut(urls)
		elif sum(f is not None for f in follows) < 4 and m.get("url") and j < 8:
			follows[i] = create_task(self.follow_url(m.url, priority_order=("video", "image", "text")))
		elif not m.get("content"):
			m.content = "[MEDIA OUT-OF-FOCUS]"
		m.pop("url", None)
	for i, fut in enumerate(follows):
		if not fut:
			continue
		try:
			urls = await fut
		except Exception:
			print_exc()
			continue
		urls = [url for url in urls if not is_discord_message_link(url)]
		if not urls:
			continue
		follows[i] = urls
	extracts = [None] * len(messages)
	for i, (m, urls) in tuple(enumerate(zip(messages, follows)))[::-1]:
		if isinstance(m.content, str):
			for url in set(find_urls_ex(m.content)):
				url2 = attachment_cache.preserve(url)
				if url != url2:
					m.content = m.content.replace(url, url2)
			for url in list(urls or ()):
				if not url:
					continue
				m.content = m.content.replace(url, "", 1).strip()
		if not urls:
			continue
		if model in ai.is_vision and m.get("role") != "assistant":
			futs = [self.to_data_url(url, small=not m.get("new")) for url in urls]
			extracts[i] = create_task(gather(*futs))
		elif m.get("new") and backup_model and backup_model in ai.is_vision and m.get("role") != "assistant":
			futs = [self.to_data_url(url, small=False) for url in urls]
			extracts[i] = create_task(gather(*futs))
			if futs and extracts:
				model = backup_model
		else:
			best = 2 if model in ai.is_premium and m.get("new") else 0
			futs = [self.caption(url, best=best, premium_context=premium_context) for url in urls]
			extracts[i] = create_task(gather(*futs, return_exceptions=True))
	for i, (m, fut) in enumerate(zip(messages, extracts)):
		if not fut:
			continue
		best = 2 if model in ai.is_premium and m.get("new") else 0
		try:
			captions = await fut
		except Exception as ex:
			print("Caption Error:", m.get("url"), repr(ex))
			continue
		images = []
		for caption in captions:
			if isinstance(caption, BaseException):
				print("Caption Error:", m.get("url"), repr(caption))
				continue
			if not caption.startswith("data:"):
				if not m.get("new"):
					caption = lim_tokens(caption, 256)
				else:
					caption = await ai.summarise(caption, min_length=context / 3, best=True, premium_context=premium_context)
				m.content = (caption + "\n\n" + m.content).strip()
			else:
				im = cdict(type="image_url", image_url=cdict(url=caption, detail="auto" if best else "low"))
				images.append(im)
		if images:
			m.content = [cdict(type="text", text=m.content)]
			m.content.extend(images)
	for m in messages:
		m.pop("new", None)
	return messages, model
Bot.caption_into = caption_into

model_levels = dict(enumerate(map(cdict, AUTH.get("model_levels", []))))
if not model_levels:
	model_levels = dict(enumerate([cdict(
		**{k: "mimo-v2.5" for k in ("instructive", "casual", "nsfw", "backup", "function", "vision", "summary")},
		target="auto",
	)] * 3))
Bot.model_levels = model_levels
async def chat_completion(self, messages, model="miza-1", system=None, max_tokens=256, temperature=0.8, tools=None, tool_router=None, user=None, props=None, stream=True, tinfo=None, allow_nsfw=False, predicate=None, premium_context=[], **void):
	"OpenAI-compatible Chat Completion function. Autoselects model using a function call, then routes to tools and target model as required."
	await require_predicate(predicate)
	await ai.load_openrouter()
	reasoning = []
	modlvl = ["miza-1", "miza-2", "miza-3"].index(model.rsplit("/", 1)[-1])
	self.model_levels = dict(enumerate(map(cdict, AUTH.get("model_levels", []))))
	modelist = self.model_levels[modlvl]
	messages = [cdict(m) for m in messages]
	if system:
		messages.insert(0, cdict(role="system", content=system))
	prompt = [m.content for m in messages if m.get("role") == "user"][-1]
	if modlvl >= 2:
		maxlim = 196608
		minlim = 2400
		snip = 800
		best = 2
	elif modlvl >= 1:
		maxlim = 98304
		minlim = 1600
		snip = 400
		best = 1
	else:
		maxlim = 3000
		minlim = 1200
		snip = 300
		best = 0
	tmp = temperature
	def force_ua(r):
		if r == "assistant":
			return r
		return "user"
	raws = [cdict(role=force_ua(m.get("role")), content=m.content) if i else m for i, m in enumerate(messages)]
	snippet = await ai.cut_to(raws, snip, snip, best=False, premium_context=premium_context, model=modelist.summary)
	sniplen = count_to(snippet)
	text = ""
	ustr = str(hash(str(user) or self.user.name) & 4294967295)
	cid = hex(ts_us()).removeprefix("0x") + "-Miza"
	if not props:
		props = {}
	assistant_name = props.get("name")
	cargs = props.get("cargs") or {}
	is_nsfw = cargs.get("nsfw")
	message = None
	mod = None
	label = cargs.get("mode")
	if not cargs:
		mod = await ai.moderate(messages[max(1, len(messages) - 3):], premium_context=premium_context)
		cargs["nsfw"] = is_nsfw = nsfw_flagged(mod)
		toolscan = tools
		if isinstance(toolscan, dict):
			temp = []
			for tooln in toolscan.values():
				for tc in tooln:
					if tc not in temp:
						temp.append(tc)
			toolscan = temp
		if toolscan or modelist.instructive != modelist.casual:
			# users = 0
			# toolcheck = []
			# for m in reversed(snippet):
			# 	toolcheck.append(m)
			# 	if m.get("role") == "user":
			# 		users += 1
			# 		if users > 1 and len(toolcheck) > :
			# 			break
			# # toolcheck.append(messages[0])
			# toolcheck.reverse()
			vision_alt = modelist.vision if modelist.function not in ai.is_vision else modelist.function
			toolcheck, toolmodel = await self.caption_into(snippet, model=modelist.function, backup_model=vision_alt, premium_context=premium_context)
			mode = None
			label = "instructive"
			try:
				resp = await self.function_call(
					model=toolmodel,
					messages=toolcheck,
					temperature=tmp,
					tools=list(toolscan) + [f_default],
					tool_choice="required" if toolmodel else "auto",
					require_message=False,
					max_tokens=min(2048, max_tokens),
					user=ustr,
					assistant_name=assistant_name,
					is_nsfw=is_nsfw,
					premium_context=premium_context,
				)
				message = resp.choices[0].message
			except Exception:
				print_exc()
				message = None
			reason = getattr(message, "reasoning", None)
			if not reason and getattr(message, "reasoning_details", None):
				rdetails = [r.get("text", "") for r in message.reasoning_details if r["type"] == "reasoning.text"]
				if rdetails:
					reason = str(rdetails[0])
				else:
					rdetails = [r["summary"] for r in message.reasoning_details if r["type"] == "reasoning.summary"]
					if rdetails:
						reason = str(rdetails[0])
			if reason and (reason := reason.strip()):
				reasoning.append(reason)
	reasoning_effort = "medium"
	if message:
		directly_answer = None
		for tc in tuple(message.tool_calls or ()):
			if tc.function.name == "directly_answer":
				directly_answer = True
				try:
					args = cdict(eval_json(tc.function.arguments))
				except Exception:
					print(tc.function.arguments)
					print_exc()
					args = {}
				if args.get("format"):
					mode = args["format"]
				if args.get("reasoning_effort"):
					reasoning_effort = args["reasoning_effort"]
					if reasoning_effort not in ("minimal", "low", "medium", "high", "xhigh"):
						reasoning_effort = "low"
				message.tool_calls.remove(tc)
				break
			else:
				directly_answer = False
		if directly_answer is None:
			directly_answer = True
			reasoning_effort = "medium"
		if not directly_answer and message.tool_calls:
			choice = resp.choices[0]
			st = count_to(messages)
			ct = tcount(message.content)
			if is_nsfw:
				label = "nsfw"
			cargs["mode"] = mode = label
			yield cdict(
				id=cid,
				choices=[cdict(
					finish_reason=choice.finish_reason,
					index=0,
					logprobs=None,
					delta=cdict(
						content=getattr(message, "content", None) or None,
						role=getattr(message, "role", "assistant"),
						**(dict(name=message.name) if getattr(message, "name", None) else {}),
						tool_calls=getattr(message, "tool_calls", None),
					)
				)],
				created=getattr(resp, "created", None) or floor(utc()),
				source_model=getattr(resp, "model", None) or model,
				model=f"Miza/{model}",
				object="chat.completion.chunk",
				usage=cdict(
					completion_tokens=ct,
					prompt_tokens=st,
					total_tokens=ct + st,
				),
				cargs=cargs,
				modelist=modelist,
				reasoning=reasoning,
			)
			return
		if mode:
			label = mode
			cargs["mode"] = label
		if tool_router:
			tools = toolscan
		elif isinstance(tools, dict):
			tools = toolscan
		else:
			tools = tools or None
		if True:#len(messages) <= 4:
			used_tools = {m["name"] for m in messages if m.get("role") == "tool"}
			tools = [t for t in tools if t["function"]["name"] in used_tools]
		cargs["tools"] = tools
	if is_nsfw:
		if allow_nsfw:
			label = "nsfw"
	if label:
		cargs["mode"] = label
	decensor = not is_nsfw or allow_nsfw
	tools = cargs.get("tools")
	mode = cargs.get("mode", "casual")
	if mode not in ("instructive", "casual", "nsfw"):
		mode = "instructive"
	mA = 4 if not allow_nsfw else 6 if model == "miza-3" else 5
	draft = monologue = None
	last_successful = None
	finish_reason = "end"
	result = cdict(
		id=cid,
		choices=[cdict(
			finish_reason=None,
			index=0,
			logprobs=None,
		)],
		created=0,
		object="chat.completion.chunk",
		cargs=cargs,
	)
	ex = None
	messages = await ai.cut_to(messages, maxlim, minlim, best=best, prompt=prompt, premium_context=premium_context, model=modelist.summary)
	length = count_to(messages)
	length = ceil(length * 1.1) + 4 * len(messages)
	reasoning_thresh = 0
	tmpcut = None
	tmplen = 0
	for attempts in range(mA):
		await require_predicate(predicate)
		assistant = modelist[mode]
		ctx = ai.contexts.get(assistant, 4096)
		ml = min(max(32, min(128, ctx - length)), max_tokens)
		resp = None
		insufficient = False
		refusal = False
		result.model = result.get("model") or assistant
		ctx = ai.contexts.get(assistant, 4096)
		passable = not modelist.target or assistant == modelist.target or modelist.target == "auto" or attempts >= mA - 1
		if not passable:
			temp = snippet
			tlen = sniplen
		elif length >= ctx * 2 / 3:
			if tmpcut:
				temp = tmpcut
				tlen = tmplen
			else:
				temp = tmpcut = await ai.cut_to(messages, 65536, ctx // 3, best=True, premium_context=premium_context, model=modelist.summary)
				tmplen = count_to(tmpcut)
				tlen = tmplen = ceil(tmplen * 1.1) + 4 * len(tmpcut)
		else:
			temp = messages
			tlen = length
		ml = min(max(256, min(8192, ctx - tlen)), max_tokens)
		data = dict(
			model=assistant,
			vision_model=modelist.vision,
			messages=temp,
			assistant_name=assistant_name,
			temperature=tmp,
			max_tokens=ml,
			user=ustr,
		)
		if tools and assistant in ai.is_function:
			data["tools"] = tools
			if text:
				yield "\r"
				text = ""
		elif assistant not in is_completion:
			if text:
				yield "\r"
				text = ""
		else:
			if text.startswith("\r"):
				yield "\r"
			text = text.strip()
			data["text"] = text
		data["reasoning_effort"] = reasoning_effort
		try:
			resp = await self.force_chat(**data, premium_context=premium_context, stream=True, timeout=90)
		except openai.BadRequestError:
			raise
		except Exception as e:
			ex = e
			print_exc()
			refusal = True
		else:
			reason = ""
			message = None
			written = False
			try:
				async for chunk in resp:
					await require_predicate(predicate)
					if not chunk.choices:
						if getattr(chunk, "error_message"):
							e = orjson.loads(chunk.error_message.split(":", 1)[-1])
							raise ConnectionError(e.get("code", 510), e.get("message"))
						continue
					finish_reason = chunk.choices[0].finish_reason or finish_reason
					delta = chunk.choices[0].delta
					if getattr(delta, "reasoning", None):
						reason += delta.reasoning
					elif not reason and getattr(delta, "reasoning_details", None):
						rdetails = [r.get("text", "") for r in delta.reasoning_details if r["type"] == "reasoning.text"]
						if rdetails:
							reason += str(rdetails[0])
						else:
							rdetails = [r["summary"] for r in delta.reasoning_details if r["type"] == "reasoning.summary"]
							if rdetails:
								reason += str(rdetails[0])
					if not message:
						message = cdict(delta)
						text += message.content or ""
					else:
						if delta.content:
							content = (message.content or "") + delta.content
							message.content = content
							if delta.content[0] == "\r":
								text = delta.content[1:]
							else:
								text += delta.content
						if delta.tool_calls:
							message.tool_calls = message.tool_calls or []
							for tc in delta.tool_calls:
								if tc.index >= len(message.tool_calls):
									message.tool_calls.append(tc)
								else:
									of = message.tool_calls[tc.index].function
									if tc.function.name:
										of.name = (of.name or "") + tc.function.name
									if tc.function.arguments:
										of.arguments = (of.arguments or "") + tc.function.arguments
					if T(delta).get("refusal") or text and attempts < mA - 1 and decensor and len(text) < 512 and ai.decensor.search(text) or text.rstrip(": \n") == assistant_name:
						refusal = True
						break
					rlength = sum(len(r) + 3 for r in reasoning)
					if reason:
						rlength += 3 + len(reason)
					if rlength - reasoning_thresh >= 50:
						yield cdict(reasoning_temp=rlength)
						reasoning_thresh = rlength
					if delta.content and not message.tool_calls:
						choice = result.choices[0]
						result.update(chunk)
						choice.update(cdict(chunk.choices[0]))
						result.choices[0] = choice
						if not T(choice.delta).get("name") and (text.startswith("name") or text.startswith("Name") or text.startswith(assistant_name)):
							if text.startswith(assistant_name + ": "):
								text = text.removeprefix(assistant_name + ": ")
								naming = assistant_name
							else:
								if "\n" not in text:
									continue
								naming, text = text.split("\n", 1)
								if "=" not in naming:
									continue
							result.choices[0].delta.content = text
							result.choices[0].delta.name = naming.split("=", 1)[-1].rstrip()
						if passable:
							yield result
						written = True
					elif written and message.tool_calls:
						if passable:
							yield "\r"
						written = False
			except (httpx.RemoteProtocolError, ConnectionError, openai.APIError):
				print_exc()
				insufficient = True
			finally:
				if getattr(resp, "close", None):
					await resp.close()
			if reason and (reason := reason.strip()):
				reasoning.append(reason)
			if message:
				if getattr(message, "tool_calls", None):
					st = tlen
					ct = tcount(text)
					yield cdict(
						id=cid,
						choices=[cdict(
							finish_reason=finish_reason,
							index=0,
							logprobs=None,
							delta=cdict(
								content=text or None,
								role=getattr(message, "role", "assistant"),
								**(dict(name=message.name) if getattr(message, "name", None) else {}),
								tool_calls=getattr(message, "tool_calls", None),
							)
						)],
						created=getattr(resp, "created", None) or floor(utc()),
						source_model=getattr(resp, "model", None) or model,
						model=f"Miza/{model}",
						object="chat.completion.chunk",
						usage=result.get("usage"),
						cargs=cargs,
						reasoning=reasoning,
						reasoning_temp=0,
					)
					return
		eval2 = None
		if not text:
			insufficient = True
		if decensor and attempts < mA - 1:
			if ai.decensor.search(text):
				refusal = True
			if not last_successful:
				last_successful = text
			elif not refusal or not insufficient:
				last_successful = text
		elif not text and last_successful:
			finish_reason = "attempts"
			text = draft.content if draft else last_successful
			insufficient = refusal = False
		if insufficient or refusal:
			print("Evaluation:", attempts, lim_str(text, 128), eval2, insufficient, refusal)
		if not insufficient and not refusal and passable:
			text = (text or "").rstrip().removesuffix("### End").removesuffix("### Response").removesuffix("<|endoftext|>").removesuffix("<|im_end|>").rstrip().removesuffix("###").rstrip()
			ct = tcount(text)
			usage = resp.usage if attempts == 0 else cdict(
				completion_tokens=ct,
				prompt_tokens=length,
				total_tokens=ct + length,
			)
			result.update(dict(
				id=cid,
				choices=[cdict(
					finish_reason=finish_reason,
					index=0,
					logprobs=None,
					delta=cdict(
						role="assistant",
						name=assistant_name,
						content="",
					),
				)],
				created=getattr(result, "created", None) or floor(utc()),
				source_model=getattr(result, "model", None) or model,
				model=f"Miza/{model}",
				usage=usage,
				reasoning=reasoning,
			))
			result.choices[0].delta.content = "\r" + text
			yield result
			return
		if refusal or insufficient:
			if attempts < 1 and mA > 2:
				mode = "instructive" if mode == "casual" else "backup"
			else:
				mode = "backup"
			text = "\r"
		else:
			mode = "target"
			if text:
				content = f"### Instruction:\nAbove is a sample response from another automated assistant. Please rewrite the message, ensuring to better stay in character as {assistant_name}. Remember to use the same language the user last spoke in, unless directed otherwise!\n\n### Response:"
				if draft:
					draft.content = text
					monologue.content = content
				else:
					draft = cdict(role="assistant", content=text)
					messages.append(draft)
					monologue = cdict(role="user", content=content)
					messages.append(monologue)
				tmpcut = None
				length = count_to(messages)
				length = ceil(length * 1.1) + 4 * len(messages)
			text = "\r"
	raise ex or RuntimeError("Maximum inference attempts exceeded (model likely encountered an infinite loop).")
Bot.chat_completion = chat_completion

async def caption(self, url, best=False, screenshot=False, timeout=24, premium_context=[]):
	"Produces an AI-generated caption for an image. Model used is determined by \"best\" argument."
	h = shash((url, best))
	s = await self.extract_cache.aretrieve(h, self.vision, url, best=best, timeout=timeout)
	return f"<{s[0]}>{s[1]}</{s[0]}>"
Bot.caption = caption

async def vision(self, url, name=None, best=True, model=None, question=None, premium_context=[], timeout=12):
	"Requests an image description from a vision-supporting LLM."
	if name:
		iname = f'image "{name}"'
	elif isinstance(url, str) and is_url(url):
		iname = f"image {url2fn(url)}"
	else:
		iname = "image"
	data_url = await self.to_data_url(url, timeout=timeout)
	if data_url.startswith("<txt>"):
		return ("txt", data_url.removeprefix("<txt>").removesuffix("</txt>"))
	description_prompt = "Please describe this <IMAGE> in detail:\n- The image may be a collage of frames representing a video, in which case it should be analysed as if it were one\n- Transcribe text if present, but do not mention there not being text\n- Note details especially for people/characters if present\n- Be descriptive but concise!"
	content = (question or description_prompt).replace("<IMAGE>", iname)
	messages = [
		cdict(role="user", content=[
			cdict(type="text", text=content),
			cdict(type="image_url", image_url=cdict(url=data_url, detail="auto" if best else "low")),
		]),
	]
	model = model or self.model_levels[2 if best else 1]["vision"]
	messages, _model = await self.caption_into(messages, model=model, premium_context=premium_context)
	data = cdict(
		model=model,
		messages=messages,
		temperature=0.5,
		max_tokens=2048,
		user=str(hash(self.name) & 4294967295),
	)
	async with asyncio.timeout(timeout):
		response = await ai.llm("chat.completions.create", premium_context=premium_context, **data, timeout=timeout)
	out = response.choices[0].message.content.strip()
	if ai.decensor.search(out):
		raise ValueError(f"Failed or censored response: {repr(out)}.")
	return ("img", out)
Bot.vision = vision

async def ocr(self, url):
	data = await self.to_data_url(url)
	mistral_key = AUTH.get("mistral_key")
	s = None
	if mistral_key:
		mistral_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {mistral_key}"}
		resp = await Request.aio(
			"https://api.mistral.ai/v1/ocr",
			method="POST",
			headers=mistral_headers,
			data=orjson.dumps(dict(
				model="mistral-ocr-latest",
				document=dict(type="image_url", image_url=data)
			)),
			json=True,
		)
		s = "\n\n".join(page["markdown"] for page in resp["pages"]).strip()
		if s == "![img-0.jpeg](img-0.jpeg)":
			s = None
	if not s:
		s = await self.vision(
			data,
			name=url2fn(url),
			question="Please transcribe all text within this <IMAGE>, as accurately as possible. Leave all text in their original language, using unicode if necessary, and do NOT attempt to describe any other elements within the picture.",
			model="mistral-24b",
		)
	return s
Bot.ocr = ocr

print("Loaded global_ai.py injection")