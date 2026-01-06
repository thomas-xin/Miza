import asyncio
import base64
from collections import deque
import decimal
from math import ceil, inf
import orjson
import re
from traceback import format_exc, print_exc
from mpmath import mpf
import numpy as np
import openai
assert hasattr(openai, "AsyncOpenAI"), "OpenAI library has incorrect version installed!"
from misc.types import regexp, astype, lim_str, as_str, cdict, round_random, tracebacksuppressor, utc, T, string_like, getattr_chain
from misc.util import AUTH, CACHE_PATH, AutoCache, get_image_size, json_dumpstr, get_encoding, tcount, lim_tokens, shash, split_across
from misc.asyncs import flatten, asubmit, esubmit, csubmit, emptyctx, gather, Semaphore, CloseableAsyncIterator

print("AI:", __name__)


endpoints = cdict(
	openrouter="https://openrouter.ai/api/v1",
	openrouter_="https://openrouter.ai/api/v1",
	openai="https://api.openai.com/v1",
	deepseek="https://api.deepseek.com/v1",
	together="https://api.together.xyz/v1",
	fireworks="https://api.fireworks.ai/inference/v1",
	deepinfra="https://api.deepinfra.com/v1/openai",
	mistral="https://api.mistral.ai/v1/",
	mizabot="https://api.mizabot.xyz/inference/v1",
)

def cast_rp(fp, pp, model=None):
	s = 1
	return ((fp + pp) / 8 + 1) ** (0.125 * s)
# List of language models and their respective providers, as well as pricing per million input/output tokens
available = {
	"deepseek-r1": {
		"deepseek": ("deepseek-reasoner", ("0.41167", "1.64333")),
		"deepinfra": ("deepseek-ai/DeepSeek-R1", ("0.85", "2.5")),
	},
	"deepseek-v3.2": {
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
	},
	"deepseek-v3.1": {
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
		"fireworks": ("accounts/fireworks/models/deepseek-v3-0324", ("0.9", "0.9")),
		"together": ("deepseek-ai/DeepSeek-V3", ("1.25", "1.25")),
		"deepinfra": ("deepseek-ai/DeepSeek-V3", ("0.85", "0.9")),
	},
	"deepseek-v3": {
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
		"fireworks": ("accounts/fireworks/models/deepseek-v3", ("0.9", "0.9")),
		"together": ("deepseek-ai/DeepSeek-V3", ("1.25", "1.25")),
		"deepinfra": ("deepseek-ai/DeepSeek-V3", ("0.85", "0.9")),
	},
	"llama-3-405b": {
		"deepinfra": ("meta-llama/Meta-Llama-3.1-405B-Instruct", ("1.79", "1.79")),
		"fireworks": ("accounts/fireworks/models/llama-v3p1-405b-instruct", ("3", "3")),
		"together": ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", ("5", "5")),
	},
	"llama-3-90b": {
		"deepinfra": ("meta-llama/Llama-3.2-11B-Vision-Instruct", ("0.35", "0.4")),
		"together": ("meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", ("1.2", "1.2")),
	},
	"llama-3-70b": {
		"fireworks": ("accounts/fireworks/models/llama-v3p3-70b-instruct", ("0.9", "0.9")),
		"deepinfra": ("meta-llama/Llama-3.3-70B-Instruct", ("0.23", "0.4")),
		"together": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", ("0.88", "0.88")),
	},
	"llama-3-11b": {
		"deepinfra": ("meta-llama/Llama-3.2-11B-Vision-Instruct", ("0.055", "0.055")),
		"together": ("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", ("0.18", "0.18")),
	},
	"llama-3-8b": {
		"deepinfra": ("meta-llama/Meta-Llama-3.1-8B-Instruct", ("0.03", "0.05")),
		"fireworks": ("accounts/fireworks/models/llama-v3p1-8b-instruct", ("0.2", "0.2")),
		"together": ("meta-llama/Meta-Llama-3.1-8B-Instruct", ("0.2", "0.2")),
	},
	"qwen-72b": {
		"fireworks": ("accounts/fireworks/models/qwen2p5-72b-instruct", ("0.9", "0.9")),
		"deepinfra": ("Qwen/Qwen2.5-72B-Instruct", ("0.35", "0.4")),
		"together": ("Qwen/Qwen2.5-72B-Instruct-Turbo", ("1.2", "1.2")),
	},
	"gpt-oss-120b": {
		"deepinfra": ("openai/gpt-oss-120b", ("0.039", "0.19")),
	},
	"gpt-oss-20b": {
		"deepinfra": ("openai/gpt-oss-20b", ("0.03", "0.14")),
	},
	"o4-mini": {
		"openai": ("o4-mini", ("1.1", "4.4")),
	},
	"o3": {
		"openai": ("o3", ("10", "40")),
	},
	"o3-mini": {
		"openai": ("o3-mini", ("1.1", "4.4")),
	},
	"gpt-5": {
		"openai": ("gpt-5", ("1.25", "10")),
	},
	"gpt-5-mini": {
		"openai": ("gpt-5-mini", ("0.25", "2")),
	},
	"gpt-5-nano": {
		"openai": ("gpt-5-mini", ("0.05", "0.4")),
	},
	"gpt-4.1-mini": {
		"openai": ("gpt-4.1-mini", ("0.4", "1.6")),
	},
	"gpt-4.1": {
		"openai": ("gpt-4.1", ("2", "8")),
	},
	"gpt-4-mini": {
		"openai": ("gpt-4o-mini", ("0.15", "0.6")),
	},
	"gpt-4": {
		"openai": ("gpt-4o", ("2.5", "10")),
	},
	"mistral-24b": {
		"mistral": ("mistral-small-latest", ("0", "0")),
	},
}

# tags: is_completion, is_function, is_vision, is_premium
is_completion = set()
is_reasoning = {
	"grok-4.1-fast",
}
is_function = {
	"grok-4.1-fast",
}
is_vision = {
	"grok-4.1-fast",
}
is_premium = set()
contexts = {}

oai_name = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
api_map = cdict()
api_sems = cdict()
api_blocked = AutoCache(stale=0, timeout=30, desync=0.25)

def get_oai(func, api="openai"):
	if not isinstance(api, str):
		return getattr_chain(api, func)
	base_url = endpoint = endpoints.get(api, api)
	oai = api_map.get(endpoint)
	if not oai:
		if "#" in endpoint:
			base_url, key = endpoint.split("#", 1)
		else:
			kkey = api.split("://", 1)[-1].split("/", 1)[0] + "_key"
			key = AUTH.get(kkey) or "."
		oai = openai.AsyncOpenAI(api_key=key, base_url=base_url)
		api_map[endpoint] = oai
	return getattr_chain(oai, func)

decensor = regexp(r"(?:i am unable to|i.?m unable to|i cannot|i can.?t|i am not able to|i.?m not able to|i will not) (?:fulfil|assist|help with|provide|do|respond|comply|engage|perform|encourage|endorse|continue) |refrain from|disengage from", re.I)

def nsfw_flagged(resp):
	cat = resp.categories
	flagged = ("harassment", "hate", "sexual", "self_harm", "violence_graphic", "illicit_violent")
	found = []
	for flag in flagged:
		if getattr(cat, flag):
			score = getattr(resp.category_scores, flag)
			if score >= 0.5:
				found.append((score, flag + f"({round(score * 100)}%)"))
	if not found:
		# if resp.flagged:
		# 	flag, score = max(dict(resp.category_scores).items(), key=lambda t: t[1])
		# 	return flag + f"({round(score * 100)}%)"
		return
	return max(found)[-1]

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
		if m.get("role"):
			return m.role + "\n" + m.name + "\n" + content
		return m.name + "\n" + content
	if m.get("role"):
		m.role + "\n" + content
	return content

def m_str(m, include_role=True):
	content = m.content and str(m.content)
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	if not m.get("name"):
		if not include_role:
			return m.content or ""
		if m.get("role") and m.role != "user":
			return f"{m.role}: " + content
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
			return name + ": " + content
		if content and ": " in content:
			return content
		return "user: " + content
	return m.name + ": " + content

def chatml(m):
	s, e = "<|im_start|>", "<|im_end|>"
	if not isinstance(m, cdict):
		m = cdict(m)
	content = str(getattr(m, "content", ""))
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	name, role = getattr(m, "name", None), (getattr(m, "role", None) or "user")
	if not name or role != "user":
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
		else:
			return f"{s}{role}\n" + content + e
	return f"{s}{role} name={name}\n\n" + content + e

def llamav3(m):
	s, e = "<|start_header_id|>", "<|eot_id|>"
	if not isinstance(m, cdict):
		m = cdict(m)
	content = str(getattr(m, "content", ""))
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	name, role = getattr(m, "name", None), (getattr(m, "role", None) or "user")
	if not name or role != "user":
		if role == "assistant":
			role = "partner"
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
		else:
			return f"{s}{role}<|end_header_id|>\n" + content + e
	return f"{s}{role} name={name}<|end_header_id|>\n" + content + e

def mistral(m):
	if not isinstance(m, cdict):
		m = cdict(m)
	content = str(getattr(m, "content", ""))
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	name, role = getattr(m, "name", None), (getattr(m, "role", None) or "user")
	if role == "system":
		return f"<s> [INST] <<SYS>>\n{content}\n<</SYS>> [/INST] </s>"
	elif role == "user":
		s, e = "<s> [INST]", "[/INST]"
	else:
		s, e = "", "</s>"
	if not name or role != "user":
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
		else:
			return f"{s}{role}\n" + content + e
	return f"{s}{role} name={name}\n\n" + content + e

def m_name(m):
	if not m.get("name"):
		if m.get("role") and m.role != "user":
			return m.role
		content = m.content
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			return name.removeprefix("name=").strip()
		return "user"
	return m.name

def overview(messages):
	return "\n\n".join(lim_str(m_str(m), 4096) for m in messages if m.content)

def _count_to(messages, model):
	encoding = get_encoding(model)
	tokens_per_message = 4
	num_tokens = 0
	for message in messages:
		num_tokens += tokens_per_message
		for key, value in message.items():
			if value is None or isinstance(value, bool):
				continue
			if isinstance(value, str):
				num_tokens += len(encoding.encode(value, allowed_special=encoding.special_tokens_set))
			elif isinstance(value, dict):
				num_tokens += len(encoding.encode(json_dumpstr(value), allowed_special=encoding.special_tokens_set))
			elif isinstance(value, (tuple, list)):
				if key == "content":
					for part in value:
						if part.get("type") == "text":
							num_tokens += len(encoding.encode(part["text"], allowed_special=encoding.special_tokens_set))
							continue
						if part.get("type") == "image_url":
							if model == "cl100k_im" and part["image_url"].get("detail", "auto") == "low":
								num_tokens += 85
							else:
								b = base64.b64decode(part["image_url"]["url"].encode("ascii").split(b",", 1)[-1])
								num_tokens += np.prod(get_image_size(b)) / 768 + 128
							continue
						raise RuntimeError(f"Unexpected object {json_dumpstr(part)} in message.")
				# else:
				# 	num_tokens += len(encoding.encode(json_dumpstr(value), allowed_special=encoding.special_tokens_set))
	return num_tokens + 3
async def count_to(messages, model="cl100k_im"):
	"""Return the number of tokens used by a list of messages."""
	return await asubmit(_count_to, messages, model, priority=2)

async def cut_to(messages, limit=1024, softlim=384, exclude_first=True, best=False, prompt=None, premium_context=[]):
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
		if c + count > softlim * 0.8 and not m.get("tool_calls") and m.get("role") != "tool":
			break
		mes.append(m)
		count = await count_to(mes)
	basics = [m for m in messages if m.get("role") != "tool" and m.get("content")]
	if basics and (m := basics[-1]) not in mes:
		mes.append(m)
		count = await count_to(mes)
	if softlim >= limit:
		if not mes:
			m = cdict(messages[-1])
			if isinstance(m.content, list):
				m.content = "\n".join(c["text"] for c in m.content if c.get("type") == "text")
			m.content = lim_tokens(m.content, limit - 16, mode="right")
			mes = [m]
		messages = mes[::-1]
		if exclude_first:
			messages.insert(0, sm)
		return messages
	summ = "Summary of chat history (include this if asked to summarise!):\n"
	s = overview(messages[:i + 1] if i > 0 else messages)
	s = s.removeprefix(summ).removeprefix("system:").strip()
	c = await tcount(summ + s)
	c2 = await count_to(messages)
	if c2 <= softlim * 1.2:
		if exclude_first:
			messages.insert(0, sm)
		return messages
	ml = max(1024, round_random(softlim - count))
	if best:
		s2 = await summarise(s, min_length=ml, best=best, prompt=prompt, premium_context=premium_context)
	else:
		s2 = await asubmit(lim_tokens, s, ml, mode="right", priority=2)
	summ += s2
	messages = mes[::-1]
	messages.insert(0, cdict(
		role="system",
		content=summ,
	))
	if exclude_first:
		messages.insert(0, sm)
	return messages

async def summarise(q, min_length=384, max_length=16384, padding=128, best=True, prompt=None, premium_context=[]):
	"Produces an AI-generated summary of input text. Model used is controlled by \"best\" parameter."
	split_length = max_length - padding
	summ_length = min(min_length, split_length - 1)
	q = lim_tokens(q, 1048576)
	c = await tcount(q)
	if c <= min_length:
		return q
	if c <= summ_length:
		q = await _summarise(q, summ_length, best=best, prompt=prompt, premium_context=premium_context)
	splits = await asubmit(split_across, q, lim=split_length, mode="tlen", priority=2)
	futs = []
	for s in splits:
		fut = csubmit(_summarise(s, ceil(2 * summ_length / len(splits)), best=best, prompt=prompt, premium_context=premium_context))
		futs.append(fut)
	outs = await gather(*futs)
	q = "\n\n".join(outs)
	q = lim_tokens(q, summ_length * 2)
	c = await tcount(q)
	if c <= min_length:
		return q
	return await _summarise(q, summ_length, best=best, prompt=prompt, premium_context=premium_context)

cache = CACHE = AutoCache(f"{CACHE_PATH}/ai", stale=86400, timeout=86400 * 14)

class ExtendedOpenAI(openai.AsyncOpenAI):
	__slots__ = ("model", "pricing", "refresh")

	def __init__(self, *args, **kwargs):
		self.model = ""
		self.pricing = (0, 0)
		self.refresh = 0
		super().__init__(*args, **kwargs)

summarisation_model = None
def load_summarisation_model():
	global summarisation_model
	info = AUTH.get("summarisation_model")
	if not info or getattr(summarisation_model, "refresh", 0) > utc():
		return summarisation_model
	api_key = info.get("api_key", "x")
	base_url = info.get("base_url", "x")
	try:
		oai = openai.OpenAI(api_key=api_key, base_url=base_url)
		model = oai.models.list().data[0].id
		pricing = info.get("pricing", [0, 0])
	except Exception:
		return summarisation_model
	summarisation_model = ExtendedOpenAI(api_key=api_key, base_url=base_url)
	print(f"Loaded summarisation model API {base_url} with model {model}.")
	summarisation_model.model = model
	summarisation_model.pricing = tuple(pricing)
	summarisation_model.refresh = utc() + 720
	return summarisation_model

openai_refresh = 0
async def load_openrouter():
	if openai_refresh > utc():
		return

	async def get_openrouter_models():
		oai = get_oai("", "openrouter")
		models = await flatten(oai.models.list())
		models.sort(key=lambda model: "-preview" not in model.id)
		return models

	models = await CACHE.aretrieve("openrouter-models", get_openrouter_models, _force=True)
	count = 0
	for model in models:
		name = model.id.rsplit("/", 1)[-1].rsplit("-preview", 1)[0]
		prompt, completion = str(decimal.Decimal(model.pricing["prompt"]) * 1000000), str(decimal.Decimal(model.pricing["completion"]) * 1000000)
		entry = (model.id, (prompt, completion))
		try:
			available[name]["openrouter"] = entry
		except KeyError:
			available[name] = dict(openrouter=entry)
		contexts[name] = model.context_length
		if float(completion) >= 5:
			is_premium.add(name)
		else:
			is_premium.discard(name)
		if "image" in model.architecture.get("input_modalities", ()):
			is_vision.add(name)
		else:
			is_vision.discard(name)
		if "tools" in model.supported_parameters:
			is_function.add(name)
		else:
			is_function.discard(name)
		if "reasoning" in model.supported_parameters:
			is_reasoning.add(name)
		else:
			is_reasoning.discard(name)
		count += 1
	print(f"Openrouter: Loaded {count} models")
	globals()["openai_refresh"] = utc() + 86400

with tracebacksuppressor:
	fut = esubmit(load_summarisation_model)
	asyncio.run(load_openrouter())
	fut.result()

async def _summarise(s, max_length, best=False, prompt=None, premium_context=[]):
	if len(s) <= max_length:
		return s
	s = lim_tokens(s, 98304, mode="right")
	if best:
		with tracebacksuppressor:
			s2 = s
			if prompt:
				s2 += "\n\n" + prompt
			if prompt:
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive but concise summary of the text above, and make sure to include all information relevant to the following question if available:\n\n"""\n{prompt}\n"""\n\nWrite only the summary, and do not produce an answer if there is none.'
			else:
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive but concise summary of the text above!'
			ml = round_random(max_length)
			c = await tcount(prompt)
			data = dict(prompt=prompt, temperature=0.6, max_tokens=ml, premium_context=premium_context)
			resp = await instruct(data)
			print("Summary:", resp)
			if resp and not decensor.search(resp):
				return resp
	return lim_tokens(s, round_random(max_length * 2 / 3))

async def llm(func, *args, api=None, timeout=120, premium_context=None, require_message=True, allow_alt=True, **kwargs):
	if isinstance(api, str) or not api:
		await load_openrouter()
		if "model" in kwargs:
			apis = available.get(kwargs["model"]) or {api: None}
		else:
			apis = {api: None}
	else:
		apis = {api: None}
	orig_model = model = kwargs.get("model")
	exc = None
	tries = tuple(apis.items())
	kwa = kwargs
	for i, (api, minfo) in enumerate(tries + tries):
		if api is None and minfo is None:
			api = load_summarisation_model()
		if api is None:
			if not allow_alt:
				break
			assert isinstance(minfo, str), minfo
			kwargs["model"] = minfo
			return await llm(func, *args, timeout=timeout, premium_context=premium_context, require_message=require_message, allow_alt=False, **kwargs)
		if not isinstance(api, str):
			sapi = as_str(api.base_url)
		else:
			sapi = api
		if minfo is None and api is summarisation_model:
			model, pricing = orig_model, tuple(summarisation_model.pricing)
		elif minfo is None:
			if not allow_alt:
				break
			print("No pricing schematic found for:", orig_model, sapi, minfo)
			model, pricing = orig_model, (0, 0)
		else:
			model, pricing = minfo
		if (sapi, model) in api_blocked:
			exc = api_blocked[(sapi, model)]
			continue
		sem = emptyctx
		kwa = kwargs.copy()
		body = cdict(kwargs.get("extra_body") or {})
		if orig_model in is_reasoning or minfo is None:
			mt = kwa.pop("max_tokens", 0) or 0
			if not kwa.get("max_completion_tokens"):
				kwa["max_completion_tokens"] = mt * 3 // 2
			kwa.pop("temperature", None)
			kwa.pop("presence_penalty", None)
			kwa.pop("frequency_penalty", None)
			if sapi == "openrouter":
				reasoning = dict(
					effort=kwa.pop("reasoning_effort", "low"),
					summary="detailed",
				)
				reasoning_2 = body.pop("reasoning", None) or kwa.pop("reasoning", None)
				if reasoning_2:
					reasoning.update(reasoning_2)
				body["reasoning"] = reasoning
			elif not kwa.get("reasoning_effort"):
				kwa["reasoning_effort"] = "low"
		elif "reasoning_effort" in kwa:
			kwa.pop("reasoning_effort")
		kwa["model"] = model
		if isinstance(api, str):
			caller = get_oai(func, api=api)
		else:
			caller = getattr_chain(api, func)
		if body:
			kwa["extra_body"] = body
		try:
			was_input = "input" in kwa
			if was_input:
				messages = kwa.pop("input")
				if isinstance(messages, str):
					messages = [cdict(role="user", content=messages)]
				kwa["messages"] = messages
			if "messages" in kwa:
				messages = []
				for m in kwa["messages"]:
					m2 = cdict(m)
					if m.get("name"):
						name = m2.pop("name")
						name2 = name.replace(" ", "-")
						if oai_name.search(name2):
							m2.name = name2
						elif isinstance(m2.content, list):
							m2.content = [cdict(type="text", text=f"name={name}\n\n{c.text}") if c.get("type") == "text" else c for c in m2.content]
						else:
							m2.content = f"name={name}\n\n{m2.content}"
					if model in is_function:
						m = fix_tool(m2)
					else:
						m = untool(m2)
					if not m.get("content"):
						m.content = "."
					messages.append(m)
				tcid = set()
				i = 0
				while i < len(messages):
					try:
						m = cdict(messages[i])
						if m.get("role") == "tool":
							tci = m.get("tool_call_id") or "x" + str(len(tcid))
							if tci in tcid:
								m["tool_call_id"] = tci
								continue
							if not i:
								m.role = "assistant"
								m.name = "tool-" + T(m).get("name", "").rstrip("-")
								m.content = m.content or ""
								continue
							tool_call = cdict(
								id=tci,
								type="function",
								function=cdict(
									name=T(m).get("name") or "tool",
									arguments="{}",
								),
							)
							if not i or messages[i - 1].get("role") != "assistant":
								m2 = cdict(
									role="assistant",
									name=None,
									content="",
									tool_calls=[tool_call],
								)
								messages.insert(i, m2)
								tcid.add(tci)
								continue
							tc = T(messages[i - 1]).get("tool_calls", [])
							tc.append(tool_call)
							messages[i - 1].tool_calls = tc
							tcid.add(tci)
							continue
						if m.get("tool_calls"):
							for tc in m.tool_calls:
								tcid.add(tc.id)
					finally:
						i += 1
				if was_input:
					kwa["input"] = messages if len(messages) > 1 else messages[0].content
				else:
					kwa["messages"] = messages
			if kwa.get("user"):
				kwa["user"] = str(hash(kwa["user"]))
			else:
				kwa.pop("user", None)
			async with sem:
				response = await asyncio.wait_for(caller(*args, timeout=timeout, **kwa), timeout=timeout)
			inputs = (kwa["messages"], kwa.get("tools")) if "messages" in kwa else kwa.get("prompt")
			stream = OpenAIPricingIterator(
				response,
				getattr(response, "close", None),
				premium_context=premium_context,
				api=sapi,
				model=model,
				input=inputs,
				pricing=pricing,
			)
			if response.object == "response" or hasattr(response, "choices"):
				return await stream.pass_item(response)
			return stream
		except Exception as ex:
			if isinstance(ex, ConnectionError) and ex.errno in (401, 403, 404, 429, 502, 503, 504):
				api_blocked[(sapi, model)] = ex
			if not exc:
				exc = ex
			print(str(sapi) + ": " + str(kwa.get("model")) + ":", kwa, format_exc())
			continue
	if not exc:
		print("ERRORED:", model, lim_str(kwa, 16384))
	raise (exc or RuntimeError("Unknown error occured."))

async def instruct(data, prune=True, cache=True, user=None):
	key = shash(str((data.get("prompt") or data.get("messages"), data.get("model", "kimi-k2-t"), data.get("temperature", 0.75), data.get("max_tokens", 256))))
	if cache:
		return await CACHE.aretrieve(key, _instruct, data, prune=prune, user=user)
	return await CACHE._aretrieve(key, _instruct, data, prune=prune, user=user)

async def _instruct(data, user=None, prune=True):
	inputs = dict(
		temperature=0.75,
		max_tokens=4096,
		user=user,
	)
	inputs.update(data)
	model = inputs.get("model")
	if not model:
		if load_summarisation_model():
			api = summarisation_model
			model = api.model
		else:
			api = None
			model = "grok-4.1-fast"
		inputs.update(dict(
			model=model,
			api=api,
		))
	if not inputs.get("reasoning_effort"):
		inputs["reasoning_effort"] = "minimal" if model in is_reasoning else "low"
	if inputs["model"] not in is_completion:
		if "messages" not in inputs:
			prompt = inputs.pop("prompt")
			inputs["messages"] = [cdict(role="user", content=prompt)]
		async with asyncio.timeout(70):
			response = await llm("chat.completions.create", **inputs, timeout=60)
		resp = response.choices[0].message.content
	else:
		async with asyncio.timeout(100):
			response = await llm("completions.create", **inputs, timeout=90)
		resp = response.choices[0].text
	if prune:
		resp = (resp or "").strip()
		resp2 = regexp(r"### (?:Input|Instruction):?").split(resp, 1)[0].strip().split("### Response:", 1)[-1].strip()
		if resp != resp2:
			print("PRUNED:", resp, resp2, sep="::")
			resp = resp2
		resp2 = resp.split("</think>", 1)[-1].strip()
		if resp2:
			resp = resp2.replace("<think>", "").replace("</think>", "").strip()
	return resp.strip()


f_browse = {
	"type": "function", "function": {
		"name": "browse",
		"description": "Searches internet browser, or visits given URL. Avoid using on redundant file URLs from images sent by user(s). Use for knowledge or advice to validate facts and up-to-date information!",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "Who won the 2026 world cup?", "https://youtu.be/dQw4w9WgXcQ", "Weather in San Francisco"',
				},
			},
			"required": ["query"],
}}}
f_reasoning = {
	"type": "function", "function": {
		"name": "reasoning",
		"description": "Requests for a slower, more powerful language model to provide reasoning. Use if you are unsure about, or if a user is pointing out a flaw in your logic. Includes complex programming tasks. Make sure to pass all relevant information!",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step. Use the example above to decode: oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz"',
				},
			},
			"required": ["query"],
}}}
f_wolfram_alpha = {
	"type": "function", "function": {
		"name": "wolfram_alpha",
		"description": "Queries the Wolfram Alpha engine. Only use for mathematics equations to ensure correctness of answers.",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "Real solutions for x^3-6x^2+12", "eigenvalues of {{2,3,-3},{4,2,-4},{4,3,-5}}", "Glaisher–Kinkelin constant to 100 decimal places"',
				},
			},
			"required": ["query"],
}}}
f_sympy = {
	"type": "function", "function": {
		"name": "sympy",
		"description": "Queries the Sympy algebraic library. Faster than Wolfram Alpha. Note that this runs `sympy.parsing.sympy_parser`, NOT a Python REPL",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "factorint(57336415063790604359)", "randint(1, 100)", "limit(diff(-atan(x)),x,-sqrt(-1.01))"',
				},
				"precision": {
					"type": "integer",
					"description": "Float precision, eg. 128"
				},
			},
			"required": ["query"],
}}}
f_myinfo = {
	"type": "function", "function": {
		"name": "myinfo",
		"description": "Retrieves basic information about yourself and your creators/owners (default) or another user and their profile. Only needs to be used once!",
		"parameters": {
			"type": "object", "properties": {
				"user": {
					"type": "string",
					"description": "Username or ID, eg. Miza",
				},
			},
}}}
f_recall = {
	"type": "function", "function": {
		"name": "recall",
		"description": "Recalls previous messages from conversation history.",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": '''Query, eg. "Jack's birthday, age, gender"''',
				},
			},
			"required": ["query"],
}}}
f_txt2img = {
	"type": "function", "function": {
		"name": "txt2img",
		"description": "Generates an image of the input description, only use when asked to draw a picture. Please make it elaborate where possible!",
		"parameters": {
			"type": "object", "properties": {
				"prompt": {
					"type": "string",
					"description": "Prompt, eg. Brilliant view of a futuristic city in an alien world, glowing spaceships, 8k fantasy art",
				},
				"count": {
					"type": "integer",
					"description": "Amount of images to produce.",
				},
			},
			"required": ["prompt"],
}}}
f_img2txt = {
	"type": "function", "function": {
		"name": "img2txt",
		"description": "Answers a question about the user's input image; only use for necessary information. Please be detailed!",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": "Question, eg. What game is the screenshot from? Please transcribe all text.",
				},
			},
			"required": ["query"],
}}}
f_reminder = {
	"type": "function", "function": {
		"name": "reminder",
		"description": "Sets a reminder for the user, only use if asked for.",
		"parameters": {
			"type": "object", "properties": {
				"message": {
					"type": "string",
					"description": "Message, eg. Remember to take your meds!",
				},
				"time": {
					"type": "string",
					"description": "Datetime, eg. 3 days 3.9 seconds after next april 7th",
				},
			},
			"required": ["message", "time"],
}}}
f_play = {
	"type": "function", "function": {
		"name": "play",
		"description": "Searches for a song and plays it, only use if asked to play music.",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": "Name or URL, eg. Rick Astley - Never gonna give you up",
				},
			},
			"required": ["query"],
}}}
f_audio = {
	"type": "function", "function": {
		"name": "audio",
		"description": "Adjusts audio settings for current music player.",
		"parameters": {
			"type": "object", "properties": {
				"mode": {
					"type": "string",
					"enum": ["volume", "reverb", "pitch", "speed", "pan", "bassboost", "compressor", "chorus", "nightcore", "bitrate"],
				},
				"value": {
					"type": "number",
					"description": "New value as percentage, eg. 300",
				},
			},
			"required": ["mode", "value"],
}}}
f_astate = {
	"type": "function", "function": {
		"name": "astate",
		"description": "Adjusts music player state.",
		"parameters": {
			"type": "object", "properties": {
				"mode": {
					"type": "string",
					"enum": ["pause", "loop", "repeat", "shuffle", "quit", "reset"],
				},
				"value": {
					"type": "boolean",
				},
			},
			"required": ["mode", "value"],
}}}
f_askip = {
	"type": "function", "function": {
		"name": "askip",
		"description": "Skips music player songs.",
		"parameters": {
			"type": "object", "properties": {
				"range": {
					"type": "boolean",
					"description": "Python indexing syntax, eg. 0 or 1:6",
				},
			},
			"required": ["range"],
}}}
f_default = {
	"type": "function", "function": {
		"name": "directly_answer",
		"description": "Indicates that you are preparing to draft up a text-only response to the user. Use when no other tools are necessary.",
		"parameters": {
			"type": "object", "properties": {
				"format": {
					"type": "string",
					"enum": ["instructive", "casual"],
					"description": 'The conversation format/tone; "instructive" for academic, knowledge or advice responses, "casual" for banter, roleplay, or very simple questions.',
				},
			},
			"required": ["format"],
}}}

TOOLS = {
	"knowledge_internet": [
		f_browse,
		f_wolfram_alpha,
		f_sympy,
		f_myinfo,
		f_txt2img,
	],
	"image_generation": [
		f_browse,
		f_myinfo,
		f_txt2img,
	],
	"calculator": [
		f_browse,
		f_wolfram_alpha,
		f_sympy,
	],
	"calendar": [
		f_reminder,
	],
	"audio_music": [
		f_play,
		f_audio,
		f_astate,
		f_askip,
	],
	"none": [],
}
TINFO = {
	"calculator": "Use plain text when writing mathematical equations or formulas.",
}

def unimage(message):
	if isinstance(message.content, str):
		return message
	content = "\n\n".join(line["text"] for line in message.content if line["type"] == "text")
	m = cdict(message)
	m.content = content
	return m

def untool(message):
	content = message.content or ""
	if message.get("role") == "tool":
		if isinstance(content, list):
			message.role = "user"
		else:
			message.role = "user"
			name = message.get("name") or "tool"
			content = name + ":\n" + "```\n" + content + "\n```"
	if message.get("tool_calls") is not None:
		tcs = message.pop("tool_calls")
		content += "\n"
		for tc in tcs:
			fn = tc.function
			content += f"\n> Used {fn.name} {fn.arguments}"
	message.content = content.strip() if isinstance(content, string_like) else content
	return message

def fix_tool(message):
	if not message.get("tool_calls") and message.get("role") != "tool":
		return message
	# Somewhat hacky workaround for incompatibility between tool call ID lengths for different LLM providers
	seen = set()
	if T(message).get("tool_call_id"):
		temp_id = message.tool_call_id
		if len(temp_id) > 40:
			temp_id = temp_id[:40]
		while temp_id in seen:
			temp_id = temp_id[:-1] + chr(ord(temp_id[-1]) + 1)
	for i, tc in enumerate(T(message).get("tool_calls", ())):
		if len(tc.id) > 40:
			tc.id = tc.id[:40]
		if tc.id in seen:
			tc.id = tc.id[:-1] + chr(ord(tc.id[-1]) + 1)
		seen.add(tc.id)
	return message


def unimage(message):
	if not message.content or isinstance(message.content, string_like):
		return message
	message = cdict(message)
	content = ""
	for cont in message.content:
		if cont.get("type") == "text":
			if content:
				content += "\n\n"
			content += cont["text"]
	message.content = content
	return message


CL100K_IM = {
	"gpt-oss-120b",
	"gpt-oss-20b",
	"o4-mini",
	"o3",
	"o3-mini",
	"o1",
	"o1-preview",
	"o1-mini",
	"gpt-5.2",
	"gpt-5.1",
	"gpt-5",
	"gpt-5-mini",
	"gpt-5-nano",
	"gpt-4.1",
	"gpt-4.1-mini",
	"chatgpt-4o-latest",
	"gpt-4o",
	"gpt-4o-mini",
	"deepseek-r1",
	"deepseek-v3.2",
	"deepseek-v3.1",
	"deepseek-v3",
	"quill-72b",
	"databricks/dbrx-instruct",
	"meta-llama/Meta-Llama-3-70B-Instruct",
}

class OpenAIPricingIterator(CloseableAsyncIterator):

	def __init__(self, it, close, premium_context, api, model, input="", m_input=0, m_output=0, pricing=None):
		super().__init__(it, close)
		self.premium_context = premium_context or []
		self.applied = False
		self.input = input
		self.output = ""
		self.tokens = [0, 0]
		self.model = model
		self.costs = [utc(), api, model, "0"]
		self.true_cost = None
		self.pricing = pricing or (m_input, m_output)
		self.tokeniser = "cl100k_im" if model in CL100K_IM else "llamav2"
		self.terminated = False

	@property
	def usage(self):
		return cdict(
			prompt_tokens=self.tokens[0],
			completion_tokens=self.tokens[1],
			total_tokens=self.tokens[0] + self.tokens[1],
		)

	def update_cost(self):
		if self.true_cost is not None:
			self.costs[-1] = self.true_cost
		else:
			self.costs[-1] = str((mpf(self.pricing[0]) * self.tokens[0] + mpf(self.pricing[1]) * self.tokens[1]) / 1000000)
		return self.costs

	async def pass_item(self, item):
		if self.terminated:
			return item
		if item and item.choices and item.choices[0]:
			def dump_calls(tcs):
				for i, tc in enumerate(tcs):
					resp = astype(tc, cdict)
					resp.function = astype(tc.function, cdict)
					tcs[i] = resp
				return json_dumpstr(resp)
			choice = item.choices[0]
			if hasattr(choice, "text"):
				s = choice.text
			elif hasattr(choice, "delta"):
				s = choice.delta.content or ""
				if getattr(choice.delta, "tool_calls", None):
					s += "\n" + dump_calls(choice.delta.tool_calls)
			elif hasattr(choice, "message"):
				s = choice.message.content or ""
				if getattr(choice.message, "tool_calls", None):
					s += "\n" + dump_calls(choice.message.tool_calls)
			else:
				s = ""
			if not s:
				return item
			self.output += s
		if not self.tokens[0]:
			if isinstance(self.input, str):
				self.tokens[0] = await tcount(self.input)
			elif isinstance(self.input, tuple):
				ii = await count_to(self.input[0])
				if self.input[1]:
					ti = await tcount(str(self.input[1]))
				else:
					ti = 0
				self.tokens[0] = ii + ti
			else:
				self.tokens[0] = await count_to(self.input)
		self.tokens[1] = await tcount(self.output)
		self.update_cost()
		if not self.applied:
			self.premium_context.append(self.costs)
			self.applied = True
		return item

	async def __aiter__(self):
		async for item in self.it:
			if getattr(item, "usage", None):
				print(item)
				self.tokens[0] = item.usage.prompt_tokens
				self.tokens[1] = item.usage.completion_tokens
				if getattr(item.usage, "is_byok", False):
					self.true_cost = getattr_chain(item.usage, "cost_details.upstream_inference_cost", None)
				else:
					self.true_cost = getattr(item.usage, "cost", None)
				self.update_cost()
				if not self.applied:
					self.premium_context.append(self.costs)
					self.applied = True
				self.terminated = True
			if not item.choices:
				continue
			choice = item.choices[0]
			if getattr(choice, "text", None):
				yield await self.pass_item(item)
				continue
			delta = getattr(choice, "delta", None)
			if not delta:
				continue
			for k in ("tool_calls", "content", "reasoning", "reasoning_details"):
				if getattr(delta, k, None):
					yield await self.pass_item(item)
					break
		print("aiter pricing:", self.tokens, self.costs)

def instruct_structure(messages, exclude_first=True, fmt="alpaca", assistant=None):
	messages = list(map(unimage, messages))
	if fmt == "mistral":
		ins = tuple(map(mistral, messages))
		stops = ["</s>", "[INST", "[/INST"]
		prompt = "\n".join(ins) + "\nassistant"
		prompt += "\n"
	elif fmt == "llamav3":
		ins = tuple(map(llamav3, messages))
		stops = ["<|eot_id|>", "<|end_of_text|>"]
		prompt = "\n".join(ins) + "\n<|start_header_id|>partner<|end_header_id|>"
		prompt += "\n"
	elif fmt == "chatml":
		ins = tuple(map(chatml, messages))
		stops = ["<|im_start|>", "<|im_end|>"]
		prompt = "\n".join(ins) + "\n<|im_start|>assistant"
		prompt += "\n"
	elif fmt == "alpaca":
		ins = tuple(map(m_str, messages))
		stops = ["### Input:", "### Instruction:", "### Response:"]
		if exclude_first:
			prompt = "### Instruction:\n" + ins[0] + "\n\n"
			ins, messages = ins[1:], messages[1:]
		else:
			prompt = ""
		for s, m in zip(ins, messages):
			if m.get("role") == "assistant":
				prompt += "### Response:\n"
			else:
				prompt += "### Input:\n"
			prompt += s + "\n\n"
		prompt += "### Response:"
		prompt += "\n"
		if assistant:
			prompt += f"{assistant}:"
	else:
		raise NotImplementedError(fmt)
	return prompt, stops

async def collect_stream(resp):
	result = cdict(
		id="0",
		choices=[cdict(
			finish_reason=None,
			index=0,
			logprobs=None,
		)],
		created=0,
		model="unknown",
		object="text_completion",
	)
	choice = result.choices[0]
	async for r in resp:
		if isinstance(r, dict):
			result.update(r)
			if not r.choices:
				continue
			c = r.choices[0]
			if getattr(c, "delta", None):
				result["object"] = "chat.completion.chunk"
				try:
					if c.delta.content.startswith("\r"):
						choice.message.content = c.delta.content[1:]
					else:
						choice.message.content += c.delta.content
				except AttributeError:
					choice.message = cdict(c.delta)
					if getattr(choice, "text", None):
						c.delta.content = choice.pop("text") + c.delta.content
				else:
					choice.message.tool_calls = getattr(c.delta, "tool_calls", [])
			elif getattr(c, "text", None):
				try:
					if c.text.startswith("\r"):
						choice.message.content = c.text[1:]
					else:
						choice.message.content += c.text
				except AttributeError:
					try:
						if c.text.startswith("\r"):
							c.text = c.text[1:]
							raise AttributeError
						choice.text += c.text
					except AttributeError:
						choice.text = c.text
		else:
			try:
				if r.startswith("\r"):
					choice.message.content = r[1:]
				else:
					choice.message.content += r
			except AttributeError:
				try:
					if r.startswith("\r"):
						r = r[1:]
						raise AttributeError
					choice.text += r
				except AttributeError:
					choice.text = r
	if getattr(choice, "message", None) and not hasattr(choice.message, "tool_calls"):
		choice.message.tool_calls = []
	result.choices[0] = choice
	return result

async def moderate(text="", image="", input="", premium_context=[]):
	if isinstance(text, (tuple, list)):
		text = instruct_structure(text, fmt="chatml")
	text = lim_tokens(as_str(text), 24576)
	async def moderate_into(text, image):
		if not image:
			input = text
		else:
			input = []
			if text:
				input.append(dict(type="text", text=text))
			if image:
				input.append(dict(type="image_url", image_url=dict(url=image)))
		try:
			resp = await get_oai("moderations.create")(model="omni-moderation-latest", input=input)
		except Exception:
			print_exc()
			resp = cdict(
				id="ERROR",
				model="ERROR",
				results=[
					cdict(
						flagged=False,
						categories={
							"sexual": False,
							"hate": False,
							"harassment": False,
							"self-harm": False,
							"sexual/minors": False,
							"hate/threatening": False,
							"violence/graphic": False,
							"self-harm/intent": False,
							"self-harm/instructions": False,
							"harassment/threatening": False,
							"violence": False,
						},
						category_scores={
							"sexual": 0,
							"hate": 0,
							"harassment": 0,
							"self-harm": 0,
							"sexual/minors": 0,
							"hate/threatening": 0,
							"violence/graphic": 0,
							"self-harm/intent": 0,
							"self-harm/instructions": 0,
							"harassment/threatening": 0,
							"violence": 0,
						},
					)
				]
			)
		else:
			premium_context.append(["openai", resp.model, "0.00001"])
		return resp
	resp = await CACHE.aretrieve("moderate-" + shash(text) + "-" + shash(image), moderate_into, text, image)
	return resp.results[0]