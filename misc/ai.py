import asyncio
import base64
import orjson
import re
import openai
import numpy as np
from collections import deque
from math import ceil, inf
from traceback import format_exc, print_exc
from mpmath import mpf
from misc.types import regexp, astype, lim_str, as_str, cdict, round_random, tracebacksuppressor, utc, T, string_like, getattr_chain
from misc.util import AUTH, CACHE_PATH, AutoCache, get_image_size, json_dumpstr, get_encoding, tcount, lim_tokens, shash, split_across
from misc.asyncs import asubmit, csubmit, emptyctx, gather, Semaphore, CloseableAsyncIterator

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
	"claude-4.1-opus": {
		"openrouter": ("anthropic/claude-opus-4.1", ("15", "75")),
	},
	"claude-4.5-sonnet": {
		"openrouter": ("anthropic/claude-4.5-sonnet", ("3", "15")),
	},
	"claude-4-sonnet": {
		"openrouter": ("anthropic/claude-sonnet-4", ("3", "15")),
	},
	"claude-4.5-haiku": {
		"openrouter": ("anthropic/claude-haiku-4.5", ("1", "5")),
	},
	"deepseek-r1": {
		"openrouter": ("deepseek/deepseek/deepseek-r1-0528:free", ("0", "0")),
		"deepseek": ("deepseek-reasoner", ("0.41167", "1.64333")),
		"deepinfra": ("deepseek-ai/DeepSeek-R1", ("0.85", "2.5")),
	},
	"deepseek-v3.2": {
		"openrouter": ("deepseek/deepseek-v3.2-exp", ("0.27", "0.41")),
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
	},
	"deepseek-v3.1": {
		"openrouter": ("deepseek/deepseek-chat-v3.1", ("0.2", "0.8")),
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
		"fireworks": ("accounts/fireworks/models/deepseek-v3-0324", ("0.9", "0.9")),
		"together": ("deepseek-ai/DeepSeek-V3", ("1.25", "1.25")),
		"deepinfra": ("deepseek-ai/DeepSeek-V3", ("0.85", "0.9")),
	},
	"deepseek-v3": {
		"openrouter": ("deepseek/deepseek-v3.2-exp", ("0.27", "0.41")),
		"deepseek": ("deepseek-chat", ("0.2025", "0.825")),
		"fireworks": ("accounts/fireworks/models/deepseek-v3", ("0.9", "0.9")),
		"together": ("deepseek-ai/DeepSeek-V3", ("1.25", "1.25")),
		"deepinfra": ("deepseek-ai/DeepSeek-V3", ("0.85", "0.9")),
	},
	"minimax-m2": {
		"openrouter": ("minimax/minimax-m2", ("0.15", "0.45")),
	},
	"minimax-m1": {
		"openrouter": ("minimax/minimax-m1", ("0.3", "1.65")),
	},
	"minimax-01": {
		"openrouter": ("minimax/minimax-01", ("0.2", "1.1")),
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
	"command-r-plus": {
		"openrouter": ("cohere/command-r-plus-08-2024", ("2.375", "9.5")),
	},
	"command-r": {
		"openrouter": ("cohere/command-r-08-2024", ("0.1425", "0.57")),
	},
	"magnum-72b": {
		"openrouter": ("anthracite-org/magnum-v4-72b", ("1.875", "2.25")),
	},
	"qwen3-235b": {
		"openrouter": ("qwen/qwen3-vl-235b-a22b-thinking", ("0.3", "1.2")),
	},
	"qwen-72b": {
		"fireworks": ("accounts/fireworks/models/qwen2p5-72b-instruct", ("0.9", "0.9")),
		"deepinfra": ("Qwen/Qwen2.5-72B-Instruct", ("0.35", "0.4")),
		"together": ("Qwen/Qwen2.5-72B-Instruct-Turbo", ("1.2", "1.2")),
	},
	"gemini-3-pro": {
		"openrouter": ("google/gemini-3-pro-preview", ("2", "12")),
	},
	"gemini-2.5-pro": {
		"openrouter": ("google/gemini-2.5-pro", ("1.25", "10")),
	},
	"gemini-2.5-flash-t": {
		"openrouter": ("google/gemini-2.5-flash-preview-09-2025", ("0.3", "2.5")),
	},
	"gemini-2.5-flash": {
		"openrouter": ("google/gemini-2.5-flash-lite-preview-09-2025", ("0.1", "0.4")),
	},
	"grok-4.1-fast": {
		"openrouter": ("x-ai/grok-4.1-fast", ("0.2", "0.5")),
	},
	"grok-4": {
		"openrouter": ("x-ai/grok-4", ("3", "15")),
	},
	"grok-4-fast": {
		"openrouter": ("x-ai/grok-4-fast", ("0.2", "0.5")),
	},
	"gpt-oss-120b": {
		"openrouter": ("openai/gpt-oss-120b", ("0.072", "0.28")),
	},
	"gpt-oss-20b": {
		"openrouter": ("openai/gpt-oss-20b", ("0.05", "0.2")),
		"fireworks": ("accounts/fireworks/models/gpt-oss-20b", ("0.1", "0.1")),
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
	"gpt-5.1": {
		"openrouter": ("openai/gpt-5.1", ("1.25", "10")),
	},
	"gpt-5": {
		"openrouter": ("openai/gpt-5", ("1.25", "10")),
		"openai": ("gpt-5", ("1.25", "10")),
	},
	"gpt-5-mini": {
		"openrouter": ("openai/gpt-5-mini", ("0.25", "2")),
		"openai": ("gpt-5-mini", ("0.25", "2")),
	},
	"gpt-5-nano": {
		"openrouter": ("openai/gpt-5-nano", ("0.05", "4")),
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
		"openrouter": ("openai/gpt-4o", ("2.5", "10")),
	},
	"mistral-24b": {
		"mistral": ("mistral-small-latest", ("0", "0")),
		"openrouter": ("mistralai/mistral-small-3.1-24b-instruct-2503", ("0.1", "0.3")),
		"openrouter_": ("cognitivecomputations/dolphin3.0-mistral-24b:free", ("0", "0")),
	},
	"kimi-k2-t": {
		"deepinfra": ("moonshotai/Kimi-K2-Thinking", ("0.55", "2.5")),
		"openrouter": ("moonshotai/kimi-k2-thinking", ("0.6", "2.5")),
	},
	"kimi-k2": {
		"openrouter": ("moonshotai/kimi-k2-0905:exacto", ("0.6", "2.5")),
	},
}

# tags: is_completion, is_function, is_vision, is_premium
is_completion = {
	"dbrx-instruct",
	"gpt-3.5-turbo-instruct",
}
is_reasoning = {
	"claude-4.1-opus",
	"claude-4.5-sonnet",
	"claude-4.5-haiku",
	"claude-3.7-sonnet:thinking",
	"claude-3.7-sonnet-t",
	"grok-4.1-fast",
	"grok-4",
	"grok-4-fast",
	"grok-3",
	"grok-3-mini",
	"gemini-3-pro",
	"gemini-2.5-pro",
	"gemini-2.5-flash-t",
	"gpt-oss-120b",
	"gpt-oss-20b",
	"gpt-5.1",
	"gpt-5",
	"gpt-5-mini",
	"gpt-5-nano",
	"o4-mini",
	"o3",
	"o3-mini",
	"o1",
	"o1-preview",
	"o1-mini",
	"deepseek-r1",
	"kimi-k2-t",
	"qwen3-235b",
}
is_function = {
	"claude-3.7-sonnet-t",
	"claude-3.7-sonnet",
	"claude-3.5-sonnet",
	"claude-3.5-haiku",
	"claude-3-opus",
	"claude-3-sonnet",
	"claude-3-haiku",
	"command-r",
	"command-r-plus",
	"35b-beta-long",
	"grok-4.1-fast",
	"grok-4",
	"grok-4-fast",
	"grok-3",
	"grok-3-mini",
	"gemini-3-pro",
	"gemini-2.5-pro",
	"gemini-2.5-flash-t",
	"gemini-2.5-flash",
	"gemini-2.0",
	"gpt-oss-120b",
	"gpt-oss-20b",
	"o4-mini",
	"o3",
	"o3-mini",
	"o1",
	"o1-preview",
	"o1-mini",
	"gpt-5.1",
	"gpt-5",
	"gpt-5-mini",
	"gpt-5-nano",
	"gpt-4.1",
	"gpt-4.1-mini",
	"gpt-4",
	"chatgpt-4o-latest",
	"gpt-4-mini",
	"gpt-4o-mini",
	"gpt-4-0125-preview",
	"gpt-3.5",
	"gpt-3.5-turbo",
	"deepseek-v3.2",
	"deepseek-v3.1",
	"mistral-24b",
	"kimi-k2-t",
	"kimi-k2",
	"caller-large",
	"firefunction-v2",
	"firefunction-v1",
}
is_vision = {
	"claude-3.7-sonnet-t",
	"claude-3.7-sonnet",
	"claude-3.5-sonnet",
	"claude-3-opus",
	"claude-3-sonnet",
	"claude-3-haiku",
	"llama-3-11b",
	"llama-3-90b",
	"grok-4.1-fast",
	"grok-4",
	"grok-4-fast",
	"gemini-3-pro",
	"gemini-2.5-pro",
	"gemini-2.5-flash-t",
	"gemini-2.5-flash",
	"gemini-2.0",
	"gpt-oss-120b",
	"gpt-oss-20b",
	"o1",
	"o1-preview",
	"gpt-5.1",
	"gpt-5",
	"gpt-5-mini",
	"gpt-5-nano",
	"gpt-4.1",
	"gpt-4.1-mini",
	"gpt-4",
	"chatgpt-4o-latest",
	"gpt-4-mini",
	"gpt-4o-mini",
	"qwen3-235b",
	"minimax-m2",
	"minimax-m1",
	"minimax-01",
	"mistral-24b",
	"firellava-13b",
	"phi-4b",
}
is_premium = {
	"claude-4.5-opus",
	"claude-4.1-opus",
	"claude-4-opus",
	"claude-4.5-sonnet",
	"claude-4-sonnet",
	"llama-3-405b",
	"gpt-5.1",
	"gpt-5",
	"grok-4",
	"gemini-3-pro",
	"gemini-2.5-pro",
	"o3",
	"o1",
	"o1-preview",
	"command-r-plus",
}
instruct_formats = {
	"reflection-llama-3-70b": "llamav3",
	"euryale-70b": "llamav3",
	"lzlv-70b": "vicuna",
	"skyfall-36b": "mistral",
	"mistral-24b": "mistral",
	"miquliz-120b": "mistral",
	"goliath-120b": "alpaca",
	"command-r": "cohere",
	"command-r-plus": "cohere",
	"command-r-plus-08-2024": "cohere",
	"command-r-plus-h6t2": "cohere",
	"magnum-72b": "chatml",
	"qwen-72b": "chatml",
	"dbrx-instruct": "chatml",
	"mixtral-8x22b-instruct": "mistral",
	"wizard-8x22b": "mistral",
	"llama-3-405b": "llamav3",
	"llama-3-90b": "llamav3",
	"llama-3-70b": "llamav3",
	"llama-3-11b": "llamav3",
	"llama-3-8b": "llamav3",
	"phi-4b": "llamav3",
}
# Default context: 4096
contexts = {
	"claude-3.7-sonnet-t": 200000,
	"claude-3.7-sonnet": 200000,
	"claude-3.5-sonnet": 200000,
	"claude-3.5-haiku": 200000,
	"claude-3-opus": 200000,
	"claude-3-sonnet": 200000,
	"claude-3-haiku": 200000,
	"command-r": 112000,
	"command-r-plus": 112000,
	"35b-beta-long": 14336,
	"magnum-72b": 16384,
	"qwen3-235b": 262144,
	"qwen-72b": 32768,
	"llama-3-8b": 131072,
	"llama-3-11b": 131072,
	"llama-3-70b": 131072,
	"llama-3-90b": 131072,
	"llama-3-405b": 131072,
	"grok-4.1-fast": 2000000,
	"grok-4": 262144,
	"grok-4-fast": 2097152,
	"grok-3": 131072,
	"grok-3-mini": 131072,
	"gemini-3-pro": 1048576,
	"gemini-2.5-pro": 1048576,
	"gemini-2.5-flash-t": 1048576,
	"gemini-2.5-flash": 1048576,
	"gemini-2.0": 1048576,
	"gpt-oss-120b": 131072,
	"gpt-oss-20b": 131072,
	"o4-mini": 200000,
	"o3": 200000,
	"o3-mini": 200000,
	"o1": 200000,
	"o1-preview": 200000,
	"o1-mini": 200000,
	"gpt-5.1": 400000,
	"gpt-5": 400000,
	"gpt-5-mini": 400000,
	"gpt-5-nano": 400000,
	"gpt-4.1": 1048576,
	"gpt-4.1-mini": 1048576,
	"gpt-4": 128000,
	"chatgpt-4o-latest": 128000,
	"gpt-4-mini": 128000,
	"gpt-4o-mini": 128000,
	"gpt-3.5": 16384,
	"gpt-3.5-turbo-instruct": 4096,
	"minimax-m2": 1000000,
	"minimax-m1": 1000000,
	"minimax-01": 1000000,
	"deepseek-r1": 64000,
	"deepseek-v3.2": 163840,
	"deepseek-v3.1": 64000,
	"deepseek-v3": 64000,
	"skyfall-36b": 32768,
	"mistral-24b": 32768,
	"dbrx-instruct": 32768,
	"miquliz-120b": 32768,
	"reflection-llama-3-70b": 8192,
	"euryale-70b": 8192,
	"lzlv-70b": 4096,
	"wizard-8x22b": 65536,
	"mixtral-8x22b-instruct": 65536,
	"nous-hermes-2-mixtral-8x7b-dpo": 32768,
	"mixtral-8x7b-instruct": 32768,
	"mixtral-8x7b": 32768,
	"kimi-k2-t": 262144,
	"kimi-k2": 262144,
	"caller-large": 32768,
	"firefunction-v2": 8192,
	"firefunction-v1": 32768,
	"firellava-13b": 4096,
	"phi-4b": 131072,
	"mythomax-13b": 4096,
}

oai_name = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
api_map = cdict()
api_sems = cdict()
api_blocked = AutoCache(stale=0, timeout=30)

def oai_method(oai, func):
	lookup = func.split(".")
	caller = oai
	for k in lookup:
		caller = getattr(caller, k)
	return caller

def get_oai(func, api="openai"):
	if not isinstance(api, str):
		return oai_method(api, func)
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
	return oai_method(oai, func)

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

def im_sep(mode="im"):
	if mode == "im":
		start = "<|im_start|>"
		end = "<|im_end|>"
	else:
		start = "▀"
		end = "▄"
	return start, end

def chatml(m, mode="im"):
	s, e = im_sep(mode)
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

def cohere(m):
	if not isinstance(m, cdict):
		m = cdict(m)
	content = str(getattr(m, "content", ""))
	if not content or not content.strip():
		temp = deque()
		for fc in m.get("tool_calls", ()):
			temp.append(fc.function.name + " " + as_str(fc.function.arguments))
		content = "\n".join(temp)
	name, role = getattr(m, "name", None), (getattr(m, "role", None) or "user")
	if role == "tool":
		return "<results>\n" + f"Document: 1\ntitle: {m.get('name', 'tool')}\ntext: {m['content']}" + "\n</results>"
	if role == "system":
		return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\n{content}<|END_OF_TURN_TOKEN|>"
	if role == "user":
		s, e = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|END_OF_TURN_TOKEN|>"
	else:
		s, e = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", "<|END_OF_TURN_TOKEN|>"
	if not name or role != "user":
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
		else:
			return f"{s}\n" + content + e
	return f"{s}name={name}\n\n" + content + e

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

def vicuna(m):
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
		return content.strip() + "\n"
	elif role == "assistant":
		s, e = "### ", "</s>"
	else:
		s, e = "### ", ""
	if not name or role != "user":
		if content.startswith("name=") and "\n" in content:
			name, content = content.split("\n", 1)
			name = name.removeprefix("name=").strip()
		else:
			return f"{s}{role}: " + content + e
	return f"{s}{role} name={name}: " + content + e

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

info = AUTH.get("summarisation_model")
if info:
	model = info.get("model", "x")
	api_key = info.get("api_key", "x")
	base_url = info.get("base_url", "x")
	pricing = info.get("pricing", [0, 0])
	summarisation_model = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
	summarisation_model.model = model
	summarisation_model.pricing = pricing
else:
	summarisation_model = None
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
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive but concise summary of the text above, and make sure to include all information relevant to the following question if available:\n\n"""\n{prompt}\n"""\n\nWrite only the summary, not an answer or acknowledgement.'
			else:
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive but concise summary of the text above!'
			ml = round_random(max_length)
			c = await tcount(prompt)
			data = dict(prompt=prompt, temperature=0.8, top_p=0.9, max_tokens=ml, premium_context=premium_context)
			resp = await instruct(data)
			print("Summary:", resp)
			if resp and not decensor.search(resp):
				return resp
	return lim_tokens(s, round_random(max_length * 2 / 3))

async def llm(func, *args, api=None, timeout=120, premium_context=None, require_message=True, allow_alt=True, **kwargs):
	if isinstance(api, str) or not api:
		# await ensure_models()
		if "model" in kwargs:
			apis = available.get(kwargs["model"]) or {api: None}
		else:
			apis = {api: None}
	else:
		apis = {api: None}
	orig_model = kwargs.get("model")
	exc = None
	tries = tuple(apis.items())
	kwa = kwargs
	for i, (api, minfo) in enumerate(tries + tries):
		if api is None and minfo is None:
			api = summarisation_model
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
		elif "reasoning_effort" in kwa:
			kwa.pop("reasoning_effort")
		kwa["model"] = model
		rl = 8, 1
		if sapi == "mizabot":
			rl = 16, 0.25
		elif sapi == "together":
			rl = 64, 4
		elif sapi == "fireworks":
			rl = 48, 6
		elif sapi == "deepinfra":
			rl = 32, 8
		elif sapi == "openai":
			rl = 5000, 60
		if rl[0]:
			sem = api_sems.get((sapi, model)) or api_sems.setdefault((sapi, model), Semaphore(rl[0], inf, rate_limit=rl[1], sync=rl[1] > 900))
			if (sem.busy or sem.active >= sem.limit / 2 + 1) and i < len(apis) - 1:
				continue
		if isinstance(api, str):
			caller = get_oai(func, api=api)
		else:
			caller = oai_method(api, func)
		if "repetition_penalty" not in kwa:
			kwa["repetition_penalty"] = cast_rp(kwa.pop("frequency_penalty", 0.25), kwa.pop("presence_penalty", 0.25), model=model)
		match sapi:
			case "openrouter":
				body["usage"] = dict(include=True)
			case "fireworks":
				kwa.pop("repetition_penalty", None)
				if kwa.get("tool_choice") == "required":
					kwa["tool_choice"] = "any"
			case "together":
				kwa.pop("frequency_penalty", None)
				kwa.pop("presence_penalty", None)
			case "mistral":
				kwa.pop("repetition_penalty", None)
				body.clear()
			case "openai" | "deepseek" | "deepinfra":
				kwa.pop("repetition_penalty", None)
				kwa.pop("top_p", None)
		if "repetition_penalty" in kwa:
			body["repetition_penalty"] = kwa.pop("repetition_penalty")
		if not kwa.get("stop") or sapi == "openrouter":
			kwa.pop("stop", None)
		if sapi not in ("openai",):
			kwa.pop("user", None)
		elif "user" in kwa:
			kwa["user"] = str(hash(str(kwa["user"])))
		if body:
			kwa["extra_body"] = body
		try:
			if "messages" in kwa:
				if model not in is_function:
					messages = []
					for m in kwa["messages"]:
						m = cdict(m)
						m = untool(m)
						if not m.get("content"):
							m.content = "."
						messages.append(m)
					kwa["messages"] = messages
				else:
					messages = []
					for m in kwa["messages"]:
						m = cdict(m)
						m = fix_tool(m)
						if not m.get("content"):
							m.content = "."
						messages.append(m)
					kwa["messages"] = messages
				if sapi in ("fireworks", "together", "deepinfra", "mistral"):
					messages = []
					for m in kwa["messages"]:
						m2 = None
						if m.get("name"):
							m2 = cdict(m)
							name = m2.pop("name")
							if isinstance(m2.content, list):
								m2.content = [cdict(type="text", text=f"name={name}\n\n{c.text}") if c.get("type") == "text" else c for c in m2.content]
							else:
								m2.content = f"name={name}\n\n{m2.content}"
							m = m2
						messages.append(m)
					kwa["messages"] = messages
				if sapi in ("openai", "deepseek", "openrouter"):
					messages = []
					for m in kwa["messages"]:
						m2 = None
						if m.get("name"):
							if 1 or not oai_name.search(m.name):
								m2 = cdict(m)
								name = m2.pop("name")
								name2 = name.replace(" ", "-")
								if oai_name.search(name2):
									m2.name = name2
								elif isinstance(m2.content, list):
									m2.content = [cdict(type="text", text=f"name={name}\n\n{c.text}") if c.get("type") == "text" else c for c in m2.content]
								else:
									m2.content = f"name={name}\n\n{m2.content}"
								m = m2
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
					kwa["messages"] = messages
				if sapi == "fireworks":
					messages = []
					system = []
					for m in kwa["messages"]:
						if m.get("role") == "system":
							system.append(m.content)
						elif isinstance(m.content, (tuple, list)):
							m = cdict(m)
							content = list(m.content)
							for i, c in enumerate(content):
								if c.get("type") == "image_url":
									content[i] = dict(type="image_url", image_url=dict(url=c["image_url"]["url"]))
							m.content = content
							messages.append(m)
						else:
							messages.append(m)
					for i, m in enumerate(tuple(messages)):
						if m.role == "assistant":
							messages.insert(i, cdict(role="user", content=""))
							break
						if m.role == "user":
							break
					if system:
						messages.insert(0, cdict(role="system", content="\n\n".join(system)))
					if messages[-1].get("role") != "user":
						messages[-1]["role"] = "user"
					kwa["messages"] = messages
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
			if hasattr(response, "choices"):
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
	key = shash(str((data.get("prompt") or data.get("messages"), data.get("model", "kimi-k2-t"), data.get("temperature", 0.75), data.get("max_tokens", 256), data.get("top_p", 0.999), data.get("frequency_penalty", 0), data.get("presence_penalty", 0))))
	if cache:
		return await CACHE.aretrieve(key, _instruct, data, prune=prune, user=user)
	return await CACHE._aretrieve(key, _instruct, data, prune=prune, user=user)

async def _instruct(data, user=None, prune=True):
	inputs = dict(
		temperature=0.75,
		max_tokens=4096,
		top_p=0.999,
		frequency_penalty=0,
		presence_penalty=0,
		user=user,
	)
	inputs.update(data)
	if not inputs.get("model"):
		if summarisation_model:
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
		resp = resp.strip()
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
		"description": "Indicates that you are preparing to draft up a text-only response to the user. You should use other tools first as required, but you MUST use this tool if none is necessary.",
		"parameters": {
			"type": "object", "properties": {
				"format": {
					"type": "string",
					"enum": ["instructive", "casual"],
					"description": 'The conversation format. Enter "instructive" for academic, knowledge or advice responses, "casual" for banter, roleplay, or very simple questions.',
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


def construct_format_parameters_prompt(parameters):
	constructed_prompt = "\n".join(f"<parameter>\n<name>{parameter['name']}</name>\n<type>{parameter['type']}</type>\n<description>{parameter['description']}</description>\n</parameter>" for parameter in parameters)
	return constructed_prompt

def construct_format_tool_for_claude_prompt(name, description, parameters):
	constructed_prompt = (
		"<tool_description>\n"
		f"<tool_name>{name}</tool_name>\n"
		+ ("<description>\n"
		f"{description}\n"
		"</description>\n" if description else "")
		+ "<parameters>\n"
		f"{construct_format_parameters_prompt(parameters)}\n"
		"</parameters>\n"
		"</tool_description>"
	)
	return constructed_prompt

def to_claude_function(tool):
	function = tool["function"]
	params = [dict(name=k, type=p["type"], description=p.get("description", "")) for k, p in function["parameters"]["properties"].items()]
	return construct_format_tool_for_claude_prompt(function["name"], function["description"], params)

def construct_tool_use_system_prompt(tools):
	return (
		"You have access to a set of tools you can use to answer the user's question. If relevant, please use them before answering, to ensure correcness of your responses!\n"
		"\n"
		"Please call them like this:\n"
		"<function_calls>\n"
		"<invoke>\n"
		"<tool_name>$TOOL_NAME</tool_name>\n"
		"<parameters>\n"
		"<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
		"...\n"
		"</parameters>\n"
		"</invoke>\n"
		"</function_calls>\n"
		"\n"
		"Example:\n"
		"<invoke>\n"
		"<tool_name>browse</tool_name>\n"
		"<parameters>\n"
		"<query>Who is Elon Musk?</query>\n"
		"</parameters>\n"
		"</invoke>\n"
		"\n"
		"Currently available tools:\n"
		"<tools>\n"
		+ '\n'.join([to_claude_function(tool) for tool in tools]) +
		"\n</tools>"
	)

def to_claude_tool(tool):
	function = tool["function"]
	return cdict(
		name=function["name"],
		description=function["description"],
		input_schema=cdict(function["parameters"]),
	)

def extract_between_tags(tag: str, string: str, strip: bool = True, capture: bool = True) -> list[str]:
	reg = f"<{tag}>(.+?)</{tag}>" if capture else f"<{tag}>.+?</{tag}>"
	ext_list = re.findall(reg, string, re.DOTALL)
	if strip:
		ext_list = [e.strip() for e in ext_list]
	return ext_list

def from_claude(message, messages=None, allowed_tools=None):
	content = message.content
	if isinstance(content, list):
		message.content = "\n".join(c["text"] for c in content if c and c.get("type") == "text").strip().replace("</thinking>", "*").replace("<thinking>", "*")
		message.tool_calls = tc = []
		for c in content:
			if c and c.get("type") == "tool_use":
				tc.append(cdict(
					index=len(tc),
					id=c.get("id", 0),
					type="function",
					function=cdict(
						name=c.get("name"),
						arguments=json_dumpstr(c.get("input")),
					),
				))
		return message
	if isinstance(content, dict):
		content = content.get("text") or ""
	if not message.get("name") and content.startswith("name=") and "\n" in content:
		name, content = content.split("\n", 1)
		message.name = name.removeprefix("name=").strip()
	if "<function_calls>" not in content and "<invoke>" not in content:
		if "### Response:" in content:
			content = content.rsplit("### Response:", 1)[-1]
		for kw in ("final", "rewrite", "revision", "draft_revision", "draft-revision", "result", "draft"):
			if f"<{kw}>" in content and f"</{kw}>" in content:
				content = "\n\n".join(extract_between_tags(kw, content)) + "\n\n" + content.rsplit(f"</{kw}>", 1)[-1].lstrip()
				break
		if messages and "<search_quality_reflection>" in content and "</search_quality_reflection>" in content:
			reflection = "\n\n".join(extract_between_tags("search_quality_reflection", content))
			content = content.split("</search_quality_reflection>", 1)[-1]
			if "</search_quality_score>" in content:
				content = content.split("</search_quality_score>", 1)[-1]
			for m in reversed(messages):
				if m.role == "tool":
					content += m.content
					break
			content += "\n\n" + reflection
		message.content = content.strip()
		return message
	if "<function_calls>" in content:
		content, calls = content.rsplit("<function_calls>", 1)
	else:
		content, calls = content.rsplit("<invoke>", 1)
		calls = "<invoke>" + calls
		if "</invoke>" in calls:
			calls, cont2 = calls.rsplit("</invoke>", 1)
			if cont2.strip():
				content = content.strip() + "\n\n" + cont2.strip()
			calls += "</invoke>"
	if "### Response:" in content:
		content = content.rsplit("### Response:", 1)[-1]
	for kw in ("final", "rewrite", "revision", "draft_revision", "draft-revision", "result"):
		if f"<{kw}>" in content and f"</{kw}>" in content:
			content = "\n\n".join(extract_between_tags(kw, content))
			break
	message.content = content.strip()
	invokes = extract_between_tags("invoke", calls)
	if not invokes:
		return message
	message.tool_calls = tc = []
	for i, call in enumerate(invokes):
		name = extract_between_tags("tool_name", call)[0]
		function = None
		if allowed_tools:
			matches = [f for f in allowed_tools if f["function"]["name"] == name]
			if matches:
				function = matches[0]["function"]
		params = ()
		if "<parameters>" in call:
			params = extract_between_tags("parameters", call)[0]
		elif "<parameter>" in call:
			params = "".join(extract_between_tags("parameter", call))
		if params:
			args = extract_between_tags(r"\w+", params, capture=False)
			kwargs = {}
			for a in args:
				k = a.split(">", 1)[0].removeprefix("<").strip()
				v = a.split(">", 1)[-1].rsplit("</", 1)[0].strip()
				kwargs[k] = v
		if ("description" in kwargs or "parameter" in kwargs or "value" in kwargs) and function:
			properties = function["parameters"]["properties"]
			props = set(properties).difference(kwargs)
			if props:
				kwargs[next(iter(props))] = kwargs.pop("description", "") or kwargs.pop("parameter", "") or kwargs.pop("value", "")
		tc.append(cdict(
			index=len(tc),
			id=str(i),
			type="function",
			function=cdict(
				name=name,
				arguments=json_dumpstr(kwargs),
			),
		))
	return message

def to_claude(messages, tools=None):
	system = ""
	has_user = False
	last_role = None
	outs = []
	ims = {}
	for i, m in enumerate(messages):
		m = cdict(m)
		if i == len(messages) - 1 and m.role == "assistant":
			m.role = "user"
		if m.role == "system":
			if system:
				system += "\n\n"
			system += m.content
			continue
		if m.role == "assistant" and not has_user:
			continue
		has_user = True
		if m.role == "tool":
			resp = f"""<function_results>
<result>
<tool_name>{m.name}</tool_name>
<stdout>
{m.content}
</stdout>
</result>
</function_results>"""
			if not outs:
				outs.append(cdict(role="user", content=resp))
			else:
				outs[-1].content += "\n\n" + resp
			continue
		elif getattr(m, "tool_calls", None):
			content = m.content or ""
			if content and content[-1] != "\n":
				content += "\n"
			content += "<function_calls>\n"
			for fc in m.pop("tool_calls"):
				content += "<invoke>\n"
				content += "<tool_name>" + fc.function.name + "</tool_name>\n"
				if fc.function.arguments:
					kwargs = orjson.loads(fc.function.arguments)
					if kwargs:
						content += "<parameters>\n"
						content += "\n".join(f"<{k}>{v}</{k}>" for k, v in kwargs.items()) + "\n"
						content += "</parameters>\n"
				content += "</invoke>\n"
			content += "</function_calls>"
			m.content = content
		m.pop("tool_call_id", None)
		images = []
		content = ""
		if isinstance(m.content, list):
			for c in m.content:
				if c.get("type") == "text":
					content += "\n" + c["text"]
					continue
				mime, data = c["image_url"]["url"].split(";base64,", 1)
				image = cdict(
					type="image",
					source=cdict(
						type="base64",
						media_type=mime.removeprefix("data:"),
						data=data,
					),
				)
				images.append(image)
		else:
			content = m.content
		content = (content or "").strip()
		if m.get("name"):
			if m.get("role") == "user":
				content = "name=" + m.pop("name") + "\n\n" + content
			else:
				m.pop("name", None)
		if images and m.get("role") == "user":
			while len(ims) + len(images) > 20:
				k = next(iter(ims))
				im2 = ims.pop(k)
				for im in im2:
					messages[k].content.remove(im)
			ims[i] = images
			images.append(cdict(type="text", text=content))
			content = images
		if not outs or m.role != last_role:
			if not content:
				continue
			m.content = content
			outs.append(m)
			last_role = m.role
			continue
		if isinstance(outs[-1].content, list):
			if isinstance(content, list):
				outs[-1].content.extend(content)
			else:
				for c in outs[-1].content:
					if c.get("type") == "text":
						c["text"] += "\n\n" + content
						break
		elif isinstance(content, list):
			for c in content:
				if c.get("type") == "text":
					c["text"] = outs[-1].content + "\n\n" + c["text"]
					break
		else:
			outs[-1].content += "\n\n" + content
	if tools:
		if system:
			system += "\n\n"
		system += construct_tool_use_system_prompt(tools)
	if outs and isinstance(outs[-1].content, list) and any(c.get("type") == "image" for c in outs[-1].content):
		print("IM:", lim_str(outs[-1], 1024))
	return system, outs

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
					ti = await tcount(construct_tool_use_system_prompt(self.input[1]))
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

	async def __anext__(self):
		while True:
			try:
				item = await self.it.__anext__()
			except StopAsyncIteration:
				print("anext pricing:", self.tokens, self.costs)
				raise
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
				break
			delta = getattr(choice, "delta", None)
			if not delta:
				continue
			for k in ("tool_calls", "content", "reasoning_content"):
				if getattr(delta, k, None):
					break
		return await self.pass_item(item)

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
			else:
				delta = getattr(choice, "delta", None)
				if not delta:
					continue
				for k in ("tool_calls", "content", "reasoning_content"):
					if getattr(delta, k, None):
						yield await self.pass_item(item)
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
	elif fmt == "cohere":
		ins = tuple(map(cohere, messages))
		stops = ["<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"]
		prompt = "<BOS_TOKEN>" + "\n".join(ins) + "\n<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
		prompt += "\n"
	elif fmt == "blockml":
		ins = [chatml(m, "cc") for m in messages]
		stops = im_sep("cc")
		prompt = "\n".join(ins) + "\n" + stops[0] + "assistant"
		prompt += "\n"
	elif fmt == "vicuna":
		ins = tuple(map(vicuna, messages))
		stops = ["</s>", "### user", "### assistant"]
		prompt = "<s>" + "\n".join(ins) + "\n### assistant"
		prompt += ":"
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


cache = CACHE = AutoCache(f"{CACHE_PATH}/ai", stale=86400, timeout=86400 * 14)

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