import asyncio
import base64
import copy
import orjson
import re
import openai
import numpy as np
from collections import deque
from math import ceil, floor, inf
from traceback import format_exc, print_exc
from mpmath import mpf
from misc.types import regexp, astype, lim_str, as_str, cdict, round_random, CommandCancelledError, CE, tracebacksuppressor, utc, T, json_dumps, string_like
from misc.util import AUTH, Cache, Request, get_image_size, json_dumpstr, get_encoding, tcount, lim_tokens, reqs, shash, split_across, wrap_iter
from misc.asyncs import asubmit, csubmit, emptyctx, gather, unflatten, Semaphore, CloseableAsyncIterator

print("AI:", __name__)


endpoints = cdict(
	openai="https://api.openai.com/v1",
	deepseek="https://api.deepseek.com/v1",
	anthropic="https://api.anthropic.com/v1",
	cohere="https://api.cohere.ai/v1",
	cohere_trial="https://api.cohere.ai/v1",
	together="https://api.together.xyz/v1",
	fireworks="https://api.fireworks.ai/inference/v1",
	deepinfra="https://api.deepinfra.com/v1/openai",
	mizabot="https://api.mizabot.xyz/inference/v1",
)

def cast_rp(fp, pp, model=None):
	if model in ("miquliz-120b", "miquliz-120b-v2.0"):
		s = 3
	else:
		s = 1
	return ((fp + pp) / 8 + 1) ** (0.125 * s)
# List of language models and their respective providers, as well as pricing per million input/output tokens
available = {
	"claude-3-opus": {
		"anthropic": ("claude-3-opus-20240229", ("15", "75")),
		None: "gpt-4",
	},
	"claude-3.5-sonnet": {
		"anthropic": ("claude-3-5-sonnet-20241022", ("3", "15")),
	},
	"claude-3.5-haiku": {
		"anthropic": ("claude-3-5-haiku-20241022", ("1", "5")),
		None: "gpt-4m",
	},
	"claude-3-haiku": {
		"anthropic": ("claude-3-haiku-20240307", ("1", "5")),
		None: "gpt-4m",
	},
	"deepseek-v3": {
		"deepseek": ("deepseek-chat", ("0.14", "0.28")),
		None: "gpt-4",
	},
	"llama-3-405b": {
		"deepinfra": ("meta-llama/Meta-Llama-3.1-405B-Instruct", ("1.79", "1.79")),
		"fireworks": ("accounts/fireworks/models/llama-v3p1-405b-instruct", ("3", "3")),
		"together": ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", ("5", "5")),
		None: "qwen-72b",
	},
	"llama-3-90b": {
		"deepinfra": ("meta-llama/Llama-3.2-11B-Vision-Instruct", ("0.35", "0.4")),
		"together": ("meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", ("1.2", "1.2")),
		None: "llama-3-70b",
	},
	"llama-3-70b": {
		"fireworks": ("accounts/fireworks/models/llama-v3p3-70b-instruct", ("0.9", "0.9")),
		"deepinfra": ("meta-llama/Llama-3.3-70B-Instruct", ("0.23", "0.4")),
		"together": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", ("0.88", "0.88")),
		None: "lzlv-70b",
	},
	"llama-3-11b": {
		"deepinfra": ("meta-llama/Llama-3.2-11B-Vision-Instruct", ("0.055", "0.055")),
		"together": ("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", ("0.18", "0.18")),
		None: "llama-3-8b",
	},
	"llama-3-8b": {
		"deepinfra": ("meta-llama/Meta-Llama-3.1-8B-Instruct", ("0.03", "0.05")),
		"fireworks": ("accounts/fireworks/models/llama-v3p1-8b-instruct", ("0.2", "0.2")),
		"together": ("meta-llama/Meta-Llama-3.1-8B-Instruct", ("0.2", "0.2")),
		None: "command-r",
	},
	"command-r-plus": {
		"cohere_trial": ("command-r-plus", ("0.25", "1.25")),
		"mizabot": ("command-r-plus-h6t2", ("0.5", "2.5")),
		"cohere": ("command-r-plus-08-2024", ("2.5", "10")),
		None: "qwen-72b",
	},
	"command-r": {
		"cohere_trial": ("command-r", ("0.05", "0.25")),
		"mizabot": ("CohereForAI/c4ai-command-r-v01", ("0.1", "0.5")),
		"cohere": ("command-r", ("0.5", "1.5")),
		None: "firefunction-v2",
	},
	"35b-beta-long": {
		"mizabot": ("CausalLM/35b-beta-long", ("0.1", "0.5")),
		None: "firefunction-v2",
	},
	"qwen-72b": {
		"fireworks": ("accounts/fireworks/models/qwen2p5-72b-instruct", ("0.9", "0.9")),
		"deepinfra": ("Qwen/Qwen2.5-72B-Instruct", ("0.35", "0.4")),
		"together": ("Qwen/Qwen2.5-72B-Instruct-Turbo", ("1.2", "1.2")),
		None: "llama-3-70b",
	},
	"o1": {
		"openai": ("o1", ("15", "60")),
		None: "gpt-4",
	},
	"o1-preview": {
		"openai": ("o1-preview", ("15", "60")),
		None: "gpt-4",
	},
	"o1-mini": {
		"openai": ("o1-mini", ("3", "12")),
		None: "gpt-4m",
	},
	"gpt-4m": {
		"openai": ("gpt-4o-mini", ("0.15", "0.6")),
		None: "claude-3-haiku",
	},
	"gpt-4": {
		"openai": ("chatgpt-4o-latest", ("2.5", "10")),
		None: "claude-3.5-sonnet",
	},
	"gpt-3.5-turbo-instruct": {
		"openai": ("gpt-3.5-turbo-instruct", ("1.5", "2")),
		None: "command-r-plus",
	},
	"gpt-3.5": {
		"openai": ("gpt-3.5-turbo-0125", ("0.5", "1.5")),
		None: "command-r-plus",
	},
	"firefunction-v1": {
		"fireworks": ("accounts/fireworks/models/firefunction-v1", ("0.5", "0.5")),
		None: "gpt-4m",
	},
	"firefunction-v2": {
		"fireworks": ("accounts/fireworks/models/firefunction-v2", ("0.9", "0.9")),
		None: "gpt-4m",
	},
	"firellava-13b": {
		"fireworks": ("accounts/fireworks/models/firellava-13b", ("0.2", "0.2")),
		None: "claude-3-haiku",
	},
	"phi-4b": {
		"fireworks": ("accounts/fireworks/models/phi-3-vision-128k-instruct", ("0.2", "0.2")),
		None: "gpt-4m",
	},
	"dbrx-instruct": {
		"deepinfra": ("databricks/dbrx-instruct", ("0.6", "0.6")),
		"together": ("databricks/dbrx-instruct", ("1.2", "1.2")),
		"fireworks": ("accounts/fireworks/models/dbrx-instruct", ("1.6", "1.6")),
	},
	"wizard-8x22b": {
		"deepinfra": ("microsoft/WizardLM-2-8x22B", ("0.65", "0.65")),
		None: "mixtral-8x22b-instruct",
	},
	"mixtral-8x22b-instruct": {
		"deepinfra": ("mistralai/Mixtral-8x22B-Instruct-v0.1", ("0.65", "0.65")),
		"together": ("mistralai/Mixtral-8x22B-Instruct-v0.1", ("1.2", "1.2")),
		"fireworks": ("accounts/fireworks/models/mixtral-8x22b-instruct", ("0.9", "0.9")),
		None: "dbrx-instruct",
	},
	"miquliz-120b": {
		"mizabot": ("wolfram/miquliz-120b-v2.0", ("1", "5")),
		None: "command-r-plus",
	},
	"reflection-llama-3-70b": {
		"deepinfra": ("mattshumer/Reflection-Llama-3.1-70B", ("0.35", "0.4")),
		None: "llama-3-70b",
	},
	"euryale-70b": {
		"deepinfra": ("Sao10K/L3-70B-Euryale-v2.1", ("0.35", "0.4")),
		None: "lzlv-70b",
	},
	"lumimaid-70b": {
		"mizabot": ("NeverSleep/Llama-3-Lumimaid-70B-v0.1-alt-GGUF", ("0.2", "1")),
		None: "llama-3-70b",
	},
	"lzlv-70b": {
		"deepinfra": ("lizpreciatior/lzlv_70b_fp16_hf", ("0.59", "0.79")),
	},
	"mixtral-8x7b-instruct": {
		"deepinfra": ("mistralai/Mixtral-8x7B-Instruct-v0.1", ("0.27", "0.27")),
		"fireworks": ("accounts/fireworks/models/mixtral-8x7b-instruct", ("0.5", "0.5")),
		"together": ("mistralai/Mixtral-8x7B-Instruct-v0.1", ("0.6", "0.6")),
		None: "llama-3-8b",
	},
	"mythomax-13b": {
		"deepinfra": ("Gryphe/MythoMax-L2-13b", ("0.12", "0.12")),
		"fireworks": ("accounts/fireworks/models/mythomax-l2-13b", ("0.2", "0.2")),
		"together": ("Gryphe/MythoMax-L2-13b", ("0.30", "0.30")),
		None: "llama-3-8b",
	},
	"stripedhyena-nous-7b": {
		"together": ("togethercomputer/StripedHyena-Nous-7B", ("0.2", "0.2")),
		None: "mythomax-13b",
	},
}

# tags: is_chat, is_completion, is_function, is_vision, is_premium
is_chat = {
	"claude-3.5-sonnet",
	"claude-3-5-sonnet-20241022",
	"claude-3.5-haiku",
	"claude-3-5-haiku-20241022",
	"claude-3-opus",
	"claude-3-opus-20240229",
	"claude-3-sonnet",
	"claude-3-sonnet-20240229",
	"claude-3-haiku",
	"claude-3-haiku-20240307",
	"command-r",
	"command-r-plus",
	"35b-beta-long",
	"llama-3-8b",
	"llama-3-11b",
	"llama-3-70b",
	"llama-3-90b",
	"llama-3-405b",
	"reflection-llama-3-70b",
	"euryale-70b",
	"lzlv-70b",
	"qwen-72b",
	"o1",
	"o1-preview",
	"o1-mini",
	"gpt-4",
	"chatgpt-4o-latest",
	"gpt-4o-2024-05-13",
	"gpt-4m",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-0125-preview",
	"gpt-4-vision-preview",
	"gpt-3.5",
	"gpt-3.5-turbo-0125",
	"gpt-3.5-turbo",
	"deepseek-v3",
	"firefunction-v2",
	"firefunction-v1",
	"firellava-13b",
	"phi-4b",
	"wizard-8x22b",
	"mixtral-8x22b-instruct",
	"nous-hermes-2-mixtral-8x7b-dpo",
	"mixtral-8x7b-instruct",
	"mythomax-13b",
	"stripedhyena-nous-7b",
}
is_completion = {
	"command-r-plus",
	"35b-beta-long",
	"qwen-72b",
	"dbrx-instruct",
	"miquliz-120b",
	"gpt-3.5-turbo-instruct",
	"llama-3-8b",
	"llama-3-11b",
	"llama-3-70b",
	"llama-3-90b",
	"llama-3-405b",
	"reflection-llama-3-70b",
	"euryale-70b",
	"lzlv-70b",
	"phi-4b",
	"wizard-8x22b",
	"mixtral-8x22b-instruct",
	"nous-hermes-2-mixtral-8x7b-dpo",
	"mixtral-8x7b-instruct",
	"mixtral-8x7b",
	"mythomax-13b",
	"stripedhyena-nous-7b",
}
is_function = {
	"claude-3.5-sonnet",
	"claude-3-5-sonnet-20241022",
	"claude-3.5-haiku",
	"claude-3-5-haiku-20241022",
	"claude-3-opus",
	"claude-3-opus-20240229",
	"claude-3-sonnet",
	"claude-3-sonnet-20240229",
	"claude-3-haiku",
	"claude-3-haiku-20240307",
	"command-r",
	"command-r-plus",
	"35b-beta-long",
	"o1",
	"o1-preview",
	"o1-mini",
	"gpt-4",
	"chatgpt-4o-latest",
	"gpt-4o-2024-05-13",
	"gpt-4m",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-0125-preview",
	"gpt-3.5",
	"gpt-3.5-turbo-0125",
	"gpt-3.5-turbo",
	"deepseek-v3",
	"firefunction-v2",
	"firefunction-v1",
}
is_vision = {
	"claude-3.5-sonnet",
	"claude-3-5-sonnet-20241022",
	"claude-3-opus",
	"claude-3-opus-20240229",
	"claude-3-sonnet",
	"claude-3-sonnet-20240229",
	"claude-3-haiku",
	"claude-3-haiku-20240307",
	"llama-3-11b",
	"llama-3-90b",
	"o1",
	"o1-preview",
	"gpt-4",
	"chatgpt-4o-latest",
	"gpt-4o-2024-05-13",
	"gpt-4m",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-vision-preview",
	"firellava-13b",
	"phi-4b",
}
is_premium = {
	"claude-3.5-sonnet",
	"claude-3-5-sonnet-20240620",
	"claude-3-opus",
	"claude-3-opus-20240229",
	"claude-3-sonnet",
	"claude-3-sonnet-20240229",
	"llama-3-405b",
	"o1",
	"o1-preview",
	"o1-mini",
	"chatgpt-4o-latest",
	"gpt-4o-2024-05-13",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-0125-preview",
	"gpt-4-vision-preview",
	"command-r-plus",
	"miquliz-120b",
}
instruct_formats = {
	"reflection-llama-3-70b": "llamav3",
	"euryale-70b": "llamav3",
	"lzlv-70b": "vicuna",
	"miquliz-120b": "mistral",
	"goliath-120b": "alpaca",
	"command-r": "cohere",
	"command-r-plus": "cohere",
	"command-r-plus-08-2024": "cohere",
	"command-r-plus-h6t2": "cohere",
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
	"claude-3.5-sonnet": 200000,
	"claude-3.5-haiku": 200000,
	"claude-3-opus": 200000,
	"claude-3-sonnet": 200000,
	"claude-3-haiku": 200000,
	"command-r": 112000,
	"command-r-plus": 112000,
	"35b-beta-long": 14336,
	"qwen-72b": 32768,
	"llama-3-8b": 131072,
	"llama-3-11b": 131072,
	"llama-3-70b": 131072,
	"llama-3-90b": 131072,
	"llama-3-405b": 131072,
	"o1": 128000,
	"o1-preview": 128000,
	"o1-mini": 128000,
	"gpt-4": 128000,
	"chatgpt-4o-latest": 128000,
	"gpt-4o-2024-05-13": 128000,
	"gpt-4m": 128000,
	"gpt-4o-mini": 128000,
	"gpt-4o-mini-2024-07-18": 128000,
	"gpt-4-turbo-2024-04-09": 128000,
	"gpt-4-0125-preview": 128000,
	"gpt-4-vision-preview": 128000,
	"gpt-3.5": 16384,
	"gpt-3.5-turbo-0125": 16384,
	"gpt-3.5-turbo-instruct": 4096,
	"deepseek-v3": 64000,
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
	"firefunction-v2": 8192,
	"firefunction-v1": 32768,
	"firellava-13b": 4096,
	"phi-4b": 131072,
	"mythomax-13b": 4096,
}

oai_name = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
api_map = cdict()
api_sems = cdict()
api_blocked = Cache(timeout=30, trash=0)

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

decensor = regexp(r"(?:i am unable to|i'm unable to|i cannot|i can't|i am not able to|i'm not able to|i will not) (?:fulfil|assist|help with|provide|do|respond|comply|engage|perform|encourage|endorse|continue) |refrain from|disengage from", re.I)

def nsfw_flagged(resp):
	cat = resp.categories
	flagged = ("harassment", "hate", "sexual", "self_harm", "violence_graphic", "illicit_violent")
	found = []
	for flag in flagged:
		if getattr(cat, flag):
			score = getattr(resp.category_scores, flag)
			found.append((score, flag + f"({round(score * 100)}%)"))
	if not found:
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
		if "role" in m:
			return m.role + "\n" + m.name + "\n" + content
		return m.name + "\n" + content
	if "role" in m:
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
				num_tokens += len(encoding.encode(value))
			elif isinstance(value, dict):
				num_tokens += len(encoding.encode(json_dumpstr(value)))
			elif isinstance(value, (tuple, list)):
				if key == "content":
					for part in value:
						if part.get("type") == "text":
							num_tokens += len(encoding.encode(part["text"]))
							continue
						if part.get("type") == "image_url":
							if model == "cl100k_im" and part["image_url"].get("detail", "auto") == "low":
								num_tokens += 85
							else:
								b = base64.b64decode(part["image_url"]["url"].encode("ascii").split(b",", 1)[-1])
								num_tokens += np.prod(get_image_size(b)) / 768 + 128
							continue
						raise RuntimeError(f"Unexpected object {json_dumpstr(part)} in message.")
				else:
					num_tokens += len(encoding.encode(json_dumpstr(value)))
	return num_tokens + 3
async def count_to(messages, model="cl100k_im"):
	"""Return the number of tokens used by a list of messages."""
	return await asubmit(_count_to, messages, model, priority=2)

async def cut_to(messages, limit=1024, softlim=256, exclude_first=True, best=False, simple=False, prompt=None, premium_context=[]):
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
		if c + count > softlim and not m.get("tool_calls") and m.get("role") != "tool":
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
	summ = "Summary of prior conversation:\n"
	s = "\n\n".join(m_str(m) for m in (messages[:i + 1] if i > 0 else messages))
	s = s.removeprefix(summ).removeprefix("system:").strip()
	c = await tcount(summ + s)
	c2 = await count_to(messages)
	if c2 <= softlim * 1.2:
		if exclude_first:
			messages.insert(0, sm)
		return messages
	ml = max(64, round_random(softlim - count))
	if best:
		s2 = await summarise(s, min_length=ml, best=True, prompt=prompt, premium_context=premium_context)
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

async def summarise(q, min_length=192, max_length=98304, padding=128, best=True, prompt=None, premium_context=[]):
	"Produces an AI-generated summary of input text. Model used is controlled by \"best\" parameter."
	split_length = max_length - padding
	summ_length = min(min_length, split_length - 1)
	c = await tcount(q)
	if c <= min_length:
		return q
	if c <= summ_length:
		return await _summarise(q, summ_length, best=best, prompt=prompt, premium_context=premium_context)
	splits = await asubmit(split_across, q, lim=split_length, mode="tlen", priority=2)
	futs = []
	for s in splits:
		fut = csubmit(_summarise(s, ceil(2 * summ_length / len(splits)), best=best, prompt=prompt, premium_context=premium_context))
		futs.append(fut)
	outs = await gather(*futs)
	q = "\n\n".join(outs)
	q = lim_tokens(q, summ_length)
	c = await tcount(q)
	if c <= min_length:
		return q
	return await _summarise(q, summ_length, best=best, prompt=prompt, premium_context=premium_context)

async def _summarise(s, max_length, prune=True, best=False, prompt=None, premium_context=[]):
	if len(s) <= max_length:
		return s
	s = lim_tokens(s, 98304, mode="right")
	if best:
		with tracebacksuppressor:
			s2 = s
			if prompt:
				s2 += "\n\n" + prompt
			if prompt:
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive summary of the text above, and make sure to include all information relevant to the following question if available:\n\n"""\n{prompt}\n"""\n\n### Response:'
			else:
				prompt = f'### Input:\n"""\n{s}\n"""\n\n### Instruction:\nPlease provide a comprehensive summary of the text above!\n\n### Response:'
			ml = round_random(max_length)
			c = await tcount(prompt)
			model = "gpt-4m" if c + ml < 8192 else "llama-3-8b"
			data = dict(model=model, prompt=prompt, temperature=0.8, top_p=0.9, max_tokens=ml, premium_context=premium_context)
			resp = await instruct(data, best=True, skip=True)
			resp = resp.strip()
			print("Summ:", prompt, resp)
			if resp and not decensor.search(resp):
				return resp
	return lim_tokens(s, round_random(max_length * 2 / 3))

local_available = False
last_ensured = None
last_ensured_time = 0
async def ensure_models(force=False):
	global local_available, last_ensured, last_ensured_time
	if utc() - last_ensured_time < 720 and not force:
		return last_ensured
	known = {
		"command-r-plus-h6t2": "command-r-plus",
		"command-r-plus-08-2024": "command-r-plus",
		"alpindale/c4ai-command-r-plus-GPTQ": "command-r-plus",
		"CohereForAI/c4ai-command-r-plus": "command-r-plus",
		"CausalLM/35b-beta-long": "35b-beta-long",
		"quill-72b-h6t2": "quill-72b",
		"NeverSleep/Llama-3-Lumimaid-70B-v0.1-alt-GGUF": "lumimaid-70b",
	}
	default = "gpt-4m"
	try:
		try:
			info = await Request(f"{endpoints['mizabot']}/models", aio=True, json=True)
		except Exception:
			print_exc()
			last_ensured = models = (default,)
			local_available = False
			return models
		print(info)
		models = [m.get("id") for m in sorted(info["data"], key=lambda m: m.get("created"), reverse=True)]
		for model in models:
			mname = known.get(model, model)
			if mname in available:
				info = available[mname]
				if "mizabot" in info:
					mizabot = (model,) + info["mizabot"][1:]
				else:
					mizabot = (model, (0.5, 2.5))
					print(f"Unknown cost, automatically using {mizabot}")
				info["mizabot"] = mizabot
			else:
				mizabot = (model, (0.5, 2.5))
				print(f"Unknown cost, automatically using {mizabot}")
				available[mname] = {
					"mizabot": (model, (0.5, 2.5)),
					None: default,
				}
		last_ensured = models
		local_available = True
		return models
	finally:
		model = models[0]
		mname = known.get(model, model)
		available["auto"] = {None: mname}
		if mname in contexts:
			contexts["auto"] = contexts[mname]
		if mname in instruct_formats:
			instruct_formats["auto"] = instruct_formats[mname]
		for k in ("is_chat", "is_completion", "is_function", "is_vision", "is_premium"):
			if mname in globals()[k]:
				globals()[k].add("auto")
		last_ensured_time = utc()

async def llm(func, *args, api="openai", timeout=120, premium_context=None, require_message=True, allow_alt=True, **kwargs):
	if isinstance(api, str):
		await ensure_models()
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
		if api is None:
			if not allow_alt:
				break
			assert isinstance(minfo, str)
			kwargs["model"] = minfo
			return await llm(func, *args, timeout=timeout, premium_context=premium_context, require_message=require_message, **kwargs)
		if not isinstance(api, str):
			sapi = as_str(api.base_url)
		else:
			sapi = api
		if minfo is None:
			if not allow_alt:
				break
			print("No pricing schematic found for:", orig_model, sapi, minfo)
			model, pricing = orig_model, (0, 0)
		else:
			model, pricing = minfo
		if (sapi, model) in api_blocked:
			exc = api_blocked[(sapi, model)]
			continue
		is_cohere = model in ("command-r", "command-r-plus", "command-r-plus-08-2024", "command-r-plus-h6t2", "35b-beta-long", "CausalLM/35b-beta-long", "CohereForAI/c4ai-command-r-v01", "CohereForAI/c4ai-command-r-plus", "c4ai-command-r-plus-GPTQ", "alpindale/c4ai-command-r-plus-GPTQ")
		sem = emptyctx
		kwa = kwargs.copy()
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
		elif sapi == "anthropic":
			rl = 50, 60
		elif sapi == "cohere_trial":
			rl = 5, 3600
		elif sapi == "cohere":
			rl = 1000, 60
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
		body = cdict()
		if "repetition_penalty" not in kwa:
			kwa["repetition_penalty"] = cast_rp(kwa.pop("frequency_penalty", 0.25), kwa.pop("presence_penalty", 0.25), model=model)
		if sapi == "mizabot":
			tpp = kwa.get("top_p", 0.9)
			mpp = max(0, 1 - tpp)
			body["min_p"] = mpp
			kwa["top_p"] = 0.999
			kwa["temperature"] = kwa.get("temperature", 1) * (1 + mpp)
			kwa.pop("frequency_penalty", None)
			kwa.pop("presence_penalty", None)
		elif sapi == "fireworks":
			kwa.pop("repetition_penalty", None)
			if kwa.get("tool_choice") == "required":
				kwa["tool_choice"] = "any"
		elif sapi == "together":
			kwa.pop("frequency_penalty", None)
			kwa.pop("presence_penalty", None)
		elif sapi == "deepinfra":
			kwa.pop("repetition_penalty", None)
		elif sapi == "openai":
			kwa.pop("repetition_penalty", None)
		elif sapi in ("cohere", "cohere_trial"):
			kwa.pop("repetition_penalty", None)
		elif sapi == "anthropic":
			pass
		if "repetition_penalty" in kwa:
			body["repetition_penalty"] = kwa.pop("repetition_penalty")
		if not kwa.get("stop"):
			kwa.pop("stop", None)
		if sapi not in ("openai", "anthropic"):
			kwa.pop("user", None)
		elif "user" in kwa:
			kwa["user"] = str(hash(str(kwa["user"])))
		if body:
			kwa["extra_body"] = body
		try:
			if "messages" in kwa:
				if model not in is_function:
					kwa["messages"] = [untool(m.copy()) for m in kwa["messages"]]
				else:
					messages = []
					for m in kwa["messages"]:
						if m.get("content") is None:
							m = cdict(m)
							m.content = ""
						messages.append(m)
					kwa["messages"] = messages
				if sapi in ("fireworks", "together", "deepinfra"):
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
				if sapi == "openai":
					messages = []
					for m in kwa["messages"]:
						m2 = None
						if m.get("name"):
							if not oai_name.search(m.name):
								m2 = cdict(m)
								name = m2.pop("name")
								if isinstance(m2.content, list):
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
			if is_cohere:
				kwa.pop("tool_choice", None)
				stop = set(kwa.get("stop") or ())
				if "tools" in kwa:
					tools = kwa.pop("tools")
					systool = cohere_construct_tool_use_system_prompt(list(map(to_cohere_tool, tools)))
					if kwa.get("messages"):
						sm = cdict(role="system", content=systool)
						if require_message:
							for i, m in tuple(enumerate(kwa["messages"])):
								if m.get("role") != "system":
									kwa["messages"].insert(i, sm)
									break
						else:
							kwa["messages"].append(sm)
						# for i, m in reversed(tuple(enumerate(kwa["messages"]))):
						# 	if m.get("role") == "user":
						# 		kwa["messages"].insert(i, cdict(role="system", content=systool))
						# 		break
					elif kwa.get("prompt"):
						kwa["prompt"] = systool + "\n\n" + kwa["prompt"]
				else:
					tools = []
				if "messages" in kwa:
					fmt = "chatml" if kwa["model"] in ("35b-beta-long", "CausalLM/35b-beta-long") else "cohere"
					prompt, stopn = instruct_structure(kwa.pop("messages"), fmt=fmt)
					stop.update(stopn)
					kwa["prompt"] = prompt
					was_chat = True
				else:
					was_chat = False
				if sapi == "cohere_trial":
					c = await tcount(kwa["prompt"])
					if c + kwa.get("max_tokens", 1024) / 2 < 16384:
						continue
				elif sapi == "mizabot":
					c = await tcount(kwa["prompt"])
					if c + kwa.get("max_tokens", 1024) / 2 > 16384:
						continue
				if tools:
					print("COHERE:", kwa["prompt"])
				was_stream = kwa.pop("stream", False)
				kwa["stream"] = True
				stop.add("<EOS_TOKEN>")
				kwa["stop"] = list(stop)
				result = cdict(
					id="0",
					choices=[cdict(
						finish_reason=None,
						index=0,
						logprobs=None,
					)],
					created=0,
					model=kwa["model"],
					object="text_completion",
				)
				if sapi in ("cohere", "cohere_trial"):
					inputs = dict(
						model=kwa["model"],
						prompt=kwa["prompt"],
						max_tokens=kwa.get("max_tokens", 1024),
						p=kwa.get("top_p", 0.999),
						temperature=kwa.get("temperature", 0.75),
						frequency_penalty=kwa.get("frequency_penalty", 0),
						truncate="START",
						stream=True,
					)
					inputs["stop_sequences"] = list(stop)
					headers = {
						"Authorization": "Bearer " + AUTH["cohere_key"],
						"content-type": "application/json",
					}
					resp = await asubmit(
						reqs.next().post,
						endpoints.get(sapi, sapi) + "/generate",
						data=json_dumps(inputs),
						headers=headers,
						stream=True,
						_timeout_=timeout,
					)
					try:
						resp.raise_for_status()
					except:
						print(resp.text)
						raise
					async def stream_iter(resp):
						it = wrap_iter(resp)
						s = ""
						text = ""
						try:
							async for b in unflatten(it):
								if not b and "\n" not in s:
									break
								s += as_str(b)
								while "\n" in s:
									temp, s = s.split("\n", 1)
									if temp.startswith("{"):
										data = orjson.loads(temp)
										if data["event_type"] == "text-generation":
											text = data["text"]
											result.choices[0].text = text
										elif data["event_type"] == "stream-end":
											result.choices[0].text = ""
											response = data["response"]
											result.id = response["id"]
											if response.get("generations"):
												gen = response["generations"][0]
												text = gen["text"]
												reason = gen.get("finish_reason")
												if reason.lower() != "complete":
													raise RuntimeError(reason)
												result.choices[0].finish_reason = reason
												meta = gen.get("meta")
												if meta and meta.get("billed_units"):
													result.usage = meta.get("billed_units")
										else:
											print("Unknown event", data)
											continue
										yield copy.deepcopy(result)
						except CommandCancelledError:
							pass
					async with sem:
						iterator = stream_iter(resp)
				else:
					caller = get_oai("completions.create", api=api)
					async with sem:
						iterator = await asyncio.wait_for(caller(*args, timeout=timeout, **kwa), timeout=timeout)
				async def pipe_iter(it):
					result = cdict(
						id="0",
						choices=[cdict(
							finish_reason=None,
							index=0,
							logprobs=None,
						)],
						created=0,
						model=kwa["model"],
						object="chat.completion.chunk",
					)
					s = ""
					text = ""
					locked = False
					try:
						async for b in it:
							if not b or not getattr(b, "choices", None) or not b.choices[0].text:
								continue
							s += b.choices[0].text
							while "\n" in s or len(s) > 48:
								if "\n" in s:
									i = s.index("\n")
									splitted = True
								else:
									i = 32
									splitted = False
								temp, s = s[:i + 1], s[i + 1:]
								text += temp
								if splitted and was_chat and temp.startswith("Action: ```"):
									locked = True
									continue
								if not locked and temp:
									if was_chat:
										result.choices[0].delta = cdict(role="assistant", content=temp, tool_calls=None)
									else:
										result.choices[0].text = temp
									yield copy.deepcopy(result)
					except CommandCancelledError:
						pass
					if s:
						temp = s
						text += temp
						if not was_chat or not locked and temp:
							if was_chat:
								result.choices[0].delta = cdict(role="assistant", content=temp, tool_calls=None)
							else:
								result.choices[0].text = temp
							yield copy.deepcopy(result)
					if not locked or not was_chat:
						return
					msg = cdict(role="assistant", content=text.strip())
					message = from_cohere(msg)
					if message.tool_calls:
						has_da = any(t["function"]["name"] == "directly_answer" for t in tools)
						directly_answer = False
						for i, tc in enumerate(tuple(message.tool_calls)):
							if tc.function.name == "directly_answer":
								directly_answer = True
								if not has_da:
									message.tool_calls.pop(i)
								break
						if directly_answer and require_message and tools and not message.tool_calls:
							inputs = dict(kwargs)
							inputs.pop("tools")
							inputs["stream"] = True
							csubmit(it.close())
							resp = await llm(func, *args, **inputs, timeout=timeout, premium_context=premium_context)
							async for r in resp:
								yield r
							return
					if message.content:
						message.content = "\r" + message.content
					result.choices[0].delta = message
					yield result
				inputs = kwa["prompt"]
				stream = OpenAIPricingIterator(
					pipe_iter(iterator),
					getattr(iterator, "close", None),
					premium_context=premium_context,
					api=sapi,
					model=model,
					input=inputs,
					pricing=pricing,
				)
				if was_stream:
					return stream
				return await collect_stream(stream)
			elif sapi == "anthropic" and "messages" in kwa:
				if kwa.get("stream"):
					system, messages = to_claude(kwa["messages"], kwa.get("tools"))
					stopn = ["\n\nHuman:", "\n\nAssistant", "</function_calls>"]
					inputs = dict(
						model=kwa["model"],
						system=system,
						messages=messages,
						max_tokens=min(4096, kwa.get("max_tokens", 1024)),
						stop_sequences=list(set(tuple(kwa.get("stop", ())) + tuple(stopn))),
						temperature=kwa.get("temperature", 0.75),
						stream=True,
					)
					headers = {
						"x-api-key": AUTH["anthropic_key"],
						"content-type": "application/json",
						"anthropic-version": "2023-06-01",
					}
					async with sem:
						resp = await asubmit(
							reqs.next().post,
							endpoints.get(sapi, sapi) + "/messages",
							data=json_dumps(inputs),
							headers=headers,
							stream=True,
							_timeout_=timeout,
						)
					try:
						resp.raise_for_status()
					except Exception:
						print("Claude Inputs:", inputs)
						print(resp.text)
						raise
					async def stream_iter(resp):
						it = wrap_iter(resp)
						result = cdict(
							id="0",
							choices=[cdict(
								finish_reason=None,
								index=0,
								logprobs=None,
							)],
							created=0,
							model=kwa["model"],
							object="chat.completion.chunk",
						)
						s = ""
						text = ""
						locked = False
						try:
							async for b in unflatten(it):
								if not b:
									break
								s += as_str(b)
								while "\n" in s:
									temp, s = s.split("\n", 1)
									if temp.startswith("data:"):
										data = orjson.loads(temp.split("data:", 1)[-1])
										if data["type"] == "message_start":
											message = data["message"]
											result.id = message["id"]
											result.usage = message["usage"]
											result.model = message["model"]
											result.choices[0].delta = cdict(
												role=message.get("role") or "assistant",
												content=message.get("content") or "",
												tool_calls=None,
											)
											text = result.choices[0].delta.content
										elif data["type"] == "content_block_start":
											delta = data["content_block"]
											if not delta.get("text"):
												continue
											result.choices[0].delta.content = delta["text"]
											text += result.choices[0].delta.content
										elif data["type"] == "content_block_delta":
											delta = data["delta"]
											if "stop_reason" in delta:
												result.choices[0].finish_reason = delta["stop_reason"]
											if "usage" in delta:
												result.usage = delta["usage"]
											if not delta.get("text"):
												continue
											result.choices[0].delta.content = delta["text"]
											text += result.choices[0].delta.content
										elif data["type"] == "message_delta":
											delta = data["delta"]
											if "stop_reason" in delta:
												result.choices[0].finish_reason = delta["stop_reason"]
											if "usage" in delta:
												result.usage = delta["usage"]
											if not delta.get("text"):
												continue
											result.choices[0].delta.content = delta["text"]
											text += result.choices[0].delta.content
										else:
											print("Unknown event", data)
											continue
										if "<function_calls>" in text or "<invoke>" in text:
											locked = True
											continue
										if "<function" in text or "<invoke" in text:
											continue
										if not locked and text:
											if text.startswith("name") or text.startswith("Name"):
												if "\n" not in text:
													continue
												naming, text = text.split("\n", 1)
												if "=" not in naming:
													text = naming + "\n" + text
													locked = True
													continue
												result.choices[0].delta.content = text
												result.choices[0].delta.name = naming.split("=", 1)[-1].rstrip()
											yield copy.deepcopy(result)
						except CommandCancelledError:
							pass
						if not locked:
							return
						if not result.choices[0].get("delta"):
							return
						msg = result.choices[0].delta
						msg.content = text.strip()
						message = from_claude(msg, messages=kwa["messages"], allowed_tools=kwa.get("tools"))
						if message.content:
							message.content = "\r" + message.content
						result.choices[0].delta = message
						yield result
					return OpenAIPricingIterator(
						stream_iter(resp),
						resp.close,
						premium_context=premium_context,
						api="anthropic",
						model=model,
						input=(kwa["messages"], kwa.get("tools")),
						pricing=pricing,
					)
				system, messages = to_claude(kwa["messages"], None)
				tools = kwa.get("tools", [])
				if tools:
					tools = list(map(to_claude_tool, tools))
				stopn = ["\n\nHuman:", "\n\nAssistant"]
				inputs = dict(
					model=kwa["model"],
					system=system,
					messages=messages,
					tools=tools,
					max_tokens=min(4096, kwa.get("max_tokens", 1024)),
					stop_sequences=list(set(tuple(kwa.get("stop", ())) + tuple(stopn))),
					temperature=kwa.get("temperature", 0.75),
				)
				headers = {
					"x-api-key": AUTH["anthropic_key"],
					"content-type": "application/json",
					"anthropic-version": "2023-06-01",
					"anthropic-beta": "tools-2024-04-04",
				}
				try:
					async with sem:
						resp = await Request(
							endpoints.get(sapi, sapi) + "/messages",
							method="POST",
							data=json_dumps(inputs),
							headers=headers,
							json=True,
							aio=True,
							timeout=timeout,
						)
				except Exception:
					print("Claude Inputs:", inputs)
					raise
				msg = cdict(role=resp.get("role", "assistant"), content=resp["content"])
				if resp.get("tool_calls"):
					msg["tool_calls"] = resp["tool_calls"]
				else:
					msg = from_claude(msg, messages=kwa["messages"], allowed_tools=kwa.get("tools"))
				resp = cdict(
					id=resp["id"],
					choices=[cdict(
						finish_reason=resp["stop_reason"],
						index=0,
						logprobs=None,
						message=msg,
					)],
					created=getattr(resp, "created", None) or floor(utc()),
					model=getattr(resp, "model", None) or model,
					object="chat.completion",
					usage=resp["usage"],
					api="anthropic",
				)
				stream = OpenAIPricingIterator(
					None,
					None,
					premium_context=premium_context,
					api="anthropic",
					model=model,
					input=(kwa["messages"], kwa.get("tools")),
					pricing=pricing,
				)
				return await stream.pass_item(resp)
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
		except CE:
			raise
		except Exception as ex:
			if isinstance(ex, ConnectionError) and ex.errno in (401, 403, 404, 429, 502, 503, 504):
				api_blocked[(sapi, model)] = ex
			if not exc:
				exc = ex
			print(str(sapi) + ": " + str(kwa.get("model")) + ":", kwa, format_exc())
			continue
	# print("ERRORED:", model, lim_str(kwa, 16384))
	raise (exc or RuntimeError("Unknown error occured."))

async def instruct(data, best=False, skip=False, prune=True, cache=True, user=None):
	data["prompt"] = data.get("prompt") or data.pop("inputs", None) or data.pop("input", None)
	key = shash(str((data["prompt"], data.get("model", "gpt-4m"), data.get("temperature", 0.75), data.get("max_tokens", 256), data.get("top_p", 0.999), data.get("frequency_penalty", 0), data.get("presence_penalty", 0))))
	if cache:
		tup = await CACHE.retrieve_from(key, _instruct2, data, best=best, skip=skip, prune=prune, user=user)
		if tup[1] >= best:
			return tup[0]
	resp, best = await _instruct2(data, best=best, skip=skip, prune=prune, user=user)
	if best and decensor.search(resp):
		resp, best = await _instruct2(data, best=False, skip=False, prune=True, user=user)
	CACHE[key] = (resp, best)
	return resp

async def _instruct2(data, best=False, skip=False, prune=True, user=None):
	resp = await _instruct(data, best=best, skip=skip, user=user)
	if prune:
		resp = resp.strip()
		resp2 = regexp(r"### (?:Input|Instruction|Response):?").split(resp, 1)[0].strip()
		if resp != resp2:
			print("PRUNED:", resp, resp2, sep="::")
			resp = resp2
	return (resp, best)

async def _instruct(data, best=False, skip=False, user=None):
	# c = await tcount(data["prompt"])
	inputs = dict(
		model="gpt-4" if best >= 2 else "gpt-4m",
		temperature=0.75,
		max_tokens=256,
		top_p=0.999,
		frequency_penalty=0,
		presence_penalty=0,
		user=user,
		stop=["### Instruction:", "### Response:", "### Input:"]
	)
	inputs.update(data)
	if best >= 1:
		res = await moderate(data["prompt"], premium_context=data.get("premium_context", []))
		dec = nsfw_flagged(res) and not skip
	else:
		dec = True and not skip
	if dec:
		inputs["model"] = "auto" if best else "llama-3-70b"
	if data.get("model", "gpt-4m") in is_chat:
		prompt = inputs.pop("prompt")
		inputs["messages"] = [cdict(role="user", content=prompt)]
		async with asyncio.timeout(70):
			response = await llm("chat.completions.create", **inputs, timeout=60)
		return response.choices[0].message.content
	async with asyncio.timeout(100):
		response = await llm("completions.create", **inputs, timeout=90)
	return response.choices[0].text


f_browse = {
	"type": "function", "function": {
		"name": "browse",
		"description": "Searches internet browser, or visits given URL. Use for knowledge or advice to validate facts and up-to-date information!",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "Who won the 2026 world cup?", "https://youtu.be/dQw4w9WgXcQ", "Weather in San Francisco"',
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
					"description": 'Query, eg. "Real solutions for x^3-6x^2+12", "eigenvalues of {{2,3,-3},{4,2,-4},{4,3,-5}}", "randint(1,100)"',
				},
			},
			"required": ["query"],
}}}
f_reasoning = {
	"type": "function", "function": {
		"name": "reasoning",
		"description": "Requests for a slower, more powerful language model to provide reasoning. Use if you are unsure about, or if a user is pointing out a flaw in your logic.",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step. Use the example above to decode: oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz"',
				},
			},
			"required": ["query"],
}}}
f_sympy = {
	"type": "function", "function": {
		"name": "sympy",
		"description": "Queries the Sympy algebraic library. Use in place of Wolfram Alpha only for single functions.",
		"parameters": {
			"type": "object", "properties": {
				"query": {
					"type": "string",
					"description": 'Query, eg. "factorint(57336415063790604359)", "randint(1,100)", "pi*2"',
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
		"description": "Generates an image of the input caption, only use when asked to draw a picture. Please be descriptive!",
		"parameters": {
			"type": "object", "properties": {
				"prompt": {
					"type": "string",
					"description": "Prompt, eg. Brilliant view of a futuristic city in an alien world, glowing spaceships, 8k fantasy art",
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
		"description": "Calls a standard chatbot to respond to the conversation. You must always use the `directly-answer` tool if calling the other tools is unnecessary.",
		"parameters": {
			"type": "object", "properties": {
				"assistant": {
					"type": "string",
					"enum": ["instructive", "casual"],
					"description": 'How the assistant should respond. Enter "instructive" for academic, knowledge or advice responses, "casual" for banter, roleplay, or simple questions.',
				},
			},
			"required": ["assistant"],
}}}

TOOLS = {
	"knowledge_internet": [
		f_browse,
		f_wolfram_alpha,
		f_reasoning,
		# f_sympy,
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
		f_reasoning,
		# f_sympy,
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
QUESTION_EXAMPLES = {
	"knowledge_internet": {
		"instructive": {
			"What's the current president of the United States?",
			"who is elon musk?",
			"can you explain the effects of artificial intelligence in the workplace",
			"have you heard of the critter tower defence mod of cave story?",
			"Please explain the terms of service",
			'Decode following text:\n"PDpzdGFyZToxMDkzNjE3NTk5OTU2Mzk0MDQ0Pg"',
			"What is the dead internet theory?",
			"help me with thesis essay on social media addiction.",
			"Best brand of laptop to buy 2025",
			"difference between string theory and m theory:",
			"what is the lifespan of an average dog?",
			"Can you recommend some car insurance companies in Germany",
			"melting point and tensile strength of iridium",
			"whats the capital of brazil",
			"cite some sources please",
			"Is it true?",
			"I'm referring to what you claimed here",
			"that's not what i asked",
		},
		"casual": {
			"google en passant",
			"Top 10 Nintendo games",
			"what celeste player just hit 10k youtube subscribers",
			"Give me an inspirational quote from a James Cameron movie.",
			"name an obscure hidden gem SNES title.",
			"which youtube channel has the most subscribers?",
			"opinion on latest language model",
		},
	},
	"image_generation": {
		"casual": {
			"can you send a picture of what you look like?",
			"/art box of puppies, cute 4k high quality",
			"Generate an image of a boy with dragon wings",
			"can you give me some example picture for inspiration?",
			"that looks horrible, try again",
			"More art please!",
			"What would a cat with 9 tails look like?",
			"Draw a fox",
		},
	},
	"calculator": {
		"instructive": {
			"What's the square root of 6561",
			"calculate pi to 700 places.",
			"what is 4637,4938576934 + 210,293857502?",
			"can you give me sample exam questions on volumes of rotation integration",
			"find the derivative of 3^(x+2)/(x-1)",
			"what number comes next in the sequence: 1 3 7 15 31",
			"step by step solution to equation in this picture",
			"Convert 100 fahrenheit to celsius",
		},
		"casual": {
			"gimme a random number from 1 to 99999",
			"can you roll a d20.",
			"how many fingers do 5 hands have?",
		},
	},
	"calendar": {
		"instructive": {
			"I have a test 3pm tomorrow, but I think I'll forget",
			"I have a dentist appointment on the 26th of August.",
		},
		"casual": {
			"Remind me in 5 minutes",
			"hey, set an alarm for 30 seconds from now",
			"The next spacex rocket launches in an hour, make an announcement when it's almost time!",
			"Can you remind me in 3 days?",
			"what's today's date?",
			"Whats the time",
			"current time in PDT.",
		},
	},
	"audio_music": {
		"casual": {
			"can you play your favourite song?",
			"what does that sound like",
			"stop the song PLEASE",
			"That's too loud!!",
			"I can't hear, can you turn the volume up",
			"can you bassboost the music, I think it would be funny",
			"Can you play Spotify songs?",
			"you can play music",
			"/play interstellar no time for caution",
			"Does this sound better in nightcore",
		},
	},
	"none": {
		"instructive": {
			"how do i support a depressed family member",
			"Are you sentient?",
			"Write me code in Rust to calculate 10000th prime number.",
			"Help me with my app in Unity, I need the background to be rotated",
			'rewrite the sentence using the word "discombobulated"',
			"what's wrong with this python code?\nimport math\nprint(math.sqrt(3 ** 2 - 4 ** 2))",
			"I don't understand",
			"Can I have advice? Nobody listens to me",
			"i need you to tell me what to do",
		},
		"casual": {
			"what's your opinion on lesbians",
			"Please roleplay as Link from Legend of Zelda.",
			"can you give him a hug?",
			"can you write me a sad story with a happy ending?",
			"what was the last thing i said",
			"Do you wish to gain a physical body?",
			"talk like you're darth vader",
			"Are you better than Clyde",
			"How are you?",
		},
	},
}
MODEL_ROUTER = []
TOOL_ROUTER = []
for t, v in QUESTION_EXAMPLES.items():
	for m, e in v.items():
		for text in e:
			MODEL_ROUTER.append(dict(label=m, text=text))
			TOOL_ROUTER.append(dict(label=t, text=text))


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


def to_cohere_function(tool):
	tmap = dict(
		string="str",
		number="float",
		integer="int",
		boolean="bool",
	)
	args = ", ".join(k + ": " + tmap[v["type"]] for k, v in tool.parameter_definitions.items())
	def get_description(v):
		if v.get("enum"):
			return f': enum ({", ".join(map(json_dumpstr, v["enum"]))}): {v.get("description", "")}'.rstrip()
		return ": " + v["description"] if "description" in v else ""
	if tool.parameter_definitions:
		argh = "\n\n\tArgs:\n\t\t" + "\n\t\t".join(k + " (" + tmap[v["type"]] + ")" + get_description(v) for k, v in tool.parameter_definitions.items()) + '\n\t"""'
	else:
		argh = '\n\t"""\n\tpass'
	return f'''```python
def {tool.name}({args}) -> str:
	"""{tool.description}{argh}
```'''

def cohere_construct_tool_use_system_prompt(tools):
	return (
		"## Available Tools\nYou have access to a set of tools you can use to answer the user's question. If relevant, please use them before answering, to ensure correcness of your responses!\n"
		"\n"
		+ "\n\n".join(to_cohere_function(tool) for tool in tools) + "\n\n"
		+ "Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. The list of actions you want to call should be formatted as a list of json objects, for example:\n"
		+ """```json
[
	{
		"tool_name": title of the tool in the specification,
		"parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} for no parameters
	}
]```"""
	)

def to_cohere_tool(tool):
	function = tool["function"]
	return cdict(
		name=function["name"],
		description=function["description"],
		parameter_definitions=cdict(function["parameters"]["properties"]),
	)

def from_cohere(message):
	content = message.content
	print("FROM:", content)
	message.tool_calls = tc = []
	while True:
		content = "\n" + content
		spl = content.split("\nAction: ```json", 1)
		if len(spl) <= 1 or "```" not in spl[1]:
			message.content = "".join(spl).strip()
			return message
		act, content = spl[1].split("```", 1)
		actions = orjson.loads(act)
		tc.extend(cdict(
			index=len(tc),
			id=f"{len(tc) + 1}",
			type="function",
			function=cdict(
				name=c.get("tool_name"),
				arguments=json_dumpstr(c.get("parameters")),
			),
		) for c in actions)
		content = spl[0].strip() + "\n\n" + content.strip()

def untool(message):
	content = message.content or ""
	if message.get("role") == "tool":
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
	"o1",
	"o1-preview",
	"o1-mini",
	"chatgpt-4o-latest",
	"gpt-4o-2024-05-13",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",
	"gpt-4-turbo-2024-04-09",
	"gpt-3.5-turbo-0125",
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
		self.costs = [utc(), api, model, "0"]
		self.pricing = pricing or (m_input, m_output)
		self.tokeniser = "cl100k_im" if model in CL100K_IM else "cohere" if model in ("command-r-", "command-r-plus", "command-r-plus-08-2024", "command-r-plus-h6t2") else "llamav2"

	async def pass_item(self, item):
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
		self.costs[-1] = str((mpf(self.pricing[0]) * self.tokens[0] + mpf(self.pricing[1]) * self.tokens[1]) / 1000000)
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
			if not item.choices:
				continue
			choice = item.choices[0]
			if getattr(choice, "text", None):
				break
			delta = getattr(choice, "delta", None)
			if delta and getattr(delta, "tool_calls", None):
				break
			if delta and getattr(delta, "content", None):
				break
		return await self.pass_item(item)

	async def __aiter__(self):
		async for item in self.it:
			if not item.choices:
				continue
			choice = item.choices[0]
			if getattr(choice, "text", None):
				yield await self.pass_item(item)
			else:
				delta = getattr(choice, "delta", None)
				if delta and getattr(delta, "tool_calls", None):
					yield await self.pass_item(item)
				elif delta and getattr(delta, "content", None):
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


cache = CACHE = Cache(timeout=86400 * 14, persist="ai.cache")

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
		resp = await get_oai("moderations.create")(model="omni-moderation-latest", input=input)
		premium_context.append(["openai", resp.model, "0.00001"])
		return resp
	resp = await CACHE.retrieve_from("moderate-" + shash(text) + "-" + shash(image), moderate_into, text, image)
	return resp.results[0]