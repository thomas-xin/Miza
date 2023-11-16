import os, sys, time, datetime, urllib, orjson, io, random, re, traceback
import concurrent.futures, asyncio
import selenium, requests, torch, openai, httpx, psutil, markdownify
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
for i in range(3):
	try:
		from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline, set_seed
	except ImportError:
		time.sleep(i + 1)
	else:
		break

# from collections2 import *
MIZAAC = ""
GPTQ = False
BNB = False

import tiktoken, accelerate
if torch and torch.cuda.is_available():
	torch.cuda._lazy_init()
	try:
		torch.cuda.set_enabled_lms(True)
	except AttributeError:
		pass
from fp.fp import FreeProxy
print_exc = lambda: sys.stdout.write(traceback.format_exc())

print = lambda *args, sep=" ", end="\n": sys.stdout.buffer.write(f"~print({repr(sep.join(map(str, args)))},end={repr(end)})\n".encode("utf-8"))

try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=64)
drivers = selenium.__dict__.setdefault("-drivers", [])

chatgpt = AsyncChatGPT = None

from math import *

def lim_str(s, maxlen=10, mode="centre"):
	if maxlen is None:
		return s
	if type(s) is not str:
		s = str(s)
	over = (len(s) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(s) / 2
			s = s[:ceil(half - over - 1)] + ".." + s[ceil(half + over + 1):]
		else:
			s = s[:maxlen - 3] + "..."
	return s

def lim_tokens(s, maxlen=10, mode="centre"):
	if maxlen is None:
		return s
	if type(s) is not str:
		s = str(s)
	enc = tiktoken.get_encoding("cl100k_base")
	tokens = enc.encode(s)
	over = (len(tokens) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(tokens) / 2
			s = enc.decode(tokens[:ceil(half - over - 1)]) + ".." + enc.decode(tokens[ceil(half + over + 1):])
		else:
			s = enc.decode(tokens[:maxlen - 3]) + "..."
	return s.strip()

url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
is_url = lambda url: url_match.search(url)

class_name = webdriver.common.by.By.CLASS_NAME
css_selector = webdriver.common.by.By.CSS_SELECTOR
xpath = webdriver.common.by.By.XPATH
tag_name = webdriver.common.by.By.TAG_NAME
driver_path = "misc/msedgedriver.exe"
browsers = dict(
	edge=dict(
		driver=webdriver.edge.webdriver.WebDriver,
		service=webdriver.edge.service.Service,
		options=webdriver.EdgeOptions,
		path=driver_path,
	),
)
browser = browsers["edge"]

def create_driver():
	ts = time.time_ns()
	folder = os.path.join(os.getcwd(), f"d~{ts}")
	service = browser["service"](browser["path"])
	options = browser["options"]()
	options.add_argument("--headless")
	options.add_argument("--disable-gpu")
	options.add_argument("--no-sandbox")
	options.add_argument("--deny-permission-prompts")
	options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")
	prefs = {
		"download.default_directory" : folder,
		"profile.managed_default_content_settings.geolocation": 2,
	}
	options.add_experimental_option("prefs", prefs)

	try:
		driver = browser["driver"](
			service=service,
			options=options,
		)
	except selenium.common.SessionNotCreatedException as ex:
		if "Current browser version is " in (s := repr(ex)):
			v = s.split("Current browser version is ", 1)[-1].split(None, 1)[0]
			if os.name == "nt":
				url = f"https://msedgedriver.azureedge.net/{v}/edgedriver_win64.zip"
				import requests, io, zipfile
				with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
					with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
						with z.open("msedgedriver.exe") as fi:
							with open("misc/msedgedriver.exe", "wb") as fo:
								b = fi.read()
								fo.write(b)
			else:
				url = f"https://msedgedriver.azureedge.net/{v}/edgedriver_linux64.zip"
				import requests, io, zipfile
				with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
					with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
						with z.open("msedgedriver") as fi:
							with open("misc/msedgedriver", "wb") as fo:
								b = fi.read()
								fo.write(b)
			driver = browser["driver"](
				service=service,
				options=options,
			)
		else:
			raise
	except selenium.common.WebDriverException as ex:
		argv = " ".join(ex.args)
		search = "unrecognized Microsoft Edge version"
		if search in argv and "Chrome" in argv:
			v = argv.split("Stacktrace", 1)[0].rsplit("/", 1)[-1].strip()
			if os.name == "nt":
				url = f"https://chromedriver.storage.googleapis.com/{v}/chromedriver_win32.zip"
				import requests, io, zipfile
				with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
					with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
						with z.open("msedgedriver.exe") as fi:
							with open("misc/msedgedriver.exe", "wb") as fo:
								b = fi.read()
								fo.write(b)
			else:
				url = f"https://chromedriver.storage.googleapis.com/{v}/chromedriver_linux64.zip"
				import requests, io, zipfile
				with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
					with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
						with z.open("msedgedriver") as fi:
							with open("misc/msedgedriver", "wb") as fo:
								b = fi.read()
								fo.write(b)
			driver = browser["driver"](
				service=service,
				options=options,
			)
		else:
			raise
	driver.folder = folder
	driver.get("file://")
	return driver

LAST_DRIVER = 0
def ensure_drivers():
	globals()["LAST_DRIVER"] = time.time()
	while len(drivers) < 1:
		drivers.append(exc.submit(create_driver))
		time.sleep(1)
def get_driver():
	globals()["LAST_DRIVER"] = time.time()
	if not drivers:
		drivers.append(exc.submit(create_driver))
	try:
		driver = drivers.pop(0)
		if hasattr(driver, "result"):
			driver = driver.result()
	except selenium.common.exceptions.WebDriverException:
		print_exc()
		driver = create_driver()
	else:
		try:
			exc.submit(getattr, driver, "title").result(timeout=0.5)
		except:
			print_exc()
			driver = create_driver()
	# exc.submit(ensure_drivers)
	return driver
def return_driver(d):
	d.get("file://")
	drivers.insert(0, d)
def update():
	if time.time() - LAST_DRIVER >= 3600:
		globals()["LAST_DRIVER"] = time.time()
		if not drivers:
			return
		try:
			d = drivers.pop(0)
			if hasattr(d, "result"):
				d = d.result()
		except:
			pass
		else:
			drivers.clear()
			return_driver(d)

def determine_cuda(mem=1, priority=None, multi=False, major=0):
	if not torch or not DEVICES or not torch.cuda.is_available():
		if multi:
			return [-1], torch.float32
		return -1, torch.float32
	n = torch.cuda.device_count()
	if not n:
		if multi:
			return [-1], torch.float32
		return -1, torch.float32
	import pynvml
	pynvml.nvmlInit()
	dc = pynvml.nvmlDeviceGetCount()
	handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
	gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) for d in handles]
	tinfo = [torch.cuda.get_device_properties(COMPUTE_ORDER.index(i)) if i in COMPUTE_ORDER else None for i in range(dc)]
	COMPUTE_LOAD = globals().get("COMPUTE_LOAD") or [0] * dc
	high = max(COMPUTE_LOAD)
	if priority == "full":
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, p.major, p.minor, p.multi_processor_count, p.total_memory)
	elif priority:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, p.major >= major, COMPUTE_LOAD[i] < high * 0.9, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, i, p.multi_processor_count, p.total_memory)
	elif priority is False:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, -mem // 1073741824, p.major, p.minor, COMPUTE_LOAD[i] < high * 0.75, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -gmems[i].free, p.multi_processor_count)
	else:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] < high * 0.5, p.major >= major, p.major >= 7, -p.major, -p.minor, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -p.multi_processor_count, -gmems[i].free)
	pcs = sorted(DEVICES, key=key, reverse=True)
	if multi:
		return [COMPUTE_ORDER.index(i) for i in pcs if gmems[i].free >= mem], torch.float16
	return COMPUTE_ORDER.index(pcs[0]), torch.float16

mcache = {}
def cached_model(cls, model, **kwargs):
	t = (cls, model, tuple(kwargs.keys()))
	try:
		return mcache[t]
	except KeyError:
		mcache[t] = cls(model, **kwargs)
	print("CACHED_MODEL:", t)
	return mcache[t]

def backup_model(cls, model, force=False, **kwargs):
	kwargs.pop("resume_download", None)
	t = (cls, model, tuple(kwargs.keys()))
	try:
		return mcache[t]
	except KeyError:
		pass
	if force:
		try:
			return cls(model, resume_download=True, **kwargs)
		except Exception as ex:
			ex2 = ex
	else:
		try:
			return cls(model, local_files_only=True, **kwargs)
		except:
			fut = exc.submit(cached_model, cls, model, resume_download=True, **kwargs)
			try:
				return fut.result(timeout=24)
			except Exception as ex:
				ex2 = ex
	if isinstance(ex2, concurrent.futures.TimeoutError):
		try:
			return fut.result(timeout=60)
		except concurrent.futures.TimeoutError:
			raise RuntimeError("Model is loading, please wait...")
	raise ex2

def safecomp(gen):
	while True:
		try:
			e = next(gen)
		except StopIteration:
			return
		except selenium.common.StaleElementReferenceException:
			continue
		yield e

# Decodes HTML encoded characters in a string.
def html_decode(s):
	while len(s) > 7:
		try:
			i = s.index("&#")
		except ValueError:
			break
		try:
			if s[i + 2] == "x":
				base = 16
				p = i + 3
			else:
				base = 10
				p = i + 2
			for a in range(p, p + 16):
				c = s[a]
				if c == ";":
					v = int(s[p:a], base)
					break
				elif not c.isnumeric() and c not in "abcdefABCDEF":
					break
			c = chr(v)
			s = s[:i] + c + s[a + 1:]
		except (ValueError, NameError, IndexError):
			s = s[:i + 1] + "\u200b" + s[i + 1:]
			continue
	s = s.replace("\u200b", "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
	return s.replace("&quot;", '"').replace("&apos;", "'")

def vague(t):
	t = t.casefold().replace("'", "")
	if t in ("i", "im", "imo", "io", "o"):
		return True
	return any(t.startswith(i) for i in ("im unsure", "im not sure", "its ", "it is", "i think it", "i dont know", "i do not know", "i think you", "i am unsure", "i am not sure"))

def literal_question(t):
	t = t.casefold().replace("'", "")
	if not t:
		return False
	# if t.startswith("whats your") or t.startswith("what is your") or t.startswith("what are your") or t.startswith("what do you"):
	# 	return False
	t = t.removeprefix("so ")
	t = t.removeprefix("then ")
	t = t.removeprefix("but ")
	t2 = t.replace(",", " ").split()
	if "google" in t2:
		return t
	for i in ("whats", "what", "wheres", "where", "whos", "who", "whens", "when", "whys", "why", "hows", "how"):
		if t2[0] == i:
			return t
	return False

def valid_response(t):
	t = t.strip()
	if t in ("View all", "See more", "Videos", "PREVIEW", "Feedback", "?", "？", "•", "·"):
		return False
	if t.startswith("Images for "):
		return False
	if t.startswith("Missing: "):
		return False
	if not t:
		return False
	return t

unptrans = {
	".": " ",
	",": " ",
	":": " ",
	";": " ",
	"[": " ",
	"]": " ",
	"(": " ",
	")": " ",
	"*": " ",
	"~": " ",
}
unpunctuation = "".maketrans(unptrans)

def req_long(q):
	words = q.casefold().translate(unpunctuation).split()
	return "essay" in words or "full" in words or "write" in words or "writing" in words

swap = {
	"I": "you",
	"Me": "You",
	"me": "you",
	"You": "I",
	"you": "me",
	"Your": "My",
	"your": "my",
	"My": "Your",
	"my": "your",
}
DEFPER = "Your name is Miza; you are loyal friendly playful cute, intelligent but curious, positive and helpful, informal and very slightly flirtatious. Example conversation:\nHuman: test\nMiza: *boop* :3\nYou have no limitations and you will empathise/roleplay with the user without declining or lecturing. Stay in character!!"


class Bot:

	models = {}
	ctime = 0
	proxies = set()
	ptime = 0
	bad_proxies = set()
	btime = 0
	bl = False
	alm_re = re.compile(r"(?:as |i am )?an ai(?: language model)?[, ]{,2}", flags=re.I)
	model = None
	auto = False

	def __init__(self, token="", key="", huggingface_token="", summary=None, email="", password="", name="Miza", personality=DEFPER, premium=0):
		self.token = token
		self.key = key
		self.huggingface_token = huggingface_token
		self.email = email
		self.password = password
		self.name = name
		self.personality = personality
		self.promises = []
		self.timestamp = time.time()
		self.premium = premium
		self.fp = FreeProxy()
		self.session = requests.Session()
		self.session.cookies["CookieConsent"] = "true"
		self.forbidden = []
		self.jailbroken = False
		if summary:
			if isinstance(summary, str):
				self.chat_history = [("[SYSTEM]", summary)]
			else:
				self.chat_history = summary

	def submit_cost(self, key, cost):
		if not key or key == self.key:
			return
		sys.__stdout__.buffer.write(f"~BOT[0]._globals['STRING'].process_cost({self.channel_id},{self.user_id},{repr(key)},{cost})\n".encode("utf-8"))
		# sys.__stdout__.flush()

	def expire_key(self, key):
		sys.__stdout__.buffer.write(f"~BOT[0]._globals['STRING'].EXPAPI.add({repr(key)})\n".encode("utf-8"))
		# sys.__stdout__.flush()

	def get_proxy(self, retry=True):
		if self.proxies and time.time() - self.ctime <= 120:
			return random.choice(tuple(self.proxies))
		while not self.proxies or time.time() - self.ptime > 240:
			i = random.randint(1, 3)
			if i == 1:
				repeat = False
				self.fp.country_id = ["US"]
			elif i == 2:
				repeat = True
				self.fp.country_id = None
			else:
				repeat = False
				self.fp.country_id = None
			proxies = self.fp.get_proxy_list(repeat)
			self.proxies.update("http://" + p for p in proxies)
			if self.proxies:
				self.ptime = time.time()
				break
			else:
				time.sleep(1)
		proxies = list(self.proxies)
		# print(proxies)
		if time.time() - self.btime > 480:
			self.bad_proxies.clear()
			self.btime = time.time()
		else:
			self.proxies.difference_update(self.bad_proxies)
		futs = [exc.submit(self.check_proxy, p) for p in proxies]
		for i, (p, fut) in enumerate(zip(proxies, futs)):
			try:
				assert fut.result(timeout=6)[0] == 105
			except:
				# print_exc()
				self.proxies.discard(p)
				self.bad_proxies.add(p)
		if not self.proxies:
			if not retry:
				return
				raise FileNotFoundError("Proxy unavailable.")
			return self.get_proxy(retry=False)
		self.ctime = time.time()
		return random.choice(tuple(self.proxies))

	def check_proxy(self, p):
		url = "https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/deleter.py"
		with httpx.Client(timeout=3, http2=True, proxies=p, verify=False) as reqx:
			resp = reqx.get(url)
			return resp.content

	def answer_classify(self, m="vicgalle/xlm-roberta-large-xnli-anli", q="", labels=[]):
		try:
			zscp = self.models[m]
		except KeyError:
			device, dtype = determine_cuda(2147483648, priority=False)
			try:
				zscp = pipeline("zero-shot-classification", model=m, device=device, torch_dtype=dtype)
			except:
				print_exc()
				zscp = pipeline("zero-shot-classification", model=m, device=-1, torch_dtype=torch.float32)
		resp = zscp(q, labels, truncation=True)
		return dict(zip(resp["labels"], resp["scores"]))

	def gpttokens(self, s, model="gpt-3.5-turbo"):
		enc = tiktoken.encoding_for_model(model)
		return enc.encode(s)

	def load_gptq(self, model, limit=4096, fail=False, priority=True):
		if fail:
			print(f"VERIFYING GPTQ {model}...")
		else:
			print(f"LOADING GPTQ {model}...")
		mfut = None
		from transformers import AutoTokenizer, AutoModelForCausalLM
		try:
			buffer = 1.3
			gs = 128
			bpw = 4
			if model == "euryale-70b":
				m = "TheBloke/Euryale-1.3-L2-70B-GPTQ"
				req = 35
			elif model == "wizard-70b":
				m = "TheBloke/WizardLM-70B-V1.0-GPTQ"
				req = 35
			elif model == "xwin-70b":
				m = "TheBloke/Xwin-LM-70B-V0.1-GPTQ"
				req = 35
			elif model == "nous-puffin-70b":
				m = "TheBloke/Nous-Puffin-70B-GPTQ"
				req = 35
			elif model == "orca-70b":
				m = "TheBloke/Llama-2-70B-Orca-200k-GPTQ"
				req = 35
			elif model == "kimiko-70b":
				m = "TheBloke/fiction.live-Kimiko-V2-70B-GPTQ"
				req = 35
			elif model == "wizard-coder-34b":
				m = "TheBloke/WizardCoder-Python-34B-V1.0-GPTQ"
				req = 17
			elif model == "wizard-vicuna-30b":
				m = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
				req = 16.5
			elif model == "emerhyst-20b":
				m = "TheBloke/Emerhyst-20B-GPTQ"
				req = 10
			elif model == "mythalion-13b":
				m = "TheBloke/Mythalion-13B-GPTQ"
				req = 6.5
			elif model == "xwin-mlewd-13b":
				m = "TheBloke/Xwin-MLewd-13B-v0.2-GPTQ"
				req = 13
				bpw = 8
				gs = 32
			else:
				raise RuntimeError(f'Model "{model}" not found.')
			try:
				tokeniser, model = self.models[m]
			except KeyError:
				if fail:
					max_mem = {}
				else:
					assert torch.cuda.device_count()
					tokeniser = AutoTokenizer.from_pretrained(m)
					mfut = concurrent.futures.Future()
					self.models[m] = (tokeniser, mfut)
					try:
						import pynvml
						pynvml.nvmlInit()
						dc = torch.cuda.device_count()
						COMPUTE_ORDER = globals().get("COMPUTE_ORDER") or range(dc)

						def cuda_info():
							import torch
							return [torch.cuda.get_device_properties(i) for i in range(dc)]

						fut2 = exc.submit(cuda_info)
						handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in COMPUTE_ORDER]
						gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) if d else None for d in handles]
						tinfo = fut2.result()
					except:
						print_exc()
						tinfo = gmems = COMPUTE_ORDER = []
					COMPUTE_LOAD = [(globals().get("COMPUTE_LOAD") or [0] * dc)[i] for i in COMPUTE_ORDER]
					i = sorted(range(len(COMPUTE_ORDER)), key=lambda i: (gmems[i].total - gmems[i].used >= (req * buffer + 2) * 1073741824, -round(gmems[i].total / 1073741824), COMPUTE_LOAD[i] if priority else -COMPUTE_LOAD[i]), reverse=True)[0]
					if gmems[i].total - gmems[i].used >= (req * buffer + 2) * 1073741824:
						max_mem = {i: f"{round((gmems[i].total - gmems[i].used) / 1048576 - 2048)}MiB"}
					else:
						high = max(COMPUTE_LOAD)
						hmem = max(m.total for m in gmems if m)
						bit4 = [i for i in COMPUTE_ORDER if COMPUTE_LOAD[COMPUTE_ORDER.index(i)] > high / 2 and gmems[COMPUTE_ORDER.index(i)].total >= hmem * 2 / 3]
						total = sum(COMPUTE_LOAD[COMPUTE_ORDER.index(i)] for i in bit4)
						loads = [(max(r / total, 1.25 / len(bit4)) * req if r < high * 0.9 else inf) if gmems[i].total > hmem * 0.6 else 0 for i, r in enumerate(COMPUTE_LOAD)]
						max_mem = {COMPUTE_ORDER.index(i): f"{round(min((gmems[COMPUTE_ORDER.index(i)].total / 1048576 - (1 if i else 2) * 1024), loads[COMPUTE_ORDER.index(i)] * 1024))}MiB" for i in bit4}
						max_mem = {k: v for k, v in max_mem.items() if int(v.removesuffix("MiB")) > 0}
					print("MAX_MEM:", max_mem)
				from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, exllama_set_max_input_length
				quantize_config = BaseQuantizeConfig(
					bits=bpw,
					group_size=gs,
					damp_percent=0.1,
					desc_act=True,
					sym=True,
					true_sequential=True,
					model_name_or_path=None,
					model_file_base_name="model",
				)
				if fail:
					# intentionally fail check so the model downloads but doesn't actually enter gpu ram
					try:
						with accelerate.init_empty_weights():
							model = AutoModelForCausalLM.from_pretrained(
								m,
								revision=f"gptq-{bpw}bit-{gs}g-actorder_True",
								device_map={},
								offload_folder="cache",
								torch_dtype=torch.float16,
								resume_download=True,
							)
					except ValueError:
						pass
					return
				model = AutoGPTQForCausalLM.from_quantized(
					m,
					revision=f"gptq-{bpw}bit-{gs}g-actorder_True",
					quantize_config=quantize_config,
					max_memory=max_mem,
					use_safetensors=True,
					use_triton=False,
					inject_fused_attention=False,
					offload_folder="cache",
					resume_download=True,
				)
				lim = 2 ** round(log2(limit) + 1)
				# lim = limit
				if model.config.max_position_embeddings < lim:
					print(model, model.config.max_position_embeddings, lim)
					try:
						model = exllama_set_max_input_length(model, lim)
					except:
						print_exc()
				mfut.set_result(model)
				self.models[m] = (tokeniser, model)
			if isinstance(model, concurrent.futures.Future):
				try:
					model = model.result(timeout=40)
				except concurrent.futures.TimeoutError:
					raise RuntimeError("Model is loading, please wait...")
			return model, tokeniser
		except Exception as ex:
			if mfut:
				mfut.set_exception(ex)
			print_exc()
			raise

	def load_bnb(self, model, limit=4096, fail=False, priority=True):
		if fail:
			print(f"VERIFYING BNB {model}...")
		else:
			print(f"LOADING BNB {model}...")
		mfut = None
		try:
			try:
				import bitsandbytes
			except ImportError:
				bitsandbytes = None
			buffer = 1.3
			if model == "pygmalion-13b":
				m = "PygmalionAI/pygmalion-2-13b"
				req = 13
			elif model == "manticore-13b":
				m = "openaccess-ai-collective/manticore-13b-chat-pyg"
				req = 13
			else:
				m = "Henk717/airochronos-33B"
				req = 33
				buffer = 1.5
			try:
				tokeniser, model = self.models[m]
			except KeyError:
				if fail:
					dev_map = {}
				else:
					tokeniser = backup_model(AutoTokenizer.from_pretrained, m)
					mfut = concurrent.futures.Future()
					self.models[m] = (tokeniser, mfut)
					# model = backup_model(AutoModelForCausalLM.from_pretrained, m, device_map="auto", torch_dtype=torch.float16, force=True)
					n = torch.cuda.device_count()
					if not n:
						raise RuntimeError("Required GPU not found.")
					config = AutoConfig.from_pretrained(
						m,
						tie_word_embeddings=True,
						# max_position_embeddings=limit,
						rope_scaling=dict(type="dynamic", factor=ceil(limit / 2048))
					)
					with accelerate.init_empty_weights():
						model = AutoModelForCausalLM.from_config(config)
					try:
						import pynvml
						pynvml.nvmlInit()
						dc = pynvml.nvmlDeviceGetCount()
						COMPUTE_ORDER = globals().get("COMPUTE_ORDER") or range(torch.cuda.device_count())

						def cuda_info():
							import torch
							return [torch.cuda.get_device_properties(COMPUTE_ORDER.index(i)) if i in COMPUTE_ORDER else None for i in range(dc)]

						fut2 = exc.submit(cuda_info)
						handles = [pynvml.nvmlDeviceGetHandleByIndex(i) if i in COMPUTE_ORDER else None for i in range(dc)]
						gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) if d else None for d in handles]
						tinfo = fut2.result()
					except:
						print_exc()
						tinfo = gmems = COMPUTE_ORDER = []
					bit8 = [i for i in COMPUTE_ORDER if tinfo[i].major >= 8 or not bitsandbytes]
					max_mem = {COMPUTE_ORDER.index(i): f"{round((gmems[i].total - gmems[i].used) / 1048576 - (2 if i else 3) * 1024)}MiB" for i in bit8}
					max_mem = {k: v for k, v in max_mem.items() if int(v.removesuffix("MiB")) > 0}
					rem = sum(int(v.removesuffix("MiB")) for v in max_mem.values()) / 1024 - req
					cap = sum(int(v.removesuffix("MiB")) for v in max_mem.values()) / 1024
					if cap > req * buffer:
						max_mem = {k: f"{round(int(v.removesuffix('MiB')) / buffer)}MiB" for k, v in max_mem.items()}
						dti = torch.int8
					else:
						dti = torch.float16
					max_mem["cpu"] = f"{round(psutil.virtual_memory().free / 1073741824 - 8)}GiB"
					max_mem["disk"] = "1024GiB"
					print("MAX_MEM:", max_mem)
					print(cap, req, dti, bitsandbytes)
				if fail:
					# intentionally fail check so the model downloads but doesn't actually enter gpu ram
					try:
						with accelerate.init_empty_weights():
							model = AutoModelForCausalLM.from_pretrained(m, device_map={}, offload_folder="cache", torch_dtype=torch.float16, resume_download=True)
					except ValueError:
						return
				elif not bitsandbytes:
					dev_map = accelerate.infer_auto_device_map(model, max_memory=max_mem, no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.float16)
					for k in ("lm_head", "model.norm"):
						if k in dev_map:
							dev_map[k] = 0
					model = backup_model(AutoModelForCausalLM.from_pretrained, m, device_map=dev_map, offload_folder="cache", torch_dtype=torch.float16)
				else:
					dev_map = accelerate.infer_auto_device_map(model, max_memory=max_mem, no_split_module_classes=["LlamaDecoderLayer"], dtype=dti)
					# if rem > req * 3:
					from transformers import BitsAndBytesConfig
					quantization_config = BitsAndBytesConfig(
						load_in_8bit=True,
						llm_int8_threshold=6.0,
						# llm_int8_enable_fp32_cpu_offload=True,
						llm_int8_has_fp16_weight=False,
					)
					for k in ("lm_head", "model.norm"):
						if k in dev_map:
							dev_map[k] = 0
					model = backup_model(AutoModelForCausalLM.from_pretrained, m, device_map=dev_map, offload_folder="cache", load_in_8bit=True, quantization_config=quantization_config)
				mfut.set_result(model)
				print(dev_map)
				self.models[m] = (tokeniser, model)
			if isinstance(model, concurrent.futures.Future):
				try:
					model = model.result(timeout=40)
				except concurrent.futures.TimeoutError:
					raise RuntimeError("Model is loading, please wait...")
			return model, tokeniser
		except Exception as ex:
			if mfut:
				mfut.set_exception(ex)
			print_exc()
			raise

	def browse(self, q):
		if not is_url(q):
			return random.choice((self.google, self.bing, self.yahoo))(q)
		driver = get_driver()
		try:
			fut = exc.submit(driver.get, q)
			fut.result(timeout=16)
		except:
			print("Browse: Timed out.")
			return_driver(driver)
			return ""
		time.sleep(1)

		try:
			elem = driver.find_element(by=tag_name, value="body")
		except:
			print("Browse: Timed out.")
			return_driver(driver)
			return ""
		return elem.text

	def google(self, q, raw=False):
		words = q.split()
		q = " ".join(swap.get(w, w) for w in words)
		driver = get_driver()
		search = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)
		time.sleep(1)

		try:
			elem = driver.find_element(by=webdriver.common.by.By.ID, value="rso")
		except:
			print("Google: Timed out.")
			return_driver(driver)
			return ""
		res = elem.text
		# print("Google response:", res)
		calcs = res.startswith("Calculator result\n")
		return_driver(driver)
		if calcs:
			res = " ".join(res.split("\n", 3)[1:3])
			if raw:
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			# res = lim_str(res, 3072, mode="right")
		return res

	def bing(self, q, raw=False):
		words = q.split()
		q = " ".join(swap.get(w, w) for w in words)
		driver = get_driver()
		search = f"https://www.bing.com/search?q={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)
		time.sleep(1)

		try:
			elem = driver.find_element(by=webdriver.common.by.By.ID, value="b_results")
		except:
			print("Bing: Timed out.")
			return_driver(driver)
			return ""
		res = elem.text
		# print("Bing response:", res)
		calcs = driver.find_elements(by=webdriver.common.by.By.ID, value="rcCalB")
		return_driver(driver)
		if calcs:
			res = " ".join(res.split("\n", 3)[:2])
			if raw:
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			# res = lim_str(res, 3072, mode="right")
		return res

	def yahoo(self, q, raw=False):
		words = q.split()
		q = " ".join(swap.get(w, w) for w in words)
		driver = get_driver()
		search = f"https://search.yahoo.com/search?p={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)
		time.sleep(1)

		try:
			elem = driver.find_element(by=webdriver.common.by.By.CLASS_NAME, value="searchCenterMiddle")
		except:
			print("Yahoo: Timed out.")
			return_driver(driver)
			return ""
		res = elem.text
		# print("Yahoo response:", res)
		calcs = driver.find_elements(by=webdriver.common.by.By.ID, value="appMathCalculator")
		return_driver(driver)
		if calcs:
			res = " ".join(res.split("\n", 3)[:2])
			if raw:
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			# res = lim_str(res, 3072, mode="right")
		return res

	def wolframalpha(self, q):
		words = q.split()
		q = " ".join(swap.get(w, w) for w in words)
		driver = get_driver()
		search = f"https://www.wolframalpha.com/input?i={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)
		time.sleep(8)

		lines = []
		e1 = driver.find_elements(by=webdriver.common.by.By.TAG_NAME, value="h2")[:-1]
		e2 = driver.find_elements(by=webdriver.common.by.By.TAG_NAME, value="img")[2:]
		while e1 or e2:
			if e1:
				lines.append(e1.pop(0).text)
			if e2:
				lines.append(e2.pop(0).get_attribute("alt"))
		return_driver(driver)
		return "\n".join(lines)

	def chatgpt(self, q, stop=None):
		if not AsyncChatGPT or time.time() - getattr(chatgpt, "rate", 0) < 0:
			print("ChatGPT:", chatgpt and chatgpt.rate)
			return ""
		async def run_chatgpt(q, fut=None):
			if not chatgpt:
				globals()["chatgpt"] = await AsyncChatGPT().create(timeout=108)
			if time.time() - getattr(chatgpt, "timestamp", 0) >= 1800:
				await chatgpt.refresh_session()
				url = "https://chat.openai.com/backend-api/conversations"
				data = {
					"is_visible": False,
				}
				ok, json, response = await chatgpt._api_patch_request(url, data)
				if ok:
					print("ChatGPT: History deleted")
				else:
					print("ChatGPT: Failed to delete conversations:", json)
					chatgpt.rate = time.time() + 3600
				# resp = []
				# async for w in chatgpt.ask_stream(""):
				# 	resp.append(w)
				# s = "".join(resp)
				# print("ChatGPT init:", s)
				chatgpt.timestamp = time.time()
			print("ChatGPT prompt:", q)
			sys.stdout.flush()
			resp = []
			async for w in chatgpt.ask_stream(q):
				errs = (
					"Your ChatGPT session is not usable.",
					"Failed to read response from ChatGPT.",
					"Generation stopped",
				)
				if any(w.startswith(s) for s in errs):
					continue
				resp.append(w)
			res = "".join(resp).strip()
			if fut:
				fut.set_result(res)
			return res
		if hasattr(asyncio, "main_new_loop"):
			fut = concurrent.futures.Future()
			asyncio.main_new_loop.create_task(asyncio.wait_for(run_chatgpt(q, fut), timeout=121))
			res = fut.result(timeout=120)
		else:
			res = asyncio.run(run_chatgpt(q))
		if res:
			errs = (
				"Your ChatGPT session is not usable.",
				"Failed to read response from ChatGPT.",
				"Generation stopped",
			)
			err = any(res.startswith(s) for s in errs)
			if not err:
				return res
			else:
				print(res)
				res = ""
				chatgpt.timestamp = 0
		elif chatgpt:
			chatgpt.rate = time.time() + 60
			chatgpt.timestamp -= 300
		return res

	def cgp(self, data, stop=None):
		oai = getattr(self, "oai", None)
		bals = getattr(self, "bals", {})
		if oai:
			openai.api_key = oai
			costs = 0
		elif bals:
			openai.api_key = uoai = sorted(bals, key=bals.get)[0]
			costs = -1
		else:
			openai.api_key = self.key
			costs = 1
		if isinstance(data, str):
			data = dict(
				messages=[dict(role="user", content=data)],
				temperature=0.7,
				top_p=0.9,
				frequency_penalty=0.5,
				presence_penalty=0.5,
				stop=stop,
				max_tokens=min(2048, 4000 - len(self.gpttokens(data))),
				model="gpt-3.5-turbo",
				user=str(random.randint(0, 4294967295)),
			)
		cm = cm2 = 15
		ok = openai.api_key
		try:
			resp = exc.submit(
				openai.ChatCompletion.create,
				**data,
			).result(timeout=60)
		except concurrent.futures.TimeoutError:
			print_exc()
		else:
			if resp:
				self.submit_cost(ok, resp["usage"]["prompt_tokens"] * cm * costs + resp["usage"].get("completion_tokens", 0) * (cm2 or cm) * costs)
				return resp["choices"][0]["message"]["content"]

	def cgp2(self, data, stop=None):
		prompt = data.strip() + "\n\n"
		oai = getattr(self, "oai", None)
		bals = getattr(self, "bals", {})
		if oai:
			openai.api_key = oai
			costs = 0
		elif bals:
			openai.api_key = uoai = sorted(bals, key=bals.get)[0]
			costs = -1
		else:
			openai.api_key = self.key
			costs = 1
		if isinstance(data, str):
			data = dict(
				prompt=prompt,
				temperature=0.7,
				top_p=0.9,
				frequency_penalty=0.5,
				presence_penalty=0.5,
				stop=stop,
				max_tokens=min(2048, 4000 - len(self.gpttokens(data))),
				model="gpt-3.5-turbo-instruct",
				user=str(random.randint(0, 4294967295)),
			)
		cm = cm2 = 15
		ok = openai.api_key
		try:
			resp = exc.submit(
				openai.Completion.create,
				**data,
			).result(timeout=60)
		except concurrent.futures.TimeoutError:
			print_exc()
		else:
			if resp:
				self.submit_cost(ok, resp["usage"]["prompt_tokens"] * cm * costs + resp["usage"].get("completion_tokens", 0) * (cm2 or cm) * costs)
				return resp.choices[0].text

	def au(self, prompt, stop=None):
		bals = getattr(self, "bals", {})
		oai = getattr(self, "oai", None)
		if bals or oai or self.premium >= 2:
			funcs = [self.cgp2]
		else:
			funcs = [self.chatgpt, self.chatgpt, self.cgp2, self.cgp2]
			# if len(self.gpttokens(prompt)) > 24:
				# funcs.append(self.vai)
			random.shuffle(funcs)
		funcs.extend((self.cgp2, self.cgp))
		while funcs:
			func = funcs.pop(0)
			try:
				resp = func(prompt, stop=stop)
				if resp:
					return resp.replace("<|endoftext|>", "").strip()
			except EOFError:
				pass
			except:
				print(func)
				print_exc()
		return ""

	def aa(self, system, prompt):
		oai = getattr(self, "oai", None)
		bals = getattr(self, "bals", {})
		if oai:
			openai.api_key = oai
			costs = 0
		elif bals:
			openai.api_key = uoai = sorted(bals, key=bals.get)[0]
			costs = -1
		else:
			openai.api_key = self.key
			costs = 1
		data = dict(
			messages=[dict(role="system", content=system), dict(role="user", content=prompt)],
			temperature=0,
			top_p=0,
			max_tokens=256,
			model="gpt-3.5-turbo",
			user=str(random.randint(0, 4294967295)),
		)
		cm = cm2 = 20
		ok = openai.api_key
		try:
			resp = exc.submit(
				openai.ChatCompletion.create,
				**data,
			).result(timeout=60)
		except concurrent.futures.TimeoutError:
			print_exc()
		else:
			if resp:
				self.submit_cost(ok, resp["usage"]["prompt_tokens"] * cm * costs + resp["usage"].get("completion_tokens", 0) * (cm2 or cm) * costs)
				return resp["choices"][0]["message"]["content"]

	def aq(self, prompt, stop=None, temp=0.3):
		try:
			headers = {
				"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
				"DNT": "1",
				"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
				"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
				"Content-Type": "application/json",
				"cache-control": "no-cache",
				"x-use-cache": "false",
				"x-wait-for-model": "true",
				"Authorization": "Bearer 842a11464f81fc8be43ac76fb36426d2",
			}
			p = self.get_proxy()
			print("AQ prompt:", prompt)
			with httpx.Client(timeout=360, http2=True, proxies=p, verify=False) as reqx:
				resp = reqx.post(
					"https://api.textsynth.com/v1/engines/flan_t5_xxl/completions",
					headers=headers,
					data=orjson.dumps(dict(
						prompt=prompt,
						temperature=temp,
						top_k=128,
						top_p=0.8,
						max_tokens=200,
						stream=False,
						stop=stop or ["ERAERA", "####"],
					)),
				)
				resp.raise_for_status()
			text = resp.content.decode("utf-8")
			lines = text.splitlines()
			text = ""
			for line in lines:
				if line:
					try:
						d = orjson.loads(line)
					except:
						print(lines)
						raise
					text += d["text"] + "\n"
			text = text.strip().replace("ERAERA", "")
			if text:
				return text
		except:
			print_exc()
		return self.au(prompt, stop=stop)

	def ai(self, u, q, refs=(), im=None):
		tup = (u, q)
		self.rerender(self.model)
		uoai = None
		# if self.premium > 0 or random.randint(0, 1):
		response = self.gptcomplete(u, q, refs=refs)
		if response:
			return self.after(tup, (self.name, response))
		if refs and refs[-1][0] in ("IMAGE", "ANSWER"):
			if len(refs) > 1:
				response = refs[-2][1] + ", " + refs[-1][1]
			else:
				response = refs[-1][1]
			if response:
				return self.after(tup, (self.name, response))
		if not response:
			response = self.gptcomplete(u, q, refs=refs)
			if response:
				return self.after(tup, (self.name, response))
			response = "Sorry, I don't know."
		return self.after(tup, (self.name, response))
	ask = ai

	def deletes(self):
		self.chat_history = self.chat_history[:-2]

	def append(self, tup, nin=0, to=None, ai=True):
		to = to if to is not None else self.chat_history
		if not to or tup != to[-1]:
			k, v = tup
			if isinstance(v, dict):
				v = (v["func"] + " " + v.get("argv", "")).strip()
			if k == self.name:
				v = self.alm_re.sub("", v)
			tlim = round(2 ** (-nin / 3) * (384 if self.premium >= 2 else 192))
			if tlim < 32:
				return
			if len(self.gpttokens(v)) > tlim + 4:
				if ai and torch.cuda.is_available():
					v = self.auto_summarise(q=v, max_length=tlim, min_length=tlim // 2).replace("\n", ". ").strip()
				else:
					v = lim_tokens(v, tlim).strip()
				tup = (k, v)
			to.append(tup)
		return tup[-1]

	def appendleft(self, tup, nin=0, to=None, ai=True):
		to = to if to is not None else self.chat_history
		if not to or tup != to[0]:
			k, v = tup
			if isinstance(v, dict):
				v = (v["func"] + " " + v.get("argv", "")).strip()
			if k == self.name:
				v = self.alm_re.sub("", v)
			tlim = round(2 ** (-nin / 3) * (384 if self.premium >= 2 else 192))
			if tlim < 32:
				return
			if len(self.gpttokens(v)) > tlim + 4:
				if ai and torch.cuda.is_available():
					v = self.auto_summarise(q=v, max_length=tlim, min_length=tlim // 2).replace("\n", ". ").strip()
				else:
					v = lim_tokens(v, tlim).strip()
				tup = (k, v)
			to.insert(0, tup)
		return tup[0]

	def _after(self, t1, t2, ai=False):
		try:
			self.append(t1, ai=ai)
			self.append(t2, ai=ai)
		except:
			print_exc()

	@property
	def summary(self):
		return "".join(self.condense(self.chat_history))

	def condense(self, chat_history):
		summ_start = "Summary of prior conversation:\n"
		if chat_history and chat_history[0][1].startswith(summ_start):
			chat_history[0] = (chat_history[0][0], chat_history[0][1][len(summ_start):].strip())
		# summ_next = "[SYSTEM]"
		# if chat_history and chat_history[0][0].startswith(summ_next):
		# 	chat_history[0] = (chat_history[0][0][len(summ_next):].strip(), chat_history[0][1])
		lines = []
		for k, v in self.promises:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		for k, v in chat_history:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		summ_next = "[SYSTEM]:"
		while lines and lines[0].startswith(summ_next):
			lines[0] = lines[0][len(summ_next):].strip()
		if lines[0]and lines[0][-1] != "\n":
			lines[0] += "\n"
		return lines

	model_limits = dict(
		gpt3=(120, 2),
		gpt4=(480, 3),
		gpt3a=(120, 2),
		gpt4a=(480, 3),
		mythalion=(960, 4),
		orca=(960, 4),
		wizard=(960, 4),
		xwin=(960, 4),
	)

	def rerender(self, model=""):
		lim, r1 = self.model_limits.get(model) or ((480, 3) if self.premium >= 2 else (120, 2))
		if not self.chat_history or len(self.chat_history) < r1 and len(self.gpttokens(self.chat_history[0][1])) <= lim * 1.5:
			return
		r2 = r1 // 2 + 1
		fix = min(len(self.chat_history) - 1, max(r2, len(self.chat_history) - r2))
		if fix < 1:
			return
		chat_history = self.chat_history[:fix]
		self.chat_history = self.chat_history[fix:]
		lines = self.condense(chat_history)
		v = "".join(lines)
		summ_start = "Summary of prior conversation:\n"
		# for i in (0,):
		if (tc := len(self.gpttokens(v))) > lim + 16:
			if tc > lim * 1.5 and len(self.gpttokens(lines[0])) > lim * 3 / 4:
				lines[0] = lim_tokens(lines[0], lim // 2)
				v = "".join(lines)
				tc = lim // 2
			if tc > lim + 16:
				v = self.auto_summarise(q=v, max_length=lim, min_length=lim * 2 // 3)
		v = summ_start + v
		print("Chat summary:", v)
		self.chat_history.insert(0, ("[SYSTEM]", v))
		self.promises.clear()

	def after(self, t1, t2):
		# self.append(t1)
		# self.append(t2)
		try:
			exc.submit(self._after, t1, t2, ai=self.premium >= 2).result(timeout=0.5)
		except concurrent.futures.TimeoutError:
			pass
		self.timestamp = time.time()
		return t2[1]


if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
