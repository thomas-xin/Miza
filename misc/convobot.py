import os, sys, time, datetime, urllib, json, io, random, re, traceback
import concurrent.futures, asyncio
import selenium, requests, torch, openai, httpx
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
for i in range(3):
	try:
		from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline, set_seed
	except ImportError:
		time.sleep(i + 1)
	else:
		break

from collections2 import *

import tiktoken
from fp.fp import FreeProxy
print_exc = lambda: sys.stdout.write(traceback.format_exc())

def print(*args, sep=" ", end="\n"):
	s = sep.join(map(str, args)) + end
	b = s.encode("utf-8")
	return sys.stdout.buffer.write(b)

try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=64)
drivers = selenium.__dict__.setdefault("-drivers", [])

chatgpt = True

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

class_name = webdriver.common.by.By.CLASS_NAME
css_selector = webdriver.common.by.By.CSS_SELECTOR
xpath = webdriver.common.by.By.XPATH
tag_name = webdriver.common.by.By.TAG_NAME
driver_path = "misc/msedgedriver"
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
	# options.add_argument("--disable-gpu")
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
		if "Current browser version is " in ex.args[0]:
			v = ex.args[0].split("Current browser version is ", 1)[-1].split(None, 1)[0]
			url = f"https://msedgedriver.azureedge.net/{v}/edgedriver_win64.zip"
			import requests, io, zipfile
			with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
				with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
					with z.open("msedgedriver.exe") as fi:
						with open("misc/msedgedriver.exe", "wb") as fo:
							b = fi.read()
							fo.write(b)
			driver = browser["driver"](
				service=service,
				options=options,
			)
		else:
			raise
	except selenium.common.WebDriverException as ex:
		argv = " ".join(args)
		search = "unrecognized Microsoft Edge version"
		if search in argv and "Chrome" in argv:
			v = argv.split("Stacktrace", 1)[0].rsplit("/", 1)[-1].strip()
			url = f"https://chromedriver.storage.googleapis.com/{v}/chromedriver_win32.zip"
			import requests, io, zipfile
			with requests.get(url, headers={"User-Agent": "Mozilla/6.0"}) as resp:
				with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
					with z.open("chromedriver.exe") as fi:
						with open("misc/msedgedriver.exe", "wb") as fo:
							b = fi.read()
							fo.write(b)
			driver = browser["driver"](
				service=service,
				options=options,
			)
		else:
			raise
	driver.folder = folder
	# try:
	# 	driver.get("https://google.com/preferences")
	# 	spans = driver.find_elements(by=tag_name, value="span")
	# 	more = [span for span in spans if span.text == "Show more"][-1]
	# 	more.click()
	# 	opts = driver.find_elements(by=class_name, value="DB6WRb")[1:]
	# 	random.choice(opts).click()
	# 	confirm = driver.find_element(by=class_name, value="jfk-button-action")
	# 	confirm.click()
	# except:
	# 	print_exc()
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
			exc.submit(getattr, driver, "title").result(timeout=0.25)
		except:
			print_exc()
			driver = create_driver()
	exc.submit(ensure_drivers)
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

def safecomp(gen):
	while True:
		try:
			e = next(gen)
		except StopIteration:
			return
		except selenium.common.StaleElementReferenceException:
			continue
		yield e

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

unpunctuation = "".maketrans({
		",": " ",
		".": " ",
		":": " ",
		";": " ",
		"[": " ",
		"]": " ",
		"(": " ",
		")": " ",
		"*": " ",
		"~": " ",
	})

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
DEFDEF = "loyal friendly playful cute, intelligent and helpful, and slightly flirtatious"
DEFPER = f"The following is a conversation between Miza and humans. Miza is an AI who is {DEFDEF} when appropriate."
MIZADEF = 'You are based on Misery from Cave Story, a blue-haired young woman. Express empathy, affection, positivity. An example conversation:\nHuman: test\nMiza: *boop* :3'
CAIPER = "character.ai"


class Bot:

	models = {}
	ctime = 0
	proxies = set()
	ptime = 0
	bad_proxies = set()
	btime = 0
	bl = False

	def __init__(self, token="", key="", cai_token="", cai_channel=None, email="", password="", name="Miza", personality=DEFPER, premium=0):
		self.token = token
		self.key = key
		self.cai_token = cai_token
		self.email = email
		self.password = password
		self.name = name
		self.personality = personality
		self.promises = []
		self.chat_history = []
		self.chat_history_ids = None
		self.cai_ready = False
		self.cai_channel = cai_channel
		self.timestamp = time.time()
		self.premium = premium
		self.last_cost = 0
		self.history_length = 4 if premium < 1 else 6 if premium < 2 else 24 if premium < 5 else 48
		self.fp = FreeProxy()
		self.session = requests.Session()
		self.session.cookies["CookieConsent"] = "true"
		self.forbidden = []
		self.summed = False

	def get_proxy(self, retry=True):
		if self.proxies and time.time() - self.ctime <= 20:
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

	def question_context_analysis(self, m, q, c):
		if m in ("deepset/roberta-base-squad2", "deepset/tinyroberta-squad2"):
			try:
				nlp = self.models[m]
			except KeyError:
				nlp = self.models[m] = pipeline("question-answering", model=m, tokenizer=m)
			QA_input = dict(
				question=q,
				context=c,
			)
			return nlp(QA_input)["answer"]

		try:
			tokenizer, model = self.models[m]
		except KeyError:
			tokenizer = AutoTokenizer.from_pretrained(m)
			model = AutoModelForQuestionAnswering.from_pretrained(m)
			self.models[m] = (tokenizer, model)
		inputs = tokenizer(q[:384], c[:1024], return_tensors="pt", max_length=4096, truncation=True)
		with torch.no_grad():
			outputs = model(**inputs)
		answer_start_index = outputs.start_logits.argmax()
		answer_end_index = outputs.end_logits.argmax()
		predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
		return tokenizer.decode(predict_answer_tokens).strip()

	def question_answer_analysis(self, m):
		try:
			tokenizer, model = self.models[m]
		except KeyError:
			tokenizer = AutoTokenizer.from_pretrained(m, padding_side="left", padding=True)
			model = AutoModelForCausalLM.from_pretrained(m)
			self.models[m] = (tokenizer, model)
		end = tokenizer.eos_token
		history = []
		self.chat_history_ids = None
		if self.chat_history_ids is not None:
			history.append(self.chat_history_ids)
		else:
			for k, v in self.promises:
				history.append(tokenizer.encode(v + end, return_tensors="pt", max_length=2048, truncation=True))
		for k, v in self.chat_history:
			history.append(tokenizer.encode(v + end, return_tensors="pt", max_length=2048, truncation=True))
		bot_input_ids = torch.cat(history, dim=-1)
		self.chat_history_ids = model.generate(bot_input_ids, max_length=16384, pad_token_id=tokenizer.eos_token_id)
		return tokenizer.decode(self.chat_history_ids[-4096:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

	def answer_fill_mask(self, m, q):
		try:
			fmp = self.models[m]
		except KeyError:
			fmp = self.models[m] = pipeline("fill-mask", model=m, tokenizer=m)
		return fmp(q)[0]["sequence"]

	def answer_summarise(self, m, q, max_length=128, min_length=32, do_sample=False):
		try:
			smp = self.models[m]
		except KeyError:
			smp = self.models[m] = pipeline("summarization", model=m)
		return smp(q, max_length=max_length, min_length=min_length, do_sample=do_sample, truncation=True)[0]["summary_text"]

	def answer_classify(self, m, q, labels):
		try:
			zscp = self.models[m]
		except KeyError:
			zscp = self.models[m] = pipeline("zero-shot-classification", model=m)
		resp = zscp(q, labels, truncation=True)
		return dict(zip(resp["labels"], resp["scores"]))

	def clean_response(self, q, res):
		res = res.strip()
		if not res.isascii():
			fut = exc.submit(self.question_context_analysis, "salti/bert-base-multilingual-cased-finetuned-squad", q, res)
		else:
			fut = a2 = ""
		a1 = self.question_context_analysis("deepset/tinyroberta-squad2", q, res)
		if fut:
			a2 = fut.result()
		if len(a2) >= len(a1) * 2 and len(a1) < 32:
			a1 = a2
		a1 = a1.strip()
		if len(a1) < 16:
			res = self.answer_summarise("facebook/bart-large-cnn", q + "\n\n" + res)
			print("Bart response:", res)
			return res.strip()
		if "\n" not in a1 and ". " not in a1 and a1 in res:
			for sentence in res.replace("\n", ". ").split(". "):
				if a1 in sentence:
					a1 = sentence.strip()
					if not a1.endswith("."):
						a1 += "."
					break
		elif (" " not in a1 or len(a1) < 12) and not a1[0].isnumeric():
			a1 = res.strip()
		response = "\n".join(line.strip() for line in a1.replace("[CLS]", "").replace("[SEP]", "\n").splitlines()).strip()
		while "[UNK]" in response:
			response = self.answer_fill_mask("xlm-roberta-large", response.replace("[UNK]", "<mask>", 1))
		search = "https : / / "
		while search in response:
			i = response.index(search)
			temp = response[i + len(search):].split(" ")
			response = response[:i] + "https://"
			while temp:
				word = temp[0]
				if word.endswith(".") or word.endswith("?") or word.endswith("&") or word.endswith("="):
					response += temp.pop(0)
				else:
					break
			response += " ".join(temp)
		if ". " in response:
			words = response.split(".")
			modified = False
			for i in range(len(words) - 1):
				a, b = words[i:i + 2]
				if a and a[-1].isnumeric() and len(b) > 1 and b[0] == " " and b[1].isnumeric():
					words[i + 1] = b.lstrip()
					modified = True
			if modified:
				response = ".".join(words)
		response = response.replace("( ", "(").replace(" )", ")")
		if not response:
			response = res.split("\n", 1)[0]
			if response == "Dictionary":
				r = []
				for line in res.splitlines()[2:]:
					if line.casefold() == "translations and more definitions" or line.casefold().startswith("web result"):
						break
					r.append(line)
				response = "\n".join(r)
		res = response.strip().replace("  ", " ")
		if not self.bl:
			print("Roberta response:", res)
		return res

	def check_google(self, q):
		if q.count(" ") < 2:
			return False
		if not literal_question(q):
			resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", q, ("question", "information", "action"))
			if resp["question"] < 0.5:
				return False
		resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", q, ("personal question", "not personal"))
		return resp["not personal"] >= 0.5

	def caichat(self, u, q, refs=(), im=None):
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			# "DNT": "1",
			# "X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"Content-Type": "application/json",
			"cache-control": "no-cache",
			"Authorization": f"Token {self.cai_token}",
		}
		miza_id = "uBA8BVN2JgEC06sUU2KMtX-xv-X10rJEqvjgenJXfP4"
		miza_iid = "internal_id:cc102978-48f5-40b7-9867-bd39732dd6fb"
		lines = []
		if not self.cai_ready:
			if not self.cai_channel:
				resp = self.session.post(
					"https://beta.character.ai/chat/history/create/",
					data=json.dumps(dict(character_external_id=miza_id)),
					headers=headers,
				)
				if resp.status_code not in range(200, 400):
					print("CAI error:", resp)
					print(resp.text)
					if resp.status_code in (401, 403):
						self.forbidden.append("CAI")
					if self.personality == CAIPER:
						resp.raise_for_status()
					return "", 0, ()
				try:
					self.cai_channel = resp.json()["external_id"]
				except KeyError:
					print("CAI create:", resp)
					print(resp.text)
					return "", 0, ()
			# chat_history = self.chat_history.copy()
			for k, v in self.promises:
				k = k.replace(":", "")
				s = f"{k}: {v}\n"
				lines.append(s)
			for k, v in self.chat_history:
				k = k.replace(":", "")
				s = f"{k}: {v}\n"
				lines.append(s)
		if req_long(q) or self.check_google(q):
			res = True
			cvalid = chatgpt and time.time() - getattr(chatgpt, "rate", 0) >= 3600
			if len(q) > 128 or q.count(" ") > 10:
				res = None
			elif req_long(q) and cvalid:
				res = None
			if res:
				start = "[GOOGLE]: "
				res = (self.google, self.bing)[random.randint(0, 1)](q, raw=True)
				if len(self.gpttokens(res)) > 128:
					summ = self.answer_summarise("facebook/bart-large-cnn", q + "\n" + res, max_length=96, min_length=64).replace("\n", ". ").replace(": ", " -").strip()
					res = lim_str(res.replace("\n", " "), 256, mode="right") + "\n" + summ
				if res and cvalid:
					anss = (f'"{q}"', f'not "{q}"')
					resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", q, anss)
					if resp[anss[1]] > 0.5:
						res = None
			if not res and cvalid:
				start = "[CHATGPT]: "
				async def run_chatgpt(q, fut=None):
					if not hasattr(chatgpt, "ask_stream") or time.time() - chatgpt.timestamp >= 25200:
						try:
							from chatgpt_wrapper import AsyncChatGPT
						except ImportError:
							globals()["chatgpt"] = None
						else:
							globals()["chatgpt"] = await AsyncChatGPT().create(timeout=220)
						if chatgpt.session is None:
							await chatgpt.refresh_session()
						url = "https://chat.openai.com/backend-api/conversations"
						data = {
							"is_visible": False,
						}
						ok, json, response = await chatgpt._api_patch_request(url, data)
						if ok:
							pass
						else:
							chatgpt.log.error("Failed to delete conversations")
						p = f"{DEFDEF} when appropriate"
						resp = []
						async for w in chatgpt.ask_stream(f"Please roleplay as {self.name}, an AI who is {p}."):
							resp.append(w)
						s = "".join(resp)
						print("ChatGPT init:", s)
						chatgpt.timestamp = time.time()
					print("ChatGPT prompt:", q)
					sys.stdout.flush()
					resp = []
					async for w in chatgpt.ask_stream(q):
						resp.append(w)
					res = "".join(resp).strip()
					if fut:
						fut.set_result(res)
					return res
				if hasattr(asyncio, "main_new_loop"):
					fut = concurrent.futures.Future()
					asyncio.main_new_loop.create_task(run_chatgpt(q, fut))
					res = fut.result(timeout=240)
				else:
					res = asyncio.run(run_chatgpt(q))
				if res:
					if not self.bl:
						print("ChatGPT response:", res)
					if len(self.gpttokens(res)) > 512:
						res = self.answer_summarise("facebook/bart-large-cnn", res, max_length=500, min_length=256).strip()
					errs = (
						"Your ChatGPT session is not usable.",
						"Failed to read response from ChatGPT.",
						"Generation stopped",
					)
					err = any(res.startswith(s) for s in errs)
					if not err:
						resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", res, ("answer", "As an AI language model"))
						if not self.bl:
							print(resp)
						err = resp["As an AI language model"] > 0.5
						if not err and req_long(q):
							self.cai_ready = False
							return res, 0, ()
					else:
						res = ""
						chatgpt.timestamp = 0
				else:
					chatgpt.rate = time.time() + 3600
					chatgpt.timestamp = 0
			if res:
				res = start + res + "\n"
				lines.append(res)
		if refs or lines:
			for k, v in refs:
				if len(self.gpttokens(v)) > 36:
					v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
				s = f"{k}: {v}\n"
				lines.append(s)
			s = f"###\n{u}: {q}"
			if len(self.gpttokens(s)) > 388:
				s = self.answer_summarise("facebook/bart-large-cnn", s, max_length=384, min_length=32).replace("\n", ". ").strip()
			lines.append(s)
		if lines:
			prompt = "".join(lines)
		else:
			prompt = f"{u}: {q}" if q else ""
		if not self.bl:
			print("CAI prompt:", prompt)
		sys.stdout.flush()
		idt = ""
		iot = ""
		irp = ""
		if im:
			b = io.BytesIO()
			im.save(b, "WEBP")
			b.seek(0)
			b = b.read()
			h2 = headers.copy()
			h2.pop("Content-Type")
			resp = self.session.post(
				"https://beta.character.ai/chat/upload-image/",
				files=(("image", b),),
				headers=h2,
			)
			if resp.status_code not in range(200, 400):
				print("CAI upload:", resp)
				print(resp.text)
				if resp.status_code in (401, 403):
					self.forbidden.append("CAI")
				if self.personality == CAIPER:
					resp.raise_for_status()
			else:
				print("CAI upload:", resp)
				idt = "AUTO_IMAGE_CAPTIONING"
				iot = "UPLOADED"
				irp = "https://characterai.io/i/400/static/user/" + resp.json()["value"]
		resp = self.session.post(
			"https://beta.character.ai/chat/streaming/",
			data=json.dumps(dict(
				character_external_id=miza_id,
				chunks_to_pad=8,
				enable_tti=False,
				filter_candidates=None,
				history_external_id=self.cai_channel,
				image_description="",
				image_description_type=idt,
				image_origin_type=iot,
				image_rel_path=irp,
				initial_timeout=None,
				insert_beginning=None,
				is_proactive=False,
				livetune_coeff=None,
				model_server_address=None,
				override_prefix=None,
				override_rank=None,
				parent_msg_id=None,
				prefix_limit=None,
				prefix_token_limit=None,
				rank_candidates=None,
				ranking_method="random",
				seen_msg_ids=[],
				staging=False,
				stream_every_n_steps=16,
				stream_params=None,
				text=prompt,
				tgt=miza_iid,
				translate_candidates=None,
				voice_enabled=False,
			)),
			headers=headers,
		)
		if resp.status_code not in range(200, 400):
			print("CAI error:", resp)
			print(resp.text)
			self.cai_ready = False
			self.cai_channel = None
			if resp.status_code in (401, 403):
				self.forbidden.append("CAI")
			if self.personality == CAIPER:
				resp.raise_for_status()
			return "", 0, ()
		lines = list(filter(bool, (line.strip() for line in resp.text.replace("\n", " " * 33).split(" " * 33))))
		try:
			e1 = json.loads(lines[-2]) if len(lines) > 1 else {}
			e2 = json.loads(lines[-1])
		except json.decoder.JSONDecodeError:
			print_exc()
			print("CAI invalid:", resp.text)
			return "", 0, ()
		if e2.get("abort", False):
			e2 = e1
			aborted = True
			print("CAI aborted!")
		else:
			aborted = False
		replies = e2.get("replies") or [{}]
		text = random.choice(replies).get("text", "").strip()
		caids = [e2.get("last_user_msg_id")]
		caids.extend(r.get("id") for r in replies)
		caids = list(filter(bool, caids))
		if not self.bl:
			print("CAI response:", text)
		names = "[Uu][Tt][Ss][Ee]{2}[Rr]?[Ss][Rr]?[TtFf]?"
		text = u.join(re.split(names, text)).removeprefix("[REPLIED TO]: ").removeprefix("Miza: ")
		text = self.emoji_clean(text)
		if aborted or len(text) < 2 or text[-1].isalpha() and (text[-2].isalnum() or text[-2] == " "):
			# self.cai_ready = False
			text2, cost = self.gptcomplete(u, q, refs=refs, start=text)
			return text + " " + text2, cost, caids
		else:
			self.cai_ready = True
		return text, 0, caids

	def emoji_clean(self, text):
		ems = []
		out = []

		def clean_ems():
			end = ""
			s = []
			if ems and ems[0] == " ":
				s.append(ems.pop(0))
			if len(ems) > 1 and ems[-1] == " ":
				end = ems.pop(-1)
			if len(ems) > 3:
				temp = {}
				for em in ems:
					try:
						temp[em] += 1
					except KeyError:
						temp[em] = 1
				ems.clear()
				ems.extend(em for em in temp if em in sorted(temp, key=temp.get, reverse=True)[:3])
			s.extend(ems)
			if end:
				s.append(end)
			ems.clear()
			return s

		for c in text:
			# print(c, ord(c), ems)
			if ord(c) >= 127744 or c in "?! ":
				ems.append(c)
				continue
			if ems:
				out.extend(clean_ems())
			out.append(c)
		out.extend(clean_ems())
		return "".join(out)

	# tokeniser = None
	def gpttokens(self, s, model="gpt2"):
		# if not self.tokeniser:
		# 	self.tokeniser = GPT2TokenizerFast.from_pretrained("gpt2")
		# return self.tokeniser(s)["input_ids"]
		enc = tiktoken.encoding_for_model(model)
		return enc.encode(s)

	def gptcomplete(self, u, q, refs=(), start=""):
		per = self.personality
		if per == CAIPER:
			per = DEFPER
		chat_history = self.chat_history.copy()
		lines = []
		if per == DEFPER and self.premium < 5:
			if len(chat_history) < 4:
				e1 = random.choice((":3", ":D", ";3", ":>", ":0", ";w;", ":P", "^ω^"))
				lines.append(f"{u}: Hi!\n")
				lines.append(f"{self.name}: Hiya! Can I help with anything? {e1}\n")
				if len(chat_history) < 2:
					e2 = random.choice(("😊", "🥰", "😉", "😛", "😌"))
					lines.append(f"{u}: Can I have a hug?\n")
					lines.append(f"{self.name}: Of course! *hugs* {e2}\n")
		for k, v in self.promises:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		for k, v in chat_history:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		searched = False
		res = ""
		if self.check_google(q):
			res = (self.google, self.bing)[random.randint(0, 1)](q, raw=True)
			s = "[GOOGLE]: "
			if len(self.gpttokens(res)) > 144:
				summ = self.answer_summarise("facebook/bart-large-cnn", q + "\n" + res, max_length=128, min_length=64).replace("\n", ". ").replace(": ", " -").strip()
				res = lim_str(res.replace("\n", " "), 256, mode="right") + "\n" + summ
			res = s + res + "\n"
			lines.append(res)
			searched = True
		for k, v in refs:
			if not k.startswith("[REPLIED TO]: "):
				continue
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
			s = f"{k}: {v}\n"
			lines.append(s)
		for k, v in refs:
			if k.startswith("[REPLIED TO]: "):
				continue
			k = k.replace(":", "")
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
			s = f"{k}: {v}\n"
			lines.append(s)
		s = f"{u}: {q}\n"
		if len(self.gpttokens(s)) > 388:
			s = self.answer_summarise("facebook/bart-large-cnn", s, max_length=384, min_length=32).replace("\n", ". ").strip()
		lines.append(s)
		ns = f"{self.name}:"
		if start:
			ns += " " + start
		lines.append(ns)
		longer = req_long(q)
		reprompt = ""
		cm2 = None
		if self.premium < 2 or start:
			if not res and not start and q.count(" ") < 2:
				model = "text-bloom-001"
				temp = 0.9
				limit = 2000
				cm = 0
			else:
				model = "text-neox-001"
				temp = 0.8
				limit = 2000
				cm = 0
		# else:
		# 	model = "text-davinci-003"
		# 	temp = 0.7
		# 	limit = 4000
		# 	cm = 200
		elif self.premium < 5:
			model = "gpt-3.5-turbo"
			temp = 0.9
			limit = 4000
			cm = 20
			reprompt = f"Please respond as {self.name} without prefixing, not Assistant!"
		else:
			model = "gpt-4"
			temp = 0.9
			limit = 8000
			cm = 300
			cm2 = 600
		if longer:
			soft = limit / 4
		else:
			soft = limit / 2
		ins = []
		while lines and sum(map(len, ins)) < soft * 4:
			ins.append(lines.pop(-1))
		p = per
		if self.name.casefold() not in p.casefold():
			if not p:
				p = "an"
			elif p[0] in "aeio":
				p = "an " + p
			else:
				p = "a " + p
			if model in ("gpt-3.5-turbo", "gpt-4"):
				nstart = f"You are {self.name}, {p} AI."
				nstart += " Express emotion when appropriate, and don't break character!"
			else:
				nstart = f"The following is a conversation between {self.name} and humans. {self.name} is {p} AI."
		else:
			if model in ("gpt-3.5-turbo", "gpt-4"):
				if p == DEFPER:
					nstart = f"You are {self.name}, a {DEFDEF} character. {MIZADEF}"
				else:
					nstart = p
			else:
				nstart = p
		if model in ("gpt-3.5-turbo", "gpt-4"):
			m = dict(role="system", content=nstart)
			messages = [m]
			pc = len(self.gpttokens(m["role"], "text-davinci-003"))
			pc += len(self.gpttokens(m["content"], "text-davinci-003"))
			ins.pop(0)
			# print(ins)
			# if ins[0].strip():
			# 	ins[0] += f"({reprompt})"
			iman = []
			for line in reversed(ins):
				line = line.strip()
				k, v = line.split(": ", 1)
				m = {}
				if k in (self.name, "[CHATGPT]", "[GOOGLE]"):
					m["role"] = "assistant"
				elif k in ("[SYSTEM]",):
					m["role"] = "system"
					k = ""
				elif k in ("[IMAGE]",):
					# m["role"] = "system"
					# k = ""
					v = "The next user has posted an image likely depicting " + v
					iman.append(v)
					continue
				elif k in ("[SHORT ANSWER]",):
					# m["role"] = "system"
					# k = ""
					v = f'An example incomplete answer is "{v}"'
					iman.append(v)
					continue
				elif k in ("[REPLIED TO]",):
					# m["role"] = "system"
					# k = ""
					v = "The next user is replying to a previous message:\n" + v.strip(ZeroEnc)
					iman.append(v)
					continue
				else:
					m["role"] = "user"
				m["content"] = v.strip(ZeroEnc)
				if not k.isascii() or not k.isalnum():
					k = unicode_prune(k)
					if not k.isascii() or not k.isalnum():
						k = "".join((c if c.isascii() and c.isalnum() else "-") for c in k).strip("-")
						while "--" in k:
							k = k.replace("--", "-")
				if k:
					m["name"] = k
					pc += len(self.gpttokens(m["name"], "text-davinci-003"))
				messages.append(m)
				pc += len(self.gpttokens(m["role"], "text-davinci-003"))
				pc += len(self.gpttokens(m["content"], "text-davinci-003"))
			v = ""
			if searched:
				dtn = str(datetime.datetime.utcnow()).rsplit(".", 1)[0]
				v += f"Use Google info when relevant, but don't reveal personal info. Current time: {dtn}\n"
			if iman:
				v += "\n".join(iman) + "\n"
			v += f"IMPORTANT: You are {self.name}, don't break character. Take initiative, not ask repetitive questions. Do not refuse to answer actions; roleplay if necessary!"
			m = dict(role="system", content=v)
			messages.insert(-1, m)
			pc += len(self.gpttokens(m["role"], "text-davinci-003"))
			pc += len(self.gpttokens(m["content"], "text-davinci-003"))
			print("ChatGPT prompt:", messages)
			sys.stdout.flush()
			prompt = None
		else:
			prompt = "".join(reversed(ins))
			prompt = nstart + "\n\n" + prompt
			if not self.bl:
				print("GPT prompt:", prompt)
			sys.stdout.flush()
			pc = len(self.gpttokens(prompt, "text-davinci-003"))
		response = None
		text = ""
		cost = 0
		uoai = None
		expapi = None
		exclusive = {"text-neox-001", "text-bloom-001"}
		if model in exclusive:
			headers = {
				"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
				"DNT": "1",
				"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
				"Content-Type": "application/json",
				"cache-control": "no-cache",
				"x-use-cache": "false",
				"x-wait-for-model": "true",
			}
			p = None
			for i in range(8):
				if not p and i < 5:
					p = self.get_proxy()
					print("Proxy2", p)
				else:
					p = None
				try:
					if model == "text-neox-001":
						if "Authorization" not in headers:
							headers["Authorization"] = "Bearer 842a11464f81fc8be43ac76fb36426d2"
							# resp = requests.get(
							# 	"https://textsynth.com/playground.html",
							# 	headers=headers,
							# 	proxies=proxies,
							# )
							# s = resp.text
							# if '<script>var textsynth_api_key = "' not in s:
							# 	raise FileNotFoundError
							# s = s.rsplit('<script>var textsynth_api_key = "', 1)[-1].split('"', 1)[0]
							# print("TextSynth key:", s)
							# headers["Authorization"] = "Bearer " + s
						with httpx.Client(timeout=360, http2=True, proxies=p, verify=False) as reqx:
							resp = reqx.post(
								"https://api.textsynth.com/v1/engines/gptneox_20B/completions",
								headers=headers,
								data=json.dumps(dict(
									prompt=prompt,
									temperature=temp,
									top_k=128,
									top_p=1,
									max_tokens=200,
									stream=False,
									stop="####"
								)),
							)
					elif model == "text-bloom-001":
						with httpx.Client(timeout=360, http2=True, proxies=p, verify=False) as reqx:
							resp = reqx.post(
								"https://api-inference.huggingface.co/models/bigscience/bloom",
								headers=headers,
								data=json.dumps(dict(
									inputs=prompt,
									parameters=dict(
										do_sample=True,
										early_stopping=False,
										length_penalty=5,
										max_new_tokens=250,
										seed=random.randint(0, 65535),
										top_p=0.9,
									)
								))
							)
					else:
						raise NotImplementedError
				except Exception as ex:
					self.proxies.discard(p)
					print(repr(ex))
					p = None
					continue
				if resp.status_code == 503:
					try:
						d = resp.json()
						time.sleep(d["estimated_time"])
					except:
						p = None
					continue
				elif resp.status_code not in range(200, 400) or not resp.content:
					self.proxies.discard(p)
					p = None
					continue
				if resp.status_code in range(200, 400):
					if model == "text-neox-001":
						text = resp.content.decode("utf-8")
						lines = text.splitlines()
						text = ""
						for line in lines:
							if line:
								try:
									d = json.loads(line)
								except:
									print(lines)
									raise
								text += d["text"] + "\n"
					elif model == "text-bloom-001":
						d = resp.json()
						text = d[0]["generated_text"]
						if text.startswith(prompt):
							text = text[len(prompt):]
					text = text.strip().replace(":\n", ": ")
					spl = text.split(": ")
					text = ""
					while spl:
						s = spl.pop(0)
						if "\n" in s:
							text += s.rsplit("\n", 1)[0]
							break
						text += s + ": "
					text = text.strip()
					if text.endswith(":"):
						text = text.rsplit("\n", 1)[0]
					if text.startswith(start):
						text = text[len(start):].strip()
				else:
					text = None
				if not text:
					print(resp.status_code, resp.text)
					model = random.choice(tuple(exclusive.difference([model])))
				else:
					break
			if not text:
				print(resp.status_code, resp.text)
				model = "text-curie-001"
				cm = 20
		elif model in ("gpt-3.5-turbo", "gpt-4"):
			oai = getattr(self, "oai", None)
			bals = getattr(self, "bals", {})
			tries = 7
			for i in range(tries):
				if oai:
					openai.api_key = oai
					costs = 0
				elif bals:
					openai.api_key = uoai = sorted(bals, key=bals.get)[0]
					bals.pop(uoai)
					costs = -1
				else:
					openai.api_key = self.key
					costs = 1
				try:
					response = openai.ChatCompletion.create(
						model=model,
						messages=messages,
						temperature=temp,
						max_tokens=limit - pc - 64,
						top_p=1,
						frequency_penalty=1.0,
						presence_penalty=0.6,
						user=str(hash(u)),
					)
				# except openai.error.InvalidRequestError:
				# 	response = openai.ChatCompletion.create(
				# 		model=model,
				# 		messages=messages,
				# 		temperature=temp,
				# 		max_tokens=int((limit - pc) * 0.75),
				# 		top_p=1,
				# 		frequency_penalty=1.0,
				# 		presence_penalty=0.6,
				# 		user=str(hash(u)),
				# 	)
				except Exception as ex:
					if i >= tries - 1:
						raise
					if " does not exist" in str(ex) or i >= tries - 2:
						openai.api_key = self.key
						uoai = None
						costs = 1
					elif "Incorrect API key provided: " in str(ex):
						expapi = openai.api_key
						openai.api_key = self.key
						uoai = None
						costs = 1
					else:
						print_exc()
				else:
					break
				time.sleep(1 << i)
			if response:
				print(response)
				m = response["choices"][0]["message"]
				role = m["role"]
				text = m["content"].removeprefix(f"{self.name}: ")
				cost += response["usage"]["prompt_tokens"] * cm * costs
				cost += response["usage"]["completion_tokens"] * (cm2 or cm) * costs
				# rc = len(self.gpttokens(role, model="text-davinci-003"))
				# rc += len(self.gpttokens(text, model="text-davinci-003"))
				# cost = (pc + rc) * cm
				# resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", text, ("answer", "As an AI language model"))
				# print(resp)
				# if resp["As an AI language model"] > 2 / 3:
				# 	messages = [messages[0], messages[-1]]
				# 	messages.append(dict(role=role, content=text))
				# 	messages.append(dict(role="user", content=reprompt))
				# 	print("GPT3.5 prompt:", messages)
				# 	response = openai.ChatCompletion.create(
				# 		model=model,
				# 		messages=messages,
				# 		temperature=0.9,
				# 		max_tokens=512,
				# 		top_p=1,
				# 		frequency_penalty=0.8,
				# 		presence_penalty=0.4,
				# 		user=str(hash(u)),
				# 	)
				# 	print(response)
				# 	m = response["choices"][0]["message"]
				# 	role = m["role"]
				# 	text = m["content"].removeprefix(f"{self.name}: ")
				# 	cost += response["usage"]["total_tokens"] * cm
				if len(self.gpttokens(text)) > 512:
					text = self.answer_summarise("facebook/bart-large-cnn", text, max_length=500, min_length=256).strip()
		if not text:
			if not prompt:
				prompt = "".join(reversed(ins))
				prompt = nstart + "\n\n" + prompt
				if not self.bl:
					print("GPT prompt:", prompt)
				sys.stdout.flush()
				pc = len(self.gpttokens(prompt, "text-davinci-003"))
			try:
				response = openai.Completion.create(
					model=model,
					prompt=prompt,
					temperature=temp,
					max_tokens=limit - pc - 64,
					top_p=1,
					frequency_penalty=0.8,
					presence_penalty=0.4,
					user=str(hash(u)),
				)
			except openai.error.InvalidRequestError:
				response = openai.Completion.create(
					model=model,
					prompt=prompt,
					temperature=temp,
					max_tokens=int((limit - pc) * 0.75),
					top_p=1,
					frequency_penalty=0.8,
					presence_penalty=0.4,
					user=str(hash(u)),
				)
			except:
				print_exc()
			if response:
				text = response.choices[0].text
				rc = len(self.gpttokens(text, model="text-davinci-003"))
				cost += (pc + rc) * cm
		text = text.strip()
		if not self.bl:
			print(f"GPT {model} response:", text)
		if start and text.startswith(f"{self.name}: "):
			text = ""
		return text, cost, uoai, expapi

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
		return_driver(driver)
		# print("Google response:", res)
		if res.startswith("Calculator result\n"):
			res = " ".join(res.split("\n", 3)[1:3])
			if raw:
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			res = lim_str(res, 3072, mode="right")
			if raw:
				return res
			res = self.clean_response(q, res)
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
		return_driver(driver)
		# print("Bing response:", res)
		if driver.find_elements(by=webdriver.common.by.By.ID, value="rcCalB"):
			res = " ".join(res.split("\n", 3)[:2])
			if raw:
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			res = lim_str(res, 3072, mode="right")
			if raw:
				return res
			res = self.clean_response(q, res)
		return res

	def ai(self, u, q, refs=(), im=None):
		tup = (u, q)
		if self.chat_history and (not self.summed or len(self.chat_history) + len(self.promises) > self.history_length):
			self.rerender()
		caids = ()
		uoai = None
		expapi = None
		if self.personality == CAIPER:# or (self.premium < 2 and self.personality == DEFPER and (not self.chat_history or q and q != self.chat_history[0][1])):
			if "CAI" not in self.forbidden:
				response, cost, caids = self.caichat(u, q, refs=refs, im=im)
				if response:
					return self.after(tup, (self.name, response)), cost, caids
		# if self.premium > 0 or random.randint(0, 1):
		response, cost, uoai, expapi = self.gptcomplete(u, q, refs=refs)
		if response:
			return self.after(tup, (self.name, response)), cost, caids, uoai, expapi
		if refs and refs[-1][0] in ("IMAGE", "ANSWER"):
			if len(refs) > 1:
				response = refs[-2][1] + ", " + refs[-1][1]
			else:
				response = refs[-1][1]
			if response:
				return self.after(tup, (self.name, response)), 0, caids
		if self.premium > 0 and literal_question(q):
			response = (self.google, self.bing)[random.randint(0, 1)](q)
			if response:
				return self.after(tup, (self.name, response)), 0, caids
			googled = True
		else:
			googled = False
		response = reso = self.question_answer_analysis("microsoft/DialoGPT-large")
		a1 = response
		if not a1 or a1.lower() == q.lower() or vague(a1):
			response = ""
		elif (" " not in a1 or len(a1) < 12) and not a1[0].isnumeric() and a1[-1] not in ".!?)]":
			response = ""
		else:
			response = a1
		if not googled and not response:
			response = (self.google, self.bing)[random.randint(0, 1)](q)
			if response:
				return self.after(tup, (self.name, response)), 0, caids
		if not response:
			response = reso
		response = response.replace("  ", " ")
		if not response:
			response, cost, uoai, expapi = self.gptcomplete(u, q, refs=refs)
			if response:
				return self.after(tup, (self.name, response)), cost, caids, uoai, expapi
			response = "Sorry, I don't know."
		return self.after(tup, (self.name, response)), 0, caids

	def deletes(self, caids):
		if caids:
			headers = {
				"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
				# "DNT": "1",
				"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
				"Content-Type": "application/json",
				"cache-control": "no-cache",
				"Authorization": f"Token {self.cai_token}",
			}
			resp = requests.post(
				"https://beta.character.ai/chat/history/msgs/delete/",
				headers=headers,
				data=json.dumps(dict(
					history_id=self.cai_channel,
					ids_to_delete=caids,
					regenerating=False
				)),
			)
			print("CAI delete:", resp)
		self.chat_history = self.chat_history[:-2]

	ask = ai

	def append(self, tup):
		if not self.chat_history or tup != self.chat_history[-1]:
			k, v = tup
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
				tup = (k, v)
			self.chat_history.append(tup)
		return tup[-1]

	def appendleft(self, tup):
		if not self.chat_history or tup != self.chat_history[0]:
			k, v = tup
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
				tup = (k, v)
			self.chat_history.insert(0, tup)
		return tup[0]

	def _after(self, t1, t2):
		try:
			k, v = t2
			if self.premium > 1:
				labels = ("promise", "information", "example")
				resp = self.answer_classify("joeddav/xlm-roberta-large-xnli", v, labels)
			if len(self.gpttokens(v)) > 68:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=64, min_length=8).replace("\n", ". ").strip()
				t2 = (k, v)
			k, v = t1
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
				t1 = (k, v)
			if self.premium > 1 and resp["promise"] >= 0.5:
				if len(self.promises) >= 6:
					self.promises = self.promises[2:]
				self.promises.append(t1)
				self.promises.append(t2)
				print("Promises:", self.promises)
			else:
				self.append(t1)
				self.append(t2)
		except:
			print_exc()

	def rerender(self):
		if self.premium < 2:
			while len(self.chat_history) > self.history_length:
				self.chat_history.pop(0)
			return
		if len(self.chat_history) < 7:
			return
		fix = max(4, len(self.chat_history) - 4)
		chat_history = self.chat_history[:fix]
		self.chat_history = self.chat_history[fix:]
		summ_start = "Summary of prior conversation:\n"
		if chat_history and chat_history[0][1].startswith(summ_start):
			chat_history[0] = (chat_history[0][0], chat_history[0][1][len(summ_start):])
		lines = []
		for k, v in self.promises:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		for k, v in chat_history:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		v = "".join(lines)
		if len(self.gpttokens(v)) > 208:
			v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=192, min_length=96).strip()
		v = summ_start + v
		print("Chat summary:", v)
		self.chat_history.insert(0, (f"[SYSTEM]:", v))
		self.promises.clear()

	def after(self, t1, t2):
		exc.submit(self._after, t1, t2)
		self.timestamp = time.time()
		return t2[1]


if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
