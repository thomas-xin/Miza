import os, time, urllib, json, io, random, re
import concurrent.futures
import selenium, requests, torch, openai, httpx
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from transformers import GPT2TokenizerFast, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline, set_seed
from fp.fp import FreeProxy
from traceback import print_exc

try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=64)
drivers = selenium.__dict__.setdefault("-drivers", [])

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
	try:
		driver.get("https://google.com/preferences")
		spans = driver.find_elements(by=tag_name, value="span")
		more = [span for span in spans if span.text == "Show more"][-1]
		more.click()
		opts = driver.find_elements(by=class_name, value="DB6WRb")[1:]
		random.choice(opts).click()
		confirm = driver.find_element(by=class_name, value="jfk-button-action")
		confirm.click()
	except:
		print_exc()
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
			d.get("file://")
			drivers.clear()
			drivers.append(d)

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
DEFPER = "loyal friendly playful cute"


class Bot:

	models = {}
	ctime = 0
	proxies = set()
	ptime = 0
	bad_proxies = set()
	btime = 0

	def __init__(self, token="", key="", cai_token="", email="", password="", name="Miza", personality=DEFPER, premium=0):
		self.token = token
		self.key = key
		self.cai_token = cai_token
		self.email = email
		self.password = password
		self.name = name
		self.personality = " ".join(personality.replace(",", " ").split())
		self.promises = []
		self.chat_history = []
		self.chat_history_ids = None
		self.cai_ready = False
		self.cai_channel = None
		self.timestamp = time.time()
		self.premium = premium
		self.last_cost = 0
		self.history_length = 2 if premium < 1 else 4
		self.fp = FreeProxy()

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
		inputs = tokenizer(q[:512], c[:2048], return_tensors="pt", max_length=4096, truncation=True)
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
		return zscp(q, labels, truncation=True)

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
		print("Roberta response:", res)
		return res

	def caichat(self, u, q, refs=(), im=None):
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			# "DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"Content-Type": "application/json",
			"cache-control": "no-cache",
			"Authorization": f"Token {self.cai_token}",
		}
		miza_id = "uBA8BVN2JgEC06sUU2KMtX-xv-X10rJEqvjgenJXfP4"
		miza_iid = "internal_id:cc102978-48f5-40b7-9867-bd39732dd6fb"
		lines = []
		if not self.cai_ready:
			if not self.cai_channel:
				resp = requests.post(
					"https://beta.character.ai/chat/history/create/",
					data=json.dumps(dict(character_external_id=miza_id)),
					headers=headers,
				)
				if resp.status_code not in range(200, 400):
					print("CAI create:", resp)
					print(resp.text)
					return
				try:
					self.cai_channel = resp.json()["external_id"]
				except KeyError:
					print("CAI create:", resp)
					print(resp.text)
					return
			chat_history = self.chat_history.copy()
			for k, v in self.promises:
				k = k.replace(":", "")
				s = f"{k}: {v}\n"
				lines.append(s)
			for k, v in chat_history:
				k = k.replace(":", "")
				s = f"{k}: {v}\n"
				lines.append(s)
		if literal_question(q):
			res = (self.google, self.bing)[random.randint(0, 1)](q, raw=True)
			start = "GOOGLE: "
			if len(self.gpttokens(res)) > 128:
				summ = self.answer_summarise("facebook/bart-large-cnn", q + "\n" + res, max_length=96, min_length=64).replace("\n", ". ").replace(": ", " -").strip()
				res = lim_str(res.replace("\n", " "), 256, mode="right") + "\n" + summ
			res = start + res + "\n"
			lines.append(res)
		if refs or lines:
			for k, v in refs:
				if len(self.gpttokens(v)) > 36:
					v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
				s = f"{k}: {v}\n"
				lines.append(s)
			s = f"###\nQuestion: {q}\n"
			if len(self.gpttokens(s)) > 388:
				s = self.answer_summarise("facebook/bart-large-cnn", s, max_length=384, min_length=32).replace("\n", ". ").strip()
			lines.append(s)
		if lines:
			prompt = "".join(lines)
		else:
			prompt = q
		print("CAI prompt:", prompt)
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
			resp = requests.post(
				"https://beta.character.ai/chat/upload-image/",
				files=(("image", b),),
				headers=h2,
			)
			if resp.status_code not in range(200, 400):
				print("CAI create:", resp)
				print(resp.text)
			else:
				print("CAI upload:", resp)
				idt = "AUTO_IMAGE_CAPTIONING"
				iot = "UPLOADED"
				irp = "https://characterai.io/i/400/static/user/" + resp.json()["value"]
		resp = requests.post(
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
			print("CAI:", resp)
			print(resp.text)
			self.cai_ready = False
			self.cai_channel = None
			return
		lines = [line.strip() for line in resp.text.splitlines()]
		e1 = json.loads(lines[-2]) if len(lines) > 1 else {}
		e2 = json.loads(lines[-1])
		if e2.get("abort", False):
			e2 = e1
			aborted = True
		else:
			aborted = False
		names = "[Uu][Tt][Ss][Ee]{2}[Ss][Rr][TtFf]"
		text = random.choice(e2.get("replies") or [{}]).get("text", "").strip()
		text = u.join(re.split(names, text))
		print("CAI response:", text)
		if aborted:
			self.cai_ready = False
			text2, cost = self.gptcomplete(u, q, refs=refs, start=text)
			return text + text2, cost
		else:
			self.cai_ready = True
		return text, 0

	tokeniser = None
	def gpttokens(self, s):
		if not self.tokeniser:
			self.tokeniser = GPT2TokenizerFast.from_pretrained("gpt2")
		return self.tokeniser(s)["input_ids"]

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
	def gptcomplete(self, u, q, refs=(), start=""):
		openai.api_key = self.key
		per = self.personality
		chat_history = self.chat_history.copy()
		lines = []
		if per == DEFPER:
			e1 = random.choice((":3", ":D", ";3", ":>", ":0", ";w;", ":P", "^ω^"))
			lines.append(f"{u}: Hi!\n")
			lines.append(f"{self.name}: Hiya! Can I help with anything? {e1}\n")
			if len(chat_history) < 2:
				e2 = random.choice(("😊", "🥰", "😉", "😛", "😌"))
				lines.append(f"{u}: What's the integral of 4x+1?\n")
				lines.append(f"{self.name}: It's 2x^2+x+C! {e2}\n")
		for k, v in self.promises:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		for k, v in chat_history:
			k = k.replace(":", "")
			s = f"{k}: {v}\n"
			lines.append(s)
		res = ""
		if (not refs and self.premium > 1 or literal_question(q)):
			res = (self.google, self.bing)[random.randint(0, 1)](q, raw=True)
			start = "GOOGLE: "
			if len(self.gpttokens(res)) > 128:
				summ = self.answer_summarise("facebook/bart-large-cnn", q + "\n" + res, max_length=96, min_length=64).replace("\n", ". ").replace(": ", " -").strip()
				res = lim_str(res.replace("\n", " "), 256, mode="right") + "\n" + summ
			res = start + res + "\n"
			lines.append(res)
		for k, v in refs:
			if not k.startswith("REPLIED TO: "):
				continue
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
			s = f"{k}: {v}\n"
			lines.append(s)
		s = f"{u}: {q}\n"
		if len(self.gpttokens(s)) > 388:
			s = self.answer_summarise("facebook/bart-large-cnn", s, max_length=384, min_length=32).replace("\n", ". ").strip()
		lines.append(s)
		for k, v in refs:
			if k.startswith("REPLIED TO: "):
				continue
			k = k.replace(":", "")
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=6).replace("\n", ". ").strip()
			s = f"{k}: {v}\n"
			lines.append(s)
		ns = f"{self.name}:"
		if start:
			ns += " " + start
		lines.append(ns)
		words = q.casefold().translate(self.unpunctuation).split()
		longer = "essay" in words or "full" in words or "write" in words or "writing" in words or "about" in words
		if self.premium < 2:
			if longer:
				model = "text-curie-001"
				temp = 0.8
				limit = 2000
				cm = 20
			else:
				model = "text-neox-001"
				temp = 0.8
				limit = 2000
				cm = 0
		else:
			model = "text-davinci-003"
			temp = 0.7
			limit = 4000
			cm = 200
		if longer:
			soft = limit / 4
		else:
			soft = limit / 2
		prompt = ""
		while lines and len(prompt) < soft * 4:
			prompt = lines.pop(-1) + prompt
		p = per
		if not p:
			p = "an"
		elif p[0] in "aeio":
			p = "an " + p
		else:
			p = "a " + p
		start = f"{self.name} is {p} AI:\n\n"
		prompt = start + prompt
		print("GPTV3 prompt:", prompt)
		pc = len(self.gpttokens(prompt))
		response = None
		text = ""
		if model in ("text-neox-001", "text-bloom-001"):
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
				break
			if resp.status_code in range(200, 400):
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
			else:
				print(resp.status_code, resp.text)
				model = "text-curie-001"
		if not text:
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
		if response and not text:
			text = response.choices[0].text
			rc = len(self.gpttokens(text))
			cost = (pc + rc) * cm
		else:
			cost = 0
		text = text.strip()
		print(f"GPTV3 {model} response:", text)
		return text, cost

	def bloom(self, prompt):
		API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
		headers = {"Authorization": f"Bearer {self.token}"}
		resp = requests.post(API_URL, headers=headers, json=dict(inputs=prompt))
		return resp.json()

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
			drivers.append(driver)
			return ""
		res = elem.text
		# print("Google response:", res)
		if res.startswith("Calculator result\n"):
			res = " ".join(res.split("\n", 3)[1:3])
			if raw:
				drivers.append(driver)
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			res = lim_str(res, 3072, mode="right")
			if raw:
				drivers.append(driver)
				return res
			res = self.clean_response(q, res)
		drivers.append(driver)
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
			drivers.append(driver)
			return ""
		res = elem.text
		# print("Bing response:", res)
		if driver.find_elements(by=webdriver.common.by.By.ID, value="rcCalB"):
			res = " ".join(res.split("\n", 3)[:2])
			if raw:
				drivers.append(driver)
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			res = lim_str(res, 3072, mode="right")
			if raw:
				drivers.append(driver)
				return res
			res = self.clean_response(q, res)
		drivers.append(driver)
		return res

	def ai(self, u, q, refs=(), im=None):
		tup = (u, q)
		while len(self.chat_history) > self.history_length:
			self.chat_history.pop(0)
		if self.personality == DEFPER:
			response, cost = self.caichat(u, q, refs=refs, im=im)
			if response:
				return self.after(tup, (self.name, response)), cost
		# if self.premium > 0 or random.randint(0, 1):
		response, cost = self.gptcomplete(u, q, refs=refs)
		if response:
			return self.after(tup, (self.name, response)), cost
		if refs and refs[-1][0] in ("IMAGE", "ANSWER"):
			if len(refs) > 1:
				response = refs[-2][1] + ", " + refs[-1][1]
			else:
				response = refs[-1][1]
			if response:
				return self.after(tup, (self.name, response)), 0
		if self.premium > 0 and literal_question(q):
			response = (self.google, self.bing)[random.randint(0, 1)](q)
			if response:
				return self.after(tup, (self.name, response)), 0
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
				return self.after(tup, (self.name, response)), 0
		if not response:
			response = reso
		response = response.replace("  ", " ")
		if not response:
			response, cost = self.gptcomplete(u, q, refs=refs)
			if response:
				return self.after(tup, (self.name, response)), cost
			response = "Sorry, I don't know."
		return self.after(tup, (self.name, response)), 0

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
				response = self.answer_classify("joeddav/xlm-roberta-large-xnli", v, labels)
				data = dict(zip(response["labels"], response["scores"]))
			if len(self.gpttokens(v)) > 36:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=32, min_length=8).replace("\n", ". ").strip()
				t2 = (k, v)
			k, v = t1
			if len(self.gpttokens(v)) > 28:
				v = self.answer_summarise("facebook/bart-large-cnn", v, max_length=24, min_length=6).replace("\n", ". ").strip()
				t1 = (k, v)
			if self.premium > 1 and data["promise"] >= 0.5:
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

	def after(self, t1, t2):
		exc.submit(self._after, t1, t2)
		return t2[1]


if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
