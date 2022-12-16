import os, time, urllib, json, random
import concurrent.futures
import selenium, requests, torch, openai
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from transformers import GPT2TokenizerFast, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline, set_seed
from traceback import print_exc

try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=16)
drivers = []

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
	options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")
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
	if not t or t.startswith("whats your") or t.startswith("what is your") or t.startswith("what are your") or t.startswith("what do you"):
		return False
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


class Bot:

	models = {}

	def __init__(self, token="", email="", password="", name="Miza", personality="friendly, playful, cute", premium=0):
		self.token = token
		self.email = email
		self.password = password
		self.name = name
		self.personality = " ".join(personality.replace(",", " ").split())
		self.curr_history = []
		self.chat_history = []
		self.chat_history_ids = None
		self.timestamp = time.time()
		self.premium = premium
		self.history_length = 1 if premium < 1 else 2 if premium < 2 else 8

	def question_context_analysis(self, m, q, c):
		if m == "deepset/roberta-base-squad2":
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
		inputs = tokenizer(q[:512], c[:512], return_tensors="pt", max_length=4096, truncation=True)
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
		if self.chat_history_ids is not None:
			history.append(self.chat_history_ids)
		for k, v in self.curr_history:
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

	def clean_response(self, q, res):
		res = res.strip()
		if not res.isascii():
			fut = exc.submit(self.question_context_analysis, "salti/bert-base-multilingual-cased-finetuned-squad", q, res)
		else:
			fut = a2 = ""
		a1 = self.question_context_analysis("deepset/roberta-base-squad2", q, res)
		if fut:
			a2 = fut.result()
		if len(a2) >= len(a1) * 2:
			a1 = a2
		a1 = a1.strip()
		if len(a1) < 16:
			res = self.answer_summarise("facebook/bart-large-cnn", res)
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
	def gptcomplete(self):
		if not self.curr_history:
			return ""
		q = self.curr_history[-1][1]
		openai.api_key = self.token
		lines = []
		res = ""
		if self.premium > 0 and (self.premium > 1 or literal_question(q)):
			if random.randint(0, 1):
				res = self.google(raw=True)
				start = "Google: "
			else:
				res = self.bing(raw=True)
				start = "Bing: "
			res = self.answer_summarise("facebook/bart-large-cnn", res, max_length=384, min_length=64).replace("\n", ". ").replace(": ", " -").strip()
			res = lim_str(start + res, 384, mode="right")
		if self.curr_history:
			for k, v in self.curr_history[:-1]:
				s = f"{k}: {v}\n"
				if len(s) > 128:
					s = self.answer_summarise("facebook/bart-large-cnn", s).replace("\n", ". ").strip()
					s = lim_str(s, 128)
				lines.append(s)
			k, v = self.curr_history[-1]
			s = f"{k}: {v}\n"
			if len(s) > 2048:
				s = self.answer_summarise("facebook/bart-large-cnn", s, max_length=2048, min_length=32).replace("\n", ". ").strip()
				if len(s) < 256:
					s = f"{k}: {v}\n"
				s = lim_str(s, 1024)
			lines.append(s)
		lines.append(f"{self.name}:")
		if self.premium < 1 or self.premium < 2 and (len(q) >= 256 or res):
			model = "text-babbage-001"
			temp = 0.9
			limit = 1000
		elif self.premium < 2:
			model = "text-curie-001"
			temp = 0.8
			limit = 2000
		else:
			model = "text-davinci-003"
			temp = 0.7
			limit = 4000
		q = self.curr_history[-1][1]
		words = q.casefold().translate(self.unpunctuation).split()
		if "essay" in words or "full" in words or "write" in words or "writing" in words or "about" in words:
			soft = limit / 4
		else:
			soft = limit / 2
		prompt = ""
		while lines and len(self.gpttokens(prompt + res)) < soft:
			prompt = lines.pop(-1) + prompt
		p = "" if self.premium < 2 else self.personality
		if not p:
			p = "an"
		elif p[0] in "aeio":
			p = "an " + p
		else:
			p = "a " + p
		start = f"{self.name} is {p} AI:\n\n"
		if res:
			start += res + "\n"
		prompt = start + prompt
		print("GPTV3 prompt:", prompt)
		response = None
		text = ""
		try:
			response = openai.Completion.create(
				model=model,
				prompt=prompt,
				temperature=temp,
				max_tokens=limit - len(self.gpttokens(prompt)) - 64,
				top_p=1,
				frequency_penalty=0.8,
				presence_penalty=0.4,
				user=str(hash(self.curr_history[-1][0])),
			)
		except openai.error.InvalidRequestError:
			response = openai.Completion.create(
				model=model,
				prompt=prompt,
				temperature=temp,
				max_tokens=int((limit - len(self.gpttokens(prompt))) * 0.75),
				top_p=1,
				frequency_penalty=0.8,
				presence_penalty=0.4,
				user=str(hash(self.curr_history[-1][0])),
			)
		except:
			print_exc()
		if response:
			text = response.choices[0].text.strip()
		print(f"GPTV3 {model} response:", text)
		return text

	def google(self, raw=False):
		if not self.curr_history:
			return ""
		q = self.curr_history[-1][1]
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
		print("Google response:", res)
		if res.startswith("Calculator result\n"):
			res = " ".join(res.split("\n", 3)[1:3])
			if raw:
				drivers.append(driver)
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			if raw:
				drivers.append(driver)
				return res
			res = self.clean_response(q, res)
		drivers.append(driver)
		return res

	def bing(self, raw=False):
		if not self.curr_history:
			return ""
		q = self.curr_history[-1][1]
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
		print("Bing response:", res)
		if driver.find_elements(by=webdriver.common.by.By.ID, value="rcCalB"):
			res = " ".join(res.split("\n", 3)[:2])
			if raw:
				drivers.append(driver)
				return res
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			if raw:
				drivers.append(driver)
				return res
			res = self.clean_response(q, res)
		drivers.append(driver)
		return res

	def ai(self):
		while len(self.chat_history) > self.history_length:
			self.chat_history.pop(0)
		self.curr_history = self.chat_history.copy()
		if self.premium > 0 or random.randint(0, 1):
			response = self.gptcomplete()
			if response:
				return self.append((self.name, response))
		q = self.curr_history[-1][-1]
		if self.premium > 0 and literal_question(q):
			response = (self.google, self.bing)[random.randint(0, 1)]
			if response:
				return self.append((self.name, response))
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
			response = (self.google, self.bing)[random.randint(0, 1)]
			if response:
				return self.append((self.name, response))
		if not response:
			response = reso
		response = response.replace("  ", " ")
		if not response:
			response = self.gptcomplete()
			if response:
				return self.append((self.name, response))
			response = "Sorry, I don't know."
		return self.append((self.name, response))

	def append(self, tup):
		if not self.chat_history or tup != self.chat_history[-1]:
			self.chat_history.append(tup)
		return tup[-1]

	def appendleft(self, tup):
		if not self.chat_history or tup != self.chat_history[0]:
			self.chat_history.insert(0, tup)
		return tup[0]


if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
