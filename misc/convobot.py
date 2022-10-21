import os, time, urllib, json
import concurrent.futures
import selenium, requests, torch
from selenium import webdriver
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline

exc = concurrent.futures.ThreadPoolExecutor(max_workers=6)
drivers = []

class_name = webdriver.common.by.By.CLASS_NAME
css_selector = webdriver.common.by.By.CSS_SELECTOR
xpath = webdriver.common.by.By.XPATH
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
	options.headless = True
	options.add_argument("--disable-gpu")
	prefs = {"download.default_directory" : folder}
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
	return driver

def ensure_drivers():
	while len(drivers) < 1:
		drivers.append(exc.submit(create_driver))
		time.sleep(1)
def get_driver():
	if not drivers:
		drivers.append(exc.submit(create_driver))
	try:
		driver = drivers.pop(0)
		if hasattr(driver, "result"):
			driver = driver.result()
	except selenium.common.exceptions.WebDriverException:
		driver = create_driver()
	else:
		try:
			exc.submit(getattr, driver, "title").result(timeout=0.25)
		except:
			from traceback import print_exc
			print_exc()
			driver = create_driver()
	exc.submit(ensure_drivers)
	return driver

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
	if t.startswith("whats your") or t.startswith("what is your") or t.startswith("what are your") or t.startswith("what do you"):
		return False
	return any(t.startswith(i) for i in ("whats ", "what ", "wheres ", "where ", "whos ", "who ", "whens ", "when ", "whys ", "why ", "hows ", "how "))

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

	def __init__(self, token=""):
		self.token = token
		self.history = {}
		self.timestamp = time.time()

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
		inputs = tokenizer(q[:512], c[:512], return_tensors="pt")
		with torch.no_grad():
			outputs = model(**inputs)
		answer_start_index = outputs.start_logits.argmax()
		answer_end_index = outputs.end_logits.argmax()
		predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
		return tokenizer.decode(predict_answer_tokens).strip()

	def question_answer_analysis(self, m, q, qh, ah):
		try:
			tokenizer, model = self.models[m]
		except KeyError:
			tokenizer = AutoTokenizer.from_pretrained(m, padding_side="left", padding=True)
			model = AutoModelForCausalLM.from_pretrained(m)
			self.models[m] = (tokenizer, model)
		new_user_input_ids = tokenizer.encode(q + tokenizer.eos_token, return_tensors="pt")
		if self.history:
			chat_history_ids = tokenizer.encode((tokenizer.eos_token.join(qh), tokenizer.eos_token.join(ah)), return_tensors="pt")
			bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).long()
		else:
			bot_input_ids = new_user_input_ids
		chat_history_ids = model.generate(bot_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
		return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

	def ask(self, q):
		driver = get_driver()

		folder = driver.folder
		search = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		try:
			elem = driver.find_element(by=webdriver.common.by.By.ID, value="rso")
		except:
			return ""
		res = elem.text
		drivers.append(driver)
		if res.startswith("Calculator result\n"):
			response = " ".join(res.split("\n", 3)[1:3])
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			# print(res)
			if not res.isascii():
				fut = exc.submit(self.question_context_analysis, "salti/bert-base-multilingual-cased-finetuned-squad", q, res)
			else:
				fut = a2 = ""
			a1 = self.question_context_analysis("deepset/roberta-base-squad2", q, res)
			if fut:
				a2 = fut.result()
			if len(a2) > len(a1):
				a1 = a2
			response = a1
			if not response:
				response = res.split("\n", 1)[0]
				if response == "Dictionary":
					r = []
					for line in res.splitlines()[2:]:
						if line.casefold() == "translations and more definitions" or line.casefold().startswith("web result"):
							break
						r.append(line)
					response = "\n".join(r)
			elif response.casefold().replace("'", "") in ("i", "im", "imo", "io"):
				response = ""

		response = response.strip().replace("  ", " ")
		# self.history[q] = response
		return response

	def talk(self, i, recursive=True):
		t = time.time()
		if t > self.timestamp + 720:
			self.history.clear()
		elif len(self.history) > 8:
			self.history.pop(next(iter(self.history)))
		self.timestamp = t
		if not recursive or literal_question(i):
			words = i.split()
			i = " ".join(swap.get(w, w) for w in words)
			response = self.ask(i)
			if response and response.casefold() != i.casefold():
				self.history[i] = response
				return response
		self.history.pop(i, None)
		a1 = self.question_answer_analysis("microsoft/DialoGPT-large", i, list(self.history.keys()), list(self.history.values()))
		if a1.lower() == i.lower() or vague(a1) or a1.lower() in (a.lower() for a in self.history.values()):
			a1 = ""
		response = a1
		if recursive and not response:
			return self.talk(i, recursive=False)
		if not response:
			response = resp.json().get("generated_text", "").strip().replace("  ", " ")
		if not response:
			response = "Sorry, I don't know."
		self.history[i] = response
		return response

if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
