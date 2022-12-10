import os, time, urllib, json, random
import concurrent.futures
import selenium, requests, torch, openai
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline, set_seed

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
	options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")
	# options.add_argument("--disable-gpu")
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

	def __init__(self, token="", email="", password=""):
		self.token = token
		self.email = email
		self.password = password
		self.history = {}
		self.chat_history = []
		self.chat_history_ids = None
		self.previous = None
		self.timestamp = time.time()
		# self.gpt2_generator = pipeline('text-generation', model='gpt2-large')

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

	def question_answer_analysis(self, m, q, qh, ah):
		try:
			tokenizer, model = self.models[m]
		except KeyError:
			tokenizer = AutoTokenizer.from_pretrained(m, padding_side="left", padding=True)
			model = AutoModelForCausalLM.from_pretrained(m)
			self.models[m] = (tokenizer, model)
		end = tokenizer.eos_token
		new_user_input_ids = tokenizer.encode(q + end, return_tensors="pt", max_length=2048, truncation=True)
		history = []
		if self.chat_history_ids is not None:
			history.append(self.chat_history_ids)
		for k, v in self.history.items():
			history.append(tokenizer.encode(k + end, return_tensors="pt", max_length=2048, truncation=True))
			if v:
				history.append(tokenizer.encode(v + end, return_tensors="pt", max_length=2048, truncation=True))
		self.history.clear()
		history.append(new_user_input_ids)
		bot_input_ids = torch.cat(history, dim=-1)
		self.chat_history_ids = model.generate(bot_input_ids, max_length=16384, pad_token_id=tokenizer.eos_token_id)
		return tokenizer.decode(self.chat_history_ids[-4096:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

	def answer_fill_mask(self, m, q):
		if m == "xlm-roberta-large":
			try:
				fmp = self.models[m]
			except KeyError:
				fmp = self.models[m] = pipeline("fill-mask", model=m, tokenizer=m)
			return fmp(q)[0]["sequence"]

	def clean_response(self, q, res, additional=()):
		if additional:
			res = "\n".join(additional) + "\n" + (res or "")
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
		if not a1:
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
		return response.strip().replace("  ", " ")

	def gptcomplete(self, q, additional=()):
		openai.api_key = self.token
		question = q.strip()
		lines = []
		if self.chat_history:
			for q, a in self.chat_history:
				q = lim_str(q, 256)
				a = lim_str(a, 256)
				lines.append(f"Human: {q}\nMiza AI: {a}\n")
		for a in additional:
			lines.append(a + "\n")
		lines.append(f"Human: {question}\n")
		if literal_question(question) and not additional:
			res = lim_str(self.google(question, raw=True).replace("\n", ". "), 256, mode="right")
			lines.append(f"Google: {res}\n")
			googled = True
		else:
			googled = False
		lines.append("Miza AI:")
		prompt = ""
		while lines and len(prompt) < 1536:
			prompt = lines.pop(-1) + prompt
		print("GPTV3 prompt:", prompt)
		model = "text-babbage-001" if len(prompt) >= 1024 else "text-curie-001" if len(prompt) >= 512 or not random.randint(0, 2) else "text-davinci-003"
		response = openai.Completion.create(
			model=model,
			prompt=prompt,
			temperature=0.8,
			max_tokens=256,
			top_p=1,
			frequency_penalty=0,
			presence_penalty=0,
			user=str(id(self)),
		)
		text = response.choices[0].text.removesuffix("Is there anything else I can help you with?").removesuffix("Can you provide more information to support your claim?").strip()
		print("GPTV3 response:", text)
		# set_seed(int(time.time() // 0.1) & 4294967295)
		# text = ""
		# while not text.endswith("."):
		# 	response = self.gpt2_generator(
		# 		prompt,
		# 		max_length=4096,
		# 		num_return_sequences=1,
		# 	)
		# 	gt = response[0]["generated_text"]
		# 	seed 
		# text = text.removesuffix("Is there anything else I can help you with?").removesuffix("Can you provide more information to support your claim?").strip()
		# print("GPTV2 response:", text)
		test = text.casefold()
		if not test or test.startswith("sorry,") or test.startswith("i'm sorry,"):
			resp = openai.Moderation.create(
				input=question,
			)
			results = resp.results[0].categories
			if results.hate or results["self-harm"] or results["sexual/minors"] or results.violence:
				print(results)
				return text
			if googled:
				return
			lines = []
			if self.chat_history:
				for q, a in self.chat_history:
					q = lim_str(q, 256)
					a = lim_str(a, 256)
					lines.append(f"Human: {q}\nMiza AI: {a}\n")
			for a in additional:
				lines.append(a + "\n")
			lines.append(f"Human: {question}\n")
			res = lim_str(self.google(question, raw=True).replace("\n", ". "), 256, mode="right")
			lines.pop(-1)
			lines.append(f"Google: {res}\n")
			lines.append("Miza AI:")
			prompt = ""
			while lines and len(prompt) < 1536:
				prompt = lines.pop(-1) + prompt
			print("GPTV3 prompt2:", prompt)
			model = "text-curie-001" if len(prompt) >= 512 or not random.randint(0, 2) else "text-davinci-003"
			response = openai.Completion.create(
				model=model,
				prompt=prompt,
				temperature=0.8,
				max_tokens=256,
				top_p=1,
				frequency_penalty=0,
				presence_penalty=0,
				user=str(id(self)),
			)
			text = response.choices[0].text.removesuffix("Is there anything else I can help you with?").removesuffix("Can you provide more information to support your claim?").strip()
			print("GPTV3 response2:", text)
		return text

	def chatgpt(self, q, additional=(), force=False):
		if not self.email or not self.password:
			return ""
		if additional:
			question = "\n".join(additional) + "\n" + q
		else:
			question = q
		question = question.strip()
		driver = d = get_driver()
		search = "https://chat.openai.com/chat"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		elems = d.find_elements(by=class_name, value="btn-primary")
		elems = [e for e in elems if "log in" in e.text.casefold()]
		if elems:
			elems[0].click()
			time.sleep(1)
			e = d.find_element(by=webdriver.common.by.By.ID, value="username")
			e.send_keys(self.email)
			e = [e for e in d.find_elements(by=class_name, value="_button-login-id") if e.text == "Continue"][0]
			e.click()
			time.sleep(1)
			e = d.find_element(by=webdriver.common.by.By.ID, value="password")
			e.send_keys(self.password)
			e = [e for e in d.find_elements(by=class_name, value="_button-login-password") if e.text == "Continue"][0]
			e.click()
		time.sleep(1)

		elems = [e for e in d.find_elements(by=class_name, value="btn-neutral") if e.text in ("Next", "Done")]
		if elems:
			elems[0].click()
			while True:
				elems = [e for e in d.find_elements(by=class_name, value="btn-neutral") if e.text in ("Next", "Done")]
				if not elems:
					break
				elems[0].click()
			elems = [e for e in d.find_elements(by=class_name, value="btn-primary") if e.text in ("Next", "Done")]
			if elems:
				elems[0].click()

		elems = [e for e in d.find_elements(by=class_name, value="resize-none") if e in d.find_elements(by=class_name, value="bg-transparent")]
		e = elems[0]
		try:
			e.send_keys(question)
		except selenium.common.exceptions.WebDriverException:
			d.execute_script("document.getElementsByClassName('resize-none')[0].focus()")
			d.execute_script(f"document.execCommand('insertText', false, {repr(question)});")
		if not e.text:
			print("ChatGPT: No input.")
			drivers.insert(0, driver)
			return
		elems = [e for e in d.find_elements(by=class_name, value="text-gray-500") if e in d.find_elements(by=class_name, value="absolute")]
		e = elems[0]
		e.click()
		time.sleep(0.5)
		t2 = q.rstrip("?").casefold().split()
		for attempt in range(3):
			for i in range(180):
				elems = [e for e in d.find_elements(by=class_name, value="btn-neutral") if e.text == "Try again"]
				if elems:
					elems = d.find_elements(by=class_name, value="text-base")
					response = elems[-1].text
					break
				time.sleep(0.5)
				if i == 20:
					elems = d.find_elements(by=class_name, value="text-base")
					if not elems:
						print("ChatGPT: No input.")
						drivers.insert(0, driver)
						return
			else:
				response = None
				elems = d.find_elements(by=class_name, value="text-base")
				if elems:
					response = elems[-1].text
					if len(response) < 4 or response == question:
						response = None
					else:
						print("ChatGPT: Recovered.")
				if not response:
					print("ChatGPT: Timed out.")
					drivers.insert(0, driver)
					return
			response = response.removesuffix("\n2 / 2").removesuffix("\n3 / 3")
			print("ChatGPT response:", response)
			test = response.casefold()
			if test.startswith("!\nan error occurred.") or test.startswith("!\ninternal server error") or test.startswith("!\ntoo many requests"):
				elems = [e for e in d.find_elements(by=class_name, value="btn-neutral") if e.text == "Try again"]
				if not elems:
					return
				elems[0].click()
				continue
			spl = test.split()
			if not force and not additional and "\n" not in test and len(test) < 1024:
				openai.api_key = self.token
				resp = openai.Moderation.create(
					input=question,
				)
				results = resp.results[0].categories
				if results.hate or results["self-harm"] or results["sexual/minors"] or results.violence:
					print(results)
					break
				filtered = (
					"i'm sorry,",
					"i am sorry,",
					"sorry,",
					"i am not sure",
					"i'm not sure",
					"it is not specified",
				)
				if any(test.startswith(stm) for stm in filtered):
					elems = [e for e in d.find_elements(by=class_name, value="btn-neutral") if e.text == "Try again"]
					if not elems or attempt >= 1:
						drivers.insert(0, driver)
						return
					elems[0].click()
					continue
				filtered = (
					"t have access",
					"s not appropriate for",
					"s not possible for me",
					"s impossible for me",
					"t have the ability to",
					"m not able to",
					"m not capable of",
					"m unable to",
				)
				if (
					any(stm in test for stm in filtered)
					or ("illegal" in spl and "legal" not in q and "ok" not in t2 and "okay" not in t2)
				):
					drivers.insert(0, driver)
					return
			break
		else:
			print("ChatGPT: Exceeded attempt limit.")
			drivers.insert(0, driver)
			return
		drivers.insert(0, driver)
		if not response:
			return
		searches = (
			"I am a large language model trained by OpenAI and ",
			"As a large language model trained by OpenAI, I ",
		)
		for search in searches:
			if response.startswith(search):
				response = "I " + response[search:]
		response = response.replace("I am Assistant", "I am Miza").replace("trained by OpenAI", "linked to OpenAI, Google, Deepset and Microsoft")
		if ". Is there" in response:
			response = response.rsplit(". Is there", 1)[0] + "."
		if "essay" in t2 or "full" in t2 or "write" in t2 or "writing" in t2 or "about" in t2 or "worth noting that" in test or "worth mentioning that" in test:
			return response
		if additional or len(q) < 32:
			if response and additional and self.previous and q.casefold() == self.previous[0]:
				additional = [a for a in additional if a.casefold() != self.previous[1]]
			response = self.clean_response(q, response, additional=additional)
		else:
			response = response.strip()
		# print(response)
		return response

	def google(self, q, additional=(), raw=False):
		words = q.split()
		q = " ".join(swap.get(w, w) for w in words)
		driver = get_driver()
		search = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		try:
			elem = driver.find_element(by=webdriver.common.by.By.ID, value="rso")
		except:
			print("Google: Timed out.")
			drivers.insert(0, driver)
			return ""
		res = elem.text
		if res.startswith("Calculator result\n"):
			response = " ".join(res.split("\n", 3)[1:3])
			if raw:
				return response
		else:
			res = "\n".join(r.strip() for r in res.splitlines() if valid_response(r))
			if raw:
				return res
			response = self.clean_response(q, res, additional=additional)
		print("Google response:", response)
		drivers.insert(0, driver)
		return response

	def talk(self, i, additional=()):
		t = time.time()
		if t > self.timestamp + 720:
			self.history.clear()
		elif len(self.history) > 8:
			self.history.pop(next(iter(self.history)))
		self.timestamp = t
		tried_chatgpt = False
		response = reso = None
		words = i.casefold().split()
		if not additional and (len(i) >= 32 and (random.randint(0, 1) or not self.chat_history) or "essay" in words):
			response = reso = self.chatgpt(i, additional=additional)
			tried_chatgpt = True
		if response and response.casefold() != i.casefold():
			return self.register(i, response)
		response = reso = self.gptcomplete(i, additional=additional)
		if response and response.casefold() != i.casefold():
			return self.register(i, response)
		if not tried_chatgpt:
			response = reso = self.chatgpt(i, additional=additional)
			tried_chatgpt = True
		if literal_question(i):
			response = self.google(i, additional=additional)
			if response and response.casefold() != i.casefold():
				return self.register(i, response)
			googled = True
		else:
			googled = False
		if additional:
			response = self.clean_response(i, response, additional=additional)
			if response and response.casefold() != i.casefold():
				return self.register(i, response)
		res = self.question_answer_analysis("microsoft/DialoGPT-large", i, list(self.history.keys()), list(self.history.values()))
		a1 = res
		tup = (i.casefold(), a1.casefold())
		if a1.lower() == i.lower() or vague(a1) or (len(i) > 5 and tup == self.previous):
			a1 = ""
		if a1 and (" " not in a1 or len(a1) < 12) and not a1[0].isnumeric() and not a1.endswith("."):
			response = ""
		else:
			response = a1
		if not googled and not response:
			response = self.google(i, additional=additional)
			if response and response.casefold() != i.casefold():
				return self.register(i, response)
		if not response:
			response = reso if reso else res
		if not response:
			print(i + ": forcing GPTV3 response...")
			response = reso = self.chatgpt(i, additional=additional, force=True)
		response = response.replace("  ", " ")
		if not response:
			response = "Sorry, I don't know."
		return self.register(i, response)

	def append(self, msg):
		self.history[msg] = self.history.pop(msg, "")
		return msg

	def register(self, q, a):
		tup = (q.casefold(), a.casefold())
		self.previous = tup
		if len(self.chat_history) >= 3:
			self.chat_history.pop(0)
		self.chat_history.append((q, a))
		self.history[q] = a
		return a

if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.talk(input()))
