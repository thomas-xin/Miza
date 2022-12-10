import os, time, urllib, json, io, random, subprocess
import concurrent.futures
import selenium, requests, torch, openai
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline
import numpy as np
from PIL import Image

try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=16)
drivers = []

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
	# options.headless = True
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


class Bot:

	models = {}

	def __init__(self, token=""):
		self.token = token
		self.cache = {}
		self.session = requests.Session()
		self.timestamp = time.time()

	def art_dalle(self, prompt, kwargs=None):
		openai.api_key = self.token
		resp = openai.Image.create(
			prompt=prompt,
			n=1,
			size="512x512",
			user=str(id(self)),
		)
		print(resp)
		with requests.get(resp.data[0].url) as resp:
			return resp.content

	def dalle_i2i(self, prompt, image_1b, image_2b=None, force=False):
		openai.api_key = self.token
		if image_2b:
			im = Image.open(io.BytesIO(image_2b))
			if "A" not in im.mode:
				im.putalpha(im.convert("L").point(lambda x: x ^ 255))
				b = io.BytesIO()
				im.save(b, format="png")
				b.seek(0)
				image_2b = b.read()
			resp = openai.Image.create_edit(
				prompt=prompt,
				image=image_1b,
				mask=image_2b,
				n=1,
				size="512x512",
				user=str(id(self)),
			)
		else:
			if not prompt or not force:
				resp = openai.Image.create_variation(
					image=image_1b,
					n=1,
					size="512x512",
					user=str(id(self)),
				)
			else:
				im = Image.new("LA", (512, 512), 0)
				b = io.BytesIO()
				im.save(b, format="png")
				b.seek(0)
				image_2b = b.read()
				resp = openai.Image.create_edit(
					prompt=prompt,
					image=image_1b,
					mask=image_2b,
					n=1,
					size="512x512",
					user=str(id(self)),
				)
		print(resp)
		with requests.get(resp.data[0].url) as resp:
			return resp.content

	def art_mage(self, prompt, kwargs=None):
		driver = get_driver()

		folder = driver.folder
		search = "https://www.mage.space/"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		bar = driver.find_element(by=webdriver.common.by.By.ID, value="search-bar")
		try:
			bar.send_keys(prompt)
		except selenium.common.exceptions.WebDriverException:
			d.execute_script("document.getElementById('search-bar').focus()")
			d.execute_script(f"document.execCommand('insertText', false, {repr(prompt)});")

		generate = driver.find_element(by=webdriver.common.by.By.ID, value="ZQvTCDloXyqgqlOiDvup")
		generate.click()

		elems = None
		while not elems:
			elems = driver.find_elements(by=tag_name, value="img")
			for i in elems:
				if any(x > 0 for x in i.size.values()):
					break
			else:
				elems.clear()
			time.sleep(1)

		if not os.path.exists(folder):
			os.mkdir(folder)

		try:
			elems = driver.find_elements(by=css_selector, value="a[download='']")
			if elems:
				elems[0].click()
			else:
				elems = driver.find_elements(by=css_selector, value="*")
				for e in reversed(elems):
					if e.text == "Download":
						e.click()
						break
				else:
					raise FileNotFoundError("Download")

			elems = None
			while not elems:
				elems = [e for e in os.listdir(folder) if e.endswith(".png")]
				time.sleep(0.5)
			print(elems[0])

			ts = time.time_ns()
			fn = f"cache/{ts}.png"
			if not os.path.exists("cache"):
				os.mkdir("cache")
			os.rename(os.path.join(folder, elems[0]), fn)
		finally:
			os.rmdir(folder)
		return fn

	def art_deepai(self, prompt, kwargs=None):
		headers = {
			"User-Agent": "Mozilla/5.0",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"api-key": "quickstart-QUdJIGlzIGNvbWluZy4uLi4K",
		}
		resp = self.session.post(
			"https://api.deepai.org/api/text2img",
			files=dict(text=prompt),
			headers=headers,
		)
		if resp.status_code in range(200, 400):
			print(resp.text)
			url = resp.json()["output_url"]
			b = self.session.get(url, headers=headers).content
			image = Image.open(io.BytesIO(b))
			ims = [
				image.crop((0, 0, 512, 512)),
				image.crop((512, 0, 1024, 512)),
				image.crop((512, 512, 1024, 1024)),
				image.crop((0, 512, 512, 1024)),
			]
			ims2 = self.cache.setdefault(prompt, [])
			for im in ims:
				p = np.sum(im.resize((32, 32)).convert("L"))
				if p > 1024:
					b = io.BytesIO()
					im.save(b, format="png")
					b.seek(0)
					ims2.append(b.read())
			if ims2:
				random.shuffle(ims2)
				return ims2.pop(0)
			else:
				self.cache.pop(prompt, None)

	def art(self, prompt, url="", url2="", kwargs={}, specified=False):
		funcs = []
		if not specified and not url and not url2 or not os.path.exists("misc/stable_diffusion.openvino"):
			if random.randint(0, 2) and self.cache.get(prompt):
				return self.cache[prompt].pop(0)
			funcs.append(self.art_mage)
			funcs.append(self.art_deepai)
			if random.randint(0, 1):
				funcs.append(self.art_dalle)
		random.shuffle(funcs)
		for func in funcs:
			try:
				im = func(prompt, kwargs)
			except:
				import traceback
				traceback.print_exc()
				im = None
			if im:
				return im

if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.art(input(), url="a"))
