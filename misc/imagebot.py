import os, sys, time, urllib, orjson, io, random, subprocess, base64, traceback
import concurrent.futures
import selenium, requests, torch, openai, httpx
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from fp.fp import FreeProxy
import numpy as np
from PIL import Image
print_exc = lambda: sys.stdout.write(traceback.format_exc())

def print(*args, sep=" ", end="\n"):
	s = sep.join(map(str, args)) + end
	b = s.encode("utf-8")
	return sys.stdout.buffer.write(b)

if torch.cuda.is_available():
	try:
		torch.cuda.set_enabled_lms(True)
	except AttributeError:
		pass
try:
	exc = concurrent.futures.exc_worker
except AttributeError:
	exc = concurrent.futures.exc_worker = concurrent.futures.ThreadPoolExecutor(max_workers=64)
drivers = selenium.__dict__.setdefault("-drivers", [])

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
		argv = " ".join(ex.args)
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
	driver.get("file://")
	driver.folder = folder
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
	ctime = 0
	proxies = set()
	ptime = 0
	bad_proxies = set()
	btime = 0

	def __init__(self, token=""):
		self.token = token
		self.cache = {}
		self.session = requests.Session()
		self.timestamp = time.time()
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

	def art_dalle(self, prompt, kwargs=None):
		openai.api_key = self.token
		resp = openai.Image.create(
			prompt=prompt,
			n=1,
			size="512x512",
			user=str(id(self)),
		)
		print(resp)
		with self.session.get(resp.data[0].url) as resp:
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
		with self.session.get(resp.data[0].url) as resp:
			return resp.content, 180000

	def art_mage(self, prompt, kwargs=None):
		driver = get_driver()

		folder = driver.folder
		search = "https://www.mage.space/"
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		# elems = driver.find_elements(by=class_name, value="mantine-1qsvvs3")
		# if elems:
		# 	elems[0].click()
		# 	elems = driver.find_elements(by=class_name, value="mantine-q5ciiw")
		# 	if elems:
		# 		elems[0].click()
		# 	elems = driver.find_elements(by=class_name, value="mantine-8jlqcf")
		# 	if elems:
		# 		elems[0].click()

		time.sleep(2)
		elems = driver.find_elements(by=webdriver.common.by.By.ID, value="mantine-R3bm")
		if elems:
			driver.execute_script("document.getElementById('mantine-R3bm').style['z-index'] = -3")
			time.sleep(1)

		bar = driver.find_element(by=webdriver.common.by.By.ID, value="search-bar")
		bar.clear()
		try:
			bar.send_keys(prompt)
		except selenium.common.exceptions.WebDriverException:
			driver.execute_script("document.getElementById('search-bar').focus()")
			driver.execute_script(f"document.execCommand('insertText', false, {repr(prompt)});")

		driver.execute_script("document.getElementById('ZQvTCDloXyqgqlOiDvup').click()")
		# generate = driver.find_element(by=webdriver.common.by.By.ID, value="ZQvTCDloXyqgqlOiDvup")
		# generate.click()

		elems = None
		i = 0
		while not elems:
			if i >= 120:
				print("Mage: unavailable")
				return
			elems = driver.find_elements(by=tag_name, value="img")
			for e in reversed(elems):
				a = e.get_attribute("src")
				if "fdf0bcda49214494b6965064309ed6cc" in a:
					continue
				if a.startswith("https://cdn.mage.space/generate/"):
					break
			else:
				elems.clear()
			time.sleep(1)
			i += 1
		time.sleep(1)
		elems = driver.find_elements(by=class_name, value="mantine-1q3qenk")
		driver.delete_all_cookies()
		return_driver(driver)
		if elems:
			print("Mage: censored")
			return False

		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
		}
		print("Mage:", a)
		resp = self.session.get(a, headers=headers)
		if resp.status_code in range(200, 400):
			return resp.content
		print(resp.status_code, resp.text)

	def art_deepai(self, prompt, kwargs=None):
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"api-key": "quickstart-QUdJIGlzIGNvbWluZy4uLi4K",
		}
		resp = self.session.post(
			"https://api.deepai.org/api/text2img",
			files=dict(text=prompt),
			headers=headers,
		)
		if resp.status_code in range(200, 400):
			print("DeepAI:", resp.text)
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
			print("DeepAI: censored")
			return False
		print(resp.status_code, resp.text)

	def art_openjourney(self, prompt, kwargs=None):
		# if not any(w in prompt for w in ("style", "stylised", "stylized")):
		# 	prompt += ", mdjrny-v4 style"
		headers = {
			"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"cache-control": "no-cache",
			"x-use-cache": "false",
			"x-wait-for-model": "true",
		}
		resp = None
		p = None
		for i in range(8):
			if not p and i < 5:
				p = self.get_proxy()
				print("Proxy2", p)
			else:
				p = None
			try:
				with httpx.Client(timeout=360, http2=True, proxies=p, verify=False) as reqx:
					resp = reqx.post(
						"https://api-inference.huggingface.co/models/prompthero/openjourney",
						headers=headers,
						data=dict(inputs=prompt, wait_for_model=True),
					)
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
			print("Openjourney:", resp)
			b = resp.content
			im = Image.open(io.BytesIO(b))
			p = np.sum(im.resize((32, 32)).convert("L"))
			if p > 1024:
				return b
			print("Openjourney: censored")
			return False
		print(resp.status_code, resp.text)

	def art_stablediffusion_local(self, prompt, kwargs=None, model="stabilityai/stable-diffusion-2-1", fail_unless_gpu=True):
		pipe = torch.cuda.is_available() and self.models.get(model)
		if pipe == False and fail_unless_gpu:
			return
		if not pipe:
			from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
			pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
			pipe.enable_attention_slicing()
			pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
			if torch.cuda.is_available() and self.models.get(model, True):
				try:
					if torch.cuda.get_device_properties(0).total_memory < 8589934592:
						raise MemoryError("CUDA: Insufficient estimated virtual memory.")
					pipe = pipe.to("cuda")
					# pipe.enable_xformers_memory_efficient_attention()
				except:
					self.models[model] = False
					print_exc()
					print("StablediffusionL: CUDA f16 init failed")
					if fail_unless_gpu:
						return
			pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
			self.models[model] = pipe
		im = pipe(
			prompt,
			num_inference_steps=int(kwargs.get("--num-inference-steps", 50)),
			guidance_scale=float(kwargs.get("--guidance-scale", 7.5)),
		).images[0]
		b = io.BytesIO()
		im.save(b, format="png")
		print("StablediffusionL:", b)
		b.seek(0)
		return b.read()

	def art_textsynth(self, prompt, kwargs=None):
		headers = {
			"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"Content-Type": "application/json",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"cache-control": "no-cache",
		}
		kwargs = kwargs or {}
		resp = None
		p = None
		for i in range(8):
			if not p and i < 5:
				p = self.get_proxy()
				print("Proxy2", p)
			else:
				p = None
			try:
				if "Authorization" not in headers:
					headers["Authorization"] = "Bearer 842a11464f81fc8be43ac76fb36426d2"
				with httpx.Client(timeout=360, http2=True, proxies=p, verify=False) as reqx:
					resp = reqx.post(
						"https://api.textsynth.com/v1/engines/stable_diffusion/text_to_image",
						headers=headers,
						data=orjson.dumps(dict(
							prompt=prompt,
							timesteps=int(kwargs.get("--num-inference-steps", 50)),
							guidance_scale=float(kwargs.get("--guidance-scale", 7.5)),
							image_count=1,
							width=512,
							height=512,
						)),
					)
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
			print("TextSynth:", resp)
			d = resp.json()
			ds = d["images"]
			ims = [base64.b64decode(b["data"].encode("ascii")) for b in ds]
			b = ims.pop(0)
			self.cache.setdefault(prompt, []).extend(ims)
			return b
		print(resp.status_code, resp.text)

	def art(self, prompt, url="", url2="", kwargs={}, specified=False, dalle2=False, openjourney=False, nsfw=False):
		funcs = []
		if not url and not dalle2 and nsfw:
			funcs.append(self.art_textsynth)
		if not specified and not url and os.name == "nt":
			if random.randint(0, 2) and self.cache.get(prompt):
				return self.cache[prompt].pop(0), 0
			funcs.append(self.art_mage)
		if not specified and not url:
			if not openjourney:
				funcs.append(self.art_openjourney)
			if random.randint(0, 4):
				funcs.append(self.art_deepai)
			else:
				funcs.insert(0, self.art_deepai)
			if openjourney:
				funcs.insert(0, self.art_openjourney)
			if dalle2:
				funcs.insert(0, self.art_dalle)
		if not url:
			funcs.append(self.art_textsynth)
		for func in funcs:
			try:
				im = exc.submit(func, prompt, kwargs).result(timeout=240)
			except:
				print_exc()
				im = dalle2 = None
			if im:
				if dalle2:
					return im, 180000
				else:
					return im, 0
			elif im is False and not nsfw:
				raise PermissionError("NSFW filter detected in non-NSFW channel. If you believe this was a mistake, please try again.")

if __name__ == "__main__":
	import sys
	token = sys.argv[1] if len(sys.argv) > 1 else ""
	bot = Bot(token)
	while True:
		print(bot.art(input(), url="a"))
