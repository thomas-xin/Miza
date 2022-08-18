import os, time
import concurrent.futures
import selenium
from selenium import webdriver

exc = concurrent.futures.ThreadPoolExecutor(max_workers=6)
drivers = []

class_name = webdriver.common.by.By.CLASS_NAME
css_selector = webdriver.common.by.By.CSS_SELECTOR
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
	while len(drivers) < 3:
		time.sleep(1)
		drivers.append(exc.submit(create_driver))
def get_driver():
	if not drivers:
		drivers.append(exc.submit(create_driver))
	try:
		driver = drivers.pop(0).result()
	except selenium.common.exceptions.WebDriverException:
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
			pass
		yield e


def yt_download(url, fmt="mp3", dir="", timeout=256):
	driver = get_driver()
	try:
		folder = driver.folder
		search = f"https://yt-download.org/api/button/{fmt}?url=" + url
		fut = exc.submit(driver.get, search)
		fut.result(timeout=16)

		t = time.time()
		elems = None
		while not elems:
			if fmt == "mp3":
				elems = driver.find_elements(by=class_name, value="bit256")
			else:
				for q in (1080, 720, 480, 360, 240, 144):
					elems = driver.find_elements(by=class_name, value=f"v{q}p")
					if elems:
						break
			if not elems:
				elems = driver.find_elements(by=class_name, value="alert")
				if elems:
					raise FileNotFoundError("Video unavailable.")
			time.sleep(0.2)
			if time.time() - t > 16:
				raise TimeoutError("Widget failed to load.")
		elems[0].click()

		t = time.time()
		elems = None
		while not elems:
			elems = list(safecomp(e for e in driver.find_elements(by=css_selector, value="*") if e.text == "CONVERT"))
			time.sleep(0.2)
			if time.time() - t > 16:
				raise TimeoutError("Convert button failed to load.")
		elems[0].click()

		titles = list(safecomp(e for e in driver.find_elements(by=css_selector, value="*") if "Download " in e.text))
		if titles:
			title = titles[0].text.split("Download ", 1)[-1].split("\n", 1)[0]
		else:
			title = url.split("?", 1)[0].rsplit("/", 1)[-1]

		t = time.time()
		while True:
			if elems:
				try:
					if elems[0].text == "DOWNLOAD":
						break
				except selenium.common.exceptions.StaleElementReferenceException:
					elems = ()
			if not elems:
				try:
					for e in driver.find_elements(by=css_selector, value="*"):
						if e.text == "DOWNLOAD":
							elems = (e,)
						elif "Convert Status: failed" in e.text:
							raise RuntimeError("Conversion failed.")
				except selenium.common.exceptions.StaleElementReferenceException:
					elems = ()
			time.sleep(0.5)
			if time.time() - t > timeout:
				raise TimeoutError("Download button failed to load.")
		if not os.path.exists(folder):
			os.mkdir(folder)
		elems[0].click()

		t = time.time()
		elems = None
		while not elems:
			elems = [e for e in os.listdir(folder) if e.endswith(f".{fmt}")]
			time.sleep(0.5)
			if time.time() - t > timeout:
				raise TimeoutError("Request timed out.")

		ts = time.time_ns()
		fn = os.path.join(dir, f"{ts}.{fmt}")
		os.rename(os.path.join(folder, elems[0]), fn)
	finally:
		if os.path.exists(folder):
			for fn in os.listdir(folder):
				os.remove(os.path.join(folder, fn))
			os.rmdir(folder)
		if not int(time.time_ns() // 1e4) % 3:
			driver.close()
		else:
			fut = concurrent.futures.Future()
			fut.set_result(driver)
			drivers.insert(0, fut)
	return title, fn