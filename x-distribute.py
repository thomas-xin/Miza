import os
# if os.name != "nt":
#     raise SystemExit("Compute distribution currently only supported on Windows due to drivers, sorry!")

import sys
from traceback import print_exc
python = sys.executable

if sys.version_info[0] < 3:
	raise ImportError("Python 3 required.")

if not os.path.exists("auth.json"):
	with open("auth.json", "wb") as f:
		f.write(b"{}")
if os.path.exists("x-compute.py"):
	sys.path.append("..")

import benchmark, json, psutil, subprocess

with open("auth.json", "rb") as f:
	AUTH = json.load(f)
compute_load = AUTH.get("compute_load") or []

IS_MAIN = False
FIRST_LOAD = True
# Spec requirements:
# ytdl											anything with internet
# math			CPU >1							multithreading support
# image			FFMPEG, CPU >3, RAM >6GB		multiprocessing support
# browse		Windows, CPU >1, RAM >3GB		webdriver support
# caption		CPU >5, RAM >14GB				cpu inference
# agpt			CPU >1, RAM >22GB				(planned) reliability
# video			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# ecdc			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# summ			GPU >200k, VRAM >4GB			RTX2060, T4, RTX3050, RTX3060m, A16
# class			GPU >400k, VRAM >11GB			V100, RTX3060, A2000, RTX4070
# sd			GPU >200k, VRAM >5GB			RTX2060, T4, RTX3050, RTX3060m, A16
# sdxl			GPU >400k, VRAM >9GB			GTX1080ti, RTX2080ti, RTX3060, RTX3080, A2000
# sdxlr			GPU >400k, VRAM >15GB			V100, RTX3090, A4000, RTX4080, L4
# gptq			GPU >700k, VRAM >44GB			2xV100, 5xRTX3080, 2xRTX3090, A6000, A40, A100, 2xRTX4090, L6000, L40
def spec2cap():
	try:
		from multiprocessing import shared_memory
		globals()["MEM_LOCK"] = shared_memory.SharedMemory(name="X-DISTRIBUTE", create=True, size=1)
	except FileExistsError:
		return
	caps = [[], "ytdl"]
	if not IS_MAIN:
		caps.append("remote")
	cc = psutil.cpu_count()
	ram = psutil.virtual_memory().total
	try:
		subprocess.run("ffmpeg")
	except FileNotFoundError:
		ffmpeg = False
	else:
		ffmpeg = True
	done = []
	try:
		import pynvml
		pynvml.nvmlInit()
		handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
		rrams = [pynvml.nvmlDeviceGetMemoryInfo(d).total for d in handles]
	except:
		rrams = []
	vrams = tuple(rrams)
	cut = 0
	if AUTH.get("discord_token") and any(v > 6 * 1073741824 and c > 700000 for v, c in zip(rrams, COMPUTE_POT)):
		vram = sum(rrams[i] for i in range(DC) if COMPUTE_POT[i] > 400000)
		if vram > 44 * 1073741824:
			cut = 44 * 1073741824
			did = []
			for i in COMPUTE_ORDER:
				v = rrams[i]
				if cut > 0:
					red = min(cut, v)
					rrams[i] = v - red
					cut -= red
					did.append(i)
				else:
					break
			if FIRST_LOAD:
				yield [[], "load", "gptq"]
				globals()["FIRST_LOAD"] = False
			yield [did, "agpt", "gptq"]
			done.append("gptq")
	if cc > 1:
		caps.append("math")
		if os.name == "nt" and ram > 3 * 1073741824:
			caps.append("browse")
		if cc > 3 and ram > 6 * 1073741824 and ffmpeg:
			caps.append("image")
		if cc > 5 and ram > 14 * 1073741824:
			caps.append("caption")
		if AUTH.get("discord_token") and cc > 1 and ram > 22 * 1073741824:
			caps.append("agpt")
	yield caps
	if cc > 2:
		caps = [[], "ytdl", "math"]
		if ram > 14 * 1073741824 and ffmpeg:
			caps.append("image")
		if ram > 46 * 1073741824:
			caps.append("caption")
		if AUTH.get("discord_token") and not cut and cc > 3 and ram > 46 * 1073741824:
			caps.append("agpt")
		yield caps
	if not DC:
		return
	for i, v in reversed(tuple(enumerate(rrams))):
		c = COMPUTE_POT[i]
		caps = [[i]]
		if c > 100000 and v > 3 * 1073741824 and ffmpeg:
			caps.append("video")
			caps.append("ecdc")
		if DC > 1 and c > 400000 and v > 11 * 1073741824 and (v > 29 * 1073741824 or "class" not in done):
			caps.append("class")
			done.append("class")
			v -= 11 * 1073741824
		if c > 200000 and v > 4 * 1073741824 and (v > 19 * 1073741824 or done.count("summ") <= DC / 2):
			caps.append("summ")
			done.append("summ")
			v -= 2 * 1073741824
		if c > 400000 and v > 15 * 1073741824:
			caps.append("sdxlr")
			caps.append("sdxl")
			# done.append("sdxlr")
			done.append("sdxl")
			v -= 15 * 1073741824
		elif c > 400000 and IS_MAIN and vrams[i] > 15 * 1073741824:
			caps.append("sdxlr")
			caps.append("ngptq")
			# done.append("sdxlr")
			v = 0
		elif c > 400000 and v > 9 * 1073741824:
			# if "sdxl" not in done or c <= 600000:
			caps.append("sdxl")
			caps.append("sd")
			done.append("sdxl")
			v -= 9 * 1073741824
		elif c > 400000 and IS_MAIN and "sdxl" not in done and vrams[i] > 9 * 1073741824 and "class" not in caps:
			caps.append("sdxl")
			caps.append("ngptq")
			done.append("sdxl")
			v = 0
		elif c > 200000 and v > 5 * 1073741824:
			if "sd" not in done or c <= 600000:
				caps.append("sd")
				done.append("sd")
				v -= 5 * 1073741824
		# if v <= 4 * 1073741824:
			# v = 0
		# vrams[i] = v
		if len(caps) > 1:
			yield caps

COMPUTE_POT = compute_load.copy()
DC = benchmark.DC
CAPS = list(spec2cap())
print(CAPS)
if not CAPS:
	raise SystemExit

req = [
	"filetype",
	"orjson",
	"requests",
	"tiktoken",
	"pillow",
]
if os.name == "nt":
	req.append("wmi")
if any("ytdl" in caps for caps in CAPS):
	req.append("yt-dlp")
if any("image" in caps for caps in CAPS):
	req.extend((
		"blend_modes",
		"colorspace",
	))
if any("math" in caps for caps in CAPS):
	req.extend((
		"matplotlib",
		"mpmath",
		"sympy",
	))
if any("caption" in caps for caps in CAPS):
	req.extend((
		"clip_interrogator",
		"pytesseract",
		"sentence_transformers",
	))
import pkg_resources, subprocess
for mn in req:
	try:
		pkg_resources.get_distribution(mn)
	except:
		subprocess.run([sys.executable, "-m", "pip", "install", mn, "--upgrade", "--user"])

if any("ecdc" in caps for caps in CAPS):
	try:
		assert pkg_resources.get_distribution("encodec").version >= "0.1.2a3"
	except (pkg_resources.DistributionNotFound, AssertionError):
		subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/encodec", "--user"])

import time, base64, orjson, threading, requests, urllib3, concurrent.futures
from math import *
session = requests.Session()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# exc = concurrent.futures.ThreadPoolExecutor(max_workers=4 + benchmark.DC)

new_tasks = {}
procs = []
PROC_RESP = {}
FORWARD = "https://api.mizabot.xyz/distribute"

def task_submit(proc, command, _timeout=12):
	ts = time.time_ns() // 1000
	while ts in PROC_RESP:
		ts += 1
	PROC_RESP[ts] = fut = concurrent.futures.Future()
	command = "[" + ",".join(map(repr, command[:2])) + "," + ",".join(map(str, command[2:])) + "]"
	s = f"~{ts}~".encode("ascii") + base64.b64encode(command.encode("utf-8")) + b"\n"
	# print(s)
	proc.busy = concurrent.futures.Future()
	try:
		proc.stdin.write(s)
		proc.stdin.flush()
		fut = PROC_RESP[ts]
		resp = fut.result(timeout=_timeout)
	except (BrokenPipeError, OSError) as ex:
		print_exc()
		proc.kill()
		procs.remove(proc)
		start_proc(proc.di, proc.caps)
		raise
	finally:
		PROC_RESP.pop(ts, None)
		try:
			proc.busy.set_result(None)
		except:
			pass
		proc.busy = None
	# create_task(wait_sub())
	return resp

def update_resps(proc):
	def func():
		print(proc, "starting...")
		while True:
			try:
				if not proc.is_running():
					return
				b = proc.stdout.readline()
				if not b:
					return
				# s = as_str(b.rstrip())
				# if s and s[0] == "~":
				#     c = as_str(evalEX(s[1:]))
				#     exec_tb(c, globals())
			except:
				print_exc()
			s = b.rstrip()
			print(proc, s)
			try:
				if s and s[:1] == b"$":
					s, r = s.split(b"~", 1)
					# print("PROC_RESP:", s, PROC_RESP.keys())
					d = {"_x": base64.b64decode(r)}
					c = eval(memoryview(s)[1:], globals(), d)
					if isinstance(c, (str, bytes, memoryview)):
						exec(c, globals(), d)
				elif s and s[:1] == b"~":
					c = eval(memoryview(s)[1:], globals())
					if isinstance(c, (str, bytes, memoryview)):
						exec(c, globals())
				else:
					print(s)
			except:
				print_exc()
				print(s)
	return func

def update_tasks(proc):
	def func():
		resps = {}
		while proc.is_running():
			data = ()
			resp = base64.urlsafe_b64encode(repr(resps).encode("utf-8")).rstrip(b"=") if resps else "{}"
			try:
				resp = session.post(
					FORWARD,
					data=dict(
						caps=orjson.dumps(proc.caps),
						resp=resp,
					),
					verify=False
				)
				data = resp.content
				# resp = session.get(f"https://api.mizabot.xyz/distribute?caps=[{proc.cap}]&resp={resp}")
				resp.raise_for_status()
				data = resp.json()
			except:
				if data:
					print(resp.text)
				data = ()
				print_exc()
				time.sleep(10)
			resps.clear()
			# if data:
				# print(data)
			for task in data:
				cap = task[1]
				new_tasks.setdefault(cap, []).append(task)
			tasks = []
			for cap in proc.caps:
				tasks.extend(new_tasks.pop(cap, ()))
			if not tasks:
				if proc.waiting:
					proc.waiting.result()
					proc.waiting = concurrent.futures.Future()
				else:
					time.sleep(1)
				continue
			if tasks:
				i, cap, command, timeout = tasks.pop(0)
				try:
					resp = task_submit(proc, command, _timeout=timeout)
				except Exception as ex:
					resps[str(i)] = "ERR:" + repr(ex)
				else:
					resps[str(i)] = "RES:" + resp if isinstance(resp, str) else resp
	return func

def start_proc(di, caps):
	args = [python, "misc/x-compute.py", ",".join(map(str, di)), ",".join(caps)]
	print(args)
	proc = psutil.Popen(
		args,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		# stderr=None,
		bufsize=262144,
	)
	proc.busy = None
	proc.waiting = concurrent.futures.Future()
	proc.di = di
	proc.caps = caps
	procs.append(proc)
	threading.Thread(target=update_tasks(proc)).start()
	threading.Thread(target=update_resps(proc)).start()
	return proc


for di, *caps in CAPS:
	start_proc(di, caps)
	time.sleep(1)
try:
	import time, requests, orjson, base64, cpuinfo
	class Self:
		_cpuinfo = None
		ip_time = 0
		up_bps = 0
		down_bps = 0
	self = Self()
	session = requests.Session()

	nex = None
	caps = []
	tasks = []
	responses = {}
	ioc = psutil.net_io_counters()
	up_old = ioc.bytes_sent
	down_old = ioc.bytes_recv
	ot = time.time() - 1
	import itertools
	for i in itertools.count(0):
		# print(i)
		ip = "<IP>"
		t = time.time()
		ioc = psutil.net_io_counters()
		up_bps = (ioc.bytes_sent - up_old) / (t - ot)
		down_bps = (ioc.bytes_recv - down_old) / (t - ot)
		up_old = ioc.bytes_sent
		down_old = ioc.bytes_recv
		ot = t
		cinfo = self._cpuinfo
		if not cinfo:
			cinfo = self._cpuinfo = cpuinfo.get_cpu_info()
		cpercent = psutil.cpu_percent()
		try:
			import pynvml
			pynvml.nvmlInit()
			dc = pynvml.nvmlDeviceGetCount()
			handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
			gname = [pynvml.nvmlDeviceGetName(d) for d in handles]
			gcore = [pynvml.nvmlDeviceGetNumGpuCores(d) for d in handles]
			gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) for d in handles]
			gutil = [pynvml.nvmlDeviceGetUtilizationRates(d) for d in handles]
			gpowa = [pynvml.nvmlDeviceGetPowerUsage(d) for d in handles]
			gpowb = [pynvml.nvmlDeviceGetEnforcedPowerLimit(d) for d in handles]
			gtempa = [pynvml.nvmlDeviceGetTemperature(d, 0) for d in handles]
			gtempb = [pynvml.nvmlDeviceGetTemperatureThreshold(d, 0) for d in handles]
		except:
			gname = []
		minfo = psutil.virtual_memory()
		sinfo = psutil.swap_memory()
		dinfo = {}
		for p in psutil.disk_partitions(all=False):
			try:
				dinfo[p.mountpoint] = psutil.disk_usage(p.mountpoint)
			except OSError:
				pass
		ram_name = globals().get("RAM_NAME") or "RAM"
		if ram_name == "RAM" and os.name == "nt" and globals().get("WMI") is not False:
			if not globals().get("WMI"):
				try:
					import wmi
					globals()["WMI"] = wmi.WMI()
				except:
					print_exc()
					globals()["WMI"] = False
			if WMI:
				OS = WMI.Win32_Operatingsystem()[0]
				cswap = (int(OS.TotalVirtualMemorySize) - int(OS.FreeVirtualMemory)) * 1024 - psutil.virtual_memory().used
				if cswap > sinfo.used:
					class mtemp:
						def __init__(self, used, total):
							self.used, self.total = used, total
					sinfo = mtemp(used=cswap, total=sinfo.total)
				ram = WMI.Win32_PhysicalMemory()[0]
				ram_speed = ram.ConfiguredClockSpeed
				ram_type = ram.SMBIOSMemoryType
				try:
					ram_class = {
						2: "DRAM",
						5: "EDO",
						9: "RAM",
						10: "ROM",
						20: "DDR1",
						21: "DDR2",
						24: "DDR3",
						26: "DDR4",
						34: "DDR5",
						35: "DDR5",
					}[ram_type]
				except KeyError:
					ram_class = "DDR" + str(max(1, ceil(math.log2(ram_speed / 250))))
				ram_name = globals()["RAM_NAME"] = f"{ram_class}-{ram_speed}"
		stats = orjson.dumps(dict(
			cpu={ip: dict(name=cinfo["brand_raw"], count=cinfo["count"], usage=cpercent / 100, max=1, time=t)},
			gpu={f"{ip}-{i}": dict(
				name=name,
				count=gcore[i],
				usage=gutil[i].gpu / 100,
				max=1,
				time=t,
			) for i, name in enumerate(gname)},
			memory={
				f"{ip}-v": dict(name=ram_name, count=1, usage=minfo.used, max=minfo.total, time=t),
				f"{ip}-s": dict(name="Swap", count=1, usage=sinfo.used, max=sinfo.total, time=t),
				**{f"{ip}-{i}": dict(
					name=name,
					count=1,
					usage=gmems[i].used,
					max=gmems[i].total,
					time=t,
				) for i, name in enumerate(gname)},
			},
			disk={f"{ip}-{k}": dict(name=k, count=1, usage=v.used, max=v.total, time=t) for k, v in dinfo.items()},
			network={
				f"{ip}-u": dict(name="Upstream", count=1, usage=up_bps, max=-1, time=t),
				f"{ip}-d": dict(name="Downstream", count=1, usage=down_bps, max=-1, time=t),
			},
			power={
				**{f"{ip}-{i}": dict(
					name=name,
					count=1,
					usage=gpowa[i] / 1000,
					max=gpowb[i] / 1000,
					time=t,
				) for i, name in enumerate(gname)},
			},
			temperature={
				**{f"{ip}-{i}": dict(
					name=name,
					count=1,
					usage=gtempa[i],
					max=gtempb[i],
					time=t,
				) for i, name in enumerate(gname)},
			},
		))
		prcs = [proc for proc in procs if not proc.busy or proc.busy.done()]
		caps = set()
		for proc in prcs:
			caps.update(proc.caps)
		caps = list(caps)
		stat = base64.urlsafe_b64encode(stats).rstrip(b"=")
		data = ()
		try:
			resp = session.post(
				"https://api.mizabot.xyz/distribute",
				data=dict(
					caps=orjson.dumps(caps),
					stat=stat,
				),
				verify=False,
			)
			FORWARD = resp.url
			data = resp.content
			# resp = session.get(f"https://api.mizabot.xyz/distribute?caps={caps}&stat={stat}")
			resp.raise_for_status()
			data = resp.json()
			if nex:
				print("Reconnected.")
				nex = None
		except Exception as ex:
			nex = ex
			if data:
				print(resp.text)
			data = ()
			print_exc()
			time.sleep(20)
			session = requests.Session()
		# if data:
			# print(data)
		for task in data:
			cap = task[1]
			new_tasks.setdefault(cap, []).append(task)
		if data:
			for proc in procs:
				if proc.waiting:
					try:
						proc.waiting.set_result(None)
					except:
						pass
					# proc.waiting = None
		time.sleep(5)
except:
	print_exc()
finally:
	for proc in procs:
		try:
			proc.kill()
		except:
			pass
	procs.clear()
	# exc.shutdown()
	psutil.Process().kill()