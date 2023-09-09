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
	data = json.load(f)
compute_load = data.get("compute_load") or []

# Spec requirements:
# ytdl											anything with internet
# math			CPU >1							multithreading support
# image			FFMPEG, CPU >3, RAM >6GB		multiprocessing support
# caption		CPU >5, RAM >14GB				cpu inference
# agpt			CPU >1, RAM >22GB				(planned) reliability
# video			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# ecdc			FFMPEG, GPU >100k, VRAM >3GB	GTX970, M60, GTX1050ti, P4, GTX1630
# sd			GPU >200k, VRAM >5GB			RTX2060, T4, RTX3050, RTX3060m, A16
# sdxl			GPU >400k, VRAM >9GB			GTX1080ti, RTX2080ti, RTX3060, RTX3080, A2000
# sdxlr			GPU >400k, VRAM >15GB			V100, RTX3090, A4000, RTX4080, L4
# gptq			GPU >800k, VRAM >43GB			2xRTX3090, A6000, A40, A100, 2xRTX4090, L6000, L40
def spec2cap():
	caps = [-1, "ytdl"]
	cc = psutil.cpu_count()
	ram = psutil.virtual_memory().total
	try:
		subprocess.run("ffmpeg")
	except FileNotFoundError:
		ffmpeg = False
	else:
		ffmpeg = True
	if cc > 1:
		caps.append("math")
		if cc > 3 and ram > 6 * 1073741824 and ffmpeg:
			caps.append("image")
		if cc > 5 and ram > 14 * 1073741824:
			caps.append("caption")
		if cc > 1 and ram > 22 * 1073741824:
			caps.append("agpt")
	yield caps
	if cc > 2:
		caps = [-1, "ytdl", "math"]
		if ram > 14 * 1073741824 and ffmpeg:
			caps.append("image")
		if ram > 46 * 1073741824:
			caps.append("caption")
		if cc > 3 and ram > 46 * 1073741824:
			caps.append("agpt")
		yield caps
	if not DC:
		return
	done = set()
	import pynvml
	pynvml.nvmlInit()
	handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
	vrams = [pynvml.nvmlDeviceGetMemoryInfo(d).total for d in handles]
	for i, v in reversed(tuple(enumerate(vrams))):
		c = COMPUTE_POT[i]
		caps = [i]
		if c > 100000 and v > 3 * 1073741824 and ffmpeg:
			caps.append("video")
			caps.append("ecdc")
		if c > 400000 and v > 15 * 1073741824:
			if "sdxlr" not in done or c <= 600000:
				caps.append("sdxlr")
				caps.append("sdxl")
				done.add("sdxlr")
				v -= 15 * 1073741824
		elif c > 400000 and v > 9 * 1073741824:
			if "sdxl" not in done or c <= 600000:
				caps.append("sdxl")
				caps.append("sd")
				done.add("sdxl")
				v -= 9 * 1073741824
		elif c > 200000 and v > 5 * 1073741824:
			if "sd" not in done or c <= 600000:
				caps.append("sd")
				done.add("sd")
				v -= 5 * 1073741824
		if v <= 4 * 1073741824:
			v = 0
		vrams[i] = v
		if len(caps) > 1:
			yield caps
	if any(v > 6 * 1073741824 and c > 800000 for v, c in zip(vrams, COMPUTE_POT)):
		vram = sum(vrams[i] for i in range(DC) if COMPUTE_POT[i] > 400000)
		if vram > 43 * 1073741824:
			yield [-1, "agpt", "gptq"]

COMPUTE_POT = compute_load.copy()
DC = benchmark.DC
CAPS = list(spec2cap())

req = [
	"filetype",
	"orjson",
	"requests",
	"tiktoken",
	"pillow",
]
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
		start_proc(proc.i, proc.caps)
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
		while True:
			print(proc, "starting...")
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
			# print(proc, s)
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
			else:
				resps.clear()
			# if data:
				# print(data)
			for task in data:
				cap = task[1]
				new_tasks.setdefault(cap, []).append(task)
			tasks = []
			for cap in proc.caps:
				tasks.extend(new_tasks.get(cap, ()))
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

def start_proc(i, caps):
	args = [python, "misc/x-compute.py", str(i), ",".join(caps)]
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
	proc.i = i
	proc.caps = caps
	procs.append(proc)
	threading.Thread(target=update_tasks(proc)).start()
	threading.Thread(target=update_resps(proc)).start()
	return proc


for i, *caps in CAPS:
	start_proc(i, caps)
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
	while True:
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
				f"{ip}-v": dict(name="RAM", count=1, usage=minfo.used, max=minfo.total, time=t),
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
		# if data:
			# print(data)
		for task in data:
			cap = task[1]
			new_tasks.setdefault(cap, []).append(task)
		if data:
			for proc in procs:
				if proc.waiting:
					proc.waiting.set_result(None)
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