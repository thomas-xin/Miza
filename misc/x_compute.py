#!/usr/bin/python3

import os
import sys
import orjson

if os.path.exists("auth.json"):
	with open("auth.json", "rb") as f:
		AUTH = orjson.loads(f.read())
	cachedir = AUTH.get("cache_path") or None
	if cachedir:
		os.environ["HF_HOME"] = f"{cachedir}/huggingface"
		os.environ["TORCH_HOME"] = f"{cachedir}/torch"
		os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cachedir}/huggingface/hub"
		os.environ["TRANSFORMERS_CACHE"] = f"{cachedir}/huggingface/transformers"
		os.environ["HF_DATASETS_CACHE"] = f"{cachedir}/huggingface/datasets"
	else:
		cachedir = os.path.expanduser("~") + "/.cache"
		if not os.path.exists(cachedir):
			os.mkdir(cachedir)

import io
import time
import concurrent.futures
import itertools
import subprocess
import psutil
import collections
import traceback
import re
import ast
import base64
import random
import urllib.parse
import urllib.request
import numpy as np
from math import inf, floor, ceil, log2, log10
from traceback import print_exc
sys.path.append("misc")
from .util import EvalPipe, PipedProcess, new_playwright_page, CODECS, CODEC_FFMPEG, CODEC_PIX, temporary_file, is_url

if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	print = interface.print

cdict = dict
utc = time.time

def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

def literal_eval(s):
	return ast.literal_eval(as_str(s).lstrip())

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if len(sys.argv) > 2 and sys.argv[2]:
	os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
	DEVICES = list(map(int, sys.argv[2].split(",")))
	DEV = DEVICES[0]
else:
	DEVICES = []
	DEV = -1
if len(sys.argv) > 3:
	CAPS = set(sys.argv[3].split(","))
else:
	CAPS = frozenset()
if len(sys.argv) > 4:
	COMPUTE_LOAD = orjson.loads(sys.argv[4])
else:
	COMPUTE_LOAD = []
if len(sys.argv) > 5:
	COMPUTE_CAPS = orjson.loads(sys.argv[5])
else:
	COMPUTE_CAPS = []
if len(sys.argv) > 6:
	COMPUTE_ORDER = [i for i in orjson.loads(sys.argv[6]) if i in DEVICES]
else:
	COMPUTE_ORDER = []
if len(sys.argv) > 7:
	IT = int(sys.argv[7])
else:
	IT = 0


if CAPS.intersection(("browse", "image", "caption", "video")):
	from math import *
	import x_image
	x_image.register_print_fn(print)
	from x_image import *
	from PIL import Image
else:
	Image = None
	exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)

	if not hasattr(time, "time_ns"):
		time.time_ns = lambda: int(time.time() * 1e9)


if "video" in CAPS:
	sizecheck = re.compile("[1-9][0-9]*x[0-9]+")
	fpscheck = re.compile("[0-9]+ fps")

if "math" in CAPS:
	import matplotlib.pyplot as plt
	import psutil
	plt.style.use("dark_background")
	plt.rcParams["figure.dpi"] = 96
	plt.rcParams["figure.figsize"] = (16, 16)
	# For the ~activity command.
	special_colours = {
		"message": (0, 0, 1),
		"typing": (0, 1, 0),
		"command": (0, 1, 1),
		"reaction": (1, 1, 0),
		"misc": (1, 0, 0),
	}

	def plt_special(d, user=None, **void):
		hours = 336
		plt.rcParams["figure.figsize"] = (24, 9)
		plt.xlim(-hours, 0)
		temp = np.zeros(len(next(iter(d.values()))))
		width = hours / len(temp)
		domain = width * np.arange(-len(temp), 0)
		for k, v in d.items():
			if len(v) > len(temp):
				v = v[-len(temp):]
			plt.bar(domain, v, bottom=temp, color=special_colours.get(k, "k"), edgecolor="white", width=width, label=k)
			temp += np.array(v)
		plt.bar(list(range(-hours, 0)), np.ones(hours) * max(temp) / 512, edgecolor="white", color="k")
		if user:
			plt.title("Recent Discord Activity for " + user)
		plt.xlabel("Time (Hours)")
		plt.ylabel("Action Count")
		plt.legend(loc="upper left")
		out = temporary_file("png")
		plt.savefig(out)
		plt.clf()
		plt.rcParams["figure.figsize"] = (16, 16)
		with open(out, "rb") as f:
			return f.read()

	def plt_mp(arr, hours, name):
		if hours >= 336:
			hours /= 24
			if hours >= 336:
				hours /= 30.436849166666665
				if hours >= 24:
					hours /= 12
					word = "years"
				else:
					word = "months"
			else:
				word = "days"
		else:
			word = "hours"
		plt.style.use("dark_background")
		plt.rcParams["figure.figsize"] = (24, 9)
		plt.rcParams["figure.dpi"] = 96
		plt.xlim(-hours, 0)
		x = np.linspace(-hours, 0, len(arr))
		plt.plot(x, arr, "-w")
		plt.xlabel(word.capitalize())
		out = f"misc/{name}.png"
		plt.savefig(out)
		plt.clf()
		with open(out, "rb") as f:
			return f.read()

if "ecdc" in CAPS:
	def ecdc_encode(b, bitrate="24", name="", source="", thumbnail=""):
		assert len(b)
		fn = temporary_file()
		with open(fn, "wb") as f:
			f.write(b)
		fo = temporary_file("ecdc")
		if name:
			name = " " + base64.b64encode(name.strip().encode("utf-8")).rstrip(b"=").decode("utf-8")
		args1 = ["ffmpeg", "-v", "error", "-hide_banner", "-vn", "-nostdin", "-i", fn, "-f", "s16le", "-ac", "2", "-ar", "48k", "-"]
		args2 = [sys.executable, "misc/ecdc_stream.py", "-g", str(DEV), "-n", name or "", "-s", source or "", "-t", thumbnail or "", "-b", str(bitrate), "-e", fo]
		print(args1)
		print(args2)
		PipedProcess(args1, args2).wait()
		with open(fo, "rb") as f:
			return f.read()

	def ecdc_decode(b, fmt="opus"):
		assert len(b)
		fn = temporary_file("ecdc")
		with open(fn, "wb") as f:
			f.write(b)
		fo = temporary_file(fmt)
		args1 = [sys.executable, "misc/ecdc_stream.py", "-b", "0", "-g", str(DEV), "-d", fn]
		args2 = ["ffmpeg", "-v", "error", "-hide_banner", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "96k", fo]
		print(args1)
		print(args2)
		PipedProcess(args1, args2).wait()
		with open(fo, "rb") as f:
			return f.read()


def get_nvml():
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
	return gname, gcore, gmems, gutil, gpowa, gpowb, gtempa, gtempb

WMT = 0
WMV = 0
def get_wmem(mused=0):
	global WMT, WMV
	t = utc()
	if t - WMT > 60:
		try:
			f1 = exc.submit(subprocess.check_output, "wmic OS get TotalVirtualMemorySize /Value")
			fvms = subprocess.check_output("wmic OS get FreeVirtualMemory /Value")
			tvms = f1.result()
			tvms = int(tvms.strip().decode("ascii").removeprefix("TotalVirtualMemorySize="))
			fvms = int(fvms.strip().decode("ascii").removeprefix("FreeVirtualMemory="))
			WMV = (tvms - fvms) * 1024 - mused
		except Exception:
			WMV = 0
		WMT = utc()
	return WMV

_cpuinfo = _diskinfo = None
_ctime = _dtime = 0
def get_current_stats(up_bps, down_bps):
	global WMI, _cpuinfo, _ctime, _diskinfo, _dtime
	import psutil
	t = utc()
	cinfo = _cpuinfo
	if t - _ctime > 3600:
		_ctime = t
		import cpuinfo
		cinfo = _cpuinfo = cpuinfo.get_cpu_info()
	f1 = psutil.cpu_percent()
	f2 = psutil.virtual_memory()
	f3 = psutil.swap_memory()
	try:
		gname, gcore, gmems, gutil, gpowa, gpowb, gtempa, gtempb = get_nvml()
	except Exception:
		gname = []
	dinfo = _diskinfo
	if t - _dtime > 60:
		_dtime = t
		dinfo = _diskinfo = {}
		for p in psutil.disk_partitions(all=False):
			try:
				dinfo[p.mountpoint] = psutil.disk_usage(p.mountpoint)
			except OSError:
				pass
	cpercent, minfo, sinfo = f1, f2, f3
	ip = "127.0.0.1"
	if os.name == "nt":
		cswap = get_wmem(minfo.used)
		if cswap > sinfo.used:
			class mtemp:
				def __init__(self, used, total):
					self.used, self.total = used, total
			sinfo = mtemp(used=cswap, total=sinfo.total)
	ram_name = globals().get("RAM_NAME") or "RAM"
	if os.name == "nt" and not globals().get("WMI"):
		try:
			import wmi
			globals()["WMI"] = WMI = wmi.WMI()
		except Exception:
			traceback.print_exc()
			globals()["WMI"] = False
	if globals().get("WMI") is not False:
		if ram_name == "RAM":
			if not globals().get("wRAM"):  
				ram = globals()["wRAM"] = WMI.Win32_PhysicalMemory()[0]
			else:
				ram = globals()["wRAM"]
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
				ram_class = "DDR" + str(max(1, ceil(log2(ram_speed / 250))))
			ram_name = globals()["RAM_NAME"] = f"{ram_class}-{ram_speed}"
	return dict(
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
	)


if "math" in CAPS:
	import x_math
	x_math.register_print_fn(print)

def max_size(w, h, maxsize, force=False):
	s = w * h
	m = maxsize * maxsize
	if s > m or force:
		r = (m / s) ** 0.5
		w = round(w * r)
		h = round(h * r)
	return w, h


if "browse" in CAPS:
	import shutil
	import playwright  # noqa: F401
	import streamshatter

	def browse(q, text=True):
		if not is_url(q):
			raise ValueError(q)
		with new_playwright_page("chromium", dict(width=960, height=540)) as page:
			page.goto(q, wait_until="domcontentloaded", timeout=30000)
			time.sleep(0.25)
			bbox = page.locator("html").bounding_box()
			w = max(960, ceil(bbox["width"]))
			h = max(540, ceil(bbox["height"]))
			if w != 960 or h != 540:
				page.set_viewport_size(dict(width=w, height=h))
				time.sleep(0.25)
				bbox = page.locator("html").bounding_box()
				w2 = max(960, ceil(bbox["width"]))
				h2 = max(540, ceil(bbox["height"]))
				if w != w2 or h != h2:
					page.set_viewport_size(dict(width=w2, height=h2))
					time.sleep(0.25)
			return page.screenshot()

	def wolframalpha(q):
		with new_playwright_page() as page:
			page.goto(f"https://www.wolframalpha.com/input?i={urllib.parse.quote_plus(q)}", timeout=8000)
			time.sleep(8)
			for i in range(30):
				time.sleep(1)
				elems = page.locator("h2, img").all()
				texts = [e.text_content() or e.get_attribute("alt") for e in elems]
				text = "\n".join(filter(bool, texts)).strip()
				if text and len(texts) >= 2:
					return text
			raise TimeoutError(q)


def write_to(fn, data):
	with open(fn, "wb") as f:
		f.write(data)

def write_video(proc, data):
	try:
		i = 0
		while i < len(data):
			proc.stdin.write(data[i:i + 65536])
			i += 65536
		proc.stdin.close()
	except Exception:
		print(traceback.format_exc(), end="")

def gifsicle(out, info=None, heavy=False):
	if "." in out:
		out2 = out.rsplit(".", 1)[0] + "~2." + out.rsplit(".", 1)[-1]
	else:
		out2 = out + "~2"
	args = ["binaries/gifsicle"]
	if heavy:
		args.extend(("-O3", "--lossy=90", "--colors=64"))
	else:
		args.extend(("-O", "--lossy=50"))
	args.extend(("--dither", "--careful", "--loopcount=forever", "-o", out2, out))
	print(args)
	try:
		subprocess.run(args, stdin=subprocess.DEVNULL, timeout=90)
	except subprocess.TimeoutExpired:
		pass
	else:
		if os.path.getsize(out2) < os.path.getsize(out):
			return out2
	return out

def avifsicle(out, q=100, s=6):
	out2 = temporary_file("avif")
	args = ["binaries/avifenc", out, "-q", str(q), "--qalpha", str(q if q == 100 else q - 10), "-s", str(s), out2]
	print(args)
	subprocess.run(args, stdin=subprocess.DEVNULL)
	return out2

def ffmpeg_opts(new, frames, count, mode, first, fmt, fs, w, h, duration, opt, vf=""):
	env = os.environ.copy()
	anim = count > 1
	command = ["-i", "-"]
	if fmt in ("gif", "apng"):
		command.extend(("-gifflags", "-offsetting"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba,"
		vf += "split[s0][s1];[s0]palettegen="
		if mode == "RGBA":
			vf += "reserve_transparent=1:"
		else:
			vf += "reserve_transparent=0:"
		if opt and first.width * first.height <= 16384:
			frames = list(frames)
			new["frames"] = frames
			fr = frames.copy()
			first = fr.pop(0)
			last = fr.pop(-1)
			cols = set()
			cols.add((0, 0, 0))
			random.shuffle(fr)
			fr = [first, last, *fr]
			c = min(len(fr), max(5, floor(sqrt(len(fr)))))
			for i, f in enumerate(fr[:c]):
				if isinstance(f, Image.Image):
					im = f.resize((7, 7), resample=Resampling.NEAREST)
					R, G, B = np.asanyarray(im).T[:3]
					for r, g, b in zip(R.ravel(), G.ravel(), B.ravel()):
						l = max(r, g, b)
						if l < 1:
							continue
						t = tuple(min(255, round(log2(x / l * 255 + 1) * 8) * 4) for x in (r, g, b))
						if t not in cols:
							cols.add(t)
			mc = min(128, max(4, 2 ** ceil(log2(len(cols)) + 1)))
			vf += f"max_colors={mc}:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra3:diff_mode=rectangle"
		elif count > 4096:
			vf += "max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
		else:
			vf += "stats_mode=diff[p];[s1][p]paletteuse=dither=sierra3:diff_mode=rectangle"
		if "A" in mode:
			vf += ":alpha_threshold=128"
		if vf:
			command.extend(("-vf", vf))
		if anim:
			if fmt == "apng":
				command.extend(("-plays", "0"))
			else:
				command.extend(("-loop", "0"))
		command.extend(("-f", fmt))
	elif fmt == "y4m":
		lossless = not anim and not opt
		if not anim:
			command.extend(("-vframes", "1", "-r", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		command.extend(("-strict", "-1", "-f", "yuv4mpegpipe"))
		if mode == "RGBA":
			command.extend(("-pix_fmt", "yuva444p"))
		else:
			command.extend(("-pix_fmt", "yuv444p"))
	elif fmt == "webp":
		lossless = not anim and not opt
		if not anim:
			command.extend(("-vframes", "1", "-r", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		pix = ("rgba" if lossless else "yuva444p") if mode == "RGBA" else ("rgb24" if lossless else "yuv444p")
		if opt:
			if pix == "yuva444p":
				pix = "yuva420p"
			else:
				pix = "yuv420p"
		if mode == "RGBA":
			command.extend(("-c:v", "libwebp_anim" if anim else "libwebp", "-pix_fmt", pix, "-pred", "mixed"))
		else:
			command.extend(("-c:v", "libwebp_anim" if anim else "libwebp", "-pix_fmt", pix, "-pred", "mixed"))
		command.extend(("-f", "webp", "-compression_level", "6" if lossless else "5"))
		if anim:
			if opt > 1:
				command.extend(("-loop", "0", "-q:v", "0"))
			elif opt > 1:
				command.extend(("-loop", "0", "-q:v", "33"))
			elif opt:
				command.extend(("-loop", "0", "-q:v", "75"))
			else:
				command.extend(("-loop", "0", "-q:v", "95"))
		elif lossless:
			command.extend(("-lossless", "1"))
		else:
			command.extend(("-q:v", "95"))
	elif fmt == "png":
		command.extend(("-vframes", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		if mode == "RGBA":
			command.extend(("-c:v", "png", "-pix_fmt", "rgba", "-pred", "mixed", "-compression_level", "5"))
		else:
			command.extend(("-c:v", "png", "-pix_fmt", "rgb24", "-pred", "mixed", "-compression_level", "5"))
		command.extend(("-f", "image2pipe"))
	elif fmt == "jpg":
		command.extend(("-vframes", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area"
			command.extend(("-vf", vf))
		if mode == "RGBA":
			command.extend(("-c:v", "mjpeg", "-pix_fmt", "yuva420p", "-pred", "mixed"))
		else:
			command.extend(("-c:v", "mjpeg", "-pix_fmt", "yuv420p", "-pred", "mixed"))
		command.extend(("-f", "image2pipe"))
	else:
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		if getattr(first, "audio", None):
			command.extend(("-i", first.audio["url"]))
			if first.audio["codec"] not in ("mp3", "mpeg", "ogg", "opus", "aac"):
				command.extend(("-c:a", "libopus", "-b:a", "224k"))
			else:
				command.extend(("-c:a", "copy"))
		else:
			command.extend(("-c:a", "copy"))
		if first.width & 1 or first.height & 1:
			w = round(first.width / 2) * 2
			h = round(first.height / 2) * 2
			command.extend(("-vf", f"scale={w}:{h}:flags=bicubic"))
		bitrate = floor(min(fs / duration * 7.5, 99999999)) # use 7.5 bits per byte
		if opt > 1:
			bitrate *= 0.8
		if opt:
			bitrate *= 0.9
		command.extend(("-b:v", str(bitrate), "-vbr", "on"))
		cdc = CODEC_FFMPEG.get(fmt, "av1_nvenc")
		fmt = CODECS.get(fmt, fmt)
		pix_fmt = CODEC_PIX.get(cdc, "yuv420p")
		if cdc == "av1_nvenc":
			if w < 144 or h < 144:
				cdc = "libaom-av1"
				pix_fmt = "yuv444p"
				command.extend(("-cpu-used", "6"))
			else:
				import pynvml
				env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
				av1_devices = [str(i) for i, info in enumerate(map(lambda n: pynvml.nvmlDeviceGetCudaComputeCapability(pynvml.nvmlDeviceGetHandleByIndex(n)), range(pynvml.nvmlDeviceGetCount()))) if info[0] > 8 or info[0] == 8 and info[1] >= 9]
				env["CUDA_VISIBLE_DEVICES"] = ",".join(av1_devices)
				command.insert(0, "-hwaccel_device")
				command.insert(1, av1_devices[0])
		command.extend(("-pix_fmt", pix_fmt, "-c:v", cdc))
		command.extend(("-f", fmt))
	return command, env, fmt

def save_into(im, size, fmt, fs, r=0, opt=False):
	assert size[0] and size[1], f"Expected non-zero size, got {size}"
	thresh = 0.984375
	heavy = r > thresh or np.prod(size) <= 1048576
	if fmt not in ("bmp", "ico", "jp2") and ("RGB" in im.mode and np.prod(size) > 65536 or fmt not in ("png", "jpg", "webp", "gif")):
		b = np.asanyarray(im, dtype=np.uint8).data
		pix = "rgb24" if im.mode == "RGB" else "rgba"
		args = ["ffmpeg", "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-video_size", "x".join(map(str, im.size))]
		is_avif = fmt == "avif" and im.mode == "RGBA"
		if is_avif:
			fmt = "y4m"
		opts, env, fmt = ffmpeg_opts({}, iter([im]), 1, im.mode, im, fmt, fs * (r or 1), size[0], size[1], 1, opt)
		args.extend(opts)
		print(im, len(b))
		if fmt in ("png", "jpg", "webp"):
			args.append("-")
			print(args)
			return subprocess.run(args, stdout=subprocess.PIPE, input=b, env=env).stdout
		else:
			out = temporary_file(fmt)
			args.append(out)
			print(args)
			subprocess.run(args, input=b, env=env)
			if is_avif:
				out = avifsicle(out, q=60 if opt else 100, s=1)
			assert os.path.exists(out) and os.path.getsize(out), f"Expected output file {out}"
			with open(out, "rb") as f:
				return f.read()
	fmt = dict(jpg="jpeg", gif="png").get(fmt.lower(), fmt)
	if im.size != tuple(size):
		im = im.resize(size, resample=Resampling.LANCZOS if np.prod(size) <= 1048676 else Resampling.BICUBIC)
	out = io.BytesIO()
	if fmt in ("webp", "jp2"):
		if heavy:
			im.save(out, format="webp", lossless=True, quality=80, method=6)
		else:
			im.save(out, format="webp", lossless=True, quality=80, method=3)
			r = fs / out.tell()
			if r > thresh:
				out = io.BytesIO()
				im.save(out, format="webp", lossless=True, quality=100, method=6)
	elif fmt in ("png", "ico"):
		if heavy:
			im.save(out, format="png", optimize=True, compress_level=9)
		else:
			im.save(out, format="png", optimize=True, compress_level=5)
			r = fs / out.tell()
			if r > thresh:
				out = io.BytesIO()
				im.save(out, format="png", optimize=True, compress_level=9)
	else:
		if fmt in ("jpeg", "bmp"):
			im, A = split_rgba(im)
		im.save(out, format=fmt, optimize=True)
	return out.getbuffer()

def anim_into(out, new, first, size, fmt, fs, r=0, opt=False):
	assert size[0] and size[1], f"Expected non-zero size, got {size}"
	command = ["ffmpeg", "-nostdin", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel]
	mode = new["mode"]
	command.extend((
		"-f", "rawvideo", "-framerate", str(new["fps"]), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
		"-video_size", "x".join(map(str, first.size)),
	))
	is_avif = fmt == "avif" and first.mode == "RGBA"
	if is_avif:
		fmt = "y4m"
	opts, env, fmt = ffmpeg_opts(new, new["frames"], new["count"], mode, first, fmt, fs, *size, new["duration"], opt)
	command.extend(opts)
	if "." in out:
		out2 = out.rsplit(".", 1)[0] + "~2." + CODECS.get(fmt, fmt)
	else:
		out2 = out + "~2"
	command.append(out2)
	print(command)
	proc = psutil.Popen(command, stdin=subprocess.PIPE, env=env)
	frames = []
	for frame in new["frames"]:
		frames.append(frame)
		b = np.asanyarray(frame, dtype=np.uint8).data
		proc.stdin.write(b)
	new["frames"] = frames
	proc.stdin.close()
	proc.wait()
	assert os.path.exists(out2) and os.path.getsize(out2), f"Expected output file {out2}"
	print(os.path.getsize(out2), fs, np.prod(size) * new["count"])
	if is_avif:
		out = avifsicle(out2, q=75, s=min(10, 1 + round(log2(new["count"]))))
	elif fmt == "gif" and (fs >= 1048576 or "A" not in mode) and np.prod(size) * new["count"] <= 67108864 and os.path.getsize(out2) < fs * 3:
		out = gifsicle(out2, new, heavy=os.path.getsize(out2) > fs * 1.5)
	else:
		out = out2
	with open(out, "rb") as f:
		return f.read()


statics = ("png", "bmp", "jpg", "heic", "ico", "icns", "j2k", "tga", "tiff", "pdf")
# Main image operation function
def evalImg(url, operation, args):
	ts = time.time_ns() // 1000
	out = temporary_file("avif", name=ts)
	fmt = "auto"
	cdc = "webp"
	fs = inf
	dur = None
	maxframes = inf
	nogif = False
	oz = False
	if len(args) > 1 and args[-2] == "-f":
		fmt = args.pop(-1) or fmt
		args.pop(-1)
	if len(args) > 1 and args[-2] == "-fs":
		fs = floor(float(args.pop(-1)))
		args.pop(-1)
	if len(args) > 1 and args[-2] == "-d":
		dur = args.pop(-1)
		args.pop(-1)
	if args and args[-1] == "-o":
		opt = True
		args.pop(-1)
	elif args and args[-1] == "-oz":
		maxframes = 16
		oz = True
		opt = False
		args.pop(-1)
	else:
		opt = False
	if fmt in statics and not oz:
		maxframes = 1
		nogif = True
	if args and args[-1] == "-bg":
		bg = True
		args.pop(-1)
	else:
		bg = False
	if operation not in ("$", "&"):
		# print("IOPER:", operation, args)
		if args and args[0] == "-nogif":
			nogif = args.pop(0)
		if args and args[-1] == "-raw":
			args.pop(-1)
			image = get_request(url)
		else:
			if args and args[-1] == "-nodel":
				nodel = args.pop(-1)
			else:
				nodel = False
			if operation == "resize_max":
				msize = args[1]
			else:
				msize = None
			image = get_image(url, out, nodel=nodel, nogif=nogif, maxframes=maxframes, msize=msize)
		new = eval(operation)(image, *args)
	else:
		new = eval(url)(*args)
		# print("OPER:", new)
	if Image and isinstance(new, Image.Image):
		if getattr(new, "audio", None):
			new = dict(count=1, duration=1, frames=[new])
		else:
			prop = properties(new)
			if prop[0] > 1:
				def iter_img(im):
					for i in range(prop[0]):
						im.seek(i)
						if i == prop[0] - 1:
							yield im
						else:
							yield im.copy()
				new = dict(count=prop[0], duration=prop[1], frames=iter_img(new))
				print("Output anim:", new, prop)
			else:
				print("Output stat:", new, prop)
	if isinstance(new, dict) and "frames" in new:
		# print(nogif, oz, new)
		frames = optimise(new["frames"])
		if not frames:
			raise EOFError("No image output detected.")
		video = False
		if nogif:
			new["frames"] = [next(iter(frames))]
		elif new["count"] == 1 or not new.get("duration"):
			temp = next(iter(frames))
			video = getattr(temp, "audio", None)
			if not video:
				new["frames"] = [temp]
			else:
				duration = dur = 3600000
				new["count"] = 16
				new["frames"] = [temp] * new["count"]
		elif oz:
			frames = list(frames)
			first = frames[0]
			count = min(new["count"], maxframes)
			if len(frames) > count:
				frames = [frames[round(i / (count - 1) * (len(frames) - 1))] for i in range(count)]
			rows = ceil(sqrt(count))
			cols = ceil(count / rows)
			im = Image.new(first.mode, (first.width * cols, first.height * rows))
			print("Template:", im)
			for i, frame in enumerate(frames):
				y = i // cols
				x = i % cols
				im.paste(frame, (x * first.width, y * first.height))
			new["frames"] = [im]
			new["count"] = 1
			new["duration"] = 1
		else:
			video = True
		duration = new["duration"]
		count = new.get("count", 1)
		if dur:
			dur *= new["count"] / (new["count"] + 1)
			if duration > dur:
				duration = dur
		if video:
			if fmt in statics:
				fmt, cdc = "zip", fmt
			print("DURATION:", duration, new["count"])
			fps = new["count"] / duration
			if round(fps, 2) == round(fps):
				fps = round(fps)
			else:
				fps = round(fps, 4)
			new["fps"] = fps
			real_frames = {}
			if issubclass(type(frames), collections.abc.Sequence):
				first = frames[0]
			else:
				it = iter(frames)
				first = next(it)
				frames = resume(first, it)
			size = first.size
			if fmt == "auto":
				if getattr(first, "audio", None) or new["count"] * np.prod(size) > 1073741824:
					fmt = "mp4"
					cdc = "av1_nvenc"
				else:
					fmt = "avif"
			out = temporary_file(CODECS.get(fmt, fmt), name=ts)
			mode = str(first.mode)
			if mode == "P":
				raise RuntimeError("Unexpected P mode image")
			if fmt == "gif" and "A" in mode:
				frames = clamp_transparency(frames)
			archive = False
			is_avif = False
			if fmt == "zip":
				import zipfile
				resp = zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
				archive = resp.writestr
			elif fmt == "tar":
				import tarfile
				resp = tarfile.TarFile(out, "w")
				def archive(fn, data):
					tarinfo = tarfile.TarInfo(name=fn)
					tarinfo.size = len(data)
					return resp.addfile(tarinfo, io.BytesIO(data))
			else:
				command = ["ffmpeg", "-nostdin", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel]
				# if hwaccel == "cuda":
				# 	if COMPUTE_CAPS:
				# 		try:
				# 			devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
				# 		except IndexError:
				# 			devid = random.randint(0, len(COMPUTE_CAPS))
				# 			fmt = "webp"
				# 	else:
				# 		devid = random.randint(0, ceil(len(COMPUTE_CAPS) / 2))
				# 	command.extend(("-hwaccel_device", str(devid)))
				command.extend([
					"-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
					"-video_size", "x".join(map(str, size)),
				])
				new["frames"] = frames
				is_avif = fmt == "avif" and mode == "RGBA"
				if is_avif:
					fmt = "y4m"
				opts, env, fmt = ffmpeg_opts(new, frames, count, mode, first, fmt, fs, *size, duration, False)
				out = temporary_file(CODECS.get(fmt, fmt), name=ts)
				frames = new.get("frames") or frames
				command.extend(opts)
				command.append(out)
				print(command)
				proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1048576, env=env)
			i = None
			futs = []
			for i, frame in enumerate(frames):
				def save_frame(i, frame):
					if isinstance(frame, Image.Image):
						if frame.size != size:
							frame = frame.resize(size)
						if frame.mode != mode:
							frame = frame.convert(mode)
						if bg and "A" in mode:
							if mode != "RGBA":
								frame = frame.convert("RGBA")
							i2 = gen_bg(size)
							i3 = Image.alpha_composite(i2, frame)
							frame = i3.convert("RGB")
						if archive:
							b = save_into(frame, frame.size, cdc, inf)
						else:
							b = np.asanyarray(frame, dtype=np.uint8).data
					elif type(frame) is io.BytesIO:
						frame.seek(0)
						im = Image.open(frame)
						if archive:
							b = save_into(im, im.size, cdc, inf)
						else:
							b = frame.read()
					else:
						im = Image.open(io.BytesIO(frame))
						if archive:
							b = save_into(im, im.size, cdc, inf)
						else:
							b = frame
					if not real_frames or fmt in ("gif", "webp", "apng", "avif"):
						real_frames[i] = frame
					return b
				workers = [fut for fut in futs if not fut.done()]
				if len(workers) >= 12:
					for fut in workers:
						fut.result()
				futs.append(exc.submit(save_frame, i, frame))
			print(i, first)
			for i, fut in enumerate(futs):
				b = fut.result()
				if archive:
					n = ceil(log10(new["count"]))
					s = f"%0{n}d" % i
					archive(f"{s}.{cdc}", data=b)
				else:
					proc.stdin.write(b)
			if archive:
				resp.close()
			else:
				proc.stdin.close()
				proc.wait()
			if is_avif:
				out = avifsicle(out, q=60 if opt else 80 if new["count"] > 1 else 100, s=min(10, round(log2(new["count"]))))
				fmt = "avif"
			if not archive:
				print(os.path.getsize(out), fs, np.prod(size) * new["count"])
				if fmt == "gif" and (fs >= 1048576 or "A" not in mode) and first.width * first.height * new["count"] <= 67108864 and os.path.getsize(out) < fs * 3:
					out = gifsicle(out, new, heavy=os.path.getsize(out) > fs * 1.5)
				new["mode"] = mode
				new["frames"] = list(map(real_frames.get, sorted(real_frames)))
				print("LEN:", len(real_frames))
				orig = out
				with open(orig, "rb") as f:
					out = f.read()
				if len(out) > fs:
					w, h = cw, ch = first.width, first.height
					scale = 1
					prec = 0.25
					lower_bound = 0
					upper_bound = 1
					seen = {}
					r = fs / len(out)
					for i in range(4):
						if r >= 1 and r <= 1 + prec:
							upper_bound = lower_bound = scale
							break
						scale *= r / (1 + prec)
						effsize = sqrt(cw * ch * scale)
						w, h = max_size(cw, ch, effsize)
						if w > 16 and h > 16:
							w = round(w / 2) * 2
							h = round(h / 2) * 2
						if (w, h) in seen:
							out, r = seen[(w, h)]
						else:
							rat = np.prod((w, h)) / np.prod(size)
							if rat < 1 / 16:
								opt = 3
							elif rat < 1 / 8:
								opt = 2
							elif rat < 1 / 4:
								opt = 1
							out = anim_into(orig, new, first, (w, h), fmt, fs, r=r, opt=opt)
							r = fs / len(out)
							print("RA:", w, h, scale, len(out), r)
							seen[(w, h)] = out, r
						if r <= 1:
							upper_bound = scale
						else:
							lower_bound = scale
					while r < 1 or upper_bound - lower_bound > max(prec, 1 / sqrt(cw * ch)):
						scale = (lower_bound + upper_bound) / 2
						effsize = sqrt(cw * ch * scale)
						w, h = max_size(cw, ch, effsize)
						if w > 16 and h > 16:
							w = round(w / 2) * 2
							h = round(h / 2) * 2
						if (w, h) in seen:
							out, r = seen[(w, h)]
						else:
							rat = np.prod((w, h)) / np.prod(size)
							if rat < 1 / 16:
								opt = 3
							elif rat < 1 / 8:
								opt = 2
							elif rat < 1 / 4:
								opt = 1
							out = anim_into(orig, new, first, (w, h), fmt, fs, r=r, opt=opt)
							r = fs / len(out)
							print("RB:", w, h, scale, len(out), r)
							seen[(w, h)] = out, r
						if r <= 1:
							upper_bound = scale
						else:
							lower_bound = scale
					print("F:", w, h, scale, len(out), r)
			if isinstance(out, str):
				assert os.path.exists(out) and os.path.getsize(out), f"Expected output file {out}"
				with open(out, "rb") as f:
					return f.read()
			return out
		else:
			new = next(iter(new["frames"]))
	if Image and isinstance(new, Image.Image):
		if fmt == "auto":
			fmt = "webp"
		new = optimise(new, keep_rgb=False)
		if bg and "A" in new.mode:
			if new.mode != "RGBA":
				new = new.convert("RGBA")
			i2 = gen_bg(new.size)
			i3 = Image.alpha_composite(i2, new)
			new = i3.convert("RGB")
		out = save_into(new, new.size, fmt, fs)
		if len(out) > fs:
			w, h = cw, ch = new.width, new.height
			scale = 1
			prec = 0.001
			lower_bound = 0
			upper_bound = 1
			seen = {}
			r = fs / len(out)
			for i in range(4):
				if r >= 1 and r <= 1 + prec:
					upper_bound = lower_bound = scale
					break
				scale *= r / (1 + prec)
				effsize = sqrt(cw * ch * scale)
				w, h = max_size(cw, ch, effsize)
				if w > 16 and h > 16:
					w = round(w / 2) * 2
					h = round(h / 2) * 2
				if (w, h) in seen:
					out, r = seen[(w, h)]
				else:
					out = save_into(new, (w, h), fmt, fs, r=r, opt=True)
					r = fs / len(out)
					print("RA:", w, h, scale, len(out), r)
					seen[(w, h)] = out, r
				if r <= 1:
					upper_bound = scale
				else:
					lower_bound = scale
			while r < 1 or upper_bound - lower_bound > max(prec, 1 / sqrt(cw * ch)):
				scale = (lower_bound + upper_bound) / 2
				effsize = sqrt(cw * ch * scale)
				w, h = max_size(cw, ch, effsize)
				if w > 16 and h > 16:
					w = round(w / 2) * 2
					h = round(h / 2) * 2
				if (w, h) in seen:
					out, r = seen[(w, h)]
				else:
					out = save_into(new, (w, h), fmt, fs, r=r, opt=True)
					r = fs / len(out)
					print("RB:", w, h, scale, len(out), r)
					seen[(w, h)] = out, r
				if r <= 1:
					upper_bound = scale
				else:
					lower_bound = scale
			print("F:", w, h, scale, len(out), r)
		return out
	elif type(new) is str and new.startswith("$"):
		return new[1:]
	return new

def evaluate_image(args):
	out = evalImg(*args)
	if out and isinstance(out, (tuple, list)) and isinstance(out[0], (io.BytesIO, memoryview, Image.Image)):
		if isinstance(out[0], io.BytesIO):
			for i in range(len(out)):
				out[i].seek(i)
				out[i] = out[i].read()
		elif isinstance(out[0], memoryview):
			for i in range(len(out)):
				out[i] = bytes(out[i])
		elif isinstance(out[0], Image.Image):
			for i in range(len(out)):
				out[i] = save_into(out[i], out[i].size, "webp", inf)
	elif isinstance(out, io.BytesIO):
		out.seek(0)
		out = out.read()
	elif isinstance(out, memoryview):
		out = bytes(out)
	return out


if __name__ == "__main__":
	interface.start(background=False)