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
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import io
import time
import concurrent.futures
import asyncio
import itertools
import subprocess
import psutil
import collections
import traceback
import re
import niquests
import ast
import base64
import hashlib
import random
import urllib
import numpy as np
from contextlib import suppress
from math import inf, floor, ceil, log2, log10
from traceback import print_exc
sys.path.append("misc")
from misc.util import EvalPipe, new_playwright_page, CODECS, CODEC_FFMPEG

if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	print = interface.print

requests = niquests.Session()
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

CACHE = {}

def wrap_future(fut, loop=None):
	if loop is None:
		loop = asyncio.main_new_loop
	wrapper = loop.create_future()

	def set_suppress(res, is_exception=False):
		try:
			if is_exception:
				wrapper.set_exception(res)
			else:
				wrapper.set_result(res)
		except (RuntimeError, asyncio.InvalidStateError):
			pass

	def on_done(*void):
		try:
			res = fut.result()
		except Exception as ex:
			loop.call_soon_threadsafe(set_suppress, ex, True)
		else:
			loop.call_soon_threadsafe(set_suppress, res)

	fut.add_done_callback(on_done)
	return wrapper

async def _await_fut(fut, ret):
	out = await fut
	ret.set_result(out)
	return ret

def await_fut(fut, timeout=None):
	return convert_fut(fut).result(timeout=timeout)

def convert_fut(fut):
	loop = asyncio.main_new_loop
	try:
		ret = asyncio.run_coroutine_threadsafe(fut, loop=loop)
	except Exception:
		ret = concurrent.futures.Future()
		loop.create_task(_await_fut(fut, ret))
	return ret

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

if CAPS.intersection(("browse", "image", "caption", "video", "sd", "sdxl", "scc")):
	from math import *
	import x_image
	x_image.print = print
	from x_image import *
	from PIL import Image
else:
	Image = None
	exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)

	if not hasattr(time, "time_ns"):
		time.time_ns = lambda: int(time.time() * 1e9)


def enumerate_with_next(iterable):
	it1, it2 = itertools.tee(iterable)
	next(it2, None)
	for index, (current, next_val) in enumerate(zip(it1, it2)):
		yield index, current, next_val
	if it1 and it2:
		for index, current in enumerate(it1, start=index + 1):
			yield index, current, None


def is_strict_running(proc):
	"Detects if a process is truly running. Zombie processes are treated as dead."
	if not proc:
		return
	try:
		if getattr(proc, "returncode", None) is not None:
			return False
		if hasattr(proc, "poll") and proc.poll() is not None:
			return False
		if not proc.is_running():
			return False
		try:
			if os.name != "nt" and proc.status() == "zombie":
				proc.wait()
				return
		except (ProcessLookupError, psutil.NoSuchProcess):
			return
		return True
	except AttributeError:
		try:
			proc = psutil.Process(proc.pid)
		except (ProcessLookupError, psutil.NoSuchProcess):
			return
		except:
			print_exc()
			return
	if not proc.is_running():
		return False
	try:
		if os.name != "nt" and proc.status() == "zombie":
			proc.wait()
			return
	except (ProcessLookupError, psutil.NoSuchProcess):
		return
	return True


if "video" in CAPS:
	sizecheck = re.compile("[1-9][0-9]*x[0-9]+")
	fpscheck = re.compile("[0-9]+ fps")

if "math" in CAPS:
	import matplotlib.pyplot as plt
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
		ts = time.time_ns() // 1000
		out = f"cache/{ts}.png"
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

	def rank_embeddings(embs, emb, temp=0.5):
		btest = base64.b64decode(emb)
		y = np.frombuffer(btest, dtype=np.float16)
		blist = [base64.b64decode(line) for line in embs]
		bt2 = b"".join(blist)
		x = np.frombuffer(bt2, dtype=np.float16)
		x = x.reshape((len(x) // len(y), len(y)))
		y = y.reshape((1, len(y)))
		norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
		z = (x * y).sum(axis=1)
		z /= norms
		top = np.max(z)
		return [i for i in np.argsort(z)[::-1] if z[i] - random.random() / 3 >= (top - temp * 2 / 3)]

if "ecdc" in CAPS:
	class PipedProcess:

		procs = ()
		stdin = stdout = stderr = None

		def __init__(self, *args, stdin=None, stdout=None, stderr=None, cwd=".", bufsize=4096):
			if not args:
				return
			self.exc = concurrent.futures.ThreadPoolExecutor(max_workers=len(args) - 1) if len(args) > 1 else None
			self.procs = []
			for i, arg in enumerate(args):
				first = not i
				last = i >= len(args) - 1
				si = stdin if first else subprocess.PIPE
				so = stdout if last else subprocess.PIPE
				se = stderr if last else None
				proc = psutil.Popen(arg, stdin=si, stdout=so, stderr=se, cwd=cwd, bufsize=bufsize * 256)
				if first:
					self.stdin = proc.stdin
					self.args = arg
				if last:
					self.stdout = proc.stdout
					self.stderr = proc.stderr
				self.procs.append(proc)
			for i in range(len(args) - 1):
				self.exc.submit(self.pipe, i, bufsize=bufsize)
			self.pid = self.procs[0].pid

		def pipe(self, i, bufsize=4096):
			try:
				proc = self.procs[i]
				proc2 = self.procs[i + 1]
				si = 0
				while proc.is_running() and proc2.is_running():
					b = proc.stdout.read(si * (si + 1) * bufsize // 8 + bufsize)
					if not b:
						break
					proc2.stdin.write(b)
					proc2.stdin.flush()
					si += 1
				if proc2.is_running():
					proc2.stdin.close()
			except:
				import traceback
				traceback.print_exc()
				if not proc.is_running() or not proc2.is_running():
					self.terminate()
			if self.exc:
				self.exc.shutdown(wait=False)

		def is_running(self):
			for proc in self.procs:
				if proc.is_running():
					return True
			return False

		def terminate(self):
			for proc in self.procs:
				proc.terminate()

		def kill(self):
			for proc in self.procs:
				proc.kill()

		def wait(self):
			for proc in self.procs:
				proc.wait()

		def status(self):
			return self.procs[-1].status()

	def ecdc_encode(b, bitrate="24", name="", source="", thumbnail=""):
		assert len(b)
		ts = time.time_ns() // 10000 * 10 + int(DEV)
		fn = "cache/" + str(ts)
		with open(fn, "wb") as f:
			f.write(b)
		fo = "cache/" + str(ts) + ".ecdc"
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
		ts = time.time_ns() // 10000 * 10 + int(DEV)
		fn = "cache/" + str(ts)
		with open(fn, "wb") as f:
			f.write(b)
		fo = "cache/" + str(ts) + "." + fmt
		args1 = [sys.executable, "misc/ecdc_stream.py", "-b", "0", "-g", str(DEV), "-d", fn]
		args2 = ["ffmpeg", "-v", "error", "-hide_banner", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "96k", fo]
		print(args1)
		print(args2)
		PipedProcess(args1, args2).wait()
		with open(fo, "rb") as f:
			return f.read()


CBOTS = {}
def cb_exists(cid):
	return cid in CBOTS

mcache = {}
def backup_model(cls, model, force=False, **kwargs):
	kwargs.pop("resume_download", None)
	t = (cls, model, str(kwargs))
	try:
		return mcache[t].result()
	except KeyError:
		mcache[t] = fut = concurrent.futures.Future()
	if force:
		try:
			fut.set_result(cls(model, resume_download=True, **kwargs))
			return fut.result()
		except Exception as ex:
			fut.set_exception(ex)
			ex2 = ex
	else:
		try:
			fut.set_result(cls(model, local_files_only=True, **kwargs))
			return fut.result()
		except Exception as ex:
			fut.set_exception(ex)
			mcache[t] = fut = exc.submit(cls, model, resume_download=True, **kwargs)
			try:
				return fut.result(timeout=24)
			except Exception as ex:
				ex2 = ex
	if isinstance(ex2, concurrent.futures.TimeoutError):
		try:
			return fut.result(timeout=60)
		except concurrent.futures.TimeoutError:
			raise RuntimeError("Model is loading, please wait...")
	raise ex2

def determine_cuda(mem=1, priority=None, multi=False, major=0):
	if not torch or not DEVICES or not torch.cuda.is_available():
		if multi:
			return [-1], torch.float32
		return -1, torch.float32
	n = torch.cuda.device_count()
	if not n:
		if multi:
			return [-1], torch.float32
		return -1, torch.float32
	global pynvml
	if not globals().get("pynvml"):
		import pynvml
		pynvml.nvmlInit()
	dc = pynvml.nvmlDeviceGetCount()
	handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
	gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) for d in handles]
	tinfo = [torch.cuda.get_device_properties(COMPUTE_ORDER.index(i)) if i in COMPUTE_ORDER else None for i in range(dc)]
	COMPUTE_LOAD = globals().get("COMPUTE_LOAD") or [0] * dc
	high = max(COMPUTE_LOAD)
	if priority == "full":
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, p.major, p.minor, p.multi_processor_count, p.total_memory)
	elif priority:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, p.major >= major, COMPUTE_LOAD[i] < high * 0.9, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, i, p.multi_processor_count, p.total_memory)
	elif priority is False:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, -mem // 1073741824, p.major, p.minor, COMPUTE_LOAD[i] < high * 0.75, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -gmems[i].free, p.multi_processor_count)
	else:
		key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] < high * 0.5, p.major >= major, p.major >= 7, -p.major, -p.minor, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -p.multi_processor_count, -gmems[i].free)
	pcs = sorted(DEVICES, key=key, reverse=True)
	if multi:
		return [COMPUTE_ORDER.index(i) for i in pcs if gmems[i].free >= mem], torch.float16
	return COMPUTE_ORDER.index(pcs[0]), torch.float16

gcancel = concurrent.futures.Future()
ot = 0
def ensure_gc(t):
	global ot, gcancel
	try:
		ot = max(ot, time.time() + t)
		if gcancel.done():
			gcancel = concurrent.futures.Future()
		else:
			try:
				gcancel.set_result(None)
			except concurrent.futures.InvalidStateError:
				pass
		try:
			gcancel.result(timeout=t)
		except concurrent.futures.TimeoutError:
			pass
		else:
			gcancel = concurrent.futures.Future()
			return
		if ot and time.time() > ot:
			with torch.no_grad():
				torch.cuda.empty_cache()
			ot = 0
	except:
		traceback.print_exc()


if "caption" in CAPS:
	import tiktoken
	try:
		import pytesseract
	except ImportError:
		pytesseract = None

	VIT = True
	dfut = None
	def download_model():
		from clip_interrogator import Config, Interrogator
		Vconfig = Config(
			clip_model_name="ViT-H-14/laion2b_s32b_b79k",
			clip_model_path="misc/Clip",
			caption_model_name="blip-base",
			cache_path="misc/Clip",
			device="cpu",
			caption_max_length=48,
		)
		globals()["VIT"] = Interrogator(Vconfig)
		VIT.dtype = torch.float32
		VIT.device = "cpu"

		from PIL import Image
		import torch
		def image_to_features(self, image: Image.Image) -> torch.Tensor:
			self._prepare_clip()
			images = self.clip_preprocess(image).unsqueeze(0).to(self.device).to(torch.float32)
			with torch.no_grad():
				image_features = self.clip_model.encode_image(images)
				image_features /= image_features.norm(dim=-1, keepdim=True)
			return image_features
		Interrogator.image_to_features = lambda self, image=None: image_to_features(self, image) if image else image_to_features(VIT, self)

		print("Interrogator:", VIT)
		im = Image.new("RGB", (4, 4), (0, 0, 255))
		VIT.caption_model = VIT.caption_model.to(torch.float32).to("cpu")
		VIT.clip_model = VIT.clip_model.to(torch.float32).to("cpu")
		description = VIT.interrogate_fast(im, max_flavors=12)#, caption=caption)
		print("VIT:", description)
		return pytesseract.image_to_string(im, config="--psm 1")
	# dfut = exc.submit(download_model)

	def caption(im, best=False):
		global dfut
		if not best:
			if not dfut:
				dfut = exc.submit(download_model)
			try:
				dfut.result(timeout=1)
			except concurrent.futures.TimeoutError:
				raise RuntimeError("Model is loading, please wait...")
		im = resize_max(im, 1024, resample="auto")
		if im.mode != "RGB":
			image = im.convert("RGB")
		else:
			image = im
		if pytesseract:
			fut = exc.submit(pytesseract.image_to_string, image, config="--psm 1", timeout=8)
		else:
			fut = None
		if not best:
			# cfut = exc.submit(VIT.generate_caption, image)
			desc = VIT.interrogate_fast(image, max_flavors=24)#, caption=" ")
			exc.submit(ensure_gc, 20)
			p1 = desc.lstrip()
			enc = tiktoken.get_encoding("cl100k_base")
			out = []
			otok = list(enc.encode(p1.strip()))
			if len(otok) >= 8:
				last = None
				count = 0
				while otok:
					c = otok.pop(0)
					if c == last:
						if count > 3:
							continue
						count += 1
					else:
						last = c
						count = 0
					out.append(c)
				p1 = enc.decode(out) if len(out) >= 8 else p1
		else:
			p1 = None
		if fut:
			p2 = fut.result().strip()
		else:
			p2 = None
		return (p1, p2)

	def canny(im):
		if "RGB" not in im.mode:
			im = im.convert("RGBA")
		a = np.asanyarray(im)
		from cv2 import Canny
		a2 = Canny(a, 100, 200)
		return fromarray(a2)

# if "summ" in CAPS:
# 	if 0:
# 		from transformers import pipeline
# 		smp = pipeline("summarization", model="Qiliang/bart-large-cnn-samsum-ChatGPT_v3", device=0, torch_dtype=torch.float16)
# 		print(smp)

# 		def summarise(s1, min_length=128, max_length=192, rm=True, do_sample=True):
# 			s2 = smp(s1, max_length=max_length, min_length=min_length, do_sample=do_sample, truncation=True)[0]["summary_text"]
# 			# exc.submit(ensure_gc, 20)
# 			if rm:
# 				return re.sub(r"(?:in )?(?:the|this|some)? *(?:article|essay|page|study|text|report|topic)[s, ]*(?:also mentions|we discuss|we look at|is about|includes|is based on)? *", "", s2, flags=re.I)
# 			return s2

# 	device, dtype = determine_cuda(1073741824, priority=None)
# 	device = f"cuda:{device}" if device >= 0 else "cpu"
# 	from sentence_transformers import SentenceTransformer
# 	Embedder = SentenceTransformer("LLukas22/all-mpnet-base-v2-embedding-all", device=device)
# 	if torch and dtype == torch.float16 and torch.cuda.get_device_properties(device).major >= 7:
# 		try:
# 			Embedder = Embedder.half()
# 		except (RuntimeError, NotImplementedError):
# 			pass
# 	def embedding(s):
# 		a = Embedder.encode(s).astype(np.float16)
# 		exc.submit(ensure_gc, 20)
# 		return a.data

if 0 and "whisper" in CAPS:
	special_languages = {
		"mandarin chinese": "chinese",
		"cantonese chinese": "chinese",
	}
	special_unmap = {
		"chinese": "mandarin chinese",
	}
	usable_languages = {'english', 'chinese', 'german', 'spanish', 'russian', 'korean', 'french', 'japanese', 'portuguese', 'turkish', 'polish', 'catalan', 'dutch', 'arabic', 'swedish', 'italian', 'indonesian', 'hindi', 'finnish', 'vietnamese', 'hebrew', 'ukrainian', 'greek', 'malay', 'czech', 'romanian', 'danish', 'hungarian', 'tamil', 'norwegian', 'thai', 'urdu', 'croatian', 'bulgarian', 'lithuanian', 'latin', 'maori', 'malayalam', 'welsh', 'slovak', 'telugu', 'persian', 'latvian', 'bengali', 'serbian', 'azerbaijani', 'slovenian', 'kannada', 'estonian', 'macedonian', 'breton', 'basque', 'icelandic', 'armenian', 'nepali', 'mongolian', 'bosnian', 'kazakh', 'albanian', 'swahili', 'galician', 'marathi', 'punjabi', 'sinhala', 'khmer', 'shona', 'yoruba', 'somali', 'afrikaans', 'occitan', 'georgian', 'belarusian', 'tajik', 'sindhi', 'gujarati', 'amharic', 'yiddish', 'lao', 'uzbek', 'faroese', 'haitian creole', 'pashto', 'turkmen', 'nynorsk', 'maltese', 'sanskrit', 'luxembourgish', 'myanmar', 'tibetan', 'tagalog', 'malagasy', 'assamese', 'tatar', 'hawaiian', 'lingala', 'hausa', 'bashkir', 'javanese', 'sundanese', 'burmese', 'valencian', 'flemish', 'haitian', 'letzeburgesch', 'pushto', 'panjabi', 'moldavian', 'moldovan', 'sinhalese', 'castilian'}
	common_languages = {"english", "mandarin chinese", "hindi", "spanish", "french", "arabic", "bengali", "portuguese", "russian", "urdu", "indonesian", "german", "japanese", "telugu", "turkish", "tamil", "vietnamese", "korean", "persian", "javanese", "italian", "thai"}

	import torch
	from transformers import AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, AutoModelForAudioClassification
	w_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id", return_attention_mask=True)
	w_classifier = AutoModelForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id", torch_dtype=torch.float16).to(0)
	print("CLASSIFIER:", w_classifier)
	w_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float16)
	w_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float16).to(0)
	print("MODEL:", w_model)

	def pipe_to(p, f):
		while True:
			b = f.read(1048576)
			if not b or not p.is_running():
				p.stdin.close()
				break
			p.stdin.write(b)
			p.stdin.flush()

	def split_audio(f):
		args = ["ffmpeg", "-hide_banner", "-v", "error", "-i", "-", "-vn", "-f", "s16le", "-ar", "16k", "-ac", "1", "-"]
		print(args)
		proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1048576)
		exc.submit(pipe_to, proc, f)
		buf = []
		emptied = 0
		n = 0
		while True:
			b = proc.stdout.read(16000 // 2)
			if not b:
				break
			a = np.frombuffer(b, dtype=np.int16)
			x = a.astype(np.float32)
			x *= 1 / 32768
			rms = np.sqrt(x.dot(x) / x.size)
			if rms < len(buf) / 256:
				if buf:
					if emptied > 3:
						buf.append(a)
						temp = np.concatenate(buf)
						buf.clear()
						yield temp
						emptied = 0
						n += 1
						continue
					emptied += 1
			if rms > 1 / 1024:
				buf.append(a)
		if buf:
			yield np.concatenate(buf)

	def lang_id(audio, favoured=None):
		floats = torch.from_numpy(audio).to(torch.float16)
		floats *= 1 / 32768
		extracted_features = w_extractor(floats, sampling_rate=16000, return_tensors="pt").input_features
		classes = w_classifier(extracted_features.to(w_classifier.dtype).to(w_classifier.device))
		logits = classes.logits
		for lang in common_languages:
			lid = int(w_classifier.config.label2id[" ".join(s.capitalize() for s in lang.split())])
			logits[:, lid] *= 2
		if favoured:
			lang = special_unmap.get(favoured, favoured)
			lid = int(w_classifier.config.label2id[" ".join(s.capitalize() for s in lang.split())])
			logits[:, lid] += 0.5
		predicted_class_ids = torch.argmax(logits, dim=-1).item()
		language = w_classifier.config.id2label[predicted_class_ids].lower()
		language = special_languages.get(language, language)
		if language not in usable_languages:
			language = "english"
		return language

	def whisper(url, prompt=""):
		if isinstance(url, str):
			f = requests.get(url, headers=header(), stream=True, verify=False, timeout=12).raw
		else:
			f = io.BytesIO(url)
		prompt_ids = w_processor.get_prompt_ids(prompt)
		language = None
		out = []
		with torch.no_grad():
			for audio in split_audio(f):
				language = lang_id(audio, favoured=language)
				input_features = w_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
				forced_decoder_ids = w_processor.get_decoder_prompt_ids(task="transcribe", language=language)
				predicted_ids = w_model.generate(input_features.to(w_model.dtype).to(w_model.device), forced_decoder_ids=forced_decoder_ids, prompt_ids=prompt_ids)
				resp = w_processor.batch_decode(predicted_ids, skip_special_tokens=True)
				out.extend(resp)
			if not out:
				return ""
		s = "".join(out)
		while True:
			m = re.search(r"(.+?)\1{4,}?", s)
			if not m:
				break
			sub = m.group(1)
			r2 = r"(" + re.escape(sub) + r"){4,}"
			s2 = re.sub(r2, lambda s: s.group(1) * 3, s)
			if s2 == s:
				break
			s = s2
		return s.strip()

if "math" in CAPS:
	import x_math
	x_image.print = print

if "ytdl" in CAPS:
	IMAGE_FORMS = {
		".gif": True,
		".png": True,
		".bmp": False,
		".jpg": True,
		".jpeg": True,
		".tiff": False,
		".webp": True,
	}
	def is_image(url):
		if url:
			url = url.split("?", 1)[0]
			if "." in url:
				url = url[url.rindex("."):]
				url = url.casefold()
				return IMAGE_FORMS.get(url)
	is_youtube_url = lambda url: url and re.findall("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+", url)
	def get_best_icon(entry):
		with suppress(KeyError):
			return entry["thumbnail"]
		with suppress(KeyError):
			return entry["icon"]
		try:
			thumbnails = entry["thumbnails"]
		except KeyError:
			try:
				url = entry["webpage_url"]
			except KeyError:
				url = entry["url"]
			if not url:
				return ""
			if is_discord_url(url):
				if not is_image(url):
					return "https://cdn.discordapp.com/embed/avatars/0.png"
			if is_youtube_url(url):
				if "?v=" in url:
					vid = url.split("?v=", 1)[-1]
				else:
					vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
				entry["thumbnail"] = f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
				return entry["thumbnail"]
			return url
		return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]

	def get_best_audio(entry):
		try:
			return entry["stream"]
		except KeyError:
			pass
		best = (-inf,)
		try:
			fmts = entry["formats"]
		except KeyError:
			fmts = ()
		try:
			url = entry["url"]
		except KeyError:
			url = entry["webpage_url"]
		replace = True
		for fmt in fmts:
			q = (
				fmt.get("acodec") in ("opus", "vorbis"),
				fmt.get("vcodec") in (None, "none"),
				-abs(fmt["audio_channels"] - 2) if isinstance(fmt.get("audio_channels"), (int, float)) else -inf,
				fmt["abr"] if isinstance(fmt.get("abr"), (int, float)) else -inf,
				fmt["tbr"] if not isinstance(fmt.get("abr"), (int, float)) and isinstance(fmt.get("tbr"), (int, float)) else -inf,
				fmt["asr"] if isinstance(fmt.get("asr"), (int, float)) else -inf,
			)
			q = fmt.get("abr", 0)
			if not isinstance(q, (int, float)):
				q = 0
			if q <= 0:
				if fmt.get("asr"):
					q = fmt["asr"] / 1000
				elif fmt.get("audio_channels"):
					q = fmt["audio_channels"]
			q = (fmt.get("acodec") in ("opus", "vorbis"), fmt.get("vcodec") in (None, "none"), fmt.get("tbr", 0) or q)
			u = as_str(fmt["url"])
			if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
				replace = False
			if q > best or replace:
				best = q
				url = fmt["url"]
		if "dropbox.com" in url:
			if "?dl=0" in url:
				url = url.replace("?dl=0", "?dl=1")
		if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			import yt_dlp
			resp = requests.get(url).content
			fmts = []
			with suppress(ValueError, KeyError):
				while True:
					search = b'<Representation id="'
					resp = resp[resp.index(search) + len(search):]
					f_id = as_str(resp[:resp.index(b'"')])
					search = b"><BaseURL>"
					resp = resp[resp.index(search) + len(search):]
					stream = as_str(resp[:resp.index(b'</BaseURL>')])
					fmt = dict(yt_dlp.extractor.youtube.YoutubeIE._formats[f_id])
					fmt["url"] = stream
					fmts.append(fmt)
			entry["formats"] = fmts
			return get_best_audio(entry)
		if not url:
			raise KeyError("URL not found.")
		return url

	def shash(s):
		return base64.urlsafe_b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).rstrip(b"=").decode("ascii")

	def format_selector(ctx):
		formats = ctx.get('formats')[::-1]
		# vcodec='none' means there is no video
		try:
			best_audio = next(f for f in formats if (f.get('acodec', 'none') != 'none' and f.get('vcodec', 'none') == 'none' and f.get('ext', 'none') in ("webm", "opus")))
		except StopIteration:
			best_audio = formats[0]
		yield {
			'format_id': best_audio["format_id"],
			'ext': "webm",
			'requested_formats': [best_audio],
			'protocol': best_audio["protocol"],
		}

	def ytdl(q, download=False):
		import yt_dlp
		print("YTDL:", q)
		params = dict(default_search="auto", source_address="0.0.0.0", final_ext="ts", format=format_selector)
		ydl = yt_dlp.ydl = getattr(yt_dlp, "ydl", None) or yt_dlp.YoutubeDL(params)
		res = ydl.extract_info(q, download=False, process=True)
		if "entries" in res:
			entries = res["entries"]
		else:
			entries = [res]
		if download:
			entry = entries[0]
			url = entry["webpage_url"]
			url = re.sub(r"https?:\/\/(?:www\.)?youtube\.com\/watch\?v=", "https://youtu.be/", url)
			fn = "cache/~" + shash(url) + ".ts"
			if not os.path.exists(fn) or not os.path.getsize(fn):
				ydl.params["outtmpl"] = dict(default=fn)
				headers = header()
				stream = get_best_audio(entry)
				sys.stderr.write(stream + "\n")
				if os.path.exists("cache") and not COMPUTE_ORDER:
					dirlist = os.listdir("cache")
					if len(dirlist) >= 1024:
						random.shuffle(dirlist)
						while len(dirlist) >= 1024:
							try:
								os.remove("cache/" + dirlist.pop(0))
							except:
								traceback.print_exc()
				if not os.path.exists("cache"):
					os.mkdir("cache")
				args = ["ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240", "-threads", "3", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", stream, "-map_metadata", "-1", "-vn", "-c:a", "copy", fn]
				try:
					print(args)
					subprocess.check_output(args)
					if not os.path.exists(fn):
						raise RuntimeWarning(fn)
				except (subprocess.CalledProcessError, RuntimeWarning):
					traceback.print_exc()
					args = ["ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240", "-threads", "3", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", stream, "-map_metadata", "-1", "-vn", "-c:a", "libopus", fn]
					print(args)
					subprocess.check_output(args)
				if not os.path.exists(fn):
					try:
						ydl.download(url)
					except Exception:
						traceback.print_exc()
				if not os.path.exists(fn):
					part = fn + ".part"
					sys.stderr.write(f"Incomplete download {part} {os.path.exists(part)}\n")
					resp = requests.get(stream, headers=headers, verify=False, stream=True)
					length = int(resp.headers["Content-Length"])
					sys.stderr.write(f"{resp} {length}\n")
					resp.raise_for_status()
					b = resp.raw.read()
					sys.stderr.write(f"LENGTH, {len(b)}, {length}\n")
					while len(b) < length:
						sys.stderr.write(f"{len(b)}\n")
						headers["Range"] = f"bytes={len(b)}-"
						resp = requests.get(stream, headers=headers, verify=False, stream=True)
						resp.raise_for_status()
						b += resp.raw.read()
					if len(b) > length:
						b = memoryview(b)[:length]
					assert len(b)
					with open(fn, "wb") as f:
						f.write(b)
					return b
			else:
				print(f"File {fn} already exists, skipping...")
			assert os.path.exists(fn) and os.path.getsize(fn)
			with open(fn, "rb") as f:
				return f.read()
		output = [dict(
			name=entry["title"],
			url=entry["webpage_url"],
			duration=entry.get("duration"),
			stream=get_best_audio(entry),
			icon=get_best_icon(entry),
		) for entry in entries]
		return output

def max_size(w, h, maxsize, force=False):
	s = w * h
	m = maxsize * maxsize
	if s > m or force:
		r = (m / s) ** 0.5
		w = round(w * r)
		h = round(h * r)
	return w, h

if "browse" in CAPS:
	import playwright  # noqa: F401

	url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
	def is_url(url):
		return url_match.search(url)

	def browse(q, text=True):
		if not is_url(q):
			raise ValueError(q)
		with new_playwright_page("chromium", dict(width=960, height=540)) as page:
			page.goto(q, timeout=30000)
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
			page.goto(f"https://www.wolframalpha.com/input?i={urllib.parse.quote_plus(q)}", timeout=4000)
			time.sleep(8)
			for i in range(30):
				time.sleep(1)
				elems = page.locator("h2, img").all()
				texts = [e.text_content() or e.get_attribute("alt") for e in elems]
				text = "\n".join(filter(bool, texts)).strip()
				if text and len(texts) >= 2:
					return text
			raise TimeoutError(q)

if CAPS.intersection(("sd", "sdxl", "scc")):
	EXT1 = None
	def depth(im):
		global EXT1, EXT2, DPT1, DPT2
		if im.mode != "RGB":
			im = im.convert("RGB")
		im = resize_max(im, 1024)
		import torch
		torch.backends.cuda.matmul.allow_tf32 = True
		from transformers import DPTImageProcessor, DPTForDepthEstimation, GLPNImageProcessor, GLPNForDepthEstimation
		if not EXT1:
			EXT1 = backup_model(DPTImageProcessor.from_pretrained, "Intel/dpt-hybrid-midas", torch_dtype=torch.float16, device=0)
			DPT1 = backup_model(DPTForDepthEstimation.from_pretrained, "Intel/dpt-hybrid-midas", torch_dtype=torch.float16).to(0)
			EXT2 = backup_model(GLPNImageProcessor.from_pretrained, "whoismikha/room-3d-scene-estimation", torch_dtype=torch.float16, device=0)
			DPT2 = backup_model(GLPNForDepthEstimation.from_pretrained, "whoismikha/room-3d-scene-estimation", torch_dtype=torch.float16).to(0)
		def process(im, ext, dpt):
			inputs = ext(im, return_tensors="pt").to(dpt.dtype).to(dpt.device)
			with torch.no_grad():
				outputs = dpt(**inputs)
				depth = outputs.predicted_depth
			return torch.nn.functional.interpolate(
				depth.unsqueeze(1),
				size=im.size[::-1],
				mode="bicubic",
				align_corners=False,
			)
		fut = exc.submit(process, im, EXT1, DPT1)
		pred2 = process(im, EXT2, DPT2)
		pred1 = fut.result()
		out1 = pred1.squeeze().cpu().numpy()
		out2 = pred2.squeeze().cpu().numpy()

		def normalise(a):
			m, M = np.min(a), np.max(a)
			a -= m
			a *= 1 / (M - m)
			return a

		m = np.min(out1)
		if m < 0:
			out1 -= m
		out1 = -np.sqrt(out1, out=out1)
		out1 = -np.min(out1) - out1
		out1 **= 2
		out1 = -out1
		formatted = normalise(out1)
		formatted2 = normalise(out2)
		formatted3 = normalise(formatted + formatted2) * 255
		return fromarray(formatted3.astype("uint8"))


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
	except:
		print(traceback.format_exc(), end="")

def gifsicle(out, info=None):
	if os.name == "nt":
		if not os.path.exists("misc/gifsicle.exe") or os.path.getsize("misc/gifsicle.exe") < 4096:
			with requests.get("https://mizabot.xyz/u/EN8_fQ2CAKs.exe") as resp:
				b = resp.content
			with open("misc/gifsicle.exe", "wb") as f:
				f.write(b)
	else:
		if not os.path.exists("misc/gifsicle") or os.path.getsize("misc/gifsicle") < 4096:
			with requests.get("https://cdn.discordapp.com/attachments/1093723058386256022/1152254899694870599/gifsicle-static") as resp:
				b = resp.content
			with open("misc/gifsicle", "wb") as f:
				f.write(b)
			subprocess.run(("chmod", "777", "misc/gifsicle"))
	if "." in out:
		out2 = out.rsplit(".", 1)[0] + "~2." + out.rsplit(".", 1)[-1]
	else:
		out2 = out + "~2"
	args = ["misc/gifsicle"]
	if info and info["count"] >= 256:
		args.append("-O3")
	args.extend(("--loopcount=forever", "--lossy=100", "-o", out2, out))
	print(args)
	try:
		subprocess.run(args, timeout=90)
	except subprocess.TimeoutExpired:
		pass
	else:
		if os.path.getsize(out2) < os.path.getsize(out):
			return out2
	return out

def ffmpeg_opts(new, frames, count, mode, first, fmt, fs, w, h, duration, opt, vf=""):
	anim = count > 1
	command = []
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
	elif fmt == "avif":
		lossless = not anim and not opt
		if not anim:
			command.extend(("-vframes", "1", "-r", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area"
			if mode == "RGBA":
				vf += "format=rgba,"
		bitrate = floor(min(fs / duration * 1000 * 7.5, 99999999)) # use 7.5 bits per byte
		pix = "rgb24" if lossless else "yuv420p"
		if mode == "RGBA":
			cv = ("-c:v:0", "libsvtav1", "-pix_fmt:v:0", "yuv420p") if not h & 1 and not w & 1 else ("-c:v:0", "libaom-av1", "-pix_fmt:v:0", pix, "-usage", "realtime", "-cpu-used", "3")
			b1 = floor(bitrate * 3 / 4)
			b2 = floor(bitrate / 4)
			command.extend(("-filter_complex", vf + "[scaled];" + "[scaled]split=2[v1][v2];[v2]alphaextract[v2]", "-map", "[v1]", "-map", "[v2]", "-f", "avif", *cv, "-b:v:0", str(b1), "-c:v:1", "libaom-av1", "-pix_fmt:1", "gray", "-b:v:1", str(b2), "-usage", "realtime", "-cpu-used", "3", "-y", "-g", "300"))
		else:
			if vf:
				command.extend(("-vf", vf))
			cv = ("-c:v", "libsvtav1", "-pix_fmt", "yuv420p") if not h & 1 and not w & 1 else ("-c:v", "libaom-av1", "-pix_fmt", pix, "-usage", "realtime", "-cpu-used", "3")
			command.extend(("-f", "avif", *cv, "-b:v", str(bitrate)))
		if anim:
			command.extend(("-loop", "0", "-q:v", "24"))
		elif lossless:
			command.extend(("-lossless", "1"))
		else:
			command.extend(("-q:v", "24"))
	elif fmt == "webp":
		lossless = not anim and not opt
		if not anim:
			command.extend(("-vframes", "1", "-r", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		pix = ("rgba" if lossless else "yuva420p") if mode == "RGBA" else ("rgb24" if lossless else "yuv420p")
		if mode == "RGBA":
			command.extend(("-c:v", "libwebp_anim" if anim else "libwebp", "-pix_fmt", pix, "-pred", "mixed"))
		else:
			command.extend(("-c:v", "libwebp_anim" if anim else "libwebp", "-pix_fmt", pix, "-pred", "mixed"))
		command.extend(("-f", "webp", "-compression_level", "6"))
		if anim:
			command.extend(("-loop", "0", "-q:v", "24"))
		elif lossless:
			command.extend(("-lossless", "1"))
		else:
			command.extend(("-q:v", "24"))
	elif fmt == "png":
		command.extend(("-vframes", "1"))
		if (w, h) != first.size:
			vf += f"scale={w}:{h}:flags=area,"
			if mode == "rgba":
				vf += "format=rgba"
			command.extend(("-vf", vf))
		if mode == "RGBA":
			command.extend(("-c:v", "png", "-pix_fmt", "rgba", "-pred", "mixed", "-compression_level", "6"))
		else:
			command.extend(("-c:v", "png", "-pix_fmt", "rgb24", "-pred", "mixed", "-compression_level", "6"))
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
		bitrate = floor(min(fs / duration * 1000 * 7.5, 99999999)) # use 7.5 bits per byte
		command.extend(("-b:v", str(bitrate)))
		cdc = CODEC_FFMPEG.get(fmt, "libsvtav1")
		fmt = CODECS.get(fmt, fmt)
		if mode == "RGBA":
			pix_fmt = "yuva444p" if cdc == "libsvtav1" else "yuva420p"
			command.extend(("-pix_fmt", pix_fmt, "-c:v", cdc))
		else:
			pix_fmt = "yuv444p" if cdc == "libsvtav1" else "yuv420p"
			command.extend(("-pix_fmt", pix_fmt, "-c:v", cdc))
		command.extend(("-f", fmt))
	return command, fmt

def save_into(im, size, fmt, fs, r=0, opt=False):
	assert size[0] and size[1], f"Expected non-zero size, got {size}"
	thresh = 0.984375
	heavy = r > thresh or np.prod(size) <= 1048576
	if "RGB" in im.mode and np.prod(size) > 65536 or fmt not in ("png", "jpg", "webp", "gif"):
		b = np.asanyarray(im, dtype=np.uint8).data
		pix = "rgb24" if im.mode == "RGB" else "rgba"
		args = ["ffmpeg", "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-video_size", "x".join(map(str, im.size)), "-i", "-"]
		opts, fmt = ffmpeg_opts({}, iter([im]), 1, im.mode, im, fmt, fs * (r or 1), *size, 1000, opt)
		args.extend(opts)
		print(im, len(b))
		if fmt in ("png", "jpg", "webp"):
			args.append("-")
			print(args)
			return subprocess.run(args, stdout=subprocess.PIPE, input=b).stdout
		else:
			ts = time.time_ns() // 1000
			out = "cache/" + str(ts) + "." + fmt
			args.append(out)
			print(args)
			subprocess.run(args, input=b)
			assert os.path.exists(out) and os.path.getsize(out), f"Expected output file {out}"
			with open(out, "rb") as f:
				return f.read()
	fmt = dict(jpg="jpeg", gif="png").get(fmt.lower(), fmt)
	if im.size != tuple(size):
		im = im.resize(size, resample=Resampling.LANCZOS if np.prod(size) <= 1048676 else Resampling.BICUBIC)
	out = io.BytesIO()
	if fmt == "webp":
		if heavy:
			im.save(out, format="webp", lossless=True, quality=80, method=6)
		else:
			im.save(out, format="webp", lossless=True, quality=80, method=3)
			r = fs / out.tell()
			if r > thresh:
				out = io.BytesIO()
				im.save(out, format="webp", lossless=True, quality=100, method=6)
	elif fmt == "png":
		if heavy:
			im.save(out, format="png", optimize=True, compress_level=9)
		else:
			im.save(out, format="png", optimize=True, compress_level=5)
			r = fs / out.tell()
			if r > thresh:
				out = io.BytesIO()
				im.save(out, format="png", optimize=True, compress_level=9)
	else:
		im.save(out, format=fmt, optimize=True)
	return out.getbuffer()

def anim_into(out, new, first, size, fmt, fs, r=0):
	assert size[0] and size[1], f"Expected non-zero size, got {size}"
	command = ["ffmpeg", "-nostdin", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel, "-i", out]
	mode = new["mode"]
	opts, fmt = ffmpeg_opts(new, new["frames"], new["count"], mode, first, fmt, fs, *size, new["duration"], True)
	command.extend(opts)
	if "." in out:
		out2 = out.rsplit(".", 1)[0] + "~2." + CODECS.get(fmt, fmt)
	else:
		out2 = out + "~2"
	command.append(out2)
	print(command)
	subprocess.run(command, timeout=240)
	assert os.path.exists(out2), f"Expected output file {out2}"
	if "A" not in mode and np.prod(size) * new["count"] <= 67108864 and os.path.getsize(out2) < fs * 3:
		out = gifsicle(out2, new)
	else:
		out = out2
	with open(out, "rb") as f:
		return f.read()


statics = ("png", "bmp", "jpg", "heic", "ico", "icns", "j2k", "tga", "tiff", "pdf")
# Main image operation function
def evalImg(url, operation, args):
	ts = time.time_ns() // 1000
	out = "cache/" + str(ts) + ".png"
	fmt = "auto"
	cdc = "webp"
	fs = inf
	dur = None
	maxframes = inf
	nogif = False
	if len(args) > 1 and args[-2] == "-f":
		fmt = args.pop(-1) or fmt
		args.pop(-1)
		if fmt in statics:
			maxframes = 1
			nogif = True
	if len(args) > 1 and args[-2] == "-fs":
		fs = floor(float(args.pop(-1)))
		args.pop(-1)
	if len(args) > 1 and args[-2] == "-d":
		dur = args.pop(-1) * 1000
		args.pop(-1)
	if args and args[-1] == "-o":
		opt = True
		args.pop(-1)
	elif args and args[-1] == "-oz":
		maxframes = 5
		opt = False
		args.pop(-1)
	else:
		opt = False
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
			new = dict(count=1, duration=1000, frames=[new])
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
				new = dict(count=prop[0], duration=prop[1] * 1000, frames=iter_img(new))
				print("Output anim:", new, prop)
			else:
				print("Output stat:", new, prop)
	if isinstance(new, dict) and "frames" in new:
		frames = optimise(new["frames"])
		if not frames:
			raise EOFError("No image output detected.")
		if nogif:
			new["frames"] = [next(iter(frames))]
			video = False
		elif new["count"] == 1 or not new.get("duration"):
			temp = next(iter(frames))
			video = getattr(temp, "audio", None)
			if not video:
				new["frames"] = [temp]
			else:
				duration = dur = 3600000
				new["count"] = 16
				new["frames"] = [temp] * new["count"]
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
			fps = 1000 * new["count"] / duration
			if issubclass(type(frames), collections.abc.Sequence):
				first = frames[0]
			else:
				it = iter(frames)
				try:
					first = next(it)
				except StopIteration:
					frames = []
				else:
					frames = resume(first, it)
			size = first.size
			if fmt == "auto":
				if getattr(first, "audio", None) or new["count"] * np.prod(size) > 1073741824:
					fmt = "mp4"
					cdc = "libsvtav1"
				else:
					fmt = "gif"
			out = "cache/" + str(ts) + "." + CODECS.get(fmt, fmt)
			mode = str(first.mode)
			if mode == "P":
				raise RuntimeError("Unexpected P mode image")
			archive = False
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
				if hwaccel == "cuda":
					if COMPUTE_CAPS:
						try:
							devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
						except IndexError:
							devid = random.randint(0, len(COMPUTE_CAPS))
							fmt = "gif"
					else:
						devid = random.randint(0, ceil(len(COMPUTE_CAPS) / 2))
					command.extend(("-hwaccel_device", str(devid)))
				if round(fps, 2) == round(fps):
					fps = round(fps)
				else:
					fps = round(fps, 4)
				command.extend([
					"-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
					"-video_size", "x".join(map(str, size)), "-i", "-",
				])
				opts, fmt = ffmpeg_opts(new, frames, count, mode, first, fmt, fs, *size, duration, opt)
				command.extend(opts)
				command.append("cache/" + str(ts) + "." + CODECS.get(fmt, fmt))
				print(command)
				env = dict(os.environ)
				env.pop("CUDA_VISIBLE_DEVICES", None)
				proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1048576, env=env)
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
						if archive:
							with Image.open(frame) as im:
								b = save_into(im, im.size, cdc, inf)
						else:
							b = frame.read()
					else:
						if archive:
							with Image.open(io.BytesIO(frame)) as im:
								b = save_into(im, im.size, cdc, inf)
						else:
							b = frame
					return b
				workers = [fut for fut in futs if not fut.done()]
				if len(workers) >= 12:
					for fut in workers:
						fut.result()
				futs.append(exc.submit(save_frame, i, frame))
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
			if not archive:
				if fmt == "gif" and "A" not in mode and first.width * first.height * new["count"] <= 67108864 and os.path.getsize(out) < fs * 3:
					out = gifsicle(out, new)
				new["mode"] = mode
				new["frames"] = frames
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
							out = anim_into(orig, new, first, (w, h), fmt, fs, r=r)
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
							out = anim_into(orig, new, first, (w, h), fmt, fs, r=r)
							r = fs / len(out)
							print("RB:", w, h, scale, len(out), r)
							seen[(w, h)] = out, r
						if r <= 1:
							upper_bound = scale
						else:
							lower_bound = scale
					print("F:", w, h, scale, len(out), r)
			if isinstance(out, str):
				assert os.path.exists(out), f"Expected output file {out}"
				with open(out, "rb") as f:
					return f.read()
			return out
		else:
			new = next(iter(new["frames"]))
	if Image and isinstance(new, Image.Image):
		if fmt == "auto":
			fmt = "png"
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
				out[i] = save_into(out[i], out[i].size, "png", inf)
	elif isinstance(out, io.BytesIO):
		out.seek(0)
		out = out.read()
	elif isinstance(out, memoryview):
		out = bytes(out)
	return out


if __name__ == "__main__":
	interface.start(background=False)