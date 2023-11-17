#!/usr/bin/python3

import os, sys, orjson

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

import io, time, concurrent.futures, asyncio, subprocess, psutil, collections, traceback, re, requests, contextlib, filetype, ast, base64, hashlib, random
import numpy as np
from math import *
sys.path.append("misc")

print = lambda *args, sep=" ", end="\n": sys.stdout.buffer.write(f"~print({repr(sep.join(map(str, args)))},end={repr(end)})\n".encode("utf-8"))

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
	# if is_main_thread():
	#     if not isinstance(fut, asyncio.Task):
	#         fut = create_task(fut, loop=loop)
	#     raise RuntimeError("This function must not be called from the main thread's asyncio loop.")
	try:
		ret = asyncio.run_coroutine_threadsafe(fut, loop=loop)
	except:
		ret = concurrent.futures.Future()
		loop.create_task(_await_fut(fut, ret))
	return ret

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if len(sys.argv) > 1 and sys.argv[1]:
	os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
	DEVICES = list(map(int, sys.argv[1].split(",")))
	DEV = DEVICES[0]
else:
	DEVICES = []
	DEV = -1
if len(sys.argv) > 2:
	CAPS = set(sys.argv[2].split(","))
else:
	CAPS = ()
if len(sys.argv) > 3:
	COMPUTE_LOAD = orjson.loads(sys.argv[3])
else:
	COMPUTE_LOAD = []
if len(sys.argv) > 4:
	COMPUTE_CAPS = orjson.loads(sys.argv[4])
else:
	COMPUTE_CAPS = []
if len(sys.argv) > 5:
	COMPUTE_ORDER = [i for i in orjson.loads(sys.argv[5]) if i in DEVICES]
else:
	COMPUTE_ORDER = []
if len(sys.argv) > 6:
	IT = int(sys.argv[6])
else:
	IT = 0

if CAPS.intersection(("image", "caption", "video", "sd", "sdxl", "sdxlr")):
	import zipfile, blend_modes
	try:
		# This module apparently does not exist on Linux
		import colorspace
	except:
		traceback.print_exc()
		colorspace = None
	import PIL
	from PIL import Image, ImageCms, ImageOps, ImageChops, ImageDraw, ImageFilter, ImageEnhance, ImageMath, ImageStat, ImageFile
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	Resampling = getattr(Image, "Resampling", Image)
	Transpose = getattr(Image, "Transpose", Image)
	Transform = getattr(Image, "Transform", Image)
	Image.MAX_IMAGE_PIXELS = 4294967296
	from PIL import GifImagePlugin
	GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY
else:
	Image = None

try:
	import torch
except:
	torch = None
	hwaccel = "d3d11va" if os.name == "nt" else "auto"
else:
	hwaccel = "cuda" if torch.cuda.is_available() else "d3d11va" if os.name == "nt" else "auto"

if not hasattr(time, "time_ns"):
	time.time_ns = lambda: int(time.time() * 1e9)

requests = requests.Session()

def is_strict_running(proc):
	if not proc:
		return
	try:
		if not proc.is_running():
			return False
		if proc.status() == "zombie":
			proc.wait()
			return
		return True
	except AttributeError:
		proc = psutil.Process(proc.pid)
	if not proc.is_running():
		return False
	if proc.status() == "zombie":
		proc.wait()
		return
	return True


def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

literal_eval = lambda s: ast.literal_eval(as_str(s).lstrip())

mpf = float
deque = collections.deque
suppress = contextlib.suppress

exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)

def load_mimes():
	with open("misc/mimes.txt") as f:
		mimedata = f.read().splitlines()
		globals()["mimesplitter"] = {}
		for line in mimedata:
			dat, ext, mime = line.split("\t")
			data = hex2bytes(dat)
			try:
				mimesplitter[len(data)][data] = (ext, mime)
			except KeyError:
				mimesplitter[len(data)] = {}
				mimesplitter[len(data)][data] = (ext, mime)

exc.submit(load_mimes)

def simple_mimes(b, mime=True):
	for k, v in reversed(mimesplitter.items()):
		out = v.get(b[:k])
		if out:
			return out[mime]
	try:
		s = b.decode("utf-8")
	except UnicodeDecodeError:
		return "application/octet-stream" if mime else "bin"
	return "text/plain" if mime else "txt"

def from_file(path, mime=True):
	path = filetype.get_bytes(path)
	if mime:
		out = filetype.guess_mime(path)
	else:
		out = filetype.guess_extension(path)
	if out and out.split("/", 1)[-1] == "zip" and type(path) is str and path.endswith(".jar"):
		return "application/java-archive"
	if not out:
		if type(path) is not bytes:
			if type(path) is str:
				raise TypeError(path)
			path = bytes(path)
		out = simple_mimes(path, mime)
	if out == "application/octet-stream" and path.startswith(b'ECDC'):
		return "audio/ecdc"
	return out

class magic:
	from_file = from_file
	from_buffer = from_file

# Converts a time interval represented using days:hours:minutes:seconds, to a value in seconds.
def time_parse(ts):
	data = ts.split(":")
	t = 0
	mult = 1
	while len(data):
		t += float(data[-1]) * mult
		data = data[:-1]
		if mult <= 60:
			mult *= 60
		elif mult <= 3600:
			mult *= 24
		elif len(data):
			raise TypeError("Too many time arguments.")
	return t

# URL string detector
url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
is_url = lambda url: url_match.search(url)
discord_match = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/")
is_discord_url = lambda url: discord_match.findall(url)

fcache = "cache" if os.path.exists("cache") else "../cache"

def header():
	return {
		"DNT": "1",
		"user-agent": f"Mozilla/5.{(time.time_ns() // 1000) % 10}",
	}

def get_request(url):
	if isinstance(url, (bytes, memoryview)):
		return url
	if isinstance(url, io.BytesIO):
		url.seek(0)
		return url.read()
	if is_discord_url(url) and "attachments/" in url[:64]:
		try:
			a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
		except ValueError:
			pass
		else:
			fn = f"{fcache}/attachment_{a_id}.bin"
			if os.path.exists(fn):
				with open(fn, "rb") as f:
					print(f"Attachment {a_id} loaded from cache.")
					return f.read()
	with requests.get(url, headers=header(), stream=True, verify=False, timeout=12) as resp:
		return resp.content


if "image" in CAPS:
	from_colour = lambda colour, size=128, key=None: Image.new("RGB" + "A" * (len(colour) > 3), (size, size), tuple(colour))

	def from_gradient(shape, count, colour):
		mode = "RGB" + "A" * (len(colour) > 3)
		s = 960
		if shape == "linear":
			data = np.linspace(0, count, num=s, dtype=np.float32)
		if shape == "radial":
			try:
				data = globals()["g-1"]
			except KeyError:
				data = np.linspace(-1, 1, num=s, dtype=np.float32)
				data **= 2
				data = np.array([data] * s)
				data += data.T
				data = np.sqrt(data, out=data)
				np.subtract(1, data, out=data)
				np.clip(data, 0, None, out=data)
				globals()["g-1"] = data
			if count != 1:
				data = data * count
		if shape == "conical":
			try:
				data = globals()["g-2"]
			except KeyError:
				m = (s - 1) / 2
				row = np.arange(s, dtype=np.float32)
				row -= m
				data = [None] * s
				for i in range(s):
					data[i] = a = np.arctan2(i - m, row)
					a *= 1 / tau
				data = np.float32(data).T
				globals()["g-2"] = data
			if count != 1:
				data = data * count
		if shape == "polygon":
			raise NotImplementedError
		if count > 1:
			data %= 1
		data = [(data * c).astype(np.uint8) for c in colour]
		spl = [fromarray(i) for i in data]
		im = Image.merge(mode, spl)
		if im.width != s or im.height != s:
			return im.resize((s,) * 2, resample=Resampling.NEAREST)
		return im

	def rgb_split(image, dtype=np.uint8, force=False):
		channels = None
		if not force and "RGB" not in str(image.mode):
			if str(image.mode) == "L":
				channels = [np.asanyarray(image, dtype=dtype)] * 3
			else:
				image = image.convert("RGB")
		if channels is None:
			a = np.asanyarray(image, dtype=dtype)
			channels = np.swapaxes(a, 2, 0)[:3]
		return channels

	def xyz_split(image, convert=True, dtype=np.uint8):
		if not colorspace:
			return rgb_split(image.convert("XYZ"), dtype=dtype, force=True)
		colorlib = colorspace.colorlib()
		out = rgb_split(image, dtype=np.float32)
		out *= 1 / 255
		for r, g, b in zip(*out):
			x, y, z = colorlib.RGB_to_XYZ(r, g, b)
			r[:] = x
			g[:] = y
			b[:] = z
		X, Y, Z = out
		X *= 255 / 96
		Y *= 255 / 100
		Z *= 255 / 109
		for c in out:
			np.round(c, out=c)
		if convert:
			out = list(fromarray(a, "L") for a in out)
		else:
			out = np.asanyarray(out, dtype=dtype)
		return out

	def hsv_split(image, convert=True, partial=False, dtype=np.uint8):
		channels = rgb_split(image, dtype=np.uint16)
		R, G, B = channels
		m = np.min(channels, 0)
		M = np.max(channels, 0)
		C = M - m #chroma
		Cmsk = C != 0

		# Hue
		H = np.zeros(R.shape, dtype=np.float32)
		for i, colour in enumerate(channels):
			mask = (M == colour) & Cmsk
			hm = np.asanyarray(channels[i - 2][mask], dtype=np.float32)
			hm -= channels[i - 1][mask]
			hm /= C[mask]
			if i:
				hm += i << 1
			H[mask] = hm
		H *= 256 / 6
		H = np.asanyarray(H, dtype=dtype)

		if partial:
			return H, M, m, C, Cmsk, channels

		# Saturation
		S = np.zeros(R.shape, dtype=dtype)
		Mmsk = M != 0
		S[Mmsk] = np.clip(256 * C[Mmsk] // M[Mmsk], None, 255)

		# Value
		V = np.asanyarray(M, dtype=dtype)

		out = [H, S, V]
		if convert:
			out = list(fromarray(a, "L") for a in out)
		return out

	def hsl_split(image, convert=True, dtype=np.uint8):
		H, M, m, C, Cmsk, channels = hsv_split(image, partial=True, dtype=dtype)

		# Luminance
		L = np.mean((M, m), 0, dtype=np.int16)

		# Saturation
		S = np.zeros(H.shape, dtype=dtype)
		Lmsk = Cmsk
		Lmsk &= (L != 1) & (L != 0)
		S[Lmsk] = np.clip((C[Lmsk] << 8) // (255 - np.abs((L[Lmsk] << 1) - 255)), None, 255)

		L = L.astype(dtype)

		out = [H, S, L]
		if convert:
			out = list(fromarray(a, "L") for a in out)
		return out

	def hsi_split(image, convert=True, dtype=np.uint8):
		H, M, m, C, Cmsk, channels = hsv_split(image, partial=True, dtype=dtype)

		# Intensity
		I = np.asanyarray(np.mean(channels, 0, dtype=np.float32), dtype=dtype)

		# Saturation
		S = np.zeros(H.shape, dtype=dtype)
		Imsk = I != 0
		S[Imsk] = 255 - np.clip((m[Imsk] << 8) // I[Imsk], None, 255)

		out = [H, S, I]
		if convert:
			out = list(fromarray(a, "L") for a in out)
		return out

	def hcl_split(image, convert=True, dtype=np.uint8):
		colorlib = colorspace.colorlib()
		out = rgb_split(image, dtype=np.float32)
		out *= 1 / 255
		for r, g, b in zip(*out):
			temp = colorlib.RGB_to_XYZ(r, g, b)
			temp = colorlib.XYZ_to_LUV(*temp)
			l, c, h = colorlib.LUV_to_polarLUV(*temp)
			r[:] = h
			g[:] = c
			b[:] = l
		H, C, L = out
		H *= 255 / 360
		C *= 255 / 180
		L *= 255 / 100
		for c in out:
			np.round(c, out=c)
		if convert:
			out = list(fromarray(a, "L") for a in out)
		else:
			out = np.asanyarray(out, dtype=dtype)
		return out

	def luv_split(image, convert=True, dtype=np.uint8):
		colorlib = colorspace.colorlib()
		out = rgb_split(image, dtype=np.float32)
		out *= 1 / 255
		for r, g, b in zip(*out):
			temp = colorlib.RGB_to_XYZ(r, g, b)
			l, u, v = colorlib.XYZ_to_LUV(*temp)
			r[:] = l
			g[:] = u
			b[:] = v
		L, U, V = out
		L *= 255 / 100
		U *= 255 / 306
		V *= 255 / 306
		U += 127.5
		V += 127.5
		for c in out:
			np.round(c, out=c)
		if convert:
			out = list(fromarray(a, "L") for a in out)
		else:
			out = np.asanyarray(out, dtype=dtype)
		return out

	mat_rgb2yiq = (
		(0.299, 0.587, 0.114),
		(0.596, -0.275, -0.321),
		(0.212, -0.523, 0.311),
	)
	def yiq_split(image, convert=True):
		out = np.swapaxes(rgb_split(image, dtype=np.float32), 0, 2)
		out *= 1 / 255
		try:
			out @= mat_rgb2yiq
		except TypeError:
			out = out @ mat_rgb2yiq
		out = np.swapaxes(out, 2, 0)
		if convert:
			out = list(fromarray(a, "L") for a in out)
		return out

	mat_rgb2yuv = (
		(0.299, 0.587, 0.114),
		(-0.14713, -0.28886, 0.436),
		(0.615, -0.51499, -0.10001),
	)
	def yuv_split(image, convert=True):
		out = np.swapaxes(rgb_split(image, dtype=np.float32), 0, 2)
		out *= 1 / 255
		try:
			out @= mat_rgb2yuv
		except TypeError:
			out = out @ mat_rgb2yuv
		out = np.swapaxes(out, 2, 0)
		if convert:
			out = list(fromarray(a, "L") for a in out)
		return out

	def rgb_merge(R, G, B, convert=True):
		out = np.empty(R.shape[::-1] + (3,), dtype=np.uint8)
		outT = np.swapaxes(out, 2, 0)
		if R.dtype in (np.float16, np.float32, np.float64):
			for a in (R, G, B):
				np.clip(a, 0, 255, out=a)
		outT[:] = (R, G, B)
		if convert:
			out = fromarray(out, "RGB")
		return out

	def xyz_merge(X, Y, Z, convert=True):
		if not colorspace:
			return fromarray(rgb_merge(X, Y, Z, convert=False), "XYZ").convert("RGB")
		colorlib = colorspace.colorlib()
		X = np.asanyarray(X, np.float32)
		Y = np.asanyarray(Y, np.float32)
		Z = np.asanyarray(Z, np.float32)
		X *= 96 / 255
		Y *= 100 / 255
		Z *= 109 / 255
		for x, y, z in zip(X, Y, Z):
			r, g, b = colorlib.XYZ_to_RGB(x, y, z)
			x[:] = r
			y[:] = g
			z[:] = b
		out = (X, Y, Z)
		for c in out:
			c *= 255
			np.round(c, out=c)
		return rgb_merge(*out, convert=convert)

	def hsv_merge(H, S, V, convert=True):
		return hsl_merge(H, S, V, convert, value=True)

	def hsl_merge(H, S, L, convert=True, value=False, intensity=False):
		S = np.asanyarray(S, dtype=np.float32)
		S *= 1 / 255
		np.clip(S, None, 1, out=S)
		L = np.asanyarray(L, dtype=np.float32)
		L *= 1 / 255
		np.clip(L, None, 1, out=L)
		H = np.asanyarray(H, dtype=np.uint8)

		Hp = H.astype(np.float32) * (6 / 256)
		Z = (1 - np.abs(Hp % 2 - 1))
		if intensity:
			C = (3 * L * S) / (Z + 1)
		elif value:
			C = L * S
		else:
			C = (1 - np.abs(2 * L - 1)) * S
		X = C * Z

		# initilize with zero
		R = np.zeros(H.shape, dtype=np.float32)
		G = np.zeros(H.shape, dtype=np.float32)
		B = np.zeros(H.shape, dtype=np.float32)

		# handle each case:
		mask = (Hp < 1)
		# mask = (Hp >= 0) == (Hp < 1)
		R[mask] = C[mask]
		G[mask] = X[mask]
		mask = (1 <= Hp) == (Hp < 2)
		# mask = (Hp >= 1) == (Hp < 2)
		R[mask] = X[mask]
		G[mask] = C[mask]
		mask = (2 <= Hp) == (Hp < 3)
		# mask = (Hp >= 2) == (Hp < 3)
		G[mask] = C[mask]
		B[mask] = X[mask]
		mask = (3 <= Hp) == (Hp < 4)
		# mask = (Hp >= 3) == (Hp < 4)
		G[mask] = X[mask]
		B[mask] = C[mask]
		mask = (4 <= Hp) == (Hp < 5)
		# mask = (Hp >= 4) == (Hp < 5)
		B[mask] = C[mask]
		R[mask] = X[mask]
		mask = (5 <= Hp)
		# mask = (Hp >= 5) == (Hp < 6)
		B[mask] = X[mask]
		R[mask] = C[mask]

		if intensity:
			m = L * (1 - S)
		elif value:
			m = L - C
		else:
			m = L - 0.5 * C
		R += m
		G += m
		B += m
		R *= 255
		G *= 255
		B *= 255
		return rgb_merge(R, G, B, convert)

	def hsi_merge(H, S, V, convert=True):
		return hsl_merge(H, S, V, convert, intensity=True)

	def hcl_merge(H, C, L, convert=True):
		colorlib = colorspace.colorlib()
		H = np.asanyarray(H, np.float32)
		C = np.asanyarray(C, np.float32)
		L = np.asanyarray(L, np.float32)
		H *= 360 / 255
		C *= 180 / 255
		L *= 100 / 255
		for h, c, l in zip(H, C, L):
			temp = colorlib.polarLUV_to_LUV(l, c, h)
			temp = colorlib.LUV_to_XYZ(*temp)
			r, g, b = colorlib.XYZ_to_RGB(*temp)
			h[:] = r
			c[:] = g
			l[:] = b
		out = (H, C, L)
		for c in out:
			c *= 255
			np.round(c, out=c)
		return rgb_merge(*out, convert=convert)

	def luv_merge(L, U, V, convert=True):
		colorlib = colorspace.colorlib()
		L = np.asanyarray(L, np.float32)
		U = np.asanyarray(U, np.float32)
		V = np.asanyarray(V, np.float32)
		U -= 127.5
		V -= 127.5
		L *= 100 / 255
		U *= 306 / 255
		V *= 306 / 255
		for l, u, v in zip(L, U, V):
			temp = colorlib.LUV_to_XYZ(l, u, v)
			r, g, b = colorlib.XYZ_to_RGB(*temp)
			l[:] = r
			u[:] = g
			v[:] = b
		out = (L, U, V)
		for c in out:
			c *= 255
			np.round(c, out=c)
		return rgb_merge(*out, convert=convert)

	mat_yiq2rgb = (
		(1, 0.956, 0.621),
		(1, -0.272, -0.647),
		(1, -1.107, 1.704),
	)
	def yiq_merge(yiq, convert=True):
		yiq = np.asanyarray(yiq, dtype=np.float32)
		out = np.swapaxes(yiq, 0, 2)
		try:
			out @= mat_yiq2rgb
		except TypeError:
			out = out @ mat_yiq2rgb
		out *= 255
		if convert:
			np.clip(out, 0, 255, out=out)
			out = out.astype(np.uint8)
			out = fromarray(out, "RGB")
		else:
			out = np.swapaxes(out, 2, 0)
		return out

	mat_yuv2rgb = (
		(1, 0, 1.13983),
		(1, -0.39465, -0.5806),
		(1, 2.03211, 0),
	)
	def yuv_merge(yuv, convert=True):
		yuv = np.asanyarray(yuv, dtype=np.float32)
		out = np.swapaxes(yuv, 0, 2)
		try:
			out @= mat_yuv2rgb
		except TypeError:
			out = out @ mat_yuv2rgb
		out *= 255
		if convert:
			np.clip(out, 0, 255, out=out)
			out = out.astype(np.uint8)
			out = fromarray(out, "RGB")
		else:
			out = np.swapaxes(out, 2, 0)
		return out

	srgb_p = ImageCms.createProfile("sRGB")
	lab_p  = ImageCms.createProfile("LAB")
	# hsv_p  = ImageCms.createProfile("HSV")
	rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
	lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
	# hsv2lab = ImageCms.buildTransformFromOpenProfiles(hsv_p, lab_p, "HSV", "LAB")
	# lab2hsv = ImageCms.buildTransformFromOpenProfiles(lab_p, hsv_p, "LAB", "HSV")


if CAPS.intersection(("image", "caption", "sd", "sdxl", "sdxlr")):
	def fromarray(arr, mode="L"):
		try:
			return Image.fromarray(arr, mode=mode)
		except TypeError:
			try:
				b = arr.tobytes()
			except TypeError:
				b = bytes(arr)
			s = tuple(reversed(arr.shape))
			try:
				return Image.frombuffer(mode, s, b, "raw", mode, 0, 1)
			except TypeError:
				return Image.frombytes(mode, s, b)

	def resume(im, *its):
		yield im
		for it in its:
			yield from it

	def optimise(im, keep_rgb=True):
		try:
			it = iter(im)
		except TypeError:
			pass
		else:
			if not im:
				return im
			try:
				i0 = next(it)
			except StopIteration:
				return []
			out = [optimise(i0, keep_rgb=keep_rgb)]
			orig = []
			mode = i0.mode
			if mode == out[0].mode:
				return resume(i0, it)
			for i2 in it:
				orig.append(i2)
				if i2.mode != mode:
					i2 = optimise(i2, keep_rgb=keep_rgb)
					if i2.mode != mode:
						return resume(i0, orig, it)
				out.append(i2)
			return out
		if im.mode == "L":
			if keep_rgb:
				return im.convert("RGB")
			return im
		if im.mode == "P":
			if keep_rgb:
				im = im.convert("RGBA")
			else:
				return im
		if im.mode == "RGBA":
			if keep_rgb:
				A = im.getchannel("A")
				if np.min(A) >= 254:
					return im.convert("RGB")
				return im
			R, G, B, A = im.split()
			r, g, b = np.asarray(R, dtype=np.uint8), np.asarray(G, dtype=np.uint8), np.asarray(B, dtype=np.uint8)
			distRG = np.abs(r.ravel() - g.ravel())
			if np.max(distRG) <= 2:
				distGB = np.abs(g.ravel() - b.ravel())
				if np.max(distGB) <= 2:
					distBR = np.abs(b.ravel() - r.ravel())
					if np.max(distBR) <= 2:
						if np.min(A) >= 254:
							return im.convert("L")
						return im.convert("LA")
			if np.min(A) >= 254:
				return im.convert("RGB")
			return im
		if keep_rgb:
			if im.mode != "RGB":
				return im.convert("RGB")
			return im
		R, G, B = im.split()
		r, g, b = np.asarray(R, dtype=np.uint8), np.asarray(G, dtype=np.uint8), np.asarray(B, dtype=np.uint8)
		distRG = np.abs(r.ravel() - g.ravel())
		if np.max(distRG) <= 2:
			distGB = np.abs(g.ravel() - b.ravel())
			if np.max(distGB) <= 2:
				distBR = np.abs(b.ravel() - r.ravel())
				if np.max(distBR) <= 2:
					return im.convert("L")
		return im

	# Autodetect max image size, keeping aspect ratio
	def max_size(w, h, maxsize, force=False):
		s = w * h
		m = maxsize * maxsize
		if s > m or force:
			r = (m / s) ** 0.5
			w = round(w * r)
			h = round(h * r)
		return w, h

	def resize_max(image, maxsize, force=False, resample=Resampling.LANCZOS, box=None, reducing_gap=None):
		w, h = max_size(image.width, image.height, maxsize, force=force)
		if w != image.width or h != image.height:
			if type(resample) is str:
				image = resize_to(image, w, h, resample)
			else:
				image = image.resize([w, h], resample, box, reducing_gap)
		if force > 1:
			return image.copy()
		return image

	resizers = dict(
		sinc=Resampling.LANCZOS,
		lanczos=Resampling.LANCZOS,
		cubic=Resampling.BICUBIC,
		bicubic=Resampling.BICUBIC,
		scale2x="scale2x",
		hamming=Resampling.HAMMING,
		linear=Resampling.BILINEAR,
		bilinear=Resampling.BILINEAR,
		nearest=Resampling.NEAREST,
		nearestneighbour=Resampling.NEAREST,
		crop="crop",
		padding="crop",
		sdxl=Resampling.LANCZOS,
	)

	def resize_mult(image, x, y, operation="auto"):
		if y == "-":
			y = x
		if x == y == 1:
			return image
		w = image.width * x
		h = image.height * y
		return resize_to(image, round(w), round(h), operation)

	def resize_to(image, w, h, operation="auto"):
		if w == "-":
			if h == "-":
				return image
			w = image.width / image.height * h
		elif h == "-":
			h = image.height / image.width * w
		if abs(w * h) > 1073741824:
			raise OverflowError("Resulting image size too large.")
		if w == image.width and h == image.height:
			return image
		op = operation.casefold().replace(" ", "").replace("_", "")
		if op in resizers:
			filt = resizers[op]
		elif op == "auto":
			# Choose resampling algorithm based on source/destination image sizes
			m = min(abs(w), abs(h))
			n = min(image.width, image.height)
			if n > m:
				filt = Resampling.LANCZOS
			elif m <= 512:
				filt = "scale2x"
			elif m <= 3072:
				filt = Resampling.LANCZOS
			elif m <= 4096:
				filt = Resampling.BICUBIC
			else:
				filt = Resampling.BILINEAR
		else:
			raise TypeError(f'Invalid image operation: "{op}"')
		if w < 0:
			w = -w
			image = ImageOps.mirror(image)
		if h < 0:
			h = -h
			image = ImageOps.flip(image)
		if filt != Resampling.NEAREST:
			if str(image.mode) == "P":
				image = image.convert("RGBA")
		if filt == "scale2x":
			if w > image.width or h > image.height:
				try:
					import pygame
				except ImportError:
					pygame = None
				if image.mode == "P":
					image = image.convert("RGBA")
				b = image.tobytes()
				surf = pygame.image.frombuffer(b, image.size, image.mode)
				factor = 0
				while w > surf.get_width() or h > surf.get_height():
					surf = pygame.transform.scale2x(surf)
					factor += 1
					if factor >= 2:
						break
				b = pygame.image.tostring(surf, image.mode)
				image = Image.frombuffer(image.mode, surf.get_size(), b)
			if image.size == (w, h):
				return image
			filt = Resampling.NEAREST if w > image.width and h > image.height else Resampling.HAMMING
		elif filt == "crop":
			if image.mode == "P":
				image = image.convert("RGBA")
			out = Image.new(image.mode, (w, h), (0,) * len(image.mode))
			pos = tuple(np.array((w, h)) - image.size >> 1)
			out.paste(image, pos)
			return out
		return image.resize([w, h], filt)

	def crop_to(image, x1, y1, x2, y2):
		if x1 == "-":
			x1 = 0
		elif isinstance(x1, str):
			x1 = int(x1)
		if y1 == "-":
			y1 = 0
		elif isinstance(y1, str):
			y1 = int(y1)
		if x2 == "-":
			x2 = image.width
		elif isinstance(x2, str):
			v = int(x2)
			if x2.startswith("+") or x2.startswith("-"):
				x2 = image.width + v
			else:
				x2 = v
		if y2 == "-":
			y2 = image.height
		elif isinstance(y2, str):
			v = int(y2)
			if y2.startswith("+") or y2.startswith("-"):
				y2 = image.height + v
			else:
				y2 = v
		if x1 == y1 == 0 and x2 == image.width and y2 == image.height:
			return image
		if x1 >= 0 and y1 >= 0 and x2 <= image.width and y2 <= image.height:
			return image.crop((x1, y1, x2, y2))
		newsize = (x2 - x1, y2 - y1)
		im = Image.new("RGBA", newsize)
		im.paste(image, (-x1, -y1))
		return im

	def rotate_to(image, angle, expand=True):
		angle %= 360
		if not angle % 90:
			if angle == 90:
				return image.transpose(Transpose.ROTATE_90)
			elif angle == 180:
				return image.transpose(Transpose.ROTATE_180)
			elif angle == 270:
				return image.transpose(Transpose.ROTATE_270)
			return image
		return image.rotate(angle, resample=Resampling.BICUBIC, expand=expand)


if "video" in CAPS:
	sizecheck = re.compile("[1-9][0-9]*x[0-9]+")
	fpscheck = re.compile("[0-9]+ fps")

	def video2img(url, maxsize, fps, out, size=None, dur=None, orig_fps=None, data=None):
		direct = any((size is None, dur is None, orig_fps is None))
		ts = time.time_ns() // 1000
		fn = "cache/" + str(ts)
		if direct:
			if data is None:
				data = get_request(url)
			with open(fn, "wb") as file:
				file.write(data if type(data) is bytes else data.read())
		try:
			if direct:
				try:
					command = ["ffprobe", "-hide_banner", "-v", "error", fn]
					resp = b""
					# Up to 3 attempts to get video duration
					for _ in range(3):
						try:
							proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
							res = proc.communicate(timeout=12)
							resp = b"".join(res)
							break
						except:
							try:
								proc.terminate()
								proc.wait(timeout=2)
							except:
								proc.kill()
					s = resp.decode("utf-8", "replace")
					if orig_fps is None:
						f = re.findall(fpscheck, s)[0][:-4]
						orig_fps = float(f)
					if size is None:
						sfind = re.finditer(sizecheck, s)
						sizestr = next(sfind).group()
						size = [int(i) for i in sizestr.split("x")]
				except (ValueError, IndexError):
					if orig_fps is None:
						orig_fps = 30
					if size is None:
						size = (960, 540)
			fn2 = fn + ".gif"
			f_in = fn if direct else url
			command = ["ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240", "-threads", "2", "-hide_banner", "-nostdin", "-v", "error", "-y", "-hwaccel", hwaccel]
			if hwaccel == "cuda":
				if out.endswith(".webm") and COMPUTE_CAPS:
					try:
						devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
					except IndexError:
						devid = random.randint(0, len(COMPUTE_CAPS))
						fmt = "gif"
				else:
					devid = random.randint(0, ceil(len(COMPUTE_CAPS) / 2))
				command.extend(("-hwaccel_device", str(devid)))
			command.extend(("-i", f_in, "-vf"))
			w, h = max_size(*size, maxsize)
			fps = fps or orig_fps or 30
			step = 1
			while fps / step >= 40:
				step += 1
			fps /= step
			if step > 1:
				vf = f'select="not(mod(n\\,{step}))",'
			else:
				vf = ""
			if w != size[0]:
				vf += "scale=" + str(round(w)) + ":-1:flags=lanczos,"
			vf += "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=sierra3:diff_mode=rectangle"
			command.extend([vf, "-loop", "0", "-framerate", str(fps)])
			if hwaccel == "cuda":
				if out.endswith(".mp4"):
					command.extend(("-c:v", "h264_nvenc"))
				elif out.endswith(".webm"):
					command.extend(("-c:v", "av1_nvenc"))
			command.append(out)
			print(command)
			env = dict(os.environ)
			env.pop("CUDA_VISIBLE_DEVICES", None)
			subprocess.check_output(command, env=env)
			if direct:
				os.remove(fn)
		except:
			if direct:
				try:
					os.remove(fn)
				except:
					pass
			raise

	def create_gif(in_type, args, delay):
		ts = time.time_ns() // 1000
		out = "cache/" + str(ts) + ".gif"
		maxsize = 960
		if in_type == "video":
			video2img(args[0], maxsize, round(1000 / delay) if delay else None, out, args[1], args[2], args[3])
			return "$" + out
		images = args
		# Detect if an image sequence or video is being inputted
		imgs = deque()
		for url in images:
			data = get_request(url)
			try:
				img = get_image(data)
			except (PIL.UnidentifiedImageError, OverflowError, TypeError):
				if len(data) < 268435456:
					video2img(data, maxsize, round(1000 / delay) if delay else None, out, data=data)
					# $ symbol indicates to return directly
					return "$" + out
				else:
					raise OverflowError("Max file size to load is 256MB.")
			else:
				length = 0
				for f in range(2147483648):
					try:
						img.seek(f)
					except EOFError:
						length = f
						break
				if length and not delay:
					delay = img.info.get("duration") or delay or 50
				step = 1
				if length:
					fps = 1000 / delay
					while fps / step >= 40:
						step += 1
					delay = 1000 / (fps / step)
				for f in range(0, 2147483648, step):
					try:
						img.seek(f)
					except EOFError:
						break
					if not imgs:
						size = max_size(img.width, img.height, maxsize)
					temp = resize_to(img, *size, operation="hamming")
					if type(temp) is ImageSequence:
						temp = temp._images[temp._position]
					if str(temp.mode) == "RGBA":
						if imgs and str(imgs[0]) != "RGBA":
							imgs[0] = imgs[0].convert("RGBA")
					if temp is img:
						temp = img.crop()
					imgs.append(temp)
		# size = list(imgs[0].size)
		# while size[0] * size[1] * len(imgs) > 25165824:
		#	 size[0] /= 2 ** 0.5
		#	 size[1] /= 2 ** 0.5
		# size = [round(size[0]), round(size[1])]
		# if imgs[0].size[0] != size[0]:
		#	 imgs = (resize_to(img, *size, operation="hamming") for img in imgs)
		if len(imgs) == 1:
			imgs *= 2
		count = len(imgs)
		delay = delay or 50
		return dict(duration=delay * count, count=count, frames=imgs)

if "image" in CAPS:
	x_image = "misc/x-image.py" if os.path.exists("misc/x-image.py") else "x-image.py"
	with open(x_image, "rb") as f:
		b = f.read()
	c = compile(b, "<x-image.py>", "exec", optimize=1)
	exec(c, globals())

if "math" in CAPS:
	import matplotlib.pyplot as plt
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
		plt.style.use("dark_background")
		plt.rcParams["figure.figsize"] = (24, 9)
		plt.rcParams["figure.dpi"] = 96
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
		return "$" + out

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
		ts = time.time_ns() // 1000
		out = f"misc/{name}.png"
		plt.savefig(out)
		plt.clf()
		return out

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

	def ecdc_encode(b, bitrate="24", name="", source=""):
		ts = time.time_ns() // 10000 * 10 + int(DEV)
		fn = "cache/" + str(ts)
		with open(fn, "wb") as f:
			f.write(b)
		fo = "cache/" + str(ts) + ".ecdc"
		if name:
			name = " " + base64.b64encode(name.strip().encode("utf-8")).rstrip(b"=").decode("utf-8")
		args1 = ["ffmpeg", "-v", "error", "-hide_banner", "-vn", "-nostdin", "-i", fn, "-f", "s16le", "-ac", "2", "-ar", "48k", "-"]
		args2 = [sys.executable, "misc/ecdc_stream.py", "-g", str(DEV), "-n", name, "-s", source, "-b", str(bitrate), "-e", fo]
		print(args1)
		print(args2)
		PipedProcess(args1, args2).wait()
		with open(fo, "rb") as f:
			return f.read()

	def ecdc_decode(b, fmt="opus"):
		ts = time.time_ns() // 10000 * 10 + int(DEV)
		fn = "cache/" + str(ts)
		with open(fn, "wb") as f:
			f.write(b)
		fo = "cache/" + str(ts) + "." + fmt
		if name:
			name = " " + base64.b64encode(name.strip().encode("utf-8")).rstrip(b"=").decode("utf-8")
		args1 = [sys.executable, "misc/ecdc_stream.py", "-g", str(DEV), "-d", fn]
		args2 = ["ffmpeg", "-v", "error", "-hide_banner", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "96k", fo]
		print(args1)
		print(args2)
		PipedProcess(args1, args2).wait()
		with open(fo, "rb") as f:
			return f.read()


discord_emoji = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/assets\\/[0-9A-Fa-f]+\\.svg")
is_discord_emoji = lambda url: discord_emoji.search(url)


CBOTS = {}
def cb_exists(cid):
	return cid in CBOTS

mcache = {}
def cached_model(cls, model, **kwargs):
	t = (cls, model, tuple(kwargs.items()))
	try:
		return mcache[t]
	except KeyError:
		mcache[t] = cls(model, **kwargs)
	print("CACHED_MODEL:", t)
	return mcache[t]

def backup_model(cls, model, force=False, **kwargs):
	kwargs.pop("resume_download", None)
	t = (cls, model, tuple(kwargs.keys()))
	try:
		return mcache[t]
	except KeyError:
		pass
	if force:
		try:
			return cls(model, resume_download=True, **kwargs)
		except Exception as ex:
			ex2 = ex
	else:
		try:
			return cls(model, local_files_only=True, **kwargs)
		except:
			fut = exc.submit(cached_model, cls, model, resume_download=True, **kwargs)
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

		def image_to_features(self, image: Image) -> torch.Tensor:
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
		# config.apply_low_vram_defaults()
		# globals()["VIT"] = CustomInterrogator(config, dtype=dtype)
		# VIT.load_caption_model()
		# config.device = "cpu"
		# globals()["VIT2"] = CustomInterrogator(config, dtype=torch.float32)
		# VIT2.load_clip_model()
		# caption = VIT.generate_caption(im)
		# with torch.no_grad():
		# 	torch.cuda.empty_cache()
		return pytesseract.image_to_string(im, config="--psm 1")
	dfut = exc.submit(download_model)

	def caption(im, best=False):
		if not best:
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

if "summ" in CAPS:
	from transformers import pipeline
	smp = pipeline("summarization", model="Qiliang/bart-large-cnn-samsum-ChatGPT_v3", device=0, torch_dtype=torch.float16)
	print(smp)

	def summarise(s1, min_length=128, max_length=192, rm=True, do_sample=True):
		with torch.autocast("cuda"):
			s2 = smp(s1, max_length=max_length, min_length=min_length, do_sample=do_sample, truncation=True)[0]["summary_text"]
		exc.submit(ensure_gc, 20)
		if rm:
			return re.sub(r"(?:in )?(?:the|this|some)? *(?:article|essay|page|study|text|report|topic)[s, ]*(?:also mentions|we discuss|we look at|is about|includes|is based on)? *", "", s2, flags=re.I)
		return s2

	device, dtype = determine_cuda(1073741824, priority=None)
	device = f"cuda:{device}" if device >= 0 else "cpu"
	from sentence_transformers import SentenceTransformer
	Embedder = SentenceTransformer("LLukas22/all-mpnet-base-v2-embedding-all", device=device)
	if torch and dtype == torch.float16 and torch.cuda.get_device_properties(device).major >= 7:
		try:
			Embedder = Embedder.half()
		except (RuntimeError, NotImplementedError):
			pass
	def embedding(s):
		a = Embedder.encode(s).astype(np.float16)
		exc.submit(ensure_gc, 20)
		return a.data

if "class" in CAPS:
	from transformers import AutoTokenizer
	from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, exllama_set_max_input_length
	m = "TheBloke/Mythalion-13B-GPTQ"
	T = backup_model(AutoTokenizer.from_pretrained, m)
	i = sorted(range(torch.cuda.device_count()), key=lambda i: ((d := torch.cuda.get_device_properties(i)).total_memory > 11, -d.total_memory))[-1]
	bpw = 4
	gs = 64
	quantize_config = BaseQuantizeConfig(
		bits=bpw,
		group_size=gs,
		damp_percent=0.1,
		desc_act=True,
		sym=True,
		true_sequential=True,
		model_name_or_path=None,
		model_file_base_name="model",
	)
	M = AutoGPTQForCausalLM.from_quantized(
		m,
		revision=f"gptq-{bpw}bit-{gs}g-actorder_True",
		max_memory={i: torch.cuda.get_device_properties(i).total_memory},
		use_safetensors=True,
		use_triton=False,
		inject_fused_attention=False,
		offload_folder="cache",
		resume_download=True,
	)
	try:
		M = exllama_set_max_input_length(M, 2048)
	except ValueError:
		traceback.print_exc()
	print(M)

	def moe_class(prompt, mocked={}):
		if not mocked:
			return
		print("Mock prompt:", prompt)
		tokens = T(prompt, return_tensors="pt").input_ids[:, -960:].to(M.device)
		pc = len(tokens)
		with torch.no_grad():
			res = M.generate(
				inputs=tokens,
				temperature=0.1,
				top_k=32,
				top_p=0.1,
				repetition_penalty=1.2,
				max_new_tokens=32,
				do_sample=True,
			)
			exc.submit(ensure_gc, 60)
		text = T.decode(res[0])
		text = text.removeprefix("<s>").strip().removeprefix(prompt).strip().split("</s>", 1)[0]
		print("MOCK:", text)
		try:
			num = int(re.search("[0-9]+", text).group())
			k = mocked.get(num)
		except:
			k = None
		return k

	def GPTQm(inputs):
		model = inputs["model"]
		prompt = inputs["prompt"]
		temperature = inputs.get("temperature", 0.8)
		max_tokens = inputs.get("max_tokens", 1024)
		top_p = inputs.get("top_p", 1)
		stop = inputs.get("stop")
		frequency_penalty = inputs.get("frequency_penalty", 0.5)
		presence_penalty = inputs.get("presence_penalty", 0.5)
		model, tokeniser = M, T
		prompt = prompt.strip()
		tokens = tokeniser(prompt, return_tensors="pt").input_ids.to(model.device)
		ex = RuntimeError("Maximum attempts exceeded.")
		for i in range(3):
			try:
				with torch.no_grad():
					res = model.generate(
						inputs=tokens,
						temperature=temperature * 2 / 3,
						top_k=round(top_p * 120),
						top_p=top_p,
						repetition_penalty=(frequency_penalty + presence_penalty) / 4 + 1,
						max_new_tokens=max_tokens,
						do_sample=True,
					)
					exc.submit(ensure_gc, 60)
			except RuntimeError as e:
				if "probability tensor" in str(ex).lower():
					print(repr(ex))
					ex = e
					continue
				raise
			break
		else:
			raise ex
		text = tokeniser.decode(res[0]).removeprefix("<s>").strip().removeprefix(prompt).strip().split("</s>", 1)[0]
		text = text.strip().replace(":\n", ": ")
		spl = text.split(": ")
		if len(spl) > 1:
			text = ""
			while spl:
				s = spl.pop(0)
				if "\n" in s:
					text += s.rsplit("\n", 1)[0]
					break
				text += s + ": "
			text = text.strip()
			if text.endswith(":"):
				text = text.rsplit("\n", 1)[0]
		return text

if "math" in CAPS:
	x_math = __import__("x-math")

if "ytdl" in CAPS:
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
		with suppress(KeyError):
			return entry["stream"]
		best = -inf
		try:
			fmts = entry["formats"]
		except KeyError:
			fmts = ()
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		replace = True
		for fmt in fmts:
			q = fmt.get("abr", 0)
			if not isinstance(q, (int, float)):
				q = 0
			if q <= 0:
				if fmt.get("asr"):
					q = fmt["asr"] / 1000
				elif fmt.get("audio_channels"):
					q = fmt["audio_channels"]
			vcodec = fmt.get("vcodec", "none")
			if vcodec not in (None, "none"):
				q -= 1
			if not fmt["url"].startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
				replace = False
			# print(fmt["url"], q)
			if q > best or replace:
				best = q
				url = fmt["url"]
		if "dropbox.com" in url:
			if "?dl=0" in url:
				url = url.replace("?dl=0", "?dl=1")
		if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			resp = requests.get(url).content
			fmts = deque()
			with suppress(ValueError, KeyError):
				while True:
					search = b'<Representation id="'
					resp = resp[resp.index(search) + len(search):]
					f_id = resp[:resp.index(b'"')].decode("utf-8")
					search = b"><BaseURL>"
					resp = resp[resp.index(search) + len(search):]
					stream = resp[:resp.index(b'</BaseURL>')].decode("utf-8")
					import yt_dlp
					fmt = dict(yt_dlp.extractor.youtube.YoutubeIE._formats[f_id])
					fmt["url"] = stream
					fmts.append(fmt)
			entry["formats"] = fmts
			return get_best_audio(entry)
		if not url:
			raise KeyError("URL not found.")
		return url

	shash = lambda s: base64.urlsafe_b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).rstrip(b"=").decode("ascii")

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
		params = dict(default_search="auto", source_address="0.0.0.0", final_ext="webm", format=format_selector)
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
			fn = "cache/~" + shash(url) + ".webm"
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
								os.remove(dirlist.pop(0))
							except:
								traceback.print_exc()
				if not os.path.exists("cache"):
					os.mkdir("cache")
				args = ["ffmpeg", "-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240", "-threads", "3", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", stream, "-map_metadata", "-1", "-vn", "-c:a", "copy", fn]
				try:
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
					except:
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

if "gptq" in CAPS or "bnb" in CAPS or "agpt" in CAPS or "browse" in CAPS:
	import convobot, torch
	convobot.COMPUTE_LOAD = COMPUTE_LOAD
	convobot.COMPUTE_CAPS = COMPUTE_CAPS
	convobot.COMPUTE_ORDER = COMPUTE_ORDER
	convobot.DEVICES = DEVICES
	BOT = convobot.Bot()

	def CBAU(inputs):
		user_id = inputs["user_id"]
		channel_id = inputs["channel_id"]
		prompt = inputs["prompt"]
		key = inputs["key"]
		ht = inputs["huggingface_token"]
		oai = inputs.get("oai")
		bals = inputs.get("bals")
		nsfw = inputs.get("nsfw")
		premium = inputs.get("premium")
		try:
			cb = CBOTS["AU"]
		except KeyError:
			cb = CBOTS["AU"] = convobot.Bot( 
				key=key,
				huggingface_token=ht,
				premium=premium,
			)
		cb.user_id = user_id
		cb.channel_id = channel_id
		cb.oai = oai
		cb.bals = bals
		cb.nsfw = nsfw
		cb.premium = premium
		return cb.au(prompt)

	def CBAA(inputs):
		user_id = inputs["user_id"]
		channel_id = inputs["channel_id"]
		system = inputs["system"]
		prompt = inputs["prompt"]
		key = inputs["key"]
		ht = inputs["huggingface_token"]
		oai = inputs.get("oai")
		bals = inputs.get("bals")
		nsfw = inputs.get("nsfw")
		premium = inputs.get("premium")
		try:
			cb = CBOTS["AU"]
		except KeyError:
			cb = CBOTS["AU"] = convobot.Bot( 
				key=key,
				huggingface_token=ht,
				premium=premium,
			)
		cb.user_id = user_id
		cb.channel_id = channel_id
		cb.oai = oai
		cb.bals = bals
		cb.nsfw = nsfw
		cb.premium = premium
		return cb.aa(system, prompt)

	if "gptq" in CAPS:
		convobot.GPTQ = True

		def load_models():
			mods = dict(
				load_gptq=(
					"euryale-70b",
					"xwin-70b",
					"wizard-70b",
					"kimiko-70b",
					"wizard-coder-34b",
					"xwin-mlewd-13b",
					"mythalion-13b",
					"emerhyst-20b",
					"wizard-vicuna-30b",
					"orca-70b",
					"nous-puffin-70b",
				),
			)
			bot = convobot.Bot()
			for k, v in mods.items():
				for m in v:
					exc.submit(getattr(bot, k), m, fail=True)
					time.sleep(1)
		if "load" in CAPS:
			time.sleep(20)
			load_models()
			raise SystemExit

		def GPTQ(inputs):
			print("GPTQI:", inputs)
			model = inputs["model"]
			prompt = inputs["prompt"]
			temperature = inputs.get("temperature", 0.8)
			max_tokens = inputs.get("max_tokens", 1024)
			top_p = inputs.get("top_p", 1)
			stop = inputs.get("stop")
			frequency_penalty = inputs.get("frequency_penalty", 0.5)
			presence_penalty = inputs.get("presence_penalty", 0.5)
			model, tokeniser = BOT.load_gptq(model)
			prompt = prompt.strip()
			tokens = tokeniser(prompt, return_tensors="pt").input_ids.to(model.device)
			ex = RuntimeError("Maximum attempts exceeded.")
			for i in range(3):
				try:
					with torch.no_grad():
						res = model.generate(
							inputs=tokens,
							temperature=temperature * 2 / 3,
							top_k=round(top_p * 120),
							top_p=top_p,
							repetition_penalty=(frequency_penalty + presence_penalty) / 4 + 1,
							max_new_tokens=max_tokens,
							do_sample=True,
						)
						exc.submit(ensure_gc, 60)
				except RuntimeError as e:
					if "probability tensor" in str(ex).lower():
						print(repr(ex))
						ex = e
						continue
					raise
				break
			else:
				raise ex
			text = tokeniser.decode(res[0]).removeprefix("<s>").strip().removeprefix(prompt).strip().split("</s>", 1)[0]
			text = text.strip().replace(":\n", ": ")
			spl = text.split(": ")
			if len(spl) > 1:
				text = ""
				while spl:
					s = spl.pop(0)
					if "\n" in s:
						text += s.rsplit("\n", 1)[0]
						break
					text += s + ": "
				text = text.strip()
				if text.endswith(":"):
					text = text.rsplit("\n", 1)[0]
			return text

	if "bnb" in CAPS:
		convobot.BNB = True

		def load_models():
			mods = dict(
				load_bnb=(
					"pygmalion-13b",
					"manticore-13b",
					"airochronos-33b",
				),
			)
			bot = convobot.Bot()
			for k, v in mods.items():
				for m in v:
					exc.submit(getattr(bot, k), m, fail=True)
					time.sleep(1)
		if "load" in CAPS:
			time.sleep(40)
			load_models()
			raise SystemExit

		def BNB(inputs):
			model = inputs["model"]
			prompt = inputs["prompt"]
			temperature = inputs.get("temperature", 0.8)
			max_tokens = inputs.get("max_tokens", 1024)
			top_p = inputs.get("top_p", 1)
			stop = inputs.get("stop")
			frequency_penalty = inputs.get("frequency_penalty", 0.5)
			presence_penalty = inputs.get("presence_penalty", 0.5)
			model, tokeniser = BOT.load_bnb(model)
			prompt = prompt.strip()
			tokens = tokeniser(prompt, return_tensors="pt").input_ids.to(model.device)
			ex = RuntimeError("Maximum attempts exceeded.")
			for i in range(3):
				try:
					with torch.no_grad():
						res = model.generate(
							inputs=tokens,
							temperature=temperature * 2 / 3,
							top_k=round(top_p * 120),
							top_p=top_p,
							repetition_penalty=(frequency_penalty + presence_penalty) / 4 + 1,
							max_new_tokens=max_tokens,
							do_sample=True,
						)
						exc.submit(ensure_gc, 60)
				except RuntimeError as e:
					if "probability tensor" in str(ex).lower():
						print(repr(ex))
						ex = e
						continue
					raise
				break
			else:
				raise ex
			text = tokeniser.decode(res[0]).removeprefix("<s>").strip().removeprefix(prompt).strip().split("</s>", 1)[0]
			text = text.strip().replace(":\n", ": ")
			spl = text.split(": ")
			if len(spl) > 1:
				text = ""
				while spl:
					s = spl.pop(0)
					if "\n" in s:
						text += s.rsplit("\n", 1)[0]
						break
					text += s + ": "
				text = text.strip()
				if text.endswith(":"):
					text = text.rsplit("\n", 1)[0]
				start = ns
			return text

	# exc.submit(convobot.Bot.answer_summarise, convobot.Bot, q="test")

	# try:
		# from chatgpt_wrapper import AsyncChatGPT
	# except ImportError:
		# convobot.AsyncChatGPT = None
	# except:
		# convobot.AsyncChatGPT = None
		# print(traceback.format_exc(), end="")
	# else:
		# convobot.AsyncChatGPT = AsyncChatGPT

if CAPS.intersection(("sd", "sdxl", "sdxlr")):
	import imagebot
	imagebot.COMPUTE_LOAD = COMPUTE_LOAD
	imagebot.COMPUTE_CAPS = COMPUTE_CAPS
	imagebot.COMPUTE_ORDER = COMPUTE_ORDER
	imagebot.DEVICES = DEVICES

	def IBASL(prompt, kwargs, nsfw=False, force=True, count=1, sdxl=False, aspect_ratio=0, negative_prompt=None, z=False):
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		return ib.art_stablediffusion_local(prompt, kwargs, nsfw=nsfw, fail_unless_gpu=not force, count=count, sdxl=sdxl, aspect_ratio=aspect_ratio, negative_prompt=negative_prompt, z=z)

	def IBASR(prompt, image, steps=64, negative_prompt=None):
		# print(prompt)
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		for i in range(3):
			il = ib.art_stablediffusion_refine(prompt, image, steps=steps, upscale=False, negative_prompt=negative_prompt)
			if il:
				break
		else:
			raise RuntimeError("Maximum attempts exceeded.")
		return il[0]

	def IBASLR(prompt, kwargs, nsfw=False, force=True, count=1, aspect_ratio=0, negative_prompt=""):
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		return ib.art_stablediffusion_local(prompt, kwargs, nsfw=nsfw, fail_unless_gpu=not force, count=count, sdxl=2, aspect_ratio=aspect_ratio, negative_prompt=negative_prompt)

	EXT1 = None
	def depth(im):
		global EXT1, EXT2, DPT1, DPT2
		if im.mode != "RGB":
			im = im.convert("RGB")
		im = resize_max(im, 1024)
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

	# def whisper(url, best=False):
	# 	ts = time.time()
	# 	fn = f"{ts}.mp3"
	# 	if best:
	# 		m = "large-v2"
	# 	else:
	# 		m = "medium"
	# 	args = [sys.executable, "-m", "whisper", "--model", m, "--device", str(fix), "--fp16", "True", fn]
	# 	proc = psutil.Popen(args, stdout=subprocess.PIPE)
	# 	lines = [line for line in proc.stdout.read().decode("utf-8").splitlines() if line.startswith("[")]
	# 	lineobjs = []


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

def from_bytes(b, save=None, nogif=False):
	if b[:4] == b"<svg" or b[:5] == b"<?xml":
		import wand, wand.image
		with wand.image.Image() as im:
			with wand.color.Color("transparent") as background_color:
				wand.api.library.MagickSetBackgroundColor(
					im.wand,
					background_color.resource,
				)
			im.read(blob=b, resolution=1024)
			ib = io.BytesIO(im.make_blob("png32"))
		return Image.open(ib)
	elif b[:4] == b"%PDF":
		import pdf2image
		if os.name == "nt":
			pages = pdf2image.convert_from_bytes(b, poppler_path="misc/poppler", use_pdftocairo=True)
		else:
			pages = pdf2image.convert_from_bytes(b, use_pdftocairo=True)
		return ImageSequence.open(*pages, copy=True)
	else:
		data = b
		out = io.BytesIO(b) if type(b) is bytes else b
	try:
		import pillow_heif
	except ImportError:
		pass
	else:
		pillow_heif.register_heif_opener()
	mime = magic.from_buffer(data)
	if mime == "application/zip":
		z = zipfile.ZipFile(io.BytesIO(data), compression=zipfile.ZIP_DEFLATED, strict_timestamps=False)
		return ImageSequence.open(*(Image.open(z.open(f.filename)) for f in z.filelist if not f.is_dir()))
	try:
		import wand, wand.image
	except ImportError:
		wand = None
	proc = None
	try:
		if not wand or mime.split("/", 1)[0] == "image" and mime.split("/", 1)[-1] in "blp bmp cur dcx dds dib emf eps fits flc fli fpx ftex gbr gd heif heic icns ico im imt iptc jpeg jpg mcidas mic mpo msp naa pcd pcx pixar png ppm psd sgi sun spider tga tiff wal wmf xbm".split():
			try:
				return Image.open(out)
			except PIL.UnidentifiedImageError:
				if not b:
					raise FileNotFoundError("image file not found")
				out.seek(0)   
		if mime.split("/", 1)[0] in ("image", "video"):
			fmt = "rgba" if mime.split("/", 1)[0] == "image" else "rgb24"
			ts = time.time_ns() // 1000
			fn = "cache/" + str(ts)
			with open(fn, "wb") as f:
				f.write(data)
			cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration", "-of", "csv=s=x:p=0", fn)
			print(cmd)
			p = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			cmd2 = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", fn, "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0"]
			if nogif:
				cmd2.extend(("-vframes", "1"))
			cmd2.append("-")
			print(cmd2)
			proc = psutil.Popen(cmd2, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
			mode = "RGBA" if fmt == "rgba" else "RGB"
			try:
				res = as_str(p.stdout.read()).strip()
				if not res:
					raise TypeError(f'Filetype "{mime}" is not supported.')
				info = res.split("x", 3)
			except:
				print(as_str(p.stderr.read()), end="")
				raise
			print(info)
			size = tuple(map(int, info[:2]))
			if info[3] == "N/A":
				cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "csv=p=0", fn)
				info[3] = subprocess.check_output(cmd)
			try:
				dur = float(info[3])
			except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
				dur = 0
			if not size[0] or not size[1] or not dur:
				raise InterruptedError(info)
			try:
				fps = eval(info[2], {}, {})
			except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
				fps = 30
			framedur = 1000 / fps
			bcount = 4 if fmt == "rgba" else 3
			bcount *= int(np.prod(size))
			bytecount = bcount * dur * fps
			if not nogif and bytecount > 32 * 1073741824:
				proc.terminate()
				scale = sqrt((32 * 1073741824) / bytecount)
				fps = max(8, fps * scale)
				w, h = round(size[0] * scale), round(size[1] * scale)
				size = (w, h)
				framedur = 1000 / fps
				bcount = 4 if fmt == "rgba" else 3
				bcount *= int(np.prod(size))
				cmd3 = ["ffmpeg", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-y", "-i", fn, "-vf", f"fps=fps={fps},scale={w}:{h}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0", "-"]
				print(cmd3)
				proc = psutil.Popen(cmd3, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
			images = deque()
			while True:
				b = proc.stdout.read(bcount)
				while len(b) < bcount:
					if not b or not is_strict_running(proc):
						break
					b += proc.stdout.read(bcount - len(b))
				if len(b) < bcount:
					break
				img = Image.frombuffer(mode, size, b)
				img.info["duration"] = framedur
				images.append(img)
			if images:
				proc.wait(timeout=2)
				return ImageSequence.open(*images)
			raise RuntimeError(as_str(proc.stderr.read()))
	# except:
	# 	pass
	except Exception as ex:
		exc = ex
		if proc and proc.is_running():
			proc.terminate()
	else:
		exc = TypeError(f'Filetype "{mime}" is not supported.')
	if not wand or mime.split("/", 1)[-1] in ("gif", "webp", "ico", "x-icon"):
		ib = io.BytesIO(b)
		return Image.open(ib)
	with wand.image.Image() as img:
		with wand.color.Color("transparent") as background_color:
			wand.api.library.MagickSetBackgroundColor(
				img.wand,
				background_color.resource,
			)
		try:
			img.read(blob=b, resolution=1024)
		except Exception as ex:
			exc.args = exc.args + (str(ex.__class__),) + ex.args
		else:
			exc = None
		if exc:
			raise exc
		frames = []
		for frame in wand.sequence.Sequence(img):
			if not frame:
				continue
			with wand.image.Image(frame) as fi:
				ib = io.BytesIO(fi.make_blob("png32"))
			im = Image.open(ib)
			frames.append(im)
		if not frames:
			with wand.image.Image(img) as fi:
				ib = io.BytesIO(fi.make_blob("png32"))
			im = Image.open(ib)
			frames.append(im)
	return ImageSequence.open(*frames)

def round_random(x):
	try:
		y = int(x)
	except (ValueError, TypeError):
		return x
	if y == x:
		return y
	x -= y
	if random.random() <= x:
		y += 1
	return y

def ImageOpIterator(image, step=1, operation=None, ts=0, args=()):
	# Attempt to perform operation on all individual frames of .gif images
	fl = 0
	for i in range(2147483648):
		f = max(fl, round_random(i * step))
		fl = f
		np.random.seed(ts & 4294967295)
		globals()["CURRENT_FRAME"] = i
		try:
			image.seek(f)
		except EOFError:
			break
		if str(image.mode) == "P":
			temp = image.convert("RGBA")
		elif str(image.mode) != "RGBA":
			temp = image.convert("RGBA")
		elif operation:
			temp = image
		else:
			temp = image.convert("RGBA")
		temp.load()
		if operation:
			func = getattr(temp, operation, None)
			if func is None:
				func = operation if callable(operation) else eval(operation)
				res = func(temp, *args)
			else:
				res = func(*args)
			yield res
		else:
			yield temp

def ImageIterator(image):
	for i in range(2147483648):
		try:
			image.seek(i)
		except EOFError:
			break
		yield image

if Image:
	class ImageSequence(Image.Image):

		@classmethod
		def open(cls, *images, copy=False, func=None, args=()):
			if len(images) == 1:
				simages = []
				im = images[0]
				im.load()
				try:
					for i in range(2147483648):
						im.seek(i)
						if i:
							im2 = im.copy()
						else:
							im2 = im
						simages.append(im2)
				except EOFError:
					pass
				images = simages
				if len(images) == 1:
					im = images[0]
					if func:
						return func(im, *args)
					if copy:
						return im.copy()
					return im
				im.seek(0)
				images[0] = im.copy()
				copy = False
			return cls(*images, copy=copy, func=func, args=args)

		def __init__(self, *images, copy=False, func=None, args=()):
			if func:
				self._images = [func(image, *args) for image in images]
			elif copy:
				self._images = [image.copy() for image in images]
			else:
				self._images = images
				[im.load() for im in images]
			for i1, i2 in zip(self._images, images):
				if "duration" in i2.info:
					i1.info["duration"] = max(i2.info.get("duration", 0), 1000 / 40)
			self._position = 0

		__len__ = lambda self: len(self._images)

		def seek(self, position):
			if position >= len(self._images):
				raise EOFError
			self._position = position

		def __getattr__(self, key):
			try:
				return self.__getattribute__(key)
			except AttributeError:
				return getattr(self._images[self._position], key)


def get_image(url, out=None, nodel=False, nogif=False):
	if isinstance(url, Image.Image):
		return url
	out = out or url
	if type(url) not in (bytes, bytearray, io.BytesIO):
		save = None
		if is_url(url):
			# if url in CACHE:
			# 	return CACHE[url]
			data = None
			if is_discord_emoji(url):
				save = f"cache/emoji_{url.rsplit('/', 1)[-1].split('.', 1)[0]}"
				if os.path.exists(save):
					with open(save, "rb") as f:
						data = f.read()
					print(f"Emoji {save} successfully loaded from cache.")
			if data is None:
				data = get_request(url)
			if len(data) > 8589934592:
				raise OverflowError("Max file size to load is 8GB.")
			image = from_bytes(data, save, nogif=nogif)
			# CACHE[url] = image
		else:
			if os.path.getsize(url) > 8589934592:
				raise OverflowError("Max file size to load is 8GB.")
			with open(url, "rb") as f:
				data = f.read()
			# if not nodel and out != url and out:
			# 	try:
			# 		os.remove(url)
			# 	except:
			# 		pass
			image = from_bytes(data, save, nogif=nogif)
	else:
		if len(url) > 8589934592:
			raise OverflowError("Max file size to load is 8GB.")
		image = from_bytes(url)
	return image


# Main image operation function
def evalImg(url, operation, args):
	globals()["CURRENT_FRAME"] = 0
	ts = time.time_ns() // 1000
	out = "cache/" + str(ts) + ".png"
	fmt = "default"
	dur = None
	if len(args) > 1 and args[-2] == "-f":
		fmt = args.pop(-1)
		args.pop(-1)
	if len(args) > 1 and args[-2] == "-d":
		dur = args.pop(-1) * 1000
		args.pop(-1)
	if args and args[-1] == "-o":
		opt = True
		args.pop(-1)
	else:
		opt = False
	if operation != "$":
		# print("IOPER:", operation, args)
		if args and args[0] == "-nogif":
			nogif = args.pop(0)
		else:
			nogif = False
		if args and args[-1] == "-raw":
			args.pop(-1)
			image = get_request(url)
		else:
			if args and args[-1] == "-nodel":
				nodel = args.pop(-1)
			else:
				nodel = False
			image = get_image(url, out, nodel=nodel, nogif=nogif)
		# print("AOPER:", image, args)
		# -gif is a special case where the output is always an animated format (gif, mp4, mkv etc)
		if args and args[-1] == "-gif":
			args.pop(-1)
			if fmt in ("default", "png", "jpg", "jpeg", "bmp", "webp"):
				fmt = "gif"
			if fmt == "gif" and np.prod(image.size) > 262144:
				size = max_size(*image.size, 512)
				if size != image.size:
					image = ImageSequence.open(image, func=resize_to, args=size)
			new = eval(operation)(image, *args)
			# print("GIF:", new)
		else:
			try:
				if nogif:
					raise EOFError
				image.seek(1)
			except EOFError:
				globals()["ANIM"] = False
				image.seek(0)
				if str(image.mode) == "P":
					temp = image.convert("RGBA")
				elif str(image.mode) != "RGBA":
					temp = image.convert("RGBA")
				else:
					temp = image
				func = getattr(temp, operation, None)
				if func is None:
					new = eval(operation)(temp, *args)
				else:
					new = func(*args)
				# print("SINGLE:", new)
			else:
				new = dict(frames=deque(), duration=0)
				globals()["ANIM"] = True
				for f in range(2147483648):
					try:
						image.seek(f)
					except EOFError:
						break
					new["duration"] += max(image.info.get("duration", 0), 1000 / 240)
				fps = 1000 * f / new["duration"]
				fpl = 15 if opt else 30
				step = 1
				while fps / step >= fpl:
					step += 0.25
				if f // step > 5000:
					step = f / 4999
				elif f // step > 1000 and fpl > 20:
					step = f / 999
				new["count"] = int(f // step)
				print("ImageOPIterator:", image, step, fps, fpl, f)
				new["frames"] = ImageOpIterator(image, step, operation=operation, ts=ts, args=args)
	else:
		new = eval(url)(*args)
		# print("OPER:", new)
	if Image and isinstance(new, Image.Image):
		if getattr(new, "audio", None):
			new = dict(count=1, duration=1, frames=[new])
	# print("NEW:", new)
	if isinstance(new, dict) and "frames" in new:
		frames = optimise(new["frames"])
		if not frames:
			raise EOFError("No image output detected.")
		if new["count"] == 1:
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
		if dur:
			dur *= new["count"] / (new["count"] + 1)
			if duration > dur:
				duration = dur
		if video:
			# print("VIDEO:", new)
			if fmt in ("default", "png", "jpg", "jpeg", "bmp"):
				fmt = "gif"
			print("DURATION:", duration, new["count"])
			# if new["count"] <= 1024:
			#	 it = iter(frames)
			#	 first = next(it)
			#	 first.save(out, save_all=True, append_images=it, include_color_table=True, disposal=2, interlace=True, optimize=True, transparency=0, duration=round(1000 * duration / new["count"]), loop=0)
			#	 return repr([out])
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
			if getattr(first, "audio", None) and fmt in ("default", "webp", "gif", "apng"):
				fmt = "mp4"
			elif fmt == "default":
				# fmt = "webp"
				fmt = "gif"
			out = "cache/" + str(ts) + "." + fmt
			mode = str(first.mode)
			if mode == "P":
				mode = "RGBA"
			size = first.size
			if fmt == "zip":
				resp = zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
			else:
				command = ["ffmpeg", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel]
				if hwaccel == "cuda":
					if mode == "RGBA" and COMPUTE_CAPS:
						try:
							devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
						except IndexError:
							devid = random.randint(0, len(COMPUTE_CAPS))
							fmt = "gif"
					else:
						devid = random.randint(0, ceil(len(COMPUTE_CAPS) / 2))
					command.extend(("-hwaccel_device", str(devid)))
				command.extend([
					"-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
					"-video_size", "x".join(map(str, size)), "-i", "-",
				])
				if fmt in ("gif", "apng"):
					command.extend(("-gifflags", "-offsetting"))
					vf = "split[s0][s1];[s0]palettegen="
					if mode == "RGBA":
						vf += "reserve_transparent=1:"
					else:
						vf += "reserve_transparent=0:"
					if opt:
						frames = list(frames)
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
								R, G, B = rgb_split(im)
								for r, g, b in zip(R.ravel(), G.ravel(), B.ravel()):
									l = max(r, g, b)
									if l < 1:
										continue
									t = tuple(min(255, round(log2(x / l * 255 + 1) * 2) * 16) for x in (r, g, b))
									if t not in cols:
										cols.add(t)
						mc = min(128, max(4, 2 ** ceil(log2(len(cols) + 2 >> 1))))
						vf += f"max_colors={mc}:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra3:diff_mode=rectangle"
					elif new["count"] > 4096:
						vf += "max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
					else:
						vf += "stats_mode=diff[p];[s1][p]paletteuse=dither=sierra3:diff_mode=rectangle"
					if "A" in mode:
						vf += ":alpha_threshold=128"
					if vf:
						command.extend(("-vf", vf))
					if fmt == "apng":
						command.extend(("-plays", "0"))
					else:
						command.extend(("-loop", "0"))
					command.extend(("-f", fmt))
				elif fmt == "webp":
					if mode == "RGBA":
						command.extend(("-c:v", "libwebp_anim", "-pix_fmt", "yuva420p"))
					else:
						command.extend(("-c:v", "libwebp", "-pix_fmt", "yuv420p"))
					command.extend(("-lossless", "0", "-q:v", "24", "-loop", "0"))
				else:
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
					if new.get("count", inf) <= 16:
						crf = 18
					else:
						crf = max(24, min(51, round(log(np.prod(size), 2) * 6 - 92)))
					command.extend(("-crf", str(crf), "-pix_fmt"))
					if mode == "RGBA":
						if hwaccel == "cuda" and devid != -1:
							command.extend(("yuv420p", "-c:v", "av1_nvenc"))
						else:
							command.extend(("yuv420p", "-c:v", "libsvtav1"))
						fmt = "webm"
					else:
						if hwaccel == "cuda":
							command.extend(("yuv420p", "-c:v", "h264_nvenc"))
						else:
							command.extend(("yuv420p", "-c:v", "h264"))
						fmt = "mp4"
					# command.append("-shortest")
					out = "cache/" + str(ts) + "." + fmt
				command.append(out)
				print(command)
				env = dict(os.environ)
				env.pop("CUDA_VISIBLE_DEVICES", None)
				proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1048576, env=env)
			for i, frame in enumerate(frames):
				if fmt == "zip":
					b = io.BytesIO()
				if isinstance(frame, Image.Image):
					if frame.size != size:
						frame = frame.resize(size)
					if frame.mode != mode:
						frame = frame.convert(mode)
					if fmt == "zip":
						if frame.entropy() > 8:
							frame.save(b, format="webp", lossless=False, quality=67)
						else:
							frame.save(b, format="webp", lossless=True, quality=80)
					else:
						b = frame.tobytes()
				elif type(frame) is io.BytesIO:
					if fmt == "zip":
						with Image.open(frame) as im:
							if im.entropy() > 8:
								im.save(b, format="webp", lossless=False, quality=67)
							else:
								im.save(b, format="webp", lossless=True, quality=80)
					else:
						b = frame.read()
				else:
					if fmt == "zip":
						with Image.open(io.BytesIO(frame)) as im:
							if im.entropy() > 8:
								im.save(b, format="webp", lossless=False, quality=67)
							else:
								im.save(b, format="webp", lossless=True, quality=80)
					else:
						b = frame
				if fmt == "zip":
					b.seek(0)
					n = len(str(new["count"]))
					s = f"%0{n}d" % i
					resp.writestr(f"{s}.webp", data=b.read())
				else:
					proc.stdin.write(b)
			if fmt == "zip":
				resp.close()
			else:
				proc.stdin.close()
				proc.wait()
			if fmt == "gif":
				if os.name == "nt":
					if not os.path.exists("misc/gifsicle.exe") or os.path.getsize("misc/gifsicle.exe") < 4096:
						import requests
						with requests.get("https://cdn.discordapp.com/attachments/1093723058386256022/1152254352963145748/gifsicle.exe") as resp:
							b = resp.content
						with open("misc/gifsicle.exe", "wb") as f:
							f.write(b)
				else:
					if not os.path.exists("misc/gifsicle") or os.path.getsize("misc/gifsicle") < 4096:
						import requests
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
				if i >= 255:
					args.append("-O3")
				args.extend(("--loopcount=forever", "--lossy=100", "-o", out2, out))
				print(args)
				subprocess.run(args)
				if os.path.getsize(out2) < os.path.getsize(out):
					out = out2
			# return [out]
			with open(out, "rb") as f:
				return f.read()
		else:
			new = next(iter(new["frames"]))
	if Image and isinstance(new, Image.Image):
		new = optimise(new, keep_rgb=False)
		if new.entropy() > 8 and fmt in ("default", "webp"):
			# out = "cache/" + str(ts) + ".webp"
			out = io.BytesIO()
			new.save(out, format="webp", lossless=False, quality=67)
			return out
		elif fmt in ("default", "webp"):
			# out = "cache/" + str(ts) + ".webp"
			out = io.BytesIO()
			new.save(out, format="webp", lossless=True, quality=80)
			return out
		else:
			out = io.BytesIO()
			new.save(out, format="png", optimize=True)
			return out
	elif type(new) is str and new.startswith("$"):
		return new[1:]
	return new


ILLEGAL_EXCS = (BrokenPipeError, OSError, RuntimeError, TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError)

def evaluate(ts, args):
	try:
		out = evalImg(*args)
		if out and isinstance(out, (tuple, list)) and isinstance(out[0], io.BytesIO):
			out[0].seek(0)
			out[0] = out[0].read()
		elif isinstance(out, io.BytesIO):
			out.seek(0)
			out = out.read()
		if isinstance(out, (bytes, memoryview)):
			b = base64.b64encode(out)
			sys.stdout.buffer.write(f"$PROC_RESP[{ts}].set_result(_x)~".encode("utf-8") + b + b"\n")
			# sys.stdout.buffer.write(b)
			# sys.stdout.buffer.write(b"\n")
			# sys.stdout.buffer.write(f"$PROC_RESP[{ts}].set_result(_x)~".encode("utf-8") + b + b"\n")
			# sys.stdout.buffer.write(b)
			# sys.stdout.buffer.write(b"\n")
		else:
			sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_result({repr(out)})\n".encode("utf-8"))
	except Exception as ex:
		if isinstance(ex, ILLEGAL_EXCS):
			ex = SystemError(ex.__class__.__name__, *ex.args)
		sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception({repr(ex)})\n".encode("utf-8"))
		sa = lim_str(args, 256)
		sys.stdout.buffer.write(f"~print({repr(sa)},{repr(traceback.format_exc())},sep='\\n',end='')\n".encode("utf-8"))
		traceback.print_exc()
	sys.stdout.flush()


# exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
loop = asyncio.new_event_loop()
if __name__ == "__main__":

	async def ensure_parent():
		parent = psutil.Process(os.getppid())
		while True:
			if not is_strict_running(parent):
				p = psutil.Process()
				for c in p.children(True):
					c.terminate()
					try:
						c.wait(timeout=2)
					except psutil.TimeoutExpired:
						c.kill()
				p.terminate()
				break
			await asyncio.sleep(12)

	async def process_cmd(argv):
		argv = argv.rstrip()
		if argv[0] == "~":
			ts, s = argv[1:].split("~", 1)
			try:
				d = base64.b64decode(s.encode("ascii"))
				if b"pickle." in d:
					import pickle
					globals()["pickle"] = pickle
				args = eval(d)
				if args[1] == "&":
					args[1] = "$"
					evaluate(ts, args)
				elif args[1] == "%":
					args.pop(1)
					x_math.evaluate(ts, args)
				else:
					exc.submit(evaluate, ts, args)
			except Exception as ex:
				if isinstance(ex, ILLEGAL_EXCS):
					ex = SystemError(ex.__class__.__name__, *ex.args)
				sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception({repr(ex)})\n".encode("utf-8"))
				sa = lim_str(s, 256)
				sys.stdout.buffer.write(f"~print({repr(sa)}, end='')\n".encode("utf-8"))
				sys.stdout.buffer.write(f"~print({repr(traceback.format_exc())}, end='')\n".encode("utf-8"))
				traceback.print_exc()
				sys.stdout.flush()
			while len(CACHE) > 32:
				try:
					CACHE.pop(next(iter(CACHE)))
				except RuntimeError:
					pass
		else:
			sys.stdout.buffer.write(f"~print({repr(argv)}, end='')\n".encode("utf-8"))
			sys.stdout.flush()

	async def update_loop():
		sys.stdout.buffer.write("~print('',end='')\n".encode("utf-8"))
		while True:
			argv = await wrap_future(exc.submit(sys.stdin.readline))
			if not argv:
				raise SystemExit
			await process_cmd(argv)

	asyncio.set_event_loop(loop)
	asyncio.main_new_loop = loop
	loop.create_task(ensure_parent())
	loop.run_until_complete(update_loop())
