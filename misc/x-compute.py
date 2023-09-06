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

if len(sys.argv) > 1 and sys.argv[1].isnumeric() and int(sys.argv[1]) >= 0:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
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
	COMPUTE_ORDER = orjson.loads(sys.argv[5])
else:
	COMPUTE_ORDER = []

if "image" in CAPS:
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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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

exc = concurrent.futures.ThreadPoolExecutor(max_workers=8)

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
	with requests.get(url, headers=header(), stream=True, timeout=12) as resp:
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


if CAPS.intersection(("image", "sd", "sdxl", "sdxlr")):
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
			distRG = (r.ravel() - g.ravel()) ** 2
			if np.max(distRG) <= 4:
				distGB = (g.ravel() - b.ravel()) ** 2
				if np.max(distGB) <= 4:
					distBR = (b.ravel() - r.ravel()) ** 2
					if np.max(distBR) <= 4:
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
		distRG = (r.ravel() - g.ravel()) ** 2
		if np.max(distRG) <= 4:
			distGB = (g.ravel() - b.ravel()) ** 2
			if np.max(distGB) <= 4:
				distBR = (b.ravel() - r.ravel()) ** 2
				if np.max(distBR) <= 4:
					return im.convert("L")
		return im


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
			command = ["./ffmpeg", "-threads", "2", "-hide_banner", "-nostdin", "-v", "error", "-y", "-hwaccel", hwaccel]
			if hwaccel == "cuda":
				if out.endswith(".webm") and COMPUTE_CAPS:
					devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
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
			vf += "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
			command.extend([vf, "-loop", "0", "-framerate", str(fps)])
			if hwaccel == "cuda":
				if out.endswith(".mp4"):
					command.extend(("-c:v", "h264_nvenc"))
				elif out.endswith(".webm"):
					command.extend(("-c:v", "av1_nvenc"))
			command.append(out)
			print(command)
			subprocess.check_output(command)
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
	with open("x-image.py", "rb") as f:
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

if "caption" in CAPS:
	def determine_cuda(mem=1, priority=None, multi=False, major=0):
		if not torch or not torch.cuda.is_available():
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
		tinfo = [torch.cuda.get_device_properties(i) for i in range(n)]
		COMPUTE_LOAD = globals().get("COMPUTE_LOAD") or [0] * dc
		high = max(COMPUTE_LOAD)
		if priority == "full":
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, p.major, p.minor, p.multi_processor_count, p.total_memory)
		elif priority:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, p.major >= major, COMPUTE_LOAD[i] < high * 0.9, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, i, p.multi_processor_count, p.total_memory)
		elif priority is False:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, p.major >= major, p.major >= 7, -p.major, -p.minor, COMPUTE_LOAD[i] < high * 0.75, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -gmems[i].free, p.multi_processor_count)
		else:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i] < high * 0.5, p.major >= major, p.major >= 7, -p.major, -p.minor, COMPUTE_LOAD[i] * (random.random() + 4.5) * 0.2, -p.multi_processor_count, -gmems[i].free)
		pcs = sorted(range(n), key=key, reverse=True)
		if multi:
			return [i for i in pcs if gmems[i].free >= mem], torch.float16
		return pcs[0], torch.float16

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
		return a.data

	import tiktoken
	from clip_interrogator import Config, Interrogator
	try:
		import pytesseract
	except ImportError:
		pytesseract = None

	class CustomInterrogator(Interrogator):
		def __init__(self, config, dtype=torch.float32):
			self.config = config
			self.device = config.device
			self.dtype = dtype
			self.caption_offloaded = True
			self.clip_offloaded = True

	VIT = VIT2 = True
	def download_model():
		if torch and torch.cuda.device_count():
			device, dtype = determine_cuda(priority=None)
			if torch.cuda.get_device_properties(device).total_memory < 9 * 1073741824:
				device, dtype = "cpu", torch.float32
		else:
			device, dtype = "cpu", torch.float32
		config = Config(
			clip_model_name="ViT-H-14/laion2b_s32b_b79k",
			clip_model_path="misc/Clip",
			cache_path="misc/Clip",
			device=device,
			caption_model_name="blip-base",
		)
		globals()["VIT"] = CustomInterrogator(config, dtype=dtype)
		VIT.load_caption_model()
		config.device = "cpu"
		globals()["VIT2"] = CustomInterrogator(config, dtype=torch.float32)
		VIT2.load_clip_model()
		im = Image.new("RGB", (4, 4), (0, 0, 255))
		caption = VIT.generate_caption(im)
		description = VIT2.interrogate_fast(im, caption=caption, max_flavors=12)
		print("VIT:", description)
		with torch.no_grad():
			torch.cuda.empty_cache()
		return pytesseract.image_to_string(im, config="--psm 1")
	dfut = exc.submit(download_model)
	def caption(im, best=False):
		im = resize_max(im, 1024, "auto")
		if im.mode != "RGB":
			image = im.convert("RGB")
		else:
			image = im
		if pytesseract:
			fut = exc.submit(pytesseract.image_to_string, image, config="--psm 1", timeout=8)
		else:
			fut = None
		if not best:
			dfut.result()
			cfut = exc.submit(VIT.generate_caption, image)
			desc = VIT2.interrogate_fast(image, caption=" ", max_flavors=24)
			p1 = cfut.result() + desc.lstrip()
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
		with torch.no_grad():
			torch.cuda.empty_cache()
		return (p1, p2)

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
			resp = reqs.get(url).content
			fmts = deque()
			with suppress(ValueError, KeyError):
				while True:
					search = b'<Representation id="'
					resp = resp[resp.index(search) + len(search):]
					f_id = resp[:resp.index(b'"')].decode("utf-8")
					search = b"><BaseURL>"
					resp = resp[resp.index(search) + len(search):]
					stream = resp[:resp.index(b'</BaseURL>')].decode("utf-8")
					fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
					fmt.url = stream
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
				try:
					# raise
					ydl.download(url)
				except:
					traceback.print_exc()
				if not os.path.exists(fn):
					part = fn + ".part"
					sys.stderr.write(f"Incomplete download {part} {os.path.exists(part)}\n")
					resp = requests.get(stream, headers=headers, stream=True)
					length = int(resp.headers["Content-Length"])
					sys.stderr.write(f"{resp} {length}\n")
					resp.raise_for_status()
					b = resp.raw.read()
					sys.stderr.write(f"LENGTH, {len(b)}, {length}\n")
					while len(b) < length:
						sys.stderr.write(f"{len(b)}\n")
						headers["Range"] = f"bytes={len(b)}-"
						resp = requests.get(stream, headers=headers, stream=True)
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

if "gptq" in CAPS:
	import convobot, torch
	convobot.COMPUTE_LOAD = COMPUTE_LOAD
	convobot.COMPUTE_CAPS = COMPUTE_CAPS
	convobot.COMPUTE_ORDER = COMPUTE_ORDER

	def CBAI(inputs):
		user_id = inputs["user_id"]
		channel_id = inputs["channel_id"]
		key = inputs["key"]
		ht = inputs["huggingface_token"]
		# vis = inputs.get("vis_session")
		name = inputs["name"]
		model = inputs["model"]
		personality = inputs["personality"]
		premium = inputs["premium"]
		summary = inputs["summary"]
		jb = inputs["jb"]
		history = inputs["history"]
		refs = inputs["refs"]
		im = inputs["im"]
		prompt = inputs["prompt"]
		bl = inputs.get("bl")
		oai = inputs.get("oai")
		bals = inputs.get("bals")
		nsfw = inputs.get("nsfw")
		vc = inputs.get("vc")
		try:
			cb = CBOTS[channel_id]
			if cb.personality != personality:
				summary = None
				raise KeyError
		except KeyError:
			cb = CBOTS[channel_id] = convobot.Bot(
				key=key,
				huggingface_token=ht,
				summary=summary,
				name=name,
				personality=personality,
				premium=premium,
			)
		else:
			cb.premium = premium
		if model == "auto":
			if not getattr(cb, "model", None):
				if premium < 2:
					cb.model = "wizard"
				elif premium < 4:
					cb.model = "gpt3+"
				else:
					cb.model = "gpt4+"
		else:
			if model.startswith("gpt3") and premium < 2:
				cb.model = "wizard"
			elif model.startswith("gpt4") and premium < 4:
				cb.model = "gpt3+"
			cb.model = model or "gpt3+"
		cb.user_id = user_id
		cb.channel_id = channel_id
		cb.bl = bl
		cb.oai = oai
		cb.bals = bals
		cb.nsfw = nsfw
		# cb.vis_s = vis
		cb.vc = vc
		if inputs.get("reset"):
			outs = []
			futs = []
			for i, t in enumerate(history):
				to = []
				fut = exc.submit(
					cb.append,
					t,
					nin=len(history) - i - 1,
					to=to,
					ai=i >= len(history) / 2 and torch.cuda.is_available(),
				)
				outs.append(to)
				futs.append(fut)
				time.sleep(0.01)
			history = []
			for fut, out in zip(futs, outs):
				fut.result()
				history.extend(out)
			cb.chat_history = history
		cb.jailbroken = jb
		if im:
			try:
				im = cb.image
			except AttributeError:
				im = get_image(im)
		res = cb.ai(*prompt, refs=refs, im=im)
		if cb.model in ("gpt3", "gpt4", "gpt3+", "gpt4+"):
			cb.model = None
		with torch.no_grad():
			torch.cuda.empty_cache()
		return res

if "agpt" in CAPS:
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

	try:
		from chatgpt_wrapper import AsyncChatGPT
	except ImportError:
		convobot.AsyncChatGPT = None
	except:
		convobot.AsyncChatGPT = None
		print(traceback.format_exc(), end="")
	else:
		convobot.AsyncChatGPT = AsyncChatGPT

if CAPS.intersection(("sd", "sdxl", "sdxlr")):
	import imagebot
	imagebot.COMPUTE_LOAD = COMPUTE_LOAD
	imagebot.COMPUTE_CAPS = COMPUTE_CAPS
	imagebot.COMPUTE_ORDER = COMPUTE_ORDER

	def IBASL(prompt, kwargs, nsfw=False, force=True, count=1, sdxl=False):
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		return ib.art_stablediffusion_local(prompt, kwargs, nsfw=nsfw, fail_unless_gpu=not force, count=count, sdxl=sdxl)

	def IBASR(prompt, image, steps=64):
		# print(prompt)
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		for i in range(3):
			il = ib.art_stablediffusion_refine(prompt, image, steps=steps, upscale=False)
			if il:
				break
		else:
			raise RuntimeError("Maximum attempts exceeded.")
		im = optimise(il[0])
		b = io.BytesIO()
		im.save(b, format="png")
		b.seek(0)
		return b.read()

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
		return ImageSequence(*pages, copy=True)
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
		return ImageSequence(*(Image.open(z.open(f.filename)) for f in z.filelist if not f.is_dir()))
	try:
		import wand, wand.image
	except ImportError:
		wand = None
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
			cmd = ("./ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=s=x:p=0", fn)
			print(cmd)
			p = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			cmd2 = ["./ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", fn, "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0"]
			if nogif:
				cmd2.extend(("-vframes", "1"))
			cmd2.append("-")
			print(cmd2)
			proc = psutil.Popen(cmd2, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
			bcount = 4 if fmt == "rgba" else 3
			mode = "RGBA" if fmt == "rgba" else "RGB"
			try:
				res = as_str(p.stdout.read()).strip()
				if not res:
					raise TypeError(f'Filetype "{mime}" is not supported.')
				info = res.split("x", 2)
			except:
				print(as_str(p.stderr.read()), end="")
				raise
			print(info)
			size = tuple(map(int, info[:2]))
			try:
				duration = 1000 / eval(info[-1], {}, {})
			except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
				duration = 33.333333333333333
			bcount *= int(np.prod(size))
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
				img.info["duration"] = duration
				images.append(img)
			if images:
				proc.wait(timeout=2)
				return ImageSequence(*images)
			print(proc.stderr.read())
	# except:
	# 	pass
	except Exception as ex:
		exc = ex
	else:
		exc = TypeError(f'Filetype "{mime}" is not supported.')
	with wand.image.Image() as im:
		with wand.color.Color("transparent") as background_color:
			wand.api.library.MagickSetBackgroundColor(
				im.wand,
				background_color.resource,
			)
		try:
			im.read(blob=b, resolution=1024)
		except Exception as ex:
			exc.args = exc.args + (str(ex.__class__),) + ex.args
		else:
			exc = None
		if exc:
			raise exc
		ib = io.BytesIO(im.make_blob("png32"))
	return Image.open(ib)

def ImageOpIterator(image, step, operation, ts, args):
	# Attempt to perform operation on all individual frames of .gif images
	for i, f in enumerate(range(0, 2147483648, step)):
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
		else:
			temp = image
		func = getattr(temp, operation, None)
		temp.load()
		if func is None:
			func = operation if callable(operation) else eval(operation)
			res = func(temp, *args)
		else:
			res = func(*args)
		yield res

def ImageIterator(image):
	for i in range(2147483648):
		try:
			image.seek(i)
		except EOFError:
			break
		yield image

class ImageSequence(Image.Image):

	def __init__(self, *images, copy=False, func=None, args=()):
		if len(images) == 1:
			images = ImageIterator(images[0])
			if not func and not copy:
				copy = True
		if func:
			self._images = [func(image, *args) for image in images]
		elif copy:
			self._images = [image.copy() for image in images]
		else:
			self._images = images
		for i1, i2 in zip(self._images, images):
			if "duration" in i2.info:
				i1.info["duration"] = max(i2.info.get("duration", 0), 50)
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
	if operation != "$":
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
		# -gif is a special case where the output is always an animated format (gif, mp4, mkv etc)
		if args and args[-1] == "-gif":
			args.pop(-1)
			if fmt in ("default", "png", "jpg", "jpeg", "bmp", "webp"):
				fmt = "gif"
			if fmt == "gif" and np.prod(image.size) > 262144:
				size = max_size(*image.size, 512)
				if size != image.size:
					image = ImageSequence(image, func=resize_to, args=size)
			new = eval(operation)(image, *args)
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
			else:
				new = dict(frames=deque(), duration=0)
				globals()["ANIM"] = True
				for f in range(2147483648):
					try:
						image.seek(f)
					except EOFError:
						break
					new["duration"] += max(image.info.get("duration", 0), 50)
				fps = 1000 * f / new["duration"]
				step = 1
				while fps / step >= 40:
					step += 1
				new["count"] = f // step
				new["frames"] = ImageOpIterator(image, step, operation=operation, ts=ts, args=args)
	else:
		new = eval(url)(*args)
	if isinstance(new, Image.Image):
		if getattr(new, "audio", None):
			new = dict(count=1, duration=1, frames=[new])
	if type(new) is dict and "frames" in new:
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
			print(duration, new["count"])
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
				fmt = "webp"
			out = "cache/" + str(ts) + "." + fmt
			mode = str(first.mode)
			if mode == "P":
				mode = "RGBA"
			size = first.size
			if fmt == "zip":
				resp = zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
			else:
				command = ["./ffmpeg", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel]
				if hwaccel == "cuda":
					if mode == "RGBA" and COMPUTE_CAPS:
						devid = random.choice([i for i, c in enumerate(COMPUTE_CAPS) if c >= [8, 9]])
					else:
						devid = random.randint(0, ceil(len(COMPUTE_CAPS) / 2))
					command.extend(("-hwaccel_device", str(devid)))
				command.extend([
					"-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
					"-video_size", "x".join(map(str, size)), "-i", "-",
				])
				if fmt in ("gif", "apng"):
					command.extend(("-gifflags", "-offsetting"))
					if new["count"] > 4096:
						# vf = None
						vf = "split[s0][s1];[s0]palettegen=reserve_transparent=1:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle:alpha_threshold=128"
					else:
						vf = "split[s0][s1];[s0]palettegen="
						if mode == "RGBA":
							vf += "reserve_transparent=1:"
						vf += "stats_mode=diff[p];[s1][p]paletteuse=alpha_threshold=128:diff_mode=rectangle"
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
						command.extend(("-vf", f"scale={w}:{h}"))
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
				proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=1048576)
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
			with open(out, "rb") as f:
				return f.read()
			# return [out]
	if isinstance(new, Image.Image):
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
		sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception({repr(ex)})\n".encode("utf-8"))
		sys.stdout.buffer.write(f"~print({args},{repr(traceback.format_exc())},sep='\\n',end='')\n".encode("utf-8"))
	sys.stdout.flush()


exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
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
				sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception({repr(ex)})\n".encode("utf-8"))
				sys.stdout.buffer.write(f"~print({s}, end='')\n".encode("utf-8"))
				sys.stdout.buffer.write(f"~print({repr(traceback.format_exc())}, end='')\n".encode("utf-8"))
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
		while True:
			argv = await wrap_future(exc.submit(sys.stdin.readline))
			if not argv:
				raise SystemExit
			await process_cmd(argv)

	asyncio.set_event_loop(loop)
	asyncio.main_new_loop = loop
	loop.create_task(ensure_parent())
	loop.run_until_complete(update_loop())
