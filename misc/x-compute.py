#!/usr/bin/python3

import os, sys, io, time, concurrent.futures, asyncio, subprocess, psutil, collections, traceback, re, requests, blend_modes, zipfile, contextlib, filetype, ast, colorspace, orjson, base64, random
import numpy as np
import PIL
from PIL import Image, ImageCms, ImageOps, ImageChops, ImageDraw, ImageFilter, ImageEnhance, ImageMath, ImageStat, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Resampling = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
Transform = getattr(Image, "Transform", Image)
Image.MAX_IMAGE_PIXELS = 4294967296
from math import *

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

if len(sys.argv) > 1 and sys.argv[1].isnumeric() and int(sys.argv[1]) >= 3:
	fix = int(sys.argv[1]) - 3
	os.environ["CUDA_VISIBLE_DEVICES"] = str(fix)
if len(sys.argv) > 2:
	COMPUTE_LOAD = orjson.loads(sys.argv[2])

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
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
	if not out:
		out = simple_mimes(path, mime)
	return out

class magic:
	from_file = from_file
	from_buffer = from_file

start = time.time()
CACHE = {}
ANIM = False

qr_bytes = [
	16, 28, 44, 64, 86, 108, 124, 154, 182, 216, 254, 290, 334, 365, 415, 453, 507, 563, 627,
	669, 714, 782, 860, 914, 1000, 1062, 1128, 1193, 1267, 1373, 1455, 1541, 1631, 1725, 1812, 1914, 1992, 2102, 2216, 2334,
]
qr_bytes_ex = [
	2434, 2566, 2702, 2812, 2956,
]

SWIRL = None

def to_qr(s, rainbow=False):
	global SWIRL
	if type(s) is str:
		s = s.encode("utf-8")
	size = len(s)
	err = "M" if size <= 2334 else "L"
	ver = None
	# if size > 125:
	#	 if size <= 2334:
	#		 for i, n in enumerate(qr_bytes):
	#			 if n >= size:
	#				 ver = i + 1
	#				 break
	#	 if ver is None:
	#		 for i, n in enumerate(qr_bytes_ex):
	#			 if n >= size:
	#				 ver = i + 36
	#				 err = "L"
	#	 if ver is None:
	#		 raise OverflowError("Input string too large for QR code encoding.")
	import pyqrcode
	img = pyqrcode.create(s, error=err, version=ver, mode=None, encoding="utf-8" if max(s) >= 80 else "ascii")
	fn = f"cache/{time.time_ns() // 1000}.png"
	if not os.path.exists(fn):
		img.png(fn, scale=1, module_color=(255,) * 3, background=(0,) * 4)
	import pillow_heif
	pillow_heif.register_heif_opener()
	imo = Image.open(fn)
	im = imo.convert("1")
	imo.close()
	im = im.resize((512, 512), resample=Resampling.NEAREST)
	if rainbow:
		if SWIRL is None:
			imo = Image.open("misc/swirl.png")
			SWIRL = imo.resize((512, 512), resample=Resampling.BILINEAR)
			imo.close()
		count = 128

		def qr_iterator(image):
			filt1 = filt2 = SWIRL
			# spl = hsv_split(SWIRL, convert=False)
			spl = SWIRL.convert("HSV").split()
			for i in range(count):
				if i:
					# hue1 = spl[0] + round(i * 256 / count)
					# hue2 = spl[0] - round(i * 256 / count)
					# filt1 = hsv_merge(hue1, *spl[1:])
					# filt2 = hsv_merge(hue2, *spl[1:])
					hue1 = spl[0].point(lambda x: round(x + 256 * i / count) & 255)
					hue2 = spl[0].point(lambda x: round(x - 256 * i / count) & 255)
					filt1 = Image.merge("HSV", (hue1, spl[1], spl[2])).convert("RGB")
					filt2 = Image.merge("HSV", (hue2, spl[1], spl[2])).convert("RGB")
				filt1 = ImageEnhance.Brightness(ImageEnhance.Contrast(filt1).enhance(0.5)).enhance(2)
				filt2 = ImageChops.invert(ImageEnhance.Brightness(ImageEnhance.Contrast(filt2).enhance(0.5)).enhance(2)).transpose(Transpose.FLIP_LEFT_RIGHT)
				filt1.paste(filt2, mask=image)
				yield filt1

		return dict(duration=4800, count=count, frames=qr_iterator(im))
	return ImageChops.invert(im).convert("RGBA")


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

def rgb_split(image, dtype=np.uint8):
	channels = None
	if "RGB" not in str(image.mode):
		if str(image.mode) == "L":
			channels = [np.asanyarray(image, dtype=dtype)] * 3
		else:
			image = image.convert("RGB")
	if channels is None:
		a = np.asanyarray(image, dtype=dtype)
		channels = np.swapaxes(a, 2, 0)[:3]
	return channels

def xyz_split(image, convert=True, dtype=np.uint8):
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
		command = ["./ffmpeg", "-threads", "2", "-hide_banner", "-nostdin", "-v", "error", "-y", "-hwaccel", hwaccel, "-i", f_in, "-vf"]
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
		command.extend([vf, "-loop", "0", "-framerate", str(fps), out])
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
					length = f
				except EOFError:
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

def rainbow_gif2(image, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	length = f
	loops = total / duration / 1000
	scale = 1
	while abs(loops * scale) < 1:
		scale <<= 1
		if length * scale >= 64:
			loops = 1 if loops >= 0 else -1
			break
	loops = round(loops * scale) / scale
	if abs(loops) < 1:
		loops = 1 if loops >= 0 else -1
	size = image.size
	# print(image, length, scale, loops)

	def rainbow_gif_iterator(image):
		for f in range(length * scale):
			image.seek(f % length)
			if str(image.mode) == "P":
				temp = image.convert("RGBA")
			else:
				temp = image
			if str(image.mode) == "RGBA":
				A = temp.getchannel("A")
			else:
				A = None
			if temp.size[0] != size[0] or temp.size[1] != size[1]:
				temp = temp.resize(size, Resampling.HAMMING)
			channels = list(temp.convert("HSV").split())
			# channels = hsv_split(temp, convert=False)
			# hue = channels[0] + round(f / length / scale * loops * 256)
			# temp = hsv_merge(hue, *channels[1:])
			channels[0] = channels[0].point(lambda x: int(((f / length / scale * loops + x / 256) % 1) * 256))
			temp = Image.merge("HSV", channels).convert("RGB")
			if A:
				temp.putalpha(A)
			yield temp

	return dict(duration=total * scale, count=length * scale, frames=rainbow_gif_iterator(image))

def rainbow_gif(image, duration):
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return rainbow_gif2(image, duration)
	# image = resize_max(image, 960, resample=Image.HAMMING)
	# size = list(image.size)
	size = image.size
	if duration == 0:
		fps = 0
	else:
		fps = round(256 / abs(duration))
	rate = 1
	while fps > 48 and rate < 8:
		fps >>= 1
		rate <<= 1
	while fps >= 64:
		fps >>= 1
		rate <<= 1
	if fps <= 0:
		raise ValueError("Invalid framerate value.")
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	if str(image.mode) == "RGBA":
		A = image.getchannel("A")
	else:
		A = None
	channels = list(image.convert("HSV").split())
	# channels = hsv_split(image, convert=False)
	if duration < 0:
		rate = -rate
	count = 256 // abs(rate)
	func = lambda x: (x + rate) & 255

	# Repeatedly hueshift image and return copies
	def rainbow_gif_iterator(image):
		for i in range(0, 256, abs(rate)):
			if i:
				# hue = channels[0] + i
				# image = hsv_merge(hue, *channels[1:])
				channels[0] = channels[0].point(func)
				image = Image.merge("HSV", channels).convert("RGBA")
				if A is not None:
					image.putalpha(A)
			yield image

	return dict(duration=1000 / fps * count, count=count, frames=rainbow_gif_iterator(image))


def spin_gif2(image, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	length = f
	loops = total / duration / 1000
	scale = 1
	while abs(loops * scale) < 1:
		scale *= 2
		if length * scale >= 64:
			loops = 1 if loops >= 0 else -1
			break
	loops = round(loops * scale) / scale
	if abs(loops) < 1:
		loops = 1 if loops >= 0 else -1
	size = image.size

	def spin_gif_iterator(image):
		for f in range(length * scale):
			image.seek(f % length)
			temp = image
			if temp.size[0] != size[0] or temp.size[1] != size[1]:
				temp = temp.resize(size, Resampling.HAMMING)
			temp = to_circle(rotate_to(temp, f * 360 / length / scale * loops, expand=False))
			yield temp

	return dict(duration=total * scale, count=length * scale, frames=spin_gif_iterator(image))


def spin_gif(image, duration):
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return spin_gif2(image, duration)
	maxsize = 960
	size = list(image.size)
	if duration == 0:
		fps = 0
	else:
		fps = round(256 / abs(duration))
	rate = 1
	while fps > 32 and rate < 8:
		fps >>= 1
		rate <<= 1
	while fps >= 64:
		fps >>= 1
		rate <<= 1
	if fps <= 0:
		raise ValueError("Invalid framerate value.")
	if duration < 0:
		rate = -rate
	count = 256 // abs(rate)

	# Repeatedly rotate image and return copies
	def spin_gif_iterator(image):
		for i in range(0, 256, abs(rate)):
			if i:
				im = rotate_to(image, i * 360 / 256, expand=False)
			else:
				im = image
			yield to_circle(im)

	return dict(duration=1000 / fps * count, count=count, frames=spin_gif_iterator(image))


def orbit_gif2(image, orbitals, duration, extras):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	length = f
	loops = total / duration / 1000
	scale = 1
	while abs(loops * scale) < 1:
		scale *= 2
		if length * scale >= 64:
			loops = 1 if loops >= 0 else -1
			break
	loops = round(loops * scale) / scale
	if abs(loops) < 1:
		loops = 1 if loops >= 0 else -1
	sources = [image]
	sources.extend(extras)

	def orbit_gif_iterator(sources):
		x = orbitals if len(sources) == 1 else 1
		diameter = max(sources[0].size)
		scale2 = orbitals / pi * (sqrt(5) + 1) / 2 + 0.5
		size = (round(diameter * scale2),) * 2
		for f in range(0, length * scale):
			im = Image.new("RGBA", size, (0,) * 4)
			if orbitals > 1:
				im2 = Image.new("RGBA", size, (0,) * 4)
				if orbitals & 1:
					im3 = Image.new("RGBA", size, (0,) * 4)
			for j in range(orbitals):
				image = sources[j % len(sources)]
				if hasattr(image, "length"):
					g = f % image.length
				else:
					g = f
				try:
					image.seek(g)
				except EOFError:
					image.length = f
					image.seek(0)
				image = resize_max(image, diameter, force=True)
				angle = f / length / scale * loops * tau / x + j / orbitals * tau
				pos = im.width / 2 + np.array((cos(angle), sin(angle))) * (diameter * scale2 / 2 - diameter / 2) - (image.width / 2, image.height / 2)
				pos = list(map(round, pos))
				if j == orbitals - 1 and orbitals & 1 and orbitals > 1:
					im3.paste(image, pos)
				elif not j & 1:
					im.paste(image, pos)
				else:
					im2.paste(image, pos)
			if orbitals > 1:
				if orbitals & 1:
					im2 = Image.alpha_composite(im3, im2)
				im = Image.alpha_composite(im, im2)
			yield im

	return dict(duration=total * scale, count=length * scale, frames=orbit_gif_iterator(sources))


def orbit_gif(image, orbitals, duration, extras):
	if extras:
		extras = [get_image(url) for url in extras[:orbitals]]
	else:
		duration /= orbitals
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return orbit_gif2(image, orbitals, duration, extras)
	maxsize = 960
	size = list(image.size)
	if duration == 0:
		fps = 0
	else:
		fps = round(256 / abs(duration))
	rate = 1
	while fps > 32 and rate < 8:
		fps >>= 1
		rate <<= 1
	while fps >= 64:
		fps >>= 1
		rate <<= 1
	if fps <= 0:
		raise ValueError("Invalid framerate value.")
	if duration < 0:
		rate = -rate
	count = 256 // abs(rate)
	sources = [image]
	sources.extend(extras)

	# Repeatedly rotate image and return copies
	def orbit_gif_iterator(sources):
		x = orbitals if len(sources) == 1 else 1
		diameter = max(sources[0].size)
		scale = orbitals / pi * (sqrt(5) + 1) / 2 + 0.5
		size = (round(diameter * scale),) * 2
		for i in range(0, 256, abs(rate)):
			im = Image.new("RGBA", size, (0,) * 4)
			if orbitals > 1:
				im2 = Image.new("RGBA", size, (0,) * 4)
				if orbitals & 1:
					im3 = Image.new("RGBA", size, (0,) * 4)
			for j in range(orbitals):
				image = sources[j % len(sources)]
				image = resize_max(image, diameter, force=True)
				angle = i / 256 * tau / x + j / orbitals * tau
				pos = im.width / 2 + np.array((cos(angle), sin(angle))) * (diameter * scale / 2 - diameter / 2) - (image.width / 2, image.height / 2)
				pos = list(map(round, pos))
				if j == orbitals - 1 and orbitals & 1 and orbitals > 1:
					im3.paste(image, pos)
				elif not j & 1:
					im.paste(image, pos)
				else:
					im2.paste(image, pos)
			if orbitals > 1:
				if orbitals & 1:
					im2 = Image.alpha_composite(im3, im2)
				im = Image.alpha_composite(im, im2)
			yield im

	return dict(duration=1000 / fps * count, count=count, frames=orbit_gif_iterator(sources))


def to_square(image):
	w, h = image.size
	d = w - h
	if not d:
		return image
	if d > 0:
		return image.crop((d >> 1, 0, w - (1 + d >> 1), h))
	return image.crop((0, -d >> 1, w, h - (1 - d >> 1)))


CIRCLE_CACHE = {}

def to_circle(image):
	global CIRCLE_CACHE
	if str(image.mode) != "RGBA":
		image = to_square(image).convert("RGBA")
	else:
		image = to_square(image)
	try:
		image_map = CIRCLE_CACHE[image.size]
	except KeyError:
		image_map = Image.new("RGBA", image.size)
		draw = ImageDraw.Draw(image_map)
		draw.ellipse((0, 0, *image.size), outline=0, fill=(255,) * 4, width=0)
		CIRCLE_CACHE[image.size] = image_map
	return ImageChops.multiply(image, image_map)


DIRECTIONS = dict(
	left=0,
	up=1,
	right=2,
	down=3,
	l=0,
	u=1,
	r=2,
	d=3,
)
DIRECTIONS.update({
	"0": 0,
	"1": 1,
	"2": 2,
	"3": 3,
})

def scroll_gif2(image, direction, duration):
	total = 0
	for f in range(2147483647):
		try:
			image.seek(f)
		except EOFError:
			break
		dur = max(image.info.get("duration", 0), 50)
		total += dur
	count = f

	def scroll_gif_iterator(image):
		if direction & 1:
			y = (direction & 2) - 1
			x = 0
		else:
			x = (direction & 2) - 1
			y = 0
		for i in range(count):
			image.seek(i)
			temp = resize_max(image, 960, resample=Resampling.HAMMING)
			if i:
				xm = round(x * temp.width / count * i)
				ym = round(y * temp.height / count * i)
				temp = ImageChops.offset(temp, xm, ym)
			yield temp

	return dict(duration=total, count=count, frames=scroll_gif_iterator(image))

def scroll_gif(image, direction, duration, fps):
	try:
		direction = DIRECTIONS[direction.casefold()]
	except KeyError:
		raise TypeError(f"Invalid direction {direction}")
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return scroll_gif2(image, direction, duration)
	image = resize_max(image, 960, resample=Resampling.HAMMING)
	count = round(duration * fps)

	def scroll_gif_iterator(image):
		yield image
		if direction & 1:
			y = (direction & 2) - 1
			x = 0
		else:
			x = (direction & 2) - 1
			y = 0
		for i in range(1, count):
			xm = round(x * image.width / count * i)
			ym = round(y * image.height / count * i)
			temp = ImageChops.offset(image, xm, ym)
			yield temp

	return dict(duration=1000 * duration, count=count, frames=scroll_gif_iterator(image))


def magik_gif2(image, cell_count, grid_distance, iterations):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	length = f
	loops = total / 2 / 1000
	scale = 1
	while abs(loops * scale) < 1:
		scale *= 2
		if length * scale >= 32:
			loops = 1 if loops >= 0 else -1
			break
	loops = round(loops * scale) / scale
	if abs(loops) < 1:
		loops = 1 if loops >= 0 else -1
	size = image.size

	def magik_gif_iterator(image):
		ts = time.time_ns() // 1000
		for f in range(length * scale):
			np.random.seed(ts & 4294967295)
			image.seek(f % length)
			temp = image
			if temp.size[0] != size[0] or temp.size[1] != size[1]:
				temp = temp.resize(size, Resampling.HAMMING)
			for _ in range(int(31 * iterations * f / length / scale)):
				dst_grid = griddify(shape_to_rect(image.size), cell_count, cell_count)
				src_grid = distort_grid(dst_grid, grid_distance)
				mesh = grid_to_mesh(src_grid, dst_grid)
				temp = temp.transform(temp.size, Transform.MESH, mesh, resample=Resampling.NEAREST)
			yield temp

	return dict(duration=total * scale, count=length * scale, frames=magik_gif_iterator(image))


def magik_gif(image, cell_count=7, iterations=1, anim=32, duration=2):
	grid_distance = int(max(1, round(sqrt(np.prod(image.size)) / cell_count / 3 / iterations)))
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return magik_gif2(image, cell_count, grid_distance, iterations)
	image = resize_max(image, 960, resample=Resampling.HAMMING)

	def magik_gif_iterator(image):
		yield image
		for _ in range(anim - 1):
			for _ in range(iterations):
				dst_grid = griddify(shape_to_rect(image.size), cell_count, cell_count)
				src_grid = distort_grid(dst_grid, grid_distance)
				mesh = grid_to_mesh(src_grid, dst_grid)
				image = image.transform(image.size, Transform.MESH, mesh, resample=Resampling.NEAREST)
			yield image

	return dict(duration=duration * 1000, count=anim, frames=magik_gif_iterator(image))


def quad_as_rect(quad):
	if quad[0] != quad[2]: return False
	if quad[1] != quad[7]: return False
	if quad[4] != quad[6]: return False
	if quad[3] != quad[5]: return False
	return True

def quad_to_rect(quad):
	assert(len(quad) == 8)
	assert(quad_as_rect(quad))
	return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
	assert(len(rect) == 4)
	return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
	assert(len(shape) == 2)
	return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
	w = rect[2] - rect[0]
	h = rect[3] - rect[1]
	x_step = w / float(w_div)
	y_step = h / float(h_div)
	y = rect[1]
	grid_vertex_matrix = deque()
	for _ in range(h_div + 1):
		grid_vertex_matrix.append(deque())
		x = rect[0]
		for _ in range(w_div + 1):
			grid_vertex_matrix[-1].append([int(x), int(y)])
			x += x_step
		y += y_step
	grid = np.array(grid_vertex_matrix)
	return grid

def distort_grid(org_grid, max_shift):
	new_grid = np.copy(org_grid)
	x_min = np.min(new_grid[:, :, 0])
	y_min = np.min(new_grid[:, :, 1])
	x_max = np.max(new_grid[:, :, 0])
	y_max = np.max(new_grid[:, :, 1])
	new_grid += np.random.randint(-max_shift, max_shift + 1, new_grid.shape)
	new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
	new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
	new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
	new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
	return new_grid

def grid_to_mesh(src_grid, dst_grid):
	assert(src_grid.shape == dst_grid.shape)
	mesh = deque()
	for i in range(src_grid.shape[0] - 1):
		for j in range(src_grid.shape[1] - 1):
			src_quad = [src_grid[i	, j	, 0], src_grid[i	, j	, 1],
						src_grid[i + 1, j	, 0], src_grid[i + 1, j	, 1],
						src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
						src_grid[i	, j + 1, 0], src_grid[i	, j + 1, 1]]
			dst_quad = [dst_grid[i	, j	, 0], dst_grid[i	, j	, 1],
						dst_grid[i + 1, j	, 0], dst_grid[i + 1, j	, 1],
						dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
						dst_grid[i	, j + 1, 0], dst_grid[i	, j + 1, 1]]
			dst_rect = quad_to_rect(dst_quad)
			mesh.append([dst_rect, src_quad])
	return list(mesh)

def magik(image, cell_count=7):
	dst_grid = griddify(shape_to_rect(image.size), cell_count, cell_count)
	src_grid = distort_grid(dst_grid, int(max(1, round(sqrt(np.prod(image.size)) / cell_count / 3))))
	mesh = grid_to_mesh(src_grid, dst_grid)
	return image.transform(image.size, Transform.MESH, mesh, resample=Resampling.NEAREST)


blurs = {
	"box": ImageFilter.BoxBlur,
	"boxblur": ImageFilter.BoxBlur,
	"gaussian": ImageFilter.GaussianBlur,
	"gaussianblur": ImageFilter.GaussianBlur,
}

def blur(image, filt="box", radius=2):
	try:
		_filt = blurs[filt.replace("_", "").casefold()]
	except KeyError:
		raise TypeError(f'Invalid image operation: "{filt}"')
	return image.filter(_filt(radius))


def invert(image):
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	if str(image.mode) == "RGBA":
		A = image.getchannel("A")
		image = image.convert("RGB")
	else:
		A = None
	image = ImageOps.invert(image)
	if A is not None:
		image.putalpha(A)
	return image

def greyscale(image):
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	if str(image.mode) == "RGBA":
		A = image.getchannel("A")
	else:
		A = None
	image = ImageOps.grayscale(image)
	if A is not None:
		if str(image.mode) != "L":
			image = image.getchannel("R")
		image = Image.merge("RGBA", (image, image, image, A))
	return image

def laplacian(image):
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	b = image.tobytes()
	try:
		import pygame
	except ImportError:
		pygame = None
	surf = pygame.image.frombuffer(b, image.size, image.mode)
	surf = pygame.transform.laplacian(surf)
	b = pygame.image.tostring(surf, image.mode)
	image = Image.frombuffer(image.mode, image.size, b)
	return image

def colourspace(image, source, dest):
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	if str(image.mode) == "RGBA":
		A = image.getchannel("A")
	else:
		A = None
	im = None
	out = None
	if source in ("xyz", "hsl", "hsi", "hcl", "luv", "yiq", "yuv"):
		spl = rgb_split(image)
		try:
			im = globals()[source + "_merge"](*spl, convert=True)
		except TypeError:
			im = globals()[source + "_merge"](spl)
	else:
		if source == "rgb":
			im = image
		elif source == "cmy":
			im = invert(image)
		elif source == "hsv":
			if image.mode != "RGB":
				image = image.convert("RGB")
			im = Image.frombuffer("HSV", image.size, image.tobytes())
			im = im.convert("RGB")
		elif source == "lab":
			if image.mode != "RGB":
				image = image.convert("RGB")
			im = Image.frombuffer("LAB", image.size, image.tobytes())
			im = ImageCms.applyTransform(im, lab2rgb)
	if not im:
		raise NotImplementedError(f"Colourspace {source} is not currently supported.")
	if dest in ("xyz", "hsl", "hsi", "hcl", "luv", "yiq", "yuv"):
		spl = globals()[dest + "_split"](im, convert=False)
		out = rgb_merge(*spl)
	else:
		if dest == "rgb":
			out = im
		elif dest == "cmy":
			out = invert(im)
		elif dest == "hsv":
			im = im.convert("HSV")
			out = Image.frombuffer("RGB", im.size, im.tobytes())
		elif dest == "lab":
			im = ImageCms.applyTransform(im, rgb2lab)
			out = Image.frombuffer("RGB", im.size, im.tobytes())
	if not out:
		raise NotImplementedError(f"Image conversion from {source} to {dest} is not currently supported.")
	if A is not None:
		out.putalpha(A)
	return out


# Autodetect max image size, keeping aspect ratio
def max_size(w, h, maxsize, force=False):
	s = w * h
	m = maxsize * maxsize
	if s > m or force:
		r = (m / s) ** 0.5
		w = round(w * r)
		h = round(h * r)
	return w, h

def resize_max(image, maxsize, resample=Resampling.LANCZOS, box=None, reducing_gap=None, force=False):
	w, h = max_size(image.width, image.height, maxsize, force=force)
	if w != image.width or h != image.height:
		if type(resample) is str:
			image = resize_to(image, w, h, resample)
		else:
			image = image.resize([w, h], resample, box, reducing_gap)
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
)

def resize_mult(image, x, y, operation):
	if x == y == 1:
		return image
	w = image.width * x
	h = image.height * y
	return resize_to(image, round(w), round(h), operation)

def resize_to(image, w, h, operation="auto"):
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


def get_colour(image):
	if "A" in str(image.mode):
		spl = deque(image.split())
		A = np.divide(spl.pop(), 255)
		sumA = np.sum(A)
		if sumA == 0:
			col = [0, 0, 0]
		else:
			col = [np.sum(np.multiply(channel, A)) / sumA for channel in spl]
	else:
		spl = image.split()
		col = [np.mean(channel) for channel in spl]
	return str(col)


channel_map = {
	"alpha": -1,
	"a": -1,
	"red": 0,
	"r": 0,
	"green": 1,
	"g": 1,
	"blue": 2,
	"b": 2,
	"cyan": 3,
	"c": 3,
	"magenta": 4,
	"m": 4,
	"yellow": 5,
	"y": 5,
	"hue": 6,
	"h": 6,
	"saturation": 7,
	"sat": 7,
	"s": 7,
	"value": 8,
	"v": 8,
	"lightness": 9,
	"luminance": 9,
	"lum": 9,
	"l": 9,
}

def fill_channels(image, colour, *channels):
	channels = list(channels)
	ops = {}
	for c in channels:
		try:
			cid = channel_map[c]
		except KeyError:
			if len(c) <= 1:
				raise TypeError("invalid colour identifier: " + c)
			channels.extend(c)
		else:
			ops[cid] = None
	ch = Image.new("L", image.size, colour)
	if "RGB" not in str(image.mode):
		image = image.convert("RGB")
	if -1 in ops:
		image.putalpha(ch)
	mode = image.mode
	rgb = False
	for i in range(3):
		if i in ops:
			rgb = True
	if rgb:
		spl = list(image.split())
		for i in range(3):
			if i in ops:
				spl[i] = ch
		image = Image.merge(mode, spl)
	cmy = False
	for i in range(3, 6):
		if i in ops:
			cmy = True
	if cmy:
		spl = list(ImageChops.invert(image).split())
		for i in range(3, 6):
			if i in ops:
				spl[i - 3] = ch
		image = ImageChops.invert(Image.merge(mode, spl))
	hsv = False
	for i in range(6, 9):
		if i in ops:
			hsv = True
	if hsv:
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) == "RGBA":
			A = image.getchannel("A")
		else:
			A = None
		spl = list(image.convert("HSV").split())
		# spl = hsv_split(image, convert=False)
		for i in range(6, 9):
			if i in ops:
				spl[i - 6] = ch
		# image = hsv_merge(*spl)
		image = Image.merge("HSV", spl).convert("RGB")
		if A is not None:
			image.putalpha(A)
	if 9 in ops:
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) == "RGBA":
			A = image.getchannel("A")
		else:
			A = None
		spl = hsl_split(image, convert=False)
		spl[-1] = np.full(tuple(reversed(image.size)), colour)
		image = hsl_merge(*spl)
		if A is not None:
			image.putalpha(A)
	return image


# Image blend operations (this is a bit of a mess)
blenders = {
	"normal": "blend",
	"blt": "blend",
	"blit": "blend",
	"blend": "blend",
	"replace": "replace",
	"+": "add",
	"add": "add",
	"addition": "add",
	"-": "subtract",
	"sub": "subtract",
	"subtract": "subtract",
	"subtraction": "subtract",
	"*": "multiply",
	"mul": "multiply",
	"mult": "multiply",
	"multiply": "multiply",
	"multiplication": "multiply",
	"/": blend_modes.divide,
	"div": blend_modes.divide,
	"divide": blend_modes.divide,
	"division": blend_modes.divide,
	"mod": "OP_X%Y",
	"modulo": "OP_X%Y",
	"%": "OP_X%Y",
	"and": "OP_X&Y",
	"&": "OP_X&Y",
	"or": "OP_X|Y",
	"|": "OP_X|Y",
	"xor": "OP_X^Y",
	"^": "OP_X^Y",
	"nand": "OP_255-(X&Y)",
	"~&": "OP_255-(X&Y)",
	"nor": "OP_255-(X|Y)",
	"~|": "OP_255-(X|Y)",
	"xnor": "OP_255-(X^Y)",
	"~^": "OP_255-(X^Y)",
	"xand": "OP_255-(X^Y)",
	"diff": "difference",
	"difference": "difference",
	"overlay": blend_modes.overlay,
	"screen": "screen",
	"soft": blend_modes.soft_light,
	"softlight": blend_modes.soft_light,
	"hard": blend_modes.hard_light,
	"hardlight": blend_modes.hard_light,
	"lighter": "lighter",
	"lighten": "lighter",
	"darker": "darker",
	"darken": "darker",
	"plusdarker": "OP_X+Y-255",
	"plusdarken": "OP_X+Y-255",
	"overflow": "OVERFLOW",
	"lighting": "LIGHTING",
	"extract": blend_modes.grain_extract,
	"grainextract": blend_modes.grain_extract,
	"merge": blend_modes.grain_merge,
	"grainmerge": blend_modes.grain_merge,
	"burn": "OP_255*(1-((255-Y)/X))",
	"colorburn": "OP_255*(1-((255-Y)/X))",
	"colourburn": "OP_255*(1-((255-Y)/X))",
	"linearburn": "OP_(X+Y)-255",
	"dodge": blend_modes.dodge,
	"colordodge": blend_modes.dodge,
	"colourdodge": blend_modes.dodge,
	"lineardodge": "add",
	"hue": "SP_HUE",
	"sat": "SP_SAT",
	"saturation": "SP_SAT",
	"lightness": "SP_LIT",
	"brightness": "SP_LIT",
	"lum": "SP_LUM",
	"luminosity": "SP_LUM",
	"val": "SP_VAL",
	"value": "SP_VAL",
	"color": "SP_COL",
	"colour": "SP_COL",
	"alpha": "SP_ALP",
}
halve = (np.arange(1, 257) >> 1).astype(np.uint8)
darken = np.concatenate((np.zeros(128, dtype=np.uint8), np.arange(128, dtype=np.uint8)))

def blend_op(image, url, operation, amount, recursive=True):
	op = operation.casefold().replace(" ", "").replace("_", "")
	if op in blenders:
		filt = blenders[op]
	elif op == "auto":
		filt = "blend"
	else:
		raise TypeError("Invalid image operation: \"" + op + '"')
	try:
		image2 = get_image(url)
	except TypeError as ex:
		s = ex.args[0]
		search = 'Filetype "audio/'
		if not s.startswith(search):
			raise
		s = s[len(search):]
		image.audio = dict(url=url, codec=s[:s.index('"')])
		return image
	if recursive:
		if not globals()["ANIM"]:
			try:
				image2.seek(1)
			except EOFError:
				image2.seek(0)
			else:
				dur = 0
				for f in range(2147483648):
					try:
						image2.seek(f)
					except EOFError:
						break
					dur += max(image2.info.get("duration", 0), 50)
				count = f

				def blend_op_iterator(image, image2, operation, amount):
					for f in range(2147483648):
						try:
							image2.seek(f)
						except EOFError:
							break
						if str(image.mode) == "P":
							image = image.convert("RGBA")
						elif str(image.mode) != "RGBA":
							temp = image.convert("RGBA")
						else:
							temp = image
						temp2 = image2._images[image2._position]
						# print(image2._position)
						# image2._images[image2._position].save(f"temp{f}.png")
						yield blend_op(temp, temp2, operation, amount, recursive=False)

				return dict(duration=dur, count=count, frames=blend_op_iterator(image, image2, operation, amount))
		try:
			n_frames = 1
			for f in range(CURRENT_FRAME + 1):
				try:
					image2.seek(f)
				except EOFError:
					break
				n_frames += 1
			image2.seek(CURRENT_FRAME % n_frames)
		except EOFError:
			image2.seek(0)
	if image2.width != image.width or image2.height != image.height:
		image2 = resize_to(image2, image.width, image.height, "auto")
	if type(filt) is not str:
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) != "RGBA":
			image = image.convert("RGBA")
		if str(image2.mode) == "P" and "transparency" in image2.info:
			image2 = image2.convert("RGBA")
		if str(image2.mode) != "RGBA":
			image2 = image2.convert("RGBA")
		imgA = np.array(image).astype(np.float64)
		imgB = np.array(image2).astype(np.float64)
		out = fromarray(np.uint8(filt(imgA, imgB, amount)), image.mode)
	else:
		# Basic blend, use second image
		if filt in ("blend", "replace"):
			out = image2
		# Image operation, use ImageMath.eval
		elif filt.startswith("OP_"):
			f = filt[3:]
			if str(image.mode) != str(image2.mode):
				if str(image.mode) == "P":
					image = image.convert("RGBA")
				if str(image.mode) != "RGBA":
					image = image.convert("RGBA")
				if str(image2.mode) == "P" and "transparency" in image2.info:
					image2 = image2.convert("RGBA")
				if str(image2.mode) != "RGBA":
					image2 = image2.convert("RGBA")
			mode = image.mode
			ch1 = image.split()
			ch2 = image2.split()
			c = len(ch1)
			ch3 = [ImageMath.eval(f, dict(X=ch1[i], Y=ch2[i])).convert("L") for i in range(3)]
			if c > 3:
				ch3.append(ImageMath.eval("max(X,Y)", dict(X=ch1[-1], Y=ch2[-1])).convert("L"))
			out = Image.merge(mode, ch3)
		# Special operation, use HSV channels
		elif filt.startswith("SP_"):
			f = filt[3:]
			if f == "ALP":
				if "A" in image2.mode:
					if amount % 1:
						out = image.copy()
					else:
						out = image
					out.putalpha(image2.getchannel("A"))
				else:
					out = image
					amount = 0
			else:
				if str(image.mode) == "P":
					image = image.convert("RGBA")
				if str(image.mode) == "RGBA":
					A = image.getchannel("A")
				else:
					A = None
				if f == "LUM":
					channels1 = yuv_split(image, convert=False)
					channels2 = yuv_split(image2, convert=False)
				elif f == "LIT":
					channels1 = hsl_split(image, convert=False)
					channels2 = hsl_split(image2, convert=False)
				else:
					channels1 = image.convert("HSV").split()
					channels2 = image2.convert("HSV").split()
				if f in ("HUE", "LUM"):
					channels = [channels2[0], channels1[1], channels1[2]]
				elif f == "SAT":
					channels = [channels1[0], channels2[1], channels1[2]]
				elif f in ("LIT", "VAL"):
					channels = [channels1[0], channels1[1], channels2[2]]
				elif f == "COL":
					channels = [channels2[0], channels2[1], channels1[2]]
				if f == "LUM":
					out = yuv_merge(channels)
				elif f == "LIT":
					out = hsl_merge(*channels)
				else:
					out = Image.merge("HSV", channels).convert("RGB")
				if A:
					out.putalpha(A)
		elif filt in ("OVERFLOW", "LIGHTING"):
			if str(image.mode) != str(image2.mode):
				if image.mode == "RGBA" or image2.mode == "RGBA":
					if image.mode != "RGBA":
						image = image.convert("RGBA")
					else:
						image2 = image2.convert("RGBA")
				else:
					mode = image.mode if image.mode != "P" else "RGBA" if "transparency" in image2.info else "RGB"
					image2 = image2.convert(mode)
					if image.mode != mode:
						image = image.convert(mode)
			if "A" in image.mode:
				spl, spl2 = image.split(), image2.split()
				A = ImageChops.add(spl[-1], spl2[-1])
				image = Image.merge("RGB", spl[:-1])
				image2 = Image.merge("RGB", spl2[:-1])
			else:
				A = None
			image = Image.blend(image, image2, 0.5)
			spl = hsl_split(image, convert=False, dtype=np.uint16)
			if filt == "OVERFLOW":
				spl[2] <<= 1
				spl[1] <<= 1
			else:
				temp = spl[2] ^ 255
				temp *= spl[2]
				temp //= 255
				spl[2] += temp
				spl[1] <<= 1
			out = hsl_merge(*spl)
			if A:
				out.putalpha(A)
		# Otherwise attempt to find as ImageChops filter
		else:
			if str(image.mode) != str(image2.mode):
				if str(image.mode) == "P":
					image = image.convert("RGBA")
				if str(image.mode) != "RGBA":
					image = image.convert("RGBA")
				if str(image2.mode) == "P" and "transparency" in image2.info:
					image2 = image2.convert("RGBA")
				if str(image2.mode) != "RGBA":
					image2 = image2.convert("RGBA")
			filt = getattr(ImageChops, filt)
			out = filt(image, image2)
		if str(image.mode) != str(out.mode):
			if str(image.mode) == "P":
				image = image.convert("RGBA")
			if str(image.mode) != "RGBA":
				image = image.convert("RGBA")
			if str(out.mode) == "P" and "transparency" in out.info:
				out = out.convert("RGBA")
			if str(out.mode) != "RGBA":
				out = out.convert("RGBA")
		if filt == "blend":
			A = out.getchannel("A")
			A.point(lambda x: round(x * amount))
			out.putalpha(A)
			out = Image.alpha_composite(image, out)
		else:
			if amount == 0:
				out = image
			elif amount != 1:
				out = Image.blend(image, out, amount)
	return out


def remove_matte(image, colour):
	if str(image.mode) == "P":
		image = image.convert("RGBA")
	if str(image.mode) != "RGBA":
		image = image.convert("RGBA")
	arr = np.asanyarrayarray(image, dtype=np.float32)
	col = np.array(colour)
	t = len(col)
	for row in arr:
		for cell in row:
			r = min(1, np.min(cell[:t] / col))
			if r > 0:
				col = cell[:t] - r * col
				if max(col) > 0:
					ratio = sum(cell) / max(col)
					cell[:t] = np.clip(col * ratio, 0, 255)
					cell[3] /= ratio
				else:
					cell[3] = 0
	image = fromarray(arr.astype(np.uint8))
	return image


colour_blind_map = dict(
	protan=(
		(
			(0.56667, 0.43333, 0),
			(0.55833, 0.44167, 0),
			(0.24167, 0.75833, 0),
		),
		(
			(0.81667, 0.18333, 0),
			(0.33333, 0.66667, 0),
			(0, 0.125, 0.875),
		),
	),
	deutan=(
		(
			(0.625, 0.375, 0),
			(0.7, 0.3, 0),
			(0, 0.3, 0.7),
		),
		(
			(0.8, 0.2, 0),
			(0.25833, 0.74167, 0),
			(0, 0.14167, 0.85833),
		),
	),
	tritan=(
		(
			(0.95, 0.05, 0),
			(0, 0.43333, 0.56667),
			(0, 0.475, 0.525),
		),
		(
			(0.96667, 0.03333, 0),
			(0, 0.73333, 0.26667),
			(0, 0.18333, 0.81667),
		),
	),
	achro=(
		(
			(0.299, 0.587, 0.114),
			(0.299, 0.587, 0.114),
			(0.299, 0.587, 0.114),
		),
		(
			(0.618, 0.32, 0.062),
			(0.163, 0.775, 0.062),
			(0.163, 0.32, 0.516),
		),
	),
)

colour_normal_map = (
	(1, 0, 0),
	(0, 1, 0),
	(0, 0, 1),
)

def colour_deficiency(image, operation, value=None):
	if value is None:
		if operation == "protanopia":
			operation = "protan"
			value = 1
		elif operation == "protanomaly":
			operation = "protan"
			value = 0.5
		if operation == "deuteranopia":
			operation = "deutan"
			value = 1
		elif operation == "deuteranomaly":
			operation = "deutan"
			value = 0.5
		elif operation == "tritanopia":
			operation = "tritan"
			value = 1
		elif operation == "tritanomaly":
			operation = "tritan"
			value = 0.5
		elif operation in ("monochromacy", "achromatopsia"):
			operation = "achro"
			value = 1
		elif operation == "achromatonomaly":
			operation = "achro"
			value = 0.5
		else:
			value = 1
	try:
		table = colour_blind_map[operation]
	except KeyError:
		raise TypeError(f"Invalid filter {operation}.")
	if value < 0.5:
		value *= 2
		ratios = [table[1][i] * value + colour_normal_map[i] * (1 - value) for i in range(3)]
	else:
		value = value * 2 - 1
		ratios = [table[0][i] * value + table[1][i] * (1 - value) for i in range(3)]
	colourmatrix = []
	for r in ratios:
		colourmatrix.extend(r)
		colourmatrix.append(0)
	if image.mode == "P":
		image = image.convert("RGBA")
	if image.mode == "RGBA":
		spl = image.split()
		image = Image.merge("RGB", spl[:3])
		A = spl[-1]
	else:
		A = None
	image = image.convert(image.mode, colourmatrix)
	if A:
		image.putalpha(A)
	return image
	# channels = list(image.split())
	# out = [None] * len(channels)
	# if len(out) == 4:
	#	 out[-1] = channels[-1]
	# for i_ratio, ratio in enumerate(ratios):
	#	 for i_colour in range(3):
	#		 if ratio[i_colour]:
	#			 im = channels[i_colour].point(lambda x: x * ratio[i_colour])
	#			 if out[i_ratio] is None:
	#				 out[i_ratio] = im
	#			 else:
	#				 out[i_ratio] = ImageChops.add(out[i_ratio], im)
	# return Image.merge(image.mode, out)

Enhance = lambda image, operation, value: getattr(ImageEnhance, operation)(image).enhance(value)

def brightness(image, value):
	if value:
		if value < 0:
			image = invert(image)
			value = -value
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) == "RGBA":
			A = image.getchannel("A")
		else:
			A = None
		H, S, L = hsl_split(image, convert=False, dtype=np.uint16)
		np.multiply(L, value, out=L, casting="unsafe")
		image = hsl_merge(H, S, L)
		if A:
			image.putalpha(A)
	return image

def luminance(image, value):
	if value:
		if value < 0:
			image = invert(image)
			value = -value
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) == "RGBA":
			A = image.getchannel("A")
		else:
			A = None
		yuv = yuv_split(image, convert=False)
		np.multiply(yuv[0], value, out=yuv[0])
		image = yuv_merge(yuv)
		if A:
			image.putalpha(A)
	return image

# Hueshift image using HSV channels
def hue_shift(image, value):
	if value:
		if str(image.mode) == "P":
			image = image.convert("RGBA")
		if str(image.mode) == "RGBA":
			A = image.getchannel("A")
		else:
			A = None
		channels = list(image.convert("HSV").split())
		# channels = hsv_split(image, convert=False)
		# channels[0] += round(value * 256)
		# image = hsv_merge(*channels)
		value *= 256
		channels[0] = channels[0].point(lambda x: (x + value) % 256)
		image = Image.merge("HSV", channels).convert("RGB")
		if A is not None:
			image.putalpha(A)
	return image


def get_mask(image):
	if image.mode != "LA":
		image = image.convert("LA")
	if image.size != (512, 512):
		image = image.resize((512, 512), resample=Resampling.LANCZOS)
	a = np.array(image, dtype=np.uint8).T
	L, A = a[0].T, a[1].T
	anytrans = A != 255
	notblank = A != 0
	anyalpha = anytrans & notblank
	at = np.sum(anytrans)
	aa = np.sum(anyalpha)
	anywhite = L == 255
	anyblack = L == 0
	aw = np.sum(anywhite)
	ab = np.sum(anyblack)
	# print(np.sum(anytrans), np.sum(notblank), np.sum(anyalpha), aw, ab)
	if at and aa < at / 2 and at > max(aw, ab):
		L[anytrans] = 255
		L[anytrans == False] = 0
	else:
		if aw and ab:
			avg = np.mean(L[notblank])
			if 255 - avg < 32:
				aw = 0
			elif avg < 32:
				ab = 0
			elif aw > ab:
				ab = 0
			else:
				aw = 0
		if aw and not ab:
			L[(anywhite == False) & notblank] = 0
		elif ab and not aw:
			L[anyblack & notblank] = 255
			L[(anyblack == False) & notblank] = 0
		else:
			raise RuntimeError("Unable to detect mask. Please use full black, white, or transparent.")
	mask = Image.fromarray(L, mode="L")
	return expand_mask(mask, radius=4)

def inpaint(image, url):
	image2 = get_image(url, nodel=True)
	if image2.mode == "LA":
		image2 = image2.getchannel("L")
	elif "RGB" in image2.mode or "P" in image2.mode:
		image2 = image2.convert("L")
	elif image2.mode != "L":
		image2 = image2.convert("L")
	mask = np.asanyarray(image2, dtype=np.uint8) >= 128
	outl = np.roll(mask, -1, axis=0)
	outu = np.roll(mask, -1, axis=1)
	outr = np.roll(mask, 1, axis=0)
	outd = np.roll(mask, 1, axis=1)
	outline = (outl | outu | outr | outd) & (mask == False)
	if image.mode != "RGB":
		image = image.convert("RGB")
	if image.size != image2.size:
		image = image.resize(image2.size, resample=Resampling.LANCZOS)
	a = np.array(image, dtype=np.uint8)
	orients = [None] * 2
	for i in range(2):
		if i:
			b = a.swapaxes(0, 1)
			m2 = mask.T
			o2 = outline.T
		else:
			b = a
			m2 = mask
			o2 = outline
		pm = np.argwhere(m2)
		om = np.argwhere(o2)
		paint_mask = np.empty(len(pm), dtype=object)
		paint_mask[:] = tuple(map(tuple, pm))
		outliner = np.empty(len(om), dtype=object)
		outliner[:] = tuple(map(tuple, om))
		nearr = np.searchsorted(outliner, paint_mask) % len(om)
		nearl = nearr - 1
		ipl = tuple(om[nearl].T)
		ipr = tuple(om[nearr].T)
		dist = np.sqrt(np.sum((pm.astype(np.float32) - om[nearl]) ** 2, axis=1))
		dist /= np.max(dist)
		grads = np.tile(dist, (3, 1)).T
		interpolated = (b[ipl] * (1 - grads) + b[ipr] * grads).astype(np.uint8) >> 1
		orients[i] = (m2, interpolated)
	a[mask] = 0
	for i, (m, o) in enumerate(orients):
		if i:
			a.swapaxes(0, 1)[m] += o
		else:
			a[mask] += o
	im = Image.fromarray(a, mode="RGB")
	filt = ImageFilter.GaussianBlur(radius=1.5)
	im2 = im.filter(filt)
	a2 = np.asanyarray(im2, dtype=np.uint8)
	a[mask] = a2[mask]
	return Image.fromarray(a, mode="RGB")

def expand_mask(image2, radius=4):
	if not radius:
		return image2
	if radius > image2.width:
		radius = image2.width
	if radius > image2.height:
		radius = image2.height
	if image2.mode == "LA":
		image2 = image2.getchannel("L")
	elif "RGB" in image2.mode or "P" in image2.mode:
		image2 = image2.convert("L")
	mask = np.asanyarray(image2, dtype=np.uint8)
	outmask = mask.copy()
	for x in range(-radius, radius + 1):
		for y in range(-radius, radius + 1):
			if x ** 2 + y ** 2 > (radius + 0.5) ** 2:
				continue
			temp = mask.copy()
			if x > 0:
				t2 = temp[:-x]
				temp[:x] = temp[-x:]
				temp[x:] = t2
			elif x < 0:
				t2 = temp[-x:]
				temp[x:] = temp[:-x]
				temp[:x] = t2
			if y > 0:
				t2 = temp.T[:-y]
				temp.T[:y] = temp.T[-y:]
				temp.T[y:] = t2
			elif y < 0:
				t2 = temp.T[-y:]
				temp.T[y:] = temp.T[:-y]
				temp.T[:y] = t2
			outmask |= temp
	outim = Image.fromarray(outmask, mode="L")
	return outim


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

discord_emoji = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/assets\\/[0-9A-Fa-f]+\\.svg")
is_discord_emoji = lambda url: discord_emoji.search(url)


sys.path.append("misc")

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
			return cls(model, **kwargs)
		except Exception as ex:
			ex2 = ex
	else:
		try:
			return cls(model, local_files_only=True, **kwargs)
		except:
			fut = exc.submit(cached_model, cls, model, **kwargs)
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

if len(sys.argv) <= 1 or int(sys.argv[1]) < 3:
	x_math = __import__("x-math")

if len(sys.argv) > 1 and sys.argv[1] == "1":
	import convobot, torch
	convobot.COMPUTE_LOAD = COMPUTE_LOAD

	def CBAI(inputs):
		user_id = inputs["user_id"]
		channel_id = inputs["channel_id"]
		key = inputs["key"]
		ht = inputs["huggingface_token"]
		vis = inputs.get("vis_session")
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
				if premium < 4:
					cb.model = "gpt3+"
				else:
					cb.model = "gpt4+"
		else:
			if model.startswith("gpt4") and premium < 4:
				cb.model = "gpt3+"
			else:
				cb.model = model or "gpt3+"
		cb.user_id = user_id
		cb.channel_id = channel_id
		cb.bl = bl
		cb.oai = oai
		cb.bals = bals
		cb.nsfw = nsfw
		cb.vis_s = vis
		if inputs.get("reset"):
			to = []
			for i, t in enumerate(history):
				cb.append(t, nin=len(history) - i - 1, to=to, ai=torch.cuda.is_available())
			cb.chat_history = to
		cb.jailbroken = jb
		if im:
			try:
				im = cb.image
			except AttributeError:
				im = get_image(im)
		res = cb.ai(*prompt, refs=refs, im=im)
		if cb.model in ("gpt3", "gpt4", "gpt3+", "gpt4+"):
			cb.model = None
		return res

	def CBAU(inputs):
		user_id = inputs["user_id"]
		channel_id = inputs["channel_id"]
		prompt = inputs["prompt"]
		key = inputs["key"]
		ht = inputs["huggingface_token"]
		vis = inputs.get("vis_session")
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
			cb.vis_s = vis
		cb.user_id = user_id
		cb.channel_id = channel_id
		cb.oai = oai
		cb.bals = bals
		cb.nsfw = nsfw
		cb.premium = premium
		return cb.au(prompt)

	try:
		from chatgpt_wrapper import AsyncChatGPT
	except ImportError:
		convobot.AsyncChatGPT = None
	except:
		convobot.AsyncChatGPT = None
		print(traceback.format_exc(), end="")
	else:
		convobot.AsyncChatGPT = AsyncChatGPT

elif len(sys.argv) > 1 and sys.argv[1] == "2":

	def determine_cuda(mem=1, priority=None, multi=False):
		if not torch.cuda.is_available():
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
		if priority == "full":
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i], p.major, p.minor, p.multi_processor_count, p.total_memory)
		elif priority:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, COMPUTE_LOAD[i], p.multi_processor_count, p.total_memory)
		elif priority is False:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, -COMPUTE_LOAD[i], -gmems[i].free, p.multi_processor_count)
		else:
			key = lambda i: (p := tinfo[i]) and (gmems[i].free >= mem, -COMPUTE_LOAD[i], -p.multi_processor_count, -gmems[i].free)
		pcs = sorted(range(n), key=key, reverse=True)
		if multi:
			return [i for i in pcs if gmems[i].free >= mem], torch.float16
		return pcs[0], torch.float16

	device, dtype = determine_cuda(1073741824, priority=False)
	device = f"cuda:{device}" if device >= 0 else "cpu"
	from sentence_transformers import SentenceTransformer
	Embedder = SentenceTransformer("LLukas22/all-mpnet-base-v2-embedding-all", device=device)
	if dtype == torch.float16:
		try:
			Embedder = Embedder.half()
		except (RuntimeError, NotImplementedError):
			pass
	def embedding(s):
		a = Embedder.encode(s).astype(np.float16)
		return a.data

elif len(sys.argv) > 1:
	import imagebot
	imagebot.COMPUTE_LOAD = COMPUTE_LOAD
	for i in range(3):
		try:
			from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViltProcessor, ViltForQuestionAnswering
			from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionImageVariationPipeline
		except ImportError:
			print(traceback.format_exc(), end="")
			time.sleep(i + 1)
		else:
			break

	VGPT = VVQA = None
	def caption(im, q=None, cid=None):
		im = resize_max(im, 512, "auto")
		if im.mode != "RGB":
			image = im.convert("RGB")
		else:
			image = im
		p1 = p2 = None
		if VGPT:
			p, m = VGPT
		else:
			p = backup_model(TrOCRProcessor.from_pretrained, "nlpconnect/vit-gpt2-image-captioning")
			m = backup_model(VisionEncoderDecoderModel.from_pretrained, "nlpconnect/vit-gpt2-image-captioning").to("cpu")
			globals()["VGPT"] = (p, m)
		impv = p(image, return_tensors="pt")
		pixel_values = impv.pixel_values
		generated_ids = m.generate(pixel_values)
		generated_text = p.batch_decode(generated_ids, skip_special_tokens=True)[0]
		p1 = generated_text.strip()
		if not q:
			return (p1, "")
		if VVQA:
			p, m = VVQA
		else:
			p = backup_model(ViltProcessor.from_pretrained, "dandelin/vilt-b32-finetuned-vqa")
			m = backup_model(ViltForQuestionAnswering.from_pretrained, "dandelin/vilt-b32-finetuned-vqa").to("cpu")
			globals()["VVQA"] = (p, m)
		spl = q.split()
		t = " ".join(w for w in spl if not is_url(w))[:32]
		encoding = p(image, t, return_tensors="pt")
		outputs = m(**encoding)
		logits = outputs.logits
		idx = logits.argmax(-1).item()
		p2 = m.config.id2label[idx].strip()
		return (p1, p2)

	def IBASL(prompt, kwargs, nsfw=False, force=False, count=1):
		try:
			ib = CBOTS[None]
		except KeyError:
			ib = CBOTS[None] = imagebot.Bot()
		return ib.art_stablediffusion_local(prompt, kwargs, nsfw=nsfw, fail_unless_gpu=not force, count=count)

else:
	del torch

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
	import pillow_heif
	pillow_heif.register_heif_opener()
	mime = magic.from_buffer(data)
	if mime == "application/zip":
		z = zipfile.ZipFile(io.BytesIO(data), compression=zipfile.ZIP_DEFLATED, strict_timestamps=False)
		return ImageSequence(*(Image.open(z.open(f.filename)) for f in z.filelist if not f.is_dir()))
	try:
		if mime.split("/", 1)[0] == "image" and mime.split("/", 1)[-1] in "blp bmp cur dcx dds dib emf eps fits flc fli fpx ftex gbr gd heif heic icns ico im imt iptc jpeg jpg mcidas mic mpo msp naa pcd pcx pixar png ppm psd sgi sun spider tga tiff wal wmf xbm".split():
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
	import wand, wand.image
	with wand.image.Image() as im:
		with wand.color.Color("transparent") as background_color:
			wand.api.library.MagickSetBackgroundColor(
				im.wand,
				background_color.resource,
			)
		try:
			im.read(blob=b, resolution=1024)
		except Exception as ex:
			exc.args = exc.args + (ex.__class__,) + ex.args
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
			if url in CACHE:
				return CACHE[url]
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
			CACHE[url] = image
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
		frames = new["frames"]
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
				first = next(it)

				def frameit(first, it):
					yield first
					with suppress(StopIteration):
						while True:
							yield next(it)

				frames = frameit(first, it)
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
				command = [
					"./ffmpeg", "-threads", "2", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel,
					"-f", "rawvideo", "-framerate", str(fps), "-pix_fmt", ("rgb24" if mode == "RGB" else "rgba"),
					"-video_size", "x".join(map(str, size)), "-i", "-",
				]
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
						command.extend(("yuva420p", "-c:v", "libsvtav1"))
						fmt = "webm"
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
		if isinstance(out, (tuple, list)) and isinstance(out[0], io.BytesIO):
			out[0].seek(0)
			out[0] = out[0].read()
		elif isinstance(out, io.BytesIO):
			out.seek(0)
			out = out.read()
		if isinstance(out, (bytes, memoryview)):
			b = base64.b64encode(out)
			sys.stdout.buffer.write(f"$PROC_RESP[{ts}].set_result(_x)~".encode("utf-8"))
			sys.stdout.buffer.write(b)
			sys.stdout.buffer.write(b"\n")
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
				elif "plt_special" in args or "plt_mp" in args:
					import matplotlib.pyplot
					globals()["plt"] = matplotlib.pyplot
					evaluate(ts, args)
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
