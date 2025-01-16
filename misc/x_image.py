import collections
import concurrent.futures
import contextlib
import fractions
import io
import itertools
from math import ceil, floor, inf, sqrt, tau, log2, sin, cos, pi, isfinite
import os
import random
import re
import subprocess
import time
from traceback import print_exc
import zipfile
import blend_modes
import cv2
import filetype
import numpy as np
import orjson
import psutil
import requests
try:
	# This module apparently does not exist on Linux
	import colorspace
except Exception:
	print_exc()
	colorspace = None
import PIL
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter, ImageEnhance, ImageMath, ImageStat, ImageFile # noqa: F401
from PIL import GifImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
Resampling = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
Transform = getattr(Image, "Transform", Image)
Image.MAX_IMAGE_PIXELS = 4294967296
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY

DC = 0
torch = None
try:
	import pynvml  # noqa: E402
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
	if not os.environ.get("AI_FEATURES", True):
		raise StopIteration("AI features disabled.")
except Exception:
	print_exc()
hwaccel = "cuda" if DC else "d3d11va" if os.name == "nt" else "auto"

# try:
# 	import torch
# except Exception:
# 	torch = None
# 	hwaccel = "d3d11va" if os.name == "nt" else "auto"
# else:
# 	hwaccel = "cuda" if torch.cuda.is_available() else "d3d11va" if os.name == "nt" else "auto"
# 	torch.backends.cuda.matmul.allow_tf32 = True

if not hasattr(time, "time_ns"):
	time.time_ns = lambda: int(time.time() * 1e9)

requests = requests.Session()

def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

mpf = float
deque = collections.deque
suppress = contextlib.suppress

exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)


def ImageOpIterator(image, step=1, operation=None, ts=0, args=()):
	# Attempt to perform operation on all individual frames of .gif images
	fl = 0
	for i in range(2147483648):
		f = max(fl, round_random(i * step) if step > 1 else round(i * step))
		fl = f
		np.random.seed(ts & 4294967295)
		try:
			image.seek(f)
		except EOFError:
			break
		if str(image.mode) == "P":
			temp = remove_p(image)
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

class ImageSequence(Image.Image):

	frameprops = None
	it = None

	@classmethod
	def pipe(cls, pipe, mode, size, frameprops, close=None):
		self = cls.fromiter(cls.iter(pipe, mode, size, frameprops), frameprops=frameprops, close=close)
		assert self.mode == mode, f"{self.mode} != {mode}"
		assert self.size == size, f"{self.size} != {size}"
		return self

	@classmethod
	def iter(cls, pipe, mode, size, frameprops):
		while True:
			_frames, _duration, fps = frameprops
			bcount = 4 if mode == "RGBA" else 3
			bcount *= int(np.prod(size))
			b = pipe.read(bcount)
			if len(b) < bcount:
				break
			img = Image.frombuffer(mode, size, b)
			img.info["duration"] = 1000 / fps
			yield img

	@classmethod
	def fromiter(cls, it, frameprops=None, close=None, **kwargs):
		it = iter(it)
		self = cls(next(it), copy=None, **kwargs)
		self.it = it
		self.frameprops = frameprops
		self.close = close
		print("FROMITER:", self, it, frameprops)
		return self

	@classmethod
	def cast(cls, image, **kwargs):
		if isinstance(image, cls):
			return image
		if isinstance(image, dict):
			if not isinstance(image["frames"], (tuple, list)):
				return cls.fromiter(image["frames"], frameprops=(image["count"], image["duration"], image["count"] / image["duration"]), **kwargs)
			return cls.fromiter(image["frames"], frameprops=(len(image["frames"]), image["duration"], len(image["frames"]) / image["duration"]), **kwargs)
		if isinstance(image, (tuple, list)):
			return cls(*image, **kwargs)
		return cls(image, **kwargs)

	@classmethod
	def open(cls, *images, copy=False, func=None, args=()):
		if len(images) == 1:
			simages = []
			im = images[0]
			im.load()
			prop = properties(im)
			try:
				for i in range(prop[0]):
					im.seek(i)
					if i == prop[0] - 1:
						im2 = im
					else:
						im2 = im.copy()
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
			copy = False
		return cls(*images, copy=copy, func=func, args=args)

	def __iter__(self):
		return iter(self._images)

	def __repr__(self):
		return f"<{self.__class__.__name__} frames={len(self)} mode={self.mode} size={self.size} extra_info={properties(self)}>"

	def __init__(self, *images, copy=False, func=None, args=()):
		if func:
			self._images = [func(image, *args) for image in images]
		elif copy:
			self._images = [image.copy() for image in images]
		elif copy is None:
			self._images = list(images)
		else:
			self._images = images
			[im.load() for im in images]
		dur2 = images[0].info.get("total_duration", 0) / len(images)
		for i1, i2 in zip(self._images, images):
			i1.info["duration"] = i2.info.get("duration", dur2) or 25
		self._position = 0

	def __len__(self):
		if self.frameprops:
			return self.frameprops[0]
		return len(self._images)

	def seek(self, position):
		if position >= len(self):
			raise EOFError
		self._position = position

	def __getattr__(self, key):
		try:
			return self.__getattribute__(key)
		except AttributeError:
			if self.it:
				while len(self._images) <= self._position:
					self._images.append(next(self.it))
					if len(self._images) >= len(self):
						if self.close:
							self.close()
						break
			im = self._images[self._position]
			if not im.im:
				im.load()
			return getattr(im, key)

def from_bytes(b, save=None, nogif=False, maxframes=inf, orig=None, msize=None):
	"""
	Convert a byte stream into an image or image sequence.

	Parameters:
	b (bytes): The byte stream to convert.
	save (str, optional): Path to save the image. Defaults to None.
	nogif (bool, optional): If True, only the first frame of the image is processed. Defaults to False.
	maxframes (int, optional): Maximum number of frames to process. Defaults to inf.
	orig (str, optional): Original file path, used for specific formats like m3u8. Defaults to None.
	msize (int, optional): Maximum size for the image. Defaults to None.

	Returns:
	Image or ImageSequence: The resulting image or sequence of images.

	Raises:
	FileNotFoundError: If the image file is not found.
	TypeError: If the file type is not supported.
	RuntimeError: If an error occurs during processing.
	"""
	if b[:4] == b"<svg" or b[:5] == b"<?xml":
		import wand
		import wand.image
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
	try:
		import pillow_avif  # noqa: F401
	except ImportError:
		pass
	mime = magic.from_buffer(data)
	if mime == "application/zip":
		z = zipfile.ZipFile(io.BytesIO(data), compression=zipfile.ZIP_DEFLATED, strict_timestamps=False)
		filenames = [f.filename for f in z.filelist if not f.is_dir()]
		return ImageSequence.fromiter((Image.open(z.open(fn)) for fn in filenames), frameprops=(len(filenames), len(filenames) / 30, 30))
	try:
		import wand
		import wand.image
	except ImportError:
		wand = None
	dur = None
	fcount = None
	proc = None
	try:
		left, right = mime.split("/", 1)[0], mime.split("/", 1)[-1]
		if not wand or left == "image" and right in "apng avif blp bmp cur dcx dds dib emf eps fits flc fli fpx ftex gbr gd heif heic icns ico im imt iptc jpeg jpg mcidas mic mpo msp naa pcd pcx pixar png ppm psd sgi sun spider tga tiff wal wmf xbm".split():
			try:
				im = Image.open(out)
			except PIL.UnidentifiedImageError:
				if not b:
					raise FileNotFoundError("image file not found")
				out.seek(0)
			else:
				try:
					im.seek(1)
				except EOFError:
					im.seek(0)
				else:
					im.seek(0)
					if not im.info.get("duration"):
						ts = time.time_ns() // 1000
						if mime == "video/m3u8" and orig:
							fn = orig
						else:
							fn = "cache/" + str(ts)
							with open(fn, "wb") as f:
								f.write(data)
						cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,duration", "-show_entries", "format=duration", "-of", "csv=s=x:p=0", fn)
						print(cmd)
						p = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
						res = as_str(p.stdout.read()).strip()
						print("RES:", res)
						if res:
							r0, r1, r2, *_ = res.replace("x", "\n").splitlines()
							dur = None
							try:
								dur = float(r1)
							except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
								try:
									dur = float(r2)
								except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
									pass
							if dur:
								im.info["total_duration"] = dur * 1000
								print("TD:", im.info["total_duration"])
							elif r0 != "N/A":
								try:
									fps = float(fractions.Fraction(r0))
								except (ZeroDivisionError, OverflowError):
									pass
								else:
									i = 0
									for i in itertools.count(1):
										try:
											im.seek(i)
										except EOFError:
											break
									im.seek(0)
									im.info["total_duration"] = i / fps * 1000
									print("TD:", im.info["total_duration"])
				return im
		if left in ("image", "video") and right != "webp":
			fmt = "rgba" if left == "image" else "rgb24"
			ts = time.time_ns() // 1000
			if mime == "video/m3u8" and orig:
				fn = orig
			else:
				fn = "cache/" + str(ts)
				with open(fn, "wb") as f:
					f.write(data)
			cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames", "-show_entries", "format=duration", "-of", "csv=s=x:p=0", fn)
			print(cmd)
			p = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			cmd2 = ["ffmpeg", "-nostdin", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-y"]
			if nogif:
				cmd2.extend(("-to", "1"))
			cmd2 += ["-i", fn, "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0"]
			if nogif:
				cmd2.extend(("-vframes", "1"))
			cmd2.append("-")
			print(cmd2)
			proc = psutil.Popen(cmd2, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None, bufsize=64 * 1048576)
			mode = "RGBA" if fmt == "rgba" else "RGB"
			try:
				res = as_str(p.stdout.read()).strip()
				if not res:
					raise TypeError(f'Filetype "{mime}" is not supported.')
				r1, r2, *_ = res.splitlines()
				info = r1.split("x", 4)
			except:
				print(as_str(p.stderr.read()), end="")
				raise
			print("Image info:", info)
			size = tuple(map(int, info[:2]))
			if info[3] == "N/A":
				info[3] = r2
			try:
				dur = float(info[3])
			except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
				dur = 0
			if not size[0] or not size[1] or not dur:
				raise InterruptedError(info)
			if info[2] == "N/A":
				fps = 0
			else:
				try:
					fps = float(fractions.Fraction(info[2]))
				except (ZeroDivisionError, OverflowError):
					fps = 0
			try:
				fcount = int(info[4])
			except ValueError:
				fcount = inf
			if not fps:
				if isfinite(fcount):
					fps = fcount / dur
				else:
					fps = 30
			framedur = 1000 / fps
			bcount = 4 if fmt == "rgba" else 3
			bcount *= int(np.prod(size))
			bytecount = bcount * dur * fps
			if fps * dur > maxframes:
				proc.terminate()
				fcount = floor(maxframes / dur / fps * fcount)
				fps = maxframes / dur
				framedur = 1000 / fps
				cmd3 = ["ffmpeg", "-nostdin", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-y", "-i", fn, "-vf", f"fps=fps={fps}", "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0", "-"]
				print(cmd3)
				proc = psutil.Popen(cmd3, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None, bufsize=64 * 1048576)
			elif dur > 1 and msize and size[0] * size[1] > msize ** 2:
				proc.terminate()
				w, h = max_size(*size, maxsize=msize)
				w = round(w / 2) * 2
				h = round(h / 2) * 2
				cmd3 = ["ffmpeg", "-nostdin", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-y", "-i", fn, "-vf", f"scale={w}:{h}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0", "-"]
				print(cmd3)
				proc = psutil.Popen(cmd3, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None, bufsize=64 * 1048576)
			elif not nogif and bytecount > 32 * 1073741824:
				proc.terminate()
				scale = sqrt((32 * 1073741824) / bytecount)
				fcount = floor(max(8, fps * scale) / fps * fcount) if isfinite(fcount) else fcount
				fps = max(8, fps * scale)
				w, h = round(size[0] * scale), round(size[1] * scale)
				size = (w, h)
				framedur = 1000 / fps
				bcount = 4 if fmt == "rgba" else 3
				bcount *= int(np.prod(size))
				cmd3 = ["ffmpeg", "-nostdin", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-y", "-i", fn, "-vf", f"fps=fps={fps},scale={w}:{h}:flags=bicublin", "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0", "-"]
				print(cmd3)
				proc = psutil.Popen(cmd3, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None, bufsize=64 * 1048576)
			if nogif:
				fcount = dur = fps = 1
			assert fcount >= 1, f"Invalid frame count: {fcount}"
			if not isfinite(fcount) or fcount < 3 or bcount * fcount < 268435456 or len(b) < 16777216:
				print("Decoding directly:", proc, mode, size, (fcount, dur, fps))
				images = deque()
				while True:
					b = proc.stdout.read(bcount)
					if len(b) < bcount:
						break
					img = Image.frombuffer(mode, size, b)
					img.info["duration"] = framedur
					images.append(img)
				if images:
					proc.wait(timeout=2)
					return ImageSequence.open(*images)
				raise RuntimeError(as_str(proc.stderr.read()))
			print("Decoding lazily:", proc, mode, size, (fcount, dur, fps))
			return ImageSequence.pipe(proc.stdout, mode, size, (fcount, dur, fps), close=proc.terminate)
	except Exception as ex:
		print(repr(ex))
		exc = ex
		if proc and proc.is_running():
			proc.terminate()
	else:
		exc = TypeError(f'Filetype "{mime}" is not supported.')
	if not wand or right in ("ico", "x-icon"):
		ib = io.BytesIO(b)
		return Image.open(ib)
	def wand_iter():
		nonlocal exc
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
			tps = img.ticks_per_second
			frames = []
			for frame in wand.sequence.Sequence(img):
				if not frame:
					continue
				with wand.image.Image(frame) as fi:
					ib = io.BytesIO(fi.make_blob("png32"))
				im = Image.open(ib)
				im.info["duration"] = 1000 * frame.delay / tps
				frames.append(im)
				yield im
			if not frames:
				with wand.image.Image(img) as fi:
					ib = io.BytesIO(fi.make_blob("png32"))
				im = Image.open(ib)
				frames.append(im)
				yield im
	if fcount and dur:
		frameprops = (fcount, dur, fcount / dur)
		return ImageSequence.fromiter(wand_iter(), frameprops=frameprops)
	return ImageSequence.open(*wand_iter())

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

discord_emoji = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/assets\\/[0-9A-Fa-f]+\\.svg")
def is_discord_emoji(url):
	return discord_emoji.search(url)

CACHE = {}
def get_image(url, out=None, nodel=False, nogif=False, maxframes=inf, msize=None, cache=False):
	if isinstance(url, Image.Image):
		return url
	out = out or url
	if type(url) not in (bytes, bytearray, io.BytesIO):
		save = None
		if is_url(url):
			if cache and url in CACHE:
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
				if is_discord_emoji(url) and not os.path.exists(save):
					with open(save, "wb") as f:
						f.write(data)
			if len(data) > 8589934592:
				raise OverflowError("Max file size to load is 8GB.")
			image = from_bytes(data, save, nogif=nogif, maxframes=maxframes, orig=url, msize=msize)
			if cache:
				CACHE[url] = image
		else:
			if os.path.getsize(url) > 8589934592:
				raise OverflowError("Max file size to load is 8GB.")
			with open(url, "rb") as f:
				data = f.read()
			image = from_bytes(data, save, nogif=nogif, maxframes=maxframes, msize=msize)
	else:
		if len(url) > 8589934592:
			raise OverflowError("Max file size to load is 8GB.")
		image = from_bytes(url, maxframes=maxframes, msize=msize)
	return image


def load_mimes():
	with open("misc/mimes.txt") as f:
		mimedata = f.read().splitlines()
		globals()["mimesplitter"] = mimesplitter = {}
		def hex2bytes(b):
			s = as_str(b).replace(" ", "")
			if len(s) & 1:
				s = s[:-1] + "0" + s[-1]
			return bytes.hex(s)
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
	mimesplitter = globals()["mimesplitter"]
	for k, v in reversed(mimesplitter.items()):
		out = v.get(b[:k])
		if out:
			return out[mime]
	try:
		b.decode("utf-8")
	except UnicodeDecodeError:
		return "application/octet-stream" if mime else "bin"
	return "text/plain" if mime else "txt"

def from_file(path, mime=True):
	"Detects mimetype of file or buffer. Includes custom .jar, .ecdc, .m3u8 detection."
	path = filetype.get_bytes(path)
	if mime:
		out = filetype.guess_mime(path)
	else:
		out = filetype.guess_extension(path)
	if out and out.split("/", 1)[-1] == "zip" and isinstance(path, str) and path.endswith(".jar"):
		return "application/java-archive"
	if not out:
		if not isinstance(path, bytes):
			if isinstance(path, str):
				raise TypeError(path)
			path = bytes(path)
		out = simple_mimes(path, mime)
	if out == "application/octet-stream" and path.startswith(b'ECDC'):
		return "audio/ecdc"
	if out == "application/octet-stream" and path.startswith(b'\x00\x00\x00,ftypavis'):
		return "image/avif"
	if out == "text/plain" and path.startswith(b"#EXTM3U"):
		return "video/m3u8"
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
def is_url(url):
	return url_match.search(url)
discord_match = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/")
def is_discord_url(url):
	return discord_match.findall(url)

fcache = "cache" if os.path.exists("cache") else "../cache"

def header():
	return {
		"DNT": "1",
		"user-agent": f"Mozilla/5.{(time.time_ns() // 1000) % 10}",
	}

def get_request(url, return_headers=False):
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
			for fn in os.listdir(f"{fcache}/attachments"):
				if fn.split("_", 1)[-1].split(".", 1)[0] == str(a_id):
					with open(f"{fcache}/attachments/{fn}", "rb") as f:
						print(f"Attachment {a_id} loaded from cache.")
						return f.read()
	with requests.get(url, headers=header(), stream=True, verify=False, timeout=12) as resp:
		if return_headers:
			return resp.content, resp.headers
		return resp.content


def from_colour(colour, size=128, key=None):
	return Image.new("RGB" + "A" * (len(colour) > 3), (size, size), tuple(colour))

def from_gradient(shape, colour, background, repetitions=1, size=960, colourspace="rgb", output="pil"):
	"Generates an image from a gradient."
	if len(colour) > 3 or len(background) > 3:
		if len(colour) < 4:
			colour = (*colour, background[3])
		elif len(background) < 4:
			background = (*background, colour[3])
		dim = 4
	else:
		dim = 3
	oob = repetitions > 1
	dtype = np.float32
	col = np.array(colour, dtype=dtype).reshape((dim, 1, 1))
	bg = np.array(background, dtype=dtype).reshape((dim, 1, 1))
	col *= 1 / 255
	bg *= 1 / 255
	if shape == "linear":
		data = np.linspace(0, repetitions, num=size, dtype=dtype)
		data = np.tile(data, (size, 1))
	elif shape == "radial":
		data = np.linspace(-1 / sqrt(2), 1 / sqrt(2), num=size, dtype=dtype)
		data **= 2
		data = np.tile(data, (size, 1))
		data += data.T
		np.sqrt(data, out=data)
		np.subtract(1, data, out=data)
		if repetitions != 1:
			data *= repetitions
	elif shape == "conical":
		mid = (size - 1) / 2
		row = np.arange(size, dtype=dtype)
		row -= mid
		data = np.stack([np.arctan2(i - mid, row) for i in range(size)])
		data *= repetitions / tau
		oob = True
	elif shape == "spiral":
		y, x = np.ogrid[:size, :size]
		center_y, center_x = (size - 1) / 2, (size - 1) / 2
		y, x = y.astype(np.float32) - center_y, x.astype(np.float32) - center_x
		r = np.hypot(x, y) * (2 * sqrt(abs(repetitions)) / size)
		theta = np.arctan2(y, x)
		data = r + theta * (repetitions / 2 / pi)
		oob = True
	elif shape == "polygon":
		angles = np.linspace(0, 2 * pi, repetitions, endpoint=False)
		vertices = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
		mid = (size - 1) / 2
		y, x = np.ogrid[:size, :size]
		y, x = y.astype(np.float32) - mid, x.astype(np.float32) - mid
		data = np.zeros((size, size), dtype=dtype)
		for i in range(size):
			for j in range(size):
				px, py = x[j, i], y[j, i]
				distances = np.linalg.norm(vertices - np.array([px, py]), axis=1)
				data[i, j] = np.min(distances)
		return data / np.max(data)
	else:
		raise NotImplementedError(shape)
	if oob:
		data %= 1
	grad1 = np.tile(data, (dim, 1, 1))
	grad2 = 1 - grad1
	if not np.all(col == (1, 1, 1)):
		grad1 *= col
	if not np.all(bg == (1, 1, 1)):
		grad2 *= bg
	data = np.add(grad1, grad2, out=grad1).swapaxes(0, 2)
	data = np.ascontiguousarray(data)
	if output != "pil":
		return data
	if colourspace.casefold() != "rgb":
		data = convert_colour(data, colourspace, "rgb")
	data *= 255
	grad = quantise_into(data)
	im = fromarray(grad, mode="RGBA" if dim == 4 else "RGB")
	return im

def gen_bg(size):
	rb = np.zeros(size[::-1], dtype=np.uint8)
	rb[::2, ::2] = 255
	rb[1::2, 1::2] = 255
	r = b = fromarray(rb)
	g = Image.new("L", size)
	a = Image.new("L", size, color=255)
	return Image.merge("RGBA", (r, g, b, a))

def quantise_into(a, clip=None, in_place=False, checkerboard=True, dtype=np.uint8):
	if issubclass(a.dtype.type, np.integer):
		return a
	if checkerboard:
		inds = np.indices(a.shape, dtype=np.uint8).sum(axis=0, dtype=np.uint8)
		inds &= 1
		inds = inds.view(bool)
	z = np.random.random_sample(a.shape)
	if checkerboard:
		z %= 0.5
	else:
		z %= 1 - 2 ** 12
	a = np.add(a, z, out=a if in_place else None)
	if checkerboard:
		a[inds] += 0.5 - 2 ** -12
	if clip:
		a = np.clip(a, *clip, out=a)
	if issubclass(getattr(dtype, "type", dtype), np.floating):
		np.floor(a, out=a)
	return np.asanyarray(a, dtype=dtype)

def detect_c2pa(url):
	if is_url(url):
		b, h = get_request(url, return_headers=True)
		mime = h.get("content-type") or "png"
		fmt = mime.split("/", 1)[-1]
		fn = f"{time.time_ns()}.{fmt}"
		with open(fn, "wb") as f:
			f.write(b)
	else:
		fn = url
	try:
		import c2pa
		resp = c2pa.read_file(fn, "cache")
		if not resp:
			raise StopIteration
		resp = orjson.loads(resp)
	except Exception:
		return
	return resp["manifests"][resp["active_manifest"]]["signature_info"]["issuer"]

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

def has_transparency(image):
	"Checks if a palette image has transparency. We assume that an image has transparency if it has a transparency index, or an even number of channels (indicating modes LA or RGBA)."
	assert image.mode == "P", "Expected a palette image."
	transparent = image.info.get("transparency", -1)
	if transparent != -1:
		return True
	for tup in image.getcolors():
		if len(tup) in (2, 4):
			alpha = tup[-1]
			# Introduce a slight fuzzy check; an alpha value of 254 is considered opaque.
			if alpha < 254:
				return True
	return False
def target_p(image):
	return ("RGBA" if has_transparency(image) else "RGB") if image.mode == "P" else image.mode
def remove_p(image):
	"Removes the palette from an image, accounting for transparency."
	if image.mode == "P":
		mode = "RGBA" if has_transparency(image) else "RGB"
		return image.convert(mode)
	return image

def optimise(im, keep_rgb=True, recurse=True, max_frames=60):
	"""
	Optimizes the given image or sequence of images by converting their modes based on certain conditions.

	Parameters:
	im (PIL.Image or iterable): The image or sequence of images to be optimized.
	keep_rgb (bool, optional): If True, keeps the image in RGB mode if possible. Defaults to True.
	recurse (bool, optional): If True, allows recursion for sequences of images. Defaults to True.
	max_frames (int, optional): The maximum number of frames to process in a sequence. Defaults to 60.

	Returns:
	PIL.Image or list: The optimized image or list of optimized images.

	Raises:
	TypeError: If the input is not an image or a sequence of images, or if the image is a dictionary.
	"""
	try:
		if not recurse:
			raise TypeError
		it = iter(im)
	except TypeError:
		pass
	else:
		if not im:
			return im
		try:
			i0 = next(it)
			if type(i0) is type(im):
				raise StopIteration
		except StopIteration:
			return []
		i0 = optimise(i0, keep_rgb=keep_rgb, recurse=False)
		out = [i0]
		orig = []
		mode = i0.mode
		changed = False
		for i, i2 in enumerate(it):
			if i >= max_frames and not changed:
				print("Unchanged:", mode, i0, i2)
				return resume(i0, out, it)
			orig.append(i2)
			if i2.mode != mode:
				changed = True
				i2 = optimise(i2, keep_rgb=keep_rgb, recurse=False)
				if i2.mode != mode:
					return [im.convert(i2.mode) for im in resume(i0, orig, it)]
			out.append(i2)
		return out
	if isinstance(im, dict):
		raise TypeError(im)
	original = im.mode
	try:
		if im.mode == "P":
			if keep_rgb:
				im = remove_p(im)
			else:
				return im
		if im.mode == "L":
			if keep_rgb:
				return (im := im.convert("RGB"))
			return im
		if im.mode == "LA":
			A = im.getchannel("A")
			if np.min(A) >= 254:
				return (im := im.convert("L"))
			return im
		if im.mode == "RGBA":
			if keep_rgb:
				A = im.getchannel("A")
				if np.min(A) >= 254:
					return (im := im.convert("RGB"))
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
						return (im := im.convert("LA"))
			if np.min(A) >= 254:
				return (im := im.convert("RGB"))
			return im
		if keep_rgb:
			if im.mode != "RGB":
				return (im := im.convert("RGB"))
			return im
		R, G, B = im.split()
		r, g, b = np.asarray(R, dtype=np.uint8), np.asarray(G, dtype=np.uint8), np.asarray(B, dtype=np.uint8)
		distRG = np.abs(r.ravel() - g.ravel())
		if np.max(distRG) <= 2:
			distGB = np.abs(g.ravel() - b.ravel())
			if np.max(distGB) <= 2:
				distBR = np.abs(b.ravel() - r.ravel())
				if np.max(distBR) <= 2:
					return (im := im.convert("L"))
		return im
	finally:
		print("OP:", original, im.mode)

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
	# print("RM:", image, maxsize, force, resample, box, reducing_gap)
	w, h = max_size(image.width, image.height, maxsize, force=force)
	if w != image.width or h != image.height:
		if type(resample) is str:
			image = resize_to(image, w, h, resample)
		else:
			image = image.resize([w, h], resample, box, reducing_gap)
	if force > 1:
		return image.copy()
	return image

def downsample(im, lenience=5, maxsize=16384, minsize=48, keep_alpha=True):
	if im.width <= minsize or im.height <= minsize:
		return im
	cache = {}
	es = {}
	A = None
	if im.mode != "RGB":
		if "RGB" not in im.mode:
			im = im.convert("RGBA")
		if keep_alpha and "A" in im.mode:
			A = im.getchannel("A")
		im = im.convert("RGB")
	im3 = im
	thresh = 2 ** (lenience + 4)

	def roundown(x):
		consts = range(2, int(sqrt(x + 1)))
		cs = [ceil(log2(x / c) - 1) for c in consts]
		return sorted((c * 2 ** n for c, n in zip(consts, cs)), key=lambda y: (x >= y, abs(x - y)))[0]

	cache[1] = w, h = lx, ly = im.size
	Me = e = im3.entropy()
	es[1] = e
	ex = 0
	r = sqrt(max(2, w * h / maxsize / maxsize))
	mr = 1
	for i in range(64):
		if w > h:
			x = roundown(min(lx - 1, w / r))
			y = round(h / w * x)
		else:
			y = roundown(min(ly - 1, h / r))
			x = round(w / h * y)
		lx, ly = x, y
		if x < minsize or y < minsize:
			break
		im2 = im3.resize((x, y), resample=Image.Resampling.BILINEAR)
		im4 = im2.resize((w, h), resample=Image.Resampling.NEAREST)
		im4 = ImageChops.difference(im4, im3)
		ex = im4.entropy()
		if ex > lenience:
			break
		cache[r] = im2.size
		e = im2.entropy()
		es[r] = e
		Me = max(Me, e)
		mr = min(mr, e / Me)
		if mr < 1 - 1 / thresh / 1.5 or ex > lenience - 1:
			r *= 2 ** 0.125
		elif mr < 1 - 1 / thresh / 2 or ex > lenience - 2:
			r *= 2 ** 0.25
		else:
			r *=  2 ** 0.5

	for k in sorted(es, reverse=True):
		v = es[k]
		if v > Me * (1 - 1 / thresh):
			break
	size = max_size(*cache[k], maxsize=maxsize)
	if im.size != size:
		im2 = im.resize(size, resample=Image.Resampling.LANCZOS)
		im3 = im.resize(size, resample=Image.Resampling.NEAREST)
		d2 = ImageChops.difference(im2.resize(im.size, resample=Image.Resampling.LANCZOS), im)
		d3 = ImageChops.difference(im3.resize(im.size, resample=Image.Resampling.LANCZOS), im)
		if d2.entropy() > d3.entropy():
			out = im3
		else:
			out = im2
	else:
		out = im
	if A:
		if A.size != out.size:
			A = A.resize(out.size, resample=Image.Resampling.LANCZOS)
		out.putalpha(A)
	print(im, out)
	return out

def image_to(im, mode="RGB", size=None):
	im = im if im.mode == mode else im.convert(mode)
	if size and size != im.size:
		im = im.resize(size, resample=Image.Resampling.LANCZOS)
	return im

def pil2pyg(image):
	global pygame
	try:
		import pygame
	except ImportError:
		pygame = None
		raise
	b = np.asanyarray(image, dtype=np.uint8).data
	return pygame.image.frombuffer(b, image.size, image.mode)

def pyg2pil(surf):
	mode = "RGBA" if surf.get_flags() & pygame.SRCALPHA else "RGB"
	b = surf.get_buffer()
	return Image.frombuffer(mode, surf.get_size(), b)

resizers = dict(
	sinc="lanczos",
	lanczos="lanczos",
	cubic="cubic",
	spline="cubic",
	scale2x="scale2x",
	hamming=Resampling.HAMMING,
	gauss=Resampling.HAMMING,
	linear="linear",
	nearest=Resampling.NEAREST,
	nearest_neighbour=Resampling.NEAREST,
	area="area",
	crop="crop",
	padding="crop",
	sdxl=Resampling.LANCZOS,
)

def resize_mult(image, x, y, mode="auto"):
	if y == "-":
		y = x
	if x == y == 1:
		return image
	w = round(image.width * x)
	h = round(image.height * y)
	return resize_to(image, w, h, mode)

def resize_to(image, w, h, mode="auto"):
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
	if w < 0:
		w = -w
		image = ImageOps.mirror(image)
	if h < 0:
		h = -h
		image = ImageOps.flip(image)
	op = mode.casefold().replace(" ", "").replace("_", "")
	if op in resizers:
		filt = resizers[op]
	elif op == "auto":
		filt = "area" if image.width * image.height > w * h else Resampling.BOX if image.width * image.height <= 16384 else "cubic" if w * h > 1048576 else "lanczos"
	else:
		raise TypeError(f'Invalid image operation: "{op}"')
	print(image, w, h, mode, filt)
	if filt != Resampling.NEAREST:
		image = remove_p(image)
	if filt == "scale2x":
		if w > image.width or h > image.height:
			surf = pil2pyg(image)
			while w > surf.get_width() or h > surf.get_height():
				surf = pygame.transform.scale2x(surf)
			image = pyg2pil(surf)
		filt = Resampling.NEAREST
	elif filt == "crop":
		out = Image.new(image.mode, (w, h), (0,) * len(image.mode))
		pos = tuple(np.array((w, h)) - image.size >> 1)
		out.paste(image, pos)
		image = out
	image = optimise(image)
	if image.size == (w, h):
		return image
	if filt == "lanczos":
		if max(w * h, np.prod(image.size)) > 1048576:
			filt = Resampling.LANCZOS
		else:
			a = np.array(image, dtype=np.float32)
			a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)
			a = quantise_into(a, clip=(0, 255), in_place=True)
			return Image.fromarray(a, image.mode)
	if filt == "area":
		if max(w * h, np.prod(image.size)) > 1048576:
			filt = Resampling.LANCZOS
		else:
			a = np.array(image, dtype=np.float32)
			a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
			a = quantise_into(a, in_place=True)
			return Image.fromarray(a, image.mode)
	if filt == "cubic":
		if max(w * h, np.prod(image.size)) > 1048576:
			filt = Resampling.BICUBIC
		else:
			a = np.array(image, dtype=np.float32)
			a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
			a = quantise_into(a, clip=(0, 255), in_place=True)
			return Image.fromarray(a, image.mode)
	if filt == "linear":
		if max(w * h, np.prod(image.size)) > 1048576:
			filt = Resampling.BILINEAR
		else:
			a = np.array(image, dtype=np.float32)
			a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
			a = quantise_into(a, in_place=True)
			return Image.fromarray(a, image.mode)
	return image.resize([w, h], filt)

def rotate_to(image, angle, expand=True, fill=False):
	angle %= 360
	if not angle % 90:
		if angle == 90:
			return image.transpose(Transpose.ROTATE_90)
		elif angle == 180:
			return image.transpose(Transpose.ROTATE_180)
		elif angle == 270:
			return image.transpose(Transpose.ROTATE_270)
		return image
	if fill and image.mode != "RGBA":
		image = image.convert("RGBA")
	im = image.rotate(angle, resample=Resampling.BICUBIC, expand=expand)
	if fill and im.mode == "RGBA":
		colour = get_colour(image)
		a = np.tile(np.float32(colour), (im.height, im.width, 1))
		a = quantise_into(a, in_place=True)
		image = Image.fromarray(a, "RGB").convert("RGBA")
		return Image.alpha_composite(image, im)
	return im

ANIM = False
SWIRL = None

def to_qr(s, repetitions=3, duration=4.8, fps=30):
	global SWIRL
	if type(s) is str:
		s = s.encode("utf-8")
	size = len(s)
	if size <= 1273:
		err = "H"
	elif size <= 1663:
		err = "Q"
	elif size <= 2331:
		err = "M"
	else:
		if size > 2953:
			raise OverflowError("Input size too large.")
		err = "L"
	ver = None
	import pyqrcode
	img = pyqrcode.create(s, error=err, version=ver, mode=None, encoding="utf-8")
	b = io.BytesIO()
	img.png(b, scale=1, module_color=(255,) * 3, background=(0,) * 4)
	b.seek(0)
	imo = Image.open(b)
	im = imo.convert("1")
	im = im.resize((512, 512), resample=Resampling.NEAREST)
	if abs(duration) > 0 and fps > 0:
		if SWIRL is None:
			SWIRL = from_gradient("spiral", (0, 255, 192), (255, 255, 192), repetitions=repetitions, size=512, colourspace="hsl", output="raw")
		count = round(duration * fps)

		def qr_iterator(image):
			bg = SWIRL.copy()
			fg = SWIRL.copy()
			fg.T[2] = 64 / 255
			mask = np.nonzero(image)
			for i in range(count):
				fg.T[0] = SWIRL.T[0] + i / count
				bg.T[0] = SWIRL.T[0].T - i / count
				bg[mask] = fg[mask]
				data = convert_colour(bg, "hsl", "rgb", in_place=False)
				data *= 255
				grad = quantise_into(data)
				yield fromarray(grad, mode="RGB")

		return dict(duration=duration * 1000, count=count, frames=qr_iterator(im))
	return ImageChops.invert(im).convert("RGBA")


def properties(im) -> tuple: # frames, duration, fps
	try:
		if im.frameprops:
			return im.frameprops
	except AttributeError:
		pass
	total_duration = im.info.get("total_duration", 0)
	duration = total_duration
	for f in range(2147483648):
		try:
			im.seek(f)
		except EOFError:
			break
		if not total_duration:
			duration += max(im.info.get("duration", 50), 1)
	duration = duration or 50
	while duration < 20:
		duration *= 2
	fps = f / duration * 1000
	props = (max(1, f), duration / 1000, fps)
	print("PROPS:", props)
	return props

def sync_fps(props, duration=None, fps=None):
	d = max(t[1] for t in props)
	prog = 1
	if duration and duration < 0:
		duration = -duration
		prog = -prog
	if fps and fps < 0:
		fps = -fps
		prog = -prog
	seconds = d * ceil(duration / d) if duration is not None else d
	fps = fps if fps is not None else max(t[2] for t in props)
	print("SYNC:", props, seconds, fps, prog)
	return seconds, fps, prog

def map_sync(images, *args, func, duration=None, fps=None, keep_size="approx", retrieve=False, **kwargs):
	"""
	Synchronizes and maps a function over a sequence of images.
	The images may be static (repeated if necessary), or animated, in which case a heuristically determined set of frames will be used.
	Args:
		images (list): List of image URLs or image objects.
		*args: Additional positional arguments to pass to the mapping function.
		func (callable): Function to apply to the synchronized images.
		duration (float, optional): Duration in seconds for the synchronization. Defaults to None.
		fps (int, optional): Frames per second for the synchronization. Defaults to None.
		keep_size (str, optional): Determines how to handle image sizes. Options are "exact", "approx", or None. Defaults to "approx".
		retrieve (bool, optional): If True, retrieves images from URLs. Defaults to False.
		**kwargs: Additional keyword arguments to pass to the mapping function.
	Returns:
		dict: A dictionary containing:
			- 'duration' (int): Duration in milliseconds.
			- 'count' (int): Number of frames.
			- 'frames' (generator): Generator yielding the mapped frames.
	"""
	if retrieve:
		sources = [get_image(url) for url in images]
	else:
		sources = images
	props = [properties(im) for im in sources]
	seconds, fps, prog = sync_fps(props, duration, fps)
	count = max(1, round(fps * seconds))
	if duration:
		prog *= max(1, round(seconds / duration))
	seed = time.time_ns() // 1000

	if keep_size == "exact":
		others = [ImageSequence.cast(resize_map(im, (), duration, fps, "set", sources[0].width, sources[0].height, mode="auto")) for im in sources[1:]]
	elif keep_size == "approx":
		width = max(source.width for source in sources)
		height = max(source.height for source in sources)
		diameter = sqrt(width * height)
		others = [ImageSequence.cast(resize_map(im, (), duration, fps, "set", *max_size(im.width, im.height, maxsize=diameter, force=True), mode="auto")) for im in sources[1:]]
	elif keep_size:
		raise NotImplementedError(keep_size)
	else:
		others = sources[1:]
	mapped_sources = (sources[0], *others)
	def map_iter():
		for i in range(count):
			ims = []
			for im, prop in zip(mapped_sources, props):
				maxframes = prop[0]
				mult = round(seconds / prop[1])
				n = floor(i / count * mult * maxframes)
				try:
					im.seek(n % maxframes)
				except Exception:
					print(f"Seek error: {n}, {n % maxframes}, {maxframes}")
					print(f"Current frame: {i}, {props}, {mapped_sources}")
					raise
				ims.append(im)
			yield func(ims, *args, props=props, progress=prog * i / count % 1 if count > 1 else 1, count=count, seed=seed, **kwargs)

	return dict(duration=1000 * seconds, count=count, frames=map_iter())

def sync_animations(func, keep_size="approx"):
	def sync_into(image, extras, duration=None, fps=None, *args, **kwargs):
		return map_sync([image, *extras], *args, func=func, duration=duration, fps=fps, retrieve=True, keep_size=keep_size, **kwargs)
	return sync_into


@sync_animations
def spin_map(images, angle, circle, progress=0, **kwargs):
	image = images[0]
	z = progress * 360 + angle % 360
	if z:
		image = rotate_to(image, z, expand=False, fill=not circle)
	if not circle:
		return image
	return to_circle(image)

@sync_animations
def rainbow_map(images, mode, progress=0, **kwargs):
	image = images[0]
	progress %= 1
	if not progress:
		return image
	im, A = split_rgba(image)
	rgb = np.array(im, dtype=np.float32)
	rgb *= 1 / 255
	space = convert_colour(rgb, "rgb", mode)
	space.T[0] += progress
	rgb = convert_colour(space, mode, "rgb")
	rgb *= 255
	rgb = quantise_into(rgb, in_place=True)
	im = fromarray(rgb, "RGB")
	return join_rgba(im, A)

@sync_animations
def pet_map(images, squeeze, progress=0, **kwargs):
	image = images[0]
	pet = get_image("https://mizabot.xyz/u/2omMy8VUGJ5GOH3AIJxw3wJ5yQrU/EBjYrqCEUDw.zip", cache=True)
	w, h = image.width * 2.5, image.height * 2.5
	if w < 256 and h < 256:
		w, h = max_size(w, h, 256, force=True)
	w, h = round_random(w), round_random(h)
	im = Image.new("RGBA", (w, h))
	sqr = (1 - cos(progress * tau)) * 2.5
	wm = 0.8 + sqr * 0.02
	hm = 0.8 - sqr * 0.05
	ox = (1 - wm) * 0.5 + 0.1
	oy = (1 - hm) - 0.08
	im.paste(image.resize((round_random(wm * w), round_random(hm * h)), resample=Resampling.LANCZOS), (round_random(ox * w), round_random(oy * h)))
	i = floor(progress * len(pet))
	pet.seek(i)
	pet2 = pet.resize((w, h), resample=Resampling.LANCZOS)
	im.paste(pet2, mask=pet2)
	return im.resize((round(w / 2), round(h / 2)), resample=Resampling.LANCZOS)

@sync_animations
def scroll_map(images, direction, progress=0, **kwargs):
	image = images[0]
	progress %= 1
	if not progress:
		return image
	if direction == "left":
		x = -1
		y = 0
	elif direction == "up":
		x = 0
		y = -1
	elif direction == "right":
		x = 1
		y = 0
	elif direction == "down":
		x = 0
		y = 1
	else:
		raise TypeError(f"Invalid direction {direction}")
	xm = round(x * image.width * progress)
	ym = round(y * image.width * progress)
	return ImageChops.offset(image, xm, ym)

def resize_map(image, extras, duration, fps, operation, x, y, mode="auto", area=None, **kwargs):
	"""
	Resize an image or a sequence of images based on the specified parameters.
	Parameters:
	image (PIL.Image.Image): The image to be resized.
	extras (list): Additional images to be processed.
	duration (int): Duration of the output in milliseconds.
	fps (int): Frames per second for the output.
	operation (str): The type of resizing operation ('rel', 'max', 'mult', 'set').
	x (int or float): The target width or scaling factor.
	y (int or float): The target height or scaling factor.
	mode (str, optional): The resizing mode ('auto', 'nearest', 'linear', 'cubic', 'area', 'hamming', 'lanczos', 'crop'). Default is 'auto'.
	area (int, optional): The target area for resizing. Default is None.
	**kwargs: Additional keyword arguments.
	Returns:
	PIL.Image.Image or dict: The resized image or a dictionary containing the resized frames and metadata if processing a sequence of images.
	Raises:
	RuntimeError: If there is an error during the resizing process.
	"""
	prop = properties(image)
	duration, fps, prog = sync_fps([prop], duration, fps)
	if operation == "rel":
		x, y = max_size(*image.size, maxsize=x, force=True)
		operation = "set"
	elif operation == "max":
		if x is None:
			x, y = image.size
		else:
			x, y = max_size(*image.size, maxsize=x)
		operation = "set"
	elif y == "-":
		if x == "-":
			if operation == "mult":
				x = y = 1
			else:
				x, y = image.size
		else:
			y = x / image.width * image.height
	elif x == "-":
		x = y / image.height * image.width
	if operation == "mult":
		w = image.width * x
		h = image.height * y
	else:
		w, h = x, y
	if area:
		w, h = max_size(w, h, maxsize=sqrt(area), force=True)
	w, h = round(w), round(h)
	if (w, h) == image.size:
		return image
	if mode == "auto":
		mode = "area" if image.width * image.height > w * h else "nearest" if image.width * image.height <= 16384 else "cubic" if w * h > 1048576 else "lanczos"
	maxframes = prop[0]
	maxsize = w * h
	if mode in ("nearest", "linear", "cubic", "area", "hamming", "lanczos", "crop") and maxframes > 1 and maxsize * maxframes > 8388608 or mode in ("spline", "gauss"):
		fmt = "rgb24" if target_p(image) in ("HSV", "RGB") else "rgba"
		if mode == "nearest":
			mode = "neighbor"
		elif mode == "linear":
			mode = "fast_bilinear"
		elif mode == "cubic":
			mode = "bicublin"
		elif mode == "hamming":
			mode = "gauss"
		if mode == "crop":
			vf = f"crop={w}:{h}:(in_w-{w})/2:(in_h-{h})/2"
		else:
			vf = f"scale={w}:{h}:flags={mode}"
		cmd3 = ["ffmpeg", "-hwaccel", hwaccel, "-hide_banner", "-v", "error", "-f", "rawvideo", "-video_size", "x".join(map(str, image.size)), "-pix_fmt", fmt, "-i", "-", "-vf", vf, "-f", "rawvideo", "-pix_fmt", fmt, "-"]
		print(cmd3)
		proc = psutil.Popen(cmd3, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
		def writer():
			for i in range(prop[0]):
				image.seek(i)
				proc.stdin.write(np.asanyarray(remove_p(image), dtype=np.uint8).data)
			proc.stdin.close()
		def resize_bufferer():
			bcount = w * h * (3 if fmt == "rgb24" else 4)
			b = None
			while True:
				b = proc.stdout.read(bcount)
				if len(b) < bcount:
					break
				yield Image.frombuffer("RGB" if fmt == "rgb24" else "RGBA", (w, h), b)
			if b is None:
				raise RuntimeError(as_str(proc.stderr.read()))
		exc.submit(writer)
		buf = resize_bufferer()
		if prog < 0:
			buf = reversed(list(buf))
		return dict(duration=1000 * duration, count=prop[0], frames=buf)

	func = resize_mult if operation == "mult" else resize_to
	return dict(duration=1000 * duration, count=prop[0], frames=map(func, ImageSequence.cast(image), [x] * prop[0], [y] * prop[0], [mode] * prop[0]))

def orbit_map(image, extras, duration, fps, count):
	symmetry = count or (1 if extras else 5)
	sources = [image, *(get_image(url) for url in extras)]
	orbitals = sources * symmetry
	orbital_count = len(orbitals)
	props = [properties(im) for im in sources]
	if any(t[0] > 1 for t in props):
		symmetry = 1
	duration /= symmetry
	diameter = max(max(source.size) for source in sources)
	scale = orbital_count / pi * (sqrt(5) + 1) / 2 + 0.5
	diameter = min(diameter, round(1920 / scale))
	size = (round(diameter * scale),) * 2

	# Repeatedly rotate image and return copies
	def orbit_iterator(images, progress=0, **kwargs):
		im = Image.new("RGBA", size, (0,) * 4)
		if orbital_count > 1:
			im2 = Image.new("RGBA", size, (0,) * 4)
			if orbital_count & 1:
				im3 = Image.new("RGBA", size, (0,) * 4)
		for i, image in enumerate(images):
			image = resize_max(image, diameter, force=True)
			angle = progress * tau * symmetry + i * tau / orbital_count
			pos = im.width / 2 + np.array((cos(angle), sin(angle))) * (diameter * scale / 2 - diameter / 2) - (image.width / 2, image.height / 2)
			pos = list(map(round, pos))
			if i == orbital_count - 1 and orbital_count & 1 and orbital_count > 1:
				im3.paste(image, pos)
			elif not i & 1:
				im.paste(image, pos)
			else:
				im2.paste(image, pos)
		if orbital_count > 1:
			if orbital_count & 1:
				im2 = Image.alpha_composite(im3, im2)
			im = Image.alpha_composite(im, im2)
		return im
	return map_sync(orbitals, func=orbit_iterator, duration=duration, fps=fps, keep_size=None)

def quad_as_rect(quad):
	if quad[0] != quad[2]:
		return False
	if quad[1] != quad[7]:
		return False
	if quad[4] != quad[6]:
		return False
	if quad[3] != quad[5]:
		return False
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
			src_quad = [
				src_grid[i, j, 0], src_grid[i, j, 1],
				src_grid[i + 1, j, 0], src_grid[i + 1, j, 1],
				src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
				src_grid[i, j + 1, 0], src_grid[i, j + 1, 1],
			]
			dst_quad = [
				dst_grid[i, j, 0], dst_grid[i, j, 1],
				dst_grid[i + 1, j, 0], dst_grid[i + 1, j, 1],
				dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
				dst_grid[i, j + 1, 0], dst_grid[i, j + 1, 1],
			]
			dst_rect = quad_to_rect(dst_quad)
			mesh.append([dst_rect, src_quad])
	return list(mesh)

@sync_animations
def magik_map(images, intensity, cell_count, progress=0, count=1, seed=0, **kwargs):
	image = images[0]
	total_distance = sqrt(np.prod(image.size)) / cell_count * intensity
	grid_distance = max(1, round(sqrt(total_distance)))
	iterations = round(total_distance / grid_distance * progress)
	for i in range(iterations):
		np.random.seed(seed + i & 4294967295)
		dst_grid = griddify(shape_to_rect(image.size), cell_count, cell_count)
		src_grid = distort_grid(dst_grid, grid_distance)
		mesh = grid_to_mesh(src_grid, dst_grid)
		image = image.transform(image.size, Transform.MESH, mesh, resample=Resampling.NEAREST)
	return image

@sync_animations
def crop_map(images, x1, y1, x2, y2, **kwargs):
	image = images[0]
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

@sync_animations
def adjust_map(images, operation, value, channels, clip, **kwargs):
	image = images[0]
	channels = set(channels)
	chans = channels.difference("rgbcmyhscvliy")
	if chans:
		raise NotImplementedError(f"Unsupported channel(s) {chans}.")
	dtype = np.float32
	im, A = split_rgba(image)
	if A:
		a = np.array(A, dtype=dtype)
		a *= 1 / 255
	rgb = np.array(im, dtype=dtype)
	rgb *= 1 / 255

	np_operations = {
		"+": np.add,
		"-": np.subtract,
		"*": np.multiply,
		"/": np.true_divide,
		"**": np.power,
		"%": np.mod,
	}
	uf_operations = {
		"&": np.bitwise_and,
		"|": np.bitwise_or,
		"^": np.bitwise_xor,
	}
	def operate_channel(c):
		if operation == "=":
			c[:] = value
		elif operation in np_operations:
			ufunc = np_operations[operation]
			ufunc(c, value, out=c)
		elif operation in uf_operations:
			ufunc = uf_operations[operation]
			c *= 65535
			temp = c.astype(np.uint32)
			ufunc(temp, value * 65535, out=temp)
			c[:] = temp
			c *= 1 / 65535
		else:
			raise NotImplementedError(operation)
		return c
	def map_operation(colours, space):
		if space != "rgb":
			colours = convert_colour(rgb, "rgb", space)
		for i, c in enumerate(space):
			if c in channels:
				operate_channel(colours.T[i])
		if space != "rgb":
			colours = convert_colour(rgb, space, "rgb")
		return colours

	if channels.intersection("rgb"):
		rgb = map_operation(rgb, "rgb")
	if channels.intersection("cmy"):
		rgb = map_operation(rgb, "cmy")
	if channels.intersection("hsv"):
		rgb = map_operation(rgb, "hsv")
	if channels.intersection("l"):
		rgb = map_operation(rgb, "hcl")
	if channels.intersection("cy"):
		rgb = map_operation(rgb, "hcy")
	if channels.intersection("a"):
		if value == 1:
			A = None
		elif A:
			a = operate_channel(a)
		else:
			a = np.full(image.size, value, dtype=dtype)
	if A:
		a *= 255
		if clip:
			bounds = (0, 255)
		else:
			bounds = None
			a %= 256
		a = quantise_into(a, clip=bounds, in_place=True)
		A = fromarray(a, "L")
	rgb *= 255
	if clip:
		bounds = (0, 255)
	else:
		bounds = None
		rgb %= 256
	rgb = quantise_into(rgb, clip=bounds, in_place=True)
	im = fromarray(rgb, "RGB")
	im = join_rgba(im, A)
	return im

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
	"additive": "add",
	"-": "subtract",
	"sub": "subtract",
	"subtract": "subtract",
	"subtraction": "subtract",
	"subtractive": "subtract",
	"*": "multiply",
	"mul": "multiply",
	"mult": "multiply",
	"multiply": "multiply",
	"multiplication": "multiply",
	"multiplicative": "multiply",
	"/": blend_modes.divide,
	"div": blend_modes.divide,
	"divide": blend_modes.divide,
	"division": blend_modes.divide,
	"divisive": blend_modes.divide,
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
def _blend_map(images, operation, opacity, props=(), **kwargs):

	def blend_mut(i1, a1, i2, a2):
		if opacity == 0:
			return i1, a1
		if operation == "blend":
			o1, o2 = min(1, opacity * 2), min(1, (1 - opacity) * 2)
			return (i1 * a1 * o1 + i2 * a2 * o2) / (a1 + a2), np.maximum(a1, a2)
		elif operation == "replace":
			return i1 * opacity + i2 * (1 - opacity), (a1 + a2) / 2
		elif operation == "add":
			return i1 * opacity * 2 + i2 * (1 - opacity) * 2, a1 + a2
		elif operation == "mul":
			return i1 ** (opacity * 2) * i2 ** ((1 - opacity) * 2), a1 * a2
		elif operation == "sub":
			return i1 * opacity - i2 * (1 - opacity), np.maximum(a1, a2)
		elif operation == "div":
			return i1 ** (opacity * 2) / i2 ** ((1 - opacity) * 2), a1 / a2
		else:
			raise NotImplementedError(operation)

	dtype = np.float32
	image = images[0]
	im, A = split_rgba(image)
	rgb = np.array(im, dtype=dtype)
	rgb *= 1 / 255
	if A is not None:
		a1 = np.array(A, dtype=dtype)
		a1 *= 1 / 255
	else:
		a1 = np.ones((im.height, im.width), dtype=dtype)
	a1 = a1.reshape((im.height, im.width, 1))

	for j in range(1, len(images)):
		image2 = images[j]
		im2, A2 = split_rgba(image2)
		rgb2 = np.array(im2, dtype=dtype)
		rgb2 *= 1 / 255
		if A2 is not None:
			a2 = np.array(A2, dtype=dtype)
			a2 *= 1 / 255
		else:
			a2 = np.ones((im.height, im.width), dtype=dtype)
		a2 = a2.reshape((im2.height, im2.width, 1))
		rgb, a1 = blend_mut(rgb, a1, rgb2, a2)
		np.nan_to_num(rgb, copy=False)
		np.nan_to_num(a1, copy=False)

	a1 *= 255
	a1 = quantise_into(a1, clip=(0, 255), in_place=True).reshape(a1.shape[:2])
	A = fromarray(a1, "L")
	rgb *= 255
	rgb = quantise_into(rgb, clip=(0, 255), in_place=True)
	im = fromarray(rgb, "RGB")
	return join_rgba(im, A)
blend_map = sync_animations(_blend_map, keep_size="exact")


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
	if image.mode != "RGBA":
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


def split_rgba(image):
	image = remove_p(image)
	if image.mode == "RGBA":
		a = np.asanyarray(image, dtype=np.uint8)
		aa = a.T[3]
		if np.min(aa) >= 254:
			A = None
		else:
			A = fromarray(aa.T)
		image = fromarray(a.T[:3].T, mode="RGB")
	else:
		A = None
	return image, A

def join_rgba(image, A=None):
	if A is not None and np.min(A) < 254:
		image.putalpha(A)
	return image

range_xyz = ((0, 0.95047), (0, 1), (0, 1.08883))
range_hcl = ((-np.pi, np.pi), (0, np.sqrt(2)), (0, 1))
range_luv = ((0, 100), (-83.07753, 175.01505), (-134.1030, 107.3985))
range_hcy = ((-np.pi, np.pi), (0, 1 / np.sqrt(2)), (0, 1))

def rgb_to_hsi(rgb_image):
	m = np.min(rgb_image, -1).T
	I = np.mean(rgb_image, -1).T # noqa: E741
	hls_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS, dst=rgb_image)
	H, L, S = np.swapaxes(hls_image, 0, 2)
	H[:] *= 1 / 360
	mask = np.nonzero(I)
	L[mask] = 1 - m[mask] / I[mask]
	S[:] = I
	return hls_image

def hsi_to_rgb(hsi_image):
	H, S, I = np.swapaxes(hsi_image, 0, 2) # noqa: E741
	H *= 2 * np.pi
	R = np.zeros_like(I)
	G = np.zeros_like(I)
	B = np.zeros_like(I)
	V = I * 3
	H %= 2 * np.pi
	# RG sector (0 <= H < 2*pi/3)
	mask1 = H < 2 * np.pi / 3
	B[mask1] = I[mask1] * (1 - S[mask1])
	R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H[mask1]) / np.cos(np.pi / 3 - H[mask1]))
	G[mask1] = V[mask1] - (R[mask1] + B[mask1])
	# GB sector (2*pi/3 <= H < 4*pi/3)
	mask2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
	H_adj = H[mask2] - 2 * np.pi / 3
	R[mask2] = I[mask2] * (1 - S[mask2])
	G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H_adj) / np.cos(np.pi / 3 - H_adj))
	B[mask2] = V[mask2] - (R[mask2] + G[mask2])
	# BR sector (4*pi/3 <= H < 2*pi)
	mask3 = H >= 4 * np.pi / 3
	H_adj = H[mask3] - 4 * np.pi/3
	G[mask3] = I[mask3] * (1 - S[mask3])
	B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H_adj) / np.cos(np.pi / 3 - H_adj))
	R[mask3] = V[mask3] - (G[mask3] + B[mask3])
	H[:], S[:], I[:] = R, G, B
	return hsi_image

def luv_to_hcl(luv_image):
	L, U, V = np.swapaxes(luv_image, 0, 2)
	H = np.arctan2(V, U)
	C = np.hypot(U, V)
	H *= 1 / (range_hcl[0][1] - range_hcl[0][0])
	H %= 1
	C *= 1 / range_hcl[1][1]
	V[:] = L
	L[:], U[:] = H, C
	return luv_image

def hcl_to_luv(hcl_image):
	H, C, L = np.swapaxes(hcl_image, 0, 2)
	H *= range_hcl[0][1] - range_hcl[0][0]
	C *= range_hcl[1][1]
	U = C * np.cos(H)
	V = C * np.sin(H)
	H[:] = L
	C[:], L[:] = U, V
	return hcl_image

def ycc_to_hcy(yvu_image):
	Y, V, U = np.swapaxes(yvu_image, 0, 2)
	U -= 0.5
	V -= 0.5
	H = np.arctan2(V, U)
	C = np.hypot(U, V)
	H *= 1 / (range_hcy[0][1] - range_hcy[0][0])
	H %= 1
	C *= 1 / range_hcy[1][1]
	U[:] = Y
	Y[:], V[:] = H, C
	return yvu_image

def hcy_to_ycc(hcy_image):
	H, C, Y = np.swapaxes(hcy_image, 0, 2)
	H *= range_hcy[0][1] - range_hcy[0][0]
	C *= range_hcy[1][1]
	U = C * np.cos(H)
	V = C * np.sin(H)
	U += 0.5
	V += 0.5
	H[:] = Y
	C[:], Y[:] = V, U
	return hcy_image

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
	image, A = split_rgba(image)
	image = ImageOps.invert(image)
	image = join_rgba(image, A)
	return image

def greyscale(image):
	image, A = split_rgba(image)
	image = ImageOps.grayscale(image)
	image = join_rgba(image, A)
	return image

def laplacian(image):
	image = remove_p(image)
	surf = pil2pyg(image)
	surf = pygame.transform.laplacian(surf)
	return pyg2pil(surf)

cv2_swaps = {k: v for k, v in cv2.__dict__.items() if k.startswith("COLOR_")}
# supported formats: RGB, BGR, CMY, XYZ, HSV, HSI, HSL, HCL, HCY, LAB, LUV, YUV
def convert_colour(im, source, dest, in_place=True):
	if not in_place:
		im = np.array(im, dtype=np.float32)
	from_cv2 = dict(
		bgr="BGR",
		xyz="XYZ",
		hsv="HSV",
		hsl="HLS",
		hcl="LUV",
		hcy="YCrCb",
		lab="LAB",
		luv="LUV",
		yuv="YCrCb",
	)
	source = source.casefold()
	dest = dest.casefold()
	src = from_cv2.get(source, "RGB")
	dst = from_cv2.get(dest, "RGB")
	if source in ("rgb", "bgr"):
		pass
	elif source == "cmy":
		np.subtract(1, im, out=im)
	elif source == "xyz":
		im.T[0] *= range_xyz[0][1]
		im.T[1] *= range_xyz[1][1]
		im.T[2] *= range_xyz[2][1]
	elif source == "hsv":
		im.T[0] %= 1
		im.T[0] *= 360 * 255 / 256
	elif source == "hsl":
		im.T[0] %= 1
		im.T[0] *= 360 * 255 / 256
		temp = im.T[1].copy()
		im.T[1] = im.T[2]
		im.T[2] = temp
	elif source == "hsi":
		im.T[0] %= 1
		im.T[0] *= 255 / 256
		im = hsi_to_rgb(im)
	elif source == "hcl":
		im.T[0] *= 255 / 256
		im = hcl_to_luv(im)
		im.T[0] *= range_luv[0][1]
		im.T[1] *= max(abs(range_luv[1][1]), abs(range_luv[1][0]))
		im.T[2] *= max(abs(range_luv[2][1]), abs(range_luv[2][0]))
	elif source == "hcy":
		im.T[0] *= 255 / 256
		im = hcy_to_ycc(im)
	elif source == "lab":
		im.T[0] *= 100
		im.T[1:] *= 254
		im.T[1:] -= 127
	elif source == "luv":
		im.T[0] *= range_luv[0][1]
		im.T[1] *= range_luv[1][1] - range_luv[1][0]
		im.T[1] += range_luv[1][0]
		im.T[2] *= range_luv[2][1] - range_luv[2][0]
		im.T[2] += range_luv[2][0]
	elif source == "yuv":
		temp = im.T[1].copy()
		im.T[1] = im.T[2]
		im.T[2] = temp
	else:
		raise NotImplementedError(source)
	if src != dst:
		k = f"COLOR_{src}2{dst}"
		im = cv2.cvtColor(im, cv2_swaps[k], dst=im)
	if dest in ("rgb", "bgr"):
		pass
	elif dest == "cmy":
		np.subtract(1, im, out=im)
	elif dest == "xyz":
		im.T[0] *= 1 / range_xyz[0][1]
		im.T[1] *= 1 / range_xyz[1][1]
		im.T[2] *= 1 / range_xyz[2][1]
	elif dest == "hsv":
		im.T[0] *= 1 / 360 * 256 / 255
		im.T[0] %= 1
	elif dest == "hsl":
		temp = im.T[1].copy()
		im.T[1] = im.T[2]
		im.T[2] = temp
		im.T[0] *= 1 / 360 * 256 / 255
		im.T[0] %= 1
	elif dest == "hsi":
		im = rgb_to_hsi(im)
		im.T[0] *= 256 / 255
		im.T[0] %= 1
	elif dest == "hcl":
		im.T[0] *= 1 / range_luv[0][1]
		im.T[1] *= 1 / max(abs(range_luv[2][1]), abs(range_luv[2][0]))
		im.T[2] *= 1 / max(abs(range_luv[2][1]), abs(range_luv[2][0]))
		im = luv_to_hcl(im)
		im.T[0] *= 256 / 255
		im.T[0] %= 1
	elif dest == "hcy":
		im = ycc_to_hcy(im)
		im.T[0] *= 256 / 255
		im.T[0] %= 1
	elif dest == "lab":
		im.T[0] *= 1 / 100
		im.T[1:] += 127
		im.T[1:] *= 1 / 254
	elif dest == "luv":
		im.T[0] *= 1 / range_luv[0][1]
		im.T[1] -= range_luv[1][0]
		im.T[1] *= 1 / (range_luv[1][1] - range_luv[1][0])
		im.T[2] -= range_luv[2][0]
		im.T[2] *= 1 / (range_luv[2][1] - range_luv[2][0])
	elif dest == "yuv":
		temp = im.T[1].copy()
		im.T[1] = im.T[2]
		im.T[2] = temp
	else:
		raise NotImplementedError(dest)
	return im

def colourspace(image, source, dest):
	image, A = split_rgba(image)
	im = np.array(image, dtype=np.float32)
	im *= 1 / 255
	im = convert_colour(im, source, dest)
	im *= 255
	im = quantise_into(im, in_place=True)
	image = fromarray(im, "RGB")
	image = join_rgba(image, A)
	return image

def get_colour(image):
	rgb, A = split_rgba(image)
	if A:
		a = np.array(A, dtype=np.float32)
		a *= 1 / 255
		sumA = np.sum(a)
		if sumA == 0:
			return [0, 0, 0]
		return [np.sum(np.multiply(c.T, a)) / sumA for c in np.asanyarray(rgb, dtype=np.uint8).T]
	return [np.mean(c) for c in np.asanyarray(rgb, dtype=np.uint8).T]


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
	image, A = split_rgba(image)
	image = image.convert(image.mode, colourmatrix)
	return join_rgba(image, A)
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

def Enhance(image, operation, value):
	return getattr(ImageEnhance, operation)(image).enhance(value)

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
		L[~anytrans] = 0
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
			L[~anywhite & notblank] = 0
		elif ab and not aw:
			L[anyblack & notblank] = 255
			L[~anywhite & notblank] = 0
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
	outline = (outl | outu | outr | outd) & ~mask
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


def ectoplasm(url, message, force=False):
	import ectoplasm
	fn = None
	if isinstance(url, str):
		if not is_url(url):
			fn = url
			with open(fn, "rb") as f:
				b = f.read()
		else:
			b = get_request(url)
	else:
		b = url
	if isinstance(b, bytes):
		image = from_bytes(b)
	else:
		image = b
		i = io.BytesIO()
		image.save(i, "png")
		i.seek(0)
		b = i.read()
	if not fn:
		ts = time.time_ns() // 1000
		fn = "cache/" + str(ts) + ".png"
	if not force or not message:
		with open(fn, "wb") as f:
			f.write(b)
		import sys
		sys.stdout.flush()
		try:
			resp = ectoplasm.decode_image(image, path=fn)
		except Exception:
			resp = b""
		if resp or not message:
			return resp
	im = ectoplasm.encode_image(image, message)
	ectoplasm.save_image(im, message, fn)
	return fn