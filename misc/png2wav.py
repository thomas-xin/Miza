import os, sys, time, subprocess

is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
ffmpeg_start = ("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto", "-vn")

hsv = sys.argv[-1] != "-hsv"
if not hsv:
	sys.argv.pop(-1)
if len(sys.argv) > 1:
	fn = sys.argv[1]
else:
	fn = input("Please input a filename or URL: ")

if len(sys.argv) > 2:
	fo = sys.argv[2]
	pcm = fo.endswith(".pcm")
else:
	fo = fn.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".wav"
	pcm = False

if is_url(fn):
	import io
	fi, fn = fn, io.BytesIO()
	try:
		import requests
		with requests.get(fi, stream=True) as resp:
			it = resp.iter_content(1048576)
			while True:
				b = next(it)
				if not b:
					raise StopIteration
				fn.write(b)
	except StopIteration:
		pass

if not pcm:
	cmd = ffmpeg_start + ("-f", "f32le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "192k", "-vbr", "on", fo)
	p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
else:
	f = open(fo, "wb")

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 4294967296

img = Image.open(fn)
dfts = img.height >> 1
ffts = dfts - 1 << 1
if hsv:
	img = img.convert("HSV")

arr = e16 = None
columns = np.asanyarray(img, dtype=np.float32)
for img in columns.swapaxes(0, 1):
	hue, sat, val = img[:len(img) // 2 << 1].T[:3]
	hue -= 128
	hue *= np.pi / 128
	phase = hue
	sat[sat == 0] = 255
	np.subtract(255, sat, out=sat)
	np.power(1.03125, val, out=val)
	val *= sat
	val *= ffts * 2 / 255 / 8192
	cpl = np.multiply(phase, 1j)
	np.exp(cpl, out=cpl)
	cpl *= val
	lft, rft = cpl[::2], cpl[1::2]
	left, right = np.fft.irfft(lft[::-1]), np.fft.irfft(rft[::-1])
	if arr is None or len(arr) != ffts << 1:
		arr = np.empty(ffts << 1, dtype=np.float32)
	arr[::2] = left
	arr[1::2] = right
	np.clip(arr, -1, 1, out=arr)
	if pcm:
		arr *= 32767
		if e16 is None or len(e16) != len(arr):
			e16 = arr.astype(np.int16)
		else:
			e16[:] = arr
		b = e16.data
		f.write(b)
	else:
		b = arr.data
		p.stdin.write(b)

if pcm:
	f.close()
else:
	p.stdin.close()