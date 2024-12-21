import os
import sys
import time
import subprocess

def is_url(url):
	return "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
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
else:
	fo = fn.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".png"

ffmpeg_probe = (
	"ffprobe",
	"-v",
	"error",
	"-select_streams",
	"a:0",
	"-show_entries",
	"format=duration",
	"-of",
	"default=nokey=1:noprint_wrappers=1",
	fn,
)
try:
	duration = float(subprocess.check_output(ffmpeg_probe))
except ValueError:
	ffmpeg_probe = (
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"stream=duration",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		fn,
	)
	duration = float(subprocess.check_output(ffmpeg_probe))
frames = duration * 48000
req = int(frames ** 0.5) * 8
ffts = req // 8
dfts = ffts // 2 + 1
# print(dfts, ffts)

fi = "temp.pcm"
if os.path.exists(fi):
	os.remove(fi)

cmd = ffmpeg_start + ("-i", fn, "-f", "f32le", "-ac", "2", "-ar", "48k", fi)
p = subprocess.Popen(cmd)

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 4294967296

time.sleep(0.1)
while True:
	if os.path.exists(fi) and os.path.getsize(fi) >= 96000:
		break
	if p.poll() is not None:
		raise RuntimeError(p.stderr.read().decode("utf-8"))
	time.sleep(0.01)
f = open(fi, "rb")

random = np.random.default_rng(0)
# Randomly rounds with weighted distribution; i.e. 1.6 rounds to 2 with 60% chance, 1 with 40% chance
def round_random(x):
	x = np.asanyarray(x)
	itemp = globals().get("itemp")
	if itemp is None or len(itemp) != len(x):
		# Needs space to store temporary values
		globals()["itemp"] = np.empty(len(x), dtype=np.uint8)
		globals()["ftemp"] = np.empty(len(x), dtype=np.float32)
	# integer part
	y = np.floor(x, out=itemp, casting="unsafe")
	# fractional part
	x -= y
	# 0~1 distribution
	z = random.random(len(x), dtype=np.float32, out=globals()["ftemp"])
	y[z <= x] += 1
	return y

amp = None
rat = np.log2(1.03125)
columns = []
while True:
	b = f.read(req)
	while len(b) < req and p.poll() is None:
		time.sleep(0.1)
		b += f.read(req - len(b))
	if not b:
		f.close()
		os.remove(fi)
		break
	if len(b) < req:
		b += b"\x00" * (req - len(b))
	arr = np.frombuffer(b, dtype=np.float32)
	left, right = arr[::2], arr[1::2]
	lft, rft = np.fft.rfft(left), np.fft.rfft(right)
	lft, rft = lft[:dfts][::-1], rft[:dfts][::-1]
	if amp is None or len(amp) != dfts << 1:
		amp = np.empty(dfts << 1, dtype=np.float32)
		phase = np.empty(dfts << 1, dtype=np.float32)
		lamp = np.empty(dfts, dtype=np.float32)
		ramp = np.empty(dfts, dtype=np.float32)
		amp2 = np.empty(dfts << 1, dtype=np.float32)
		amp3 = np.empty(dfts << 1, dtype=np.uint8)
		ampi = np.empty(dfts << 1, dtype=np.uint8)
		ampe = np.empty(dfts << 1, dtype=np.float32)
	lamp, lpha = np.abs(lft, out=lamp), np.angle(lft)
	ramp, rpha = np.abs(rft, out=ramp), np.angle(rft)
	amp[::2] = lamp
	amp[1::2] = ramp
	phase[::2] = lpha
	phase[1::2] = rpha
	amp *= 255 * 8192 / len(arr)
	norm = min(np.mean(amp) / 32, np.max(amp) / 512)
	mask = amp > norm
	amp2 = np.log2(amp / 96, out=amp2)
	np.multiply(amp2, 1 / rat, out=amp2)
	np.clip(amp2, 0, 255, out=amp2)
	amp2[mask] = np.clip(amp2[mask], 32, None)
	ampi[:] = np.ceil(amp2, out=amp3, casting="unsafe")
	np.power(1 / 1.03125, ampi, out=ampe)
	amp *= ampe
	np.round(amp, out=amp)
	phase *= 128 / np.pi
	phase += 128
	size = (1, len(amp))
	temp = round_random(phase)
	hue = Image.frombuffer("L", size, temp.tobytes())
	temp = amp.astype(np.uint8) ^ 255
	temp[temp == 0] = 1
	sat = Image.frombuffer("L", size, temp.tobytes())
	temp = amp3
	val = Image.frombuffer("L", size, temp.data)
	img = Image.merge("HSV" if hsv else "RGB", (hue, sat, val))
	columns.append(img)

out = Image.new("HSV" if hsv else "RGB", (len(columns), dfts << 1))
for i, img in enumerate(columns):
	out.paste(img, (i, 0))
if hsv:
	out = out.convert("RGB")

out.save(fo)