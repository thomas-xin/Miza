import os, sys, time, subprocess, psutil, numpy
from PIL import Image
np = numpy

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
duration = float(subprocess.check_output(ffmpeg_probe))
frames = duration * 48000
req = int(np.sqrt(frames * 64) / 8) * 8
ffts = req // 8
dfts = ffts // 2 + 1
# print(dfts, ffts)

fi = "temp.pcm"
if os.path.exists(fi):
    os.remove(fi)

cmd = ffmpeg_start + ("-i", fn, "-f", "f32le", "-ac", "2", "-ar", "48k", fi)
p = psutil.Popen(cmd, stderr=subprocess.PIPE)

time.sleep(0.1)
while True:
    if os.path.exists(fi) and os.path.getsize(fi) >= 96000:
        break
    if not p.is_running():
        raise RuntimeError(p.stderr.read().decode("utf-8"))
    time.sleep(0.01)
f = open(fi, "rb")

rat = np.log2(1.03125)
columns = []
while True:
    b = f.read(req)
    while len(b) < req and p.is_running():
        time.sleep(0.1)
        b += f.read(req - len(b))
    if not b:
        break
    if len(b) < req:
        b += b"\x00" * (req - len(b))
    arr = np.frombuffer(b, dtype=np.float32)
    left, right = arr[::2], arr[1::2]
    lft, rft = np.fft.rfft(left), np.fft.rfft(right)
    lft, rft = lft[:dfts][::-1], rft[:dfts][::-1]
    lamp, lpha = np.abs(lft), np.angle(lft)
    ramp, rpha = np.abs(rft), np.angle(rft)
    amplitude, phase = np.empty(dfts << 1, dtype=np.float32), np.empty(dfts << 1, dtype=np.float32)
    amplitude[::2] = lamp
    amplitude[1::2] = ramp
    phase[::2] = lpha
    phase[1::2] = rpha
    amplitude *= 255 / len(arr) * 4096
    amp = amplitude
    amp2 = np.log2(amp / 255)
    np.multiply(amp2, 1 / rat, out=amp2)
    np.clip(amp2, 0, None, out=amp2)
    amp2 = np.ceil(amp2, out=amp2).astype(np.uint8)
    np.divide(amp, 1.03125 ** amp2, out=amp)
    np.clip(amp, None, 255, out=amp)
    np.round(amp, out=amp)
    np.subtract(255, amp2, out=amp2)
    phase *= 128 / np.pi
    phase += 128
    size = (1, len(amplitude))
    hue = Image.frombuffer("L", size, phase.astype(np.uint8).tobytes())
    sat = Image.frombuffer("L", size, amp2.astype(np.uint8).tobytes())
    val = Image.frombuffer("L", size, amp.astype(np.uint8).tobytes())
    img = Image.merge("HSV" if hsv else "RGB", (hue, sat, val))
    columns.append(img)

out = Image.new("HSV" if hsv else "RGB", (len(columns), dfts << 1))
for i, img in enumerate(columns):
    out.paste(img, (i, 0))
if hsv:
    out = out.convert("RGB")

out.save(fo)