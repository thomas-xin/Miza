import os, sys, random, time, base64
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pillow_heif
pillow_heif.register_heif_opener()
import numpy as np

Resampling = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
Transform = getattr(Image, "Transform", Image)

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

invert = lambda b: bytes(i ^ 255 for i in b)

# Reduces an image to a 4-bit black and white 4x4 square, then bitcrushes to 1-bit leaving 2 bytes total
def hash_reduce(l):
	# amax = np.max(l)
	aavg = np.mean(l)
	# amin = np.min(l)
	# if amax <= amin:
		# amax = amin + 1
	im = Image.fromarray(l, mode="L")
	im = im.resize((4, 4), resample=Resampling.LANCZOS)
	im = im.point(lambda x: int(x >= aavg))
	bi = np.array(im, dtype=np.uint8).ravel()
	bo = np.zeros(len(bi) // 8, dtype=np.uint8)
	for i in range(8):
		bo += bi[i::8] << (7 - i)
	return bo

# Reduces an image to a 12-bit rgb 32x32 square, returning an additional greyscale luma component
def split_to(im):
	hi = im.resize((32, 32), resample=Resampling.LANCZOS)
	g = hi.convert("L")
	h, s, l = hsl_split(hi, convert=False, dtype=np.float32)
	return (np.uint8(a) >> 4 for a in (h, s, l, g))

def hash_to(im, msg, skip=False):
	if skip:
		h, s, l, g = split_to(im)
		rh = hash_reduce(g)
	else:
		h, s, l, g, rh = compare_to(im, msg)
	sl = s + (l << 4)
	hg = h + (g << 4)
	hb = hg.tobytes() + sl.tobytes()
	s = base64.b64encode(hb).rstrip(b"=").decode("ascii") + ":" + msg + "\n"
	for fd in (f"iman/{rh[0]}/{rh[1]}.txt", f"iman/{255 - rh[0]}/{255 - rh[1]}.txt"):
		folder = fd.rsplit("/", 1)[0]
		if not os.path.exists(folder):
			os.mkdir(folder)
		with open(fd, "a", encoding="utf-8") as f:
			f.write(s)

def compare_to(im, msg):
	h, s, l, g = split_to(im)
	rh = hash_reduce(g)
	fd = f"iman/{rh[0]}/{rh[1]}.txt"

	if os.path.exists(fd):
		with open(fd, "r", encoding="utf-8") as f:
			d = f.readlines()

		for line in d:
			k, v = line.split(":", 1)
			v = v[:-1]
			hb2 = np.frombuffer(base64.b64decode(k.encode("ascii") + b"=="), dtype=np.uint8)
			half = len(hb2) >> 1
			hg2, sl2 = hb2[:half].reshape((32, 32)), hb2[half:].reshape((32, 32))
			h2 = hg2 & 15
			g2 = hg2 >> 4
			s2 = sl2 & 15
			l2 = sl2 >> 4

			heff = np.sqrt(65536 - (255 - s) * (255 - s2)) / 256
			hd = np.sum((128 - np.abs((h - h2) - 128)) * heff) / 128 / 1024
			seff = np.sqrt(65536 - (255 - l) * (255 - l2)) / 256
			sd = np.sum(np.abs(s - s2) * seff) / 15 / 1024
			ld = np.sum(np.abs(l - l2)) / 15 / 1024
			gd = np.sum(np.abs(g - g2)) / 15 / 1024
			gd2 = np.sum(np.abs(15 - g - g2)) / 15 / 1024

			# print(hd, sd, ld, gd, gd2)
			R = (hd + sd) / 2 + min(ld, gd, gd2)
			print(rh, R)
			if R <= 3:
				if v == msg:
					print("No copyright detected.")
					raise SystemExit
				print("Copyright detected in hashing:", v)
				raise SystemExit

	return h, s, l, g, rh

if not os.path.exists("iman"):
	os.mkdir("iman")

test = "-t" in sys.argv
if test:
	sys.argv.remove("-t")

fn = sys.argv[1]
msg = " ".join(sys.argv[2:])
im = Image.open(fn)
if getattr(im, "text", None) and im.text.get("copyright"):
	if im.text["copyright"] != msg:
		print("Copyright detected in metadata:", im.text["copyright"])
		raise SystemExit

if "RGB" not in im.mode:
	im = im.convert("RGBA")
area = im.size[0] * im.size[1]
if area < 1024:
	raise ValueError("Input image too small.")
ar = im.size[0] / im.size[1]
# print(im.size, area)

i_entropy = im.entropy()
ie_req = 4
entropy = min(1, abs(i_entropy) ** 3 / 384)
# print(entropy, im.entropy())

write = bool(msg)
mb = msg.encode("utf-8")
b = b"\xff" + b"\xff".join((mb, invert(mb), mb)) + b"\xff" * 2
bb = list(bool(i & 1 << j) for i in b for j in range(8))
bs = len(bb)
it = iter(bb)
ic = 0
reader = []

if write and i_entropy >= ie_req:
	hash_to(im, msg)
else:
	compare_to(im, msg)

lim = 60 * 8 / 3
w = h = int(np.ceil(np.sqrt(lim)))
while True:
	if abs(w / h - ar) / ar < 1 / 32:
		break
	if w / h > ar:
		if (w - 1) * h >= lim:
			w -= 1
		else:
			h += 1
			lim *= np.sqrt(2)
	else:
		if w * (h - 1) >= lim:
			h -= 1
		else:
			w += 1
			lim *= np.sqrt(2)
# print(bs, w, h)

copydetect = True
np.random.seed(time.time_ns() & 4294967295)
spl = list(im.split())
if len(spl) > 3:
	spl[-1] = spl[-1].point(lambda x: max(x, 8))
for i in (2, 0, 1):
	pl = spl[i]
	a = np.array(pl, dtype=np.uint8)
	ex = 0
	for x in range(1, w + 1):
		sx = ex
		ex = round_random(x * im.width / w)
		ey = 0
		for y in range(1, h + 1):
			sy = ey
			ey = round_random(y * im.height / h)
			pa = (ey - sy) * (ex - sx)
			target = a[sy:ey].T[sx:ex]
			if copydetect:
				reader.append(np.sum(target & 2 > 0) + np.sum(target & 1) >= pa)
			if len(reader) == 8 and copydetect and reader != [True] * 8:
				if not write:
					print("No copyright detected.")
					raise SystemExit
				copydetect = False

			bit = next(it, False)
			if test:
				if bit:
					target[:] = 255
				else:
					target[:] = 0
			elif write:
				rv = target.ravel()

				if entropy != 1:
					r1 = np.random.randint(-1, 1, pa, dtype=np.int8)
					r1 |= 1
					r1 <<= 1
					r2 = np.random.randint(0, 4, pa, dtype=np.int8)
					r2[r2 == 0] = -3
					r2[r2 > 0] = 1

				if entropy != 0:
					rind = np.zeros(len(rv), dtype=np.bool_)
					rind[:int(len(rv) * entropy)] = True
					np.random.shuffle(rind)

				if bit:
					v = np.clip(rv, 3, None, out=rv)
					if entropy != 0:
						v[rind] |= 3
					if entropy != 1:
						ind = v & 3
						mask = ind == 0
						t = v[mask]
						v[mask] = np.subtract(t, r2[:len(t)], out=t, casting="unsafe")
						mask = ind == 1
						t = v[mask]
						v[mask] = np.add(t, r1[:len(t)], out=t, casting="unsafe")
						mask = ind == 2
						t = v[mask]
						v[mask] = np.add(t, r2[:len(t)], out=t, casting="unsafe")
				else:
					v = np.clip(rv, None, 252, out=rv)
					if entropy != 0:
						v[rind] &= 252
					if entropy != 1:
						ind = v & 3
						mask = ind == 1
						t = v[mask]
						v[mask] = np.subtract(t, r2[:len(t)], out=t, casting="unsafe")
						mask = ind == 2
						t = v[mask]
						v[mask] = np.add(t, r1[:len(t)], out=t, casting="unsafe")
						mask = ind == 3
						t = v[mask]
						v[mask] = np.add(t, r2[:len(t)], out=t, casting="unsafe")
				target[:] = v.reshape(target.shape)

	spl[i] = Image.fromarray(a, mode="L")

try:
	next(it)
except StopIteration:
	pass
else:
	raise EOFError

while len(reader) & 7:
	reader.pop(-1)
# print(reader)

if copydetect:
	bitd = np.array(reader, dtype=np.bool_)
	byted = np.zeros(len(reader) // 8, dtype=np.uint8)
	for i in range(8):
		byted |= bitd[i::8] << np.uint8(i)
	b = byted.tobytes()
	# print(b)
	while b and b[-1] != 255:
		b = b[:-1]
else:
	b = b""

try:
	if not b or b[0] != 255 or b[-1] != 255:
		raise ValueError
	b = b[1:-1]
	if len(b) % 3:
		raise ValueError
	l = len(b) // 3
	x = b[:l - 1]
	y = invert(b[l:l * 2 - 1])
	z = b[l * 2:l * 3 - 1]
	# print(x, y, z)
	if x == y:
		b = x
	elif x == z:
		b = x
	elif y == z:
		b = z
	else:
		d = []
		for i in range(l - 1):
			if x[i] == y[i]:
				d.append(x[i])
			elif x[i] == z[i]:
				d.append(x[i])
			else:
				d.append(z[i])
		b = bytes(d)
	s = b.decode("utf-8")
	if s == msg:
		raise ValueError
except (ValueError, UnicodeDecodeError):
	# print(b)
	if write:
		im = Image.merge(im.mode, spl)
		fn = fn.rsplit(".", 1)[0] + "~1.png"
		meta = PngInfo()
		meta.add_text("copyright", msg)
		im.save(fn, pnginfo=meta)
	print("No copyright detected.")
else:
	if i_entropy >= ie_req:
		hash_to(im, s, skip=True)
	print("Copyright detected in steganography:", s)