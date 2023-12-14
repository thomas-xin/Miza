import concurrent.futures
exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)
fut = exc.submit(__import__, "numpy")
import os, sys, random, time, base64
from PIL import Image
from PIL.PngImagePlugin import PngInfo
try:
	import pillow_heif
	pillow_heif.register_heif_opener()
except:
	pass

Resampling = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
Transform = getattr(Image, "Transform", Image)
fromarray = Image.fromarray

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
	out = []
	for ia in (np.floor(aavg), np.ceil(aavg)):
		i2 = im.point(lambda x: int(x >= ia))
		bi = np.array(i2, dtype=np.uint8).ravel()
		bo = np.zeros(len(bi) // 8, dtype=np.uint8)
		for i in range(8):
			bo += bi[i::8] << (7 - i)
		out.append(bo)
	# print(aavg, out)
	return out

# Reduces an image to a 12-bit rgb 32x32 square, returning an additional greyscale luma component
def split_to(im):
	hi = im.resize((32, 32), resample=Resampling.LANCZOS)
	g = hi.convert("L")
	h, s, l = hsl_split(hi, convert=False, dtype=np.float32)
	return (np.uint8(a) >> 4 for a in (h, s, l, g))

def hash_to(im, msg, skip=False):
	if skip:
		h, s, l, g = split_to(im)
		rhs = hash_reduce(g)
	else:
		h, s, l, g, rhs = compare_to(im, msg)
	sl = s + (l << 4)
	hg = h + (g << 4)
	hb = hg.tobytes() + sl.tobytes()
	for rh in rhs:
		i = 0
		fd = f"iman/{rh[0]}/{rh[1]}.txt"
		# for i, fd in enumerate((f"iman/{rh[0]}/{rh[1]}.txt", f"iman/{255 - rh[0]}/{255 - rh[1]}.txt")):
		s = base64.b64encode(hb).rstrip(b"=").decode("ascii") + f";{i}:" + msg + "\n"
		folder = fd.rsplit("/", 1)[0]
		if not os.path.exists(folder):
			os.mkdir(folder)
		with open(fd, "a", encoding="utf-8") as f:
			f.write(s)

def compare_to(im, msg):
	tups = list(split_to(im))
	g = tups[-1]
	rhs = hash_reduce(g)
	# rhs = [[18, 237]]
	# rhs = [[247, 19]]
	for rh in rhs:
		fd = f"iman/{rh[0]}/{rh[1]}.txt"
		if os.path.exists(fd):
			break

	if os.path.exists(fd):
		with open(fd, "r", encoding="utf-8") as f:
			d = f.readlines()

		h, s, l, g = map(np.float32, tups)
		for line in d:
			i = line.index(":")
			k, v = line[:i], line[i + 1:]
			if k[-2] == ";":
				inverted = int(k[-1])
				k = k[:-2]
			else:
				inverted = False
			v = v[:-1]
			hb2 = np.frombuffer(base64.b64decode(k.encode("ascii") + b"=="), dtype=np.uint8)
			half = len(hb2) >> 1
			hg2, sl2 = hb2[:half].reshape((32, 32)), hb2[half:].reshape((32, 32))
			h2 = hg2 & 15
			g2 = hg2 >> 4
			s2 = sl2 & 15
			l2 = sl2 >> 4

			# print(h)
			# print(s)
			# print(l)
			h2, l2, s2, g2 = map(np.float32, (h2, l2, s2, g2))
			heff = np.sqrt(900 - (30 - s) * (30 - s2)) / 30
			# print(np.abs(np.abs(h - h2) - 8))
			hd = np.sum((8 - np.abs(np.abs(h - h2) - 8)) * heff) / 8 / 1024
			seff = np.sqrt(900 - (30 - l) * (30 - l2)) / 30
			sd = np.sum(np.abs(s - s2) * seff) / 15 / 1024
			if inverted:
				gd = np.sum(np.abs(15 - g - g2)) / 15 / 1024
				ld = np.sum(np.abs(15 - l - l2)) / 15 / 1024
			else:
				gd = np.sum(np.abs(g - g2)) / 15 / 1024
				ld = np.sum(np.abs(l - l2)) / 15 / 1024
			# print(np.abs(l - l2))

			# print(hd, sd, ld, gd, inverted)
			R = (hd + sd) / 2 + min(ld, gd * 2)
			# print(rh, R, v)
			if R <= 0.05:
				if v == msg:
					print("No copyright detected.")
					raise SystemExit
				print("Copyright detected in hashing:", v)
				raise SystemExit

	return *tups, rhs

test = "-t" in sys.argv
if test:
	sys.argv.remove("-t")

if "-i" in sys.argv:
	i = sys.argv.index("-i")
	its = int(sys.argv[i + 1])
	sys.argv = sys.argv[:i] + sys.argv[i + 2:]
else:
	its = None

if "-o" in sys.argv:
	i = sys.argv.index("-o")
	ofn = sys.argv[i + 1]
	sys.argv = sys.argv[:i] + sys.argv[i + 2:]
else:
	ofn = None

fn = sys.argv[1]
msg = " ".join(sys.argv[2:])
if msg and len(msg) < 2:
	msg += "\u200b"
if not its:
	its = 100 // (len(msg.encode("utf-8")) + 2)
	# print(its)
if fn.startswith("https://") or fn.startswith("http://"):
	import requests, io
	im = Image.open(io.BytesIO(requests.get(fn).content))
	fn = fn.rsplit("/", 1)[-1]
else:
	im = Image.open(fn)
if getattr(im, "text", None) and im.text.get("copyright") and not test:
	if im.text["copyright"] != msg:
		print("Copyright detected in metadata:", im.text["copyright"])
		raise SystemExit
Ms = 156
if not msg and (im.width > Ms or im.height > Ms):

	def max_size(w, h, maxsize, force=False):
		s = w * h
		m = maxsize * maxsize
		if s > m or force:
			r = (m / s) ** 0.5
			w = round(w * r)
			h = round(h * r)
		return w, h

	im = im.resize(max_size(im.width, im.height, Ms), resample=Resampling.NEAREST)

if "RGB" not in im.mode:
	im = im.convert("RGBA")
area = im.size[0] * im.size[1]
if area < 1024:
	raise ValueError("Input image too small.")
ar = im.size[0] / im.size[1]
# print(im.size, area)

i_entropy = im.entropy()
e_entropy = abs(i_entropy) ** 2 / 256
ie_req = 4
entropy = (min(1.5, e_entropy) + 0.5) / 2
# print(entropy, e_entropy, i_entropy)

write = bool(msg)
mb = msg.encode("utf-8")
b = [bytes([its])]
for n in range(its):
	b.append(mb)
	mb = invert(mb)
b.append(bytes([its, 170] * its)[1:])
b = b"".join(b)
# print(len(b))
bb = list(bool(i & 1 << j) for i in b for j in range(8))
bs = len(bb)
it = iter(bb)
ic = 0
reader = []

# if not os.path.exists("iman"):
	# os.mkdir("iman")
# if write and i_entropy >= ie_req:
	# hash_to(im, msg)
# else:
	# compare_to(im, msg)

w = h = 17
# while True:
	# if abs(w / h - ar) / ar < 1 / 32:
		# break
	# if w / h > ar:
		# if (w - 1) * h >= lim:
			# w -= 1
		# else:
			# h += 1
			# lim *= np.sqrt(2)
	# else:
		# if w * (h - 1) >= lim:
			# h -= 1
		# else:
			# w += 1
			# lim *= np.sqrt(2)
# print(bs, w, h)

counted = True
copydetect = True
inverted = False
spl = list(im.split())
if len(spl) > 3:
	spl[-1] = spl[-1].point(lambda x: max(x, 8))

np = fut.result()
np.random.seed(time.time_ns() & 4294967295)
ars = [np.array(pl, dtype=np.uint8) for pl in spl[:3]]


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

# sx, ex = round_random((w - 1) / 2 * im.width / w), round_random((w + 1) / 2 * im.width / w)
# sy, ey = round_random((h - 1) / 2 * im.height / h), round_random((h + 1) / 2 * im.height / h)
# pa = (ey - sy) * (ex - sx)
# core = []
# for a in ars:
	# target = a[sy:ey].T[sx:ex]
	# core.append(np.sum(target & 2 > 0) + np.sum(target & 1) >= pa)
# if not core[0] == core[1] == core[2]:
	# if not write:
		# print("No copyright detected.")
		# raise SystemExit
	# copydetect = False

vis = {}
x = y = 0
di = 3
for p in range(w * h):
	# print((x, y))
	if (x, y) in vis:
		break
	vis[(x, y)] = True
	if di == 0:
		if x <= 0 or (x - 1, y) in vis:
			di = 3
	elif di == 1:
		if y <= 0 or (x, y - 1) in vis:
			di = 0
	elif di == 2:
		if x >= w - 1 or (x + 1, y) in vis:
			di = 1
	elif di == 3:
		if y >= h - 1 or (x, y + 1) in vis:
			di = 2
	if di == 0:
		x -= 1
	elif di == 1:
		y -= 1
	elif di == 2:
		x += 1
	elif di == 3:
		y += 1

def encode(a, target, rc, bit):
	if test:
		if bit:
			target[:] = 255
		else:
			target[:] = 0
	elif write:
		rv = target.ravel()

		if entropy != 1:
			r1 = np.random.randint(-1, 1, int(np.ceil(pa / 2)), dtype=np.int8)
			r1 |= 1
			r1 <<= 1
			r1 = np.tile(r1, (2, 1)).T.ravel()[:pa]
			r2 = np.random.randint(0, 4, int(np.ceil(pa / 2)), dtype=np.int8)
			r2[r2 == 0] = -3
			r2[r2 > 0] = 1
			r2 = np.tile(r2, (2, 1)).T.ravel()[:pa]
			# r1 = (-2, 2)
			# r2 = (-3, 1)
			# print(r1, r2)

		if entropy != 0:
			rind = np.random.binomial(1, entropy, len(rv)).astype(bool)

		if bit:
			# rv[:] = 255
			v = np.clip(rv, 2, 253, out=rv)
			if entropy != 0:
				v[rind] |= 2
				v[rind] &= 254
			if entropy != 1:
				ind = v & 3 # must be &2 = 2
				mask = ind == 0
				t = v[mask]
				v[mask] = np.subtract(t, r1[:len(t)], out=t, casting="unsafe")
				mask = ind == 1
				t = v[mask]
				v[mask] = np.add(t, r2[:len(t)], out=t, casting="unsafe")
				mask = ind == 3
				t = v[mask]
				v[mask] = np.subtract(t, r2[:len(t)], out=t, casting="unsafe")
		else:
			# rv[:] = 0
			v = np.clip(rv, 0, 251, out=rv)
			if entropy != 0:
				v[rind] &= 252
			if entropy != 1:
				ind = v & 3 # must be &2 = 0
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
		# print(bit, v[:20] & 2, v[:5], entropy)

corners = [
	(0, 0),
	(0, h - 1),
	(w - 1, h - 1),
	(w - 1, 0),
]
if w & 1 and h & 1:
	corners.append((w >> 1, h >> 1))
for p in corners:
	vis.pop(p, None)
cornerdata = []
futs = []
tiles = corners + list(reversed(vis))
for p, (x, y) in enumerate(tiles):
	if p == len(corners) and copydetect:
		nc = sum(cornerdata)
		if nc >= len(corners) - 1:
			pass
		else:
			ic = len(corners) - nc
			if ic >= len(corners) - 1:
				inverted = True
			else:
				if not write:
					print("No copyright detected.")
					raise SystemExit
				copydetect = False
	counted = p >= len(corners)
	sx = round(x * im.width / w)
	ex = round((x + 1) * im.width / w)
	sy = round(y * im.height / h)
	ey = round((y + 1) * im.height / h)
	pa = (ey - sy) * (ex - sx)
	order = [0, 1, 2]
	seed = p
	if seed & 1:
		order = order[::-1]
	seed = (seed >> 1) % 3
	for i in order[seed:] + order[:seed]:
		a = ars[i]
		target = a[sy:ey].T[sx:ex]
		rc = np.sum(target & 2 > 0) * 2
		# print(rc, pa)
		if copydetect:
			if not counted:
				rr = rc >= pa / np.sqrt(2)
				cornerdata.append(rr)
			else:
				reader.append(rc >= pa)
		if counted:
			bit = next(it, False if x * h + y & 8 else True)
		else:
			bit = True
		encode(a, target, rc, bit)
		# fut = exc.submit(encode, a, target, rc, bit)
		# futs.append(fut)

for fut in futs:
	fut.result()

if write:
	for i in range(len(ars)):
		spl[i] = Image.fromarray(ars[i], mode="L")

try:
	next(it)
except StopIteration:
	pass
else:
	i = 1
	try:
		for i in range(1, 4096):
			next(it)
	except StopIteration:
		pass
	raise EOFError(i)

while len(reader) & 7:
	reader.pop(-1)
# print(reader)

if copydetect:
	# print(reader)
	bitd = np.array(reader, dtype=np.bool_)
	b = np.zeros(len(reader) // 8, dtype=np.uint8)
	for i in range(8):
		b |= bitd[i::8] << np.uint8(i)
	if inverted:
		b ^= 255
	# print(b)
	while len(b) and b[-1] != 170:
		b = b[:-1]
else:
	b = b""

try:
	# print(b)
	if len(b) < 1 or b[-1] != 170:
		raise ValueError
	ita = [b[0]] if b[0] != 0 else []
	b = b[1:]
	for i in range(len(b)):
		if i & 1 or b[-1] == 170:
			if b[-1] != 170:
				ita.append(b[-1])
			elif (len(b) < 3 or b[-3] != 170):
				b = b[:-1]
				break
			b = b[:-1]
		else:
			break
	if not ita:
		raise ValueError
	u, c = np.unique(ita, return_counts=True)
	a = np.argsort(c)
	its = u[a[-1]]
	if its > 50 or its < 2:
		raise ValueError
	if len(b) * its < 10:
		raise ValueError
	if len(b) < its:
		raise ValueError
	l = len(b) // its
	dups = []
	for i in range(its):
		bi = b[i * l:i * l + l]
		if i & 1:
			bi ^= 255
		dups.append(bi)
	dups = np.asanyarray(dups, dtype=np.uint8).T
	# print(dups)
	errs = 0
	confidences = []
	d = []
	for i in range(l):
		u, c = np.unique(dups[i], return_counts=True)
		a = np.argsort(c)
		m = np.max(c)
		confidences.append(m)
		if sum(c == m) > 1:
			if errs >= l / 2:
				raise ValueError
			errs += 1
			li = sorted(n for n in u if 32 <= n < 128)
			if not li:
				c = u[-1]
			else:
				c = li[0]
		else:
			c = u[a[-1]]
		d.append(c)
		# print(dups[i])
		# print(np.unique(dups[i], return_counts=True))
	b = bytes(d)
	if not b or len(b) < 2:
		raise ValueError
	# print(b)
	s = b.decode("utf-8").removesuffix("\u200b")
	if s == msg:
		raise ValueError
	confidence = round(sum(confidences) / len(confidences) / 5 * 100)
	if confidence < 40:
		raise ValueError
except (ValueError, UnicodeDecodeError):
	# print(b)
	if write:
		im = Image.merge(im.mode, spl)
		fn = ofn or fn.rsplit(".", 1)[0] + "~1.png"
		# im.save(fn)
		meta = PngInfo()
		meta.add_text("copyright", msg)
		level = round((1 - entropy) * 8 + 1)
		im.save(fn, format="png", compress_level=level, pnginfo=meta)
	print("No copyright detected.")
else:
	# if i_entropy >= ie_req:
		# hash_to(im, s, skip=True)
	print(f"Copyright detected in steganography:", s)