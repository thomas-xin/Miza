import sys, random, time
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

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

test = "-t" in sys.argv
if test:
	sys.argv.remove("-t")

fn = sys.argv[1]
msg = " ".join(sys.argv[2:])
im = Image.open(fn)
if getattr(im, "text", None) and im.text.get("copyright"):
	if im.text["copyright"] != msg:
		print("Copyright detected:", im.text["copyright"])
		raise SystemExit

if "RGB" not in im.mode:
	im = im.convert("RGBA")
area = im.size[0] * im.size[1]
if area < 1024:
	raise ValueError("Input image too small.")
ar = im.size[0] / im.size[1]
# print(im.size, area)

entropy = max(0, min(1, im.entropy() ** 3 / 512 + 0.125))
# print(entropy, im.entropy())

write = bool(msg)
b = b"\xff" + (msg.encode("utf-8") + b"\xff") * 3 + b"\xff"
bb = list(bool(i & 1 << j) for i in b for j in range(8))
bs = len(bb)
it = iter(bb)
ic = 0
reader = []


lim = 48 * 8 / 3
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

np.random.seed(time.time_ns() & 4294967295)
spl = list(im.split())
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
			reader.append(np.sum(a[sy:ey].T[sx:ex] & 2 > 0) + np.sum(a[sy:ey].T[sx:ex] & 1) >= pa)
			try:
				bit = next(it)
			except StopIteration:
				bit = 0
			weight = 16
			target = a[sy:ey].T[sx:ex]
			rv = target.ravel()
			if test:
				if bit:
					rv[:] = 255
				else:
					rv[:] = 0
			elif write:
				r1 = np.random.randint(-1, 1, pa, dtype=np.int8)
				r1 |= 1
				r1 <<= 1
				r2 = np.random.randint(0, 4, pa, dtype=np.int8)
				r2[r2 == 0] = -3
				r2[r2 > 0] = 1
				rind = np.zeros(len(rv), dtype=np.bool_)
				if entropy:
					rind[:int(len(rv) * entropy)] = True
				np.random.shuffle(rind)

				if bit:
					v = np.clip(rv, 3, None, out=rv)
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
						target[:] = v.reshape(target.shape)
				else:
					v = np.clip(rv, None, 252, out=rv)
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

bitd = np.array(reader, dtype=np.bool_)
byted = np.zeros(len(reader) // 8, dtype=np.uint8)
for i in range(8):
	byted |= bitd[i::8] << np.uint8(i)
b = byted.tobytes()
while b and b[-1] != 255:
	b = b[:-1]

try:
	if not b or b[0] != 255 or b[-1] != 255:
		raise ValueError
	b = b[1:-1]
	if len(b) % 3:
		raise ValueError
	l = len(b) // 3
	x = b[:l - 1]
	y = b[l:l * 2 - 1]
	z = b[l * 2:l * 3 - 1]
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
	print("Copyright detected:", s)