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
	import pyqrcode
	img = pyqrcode.create(s, error=err, version=ver, mode=None, encoding="utf-8" if max(s) >= 80 else "ascii")
	fn = f"cache/{time.time_ns() // 1000}.png"
	if not os.path.exists(fn):
		img.png(fn, scale=1, module_color=(255,) * 3, background=(0,) * 4)
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


def rainbow_gif2(image, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
			channels[0] = channels[0].point(lambda x: round_random((f / length / scale * loops % 1) * 256) + x & 255)
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
	if duration < 0:
		rate = -rate
	count = 256 // abs(rate)

	# Repeatedly hueshift image and return copies
	def rainbow_gif_iterator(image):
		hue = channels[0]
		for i in range(0, 256, abs(rate)):
			if i:
				channels[0] = hue.point(lambda x: i + x & 255)
				image = Image.merge("HSV", channels).convert("RGBA")
				if A is not None:
					image.putalpha(A)
			yield image

	return dict(duration=1000 / fps * count, count=count, frames=rainbow_gif_iterator(image))


def shiny_gif2(image, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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

	def shiny_gif_iterator(image):
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
			func = lambda x: round_random((abs((f / length / scale * loops / 3) % (1 / 3) - 1 / 6) - 1 / 12) * 256) + x & 255
			channels[0] = channels[0].point(func)
			temp = Image.merge("HSV", channels).convert("RGB")
			if A:
				temp.putalpha(A)
			yield temp

	return dict(duration=total * scale, count=length * scale, frames=shiny_gif_iterator(image))

def shiny_gif(image, duration):
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return shiny_gif2(image, duration)
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
	if duration < 0:
		rate = -rate
	count = 256 // abs(rate)

	# Repeatedly hueshift image and return copies
	def shiny_gif_iterator(image):
		hue = channels[0].copy()
		for i in range(0, 256, abs(rate)):
			func = lambda x: round_random((abs(i / 256 / 3 % (1 / 3) - 1 / 6) - 1 / 12) * 256) + x & 255
			channels[0] = hue.point(func)
			image = Image.merge("HSV", channels).convert("RGBA")
			if i < 32:
				if i < 8:
					pass
			if A is not None:
				image.putalpha(A)
			yield image

	return dict(duration=1000 / fps * count, count=count, frames=shiny_gif_iterator(image))


def pet_gif2(image, squeeze, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
	pet = get_image("https://mizabot.xyz/u/EBjYrqCEUDw")
	count = len(pet._images)
	iters = round(length * scale / count)

	def pet_gif_iterator(image):
		lastpet = 0
		for f in range(length * scale):
			image.seek(f % length)
			w, h = image.width * 2.5, image.height * 2.5
			if w < 256 and h < 256:
				w, h = max_size(w, h, 256, force=True)
			w, h = round_random(w), round_random(h)
			im = Image.new("RGBA", (w, h))
			sqr = (1 - cos(f / length / scale * iters * tau)) * 2.5
			wm = 0.8 + sqr * 0.02
			hm = 0.8 - sqr * 0.05
			ox = (1 - wm) * 0.5 + 0.1
			oy = (1 - hm) - 0.08
			im.paste(image.resize((round_random(wm * w), round_random(hm * h)), resample=Resampling.LANCZOS), (round_random(ox * w), round_random(oy * h)))
			lastpet = max(round_random(f / length / scale * loops * count), lastpet)
			pet2 = pet._images[lastpet % count].resize((w, h), resample=Resampling.LANCZOS)
			im.paste(pet2, mask=pet2)
			yield im.resize((round(w / 2), round(h / 2)), resample=Resampling.LANCZOS)

	return dict(duration=total * scale, count=length * scale, frames=pet_gif_iterator(image))

def pet_gif(image, squeeze, duration):
	try:
		image.seek(1)
	except EOFError:
		image.seek(0)
	else:
		return pet_gif2(image, squeeze, duration)
	pet = get_image("https://mizabot.xyz/u/EBjYrqCEUDw")
	count = len(pet._images)

	def pet_gif_iterator(image):
		w, h = image.width * 2.5, image.height * 2.5
		if w < 256 and h < 256:
			w, h = max_size(w, h, 256, force=True)
		w, h = round_random(w), round_random(h)
		for f in range(count):
			im = Image.new("RGBA", (w, h))
			sqr = (1 - cos(f / count * tau)) * 2.5
			wm = 0.8 + sqr * 0.02
			hm = 0.8 - sqr * 0.05
			ox = (1 - wm) * 0.5 + 0.1
			oy = (1 - hm) - 0.08
			im.paste(image.resize((round_random(wm * w), round_random(hm * h)), resample=Resampling.LANCZOS), (round_random(ox * w), round_random(oy * h)))
			pet2 = pet._images[f].resize((w, h), resample=Resampling.LANCZOS)
			im.paste(pet2, mask=pet2)
			yield im.resize((round(w / 2), round(h / 2)), resample=Resampling.LANCZOS)

	return dict(duration=1000 * duration, count=count, frames=pet_gif_iterator(image))


def spin_gif2(image, duration):
	total = 0
	for f in range(2147483648):
		try:
			image.seek(f)
		except EOFError:
			break
		total += max(image.info.get("duration", 0), 50)
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
		diameter = min(diameter, round(2048 / scale2))
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
	while fps >= 64 and rate <= 64:
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
		diameter = min(diameter, round(2048 / scale))
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
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
	fps = f / total * 1000
	if fps < 24:
		step = fps / 24
		image = ImageSequence(*ImageOpIterator(image, step=step))
		f = len(image)
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
			# A = out.getchannel("A")
			# A.point(lambda x: round(x * amount))
			# out.putalpha(A)
			out = Image.alpha_composite(image, out)
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