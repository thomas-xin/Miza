from main import *
import PIL, colorsys, io
from PIL import Image, ImageDraw, ImageMath, ImageOps, ImageChops, ImageFont
Resampling = getattr(Image, "Resampling", Image)

print = lambda *args, sep=" ", end="\n": sys.__stderr__.write(str(sep).join(str(i) for i in args) + str(end))


# Read input from the main process, set screen size and decide what to use as the particle
screensize = [int(x) for x in sys.argv[2:4]]
barcount = int(sys.argv[4])
highest_note = int(sys.argv[5])
particles = sys.argv[1].casefold()
IMAGE = None
try:
	particles = eval(particles)
except (SyntaxError, NameError):
	if particles in ("bar", "piano"):
		pid = 1
	elif particles in ("bubble", "hexagon"):
		pid = 2
	elif particles == "trail":
		pid = 3
	elif is_url(particles):
		import requests
		IMAGE = Image.open(io.BytesIO(requests.get(particles).content))
		pid = -2
	else:
		try:
			IMAGE = Image.open(particles)
			pid = -1
		except (OSError, FileNotFoundError):
			raise TypeError(f"Unable to find particle image file: <{particles}>")
# If the particle image was specified, convert to a greyscale image, taking alpha values into account with a black background (since we are using an additive-like blending operation)
if IMAGE:
	if "A" in str(IMAGE.mode):
		A = IMAGE.getchannel("A")
	else:
		A = None
	if str(IMAGE.mode) != "L":
		IMAGE = ImageOps.grayscale(IMAGE)
	if A:
		IMAGE = ImageChops.multiply(IMAGE, A)
	if IMAGE.getextrema() == 255:
		IMAGE = IMAGE.point(lambda x: x if x < 255 else 254)


halve = (np.arange(1, 257) >> 1).astype(np.uint8)
darken = np.concatenate((np.zeros(128, dtype=np.uint8), np.arange(128, dtype=np.uint8)))

def overflow(image, sprite):
	# Custom "overflow" blend operation, similar to additive but taking spectral sensitivity into account and scaling up other channels instead of simply capping at 255
	# This results in colours like two pure reds adding to a pink, instead of the same red
	out = list(image)
	for i, s, I in zip(range(3), sprite, image):
		if not s:
			continue
		if out[i] == I:
			out[i] = ImageChops.add(I, s)
		else:
			out[i] = ImageChops.add(out[i], s)
		# As the triple add is an expensive operation, only calculate it if it would be required
		extrema = out[i].getextrema()
		if extrema[-1] == 255:
			# The value to add to other channels is currently (x / 2 + y / 2) - 128
			im1 = I.point(halve)
			im2 = s.point(halve)
			overflow = ImageChops.add(im1, im2)
			overflow = overflow.point(darken)
			extrema = overflow.getextrema()
			# If the output scaled channels have values in them, place the resulting scaled images back into output
			if extrema[-1]:
				for j in (x for x in range(3) if x != i):
					out[j] = ImageChops.add(out[j], overflow)
	return Image.merge("RGB", out)


# This was originally only for bubble particles but I was too lazy to change the variable name
CIRCLES = {}

class Circles:

	@classmethod
	def sprite(cls, radius, colour, angle=0):
		if angle:
			tup = (radius, colour, angle)
		else:
			tup = (radius, colour)
		try:
			# Use cached sprite if possible
			return CIRCLES[tup]
		except KeyError:
			# Otherwise attempt to create a new sprite and add to the cache, deleting the oldest one if the system has run out of memory
			while True:
				try:
					if len(CIRCLES) >= 65536:
						raise MemoryError
					try:
						# Attempt to pull out cached greyscale image of a certain size, generating a new one if required
						surf = CIRCLES[radius]
					except KeyError:
						if particles == "trail":
							r = radius + 2
							h = 8
							s = 8
							surf = Image.new("L", (r * h + r, r * 2), 0)
							draw = ImageDraw.Draw(surf)
							for i in range(1, radius * s):
								c = round(i / s)
								draw.ellipse((r * h - c - round(i / s * h), r - c, r * h + c - round(i / s * h), r + c), round((i / s * 223) / (radius - 1)))
							draw.ellipse((3, 3, r * 2 - 3, r * 2 - 3), 254)
						elif IMAGE:
							# If the particle type is a source image, resize it to the required size
							surf = IMAGE.resize((radius << 1,) * 2, resample=Resampling.LANCZOS)
						else:
							# Otherwise draw a series of concentric hexagons or circles to produce particle shape
							r = radius + 2
							surf = Image.new("L", (r * 2,) * 2, 0)
							draw = ImageDraw.Draw(surf)
							for c in range(1, radius):
								# Radii of shapes are a bit finnicky to manage in order to ensure that they fit together with no gaps
								if particles == "hexagon":
									draw.regular_polygon((radius, radius, r - c), 6, 0, (c * 192) // (radius - 1) + 63 if c > 5 else c * 51, None)
								else:
									draw.ellipse((r - c, r - c, r + c, r + c), None, (c * 254) // (radius - 1), 2)
							if particles == "bubble":
								draw.ellipse((3, 3, r * 2 - 3, r * 2 - 3), None, 192, 1)
						CIRCLES[radius] = surf
					if angle:
						surf = surf.rotate(angle, resample=Resampling.BICUBIC, expand=True)
					# Convert greyscale to colour image by scaling RGB channels as required
					curr = cdict(size=surf.size)
					for c, v in zip("RGB", colour):
						if v == 255:
							curr[c] = surf
						elif v:
							curr[c] = surf.point((np.arange(256) * v / 255).astype(np.uint8))
					CIRCLES[tup] = curr
					return curr
				except MemoryError:
					# Upon running out of memory, remove the oldest cached image
					try:
						temp = CIRCLES.pop(next(iter(CIRCLES)), None)
					except RuntimeError:
						pass
					else:
						if not isinstance(temp, dict):
							temp.close()
							del temp
						else:
							for v in temp.values():
								if v and isinstance(v, Image.Image):
									v.close()
								del v
							del temp


# Main particle renderer
PARTICLES = set()
P_ORDER = 0
TICK = 0

class Particles:

	def __init__(self):
		# Determine particle type to use
		self.Particle = Particle = (None, Bar, Bubble, Trail)[pid]
		# Use an index count equal to half the screen height (one position every two pixels)
		s2 = screensize[1] >> 1 if Particle != Bar else barcount << 1
		# Use an appropriate buffer for exponentially decreasing values, similar to how typical audio volume bars are designed
		if Particle == Bar:
			self.bars = [Bar(i) for i in range(s2 + 1 >> 1)]
		elif Particle in (Bubble, Trail):
			# Calculate array of hues to render particles as
			self.colours = [tuple(round(x * 255) for x in colorsys.hsv_to_rgb(i / s2, 1, 1)) for i in range(s2 + 2)]
			self.hits = np.zeros(s2 + 3 >> 1, dtype=float)

	def animate(self, spawn):
		Particle = self.Particle
		if Particle:
			# Initialize next frame to be an empty black image
			sfx = Image.new("RGB", screensize, (0,) * 3)
			if particles == "piano":
				globals()["DRAW"] = ImageDraw.Draw(sfx)
			if Particle == Bar:
				# Raise bars to their current values as required
				for i, pwr in enumerate(spawn):
					# print(len(spawn), len(self.bars))
					self.bars[i].ensure(pwr * 24)
				# Display and update bars
				for bar in self.bars:
					bar.render(sfx=sfx)
				highbars = sorted(self.bars, key=lambda bar: bar.height, reverse=True)[:32]
				high = highbars[0]
				for bar in reversed(highbars):
					bar.post_render(sfx=sfx, scale=bar.height / max(1, high.height))
				for bar in self.bars:
					bar.update()
			elif Particle in (Bubble, Trail):
				# Calculate appropriate particle size and colour, create particle at corresponding pixel positions
				mins = screensize[0] / 1048576
				minp = screensize[0] / 16384
				np.multiply(self.hits, 63 / 64, out=self.hits)
				for x in sorted(range(len(spawn)), key=lambda z: -spawn[z])[:64]:
					pwr = spawn[x]
					if pwr >= mins and pwr >= self.hits[x >> 1] * 1.5:
						self.hits[x >> 1] = pwr
						pwr /= 1.5
						pwr += 1 / 64
						p = Particle((screensize[0], x * 4 + 2), colour=self.colours[x << 1], intensity=pwr)
						PARTICLES.add(p)
				# Display and update particles
				# for particle in PARTICLES:
				for particle in sorted(PARTICLES, key=lambda p: getattr(p, "order", 0)):
					particle.render(sfx=sfx)
					particle.update()
		# print(len(CIRCLES))
		return sfx.tobytes()

	def start(self):
		global TICK
		# Byte count, we are reading one 4 byte float per pixel of screen height
		count = screensize[1] // 4 << 2 if pid != 1 else barcount << 2
		while True:
			# Read input amplitude array for current frame, render and animate it, then write output data back to main process
			arr = np.frombuffer(sys.stdin.buffer.read(count), dtype=np.float32)
			if not len(arr):
				break
			temp = self.animate(arr)
			TICK += 1
			sys.stdout.buffer.write(temp)


# Default (abstract) particle class, does nothing
class Particle(collections.abc.Hashable):

	__slots__ = ("hash", "order")

	def __init__(self):
		global P_ORDER
		self.order = P_ORDER
		P_ORDER += 1
		self.hash = random.randint(-2147483648, 2147483647)

	__hash__ = lambda self: self.hash
	update = lambda self: None
	render = lambda self, surf: None


# Bar particle class, simulates an exponentially decreasing bar with a gradient
class Bar(Particle):

	__slots__ = ("y", "colour", "width", "height", "surf", "line")

	font = ImageFont.truetype(f"{PATH}/Pacifico.ttf", 12)

	def __init__(self, x):
		super().__init__()
		dark = False
		self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / barcount, 1, 1))
		self.y = round(screensize[1] / barcount * x)
		self.width = min(screensize[1], round(screensize[1] / barcount * (x + 1))) - self.y
		if particles == "bar":
			if x & 1:
				dark = True
			self.line = Image.new("RGB", (1, self.width), 16777215)
		else:
			note = highest_note - x + 9
			if note % 12 in (1, 3, 6, 8, 10):
				dark = True
			name = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[note % 12]
			octave = note // 12
			self.line = name + str(octave)
			# print(self.line)
		if dark:
			self.colour = tuple(i + 1 >> 1 for i in self.colour)
			self.surf = Image.new("RGB", (3, 1), self.colour)
		else:
			self.surf = Image.new("RGB", (2, 1), self.colour)
		self.surf.putpixel((0, 0), 0)
		self.height = 0

	def update(self):
		if self.height:
			if type(self.line) is str:
				self.height = self.height * 0.93 - 1
			else:
				self.height = self.height * 0.97 - 1
			if self.height < 0:
				self.height = 0

	def ensure(self, value):
		if self.height < value:
			self.height = value

	def render(self, sfx, **void):
		size = round(self.height)
		if size:
			# Resize gradient to form a bar, pasting onto the current frame
			surf = self.surf.resize((size, self.width), resample=Resampling.BILINEAR)
			sfx.paste(surf, (screensize[0] - size, self.y))
			if type(self.line) is not str:
				pos = max(0, screensize[0] - size)
				sfx.paste(self.line, (pos, self.y))

	def post_render(self, sfx, scale, **void):
		size = round(self.height)
		if size > 8:
			if type(self.line) is str:
				y = self.y + (self.width >> 1) - 12
				try:
					width = DRAW.textlength(self.line, self.font)
				except (TypeError, AttributeError):
					width = 8 * len(self.line)
				pos = max(0, screensize[0] - size - width)
				factor = round(255 * scale)
				col = sum(factor << (i << 3) for i in range(3))
				DRAW.text((pos, y), self.line, col, self.font)


# Bubble particle class, simulates a growing and fading particle that floats in a random direction upon spawning
class Bubble(Particle):

	__slots__ = ("pos", "vel", "colour", "intensity", "tick", "rotation", "rotvel")

	def __init__(self, pos, colour=(255, 255, 255), intensity=1, speed=-3, spread=1.6):
		super().__init__()
		self.pos = np.array(pos, dtype=float)
		if speed:
			angle = (random.random() - 0.5) * spread
			self.vel = np.array([speed * cos(angle), speed * sin(angle)])
			self.rotvel = (random.random() - 0.5)
		else:
			self.vel = 0
			self.rotvel = 0
		self.rotation = -angle if pid < 0 or particles == "hexagon" else None
		self.colour = colour
		self.intensity = intensity
		self.tick = 0.

	def update(self):
		if issubclass(type(self.vel), collections.abc.Sized) and len(self.vel):
			self.pos += self.vel
		maxsize = min(256, sqrt(self.intensity) * 16)
		if self.tick >= maxsize:
			self.intensity *= 0.93
			if self.intensity <= 1 / 64:
				self.intensity = 0
				PARTICLES.discard(self)
		else:
			self.tick += sqrt(maxsize) / 4
		if particles == "hexagon":
			self.rotation += self.rotvel
			self.rotvel *= 0.95

	def render(self, sfx, **void):
		intensity = min(self.intensity / 16, 3)
		colour = [x * intensity / screensize[0] * 256 for x in self.colour]
		for i, x in enumerate(colour):
			if x > 255:
				temp = (x - 256) / 4
				colour[i] = 255
				for j in range(len(colour)):
					if i != j:
						colour[j] = min(255, temp + colour[j])
		if self.rotation:
			sprite = Circles.sprite(max(4, int(self.tick)), tuple(min(255, round(x / 8) << 3) for x in colour), round(self.rotation % tau * 30 / pi) * 6)
		else:
			sprite = Circles.sprite(max(4, int(self.tick)), tuple(min(255, round(x / 8) << 3) for x in colour))
		if len(sprite) > 1:
			size = sprite.size
			offs = [x >> 1 for x in size]
			pos = tuple(int(x) for x in self.pos - offs)
			crop = sfx.crop(pos + tuple(pos[i] + size[i] for i in range(2)))
			if crop.size[0] and crop.size[1]:
				if crop.size != size:
					sprite = {k: v.crop((0,) * 2 + crop.size) for k, v in sprite.items() if type(v) is Image.Image}
				spriteit = (sprite.get(k) for k in "RGB")
				result = overflow(crop.split(), spriteit)
				# Paste clipped output area onto current frame as a sprite
				sfx.paste(result, pos)


class Trail(Particle):

	__slots__ = ("pos", "vel", "colour", "intensity", "tick", "angle")

	def __init__(self, pos, colour=(255, 255, 255), intensity=1, speed=-3, spread=0.2):
		super().__init__()
		self.pos = np.array(pos, dtype=float)
		if speed:
			self.angle = (random.random() - 0.5) * spread
			self.vel = np.array([speed * cos(self.angle), speed * sin(self.angle)])
		else:
			self.vel = 0
		self.colour = colour
		self.intensity = intensity * 3
		self.tick = 0

	def update(self):
		if issubclass(type(self.vel), collections.abc.Sized) and len(self.vel):
			self.pos += self.vel
		if self.pos[0] < -self.intensity * 16:
			PARTICLES.discard(self)
		self.tick += 1

	def render(self, sfx, **void):
		intensity = min(self.intensity / 48, 3)
		colour = [x * intensity / screensize[0] * 256 for x in self.colour]
		for i, x in enumerate(colour):
			if x > 255:
				temp = (x - 256) / 4
				colour[i] = 255
				for j in range(len(colour)):
					if i != j:
						colour[j] = min(255, temp + colour[j])
		sprite = Circles.sprite(max(4, round(self.intensity)), tuple(min(255, round(x / 8) << 3) for x in colour), round(-self.angle % tau * 30 / pi) * 6)
		if len(sprite) > 1:
			size = sprite.size
			offs = (-size[0] >> 5, -size[1] >> 1)
			pos = tuple(int(x) for x in self.pos + offs)
			crop = sfx.crop(pos + tuple(pos[i] + size[i] for i in range(2)))
			if crop.size[0] and crop.size[1]:
				if crop.size != size:
					sprite = {k: v.crop((0,) * 2 + crop.size) for k, v in sprite.items() if type(v) is Image.Image}
				spriteit = (sprite.get(k) for k in "RGB")
				result = overflow(crop.split(), spriteit)
				# Paste clipped output area onto current frame as a sprite
				sfx.paste(result, pos)


if __name__ == "__main__":
	engine = Particles()
	engine.start()