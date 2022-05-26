#!/usr/bin/env python

import os, sys

python = [sys.executable]
arg = sys.argv[0].replace("\\", "/")
if "/" in arg:
	PATH = os.path.join(os.getcwd(), arg.rsplit("/", 1)[0])
else:
	PATH = "."

from install_update import traceback, subprocess
if os.name == "nt":
	os.system("color")
import numpy, time, psutil, collections, random, contextlib, re, itertools, concurrent.futures
suppress = contextlib.suppress
from math import *
from PIL import Image
if __name__ == "__main__":
	# Requires a thread pool to manage concurrent pipes
	exc = concurrent.futures.ThreadPoolExecutor(max_workers=8)
	import requests

np = numpy
deque = collections.deque

# Simple function to detect URLs
url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s`|\"'\\])>]+$")
is_url = lambda url: url_match.search(url)

# Simple function to evaluate json inputs
jdict = dict(true=True, false=False, null=None, none=None)
globals().update(jdict)
eval_json = lambda s: eval(s, jdict, {})


# Simple dictionary-like hash table that additionally allows class-like indexing
class cdict(dict):

	__slots__ = ()

	__init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
	__repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
	__str__ = lambda self: super().__repr__()
	__iter__ = lambda self: iter(tuple(super().__iter__()))
	__call__ = lambda self, k: self.__getitem__(k)

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not k.startswith("__") or not k.endswith("__"):
			try:
				return self.__getitem__(k)
			except KeyError as ex:
				raise AttributeError(*ex.args)
		raise AttributeError(k)

	def __setattr__(self, k, v):
		if k.startswith("__") and k.endswith("__"):
			return object.__setattr__(self, k, v)
		return self.__setitem__(k, v)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(self)
		return data

	@property
	def __dict__(self):
		return self

	___repr__ = lambda self: super().__repr__()
	to_dict = lambda self: dict(**self)
	to_list = lambda self: list(super().values())


def rgb_split(image, dtype=np.uint8):
	channels = None
	if "RGB" not in str(image.mode):
		if str(image.mode) == "L":
			channels = [np.asarray(image, dtype=dtype)] * 3
		else:
			image = image.convert("RGB")
	if channels is None:
		a = np.asarray(image, dtype=dtype)
		channels = np.moveaxis(a, -1, 0)[:3]
	return channels

def hsv_split(image, convert=True, partial=False, dtype=np.uint8):
	channels = rgb_split(image, dtype=np.uint32)
	R, G, B = channels
	m = np.min(channels, 0)
	M = np.max(channels, 0)
	C = M - m #chroma
	Cmsk = C != 0

	# Hue
	H = np.zeros(R.shape, dtype=np.float32)
	for i, colour in enumerate(channels):
		mask = (M == colour) & Cmsk
		hm = channels[i - 2][mask].astype(np.float32)
		hm -= channels[i - 1][mask]
		hm /= C[mask]
		if i:
			hm += i << 1
		H[mask] = hm
	H *= 256 / 6
	H = H.astype(dtype)

	if partial:
		return H, M, m, C, Cmsk, channels

	# Saturation
	S = np.zeros(R.shape, dtype=dtype)
	# S = np.full(R.shape, 255, dtype=dtype)
	Mmsk = M != 0
	S[Mmsk] = np.clip(256 * C[Mmsk] // M[Mmsk], None, 255)

	# Value
	V = M.astype(dtype)

	out = [H, S, V]
	if convert:
		out = list(fromarray(a, "L") for a in out)
	return out

def hsl_split(image, convert=True, dtype=np.uint8):
	H, M, m, C, Cmsk, channels = hsv_split(image, partial=True, dtype=dtype)

	# Luminance
	L = np.mean((M, m), 0, dtype=np.int32)

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
	I = np.mean(channels, 0, dtype=np.float32).astype(dtype)

	# Saturation
	S = np.zeros(H.shape, dtype=dtype)
	Imsk = I != 0
	S[Imsk] = 255 - np.clip((m[Imsk] << 8) // I[Imsk], None, 255)

	out = [H, S, I]
	if convert:
		out = list(fromarray(a, "L") for a in out)
	return out

def rgb_merge(R, G, B, convert=True):
	out = np.empty(R.shape + (3,), dtype=np.uint8)
	outT = np.moveaxis(out, -1, 0)
	outT[:] = [np.clip(a, None, 255) for a in (R, G, B)]
	if convert:
		out = fromarray(out, "RGB")
	return out

def hsv_merge(H, S, V, convert=True):
	return hsl_merge(H, S, V, convert, value=True)

def hsl_merge(H, S, L, convert=True, value=False, intensity=False):
	S = np.asarray(S, dtype=np.float32)
	S *= 1 / 255
	np.clip(S, None, 1, out=S)
	L = np.asarray(L, dtype=np.float32)
	L *= 1 / 255
	np.clip(L, None, 1, out=L)
	H = np.asarray(H, dtype=np.uint8)

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


if __name__ == "__main__":

	# Converts an amount of seconds into a time display {days:hours:minutes:seconds}
	def time_disp(s):
		if not isfinite(s):
			return str(s)
		s = round(s)
		output = str(s % 60)
		if len(output) < 2:
			output = "0" + output
		if s >= 60:
			temp = str((s // 60) % 60)
			if len(temp) < 2 and s >= 3600:
				temp = "0" + temp
			output = temp + ":" + output
			if s >= 3600:
				temp = str((s // 3600) % 24)
				if len(temp) < 2 and s >= 86400:
					temp = "0" + temp
				output = temp + ":" + output
				if s >= 86400:
					output = str(s // 86400) + ":" + output
		else:
			output = "0:" + output
		return output


	# Maps colours to their respective terminal escape codes
	C = COLOURS = cdict(
		red="\x1b[38;5;196m",
		orange="\x1b[38;5;208m",
		yellow="\x1b[38;5;226m",
		chartreuse="\x1b[38;5;118m",
		green="\x1b[38;5;46m",
		spring_green="\x1b[38;5;48m",
		cyan="\x1b[38;5;51m",
		azure="\x1b[38;5;33m",
		blue="\x1b[38;5;21m",
		violet="\x1b[38;5;93m",
		magenta="\x1b[38;5;201m",
		rose="\x1b[38;5;198m",
		black="\u001b[30m",
		white="\u001b[37m",
		reset="\u001b[0m",
	)
	# Removes terminal colour codes from text
	def nocol(s):
		for i in C.values():
			s = s.replace(i, "")
		return s

	# Generates a progress bar of terminal escape codes and various block characters
	bar = "∙░▒▓█"
	col = [C.red, C.orange, C.yellow, C.chartreuse, C.green, C.spring_green, C.cyan, C.azure, C.blue, C.violet, C.magenta, C.rose]
	def create_progress_bar(ratio, length=32, offset=None):
		# there are 4 possible characters for every position, meaning that the value of a single bar is 4
		high = length * 4
		position = min(high, round(ratio * high))
		items = deque()
		if offset is not None:
			offset = round(offset * len(col))
		for i in range(length):
			new = min(4, position)
			if offset is not None:
				items.append(col[offset % len(col)])
				offset += 1
			items.append(bar[new])
			position -= new
		return "".join(items)


	# Default settings for the program
	dest = None
	sample_rate = 48000
	fps = 30
	amplitude = 0.1
	smudge_ratio = 0.9
	render = display = particles = play = image = 0
	higher_bound = lower_bound = None
	skip = 1
	speed = resolution = 1
	screensize = size = (1280, 720)


	# Main class that implements the program's functionality
	class Render:

		def __init__(self, f_in):
			global higher_bound, lower_bound
			# Cutoff between particle and spectrum display is 1/4 of the screen
			self.cutoff = screensize[0] >> 2
			# Start ffmpeg process to calculate single precision float samples from the audio if required
			opt = "-n"
			if os.path.exists(f2):
				if is_url(source) or os.path.getmtime(f2) < max(os.path.getctime(source), os.path.getmtime(source)):
					opt = "-y"
			args = ["ffmpeg", opt, "-hide_banner", "-loglevel", "error", "-i", f_in, "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", f2]
			print(" ".join(args))
			fut1 = exc.submit(psutil.Popen, args, stderr=subprocess.PIPE)
			# Start ffmpeg process to convert audio to wav if required
			opt = "-n"
			if os.path.exists(f3):
				if is_url(source) or os.path.getmtime(f3) < max(os.path.getctime(source), os.path.getmtime(source)):
					opt = "-y"
			args = ["ffmpeg", opt, "-hide_banner", "-loglevel", "error", "-i", f_in, "-f", "wav", f3]
			print(" ".join(args))
			proc = psutil.Popen(args, stderr=subprocess.PIPE)
			# Wait for wav file to appear before continuing
			try:
				fl = os.path.getsize(f3)
			except FileNotFoundError:
				fl = 0
			while fl < res_scale:
				if not proc.is_running():
					err = proc.stderr.read().decode("utf-8", "replace")
					if err:
						ex = RuntimeError(err)
					else:
						ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
					raise ex
				time.sleep(0.1)
				try:
					fl = os.path.getsize(f3)
				except FileNotFoundError:
					fl = 0
			if render:
				# Start ffmpeg process to convert output bitmap images and wav audio into a mp4 video
				args = ["ffmpeg", "-y", "-hwaccel", "auto", "-hide_banner", "-loglevel", "error", "-r", str(fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "x".join(str(i) for i in screensize), "-i", "-"]
				# if play:
				args.extend(("-f", "wav", "-i", f3, "-b:a", "256k"))
				args.extend(("-c:v", "h264", "-crf", "20"))
				if not skip:
					d = round((screensize[0] - self.cutoff) / speed / fps * 1000)
					args.extend(("-af", f"adelay=delays={d}:all=1"))
				args.append(f4)
				print(" ".join(args))
				fut2 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE)
			if display:
				# Start python process running display.py to display the preview
				args = python + [f"{PATH}/display.py", *[str(x) for x in screensize]]
				print(" ".join(args))
				fut3 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
			if not higher_bound:
				higher_bound = "F#10"
			if str(higher_bound).isnumeric():
				highest_note = int(higher_bound)
			else:
				highest_note = "C~D~EF~G~A~B".index(higher_bound[0].upper()) - 9 + ("#" in higher_bound)
				while higher_bound[0] not in "0123456789-":
					higher_bound = higher_bound[1:]
					if not higher_bound:
						raise ValueError("Octave not found.")
				highest_note += int(higher_bound) * 12
			if not lower_bound:
				lower_bound = "A0"
			if str(lower_bound).isnumeric():
				lowest_note = int(lower_bound)
			else:
				lowest_note = "C~D~EF~G~A~B".index(lower_bound[0].upper()) - 9 + ("#" in lower_bound)
				while lower_bound[0] not in "0123456789-":
					lower_bound = lower_bound[1:]
					if not lower_bound:
						raise ValueError("Octave not found.")
				lowest_note += int(lower_bound) * 12
			maxfreq = 27.5 * 2 ** ((highest_note + 0.5) / 12)
			minfreq = 27.5 * 2 ** ((lowest_note - 0.5) / 12)
			globals()["barcount"] = int(highest_note - lowest_note) + 1
			freqmul = 1 / (1 - log(minfreq, maxfreq))
			print(maxfreq, minfreq, freqmul)
			if particles:
				# Start python process running particles.py to render the particles using amplitude sample data
				args = python + [f"{PATH}/particles.py", str(particles), str(self.cutoff), str(screensize[1]), str(barcount), str(highest_note)]
				print(" ".join(args))
				fut4 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
			if play:
				# Start ffmpeg and ffplay piping 16 bit int samples to simulate audio being played
				args = ["ffplay", "-loglevel", "error", "-hide_banner", "-nodisp", "-autoexit", "-f", "s16le", "-ar", "48k", "-ac", "2", "-i", "-"]
				print(" ".join(args))
				fut5 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE)
				args = ["ffmpeg", "-loglevel", "error", "-i", f3, "-f", "s16le", "-ar", "48k", "-ac", "2", "-"]
				print(" ".join(args))
				fut6 = exc.submit(psutil.Popen, args, stdout=subprocess.PIPE)
			# Buffer to store events waiting on audio playback
			self.player_buffer = None
			self.effects = deque()
			self.glow_buffer = deque()
			# Buffer to store empty input data (in single precision floating point)
			self.emptybuff = np.zeros(res_scale, dtype=np.float32)
			# Initialize main input data to be empty
			self.expanded = np.zeros(res_scale, dtype=np.float32)
			self.buffer = self.emptybuff#[:sample_rate // fps]
			# Size of the discrete fourier transform of the input frames
			dfts = (res_scale >> 1) + 1
			# Frequency list of the fast fourier transform algorithm output
			self.fff = np.fft.fftfreq(res_scale, 1 / sample_rate)[:dfts]
			# FFT returns the values along a linear scale, we want the display the data as a logarithmic scale (because that's how pitch in music works)
			self.fftrans = np.zeros(dfts, dtype=int)
			for i, x in enumerate(self.fff):
				if x <= 0:
					continue
				else:
					x = round((1 - log(x, maxfreq)) * freqmul * (screensize[1] - 1))
				if x > screensize[1] - 1 or x < 0:
					continue
				self.fftrans[i] = x
			# Linearly scale amplitude data (unused)
			self.linear_scale = np.arange(screensize[1], dtype=np.float64) / screensize[1]
			# Initialize hue of output image to a vertical rainbow
			self.hue = fromarray(np.expand_dims((self.linear_scale * 256).astype(np.uint8), 0))
			# Initialize saturation of output image to be maximum
			self.sat = Image.new("L", (screensize[1], 1), 255)
			# Initialize value of output image to be maximum
			self.val = self.sat
			# Amplitude scale from config, divided by DFT size
			self.scale = ascale / dfts
			self.fut = None
			self.playing = True
			proc = fut1.result()
			# Wait for float sample file to appear before continuing
			try:
				fl = os.path.getsize(f2)
			except FileNotFoundError:
				fl = 0
			while fl < res_scale:
				if not proc.is_running():
					err = proc.stderr.read().decode("utf-8", "replace")
					if err:
						ex = RuntimeError(err)
					else:
						ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
					raise ex
				time.sleep(0.1)
				try:
					fl = os.path.getsize(f2)
				except FileNotFoundError:
					fl = 0
			# Open float sample file to read as input
			self.file = open(f2, "rb")
			# Generate blank (fully black) image to begin
			self.image = np.zeros((screensize[1], screensize[0], 3), dtype=np.uint8)
			# Use matrix transposition to paste horizontal images vertically
			self.trans = np.swapaxes(self.image, 0, 1)
			# Wait for remaining processes to start before continuing
			if render:
				self.rend = fut2.result()
			if display:
				self.disp = fut3.result()
			if particles:
				self.part = fut4.result()
				exc.submit(self.animate)
			if play:
				self.player = (fut5.result(), fut6.result())

		def animate(self):
			# Read one frame of animation from the particle renderer process, store in the buffer for retrieval and rendering when the spectrum lines on the display hit the particle display area
			shape = (self.cutoff, screensize[1], 3)
			size = np.prod(shape)
			while True:
				img = bytes()
				while len(img) < size:
					self.playing = False
					temp = self.part.stdout.read(size - len(img))
					if not temp:
						break
					img += temp
				self.playing = True
				img = np.frombuffer(img, dtype=np.uint8)
				# Must transpose image data as X and Y coordinates are swapped
				ordered = np.reshape(img, (shape[1], shape[0], shape[2]))
				if ordered.shape[0] != shape[0]:
					if ordered.shape[1] == shape[0]:
						ordered = np.swapaxes(ordered, 0, 1)
					else:
						ordered = np.swapaxes(ordered, 0, 2)
				if ordered.shape[1] != shape[1]:
					ordered = np.swapaxes(ordered, 1, 2)
				self.effects.append(ordered)

		def read(self):
			# Calculate required amount of input data to read, converting to 32 bit floats
			req = (res_scale) - len(self.buffer)
			if req > 0:
				data = self.file.read(req << 2)
				self.buffer = np.concatenate((self.buffer, np.frombuffer(data, dtype=np.float32)))
			else:
				data = True
			# Calculate required amount of input data again, in case the file was exhausted, appending empty data if required
			req = (res_scale) - len(self.buffer)
			if req > 0:
				self.buffer = np.concatenate((self.buffer, self.emptybuff[:req]))
			if not particles and not data:
				# If no data was found at all, stop the program as we have reached the end of audio input
				if all(self.buffer == 0):
					raise StopIteration
			# Calculate real fast fourier transform of input samples
			dft = np.fft.rfft(self.buffer[:res_scale])
			# Advance sample buffer by sample rate divided by output fps
			self.buffer = self.buffer[sample_rate // fps:]
			np.multiply(self.buffer, smudge_ratio, out=self.buffer)
			# Real fft algorithm returns complex numbers as polar coordinate pairs, initialize empty array to store their sums across a log scale
			arr = np.zeros(screensize[1], dtype=np.complex64)
			# This function took me way too long to find lmao, extremely useful here as there may be more than one of the same output index per input position due to the log scale
			np.add.at(arr, self.fftrans, dft)
			arr[0] = 0
			# After the addition, we no longer require the phase of the complex numbers as their waves (and thus interference) have been summed, take absolute value of data array
			amp = np.abs(arr).astype(np.float32)
			# Multiply array by amplitude scale
			amp = np.multiply(amp, self.scale * 256, out=amp)
			# amp = np.multiply(amp, 256, out=amp)
			# Saturation decreases when input is above 255, becoming fully desaturated and maxing out at 511
			sat = np.clip(511 - amp, 0, 255).astype(np.uint8)
			# Value increases as input is above 0, becoming full brightness and maxing out at 255
			val = np.clip(amp, 0, 255).astype(np.uint8)
			# Glow buffer is the brightness of the line separating particles and spectrum lines, it changes as data passes it
			if len(self.glow_buffer) >= (screensize[0] - self.cutoff) / speed:
				self.trans[self.cutoff] = self.glow_buffer.popleft()
			self.glow_buffer.append(min(255, int(sum(amp) / self.scale / 524288) + 127))
			# Write a copy of the resulting spectrum data to particles subprocess if applicable
			if particles and data:
				np.multiply(amp, 1 / 64, out=amp)
				np.clip(amp, 0, 64, out=amp)
				if getattr(self.part, "fut", None):
					self.part.fut.result()
				if str(particles) in ("bar", "piano"):
					compat = np.zeros(barcount, dtype=np.float32)
					try:
						bartrans = self.bartrans
					except AttributeError:
						bartrans = self.bartrans = (np.arange(len(amp)) * (barcount / len(amp))).astype(np.uint32)
					np.add.at(compat, bartrans, amp)
				else:
					compat = None
					c = 4
					for i in range(c):
						temp = amp[i::c]
						if compat is None:
							compat = temp
						else:
							compat[:len(temp)] += temp
				self.part.fut = exc.submit(self.part.stdin.write, compat.tobytes())
			# Convert saturation and brightness arrays into 2D arrays of length 1, to prepare them for image conversion
			self.sat = fromarray(np.expand_dims(sat, 0))
			self.val = fromarray(np.expand_dims(val, 0))
			# Merge arrays into a single HSV image, converting to RGB and extracting as a 1D array
			return hsv_merge(self.hue, self.sat, self.val, convert=False)[0]
			# return np.uint8(Image.merge("HSV", (self.hue, self.sat, self.val)).convert("RGB"))[0]

		def start(self):
			with suppress(StopIteration):
				# Default bar colour to (127, 127, 127) grey for no data
				self.trans[self.cutoff] = 127
				# List of futures to wait for at each frame
				futs = None
				# Current time in nanoseconds, timestamps to wait for every frame as well as estimate remaining render time
				ts = time.time_ns()
				timestamps = deque()
				if image:
					yvals = deque()
				for i in range(2147483648):
					# Force the first frame to calculate immediately if not yet set
					if self.fut is None:
						self.fut = exc.submit(self.read)
					# A single line of RGB pixel values from the read() method
					line = self.fut.result()
					# Signal to concurrently begin the next frame's render
					self.fut = exc.submit(self.read)
					if not skip or i >= (screensize[0] - self.cutoff) / speed:
						# Shift entire image {speed} pixels to the left
						self.trans[self.cutoff + 1:-speed] = self.trans[self.cutoff + speed + 1:]
					# If the current iteration of the loop would indicate that the spectrum lines have passed the bar, begin rendering buffered particle data
					if i >= (screensize[0] - self.cutoff) / speed:
						if particles:
							# Wait for particle render if unavailable
							while not self.effects:
								time.sleep(0.01)
								if not self.playing:
									raise StopIteration
						if getattr(self, "player", None):
							# Wait for audio playback if audio is lagging for whatever reason
							if self.player_buffer:
								self.player_buffer.result()
							self.player_buffer = exc.submit(self.play_audio)
						if particles:
							img = self.effects.popleft()
							self.trans[:self.cutoff] = img
					if not skip or i >= (screensize[0] - self.cutoff) / speed:
						# Fill right side of the image that's now been shifted across with the RGB values calculated earlier
						for x in range(speed):
							self.trans[-x - 1] = line
					elif skip:
						for x in range(speed):
							self.trans[self.cutoff + i * speed + x] = line
					if image:
						yvals.append(line)
					# Ensure that all subprocesses are functioning correctly
					for p in ("render", "display", "particles"):
						if globals().get(p):
							proc = getattr(self, p[:4], None)
							if not (proc and proc.is_running() and not proc.stdin.closed):
								raise StopIteration
					# Wait for current frame to complete for all subprocesses
					if futs:
						for fut in futs:
							fut.result()
					# Initialize work buffer for next frame
					futs = deque()
					# Enqueue rendering the display and video to the worker processes
					if not skip or i >= (screensize[0] - self.cutoff) / speed:
						# Convert entire frame's image to byte object, preparing to send to render and display subprocesses
						b = self.image.tobytes()
						for p in ("display", "render"):
							if globals().get(p):
								proc = getattr(self, p[:4], None)
								if proc:
									futs.append(exc.submit(proc.stdin.write, b))
						# Calculate current audio position, total audio duration, estimated remaining render time
						billion = 1000000000
						t = time.time_ns()
						while timestamps and timestamps[0] < t - 60 * billion:
							timestamps.popleft()
						timestamps.append(t)
						fs = os.path.getsize(f2)
						x = max(1, fs / (sample_rate // fps << 2))
						t = max(0, i - (screensize[0] - self.cutoff) / speed)
						ratio = min(1, t / x)
						rem = inf
						with suppress(OverflowError, ZeroDivisionError):
							rem = (fs / sample_rate / 4 - t / fps) / (len(timestamps) / fps / 60)
						# Display output as a progress bar on the console
						out = f"\r{C.white}|{create_progress_bar(ratio, 64, ((-t * 6 / fps) % 6 / 6))}{C.white}| ({C.green}{time_disp(t / fps)}{C.white}/{C.red}{time_disp(fs / sample_rate / 4)}{C.white}) | Estimated time remaining: {C.magenta}[{time_disp(rem)}]"
						overflow = 120 - len(nocol(out))
						out = out[:len(out) + overflow] + " " * (overflow) + C.white
						sys.stdout.buffer.write(out.encode("utf-8"))
						if play:
							# Wait until the time for the next frame
							while time.time_ns() < ts + billion / fps:
								time.sleep(0.001)
						ts = max(ts + billion / fps, time.time_ns() - billion / fps)
			if image:
				dim = list(itertools.chain(*(itertools.repeat(y, speed) for y in yvals)))
				img = np.array(dim, dtype=np.uint8)
				img = np.swapaxes(img, 0, 1)
				img = fromarray(img, mode="RGB")
				img.save(f5)
			# Close everything and exit
			self.file.close()
			if render:
				self.rend.stdin.close()
			if display:
				self.disp.stdin.close()
			if render:
				self.rend.wait()
				if self.rend.returncode:
					raise RuntimeError(self.rend.returncode)
			if display:
				self.disp.wait()
				if self.disp.returncode:
					raise RuntimeError(self.disp.returncode)
			proc = psutil.Process()
			for child in proc.children(True):
				with suppress(psutil.NoSuchProcess):
					child.kill()
			proc.kill()

		def play_audio(self):
			# Plays one frame of audio calculated by fps and sample rate
			req = int(48000 / fps * 4)
			self.player[0].stdin.write(self.player[1].stdout.read(req))


	ytdl = None
	# Get config file data, create a new one if unavailable
	if not os.path.exists(f"{PATH}/config.json"):
		data = "{" + "\n\t" + "\n\t".join((
				'"source": "", # This field may be omitted to be prompted for an input at runtime; may be a file path or URL.',
				'"dest": "", # This field may be omitted to use the source filename.'
				'"size": [1280, 720], # Both dimensions should be divisible by 4 for best results.',
				'"fps": 30, # Framerate of the output video.',
				'"sample_rate": 48000, # Sample rate to evaluate fourier transforms at, should be a multiple of fps.',
				'"amplitude": 0.1, # Amplitude to scale audio volume, adjust as necessary.',
				'"smudge_ratio": 0.9, # Redirects vertical blurriness horizontally; should be a value between 0 and 1.',
				'"speed": 2, # Speed of screen movemement in pixels per frame, does not change audio playback speed.',
				'"resolution": 192, # Resolution of DFT in bars per pixel, this should be a relatively high number due to the logarithmic scale.',
				'"lower_bound": "A0", # Lowest musical note displayed on the spectrogram, may optionally be the ID of the note, with respect to A0.',
				'"higher_bound": "F#10", # Highest musical note displayed on the spectrogram, may optionally be the ID of the note, with respect to A0.',
				'"particles": "piano", # May be one of None, "bar", "bubble", "piano", "hexagon", or a file path/URL in quotes to indicate image to use for particles.',
				'"skip": true, # Whether to seek video to when audio begins playing.',
				'"display": true, # Whether to preview the rendered video in a separate window.',
				'"render": true, # Whether to output the result to a video file.',
				'"play": true, # Whether to play the actual audio being rendered.',
				'"image": false, # Whether to render entire spectrogram as an image.',
			)) + "\n}"
		with open(f"{PATH}/config.json", "w") as f:
			f.write(data)
	else:
		with open(f"{PATH}/config.json", "rb") as f:
			data = f.read()
	config = eval_json(data)
	# Send settings to global variables (because I am lazy lol)
	globals().update(config)
	# Take command-line flags such as "-size" to interpret as setting overrides
	argv = sys.argv[1:]
	while len(argv) >= 2:
		if argv[0].startswith("-"):
			arg = argv[1]
			with suppress(SyntaxError, NameError):
				arg = eval(arg)
			s = argv[0][1:]
			config[s] = arg
			globals()[s] = arg
			argv = argv[2:]
		else:
			break
	if len(argv):
		inputs = argv
		config["source"] = argv
	else:
		if not source:
			source = input("No audio input found in config.json; please input audio file by path or URL: ")
		inputs = [source]
	print(config)
	# Calculate remaining required variables from the input settings
	screensize = size
	res_scale = resolution * screensize[1]
	ascale = amplitude * screensize[1] / 2
	sources = deque()
	futs = deque()
	# Render all input sources, initializing audio downloader if any of them are download URLs
	for path in inputs:
		if is_url(path):
			if ytdl is None:
				if re.findall("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/", path):
					title = path.split("?", 1)[0].rsplit("/", 1)[-1]
					if title.rsplit(".", 1)[-1] in ("ogg", "webm"):
						url2 = path.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
						with requests.get(url2, stream=True) as resp:
							if resp.status_code in range(200, 400):
								futs.append(url2)
								continue
							headers = {k.casefold(): v for k, v in resp.headers.items()}
						mime = headers.get("content-type", "")
						if mime.startswith("audio/") or mime.startswith("video/"):
							if mime != "audio/midi":
								futs.append(path)
								continue
				with requests.head(path, stream=True) as resp:
					headers = {k.casefold(): v for k, v in resp.headers.items()}
				mime = headers.get("content-type", "")
				if mime.startswith("audio/") or mime.startswith("video/"):
					if mime != "audio/midi":
						futs.append(path)
						continue
				from audio_downloader import AudioDownloader
				ytdl = AudioDownloader()
			futs.append(exc.submit(ytdl.extract, path))
		else:
			futs.append(path)
	for fut in futs:
		if type(fut) is not str:
			sources.extend(fut.result())
		else:
			sources.append(fut)
	if dest:
		sources = (sources[0],)
	for entry in sources:
		if type(entry) is not str:
			ytdl.extract_single(entry)
			source = entry.stream
			fn = re.sub('[\\\\/*?:"<>|]', "-", entry.name)
		else:
			source = entry
			fn = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
		if dest:
			fn = dest
		f2 = fn + ".pcm"
		f3 = fn + ".riff"
		f4 = fn + ".mp4"
		f5 = fn + ".png"
		print("Loading", source)
		print(skip, display, render, play)
		r = Render(source)
		if render:
			print("Rendering", f4)
		r.start()