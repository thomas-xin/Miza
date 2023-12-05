# Better than Curl 🥌

import os, sys, subprocess, time, math, random, concurrent.futures, urllib.request
if os.name == "nt":
	os.system("color")

from traceback import print_exc
from math import *
from concurrent.futures import thread

def _adjust_thread_count(self):
	# if idle threads are available, don't spin new threads
	try:
		if self._idle_semaphore.acquire(timeout=0):
			return
	except AttributeError:
		pass

	# When the executor gets lost, the weakref callback will wake up
	# the worker threads.
	def weakref_cb(_, q=self._work_queue):
		q.put(None)

	num_threads = len(self._threads)
	if num_threads < self._max_workers:
		thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
		t = thread.threading.Thread(
			name=thread_name,
			target=thread._worker,
			args=(
				thread.weakref.ref(self, weakref_cb),
				self._work_queue,
				self._initializer,
				self._initargs,
			),
			daemon=True
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = lambda self: _adjust_thread_count(self)

utc = time.time
math.round = round

def round(x, y=None):
	try:
		if isfinite(x):
			try:
				if x == int(x):
					return int(x)
				if y is None:
					return int(math.round(x))
			except:
				pass
			return round_min(math.round(x, y))
		else:
			return x
	except:
		pass
	if type(x) is complex:
		return round(x.real, y) + round(x.imag, y) * 1j
	try:
		return math.round(x, y)
	except:
		pass
	return x

def round_min(x):
	if type(x) is str:
		if "." in x:
			x = x.strip("0")
			if len(x) > 8:
				x = mpf(x)
			else:
				x = float(x)
		else:
			try:
				return int(x)
			except ValueError:
				return float(x)
	if type(x) is int:
		return x
	if type(x) is not complex:
		if isfinite(x):
			y = math.round(x)
			if x == y:
				return int(y)
		return x
	else:
		if x.imag == 0:
			return round_min(x.real)
		else:
			return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)


def time_disp(s, rounded=True):
	if not isfinite(s):
		return str(s)
	if rounded:
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


def header():
	return {
		"User-Agent": f"Mozilla/5.{random.randint(1, 9)} (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
		"DNT": "1",
		"X-Forwarded-For": ".".join(str(random.randint(0, 255)) for _ in range(4)),
	}


COLOURS = ["\x1b[38;5;16m█"]
COLOURS.extend(f"\x1b[38;5;{i}m█" for i in range(232, 256))
COLOURS.append("\x1b[38;5;15m█")

def download(url, fn, resp=None, index=0, start=None, end=None, tn=None):
	size = 0
	packet = 131072
	# print(index, start, end)
	if os.path.exists(fn) and end is not None and start is not None and os.path.getsize(fn) == end - start:
		progress[index] = end - start
		if resp:
			resp.close()
		return fn
	try:
		f = open(fn, "wb")
	except:
		f = open(fn, "rb+")
	err1 = None
	rheader = None
	with f:
		while tn is None or threaders[tn]:
			try:
				if not resp:
					rheader = header()
					if size or start or end:
						if err1 and size:
							print(f"Corrupted range likely, retrying chunk {index}...")
							size = 0
							f.seek(0)
						rstart = size
						if start:
							rstart += start
						rend = end - 1 if end else ""
						r = f"bytes={rstart}-{rend}"
						rheader["Range"] = r
					req = urllib.request.Request(url, headers=rheader)
					try:
						resp = urllib.request.urlopen(req)
					except:
						if err1 is not None:
							err1 = True
						print("Header:", rheader, start, end, size)
						raise
				with resp:
					if index and resp.code >= 400:
						if resp.code in (429, 500, 503):
							time.sleep(7 + random.random() * 4 + index / 2)
						else:
							globals()["attempts"] += 1
							if err1 is not None:
								err1 = True
						err = prio.submit(resp.read)
						print("Header:", rheader, start, end, size)
						raise ConnectionError(resp.code, err.result(timeout=4).decode("utf-8").rstrip())
					err1 = False
					while tn is None or threaders[tn]:
						try:
							fut = submit(resp.read, packet)
						except RuntimeError:
							print_exc()
							return
						try:
							b = fut.result(timeout=24)
						except (ValueError, AttributeError):
							raise StopIteration
						if end and len(b) + size > end:
							b = b[:end - len(b) - size]
						if not b:
							if not start or not end or start + size >= end:
								raise StopIteration
							raise EOFError(start, size, end)
						f.write(b)
						size += len(b)
						progress[index] = size
						if quiet:
							continue
						total = sum(progress.values())
						percentage = round(total / fsize * 100, 4)
						s = f"\r{percentage}%"
						box = lambda i: COLOURS[round(min(1, i) * (len(COLOURS) - 1))]
						s += " " * (10 - len(s))
						prog = "".join(box(v * threads / fsize) for v in progress.values())
						s += prog
						if verbose and prog != last_progress:
							globals()["last_progress"] = prog
							s = "\n" + s[1:]
						s += "\x1b[38;5;7m"
						print(s, end="")
			except (StopIteration, KeyboardInterrupt):
				break
			except:
				print_exc()
				time.sleep(5)
				print(f"\nThread {index} errored, retrying...")
				packet = max(8192, packet >> 1)
			resp = None
	return fn

def upload(url, fn, resp=None, index=0, start=None, end=None, tn=None):
	start = start or 0
	end = end or os.path.getsize(fn)
	with open(fn, "rb") as f:
		f.seek(start)
		data = f.read(end - start)
		while tn is None or threaders[tn]:
			try:
				rheader = header()
				rheader["x-file-name"] = fn.rsplit("/", 1)[-1]
				rheader["x-index"] = str(index)
				resp = requests.post(url, data=data, headers=rheader, timeout=32)
				# req = urllib.request.Request(url, data, headers=rheader, method="POST")
				# resp = urllib.request.urlopen(req)
				if index and resp.status_code >= 400:
					if resp.status_code in (429, 500, 503):
						time.sleep(7 + random.random() * 4 + index / 2)
					else:
						globals()["attempts"] += 1
					raise ConnectionError(resp.status_code, resp.text.rstrip())
				size = end - start
				progress[index] = size
				if quiet:
					continue
				total = sum(progress.values())
				ratio = total / fsize
				percentage = round(ratio * 100, 4)
				s = f"\r{percentage}%"
				box = lambda i: COLOURS[round(min(1, i) * (len(COLOURS) - 1))]
				s += " " * (10 - len(s))
				x = ratio * 64
				prog = box(1) * int(x) + box(x % 1)
				s += prog
				if verbose and prog != last_progress:
					globals()["last_progress"] = prog
					s = "\n" + s[1:]
				s += "\x1b[38;5;7m"
				print(s, end="")
				raise StopIteration
			except (StopIteration, KeyboardInterrupt):
				break
			except:
				print_exc()
				time.sleep(5)
				print(f"\nThread {index} errored, retrying...")
			resp = None
	return fn


uploading = verbose = notrain = quiet = False
attempts = 0
max_attempts = float("inf")
fn = None
if len(sys.argv) < 2:
	url = input("Please enter a URL to download from: ")
	threads = 0
	chunked = False
else:
	args = list(sys.argv)
	if "-v" in args:
		args.remove("-v")
		verbose = True
	if "-q" in args:
		args.remove("-q")
		quiet = True
	if "-u" in args:
		args.remove("-u")
		uploading = True
	if "-n" in args:
		args.remove("-n")
		notrain = True
	if "-threads" in args:
		i = args.index("-threads")
		threads = int(args[i + 1])
		args = args[:i] + args[i + 2:]
	else:
		threads = 0
	if "-attempts" in args:
		i = args.index("-attempts")
		max_attempts = int(args[i + 1])
		args = args[:i] + args[i + 2:]
	else:
		max_attempts = 1
	if "-c" in args:
		args.remove("-c")
		if args[-1].startswith("http") and "://" in args[-1]:
			urls = args[1:]
		else:
			urls = args[1:-1]
			fn = args[-1]
		url = urls[0]
		chunked = True
	else:
		url = args[1] if len(args) > 1 else None
		if not url:
			url = input("Please enter a URL to download from: ")
		if len(args) >= 3:
			fn = " ".join(args[2:])
		chunked = False
url = url.replace("\\", "/")


t = utc()
progress = {}

if uploading:
	try:
		import requests
	except ModuleNotFoundError:
		subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "requests"])
		import requests
	fsize = fs = os.path.getsize(url)
	threads = math.ceil(fsize / 1048576)
	load = fsize / threads
	delay = 1
	exc = concurrent.futures.ThreadPoolExecutor(max_workers=threads + 3)
	submit = exc.submit
	print(f"Splitting into {threads} threads...")
	workers = [None] * threads
	for i in range(threads):
		start = math.ceil(i * load)
		if i == threads - 1:
			end = None
		else:
			end = min(math.ceil(i * load + load), fsize)
		if start >= end:
			continue
		workers[i] = submit(upload, "https://mizabot.xyz/upload_chunk", url, None, index=i, start=start, end=end)
		try:
			workers[i].result(timeout=delay)
		except concurrent.futures.TimeoutError:
			pass
		if workers[i].done() or i >= 1 and workers[i - 1].done() or i >= 2 and workers[i - 2].done():
			delay /= 2
		else:
			delay *= math.sqrt(2)
	for fut in workers:
		fut.result()
	rheader = header()
	rheader["x-file-name"] = url.rsplit("/", 1)[-1]
	rheader["x-index"] = threads
	url = "https://mizabot.xyz/merge"
	req = urllib.request.Request(url, headers=rheader)
	resp = urllib.request.urlopen(req)
	print("\nhttps://mizabot.xyz/file" + resp.read()[2:].decode("utf-8", "replace"))
	exc.shutdown(wait=True)
	s = utc() - t
	e = ""
	bps = fs / s * 8
	if bps >= 1 << 10:
		if bps >= 1 << 20:
			if bps >= 1 << 30:
				if bps >= 1 << 40:
					e = "T"
					bps /= 1 << 40
				else:
					e = "G"
					bps /= 1 << 30
			else:
				e = "M"
				bps /= 1 << 20
		else:
			e = "k"
			bps /= 1 << 10
	bps = str(round(bps, 4)) + " " + e
	print(f"{fs} bytes successfully uploaded in {time_disp(s)}, average upload speed {bps}bps")
	raise SystemExit

import base64, hashlib
shash = lambda s: base64.urlsafe_b64encode(hashlib.md5(s if type(s) is bytes else str(s).encode("utf-8")).digest()).rstrip(b"=").decode("ascii")
if not os.path.exists("cache"):
	os.mkdir("cache")
if not os.path.exists("files"):
	os.mkdir("files")
print("Sending sampler request...")
rheader = header()
req = urllib.request.Request(url, headers=rheader)
resp = urllib.request.urlopen(req)
o_url = url
url = resp.url
head = {k.casefold(): v for k, v in resp.headers.items()}
print(head)
if verbose:
	last_progress = ""
if chunked:
	r2 = urllib.request.Request(urls[-1], headers=rheader)
	re2 = urllib.request.urlopen(r2)
	fsize = int(head.get("content-length", 1073741824)) * (len(urls) - 1) + int(re2.headers.get("content-length", 1073741824))
elif head.get("content-length"):
	fsize = int(head["content-length"])
elif head.get("content-range"):
	fsize = int(head["content-range"].rsplit("/", 1)[-1])
else:
	fsize = 1073741824
if chunked:
	threads = len(urls)
elif "bytes" in head.get("accept-ranges", ""):
	print("Accept-Ranges header found.")
	if threads < 1:
		try:
			with open("training.txt", "r", encoding="utf-8") as f:
				s = f.read()
		except FileNotFoundError:
			s = ""
		decision = {}
		for line in s.splitlines():
			spl = line.rstrip().split()
			fs, tc, tm = int(spl[0]), int(spl[1]), float(spl[2])
			try:
				ratio = decision[fs]
			except KeyError:
				ratio = decision[fs] = {}
			try:
				ratio[tc].append(tm)
			except KeyError:
				ratio[tc] = [tm]
		if decision:
			distances = ((abs(fs - fsize) / (2 + log(len(decision[fs]))), fs) for fs in decision)
			LS = sorted(distances)[0][1]
			sizes = {sum(v) / len(v) * (12 + log(k, 2)): k for k, v in decision[LS].items()}
			sizes = {k: v for k, v in sorted(sizes.items())}
			k = next(iter(sizes))
			threads = round(sizes[k] / LS * fsize)
			if LS == fsize:
				print(f"Decision tree hit: {threads}")
			else:
				print(f"Decision tree miss: {threads}")
			if random.random() >= 0.125:
				lr = max(1, round(threads / 8), round(256 / len(sizes)))
			else:
				lr = 0
			threads += random.randint(-lr, lr)
			if threads <= 1:
				threads = random.randint(1, 3)
		else:
			n = round(fsize / 4194304)
			print(f"Decision tree empty: {n}")
			threads = n
		threads = max(3, min(64, threads))
	if threads > fsize / 65536:
		threads = -(-fsize // 65536)
else:
	threads = 1
if not fn:
	fn = head.get("attachment-filename") or url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0] or "file"
	fn = "files/" + fn.rsplit("/", 1)[-1]
fn2 = fn + "~"
h = o_url + f" size={fsize} threads={threads}"
# print("Hash:", h)
# raise
FID = shash(h)
exc = concurrent.futures.ThreadPoolExecutor(max_workers=threads + 1 << 1)
prio = concurrent.futures.ThreadPoolExecutor(max_workers=8)
submit = exc.submit
threaders = {}
if threads > 1:
	print(f"Splitting into {threads} threads...")
	workers = [None] * threads
	load = fsize / threads
	delay = 1
	tt = utc() - t
	if tt < 1:
		time.sleep(1 - tt)
	tt = None
	for i in range(threads):
		if chunked:
			url = urls[i]
			tn = len(threaders)
			threaders[tn] = True
			workers[i] = submit(download, url, f"cache/${FID}-{i}", resp, index=i, start=0, end=None, tn=tn)
			workers[i].tn = tn
		else:
			start = math.ceil(i * load)
			if i == threads - 1:
				end = None
			else:
				end = min(math.ceil(i * load + load), fsize)
			tn = len(threaders)
			threaders[tn] = True
			workers[i] = submit(download, url, f"cache/${FID}-{i}", resp, index=i, start=start, end=end, tn=tn)
			workers[i].tn = tn
		resp = None
		try:
			j = max(0, i - 2)
			workers[j].result(timeout=delay)
		except concurrent.futures.TimeoutError:
			pass
		for j in range(math.ceil(threads / 4) + 8):
			if j > i:
				continue
			if workers[i - j].done():
				delay = max(0.0625, delay / 2)
				break
		else:
			delay += min(delay / 4, 2)
		if delay > 5:
			delay = 5
		time.sleep(0.5)
		fut = workers[0]
		if tt is None and fut.done():
			tt = utc() - t
		if attempts > max_attempts:
			try:
				import psutil
			except ModuleNotFoundError:
				subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "psutil"])
				import psutil
			psutil.Process().kill()
	if os.path.exists(fn2):
		os.remove(fn2)
	fi = fut.result()
	if tt is None:
		tt = utc() - t + 20
	try:
		os.rename(fi, fn2)
	except OSError:
		with open(fi, "rb") as g:
			with open(fn2, "wb") as f:
				while True:
					b = g.read(4194304)
					if not b:
						break
					f.write(b)
	with open(fn2, "ab") as f:
		for i, fut in enumerate(workers[1:]):
			if attempts > max_attempts:
				try:
					import psutil
				except ModuleNotFoundError:
					subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "psutil"])
					import psutil
				psutil.Process().kill()
			for x in range(2147483648):
				try:
					fi = fut.result(timeout=tt)
				except concurrent.futures.TimeoutError:
					if i + 2 >= len(workers) or workers[i + 2].done() or x > 2:
						print(f"Thread {i + 1} timed out, restarting...")
						threaders[workers[i + 1].tn] = False
						tt += 5
						if chunked:
							tn = len(threaders)
							threaders[tn] = True
							fut = workers[i + 1] = prio.submit(download, urls[i + 1], f"cache/${FID}-{i + 1}", None, index=i + 1, start=0, end=None, tn=tn)
							fut.tn = tn
						else:
							start = math.ceil((i + 1) * load)
							if i + 1 == threads - 1:
								end = None
							else:
								end = min(math.ceil((i + 1) * load + load), fsize)
							tn = len(threaders)
							threaders[tn] = True
							fut = workers[i + 1] = prio.submit(download, url, f"cache/${FID}-{i + 1}", None, index=i + 1, start=start, end=end, tn=tn)
							fut.tn = tn
					continue
				else:
					if x:
						tt = 20
					break
			with open(fi, "rb") as g:
				while True:
					b = g.read(4194304)
					if not b:
						break
					f.write(b)
			# try:
				# os.remove(fi)
			# except KeyboardInterrupt:
				# break
			# except:
				# pass
else:
	print("Resuming request using 1 thread...")
	download(url, fn2, resp)
s = utc() - t
if os.path.exists(fn):
	os.remove(fn)
os.rename(fn2, fn)
fs = os.path.getsize(fn)
if not notrain:
	with open("training.txt", "a", encoding="utf-8") as f:
		f.write(f"{fs} {threads} {s}\n")
e = ""
bps = fs / s * 8
if bps >= 1 << 10:
	if bps >= 1 << 20:
		if bps >= 1 << 30:
			if bps >= 1 << 40:
				e = "T"
				bps /= 1 << 40
			else:
				e = "G"
				bps /= 1 << 30
		else:
			e = "M"
			bps /= 1 << 20
	else:
		e = "k"
		bps /= 1 << 10
bps = str(round(bps, 4)) + " " + e
print(f"\n{fs} bytes successfully downloaded in {time_disp(s)}, average download speed {bps}bps")

if fs == fsize:
	import shutil
	try:
		shutil.rmtree("cache")
	except:
		pass
# try:
	# shutil.rmtree("files")
# except:
	# pass

try:
	prio.submit(exc.shutdown, wait=True).result(timeout=1)
except:
	pass
else:
	raise SystemExit(0)
print("Force killing due to frozen threads...")
try:
	import psutil
except ModuleNotFoundError:
	subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "psutil"])
	import psutil
psutil.Process().kill()
