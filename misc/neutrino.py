import os


def copyfileobj(fsrc, fdst, length=1048576, size=None):
	pos = 0
	while True:
		if size is not None:
			bufsize = min(length, size - pos)
		else:
			bufsize = length
		buf = fsrc.read(bufsize)
		if not buf:
			break
		pos += len(buf)
		fdst.write(buf)

def write_into(out, i, pos):
	with open(out, "rb+") as fo:
		if pos:
			fo.seek(pos)
		with open(i, "rb") as fi:
			copyfileobj(fi, fo)

def read_into(out, argv, pos, size):
	with open(out, "wb") as fo:
		with open(argv, "rb") as fi:
			fi.seek(pos)
			copyfileobj(fi, fo, size=size)

def filecmp(fsrc, fdst, length=1048576):
	while True:
		fut = submit(fsrc.read, length)
		y = fdst.read(length)
		x = fut.result()
		if x != y:
			return
		if not x:
			return True

def deflate(fsrc, fdst, pos, size=None):
	import zipfile
	with zipfile.ZipFile(fdst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, strict_timestamps=False) as z:
		with z.open("M", "w") as fo:
			with open(fsrc, "rb") as fi:
				fi.seek(pos)
				copyfileobj(fi, fo, size=size)
	if os.path.getsize(fdst) > size * 0.95:
		with open(fdst, "wb") as fo:
			fo.write(b"\x80")
			with open(fsrc, "rb") as fi:
				fi.seek(pos)
				copyfileobj(fi, fo, size=size)

def inflate(fsrc, fdst, pos):
	import zipfile
	with open(fdst, "rb+") as fo:
		fo.seek(pos)
		try:
			with zipfile.ZipFile(fsrc, "r") as z:
				with z.open("M", "r") as fi:
					copyfileobj(fi, fo)
		except:
			with open(fsrc, "rb") as fi:
				fi.seek(1)
				copyfileobj(fi, fo)

def ensure_compressor():
	if os.path.exists("4x4") or os.path.exists("4x4.exe"):
		return
	if os.name == "nt":
		url = "https://cdn.discordapp.com/attachments/682561514221338690/890225317090844692/4x4.zip"
	else:
		url = "https://cdn.discordapp.com/attachments/682561514221338690/890225686302851162/4x4.zip"
	import urllib.request, zipfile, io
	req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
	with urllib.request.urlopen(req) as resp:
		b = io.BytesIO(resp.read())
	with zipfile.ZipFile(b, "r") as z:
		z.extractall()
	if os.name != "nt":
		subprocess.run(("chmod", "777", "4x4"))

def encrypt(fsrc, fdst, pos, size=None, password="", total=-1, emoji=True):
	import base64, hashlib, itertools
	h = int.from_bytes(hashlib.sha512(password.encode("utf-8")).digest(), "little") + (pos + total)
	try:
		import numpy as np
	except ModuleNotFoundError:
		subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "numpy"])
		import numpy as np
	chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
	emojis = np.fromiter(itertools.chain(
		range(128512, 128568),
		range(128577, 128580),
		range(129296, 129302),
		range(129312, 129318),
		range(129319, 129328),
		range(129392, 129399),
		(129303, 129400, 129402),
	), dtype=np.uint32)
	rand = np.random.default_rng(h)
	def __encrypt(b):
		rand.shuffle(emojis)
		it = iter(emojis)
		_encrypt = {ord(c): chr(next(it)) for c in chars}
		a = np.frombuffer(b, dtype=np.uint8)
		n = np.arange(len(b), dtype=np.uint32)
		y = rand.integers(0, 256, size=len(b), dtype=np.uint8)
		r = rand.integers(0, n, dtype=np.uint32, endpoint=True)
		x = np.empty(len(b), dtype=np.uint8)
		for i, j in enumerate(r):
			if i != j:
				x[i] = x[j]
			x[j] = a[i]
		x -= y
		s = base64.b85encode(x.tobytes()).decode("ascii")
		return s.translate(_encrypt).encode("utf-8")
	i = 0
	with open(fdst, "rb+") as fo:
		fo.seek(pos * 5)
		with open(fsrc, "rb") as fi:
			fi.seek(pos)
			while True:
				count = min(65536, size - i)
				if count <= 0:
					break
				b = fi.read(count)
				if not b:
					break
				i += len(b)
				bc = __encrypt(b)
				fo.write(bc)

def decrypt(fsrc, fdst, pos, size=None, password="", total=-1, emoji=True):
	import base64, hashlib, itertools
	h = int.from_bytes(hashlib.sha512(password.encode("utf-8")).digest(), "little") + (pos + total)
	try:
		import numpy as np
	except ModuleNotFoundError:
		subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "numpy"])
		import numpy as np
	chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
	emojis = np.fromiter(itertools.chain(
		range(128512, 128568),
		range(128577, 128580),
		range(129296, 129302),
		range(129312, 129318),
		range(129319, 129328),
		range(129392, 129399),
		(129303, 129400, 129402),
	), dtype=np.uint32)
	rand = np.random.default_rng(h)
	def __decrypt(b):
		rand.shuffle(emojis)
		it = iter(emojis)
		_decrypt = {next(it): c for c in chars}
		s = b.decode("utf-8").translate(_decrypt)
		b = base64.b85decode(s.encode("ascii"))
		a = np.frombuffer(b, dtype=np.uint8)
		n = np.arange(len(b), dtype=np.uint32)
		y = rand.integers(0, 256, size=len(b), dtype=np.uint8)
		r = rand.integers(0, n, dtype=np.uint32, endpoint=True)
		x = a + y
		for i, j in zip(n[::-1], r[::-1]):
			x[i], x[j] = x[j], x[i]
		return x.tobytes()
	i = 0
	with open(fdst, "rb+") as fo:
		fo.seek(pos // 5)
		with open(fsrc, "rb") as fi:
			fi.seek(pos)
			while True:
				count = min(327680, size - i)
				if count <= 0:
					break
				b = fi.read(count)
				if not b:
					break
				i += len(b)
				bc = __decrypt(b)
				fo.write(bc)


if __name__ == "__main__":
	import time
	orig = time.time()
	import sys, collections, zipfile, io, concurrent.futures, subprocess
	from concurrent.futures import thread
	from collections import deque

	try:
		import orjson as json
	except ModuleNotFoundError:
		subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "orjson"])
		try:
			import orjson as json
		except ModuleNotFoundError:
			import json
	as_bytes = lambda s: s.encode("utf-8") if isinstance(s, str) else s

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

	ppe = concurrent.futures.ProcessPoolExecutor(max_workers=32)
	tpe = concurrent.futures.ThreadPoolExecutor(max_workers=192)
	submit = tpe.submit

	def create_file(name, size):
		try:
			f = open(name, "rb+")
		except:
			f = open(name, "wb")
		with f:
			f.truncate(size)

	this = sys.argv[0]
	lzt = None

	compress = None
	for i, a in enumerate(sys.argv):
		if a.startswith("-c"):
			a = a[2:]
			if a:
				compress = int(a)
			else:
				compress = 5
			sys.argv.pop(i)
			break
	if "-y" in sys.argv:
		sys.argv.remove("-y")
		yes = True
	else:
		yes = False
	if "-e" in sys.argv:
		sys.argv.remove("-e")
		encode = True
	else:
		encode = False
	if "-d" in sys.argv:
		sys.argv.remove("-d")
		decompress = True
	else:
		decompress = False
	try:
		i = sys.argv.index("-s")
	except ValueError:
		isize = None
	else:
		isize = int(sys.argv[i + 1])
		sys.argv.pop(i)
		sys.argv.pop(i)
	try:
		i = sys.argv.index("-f")
	except ValueError:
		target = None
	else:
		target = os.path.normcase(sys.argv[i + 1])
		sys.argv.pop(i)
		sys.argv.pop(i)
	try:
		i = sys.argv.index("--encrypt")
	except ValueError:
		encryptp = None
	else:
		encode = True
		encryptp = sys.argv[i + 1]
		sys.argv.pop(i)
		sys.argv.pop(i)
	try:
		i = sys.argv.index("--decrypt")
	except ValueError:
		decryptp = None
	else:
		encode = False
		decryptp = sys.argv[i + 1]
		sys.argv.pop(i)
		sys.argv.pop(i)
	argv = sys.argv[1] if len(sys.argv) > 1 else None

	while not argv:
		argv = input("Please enter file or folder to process: ")
	argv = argv.replace("\\", "/")

	isdir = os.path.isdir(argv)
	if encode or isdir:
		if compress:
			lzt = ppe.submit(ensure_compressor)
			lzt.result()
		out = sys.argv[2] if len(sys.argv) > 2 else argv.rsplit("/", 1)[-1] + ".wb" or "output.wb"
		if not yes and os.path.exists(out):
			if "." in out:
				out = "~1.".join(out.rsplit(".", 1))
				while os.path.exists(out):
					spl = out.rsplit(".", 1)
					spl2 = spl[0].rsplit("~", 1)
					out = spl2[0] + "~" + str(int(spl2[1]) + 1) + "." + spl[-1]
			else:
				out += "~1"
				while os.path.exists(out):
					spl = out.rsplit("~", 1)
					out = spl[0] + "~" + str(int(spl[1]) + 1)
		info = deque((deque(),))
		names = {}
		sizes = {}
		hashes = {}
		extras = deque()

		hsize = 16384

		def _get_hash(path, pos):
			with open(path, "rb") as f:
				f.seek(pos)
				return f.read()

		def get_hash(path=".", size=0):
			if size > hsize:
				pos = max(hsize, size - hsize)
				fut = submit(_get_hash, path, pos)
				y = True
			else:
				y = False
			with open(path, "rb") as f:
				x = f.read(hsize)
			if y:
				return hash(x) + hash(fut.result()) + size
			return hash(x) + size

		def recursive_scan(path=".", pos=0):
			files = deque()
			for f in os.scandir(path):
				if f.is_file(follow_symlinks=False):
					s = f.stat()
					size = s.st_size
					if size:
						if compress:
							h = get_hash(f.path, size=size)
						if compress and size in sizes:
							try:
								for f2 in sizes[size]:
									try:
										h2 = hashes[f2]
									except KeyError:
										h2 = hashes[f2] = get_hash(f2, size=size)
									if h == h2:
										with open(f.path, "rb") as fi:
											with open(f2, "rb") as fo:
												if filecmp(fi, fo):
													extras.append((os.path.relpath(f.path, argv), names[f2], size))
													raise StopIteration
							except StopIteration:
								continue
							hashes[f.path] = h
						files.append(f.path)
						info.append((os.path.relpath(f.path, argv), pos, size))
						names[f.path] = pos
						if compress:
							try:
								sizes[size].append(f.path)
							except KeyError:
								sizes[size] = deque((f.path,))
						pos += size
					else:
						extras.append((os.path.relpath(f.path, argv), 0, 0))
				elif f.is_dir(follow_symlinks=False):
					fp = os.path.relpath(f.path, argv)
					info[0].append(fp)
					try:
						sub, pos = recursive_scan(path=f.path, pos=pos)
					except PermissionError:
						pass
					else:
						files.extend(sub)
						pos = pos
			return files, pos

		sys.stdout.write("Scanning...")

		if isdir:
			fut = submit(recursive_scan, argv)
			count = 0
			while not fut.done():
				count2 = len(info) - 1 + len(extras)
				if count < count2:
					count = count2
					sys.stdout.write(f"\rScanning ({count})")
					time.sleep(0.03)
			sys.stdout.write(f"\rScanned ({len(info) - 1 + len(extras)}) \n")
			files, pos = fut.result()
			info.extend(extras)

			fs = fsize = pos
			if decompress:
				info.append(None)
			info[0] = list(info[0])
			info = list(info)
			infodata = as_bytes(json.dumps(info))
			if not compress:
				b = io.BytesIO()
				try:
					z = zipfile.ZipFile(b, "w", compression=zipfile.ZIP_LZMA, compresslevel=9, strict_timestamps=False)
				except:
					z = zipfile.ZipFile(b, "w", compression=zipfile.ZIP_LZMA, compresslevel=9)
				with z:
					z.writestr("M", infodata)
				b.seek(0)
				infodata = b.read()
			infolen = len(infodata).to_bytes(len(infodata).bit_length() + 7 >> 3, "little")
			infodata += b"\x80" * 2 + b"\x80".join(bytes((i,)) for i in infolen)
			fs += len(infodata)

		else:
			sys.stdout.write(f"\rScanned (1/1) \n")
			fsize = os.path.getsize(argv)
			name = argv.replace("\\", "/").split("/", 1)[-1]
			pos = 0
			files = [name]
			info.append((name, pos, fsize))
			names[name] = pos
			infodata = b"\x01\x80"
			fs = fsize + 2

		t = time.time()

		create_file(out, fs)

		sys.stdout.write("Combining...")
		futs = deque()
		pfuts = deque()
		indices = sorted(range(len(files)), key=lambda i: info[i + 1][2])
		quarter = len(indices) >> 2
		for f in map(files.__getitem__, reversed(indices[-quarter:])):
			pfuts.appendleft(ppe.submit(write_into, out, f, names[f]))
		for f in map(files.__getitem__, indices[:-quarter]):
			futs.append(submit(write_into, out, f, names[f]))
		futs.extend(pfuts)

		offs = 0
		for i, fut in enumerate(futs):
			try:
				fut.result()
				sys.stdout.write(f"\rCombining ({i}/{len(futs)})")
			except FileNotFoundError:
				pass
		sys.stdout.write(f"\rCombined ({len(futs)}/{len(futs)}) \n")

		with open(out, "rb+") as f:
			f.seek(fs - len(infodata))
			f.write(infodata)

		print(f"{fs} bytes written in {round(time.time() - t, 4)} seconds; {len(files)} unique files/folders, {len(extras)} duplicate/empty files.")

		if compress:
			sys.stdout.write("Compressing...")
			try:
				lzt.result()
			except:
				if os.path.exists(".cache"):
					for n in os.listdir(".cache"):
						submit(os.remove, ".cache/" + n)
				else:
					os.mkdir(".cache")

				cc = fs / (1 << 28)
				cc += bool(cc % 1)
				futs = deque()
				for i in range(int(cc)):
					j = i << 28
					futs.append(ppe.submit(deflate, out, f".cache/{j}", j, 1 << 28))
				with open(f".cache/{fs}", "wb"):
					pass
				for i, fut in enumerate(futs):
					fut.result()
					sys.stdout.write(f"\rCompressing ({i}/{len(futs)})")
				sys.stdout.write(f"\rCompressed ({len(futs)}/{len(futs)}) \n")
				if not encryptp:
					ppe.shutdown(wait=True)
				subprocess.run((sys.executable, this, "-d", ".cache", out + "c"))

				futs = deque()
				if os.path.exists(".cache"):
					for n in os.listdir(".cache"):
						futs.append(submit(os.remove, ".cache/" + n))
				for fut in futs:
					fut.result()

				if os.path.getsize(out + "c") < fs:
					os.remove(out)
					os.rename(out + "c", out)
				else:
					os.remove(out + "c")
				if not encryptp and os.path.exists(".cache"):
					os.rmdir(".cache")
			else:
				if not encryptp:
					ppe.shutdown(wait=True)
					tpe.shutdown(wait=True)
				print()
				if compress <= 1:
					c = 1
				elif compress == 2:
					c = 2
				elif compress == 3:
					c = 4
				elif compress == 4:
					c = 6
				elif compress >= 5:
					c = min(12, compress + 3)
				subprocess.run(("./4x4", str(c), "-p16", "-i48", out, out + ".lz"))
				if os.path.getsize(out + ".lz") + 2 < fs:
					os.remove(out)
					os.rename(out + ".lz", out)
					with open(out, "ab") as f:
						f.write(b"\x00\x80")
				else:
					os.remove(out + ".lz")

		if encryptp:
			sys.stdout.write("Encrypting...")
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					submit(os.remove, ".cache/" + n)
			else:
				os.mkdir(".cache")

			o2 = ".cache/.out"
			fis = os.path.getsize(out)
			create_file(o2, fis * 5)
			cc = fs / 5242880
			cc += bool(cc % 1)
			futs = deque()
			for i in range(int(cc)):
				j = i * 5242880
				futs.append(ppe.submit(encrypt, out, o2, j, 5242880, total=fis, password=encryptp))
			for i, fut in enumerate(futs):
				fut.result()
				sys.stdout.write(f"\nEncrypting ({i}/{len(futs)})")
			sys.stdout.write(f"\nEncrypted ({len(futs)}/{len(futs)}) \n")
			ppe.shutdown(wait=True)
			os.remove(out)
			os.rename(o2, out)
			if os.path.exists(".cache"):
				os.rmdir(".cache")

		osize = os.path.getsize(out)

	else:
		out = sys.argv[2] if len(sys.argv) > 2 else argv.rsplit(".", 1)[0] or "output"
		if not yes and os.path.exists(out):
			out += "~1"
			while os.path.exists(out):
				spl = out.rsplit("~", 1)
				out = spl[0] + "~" + str(int(spl[1]) + 1)
		fs = fsize = os.path.getsize(argv)
		infolen = b""

		if not target:
			sys.stdout.write("Scanning...")

		if decryptp:
			if not target:
				sys.stdout.write("\nDecrypting...")
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					submit(os.remove, ".cache/" + n)
			else:
				os.mkdir(".cache")

			i2 = ".cache/.in"
			create_file(i2, fs // 5)
			cc = fs / 26214400
			cc += bool(cc % 1)
			futs = deque()
			for i in range(int(cc)):
				j = i * 26214400
				futs.append(ppe.submit(decrypt, argv, i2, j, 26214400, total=fs // 5, password=decryptp))
			for i, fut in enumerate(futs):
				fut.result()
				if not target:
					sys.stdout.write(f"\rDecrypting ({i}/{len(futs)})")
			if not target:
				sys.stdout.write(f"\rDecrypted ({len(futs)}/{len(futs)}) \n")
			args = (sys.executable, this, "-s", str(fs), i2)
			if yes:
				args += ("-y",)
			if target:
				args += ("-f", target)
			else:
				args += (out,)
			subprocess.run(args)
			os.remove(i2)
			os.rmdir(".cache")
			raise SystemExit

		with open(argv, "rb+") as f:
			f.seek(fs - 2)
			b = f.read(2)
			if b == b"\x00\x80":
				ensure_compressor()
				f.truncate(fs - 2)
				f.close()
				if not target:
					print("\nDecompressing...")
				subprocess.run(("./4x4", "d", "-p16", "-i48", argv, argv + ".lz"))
				with open(argv, "ab") as f:
					f.write(b"\00\x80")
				args = (sys.executable, this, "-s", str(fs), argv + ".lz")
				if yes:
					args += ("-y",)
				if target:
					args += ("-f", target)
				else:
					args += (out,)
				subprocess.run(args)
				os.remove(argv + ".lz")
				raise SystemExit
			if b == b"\x01\x80":
				infodata = as_bytes(json.dumps([(), [out, 0, fs - 2]]))
				out = "./"
			else:
				b = c = b""
				for i in range(fs - 1, -1, -1):
					f.seek(i)
					b = c
					c = f.read(1)
					if b == c == b"\x80":
						break
					infolen = c + infolen
				infolen = int.from_bytes(infolen[1::2], "little")
				i -= infolen
				f.seek(i)
				infodata = f.read(infolen)
		if infodata[0] not in (91, 123, 128):
			b = io.BytesIO(infodata)
			with zipfile.ZipFile(b, "r") as z:
				infodata = z.read("M")
		if infodata[0] == 128:
			import pickle
			info = pickle.loads(infodata)
		else:
			info = json.loads(infodata)
			info = deque(info)

		if not target:
			sys.stdout.write(f"\rScanned ({len(info) - 1 + len(info[0])}) \n")

		fs = 0
		t = time.time()
		filec = extrac = 0

		entries = info.popleft()
		if info[-1] is None:
			decompress = True
			info.pop()
			rout, out = out, ".cache"
		else:
			decompress = False

		if decompress or not target:
			if not os.path.exists(out):
				os.mkdir(out)
			for f in sorted(entries, key=len):
				fn = os.path.join(out, f)
				try:
					os.mkdir(fn)
				except FileExistsError:
					extrac += 1
				else:
					filec += 1

		tuples = sorted(info, key=lambda t: t[2])

		if target and not decompress:
			for path, pos, size in tuples:
				if os.path.normcase(path) == target:
					with open(argv, "rb") as f:
						f.seek(pos)
						pos = 0
						while True:
							b = f.read(min(1048576, size - pos))
							if not b:
								raise SystemExit
							pos += len(b)
							sys.stdout.buffer.write(b)
			raise FileNotFoundError(target)

		futs = deque()
		pfuts = deque()
		quarter = len(tuples) >> 2
		osize = 0
		for path, pos, size in reversed(tuples[-quarter:]):
			if not path:
				path = out.rsplit("/", 1)[-1]
			if not size:
				with open(os.path.join(out, path), "wb"):
					extrac += 1
			else:
				pfuts.appendleft(ppe.submit(read_into, f"{out}/{path}", argv, pos, size))
				filec += 1
				fs += size
				osize += size
		for path, pos, size in tuples[:-quarter]:
			if not size:
				with open(os.path.join(out, path), "wb"):
					extrac += 1
			else:
				futs.append(submit(read_into, f"{out}/{path}", argv, pos, size))
				filec += 1
				fs += size
				osize += size
		futs.extend(pfuts)
		for i, fut in enumerate(futs):
			fut.result()
			if not target:
				sys.stdout.write(f"\rExtracting {i}/{len(futs)}")
		if not target:
			sys.stdout.write(f"\rExtracted {len(futs)}/{len(futs)} \n")
			print(f"{fs} bytes written in {round(time.time() - t, 4)} seconds; {filec} valid files/folders, {extrac} empty files.")

		if decompress:
			if os.path.exists(f"{out}/.0"):
				os.remove(f"{out}/.0")
			li = sorted(map(int, os.listdir(out)))
			fs = li.pop(-1)
			create_file(f"{out}/.0", fs)

			if not target:
				sys.stdout.write("Decompressing...")
			futs = deque()
			for i in li:
				futs.append(ppe.submit(inflate, f"{out}/{i}", f"{out}/.0", i))
			for i, fut in enumerate(futs):
				fut.result()
				if not target:
					sys.stdout.write(f"\rDecompressing ({i}/{len(futs)})")
			if not target:
				sys.stdout.write(f"\rDecompressed ({len(futs)}/{len(futs)}) \n")
			ppe.shutdown(wait=True)
			args = (sys.executable, this, ".cache/.0", rout)
			if target:
				args += ("-f", target)
			subprocess.run(args)

			futs = deque()
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					futs.append(submit(os.remove, ".cache/" + n))
			for fut in futs:
				fut.result()
			os.rmdir(".cache")
			raise SystemExit

	print(f"Total elapsed time: {round(time.time() - orig, 4)} seconds, output size ratio: {round(osize / (isize or fsize), 4)}")