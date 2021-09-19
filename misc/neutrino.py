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


if __name__ == "__main__":
	import sys, time, collections, pickle, zipfile, io, concurrent.futures, subprocess
	from concurrent.futures import thread
	from collections import deque

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
		if os.name == "nt":
			if os.path.exists(name):
				os.remove(name)
			subprocess.run(("fsutil", "file", "createnew", name, f"{size}"), stdout=subprocess.DEVNULL)
		else:
			subprocess.run(f"truncate -s {size} {name}", shell=True, stdout=subprocess.DEVNULL)

	this = sys.argv[0]

	if "-c" in sys.argv:
		sys.argv.remove("-c")
		compress = True
	else:
		compress = False
	if "-d" in sys.argv:
		sys.argv.remove("-d")
		decompress = True
	else:
		decompress = False
	try:
		i = sys.argv.index("-f")
	except ValueError:
		target = None
	else:
		target = os.path.normcase(sys.argv[i + 1])
		sys.argv = sys.argv[:i] + sys.argv[i + 1:]
	argv = sys.argv[1] if len(sys.argv) > 1 else None

	while not argv:
		argv = input("Please enter file or folder to process: ")
	argv = argv.replace("\\", "/")

	if os.path.isdir(argv):
		out = sys.argv[2] if len(sys.argv) > 2 else argv.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".wb" or "output.wb"
		if os.path.exists(out):
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
						if compress and size in sizes:
							h = get_hash(f.path, size=size)
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

		fs = pos
		if decompress:
			info.append(None)
		infodata = pickle.dumps(info)
		if not compress:
			b = io.BytesIO()
			with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=False) as z:
				z.writestr("M", infodata)
			b.seek(0)
			infodata = b.read()
		infolen = len(infodata).to_bytes(len(infodata).bit_length() + 7 >> 3, "little")
		infodata += b"\x80" * 2 + b"\x80".join(bytes((i,)) for i in infolen)
		fs += len(infodata)

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

		if compress:
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					submit(os.remove, ".cache/" + n)
			else:
				os.mkdir(".cache")

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
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					submit(os.remove, ".cache/" + n)
			else:
				os.mkdir(".cache")

			sys.stdout.write("Compressing...")
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
			subprocess.run((sys.executable, this, "-d", ".cache", out + "c"))
			os.remove(out)

			futs = deque()
			if os.path.exists(".cache"):
				for n in os.listdir(".cache"):
					futs.append(submit(os.remove, ".cache/" + n))
			for fut in futs:
				fut.result()
			os.rmdir(".cache")
			os.rename(out + "c", out)

	else:
		out = sys.argv[2] if len(sys.argv) > 2 else argv.rsplit(".", 1)[0] or "output"
		if os.path.exists(out):
			out += "~1"
			while os.path.exists(out):
				spl = out.rsplit("~", 1)
				out = spl[0] + "~" + str(int(spl[1]) + 1)
		fs = os.path.getsize(argv)
		infolen = b""

		if not target:
			sys.stdout.write("Scanning...")

		with open(argv, "rb") as f:
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
		if infodata[0] != 128:
			b = io.BytesIO(infodata)
			with zipfile.ZipFile(b, "r") as z:
				infodata = z.read("M")
		info = pickle.loads(infodata)

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
		else:
			futs = deque()
			pfuts = deque()
			quarter = len(tuples) >> 2
			for path, pos, size in reversed(tuples[-quarter:]):
				if not size:
					with open(os.path.join(out, path), "wb"):
						extrac += 1
				else:
					pfuts.appendleft(ppe.submit(read_into, f"{out}/{path}", argv, pos, size))
					filec += 1
					fs += size
			for path, pos, size in tuples[:-quarter]:
				if not size:
					with open(os.path.join(out, path), "wb"):
						extrac += 1
				else:
					futs.append(submit(read_into, f"{out}/{path}", argv, pos, size))
					filec += 1
					fs += size
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