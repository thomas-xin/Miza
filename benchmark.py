import os
import sys
python = sys.executable

os.system("color")
def srgb(r, g, b, s):
	return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"

if "-nw" in sys.argv:
	sys.argv.remove("-nw")
	rewrite = False
else:
	rewrite = True
keep = True

if "-i" in sys.argv:
	INT = True
	sys.argv.remove("-i")
else:
	INT = False

is_sub = False
if len(sys.argv) > 1:
	device, name, core, mem = sys.argv[1:]
	core, mem = int(core), int(mem)
	if __name__ == "__main__":
		print(f"Benchmarking device {device}...")
	is_cuda = device.startswith("cuda:")
	if is_cuda:
		import os
		os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
		cuda = "cuda:0"
	else:
		cuda = "cpu"
	import torch
	import time
	import math
	from torch.utils import benchmark
	BATCH = 256
	def walltime(stmt, arg_dict, duration=1):
		return sum(benchmark.Timer(stmt=stmt, globals=arg_dict).timeit(BATCH).raw_times) / BATCH
	if not is_cuda:
		import concurrent.futures
		exc = concurrent.futures.ProcessPoolExecutor(max_workers=core)
	if __name__ == "__main__":
		count = 64 * (math.cbrt(core) if is_cuda else 1)
		def mark(*args):
			return temp.append(time.time())
		while True:
			taken = 0
			temp = []
			if INT:
				dtype = torch.int8 if is_cuda else torch.int16
			else:
				dtype = torch.float16 if is_cuda else torch.float32
			m = math.ceil(math.log2(count))
			n = min(12, m)
			if n < m:
				it = 2 ** (m - n)
			else:
				it = 1
			c = 2 ** n
			a = torch.randn(c, c, dtype=dtype, device=cuda)
			b = torch.randn(c, c, dtype=dtype, device=cuda)
			sys.stderr.write(str(("DEVICE:", device, count, len(a), c, it)) + "\n")
			sys.stderr.flush()
			for i in range(it):
				if is_cuda:
					t = walltime('a @ b', dict(a=a, b=b))
				else:
					t = 0
					futs = []
					for thread in range(core):
						fut = exc.submit(walltime, 'a @ b', dict(a=a, b=b))
						# fut = exc.submit(time.sleep, 1)
						futs.append(fut)
					for fut in futs:
						t += fut.result()
					t /= core ** 2
				# sys.stderr.write(f"{t} {core}\n")
				# sys.stderr.flush()
				temp.append(2 * c ** 3 / t)
				if not is_cuda:
					t *= core
				taken += t
				# t1 = a * b
				# t2 = a * b
				# assert torch.all(t1 == t2)
			del a
			del b
			req = 5 / BATCH
			if taken < req:
				count = round(max(count * math.cbrt(2), count * math.cbrt(req / (taken + 1 / 1000))))
				sys.stderr.write(str((device, taken, count)) + "\n")
				sys.stderr.flush()
				continue
			break
		if not is_cuda:
			exc.shutdown(wait=False)
		memc = round(mem / 1073741824, 2)
		# im = data.images[0]
		# im.save(f"{name} ({core}-core, {memc} GB).png")
		iavg = sum(temp) / len(temp)
		wavg = [n for n in temp if n >= iavg]
		ops = sum(wavg) / len(wavg)
		op = "I" if INT else "FL"
		score = ops / 50000000
		# diffs = [temp[i] - temp[i - 1] for i in range(1, len(temp))]
		# iavg = sum(diffs) / len(diffs)
		# wavg = [n for n in diffs if n <= iavg]
		# avg = sum(wavg) / len(wavg)
		# score = 100000 / avg
		cc = f"{core}-core"
		cc = srgb(0, 255, 0, cc) if core >= 4096 else srgb(255, 255, 0, cc) if core >= 16 else srgb(255, 127, 0, cc) if core >= 8 else srgb(255, 0, 0, cc)
		gb = f"{memc} GB"
		gb = srgb(0, 255, 0, gb) if memc > 11 else srgb(255, 255, 0, gb) if memc >= 7 else srgb(255, 127, 0, gb) if memc > 3 else srgb(255, 0, 0, gb)
		FP = "FP16" if dtype == torch.float16 else "FP32"
		print(f"Benchmarked {srgb(0, 255, 255, name)} ({cc}, {gb}). Average peak {FP} performance: {ops / 1e12} T{op}OPS.")
		sc = f"Score: {round(score, 2)}"
		sc = srgb(0, 255, 0, sc) if score >= 1000000 else srgb(255, 255, 0, sc) if score >= 300000 else srgb(255, 127, 0, sc) if score >= 90000 else srgb(255, 0, 0, sc)
		print(sc)
		print(score)
		raise SystemExit
	else:
		is_sub = True

DC = 0
if not is_sub:
	import subprocess
	print(srgb(0, 0, 255, "Scanning hardware..."))

	try:
		import pynvml
	except ImportError:
		subprocess.run([python, "-m", "pip", "install", "pynvml", "--upgrade"])

	print(srgb(0, 255, 255, "Loading..."))
	import pynvml
	try:
		pynvml.nvmlInit()
		DC = pynvml.nvmlDeviceGetCount()
	except Exception:
		DC = 0
	if not DC:
		print(srgb(255, 0, 0, "WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration."))

	try:
		import cpuinfo
		import psutil
		if DC:
			import torch
			import torchvision
			if not torch.cuda.is_available() or not torchvision._HAS_OPS:
				raise ImportError
	except ImportError:
		subprocess.run([python, "-m", "pip", "install", "py-cpuinfo", "--upgrade"])
		subprocess.run([python, "-m", "pip", "install", "psutil", "--upgrade"])
		if DC:
			subprocess.run([python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cu121"])
			import torch
		import cpuinfo
		import psutil

	compute_load = []
	compute_order = []

	import json
	if __name__ != "__main__" and os.path.exists("auth.json"):
		import importlib.metadata

		req = []
		if DC:
			req.append("accelerate")
		for mn in req:
			try:
				importlib.metadata.version(mn)
			except Exception:
				subprocess.run([python, "-m", "pip", "install", mn, "--upgrade"])

		with open("auth.json", "rb") as f:
			try:
				data = json.load(f)
			except json.JSONDecodeError:
				data = {}
		compute_load = data.get("compute_load")
		compute_order = data.get("compute_order")
		if compute_load is not None and compute_order is not None and len(compute_load) == len(compute_order) == DC:
			print(srgb(0, 255, 0, "No benchmark required, skipping..."))
			keep = False

	if keep:
		compute_queue = []
		compute_order = []

		import time
		handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(DC)]
		bwidths = [1073741824 / 8 / 8 * 2 ** pynvml.nvmlDeviceGetCurrPcieLinkGeneration(d) * pynvml.nvmlDeviceGetCurrPcieLinkWidth(d) for d in handles]
		if DC > 1:
			import torch
			import time
			devices = list(range(DC - 1, -1, -1))

			for i in devices:
				print(f"Initialising device {i}...")
				try:
					a = torch.ones(1, dtype=torch.uint8, device=i)
				except RuntimeError as ex:
					print(f"Device {i}:", repr(ex))
					a = torch.ones(1, dtype=torch.float32, device=i)
				del a

			i = sorted(reversed(devices), key=lambda i: (d := torch.cuda.get_device_properties(i)) and (d.major * d.multi_processor_count * d.total_memory), reverse=True)[0]
			compute_order.append(i)
			devices.remove(i)
			f = i
			while len(devices) > 1:
				print(f"Transferring device {i}...")
				best = next(iter(devices))
				shortest = torch.inf
				j = i
				for i in devices:
					t = 1 / bwidths[i] / torch.cuda.get_device_properties(i).total_memory
					# a = torch.randint(0, 255, (1073741824,), dtype=torch.uint8, device=j)
					# t = time.time()
					# b = a.to(i)
					# t = time.time() - t
					# del a
					# print(i, t)
					if t < shortest:
						shortest = t
						best = i
				compute_order.append(best)
				devices.remove(best)
				i = best
			compute_order.append(next(iter(devices)))
			print("Optimal device order:", compute_order)
		elif DC:
			compute_order = list(range(DC - 1, -1, -1))

		print("Starting benchmarks...")
		total = 0
		procs = []
		# avgs = []
		# mems = []

		info = cpuinfo.get_cpu_info()
		mem = psutil.virtual_memory().total
			# mems.append(mem)
		if __name__ == "__main__":
			args = [python, __file__, "cpu", info["brand_raw"], str(info["count"]), str(mem)]
			if INT:
				args.append("-i")
			print(args)
			proc = subprocess.Popen(args, stdout=subprocess.PIPE)
			proc.i = -1
			procs.append(proc)
			proc.wait()
		if DC:
			last = None
			for i in list(range(DC)[::2]) + list(range(DC)[1::2]):
				info = pynvml.nvmlDeviceGetHandleByIndex(i)
				mem = torch.cuda.get_device_properties(i).total_memory
				# mems.append(mem)
				args = [python, __file__, f"cuda:{i}", pynvml.nvmlDeviceGetName(info), str(pynvml.nvmlDeviceGetNumGpuCores(info)), str(mem)]
				if INT:
					args.append("-i")
				print(args)
				proc = subprocess.Popen(args, stdout=subprocess.PIPE)
				proc.i = i
				procs.append(proc)
				if last:
					try:
						last.wait(timeout=1)
					except subprocess.TimeoutExpired:
						pass
				else:
					time.sleep(1)
				last = proc
		if procs:
			half = len(procs) >> 1
			outs = []
			for a, b in zip(procs[:half], procs[half:half * 2]):
				outs.append(a)
				outs.append(b)
			outs.extend(procs[half * 2:])
			compute_load = [0] * DC
			olines = [""] * (DC + 1)
			# print(outs)
			for n, proc in enumerate(outs):
				s = proc.stdout.readlines()
				avg = float(s.pop(-1))
				# avgs.append(avg)
				total += avg
				if proc.i > -1:# or DC >= 3:
					compute_load[proc.i] = avg
				olines[proc.i + 1] = b"".join(s).decode("utf-8")
			ops = total * 50000000
			op = "I" if INT else "FL"
			print("\n" + "\n".join(olines).strip())
			print(srgb(0, 255, 0, f"Benchmark complete. Total score: {round(total, 2)}, {round(ops / 1e12, 2)} T{op}OPS."))

		if not os.path.exists("auth.json"):
			keep = False

	if keep and rewrite:
		with open("auth.json", "rb+") as f:
			try:
				data = json.load(f)
			except json.JSONDecodeError:
				data = {}
			data["compute_load"] = compute_load
			data["compute_order"] = compute_order
			b = json.dumps(data, indent="\t").encode("utf-8")
			f.truncate(len(b))
			f.seek(0)
			f.write(b)
			print(srgb(0, 255, 0, "Results written to `auth.json`."))