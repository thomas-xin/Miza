import os, sys
python = sys.executable

os.system("color")
srgb = lambda r, g, b, s: f"\033[38;2;{r};{g};{b}m{s}\033[0m"

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
		device = "cuda:0"
	import torch, time, math
	from torch.utils import benchmark
	def walltime(stmt, arg_dict, duration=1):
		return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(min_run_time=duration).median
	if not is_cuda:
		import concurrent.futures
		exc = concurrent.futures.ProcessPoolExecutor(max_workers=core)
	if __name__ == "__main__":
		count = 4096
		mark = lambda *args: temp.append(time.time())
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
			# print(c)
			a = torch.randn(c, c, dtype=torch.float32, device="cuda" if is_cuda else "cpu").to(dtype)
			b = torch.randn(c, c, dtype=torch.float32, device="cuda" if is_cuda else "cpu").to(dtype)
			# print("DEVICE:", device, count, len(a), c, it)
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
						# sys.stderr.write(f"{t} {core}\n")
					t /= core ** 2
				temp.append(2 * c ** 3 / t)
				if not is_cuda:
					t *= core
				taken += t
			del a, b
			if taken < 5 and count < 65536:
				count = min(65536, max(count * 2, round(math.sqrt(5 / taken) * count)))
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
	print(srgb(0, 0, 255, "Scanning hardware..."))

	import subprocess
	try:
		import pynvml
	except ImportError:
		subprocess.run([python, "-m", "pip", "install", "pynvml", "--upgrade", "--user"])

	print(srgb(0, 255, 255, "Loading..."))
	import pynvml
	try:
		# raise
		pynvml.nvmlInit()
		DC = pynvml.nvmlDeviceGetCount()
	except:
		DC = 0
	# if not DC:
		# print(srgb(255, 0, 0, "WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration."))

	compute_load = []
	compute_order = []

	import json
	keep = True
	if __name__ != "__main__" and os.path.exists("auth.json"):
		try:
			import pkg_resources
		except:
			import traceback
			print(srgb(255, 0, 0, traceback.format_exc()), end="")
			subprocess.run(["pip", "install", "setuptools", "--upgrade", "--user"])
			import pkg_resources

		try:
			import cpuinfo, psutil
			if DC:
				import torch, torchvision, xformers
				if not torch.cuda.is_available() or not torchvision._HAS_OPS:
					raise ImportError
		except ImportError:
			subprocess.run([python, "-m", "pip", "install", "py-cpuinfo", "--upgrade", "--user"])
			subprocess.run([python, "-m", "pip", "install", "psutil", "--upgrade", "--user"])
			if DC:
				subprocess.run([python, "-m", "pip", "install", "xformers", "--upgrade", "--user", "--index-url", "https://download.pytorch.org/whl/cu121"])
				subprocess.run([python, "-m", "pip", "install", "torchvision", "torchaudio", "--upgrade", "--user", "--index-url", "https://download.pytorch.org/whl/cu121"])
				try:
					assert pkg_resources.get_distribution("exllamav2").version >= "0.0.8"
				except (pkg_resources.DistributionNotFound, AssertionError):
					vi = f"{sys.version_info.major}{sys.version_info.minor}"
					oi = "win_amd64" if os.name == "nt" else "linux_x86_64"
					subprocess.run([python, "-m", "pip", "install", "--upgrade", "--user", f"https://github.com/turboderp/exllamav2/releases/download/v0.0.8/exllamav2-0.0.8+cu121-cp{vi}-cp{vi}-{oi}.whl"])
				import torch
			import cpuinfo, psutil

		req = ["diffusers"]
		if DC:
			req.append("accelerate")
		# if os.name == "nt":
		# 	req.append("bitsandbytes-windows")
		# else:
		# 	req.append("bitsandbytes")
		for mn in req:
			try:
				pkg_resources.get_distribution(mn)
			except:
				subprocess.run([python, "-m", "pip", "install", mn, "--upgrade", "--user"])
		# if os.name == "nt":
		# 	try:
		# 		pkg_resources.get_distribution("bitsandbytes")
		# 	except:
		# 		try:
		# 			dist = pkg_resources.get_distribution("bitsandbytes-windows")
		# 			fold = dist.module_path + "/bitsandbytes_windows-" + dist.version + ".dist-info"
		# 			if os.path.exists(fold):
		# 				os.rename(fold, fold.replace("_windows", ""))
		# 		except:
		# 			import traceback
		# 			print(srgb(255, 0, 0, traceback.format_exc()), end="")

		with open("auth.json", "rb") as f:
			try:
				data = json.load(f)
			except:
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
		if DC > 2:
			import torch, time
			devices = list(range(DC - 1, -1, -1))

			for i in devices:
				print(f"Initialising device {i}...")
				a = torch.zeros(1, dtype=torch.uint8, device=i)
				del a

			i = sorted(devices, key=lambda i: (d := torch.cuda.get_device_properties(i)) and (d.major * d.multi_processor_count * d.total_memory), reverse=True)[0]
			compute_order.append(i)
			devices.remove(i)
			f = i
			while len(devices) > 1:
				print(f"Transferring device {i}...")
				best = next(iter(devices))
				shortest = torch.inf
				j = i
				for i in devices:
					a = torch.randint(0, 255, (1073741824,), dtype=torch.uint8, device=j)
					t = time.time()
					b = a.to(i)
					t = time.time() - t
					del a
					print(i, t)
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

	if keep:
		with open("auth.json", "rb+") as f:
			try:
				data = json.load(f)
			except:
				data = {}
			data["compute_load"] = compute_load
			data["compute_order"] = compute_order
			b = json.dumps(data, indent="\t").encode("utf-8")
			f.truncate(len(b))
			f.seek(0)
			f.write(b)
			print(srgb(0, 255, 0, "Results written to `auth.json`."))