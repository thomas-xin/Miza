import os, sys

os.system("color")
srgb = lambda r, g, b, s: f"\033[38;2;{r};{g};{b}m{s}\033[0m"

if len(sys.argv) > 1:
	device, name, core, mem = sys.argv[1:]
	core, mem = int(core), int(mem)
	print(f"Benchmarking device {device}...")
	is_cuda = device.startswith("cuda:")
	if is_cuda:
		import os
		os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
		device = "cuda:0"
	if os.path.exists("auth.json"):
		import json
		with open("auth.json", "rb") as f:
			AUTH = json.loads(f.read())
		cachedir = AUTH.get("cache_path") or None
		if cachedir:
			os.environ["HF_HOME"] = f"{cachedir}/huggingface"
			os.environ["TORCH_HOME"] = f"{cachedir}/torch"
			os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cachedir}/huggingface/hub"
			os.environ["TRANSFORMERS_CACHE"] = f"{cachedir}/huggingface/transformers"
			os.environ["HF_DATASETS_CACHE"] = f"{cachedir}/huggingface/datasets"
	import torch, time, math
	from torch.utils import benchmark
	def walltime(stmt, arg_dict, duration=1):
		return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
			min_run_time=duration).median
	# prompt = " ".join(["water"] * 64)
	# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
	# model = "runwayml/stable-diffusion-v1-5"
	# pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16 if is_cuda else torch.float32)
	# if is_cuda:
		# pipe = pipe.to(device)
	# pipe.enable_attention_slicing()
	# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	# if is_cuda:
		# try:
			# pipe.enable_model_cpu_offload()
		# except AttributeError:
			# pass
	# pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
	count = 4096
	mark = lambda *args: temp.append(time.time())
	while True:
		taken = 0
		temp = []
		# data = pipe(prompt, num_inference_steps=count + 1, callback=mark)
		matmul_tflops = {}
		dtype = torch.float16
		m = math.ceil(math.log2(count))
		n = min(12, m)
		if n < m:
			it = 2 ** (m - n)
		else:
			it = 1
		c = 2 ** n
		# print(c)
		a = torch.randn(c, c, dtype=dtype, device="cuda" if is_cuda else "cpu")
		b = torch.randn(c, c, dtype=dtype, device="cuda" if is_cuda else "cpu")
		# print("DEVICE:", device, count, len(a), c, it)
		for i in range(it):
			t = walltime('a @ b', dict(a=a, b=b))
			temp.append(2 * c ** 3 / t)
			taken += t
		del a, b
		if taken < 5 and count < 65536:
			count = min(65536, max(count * 2, round(math.sqrt(5 / taken) * count)))
			continue
		break
	memc = round(mem / 1073741824, 2)
	# im = data.images[0]
	# im.save(f"{name} ({core}-core, {memc} GB).png")
	iavg = sum(temp) / len(temp)
	wavg = [n for n in temp if n >= iavg]
	flops = sum(wavg) / len(wavg)
	score = flops / 50000000
	# diffs = [temp[i] - temp[i - 1] for i in range(1, len(temp))]
	# iavg = sum(diffs) / len(diffs)
	# wavg = [n for n in diffs if n <= iavg]
	# avg = sum(wavg) / len(wavg)
	# score = 100000 / avg
	cc = f"{core}-core"
	cc = srgb(0, 255, 0, cc) if core >= 4096 else srgb(255, 255, 0, cc) if core >= 16 else srgb(255, 127, 0, cc) if core >= 8 else srgb(255, 0, 0, cc)
	gb = f"{memc} GB"
	gb = srgb(0, 255, 0, gb) if memc > 11 else srgb(255, 255, 0, gb) if memc >= 7 else srgb(255, 127, 0, gb) if memc > 3 else srgb(255, 0, 0, gb)
	print(f"Benchmarked {srgb(0, 255, 255, name)} ({cc}, {gb}). Average peak FP16 performance: {flops / 1e12} TFLOPS.")
	sc = f"Score: {round(score, 2)}"
	sc = srgb(0, 255, 0, sc) if score >= 1000000 else srgb(255, 255, 0, sc) if score >= 300000 else srgb(255, 127, 0, sc) if score >= 90000 else srgb(255, 0, 0, sc)
	print(sc)
	print(score)
	raise SystemExit

print(srgb(0, 0, 255, "Scanning hardware..."))

import subprocess
try:
	import pynvml
except ImportError:
	subprocess.run([sys.executable, "-m", "pip", "install", "pynvml", "--upgrade", "--user"])

print(srgb(0, 255, 255, "Loading..."))
import pynvml
try:
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
except:
	DC = 0
# if not DC:
	# print(srgb(255, 0, 0, "WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration."))

keep = True
if __name__ != "__main__" and os.path.exists("auth.json"):
	import json
	with open("auth.json", "rb") as f:
		data = json.load(f)
	compute_load = data.get("compute_load")
	compute_order = data.get("compute_order")
	if compute_load is not None and compute_order is not None and len(compute_load) == len(compute_order) == DC:
		print(srgb(0, 255, 0, "No benchmark required, skipping..."))
		keep = False

if keep:
	try:
		import cpuinfo, psutil
		if DC:
			import torch
			if not torch.cuda.is_available():
				raise ImportError
	except ImportError:
		subprocess.run([sys.executable, "-m", "pip", "install", "py-cpuinfo", "--upgrade", "--user"])
		subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "--upgrade", "--user"])
		if DC:
			subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118", "--upgrade", "--user"])
			import torch
		# else:
		# 	subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--upgrade", "--user"])
		import cpuinfo, psutil

	try:
		import pkg_resources
	except:
		import traceback
		print(srgb(255, 0, 0, traceback.format_exc()), end="")
		subprocess.run(["pip", "install", "setuptools", "--upgrade", "--user"])
		import pkg_resources
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
			subprocess.run([sys.executable, "-m", "pip", "install", mn, "--upgrade", "--user"])
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
			print(f"Testing device {i}...")
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
	if DC:
		# mems.append(mem)
		if DC < 3:
			args = [sys.executable, sys.argv[0], "cpu", info["brand_raw"], str(info["count"]), str(mem)]
			proc = subprocess.Popen(args, stdout=subprocess.PIPE)
			procs.append(proc)
		last = None
		for i in list(range(DC)[::2]) + list(range(DC)[1::2]):
			info = pynvml.nvmlDeviceGetHandleByIndex(i)
			mem = torch.cuda.get_device_properties(i).total_memory
			# mems.append(mem)
			args = [sys.executable, sys.argv[0], f"cuda:{i}", pynvml.nvmlDeviceGetName(info), str(pynvml.nvmlDeviceGetNumGpuCores(info)), str(mem)]
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
		half = DC + 1 >> 1
		outs = []
		for a, b in zip(procs[:half], procs[half:]):
			outs.append(a)
			outs.append(b)
		compute_load = [0] * len(outs)
		olines = []
		for n, proc in enumerate(outs):
			s = proc.stdout.readlines()
			avg = float(s.pop(-1))
			# avgs.append(avg)
			total += avg
			if n or DC >= 3:
				compute_load[proc.i] = avg
			olines.append(b"".join(s).decode("utf-8"))
		print("\n" + "\n".join(olines))
		print(srgb(0, 255, 0, f"Benchmark complete. Total score: {round(total, 2)}"))

	if not os.path.exists("auth.json"):
		keep = False

	import json
	with open("auth.json", "rb+") as f:
		data = json.load(f)
		data["compute_load"] = compute_load
		data["compute_order"] = compute_order
		b = json.dumps(data, indent="\t").encode("utf-8")
		f.truncate(len(b))
		f.seek(0)
		f.write(b)
		print(srgb(0, 255, 0, "Results written to `auth.json`."))