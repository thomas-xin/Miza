import os, sys

os.system("color")
srgb = lambda r, g, b, s: f"\033[38;2;{r};{g};{b}m{s}\033[0m"

print(srgb(0, 0, 255, "Scanning hardware..."))

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
	prompt = " ".join(["water"] * 64)
	import torch, time
	if 0:
		from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
		import accelerate
		m = "Neko-Institute-of-Science/pygmalion-7b"
		tokenizer = AutoTokenizer.from_pretrained(m)
		import bitsandbytes
		config = AutoConfig.from_pretrained(m)
		# dev_map = accelerate.infer_auto_device_map(model, max_memory=max_mem, dtype=torch.float16)
		# from transformers import BitsAndBytesConfig
		# quantization_config = BitsAndBytesConfig(
			# load_in_8bit=True,
			# llm_int8_threshold=8.0,
			# llm_int8_enable_fp32_cpu_offload=True,
			# llm_int8_has_fp16_weight=True
		# )
		model = AutoModelForCausalLM.from_pretrained(m, device_map="auto", load_in_8bit=is_cuda)
		tokens = tokenizer.encode(prompt, return_tensors="pt").cuda()
		res = model.generate(
			tokens,
			temperature=1,
			top_k=192,
			top_p=1,
			max_length=1024,
		)
		text = tokenizer.decode(res[0])
		print(text)
		print(0)
		raise SystemExit
	from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
	model = "runwayml/stable-diffusion-v1-5"
	pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16 if is_cuda else torch.float32)
	if is_cuda:
		pipe = pipe.to(device)
	pipe.enable_attention_slicing()
	pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	if is_cuda:
		try:
			pipe.enable_model_cpu_offload()
		except AttributeError:
			pass
	pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
	count = 3
	taken = 0
	temp = []
	mark = lambda *args: temp.append(time.time())
	while True:
		temp.clear()
		data = pipe(prompt, num_inference_steps=count + 1, callback=mark)
		taken = temp[-1] - temp[1]
		if taken < 30 and count < 999:
			count = min(999, max(count * 2, round(30 / taken * count)))
			continue
		break
	memc = round(mem / 1073741824, 2)
	im = data.images[0]
	im.save(f"{name} ({core}-core, {memc} GB).png")
	tavg = taken
	diffs = [temp[i] - temp[i - 1] for i in range(1, len(temp))]
	iavg = sum(diffs) / len(diffs)
	wavg = [n for n in diffs if n <= iavg]
	avg = sum(wavg) / len(wavg)
	score = 100000 / avg
	cc = f"{core}-core"
	cc = srgb(0, 255, 0, cc) if core >= 4096 else srgb(255, 255, 0, cc) if core >= 16 else srgb(255, 127, 0, cc) if core >= 8 else srgb(255, 0, 0, cc)
	gb = f"{memc} GB"
	gb = srgb(0, 255, 0, gb) if memc > 11 else srgb(255, 255, 0, gb) if memc >= 7 else srgb(255, 127, 0, gb) if memc > 3 else srgb(255, 0, 0, gb)
	print(f"Benchmarked {srgb(0, 255, 255, name)} ({cc}, {gb}). Weighted average time taken across {count} iteration(s): {avg}s")
	sc = f"Score: {round(score, 2)}"
	sc = srgb(0, 255, 0, sc) if score >= 1000000 else srgb(255, 255, 0, sc) if score >= 300000 else srgb(255, 127, 0, sc) if score >= 90000 else srgb(255, 0, 0, sc)
	print(sc)
	print(score)
	raise SystemExit

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
if not DC:
	print(srgb(255, 0, 0, "WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration."))

keep = True
if __name__ != "__main__" and os.path.exists("auth.json"):
	import json
	with open("auth.json", "rb") as f:
		data = json.load(f)
	compute_load = data.get("compute_load")
	if compute_load is not None and len(compute_load) == DC:
		print(srgb(0, 255, 0, "No benchmark required, skipping..."))
		keep = False

if keep:
	try:
		import torch, cpuinfo, psutil
		if DC and not torch.cuda.is_available():
			raise ImportError
	except ImportError:
		subprocess.run([sys.executable, "-m", "pip", "install", "py-cpuinfo", "--upgrade", "--user"])
		subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "--upgrade", "--user"])
		if DC:
			subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--upgrade", "--user"])
		else:
			subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118", "--upgrade", "--user"])
		import torch, cpuinfo, psutil

	try:
		import pkg_resources
	except:
		import traceback
		print(srgb(255, 0, 0, traceback.format_exc()), end="")
		subprocess.run(["pip", "install", "setuptools", "--upgrade", "--user"])
		import pkg_resources
	req = ["diffusers", "accelerate"]
	if os.name == "nt":
		req.append("bitsandbytes-windows")
	else:
		req.append("bitsandbytes")
	for mn in req:
		try:
			pkg_resources.get_distribution(mn)
		except:
			subprocess.run([sys.executable, "-m", "pip", "install", mn, "--upgrade", "--user"])

	total = 0
	procs = []
	# avgs = []
	# mems = []

	info = cpuinfo.get_cpu_info()
	mem = psutil.virtual_memory().total
	# mems.append(mem)
	args = [sys.executable, sys.argv[0], "cpu", info["brand_raw"], str(info["count"]), str(mem)]
	proc = subprocess.Popen(args, stdout=subprocess.PIPE)
	procs.append(proc)
	outs = []
	for i in list(range(DC)[::2]) + list(range(DC)[1::2]):
		# if len(procs) > 2:
			# proc = procs.pop(0)
			# proc.wait()
			# outs.append(proc)
		info = pynvml.nvmlDeviceGetHandleByIndex(i)
		mem = torch.cuda.get_device_properties(i).total_memory
		# mems.append(mem)
		args = [sys.executable, sys.argv[0], f"cuda:{i}", pynvml.nvmlDeviceGetName(info), str(pynvml.nvmlDeviceGetNumGpuCores(info)), str(mem)]
		proc = subprocess.Popen(args, stdout=subprocess.PIPE)
		proc.i = i
		procs.append(proc)
	outs.extend(procs)

	print()
	compute_load = [0] * len(outs)
	for n, proc in enumerate(outs):
		s = proc.stdout.readlines()
		avg = float(s.pop(-1))
		# avgs.append(avg)
		total += avg
		if n:
			compute_load[proc.i] = avg
		print(b"".join(s).decode("utf-8"))
	print(srgb(0, 255, 0, f"Benchmark complete. Total score: {round(total, 2)}"))

	if not os.path.exists("auth.json"):
		keep = False

if keep:
	import json
	with open("auth.json", "rb+") as f:
		data = json.load(f)
		data["compute_load"] = compute_load
		b = json.dumps(data, indent="\t").encode("utf-8")
		f.truncate(len(b))
		f.seek(0)
		f.write(b)
		print(srgb(0, 255, 0, "Results written to `auth.json`."))