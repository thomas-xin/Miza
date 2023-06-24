import sys
if len(sys.argv) > 1:
	device, name, core = sys.argv[1:]
	print(f"Benchmarking device {device}...")
	is_cuda = device.startswith("cuda:")
	if is_cuda:
		import os
		os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
		device = "cuda:0"
	prompt = " ".join(["water"] * 64)
	import torch, time
	if 1:
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
		model = AutoModelForCausalLM.from_pretrained(m, device_map="auto", load_in_8bit=True)
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
	im = data.images[0]
	im.save(f"{name} ({core}-core).png")
	tavg = taken
	avg = count / tavg * 100000
	print(f"Benchmarked {name} ({core}-core). Average time taken for {count} iteration(s): {tavg}s")
	print(f"Score: {round(avg, 2)}")
	print(avg)
	raise SystemExit

import subprocess
try:
	import pynvml
except ImportError:
	subprocess.run([sys.executable, "-m", "pip", "install", "pynvml", "--upgrade", "--user"])

print("Loading...")
import pynvml
try:
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
except:
	DC = 0
if not DC:
	print("WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration.")

try:
	import torch, diffusers, cpuinfo
	if DC and not torch.cuda.is_available():
		raise ImportError
except ImportError:
	subprocess.run([sys.executable, "-m", "pip", "install", "py-cpuinfo", "--upgrade", "--user"])
	subprocess.run([sys.executable, "-m", "pip", "install", "diffusers", "--upgrade", "--user"])
	if DC:
		subprocess.run([sys.executable, "-m", "pip", "install", "pynvml", "--upgrade", "--user"])
	else:
		subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118", "--upgrade", "--user"])

import cpuinfo


total = 0
procs = []

info = cpuinfo.get_cpu_info()
args = [sys.executable, sys.argv[0], "cpu", info["brand_raw"], str(info["count"])]
proc = subprocess.Popen(args, stdout=subprocess.PIPE)
procs.append(proc)
for i in range(DC):
	info = pynvml.nvmlDeviceGetHandleByIndex(i)
	args = [sys.executable, sys.argv[0], f"cuda:{i}", pynvml.nvmlDeviceGetName(info), str(pynvml.nvmlDeviceGetNumGpuCores(info))]
	proc = subprocess.Popen(args, stdout=subprocess.PIPE)
	procs.append(proc)

for proc in procs:
	s = proc.stdout.readlines()
	total += float(s.pop(-1))
	print(b"\n".join(s).decode("utf-8"))
print(f"Benchmark complete. Total score: {round(total, 2)}")