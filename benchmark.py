import sys
if len(sys.argv) > 1:
    device, name, core = sys.argv[1:]
    print(f"Benchmarking device {device}...")
    is_cuda = device.startswith("cuda:")
    if is_cuda:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
        device = "cuda:0"
    import torch, time
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
    count = 4
    taken = 0
    temp = []
    mark = lambda *args: temp.append(time.time())
    while True:
        temp.clear()
        data = pipe(" ".join(["water"] * 64), num_inference_steps=count, callback=mark)
        delays = [temp[i] - temp[i - 1] for i in range(2, len(temp))]
        taken = sum(delays)
        if taken < 45:
            count = max(count * 2, round(60 / taken * count))
            continue
        break
    tavg = taken
    avg = count / tavg * 100000
    print(f"Benchmarked {name} ({core}-core). Average time taken for {count} iteration(s): {tavg}")
    print(f"Score: {round(avg, 2)}")
    print(avg)
    raise SystemExit

print("Loading...")
import pynvml
try:
    pynvml.nvmlInit()
    DC = pynvml.nvmlDeviceGetCount()
except:
    DC = 0
if not DC:
    print("WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration.")

import cpuinfo, subprocess


total = 0
info = cpuinfo.get_cpu_info()
args = [sys.executable, sys.argv[0], "cpu", info["brand_raw"], str(info["count"])]
proc = subprocess.Popen(args, stdout=subprocess.PIPE)
s = proc.stdout.readlines()
total += float(s.pop(-1))
print(b"\n".join(s).decode("utf-8"))
for i in range(DC):
    info = pynvml.nvmlDeviceGetHandleByIndex(i)
    args = [sys.executable, sys.argv[0], f"cuda:{i}", pynvml.nvmlDeviceGetName(info), str(pynvml.nvmlDeviceGetNumGpuCores(info))]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    s = proc.stdout.readlines()
    total += float(s.pop(-1))
    print(b"\n".join(s).decode("utf-8"))
print(f"Benchmark complete. Total score: {round(total, 2)}")