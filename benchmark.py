print("Loading...")
import pynvml
try:
    pynvml.nvmlInit()
    DC = pynvml.nvmlDeviceGetCount()
except:
    DC = 0
if not DC:
    print("WARNING: No NVIDIA GPUs detected. Please install one for AI compute acceleration.")

import torch, time, cpuinfo
compute_load = []
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
model = "runwayml/stable-diffusion-v1-5"

def bench(device, name, core):
    print(f"Benchmarking device {device}...")
    is_cuda = device.startswith("cuda:")
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16 if is_cuda else torch.float32)
    if is_cuda:
        pipe = pipe.to(device)
    # pipe.enable_attention_slicing()
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # if is_cuda:
    #     try:
    #         pipe.enable_model_cpu_offload()
    #     except AttributeError:
    #         pass
    count = 1
    taken = 0
    while True:
        t = time.time()
        data = pipe(" ".join(["water"] * 64), num_inference_steps=count)
        taken = time.time() - t
        if taken < 15:
            count <<= 1
            continue
        break
    for i in range(3):
        t = time.time()
        data = pipe(" ".join(["water"] * 64), num_inference_steps=count)
        taken += time.time() - t
    tavg = taken / 4
    avg = count / tavg * 100000
    print(f"Benchmarked {name} ({core}-core). Average time taken for {count} iteration(s): {tavg}")
    print(f"Score: {round(avg, 2)}")
    return avg

total = 0
info = cpuinfo.get_cpu_info()
total += bench("cpu", info["brand_raw"], info["count"])
for i in range(torch.cuda.device_count()):
    info = pynvml.nvmlDeviceGetHandleByIndex(i)
    total += bench(f"cuda:{i}", pynvml.nvmlDeviceGetName(info), pynvml.nvmlDeviceGetNumGpuCores(info))
print(f"Benchmark complete. Total score: {round(total, 2)}")