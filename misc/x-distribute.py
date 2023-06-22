import os
if os.name != "nt":
    raise SystemExit("Compute distribution currently only supported on Windows due to drivers, sorry!")

import sys, traceback, subprocess, concurrent.futures
from traceback import print_exc
python = sys.executable

if sys.version_info[0] < 3:
    raise ImportError("Python 3 required.")

os.system("color")
srgb = lambda r, g, b, s: f"\033[38;2;{r};{g};{b}m{s}\033[0m"

try:
    import psutil, cpuinfo, gpustat
except ImportError:
    subprocess.run([python, "-m", "pip", "install", "psutil", "--upgrade", "--user"])
    subprocess.run([python, "-m", "pip", "install", "py-cpuinfo", "--upgrade", "--user"])
    subprocess.run([python, "-m", "pip", "install", "gpustat", "--upgrade", "--user"])
    import psutil, cpuinfo, gpustat
import pynvml
try:
    q = gpustat.new_query()
except pynvml.NVMLError_LibraryNotFound:
    q = None

print(srgb(0, 0, 255, "Scanning hardware..."))
exc = concurrent.futures.ThreadPoolExecutor(max_workers=4)
memories = {}
fut = exc.submit(cpuinfo.get_cpu_info)

if not q:
    print(srgb(255, 0, 0, "WARNING: No NVIDIA GPU(s) detected. Hardware-acclerated compute will not be available."))
else:
    try:
        import torch
    except ImportError:
        subprocess.run([python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118", "--upgrade", "--user"])
        import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            memories[gpu.name] = gpu.total_memory * 1048576
    else:
        print(srgb(255, 0, 0, "WARNING: Pytorch is installed but no CUDA support enabled. If you have NVIDIA GPU(s) installed, please make sure your drivers are functional, then reinstall pytorch-cuda using") + srgb(0, 255, 0, "`py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`") + srgb(255, 0, 0, "."))

cpu = fut.result()
memories[cpu["brand_raw"]] = psutil.virtual_memory().total
total_memory = sum(memories.values())

if total_memory <= 3 * 1073741824:
    raise RuntimeError("Insufficient usable memory on device. Recommended minimum total is 4GB.")

libs = "asyncio, psutil, requests, blend_modes, filetype, colorspace, orjson, numpy, pillow, pillow_heif, pyqrcode, pygame, matplotlib, wand, pdf2image, sympy, mpmath".split(", ")
if len(memories) > 1:
    libs += "selenium, openai, httpx, markdownify, diffusers, transformers, tiktoken, accelerate, freeproxy".split(", ")
try:
    import pkg_resources
except:
    print_exc()
    subprocess.run(["pip", "install", "setuptools", "--upgrade", "--user"])
    import pkg_resources
for lib in libs:
    try:
        pkg_resources.get_distribution(lib)
    except ImportError:
        subprocess.run([python, "-m", "pip", "install", lib, "--upgrade", "--user"])
if len(memories) > 1:
    try:
        import bitsandbytes
    except ImportError:
        subprocess.run([python, "-m", "pip", "install", "bitsandbytes-windows", "--upgrade", "--user"])


print(srgb(255, 0, 255, "Detected compute devices:"))
for k, v in memories.items():
    gb = v / 1073741824
    if gb == int(gb):
        gb = int(gb)
    c2 = (0, 255, 0) if gb > 11 else (255, 255, 0) if gb > 7 else (255, 127, 0) if gb > 3 else (255, 0, 0)
    print(srgb(0, 255, 255, k) + "\t\t\t" + srgb(0, 255, 0, gb) + " GB")


import threading
new_tasks = {}

def task_submit(ptype, command, fix=None, _timeout=12):
    ts = ts_us()
    proc = random.choice([proc for proc in procs if not proc.busy or proc.busy.done()])
    proc = await get_idle_proc(ptype, fix=fix)
    while ts in PROC_RESP:
        ts += 1
    PROC_RESP[ts] = fut = concurrent.futures.Future()
    # print("PROC_RESP:", PROC_RESP.keys())
    command = "[" + ",".join(map(repr, command[:2])) + "," + ",".join(map(str, command[2:])) + "]"
    s = f"~{ts}~".encode("ascii") + base64.b64encode(command.encode("utf-8")) + b"\n"
    # s = f"~{ts}~{repr(command.encode('utf-8'))}\n".encode("utf-8")
    sem = proc.sem
    if fix:
        # sem.clear()
        sem = emptyctx
    else:
        await sem()
    if not is_strict_running(proc):
        proc = await get_idle_proc(ptype, fix=fix)
    async with sem:
        try:
            proc.stdin.write(s)
            await proc.stdin.drain()
            fut = PROC_RESP[ts]
            tries = ceil(_timeout / 3) if _timeout and is_finite(_timeout) else 3600
            for i in range(tries):
                if ts not in PROC_RESP:
                    raise ConnectionResetError("Response disconnected.")
                try:
                    resp = await asyncio.wait_for(wrap_future(fut), timeout=3)
                except T1:
                    if i >= tries - 1:
                        raise
                else:
                    break
            else:
                raise OSError("Max waits exceeded.")
        except (BrokenPipeError, OSError) as ex:
            try:
                i = PROCS[ptype].index(proc)
            except (LookupError, ValueError):
                raise ex
            force_kill(proc)
            PROCS[ptype][i] = None
            raise
        finally:
            PROC_RESP.pop(ts, None)
    create_task(wait_sub())
    return resp

def update_resps(proc):
    def func():
        while True:
            with tracebacksuppressor:
                if not is_strict_running(proc):
                    return
                b = await proc.stdout.readline()
                if not b:
                    return
                # s = as_str(b.rstrip())
                # if s and s[0] == "~":
                #     c = as_str(evalEX(s[1:]))
                #     exec_tb(c, globals())
            s = b.rstrip()
            try:
                if s and s[:1] == b"$":
                    s, r = s.split(b"~", 1)
                    # print("PROC_RESP:", s, PROC_RESP.keys())
                    d = {"_x": base64.b64decode(r)}
                    c = evalex(memoryview(s)[1:], globals(), d)
                    if isinstance(c, (str, bytes, memoryview)):
                        exec_tb(c, globals(), d)
                elif s and s[:1] == b"~":
                    c = evalex(memoryview(s)[1:], globals())
                    if isinstance(c, (str, bytes, memoryview)):
                        exec_tb(c, globals())
                else:
                    print(lim_str(as_str(s), 1048576))
            except:
                print_exc()
                print(s)
    return func

def update_tasks(proc):
    def func():
        resps = {}
        while proc.is_running():
            resp = base64.b64encode(orjson.dumps(resps))
            resp = session.get(f"https://mizabot.xyz/api/distribute?caps=[{proc.cap}]&resp={resp}")
            data = resp.json()
            print(data)
            for task in data:
                cap = task[1]
                new_tasks.setdefault(cap, []).append(task)
            tasks = new_tasks.get(cap, [])
            if not tasks:
                if proc.waiting:
                    proc.waiting.result()
                    proc.waiting = concurrent.futures.Future()
                else:
                    time.sleep(0.5)
                continue
            if tasks:
                i, cap, command, timeout = tasks.pop(0)
                try:
                    resp = task_submit("compute", command, fix=proc.i, _timeout=timeout)
                except Exception as ex:
                    resps[i] = "ERR:" + repr(ex)
                else:
                    resps[i] = "RES:" + resp if isinstance(resp, str) else resp
    return func

procs = []
for k, v in memories.items():
    if not i:
        cap = "0" if total_memory < 7 * 1073741824 else "2"
    else:
        cap = str(i + 3)
    args = [python, "x-compute.py", cap]
    proc = psutil.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=262144,
    )
    proc.busy = None
    proc.waiting = concurrent.futures.Future()
    proc.cap = min(3, int(cap))
    proc.i = len(procs)
    procs.append(proc)
    threading.Thread(target=update_tasks(proc)).start()
try:
    import time, requests, orjson, base64
    class Self:
        _cpuinfo = None
        ip_time = 0
        up_bps = 0
        down_bps = 0
    self = Self()
    session = requests.Session()

    caps = []
    tasks = []
    responses = {}
    while True:
        # if time.time() - self.ip_time > 60:
        #     fut = exc.submit(requests.get, "https://api.ipify.org", verify=False)
        #     self.ip_time = time.time()
        # else:
        #     fut = None
        cinfo = self._cpuinfo
        if not cinfo:
            cinfo = self._cpuinfo = cpuinfo.get_cpu_info()
        cpercent = psutil.cpu_percent()
        try:
            import torch, gpustat
            ginfo = gpustat.new_query()
        except:
            ginfo = []
        minfo = psutil.virtual_memory()
        sinfo = psutil.swap_memory()
        dinfo = {}
        for p in psutil.disk_partitions(all=False):
            try:
                dinfo[p.mountpoint] = psutil.disk_usage(p.mountpoint)
            except OSError:
                pass
        # if fut:
        #     resp = fut.result()
        #     self.ip = resp.text
        # ip = self.ip
        ip = "<IP>"
        t = time.time()
        def get_usage(gi):
            try:
                return float(gi["utilization.gpu"]) / 100
            except ValueError:
                pass
            try:
                return gi.power_draw / gi.power_limit
            except (ValueError, TypeError, ZeroDivisionError):
                return 0
        def try_float(f):
            try:
                return float(f)
            except ValueError:
                return 0
        stats = orjson.dumps(dict(
            cpu={ip: dict(name=cinfo["brand_raw"], count=cinfo["count"], usage=cpercent / 100, max=1, time=t)},
            gpu={f"{ip}-{gi['index']}": dict(
                name=gi["name"],
                count=torch.cuda.get_device_properties(gi["index"]).multi_processor_count,
                usage=get_usage(gi),
                max=1,
                time=t,
            ) for gi in ginfo},
            memory={
                f"{ip}-v": dict(name="RAM", count=1, usage=minfo.used, max=minfo.total, time=t),
                f"{ip}-s": dict(name="Swap", count=1, usage=sinfo.used, max=sinfo.total, time=t),
                **{f"{ip}-{gi['index']}": dict(
                    name=gi["name"],
                    count=1,
                    usage=try_float(gi["memory.used"]) * 1048576,
                    max=try_float(gi["memory.total"]) * 1048576,
                    time=t,
                ) for gi in ginfo},
            },
            disk={f"{ip}-{k}": dict(name=k, count=1, usage=v.used, max=v.total, time=t) for k, v in dinfo.items()},
            network={
                ip: dict(name="Upstream", count=1, usage=self.up_bps, max=-1, time=t),
                ip: dict(name="Downstream", count=1, usage=self.down_bps, max=-1, time=t),
            },
        ))
        caps = [proc.cap for proc in procs if not proc.busy or proc.busy.done()]
        stat = base64.b64encode(stats).rstrip(b"=").decode("ascii")
        resp = session.get(f"https://mizabot.xyz/api/distribute?caps={caps}&stat={stat}")
        data = resp.json()
        print(data)
        for task in data:
            cap = task[1]
            new_tasks.setdefault(cap, []).append(task)
        if data:
            for proc in procs:
                if proc.waiting:
                    proc.waiting.set_result(None)
                    proc.waiting = None
        time.sleep(1)
finally:
    for proc in procs:
        try:
            proc.kill()
        except:
            pass
    exc.shutdown()