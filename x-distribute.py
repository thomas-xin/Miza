import os
if os.name != "nt":
    raise SystemExit("Compute distribution currently only supported on Windows due to drivers, sorry!")

import sys
from traceback import print_exc
python = sys.executable

if sys.version_info[0] < 3:
    raise ImportError("Python 3 required.")

if not os.path.exists("auth.json"):
    with open("auth.json", "wb") as f:
        f.write(b"{}")
if os.path.exists("x-compute.py"):
    sys.path.append("..")

import benchmark, json

with open("auth.json", "rb") as f:
    data = json.load(f)
compute_load = data.get("compute_load", [])

if benchmark.DC:
    caps = [0, 2] if benchmark.DC > 1 else [2]
    import pynvml
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(benchmark.DC)]
    gmems = [pynvml.nvmlDeviceGetMemoryInfo(d).total for d in handles]
    if sum(gmems) > 35 * 1073741824:
        caps.insert(-1, 1)
    for i, mem in enumerate(gmems):
        if mem <= 3 * 1073741824:
            continue
        if compute_load[i] < 300000:
            continue
        caps.append(i + 3)
else:
    caps = [0]

req = [
    "orjson",
    "requests",
    "pillow",
    "blend_modes",
    "sympy",
    "mpmath",
    "matplotlib",
]
import pkg_resources
for mn in req:
    try:
        pkg_resources.get_distribution(mn)
    except:
        subprocess.run([sys.executable, "-m", "pip", "install", mn, "--upgrade", "--user"])

import time, base64, orjson, psutil, subprocess, threading, requests, concurrent.futures
from math import *
session = requests.Session()

exc = concurrent.futures.ThreadPoolExecutor(max_workers=4 + benchmark.DC)

new_tasks = {}
procs = []
PROC_RESP = {}

def task_submit(proc, command, _timeout=12):
    ts = time.time_ns() // 1000
    while ts in PROC_RESP:
        ts += 1
    PROC_RESP[ts] = fut = concurrent.futures.Future()
    command = "[" + ",".join(map(repr, command[:2])) + "," + ",".join(map(str, command[2:])) + "]"
    s = f"~{ts}~".encode("ascii") + base64.b64encode(command.encode("utf-8")) + b"\n"
    print(s)
    proc.busy = concurrent.futures.Future()
    try:
        proc.stdin.write(s)
        proc.stdin.flush()
        fut = PROC_RESP[ts]
        resp = fut.result(timeout=_timeout)
    except (BrokenPipeError, OSError) as ex:
        proc.kill()
        procs.remove(proc)
        start_proc(proc.cap)
        raise
    finally:
        PROC_RESP.pop(ts, None)
        try:
            proc.busy.set_result(None)
        except:
            pass
        proc.busy = None
    # create_task(wait_sub())
    return resp

def update_resps(proc):
    def func():
        while True:
            try:
                if not proc.is_running():
                    return
                b = proc.stdout.readline()
                if not b:
                    return
                # s = as_str(b.rstrip())
                # if s and s[0] == "~":
                #     c = as_str(evalEX(s[1:]))
                #     exec_tb(c, globals())
            except:
                print_exc()
            s = b.rstrip()
            try:
                if s and s[:1] == b"$":
                    s, r = s.split(b"~", 1)
                    # print("PROC_RESP:", s, PROC_RESP.keys())
                    d = {"_x": base64.b64decode(r)}
                    c = eval(memoryview(s)[1:], globals(), d)
                    if isinstance(c, (str, bytes, memoryview)):
                        exec(c, globals(), d)
                elif s and s[:1] == b"~":
                    c = eval(memoryview(s)[1:], globals())
                    if isinstance(c, (str, bytes, memoryview)):
                        exec(c, globals())
                else:
                    print(s)
            except:
                print_exc()
                print(s)
    return func

def update_tasks(proc):
    def func():
        resps = {}
        while proc.is_running():
            resp = base64.urlsafe_b64encode(orjson.dumps(resps)).decode("utf-8", "replace")
            resp = session.post(
                "https://mizabot.xyz/api/distribute",
                data=dict(
                    caps=[proc.cap],
                    resp=resp,
                )
            )
            # resp = session.get(f"https://mizabot.xyz/api/distribute?caps=[{proc.cap}]&resp={resp}")
            resp.raise_for_status()
            data = resp.json()
            print(data)
            for task in data:
                cap = task[1]
                new_tasks.setdefault(cap, []).append(task)
            tasks = new_tasks.get(proc.cap, [])
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
                    resp = task_submit(proc, command, _timeout=timeout)
                except Exception as ex:
                    resps[str(i)] = "ERR:" + repr(ex)
                else:
                    resps[str(i)] = "RES:" + resp if isinstance(resp, str) else resp
    return func

def start_proc(cap):
    args = [python, "misc/x-compute.py", str(cap)]
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
    procs.append(proc)
    threading.Thread(target=update_tasks(proc)).start()
    threading.Thread(target=update_resps(proc)).start()
    return proc


for cap in caps:
    start_proc(cap)
try:
    import time, requests, orjson, base64, cpuinfo
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
        ip = "<IP>"
        t = time.time()
        cinfo = self._cpuinfo
        if not cinfo:
            cinfo = self._cpuinfo = cpuinfo.get_cpu_info()
        cpercent = psutil.cpu_percent()
        try:
            import pynvml
            dc = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(dc)]
            gname = [pynvml.nvmlDeviceGetName(d) for d in handles]
            gcore = [pynvml.nvmlDeviceGetNumGpuCores(d) for d in handles]
            gmems = [pynvml.nvmlDeviceGetMemoryInfo(d) for d in handles]
            gutil = [pynvml.nvmlDeviceGetUtilizationRates(d) for d in handles]
            gpowa = [pynvml.nvmlDeviceGetPowerUsage(d) for d in handles]
            gpowb = [pynvml.nvmlDeviceGetEnforcedPowerLimit(d) for d in handles]
        except:
            gname = []
        minfo = psutil.virtual_memory()
        sinfo = psutil.swap_memory()
        dinfo = {}
        for p in psutil.disk_partitions(all=False):
            try:
                dinfo[p.mountpoint] = psutil.disk_usage(p.mountpoint)
            except OSError:
                pass
        stats = orjson.dumps(dict(
            cpu={ip: dict(name=cinfo["brand_raw"], count=cinfo["count"], usage=cpercent / 100, max=1, time=t)},
            gpu={f"{ip}-{i}": dict(
                name=name,
                count=gcore[i],
                usage=gutil[i].gpu / 100,
                max=1,
                time=t,
            ) for i, name in enumerate(gname)},
            memory={
                f"{ip}-v": dict(name="RAM", count=1, usage=minfo.used, max=minfo.total, time=t),
                f"{ip}-s": dict(name="Swap", count=1, usage=sinfo.used, max=sinfo.total, time=t),
                **{f"{ip}-{i}": dict(
                    name=name,
                    count=1,
                    usage=gmems[i].used,
                    max=gmems[i].total,
                    time=t,
                ) for i, name in enumerate(gname)},
            },
            disk={f"{ip}-{k}": dict(name=k, count=1, usage=v.used, max=v.total, time=t) for k, v in dinfo.items()},
            network={
                f"{ip}-u": dict(name="Upstream", count=1, usage=self.up_bps, max=-1, time=t),
                f"{ip}-d": dict(name="Downstream", count=1, usage=self.down_bps, max=-1, time=t),
            },
            power={
                **{f"{ip}-{i}": dict(
                    name=name,
                    count=1,
                    usage=gpowa[i] / 1000,
                    max=gpowb[i] / 1000,
                    time=t,
                ) for i, name in enumerate(gname)},
            },
        ))
        caps = [proc.cap for proc in procs if not proc.busy or proc.busy.done()]
        stat = base64.urlsafe_b64encode(stats).rstrip(b"=").decode("ascii")
        resp = session.post(
            "https://mizabot.xyz/api/distribute",
            data=dict(
                caps=caps,
                stat=stat,
            )
        )
        # resp = session.get(f"https://mizabot.xyz/api/distribute?caps={caps}&stat={stat}")
        resp.raise_for_status()
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
    procs.clear()
    exc.shutdown()
    psutil.Process().kill()