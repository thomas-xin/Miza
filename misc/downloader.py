# Better than Curl 🥌

import os, sys, subprocess, time, math, random, concurrent.futures

try:
    import requests
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "requests"])
    import requests
os.system("color")

from traceback import print_exc
from math import *
from concurrent.futures import thread

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

utc = time.time
math.round = round

def round(x, y=None):
    try:
        if isfinite(x):
            try:
                if x == int(x):
                    return int(x)
                if y is None:
                    return int(math.round(x))
            except:
                pass
            return round_min(math.round(x, y))
        else:
            return x
    except:
        pass
    if type(x) is complex:
        return round(x.real, y) + round(x.imag, y) * 1j
    try:
        return math.round(x, y)
    except:
        pass
    return x

def round_min(x):
    if type(x) is str:
        if "." in x:
            x = x.strip("0")
            if len(x) > 8:
                x = mpf(x)
            else:
                x = float(x)
        else:
            try:
                return int(x)
            except ValueError:
                return float(x)
    if type(x) is int:
        return x
    if type(x) is not complex:
        if isfinite(x):
            y = math.round(x)
            if x == y:
                return int(y)
        return x
    else:
        if x.imag == 0:
            return round_min(x.real)
        else:
            return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)


def time_disp(s, rounded=True):
    if not isfinite(s):
        return str(s)
    if rounded:
        s = round(s)
    output = str(s % 60)
    if len(output) < 2:
        output = "0" + output
    if s >= 60:
        temp = str((s // 60) % 60)
        if len(temp) < 2 and s >= 3600:
            temp = "0" + temp
        output = temp + ":" + output
        if s >= 3600:
            temp = str((s // 3600) % 24)
            if len(temp) < 2 and s >= 86400:
                temp = "0" + temp
            output = temp + ":" + output
            if s >= 86400:
                output = str(s // 86400) + ":" + output
    else:
        output = "0:" + output
    return output


def header():
    return {
        "User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
        "DNT": "1",
        "X-Forwarded-For": ".".join(str(random.randint(0, 255)) for _ in range(4)),
    }


COLOURS = ["\x1b[38;5;16m█"]
COLOURS.extend(f"\x1b[38;5;{i}m█" for i in range(232, 256))
COLOURS.append("\x1b[38;5;15m█")
updated = False
def download(url, fn, resp=None, index=0, start=None, end=None):
    size = 0
    packet = 131072
    with open(fn, "wb") as f:
        while True:
            try:
                if not resp:
                    rheader = header()
                    if size or start or end:
                        rstart = size
                        if start:
                            rstart += start
                        rend = end - 1 if end else ""
                        r = f"bytes={rstart}-{rend}"
                        rheader["Range"] = r
                    resp = requests.get(url, headers=rheader, timeout=32, stream=True)
                with resp:
                    if index and resp.status_code >= 400:
                        if resp.status_code in (429, 503):
                            time.sleep(7 + random.random() * 4 + index / 2)
                        raise ConnectionError(resp.status_code, resp.text.rstrip())
                    try:
                        it = resp.iter_content(packet)
                    except requests.exceptions.StreamConsumedError:
                        break
                    while True:
                        try:
                            fut = submit(next, it)
                        except RuntimeError:
                            return
                        try:
                            b = fut.result(timeout=24)
                        except (ValueError, AttributeError):
                            raise StopIteration
                        if end and len(b) + size > end:
                            b = b[:end - len(b) - size]
                        if not b:
                            raise StopIteration
                        f.write(b)
                        size += len(b)
                        progress[index] = size
                        total = sum(progress.values())
                        percentage = round(total / fsize * 100, 4)
                        s = f"\r{percentage}%"
                        box = lambda i: COLOURS[round(i * (len(COLOURS) - 1))]
                        s += " " * (10 - len(s))
                        prog = "".join(box(v * threads / fsize) for v in progress.values())
                        s += prog
                        if verbose and prog != last_progress:
                            globals()["last_progress"] = prog
                            s = "\n" + s[1:]
                        s += "\x1b[38;5;7m"
                        print(s, end="")
                        updated = True
            except StopIteration:
                break
            except:
                print_exc()
                time.sleep(5)
                print(f"\nThread {index} errored, retrying...")
                packet = max(8192, packet >> 1)
            resp = None
    return fn


verbose = False
fn = None
if len(sys.argv) < 2:
    url = input("Please enter a URL to download from: ")
    threads = 1
else:
    args = list(sys.argv)
    if "-v" in args[1:]:
        args.remove("-v")
        verbose = True
    url = args[1]
    if url == "-threads":
        args.pop(1)
        threads = int(args.pop(1))
        if len(args) < 2:
            url = input("Please enter a URL to download from: ")
        else:
            url = args[1]
    else:
        threads = 1
    if len(args) >= 3:
        fn = " ".join(args[2:])


if not os.path.exists("cache"):
    os.mkdir("cache")
if not os.path.exists("files"):
    os.mkdir("files")
print("Sending sampler request...")
t = utc()
rheader = header()
resp = requests.get(url, headers=rheader, stream=True)
url = resp.url
head = {k.casefold(): v for k, v in resp.headers.items()}
progress = {}
if verbose:
    last_progress = ""
fsize = int(head.get("content-length", 1073741824))
if "bytes" in head.get("accept-ranges", ""):
    print("Accept-Ranges header found.")
    if threads == 1:
        try:
            with open("training.txt", "r", encoding="utf-8") as f:
                s = f.read()
        except FileNotFoundError:
            s = ""
        decision = {}
        for line in s.splitlines():
            spl = line.rstrip().split()
            fs, tc, tm = int(spl[0]), int(spl[1]), float(spl[2])
            try:
                ratio = decision[fs]
            except KeyError:
                ratio = decision[fs] = {}
            try:
                ratio[tc].append(tm)
            except KeyError:
                ratio[tc] = [tm]
        if decision:
            distances = ((abs(fs - fsize) / (2 + log(len(decision[fs]))), fs) for fs in decision)
            LS = sorted(distances)[0][1]
            sizes = {sum(v) / len(v) * (12 + log(k, 2)): k for k, v in decision[LS].items()}
            sizes = {k: v for k, v in sorted(sizes.items())}
            k = next(iter(sizes))
            threads = round(sizes[k] / LS * fsize)
            if LS == fsize:
                print(f"Decision tree hit: {threads}")
            else:
                print(f"Decision tree miss: {threads}")
            if random.random() >= 0.125:
                lr = max(1, round(threads / 8), round(256 / len(sizes)))
            else:
                lr = 0
            threads += random.randint(-lr, lr)
            if threads <= 1:
                threads = random.randint(1, 3)
        else:
            n = round(fsize / 4194304)
            print(f"Decision tree empty: {n}")
            threads = n
        threads = max(1, min(64, threads))
else:
    threads = 1
if not fn:
    fn = head.get("attachment-filename") or url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0] or "file"
    fn = "files/" + fn.rsplit("/", 1)[-1]
exc = concurrent.futures.ThreadPoolExecutor(max_workers=threads << 1)
submit = exc.submit
if threads > 1:
    print(f"Splitting into {threads} threads...")
    workers = [None] * threads
    load = math.ceil(fsize / threads)
    delay = 1
    for i in range(threads):
        start = i * load
        if i == threads - 1:
            end = None
        else:
            end = min(start + load, fsize)
        workers[i] = submit(download, url, f"cache/thread-{i}", resp, index=i, start=start, end=end)
        resp = None
        try:
            workers[i].result(timeout=delay)
        except concurrent.futures.TimeoutError:
            pass
        if workers[i].done() or i >= 1 and workers[i - 1].done() or i >= 2 and workers[i - 2].done():
            delay /= 2
    fut = workers[0]
    if os.path.exists(fn):
        os.remove(fn)
    fi = fut.result()
    os.rename(fi, fn)
    with open(fn, "ab") as f:
        for fut in workers[1:]:
            fi = fut.result()
            with open(fi, "rb") as g:
                while True:
                    b = g.read(4194304)
                    if not b:
                        break
                    f.write(b)
            submit(os.remove, fi)
    exc.shutdown(wait=True)
else:
    print("Resuming request using 1 thread...")
    download(url, fn, resp)
s = utc() - t
fs = os.path.getsize(fn)
with open("training.txt", "a", encoding="utf-8") as f:
    f.write(f"{fs} {threads} {s}\n")
e = ""
bps = fs / s * 8
if bps >= 1 << 10:
    if bps >= 1 << 20:
        if bps >= 1 << 30:
            if bps >= 1 << 40:
                e = "T"
                bps /= 1 << 40
            else:
                e = "G"
                bps /= 1 << 30
        else:
            e = "M"
            bps /= 1 << 20
    else:
        e = "k"
        bps /= 1 << 10
bps = str(round(bps, 4)) + " " + e
print(f"\n{fs} bytes successfully downloaded in {time_disp(s)}, average download speed {bps}bps")
