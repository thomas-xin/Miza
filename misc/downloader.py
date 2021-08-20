# Better than Curl 🥌

import os, sys, requests, time, math, random, concurrent.futures
from traceback import print_exc
from math import *

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


class cdict(dict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        if not k.startswith("__") or not k.endswith("__"):
            try:
                return self.__getitem__(k)
            except KeyError as ex:
                raise AttributeError(*ex.args)
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k.startswith("__") and k.endswith("__"):
            return object.__setattr__(self, k, v)
        return self.__setitem__(k, v)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(self)
        return data

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


def header():
    return {
        "User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
        "DNT": "1",
        "X-Forwarded-For": ".".join(str(random.randint(0, 255)) for _ in range(4)),
    }


updated = False
def download(url, fn, resp=None, index=0, start=None, end=None):
    size = 0
    packet = 1048576
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
                    try:
                        it = resp.iter_content(packet)
                    except requests.exceptions.StreamConsumedError:
                        break
                    while True:
                        try:
                            #b = next(it)
                            fut = submit(next, it)
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
                        s += " " * (64 - len(s))
                        print(s, end="")
                        updated = True
            except StopIteration:
                break
            except:
                print_exc()
                time.sleep(5)
                print(f"Thread {index} errored, retrying...")
                packet = 65536
            resp = None
    return fn


fn = None
if len(sys.argv) < 2:
    url = input("Please enter a URL to download from: ")
    threads = 1
else:
    args = list(sys.argv)
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
head = cdict((k.casefold(), v) for k, v in resp.headers.items())
progress = {}
fsize = int(head.get("content-length", 0))
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
            sizes = {sum(v) / len(v) * (6 + log(k)): k for k, v in decision[LS].items()}
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
    load = fsize // threads
    for i in range(threads):
        start = i * load
        if i == threads - 1:
            end = None
        else:
            end = start + load
        workers[i] = submit(download, url, f"cache/thread-{i}", resp, index=i, start=start, end=end)
        resp = None
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
