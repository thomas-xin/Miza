import requests, logging, random, time, sys, json, concurrent.futures
import cherrypy as cp


exc = concurrent.futures.ThreadPoolExecutor(max_workers=128)
ADDRESS = "0.0.0.0"
PORT = 8080
config = {
	"global": {
		"server.socket_host": ADDRESS,
		"server.socket_port": PORT,
		"server.thread_pool": 128,
		"server.max_request_body_size": 0,
		"server.socket_timeout": 65,
		"server.ssl_module": "builtin",
		"engine.autoreload_on": False,
	},
}


class Server:

	cache = {}

	_cpuinfo = None
	up_bps = down_bps = 0
	ip_time = 0
	ip = "127.0.0.1"
	@cp.expose
	def stat(self, api=None, **kwargs):
		if api == "ytdl" and "q" in kwargs:
			if not getattr(self, "downloader", None):
				spectralpulse = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/spectralpulse"
				sys.path.append(spectralpulse)
				import audio_downloader
				self.downloader = audio_downloader.AudioDownloader()
			entries = self.downloader.search(kwargs["q"])
			self.downloader.get_stream(entries[0], force=True, download=False)
			return json.dumps(entries)
		import psutil, cpuinfo
		if time.time() - self.ip_time > 60:
			fut = exc.submit(requests.get, "https://api.ipify.org")
			self.ip_time = time.time()
		else:
			fut = None
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
		dinfo = {p.mountpoint: psutil.disk_usage(p.mountpoint) for p in psutil.disk_partitions(all=False)}
		if fut:
			resp = fut.result()
			self.ip = resp.text
		ip = self.ip
		t = time.time()
		def get_usage(gi):
            try:
                return float(gi["utilization.gpu"]) / 100
            except ValueError:
                pass
            try:
                return = gi.power_draw / gi.power_limit
            except:
                return 0
		return json.dumps(dict(
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
					usage=gi["memory.used"] * 1048576,
					max=gi["memory.total"] * 1048576,
					time=t,
				) for gi in ginfo},
			},
			disk={f"{ip}-{k}": dict(name=k, count=1, usage=v.used, max=v.total, time=t) for k, v in dinfo.items()},
			network={
				ip: dict(name="Upstream", count=1, usage=self.up_bps, max=-1, time=t),
				ip: dict(name="Downstream", count=1, usage=self.down_bps, max=-1, time=t),
			},
		))

	def update_net(self):
		ninter = 3
		while True:
			t = time.time()
			net = psutil.net_io_counters()
			if not hasattr(self, "up_bytes"):
				self.up_bytes = deque(maxlen=3)
				self.down_bytes = deque(maxlen=3)
				self.start_up = -net.bytes_sent
				self.start_down = -net.bytes_recv
			self.up_bytes.append(net.bytes_sent)
			self.down_bytes.append(net.bytes_recv)
			self.up_bps = (self.up_bytes[-1] - self.up_bytes[0]) * 8 / len(self.up_bytes) / ninter
			self.down_bps = (self.down_bytes[-1] - self.down_bytes[0]) * 8 / len(self.down_bytes) / ninter
			time.sleep(max(0, t - time.time() + ninter))

	@cp.expose
	def proxy(self, url=None):
		if not url:
			return "Expected proxy URL."
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
		}
		if cp.request.headers.get("Range"):
			headers["Range"] = cp.request.headers["Range"]
		resp = requests.get(url, headers=headers, stream=True)
		cp.response.headers.update(resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		return resp.iter_content(65536)

	@cp.expose
	def stream(self, info=None):
		if not info:
			return "Expected info URL."
		try:
			data = self.cache[info]
		except KeyError:
			if len(self.cache) > 128:
				self.cache.pop(next(iter(self.cache)))
			data = self.cache[info] = requests.get(info).json()
		info = [data["filename"], data["size"], data["mimetype"]]
		urls = data.get("chunks") or [data["dl"]]
		size = info[1]
		disp = "filename=" + info[0]
		cp.response.headers["Content-Disposition"] = disp
		cp.response.headers["Content-Type"] = info[2]
		cp.response.headers["Attachment-Filename"] = info[0]
		brange = cp.request.headers.get("Range", "").removeprefix("bytes=")
		headers = cp.request.headers.copy()
		headers.pop("Remote-Addr", None)
		headers.pop("Host", None)
		headers.pop("Range", None)
		headers.update({
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
		})
		ranges = []
		length = 0
		if brange:
			try:
				branges = brange.split(",")
				for s in branges:
					start, end = s.split("-", 1)
					if not start:
						if not end:
							continue
						start = size - int(end)
						end = size - 1
					elif not end:
						end = size - 1
					start = int(start)
					end = int(end) + 1
					length += end - start
					ranges.append((start, end))
			except:
				pass
		if ranges:
			cp.response.status = 206
		else:
			cp.response.status = 200
			ranges.append((0, size))
			length = size
		if not size:
			size = "*"
		cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size}" for start, end in ranges)
		cp.response.headers["Content-Range"] = cr
		cp.response.headers["Content-Length"] = str(length)
		cp.response.headers["Accept-Ranges"] = "bytes"
		return self._dyn_serve(urls, ranges, headers)

	def _dyn_serve(self, urls, ranges, headers):
		reqs = requests.Session()
		try:
			for start, end in ranges:
				pos = 0
				rems = urls.copy()
				futs = []
				big = False
				while rems:
					u = rems.pop(0)
					if "?size=" in u:
						u, ns = u.split("?size=", 1)
						ns = int(ns)
					elif u.startswith("https://s3-us-west-2"):
						ns = 503316480
					elif u.startswith("https://cdn.discord"):
						ns = 8388608
					else:
						resp = reqs.head(u, headers=headers)
						ns = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
					if pos + ns <= start:
						pos += ns
						continue
					if pos >= end:
						break

					def get_chunk(u, h, start, end, pos, ns, big):
						s = start - pos
						e = end - pos
						if e >= ns:
							e = ""
						else:
							e -= 1
						h2 = dict(h.items())
						h2["range"] = f"bytes={s}-{e}"
						ex2 = None
						for i in range(3):
							resp = reqs.get(u, headers=h2, stream=True)
							if resp.status_code == 416:
								yield b""
								return
							try:
								resp.raise_for_status()
							except Exception as ex:
								ex2 = ex
							else:
								break
						if ex2:
							raise ex2
						if resp.status_code != 206:
							ms = min(ns, end - pos - s)
							if len(resp.content) > ms:
								yield resp.content[s:(e or len(resp.content))]
								return
							yield resp.content
							return
						if big:
							yield from resp.iter_content(262144)
							return
						yield from resp.iter_content(65536)

					if len(futs) > 1:
						yield from futs.pop(0).result()
					fut = exc.submit(get_chunk, u, headers, start, end, pos, ns, big)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns
					big = True
				for fut in futs:
					yield from fut.result()
		except GeneratorExit:
			pass


if __name__ == "__main__":
	logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
	app = Server()
	self = server = cp.Application(app, "/", config)
	exc.submit(app.update_net)
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
