import os, simplejson, requests, hashlib, logging, waitress, random, base64, concurrent.futures
import cherrypy as cp


exc = concurrent.futures.ThreadPoolExecutor(max_workers=128)
config = {}


class Server:

	cache = {}

	@cp.expose
	def stream(self, info):
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
					if u.startswith("https://s3-us-west-2"):
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
	waitress.serve(server, threads=128, host="0.0.0.0", port=8080, url_scheme="https")
