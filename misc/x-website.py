import os, simplejson, requests, hashlib, logging, waitress, random, base64, concurrent.futures
import cherrypy as cp


exc = concurrent.futures.ThreadPoolExecutor(max_workers=128)
nhash = lambda s: int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little")

def create_etag(data):
	n = len(data)
	s = str(nhash(data[:128] + data[n // 2 - 128:(n + 1) // 2 + 128] + data[-128:]) + n & 4294967295)
	return '"' + "0" * (10 - len(s)) + s + '"'

config = {}

HEADERS = {
	"X-Content-Type-Options": "nosniff",
	"Server": "Miza",
	"Vary": "Accept-Encoding",
	"Accept-Ranges": "bytes",
	"Access-Control-Allow-Headers": "*",
	"Access-Control-Allow-Methods": "*",
	"Access-Control-Allow-Origin": "*",
}
CHEADERS = {"Cache-Control": "public, max-age=3600, stale-while-revalidate=1073741824, stale-if-error=1073741824"}

MIMES = {}
STATIC = {}

def fetch_static(path, ignore=False):
	while path.startswith("../"):
		path = path[3:]
	try:
		data = STATIC[path]
	except KeyError:
		fn = path
		fn2 = fn + ".zip"
		with open(fn, "rb") as f:
			data = f.read()
		STATIC[path] = data
	fmt = path.rsplit(".", 1)[-1].casefold()
	try:
		mime = MIMES[fmt]
	except KeyError:
		mime = "text/html"
	return data, mime


class Server:

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	def index(self, *args, **kwargs):
		url = cp.url(qs=cp.request.query_string).rstrip("/")
		if "/upload" in url:
			raise cp.HTTPRedirect(url.replace("/upload", "/files"), status=307)
		data, mime = fetch_static("index.html")
		if url.split("//", 1)[-1].count("/") > 1:
			raise RuntimeError("Unexpected proxy URL.")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	def stream(self, info):
		data = requests.get(info).json()
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
						end = size
					elif not end:
						end = size
					start = int(start)
					end = int(end)
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
	waitress.serve(server, host="0.0.0.0", port=8080, url_scheme="https")
