import orjson, requests
import cherrypy as cp

class Server:

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	@hostmap
	def index(self, path=None, filename=None, *args, code=None, **kwargs):
		url = cp.url(qs=cp.request.query_string).rstrip("/")
		if "/p/" in url:
			raise cp.HTTPRedirect(url.replace("/p/", "/file/"), status=307)
		if "/preview/" in url:
			raise cp.HTTPRedirect(url.replace("/preview/", "/file/"), status=307)
		if "/upload" in url:
			raise cp.HTTPRedirect(url.replace("/upload", "/files"), status=307)
		data, mime = fetch_static("index.html")
		if url.split("//", 1)[-1].count("/") > 1:
			meta = '<meta property="og:title" content="Miza"><meta property="og:description" content="A multipurpose Discord bot.">'
			if "/file" in url or "/files" in url:
				meta += '<meta property="og:image" content="/mizaleaf.png">'
			else:
				meta += '<meta property="og:image" content="/logo256.png">'
			meta += '<meta property="og:site_name" content="Miza">'
			if not xrand(2) and (dt := datetime.datetime.utcnow()) and (dt.month, dt.day) in ((3, 31), (4, 1), (4, 2)):
				meta += f'<meta http-equiv="refresh" content={xrand(15, 31)};url=https://{cp.request.headers["Host"]}/teapot">'
			if path:
				ind = IND
				p = None
				if path.startswith("!"):
					ind = "!"
					path = path[1:]
				elif not path.startswith("@"):
					b = path.lstrip("~").split(".", 1)[0].encode("utf-8") + b"=="
					if b.startswith(b"dQ"):
						c = b[2:]
						path = str(p2n(c))
						try:
							p = find_file(path, ind=ind)
						except FileNotFoundError:
							pass
						else:
							url = cp.request.base + "/i/" + c.rstrip(b"=").decode("utf-8", "replace") + ".gif"
							return f"""<!DOCTYPE html>
	<html><head>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
	<meta property="og:url" content="/">
	<meta property="og:type" content="video.other">
	<meta property="twitter:player" content="https://www.youtube.com/embed/dQw4w9WgXcQ">
	<meta property="og:video:type" content="text/html">
	<meta property="og:video:width" content="960">
	<meta property="og:video:height" content="720">
	<meta name="twitter:image" content="{url}">
	<meta http-equiv="refresh" content="0;url=https://www.youtube.com/watch?v=dQw4w9WgXcQ">
	</head><body></body></html>""".encode("utf-8")
					path = str(p2n(b))
				else:
					path = path[1:]
				if not p:
					p = find_file(path, cwd=("cache", "saves/filehost"), ind=ind)
				sem = SEMAPHORES.get(p)
				if not sem:
					while len(SEMAPHORES) >= 4096:
						sem = SEMAPHORES.pop(next(iter(SEMAPHORES)))
						if sem.is_busy():
							raise SemaphoreOverflowError
					sem = SEMAPHORES[p] = Semaphore(256, 256, rate_limit=60)
				with sem:
					fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
					attachment = filename or fn
					a2 = url_unparse(attachment).removeprefix(".temp$@")
					f_url = url.replace("/file/", "/f/")
					mim = get_mime(p)
					description = mim + f", {byte_scale(os.path.getsize(p))}B"
					meta = '<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">'
					if mim.startswith("video/"):
						i_url = url.replace("/file/", "/i/") + ".gif"
						meta += f"""<meta property="og:type" content="video.other">\
	<meta property="twitter:player" content="{f_url}">\
	<meta property="og:video:type" content="{mim}">\
	<meta property="og:video:width" content="960">\
	<meta name="twitter:image" content="{i_url}">"""
					else:
						i_url = url.replace("/file/", "/r/") + ".gif"
						meta += f"""<meta name="twitter:image:src" content="{i_url}">\
	<meta name="twitter:card" content="summary_large_image">\
	<meta name="twitter:title" content="{a2}"><meta property="twitter:url" content="{f_url}"><meta property="og:image" content="{i_url}">\
	<meta property="og:image:type" content="{mim}"><meta property="og:url" content="{f_url}"><meta name="og:description" content="{description}">"""
			else:
				a2 = "Miza"
				description = "A multipurpose Discord bot."
			i = data.index(b'</title>') + 8
			s = """<!doctype html><html lang="en"><head>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
	<meta charset="utf-8"/><link rel="icon" href="/logo256.png"/>\
	<meta charset="utf-8"><meta name="author" content="Miza">\
	<meta name="viewport" content="width=device-width,initial-scale=1"/>\
	<meta name="theme-color" content="#694777"/>\
	<link rel="apple-touch-icon" href="/logo256.png"/>\
	<link rel="manifest" href="/manifest.json"/>""" + meta
			t = f'<title>{a2}</title><meta name="description" content="{description}"/>'
			data = s.encode("utf-8") + t.encode("utf-8") + data[i:]
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

async def main(req):
	rhead = req.headers
	urls = orjson.loads(req.get_body())
	size = rhead.get("X-File-Size", 0)
	rhead.pop("Remote-Addr", None)
	rhead.pop("Host", None)
	rhead.pop("Range", None)
	brange = req.headers.get("Range", "")
	brange = brange.removeprefix("bytes=")
	ranges = []
	length = 0
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
	if ranges:
		status = 206
	else:a
		status = 200
		ranges.append((0, size))
		length = size
	if not size:
		size = "*"
	cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size}" for start, end in ranges)
	headers = {}
	headers["Content-Range"] = cr
	headers["Content-Length"] = str(length)
	headers["Accept-Range"] = "bytes"
	return func.HttpResponse(
		body=_dyn_serve(urls, ranges, rhead)
		status_code=status,
		headers=headers,
	)

def _dyn_serve(urls, ranges, headers):
	reqs = requests.Session()
	try:
		for start, end in ranges:
			pos = 0
			rems = urls.copy()
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
							yield resp.content[s:e]
							return
						yield resp.content
						return
					if big:
						yield from resp.iter_content(262144)
						return
					yield from resp.iter_content(65536)

				yield from get_chunk(u, headers, start, end, pos, ns, big)
				pos = 0
				start = 0
				end -= start + ns
				big = True
	except GeneratorExit:
		pass