import requests, logging, random, datetime, time, os, sys, re, json, concurrent.futures, urllib.parse
import cherrypy as cp
from traceback import print_exc
url_unparse = urllib.parse.unquote_plus

exc = concurrent.futures.ThreadPoolExecutor(max_workers=128)
ADDRESS = "0.0.0.0"
PORT = 443
from cherrypy._cpdispatch import Dispatcher

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		p = path.strip("/")
		first = p.split("/", 1)[0]
		# print(path, p, first)
		if not p or p == "dummy.html":
			p = "raw/index.html"
		elif first in ("home", "index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"):
			if first not in ("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"):
				p = "raw/index.html"
		elif os.path.exists(f"misc/web/{p}"):
			p = "raw/" + p
		elif first not in ("proxy", "u", "unproxy", "stream", "heartbeat", "backend", "debug"):
			p = "backend/" + p
		p = "/" + p
		return super().__call__(p)

config = {
	"global": {
		"server.socket_host": ADDRESS,
		"server.socket_port": PORT,
		"server.thread_pool": 128,
		"server.max_request_body_size": 0,
		"server.socket_timeout": 65,
		"server.ssl_module": "builtin",
		"engine.autoreload_on": True,
		"tools.gzip.on": True,
		"tools.gzip.mime_types": ["text/*", "application/json", "application/javascript"],
	},
	"/": {
		"request.dispatch": EndpointRedirects(),
	},
}
if os.path.exists("domain.cert.pem") and os.path.exists("private.key.pem"):
	config["global"]["server.ssl_certificate"] = "domain.cert.pem"
	config["global"]["server.ssl_private_key"] = "private.key.pem"
if os.path.exists("auth.json"):
	with open("auth.json", "rb") as f:
		AUTH = json.load(f)
	discord_secret = AUTH.get("discord_secret") or ""
	webserver_port = AUTH.get("webserver_port") or "9801"
else:
	discord_secret = ""
	webserver_port = "9801"

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
SHEADERS = {"Cache-Control": "public, max-age=5, stale-while-revalidate=1073741824, stale-if-error=1073741824"}
CHEADERS.update(HEADERS)
SHEADERS.update(HEADERS)

MIMES = dict(
	bin="application/octet-stream",
	css="text/css",
	json="application/json",
	js="application/javascript",
	txt="text/plain",
	html="text/html",
	svg="image/svg+xml",
	ico="image/x-icon",
	png="image/png",
	jpg="image/jpeg",
	gif="image/gif",
	webp="image/webp",
	webm="video/webm",
	mp3="audio/mpeg",
	ogg="audio/ogg",
	opus="audio/opus",
	flac="audio/flac",
	wav="audio/x-wav",
	mp4="video/mp4",
)

__scales = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")

def byte_scale(n, ratio=1024):
	e = 0
	while n >= ratio:
		n /= ratio
		e += 1
		if e >= len(__scales) - 1:
			break
	return f"{round(n, 4)} {__scales[e]}"

is_discord_attachment = lambda url: url and re.search("^https?:\\/\\/(?:[A-Za-z]{3,8}\\.)?discord(?:app)?\\.(?:com|net)\\/attachments\\/", url)
def discord_expired(url):
	if is_discord_attachment(url):
		if "?ex=" not in url and "&ex=" not in url:
			return True
		temp = url.replace("?ex=", "&ex=").split("&ex=", 1)[-1].split("&", 1)[0]
		try:
			ts = int(temp, 16)
		except ValueError:
			return True
		return ts < time.time() + 60


class Server:

	cache = {}
	ucache = {}
	if os.path.exists("temp.json") and os.path.getsize("temp.json"):
		with open("temp.json", "rb") as f:
			state = json.load(f)
	else:
		state = {"/": f"https://api.mizabot.xyz:{webserver_port}"}
	session = requests.Session()

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	def index(self, path=None, filename=None, *args, code=None, **kwargs):
		HOST = self.state["/"] if self.state["/"].startswith("https://") else cp.request.base
		url = HOST + "/" + cp.url(qs=cp.request.query_string).rstrip("/").split("//", 1)[-1].split("/", 1)[-1]
		if "/p/" in url:
			raise cp.HTTPRedirect(url.replace("/p/", "/file/"), status=307)
		if "/preview/" in url:
			raise cp.HTTPRedirect(url.replace("/preview/", "/file/"), status=307)
		if "/upload" in url:
			raise cp.HTTPRedirect(url.replace("/upload", "/files"), status=307)
		rpath = "misc/web/index.html"
		mime = MIMES.get(rpath.rsplit(".", 1)[-1]) or "text/html"
		if rpath in self.cache:
			data = self.cache[rpath]
		else:
			with open(rpath, "rb") as f:
				self.cache[rpath] = data = f.read()
		if url.split("//", 1)[-1].count("/") > 1:
			meta = '<meta property="og:title" content="Miza"><meta property="og:description" content="A multipurpose Discord bot.">'
			if "/file" in url or "/files" in url:
				meta += '<meta property="og:image" content="/mizaleaf.png">'
			else:
				meta += '<meta property="og:image" content="/logo256.png">'
			meta += '<meta property="og:site_name" content="Miza">'
			if not random.randint(0, 1) and (dt := datetime.datetime.utcnow()) and (dt.month, dt.day) in ((3, 31), (4, 1), (4, 2)):
				meta += f'<meta http-equiv="refresh" content={random.randint(15, 31)};url=https://{cp.request.headers["Host"]}/teapot">'
			if path:
				irl = HOST + "/fi/" + path
				if irl not in self.ucache or time.time() - self.ucache[irl][0] > 3600:
					with self.session.get(irl, timeout=30) as resp:
						info = resp.json()
					self.ucache[irl] = [time.time(), info]
				elif time.time() - self.ucache[irl][0] > 60:
					def cache_temp():
						with self.session.get(irl, timeout=30) as resp:
							info = resp.json()
						self.ucache[irl] = [time.time(), info]
					exc.submit(cache_temp)
					info = self.ucache[irl][1]
				else:
					info = self.ucache[irl][1]
				fn = info["filename"]
				mim = info["mimetype"]
				attachment = filename or fn
				size = info["size"]
				a2 = url_unparse(attachment).removeprefix(".temp$@")
				f_url = info["raw"]
				description = mim + f", {byte_scale(size)}B"
				meta = '<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">'
				if mim.startswith("image/") and mim.split("/", 1)[-1] in ("png", "jpg", "jpeg", "webp", "gif") and size < 1048576:
					i_url = f_url
					meta += f"""<meta name="twitter:image:src" content="{i_url}"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{a2}"><meta property="twitter:url" content="{f_url}"><meta property="og:image" content="{i_url}"><meta property="og:image:type" content="{mim}"><meta property="og:url" content="{f_url}"><meta name="og:description" content="{description}">"""
				elif mim.split("/", 1)[0] in ("image", "video", "audio"):
					i_url = f_url.replace("/f/", "/i/") + ".png" if mim.startswith("video") else HOST + "/mizaleaf.png"
					r_url = f_url.replace("/f/", "/r/") + ".webm" if mim.startswith("audio") else f_url + "." + mim.split("/", 1)[-1]
					if mim.startswith("audio/"):
						dims = '<meta property="og:video:width" content="640"><meta property="og:video:height" content="64">'
					else:
						dims = '<meta property="og:video:width" content="960"><meta property="og:video:height" content="540">'
					meta += f"""<meta property="og:type" content="video.other"><meta property="twitter:player" content="{r_url}"><meta property="og:video:type" content="{mim}"><meta property="og:url" content="{f_url}">{dims}<meta name="twitter:image" content="{i_url}">"""
			else:
				a2 = "Miza"
				description = "A multipurpose Discord bot."
			i = data.index(b'</title>') + 8
			s = """<!doctype html><html lang="en"><head>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
	<meta charset="utf-8"/><link rel="icon" href="/logo256.png"/><meta charset="utf-8"><meta name="author" content="Miza"><meta name="viewport" content="width=device-width,initial-scale=1"/><meta name="theme-color" content="#694777"/><link rel="apple-touch-icon" href="/logo256.png"/><link rel="manifest" href="/manifest.json"/>""" + meta
			t = f'<title>{a2}</title><meta name="description" content="{description}"/>'
			data = s.encode("utf-8") + t.encode("utf-8") + data[i:]
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		return data

	@cp.expose
	def heartbeat(self, key, uri=""):
		assert key == discord_secret
		uri = uri or f"https://IP:{webserver_port}"
		uri = uri.replace("IP", cp.request.remote.ip)
		if self.state["/"] != uri:
			self.state["/"] = uri
			with open("temp.json", "w") as f:
				json.dump(self.state, f)
		if len(self.ucache) > 1048576:
			for k, v in tuple(self.ucache.items()):
				if isinstance(v, list) and discord_expired(v[1]):
					self.ucache.pop(k, None)
		return "💜"

	@cp.expose
	def audio(self, key, uri=""):
		assert key == discord_secret
		uri = uri or f"https://IP:{webserver_port}"
		uri = uri.replace("IP", cp.request.remote.ip)
		if self.state["/"] != uri:
			self.state["/"] = uri
			with open("temp.json", "w") as f:
				json.dump(self.state, f)
		return "💜"

	@cp.expose
	def raw(self, *path, **query):
		rpath = "/".join(path)
		rpath = "misc/web/" + (rpath or "index.html")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = MIMES.get(rpath.rsplit(".", 1)[-1]) or "text/html"
		if rpath.strip("/") == "notfound.png":
			cp.response.status = 404
		if rpath in self.cache:
			return self.cache[rpath]
		with open(rpath, "rb") as f:
			self.cache[rpath] = b = f.read()
		return b

	@cp.expose(("u",))
	def unproxy(self, *path, **query):
		rpath = "/".join(path)
		if rpath:
			rpath = "/" + rpath
		rquery = cp.request.query_string
		irl = f"{self.state['/']}/u{rpath}"
		if irl not in self.ucache or discord_expired(self.ucache[irl][1]) or (irl == self.ucache[irl][1] or self.ucache[irl][1] == "https://mizabot.xyz/notfound.png" and time.time() - self.ucache[irl][0] > 30):
			headers = dict(cp.request.headers)
			headers.pop("Connection", None)
			headers.pop("Transfer-Encoding", None)
			headers["X-Real-Ip"] = cp.request.remote.ip
			try:
				with self.session.head(irl, headers=headers, verify=False, allow_redirects=False, timeout=30) as resp:
					resp.raise_for_status()
					url = resp.headers.get("Location") or irl
			except Exception as ex:
				print("Error:", repr(ex))
				if irl in self.ucache:
					url = self.ucache[irl][1]
				else:
					url = "https://mizabot.xyz/notfound.png"
			self.ucache[irl] = [time.time(), url]
		elif time.time() - self.ucache[irl][0] > 43200:
			def cache_temp():
				headers = dict(cp.request.headers)
				headers.pop("Connection", None)
				headers.pop("Transfer-Encoding", None)
				headers["X-Real-Ip"] = cp.request.remote.ip
				with self.session.head(irl, headers=headers, verify=False, allow_redirects=False, timeout=30) as resp:
					resp.raise_for_status()
					url = resp.headers.get("Location") or irl
				self.ucache[irl] = [time.time(), url]
			exc.submit(cache_temp)
			url = self.ucache[irl][1]
		else:
			url = self.ucache[irl][1]
		if rquery:
			rquery = "?" + rquery if "?" not in url else "&" + rquery
			url += rquery
		raise cp.HTTPRedirect(url, 307)

	@cp.expose
	# @cp.tools.accept(media="multipart/form-data")
	def backend(self, *path, **query):
		rpath = "/".join(path)
		if rpath:
			rpath = "/" + rpath
		rquery = cp.request.query_string
		if rquery:
			rquery = "?" + rquery
		url = f"{self.state['/']}{rpath}{rquery}"
		if cp.request.method.upper() != "GET" or self.state["/"].startswith("https://"):
			raise cp.HTTPRedirect(url, 307)
		headers = dict(cp.request.headers)
		headers.pop("Connection", None)
		headers.pop("Transfer-Encoding", None)
		headers["X-Real-Ip"] = cp.request.remote.ip
		print("BACKEND:", url)
		resp = self.session.get(
			url,
			headers=headers,
			stream=True,
			verify=False,
			allow_redirects=False,
			timeout=3600,
		)
		if resp.status_code in range(300, 400):
			raise cp.HTTPRedirect(resp.headers.get("Location") or url, resp.status_code)
		cp.response.headers.update(resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		if int(resp.headers.get("Content-Length") or 262145) <= 262144:
			print("HEADERS:", cp.response.headers)
			return resp.content
		print("HEADERS:", cp.response.headers)
		return resp.iter_content(65536)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def proxy(self, url=None, **void):
		if not url:
			return "Expected proxy URL."
		try:
			body = cp.request.body.fp.read()
		except:
			print_exc()
			body = None
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
			"DNT": "1",
			"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
			"X-Real-Ip": ".".join(str(random.randint(1, 254)) for _ in range(4)),
		}
		headers.pop("Connection", None)
		headers.pop("Transfer-Encoding", None)
		if cp.request.headers.get("Range"):
			headers["Range"] = cp.request.headers["Range"]
		resp = self.session.request(
			cp.request.method.upper(),
			url,
			headers=headers,
			data=body,
			stream=True,
			verify=False,
			timeout=60,
		)
		cp.response.headers.update(resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		return resp.iter_content(65536)

	@cp.expose
	def debug(self):
		cp.response.headers["Content-Type"] = "application/json"
		return json.dumps(self.ucache).encode("utf-8")

	@cp.expose
	def stream(self, info=None):
		if not info:
			return "Expected info URL."
		try:
			data = self.cache[info]
		except KeyError:
			if len(self.cache) > 128:
				self.cache.pop(next(iter(self.cache)))
			data = self.cache[info] = requests.get(info, timeout=30).json()
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
						resp = reqs.head(u, headers=headers, timeout=45)
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
							resp = reqs.get(u, headers=h2, stream=True, timeout=30)
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
	# exc.submit(app.update_net)
	# if os.path.exists("x-distribute.py"):
	# 	import subprocess
	# 	subprocess.Popen([sys.executable, "x-distribute.py"])
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
