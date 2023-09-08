import requests, logging, random, time, os, sys, json, concurrent.futures
import cherrypy as cp


exc = concurrent.futures.ThreadPoolExecutor(max_workers=128)
ADDRESS = "0.0.0.0"
PORT = 443
from cherrypy._cpdispatch import Dispatcher

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		p = path.strip("/")
		if p in (
			"", "index", "home", "p", "preview", "files", "file",
			"chat", "tester", "atlas", "mizatlas", "user", "login",
			"logout", "mpinsights", "createredirect",
		):
			p = "index.html"
		if os.path.exists(f"misc/web/{p}"):
			p = "raw/" + p
		elif p.split("/", 1)[0] not in ("proxy", "stream", "heartbeat", "backend"):
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

if os.path.exists("misc/x-audio.py"):
	import subprocess, threading, psutil, base64
	from traceback import print_exc
	python = sys.executable

	class AudioClientInterface:

		clients = {}
		returns = {}
		written = False
		killed = False
		communicating = None

		def __init__(self):
			self.proc = psutil.Popen([python, "x-audio.py"], cwd=os.getcwd() + "/misc", stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=65536)
			if self.communicating:
				self.communicating.join()
			self.communicating = threading.Thread(target=self.communicate, daemon=True)
			self.communicating.start()
			try:
				if os.name == "nt":
					self.proc.ionice(psutil.IOPRIO_HIGH)
				else:
					self.proc.ionice(psutil.IOPRIO_CLASS_RT, value=7)
			except:
				pass
			self.fut = concurrent.futures.Future()
			self.audio = self

		__bool__ = lambda self: self.written

		def submit(self, s, aio=False, ignore=False, timeout=48):
			key = time.time_ns() // 1000
			while key in self.returns:
				key += 1
			self.returns[key] = None
			if type(s) not in (bytes, memoryview):
				s = str(s).encode("utf-8")
			if aio:
				s = b"await " + s
			if ignore:
				s = b"!" + s
			out = (b"~", json.dumps(key).encode("ascii"), b"~", base64.b85encode(s), b"\n")
			b = b"".join(out)
			self.returns[key] = concurrent.futures.Future()
			try:
				self.fut.result()
				self.proc.stdin.write(b)
				self.proc.stdin.flush()
				resp = self.returns[key].result(timeout=timeout)
			except:
				raise
			finally:
				self.returns.pop(key, None)
			return resp

		def communicate(self):
			try:
				proc = self.proc
				i = b"~0~Fa\n" # DO NOT CHANGE THIS IS BASE64
				proc.stdin.write(i)
				proc.stdin.flush()
				while proc.is_running():
					s = proc.stdout.readline().rstrip()
					if s:
						if s.startswith(b"~"):
							s = base64.b85decode(s[1:])
							if s == b"bot.audio.returns[0].set_result(0)":
								break
						print(s)
					time.sleep(0.2)
				self.written = True
				print("Audio client responded.")
				self.fut.set_result(self)
				while proc.is_running():
					s = proc.stdout.readline()
					if not s:
						raise EOFError
					s = s.rstrip()
					if s:
						if s[:1] == b"~":
							c = memoryview(base64.b85decode(s[1:]))
							if c[:18] == b"bot.audio.returns[":
								out = False
								if c[-18:] == b"].set_result(None)":
									out = None
								elif c[-18:] == b"].set_result(True)":
									out = True
								if out is not False:
									k = int(c[18:-18])
									try:
										self.returns[k].set_result(out)
									except:
										print_exc()
									continue
							exc.submit(exec, c, bot._globals)
						else:
							print(s)
			except:
				print_exc()

		def kill(self):
			if not self.proc.is_running():
				return
			exc.submit(self.submit, "await kill()", priority=True).result(timeout=2)
			time.sleep(0.5)
			if self.proc.is_running():
				try:
					return self.proc.kill()
				except psutil.NoSuchProcess:
					pass

	# bot = AudioClientInterface()


class Server:

	cache = {}
	if os.path.exists("temp.json") and os.path.getsize("temp.json"):
		with open("temp.json", "rb") as f:
			state = json.load(f)
	else:
		state = {"/": f"https://api.mizabot.xyz:{webserver_port}"}
	session = requests.Session()

	@cp.expose
	def heartbeat(self, key, uri=""):
		assert key == discord_secret
		uri = uri or f"https://IP:{webserver_port}"
		uri = uri.replace("IP", cp.request.remote.ip)
		if self.state["/"] != uri:
			self.state["/"] = uri
			with open("temp.json", "w") as f:
				json.dump(self.state, f)
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
		rpath = "misc/web/" + rpath
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = MIMES.get(rpath.rsplit(".", 1)[-1]) or "text/html"
		if rpath in self.cache:
			return self.cache[rpath]
		with open(rpath, "rb") as f:
			self.cache[rpath] = b = f.read()
		return b

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
		if cp.request.method.upper() != "GET":
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
		)
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
	def proxy(self, url=None):
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
		)
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
	# exc.submit(app.update_net)
	# if os.path.exists("x-distribute.py"):
	# 	import subprocess
	# 	subprocess.Popen([sys.executable, "x-distribute.py"])
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
