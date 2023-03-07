try:
	from common import *
except ModuleNotFoundError:
	try:
		f = open("common.py", "rb")
	except FileNotFoundError:
		import os, sys
		sys.path.append(os.path.abspath('..'))
		os.chdir("..")
		f = open("common.py", "rb")
	b = f.read()
	code = compile(b, "common.py", "exec", optimize=1)
	exec(code, globals())
import weakref


try:
	RAPIDAPI_SECRET = AUTH["rapidapi_secret"]
except KeyError:
	RAPIDAPI_SECRET = None
try:
	KOFI_SECRET = AUTH["kofi_secret"]
except KeyError:
	KOFI_SECRET = None


HOST = "https://mizabot.xyz"
ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
PORT = AUTH.get("webserver_port") or 80
IND = "\x7f"


def send(*args, escape=True):
	try:
		s = " ".join(str(i) for i in args)
		if escape:
			s = "\x00" + s
		if s:
			if s[-1] != "\n":
				s += "\n"
			s = s.encode("utf-8")
			# sys.stdout.buffer.write(s)
			sys.__stderr__.buffer.write(s)
			sys.__stderr__.flush()
	except OSError:
		force_kill(psutil.Process())

print = send
print_exc = lambda: send(traceback.format_exc())

def create_etag(data):
	n = len(data)
	s = str(nhash(data[:128] + data[n // 2 - 128:(n + 1) // 2 + 128] + data[-128:]) + n & 4294967295)
	return '"' + "0" * (10 - len(s)) + s + '"'


SEMAPHORES = {}
STATIC = {}
TZCACHE = {}
RESPONSES = {}
RESPONSES[0] = cdict(set_result=lambda *args: None)

PREVIEW = {}
prev_date = utc_dt().date()
zfailed = set()


import cherrypy, cheroot, logging, ssl
from cherrypy._cpdispatch import Dispatcher
cp = cherrypy
from cheroot import errors
errors.SSLEOFError = ssl.SSLEOFError

def serve(self):
	while self.ready and not self.interrupt:
		try:
			self._connections.run(self.expiration_interval)
		except (KeyboardInterrupt, SystemExit):
			raise
		except Exception as ex:
			self.error_log(
				'Error in HTTPServer.serve', level=logging.ERROR,
				traceback=False,
			)
			print(repr(ex))
cheroot.server.HTTPServer.serve = serve

def communicate(self):
	request_seen = False
	try:
		req = self.RequestHandlerClass(self.server, self)
		req.parse_request()
		if self.server.stats['Enabled']:
			self.requests_seen += 1
		if not req.ready:
			return False
		request_seen = True
		req.respond()
		if not req.close_connection:
			return True
	except socket.error as ex:
		errnum = ex.args[0]
		# sadly SSL sockets return a different (longer) time out string
		timeout_errs = 'timed out', 'The read operation timed out', 'EOF occurred in violation of protocol'
		if errnum in timeout_errs:
			if (not request_seen) or (req and req.started_request):
				self._conditional_error(req, '408 Request Timeout')
		elif errnum not in errors.socket_errors_to_ignore:
			self.server.error_log(
				'socket.error %s' % repr(errnum),
				level=logging.WARNING, traceback=True,
			)
			self._conditional_error(req, '500 Internal Server Error')
	except (KeyboardInterrupt, SystemExit):
		raise
	except (errors.FatalSSLAlert, errors.SSLEOFError) as ex:
		print(repr(ex))
	except errors.NoSSLError:
		self._handle_no_ssl(req)
	except Exception as ex:
		self.server.error_log(
			repr(ex), level=logging.ERROR, traceback=True,
		)
		self._conditional_error(req, '500 Internal Server Error')
	return False
cheroot.server.HTTPServer.communicate = communicate

actually_static = set(os.listdir("misc/static"))
mapped_static = {k[:-5]: k for k in actually_static if k.endswith(".html")}

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		send("@@@", escape=False)
		if path[1:] == AUTH.get("discord_token"):
			path = "/backup"
		elif path == "/ip":
			path = "/get_ip"
		elif path[:3] == "/f/":
			path = "/raw/" + path[3:]
		elif path == "/upload":
			path = "/files"
		elif path == "/api/mpinsights":
			path = "/api_mpinsights"
		elif path == "/api/status":
			path = "/api_status"
		else:
			p = path.lstrip("/")
			if p in actually_static:
				path = "/static/" + p
			elif p in mapped_static:
				path = "/static/" + mapped_static[p]
		return Dispatcher.__call__(self, path)

config = {
	"global": {
		"server.socket_host": ADDRESS,
		"server.socket_port": PORT,
		"server.thread_pool": 96,
		"server.max_request_body_size": 0,
		"server.socket_timeout": 60,
		"server.ssl_module": "builtin",
		"engine.autoreload_on": False,
	},
	"/": {
		"request.dispatch": EndpointRedirects(),
	}
}
if os.path.exists("domain.cert.pem") and os.path.exists("private.key.pem"):
	config["global"]["server.ssl_certificate"] = "domain.cert.pem"
	config["global"]["server.ssl_private_key"] = "private.key.pem"
	def hostmap(func):
		def decorator(*args, **kwargs):
			url = cp.url(qs=cp.request.query_string)
			if not url.startswith("https://mizabot.") and not url.startswith("https://i.mizabot."):
				if url.startswith("https://csnftcg.mizabot."):
					raise cp.HTTPRedirect(f"https://csnftcg.mizabot.xyz:9604/{url.rsplit('/', 1)[-1]}", 307)
				# time.sleep(10)
				# raise cp.HTTPRedirect(f"https://mizabot.xyz/{url.rsplit('/', 1)[-1]}", 307)
			return func(*args, **kwargs)
		return decorator
	def e404(status, message, traceback, version):
		url = cp.url(qs=cp.request.query_string)
		if not url.startswith("https://mizabot.") and not url.startswith("https://i.mizabot."):
			time.sleep(3600)
		return message
	cp.config["error_page.404"] = e404
else:
	def hostmap(func):
		def decorator(*args, **kwargs):
			return func(*args, **kwargs)
		return decorator

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


def fetch_static(path, ignore=False):
	while path.startswith("../"):
		path = path[3:]
	try:
		try:
			data = STATIC[path]
		except KeyError:
			fn = f"misc/{path}"
			fn2 = fn + ".zip"
			if os.path.exists(fn2) and zipfile.is_zipfile(fn2):
				with ZipFile(fn2, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as z:
					data = z.read(path.rsplit("/", 1)[-1])
			else:
				with open(fn, "rb") as f:
					data = f.read()
			STATIC[path] = data
		fmt = path.rsplit(".", 1)[-1].casefold()
		try:
			mime = MIMES[fmt]
		except KeyError:
			mime = "text/html"
		return data, mime
	except:
		if not ignore:
			send(path)
			print_exc()
		raise

est_time = utc()
est_last = -inf

def estimate_life():
	return inf
	global est_time, est_last
	with tracebacksuppressor:
		hosted = sorted(int(f[1:].split("~", 1)[0]) / 1e6 for f in os.listdir("cache") if f.startswith(IND))
		if not hosted:
			est_last = -inf
			ts = 0
		else:
			ts = hosted[0]
		res = self.bot_exec(f"bot.storage_ratio")
		try:
			last = (utc() - ts) / res
		except ZeroDivisionError:
			last = inf
		send(last)
		est_time = utc() - last
		est_last = utc()

# estimate_life_after = lambda t: time.sleep(t) or estimate_life()

# create_future_ex(estimate_life_after, 10)

geo_sem = Semaphore(90, 256, rate_limit=60)
geo_count = 0
IP = None

def get_geo(ip):
	global geo_count, IP
	if ip.startswith("192.168."):
		ip = IP
		if not ip:
			ip = IP = reqs.next().get("https://api.ipify.org").text
	try:
		resp = TZCACHE[ip]
	except KeyError:
		if geo_count & 1:
			url = f"http://ip-api.com/json/{ip}?fields=256"
		else:
			url = f"https://demo.ip-api.com/json/{ip}?fields=256&key=test-demo-pro"
		geo_count += 1
		with geo_sem:
			resp = reqs.next().get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}", "Origin": "https://members.ip-api.com"})
		send("@@@", escape=False)
		resp.raise_for_status()
		resp = cdict(resp.json())
		if not resp.get("timezone"):
			resp.timezone = "N/A"
		TZCACHE[ip] = resp
		send(ip + "\t" + "\t".join(resp.values()))
	return resp


class Server:

	serving = {}

	@cp.expose
	@hostmap
	def fileinfo(self, path, **void):
		orig_path = path
		ind = IND
		p = None
		cp.response.headers.update(SHEADERS)
		if path.startswith("!"):
			ind = "!"
			path = path[1:]
		elif not path.startswith("@"):
			b = path.lstrip("~").split(".", 1)[0].encode("utf-8") + b"=="
			if (len(b) - 1) & 3 == 0:
				b += b"="
			path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
		else:
			path = path[1:]
		if not p:
			p = find_file(path, ind=ind)
		mime = get_mime(p)
		f_url = cp.url(qs=cp.request.query_string).replace("/fileinfo/", "/f/")
		st = os.stat(p)
		fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
		if fn.startswith(".temp$@"):
			fn = fn[7:]
			a3 = True
		else:
			a3 = False
		t = utc()
		ti = max(st.st_atime + 30 * 86400, st.st_ctime + 60 * 86400, t)
		d = dict(
			id=p.rsplit("/", 1)[-1].split("~", 1)[0].lstrip(IND),
			filename=fn,
			ttl=-1,
			size=st.st_size,
			mimetype=mime,
			raw=f_url,
			dl=f_url.replace("/f/", "/d/", 1),
			timestamp=max(st.st_mtime, st.st_ctime),
			in_cache=a3,
		)
		if p.endswith("~.forward$") and mime == "text/html":
			with open(p, "r", encoding="utf-8") as f:
				resp = f.read(1048576)
			s = resp
			search = "<!DOCTYPE HTML><!--"
			if s.startswith(search):
				s = s[len(search):]
				search = '--><html><meta'
				try:
					s = s[:s.index(search)]
				except ValueError:
					pass
				else:
					url, code, ftype = orjson.loads(s)
					if ftype == 3:
						s = resp.split("/>", 1)[-1]
						infd, urld, _ = s.split("-->", 2)
						info = orjson.loads(infd.removeprefix("<!--"))
						urls = orjson.loads(urld.removeprefix("<!--"))
						d["filename"] = info[0]
						d["size"] = info[1]
						d["mimetype"] = info[2]
					else:
						d["original_url"] = url
		return orjson.dumps(d)

	image_loaders = {}

	@cp.expose(("animate", "animation", "a", "images", "image", "i", "view", "v", "raw", "f", "download", "d"))
	@hostmap
	def files(self, path, filename=None, download=None, **void):
		if path in ("hacks", "mods", "files", "download", "static"):
			send(cp.request.remote.ip + " was rickrolled 🙃")
			raise cp.HTTPRedirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ", status=301)
		orig_path = path
		ind = IND
		p = None
		cp.response.headers.update(CHEADERS)
		if path.startswith("!"):
			ind = "!"
			path = path[1:]
		elif not path.startswith("@"):
			b = path.lstrip("~").split(".", 1)[0].encode("ascii") + b"=="
			if b.startswith(b"dQ"):
				c = b[2:]
				if (len(c) - 1) & 3 == 0:
					c += b"="
				path = str(int.from_bytes(base64.urlsafe_b64decode(c), "big"))
				try:
					p = find_file(path, ind=ind)
				except FileNotFoundError:
					pass
				else:
					url = cp.request.base + "/i/" + c.rstrip(b"=").decode("ascii", "replace") + ".gif"
					return f"""<!DOCTYPE html>
<html><head>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta property="og:type" content="video.other">
<meta property="twitter:player" content="https://www.youtube.com/embed/dQw4w9WgXcQ">
<meta property="og:video:type" content="text/html">
<meta property="og:video:width" content="960">
<meta property="og:video:height" content="720">
<meta name="twitter:image" content="{url}">
<meta http-equiv="refresh" content="0;url=https://www.youtube.com/watch?v=dQw4w9WgXcQ">
</head><body></body></html>"""
			if (len(b) - 1) & 3 == 0:
				b += b"="
			path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
		else:
			path = path[1:]
		if not p:
			p = find_file(path, ind=ind)
		sem = SEMAPHORES.get(p)
		if not sem:
			while len(SEMAPHORES) >= 4096:
				sem = SEMAPHORES.pop(next(iter(SEMAPHORES)))
				if sem.is_busy():
					raise SemaphoreOverflowError
			sem = SEMAPHORES[p] = Semaphore(256, 256, rate_limit=60)
		with sem:
			endpoint = cp.url(qs=cp.request.query_string, base="")[1:].split("/", 1)[0]
			download = download and download[0] not in "0fFnN" or endpoint.startswith("d")
			mime = get_mime(p)
			st = os.stat(p)
			fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
			attachment = filename or fn
			a2 = url_unparse(attachment)
			if a2.startswith(".temp$@"):
				a2 = a2[7:]
				a3 = True
			else:
				a3 = False
			cp.response.headers["Attachment-Filename"] = a2
			if endpoint.startswith("r") and (mime in ("image/webp", "image/apng") or mime.split("/", 1)[0] == "video"):
				preview = "cache/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".gif"
				image_loaders = self.image_loaders
				if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
					args = (
						"./ffmpeg",
						"-nostdin",
						"-hide_banner",
						"-v",
						"error",
						"-err_detect",
						"ignore_err",
						"-fflags",
						"+discardcorrupt+genpts+igndts+flush_packets",
						"-hwaccel",
						hwaccel,
						"-an",
						"-i",
						p,
						"-loop",
						"0",
						"-fs",
						"1048576",
						"-vf",
						"scale=240:-1",
						preview,
					)
					print(args)
					proc = psutil.Popen(args)
					image_loaders[preview] = proc
				cp.response.headers["Content-Type"] = "image/gif"
				cp.response.headers["ETag"] = create_etag(p)
				while preview in image_loaders and (not os.path.exists(preview) or os.path.getsize(preview) < 524288) and is_strict_running(image_loaders[preview]):
					time.sleep(0.05)
				f = None
				if preview in image_loaders and not is_strict_running(image_loaders[preview]) or preview not in image_loaders and os.path.exists(preview):
					cp.response.headers["Content-Length"] = os.path.getsize(preview)
				elif preview in image_loaders:
					f = DownloadingFile(
						preview,
						lambda: not is_strict_running(image_loaders[preview]),
					)
				if not f:
					if os.path.getsize(preview):
						f = open(preview, "rb")
					else:
						cp.response.headers["Content-Type"] = get_mime(p)
						f = open(p, "rb")
				return cp.lib.file_generator(f, 262144)
			elif endpoint.startswith("i") and (mime in ("image/webp", "image/apng") or mime.split("/", 1)[0] == "video"):
				preview = "cache/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".png"
				image_loaders = self.image_loaders
				if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
					args = (
						"./ffmpeg",
						"-nostdin",
						"-hide_banner",
						"-v",
						"error",
						"-err_detect",
						"ignore_err",
						"-fflags",
						"+discardcorrupt+genpts+igndts+flush_packets",
						"-hwaccel",
						hwaccel,
						"-an",
						"-i",
						p,
						"-vframes",
						"1",
						preview,
					)
					print(args)
					proc = psutil.Popen(args)
					image_loaders[preview] = proc
				cp.response.headers["Content-Type"] = "image/png"
				cp.response.headers["ETag"] = create_etag(p)
				while preview in image_loaders and (not os.path.exists(preview) or os.path.getsize(preview) < 524288) and is_strict_running(image_loaders[preview]):
					time.sleep(0.05)
				f = None
				if preview in image_loaders and not is_strict_running(image_loaders[preview]) or preview not in image_loaders and os.path.exists(preview):
					cp.response.headers["Content-Length"] = os.path.getsize(preview)
				elif preview in image_loaders:
					f = DownloadingFile(
						preview,
						lambda: not is_strict_running(image_loaders[preview]),
					)
				if not f:
					if os.path.getsize(preview):
						f = open(preview, "rb")
					else:
						cp.response.headers["Content-Type"] = get_mime(p)
						f = open(p, "rb")
				return cp.lib.file_generator(f, 262144)
			elif endpoint.startswith("a") and mime.split("/", 1)[0] in "video":
				f_url = cp.url(qs=cp.request.query_string).replace(f"/{endpoint}/", "/f/")
				i_url = f_url.replace("/f/", "/r/") + ".gif"
				b = ("""<!DOCTYPE html>
<html>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
<style>
.center {
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}
</style>""" + f"""
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta name="twitter:image:src" content="{i_url}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{a2}">
<meta property="og:image" content="{i_url}">
<body style="background-color:black;">
<video class="center" playsinline autoplay muted loop>
  <source src="{f_url}">
</video>
</body>
</html>""").encode("utf-8")
				cp.response.headers["Content-Type"] = "text/html"
				cp.response.headers["Content-Length"] = len(b)
				cp.response.headers["ETag"] = create_etag(b)
				return b
			elif not os.path.exists(p):
				raise FileNotFoundError(p)
			elif p.endswith("~.forward$") and mime == "text/html" and os.path.getsize(p) < 1048576:
				with open(p, "r", encoding="utf-8") as f:
					resp = f.read(1048576)
				s = resp
				search = "<!DOCTYPE HTML><!--"
				if s.startswith(search):
					s = s[len(search):]
					search = '--><html><meta'
					try:
						s = s[:s.index(search)]
					except ValueError:
						pass
					else:
						url, code, ftype = orjson.loads(s)
						if ftype == 1:
							data = resp.encode("utf-8")
							cp.response.headers["Location"] = url
							cp.response.headers["Content-Type"] = mime
							cp.response.headers["Content-Length"] = len(data)
							cp.response.headers["ETag"] = create_etag(data)
							cp.response.status = int(code)
							return data
						elif ftype == 2:
							headers = fcdict(cp.request.headers)
							headers.pop("Remote-Addr", None)
							headers.pop("Host", None)
							headers.update(Request.header())
							resp = reqs.next().get(url, headers=headers, stream=True)
							resp.raw.decode_content = False
							headers = fcdict(resp.headers)
							cd = headers.get("Content-Disposition")
							if cd:
								if not download:
									headers["Content-Disposition"] = cd.replace("attachment;", "").strip()
								elif "attachment" not in cd:
									headers["Content-Disposition"] = "attachment;" + cd
							elif download:
								headers["Content-Disposition"] = "attachment"
							cp.response.headers.update(("-".join(w.capitalize() for w in k.split("-")), v) for k, v in headers.items())
							send(s)
							send(headers)
							send(cp.response.headers)
							cp.response.status = int(resp.status_code)
							if float(fcdict(resp.headers).get("Content-Length", inf)) <= 16777216:
								b = resp.content
								cp.response.headers["Content-Type"] = magic.from_buffer(b)
								return b
							f = resp.raw
							# f = ForwardedRequest(resp, 98304)
							return cp.lib.file_generator(f, 262144)
						elif ftype == 3:
							s = resp.split("/>", 1)[-1]
							infd, urld, _ = s.split("-->", 2)
							info = orjson.loads(infd.removeprefix("<!--"))
							urls = orjson.loads(urld.removeprefix("<!--"))
							disp = "filename=" + info[0]
							cp.response.headers["Content-Disposition"] = disp
							# cp.response.headers["Content-Length"] = info[1]
							cp.response.headers["Content-Type"] = info[2]
							referrer = cp.request.headers.get("Referer")
							print(p, len(urls), referrer)
							cp.response.headers["Attachment-Filename"] = info[0]
							if download and len(urls) == 1 and not referrer:
								raise cp.HTTPRedirect("https://cdn.discordapp.com/attachments/" + urls[0][2:], status="307")
							cp.response.headers.pop("Accept-Ranges", None)
							return self.concat(p, urls, name=info[0], mime=info[2])
			f = open(p, "rb")
			resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=a2)
			if a3:
				self.serving.setdefault(p, weakref.WeakSet()).add(f)
			return resp
	files._cp_config = {"response.stream": True}

	def concat(self, fn, urls, name="", download=False, mime=None):
		on = fn.replace("~.forward$", "!.temp$@" + name)
		pn = fn.replace("~.forward$", "~.temp$@" + name)
		try:
			fut = self.serving[on]
			for i in range(3):
				if os.path.exists(on):
					break
				time.sleep(1)
			else:
				if not os.path.exists(on) and not os.path.exists(pn):
					self.serving.pop(on, None)
					raise KeyError
		except KeyError:
			fut = create_future_ex(self._concat, urls, on, pn)
			self.serving[on] = fut
		for i in range(120):
			if os.path.exists(pn):
				break
			if os.path.exists(on) and self.serving.get(on + "~buffer") or fut.done():
				break
			time.sleep(0.5)
		if os.path.exists(pn):
			with open(pn, "rb") as f:
				resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=name)
				self.serving.setdefault(pn, weakref.WeakSet()).add(f)
				yield from resp
			self.serving.setdefault(pn, weakref.WeakSet()).discard(f)
			return
		with open(on, "rb") as f:
			while not fut.done() and on in self.serving:
				b = f.read(262144)
				if not b:
					continue
				yield b
			while True:
				b = f.read(262144)
				if not b:
					return
				yield b

	def _concat(self, urls, on, pn):
		print("Cat", urls)
		with tracebacksuppressor:
			headers = fcdict(cp.request.headers)
			headers.pop("Remote-Addr", None)
			headers.pop("Host", None)
			headers.update(Request.header())
			pos = 0
			futs = []
			with open(on, "wb") as f:
				for url in urls:
					if len(futs) >= 8:
						futs.pop(0).result()
					if url.startswith("D$"):
						url = "https://cdn.discordapp.com/attachments/" + url[2:]
					for i in range(6):
						try:
							resp = reqs.next().get(url, headers=headers, stream=True)
							resp.raise_for_status()
							break
						except:
							print_exc()
						time.sleep(i ** 2 + 1)
					fs = pos + int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length"))
					f.truncate(fs)
					futs.append(create_future_ex(self.chunk_into, resp, on, pos))
					pos = fs
			for fut in futs:
				fut.result()
				self.serving[on + "~buffer"] = True
			create_future_ex(self.rename_after, on, pn)

	def rename_after(self, on, pn):
		try:
			while True:
				try:
					os.rename(on, pn)
				except PermissionError:
					time.sleep(1)
		finally:
			self.serving.pop(on, None)
			self.serving.pop(on + "~buffer", None)

	def chunk_into(self, resp, on, pos):
		with open(on, "rb+") as f:
			f.seek(pos)
			f.write(resp.content)

	@cp.expose
	@hostmap
	def static(self, *filepath):
		if not filepath:
			if cp.request.remote_ip == "127.0.0.1":
				STATIC.clear()
				send("Webserver cache cleared.")
				return b"\xf0\x9f\x92\x9c"
			raise PermissionError
		filename = "/".join(filepath)
		try:
			data, mime = fetch_static("static/" + filename, ignore=True)
		except FileNotFoundError:
			data, mime = fetch_static(filename)
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@hostmap
	def apidoc(self):
		apidoc = getattr(self, "api", None)
		if not apidoc:
			apidoc = fetch_static(f"apidoc.html")
			data, mime = apidoc
			data2 = data.replace("⟨MIZA⟩".encode("utf-8"), cp.request.base.split("//", 1)[-1].encode("utf-8"))
			self.api = apidoc = (data2, mime)
		data, mime = apidoc
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@hostmap
	def models(self, *filepath):
		filename = "/".join(filepath)
		data, mime = fetch_static(f"waifu2x/models/{filename}")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@hostmap
	def w2wbinit(self):
		data, mime = fetch_static("waifu2x/w2wbinit.png")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@hostmap
	def waifu2x(self, *filepath, source=None):
		if filepath:
			filename = "/".join(filepath)
			if not source:
				source = "https://gitlab.com/20kdc/waifu2x-upconv7-webgl/-/raw/master/w2wbinit.png"
			if not is_url(source):
				raise FileNotFoundError
			if not regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").match(source) and not source.startswith("https://media.discordapp.net/"):
				if not source.startswith(cp.request.base):
					source = self.bot_exec(f"bot.data.exec.proxy({repr(source)})")
			data, mime = fetch_static("waifu2x/main.js")
			srcline = f'currentImage.src = "{source}";\n	currentImage.crossOrigin = "";'
			data = data.replace(b'currentImage.src = "w2wbinit.png";', srcline.encode("utf-8", "replace"))
			cp.response.headers.update(CHEADERS)
			cp.response.headers["Content-Type"] = mime
			cp.response.headers["Content-Length"] = len(data)
			cp.response.headers["ETag"] = create_etag(data)
			return data
		if source:
			data = f"""<!DOCTYPE html>
<html>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
	<meta property="og:image" content="{source}">""" + """
	<style>
		.center {
			margin: 0;
			position: absolute;
			top: 50%;
			left: 50%;
			-ms-transform: translate(-50%, -50%);
			transform: translate(-50%, -50%);
			color: white;
		}
	</style>""" + f"""
	<body style="background-color: black;">
		<div class="center">
			<input hidden type="file" id="imageIn" accept="image/png, image/jpeg"/>
			<select hidden id="modelName">
				<option value="models/upconv_7/art/scale2.0x_model">Waifu2x Upconv7 Art</option>
			</select>
			<button hidden id="runButton">Run</button>
			<button hidden id="cancelButton">Cancel</button>
			<p hidden id="statusDiv">JS not loaded yet...</p>
			<canvas id="canvas"></canvas>
			<script src="{cp.url()}/main.js?source={urllib.parse.quote(source)}"></script>
		</div>
	</body>
</html>"""
		else:
			data = """<!DOCTYPE html>
<html>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
	<style>
		.center {
			margin: 0;
			position: absolute;
			top: 50%;
			left: 50%;
			-ms-transform: translate(-50%, -50%);
			transform: translate(-50%, -50%);
			color: white;
		}
		.home {
			position: absolute;
			top: 10px;
			right: 10px;
			font-size: 20px;
		}
	</style>""" + f"""
	<body style="background-color: black;">
		<div class="center">
			<p>Open file</p>
			<input type="file" id="imageIn" accept="image/png, image/jpeg"/>
			<p>Select model</p>
			<select id="modelName">
				<option value="models/upconv_7/art/scale2.0x_model">Waifu2x Upconv7 Art</option>
				<option value="models/upconv_7/art/noise0_scale2.0x_model">Waifu2x Upconv7 Art (noise0)</option>
				<option value="models/upconv_7/art/noise1_scale2.0x_model">Waifu2x Upconv7 Art (noise1)</option>
				<option value="models/upconv_7/art/noise2_scale2.0x_model">Waifu2x Upconv7 Art (noise2)</option>
				<option value="models/upconv_7/art/noise3_scale2.0x_model">Waifu2x Upconv7 Art (noise3)</option>
				<option value="models/upconv_7/art/scale2.0x_model">Waifu2x Upconv7 Photo</option>
				<option value="models/upconv_7/art/noise0_scale2.0x_model">Waifu2x Upconv7 Photo (noise0)</option>
				<option value="models/upconv_7/art/noise1_scale2.0x_model">Waifu2x Upconv7 Photo (noise1)</option>
				<option value="models/upconv_7/art/noise2_scale2.0x_model">Waifu2x Upconv7 Photo (noise2)</option>
				<option value="models/upconv_7/art/noise3_scale2.0x_model">Waifu2x Upconv7 Photo (noise3)</option>
			</select>
			<p>Code from <a href="https://gitlab.com/20kdc/waifu2x-upconv7-webgl">20kdc waifu2x</a></p>
			<p>Model data from <a href="https://github.com/nagadomi/waifu2x/">nagadomi waifu2x</a></p>
			<button id="runButton">Run</button>
			<button id="cancelButton">Cancel</button>
			<p id="statusDiv">JS not loaded yet...</p>
			<p>experimental. exposure to high amounts of data may result in hazardous levels of memory usage, which may result in system OOM.</p>
			<p>View</p>
			<canvas id="canvas"></canvas>
			<script src="{cp.url()}/main.js"></script>
		</div>
		<a class="home" href="/miscellaneous">Back</a>
	</body>
</html>"""
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = "text/html"
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@hostmap
	def ytdl(self, **kwargs):
		d = kwargs.get("d") or kwargs.get("download")
		v = d or kwargs.get("v") or kwargs.get("view")
		q = d or v or kwargs.get("q") or kwargs.get("query") or kwargs.get("s") or kwargs.get("search")
		if not q:
			cp.response.status = 204
			return
		t = ts_us()
		while t in RESPONSES:
			t += 1
		if v:
			fmt = kwargs.get("fmt")
			if not fmt:
				fmt = "opus" if d else "mp3"
			if fmt not in ("mp3", "opus", "ogg", "wav"):
				raise TypeError
			fmt = "." + fmt
			self.bot_exec(f"bot.audio.returns[{t}]=VOICE.ytdl.search({repr(q)})[0]")
			self.bot_exec(f"VOICE.ytdl.get_stream(bot.audio.returns[{t}],force=True,download=False)")
			name, url = self.bot_exec(f"(bot.audio.returns[{t}].get('name'),bot.audio.returns[{t}].get('url'))")
			if not name or not url:
				raise FileNotFoundError
			h = shash(url)
			fn = "~" + h + fmt
			self.bot_exec(f"bot.audio.returns[{t}]=VOICE.ytdl.get_stream(bot.audio.returns[{t}],download={repr(fmt)},asap=True)")
			fni = "cache/" + fn

			def af():
				if not os.path.exists(fni):
					return
				if not os.path.getsize(fni):
					return
				try:
					res = self.bot_exec(f"bool(getattr(bot.audio.returns[{t}], 'loaded', None))")
				except:
					print_exc()
					return True
				return res is not False

			cp.response.headers.update(CHEADERS)
			if af():
				f = open(fni, "rb")
			else:
				f = DownloadingFile(fni, af=af)
				if d:
					cp.response.status = 202
				cp.response.headers["Content-Type"] = f"audio/{fmt[1:]}"
				cp.response.headers["Content-Disposition"] = "attachment; " * bool(d) + "filename=" + json.dumps(name + fmt)
				cp.response.headers.pop("Accept-Ranges", None)
				return cp.lib.file_generator(f, 262144)
			# cp.response.headers["Content-Type"] = f"audio/{fmt[1:]}"
			return cp.lib.static.serve_fileobj(f, content_type=f"audio/{fmt[1:]}", disposition="attachment" if d else "", name=name + fmt)
		else:
			count = 1 if is_url(q) else 10
			res = self.bot_exec(f"[VOICE.copy_entry(e) for e in VOICE.ytdl.search({repr(q)},count={count})]")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return orjson.dumps(res)
	ytdl._cp_config = {"response.stream": True}

	@cp.expose
	@hostmap
	def ytdlp(self, url, fmt="mp4", start="", end=""):
		if not url:
			cp.response.status = 204
			return
		t = ts_us()
		while t in RESPONSES:
			t += 1
		fmt = fmt.strip() or "mp4"
		if fmt not in ("mp3", "ogg", "opus", "m4a", "flac", "wav", "wma", "mp2", "weba", "vox", "adpcm", "pcm", "8bit", "mid", "midi", "webm", "mp4", "avi", "mov", "m4v", "mkv", "f4v", "flv", "wmv", "gif", "apng", "webp"):
			raise TypeError
		start = start.strip() or "-"
		end = end.strip() or "-"
		b = self.command(input=f"trim {url} {start} {end} as {fmt}")
		data = orjson.loads(b)
		url = data[0]["content"]
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	@hostmap
	def ytdlc(self, *urls, fmt="mp4", multi="true", **kwargs):
		m = "" if multi in ("false", "False") else "-m"
		kwurls = []
		for k, v in kwargs.items():
			if k.startswith("u") and k[1:].isnumeric():
				kwurls.append((int(k[1:]), v))
		urls = astype(urls, list)
		urls.extend(v for k, v in sorted(kwurls))
		urls = [url.strip() for url in urls]
		urls = [url for url in urls if url]
		if not any(urls):
			cp.response.status = 204
			return
		t = ts_us()
		while t in RESPONSES:
			t += 1
		fmt = fmt.strip() or "mp4"
		if fmt not in ("mp3", "ogg", "opus", "m4a", "flac", "wav", "wma", "mp2", "weba", "vox", "adpcm", "pcm", "8bit", "mid", "midi", "webm", "mp4", "avi", "mov", "m4v", "mkv", "f4v", "flv", "wmv", "gif", "apng", "webp"):
			raise TypeError
		url = " ".join(urls)
		b = self.command(input=f"concat {m} {url} as {fmt}")
		data = orjson.loads(b)
		url = data[0]["content"]
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	@hostmap
	def specexec(self, url, **kwargs):
		argv = " ".join(itertools.chain(*kwargs.items()))
		b = self.command(input=f"spectralpulse {url} {argv}")
		data = orjson.loads(b)
		url = data[0]["content"].replace("/d/", "/f/")
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "time", "mpinsights"))
	@hostmap
	def index(self, path=None, filename=None, *args, **kwargs):
		url = cp.url(qs=cp.request.query_string)
		if "/p/" in url:
			raise cp.HTTPRedirect(url.replace("/p/", "/file/"), status=307)
		if "/preview/" in url:
			raise cp.HTTPRedirect(url.replace("/preview/", "/file/"), status=307)
		if "/upload" in url:
			raise cp.HTTPRedirect(url.replace("/upload", "/files"), status=307)
		data, mime = fetch_static("index.html")
		meta = '<meta property="og:title" content="Miza"><meta property="og:description" content="A multipurpose Discord bot.">'
		if "/file" in url or "/files" in url:
			meta += '<meta property="og:image" content="/mizaleaf.png">'
		else:
			meta += '<meta property="og:image" content="/logo256.png">'
		meta += '<meta property="og:site_name" content="Miza">'
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
					if (len(c) - 1) & 3 == 0:
						c += b"="
					path = str(int.from_bytes(base64.urlsafe_b64decode(c), "big"))
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
				if (len(b) - 1) & 3 == 0:
					b += b"="
				path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
			else:
				path = path[1:]
			if not p:
				p = find_file(path, ind=ind)
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

	@cp.expose(("favicon", "favicon.ico"))
	@hostmap
	def favicon_ico(self, *args, **kwargs):
		data, mime = fetch_static("icon.ico")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	ip_sem = Semaphore(1, 0, rate_limit=150)

	def get_ip_ex(self):
		with suppress(SemaphoreOverflowError):
			with self.ip_sem:
				resp = Request("https://api.ipify.org", bypass=False, decode=True)
				if resp:
					self.ip = resp
				return resp

	@cp.expose
	@hostmap
	def get_ip(self, *args, **kwargs):
		data = orjson.dumps(dict(
			remote=cp.request.remote.ip,
			host=getattr(self, "ip", "127.0.0.1"),
		))
		cp.response.headers.update(SHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose(("upload_single",))
	@hostmap
	def upload_chunk(self, **kwargs):
		name = cp.request.headers.get("x-file-name", "untitled")
		s = cp.request.remote.ip + "%" + name
		h = hash(s) % 2 ** 48
		single = "/upload_single" in cp.url()
		xi = cp.request.headers.get("x-index", "0")
		fn = f"cache/{h}%" + xi
		with open(fn, "wb") as f:
			if single:
				pos = 0
				csize = 83886080
				g = cp.request.body.fp
				urls = []
				mids = []
				while True:
					b = g.read(csize)
					if not b:
						break
					f.write(b)
					if f.tell() > pos + csize:
						url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(fn)}, start={pos}, end={pos + csize})")
						urls.extend(url1)
						mids.extend(mid1)
						pos += csize
				if f.tell() > pos:
					url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(fn)}, start={pos})")
					urls.extend(url1)
					mids.extend(mid1)
				size = os.path.getsize(fn)
				mime = get_mime(fn)
				fn = f"cache/{h}%!"
				urls = [url.replace("https://cdn.discordapp.com/attachments/", "D$") for url in urls]
				print(urls)
				assert urls
				code = 307
				ftype = 3
				url = ""
				n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
				key = base64.urlsafe_b64encode(n.to_bytes(8, "little")).rstrip(b"=").decode("ascii")
				s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta/><!--["{name}","{size}","{mime}"]--><!--{json.dumps(urls)}--><!--KEY={key}--><!--MID={mids}--></html>'
				with open(fn, "w", encoding="utf-8") as f:
					f.write(s)
				return
			shutil.copyfileobj(cp.request.body.fp, f)
		if single:
			return self.merge(name=name, index=0)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def merge(self, **kwargs):
		key = kwargs.get("?key")
		ts = int(kwargs.get("?ts") or time.time_ns() // 1000)
		x_name = kwargs.get("x-file-name") or cp.request.headers.get("x-file-name", "untitled")
		name = kwargs.get("name") or x_name
		s = cp.request.remote.ip + "%" + x_name
		h = hash(s) % 2 ** 48
		n = f"cache/{h}%"
		fn = f"saves/filehost/{IND}{ts}~" + name
		r = n + "!"
		tn = fn.split("~", 1)[0] + "~.forward$"
		b = ts.bit_length() + 7 >> 3
		q = ""
		high = int(kwargs.get("index") or cp.request.headers.get("x-index", 0))
		if high == 0 and os.path.exists(r):
			with open(r, "r", encoding="utf-8") as f:
				with open(tn, "w", encoding="utf-8") as g:
					s = f.read()
					url = HOST + "/f/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
					s = s.replace('""', f'"{url}"', 1)
					g.write(s)
			key = key or s.split("<!--KEY=", 1)[-1].split("-->", 1)[0]
			q = f"?key={key}"
			if os.path.exists(n + "0"):
				os.rename(n + "0", fn.split("~", 1)[0] + "~.temp$@" + name)
		else:
			pos = 0
			csize = 83886080
			urls = []
			mids = []
			# os.rename(n + "0", fn)
			of = n + "0"
			if high > 1:
				with open(of, "ab") as f:
					for i in range(1, high):
						gn = n + str(i)
						with open(gn, "rb") as g:
							shutil.copyfileobj(g, f)
						while f.tell() > pos + csize:
							url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(of)}, start={pos}, end={pos + csize})")
							urls.extend(url1)
							mids.extend(mid1)
							pos += csize
						os.remove(gn)
			if os.path.getsize(of) > pos:
				url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(of)}, start={pos})")
				urls.extend(url1)
				mids.extend(mid1)
			size = os.path.getsize(of)
			mime = get_mime(of)
			urls = [url.replace("https://cdn.discordapp.com/attachments/", "D$") for url in urls]
			print(urls)
			assert urls
			try:
				ts = int(of.split("~", 1)[0].rsplit(IND, 1)[-1])
			except ValueError:
				ts = time.time_ns() // 1000
			fn = f"saves/filehost/{IND}{ts}~.forward$"
			code = 307
			ftype = 3
			b = ts.bit_length() + 7 >> 3
			url = HOST + "/f/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
			if not key:
				n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
				key = base64.urlsafe_b64encode(n.to_bytes(8, "little")).rstrip(b"=").decode("ascii")
			s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0; URL={url}"/><!--["{name}","{size}","{mime}"]--><!--{json.dumps(urls)}--><!--KEY={key}--><!--MID={json.dumps(mids)}--></html>'
			with open(fn, "w", encoding="utf-8") as f:
				f.write(s)
			q = f"?key={key}"
			os.rename(of, f"saves/filehost/{IND}{ts}~.temp$@" + name)
		return "/p/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=") + q

	@cp.expose
	@hostmap
	def upload_url(self, **kwargs):
		ts = time.time_ns() // 1000
		url = kwargs["url"]
		fn = f"../saves/filehost/{IND}{ts}~" + url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
		subprocess.run([sys.executable, "downloader.py", url, fn], cwd="misc")
		b = ts.bit_length() + 7 >> 3
		with tracebacksuppressor:
			self.replace_file(fn[3:])
		return HOST + "/p/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")

	@cp.expose
	@hostmap
	def replace_file(self, fn, key=None):
		print("Replace", fn)
		of = fn
		size = os.path.getsize(of)
		name = of.rsplit("/", 1)[-1].split("~", 1)[-1]
		if name.startswith(".temp$") or name.startswith(".forward$"):
			raise PermissionError
		mime = get_mime(of)
		urls, mids = self.bot_exec(f"bot.data.exec.stash({repr(of)})")
		urls = [url.replace("https://cdn.discordapp.com/attachments/", "D$") for url in urls]
		print(urls)
		assert urls
		try:
			ts = int(of.split("~", 1)[0].rsplit(IND, 1)[-1])
		except ValueError:
			ts = time.time_ns() // 1000
		fn = f"saves/filehost/{IND}{ts}~.forward$"
		code = 307
		ftype = 3
		b = ts.bit_length() + 7 >> 3
		url = HOST + "/f/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
		n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
		key = key or base64.urlsafe_b64encode(n.to_bytes(8, "little")).rstrip(b"=").decode("ascii")
		s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0; URL={url}"/><!--["{name}","{size}","{mime}"]--><!--{json.dumps(urls)}--><!--KEY={key}--><!--MID={json.dumps(mids)}--></html>'
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
		os.rename(of, f"saves/filehost/{IND}{ts}~.temp$@" + name)
		return url + f"?key={key}"

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def edit(self, path, key=None, **kwargs):
		if not key:
			raise PermissionError("Key not found.")
		ots = int.from_bytes(base64.urlsafe_b64decode(path.encode("ascii") + b"=="), "big")
		path = str(ots)
		p = find_file(path)
		if p.split("~", 1)[-1].startswith(".temp$@"):
			if p in self.serving:
				for f in self.serving.pop(p):
					f.close()
				time.sleep(0.2)
			with tracebacksuppressor:
				os.remove(p)
			p = find_file(path)
		if not p.split("~", 1)[-1].startswith(".forward$"):
			raise TypeError("File is not editable.")
		with open(p, "r", encoding="utf-8") as f:
			orig = f.read()
		if key != orig.split("<!--KEY=", 1)[-1].split("-->", 1)[0]:
			raise PermissionError("Incorrect key.")
		os.remove(p)
		mids = orjson.loads(orig.split("<!--MID=", 1)[-1].split("-->", 1)[0])
		self.bot_exec(f"bot.data.exec.delete({repr(mids)})")
		kwargs["?ts"] = ots
		kwargs["?key"] = key
		return self.merge(**kwargs)
		# ts = ots
		# x_name = kwargs.get("x-file-name") or cp.request.headers.get("x-file-name", "untitled")
		# name = kwargs.get("name") or x_name
		# s = cp.request.remote.ip + "%" + x_name
		# h = hash(s) % 2 ** 48
		# n = f"cache/{h}%"
		# fn = f"saves/filehost/{IND}{ts}~" + name
		# r = n + "!"
		# tn = fn.split("~", 1)[0] + "~.forward$"
		# b = ts.bit_length() + 7 >> 3
		# q = ""
		# if os.path.exists(r):
		# 	with open(r, "r", encoding="utf-8") as f:
		# 		with open(tn, "w", encoding="utf-8") as g:
		# 			s = f.read()
		# 			url = HOST + "/f/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
		# 			s = s.replace('""', f'"{url}"', 1)
		# 			t1, t2 = s.split("<!--KEY=", 1)
		# 			s = t1 + f"<!--KEY={key}-->" + t2.split("-->", 1)[-1]
		# 			g.write(s)
		# 	q = f"?key={key}"
		# 	if os.path.exists(n + "0"):
		# 		os.rename(n + "0", fn.split("~", 1)[0] + "~.temp$@" + name)
		# else:
		# 	high = int(kwargs.get("index") or cp.request.headers.get("x-index", "0"))
		# 	os.rename(n + "0", fn)
		# 	if high > 1:
		# 		with open(fn, "ab") as f:
		# 			for i in range(1, high):
		# 				gn = n + str(i)
		# 				with open(gn, "rb") as g:
		# 					shutil.copyfileobj(g, f)
		# 				os.remove(gn)
		# 	with tracebacksuppressor:
		# 		url = self.replace_file(fn, key=key)
		# 		return "/p/" + url.split("/f/", 1)[-1]
		# return "/p/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=") + q

	@cp.expose
	@hostmap
	def delete(self, path, key=None, **kwargs):
		if not key:
			raise PermissionError("Key not found.")
		path = str(int.from_bytes(base64.urlsafe_b64decode(path.encode("ascii") + b"=="), "big"))
		p = find_file(path)
		if p.split("~", 1)[-1].startswith(".temp$@"):
			if p in self.serving:
				for f in self.serving.pop(p):
					f.close()
				time.sleep(0.2)
			os.remove(p)
			p = find_file(path)
		if not p.split("~", 1)[-1].startswith(".forward$"):
			return os.remove(p)
		with open(p, "r", encoding="utf-8") as f:
			orig = f.read()
		if key != orig.split("<!--KEY=", 1)[-1].split("-->", 1)[0]:
			raise PermissionError("Incorrect key.")
		os.remove(p)
		mids = orjson.loads(orig.split("<!--MID=", 1)[-1].split("-->", 1)[0])
		self.bot_exec(f"bot.data.exec.delete({repr(mids)})")
		return """<!DOCTYPE html>
<html>
<meta http-equiv="refresh" content="0; URL=/">
<body onload="myFunction()" style="background-color:#000">
<script>
function myFunction() {
  alert("File successfully deleted. Returning to home.");
}
</script>
</body>
</html>
"""

	@cp.expose(("proxy",))
	@hostmap
	def redirect(self):
		data = """<!doctype HTML><html>
<link href="https://unpkg.com/boxicons@2.0.7/css/boxicons.min.css" rel="stylesheet">
<style>
body {
	font-family: 'Rockwell';
	color: #00ffff;
	background: black;
}
.center {
	margin: 0;
	position: absolute;
	top: 50%;
	left: 50%;
	-ms-transform: translate(-50%, -50%);
	transform: translate(-50%, -50%);
}
.tooltip {
  position: relative;
  display: inline-block;
  border-bottom: 1px dotted black;
}
.tooltip .tooltiptext {
  visibility: hidden;
  width: 120px;
  background-color: rgba(0, 0, 0, 0.5);
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 5px 0;
  position: absolute;
  z-index: 1;
}
.tooltip:hover .tooltiptext {
  visibility: visible;
}
</style>
<body>
	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000"
	 crossorigin="anonymous"></script>
	<video playsinline autoplay muted loop poster="https://cdn.discordapp.com/attachments/691915140198826005/846945647873490944/GpAy.webp" style="position:fixed;right:0;bottom:0;min-width:100%;min-height:100%;z-index:-1;">
		<source src="https://cdn.discordapp.com/attachments/691915140198826005/846587863797203004/GpAy.mp4" type="video/mp4">
	</video>
	<link href="/static/hamburger.css" rel="stylesheet">
	<div class="hamburger">
		<input
			type="checkbox"
			title="Toggle menu"
		/>
		<div class="items select">
			<a href="/" data-popup="Home">
				<video playsinline autoplay muted loop width="36" height="36" style="z-index:-1;">
					<source src="https://cdn.discordapp.com/attachments/691915140198826005/846592940075515904/miza_by_smudgedpasta_de1q8lu-pre.jpgtokeneyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOj.mp4" type="video/mp4">
				</video>
			</a>
			<a href="/mizatlas" data-popup="Command Atlas">
				<video playsinline autoplay muted loop width="36" height="36" style="z-index:-1;">
					<source src="https://cdn.discordapp.com/attachments/691915140198826005/846593904635281408/miza_has_a_leaf_blower_by_smudgedpasta_de6t2dl-pre.jpgtokeneyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJz.mp4" type="video/mp4">
				</video>
			</a>
			<a href="/upload" data-popup="File Host">
				<video playsinline autoplay muted loop width="36" height="36" style="z-index:-1;">
					<source src="https://cdn.discordapp.com/attachments/691915140198826005/846593561444745226/magical_babey_mode_by_smudgedpasta_de1q8ky-pre.jpgtokeneyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIi.mp4" type="video/mp4">
				</video>
			</a>
			<a href="/apidoc" data-popup="API Documentation">
				<video playsinline autoplay muted loop width="36" height="36" style="z-index:-1;">
					<source src="https://cdn.discordapp.com/attachments/691915140198826005/846590061901381632/deahc7l-a9773147-259d-4226-b0b6-195c6eb1f3c0.pngtokeneyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOi.mp4" type="video/mp4">
				</video>
			</a>
			<a 
				href="/time"
				data-popup="Clock"
				class='bx bx-time'></a>
		</div>
		<div class="hambg"></div>
	</div>
	<div class="center">
		<h1 style="color: white;">Create a redirect or proxy URL here!</h1>
		<form action="" method="get" class="form-example">
			<h2>URL</h2>
			<div class="tooltip">
				<input type=url name="url" style="width:240;"/>
				<span class="tooltiptext"> The URL to forward.</span>
			</div>
			<h2>Status Code</h2>
			<div class="tooltip">
				<input type=number name="code" value="307" min="100" max="599"/>
				<span class="tooltiptext"> The status code returned by the response. Should be one of {301, 302, 303, 307, 308} for a redirect.</span>
			</div>
			<h2>Type</h2>
			<div class="tooltip">
				<input type=radio value="1" name="ftype" checked=true/>Redirect
				<span class="tooltiptext"> A redirect page will simply forward users to the destination.</span>
			</div>
			<div class="tooltip">
				<input type=radio value="2" name="ftype"/>Proxy
				<span class="tooltiptext"> A proxy page will forward the data from the destination to the user.</span>
			</div>
			<br><br>
			<input type=submit value="Create" formaction="/forward" formenctype="application/x-www-form-urlencoded" formmethod="post"/>
		</form>
	</div>
</body>
</html>"""
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = "text/html"
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def forward(self, **kwargs):
		ts = time.time_ns() // 1000
		fn = f"saves/filehost/{IND}{ts}~.forward$"
		url = kwargs.get("url")
		if not url:
			raise FileNotFoundError
		code = int(kwargs.get("code", 307))
		ftype = int(kwargs.get("ftype", 1))
		s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0; URL={url}"/></html>'
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
		b = ts.bit_length() + 7 >> 3
		url = f"/p/" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
		raise cp.HTTPRedirect(url, status=307)

	@cp.expose
	@hostmap
	def backup(self, token="~"):
		at = AUTH.get("discord_token")
		if token != at:
			if cp.url(base="").strip("/") != at:
				raise InterruptedError
		backup = self.bot_exec(f"bot.backup()")
		cp.response.headers.update(CHEADERS)
		return cp.lib.static.serve_file(os.getcwd() + "/" + backup, content_type="application/octet-stream", disposition="attachment")
	backup._cp_config = {"response.stream": True}

	@cp.expose(("eval", "exec"))
	@hostmap
	def execute(self, token, *args, **kwargs):
		if token != AUTH.get("discord_token"):
			raise InterruptedError
		url = cp.url(base="", qs=cp.request.query_string)
		content = urllib.parse.unquote(url.lstrip("/").split("/", 2)[-1])
		res = self.bot_exec(content)
		return orjson.dumps(res)

	@cp.expose
	@hostmap
	def eval2(self, token, *args, **kwargs):
		if token != AUTH.get("discord_token"):
			raise InterruptedError
		url = cp.url(base="", qs=cp.request.query_string)
		content = urllib.parse.unquote(url.split("?", 1)[0].lstrip("/").split("/", 2)[-1])
		return str(eval(content, globals())).encode("utf-8")

	try:
		with open("saves/mpdata.json", "rb") as f:
			mpdata = json.load(f)
	except FileNotFoundError:
		mpdata = {}
	except:
		if os.path.exists("saves/mpdata\x7f\x7f.json"):
			with open("saves/mpdata\x7f\x7f.json", "rb") as f:
				mpdata = json.load(f)
		else:
			mpdata = {}
			print_exc()
	mpdata_updated = False

	try:
		with open("saves/mpact.json", "rb") as f:
			mpact = pickle.load(f)
	except FileNotFoundError:
		mpact = {}
	except:
		if os.path.exists("saves/mpact\x7f\x7f.json"):
			with open("saves/mpact\x7f\x7f.json", "rb") as f:
				mpact = pickle.load(f)
		else:
			mpact = {}
			print_exc()

	mpresponse = {}

	@cp.expose
	@hostmap
	def mphb(self, playing=None):
		mpdata = self.mpdata
		ip = cp.request.remote.ip
		t = utc()
		try:
			if playing is None or cp.request.method.casefold() != "patch" or cp.request.headers["User-Agent"] != "Miza Player":
				raise KeyError
		except KeyError:
			if ip in mpdata:
				d = t - mpdata[ip][1]
				if d < 60:
					mpdata[ip][0] += d
				mpdata[ip][1] = min(mpdata[ip][1], t - 60)
				d = t - mpdata[ip][3]
				if d < 60:
					mpdata[ip][2] += d
				mpdata[ip][3] = min(mpdata[ip][3], t - 60)
			cp.response.status = 450
			return ""
		if ip not in mpdata:
			mpdata[ip] = [0,] * 4
		d = t - mpdata[ip][1]
		if d < 60:
			mpdata[ip][0] += d
		mpdata[ip][1] = t
		if full_prune(playing) == "true":
			d = t - mpdata[ip][3]
			if d < 60:
				mpdata[ip][2] += d
			mpdata[ip][3] = t
		else:
			d = t - mpdata[ip][3]
			if d < 60:
				mpdata[ip][2] += d
			mpdata[ip][3] = min(mpdata[ip][3], t - 60)
		if not self.mpdata_updated:
			self.mpdata_updated = True
			create_future_ex(self.mpdata_update)
		try:
			resp = self.mpresponse.pop(ip)
		except KeyError:
			try:
				resp = self.mpresponse[None]
			except KeyError:
				return "💜"
		return "".join(chr(xrand(32, 127)) + chr(xrand(48, 96)) + c for c in resp.replace("$$$", "\n")[::-1])

	mpimg = {}

	@cp.expose
	@hostmap
	def api_mpinsights(self):
		values = self.mpget()
		for i in range(3):
			values[i] = int(values[i])
		self.ensure_mpins()
		histories = [None] * len(values)
		hours = histories.copy()
		for k in range(len(histories)):
			width = np.clip(len(self.ins_data[k]), 3, 96)
			histories[k] = list(supersample(self.ins_data[k], width))
			hours[k] = len(self.ins_data[k])
		return orjson.dumps(dict(
			current=dict(
				live_users=values[2],
				active_users=values[1],
				total_users=values[0],
				total_playtime=values[4],
				total_use_time=values[3],
				average_playtime=values[5],
			),
			historical=dict(
				live_users=[histories[2], hours[2]],
				active_users=[histories[1], hours[2]],
				total_users=[histories[0], hours[2]],
				total_playtime=[histories[4], hours[2]],
				total_use_time=[histories[3], hours[2]],
				average_playtime=[histories[5], hours[2]],
			),
		))

	@cp.expose
	@hostmap
	def api_status(self):
		status = self.bot_exec(f"bot.status()")
		return orjson.dumps(status)

	def ensure_mpins(self):
		try:
			ins_time = getattr(self, "ins_time", 0)
			t = utc()
			if t - ins_time >= 30:
				self.mpimg.clear()
				self.ins_wait = concurrent.futures.Future()
				k = self.mpact.keys()
				data = [deque() for i in range(len(next(reversed(self.mpact.values()))))]
				for i in range(min(k), max(k) + 1):
					values = self.mpact.get(i) or values
					for j, v in enumerate(values):
						data[j].append(v)
				self.ins_data = data
				self.ins_time = t
				self.ins_wait.set_result(None)
				self.ins_wait = None
		except:
			print_exc()

	# names = ("total_users", "active_users", "live_users", "active_seconds", "live_seconds", "seconds_per_user")
	def mpget(self):
		mpdata = self.mpdata
		values = [len(mpdata), 0, 0, 0, 0, 0]
		t = utc()
		for active, atime, listen, ltime in mpdata.values():
			values[1] += t - atime < 60
			values[2] += t - ltime < 60
			values[3] += active
			values[4] += listen
		values[5] = values[4] / values[0]
		return values

	def mpdata_update(self):
		try:
			time.sleep(20)
			self.mpdata_updated = False
			if not os.path.exists("saves/mpdata.json"):
				with open("saves/mpdata.json", "w") as f:
					json.dump(self.mpdata, f)
			else:
				with open("saves/mpdata\x7f.json", "w") as f:
					json.dump(self.mpdata, f)
				if os.path.exists("saves/mpdata\x7f\x7f.json"):
					os.remove("saves/mpdata\x7f\x7f.json")
				os.rename("saves/mpdata.json", "saves/mpdata\x7f\x7f.json")
				os.rename("saves/mpdata\x7f.json", "saves/mpdata.json")
		except:
			print_exc()

	def mpact_update(self):
		try:
			time.sleep(20)
			self.mpact_updated = False
			if not os.path.exists("saves/mpact.json"):
				with open("saves/mpact.json", "wb") as f:
					pickle.dump(self.mpact, f)
			else:
				with open("saves/mpact\x7f.json", "wb") as f:
					pickle.dump(self.mpact, f)
				if os.path.exists("saves/mpact\x7f\x7f.json"):
					os.remove("saves/mpact\x7f\x7f.json")
				os.rename("saves/mpact.json", "saves/mpact\x7f\x7f.json")
				os.rename("saves/mpact\x7f.json", "saves/mpact.json")
		except:
			print_exc()

	def mp_activity(self):
		try:
			self.act_time = os.path.getmtime("saves/mpact.json") // 3600
		except FileNotFoundError:
			self.act_time = 0
		while True:
			try:
				t = int(utc() // 3600)
				if t > self.act_time:
					self.act_time = t
					values = self.mpget()
					self.mpact[t] = values
					self.mpact_update()
					self.mpimg.clear()
			except:
				print_exc()
			time.sleep(60)

	@cp.expose
	@hostmap
	def donation(self, data=None):
		ip = cp.request.remote.ip
		data = data or cp.request.json
		if isinstance(data, str):
			data = orjson.loads(data)
		self.last_donation = data
		try:
			secret = data["verification_token"]
			if secret != KOFI_SECRET:
				raise KeyError
		except KeyError:
			raise PermissionError("Ko-fi Proxy Secret not detected.")
		send(data)
		if data["type"] != "Donation":
			return
		amount = round_min(float(data["amount"]))
		if amount <= 0:
			return
		msg = data["message"]
		name = data["from_name"]
		r = regexp("[^\\s@#:`][^@#:`]{0,30}[^\\s@#:`]#[0-9]{4}")
		found = r.findall(msg)
		if not found:
			r = regexp("<@[!&]?[0-9]+>")
			found = r.findall(msg)
			if not found:
				r = regexp("[0-9]{8,20}")
				found = r.findall(msg)
		if found:
			uid = found[0]
		else:
			uid = None
		res = self.bot_exec(f"bot.donate({repr(name)},{repr(uid)},{amount},{repr(msg)})")
		print(res)
		return as_str(res)

	def bot_exec(self, s):
		t = ts_us()
		while t in RESPONSES:
			t += 1
		RESPONSES[t] = fut = concurrent.futures.Future()
		send(f"!{t}\x7f{s}", escape=False)
		try:
			j, after = fut.result()
		finally:
			RESPONSES.pop(t, None)
		return j["result"]

	rapidapi = 0
	@cp.expose(("commands",))
	@hostmap
	def command(self, content="", input="", timeout=420, redirect=""):
		ip = cp.request.remote.ip
		send("/command", ip)
		if "\x7f" in content and ip in ("127.0.0.1", ADDRESS, getattr(self, "ip", None)):
			t, after = content.split("\x7f", 1)
			t = int(t)
			after = float(after)
			cl = int(cp.request.headers["Content-Length"])
			j = orjson.loads(cp.request.body.read(cl))
			if t in RESPONSES:
				RESPONSES[t].set_result((j, after))
				return b"\xf0\x9f\x92\x9c"
		content = input or urllib.parse.unquote(cp.url(base="", qs=cp.request.query_string).rstrip("?").split("/", 2)[-1])
		# if "DNT" in (k.upper() for k in cp.request.headers):
		#	 random.seed(ip)
		#	 ip = ".".join(str(xrand(1, 255)) for _ in loop(4))
		#	 random.seed(ts_us())
		#	 tz = "Anonymous (DNT enabled)"
		# else:
		try:
			secret = cp.request.headers["X-RapidAPI-Proxy-Secret"]
			if secret != RAPIDAPI_SECRET:
				raise KeyError
		except KeyError:
			data = get_geo(ip)
			tz = data["timezone"]
		else:
			ip = ".".join(str(xrand(1, 255)) for _ in loop(4))
			tz = "Anonymous (DNT enabled)"
			self.rapidapi += 1
		if " " not in content:
			content += " "
		t = ts_us()
		while t in RESPONSES:
			t += 1
		RESPONSES[t] = fut = concurrent.futures.Future()
		send(f"~{t}\x7f{ip}\x7f{tz}\x7f{content}", escape=False)
		j, after = fut.result(timeout=max(1, float(timeout)))
		RESPONSES.pop(t, None)
		a = after - utc()
		if a > 0:
			cp.response.headers["Retry-After"] = a
		cp.response.headers.update(HEADERS)
		return orjson.dumps(j)

	@cp.expose(("cat", "cats", "dog", "dogs", "neko", "nekos", "giphy"))
	@hostmap
	def imagepool(self, tag="", refresh=60):
		name = cp.url(base="").rsplit("/", 1)[-1]
		command = name.rstrip("s")
		argv = tag
		try:
			args = smart_split(argv)
		except ValueError:
			args = argv.split()
		url = self.bot_exec(f"bot.commands.{command}[0](bot=bot,channel=None,flags='v',args={repr(args)},argv={repr(argv)})")
		refresh = float(refresh or 60)
		if fcdict(cp.request.headers).get("Accept") == "application/json":
			return url
		cp.response.headers.update(HEADERS)
		return f"""<!DOCTYPE html>
<html>
<head>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script>
<meta property="og:image" content="{url}">
<meta http-equiv="refresh" content="{refresh}; URL={cp.url(qs=cp.request.query_string)}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>""" + """
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  margin-top: auto;
  margin-bottom: auto;
}
.center {
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
  max-width: 100%;
  max-height: 100%;		
}""" + f"""
</style>
</head>
<body style="background-color:black;">
<img src="{url}" class="center">
</body>
</html>"""

# @app.after_request
# def custom_header(response):
#	 response.headers.update(HEADERS)
#	 return response


def ensure_parent(proc, parent):
	while True:
		while len(RESPONSES) > 65536:
			try:
				RESPONSES.pop(next(iter(RESPONSES)))
			except:
				pass
		if not is_strict_running(parent):
			force_kill(psutil.Process())
			break
		time.sleep(6)

if __name__ == "__main__":
	pid = os.getpid()
	ppid = os.getppid()
	send(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	create_thread(ensure_parent, proc, parent)
	self = server = Server()
	create_thread(server.mp_activity)
	create_future_ex(server.get_ip_ex)
	cp.quickstart(server, "/", config)
