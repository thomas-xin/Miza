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


HOST = AUTH.get("webserver") or "https://mizabot.xyz"
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


import cherrypy, cheroot, logging, ssl, socket, waitress
from cherrypy._cpdispatch import Dispatcher
cp = cherrypy
httputil = cp.lib.httputil
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
		if "EOF occurred in violation of protocol" in str(ex):
			print(repr(ex))
		else:
			errnum = ex.args[0]
			# sadly SSL sockets return a different (longer) time out string
			timeout_errs = ('timed out', 'The read operation timed out')
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
cheroot.server.HTTPConnection.communicate = communicate

def process_headers(self):
	headers = self.headers
	for name, value in self.header_list:
		name = name.title()
		value = value.strip()
		headers[name] = httputil.decode_TEXT_maybe(value)
		if name == 'Cookie':
			with suppress():
				self.cookie.load(value)
				# raise cherrypy.HTTPError(400, str(exc))
	if not dict.__contains__(headers, 'Host'):
		if self.protocol >= (1, 1):
			msg = "HTTP/1.1 requires a 'Host' request header."
			raise cherrypy.HTTPError(400, msg)
	else:
		headers['Host'] = httputil.SanitizedHost(dict.get(headers, 'Host'))
	host = dict.get(headers, 'Host')
	if not host:
		host = self.local.name or self.local.ip
	self.base = '%s://%s' % (self.scheme, host)
cp._cprequest.Request.process_headers = process_headers

actually_static = set(os.listdir("misc/static"))
mapped_static = {k[:-5]: k for k in actually_static if k.endswith(".html")}

def map_url(url):
	if not isinstance(url, str):
		return url
	return url.removeprefix(
		"https://"
	).replace(
		"cdn.discordapp.com/attachments/", "D$"
	).replace(
		".amazonaws.com/www.guilded.gg/ContentMediaGenericFiles/", "G$"
	)
def remap_url(url):
	if not isinstance(url, str) or url.startswith("https://"):
		return url
	return "https://" + url.replace(
		"D$", "cdn.discordapp.com/attachments/"
	).replace(
		"G$", ".amazonaws.com/www.guilded.gg/ContentMediaGenericFiles/"
	)

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		send("@@@", escape=False)
		p = path.lstrip("/")
		while p:
			if p == "ip":
				p = "get_ip"
			elif p[:2] == "f/":
				p = "raw/" + p[2:]
			elif p == "upload":
				p = "files"
			elif p == "api/mpinsights":
				p = "api_mpinsights"
			elif p.startswith("api/"):
				p = p.removeprefix("api/")
			elif p in actually_static:
				p = "static/" + p
			elif p in mapped_static:
				p = "static/" + mapped_static[p]
			else:
				break
		p = "/" + p
		# if p != path:
		# 	print("REDIR:", path, p)
		return super().__call__(p)

error_map = {
	SyntaxError: 400,
	PermissionError: 401,
	InterruptedError: 403,
	FileNotFoundError: 404,
	FileExistsError: 409,
	AssertionError: 412,
	OverflowError: 413,
	TypeError: 415,
	ValueError: 417,
	IsADirectoryError: 418,
	IOError: 422,
	RuntimeError: 500,
	ConnectionError: 502,
	SystemError: 503,
	TimeoutError: 504,
	OSError: 507,
	RecursionError: 508,
	GeneratorExit: 510,
	ReferenceError: 523,
}
errdata = {}
def error_handler(exc=None):
	if not exc:
		exc = sys.exc_info()[1]
		if not exc:
			exc = RuntimeError("An unknown error occured.")
	if (dt := datetime.datetime.utcnow()) and (dt.month, dt.day) in ((3, 31), (4, 1), (4, 2)):
		status = 418
	else:
		status = error_map.get(exc) or error_map.get(exc.__class__) or 500
	if status == 418:
		head = {}
		vid = "dQw4w9WgXcQ"
		url = f"https://http.cat/{status}"
		mime = "text/html"
		embed = f"https://www.youtube.com/embed/{vid}"
		video = f"https://www.youtube.com/watch?v={vid}"
		w = 1280
		h = 720
		body = f"""<!DOCTYPE html>
<html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta property="og:type" content="video.other">
<meta property="twitter:player" content="{embed}">
<meta property="og:video:type" content="{mime}">
<meta property="og:video:width" content="{w}">
<meta property="og:video:height" content="{h}">
<meta name="twitter:image" content="{url}">
<meta http-equiv="refresh" content=0;url={video}">
</head><body></body></html>""".encode("utf-8")
	else:
		resp = errdata.get(status) or errdata.setdefault(status, reqs.next().get(f"https://http.cat/{status}"))
		head = resp.headers
		body = resp.content
	head.update(HEADERS)
	head["Content-Length"] = len(body)
	cp.response.status = status
	cp.response.headers.update(head)
	cp.response.body = body

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
		"request.error_response": error_handler,
	}
}
if os.path.exists("domain.cert.pem") and os.path.exists("private.key.pem"):
	config["global"]["server.ssl_certificate"] = "domain.cert.pem"
	config["global"]["server.ssl_private_key"] = "private.key.pem"
	def hostmap(func):
		def decorator(*args, **kwargs):
			url = cp.url(qs=cp.request.query_string)
			if not url.startswith("https://mizabot.") and not url.startswith("https://api.mizabot."):
				if url.startswith("https://csnftcg.mizabot."):
					raise cp.HTTPRedirect(f"https://csnftcg.mizabot.xyz:9604/{url.rsplit('/', 1)[-1]}", 307)
				# time.sleep(10)
				# raise cp.HTTPRedirect(f"https://mizabot.xyz/{url.rsplit('/', 1)[-1]}", 307)
			return func(*args, **kwargs)
		return decorator
	def e404(status, message, traceback, version):
		url = cp.url(qs=cp.request.query_string)
		if not url.startswith("https://mizabot.") and not url.startswith("https://api.mizabot."):
			print("Not Found:", url)
			time.sleep(3600)
		return message
	cp.config["error_page.404"] = e404
else:
	def hostmap(func):
		return func

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
	def _fileinfo(self, path, **void):
		orig_path = path
		ind = IND
		if path.startswith("!"):
			ind = "!"
			path = path[1:]
		elif not path.startswith("@"):
			b = path.lstrip("~").split(".", 1)[0]
			path = str(p2n(b))
		else:
			path = path[1:]
		p = find_file(path, cwd=("saves/filehost", "cache"), ind=ind)
		mime = get_mime(p)
		f_url = cp.url(qs=cp.request.query_string).replace("/fileinfo/", "/f/")
		st = os.stat(p)
		fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
		if fn.startswith(".temp$@"):
			fn = fn[7:]
			a3 = True
		else:
			try:
				p2 = find_file(path, cwd="cache", ind=ind)
			except FileNotFoundError:
				a3 = False
			else:
				a3 = True
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
		# print(p)
		if "$" in p and p.split("$", 1)[0].endswith("~.forward") and mime == "text/html":
			with open(p, "r", encoding="utf-8") as f:
				resp = f.read(1048576)
			s = resp
			# print(s)
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
						urls = orjson.loads(urld.removeprefix("<!--").removeprefix("URL="))
						d["filename"] = info[0]
						d["size"] = info[1]
						d["mimetype"] = info[2]
						urls = [remap_url(url) for url in urls]
						# print(urls)
						if len(urls) == 1 and not is_url(urls[0]):
							urls = self._fileinfo(f"@{urls[0]}").get("chunks", ())
						d["chunks"] = urls
					else:
						d["original_url"] = url
		return d

	@cp.expose
	@hostmap
	def fileinfo(self, path, **void):
		cp.response.headers.update(SHEADERS)
		d = self._fileinfo(path)
		cp.response.headers["Content-Type"] = "application/json"
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
		cp.response.headers.update(SHEADERS)
		if path.startswith("!"):
			ind = "!"
			path = path[1:]
		elif not path.startswith("@"):
			b = path.lstrip("~").split(".", 1)[0].encode("ascii") + b"=="
			if b.startswith(b"dQ"):
				c = b[2:]
				path = str(p2n(c))
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
				raise FileNotFoundError(404, p)
			elif "$" in p and p.split("$", 1)[0].endswith("~.forward") and mime == "text/html" and os.path.getsize(p) < 1048576:
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
							urls = orjson.loads(urld.removeprefix("<!--URL="))
							disp = "filename=" + info[0]
							cp.response.headers["Content-Disposition"] = disp
							# cp.response.headers["Content-Length"] = info[1]
							cp.response.headers["Content-Type"] = info[2]
							referrer = cp.request.headers.get("Referer")
							# print(p, len(urls), referrer)
							cp.response.headers["Attachment-Filename"] = info[0]
							if cp.request.method == "HEAD":
								cp.response.headers["Content-Length"] = info[1]
								return
							urls = [remap_url(url) for url in urls]
							# print(urls)
							if len(urls) == 1:
								if not is_url(urls[0]):
									p = find_file(urls[0], cwd=("cache", "saves/filehost"))
									urls = self._fileinfo(f"@{urls[0]}").get("chunks", ())
							if download and len(urls) == 1 and not referrer:
								raise cp.HTTPRedirect(url, status="307")
							cp.response.headers.pop("Accept-Ranges", None)
							stn = p.rsplit("~.forward$", 1)[0].replace("saves/filehost/", "cache/")
							pn = stn + "~.temp$@" + info[0]
							if os.path.exists(pn):
								f = open(pn, "rb")
								resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=info[0])
								if a3:
									self.serving.setdefault(p, weakref.WeakSet()).add(f)
								return resp
							if info[1] > 64 * 1048576:
								return self.dyn_serve(urls, size=info[1])
							return self.concat(p, urls, name=info[0], mime=info[2], stn=stn)
			f = open(p, "rb")
			resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=a2)
			if a3:
				self.serving.setdefault(p, weakref.WeakSet()).add(f)
			return resp
	files._cp_config = {"response.stream": True}

	def concat(self, fn, urls, name="", download=False, mime=None, stn="", waiter=False):
		on = stn + "!.temp$@" + name
		pn = stn + "~.temp$@" + name
		if waiter and os.path.exists(pn) and os.path.getsize(pn):
			return newfut
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
			if waiter:
				return create_future_ex(self._concat, urls, on, pn)
			return self._peek(urls, on, pn, name, download, mime)
		if waiter:
			return fut
		return self.wconcat(on, pn, name, download, mime, fut)

	def wconcat(self, on, pn, name, download, mime, fut, start=0):
		for i in range(120):
			if os.path.exists(pn):
				break
			if os.path.exists(on) and self.serving.get(on + "~buffer", 0) >= start or fut.done():
				break
			time.sleep(0.5)
		if os.path.exists(pn):
			with open(pn, "rb") as f:
				if start:
					f.seek(start)
				resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=name)
				self.serving.setdefault(pn, weakref.WeakSet()).add(f)
				yield from resp
			self.serving.setdefault(pn, weakref.WeakSet()).discard(f)
			return
		with open(on, "rb") as f:
			if start:
				f.seek(start)
			while not fut.done() and on in self.serving:
				if f.tell() + 262144 >= self.serving.get(on + "~buffer", 0):
					time.sleep(2)
					continue
				b = f.read(262144)
				if not b:
					continue
				yield b
			while True:
				b = f.read(262144)
				if not b:
					return
				yield b

	def dyn_serve(self, urls, size=0):
		# print("Serve", urls, size)
		with tracebacksuppressor:
			headers = fcdict(cp.request.headers)
			headers.pop("Remote-Addr", None)
			headers.pop("Host", None)
			headers.pop("Range", None)
			headers.update(Request.header())
			brange = cp.request.headers.get("Range", "")
			brange = brange.removeprefix("bytes=")
			ranges = []
			length = 0
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
				print_exc()
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
			cp.response.headers["Accept-Range"] = "bytes"
			return self._dyn_serve(urls, ranges, headers)

	def _dyn_serve(self, urls, ranges, headers):
		# print("Dynamic Serve:", urls, ranges, headers)
		with tracebacksuppressor(GeneratorExit):
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
						resp = reqs.next().head(u, headers=headers)
						ns = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
					# print(len(rems), pos, ns, start, end)
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
						# print("Range:", h2["range"])
						ex2 = None
						for i in range(3):
							resp = reqs.next().get(u, headers=h2, stream=True)
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

					if len(futs) > 2:
						yield from futs.pop(0).result()
					fut = create_future_ex(get_chunk, u, headers, start, end, pos, ns, big)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns
					big = True
				for fut in futs:
					yield from fut.result()

	def _peek(self, urls, on, pn, name, download, mime):
		headers = fcdict(cp.request.headers)
		headers.pop("Remote-Addr", None)
		headers.pop("Host", None)
		headers.pop("Range", None)
		headers.update(Request.header())
		with reqs.next().get(urls[0], headers=headers, stream=True) as resp:
			resp.raise_for_status()
			b = []
			it = resp.iter_content(65536)
			with suppress(StopIteration):
				while sum(map(len, b)) < 8388608:
					temp = next(it)
					b.append(temp)
					yield temp
			b = b"".join(b)
			print("PreCat", urls[0], resp, len(b))
		fut = create_future_ex(self._concat, urls, on, pn)
		self.serving[on] = fut
		yield from self.wconcat(on, pn, name, download, mime, fut, start=len(b))

	def _concat(self, urls, on, pn):
		print("Cat", urls)
		with tracebacksuppressor:
			headers = fcdict(cp.request.headers)
			headers.pop("Remote-Addr", None)
			headers.pop("Host", None)
			headers.pop("Range", None)
			headers.update(Request.header())
			buf = 0
			pos = 0
			futs = []
			with open(on, "wb") as f:
				for url in urls:
					if len(futs) >= 16 or futs and futs[0].done():
						fut = futs.pop(0)
						fut.result()
						buf += fut.buf
						self.serving[on + "~buffer"] = buf
					for i in range(16):
						try:
							resp = reqs.next().get(url, headers=headers, stream=True)
							if resp.status_code in (403, 404):
								raise FileNotFoundError
							resp.raise_for_status()
							break
						except FileNotFoundError:
							break
						except:
							print_exc()
						time.sleep(i ** 2 + 1)
					bsize = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
					fs = pos + bsize
					f.truncate(fs)
					fut = create_future_ex(self.chunk_into, resp, on, pos)
					fut.buf = bsize
					futs.append(fut)
					pos = fs
			for fut in futs:
				fut.result()
				buf += fut.buf
				self.serving[on + "~buffer"] = buf
			try:
				os.rename(on, pn)
			except PermissionError:
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
			if fmt not in ("mp3", "opus", "webm", "weba", "ogg", "wav"):
				raise TypeError(fmt)
			fmt = "." + fmt
			self.bot_exec(f"bot.audio.returns[{t}]=VOICE.ytdl.search({repr(q)})[0]")
			self.bot_exec(f"VOICE.ytdl.get_stream(bot.audio.returns[{t}],force=True,download=False)")
			name, url = self.bot_exec(f"(bot.audio.returns[{t}].get('name'),bot.audio.returns[{t}].get('url'))")
			if not name or not url:
				raise FileNotFoundError(500, v)
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

	@cp.expose
	@hostmap
	def filelist(self, path=None):
		cp.response.headers["Content-Type"] = "application/json"
		try:
			sessid = int(cp.request.cookie["sessid"].value)
		except (KeyError, ValueError):
			return "[]"
		else:
			adata = cdict(self.bot_exec(f"bot.data.sessions.get({repr(sessid)})"))
		if not adata:
			cp.response.cookie["sessid"] = ""
			return "[]"
		if "email" not in adata:
			if "id" not in adata:
				return "[]"
			fdata = self.bot_exec(f"bot.data.drives.get({adata.id},[])")
		else:
			if "id" in adata:
				fdata = self.bot_exec(
					f"bot.data.drives.setdefault({repr(adata.email)},set()).update(bot.data.drives.pop({adata.id},[]))\n"
					+ f"return bot.data.drives.get({repr(adata.email)},[])"
				)
			else:
				fdata = self.bot_exec(f"bot.data.drives.get({repr(adata.email)},[])")
		if not path:
			return orjson.dumps(fdata)
		cpath = path.split("/")
		while cpath:
			fold = cpath.pop(0)
			for e in fdata:
				if isinstance(e, dict) and e.get("i") == fold:
					fdata = e.get("f")
					break
			else:
				raise FileNotFoundError(404, path, fold)
		return orjson.dumps(fdata)

	@cp.expose
	@hostmap
	def teapot(self, *args, **kwargs):
		raise IsADirectoryError("I'm a teapot.")

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	@hostmap
	def index(self, path=None, filename=None, *args, code=None, **kwargs):
		url = cp.url(qs=cp.request.query_string).rstrip("/")
		if "/user" in url:
			try:
				sessid = int(cp.request.cookie["sessid"].value)
			except (KeyError, ValueError):
				adata = None
			else:
				adata = cdict(self.bot_exec(f"bot.data.sessions.get({repr(sessid)})"))
			t = utc()
			if not adata:
				if not code:
					cp.response.cookie["sessid"] = ""
					raise cp.HTTPRedirect(url.replace("/user", "/login"))
				resp = reqs.next().post(
					f"https://discord.com/api/oauth2/token",
					data=dict(
						client_id=AUTH.get("discord_id") or self.bot_exec("bot.id"),
						client_secret=AUTH["discord_secret"],
						grant_type="authorization_code",
						code=code,
						redirect_uri=f"{HOST}/user",
					),
				)
				resp.raise_for_status()
				adata = cdict(resp.json())
				adata.ts = round_random(t * 1e6)
				adata.expiry = t + adata.expires_in
				adata.refreshed = 0
			t = utc()
			if t > adata.expiry:
				resp = reqs.next().post(
					f"https://discord.com/api/oauth2/token",
					data=dict(
						client_id=AUTH.get("discord_id") or self.bot_exec("bot.id"),
						client_secret=AUTH["discord_secret"],
						grant_type="refresh_token",
						refresh_token=adata.refresh_token,
					),
				)
				resp.raise_for_status()
				adata.update(resp.json())
				adata.expiry = t + adata.expires_in
			t = utc()
			if t > adata.refreshed + 30:
				resp = reqs.next().get(
					"https://discord.com/api/users/@me",
					headers={"Authorization": f"{adata.token_type} {adata.access_token}"},
				)
				resp.raise_for_status()
				adata.update(resp.json())
				adata.refreshed = t
			sessid = adata.id
			if "email" in adata and "id" in adata:
				self.bot_exec(
					f"bot.data.accounts.setdefault({repr(adata.email)},{{}})['uid']={adata.id};"
					+ f"bot.data.users.setdefault({adata.id},{{}})['email']={repr(adata.email)}"
				)
			cp.response.cookie["sessid"] = sessid
			cp.response.cookie["email"] = adata.get("email") or ""
			cp.response.cookie["name"] = adata.get("username") or adata.email.split("@", 1)[0]
			cp.response.cookie["uid"] = int(adata.get("id", 0)) or ""
			if "id" in adata:
				adata["icon"] = self.bot_exec(
					f"bot.cache.users[{adata.id}]=await bot.fetch_user({adata.id});"
					+ f"return best_url(bot.cache.users[{adata.id}])"
				)
			else:
				adata.pop("icon")
			self.bot_exec(f"bot.data.sessions[{repr(sessid)}]={repr(adata)}")
			cp.response.cookie["icon"] = adata.get("icon")
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

	chunking = {}
	@cp.expose(("upload_single",))
	@hostmap
	def upload_chunk(self, **kwargs):
		name = cp.request.headers.get("x-file-name", "untitled")
		s = cp.request.remote.ip + "%" + name
		h = ihash(s) % 2 ** 48
		single = "/upload_single" in cp.url()
		xi = int(cp.request.headers.get("x-index", 0))
		mfs = int(cp.request.headers.get("x-total", 0))
		if mfs > 1125899906842624:
			raise OverflowError
		if not xi:
			print(s)
		n = f"cache/{h}%"
		fn = n + str(xi)
		csize = 83886080
		with open(fn, "wb") as f:
			if single:
				pos = 0
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
				urls = [map_url(url) for url in urls]
				print(urls)
				assert urls
				code = 307
				ftype = 3
				url = ""
				n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
				key = n2p(n)
				s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta/><!--["{name}","{size}","{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--><!--KEY={key}--><!--MID={mids}--></html>'
				with open(fn, "w", encoding="utf-8") as f:
					f.write(s)
				return self.merge(name=name, index=1)
			if mfs > 512 * 1048576:
				fut = create_future_ex(shutil.copyfileobj, cp.request.body.fp, f)
				try:
					info = cdict(self.chunking[n])
				except KeyError:
					info = self.chunking[n] = cdict(
						mime="application/octet-stream",
					)
				if xi % 8 < 2:
					fut.result()
					f.close()
					if xi == 0:
						info.mime = get_mime(fn)
					url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(fn)})")
					self.chunking[fn] = (url1, mid1)
					with suppress(PermissionError, FileNotFoundError):
						os.remove(fn)
				elif xi % 8 == 7:
					fut.result()
					f.close()
					fns = []
					for i in range(5):
						ft = n + str(xi - 5 + i)
						if ft not in self.chunking:
							self.chunking[ft] = concurrent.futures.Future()
						while ft in self.chunking:
							try:
								self.chunking[ft].result(timeout=720)
							except:
								time.sleep(8)
							else:
								break
						assert os.path.exists(ft)
						fns.append(ft)
					fns.append(fn)
					url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(fns)})")
					self.chunking[fn] = (url1, mid1)
					for ft in fns:
						with suppress(PermissionError, FileNotFoundError):
							os.remove(ft)
				else:
					if fn in self.chunking:
						if self.chunking[fn].done():
							try:
								self.chunking[fn].result()
							except:
								self.chunking[fn] = fut
							else:
								fut.result()
								return
						fut.add_done_callback(self.chunking[fn].set_result)
					self.chunking[fn] = fut
					fut.result()
				self.update_merge()
				return
			shutil.copyfileobj(cp.request.body.fp, f)

	chunk_file = "cache/chunking.json"
	try:
		if not os.path.exists(chunk_file):
			chunking = {}
		else:
			with open(chunk_file, "rb") as f:
				b = f.read()
			chunking = orjson.loads(b)
	except:
		print_exc()
		chunking = {}
	merge_sem = Semaphore(1, inf)
	def update_merge(self):
		with tracebacksuppressor:
			d = self.chunking.copy()
			for k, v in tuple(d.items()):
				if isinstance(v, concurrent.futures.Future):
					d.pop(k)
			with self.merge_sem:
				b = orjson.dumps(d)
				if not os.path.exists(self.chunk_file):
					with open(self.chunk_file, "wb") as f:
						f.write(b)
				else:
					with open(self.chunk_file, "rb+") as f:
						f.truncate(len(b))
						f.seek(0)
						f.write(b)

	merged = {}
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def merge(self, **kwargs):
		kwargs = fcdict(kwargs)
		key = kwargs.get("?key")
		ts = int(kwargs.get("?ts") or time.time_ns() // 1000)
		x_name = kwargs.get("x-file-name") or cp.request.headers.get("x-file-name", "untitled")
		name = kwargs.get("name") or x_name
		s = cp.request.remote.ip + "%" + x_name
		print(s)
		mfs = int(kwargs.get("x-total") or cp.request.headers.get("x-total", 0))
		h = ihash(s) % 2 ** 48
		nh = n = f"cache/{h}%"
		if self.merged.get(nh):
			return
		self.merged[nh] = True
		try:
			fn = f"saves/filehost/{IND}{ts}~" + name
			r = n + "!"
			q = ""
			print("Merge", fn)
			high = int(kwargs.get("index") or cp.request.headers.get("x-index", 0))
			if high == 0 and os.path.exists(r):
				tn = fn.split("~", 1)[0] + "~.forward$" + str(os.path.getsize(r))
				with open(r, "r", encoding="utf-8") as f:
					with open(tn, "w", encoding="utf-8") as g:
						s = f.read()
						url = HOST + "/f/" + n2p(ts)
						s = s.replace('""', f'"{url}"', 1)
						g.write(s)
				key = key or s.split("<!--KEY=", 1)[-1].split("-->", 1)[0]
				q = f"?key={key}"
				if os.path.exists(n + "0"):
					os.rename(n + "0", f"cache/{IND}{ts}" + "~.temp$@" + name)
			elif nh in self.chunking:
				info = cdict(self.chunking.pop(n))
				if not key:
					n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
					key = n2p(n)
				q = f"?key={key}"
				url = HOST + "/f/" + n2p(ts)
				na2 = lim_str(name, 96).replace("$", "-")
				size = mfs
				tn = f"saves/filehost/{IND}{ts}~.forward${size}$ ${na2}.$"
				urls = []
				mids = []
				mime = info.mime
				for i in range(high):
					gn = nh + str(i)
					fut = self.chunking.pop(gn, None)
					if not fut:
						if not os.path.exists(gn):
							raise FileNotFoundError(gn)
						url1, mid1 = self.bot_exec(f"bot.data.exec.stash({repr(gn)})")
						urls.extend(url1)
						mids.extend(mid1)
					elif isinstance(fut, (tuple, list)):
						urls.extend(fut[0])
						mids.extend(fut[1])
				urls = [map_url(url) for url in urls]
				code = 307
				ftype = 3
				jdn = json.dumps(name).replace("<", '"\u003c"').replace(">", '"\u003e"')
				s = (
					f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={url}"/>'
					+ f'<!--[{jdn},{size},"{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--><!--KEY={key}--><!--MID={json.dumps(mids)}-->'
					+ f'<!--SHA=A-->'
					+ '</html>'
				)
				with open(tn, "w", encoding="utf-8") as f:
					f.write(s)
			else:
				of = nh + "0"
				fn = f"cache/{IND}{ts}~.temp$@{name}"
				os.rename(of, fn)
				if high > 1:
					with open(fn, "ab") as f:
						for i in range(1, high):
							gn = nh + str(i)
							with open(gn, "rb") as g:
								shutil.copyfileobj(g, f)
				if not key:
					n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
					key = n2p(n)
				q = f"?key={key}"
				self.register_replacer(ts, key)
		finally:
			self.merged.pop(nh, None)
		self.update_merge()
		return "/p/" + n2p(ts)

	@cp.expose
	@hostmap
	def upload_url(self, **kwargs):
		ts = time.time_ns() // 1000
		url = kwargs["url"]
		fn = f"../cache/{IND}{ts}~.temp$@" + url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
		subprocess.run([sys.executable, "downloader.py", url, fn], cwd="misc")
		b = ts.bit_length() + 7 >> 3
		ts, key = self.register_replacer(ts)
		return HOST + "/p/" + n2p(ts)

	def optimise_video(self, of, size, mime):
		print("Convert", of, mime, size)
		dur = get_duration(of)
		if mime.split("/", 1)[-1] in ("mp4", "webm"):
			if dur > 3600 and size > 524288000:
				raise StopIteration
			if size / dur <= 1048576:
				args = [
					"./ffprobe",
					"-v",
					"error",
					"-show_entries",
					"stream=pix_fmt",
					"-of",
					"default=noprint_wrappers=1:nokey=1",
					of,
				]
				resp = None
				try:
					proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
					fut = create_future_ex(proc.wait, timeout=8)
					res = fut.result(timeout=8)
					resp = proc.stdout.read()
				except:
					with suppress():
						force_kill(proc)
					with suppress():
						resp = proc.stdout.read()
					print_exc()
				if resp and resp.strip() == "yuv420p":
					raise StopIteration
		done = False
		if dur > 60 and size <= 524288000 and mime.split("/", 1)[0] == "video":
			with tracebacksuppressor:
				header = Request.header()
				header["origin"] = header["referer"] = "https://www.mp4compress.com/"
				with open(of, "rb") as f:
					resp = reqs.next().post(
						"https://www.mp4compress.com/",
						files=dict(upfile=(f"record{ts_us()}.mp4", f, "video/mp4"), submitfile=(None, "")),
						headers=header,
					)
				resp.raise_for_status()
				if 'Completed: <a href="' not in resp.text:
					s = resp.text.split('<form class="form" role="form" action="https://www.mp4compress.com/" method="post" enctype="multipart/form-data">', 1)[0]
					s = s.rsplit("</p>", 1)[-1].strip()
					if s:
						print(s)
					raise FileNotFoundError('Completed: <a href="')
				url = resp.text.split('Completed: <a href="', 1)[-1].split('"', 1)[0]
				print(url)
				with reqs.next().get(
					url,
					headers=header,
					stream=True,
				) as resp:
					resp.raise_for_status()
					it = resp.iter_content(65536)
					fmt = "mp4"
					fo = f"{of}.{fmt}"
					with open(fo, "wb") as f:
						with suppress(StopIteration):
							while True:
								b = next(it)
								if not b:
									break
								f.write(b)
				done = True
		if not done:
			args = [
				"./ffmpeg",
				"-hide_banner",
				"-v",
				"error",
				"-nostdin",
				"-y",
				"-hwaccel",
				hwaccel,
				"-i",
				of,
				"-pix_fmt",
				"yuv420p",
			]
			if dur <= 60:
				fmt = "webm"
				fo = f"{of}.{fmt}"
				args.extend((
					"-c:v",
					"libsvtav1",
					"-crf",
					"42",
					fo,
				))
			else:
				fmt = "mp4"
				fo = f"{of}.{fmt}"
				args.extend((
					"-c:v",
					"h264",
					"-crf",
					"30",
					fo,
				))
			print(args)
			proc = psutil.Popen(args, stdin=subprocess.DEVNULL)
			fut = create_future_ex(proc.wait, timeout=120)
			try:
				fut.result(timeout=120)
			except concurrent.futures.TimeoutError:
				if proc.is_running() and os.path.exists(fo) and os.path.getsize(fo):
					fut = None
			if not fut:
				fut = create_future_ex(proc.wait, timeout=3600)
				fut.result(timeout=3600)
		assert os.path.exists(fo) and os.path.getsize(fo) and os.path.getsize(fo) < size
		name = of.rsplit("/", 1)[-1].split("~", 1)[-1]
		if name.startswith(".temp$@"):
			name = name[7:]
		try:
			ts = int(of.split("~", 1)[0].rsplit(IND, 1)[-1])
		except ValueError:
			ts = time.time_ns() // 1000
		name = name.rsplit(".", 1)[0]
		os.remove(of)
		of = f"cache/{IND}{ts}~.temp$@{name}.{fmt}"
		os.rename(fo, of)
		size = os.path.getsize(of)
		mime = get_mime(of)
		return of

	def optimise_image(self, of, size, mime):
		fmt = "webp"
		fo = self.bot_exec(f"process_image({repr(of)},resize_mult,[1,1,'auto','-f',{repr(fmt)}],timeout=120)")
		assert os.path.exists(fo) and os.path.getsize(fo) and os.path.getsize(fo) < size
		name = of.rsplit("/", 1)[-1].split("~", 1)[-1]
		if name.startswith(".temp$@"):
			name = name[7:]
		try:
			ts = int(of.split("~", 1)[0].rsplit(IND, 1)[-1])
		except ValueError:
			ts = time.time_ns() // 1000
		name = name.rsplit(".", 1)[0]
		os.remove(of)
		of = f"cache/{IND}{ts}~.temp$@{name}.{fmt}"
		os.rename(fo, of)
		size = os.path.getsize(of)
		mime = get_mime(of)
		return of

	def hash_file(self, fn):
		if os.name != "nt":
			return ""
		args = ["certutil", "-hashfile", fn, "sha384"]
		print(args)
		s = subprocess.check_output(args)
		try:
			s = s.splitlines()[1].decode("ascii")
			print(s)
			return s
		except:
			print(s)
			print_exc()
		return ""

	def replace_file(self, fn, key=None, delay=0, name=None, urls=None, mids=None):
		if not os.path.exists(fn):
			return
		print("Replace", fn)
		of = fn
		try:
			ts = int(of.split("~", 1)[0].rsplit(IND, 1)[-1])
		except ValueError:
			ts = time.time_ns() // 1000
		try:
			assert os.path.exists(find_file(ts, cwd="saves/filehost"))
		except:
			pass
		else:
			return
		# if os.path.exists(fn) and not urls:
		# 	return url + f"?key={key}"
		ha1 = self.hash_file(of)
		if ha1:
			ha2 = ha1[:4] + ha1[-4:]
			for fp in os.listdir("saves/filehost"):
				fl = fp.split("~", 1)[-1]
				if not fl.startswith(".forward$"):
					continue
				fl = fl.split("$", 1)[-1]
				if "$" not in fl:
					continue
				fh2 = fl.split("$", 2)[1].strip()
				if fh2 != ha2:
					continue
				with open("saves/filehost/" + fp, "r+", encoding="utf-8") as f:
					sn = f.read()
					if "<!--SHA=" not in sn:
						continue
					s = sn.split("<!--SHA=", 1)[-1]
					fh1, s = s.split("-->", 1)
					if fh1 != ha1:
						continue
					if s.startswith("<!--REF="):
						s = s.removeprefix("<!--REF=")
						refs = set(orjson.loads(s.split("-->", 1)[0]))
						refs.add(ts)
					else:
						refs = [ts]
					i = sn.index("<!--SHA=") + len("<!--SHA=") + len(fh1) + len("-->")
					f.seek(i)
					f.write(f'<!--REF=[{",".join(map(str, refs))}]--></html>')
					t2 = int(fp.split("~", 1)[0].rsplit(IND, 1)[-1])
					urls = [t2]
					mids = []
					ha1 = ha2 = ""
					break
		size = os.path.getsize(of)
		mime = get_mime(of)
		if not urls:
			if size > 8388608:
				with tracebacksuppressor(StopIteration):
					if mime.split("/", 1)[0] == "video" or mime in ("image/gif", "image/apng"):
						of = self.optimise_video(of, size, mime)
					elif mime in ("image/bmp", "image/tiff", "image/xbm", "image/heic", "image/heif"):
						of = self.optimise_image(of, size, mime)
					else:
						raise StopIteration
					size = os.path.getsize(of)
					mime = get_mime(of)
		if not name:
			name = of.rsplit("/", 1)[-1].split("~", 1)[-1]
			if name.startswith(".forward$"):
				raise PermissionError
			if name.startswith(".temp$@"):
				name = name[7:]
		url = HOST + "/f/" + n2p(ts)
		n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
		key = key or n2p(n)
		na2 = lim_str(name, 96).replace("$", "-")
		fn = f"saves/filehost/{IND}{ts}~.forward${size}${ha2 or ' '}${na2}.$"
		if urls:
			pass
		elif size > 0:
			self.register_replacer(ts, key)
			urls, mids = self.bot_exec(f"bot.data.exec.stash({repr(of)})")
			if not self.in_replacer(ts, key):
				self.bot_exec(f"bot.data.exec.delete({repr(mids)})")
				return
			print(urls)
			assert urls
		else:
			urls = mids = []
		urls = [map_url(url) for url in urls]
		code = 307
		ftype = 3
		jdn = json.dumps(name).replace("<", '"\u003c"').replace(">", '"\u003e"')
		s = (
			f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={url}"/>'
			+ f'<!--[{jdn},{size},"{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--><!--KEY={key}--><!--MID={json.dumps(mids)}-->'
			+ (f'<!--SHA={ha1}-->' if ha1 else "")
			+ '</html>'
		)
		with suppress(FileNotFoundError):
			while True:
				pn = find_file(ts, cwd="saves/filehost")
				os.remove(pn)
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
		on = f"cache/{IND}{ts}~.temp$@" + name
		if of != on:
			os.rename(of, on)
		self.remove_replacer(ts, key)
		return url + f"?key={key}"

	def bump(self):
		for fp in os.listdir("saves/filehost"):
			p = "saves/filehost/" + fp
			with open(p, "r", encoding="utf-8") as f:
				sn = f.read()
			s = sn.split("/>", 1)[-1]
			spl = s.split("-->", 4)
			if len(spl) >= 5:
				infd, urld, key, midd, _ = spl[:5]
			elif len(spl) >= 3:
				infd, urld, _ = spl[:3]
				key = ""
				midd = "[]"
			else:
				continue
			info = orjson.loads(infd.removeprefix("<!--"))
			urls = orjson.loads(urld.removeprefix("<!--").removeprefix("URL="))
			mids = orjson.loads(midd.removeprefix("<!--").removeprefix("MID="))
			while key.startswith("<!--KEY="):
				key = key[8:]
			urls = [remap_url(url) for url in urls]
			if not is_url(urls[0]):
				continue
			stn = p.rsplit("~.forward$", 1)[0].replace("saves/filehost/", "cache/")
			pn = stn + "~.temp$@" + info[0]
			self.concat(pn, urls, name=info[0], mime=info[2], stn=stn, waiter=True).result()
			if fp.count("$") >= 4:
				continue
			self.replace_file(pn, name=info[0], key=key, urls=urls, mids=mids)

	@cp.expose
	@hostmap
	def api_register_replacer(self, ts, key):
		if cp.request.remote.ip != "127.0.0.1":
			raise PermissionError
		return self.register_replacer(ts, key)

	replacer_sem = Semaphore(1, inf, rate_limit=0.0625)
	def register_replacer(self, ts, key=None):
		if not key:
			n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
			n2p(n)
		with self.replacer_sem:
			with open("saves/filehost/-1.txt", "a", encoding="ascii") as f:
				f.write(f"{ts}:{key}\n")
		return (ts, key)

	def in_replacer(self, ts, key):
		if not os.path.exists("saves/filehost/-1.txt") or not os.path.getsize("saves/filehost/-1.txt"):
			return
		with self.replacer_sem:
			with open("saves/filehost/-1.txt", "r", encoding="ascii") as f:
				lines = f.readlines()
		line = f"{ts}:{key}\n"
		return line in lines

	def remove_replacer(self, ts, key):
		if not os.path.exists("saves/filehost/-1.txt") or not os.path.getsize("saves/filehost/-1.txt"):
			return
		with self.replacer_sem:
			with open("saves/filehost/-1.txt", "r+", encoding="ascii") as f:
				lines = set(f.readlines())
				line = f"{ts}:{key}\n"
				if line not in lines:
					return False
				lines.discard(line)
				s = "".join(lines)
				f.truncate(len(s))
				f.seek(0)
				f.write(s)
		return True

	replace_fut = None
	@cp.expose
	@hostmap
	def api_update_replacers(self):
		if cp.request.remote.ip != "127.0.0.1":
			raise PermissionError
		if self.replace_fut and not self.replace_fut.done():
			return
		self.replace_fut = create_future_ex(self.update_replacers)

	def update_replacers(self):
		if not os.path.exists("saves/filehost/-1.txt") or not os.path.getsize("saves/filehost/-1.txt"):
			return
		with self.replacer_sem:
			with open("saves/filehost/-1.txt", "r", encoding="ascii") as f:
				lines = f.readlines()
		for line in lines:
			ts, key = line.rstrip().split(":", 1)
			try:
				fn = find_file(ts, cwd="cache", ind=IND)
				self.replace_file(fn, key)
			except:
				print_exc()
			finally:
				self.remove_replacer(ts, key)

	edited = {}
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def edit(self, path, key=None, **kwargs):
		if not key:
			raise PermissionError("Key not found.")
		ots = p2n(path)
		path = str(ots)
		p = find_file(path, cwd=("cache", "saves/filehost"))
		replaceable = self.remove_replacer(ots, key)
		if not replaceable:
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
			self.delete_link(p, orig)
		else:
			os.remove(p)
		if self.edited.get(ots):
			return
		self.edited[ots] = True
		try:
			if not replaceable:
				mids = orjson.loads(orig.split("<!--MID=", 1)[-1].split("-->", 1)[0])
				self.bot_exec(f"bot.data.exec.delete({repr(mids)})")
			kwargs["?ts"] = ots
			kwargs["?key"] = key
			url = self.merge(**kwargs)
		finally:
			self.edited.pop(ots, None)
		print("Edited", url)
		return url

	def delete_link(self, p, text=None):
		if not text:
			with open(p, "r", encoding="utf-8") as f:
				text = f.read()
		os.remove(p)
		s = text.split("/>", 1)[-1]
		infd, urld, _k, midd, ext  = s.split("-->", 4)
		# info = orjson.loads(infd.removeprefix("<!--"))
		urls = orjson.loads(urld.removeprefix("<!--URL="))
		mids = orjson.loads(midd.removeprefix("<!--MID="))
		spl = ext.split("-->")
		if len(spl) > 1:
			ha1 = spl[0].removeprefix("<!--SHA=")
			if len(spl) > 2:
				fids = orjson.loads(spl[1].removeprefix("<!--REF="))
				newref = None
				for fid in fids:
					with tracebacksuppressor(FileNotFoundError):
						p2 = find_file(fid, "saves/filehost")
						with open(p2, "r+", encoding="utf-8") as f:
							sn = f.read()
							if not newref:
								i = sn.index("<!--URL=") + len("<!--URL=")
								u, s = sn[i:].split("-->", 1)
								url1 = orjson.loads(u)
								if is_url(url1[0]):
									continue
								f.seek(i)
								f.write(json.dumps(urls, separators=(",", ":")))
								f.write(s)
								f.seek(0)
								sn = f.read()
								i = sn.index("<!--MID=") + len("<!--MID=")
								s = sn[i:].split("-->", 1)[-1]
								f.seek(i)
								f.write(json.dumps(mids, separators=(",", ":")))
								f.write(f"<!--SHA={ha1}-->")
								f.write(s)
							else:
								i = sn.index("<!--URL=") + len("<!--URL=")
								s = sn[i:].split("-->", 1)[-1]
								f.seek(i)
								f.write(json.dumps([newref], separators=(",", ":")))
								f.write(s)
						if not newref:
							newref = fid
							os.rename(p2, p2.replace("$ $", "$" + ha1[:4] + ha1[-4:] + "$", 1))

	@cp.expose
	@hostmap
	def delete(self, path, key=None, **kwargs):
		if not key:
			raise PermissionError("Key not found.")
		ots = p2n(path)
		path = str(ots)
		p = find_file(path, cwd=("cache", "saves/filehost"))
		replaceable = self.remove_replacer(ots, key)
		if not replaceable:
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
			self.delete_link(p, text=orig)
		else:
			os.remove(p)
		if not replaceable:
			mids = orjson.loads(orig.split("<!--MID=", 1)[-1].split("-->", 1)[0])
			self.bot_exec(f"bot.data.exec.delete({repr(mids)})")
		return """<!DOCTYPE html><html>
<meta http-equiv="refresh" content="0;URL=/">
<body onload="myFunction()" style="background-color:#000">
<script>
function myFunction() {
alert("File successfully deleted. Returning to home.");
}
</script>
</body>
</html>"""

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
	def forward(self, **kwargs):
		ts = time.time_ns() // 1000
		fn = f"saves/filehost/{IND}{ts}~.forward$"
		urls = kwargs.get("urls")
		if not urls:
			url = kwargs.get("url")
			if not url:
				raise IOError("Missing urls field")
			urls = [url]
		elif isinstance(urls, str):
			urls = orjson.loads(urls)
		code = int(kwargs.get("code", 307))
		ftype = int(kwargs.get("ftype", 1))
		url = f"/p/" + n2p(ts)
		if len(urls) <= 1:
			s = f'<!DOCTYPE HTML><!--["{urls[0]}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={urls[0]}"/></html>'
		else:
			with reqs.next().head(urls[0], headers=Request.header(), stream=True) as resp:
				mime = resp.headers.get("Content-Type") or "text/html"
			ftype = 3
			s =  (
				f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={url}"/>'
				+ f'<!--["Multi-redirect",0,"{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--></html>'
			)
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
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
		return cp.lib.static.serve_file(backup, content_type="application/octet-stream", disposition="attachment")
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
	def status(self):
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
		if not values[0]:
			values[5] = 0
		else:
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
		if "\x7f" in content and ip in ("127.0.0.1", ADDRESS, getattr(self, "ip", None)):
			t, after = content.split("\x7f", 1)
			t = int(t)
			after = float(after)
			cl = int(cp.request.headers["Content-Length"])
			j = orjson.loads(cp.request.body.read(cl))
			if t in RESPONSES:
				RESPONSES[t].set_result((j, after))
				return b"\xf0\x9f\x92\x9c"
		content = input or urllib.parse.unquote(cp.url(base="", qs=cp.request.query_string).rstrip("?").split("/", 1)[-1].removeprefix("api/").split("/", 1)[-1])
		send("/command", ip, content)
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
	logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
	pid = os.getpid()
	ppid = os.getppid()
	send(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	create_thread(ensure_parent, proc, parent)
	app = Server()
	self = server = cherrypy.Application(app, "/", config)
	create_thread(app.mp_activity)
	create_future_ex(app.get_ip_ex)
	waitress.serve(server, host=ADDRESS, port=PORT, url_scheme="https")
