import datetime
import functools
import io
import itertools
import json
import logging
import os
import random
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib
import zipfile
import cheroot
import cherrypy
import niquests
import orjson
import psutil
import requests
from concurrent.futures import Future
from traceback import print_exc
from cheroot import errors
from cherrypy._cpdispatch import Dispatcher
from .asyncs import Semaphore, SemaphoreOverflowError, eloop, esubmit, tsubmit, csubmit, await_fut
from .types import byte_like, as_str, cdict, suppress, round_min, regexp, json_dumps, resume, MemoryBytes
from .util import fcdict, nhash, shash, uhash, EvalPipe, AUTH, TEMP_PATH, MIMES, tracebacksuppressor, utc, is_url, p2n, ecdc_dir, quote_plus, rename, url_unparse, url2fn, is_youtube_url, seq, Request, magic, is_discord_attachment, is_miza_attachment, unyt, ecdc_exists, CACHE_PATH, T, byte_scale, decode_attachment, update_headers, temporary_file, CODEC_FFMPEG
from .caches import attachment_cache, colour_cache
from .audio_downloader import AudioDownloader, get_best_icon

try:
	RAPIDAPI_SECRET = AUTH["rapidapi_secret"]
except KeyError:
	RAPIDAPI_SECRET = None
try:
	KOFI_SECRET = AUTH["kofi_secret"]
except KeyError:
	KOFI_SECRET = None

HOST = AUTH.get("webserver") or "https://mizabot.xyz"
API = AUTH.get("api") or "https://api.mizabot.xyz"
ADDRESS = AUTH.get("webserver_address") or "0.0.0.0"
PORT = AUTH.get("webserver_port") or 80
IND = "\x7f"


csubmit(Request._init_())
tsubmit(eloop.run_forever)
if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	print = interface.print


def create_etag(data):
	n = len(data)
	s = str(nhash(data[:128] + data[n // 2 - 128:(n + 1) // 2 + 128] + data[-128:]) + n & 4294967295)
	return '"' + "0" * (10 - len(s)) + s + '"'

rickroll = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

SEMAPHORES = {}
STATIC = {}
TZCACHE = {}

PREVIEW = {}
def utc_dt():
	return datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
prev_date = utc_dt().date()
zfailed = set()


cp = cherrypy
httputil = cp.lib.httputil
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

def access(self):
	request = cp.serving.request
	if T(request).get("no_log"):
		return
	url = cp.url(qs=request.query_string)
	print(true_ip(), request.method, url, request.headers)
	return
cp._cplogging.LogManager.access = access

def process_headers(self):
	headers = self.headers
	for name, value in self.header_list:
		name = name.title()
		value = value.strip()
		headers[name] = httputil.decode_TEXT_maybe(value)
		if name == 'Cookie':
			with suppress():
				self.cookie.load(value)
				# raise cp.HTTPError(400, str(exc))
	if not dict.__contains__(headers, 'Host'):
		if self.protocol >= (1, 1):
			msg = "HTTP/1.1 requires a 'Host' request header."
			raise cp.HTTPError(400, msg)
	else:
		headers['Host'] = httputil.SanitizedHost(dict.get(headers, 'Host'))
	host = dict.get(headers, 'Host')
	if not host:
		host = self.local.name or self.local.ip
	self.base = '%s://%s' % (self.scheme, host)
cp._cprequest.Request.process_headers = process_headers

actually_static = set(os.listdir("misc/web"))
mapped_static = {k[:-5]: k for k in actually_static if k.endswith(".html")}

@functools.lru_cache(maxsize=256)
def get_size_mime(head, tail, count, chunksize):
	fut = esubmit(requests.head, head)
	resp = requests.head(tail)
	lastsize = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
	size = chunksize * (count - 1) + lastsize
	resp = fut.result()
	mimetype = resp.headers.get("Content-Type", "application/octet-stream")
	return mimetype, size, lastsize

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		p = path.lstrip("/")
		while p:
			if p == "ip":
				p = "get_ip"
			elif p.split("/", 1)[0] in ("f", "d"):
				p = "download/" + p.split("/", 1)[-1]
			elif p in actually_static:
				p = "static/" + p
			elif p in mapped_static:
				p = "static/" + mapped_static[p]
			else:
				break
		p = "/" + p
		return super().__call__(p)

error_map = {
	SyntaxError: 400,
	PermissionError: 401,
	InterruptedError: 403,
	FileNotFoundError: 404,
	UnicodeError: 406,
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
	head = {}
	update_headers(head, **HEADERS)
	if not exc:
		exc = sys.exc_info()[1]
		if not exc:
			exc = RuntimeError("An unknown error occured.")
	if (dt := utc_dt()) and (dt.month, dt.day) in ((3, 31), (4, 1), (4, 2)):
		status = 418
	elif isinstance(exc, ConnectionError) or isinstance(exc, type) and issubclass(exc, ConnectionError) and exc.args and isinstance(exc.args[0], int):
		status = exc.args[0]
	elif isinstance(exc, niquests.exceptions.HTTPError):
		status = exc.response.status_code
	else:
		status = error_map.get(exc) or error_map.get(exc.__class__) or 500
	cp.response.status = status
	print_exc()
	if "application/json" in cp.request.headers.get("Accept", ""):
		head["Content-Type"] = "application/json"
		body = orjson.dumps(dict(
			exception=str(exc.__class__),
			message=str(exc),
		))
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
		resp = errdata.get(status) or errdata.setdefault(status, niquests.get(f"https://http.cat/{status}", timeout=5))
		head = resp.headers.copy()
		body = resp.content
	head["Content-Length"] = len(body)
	update_headers(cp.response.headers, **head)
	cp.response.headers.pop("Connection", None)
	print(cp.response.headers)
	cp.response.body = body

config = {
	"global": {
		"server.socket_host": ADDRESS,
		"server.socket_port": PORT,
		"server.thread_pool": 256,
		"server.max_request_body_size": 0,
		"server.socket_timeout": 65,
		"server.ssl_module": "builtin",
		"engine.autoreload_on": False,
		"environment": "production",
		"tools.gzip.on": True,
		"tools.gzip.mime_types": ["text/plain", "text/html", "text/css", "text/csv", "text/xml", "text/md", "text/markdown", "text/javascript", "application/json", "application/javascript"],
	},
	"/": {
		"request.dispatch": EndpointRedirects(),
		"request.error_response": error_handler,
	},
}
DOMAIN_CERT = "domain.cert.pem"
PRIVATE_KEY = "private.key.pem"

HEADERS = {
	"X-Content-Type-Options": "nosniff",
	"Server": "Miza",
	"Vary": "Accept-Encoding",
	"Access-Control-Allow-Headers": "*",
	"Access-Control-Allow-Methods": "*",
	"Access-Control-Allow-Origin": "*",
}

CHEADERS = {
	"Cache-Control": "public,max-age=21600, stale-while-revalidate=1073741824, stale-if-error=1073741824",
	"Accept-Ranges": "bytes",
}
SHEADERS = {
	"Cache-Control": "public, max-age=60, stale-while-revalidate=1073741824, stale-if-error=1073741824",
	"Accept-Ranges": "bytes",
}
CHEADERS.update(HEADERS)
SHEADERS.update(HEADERS)


def fetch_static(path, ignore=False):
	while path.startswith("../"):
		path = path[3:]
	try:
		fn = f"misc/web/{path}"
		fn2 = fn + ".zip"
		if os.path.exists(fn2):
			ts = os.path.getmtime(fn2)
		else:
			ts = os.path.getmtime(fn)
		try:
			data, ts2 = STATIC[path]
			if ts2 < ts:
				raise KeyError
		except KeyError:
			if os.path.exists(fn2) and zipfile.is_zipfile(fn2):
				with zipfile.ZipFile(fn2, allowZip64=True, strict_timestamps=False) as z:
					data = z.read(path.rsplit("/", 1)[-1])
			else:
				with open(fn, "rb") as f:
					data = f.read()
			STATIC[path] = (data, ts)
		fmt = path.rsplit(".", 1)[-1].casefold()
		try:
			mime = MIMES[fmt]
		except KeyError:
			mime = "text/html"
		return data, mime
	except:
		if not ignore:
			print(path)
			print_exc()
		raise

ytdl = None

geo_sem = Semaphore(90, 256, rate_limit=60)
geo_count = 0
IP = None

def get_geo(ip):
	global geo_count, IP
	if ip.startswith("192.168."):
		ip = IP
		if not ip:
			ip = IP = niquests.get("https://api.ipify.org", verify=False, timeout=5).text
	try:
		resp = TZCACHE[ip]
	except KeyError:
		if geo_count & 1:
			url = f"http://ip-api.com/json/{ip}?fields=256"
		else:
			url = f"https://demo.ip-api.com/json/{ip}?fields=256&key=test-demo-pro"
		geo_count += 1
		with geo_sem:
			resp = niquests.get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}", "Origin": "https://members.ip-api.com"}, timeout=5)
		resp.raise_for_status()
		resp = cdict(resp.json())
		if not resp.get("timezone"):
			resp.timezone = "N/A"
		TZCACHE[ip] = resp
		print(ip + "\t" + "\t".join(resp.values()))
	return resp

def true_ip(request=None):
	request = request or cp.request
	ip = request.headers["Remote-Addr"]
	if ip == "127.0.0.1":
		ip = request.headers.get("X-Real-Ip") or ip
	if ip == "127.0.0.1":
		ip = request.remote.ip
	cp.serving.request.remote.ip = request.remote.ip = ip
	return ip


class Server:

	session = niquests.Session()

	@cp.expose(("0",))
	def rickroll(self, *void1, **void2):
		raise cp.HTTPRedirect(rickroll, status=301)

	opipe = None
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def detect(self, image=None):
		if image is None:
			from PIL import Image
			image = Image.open(cp.request.body.fp)
			image.load()
		elif isinstance(image, (bytes, memoryview)):
			from PIL import Image
			image = Image.open(io.BytesIO(image))
		elif isinstance(image, cherrypy._cpreqbody.Part):
			from PIL import Image
			image = Image.open(image.file)
		if not isinstance(self.opipe, Future):
			self.opipe = Future()
			from transformers import pipeline
			pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device=0)
			self.opipe.set_result(pipe)
		image = image.rotate(-90)
		image.save("test.png")
		pipe = self.opipe.result()
		data = pipe(image, threshold=1 / 3)
		cp.response.headers["Content-Type"] = "application/json"
		out = json_dumps(data)
		return out

	def get_with_retries(self, url, headers={}, data=None, timeout=3, retries=5):
		for i in range(retries):
			try:
				session = self.session if url.startswith("https://") and not is_discord_attachment(url) and i == 0 else requests
				resp = session.get(url, headers=headers, data=data, verify=i <= 1, timeout=timeout + i ** 2)
				resp.raise_for_status()
			except Exception:
				if i < retries - 1:
					continue
				raise
			else:
				return resp
		return resp

	def dyn_serve(self, urls, size=0, head=None, callback=None):
		with tracebacksuppressor:
			brange = cp.request.headers.get("Range", "").removeprefix("bytes=")
			headers = fcdict(cp.request.headers)
			headers.pop("Content-Length", None)
			headers.pop("Content-Type", None)
			headers.pop("Remote-Addr", None)
			headers.pop("Host", None)
			headers.pop("Range", None)
			update_headers(headers, **Request.header())
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
				except Exception:
					pass
			if ranges:
				cp.response.status = 206
			else:
				cp.response.status = 200
				ranges.append((0, size))
				length = size
			if not size:
				size = "*"
			if ranges == [(0, size)]:
				cp.response.headers["Content-Length"] = str(length)
			if brange:
				cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size}" for start, end in ranges)
				cp.response.headers["Content-Range"] = cr
			cp.response.headers["Accept-Ranges"] = "bytes"
			print(brange, ranges)
			return self._dyn_serve(urls, ranges, headers, head=head, callback=callback)

	def _dyn_serve(self, urls, ranges, headers, head=None, callback=None):
		with tracebacksuppressor(GeneratorExit):
			if head:
				data = head.data[head.index:]
				urls.insert(0, bytes(data))
				if callback:
					callback(head.data)
			for i, (start, end) in enumerate(ranges):
				pos = 0
				rems = urls.copy()
				futs = []
				big = False
				while rems:
					u = rems.pop(0)
					if isinstance(u, byte_like):
						ns = len(u)
					elif "?size=" in u or "&size=" in u:
						u, ns = u.replace("?size=", "&size=").split("&size=", 1)
						ns = int(ns)
					elif "?S=" in u or "&S=" in u:
						u, ns = u.replace("?S=", "&S=").split("&S=", 1)
						ns = int(ns)
					elif u.startswith("https://s3-us-west-2"):
						ns = 503316480
					elif u.startswith("https://cdn.discord"):
						ns = 8388608
					else:
						resp = requests.head(u, timeout=3)
						ns = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
					if pos + ns <= start:
						pos += ns
						continue
					if pos >= end:
						break

					def get_chunk(u, h, start, end, pos, ns, big):
						s = start - pos
						e = end - pos
						if isinstance(u, byte_like):
							yield u[s:e]
							return
						if is_miza_attachment(u) and (path := u.split("?", 1)[0].split("/u/", 1)[-1]) and len(path.split("/")) == 2 and path.count("~") == 0:
							c_id, m_id, a_id, fn = decode_attachment(path)
							u = await_fut(attachment_cache.obtain(c_id, m_id, a_id, fn))
						print(u)
						if e >= ns:
							e = ""
						else:
							e -= 1
						h2 = dict(h.items())
						h2["range"] = f"bytes={s}-{e}"
						resp = self.get_with_retries(u, headers=h2, timeout=3)
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
						yield from resp.iter_content(49152)

					if len(futs) > i + 1:
						yield from futs.pop(0).result()
					fut = esubmit(get_chunk, u, headers, start, end, pos, ns, big)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns
					big = True
				for fut in futs:
					yield from fut.result()

	@cp.expose(("c",))
	def chunked_proxy(self, path, *void):
		with tracebacksuppressor:
			fut = csubmit(attachment_cache.obtains(path))
			urls, chunksize = await_fut(fut)
			mimetype, size, lastsize = get_size_mime(urls[0], urls[-1], len(urls), chunksize)
			update_headers(cp.response.headers, **CHEADERS)
			cp.response.headers["Content-Type"] = mimetype
			new_urls = [f"{url}&S={lastsize if i >= len(urls) - 1 else chunksize}" for i, url in enumerate(urls)]
			return self.dyn_serve(new_urls, size)
	chunked_proxy._cp_config = {"response.stream": True}

	@cp.expose(("u",))
	def unproxy(self, *path, url=None, **query):
		if url:
			return self.proxy_if(url)
		rpath = "/".join(path)
		if rpath:
			rpath = "/" + rpath
		rquery = cp.request.query_string and "?" + cp.request.query_string
		if len(path) == 1 and path[0].count("~") == 2:
			fut = csubmit(attachment_cache.obtain(*path[0].split(".", 1)[0].split("~", 2)))
			return self.proxy_if(await_fut(fut))
		if len(path) == 2 and path[0].count("~") == 0:
			c_id, m_id, a_id, fn = decode_attachment("/".join(path))
			fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
			return self.proxy_if(await_fut(fut))
		if hasattr(self, "state"):
			url = f"{self.state['/']}/u{rpath}{rquery}"
			raise cp.HTTPRedirect(url, 307)
		assert len(path) == 1
		aid = p2n(path[0])
		resp = interface.run(f"await bot.renew_attachment({aid})")
		return self.proxy_if(resp)
	unproxy._cp_config = {"response.stream": True}

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def reupload(self, url=None, filename=None, **void):
		try:
			resp = cp.request.body.fp
		except Exception:
			print_exc()
			resp = None
		if not resp or int(cp.request.headers.get("Content-Length", 0)) < 1:
			if not url:
				return "Expected input URL or data."
			headers = Request.header()
			if cp.request.headers.get("Range"):
				headers["Range"] = cp.request.headers["Range"]
			resp = self.get_with_retries(url, headers=headers, timeout=3)
		fn = filename or (url2fn(url) if url else None)
		fut = attachment_cache.create(seq(resp), filename=fn)
		return await_fut(fut)

	def proxy_if(self, url):
		assert isinstance(url, str), url

		def requires_proxy():
			if not is_discord_attachment(url):
				return False
			if "Cf-Worker" in cp.request.headers:
				return True
			if "bot" in cp.request.headers.get("User-Agent", ""):
				return False
			if cp.request.headers.get("X-Real-Ip", "")[:3] in ("34.", "35."):
				return True
			if cp.request.headers.get("Sec-Fetch-Dest", "").casefold() == "document" and url.split("?", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[-1] not in ("zip", "7z", "tar", "bin", "png", "gif", "webp", "jpg", "jpeg", "heic", "heif", "avif"):
				return True
			if (mode := cp.request.headers.get("Sec-Fetch-Mode")):
				return mode.casefold() not in ("cors", "navigate") or cp.request.headers.get("Sec-Fetch-Site", "").casefold() not in ("none", "cross-site")
			if cp.request.headers.get("Referer"):
				return True
			return False

		if requires_proxy():
			return self.proxy(url=url)
		raise cp.HTTPRedirect(url, 307)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def proxy(self, url=None, **void):
		if not url:
			return "Expected proxy URL."
		try:
			body = cp.request.body.fp.read()
		except Exception:
			print_exc()
			body = None
		headers = Request.header()
		if cp.request.headers.get("Range"):
			headers["Range"] = cp.request.headers["Range"]
		resp = self.get_with_retries(url, headers=headers, data=body, timeout=2)
		cp.response.status = resp.status_code
		update_headers(cp.response.headers, **resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		if cp.response.headers.pop("Content-Encoding", None):
			cp.response.headers.pop("Content-Length", None)
		cp.response.headers.pop("Server", None)
		cp.response.headers.pop("Alt-Svc", None)
		if is_discord_attachment(url):
			cp.response.headers.pop("Content-Disposition", None)
			update_headers(cp.response.headers, **CHEADERS)
		ctype = resp.headers.get("Content-Type", "application/octet-stream")
		if ctype in ("text/html", "text/html; charset=utf-8", "application/octet-stream"):
			it = resp.iter_content(262144)
			b = next(it)
			mime = magic.from_buffer(b)
			if mime == "application/octet-stream":
				a = MemoryBytes(b)[:128]
				if sum(32 <= c < 128 for c in a) >= len(a) * 7 / 8:
					mime = "text/plain"
			cp.response.headers.pop("Content-Type", None)
			cp.response.headers["Content-Type"] = mime
			return resume(b, it)
		return resp.iter_content(262144)
	proxy._cp_config = {"response.stream": True}

	@cp.expose
	def static(self, *filepath, **kwargs):
		if not filepath:
			if cp.request.remote.ip == "127.0.0.1":
				STATIC.clear()
				print("Webserver cache cleared.")
				return b"\xf0\x9f\x92\x9c"
			raise PermissionError
		filename = "/".join(filepath)
		data = None
		try:
			data, mime = fetch_static("static/" + filename, ignore=True)
		except FileNotFoundError:
			try:
				data, mime = fetch_static(filename)
			except FileNotFoundError as ex:
				print(true_ip(), repr(ex))
		if data is None:
			raise FileNotFoundError(500, filepath)
		if filename.strip("/") == "notfound.png":
			cp.response.status = 404
		update_headers(cp.response.headers, **CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	def summarise(self, s, min_length=128, max_length=192):
		v = interface.run(f"STRING.summarise({json.dumps(s)},min_length={min_length},max_length={max_length})", cache=60)
		b = v.encode("utf-8")
		update_headers(cp.response.headers, **CHEADERS)
		cp.response.headers["Content-Type"] = "text/plain"
		cp.response.headers["Content-Length"] = len(b)
		cp.response.headers["ETag"] = create_etag(b)
		return b

	ecdc_running = {}
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def encodec(self, url="", name="", source="", thumbnail="", bitrate="auto", inference=False, urls=()):
		update_headers(cp.response.headers, **SHEADERS)
		if not os.path.exists(ecdc_dir):
			os.mkdir(ecdc_dir)
		if urls:
			urls = urls.split() if isinstance(urls, str) else urls
			outs = []
			for url in urls:
				url = unyt(url)
				if isinstance(bitrate, int):
					br = str(bitrate)
				elif bitrate == "auto":
					out = ecdc_exists(url)
					if out:
						br = out.rsplit("~", 1)[-1].rsplit(".", 1)[0]
					else:
						br = 24
				else:
					br = round_min(bitrate.removesuffix("k"))
				out = ecdc_dir + "!" + shash(url) + "~" + str(br) + ".ecdc"
				if out in self.ecdc_running or os.path.exists(out) and os.path.getsize(out):
					outs.append(1)
				else:
					outs.append(0)
			update_headers(cp.response.headers, **HEADERS)
			cp.response.headers["Content-Type"] = "application/json"
			return json_dumps(outs)
		if isinstance(url, list):
			url = url[0]
		url = unyt(url)
		if isinstance(bitrate, int):
			br = str(bitrate)
		elif bitrate == "auto":
			out = ecdc_exists(url)
			if out:
				br = out.rsplit("~", 1)[-1].rsplit(".", 1)[0]
			else:
				br = 24
		else:
			br = round_min(bitrate.removesuffix("k"))
		out = ecdc_dir + "!" + shash(url) + "~" + str(br) + ".ecdc"
		while out in self.ecdc_running:
			self.ecdc_running[out].result()
			time.sleep(random.random() / 16)
		mime = None
		if cp.request.body:
			b = cp.request.body.fp.read()
			if b:
				mime = magic.from_buffer(b)
			if mime in ("audio/ecdc", "application/octet-stream"):
				with open(out, "wb") as f:
					f.write(b)
		else:
			b = b""
		if b and url.startswith(API + "/ytdl") and mime not in ("audio/ecdc", "application/octet-stream"):
			u = url.replace("v=", "d=").split("d=", 1)[-1].split("&", 1)[0]
			if is_url(u):
				h = shash(u)
				fn = "~" + h + ".webm"
				fni = f"{TEMP_PATH}/filehost/" + fn
				if not os.path.exists(fni) or not os.path.getsize(fni) or os.path.getmtime(fni) >= 3600:
					with open(fni, "wb") as f:
						f.write(b)
					interface.run(f"VOICE.ytdl.cache[{repr(fn)}]=VOICE.AudioFileLink({repr(fn)},{repr(fni)},wasfile=True)")
			b = b""
		elif not b and url.startswith(API + "/ytdl") and mime not in ("audio/ecdc", "application/octet-stream"):
			u = url.replace("v=", "d=").split("d=", 1)[-1].split("&", 1)[0]
			if is_url(u):
				h = shash(u)
				fn = ecdc_dir + "!" + shash(url) + "~" + br + ".ecdc"
				if os.path.exists(fn) and os.path.getsize(fn):
					f = open(fn, "rb")
					return cp.lib.static.serve_fileobj(f, content_type="audio/ecdc", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".ecdc")
		try:
			if inference in ("True", "true", True):
				raise KeyError
			if not os.path.exists(out) or not os.path.getsize(out):
				raise KeyError
		except KeyError:
			pass
		else:
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type="audio/ecdc", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".ecdc")
		if b or inference in (None, "None", "none", "null", ""):
			cp.response.status = 204
			return b""
		self.ecdc_running[out] = Future()
		try:
			fn = temporary_file()
			if url.startswith(API):
				url2 = url
			elif is_discord_attachment(url):
				url2 = API + "/u?url=" + quote_plus(url)
			else:
				url2 = API + "/ytdl?d=" + quote_plus(url)
			with self.session.get(url2, timeout=1800, stream=True) as resp:
				resp.raise_for_status()
				with open(fn, "wb") as f:
					shutil.copyfileobj(resp.raw, f, 65536)
			assert os.path.exists(fn) and os.path.getsize(fn)
			res = interface.run(f"VOICE.ecdc_encode({repr(fn)},{repr(bitrate)},{repr(name)},{repr(source)},{repr(thumbnail)})")
			assert os.path.exists(res)
			rename(res, out)
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type="audio/ecdc", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".ecdc")
		finally:
			fut = self.ecdc_running.pop(out, None)
			fut.set_result(None)

	@cp.expose
	def decodec(self, url, fmt="opus"):
		update_headers(cp.response.headers, **SHEADERS)
		if isinstance(url, list):
			url = url[0]
		if is_discord_attachment(url):
			url = API + "/u?url=" + quote_plus(url)
		out = f"{TEMP_PATH}/!" + shash(url) + "~." + fmt
		while out in self.ecdc_running:
			self.ecdc_running[out].result()
			time.sleep(random.random() / 16)
		try:
			if not os.path.exists(out) or not os.path.getsize(out):
				raise KeyError
		except KeyError:
			pass
		else:
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type=f"audio/{fmt}", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".wav")
		self.ecdc_running[out] = Future()
		try:
			fn = temporary_file("ecdc")
			with self.session.get(url, timeout=1800, stream=True) as resp:
				with open(fn, "wb") as f:
					shutil.copyfileobj(resp.raw, f, 65536)
			out = interface.run(f"VOICE.ecdc_decode({repr(fn)},{repr(out)})")
			assert os.path.exists(out)
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type=f"audio/{fmt}", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0] + ".wav")
		finally:
			fut = self.ecdc_running.pop(out, None)
			fut.set_result(None)

	ydl_sems = {}
	ydl = None
	@cp.expose
	def ytdl(self, **kwargs):
		update_headers(cp.response.headers, **HEADERS)
		d = kwargs.get("d") or kwargs.get("download")
		v = d or kwargs.get("v") or kwargs.get("view")
		q = d or v or kwargs.get("q") or kwargs.get("query") or kwargs.get("s") or kwargs.get("search")
		if not q:
			cp.response.status = 204
			return
		if is_url(q):
			q = unyt(q)
		if is_url(v):
			v = unyt(v)
		if is_url(d):
			d = unyt(d)
		ip = true_ip()
		print("/ytdl", ip, q)
		if not self.ydl:
			self.ydl = globals()["ytdl"] = AudioDownloader()
		if v:
			fmt = kwargs.get("fmt")
			if not fmt:
				accept = cp.request.headers.get("Accept", "").lower().replace(";", ",").split(",")
				if "audio/ogg" in accept:
					fmt = "ogg"
				elif "audio/mp3" in accept:
					fmt = "mp3"
				elif "text/html" in accept:
					# Most likely a browser; browsers tend to support ogg but may refuse opus, so we pass opus codec in ogg container. Avoid passing webm as that would produce the full video.
					fmt = "ogg"
				else:
					fmt = "opus"
			assert fmt in ("h264", "h265", "h266", "av1", "mp4", "mkv", "webm", "avif", "webp", "gif", "ogg", "opus", "mp3", "flac", "wav"), f"Format {fmt} currently not supported."
			entry = self.ydl.search(v)[0]
			url = entry["url"]
			v2 = unyt(url)
			start = kwargs.get("start")
			end = kwargs.get("end")
			if start == "-":
				start = None
			if end == "-":
				end = None
			if start is not None:
				start = round_min(round(float(start), 3))
			if end is not None:
				end = round_min(round(float(end), 3))
			is_audio = fmt in ("ogg", "opus", "mp3", "flac", "wav")
			if not is_audio:
				codec = CODEC_FFMPEG.get(fmt)
				if codec:
					cdc = fmt
				if fmt in ("h264", "h265", "h266", "av1"):
					fmt = "mp4"
			else:
				codec = None
			tmpl = f"{CACHE_PATH}/{uhash(v2)}.{fmt}"
			name = tmpl.rsplit(".", 1)[0]
			if start is not None or end is not None:
				name += f"~{start}-{end}"
			if codec:
				fn = name + "~" + cdc + "." + fmt
			else:
				fn = name + "." + fmt
			print(tmpl, fn)
			if not os.path.exists(fn) or not os.path.getsize(fn):
				sem = self.ydl_sems.setdefault(ip, Semaphore(64, 256, rate_limit=8))
				with sem:
					# Separate video and audio formats
					if is_audio:
						fn2, _cdc, _dur, _ac = self.ydl.get_audio(entry, fmt=fmt, start=start, end=end)
						if fn != fn2:
							rename(fn2, fn)
						title = entry["name"]
					else:
						if fmt in ("avif", "webp", "gif"):
							fstr = f"bestvideo[ext={fmt}]/bestvideo[acssodec=none]/bestvideo"
						else:
							fstr = f"bestvideo[ext={fmt}]+bestaudio[acodec=opus]/best[ext={fmt}]/best/bestvideo+bestaudio/bestvideo"
						postprocessors = [dict(
							key="FFmpegCustomVideoConvertor",
							format=fmt,
							codec=codec,
							start=start,
							end=end,
							final=fn,
						)]
						ydl_opts = dict(
							format=fstr,
							default_search="auto",
							source_address="0.0.0.0",
							remote_components=["ejs:github"],
							final_ext=fmt,
							cachedir=CACHE_PATH,
							outtmpl=tmpl,
							windowsfilenames=True,
							cookiesfrombrowser=["firefox"],
							postprocessors=postprocessors,
						)
						title = self.ydl.run(f"ytd.YoutubeDL({repr(ydl_opts)}).extract_info({repr(url)},download=True)['title']", timeout=3600)
					assert os.path.exists(fn), f"Download unsuccessful: {fn}."
			else:
				entry = self.ydl.search(q)[0]
				title = entry["name"]
			update_headers(cp.response.headers, **CHEADERS)
			f = open(fn, "rb")
			return cp.lib.static.serve_fileobj(f, name=f"{title}.{fmt}", content_type=MIMES[fmt], disposition="inline")
		sem = self.ydl_sems.setdefault(ip, Semaphore(64, 256, rate_limit=8))
		with sem:
			entries = self.ydl.search(q, count=12)
		res = [dict(name=e["name"], url=e["url"], duration=e.get("duration"), icon=get_best_icon(e)) for e in entries]
		update_headers(cp.response.headers, **CHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return json_dumps(res)
	ytdl._cp_config = {"response.stream": True}

	@cp.expose
	def youtube_thumbnail(self, url, pos=0):
		if not is_youtube_url(url):
			raise TypeError("Invalid YouTube URL.")
		url = unyt(url)
		entry = self.ydl.search(url)[0]
		try:
			resp = self.ydl.get_thumbnail(entry, pos=pos)
		except Exception:
			print_exc()
			resp = get_best_icon(entry)
		update_headers(cp.response.headers, **CHEADERS)
		if isinstance(resp, byte_like):
			cp.response.headers["Content-Type"] = "image/jpeg"
			return resp
		assert is_url(resp)
		return self.proxy(url=resp)

	@cp.expose
	def mean_colour(self, url=""):
		resp = colour_cache.obtain(url)
		cp.response.headers["Content-Type"] = "application/json"
		update_headers(cp.response.headers, **SHEADERS)
		return json_dumps(dict(colour=resp))

	@cp.expose
	def specexec(self, url, **kwargs):
		update_headers(cp.response.headers, **HEADERS)
		argv = " ".join(itertools.chain.from_iterable(kwargs.items()))
		b = self.command(input=f"spectralpulse {url} {argv}")
		data = orjson.loads(b)
		url = data[0]["content"].replace("/d/", "/f/")
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	def filelist(self, path=None):
		update_headers(cp.response.headers, **HEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		try:
			sessid = int(cp.request.cookie["sessid"].value)
		except (KeyError, ValueError):
			return "[]"
		else:
			adata = cdict(interface.run(f"bot.data.sessions.get({repr(sessid)})"))
		if not adata:
			cp.response.cookie["sessid"] = ""
			return "[]"
		if "email" not in adata:
			if "id" not in adata:
				return "[]"
			fdata = interface.run(f"bot.data.drives.get({adata.id},[])")
		else:
			if "id" in adata:
				fdata = interface.run(
					f"bot.data.drives.setdefault({repr(adata.email)},set()).update(bot.data.drives.pop({adata.id},[]))\n"
					+ f"return bot.data.drives.get({repr(adata.email)},[])"
				)
			else:
				fdata = interface.run(f"bot.data.drives.get({repr(adata.email)},[])")
		if not path:
			return json_dumps(fdata)
		cpath = path.split("/")
		while cpath:
			fold = cpath.pop(0)
			for e in fdata:
				if isinstance(e, dict) and e.get("i") == fold:
					fdata = e.get("f")
					break
			else:
				raise FileNotFoundError(404, path, fold)
		return json_dumps(fdata)

	@cp.expose
	def teapot(self, *args, **kwargs):
		update_headers(cp.response.headers, **HEADERS)
		raise IsADirectoryError("I'm a teapot.")

	@cp.expose
	def status(self):
		cp.response.headers.update(SHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return orjson.dumps(interface.run("bot.status()"))

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	def index(self, path=None, filename=None, *args, code=None, **kwargs):
		url = HOST + "/" + cp.url(qs=cp.request.query_string).rstrip("/").split("//", 1)[-1].split("/", 1)[-1]
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
			if not random.randint(0, 1) and (dt := utc_dt()) and (dt.month, dt.day) in ((3, 31), (4, 1), (4, 2)):
				meta += f'<meta http-equiv="refresh" content={random.randint(15, 31)};url=https://{cp.request.headers["Host"]}/teapot">'
			if path:
				info = self._fileinfo(path)
				fn = info["filename"]
				if fn.startswith(".forward$"):
					info = self._fileinfo(f"@{path}")
					attachment = info["filename"]
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
		update_headers(cp.response.headers, **CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose(("favicon", "favicon.ico"))
	def favicon_ico(self, *args, **kwargs):
		data, mime = fetch_static("icon.ico")
		update_headers(cp.response.headers, **CHEADERS)
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
	def get_ip(self, *args, **kwargs):
		esubmit(app.get_ip_ex)
		data = json_dumps(dict(
			remote=true_ip(),
			host=T(self).get("ip", "127.0.0.1"),
		))
		update_headers(cp.response.headers, **SHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	def error(self, code=400):
		raise ConnectionError(int(code))

	def hash_file(self, fn):
		if os.name != "nt":
			return ""
		args = ["certutil", "-hashfile", fn, "sha384"]
		print(args)
		s = ""
		try:
			s = subprocess.check_output(args)
			s = s.splitlines()[1].decode("ascii")
			print(s)
			return s
		except Exception:
			if s:
				print(s)
			print_exc()
		return ""

	@cp.expose
	def backup(self, token="~"):
		at = AUTH.get("discord_token")
		if token != at:
			if cp.url(base="").strip("/") != at:
				raise InterruptedError
		backup = interface.run("bot.backup()", cache=60)
		update_headers(cp.response.headers, **CHEADERS)
		return cp.lib.static.serve_file(backup, content_type="application/octet-stream", disposition="attachment")
	backup._cp_config = {"response.stream": True}

	@cp.expose(("eval", "exec"))
	def execute(self, token, *args, **kwargs):
		if token != AUTH.get("discord_token"):
			raise InterruptedError
		url = cp.url(base="", qs=cp.request.query_string)
		content = urllib.parse.unquote(url.lstrip("/").split("/", 2)[-1])
		res = interface.run(content, timeout=3600)
		if isinstance(res, (io.IOBase, bytes, memoryview)):
			return res
		res = str(res)
		if res.upper().startswith("<!DOCTYPE HTML>"):
			cp.response.headers["Content-Type"] = "text/html"
		else:
			cp.response.headers["Content-Type"] = "text/plain"
		return res.encode("utf-8")

	@cp.expose
	def eval2(self, token, *args, **kwargs):
		if token != AUTH.get("discord_token"):
			raise InterruptedError
		url = cp.url(base="", qs=cp.request.query_string)
		content = urllib.parse.unquote(url.split("?", 1)[0].lstrip("/").split("/", 2)[-1])
		res = eval(content, globals())
		if isinstance(res, (io.IOBase, bytes, memoryview)):
			return res
		res = str(res)
		if res.upper().startswith("<!DOCTYPE HTML>"):
			cp.response.headers["Content-Type"] = "text/html"
		else:
			cp.response.headers["Content-Type"] = "text/plain"
		return res.encode("utf-8")

	@cp.expose
	def donation(self, data=None):
		# ip = true_ip()
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
		print(data)
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
		res = interface.run(f"bot.donate({repr(name)},{repr(uid)},{amount},{repr(msg)})")
		print(res)
		return as_str(res)

	def delay_flush(self, delay):
		time.sleep(delay)
		if utc() < self.lastflush:
			sys.__stdout__.flush()

	rapidapi = 0
	@cp.expose(("commands",))
	def command(self, content="", input="", timeout=420, redirect=""):
		ip = true_ip()
		content = input or urllib.parse.unquote(cp.url(base="", qs=cp.request.query_string).rstrip("?").split("/", 1)[-1].removeprefix("api/").split("/", 1)[-1])
		if not content.startswith("~"):
			content = "~" + content
		print("/command", ip, content)
		try:
			secret = cp.request.headers["X-RapidAPI-Proxy-Secret"]
			if secret != RAPIDAPI_SECRET:
				raise KeyError
		except KeyError:
			_data = get_geo(ip)
		else:
			ip = "0.0.0.0"
			self.rapidapi += 1
		if " " not in content:
			content += " "
		update_headers(cp.response.headers, **HEADERS)
		cp.response.headers["content-type"] = "application/json"
		try:
			resp = interface.run(f"await bot.run_simulate({repr(ip)},{repr(content)})", timeout=3600)
		except Exception as ex:
			cp.response.status = 500
			return json_dumps(dict(exception=ex.__class__.__name__, message=str(ex)))
		return json_dumps(resp)

	@cp.expose(("cat", "cats", "dog", "dogs", "neko", "nekos"))
	def imagepool(self, tag="", refresh=60):
		name = cp.url(base="").rsplit("/", 1)[-1]
		command = name.rstrip("s")
		info = interface.run(f"bot.commands.{command}[0](bot=bot,embed=False)")
		if fcdict(cp.request.headers).get("Accept") == "application/json":
			return info
		url = info["url"]
		if refresh:
			refresh_url = f"{API}/{name}"
			if tag:
				refresh_url += f"/{tag}"
			refresh_info = f'<meta http-equiv="refresh" content="{refresh};URL={refresh_url}">'
		update_headers(cp.response.headers, **HEADERS)
		return f"""<!DOCTYPE html><html><head><script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7025724554077000" crossorigin="anonymous"></script><meta property="og:image" content="{url}">{refresh_info}<meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="background-color:black;"><img src="{url}" style="margin:0;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);max-width:100%;max-height:100%"></body></html>"""


def terminate():
	if ytdl:
		ytdl.close()
	return cp.engine.exit()


if __name__ == "__main__":
	# logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
	pid = os.getpid()
	ppid = os.getppid()
	print(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	app = Server()
	self = server = cp.Application(app, "/", config)
	esubmit(app.get_ip_ex)
	interface.start()
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
