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
import urllib, urllib.parse
import zipfile
import cheroot, cheroot.server
import cherrypy
import diskcache
import niquests
import orjson
import psutil
from concurrent.futures import Future
from traceback import print_exc
from cheroot import errors
from cherrypy._cpdispatch import Dispatcher
from .asyncs import Semaphore, SemaphoreOverflowError, eloop, submit_thread, create_thread, create_task, await_fut
from .types import ts_us, byte_like, as_str, cdict, suppress, round_min, regexp, json_dumps, resume, getattr_chain, MemoryBytes
from .util import fcdict, nhash, uhash, EvalPipe, AUTH, TEMP_PATH, MIMES, tracebacksuppressor, utc, is_url, p2n, n2p, mime_into, rename, url_unparse, url2fn, is_youtube_url, seq, Request, get_mime, mime_from_file, is_discord_attachment, is_miza_attachment, unyt, CACHE_PATH, AutoCache, T, byte_scale, decode_attachment, update_headers, CODEC_FFMPEG, VISUAL_FORMS
from .caches import attachment_cache, colour_cache
from .audio_downloader import AudioDownloader, get_best_icon

ytdl_fut = submit_thread(AudioDownloader, workers=1)

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


create_task(Request._init_())
create_thread(eloop.run_forever)
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

@functools.lru_cache(maxsize=256)
def get_size_mime(head, tail, count, chunksize):
	fut = submit_thread(niquests.head, head)
	resp = niquests.head(tail)
	lastsize = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
	size = chunksize * (count - 1) + lastsize
	resp = fut.result()
	mimetype = resp.headers.get("Content-Type", "application/octet-stream")
	return mimetype, size, lastsize

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		p = path.lstrip("/")
		print(json.dumps(p))
		if p.split("/", 1)[0] in (".git", ".env", "admin", "private", "internal", "administrator"):
			return super().__call__("/rickroll")
		while p:
			if p == "ip":
				p = "get_ip"
			elif p.split("/", 1)[0] in ("f", "d"):
				p = "download/" + p.split("/", 1)[-1]
			else:
				break
		p = "/" + p
		return super().__call__(p)

error_map: dict[type[BaseException], int] = {
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
		"tools.gzip.mime_types": ["text/plain", "text/html", "text/css", "text/csv", "image/svg+xml", "text/xml", "text/md", "text/markdown", "text/javascript", "application/json", "application/javascript"],
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
	"Access-Control-Expose-Headers": "*",
}

CHEADERS = {
	"Cache-Control": "public,max-age=21600,stale-while-revalidate=1073741824,stale-if-error=1073741824",
	"Accept-Ranges": "bytes",
}
SHEADERS = {
	"Cache-Control": "public, max-age=60,stale-while-revalidate=1073741824,stale-if-error=1073741824",
	"Accept-Ranges": "bytes",
}
CHEADERS.update(HEADERS)
SHEADERS.update(HEADERS)


def fetch_static(path):
	while path.startswith("../"):
		path = path[3:]
	fn = "misc/web/" + path.lstrip("/")
	for exists in (fn, fn + ".zip", fn + ".html"):
		if os.path.exists(exists):
			break
	else:
		raise FileNotFoundError(fn)
	ts = os.path.getmtime(exists)
	try:
		data, mime, ts2 = STATIC[path]
		if ts2 < ts:
			raise KeyError
	except KeyError:
		if zipfile.is_zipfile(exists):
			with zipfile.ZipFile(exists) as z:
				data = z.read(path.rsplit("/", 1)[-1])
		else:
			with open(exists, "rb") as f:
				data = f.read()
		mime = mime_from_file(data)
		if mime == "text/plain" and "." in exists:
			fmt = exists.rsplit(".", 1)[-1]
			match fmt:
				case "js":
					mime = "application/javascript"
				case "svg":
					mime = "image/svg+xml"
				case _:
					mime = "text/" + fmt
		STATIC[path] = (data, mime, ts)
	return data, mime

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

	def get_with_retries(self, url, headers={}, data=None, timeout=3, retries=5):
		for i in range(retries):
			try:
				session = self.session if url.startswith("https://") and not is_discord_attachment(url) and i == 0 else niquests
				resp = session.get(url, headers=headers, data=data, verify=i <= 1, timeout=timeout + i ** 2)
				resp.raise_for_status()
			except niquests.exceptions.HTTPError:
				if resp.status_code in (400, 401, 402, 403, 404, 405):
					raise
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
						resp = niquests.head(u, timeout=3)
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
					fut = submit_thread(get_chunk, u, headers, start, end, pos, ns, big)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns
					big = True
				for fut in futs:
					yield from fut.result()

	@cp.expose(alias=("c",))
	def chunked_proxy(self, path, *void):
		with tracebacksuppressor:
			urls, chunksize = await_fut(attachment_cache.obtains(path))
			mimetype, size, lastsize = get_size_mime(urls[0], urls[-1], len(urls), chunksize)
			update_headers(cp.response.headers, **CHEADERS)
			cp.response.headers["Content-Type"] = mimetype
			new_urls = [f"{url}&S={lastsize if i >= len(urls) - 1 else chunksize}" for i, url in enumerate(urls)]
			return self.dyn_serve(new_urls, size)
	chunked_proxy._cp_config = {"response.stream": True}

	@cp.expose(alias=("u",))
	def unproxy(self, *path, url=None, force=False, download=False, **query):
		if url:
			return self.proxy_if(url, force=force, download=download)
		rpath = "/".join(path)
		if rpath:
			rpath = "/" + rpath
		rquery = cp.request.query_string and "?" + cp.request.query_string
		if len(path) == 1 and path[0].count("~") == 2:
			return self.proxy_if(await_fut(attachment_cache.obtain(*path[0].split(".", 1)[0].split("~", 2))), force=force, download=download)
		if len(path) in (1, 2) and path[0].count("~") == 0:
			c_id, m_id, a_id, fn = decode_attachment("/".join(path))
			try:
				return self.proxy_if(await_fut(attachment_cache.obtain(c_id, m_id, a_id, fn)), force=force, download=download)
			except ConnectionError as ex:
				if ex.errno == 404:
					attachment_cache.remove_cached(cp.url())
					return self.proxy_if(await_fut(attachment_cache.obtain(c_id, m_id, a_id, fn)), force=force, download=download)
				raise
		if hasattr(self, "state"):
			url = f"{self.state['/']}/u{rpath}{rquery}"
			raise cp.HTTPRedirect(url, 307)
		raise FileNotFoundError(*path)
	unproxy._cp_config = {"response.stream": True}

	upload_cache = attachment_cache.secondary
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def upload(self, url=None, filename=None, persistent=False, file=None, **void):
		update_headers(cp.response.headers, **HEADERS)
		try:
			fp = getattr_chain(file, "file.file", None) or cp.request.body.fp
		except Exception:
			print_exc()
			fp = None
		if not fp or int(cp.request.headers.get("Content-Length", 0)) < 1:
			if not url:
				return "Expected input URL or data."
			fp = await_fut(attachment_cache.download(url, read=True))
		fn = filename or (url2fn(url) if url else None)
		if persistent not in (0, "0", ""):
			return await_fut(attachment_cache.create_dynamic(fp, filename=fn))
		ts = n2p(ts_us()).decode("ascii")
		self.upload_cache.set(ts, fp, tag=fn, read=True)
		fp, fn = diskcache.FanoutCache.get(self.upload_cache, ts, tag=True, read=True)
		mimetype = get_mime(fp)
		ext = mime_into(mimetype)
		url = API + f"/f/{ts}.{ext}"
		return url

	@cp.expose
	def download(self, path, download=False):
		update_headers(cp.response.headers, **HEADERS)
		ts = path.rsplit("/", 1)[-1].split(".", 1)[0]
		fp, fn = diskcache.FanoutCache.get(self.upload_cache, ts, tag=True, read=True)
		if not fp:
			raise FileNotFoundError(404, path)
		mimetype = get_mime(fp)
		fp.seek(0)
		return cp.lib.static.serve_fileobj(fp, name=fn, content_type=mimetype, disposition="attachment" if download else "inline")
	download._cp_config = {"response.stream": True}

	def proxy_if(self, url, force=False, download=False):
		assert isinstance(url, str), url

		def requires_proxy():
			if "Cf-Worker" in cp.request.headers:
				return True
			ua = cp.request.headers.get("User-Agent", "")
			if "bot" in ua or "Bot" in ua:
				return False
			if download and is_discord_attachment(url):
				if url.split("?", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[-1] in VISUAL_FORMS:
					return True
				return False
			return True

		if requires_proxy():
			return self.proxy(url=url, force=force, download=download)
		raise cp.HTTPRedirect(url, 307)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def proxy(self, url=None, force=False, download=False, **void):
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
		heads = resp.headers
		cp.response.status = resp.status_code
		update_headers(cp.response.headers, **heads)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		if cp.response.headers.pop("Content-Encoding", None):
			cp.response.headers.pop("Content-Length", None)
		cp.response.headers.pop("Server", None)
		cp.response.headers.pop("Alt-Svc", None)
		filename = heads.get("attachment-filename") or urllib.parse.unquote(heads.get("content-disposition", "").split("filename=", 1)[-1].lstrip('"').split('"', 1)[0].strip().strip('"').strip("'") or url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0])
		disposition = "attachment" if download else "inline"
		cp.response.headers.pop("Content-Disposition", None)
		if filename:
			cp.response.headers["Content-Disposition"] = f"{disposition}; filename={urllib.parse.quote(filename)}"
		ctype = heads.get("Content-Type", "application/octet-stream")
		if ctype in ("text/html", "text/html; charset=utf-8", "application/octet-stream"):
			it = resp.iter_content(262144)
			b = next(it)
			mime = get_mime(b)
			if mime == "application/octet-stream":
				a = MemoryBytes(b)[:128]
				if sum(32 <= c < 128 for c in a) >= len(a) * 7 / 8:
					mime = "text/plain"
			cp.response.headers.pop("Content-Type", None)
			cp.response.headers["Content-Type"] = mime
			return resume(b, it)
		return resp.iter_content(262144)
	proxy._cp_config = {"response.stream": True}

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
			self.ydl = globals()["ytdl"]
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
						if is_url(fn2):
							subprocess.run(["streamshatter", fn2, fn])
						elif fn != fn2:
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
		resp = await_fut(colour_cache.obtain(url))
		cp.response.headers["Content-Type"] = "application/json"
		update_headers(cp.response.headers, **SHEADERS)
		return json_dumps(dict(colour=resp))

	@cp.expose
	def status(self):
		cp.response.headers.update(SHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return orjson.dumps(interface.run("bot.status()"))

	alias = tuple([fn.split("/", 1)[0].rsplit(".", 1)[0] for fn in os.listdir("misc/web")])
	alias += tuple(a + ".html" for a in alias)
	@cp.expose(alias=alias)
	def index(self, *args, **kwargs):
		path_info = cp.request.path_info
		if path_info == "/":
			path_info = "index.html"
		data, mime = fetch_static(path_info)
		update_headers(cp.response.headers, **CHEADERS)
		if "files" in path_info.split("/"):
			url = kwargs.get("url")
			print(url)
			if url:
				headers = await_fut(attachment_cache.scan_headers(url, fc=True))
				_fn = urllib.parse.unquote(headers["content-disposition"].split("filename=", 1)[-1])
				_size = byte_scale(headers["content-length"]) + "B"
				_mime = headers.get("content-type", "application/octet-stream")
				data = (
					f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="description" content="Your local loyal multipurpose Discord bot.">

	<!-- Open Graph Meta Tags -->
	<meta property="og:url" content="https://mizabot.xyz">
	<meta property="og:type" content="website">
	<meta property="og:title" content="{_fn}">
	<meta property="og:description" content="{_mime}, {_size}">
	<meta property="og:image" content="{url}">

	<!-- Twitter Meta Tags -->
	<meta name="twitter:card" content="summary_large_image">
	<meta property="twitter:domain" content="mizabot.xyz">
	<meta property="twitter:url" content="https://mizabot.xyz">
	<meta name="twitter:title" content="{_fn}">
	<meta name="twitter:description" content="{_mime}, {_size}">
	<meta name="twitter:image" content="{url}">

	<link rel="icon" type="image/png" href="/assets/images/logo512.webp">
	<link rel="stylesheet" href="/assets/css/global.css">
	<script defer="defer" src="/assets/js/global.js"></script>

	<!-- Page specific: -->
	<title>Files: Miza</title>
	<meta content="#FF7FBF" data-react-helmet="true" name="theme-color">
	<link rel="stylesheet" href="/assets/css/files.css">
	<script defer="defer" src="/assets/js/files.js"></script>
</head>
""".encode("utf-8")
					+ data.split(b"</head>", 1)[-1]
				)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose(alias=("favicon", "favicon.ico"))
	def favicon_ico(self, *args, **kwargs):
		data, mime = fetch_static("assets/images/mizaleaf.webp")
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
		submit_thread(app.get_ip_ex)
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

	git_cache = AutoCache(f"{CACHE_PATH}/git", stale=3600, timeout=86400 * 7)
	@cp.expose
	def git_stats(self):
		def get_git():
			s = as_str(subprocess.check_output(["git", "log" , "--shortstat", "--format=oneline"]))
			lines = s.splitlines()
			commit_count = sum(bool(line and line[0].strip()) for line in lines)
			changed = [line for line in lines if line and not line[0].strip()]
			line_changes = 0
			for line in changed:
				args = line.split(", ", 3)
				inserts = deletions = "0"
				if len(args) > 1:
					inserts = args[1]
					if len(args) > 2:
						deletions = args[2]
				line_changes += int(inserts.split(None, 1)[0])
				line_changes += int(deletions.split(None, 1)[0])
			return commit_count, line_changes
		commit_count, changed = self.git_cache.retrieve("shortstat", get_git)
		update_headers(cp.response.headers, **SHEADERS)
		cp.response.headers["content-type"] = "application/json"
		return orjson.dumps([commit_count, changed])

	@cp.expose(alias=("eval", "exec"))
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

	rapidapi = 0
	@cp.expose
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

	@cp.expose
	def rickroll(self, *args, **kwargs):
		raise cp.HTTPRedirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 308)


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
	submit_thread(app.get_ip_ex)
	interface.start()
	ytdl = ytdl_fut.result()
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
