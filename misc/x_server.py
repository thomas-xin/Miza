import base64
import concurrent.futures
import datetime
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
import orjson
import psutil
import niquests
from concurrent.futures import Future
from math import ceil
from traceback import print_exc
from cheroot import errors
from cherrypy._cpdispatch import Dispatcher
from .asyncs import Semaphore, SemaphoreOverflowError, eloop, esubmit, tsubmit, csubmit, await_fut, gather
from .types import byte_like, as_str, cdict, suppress, round_min, regexp, json_dumps, resume, RangeSet, MemoryBytes
from .util import hwaccel, fcdict, nhash, shash, uhash, bytes2zip, zip2bytes, enc_box, EvalPipe, AUTH, TEMP_PATH, reqs, MIMES, tracebacksuppressor, force_kill, utc, ts_us, is_url, p2n, n2p, leb128, decode_leb128, get_mime, ecdc_dir, url_parse, rename, url_unparse, url2fn, is_youtube_url, seq, Cache, Request, magic, is_discord_attachment, is_miza_attachment, unyt, ecdc_exists, get_duration, CACHE_PATH, T, byte_scale, decode_attachment, expand_attachment, shorten_attachment, update_headers
from .caches import attachment_cache, upload_cache, download_cache, colour_cache
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
utc_dt = datetime.datetime.utcnow
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

def map_url(url):
	if not isinstance(url, str):
		return url
	if url.startswith(HOST + "/u"):
		return url.replace(HOST + "/u", "M$")
	if url.startswith(API + "/u"):
		return url.replace(API + "/u", "M$")
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
	if url.startswith("M$"):
		return HOST + "/u" + url[2:]
	return "https://" + url.replace(
		"D$", "cdn.discordapp.com/attachments/"
	).replace(
		"G$", ".amazonaws.com/www.guilded.gg/ContentMediaGenericFiles/"
	)

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
		resp = errdata.get(status) or errdata.setdefault(status, reqs.next().get(f"https://http.cat/{status}", timeout=5))
		head = resp.headers.copy()
		body = resp.content
	update_headers(head, **HEADERS)
	head["Content-Length"] = len(body)
	cp.response.status = status
	update_headers(cp.response.headers, **head)
	cp.response.headers.pop("Connection", None)
	print_exc()
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
		"tools.gzip.mime_types": ["text/plain", "application/json", "application/javascript"],
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
			ip = IP = reqs.next().get("https://api.ipify.org", verify=False, timeout=5).text
	try:
		resp = TZCACHE[ip]
	except KeyError:
		if geo_count & 1:
			url = f"http://ip-api.com/json/{ip}?fields=256"
		else:
			url = f"https://demo.ip-api.com/json/{ip}?fields=256&key=test-demo-pro"
		geo_count += 1
		with geo_sem:
			resp = reqs.next().get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}", "Origin": "https://members.ip-api.com"}, timeout=5)
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
				resp = self.session.get(url, headers=headers, data=data, verify=i == 0, timeout=timeout + i ** 2)
				resp.raise_for_status()
			except Exception:
				if i < retries - 1:
					continue
				raise
			else:
				return resp
		return resp

	@cp.expose(("fi",))
	def fileinfo(self, *path, **void):
		update_headers(cp.response.headers, **SHEADERS)
		assert len(path) in (1, 2) and path[0].count("~") == 0
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		try:
			info = download_cache[url]
		except KeyError:
			resp = self.get_with_retries(url, timeout=3)
			data = seq(resp)
		else:
			data = MemoryBytes(info)
		length, i = decode_leb128(data, mode="index")
		content = data[i:i + length]
		try:
			encoded = zip2bytes(content)
		except Exception:
			encoded = bytes(content)
		cp.response.headers["Content-Type"] = "application/json"
		return bytes(encoded)

	@cp.expose
	def download(self, *path, download="1", **void):
		update_headers(cp.response.headers, **CHEADERS)
		assert len(path) in (1, 2) and path[0].count("~") == 0
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		callback = None
		try:
			info = download_cache[url]
		except KeyError:
			resp = self.get_with_retries(url, timeout=3)
			data = seq(resp)

			def callback(data):
				data.seek(0)
				download_cache[url] = bytes(data.read())
		else:
			data = MemoryBytes(info)
		length, i = decode_leb128(data, mode="index")
		content = data[i:i + length]
		try:
			encoded = zip2bytes(content)
		except Exception:
			encoded = bytes(content)
		info = cdict(orjson.loads(encoded))
		endpoint = cp.url(qs=cp.request.query_string, base="")[1:].split("/", 1)[0]
		download = (download and download[0] not in "0fFnN") and endpoint.startswith("d")
		if download:
			cp.response.headers["Content-Disposition"] = "attachment; " * bool(download) + "filename=" + json.dumps(info.filename)
		cp.response.headers["Attachment-Filename"] = info.filename
		cp.response.headers["Content-Type"] = info.mimetype
		cp.response.headers["ETag"] = json_dumps(f"{info.get('timestamp', 0)};{info.get('hash', info.filename)}")
		return self.dyn_serve(list(info.chunks), size=info.size, head=cdict(data=data, index=i + length), callback=callback)
	download._cp_config = {"response.stream": True}

	@tracebacksuppressor
	def dyn_serve(self, urls, size=0, head=None, callback=None):
		brange = cp.request.headers.get("Range", "").removeprefix("bytes=")
		headers = fcdict(cp.request.headers)
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
		cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size}" for start, end in ranges)
		cp.response.headers["Content-Range"] = cr
		cp.response.headers["Content-Length"] = str(length)
		cp.response.headers["Accept-Ranges"] = "bytes"
		return self._dyn_serve(urls, ranges, headers, head=head, callback=callback)

	@tracebacksuppressor(GeneratorExit)
	def _dyn_serve(self, urls, ranges, headers, head=None, callback=None):
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
					resp = reqs.next().head(u, headers=headers, timeout=3)
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
						yield from resp.iter_content(1048576)
						return
					yield from resp.iter_content(262144)

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

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def upload(self, filename="b", hash="", position=0, size=0):
		position = int(position)
		size2 = int(cp.request.headers.get("Content-Length", 0))
		size = int(size) or size2
		h = true_ip() + hash
		try:
			fp = cp.request.body.fp
		except Exception:
			print_exc()
			fp = None
		update_headers(cp.response.headers, **HEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		if not size or not fp:
			if h in upload_cache:
				return json_dumps(list(upload_cache[h].keys()))
			return b"Expected input data."
		if size > 1 << 40:
			return b"Maximum filesize is 1TB."
		assert position == 0 or h in upload_cache
		assert not position % attachment_cache.max_size
		if h not in upload_cache:
			private_key = os.urandom(8)
			hashable = private_key + b"~" + enc_box._key
			public_key = shash(hashable)
			upload_cache[h] = cdict(info=cdict(filename=filename, mimetype=cp.request.headers.get("mimetype", "application/octet-stream"), hash="", timestamp=utc(), size=size, key=public_key, chunks=[]), chunkinfo={}, required=RangeSet(0, size), key=private_key)

		def start_upload(position):
			if position == 0:
				head = fp.read(attachment_cache.max_size // 2)
				upload_cache[h].info.mimetype = magic.from_buffer(head)
				upload_cache[h].info.hash = shash(head)
				upload_cache[h].head = head
				upload_cache[h].required.remove(0, len(head))
				position += len(head)
			yield b"{"
			while upload_cache[h].required:
				fn = "c"
				b = fp.read(attachment_cache.max_size)
				if not b:
					break
				fut = attachment_cache.create(b, filename=fn)
				url = await_fut(fut)
				cid, mid, aid, fn = expand_attachment(url)
				uchunk = shorten_attachment(cid, mid, aid, fn, size=len(b))
				upload_cache[h].chunkinfo[position] = uchunk
				upload_cache[h].required.remove(position, position + len(b))
				position += len(b)
			if not upload_cache[h].required:
				upload_cache[h].info["chunks"] = [v for k, v in sorted(upload_cache[h].chunkinfo.items())]
				debug_info = upload_cache[h].info.copy()
				debug_info.pop("head", None)
				print("INFO:", debug_info)
				data = json_dumps(upload_cache[h].info)
				if len(data) > 262144:
					data2 = bytes2zip(data)
					if len(data2) <= len(data) / 2 or len(data) >= attachment_cache.max_size // 4:
						data = data2
				encoded = leb128(len(data)) + data
				head = upload_cache[h].head
				assert len(encoded) + len(head) <= attachment_cache.max_size, f"Unable to encode file header: {len(encoded) + len(head)} > {attachment_cache.max_size}"
				fut = attachment_cache.create(encoded + head, filename=filename, editable=True)
				url = await_fut(fut)
				cid, mid, aid, fn = expand_attachment(url)
				uhead = shorten_attachment(cid, mid, aid, fn, mode="p") + "?key=" + base64.urlsafe_b64encode(upload_cache[h].key).rstrip(b"=").decode("ascii")
				yield b'"url":"' + uhead.encode("utf-8") + b'"'
			yield b"}"
		return start_upload(position)
	upload._cp_config = {"response.stream": True}

	@cp.expose
	def delete(self, *path, key=None, **void):
		assert len(path) in (1, 2) and path[0].count("~") == 0
		assert key, "File Key Required."
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		resp = self.get_with_retries(url, timeout=3)
		data = seq(resp)
		length, i = decode_leb128(data, mode="index")
		content = data[i:i + length]
		try:
			data = zip2bytes(content)
		except Exception:
			data = content
		info = orjson.loads(data)
		hashable = base64.urlsafe_b64decode(key + "=") + b"~" + enc_box._key
		public_key = shash(hashable)
		assert public_key == info["key"], "File Key Mismatch."
		deletes = set()
		deletes.add((c_id, m_id))
		for url in info["chunks"]:
			c_id, m_id, a_id, fn = expand_attachment(url)
			deletes.add((c_id, m_id))
		print("DELETES:", deletes)
		futs = []
		for c_id, m_id in deletes:
			fut = csubmit(attachment_cache.delete(c_id, m_id))
			futs.append(fut)
		await_fut(gather(*futs))

	@cp.expose
	def edit(self, *path, key=None, **void):
		update_headers(cp.response.headers, **CHEADERS)
		assert len(path) in (1, 2) and path[0].count("~") == 0
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		callback = None
		try:
			info = download_cache[url]
		except KeyError:
			resp = self.get_with_retries(url, timeout=3)
			data = seq(resp)

			def callback(data):
				data.seek(0)
				download_cache[url] = bytes(data.read())
		else:
			data = MemoryBytes(info)
		length, i = decode_leb128(data, mode="index")
		content = data[i:i + length]
		try:
			encoded = zip2bytes(content)
		except Exception:
			encoded = bytes(content)
		info = cdict(orjson.loads(encoded))
		return str(info)

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
		resp = interface.run(f"bot.renew_attachment({aid})")
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
			if (mode := cp.request.headers.get("Sec-Fetch-Mode")):
				return mode.casefold() != "cors"
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
	def static(self, *filepath):
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
			t = ts_us()
			fn = f"{TEMP_PATH}/{t}"
			if url.startswith(API):
				url2 = url
			elif is_discord_attachment(url):
				url2 = API + "/u?url=" + url_parse(url)
			else:
				url2 = API + "/ytdl?d=" + url_parse(url)
			with reqs.next().get(url2, timeout=1800, stream=True) as resp:
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
			url = API + "/u?url=" + url_parse(url)
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
			t = ts_us()
			fn = f"{TEMP_PATH}/{t}.ecdc"
			with reqs.next().get(url, timeout=1800, stream=True) as resp:
				with open(fn, "wb") as f:
					shutil.copyfileobj(resp.raw, f, 65536)
			out = interface.run(f"VOICE.ecdc_decode({repr(fn)},{repr(out)})")
			assert os.path.exists(out)
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type=f"audio/{fmt}", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0] + ".wav")
		finally:
			fut = self.ecdc_running.pop(out, None)
			fut.set_result(None)

	ydl_sems = Cache(timeout=8)
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
			assert fmt in ("mp4", "mkv", "webm", "avif", "webp", "gif", "ogg", "opus", "mp3", "flac", "wav"), f"Format {fmt} currently not supported."
			tmpl = f"{CACHE_PATH}/{uhash(v)}.{fmt}"
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
			name = tmpl.rsplit(".", 1)[0]
			if start is not None or end is not None:
				name += f"~{start}-{end}"
			fn = name + "." + fmt
			print(tmpl, fn)
			if not os.path.exists(fn) or not os.path.getsize(fn):
				sem = self.ydl_sems.setdefault(ip, Semaphore(64, 256, rate_limit=8))
				with sem:
					# Separate video and audio formats
					if fmt in ("ogg", "opus", "mp3", "flac", "wav"):
						entry = self.ydl.search(v)[0]
						fn2, _cdc, _dur, _ac = self.ydl.get_audio(entry, fmt=fmt, start=start, end=end)
						if fn != fn2:
							rename(fn2, fn)
						title = entry["name"]
					else:
						if fmt in ("avif", "webp", "gif"):
							fstr = f"bestvideo[ext={fmt}]/bestvideo[acodec=none]/bestvideo"
						else:
							fstr = f"bestvideo[ext={fmt}]+bestaudio[acodec=opus]/best[ext={fmt}]/best/bestvideo+bestaudio/bestvideo"
						postprocessors = [dict(
							key="FFmpegCustomVideoConvertor",
							format=fmt,
							codec="libsvtav1",
							start=start,
							end=end,
						)]
						ydl_opts = dict(
							format=fstr,
							default_search="auto",
							source_address="0.0.0.0",
							final_ext=fmt,
							cachedir=CACHE_PATH,
							outtmpl=tmpl,
							windowsfilenames=True,
							cookiesfrombrowser=["firefox"],
							postprocessors=postprocessors,
						)
						title = self.ydl.run(f"ytd.YoutubeDL({repr(ydl_opts)}).extract_info({repr(v)},download=True)['title']", timeout=3600)
					assert os.path.exists(fn), "Download unsuccessful."
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

	def optimise_video(self, of, size, mime):
		print("Convert", of, mime, size)
		dur = get_duration(of)
		if mime.split("/", 1)[-1] in ("mp4", "webm"):
			if (not dur or dur > 3600) and (not size or size > 524288000):
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
					fut = esubmit(proc.wait, timeout=8)
					_res = fut.result(timeout=8)
					resp = proc.stdout.read()
				except Exception:
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
						timeout=720,
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
					timeout=720,
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
				if hwaccel == "cuda":
					args.extend(("-c:v", "av1_nvenc"))
					import torch
					devid = random.choice([i for i in range(torch.cuda.device_count()) if (torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor) >= (8, 9)])
					args = args[:1] + ["-hwaccel_device", str(devid)] + args[1:]
				else:
					args.extend(("-c:v", "libsvtav1"))
				args.extend((
					"-crf",
					"42",
					fo,
				))
			else:
				fmt = "mp4"
				fo = f"{of}.{fmt}"
				if hwaccel == "cuda":
					args.extend(("-c:v", "h264_nvenc"))
					devid = random.randint(0, ceil(torch.cuda.device_count() / 2))
					args = args[:1] + ["-hwaccel_device", str(devid)] + args[1:]
				else:
					args.extend(("-c:v", "h264"))
				args.extend((
					"-crf",
					"30",
					fo,
				))
			print(args)
			proc = psutil.Popen(args, stdin=subprocess.DEVNULL)
			fut = esubmit(proc.wait, timeout=120)
			try:
				fut.result(timeout=120)
			except concurrent.futures.TimeoutError:
				if proc.is_running() and os.path.exists(fo) and os.path.getsize(fo):
					fut = None
			if not fut:
				fut = esubmit(proc.wait, timeout=3600)
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
		of = f"{TEMP_PATH}/filehost/{IND}{ts}~.temp$@{name}.{fmt}"
		os.rename(fo, of)
		size = os.path.getsize(of)
		mime = get_mime(of)
		return of

	def optimise_image(self, of, size, mime):
		fmt = "webp"
		fo = interface.run(f"process_image({repr(of)},resize_mult,[1,1,'auto','-f',{repr(fmt)}],timeout=120)")
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
		of = f"{TEMP_PATH}/filehost/{IND}{ts}~.temp$@{name}.{fmt}"
		os.rename(fo, of)
		size = os.path.getsize(of)
		# mime = get_mime(of)
		return of

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
	@cp.tools.accept(media="multipart/form-data")
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
		url = f"/p/{n2p(ts)}"
		if len(urls) <= 1:
			s = f'<!DOCTYPE HTML><!--["{urls[0]}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={urls[0]}"/></html>'
		else:
			with reqs.next().head(urls[0], headers=Request.header(), stream=True, timeout=10) as resp:
				mime = resp.headers.get("Content-Type") or "text/html"
			ftype = 3
			s =  (
				f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={url}"/>'
				+ f'<!--["Multi-redirect",0,"{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--></html>'
			)
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
		update_headers(cp.response.headers, **HEADERS)
		raise cp.HTTPRedirect(url, status=307)

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
		if "\x7f" in content and ip in ("127.0.0.1", ADDRESS, T(self).get("ip", None)):
			cp.request.no_log = True
			cl = int(cp.request.headers["Content-Length"])
			d = cp.request.body.read(cl)
			try:
				j = orjson.loads(d)
			except Exception:
				try:
					j = eval(d, {}, {})
				except Exception:
					j = dict(error=d)
			if isinstance(j, dict) or not j:
				return
			else:
				return b"\xf0\x9f\x92\x9c"
		content = input or urllib.parse.unquote(cp.url(base="", qs=cp.request.query_string).rstrip("?").split("/", 1)[-1].removeprefix("api/").split("/", 1)[-1])
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
		j, after = interface.run(f"bot.run_simulate({repr(content)})")
		a = after - utc()
		if a > 0:
			cp.response.headers["Retry-After"] = a
		update_headers(cp.response.headers, **HEADERS)
		return json_dumps(j)

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
