import base64
import concurrent.futures
import datetime
import io
import itertools
import json
import logging
import os
import pickle
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
import numpy as np
import orjson
import psutil
import requests
from collections import deque
from concurrent.futures import Future
from math import ceil
from traceback import print_exc
from cheroot import errors
from cherrypy._cpdispatch import Dispatcher
from .asyncs import Semaphore, SemaphoreOverflowError, eloop, esubmit, tsubmit, csubmit, await_fut, gather, CloseableAsyncIterator
from .smath import supersample, xrand
from .types import byte_like, as_str, astype, cdict, suppress, round_min, full_prune, literal_eval, regexp, loop, json_dumps, alist, resume, RangeSet, MemoryBytes
from .util import hwaccel, fcdict, nhash, shash, bytes2zip, zip2bytes, enc_box, EvalPipe, AUTH, TEMP_PATH, reqs, MIMES, tracebacksuppressor, is_strict_running, force_kill, utc, ts_us, is_url, p2n, n2p, leb128, decode_leb128, get_mime, ecdc_dir, url_parse, url_unparse, url2fn, smart_split, seq, Request, magic, is_discord_attachment, unyt, ecdc_exists, get_duration, evalex, evalEX, DownloadingFile, T, tik_encode, tik_decode, longest_prefix, longest_common_substring, sublist_index, byte_scale, decode_attachment, attachment_cache, expand_attachment, shorten_attachment, upload_cache, download_cache


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
interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
# print = interface.print


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
	"""
	request = cp.serving.request
	if T(request).get("no_log"):
		return
	# remote = request.remote
	response = cp.serving.response
	outheaders = response.headers
	inheaders = request.headers
	if response.output_status is None:
		status = '-'
	else:
		status = response.output_status.split(b' ', 1)[0]
		status = status.decode('ISO-8859-1')

	atoms = {
		'h': true_ip(),
		'l': '-',
		'u': T(request).get('login') or '-',
		't': self.time(),
		'r': request.request_line,
		's': status,
		'b': str(dict.get(inheaders, 'Range', '') or '-') + "/" + str(dict.get(outheaders, 'Content-Length', '') or '-'),
		'f': dict.get(inheaders, 'Referer', ''),
		'a': dict.get(inheaders, 'User-Agent', ''),
		'o': dict.get(inheaders, 'Host', '-'),
		'i': request.unique_id,
		'z': cp._cplogging.LazyRfc3339UtcTime(),
	}
	for k, v in atoms.items():
		if not isinstance(v, str):
			v = str(v)
		v = v.replace('"', '\\"').encode('utf8')
		# Fortunately, repr(str) escapes unprintable chars, \n, \t, etc
		# and backslash for us. All we have to do is strip the quotes.
		v = repr(v)[2:-1]

		# in python 3.0 the repr of bytes (as returned by encode)
		# uses double \'s.  But then the logger escapes them yet, again
		# resulting in quadruple slashes.  Remove the extra one here.
		v = v.replace('\\\\', '\\')

		# Escape double-quote.
		atoms[k] = v

	try:
		self.access_log.log(
			logging.INFO, self.access_log_format.format(**atoms))
	except Exception:
		self(traceback=True)
	"""
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
	head.update(HEADERS)
	head["Content-Length"] = len(body)
	cp.response.status = status
	cp.response.headers.update(head)
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
		try:
			data = STATIC[path]
		except KeyError:
			fn = f"misc/web/{path}"
			fn2 = fn + ".zip"
			if os.path.exists(fn2) and zipfile.is_zipfile(fn2):
				with zipfile.ZipFile(fn2, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as z:
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
			print(path)
			print_exc()
		raise

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

	session = requests.Session()

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
		print("DET:", image.size, out)
		return out

	@cp.expose(("fi",))
	def fileinfo(self, *path, **void):
		cp.response.headers.update(SHEADERS)
		assert len(path) in (1, 2) and path[0].count("~") == 0
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		try:
			info = download_cache[url]
		except KeyError:
			resp = self.session.get(
				url,
				headers=Request.header(),
				verify=False,
				timeout=60,
				stream=True,
			)
			resp.raise_for_status()
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
		cp.response.headers.update(CHEADERS)
		assert len(path) in (1, 2) and path[0].count("~") == 0
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		callback = None
		try:
			info = download_cache[url]
		except KeyError:
			resp = self.session.get(
				url,
				headers=Request.header(),
				verify=False,
				timeout=60,
				stream=True,
			)
			resp.raise_for_status()
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
		headers.update(Request.header())
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
		for start, end in ranges:
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
					resp = reqs.next().head(u, headers=headers, timeout=20)
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
					print(u)
					if e >= ns:
						e = ""
					else:
						e -= 1
					h2 = dict(h.items())
					h2["range"] = f"bytes={s}-{e}"
					ex2 = None
					for i in range(3):
						resp = reqs.next().get(u, headers=h2, stream=True, timeout=20)
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
		cp.response.headers.update(HEADERS)
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
				print("INFO:", upload_cache[h])
				data = json_dumps(upload_cache[h].info)
				if len(data) > 262144:
					data2 = bytes2zip(data)
					if len(data2) <= len(data) / 2 or len(data) >= attachment_cache.max_size // 4:
						data = data2
				encoded = leb128(len(data)) + data
				head = upload_cache[h].head
				assert len(encoded) + len(head) <= attachment_cache.max_size
				fut = attachment_cache.create(encoded + head, filename=filename, editable=True)
				url = await_fut(fut)
				cid, mid, aid, fn = expand_attachment(url)
				uhead = shorten_attachment(cid, mid, aid, fn, mode="p") + "?key=" + base64.urlsafe_b64encode(upload_cache[h].key).rstrip(b"=").decode("ascii")
				yield b'"url":"' + uhead.encode("utf-8") + b'"'
			yield b"}"
		return start_upload(position)

	@cp.expose
	def delete(self, *path, key=None):
		assert len(path) in (1, 2) and path[0].count("~") == 0
		assert key, "File Key Required."
		c_id, m_id, a_id, fn = decode_attachment("/".join(path))
		fut = csubmit(attachment_cache.obtain(c_id, m_id, a_id, fn))
		url = await_fut(fut)
		resp = self.session.get(
			url,
			headers=Request.header(),
			verify=False,
			timeout=60,
			stream=True,
		)
		resp.raise_for_status()
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
			resp = self.session.request(
				cp.request.method.upper(),
				url,
				headers=headers,
				stream=True,
				verify=False,
				timeout=60,
			)
			resp.raise_for_status()
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
		resp = self.session.request(
			cp.request.method.upper(),
			url,
			headers=headers,
			data=body,
			stream=True,
			verify=False,
			timeout=60,
		)
		cp.response.status = resp.status_code
		cp.response.headers.update(resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		if is_discord_attachment(url):
			cp.response.headers.pop("Content-Disposition", None)
			cp.response.headers.update(CHEADERS)
		ctype = resp.headers.get("Content-Type", "application/octet-stream")
		if ctype in ("text/html", "text/html; charset=utf-8", "application/octet-stream"):
			it = resp.iter_content(65536)
			b = next(it)
			mime = magic.from_buffer(b)
			if mime == "application/octet-stream":
				a = MemoryBytes(b)[:128]
				if sum(32 <= c < 128 for c in a) >= len(a) * 7 / 8:
					mime = "text/plain"
			cp.response.headers.pop("Content-Type", None)
			cp.response.headers["Content-Type"] = mime
			return resume(b, it)
		return resp.iter_content(65536)

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
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose
	def summarise(self, s, min_length=128, max_length=192):
		v = interface.run(f"STRING.summarise({json.dumps(s)},min_length={min_length},max_length={max_length})", cache=60)
		b = v.encode("utf-8")
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = "text/plain"
		cp.response.headers["Content-Length"] = len(b)
		cp.response.headers["ETag"] = create_etag(b)
		return b

	ecdc_running = {}
	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def encodec(self, url="", name="", source="", thumbnail="", bitrate="auto", inference=False, urls=()):
		cp.response.headers.update(SHEADERS)
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
			cp.response.headers.update(HEADERS)
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
			try:
				os.rename(res, out)
			except OSError:
				with open(res, "rb") as f:
					b = f.read()
				with open(out, "wb") as f:
					f.write(b)
			f = open(out, "rb")
			return cp.lib.static.serve_fileobj(f, content_type="audio/ecdc", disposition="", name=url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0] + ".ecdc")
		finally:
			fut = self.ecdc_running.pop(out, None)
			fut.set_result(None)

	@cp.expose
	def decodec(self, url, fmt="opus"):
		cp.response.headers.update(SHEADERS)
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

	ydl_sems = {}
	@cp.expose
	def ytdl(self, **kwargs):
		cp.response.headers.update(HEADERS)
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
		t = ts_us()
		ip = true_ip()
		if v:
			sem = self.ydl_sems.setdefault(ip, Semaphore(64, 256, rate_limit=8))
			asap = kwargs.get("asap") or not sem.active
			with sem:
				exists = False
				fmt = kwargs.get("fmt")
				if not fmt:
					fmt = "opus" if d else "webm"
				if fmt == "weba":
					fmt = "webm"
				if fmt not in ("mp3", "opus", "webm", "ts", "ogg", "wav"):
					raise TypeError(fmt)
				name = q.rsplit("/", 1)[-1].rsplit(".", 1)[0]
				fmt2 = "." + fmt
				if is_url(q):
					url = q
					url = unyt(url)
					h = shash(url)
					fn = "~" + h + fmt2
					fni = f"{TEMP_PATH}/audio/" + fn
					if os.path.exists(fni) and os.path.getsize(fni) and utc() - os.path.getmtime(fni) >= 3:
						print("Cache hit, skipping...")
						exists = True
				if not exists:
					interface.run(f"bot.audio.returns[{t}]=VOICE.ytdl.search({repr(q)})[0]")
					stream = interface.run(f"str(VOICE.ytdl.get_stream(bot.audio.returns[{t}],force=True,download=False))", timeout=120)
					if fmt in ("ts", "webm", "weba") and is_url(stream):
						raise cp.HTTPRedirect(stream, status="307")
					test = interface.run(f"str(bot.audio.returns[{t}])")
					if not test or test[0] not in ("([{"):
						print("YTDL Invalid:", test)
						raise FileNotFoundError
					name, url = interface.run(f"(bot.audio.returns[{t}].get('name'),bot.audio.returns[{t}].get('url'))")
					if not name or not url:
						raise FileNotFoundError(500, v)
					url = unyt(url)
					h = shash(url)
					fn = "~" + h + fmt2
					fni = f"{TEMP_PATH}/audio/" + fn
					if os.path.exists(fni) and os.path.getsize(fni) and utc() - os.path.getmtime(fni) >= 3:
						print("Cache hit, skipping...")
						exists = True
				if not exists:
					interface.run(f"bot.audio.returns[{t}]=VOICE.ytdl.get_stream(bot.audio.returns[{t}],download={repr(fmt2)},asap={asap},force=1)", timeout=120)

				def af():
					if not os.path.exists(fni):
						return
					if not os.path.getsize(fni):
						return
					if utc() - os.path.getmtime(fni) < 3:
						return
					try:
						res = interface.run(f"bool(bot.audio.returns[{t}].is_finished())")
						# print(t, res, type(res))
					except Exception:
						print_exc()
						return True
					return res is not False

				cp.response.headers.update(CHEADERS)
				if exists or af():
					f = open(fni, "rb")
					interface.run(f"bot.audio.returns.pop({t},None)")
					print("Cache exists, skipping...")
				else:
					if not os.path.exists(fni):
						fni = interface.run(f"bot.audio.returns[{t}].file")
					req = 1073741824 if fmt == "mp3" else 4096
					f = DownloadingFile(fni, af=af, min_buffer=req)
					if d:
						cp.response.status = 202
					cp.response.headers["Content-Type"] = f"audio/{fmt}"
					cp.response.headers["Content-Disposition"] = "attachment; " * bool(d) + "filename=" + json.dumps(name + fmt2)
					cp.response.headers.pop("Accept-Ranges", None)
					print("Cache miss, waiting...")
					return cp.lib.file_generator(f, req)
				# cp.response.headers["Content-Type"] = f"audio/{fmt}"
			return cp.lib.static.serve_fileobj(f, content_type=f"audio/{fmt}", disposition="attachment" if d else "", name=name + fmt2)
		else:
			count = 1 if is_url(q) else kwargs.get("count", 10)
			res = interface.run(f"[VOICE.copy_entry(e) for e in VOICE.ytdl.search({repr(q)},count={count}) if isinstance(e, dict)]")
			if not res:
				res = interface.run(f"VOICE.ytdl.search({repr(q)},count={count})")
				if isinstance(res, str):
					res = evalEX(res)
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return json_dumps(res)
	ytdl._cp_config = {"response.stream": True}

	@cp.expose
	def ytdlp(self, url, fmt="mp4", start="", end=""):
		cp.response.headers.update(HEADERS)
		if not url:
			cp.response.status = 204
			return
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
	def ytdlc(self, *urls, fmt="mp4", multi="true", **kwargs):
		cp.response.headers.update(HEADERS)
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
		fmt = fmt.strip() or "mp4"
		if fmt not in ("mp3", "ogg", "opus", "m4a", "flac", "wav", "wma", "mp2", "weba", "vox", "adpcm", "pcm", "8bit", "mid", "midi", "webm", "mp4", "avi", "mov", "m4v", "mkv", "f4v", "flv", "wmv", "gif", "apng", "webp"):
			raise TypeError
		url = " ".join(urls)
		b = self.command(input=f"concat {m} {url} as {fmt}")
		data = orjson.loads(b)
		url = data[0]["content"]
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	def specexec(self, url, **kwargs):
		cp.response.headers.update(HEADERS)
		argv = " ".join(itertools.chain(*kwargs.items()))
		b = self.command(input=f"spectralpulse {url} {argv}")
		data = orjson.loads(b)
		url = data[0]["content"].replace("/d/", "/f/")
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	def filelist(self, path=None):
		cp.response.headers.update(HEADERS)
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
		cp.response.headers.update(HEADERS)
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
		cp.response.headers.update(CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		cp.response.headers["ETag"] = create_etag(data)
		return data

	@cp.expose(("favicon", "favicon.ico"))
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
	def get_ip(self, *args, **kwargs):
		esubmit(app.get_ip_ex)
		data = json_dumps(dict(
			remote=true_ip(),
			host=T(self).get("ip", "127.0.0.1"),
		))
		cp.response.headers.update(SHEADERS)
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
		cp.response.headers.update(HEADERS)
		raise cp.HTTPRedirect(url, status=307)

	@cp.expose
	def backup(self, token="~"):
		at = AUTH.get("discord_token")
		if token != at:
			if cp.url(base="").strip("/") != at:
				raise InterruptedError
		backup = interface.run("bot.backup()", cache=60)
		cp.response.headers.update(CHEADERS)
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

	try:
		with open("saves/mpdata.json", "rb") as f:
			mpdata = json.load(f)
	except FileNotFoundError:
		mpdata = {}
	except Exception:
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
	except Exception:
		if os.path.exists("saves/mpact\x7f\x7f.json"):
			with open("saves/mpact\x7f\x7f.json", "rb") as f:
				mpact = pickle.load(f)
		else:
			mpact = {}
			print_exc()

	mpresponse = {}

	@cp.expose
	def mphb(self, playing=None):
		mpdata = self.mpdata
		ip = true_ip()
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
		cp.request.no_log = True
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
			esubmit(self.mpdata_update)
		cp.response.headers.update(HEADERS)
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
		cp.response.headers.update(HEADERS)
		return json_dumps(dict(
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
	def status(self, interval=None):
		if not interval:
			cp.request.no_log = True
		status = interface.run(f"bot.status(interval={interval})", cache=3)
		cp.response.headers.update(HEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return json_dumps(status)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	def distribute(self, caps="[]", stat="{}", resp="{}", token="", id=""):
		ip = true_ip()
		at = AUTH.get("discord_token")
		verified = int(at == token)
		if not caps.startswith("["):
			caps = base64.urlsafe_b64decode(caps + "==")
		caps = orjson.loads(caps)
		if not verified:
			caps = list(set(caps).difference(("gptq", "agpt")))
		if not stat.startswith("{"):
			stat = base64.urlsafe_b64decode(stat + "==").decode("utf-8", "replace")
		stat = orjson.loads(stat.replace("<IP>", ip))
		t = utc()
		for k, v in stat.items():
			for i, d in v.items():
				if abs(t - d.get("time", 0)) > 10:
					d["time"] = t - 10
		if not resp:
			resp = cp.request.body.fp.read()
		elif not resp.startswith("{"):
			resp = base64.urlsafe_b64decode(resp + "==")
		resp = literal_eval(resp)
		for k, v in resp.items():
			if isinstance(v, str):
				if v.startswith("ERR:"):
					resp[k] = evalex(v[4:])
				elif v.startswith("RES:"):
					resp[k] = v[4:]
		if not resp:
			cp.request.no_log = True
		caps = json_dumps(caps).decode("ascii")
		stat = json_dumps(stat).decode("utf-8", "replace")
		resp = repr(resp)
		idp = ip + "-" + id
		tasks, shards = interface.run(f"(bot.distribute({caps},{stat},{resp},{repr(idp)}),len(bot.status_data.system['cpu']))")
		data = dict(tasks=tasks, next_delay=max(1, shards - 1))
		cp.response.headers.update(HEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return json_dumps(data)

	def ensure_mpins(self):
		try:
			ins_time = T(self).get("ins_time", 0)
			t = utc()
			if t - ins_time >= 30:
				self.mpimg.clear()
				self.ins_wait = Future()
				k = self.mpact.keys()
				data = [deque() for i in range(len(next(reversed(self.mpact.values()))))]
				values = ()
				for i in range(min(k), max(k) + 1):
					values = self.mpact.get(i) or values
					for j, v in enumerate(values):
						data[j].append(v)
				self.ins_data = data
				self.ins_time = t
				self.ins_wait.set_result(None)
				self.ins_wait = None
		except Exception:
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
				with open("saves/mpdata\x7f.json", "rb") as f:
					if f.read(1) in (b"\x00", b" ", b""):
						raise ValueError
				if os.path.exists("saves/mpdata\x7f\x7f.json"):
					os.remove("saves/mpdata\x7f\x7f.json")
				os.rename("saves/mpdata.json", "saves/mpdata\x7f\x7f.json")
				os.rename("saves/mpdata\x7f.json", "saves/mpdata.json")
		except Exception:
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
				with open("saves/mpact\x7f.json", "rb") as f:
					if f.read(1) in (b"\x00", b" ", b""):
						raise ValueError
				if os.path.exists("saves/mpact\x7f\x7f.json"):
					os.remove("saves/mpact\x7f\x7f.json")
				os.rename("saves/mpact.json", "saves/mpact\x7f\x7f.json")
				os.rename("saves/mpact\x7f.json", "saves/mpact.json")
		except Exception:
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
			except Exception:
				print_exc()
			time.sleep(60)

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
			_data = get_geo(ip)
		else:
			ip = ".".join(str(xrand(1, 255)) for _ in loop(4))
			self.rapidapi += 1
		if " " not in content:
			content += " "
		j, after = interface.run(f"bot.run_simulate({repr(content)})")
		a = after - utc()
		if a > 0:
			cp.response.headers["Retry-After"] = a
		cp.response.headers.update(HEADERS)
		return json_dumps(j)

	@cp.expose(("cat", "cats", "dog", "dogs", "neko", "nekos", "giphy"))
	def imagepool(self, tag="", refresh=60):
		name = cp.url(base="").rsplit("/", 1)[-1]
		command = name.rstrip("s")
		argv = tag
		try:
			args = smart_split(argv)
		except ValueError:
			args = argv.split()
		url = interface.run(f"bot.commands.{command}[0](bot=bot,channel=None,flags='v',args={repr(args)},argv={repr(argv)})")
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

	class V1Cache:
		def __init__(self, end=2048, soft=8192, hard=30720):
			self.end = end
			self.soft = soft
			self.hard = hard
			self.data = alist()
			self.cache = set()
		def pad_into(self, tokens, max_tokens=0, padding=(), sentinel=()):
			if not tokens:
				return tokens
			t = utc()
			tokens = astype(tokens, tuple)
			if tokens in self.cache:
				return tokens
			max_context = min(len(tokens) * 2, self.hard - max_tokens)
			try:
				if len(tokens) <= self.soft:
					# Within soft limit; directly forwarded
					return tokens
				target = None
				high = 0
				for t, v in reversed(self.data):
					i = longest_prefix(tokens, v)
					if i > high:
						high = i
						target = v
				out = []
				if high:
					if len(tokens) - high <= self.soft:
						# Sufficiently matched; directly forwarded
						return tokens
					if sentinel:
						try:
							sublist_index(tokens[:high], sentinel)
						except ValueError:
							tokens = tokens[high:]
							left = self.soft - self.end
							tokens = target[:high] + tokens[:left] + padding + tokens[-self.end:]
							print("Sentinel mismatch:", sentinel, len(tokens))
							return tokens
					# Match existing cached string; matched part must be `>= input / 2 and >= 8k`, unmatched part must be `<= 8k`
					out.append(target[:high])
					tokens, target = tokens[high:], target[high:]
					sub = longest_common_substring(tokens, target)
					if high + sub >= len(tokens) / 2 and high + sub >= self.soft:
						i2 = sublist_index(tokens, sub)
						if i2 <= self.soft:
							i3 = sublist_index(target, sub)
							if i3 <= self.soft:
								right = tokens[i2 + len(sub):]
								out.append(target[:i3])
								out.append(sub)
								if len(right) <= self.soft:
									out.append(right)
								else:
									left = self.soft - self.end
									out.append(right[:left])
									out.append(padding)
									out.append(right[-self.end:])
								res = tuple(itertools.chain(*out))
								overflow = len(res) - max_context
								if overflow > 0:
									tokens = res[:-self.soft] + padding + res[-self.end + overflow:]
								else:
									tokens = res
								return tokens
					left = self.soft - self.end
					out.append(tokens[:left])
					out.append(padding)
					out.append(tokens[-self.end:])
					tokens = tuple(itertools.chain(*out))
					return tokens
				# Full cache miss; use `first 6k + padding + last 2k` tokens
				left = self.soft - self.end
				tokens = tokens[:left] + padding + tokens[-self.end:]
				return tokens
			finally:
				if tokens in self.cache:
					self.data.remove(tokens, key=lambda t: t[1])
				else:
					self.cache.add(tokens)
				self.data.append(([t, tokens]))
				while sum(len(t[1]) for t in self.data) + max_tokens > self.hard + self.soft:
					temp = self.data.popleft()
					self.cache.discard(temp[1])
	v1_cache = V1Cache()
	temp_model = "command-r-plus"
	token_bans = set()#{153083, 165936, 182443, 205177, 253893, 255999}
	@cp.expose
	def inference(self, version=None, *path, **kwargs):
		if version != "v1":
			raise NotImplementedError(version)
		model = self.temp_model
		fmt = "cl100k_im"
		if not path:
			path = ["models"]
		endpoint = "/".join(path)
		url = "http://127.0.0.1:2242/v1/" + endpoint
		headers = cp.request.headers
		headers["X-Forwarded-For"] = true_ip()
		data = cp.request.body.fp.read()
		if data and endpoint.casefold() == "completions":
			try:
				d = orjson.loads(data)
			except Exception:
				pass
			else:
				if AUTH.get("mizabot_key") and headers.get("Authorization") == f"Bearer {AUTH['mizabot_key']}":
					print("Authorised LLM")
				else:
					prompt = d.get("prompt")
					if prompt and len(prompt) > 8192:
						tokens = tik_encode(prompt, encoding=fmt)
						padding = tuple(tik_encode("...\n\n...", encoding=fmt))
						sentinel = tuple(tik_encode("<START>", encoding=fmt))
						print(len(tokens))
						tokens = self.v1_cache.pad_into(tokens, max_tokens=d.get("max_tokens", 256), padding=padding, sentinel=sentinel)
						print(len(tokens))
						prompt = tik_decode(tokens, encoding=fmt)
						d["prompt"] = prompt
				d["model"] = model
				if self.token_bans:
					d["custom_token_bans"] = set(d.get("custom_token_bans", [])).union(self.token_bans)
				data = json_dumps(d)
		method = cp.request.method
		resp = getattr(reqs.next(), method.casefold())(
			url,
			headers=headers,
			data=data,
			stream=True,
		)
		headers = fcdict(resp.headers)
		headers.update(HEADERS)
		headers.pop("Connection", None)
		headers.pop("Content-Length", None)
		headers.pop("Transfer-Encoding", None)
		for k, v in headers.items():
			cp.response.headers[k] = v
		print(resp, cp.response.headers)
		if resp.headers.get("Transfer-Encoding") == "chunked":
			def gen(resp):
				try:
					for line in resp.iter_lines():
						if cp.request.closed:
							break
						yield line + b"\n"
				except (StopIteration, GeneratorExit):
					pass
				resp.close()
			return CloseableAsyncIterator(gen(resp), resp.close)
		if astype(path, list) == ["models"]:
			data = resp.json()
			self.temp_model = model = data["data"][0]["id"]
			data["data"][0]["id"] = f"{model}-h6t2"
			return json_dumps(data)
		return resp.content
	inference._cp_config = {"response.stream": True}


def ensure_parent(proc, parent):
	while True:
		if not is_strict_running(parent):
			force_kill(proc)
			break
		time.sleep(12)

if __name__ == "__main__":
	# logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
	pid = os.getpid()
	ppid = os.getppid()
	print(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
	proc = psutil.Process(pid)
	parent = psutil.Process(ppid)
	tsubmit(ensure_parent, proc, parent)
	app = Server()
	self = server = cp.Application(app, "/", config)
	tsubmit(app.mp_activity)
	esubmit(app.get_ip_ex)
	interface.start()
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
