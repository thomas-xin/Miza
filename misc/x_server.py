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
import weakref
import zipfile
import cheroot
import cherrypy
import numpy as np
import orjson
import psutil
import requests
from collections import deque
from concurrent.futures import Future
from math import inf, ceil
from traceback import print_exc
from cheroot import errors
from cherrypy._cpdispatch import Dispatcher
from .asyncs import Semaphore, SemaphoreOverflowError, eloop, newfut, esubmit, tsubmit, csubmit, await_fut, CloseableAsyncIterator
from .smath import supersample, xrand
from .types import as_str, astype, cdict, suppress, round_min, lim_str, full_prune, literal_eval, regexp, loop, json_dumps, alist, resume
from .util import hwaccel, fcdict, ihash, nhash, shash, EvalPipe, AUTH, TEMP_PATH, reqs, MIMES, tracebacksuppressor, is_strict_running, force_kill, utc, ts_us, is_url, p2n, n2p, find_file, get_mime, ecdc_dir, url_parse, url_unparse, url2fn, smart_split, seq, proxy, Request, magic, is_discord_attachment, discord_expired, unyt, ecdc_exists, get_duration, evalex, evalEX, DownloadingFile, T, tik_encode, tik_decode, longest_prefix, longest_common_substring, sublist_index, byte_scale, decode_attachment, attachment_cache


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
def hostmap(func):
	return func

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

est_time = utc()
est_last = -inf

@tracebacksuppressor
def estimate_life():
	return inf
	global est_time, est_last
	hosted = sorted(int(f[1:].split("~", 1)[0]) / 1e6 for f in os.listdir(f"{TEMP_PATH}/filehost") if f.startswith(IND))
	if not hosted:
		est_last = -inf
		ts = 0
	else:
		ts = hosted[0]
	res = interface.run("bot.storage_ratio", cache=60)
	try:
		last = (utc() - ts) / res
	except ZeroDivisionError:
		last = inf
	print(last)
	est_time = utc() - last
	est_last = utc()

# estimate_life_after = lambda t: time.sleep(t) or estimate_life()

# esubmit(estimate_life_after, 10)

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

	serving = {}
	def _fileinfo(self, path, **void):
		# orig_path = path
		ind = IND
		if path.startswith("!"):
			ind = "!"
			path = path[1:]
		elif not path.startswith("@"):
			b = path.lstrip("~").split(".", 1)[0]
			path = str(p2n(b))
		else:
			path = path[1:]
		p = find_file(path, cwd=("saves/filehost", f"{TEMP_PATH}/filehost"), ind=ind)
		mime = get_mime(p)
		f_url = API + "/f/" + n2p(int(path))
		st = os.stat(p)
		fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
		if fn.startswith(".temp$@"):
			fn = fn[7:]
			a3 = True
		else:
			try:
				_p2 = find_file(path, cwd=f"{TEMP_PATH}/filehost", ind=ind)
			except FileNotFoundError:
				a3 = False
			else:
				a3 = True
		# t = utc()
		# ti = max(st.st_atime + 30 * 86400, st.st_ctime + 60 * 86400, t)
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
					url, _code, ftype = orjson.loads(s)
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

	lfc = {}
	@cp.expose(("fi",))
	@hostmap
	def fileinfo(self, path, **kwargs):
		cp.response.headers.update(SHEADERS)
		d = self._fileinfo(path)
		cp.response.headers["Content-Type"] = "application/json"
		if d.get("chunks") and ("Cf-Worker" in cp.request.headers or kwargs.get("cf")):
			if path in self.lfc and utc() - self.lfc[path].get("t", 0) <= 80000:
				d["chunks"] = self.lfc[path]["chunks"]
			else:
				chunkt = []
				o_urls = []
				chunkset = []
				chunkmap = {}
				for url in d["chunks"]:
					if not url.startswith(API) or "/u/" not in url or "?S=" not in url:
						chunkt.append(url)
						continue
					id = url.rsplit("/", 1)[-1].split("?", 1)[0]
					with tracebacksuppressor:
						id = int.from_bytes(base64.urlsafe_b64decode(id + "=="), "big")
					chunkset.append(id)
					chunkt.append(id)
					o_urls.append(url)
				if chunkset:
					a_urls = interface.run(f"bot.renew_attachments({chunkset})", cache=60)
					print("CHUNKSET:", len(chunkset), len(a_urls))
					for id, aurl, url in zip(chunkset, a_urls, o_urls):
						if int(url.rsplit("?S=", 1)[-1]) <= 25165824:
							purl = "https://proxy.mizabot.xyz/proxy?S=" + url.rsplit("?S=", 1)[-1] + "&url=" + urllib.parse.quote_plus(aurl)
						elif "?S=" not in aurl:
							purl = aurl + "?S=" + url.rsplit("?S=", 1)[-1]
						else:
							purl = aurl
						chunkmap[id] = purl
				d["chunks"] = chunks = [chunkmap.get(u, u) for u in chunkt]
				self.lfc[path] = cdict(t=utc(), chunks=chunks)
				print(ts_us(), chunks)
		return json_dumps(d)

	image_loaders = {}

	@cp.expose(("0",))
	@hostmap
	def rickroll(self, *void1, **void2):
		raise cp.HTTPRedirect(rickroll, status=301)

	opipe = None
	@cp.expose
	@hostmap
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

	@cp.expose(("images", "image", "i", "view", "v", "raw", "r", "n", "f", "d"))
	@hostmap
	def download(self, path, filename=None, download=None, **void):
		if path in ("hacks", "mods", "files", "download", "static"):
			print(true_ip() + " was rickrolled 🙃")
			return self.rickroll()
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
					url = HOST + "/i/" + c.rstrip(b"=").decode("ascii", "replace") + ".gif"
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
			p = find_file(path, cwd=(f"{TEMP_PATH}/filehost", "saves/filehost"), ind=ind)
		sem = SEMAPHORES.get(p)
		if not sem:
			while len(SEMAPHORES) >= 4096:
				sem = SEMAPHORES.pop(next(iter(SEMAPHORES)))
				if sem.is_busy():
					raise SemaphoreOverflowError
			sem = SEMAPHORES[p] = Semaphore(256, 256, rate_limit=4)
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
				if not st.st_size or st.st_size > 48 * 1048576 or st.st_size == 147408:
					try:
						p2 = find_file(path, cwd=("saves/filehost"), ind=ind)
					except FileNotFoundError:
						p2 = None
					if p2:
						with tracebacksuppressor:
							os.remove(p)
						p = p2
						mime = get_mime(p)
						st = os.stat(p)
						a3 = False
			else:
				a3 = False
			cp.response.headers["Attachment-Filename"] = a2
			proc = None
			if endpoint.startswith("r") and mime.split("/", 1)[0] in ("video", "audio"):
				preview = f"{TEMP_PATH}/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".webm"
				image_loaders = self.image_loaders
				if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
					args = ("./ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-hwaccel", hwaccel, "-f", "lavfi", "-an", "-i", "color=size=640x96:rate=1:color=#bf7fff", "-vn", "-i", p, "-pix_fmt", "yuv420p", "-f", "webm", "-crf", "50", "-c:a", "libopus", "-b:a", "96k", "-shortest", preview)
					print(args)
					proc = psutil.Popen(args)
					image_loaders[preview] = proc
				else:
					proc = preview
				cp.response.headers["Content-Type"] = "video/webm"
			elif endpoint.startswith("i") and mime.split("/", 1)[0] in ("image", "video"):
				preview = f"{TEMP_PATH}/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".png"
				image_loaders = self.image_loaders
				if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
					args = ("./ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-hwaccel", hwaccel, "-an", "-i", p, "-loop", "0", "-fs", "1048576", "-vf", "scale=240:-1", "-vframes", "1", preview)
					print(args)
					proc = psutil.Popen(args)
					image_loaders[preview] = proc
				else:
					proc = preview
				cp.response.headers["Content-Type"] = "image/png"
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
							resp = proxy.get(url, headers=headers, stream=True, timeout=60)
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
							print(s)
							print(headers)
							print(cp.response.headers)
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
							cp.response.headers["Content-Type"] = mime = info[2]
							# referrer = cp.request.headers.get("Referer")
							# print(p, len(urls), referrer)
							cp.response.headers["Attachment-Filename"] = info[0]
							if cp.request.method == "HEAD":
								cp.response.headers["Content-Length"] = info[1]
								return
							urls = [remap_url(url) for url in urls]
							# print(urls)
							if len(urls) == 1:
								url = urls[0]
								if not is_url(url):
									p = find_file(url, cwd=(f"{TEMP_PATH}/filehost", "saves/filehost"), ind=ind)
									fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
									ax = filename or fn
									a2 = url_unparse(ax)
									if a2.startswith(".temp$@"):
										a2 = a2[7:]
										a3 = True
										if not st.st_size or st.st_size > 48 * 1048576 or st.st_size == 147408:
											with tracebacksuppressor:
												os.remove(p)
											p = find_file(path, cwd=("saves/filehost"), ind=ind)
											mime = get_mime(p)
											st = os.stat(p)
											a3 = False
									urls = self._fileinfo(f"@{url}").get("chunks", ())
									url = urls[0]
								if mime.split("/", 1)[-1] not in ("html", "txt") and (url.startswith(HOST) or url.startswith(API)):
									u = url.split("://", 1)[-1].split("/", 2)[-1]
									print("U:", u)
									if "?S=" in u:
										u, s = u.split("?S=", 1)
									# else:
									# 	s = inf
									# if float(s) < 1048576:
									return self.unproxy(u)
									# raise cp.HTTPRedirect(url, status="307")
								if download and ("Cf-Worker" not in cp.request.headers or not is_discord_attachment(url)):
									raise cp.HTTPRedirect(url, status="307")
							cp.response.headers.pop("Accept-Ranges", None)
							stn = p.rsplit("~.forward$", 1)[0].replace("saves/filehost/", f"{TEMP_PATH}/filehost/")
							pn = stn + "~.temp$@" + info[0]
							dl = HOST + "/d/" + orig_path
							print("END:", endpoint, mime, dl)
							if endpoint.startswith("r") and mime.split("/", 1)[0] in ("video", "audio"):
								preview = f"{TEMP_PATH}/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".webm"
								image_loaders = self.image_loaders
								if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
									args = ("./ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-hwaccel", hwaccel, "-f", "lavfi", "-an", "-i", "color=size=640x96:rate=1:color=#bf7fff", "-vn", "-i", dl, "-pix_fmt", "yuv420p", "-f", "webm", "-crf", "50", "-c:a", "libopus", "-b:a", "96k", "-shortest", preview)
									print(args)
									proc = psutil.Popen(args)
									image_loaders[preview] = proc
								else:
									proc = preview
								cp.response.headers["Content-Type"] = "video/webm"
							elif endpoint.startswith("i") and mime.split("/", 1)[0] in ("image", "video"):
								preview = f"{TEMP_PATH}/%" + p.rsplit("/", 1)[-1].split(".", 1)[0] + ".png"
								image_loaders = self.image_loaders
								if (not os.path.exists(preview) or not os.path.getsize(preview)) and preview not in image_loaders:
									args = ("./ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-hwaccel", hwaccel, "-an", "-to", "1", "-i", dl, "-loop", "0", "-fs", "1048576", "-vf", "scale=240:-1", "-vframes", "1", preview)
									print(args)
									proc = psutil.Popen(args)
									image_loaders[preview] = proc
								else:
									proc = preview
								cp.response.headers["Content-Type"] = "image/png"
							if not proc:
								if os.path.exists(pn):
									f = open(pn, "rb")
									resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=info[0])
									if a3:
										self.serving.setdefault(p, weakref.WeakSet()).add(f)
									return resp
								if info[1] > 48 * 1048576:
									if endpoint != "n":
										raise cp.HTTPRedirect(f"https://stream.miza-stream.workers.dev/?i={orig_path}", status=307)
									return self.dyn_serve(urls, size=info[1])
								return self.concat(p, urls, name=info[0], mime=info[2], stn=stn)
			if proc:
				cp.response.headers["ETag"] = create_etag(p)
				while preview in image_loaders and (not os.path.exists(preview) or os.path.getsize(preview) < 4096) and is_strict_running(image_loaders[preview]):
					time.sleep(0.05)
				f = None
				if preview in image_loaders and not is_strict_running(image_loaders[preview]) or preview not in image_loaders and os.path.exists(preview):
					cp.response.headers["Content-Length"] = os.path.getsize(preview)
				elif preview in image_loaders:
					f = DownloadingFile(
						preview,
						af=lambda: not is_strict_running(image_loaders[preview]),
					)
					print("PREVIEW:", f)
				s = inf
				if not f:
					s = os.path.getsize(preview)
					if s:
						f = open(preview, "rb")
					else:
						cp.response.headers["Content-Type"] = get_mime(p)
						f = open(p, "rb")
						s = os.path.getsize(p)
				if s < 1048576:
					return f.read()
				if s < 67108864:
					f = io.BytesIO(f.read())
				return cp.lib.file_generator(f, 262144)
			f = open(p, "rb")
			s = os.path.getsize(p)
			if s < 67108864:
				cp.response.headers["Content-Type"] = mime
				cp.response.headers["Content-Disposition"] = "attachment; " * bool(download) + "filename=" + json.dumps(a2)
				if s < 1048576:
					return f.read()
				f = io.BytesIO(f.read())
				return cp.lib.static._serve_fileobj(f, content_type=mime, content_length=s)
			resp = cp.lib.static.serve_fileobj(f, content_type=mime, disposition="attachment" if download else None, name=a2)
			if a3:
				self.serving.setdefault(p, weakref.WeakSet()).add(f)
			return resp
	download._cp_config = {"response.stream": True}

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
				return esubmit(self._concat, urls, on, pn)
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
		if resp.headers.get("Content-Type", "application/octet-stream") == "application/octet-stream":
			it = resp.iter_content(65536)
			b = next(it)
			mime = magic.from_buffer(b)
			cp.response.headers.pop("Content-Type", None)
			cp.response.headers["Content-Type"] = mime
			return resume(b, it)
		return resp.iter_content(65536)

	@tracebacksuppressor
	def renew_url(self, url, mid=None):
		if discord_expired(url):
			if not mid and "&mid=" in url:
				mid = int(url.split("&mid=", 1)[-1].split("&", 1)[0])
			if mid:
				return interface.run(f"bot.renew_attachment({repr(url)},m_id={mid})", timeout=60, cache=21600)
			return interface.run(f"bot.renew_attachment({repr(url)})", timeout=60, cache=21600)
		return url

	def dstream(self, url, fn=None):
		headers = fcdict(cp.request.headers)
		headers.pop("Remote-Addr", None)
		headers.pop("Host", None)
		r = headers.pop("Range", None)
		headers.update(Request.header())
		if fn:
			f = open(fn + "~", "wb")
		with reqs.next().get(url, headers=headers, stream=True, timeout=30) as resp:
			resp.raise_for_status()
			size = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 1))
			if r:
				start, end = r.removeprefix("bytes=").split("-", 1)
				if not start:
					if not end:
						return
					start = size - int(end)
					end = size - 1
				elif not end:
					end = size - 1
				start, end = int(start), min(int(end) + 1, size)
			else:
				start = 0
				end = size
			rr = f"bytes {start}-{end - 1}/{size}"
			print(r, rr)
			cp.response.headers["Range"] = rr
			cp.response.headers["Content-Length"] = end - start + 1
			cp.response.headers["x-goog-stored-content-length"] = size
			it = resp.iter_content(65536)
			pos = 0
			try:
				while True:
					b = ob = next(it)
					if pos + len(ob) > end:
						if pos >= end:
							b = b""
						else:
							b = b[:end - pos]
					pos += len(ob)
					if start:
						b2 = b[start:]
						start -= len(b) - len(b2)
						b = b2
					if b:
						yield b
					if fn:
						f.write(ob)
			except StopIteration:
				pass
		if fn:
			f.close()
			with tracebacksuppressor:
				if os.path.exists(fn):
					return
					# os.remove(fn)
				os.rename(fn + "~", fn)

	@tracebacksuppressor
	def dyn_serve(self, urls, size=0):
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
		return self._dyn_serve(urls, ranges, headers)

	@tracebacksuppressor(GeneratorExit)
	def _dyn_serve(self, urls, ranges, headers):
		# print(headers, ranges)
		for start, end in ranges:
			pos = 0
			rems = urls.copy()
			futs = []
			big = False
			while rems:
				uo = rems.pop(0)
				u = self.renew_url(uo)
				# print(uo, u)
				if "?size=" in u or "&size=" in u:
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
					if e >= ns:
						e = ""
					else:
						e -= 1
					h2 = dict(h.items())
					h2["range"] = f"bytes={s}-{e}"
					# print(u, r)
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

	def _peek(self, urls, on, pn, name, download, mime):
		headers = fcdict(cp.request.headers)
		headers.pop("Remote-Addr", None)
		headers.pop("Host", None)
		headers.pop("Range", None)
		headers.update(Request.header())
		urls[0] = self.renew_url(urls[0])
		with reqs.next().get(urls[0], headers=headers, stream=True, timeout=30) as resp:
			resp.raise_for_status()
			b = []
			it = resp.iter_content(65536)
			with suppress(StopIteration):
				while sum(map(len, b)) < 25165824:
					temp = next(it)
					b.append(temp)
					yield temp
			b = b"".join(b)
			print("PreCat", urls[0], resp, len(b))
		fut = esubmit(self._concat, urls, on, pn)
		self.serving[on] = fut
		yield from self.wconcat(on, pn, name, download, mime, fut, start=len(b))

	@tracebacksuppressor
	def _concat(self, urls, on, pn):
		print("Cat", urls)
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
				url = self.renew_url(url)
				if len(futs) >= 16 or futs and futs[0].done():
					fut = futs.pop(0)
					fut.result()
					buf += fut.buf
					self.serving[on + "~buffer"] = buf
				for i in range(16):
					try:
						resp = reqs.next().get(url, headers=headers, stream=True, timeout=30)
						if resp.status_code in (403, 404):
							raise FileNotFoundError
						resp.raise_for_status()
						break
					except FileNotFoundError:
						break
					except Exception:
						print_exc()
					time.sleep(i ** 2 + 1)
				bsize = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
				fs = pos + bsize
				f.truncate(fs)
				fut = esubmit(self.chunk_into, resp, on, pos)
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
			esubmit(self.rename_after, on, pn)

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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
	def specexec(self, url, **kwargs):
		cp.response.headers.update(HEADERS)
		argv = " ".join(itertools.chain(*kwargs.items()))
		b = self.command(input=f"spectralpulse {url} {argv}")
		data = orjson.loads(b)
		url = data[0]["content"].replace("/d/", "/f/")
		raise cp.HTTPRedirect(url, status="307")

	@cp.expose
	@hostmap
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
	@hostmap
	def teapot(self, *args, **kwargs):
		cp.response.headers.update(HEADERS)
		raise IsADirectoryError("I'm a teapot.")

	@cp.expose(("index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "user", "login", "logout", "mpinsights", "createredirect"))
	@hostmap
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
	@hostmap
	def error(self, code=400):
		raise ConnectionError(int(code))

	@cp.expose(("upload_single",))
	@hostmap
	def upload_chunk(self, **kwargs):
		name = cp.request.headers.get("x-file-name", "untitled")
		cp.response.headers.update(HEADERS)
		s = true_ip() + "%" + name
		h = ihash(s) % 2 ** 48
		single = "/upload_single" in cp.url()
		xi = int(cp.request.headers.get("x-index", 0))
		mfs = int(cp.request.headers.get("x-total", 0))
		if mfs > 1125899906842624:
			raise OverflowError
		if not xi:
			print(s)
		n = f"{TEMP_PATH}/{h}%"
		fn = n + str(xi)
		print(fn)
		csize = 83886080
		with open(fn, "wb") as f:
			if single:
				pos = 0
				g = cp.request.body.fp
				urls = []
				while True:
					b = g.read(csize)
					if not b:
						break
					f.write(b)
					if f.tell() > pos + csize:
						url1 = interface.run(f"bot.data.exec.stash({repr(fn)}, start={pos}, end={pos + csize})")
						urls.extend(url1)
						pos += csize
				if f.tell() > pos:
					url1 = interface.run(f"bot.data.exec.stash({repr(fn)}, start={pos})")
					urls.extend(url1)
				size = os.path.getsize(fn)
				mime = get_mime(fn)
				fn = f"{TEMP_PATH}/{h}%!"
				urls = [map_url(url) for url in urls]
				print(urls)
				assert urls
				code = 307
				ftype = 3
				url = ""
				n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
				key = n2p(n)
				s = f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta/><!--["{name}","{size}","{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--><!--KEY={key}--></html>'
				with open(fn, "w", encoding="utf-8") as f:
					f.write(s)
				return self.merge(name=name, index=1)
			shutil.copyfileobj(cp.request.body.fp, f, 262144 if mfs >= 1073741824 else 65536)

	chunk_file = "misc/chunking.json"
	try:
		if not os.path.exists(chunk_file):
			chunking = {}
		else:
			with open(chunk_file, "rb") as f:
				b = f.read()
			chunking = orjson.loads(b)
	except Exception:
		print_exc()
		chunking = {}
	merge_sem = Semaphore(1, inf)

	@tracebacksuppressor
	def update_merge(self):
		d = self.chunking.copy()
		available = set()
		for k, v in tuple(d.items()):
			if isinstance(v, Future):
				d.pop(k)
			if k[-1] == "%":
				if utc() - v.setdefault("timestamp", utc()) > 86400 * 7:
					d.pop(k)
				else:
					available.add(k)
		for k in tuple(d):
			if k[-1] != "%" and k.split("%", 1)[0] + "%" not in available:
				d.pop(k)
		with self.merge_sem:
			b = json_dumps(d)
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
		cp.response.headers.update(HEADERS)
		kwargs = fcdict(kwargs)
		key = kwargs.get("?key")
		ts = int(kwargs.get("?ts") or time.time_ns() // 1000)
		x_name = kwargs.get("x-file-name") or cp.request.headers.get("x-file-name", "untitled")
		name = kwargs.get("name") or x_name
		s = true_ip() + "%" + x_name
		print(s)
		# mfs = int(kwargs.get("x-total") or cp.request.headers.get("x-total", 0))
		h = ihash(s) % 2 ** 48
		nh = n = f"{TEMP_PATH}/{h}%"
		if self.merged.get(nh):
			return
		self.merged[nh] = True
		q = ""
		try:
			fn = f"saves/filehost/{IND}{ts}~" + name
			r = n + "!"
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
					fn = f"{TEMP_PATH}/filehost/{IND}{ts}" + "~.temp$@" + name
					if os.path.exists(fn):
						os.remove(fn)
					os.rename(n + "0", fn)
			else:
				of = nh + "0"
				fn = f"{TEMP_PATH}/filehost/{IND}{ts}~.temp$@{name}"
				if os.path.exists(fn):
					os.remove(fn)
				os.rename(of, fn)
				if high > 1:
					with open(fn, "ab") as f:
						for i in range(1, high):
							gn = nh + str(i)
							with open(gn, "rb") as g:
								shutil.copyfileobj(g, f, 65536)
				if not key:
					n = (ts_us() * random.randint(1, time.time_ns() % 65536) ^ random.randint(0, 1 << 63)) & (1 << 64) - 1
					key = n2p(n)
				q = f"?key={key}"
				if os.path.getsize(fn) < 256 * 1048576:
					self.register_replacer(ts, key)
		finally:
			self.merged.pop(nh, None)
		self.update_merge()
		return "/p/" + n2p(ts) + q

	@cp.expose
	@hostmap
	def upload_url(self, **kwargs):
		ts = time.time_ns() // 1000
		url = kwargs["url"]
		fn = f"{TEMP_PATH}/filehost/{IND}{ts}~.temp$@" + url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
		subprocess.run([sys.executable, "downloader.py", "-threads", "3", url, fn], cwd="misc")
		# b = ts.bit_length() + 7 >> 3
		ts, _key = self.register_replacer(ts)
		cp.response.headers.update(HEADERS)
		return HOST + "/p/" + n2p(ts)

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
		except Exception:
			pass
		else:
			return
		# if os.path.exists(fn) and not urls:
		# 	return url + f"?key={key}"
		ha1 = ha2 = self.hash_file(of)
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
					# mids = []
					ha1 = ha2 = ""
					break
		size = os.path.getsize(of)
		mime = get_mime(of)
		if not urls:
			if size > 256 * 1048576:
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
			urls = interface.run(f"bot.data.exec.stash({repr(of)})")
			if not self.in_replacer(ts, key):
				interface.run(f"bot.data.exec.delete({repr(urls)})")
				return
			print(urls)
			assert urls
		else:
			urls = []
		urls = [map_url(url) for url in urls]
		code = 307
		ftype = 3
		jdn = json.dumps(name).replace("<", '"\u003c"').replace(">", '"\u003e"')
		s = (
			f'<!DOCTYPE HTML><!--["{url}",{code},{ftype}]--><html><meta http-equiv="refresh" content="0;URL={url}"/>'
			+ f'<!--[{jdn},{size},"{mime}"]--><!--URL={json.dumps(urls, separators=(",", ":"))}--><!--KEY={key}-->'
			+ (f'<!--SHA={ha1}-->' if ha1 else "")
			+ '</html>'
		)
		pn = True
		with suppress(FileNotFoundError):
			while pn:
				pn = find_file(ts, cwd="saves/filehost")
				os.remove(pn)
		with open(fn, "w", encoding="utf-8") as f:
			f.write(s)
		on = f"{TEMP_PATH}/filehost/{IND}{ts}~.temp$@" + name
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
				infd, urld, key, _midd, _ = spl[:5]
			elif len(spl) >= 3:
				infd, urld, _ = spl[:3]
				key = ""
				# midd = "[]"
			else:
				continue
			info = orjson.loads(infd.removeprefix("<!--"))
			urls = orjson.loads(urld.removeprefix("<!--").removeprefix("URL="))
			while key.startswith("<!--KEY="):
				key = key[8:]
			urls = [remap_url(url) for url in urls]
			if not is_url(urls[0]):
				continue
			stn = p.rsplit("~.forward$", 1)[0].replace("saves/filehost/", f"{TEMP_PATH}/")
			pn = stn + "~.temp$@" + info[0]
			self.concat(pn, urls, name=info[0], mime=info[2], stn=stn, waiter=True).result()
			if fp.count("$") >= 4:
				continue
			self.replace_file(pn, name=info[0], key=key, urls=urls)

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
		self.replace_fut = esubmit(self.update_replacers)

	def update_replacers(self):
		if not os.path.exists("saves/filehost/-1.txt") or not os.path.getsize("saves/filehost/-1.txt"):
			return
		with self.replacer_sem:
			with open("saves/filehost/-1.txt", "r", encoding="ascii") as f:
				lines = f.readlines()
		for line in set(lines):
			ts, key = line.rstrip().split(":", 1)
			try:
				fn = find_file(ts, cwd=TEMP_PATH, ind=IND)
				self.replace_file(fn, key)
			except Exception:
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
		cp.response.headers.update(HEADERS)
		ots = p2n(path)
		path = str(ots)
		p = find_file(path, cwd=(f"{TEMP_PATH}/filehost", "saves/filehost"))
		replaceable = self.remove_replacer(ots, key)
		if not replaceable:
			if p.split("~", 1)[-1].startswith(".temp$@"):
				if p in self.serving:
					for f in self.serving.pop(p):
						f.close()
					time.sleep(0.5)
				if os.path.exists(p):
					try:
						os.remove(p)
					except Exception:
						print_exc()
						return f"{HOST}/error/500"
				p = find_file(path)
			if not p.split("~", 1)[-1].startswith(".forward$"):
				raise TypeError("File is not editable.")
			with open(p, "r", encoding="utf-8") as f:
				orig = f.read()
			if key != orig.split("<!--KEY=", 1)[-1].split("-->", 1)[0]:
				raise PermissionError("Incorrect key.")
			self.delete_link(p, orig)
		else:
			if p.split("~", 1)[-1].startswith(".temp$@"):
				if p in self.serving:
					for f in self.serving.pop(p):
						f.close()
					time.sleep(0.2)
				with tracebacksuppressor:
					os.remove(p)
			else:
				os.remove(p)
		if self.edited.get(ots):
			return
		self.edited[ots] = True
		try:
			if not replaceable:
				urls = orjson.loads(orig.split("<!--URL=", 1)[-1].split("-->", 1)[0])
				interface.run(f"bot.data.exec.delete({repr(urls)})")
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
		_infd, urld, _k, _midd, ext  = s.split("-->", 4)
		# info = orjson.loads(infd.removeprefix("<!--"))
		urls = orjson.loads(urld.removeprefix("<!--URL="))
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
								i = sn.index("<!--SHA=") + len("<!--SHA=")
								s = sn[i:].split("-->", 1)[-1]
								f.seek(i)
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
		p = find_file(path, cwd=(f"{TEMP_PATH}/filehost", "saves/filehost"))
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
			urls = orjson.loads(orig.split("<!--URL=", 1)[-1].split("-->", 1)[0])
			interface.run(f"bot.data.exec.delete({repr(urls)})")
		cp.response.headers.update(HEADERS)
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
	def status(self, interval=None):
		if not interval:
			cp.request.no_log = True
		status = interface.run(f"bot.status(interval={interval})", cache=3)
		cp.response.headers.update(HEADERS)
		cp.response.headers["Content-Type"] = "application/json"
		return json_dumps(status)

	@cp.expose
	@cp.tools.accept(media="multipart/form-data")
	@hostmap
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
	@hostmap
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
	@hostmap
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
	@hostmap
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
