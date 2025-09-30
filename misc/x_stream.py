import base64
import diskcache
import functools
import json
import logging
import os
import random
from traceback import print_exc
import cherrypy as cp
from cherrypy._cpdispatch import Dispatcher
import niquests
import orjson
import requests
from .asyncs import eloop, tsubmit, esubmit, csubmit, await_fut
from .types import resume, fcdict, byte_like, MemoryBytes
from .util import AUTH, tracebacksuppressor, magic, decrypt, save_auth, decode_attachment, is_discord_attachment, is_miza_attachment, discord_expired, url2fn, p2n, seq, Request, DOMAIN_CERT, PRIVATE_KEY, update_headers, USER_AGENT, CACHE_PATH
from .caches import attachment_cache

interface = None
csubmit(Request._init_())
tsubmit(eloop.run_forever)
ADDRESS = "0.0.0.0"
PORT = 443

class EndpointRedirects(Dispatcher):

	def __call__(self, path):
		p = path.strip("/")
		first = p.split("/", 1)[0]
		if not p or p == "dummy.html":
			p = "static_backend/index"
		elif first in ("home", "index", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "static"):
			p = "static_backend/" + p
		elif first not in ("proxy", "upload", "delete", "edit", "c", "u", "unproxy", "reupload", "stream", "heartbeat", "backend", "debug"):
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
		"tools.gzip.mime_types": ["text/plain", "text/html", "text/css", "text/csv", "text/xml", "text/md", "text/markdown", "text/javascript", "application/json", "application/javascript"],
	},
	"/": {
		"request.dispatch": EndpointRedirects(),
	},
}
if os.path.exists(DOMAIN_CERT) and os.path.exists(PRIVATE_KEY):
	config["global"]["server.ssl_certificate"] = DOMAIN_CERT
	config["global"]["server.ssl_private_key"] = PRIVATE_KEY
else:
	print("WARNING: SSL keys not found!")
if AUTH:
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

def true_ip(request=None):
	request = request or cp.request
	ip = request.headers["Remote-Addr"]
	if ip == "127.0.0.1":
		ip = request.headers.get("X-Real-Ip") or ip
	if ip == "127.0.0.1":
		ip = request.remote.ip
	cp.serving.request.remote.ip = request.remote.ip = ip
	return ip

@functools.lru_cache(maxsize=256)
def get_size_mime(head, tail, count, chunksize):
	fut = esubmit(requests.head, head)
	resp = requests.head(tail)
	lastsize = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))
	size = chunksize * (count - 1) + lastsize
	resp = fut.result()
	mimetype = resp.headers.get("Content-Type", "application/octet-stream")
	return mimetype, size, lastsize


class Server:

	token = ""
	alt_token = ""
	channels = []
	cache = {}
	ucache = {}
	if os.path.exists("temp.json") and os.path.getsize("temp.json"):
		with open("temp.json", "rb") as f:
			state = json.load(f)
	else:
		state = {"/": f"https://api.mizabot.xyz:{webserver_port}"}
	session = niquests.Session()

	@cp.expose
	@cp.tools.json_in()
	def heartbeat(self, key, uri="", **kwargs):
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
		data = cp.request.json or {}
		data = orjson.loads(decrypt(base64.b64decode(data["data"].encode("ascii") + b"==")))
		if data:
			print("Authorised:", data)
		self.token = data.get("token") or self.token
		self.alt_token = data.get("alt_token") or self.alt_token
		domain_cert = data.get("domain_cert")
		private_key = data.get("private_key")
		self.channels = data.get("channels") or self.channels
		AUTH["discord_token"] = self.token
		AUTH["alt_token"] = self.alt_token
		AUTH["proxy_channels"] = self.channels
		save_auth(AUTH)
		if domain_cert and private_key:
			with open(DOMAIN_CERT, "w") as f:
				f.write(domain_cert)
			with open(PRIVATE_KEY, "w") as f:
				f.write(private_key)
		attachment_cache.init()
		return "💜"

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
		resp = self.get_with_retries(url, data=body, headers=headers, timeout=2)
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
			timeout=60,
		)
		if resp.status_code in range(300, 400):
			raise cp.HTTPRedirect(resp.headers.get("Location") or url, resp.status_code)
		update_headers(cp.response.headers, **resp.headers)
		cp.response.headers.pop("Connection", None)
		cp.response.headers.pop("Transfer-Encoding", None)
		if int(resp.headers.get("Content-Length") or 262145) <= 262144:
			return resp.content
		return resp.iter_content(65536)
	backend._cp_config = {"response.stream": True}

	statics = diskcache.Cache(directory=f"{CACHE_PATH}/upload", expiry=86400 * 30)
	@cp.expose
	def static_backend(self, *path, **query):
		rpath = "/".join(path)
		if rpath:
			rpath = "/" + rpath
		rquery = cp.request.query_string
		if rquery:
			rquery = "?" + rquery
		url = f"{self.state['/']}{rpath}{rquery}"
		try:
			headers, content = self.statics[url]
		except LookupError:
			pass
		else:
			update_headers(cp.response.headers, **headers)
			return content
		headers = dict(cp.request.headers)
		headers.pop("Connection", None)
		headers.pop("Transfer-Encoding", None)
		headers["X-Real-Ip"] = cp.request.remote.ip
		resp = self.session.get(
			url,
			headers=headers,
			verify=False,
			timeout=60,
		)
		headers = dict(resp.headers)
		headers.pop("Connection", None)
		headers.pop("Transfer-Encoding", None)
		self.statics[url] = [headers, resp.content]
		update_headers(cp.response.headers, **headers)
		return resp.content

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
			data = self.cache[info] = self.get_with_retries(info, timeout=5, retries=3).json()
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
		update_headers(headers, **{
			"User-Agent": USER_AGENT,
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
	stream._cp_config = {"response.stream": True}

if __name__ == "__main__":
	logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
	app = Server()
	self = server = cp.Application(app, "/", config)
	cp.quickstart(server, "/", config)
	# waitress.serve(server, threads=128, host=ADDRESS, port=PORT, url_scheme="https")
