import base64
import datetime
import json
import logging
import os
import random
import time
from traceback import print_exc
from urllib.parse import unquote_plus
import cherrypy as cp
from cherrypy._cpdispatch import Dispatcher
import orjson
import niquests
from .asyncs import eloop, tsubmit, esubmit, csubmit, await_fut, gather
from .types import resume, cdict, fcdict, json_dumps, byte_like, utc, RangeSet, MemoryBytes
from .util import AUTH, tracebacksuppressor, magic, shash, decrypt, zip2bytes, bytes2zip, enc_box, save_auth, decode_attachment, expand_attachment, shorten_attachment, is_discord_attachment, is_miza_attachment, discord_expired, url2fn, p2n, byte_scale, leb128, decode_leb128, seq, MIMES, Request, reqs, DOMAIN_CERT, PRIVATE_KEY, update_headers
from .caches import attachment_cache, upload_cache, download_cache

interface = None
csubmit(Request._init_())
tsubmit(eloop.run_forever)
ADDRESS = "0.0.0.0"
PORT = 443

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
		elif first in ("f", "d"):
			p = "download/" + p.split("/", 1)[-1]
		elif first not in ("fi", "fileinfo", "proxy", "upload", "delete", "edit", "u", "unproxy", "reupload", "stream", "heartbeat", "backend", "debug"):
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
if os.path.exists(DOMAIN_CERT) and os.path.exists(PRIVATE_KEY):
	config["global"]["server.ssl_certificate"] = DOMAIN_CERT
	config["global"]["server.ssl_private_key"] = PRIVATE_KEY
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
					esubmit(cache_temp)
					info = self.ucache[irl][1]
				else:
					info = self.ucache[irl][1]
				fn = info["filename"]
				mim = info["mimetype"]
				attachment = filename or fn
				size = info["size"]
				a2 = unquote_plus(attachment).removeprefix(".temp$@")
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
		update_headers(cp.response.headers.update, **CHEADERS)
		cp.response.headers["Content-Type"] = mime
		cp.response.headers["Content-Length"] = len(data)
		return data

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
		update_headers(cp.response.headers, **CHEADERS)
		cp.response.headers["Content-Type"] = MIMES.get(rpath.rsplit(".", 1)[-1]) or "text/html"
		if rpath.strip("/") == "notfound.png":
			cp.response.status = 404
		if rpath in self.cache:
			return self.cache[rpath]
		with open(rpath, "rb") as f:
			self.cache[rpath] = b = f.read()
		return b

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
		update_headers(headers.update, **Request.header())
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
				assert len(encoded) + len(head) <= attachment_cache.max_size
				fut = attachment_cache.create(encoded + head, filename=filename, editable=True)
				url = await_fut(fut)
				cid, mid, aid, fn = expand_attachment(url)
				uhead = shorten_attachment(cid, mid, aid, fn, mode="p") + "?key=" + base64.urlsafe_b64encode(upload_cache[h].key).rstrip(b"=").decode("ascii")
				yield b'"url":"' + uhead.encode("utf-8") + b'"'
			yield b"}"
		return start_upload(position)
	upload._cp_config = {"response.stream": True}

	@cp.expose
	def delete(self, *path, key=None):
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
			print("HEADERS:", cp.response.headers)
			return resp.content
		print("HEADERS:", cp.response.headers)
		return resp.iter_content(65536)
	backend._cp_config = {"response.stream": True}

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
