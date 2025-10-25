import base64
import diskcache
import functools
import json
import logging
import os
from traceback import print_exc
from typing import Optional, AsyncIterator
import filetype
import niquests
import orjson
import requests
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from .asyncs import asubmit, esubmit, csubmit
from .types import fcdict, byte_like, MemoryBytes
from .util import (
	AUTH, tracebacksuppressor, magic, decrypt, save_auth, decode_attachment,
	is_discord_attachment, url2fn, seq,
	Request as MizaRequest, DOMAIN_CERT, PRIVATE_KEY, update_headers,
	CACHE_PATH,
)
from .caches import attachment_cache

csubmit(MizaRequest._init_())

ADDRESS = "0.0.0.0"
PORT = 443

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


def true_ip(request: Request) -> str:
	"""Extract the true client IP from request headers."""
	ip = request.headers.get("X-Real-Ip") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
	if not ip or ip == "127.0.0.1":
		ip = request.client.host if request.client else "127.0.0.1"
	return ip


async def get_size_mime(head, tail, count, chunksize):
	fut = csubmit(attachment_cache.download, tail)
	HEAD = await attachment_cache.download(head)
	TAIL = await fut
	firstsize = len(HEAD)
	mimetype = filetype.guess_mime(HEAD)
	lastsize = len(TAIL)
	if count >= 2:
		size = firstsize + chunksize * (count - 2) + lastsize
	else:
		size = firstsize
	return mimetype, size, firstsize, lastsize


class Server:
	"""Main server class containing all endpoint handlers."""

	token = ""
	alt_token = ""
	channels = []

	if os.path.exists("temp.json") and os.path.getsize("temp.json"):
		with open("temp.json", "rb") as f:
			state = json.load(f)
	else:
		state = {"/": f"https://api.mizabot.xyz:{webserver_port}"}

	session = niquests.Session()
	asession = niquests.AsyncSession()
	statics = diskcache.Cache(directory=f"{CACHE_PATH}/statics", expiry=86400 * 30)
	dynamics = diskcache.Cache(directory=f"{CACHE_PATH}/dynamics", expiry=86400 * 30)

	def get_with_retries(self, url, headers={}, data=None, timeout=3, retries=5):
		"""HTTP GET with automatic retries."""
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

	async def dyn_serve(
		self,
		urls: list,
		size: int = 0,
		head=None,
		callback=None,
		request: Request = None,
		mimetype: str = "application/octet-stream"
	) -> StreamingResponse:
		"""Dynamically serve content from multiple URLs with range support."""
		with tracebacksuppressor:
			brange = request.headers.get("Range", "").removeprefix("bytes=") if request else ""
			headers = fcdict(request.headers) if request else fcdict()
			headers.pop("Content-Length", None)
			headers.pop("Content-Type", None)
			headers.pop("Remote-Addr", None)
			headers.pop("Host", None)
			headers.pop("Range", None)
			update_headers(headers, **MizaRequest.header())
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

			status_code = 206 if ranges else 200
			if not ranges:
				ranges.append((0, size))
				length = size
			response_headers = dict(HEADERS)
			if ranges == [(0, size)]:
				response_headers["Content-Length"] = str(length)
			if brange:
				cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size or '*'}" for start, end in ranges)
				response_headers["Content-Range"] = cr
			print(brange, ranges)

			return StreamingResponse(
				self._dyn_serve(urls, ranges, headers, head=head, callback=callback),
				status_code=status_code,
				headers=response_headers,
				media_type=mimetype,
			)

	async def _dyn_serve(self, urls, ranges, headers, head=None, callback=None) -> AsyncIterator[bytes]:
		"""Internal generator for dynamic serving."""

		def content_generator(b, chunksize=262144):
			for i in range(0, len(b), chunksize):
				yield b[i:i + chunksize]

		with tracebacksuppressor(GeneratorExit):
			if head:
				data = head.data[head.index:]
				urls.insert(0, bytes(data))
				if callback:
					callback(head.data)

			counter = -1
			for i, (start, end) in enumerate(ranges):
				pos = 0
				rems = urls.copy()
				futs = []
				big = False

				while rems:
					u = rems.pop(0)
					counter += 1
					if isinstance(u, byte_like):
						ns = len(u)
					elif "?size=" in u or "&size=" in u:
						u, ns = u.replace("?size=", "&size=").split("&size=", 1)
						ns = int(ns)
					elif "?S=" in u or "&S=" in u:
						u, ns = u.replace("?S=", "&S=").split("&S=", 1)
						ns = int(ns)
					else:
						resp = await asubmit(requests.head, u, timeout=3)
						ns = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))

					if pos + ns <= start:
						pos += ns
						continue
					if pos >= end:
						break

					async def get_chunk(u, h, start, end, pos, ns, big):
						s = start - pos
						e = end - pos
						if not isinstance(u, byte_like):
							print("get_chunk:", u)
							b = await attachment_cache.download(u)
						if s <= 0 and e >= len(b):
							return b
						return memoryview(b)[s:e]

					if len(futs) > i + 1:
						b = await futs.pop(0)
						for chunk in content_generator(b, chunksize=262144 if counter else 65536):
							yield chunk

					fut = asubmit(get_chunk, u, headers, start, end, pos, ns, big)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns
					big = True

				for fut in futs:
					b = await fut
					for chunk in content_generator(b, chunksize=262144 if counter else 65536):
						yield chunk


# Create FastAPI app
app = FastAPI(title="Miza Proxy Server", version="2.0")
server = Server()


@app.middleware("http")
async def add_headers_middleware(request: Request, call_next):
	"""Add standard headers to all responses."""
	response = await call_next(request)
	for key, value in HEADERS.items():
		response.headers[key] = value
	return response


@app.post("/authorised-heartbeat")
async def authorised_heartbeat(request: Request, key: str = Query(...), uri: str = Query("")):
	"""Receive configuration updates from Discord bot."""
	if key != discord_secret:
		raise HTTPException(status_code=403, detail="Invalid key")

	uri = uri or f"https://{true_ip(request)}:{webserver_port}"

	if server.state["/"] != uri:
		server.state["/"] = uri
		with open("temp.json", "w") as f:
			json.dump(server.state, f)

	body = await request.json()
	data = orjson.loads(decrypt(base64.b64decode(body["data"].encode("ascii") + b"==")))

	if data:
		print("Authorised update:", data)

	server.token = data.get("token") or server.token
	server.alt_token = data.get("alt_token") or server.alt_token
	domain_cert = data.get("domain_cert")
	private_key = data.get("private_key")
	server.channels = data.get("channels") or server.channels

	AUTH["discord_token"] = server.token
	AUTH["alt_token"] = server.alt_token
	AUTH["proxy_channels"] = server.channels
	save_auth(AUTH)

	if domain_cert and private_key:
		with open(DOMAIN_CERT, "w") as f:
			f.write(domain_cert)
		with open(PRIVATE_KEY, "w") as f:
			f.write(private_key)

	attachment_cache.init()
	return "💜"


@app.get("/c/{path:path}")
async def chunked_proxy(path: str, request: Request):
	"""Serve chunked/split files with range support."""
	try:
		urls, chunksize = await attachment_cache.obtains(path.split("/", 1)[0])
	except ConnectionError as ex:
		raise HTTPException(status_code=ex.errno, detail=str(ex))
	mimetype, size, firstsize, lastsize = await get_size_mime(urls[0], urls[-1], len(urls), chunksize)

	new_urls = [f"{url}&S={firstsize if not i else lastsize if i >= len(urls) - 1 else chunksize}" for i, url in enumerate(urls)]

	response = await server.dyn_serve(new_urls, size, request=request, mimetype=mimetype)
	return response


@app.get("/u/{path:path}")
async def unproxy(path: str, request: Request, url: Optional[str] = None, force: bool = False):
	"""Unproxy Discord attachments or redirect to direct URLs."""
	if url:
		return await proxy_if(url, request, force=force)

	path_parts = path.split("/")

	if len(path_parts) == 1 and path_parts[0].count("~") == 2:
		try:
			segments = path_parts[0].split(".", 1)[0].split("~", 2)
		except Exception as ex:
			raise HTTPException(status_code=400, detail=str(ex))
		try:
			resp = await attachment_cache.obtain(*segments)
		except ConnectionError as ex:
			raise HTTPException(status_code=ex.errno, detail=str(ex))
		return await proxy_if(resp, request, force=force)

	if len(path_parts) == 2 and path_parts[0].count("~") == 0:
		try:
			c_id, m_id, a_id, fn = decode_attachment(path)
		except Exception as ex:
			raise HTTPException(status_code=400, detail=str(ex))
		try:
			resp = await attachment_cache.obtain(c_id, m_id, a_id, fn)
		except ConnectionError as ex:
			raise HTTPException(status_code=ex.errno, detail=str(ex))
		return await proxy_if(resp, request, force=force)

	if hasattr(server, "state"):
		query_string = str(request.url.query) if request.url.query else ""
		redirect_url = f"{server.state['/']}/u/{path}"
		if query_string:
			redirect_url += f"?{query_string}"
		return RedirectResponse(url=redirect_url, status_code=307)

	raise HTTPException(status_code=404, detail="Invalid path")


@app.post("/reupload")
async def reupload(
	request: Request,
	url: Optional[str] = Query(None),
	filename: Optional[str] = Query(None),
	file: Optional[UploadFile] = File(None)
):
	"""Re-upload files to Discord storage."""
	try:
		if file:
			resp = file.file
		else:
			resp = None
	except Exception:
		print_exc()
		resp = None

	content_length = int(request.headers.get("Content-Length", 0))

	if not resp or content_length < 1:
		if not url:
			return "Expected input URL or data."
		headers = MizaRequest.header()
		if request.headers.get("Range"):
			headers["Range"] = request.headers["Range"]
		resp = server.get_with_retries(url, headers=headers, timeout=3)

	fn = filename or (url2fn(url) if url else None)
	return await attachment_cache.create(seq(resp), filename=fn)


async def proxy_if(url: str, request: Request, force=False):
	"""Proxy if needed, otherwise redirect."""
	assert isinstance(url, str), url

	def requires_proxy():
		if "Cf-Worker" in request.headers:
			return True
		if "bot" in request.headers.get("User-Agent", ""):
			return False
		return True

	if force or requires_proxy():
		return await proxy(url=url, request=request)
	return RedirectResponse(url=url, status_code=307)


@app.api_route("/proxy", methods=["GET", "POST"])
async def proxy(request: Request, url: Optional[str] = None):
	"""Proxy any URL with optional body forwarding."""
	if not url:
		return "Expected proxy URL."

	try:
		body = await request.body()
	except Exception:
		print_exc()
		body = None

	if request.method.lower() == "get" and not body:
		print("get_download:", url)
		try:
			data = await attachment_cache.download(url)
		except ConnectionError as ex:
			raise HTTPException(status_code=ex.errno, detail=str(ex))
		brange = request.headers.get("Range", "").removeprefix("bytes=") if request else ""
		size = len(data)
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

		status_code = 206 if ranges else 200
		if not ranges:
			ranges.append((0, size))
			length = size
		response_headers = dict(HEADERS)
		if ranges == [(0, size)]:
			response_headers["Content-Length"] = str(length)
			direct = True
		else:
			direct = False
		if brange:
			cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size or '*'}" for start, end in ranges)
			response_headers["Content-Range"] = cr

		async def content_generator(b, chunksize=65536):
			for i in range(0, len(b), chunksize):
				yield b[i:i + chunksize]

		return StreamingResponse(
			content_generator(b"".join(data[r[0]:r[1]] for r in ranges) if not direct else data),
			status_code=status_code,
			headers=response_headers,
			media_type=filetype.guess_mime(data),
		)

	headers = MizaRequest.header()
	if request.headers.get("Range"):
		headers["Range"] = request.headers["Range"]

	resp = server.get_with_retries(url, data=body, headers=headers, timeout=2)

	response_headers = fcdict(resp.headers)
	response_headers.pop("Connection", None)
	response_headers.pop("Transfer-Encoding", None)
	if response_headers.pop("Content-Encoding", None):
		response_headers.pop("Content-Length", None)
	response_headers.pop("Date", None)
	response_headers.pop("Server", None)

	if is_discord_attachment(url):
		response_headers.pop("Content-Disposition", None)
		update_headers(response_headers, **CHEADERS)

	ctype = resp.headers.get("Content-Type", "application/octet-stream")
	it = await asubmit(resp.iter_content, 65536)

	if ctype in ("text/html", "text/html; charset=utf-8", "application/octet-stream"):
		b = await asubmit(next, it)
		mime = magic.from_buffer(b)
		if mime == "application/octet-stream":
			a = MemoryBytes(b)[:128]
			if sum(32 <= c < 128 for c in a) >= len(a) * 7 / 8:
				mime = "text/plain"
		response_headers["Content-Type"] = mime

		async def content_generator():
			yield b
			try:
				while True:
					yield await asubmit(next, it)
			except StopIteration:
				pass

		return StreamingResponse(
			content_generator(),
			status_code=resp.status_code,
			headers=response_headers,
			media_type=mime,
		)

	async def content_generator():
		try:
			while True:
				yield await asubmit(next, it)
		except StopIteration:
			pass

	return StreamingResponse(
		content_generator(),
		status_code=resp.status_code,
		headers=response_headers,
		media_type=ctype,
	)


@app.get("/backend/{path:path}")
async def backend(path: str, request: Request):
	"""Proxy requests to backend API server."""
	query_string = str(request.url.query) if request.url.query else ""
	url = f"{server.state['/']}/{path}"
	if query_string:
		url += f"?{query_string}"

	if request.method != "GET" or server.state["/"].startswith("https://"):
		return RedirectResponse(url=url, status_code=307)

	headers = fcdict(request.headers)
	headers.pop("Connection", None)
	headers.pop("Transfer-Encoding", None)
	headers.pop("Cache-Control", None)
	headers.pop("If-Modified-Since", None)
	headers.pop("If-None-Match", None)
	headers["X-Real-Ip"] = true_ip(request)

	print("BACKEND:", url)

	resp = await asubmit(
		server.session.get,
		url,
		headers=headers,
		stream=True,
		verify=False,
		allow_redirects=False,
		timeout=60,
	)

	if resp.status_code in range(300, 400):
		return RedirectResponse(url=resp.headers.get("Location") or url, status_code=resp.status_code)

	response_headers = fcdict(resp.headers)
	response_headers.pop("Connection", None)
	response_headers.pop("Transfer-Encoding", None)
	if response_headers.pop("Content-Encoding", None):
		response_headers.pop("Content-Length", None)
	response_headers.pop("Date", None)
	response_headers.pop("Server", None)

	if int(resp.headers.get("Content-Length") or 0) <= 262144:
		return Response(
			content=resp.content,
			status_code=resp.status_code,
			headers=response_headers
		)

	return StreamingResponse(
		resp.iter_content(65536),
		status_code=resp.status_code,
		headers=response_headers
	)


@app.get("/static_backend/{path:path}")
async def static_backend(path: str, request: Request):
	"""Serve cached static content from backend."""
	query_string = str(request.url.query) if request.url.query else ""
	url = f"{server.state['/']}/{path}"
	if query_string:
		url += f"?{query_string}"

	try:
		headers, content = server.statics[url]
	except LookupError:
		pass
	else:
		return Response(content=content, headers=headers)

	headers = fcdict(request.headers)
	headers.pop("Connection", None)
	headers.pop("Transfer-Encoding", None)
	headers.pop("Cache-Control", None)
	headers.pop("If-Modified-Since", None)
	headers.pop("If-None-Match", None)
	headers["X-Real-Ip"] = true_ip(request)

	print(url, headers)
	resp = await asubmit(server.session.get, url, headers=dict(headers), verify=False, timeout=60)

	response_headers = fcdict(resp.headers)
	response_headers.pop("Connection", None)
	response_headers.pop("Transfer-Encoding", None)
	if response_headers.pop("Content-Encoding", None):
		response_headers.pop("Content-Length", None)
	response_headers.pop("Date", None)
	response_headers.pop("Server", None)
	print(resp, resp.headers, len(resp.content))

	server.statics[url] = [response_headers, resp.content]

	return Response(content=resp.content, headers=response_headers)


# Catch-all route for custom routing logic
@app.get("/{path:path}")
async def catch_all(path: str, request: Request):
	"""Handle custom routing for static files and redirects."""
	p = path.strip("/")
	first = p.split("/", 1)[0] if p else ""

	if not p or p == "dummy.html" or p == "index":
		return await static_backend("index.html", request=request)
	elif first in ("favicon.ico", "logo256.png", "logo512.png", "home", "p", "preview", "files", "file", "chat", "tester", "atlas", "mizatlas", "static"):
		return await static_backend(p, request=request)
	elif first not in ("static_backend", "proxy", "c", "u", "unproxy", "reupload", "authorised-heartbeat", "backend", "debug"):
		return await backend(p, request=request)
	return await globals()[first](request=request)

	# Redirect to appropriate handler
	query_string = str(request.url.query) if request.url.query else ""
	redirect_url = f"/{p}"
	if query_string:
		redirect_url += f"?{query_string}"

	return RedirectResponse(url=redirect_url, status_code=307)


if __name__ == "__main__":
	# Configure logging
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)
	logger = logging.getLogger(__name__)

	from hypercorn.config import Config
	from hypercorn.asyncio import serve

	# Configure Hypercorn
	config = Config()
	config.bind = config.quic_bind = [f"0.0.0.0:{PORT}"]
	config.worker_class = "asyncio"
	config.workers = 1  # Single worker, uses async for concurrency
	config.keep_alive_timeout = 65
	config.graceful_timeout = 30

	# SSL Configuration
	if os.path.exists(DOMAIN_CERT) and os.path.exists(PRIVATE_KEY):
		config.certfile = DOMAIN_CERT
		config.keyfile = PRIVATE_KEY
		logger.info(f"SSL enabled with cert: {DOMAIN_CERT}")
	else:
		logger.warning("WARNING: SSL keys not found! Running without HTTPS.")
		config.bind = config.quic_bind = ["0.0.0.0:80"]  # Fallback to HTTP

	# Additional configuration
	config.accesslog = "-"  # Log to stdout
	config.errorlog = "-"   # Log to stdout
	config.loglevel = "INFO"

	logger.info(f"Starting Miza Proxy Server on {config.bind}")
	logger.info("Press Ctrl+C to stop")

	from .asyncs import eloop
	# Run the server
	try:
		eloop.run_until_complete(serve(app, config))
	except KeyboardInterrupt:
		logger.info("Server stopped by user")
	except Exception as e:
		logger.error(f"Server error: {e}", exc_info=True)
		raise SystemExit