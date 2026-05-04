import base64
import diskcache
import json
import logging
import os
import time
from traceback import print_exc
from typing import Optional, AsyncIterator
from urllib.parse import quote, unquote
import niquests
import orjson
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette_compress import CompressMiddleware
from .asyncs import run_async, create_task
from .types import fcdict, byte_like, MemoryBytes
from .util import (
	AUTH, tracebacksuppressor, magic, decrypt, save_auth, decode_attachment, discord_expired,
	is_discord_attachment, url2fn, getsize, mime_from_file,
	Request as RequestManager, DOMAIN_CERT, PRIVATE_KEY, update_headers,
	AutoCache, CACHE_PATH, VISUAL_FORMS, RNGFile,
)
from .caches import attachment_cache, colour_cache

create_task(RequestManager._init_())

ADDRESS = "0.0.0.0"
PORT = 443

if AUTH:
	discord_secret = AUTH.get("discord_secret") or ""
	webserver_port = AUTH.get("webserver_port") or "9801"
else:
	discord_secret = ""
	webserver_port = "9801"

HEADERS = {
	"Server": "Miza",
	"Vary": "Accept-Encoding",
	"Accept-Ranges": "bytes",
	"Access-Control-Expose-Headers": "*",
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
	fut = create_task(attachment_cache.scan_headers(tail, base="mizabot.xyz", fc=True))
	HEAD = await attachment_cache.download(head, read=True, fc=True)
	TAIL_headers = await fut
	firstsize = getsize(HEAD)
	mimetype = mime_from_file(HEAD)
	lastsize = int(TAIL_headers.get("content-length") or 1)
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
	statics = AutoCache(directory=f"{CACHE_PATH}/statics", stale=300, timeout=86400 * 30)
	dynamics = diskcache.Cache(directory=f"{CACHE_PATH}/dynamics", expiry=86400 * 30)

	async def dyn_serve(
		self,
		urls: list,
		size: int = 0,
		head=None,
		response_headers={},
		callback=None,
		request: Optional[Request] = None,
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
			update_headers(headers, **RequestManager.header())
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
			response_headers.update(HEADERS)
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
		with tracebacksuppressor(GeneratorExit):
			if head:
				data = head.data[head.index:]
				urls.insert(0, bytes(data))
				if callback:
					callback(head.data)

			counter = 0
			for i, (start, end) in enumerate(ranges):
				pos = 0
				rems = urls.copy()
				futs = []

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
					else:
						resp = await run_async(niquests.head, u, timeout=3)
						ns = int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 0))

					if pos + ns <= start:
						pos += ns
						continue
					if pos >= end:
						break

					async def get_chunk(u, h, start, end, pos, ns):
						s = start - pos
						e = end - pos
						print("get_chunk:", u)
						fp = await attachment_cache.download(u, read=True)

						async def content_generator():
							chunksize = 262144 if counter else 65536
							fp.seek(s)
							for i in range(s, e, chunksize):
								yield fp.read(min(chunksize, e - i))

						return content_generator()

					if len(futs) > 1:
						gen = await futs.pop(0)
						async for chunk in gen:
							yield chunk
						counter += 1

					fut = run_async(get_chunk, u, headers, start, end, pos, ns)
					futs.append(fut)
					pos = 0
					start = 0
					end -= start + ns

				for fut in futs:
					gen = await fut
					async for chunk in gen:
						yield chunk
					counter += 1


def stream_fp(request, fp, response_headers={}):
	brange = request.headers.get("Range", "").removeprefix("bytes=") if request else ""
	size = getsize(fp)
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
	response_headers.update(HEADERS)
	if ranges == [(0, size)]:
		response_headers["Content-Length"] = str(length)
	if brange:
		cr = "bytes " + ", ".join(f"{start}-{end - 1}/{size or '*'}" for start, end in ranges)
		response_headers["Content-Range"] = cr

	async def content_generator(chunksize=262144 if length > 64 * 1048576 else 65536):
		for r in ranges:
			fp.seek(r[0])
			for i in range(r[0], r[1], chunksize):
				yield fp.read(min(chunksize, r[1] - i))

	mime = mime_from_file(fp)
	if not mime:
		fp.seek(0)
		b = fp.read(65536)
		try:
			s = b.decode("utf-8")
		except Exception:
			mime = "application/octet-stream"
		else:
			mime = (
				"text/html" if s.lower().startswith("<!doctype html>")
				else "image/svg+xml" if s.lower().startswith("<svg ")
				else "text/plain"
			)

	return StreamingResponse(
		content_generator(),
		status_code=status_code,
		headers=response_headers,
		media_type=mime,
	)


# Create FastAPI app
app = FastAPI(title="Miza Proxy Server", version="2.0")
app.add_middleware(
	CORSMiddleware,
	allow_credentials=True,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)
app.add_middleware(
	CompressMiddleware,
	minimum_size=1024,
)
server = Server()


@app.middleware("http")
async def add_headers_middleware(request: Request, call_next):
	"""Add standard headers to all responses."""
	response = await call_next(request)
	del response.headers["Server"]
	for key, value in HEADERS.items():
		response.headers[key] = value
	return response


global_ip = "127.0.0.1"
last_ip_check = 0
@app.head(path="/ip")
@app.get("/ip")
async def ip(request: Request):
	global global_ip, last_ip_check
	if time.time() - last_ip_check > 3600:
		async with niquests.AsyncSession() as asession:
			resp = await asession.get("https://api.ipify.org")
			global_ip = resp.text
		last_ip_check = time.time()
	return dict(host=global_ip, remote=true_ip(request))


@app.head(path="/random")
@app.get("/random")
async def prandom(request: Request, count: int = 1048576):
	return stream_fp(request, RNGFile(count), {"Content-Disposition": "attachment; filename=random.bin"})


@app.get("/mean-color")
@app.get("/mean-colour")
async def mean_colour(request: Request, url: Optional[str] = None):
	resp = await colour_cache.obtain(url)
	return list(resp)

@app.head("/mean-color")
@app.head("/mean-colour")
async def head_mc(response: Response):
	response.headers["Content-Type"] = "application/json"
	return


@app.post("/authorised-heartbeat")
async def authorised_heartbeat(request: Request, key: Optional[str] = None, uri: Optional[str] = ""):
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

	ac = data.get("attachment_cache")
	if ac:
		for k, v in ac.items():
			attachment_cache.store(v)
	attachment_cache.init()
	return {str(k): v for k, v in attachment_cache.items() if isinstance(k, int) and v and not discord_expired(v)}


@app.head(path="/c/{path:path}")
@app.get("/c/{path:path}")
@app.get("/chunked-proxy/{path:path}")
async def chunked_proxy(request: Request, path: str):
	"""Serve chunked/split files with range support."""
	try:
		urls, chunksize = await attachment_cache.obtains(path.split("/", 1)[0])
	except ConnectionError as ex:
		raise HTTPException(status_code=ex.errno or 500, detail=str(ex))
	mimetype, size, firstsize, lastsize = await get_size_mime(urls[0], urls[-1], len(urls), chunksize)
	new_urls = [f"{url}&S={firstsize if not i else lastsize if i >= len(urls) - 1 else chunksize}" for i, url in enumerate(urls)]
	heads = await attachment_cache.scan_headers(urls[0], base="mizabot.xyz", fc=True)
	response_headers = {}
	filename = heads.get("attachment-filename") or unquote(heads.get("content-disposition", "").split("filename=", 1)[-1].lstrip('"').split('"', 1)[0].strip().strip('"').strip("'") or urls[0].rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0])
	if filename:
		response_headers["Content-Disposition"] = f"inline; filename={quote(filename)}"
	response = await server.dyn_serve(new_urls, size, request=request, mimetype=mimetype, response_headers=response_headers)
	return response


@app.head(path="/u/{path:path}")
@app.head(path="/u")
@app.get("/u/{path:path}")
@app.get("/u")
@app.get("/unproxy/{path:path}")
async def unproxy(request: Request, path: Optional[str] = None, url: Optional[str] = None, force: bool = False, download: bool = False):
	"""Unproxy Discord attachments or redirect to direct URLs."""
	assert request is not None
	if request.method.upper() == "HEAD":
		force = True
	if url:
		return await proxy_if(url, request, force=force, download=download)
	if not path:
		raise HTTPException(status_code=400, detail="Must provide path or URL.")
	try:
		c_id, m_id, a_id, fn = decode_attachment(path)
	except Exception as ex:
		raise HTTPException(status_code=400, detail=str(ex))
	try:
		resp = await attachment_cache.obtain(c_id, m_id, a_id, fn)
	except ConnectionError as ex:
		raise HTTPException(status_code=ex.errno or 500, detail=f"{url}: {ex}")
	try:
		return await proxy_if(resp, request, force=force, download=download)
	except ConnectionError as ex:
		if ex.errno == 404:
			attachment_cache.remove_cached(request.url)
			resp = await attachment_cache.obtain(c_id, m_id, a_id, fn)
			return await proxy_if(resp, request, force=force, download=download)
		raise


@app.post("/upload")
async def upload(
	request: Request,
	url: Optional[str] = None,
	filename: Optional[str] = None,
	file: Optional[UploadFile] = None,
):
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
			return Response(
				content="Expected input URL or data.",
				media_type="text/plain",
			)
		headers = RequestManager.header()
		if request.headers.get("Range"):
			headers["Range"] = request.headers["Range"]
		resp = await server.asession.get(
			url,
			headers=RequestManager.header(),
		)
		filename = filename or unquote(resp.headers.get("content-disposition", "").split("filename=", 1)[-1])
		resp = resp.content

	fn = filename or getattr(resp, "filename", None) or (url2fn(url) if url else None)

	url = await attachment_cache.create_dynamic(resp, filename=fn)
	return Response(
		content=url,
		media_type="text/plain",
	)


async def proxy_if(url: str, request: Request, force: bool = False, download: bool = False):
	"""Proxy if needed, otherwise redirect."""
	assert isinstance(url, str), url

	def requires_proxy():
		if "Cf-Worker" in request.headers:
			return True
		ua = request.headers.get("User-Agent", "")
		if "bot" in ua or "Bot" in ua:
			return False
		if download and is_discord_attachment(url):
			if url.split("?", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[-1] in VISUAL_FORMS:
				return True
			return False
		return True

	if force or requires_proxy():
		return await proxy(url=url, request=request, force=force, download=download)
	return RedirectResponse(url=url, status_code=307)


@app.api_route("/proxy", methods=["GET", "POST"])
async def proxy(request: Request, url: Optional[str] = None, force: bool = False, download: bool = False):
	"""Proxy any URL with optional body forwarding."""
	if not url:
		return Response(
			content="Expected proxy URL.",
			status_code=400,
			media_type="text/plain",
		)

	try:
		fp = await attachment_cache.download(url, read=True)
	except ConnectionError as ex:
		raise HTTPException(status_code=ex.errno or 500, detail=f"{url}: {ex}")
	heads = await attachment_cache.scan_headers(url, base="mizabot.xyz", fc=True)

	response_headers = {}
	filename = heads.get("attachment-filename") or unquote(heads.get("content-disposition", "").split("filename=", 1)[-1].lstrip('"').split('"', 1)[0].strip().strip('"').strip("'") or url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0])
	disposition = "attachment" if download else "inline"
	if filename:
		response_headers["Content-Disposition"] = f"{disposition}; filename={quote(filename)}"
	response_headers["Cache-Control"] = "public,max-age=21600,stale-while-revalidate=1073741824,stale-if-error=1073741824"

	if not force and heads.get("content-type").split(";", 1)[0] == "text/markdown":
		new_url = str(request.url.include_query_params(force="1"))
		return Response(
			"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script><style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6;max-width:800px;margin:0 auto;padding:20px;color:#333}#viewer blockquote{border-left:4px solid #ccc;margin-left:0;padding-left:16px;color:#666}#viewer code{background-color:#f4f4f4;padding:2px 4px;border-radius:4px}#viewer pre{background-color:#f4f4f4;padding:16px;border-radius:4px;overflow-x:auto}</style></head><body><div id="viewer"><p><em>Loading...</em></p></div><script>async function renderMarkdown(){try{const r=await fetch(URL);if(!r.ok)throw new Error(`HTTP error! status: ${r.status}`);const e=await r.text();viewer.innerHTML=marked.parse(e)}catch(r){viewer.innerHTML=`<p style="color: red;"><strong>Failed to load Markdown:</strong> ${r.message}</p>\n                                    <p><small>Note: The server hosting the file must allow Cross-Origin Resource Sharing (CORS).</small></p>`}}renderMarkdown();</script></body></html>""".replace("URL", json.dumps(new_url)),
			headers=response_headers,
			media_type="text/html",
		)
	return stream_fp(request, fp, response_headers)


ytdownloader = None
@app.get("/ytdl")
async def ytdl(query: Optional[str] = None):
	global ytdownloader
	if not query:
		return
	if not ytdownloader:
		import yt_dlp as ytd
		ydl_opts = {
			"quiet": 1,
			"format": "bestvideo+bestaudio/best*",
			"overwrites": 1,
			"nocheckcertificate": 1,
			"no_call_home": 1,
			"nooverwrites": 1,
			"noplaylist": 1,
			"logtostderr": 0,
			"ignoreerrors": 0,
			"default_search": "auto",
			"source_address": "0.0.0.0",
			"remote_components": ["ejs:github"],
		}
		ytdownloader = ytd.YoutubeDL(ydl_opts)
	return await run_async(ytdownloader.extract_info, query, download=False)

@app.head("/ytdl")
async def head_ytdl(response: Response):
	response.headers["Content-Type"] = "application/json"
	return


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

	resp = await server.asession.get(
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

	async def backend_request(url):
		headers = fcdict(request.headers)
		headers.pop("Connection", None)
		headers.pop("Transfer-Encoding", None)
		headers.pop("Cache-Control", None)
		headers.pop("If-Modified-Since", None)
		headers.pop("If-None-Match", None)
		headers["X-Real-Ip"] = true_ip(request)

		print(url, headers)
		resp = await run_async(server.session.get, url, headers=dict(headers), verify=False, timeout=60)

		response_headers = fcdict(resp.headers)
		response_headers.pop("Connection", None)
		response_headers.pop("Transfer-Encoding", None)
		if response_headers.pop("Content-Encoding", None):
			response_headers.pop("Content-Length", None)
		response_headers.pop("Date", None)
		response_headers.pop("Server", None)
		print(resp, resp.headers, len(resp.content))

		return [response_headers, resp.content, resp.status_code]

	try:
		headers, content, status_code = await server.statics.aretrieve(url, backend_request, url)
		if status_code not in range(200, 400):
			raise ValueError(status_code)
	except (LookupError, ValueError):
		headers, content, status_code = await server.statics._aretrieve(url, backend_request, url)
	return Response(content=content, headers=headers, status_code=status_code)


alias = tuple([fn.split("/", 1)[0].rsplit(".", 1)[0] for fn in os.listdir("misc/web")])
# Catch-all route for custom routing logic
@app.get("/{path:path}")
async def catch_all(path: str, request: Request):
	"""Handle custom routing for static files and redirects."""
	p = path.strip("/")
	first = p.split("/", 1)[0] if p else ""

	if not p or p in ("home", "index", "dummy.html", "index.html"):
		return FileResponse("misc/web/index.html", media_type="text/html")
	if p in ("robots", "robots.txt"):
		return FileResponse("misc/web/robots.txt", media_type="text/plain")
	if p in ("favicon", "favicon.ico"):
		return FileResponse("misc/web/assets/images/mizaleaf.webp", media_type="image/webp")
	if p == "files":
		return FileResponse("misc/web/files.html", media_type="text/html")
	if p in ("commands", "atlas", "mizatlas"):
		return FileResponse("misc/web/commands.html", media_type="text/html")
	if first == "assets":
		return FileResponse(f"misc/web/{p}")
	if first in (".git", ".env", "admin", "private", "internal", "administrator"):
		return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", status_code=308)
	if first in alias:
		return await static_backend(p, request=request)
	return Response(
		status_code=404,
	)


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
	config.workers = 3
	config.backlog = 1024
	config.keep_alive_timeout = 10
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