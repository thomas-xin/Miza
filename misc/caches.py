import asyncio
import concurrent.futures
import fractions
import io
import os
import random
import re
import shutil
import subprocess
import time
import traceback
import zipfile
import aiohttp
import niquests
import numpy as np
from PIL import Image
import psutil
import requests
import streamshatter
from misc.types import utc, as_str, byte_like, cdict
from misc.asyncs import asubmit, esubmit, wrap_future, await_fut, Future
from misc.smath import get_closest_heart
from misc.util import (
    CACHE_FILESIZE, CACHE_PATH, AUTH, Request, api, AutoCache, read_file_a, download_file, header_test,
    tracebacksuppressor, choice, json_dumps, json_dumpstr, b64, scraper_blacklist,
	ungroup_attachments, is_discord_url, temporary_file, url2ext, is_discord_attachment, is_miza_url,
    snowflake_time_2, shorten_attachment, expand_attachment, merge_url, split_url, discord_expired, unyt,
)

def has_transparency(image):
	assert image.mode == "P", "Expected a palette image."
	transparent = image.info.get("transparency", -1)
	if transparent != -1:
		return True
	for tup in image.getcolors():
		if len(tup) in (2, 4):
			alpha = tup[-1]
			if alpha < 254:
				return True
	return False
def remove_p(image):
	if image.mode == "P":
		mode = "RGBA" if has_transparency(image) else "RGB"
		return image.convert(mode)
	return image
def split_rgba(image):
	image = remove_p(image)
	if image.mode == "RGBA":
		a = np.asanyarray(image, dtype=np.uint8)
		aa = a.T[3]
		if np.min(aa) >= 254:
			A = None
		else:
			A = Image.fromarray(aa.T)
		image = Image.fromarray(a.T[:3].T, mode="RGB")
	else:
		A = None
	return image, A
def get_colour(image):
	rgb, A = split_rgba(image)
	if A:
		a = np.array(A, dtype=np.float32)
		a *= 1 / 255
		sumA = np.sum(a)
		if sumA == 0:
			return [0, 0, 0]
		return [(np.sum(np.multiply(c.T, a)) / sumA).item() for c in np.asanyarray(rgb, dtype=np.uint8).T]
	return [np.mean(c).item() for c in np.asanyarray(rgb, dtype=np.uint8).T]
def parse_colour(s):
	if s == "black":
		return (0, 0, 0)
	if s == "white":
		return (255, 255, 255)
	hex_match = re.match(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$', s)
	if hex_match:
		hex_value = hex_match.group(1)
		if len(hex_value) == 3:
			r, g, b = [int(char * 2, 16) for char in hex_value]
		else:
			r, g, b = [int(hex_value[i:i+2], 16) for i in (0, 2, 4)]
		return (r, g, b)
	rgb_match = re.match(r'^rgb\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)$', s)
	if rgb_match:
		r, g, b = map(int, rgb_match.groups())
		return (r, g, b)
	raise ValueError(s)

class ColourCache(AutoCache):

	def _obtain(self, url):
		with niquests.get(url, headers=Request.header(), stream=True) as resp:
			mime = resp.headers.get("Content-Type", "")
			if "text/html" in mime:
				it = resp.iter_content(65536)
				s = as_str(next(it))
				try:
					bc = s.split("background-color:", 1)[1]
				except IndexError:
					c = (255, 255, 255)
				else:
					bc = bc.replace(";", " ").split(None, 1)[0]
					c = parse_colour(bc)[:3]
			else:
				im = Image.open(resp.raw)
				c = get_colour(im)[:3]
		print("GC:", url, c)
		return c

	def obtain(self, url):
		if not url:
			return (0, 0, 0)
		if isinstance(url, byte_like):
			im = Image.open(io.BytesIO(url))
			return get_colour(im)[:3]
		try:
			return self.retrieve(unyt(url), self._obtain, url)
		except Exception as ex:
			print(repr(ex))
			return (0, 0, 0)

	def obtain_heart(self, url):
		rgb = self.obtain(url)
		return get_closest_heart(rgb)


class AttachmentCache(AutoCache):
	min_size = 262144
	max_size = CACHE_FILESIZE
	attachment_count = 10
	embed_count = 10
	discord_token = AUTH["discord_token"]
	alt_token = AUTH.get("alt_token", discord_token)
	headers = {"Content-Type": "application/json", "Authorization": "Bot " + discord_token}
	alt_headers = {"Content-Type": "application/json", "Authorization": "Bot " + alt_token}
	exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)
	sess = None
	fut = None
	queue = []
	channels = set()

	def __init__(self, *args, secondary=None, tertiary=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.init(secondary, tertiary)

	def init(self, secondary=None, tertiary=None):
		self.channels.clear()
		self.channels.update(AUTH.get("proxy_channels", ()))
		if secondary:
			self.secondary = AutoCache(secondary, size_limit=128 * 1073741824, stale=86400 * 7, timeout=86400 * 30)
		if tertiary:
			self.tertiary = AutoCache(tertiary, size_limit=1073741824, stale=86400 * 7, timeout=86400 * 30)
		return self.channels

	@property
	def last(self):
		return self.setdefault("__last__", set())
	@last.setter
	def last(self, value):
		self["__last__"] = value

	@tracebacksuppressor
	def update_queue(self):
		ec = self.embed_count
		while self.queue:
			tasks, self.queue = self.queue[:ec], self.queue[ec:]
			urls = [task[1].split("?url=", 1)[-1].replace("?", "&").replace("%", "&").split("&", 1)[0] for task in tasks]
			embeds = [dict(image=dict(url=url)) for url in urls]
			last = self.last
			n = 0
			resp = None
			try:
				if last:
					tup = choice(last)
					cid, mid, n = tup
					last.discard(tup)
					heads = self.alt_headers if n else self.headers
					try:
						resp = requests.patch(
							f"https://discord.com/api/{api}/channels/{cid}/messages/{mid}",
							data=json_dumps(dict(embeds=embeds)),
							headers=heads,
							timeout=3,
						)
					except Exception as ex:
						print(repr(ex))
						last.add(tup)
						self.last = last
				if not resp:
					cid = choice(self.channels)
					n = random.randint(0, 2)
					heads = self.alt_headers if n else self.headers
					resp = requests.post(
						f"https://discord.com/api/{api}/channels/{cid}/messages",
						data=json_dumps(dict(embeds=embeds)),
						headers=heads,
						timeout=4.5,
					)
				resp.raise_for_status()
				message = resp.json()
				mid = message["id"]
			except Exception as ex:
				if resp is not None:
					print(resp.content)
				for task in tasks:
					task[0].set_exception(ex)
				continue
			esubmit(self.clear_last, (cid, mid, n))
			for emb in message["embeds"]:
				url = emb["image"]["url"].rstrip("&")
				tasks.pop(0)[0].set_result(url)
				if not tasks:
					break
			for task in tasks:
				task[0].set_exception(ConnectionError(404, "Missing attachment embed!"))
		self.fut = None

	def clear_last(self, tup=None):
		time.sleep(1)
		with tracebacksuppressor:
			last = self.last
			if tup:
				last.add(tup)
			for (c, k, n) in tuple(last):
				if utc() - snowflake_time_2(int(k)).timestamp() > 86400 * 6:
					last.remove((c, k, n))
					with tracebacksuppressor:
						heads = self.headers if c not in self.channels else self.alt_headers
						resp = requests.delete(
							f"https://discord.com/api/{api}/channels/{c}/messages/{k}",
							headers=heads,
							timeout=5,
						)
						resp.raise_for_status()
			self.last = last

	def preserve(self, url, mid=0, minimise=False):
		if not is_discord_attachment(url):
			return url
		early = 43200 + 60
		if not discord_expired(url, early):
			cid, mid, aid, fn = split_url(url, mid)
			key = aid
			self[key] = url
		return shorten_attachment(url, mid, minimise=minimise)

	async def get_direct(self, c_id, m_id, a_id=None):
		if not m_id:
			raise LookupError("Insufficient information to retrieve attachment.")
		heads = self.headers if c_id not in self.channels else self.alt_headers
		data = await Request.aio(
			f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}",
			headers=heads,
			bypass=False,
			json=True,
		)
		for i, a in enumerate(data["attachments"]):
			if a_id in (None, m_id, i, int(a["id"])):
				return a["url"].rstrip("&")
		raise ConnectionError(404, a_id)

	async def get_attachment(self, c_id, m_id, a_id, fn):
		ac = self.attachment_count
		if not self.channels or not fn or a_id < ac:
			if not m_id:
				raise LookupError("Insufficient information to retrieve attachment.")
			return await self.get_direct(c_id, m_id, a_id)
		fut = Future()
		url, _ = merge_url(c_id, m_id, a_id, fn)
		task = [fut, url]
		self.queue.append(task)
		if self.fut is None or self.fut.done() or len(self.queue) > ac:
			self.fut = self.exc.submit(self.update_queue)
		try:
			return await asyncio.wait_for(wrap_future(fut), timeout=8)
		except asyncio.TimeoutError:
			if task in self.queue:
				if self.fut is None or self.fut.done() or len(self.queue) > ac:
					self.fut = self.exc.submit(self.update_queue)
				return await asyncio.wait_for(wrap_future(fut), timeout=8)
		return await self.get_direct(c_id, m_id, a_id)

	async def obtain(self, c_id=None, m_id=None, a_id=None, fn=None, url=None):
		if url:
			c_id, m_id, a_id, fn = split_url(url, m_id)
		ac = self.attachment_count
		if isinstance(c_id, str) and not c_id.isnumeric():
			c_id = int.from_bytes(b64(c_id), "big")
			m_id = int.from_bytes(b64(m_id), "big")
			a_id = int.from_bytes(b64(a_id), "big")
		if not a_id or a_id == m_id:
			a_id = 0
		if a_id >= ac:
			key = a_id
			early = 43200 + 60
		else:
			key = m_id * ac + a_id
			early = 86400 - 60
		resp = await self.aretrieve(key, self.get_attachment, c_id, m_id, a_id, fn)
		try:
			assert isinstance(resp, str) and not discord_expired(resp, early)
		except AssertionError:
			resp = await self._aretrieve(key, self.get_attachment, c_id, m_id, a_id, fn)
		return resp

	async def get_attachments(self, path):
		size_mb, c_id, m_ids = ungroup_attachments(path)
		urls = []
		for m_id in m_ids:
			data = await Request.aio(
				f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}",
				headers=self.headers,
				bypass=False,
				json=True,
			)
			if data.get("attachments"):
				for a in data["attachments"]:
					urls.append(a["url"].rstrip("&"))
			else:
				for e in data["embeds"]:
					try:
						url = e["author"]["icon_url"]
					except KeyError:
						url = e["image"]["url"]
					urls.append(url.rstrip("&"))
		return urls, size_mb * 1048576

	async def obtains(self, path):
		resp = await self.aretrieve(path, self.get_attachments, path)
		try:
			assert isinstance(resp, tuple) and not discord_expired(resp[0][0], 43200 - 120)
		except AssertionError:
			resp = await self._aretrieve(path, self.get_attachments, path)
		return resp

	async def _download(self, url, m_id=None, timeout=12):
		try:
			target = url
			raw_fn = temporary_file(url2ext(url))
			if is_discord_url(url):
				if is_discord_attachment(url):
					target = await self.obtain(url=url, m_id=m_id)
			elif is_miza_url(url):
				if "/u/" in url:
					c_id, m_id2, a_id, fn = expand_attachment(url)
					target = await self.obtain(c_id, m_id2 or m_id, a_id, fn)
					fn, head = await asubmit(download_file, target, filename=raw_fn, timeout=timeout, return_headers=True)
					self.tertiary[url] = head
					return open(fn, "rb")
				elif "/c/" in url:
					path = url.split("/c/", 1)[-1].split("/", 1)[0]
					urls = await self.obtains(path)
					fn, head = await asubmit(download_file, *urls, filename=raw_fn, timeout=timeout, return_headers=True)
					self.tertiary[url] = head
					return open(fn, "rb")
			try:
				f, head = await streamshatter.shatter_request(target, filename=raw_fn, log_progress=False, timeout=timeout, max_attempts=3, return_headers=True)
			except niquests.exceptions.HTTPError as ex:
				code, msg = ex.response.status_code, ex.response.reason
				raise ConnectionError(code, msg)
			self.tertiary[url] = head
		except ConnectionError:
			raise
		except:
			traceback.print_exc()
			raise
		return f
	async def download(self, url, m_id=None, filename=None, read=False, return_headers=False):
		url = unyt(url)
		if (match := scraper_blacklist.search(url)):
			raise InterruptedError(match)
		fp = await self.secondary.aretrieve(url, self._download, url, m_id=m_id, timeout=16, _read=True)
		if return_headers:
			headers = await self.scan_headers(url, m_id=m_id)
		if filename:
			try:
				if isinstance(filename, bool):
					if hasattr(fp, "name") and os.path.exists(fp.name):
						return (fp.name, headers) if return_headers else fp.name
					filename = temporary_file(url2ext(url))
				with open(filename, "wb") as f2:
					await asubmit(shutil.copyfileobj, fp, f2, 262144)
				return (filename, headers) if return_headers else filename
			finally:
				fp.close()
		if read:
			return (fp, headers) if return_headers else fp
		try:
			if hasattr(fp, "name") and os.path.exists(fp.name):
				data = await read_file_a(fp.name)
				return (data, headers) if return_headers else data
			data = await asubmit(fp.read)
			return (data, headers) if return_headers else data
		finally:
			fp.close()

	async def _scan_headers(self, url, m_id=None):
		if is_discord_attachment(url):
			url = await self.obtain(url=url, m_id=m_id)
		headers = await asubmit(header_test, url)
		return dict(headers)
	async def scan_headers(self, url, m_id=None):
		url = unyt(url)
		if (match := scraper_blacklist.search(url)):
			raise InterruptedError(match)
		return await self.tertiary.aretrieve(url, self._scan_headers, url, m_id=m_id)

	async def delete(self, c_id, m_id, url=None):
		if url:
			c_id, m_id, *_ = split_url(url, m_id)
		self.sess = self.sess or aiohttp.ClientSession()
		heads = dict(choice((self.headers, self.alt_headers)))
		url = f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}"
		resp = await self.sess.request("DELETE", url, headers=heads, timeout=120)
		resp.raise_for_status()

	async def create(self, *data, filename=None, channel=None, content="", collapse=True, editable=False, minimise=False):
		if not self.channels:
			raise RuntimeError("Proxy channel list required.")
		ac = self.attachment_count
		self.sess = self.sess or aiohttp.ClientSession()
		form_data = aiohttp.FormData(quote_fields=False)
		filename = filename or "b"
		out = []
		while data:
			temp, data = data[:ac], data[ac:]
			for fp in temp:
				if hasattr(fp, "seek"):
					fp.seek(0)
			payload = dict(
				content=content,
				attachments=[dict(
					id=i,
					filename=filename if not i else "b",
				) for i in range(len(temp))],
			)
			form_data.add_field(name="payload_json", value=json_dumpstr(payload))
			for i, b in enumerate(temp):
				form_data.add_field(
					name=f"files[{i}]",
					value=b,
					filename=filename if not i else "b",
					content_type="application/octet-stream",
				)
			cid = getattr(channel, "id", channel) if channel else choice(self.channels)
			url = f"https://discord.com/api/{api}/channels/{cid}/messages"
			if editable:
				heads = dict(self.headers)
			else:
				heads = dict(choice((self.headers, self.alt_headers)))
			heads.pop("Content-Type")
			resp = await self.sess.request("POST", url, headers=heads, data=form_data, timeout=120)
			resp.raise_for_status()
			message = await resp.json()
			cid = int(message["channel_id"])
			mid = int(message["id"])
			for i, a in enumerate(message["attachments"]):
				aid = i if editable else int(a["id"])
				fn = a["filename"]
				out.append(shorten_attachment(cid, mid, aid, fn, minimise=minimise))
			filename = "b"
		if len(out) == 1 and collapse:
			return out[0]
		return out

	async def edit(self, c_id, m_id, *data, url=None, filename=None, content="", collapse=True, minimise=False):
		if url:
			c_id, m_id, *_ = split_url(url, m_id)
		ac = self.attachment_count
		self.sess = self.sess or aiohttp.ClientSession()
		form_data = aiohttp.FormData(quote_fields=False)
		filename = filename or "b"
		out = []
		assert len(data) <= ac
		payload = dict(
			content=content,
			attachments=[dict(
				id=i,
				filename=filename if not i else "b",
			) for i in range(len(data))],
		)
		form_data.add_field(name="payload_json", value=json_dumpstr(payload))
		for i, b in enumerate(data):
			form_data.add_field(
				name=f"files[{i}]",
				value=b,
				filename=filename if not i else "b",
				content_type="application/octet-stream",
			)
		cid, mid = c_id, m_id
		url = f"https://discord.com/api/{api}/channels/{cid}/messages/{mid}"
		heads = self.headers
		heads.pop("Content-Type")
		resp = await self.sess.request("PATCH", url, headers=heads, data=form_data, timeout=120)
		resp.raise_for_status()
		message = await resp.json()
		cid = int(message["channel_id"])
		mid = int(message["id"])
		for i, a in enumerate(message["attachments"]):
			aid = i
			fn = a["filename"]
			out.append(shorten_attachment(cid, mid, aid, fn, minimise=minimise))
		if len(out) == 1 and collapse:
			return out[0]
		return out


audio_meta_cache = AutoCache(f"{CACHE_PATH}/audio_meta", stale=0, timeout=300)
def _audio_meta(path, _timeout=12) -> dict:
	command = (
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_streams",
		"-show_format",
		path,
	)
	lines = ()
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = esubmit(proc.wait, timeout=_timeout)
		_res = fut.result(timeout=_timeout)
		lines = [line for line in reversed(proc.stdout.read().decode("utf-8").splitlines()) if "=" in line]
	except Exception:
		try:
			proc.terminate()
		except Exception:
			pass
		traceback.print_exc()
	info = {(t := line.split("=", 1))[0]: t[1] for line in lines if "=" in line and line.rsplit("=", 1)[-1] not in ("N/A", "0/0")}
	name = info.get("TAG:album") or info.get("TAG:icy-name")
	title = info.get("TAG:title") or info.get("TAG:StreamTitle")
	return dict(
		name=f"{name}: {title}" if name and title else None,
		duration=(dur := float(info.get("duration") or int(info.get("duration_ts") or 0) * fractions.Fraction(info.get("time_base") or 0))) or None,
		format=info.get("format_name") or url2ext(info.get("filename") or path),
		codec=info.get("codec_name", "auto"),
		channels=int(info.get("channels") or 1),
		bitrate=float(info.get("bit_rate") or float(info.get("size", 0)) * 8 / (dur or 1) or 0),
		sample_rate=int(info.get("sample_rate") or info.get("TAG:icy-samplerate") or 0),
	)
def audio_meta(path, _timeout=12) -> cdict:
	return cdict(audio_meta_cache.retrieve(path, _audio_meta, path, _timeout=_timeout))

colour_cache = ColourCache(f"{CACHE_PATH}/colour", stale=86400, timeout=86400 * 7)
attachment_cache = AttachmentCache(
	f"{CACHE_PATH}/attachment",
	secondary=f"{CACHE_PATH}/attachment_contents",
	tertiary=f"{CACHE_PATH}/attachment_headers",
	stale=0,
	timeout=3600 * 18,
)


def acquire_from_archive(url, arcnames, filenames):
	with tracebacksuppressor:
		try:
			for fn in filenames:
				args = [fn, "--version"]
				subprocess.run(args)
		except FileNotFoundError:
			fn = await_fut(attachment_cache.download(url, filename=True))
			assert os.path.exists(fn) and os.path.getsize(fn), "Download unsuccessful!"
			if not zipfile.is_zipfile(fn):
				assert len(filenames) == 1
				fn = filenames[0]
				path = fn.replace("\\", "/").rsplit("/", 1)[0]
				os.makedirs(path, exist_ok=True)
				shutil.copyfile(fn, fn)
				if os.name != "nt":
					subprocess.run(("chmod", "777", fn))
				return
			with zipfile.ZipFile(fn) as z:
				for an, fn in zip(arcnames, filenames):
					path = fn.replace("\\", "/").rsplit("/", 1)[0]
					os.makedirs(path, exist_ok=True)
					with z.open(an, "r") as fi:
						with open(fn, "wb") as fo:
							shutil.copyfileobj(fi, fo)
					if os.name != "nt":
						subprocess.run(("chmod", "777", fn))

def download_binary_dependencies():
	if os.name == "nt":
		acquire_from_archive("https://eternallybored.org/misc/gifsicle/releases/gifsicle-1.95-win64.zip", ["gifsicle.exe"], ["binaries/gifsicle.exe"])
	else:
		print("gifsicle binary not implemented in loader! Feature will not be available.")

	if os.name == "nt":
		acquire_from_archive("https://github.com/AOMediaCodec/libavif/releases/download/v1.3.0/windows-artifacts.zip", ["avifenc.exe"], ["binaries/avifenc.exe"])
	else:
		acquire_from_archive("https://github.com/AOMediaCodec/libavif/releases/download/v1.3.0/linux-artifacts.zip", ["avifenc"], ["binaries/avifenc"])

	# if os.name == "nt":
	# 	acquire_from_archive("https://pngquant.org/pngquant-windows.zip", ["pngquant/pngquant.exe"], ["binaries/pngquant.exe"])
	# else:
	# 	print("pngquant binary not implemented in loader! Feature will not be available.")
	# 	acquire_from_archive("https://pngquant.org/pngquant-linux.tar.bz2", ["pngquant"], ["binaries/pngquant"])

	if os.name == "nt":
		acquire_from_archive("https://mizabot.xyz/u/_bjH2NMEAslNEANIlNsAJhJJoNFl/ecm.exe", [], ["binaries/ecm.exe"])
	else:
		acquire_from_archive("https://mizabot.xyz/u/vaWqov8EAslNEANIlpskJhpJMplF/ecm", [], ["binaries/ecm"])