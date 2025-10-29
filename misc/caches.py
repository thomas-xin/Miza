import asyncio
import concurrent.futures
import os
import random
import re
import shutil
import subprocess
import time
import zipfile
import aiofiles
import aiohttp
import numpy as np
from PIL import Image
from misc.types import utc, as_str
from misc.asyncs import asubmit, esubmit, wrap_future, await_fut, Future
from misc.smath import get_closest_heart
from misc.util import (
    CACHE_FILESIZE, CACHE_PATH, TEMP_PATH, AUTH, Request, api, AutoCache, download_file, header_test,
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
		req = Request.compat_session if is_discord_url(url) else Request.session
		with req.get(url, headers=Request.header(), stream=True) as resp:
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
					last.discard(tup)
					cid, mid, n = tup
					heads = self.alt_headers if n else self.headers
					try:
						resp = Request.compat_session.patch(
							f"https://discord.com/api/{api}/channels/{cid}/messages/{mid}",
							data=json_dumps(dict(embeds=embeds)),
							headers=heads,
							timeout=3,
						)
					except Exception as ex:
						print(repr(ex))
				if not resp:
					cid = choice(self.channels)
					n = random.randint(0, 2)
					heads = self.alt_headers if n else self.headers
					resp = Request.compat_session.post(
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
			esubmit(self.set_last, (cid, mid, n))
			for emb in message["embeds"]:
				url = emb["image"]["url"].rstrip("&")
				tasks.pop(0)[0].set_result(url)
				if not tasks:
					break
			for task in tasks:
				task[0].set_exception(ConnectionError(404, "Missing attachment embed!"))
		self.fut = None

	def set_last(self, tup):
		time.sleep(1)
		with tracebacksuppressor:
			last = self.last
			last.add(tup)
			for (c, k, n) in tuple(last):
				if utc() - snowflake_time_2(int(k)).timestamp() > 86400 * 6:
					last.remove((c, k, n))
					heads = self.headers if c not in self.channels else self.alt_headers
					resp = Request.compat_session.delete(
						f"https://discord.com/api/{api}/channels/{c}/messages{k}",
						headers=heads,
						timeout=5,
					)
					resp.raise_for_status()

	async def get_direct(self, c_id, m_id, a_id=None):
		if not m_id:
			raise LookupError("Insufficient information to retrieve attachment.")
		heads = self.headers if c_id not in self.channels else self.alt_headers
		data = await Request(
			f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}",
			headers=heads,
			bypass=False,
			aio=True,
			json=True,
		)
		for i, a in enumerate(data["attachments"]):
			if not a_id or a_id in (i, int(a["id"])):
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
		if not a_id:
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
			data = await Request(
				f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}",
				headers=self.headers,
				bypass=False,
				aio=True,
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

	async def _download(self, url, m_id=None):
		raw_fn = temporary_file(url2ext(url))
		if is_discord_url(url):
			if is_discord_attachment(url):
				url = await self.obtain(url=url, m_id=m_id)
			fn, head = await asubmit(download_file, url, filename=raw_fn, return_headers=True)
			self.tertiary[url] = head
			return open(fn, "rb")
		if is_miza_url(url):
			if "/u/" in url:
				c_id, m_id, a_id, fn = expand_attachment(url)
				target = await self.obtain(c_id, m_id, a_id, fn)
				fn, head = await asubmit(download_file, target, filename=raw_fn, return_headers=True)
				self.tertiary[url] = head
				return open(fn, "rb")
			elif "/c/" in url:
				path = url.split("/c/", 1)[-1].split("/", 1)[0]
				urls = await self.obtains(path)
				fn, head = await asubmit(download_file, *urls, filename=raw_fn, return_headers=True)
				self.tertiary[url] = head
				return open(fn, "rb")
			fn, head = await asubmit(download_file, url, filename=raw_fn, return_headers=True)
			self.tertiary[url] = head
			return open(fn, "rb")
		args = ["streamshatter", "--no-log-progress", "-c", TEMP_PATH, url, raw_fn]
		proc = await asyncio.create_subprocess_exec(*args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
		await proc.wait()
		if proc.returncode:
			err = await proc.stderr.read()
			line = as_str(err.strip().rsplit(b"\n", 1)[-1])
			if line.startswith("niquests.exceptions.HTTPError:"):
				try:
					curr = line.split(":", 1)[1].strip()
					code, msg = curr.split(None, 1)
				except ValueError:
					raise ConnectionError(502, line)
				else:
					raise ConnectionError(code, msg)
			raise ConnectionError(501, line)
		fn = raw_fn
		assert os.path.exists(fn)
		return open(fn, "rb")
	async def download(self, url, m_id=None, filename=None, read=False):
		url = unyt(url)
		if (match := scraper_blacklist.search(url)):
			raise InterruptedError(match)
		fp = await self.secondary.aretrieve(url, self._download, url, m_id=m_id, _read=True)
		if filename:
			try:
				if isinstance(filename, bool):
					if hasattr(fp, "name") and os.path.exists(fp.name):
						return fp.name
					filename = temporary_file(url2ext(url))
				with open(filename, "wb") as f2:
					await asubmit(shutil.copyfileobj, fp, f2, 262144)
				return filename
			finally:
				fp.close()
		if read:
			return fp
		try:
			if hasattr(fp, "name") and os.path.exists(fp.name):
				async with aiofiles.open(fp.name, "rb") as f:
					return await f.read()
			return await asubmit(fp.read)
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
		try:
			return self.tertiary[url]
		except KeyError:
			return await self.tertiary.aretrieve(url, self._scan_headers, url, m_id=m_id)

	async def delete(self, c_id, m_id, url=None):
		if url:
			c_id, m_id, *_ = split_url(url, m_id)
		self.sess = self.sess or aiohttp.ClientSession()
		heads = dict(choice((self.headers, self.alt_headers)))
		url = f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}"
		resp = await self.sess.request("DELETE", url, headers=heads, timeout=120)
		resp.raise_for_status()

	async def create(self, *data, filename=None, channel=None, content="", collapse=True, editable=False):
		if not self.channels:
			raise RuntimeError("Proxy channel list required.")
		ac = self.attachment_count
		self.sess = self.sess or aiohttp.ClientSession()
		form_data = aiohttp.FormData(quote_fields=False)
		filename = filename or "b"
		out = []
		if hasattr(data, "read"):
			if hasattr(data, "seek"):
				data = data.seek(0)
			data = await asubmit(data.read)
		while data:
			temp, data = data[:ac], data[ac:]
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
				out.append(shorten_attachment(cid, mid, aid, fn))
			filename = "b"
		if len(out) == 1 and collapse:
			return out[0]
		return out

	async def edit(self, c_id, m_id, *data, url=None, filename=None, content="", collapse=True):
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
			out.append(shorten_attachment(cid, mid, aid, fn))
		if len(out) == 1 and collapse:
			return out[0]
		return out


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


colour_cache = ColourCache(f"{CACHE_PATH}/colour", stale=86400, timeout=86400 * 7)
attachment_cache = AttachmentCache(
	f"{CACHE_PATH}/attachment",
	secondary=f"{CACHE_PATH}/attachment_contents",
	tertiary=f"{CACHE_PATH}/attachment_headers",
	stale=0,
	timeout=3600 * 18,
)