import aiohttp
import concurrent.futures
import random
import time
import requests
from misc.types import utc
from misc.asyncs import esubmit, wrap_future, Future
from misc.util import (
    Cache, AUTH, Request, api,
    tracebacksuppressor, choice, json_dumps, json_dumpstr, b64,
    snowflake_time_2, shorten_attachment, merge_url, split_url, discord_expired, url2fn
)


class AttachmentCache(Cache):
	min_size = 262144
	max_size = 25165824
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
	last = set()

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.init()

	def init(self):
		self.channels.clear()
		self.channels.update(AUTH.get("proxy_channels", ()))
		return self.channels

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
					resp = requests.patch(
						f"https://discord.com/api/{api}/channels/{cid}/messages/{mid}",
						data=json_dumps(dict(embeds=embeds)),
						headers=heads,
					)
				else:
					cid = choice(self.channels)
					n = random.randint(0, 1)
					heads = self.alt_headers if n else self.headers
					resp = requests.post(
						f"https://discord.com/api/{api}/channels/{cid}/messages",
						data=json_dumps(dict(embeds=embeds)),
						headers=heads,
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
			for task, emb in zip(tasks, message["embeds"]):
				url = emb["image"]["url"]
				task[0].set_result(url)
			# futs = []
			# for emb in message["embeds"]:
			# 	url = emb["image"]["url"]
			# 	futs.append(esubmit(requests.head, url, verify=False))
			# for task, fut in zip(tasks, futs):
			# 	resp = fut.result()
			# 	try:
			# 		resp.raise_for_status()
			# 	except Exception as ex:
			# 		task[0].set_exception(ex)
			# 	else:
			# 		task[0].set_result(resp.url)

	def set_last(self, tup):
		time.sleep(1)
		last = self.last
		last.add(tup)
		for (c, k, n) in tuple(last):
			if utc() - snowflake_time_2(int(k)).timestamp() > 3600 * 12:
				last.remove((c, k, n))

	async def get_attachment(self, c_id, m_id, a_id, fn):
		ac = self.attachment_count
		heads = self.headers if c_id not in self.channels else self.alt_headers
		if not self.channels or not fn or a_id < ac:
			if not m_id:
				raise LookupError("Insufficient information to retrieve attachment.")
			data = await Request(
				f"https://discord.com/api/{api}/channels/{c_id}/messages/{m_id}",
				headers=heads,
				bypass=False,
				aio=True,
				json=True,
			)
			for i, a in enumerate(data["attachments"]):
				if a_id in (i, int(a["id"])):
					return a["url"]
			raise KeyError(a_id)
		fut = Future()
		url, _ = merge_url(c_id, m_id, a_id, fn)
		task = [fut, url]
		self.queue.append(task)
		if self.fut is None or self.fut.done() or len(self.queue) > ac:
			self.fut = self.exc.submit(self.update_queue)
		return await wrap_future(fut)

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
		try:
			resp = self[key]
			assert isinstance(resp, str) and not discord_expired(resp, early) and url2fn(resp) == fn
		except KeyError:
			resp = await self.retrieve_from(key, self.get_attachment, c_id, m_id, a_id, fn)
		except AssertionError:
			resp = await self.get_attachment(c_id, m_id, a_id, fn)
			self[key] = resp
		return resp

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

attachment_cache = AttachmentCache(timeout=3600 * 12 , trash=0, persist="attachment.cache")
upload_cache = Cache(timeout=86400 * 30, trash=1, persist="upload.cache")
download_cache = Cache(timeout=60, trash=0, persist="download.cache")


# class ChunkSystem(FileHashDict):

# 	session = requests.Session()

# 	def combine(self, urls):
# 		size = 0
# 		url = urls[0]
# 		if len(urls) > 1:
# 			last = urls[-1]
# 			fut = esubmit(self.session.head(
# 				last,
# 				headers=Request.header(),
# 				stream=True,
# 				verify=False,
# 				timeout=60,
# 			))
# 		resp = self.session.get(
# 			url,
# 			headers=Request.header(),
# 			stream=True,
# 			verify=False,
# 			timeout=60,
# 		)
# 		if resp.headers.get("Content-Type", "application/octet-stream") == "application/octet-stream":
# 			it = resp.iter_content(65536)
# 			b = next(it)
# 			mime = magic.from_buffer(b)
# 		else:
# 			mime = resp.headers["Content-Type"]
# 		size += int(resp.headers.get("Content-Length") or resp.headers.get("x-goog-stored-content-length", 1))
# 		if len(urls) > 1:
# 			size += attachment_cache.max_size * len(urls) - 2
# 			resp2 = fut.result()
# 			size += int(resp2.headers.get("Content-Length") or resp2.headers.get("x-goog-stored-content-length", 1))
# 		return cdict(
# 			filename=url2fn(url),
# 			size=size,
# 			mimetype=mime,
# 			chunks=urls,
# 		)

# chunk_system = ChunkSystem(path=TEMP_PATH + "/chunks")