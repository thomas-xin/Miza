import asyncio
from collections import deque
from concurrent.futures import Future
from contextlib import suppress
import io
import itertools
from math import ceil, inf, isfinite
import os
import random
import re
import subprocess
import time
from traceback import print_exc
from urllib.parse import quote_plus
import zipfile
import orjson
import niquests
import yt_dlp as ytd
assert hasattr(ytd.extractor, "youtube")
from .types import alist, as_str, cdict, fcdict, full_prune, json_dumps, round_min, to_alphanumeric, tracebacksuppressor
from .smath import time_parse
from .asyncs import esubmit
from .util import (
	python, compat_python, shuffle, utc, leb128, string_similarity, verify_search, json_dumpstr, get_free_port,
	find_urls, url2fn, url2ext, discord_expired, expired, shorten_attachment, unyt, html_decode,
	is_image, is_url, is_discord_attachment, is_miza_url, is_miza_attachment, is_youtube_url, is_spotify_url, AUDIO_FORMS,
	EvalPipe, PipedProcess, AutoCache, Request, Semaphore, TEMP_PATH, CACHE_PATH, magic, rename, temporary_file, replace_ext, select_and_loads, extract_archive,
)
from .caches import audio_meta, attachment_cache

# Gets the best icon/thumbnail for a queue entry.
def get_best_icon(entry):
	try:
		return entry["thumbnail"]
	except KeyError:
		try:
			return entry["icon"]
		except KeyError:
			pass
	try:
		thumbnails = list(filter(bool, entry["thumbnails"]))
		if not thumbnails:
			raise KeyError(thumbnails)
	except KeyError:
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		if not url:
			return ""
		if is_discord_attachment(url):
			if not is_image(url):
				return "https://cdn.discordapp.com/embed/avatars/0.png"
		if is_youtube_url(url):
			if "?v=" in url:
				vid = url.split("?v=", 1)[-1]
			else:
				vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
			entry["icon"] = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
			return entry["icon"]
		if is_miza_url(url):
			return "https://mizabot.xyz/static/mizaleaf.png"
		return ""
	return sorted(
		thumbnails,
		key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)),
		reverse=True,
	)[0]["url"]
# Gets the best audio file download link for a queue entry.
def get_best_audio(entry):
	try:
		a = entry["audio"]
	except KeyError:
		pass
	else:
		if isinstance(a, str):
			a = [a]
		if len(a) < 2:
			return a[0], None, 0
		if len(a) < 3:
			return a[0], a[1], 0
		return a[:3]
	best = (-inf,)
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	url = None
	cdc = None
	ac = 0
	for fmt in fmts:
		q = (fmt.get("protocol", "m3u8_native") != "m3u8_native", fmt.get("acodec") in ("opus", "vorbis"), fmt.get("vcodec") in (None, "none") and fmt.get("acodec") not in (None, "none"), fmt.get("abr", 0) or 0, fmt.get("tbr", 0) or 0)
		u = as_str(fmt["url"])
		if q > best:
			ac = fmt.get("audio_channels", 0)
			cdc = fmt.get("acodec")
			best = q
			url = fmt["url"]
	return url, cdc, ac
# Gets the best video file download link for a queue entry.
def get_best_video(entry, hq=True):
	try:
		return entry["video"]
	except KeyError:
		pass
	best = (-inf,)
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	url = None
	cdc = None
	for fmt in fmts:
		q = (
			fmt.get("protocol", "m3u8_native") != "m3u8_native",
			fmt.get("vcodec") not in (None, "none"),
			-abs(fmt["fps"] - (90 if hq else 42)) if isinstance(fmt.get("fps"), (int, float)) else -inf,
			-abs(fmt["height"] - (1600 if hq else 720)) if isinstance(fmt.get("height"), (int, float)) else -inf,
			fmt["tbr"] if isinstance(fmt.get("tbr"), (int, float)) else -inf,
		)
		u = as_str(fmt["url"])
		if q > best:
			cdc = fmt.get("vcodec")
			best = q
			url = fmt["url"]
	return url, cdc
def get_best_lyrics(resp):
	if "description" in resp:
		lyr = []
		spl = resp["description"].splitlines()
		for i, line in enumerate(spl):
			if to_alphanumeric(full_prune(line)).strip() == "lyrics":
				para = []
				for j, line in enumerate(spl[i + 1:]):
					line = line.strip()
					if line and not to_alphanumeric(line).strip():
						break
					if find_urls(line):
						if para and para[-1].endswith(":") or para[-1].startswith("#"):
							para.pop(-1)
						break
					para.append(line)
				if len(para) >= 3:
					lyr.extend(para)
		lyrics = "\n".join(lyr).strip()
		if lyrics:
			print("lyrics_raw", lyrics)
			return lyrics
	if resp.get("automatic_captions"):
		lang = "en"
		if "formats" in resp:
			lang = None
			for fmt in resp["formats"]:
				if fmt.get("language"):
					lang = fmt["language"]
					break
		if lang in resp["automatic_captions"]:
			for cap in shuffle(resp["automatic_captions"][lang]):
				if "json" in cap["ext"]:
					break
			with tracebacksuppressor:
				data = Request(cap["url"], json=True, timeout=18)
				assert isinstance(data, dict)
				lyr = []
				for event in data["events"]:
					para = "".join(seg.get("utf8", "") for seg in event.get("segs", ()))
					lyr.append(para)
				lyrics = "".join(lyr).strip()
				if lyrics:
					print("lyrics_captions", lyrics)
					return lyrics


class AudioDownloader:

	def __init__(self, workers=1):
		self.session = niquests.Session()
		self.search_cache = AutoCache(f"{CACHE_PATH}/ytdl.search", stale=300, timeout=86400, desync=0.05)
		self.thumbnail_cache = AutoCache(f"{CACHE_PATH}/ytdl.thumbnail", stale=300, timeout=86400 * 7, desync=0.05)
		self.extract_cache = AutoCache(f"{CACHE_PATH}/ytdl.extract", stale=60, timeout=120, desync=0.05)
		self.futs = [
			esubmit(self.set_cookie),
		]
		self.worker_count = workers
		self.workers = alist()
		self.start_workers()

	def start_workers(self):
		while len(self.workers) < self.worker_count:
			port = get_free_port()
			worker = EvalPipe.connect(
				[python, "-m", "misc.x_ytdl", str(port)],
				port,
				glob=globals(),
				independent=False,
			)
			self.workers.append(worker)

	def restart(self):
		self.futs = [
			esubmit(self.set_cookie),
		]
		for worker in self.workers:
			worker.kill()
		self.workers.clear()
		return self.start_workers()

	def submit(self, s):
		if not self.workers:
			self.start_workers()
		return self.workers.next().submit(s)

	def run(self, s, timeout=None, priority=False):
		if not self.workers:
			self.start_workers()
		return self.workers.next().run(s, timeout=timeout, priority=priority)

	def extract_info(self, url, download=False, process=True):
		resp = self.extract_cache.retrieve(url, self.run, f"extract_info({json_dumpstr(url)},download={download},process={process})")
		return resp

	def get_thumbnail(self, entry, pos=0):
		url = entry.get("url")
		if not is_youtube_url(url):
			return get_best_icon(entry)
		info = self.extract_info(url)
		b = self.thumbnail_cache.retrieve(url, self.run, f"get_full_storyboard({repr(info)})")
		if isinstance(b, str):
			return b
		with io.BytesIO(b) as b:
			with zipfile.ZipFile(b, "r") as z:
				filelist = z.filelist
				found = [((diff := pos - int(zi.filename.split(".", 1)[0])) < 0, abs(diff), zi.filename) for zi in filelist]
				closest = min(found)
				return z.read(closest[-1])

	def set_cookie(self):
		self.youtube_base = "CONSENT=YES+cb.20210328-17-p0.en+FX"
		s = self.session.get("https://www.youtube.com").text
		assert isinstance(s, str)
		if "<title>Before you continue to YouTube</title>" in s:
			s = s.split('<input type="hidden" name="v" value="', 1)[-1]
			s = s[:s.index('">')].rsplit("+", 1)[0]
			self.youtube_base = f"CONSENT=YES+{s}"

	@property
	def youtube_header(self):
		headers = Request.header()
		if self.youtube_base:
			headers["Cookie"] = self.youtube_base + "%03d" % random.randint(0, 999) + ";"
		return headers

	# Returns a list of formatted queue entries from a YouTube playlist renderer.
	def extract_playlist_items(self, items):
		token = None
		out = []
		for data in items:
			try:
				video = data["playlistVideoRenderer"]
			except KeyError:
				try:
					continuation_endpoint = data["continuationItemRenderer"]["continuationEndpoint"]
					try:
						token = continuation_endpoint["commandExecutorCommand"]["commands"][-1]["continuationCommand"]["token"]
					except LookupError:
						token = continuation_endpoint["continuationCommand"]["token"]
				except LookupError:
					print(data)
				continue
			v_id = video['videoId']
			try:
				dur = round_min(float(video["lengthSeconds"]))
			except (KeyError, ValueError):
				try:
					dur = time_parse(video["lengthText"]["simpleText"])
				except KeyError:
					dur = None
			try:
				name = video["title"]["runs"][0]["text"]
			except LookupError:
				name = v_id
			temp = cdict(
				name=name,
				url=f"https://youtu.be/{v_id}",
				duration=dur,
			)
			out.append(temp)
		return out, token
	yt_client = dict(name="WEB", version="2.20251017", clientName="WEB", clientVersion="2.20251017.01.00")
	yt_session = niquests.Session(multiplexed=True)
	yt_asession = niquests.AsyncSession(multiplexed=True)
	# Returns a subsequent page of a youtube playlist from a page token.
	def get_youtube_continuation(self, token, ctx=None):
		resp = self.yt_session.post(
			"https://www.youtube.com/youtubei/v1/browse?prettyPrint=false",
			headers=self.youtube_header,
			data=json_dumps(dict(
				context=ctx or dict(client=self.yt_client),
				continuation=token,
			)),
		)
		resp.raise_for_status()
		data = resp.json()
		try:
			items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		except LookupError:
			return [], None
		return self.extract_playlist_items(items)
	# Async version of the previous function, used when possible to minimise thread pool wastage.
	async def get_youtube_continuation_async(self, token, ctx=None):
		resp = await self.yt_asession.post(
			"https://www.youtube.com/youtubei/v1/browse?prettyPrint=false",
			headers=self.youtube_header,
			data=json_dumps(dict(
				context=ctx or dict(client=self.yt_client),
				continuation=token,
			)),
		)
		resp.raise_for_status()
		data = resp.json()
		try:
			items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		except LookupError:
			return [], None
		return self.extract_playlist_items(items)
	# Generates a playlist continuation token purely from ID and page number.
	def produce_continuation(self, p, i):
		if not isinstance(p, (bytes, bytearray, memoryview)):
			p = str(p).encode("ascii")
		parts = []
		if i == 1:
			parts.append(b"\xe2\xa9\x85\xb2\x02\x81\x01\x12$VL")
		else:
			parts.append(b"\xe2\xa9\x85\xb2\x02\x7f\x12$VL")
		parts.append(p)
		if i == 1:
			parts.append(b"\x1a4")
		else:
			parts.append(b"\x1a2")
		import base64
		key = bytes((8, i, 0x7a, 0x1e if i == 1 else 0x1f)) + b"PT:" + base64.b64encode(b"\x08" + leb128(i * 100) + b'"\x10780605BC69C6C250').rstrip(b"=")
		obj = base64.b64encode(key).replace(b"=", b"%3D")
		parts.append(obj)
		parts.append(b"\x9a\x02\x22")
		parts.append(p)
		code = b"".join(parts)
		return base64.urlsafe_b64encode(code).replace(b"=", b"%3D").decode("ascii")
	# Returns a full youtube playlist.
	def get_youtube_playlist(self, p_id):
		resp = self.yt_session.get(f"https://www.youtube.com/playlist?list={p_id}", headers=self.youtube_header)
		resp.raise_for_status()
		resp = resp.content
		assert isinstance(resp, bytes)
		try:
			try:
				resp = resp[resp.index(b'{"responseContext":{'):]
			except ValueError:
				search = b"var ytInitialData = "
				try:
					resp = resp[resp.index(search) + len(search):]
				except ValueError:
					search = b'window["ytInitialData"] = '
					resp = resp[resp.index(search) + len(search):]
			try:
				resp = resp[:resp.index(b';</script><')]
			except ValueError:
				resp = resp[:resp.index(b'window["ytInitialPlayerResponse"] = null;')]
				resp = resp[:resp.rindex(b";")]
			data = orjson.loads(resp)
			client = self.yt_client
			for tracker in data["responseContext"]["serviceTrackingParams"]:
				for param in tracker.get("params", ()):
					if param["key"].startswith("client."):
						client[param["key"].removeprefix("client.")] = param["value"]
		except:
			print(resp)
			raise
		context = dict(client=client)
		count = int(data["sidebar"]["playlistSidebarRenderer"]["items"][0]["playlistSidebarPrimaryInfoRenderer"]["stats"][0]["runs"][0]["text"].replace(",", ""))
		items = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"]["contents"]
		entries, token = self.extract_playlist_items(items)
		if count > 100:
			futs = deque()
			if not token:
				token = self.produce_continuation(p_id, 1)
			for page in range(1, ceil(count / 100)):
				fut = esubmit(self.get_youtube_continuation, token, context)
				futs.append(fut)
				token = self.produce_continuation(p_id, page + 1)
			for i, fut in enumerate(futs):
				try:
					entries.extend(fut.result()[0])
				except Exception as ex:
					print(i, repr(ex))
					raise
		out = []
		urls = set()
		for entry in entries:
			if entry.url not in urls:
				urls.add(entry.url)
				out.append(entry)
		return out

	spothead_sem = Semaphore(10, inf, rate_limit=2)
	spot_backs = []
	@property
	def spotify_header(self):
		headers = Request.header()
		backs = shuffle(self.spot_backs)
		for head in tuple(backs):
			if utc() + 60 > float(head.accessTokenExpirationTimestampMs) / 1000:
				self.spot_backs.remove(head)
			if not head.sem.busy:
				with head.sem:
					headers.Authorization = "Bearer " + head.accessToken
				return headers
		with self.spothead_sem:
			resp = self.session.get("https://open.spotify.com/get_access_token", headers=Request.header(), timeout=20)
		resp.raise_for_status()
		assert isinstance(resp.content, bytes)
		head = cdict(orjson.loads(resp.content))
		head.sem = Semaphore(20, 1, rate_limit=5)
		self.spot_backs.append(head)
		with head.sem:
			headers.Authorization = "Bearer " + head.accessToken
		return headers
	soundcloud_token = "7g7gIkrcAS05cJVf2FlIsnkOXtg4JdSe"
	def get_soundcloud_playlist(self, url):
		parts = url.split("?", 1)[0].split("/")
		if parts[0] != "https:" or parts[2] not in ("soundcloud.com", "api-v2.soundcloud.com"):
			raise TypeError("Not a SoundCloud playlist.")
		if parts[-1] == "likes":
			return self.get_soundcloud_likes(url)
		api = "https://api-v2.soundcloud.com/"

		resp = self.session.get(url, headers=Request.header(), timeout=10)
		resp.raise_for_status()
		s = resp.text
		assert isinstance(s, str)
		if s[0] == "{" and s[-1] == "}":
			t = resp.json()
			return [cdict(
				name=t["title"],
				url=t["permalink_url"],
				duration=t["duration"] / 1000,
				thumbnail=t["artwork_url"],
			)]
		search = "<script>window.__sc_hydration = "
		s = s[s.index(search) + len(search):]
		s = s[:s.index(";</script>")]
		data = orjson.loads(s)

		emap = {}
		entries = []
		for hydratable in data:
			if hydratable["hydratable"] == "playlist":
				for t in hydratable["data"]["tracks"]:
					try:
						t["title"]
					except KeyError:
						tid = t["id"]
						emap[tid] = len(entries)
						entries.append(None)
					else:
						entry = cdict(
							name=t["title"],
							url=t["permalink_url"],
							duration=t["duration"] / 1000,
							thumbnail=t["artwork_url"],
						)
						entries.append(entry)

		if emap:
			ids = ",".join(map(str, emap))
			url = f"{api}tracks?ids={ids}&client_id={self.soundcloud_token}"
			resp = self.session.get(url, headers=Request.header(), timeout=10)
			if not resp.content:
				resp.raise_for_status()
			for t, p in zip(resp.json(), emap.values()):
				entry = cdict(
					name=t["title"],
					url=t["permalink_url"],
					duration=t["duration"] / 1000,
					thumbnail=t["artwork_url"],
				)
				entries[p] = entry
		return [e for e in entries if e]
	def get_soundcloud_likes(self, url):
		api = "https://api-v2.soundcloud.com/"
		lim = 1000

		uapi = api + "users/"
		if url.startswith(uapi):
			uid = url[len(uapi):].split("?", 1)[0]
		else:
			resp = self.session.get(url, headers=Request.header(), timeout=20)
			resp.raise_for_status()
			s = resp.text
			assert isinstance(s, str)
			search = 'content="soundcloud://users:'
			s = s[s.index(search) + len(search):]
			uid = s[:s.index('"')]

		futs = []
		entries = []
		url = f"{api}users/{uid}/likes?client_id={self.soundcloud_token}&limit={lim}"
		while True:
			resp = self.session.get(url, headers=Request.header(), timeout=20)
			if not resp.content:
				resp.raise_for_status()
			data = resp.json()
			for e in data["collection"]:
				try:
					t = e["track"]
				except KeyError:
					p = e["playlist"]
					url = p["permalink_url"]
					if len(futs) >= 12:
						futs.pop(0).result()
					fut = esubmit(self.get_soundcloud_playlist, url)
					futs.append(fut)
					entries.append(fut)
				else:
					entry = cdict(
						name=t["title"],
						url=t["permalink_url"],
						duration=t["duration"] / 1000,
						thumbnail=t["artwork_url"],
					)
					entries.append(entry)
			url = data.get("next_href")
			if len(entries) < lim or not url:
				break
			url += f"client_id={self.soundcloud_token}&limit={lim}"

		while True:
			for i, e in enumerate(entries):
				if isinstance(e, Future):
					entries = entries[:i] + e.result() + entries[i + 1:]
					break
			else:
				break
		return entries
	# Returns part of a spotify playlist.
	def get_spotify_part(self, url):
		resp = self.session.get(url, headers=self.spotify_header, timeout=20)
		if resp.status_code not in range(200, 400):
			# resp = proxy.content_or(url, headers=self.spotify_header, timeout=25)
			resp.raise_for_status()
		return self.export_spotify_part(resp.json())
	def export_spotify_part(self, d):
		out = []
		try:
			d = d["tracks"]
		except KeyError:
			pass
		try:
			items = d["items"]
			total = d.get("total", 0)
		except KeyError:
			if "type" in d:
				items = (d,)
				total = 1
			else:
				items = []
				total = 0
		for item in items:
			try:
				track = item["track"]
			except KeyError:
				try:
					track = item["episode"]
				except KeyError:
					if "id" in item:
						track = item
					else:
						continue
			name = track.get("name", track["id"])
			dur = track.get("duration_ms")
			if dur:
				dur /= 1000
			temp = cdict(
				name=name,
				url=f"https://open.spotify.com/track/{track['id']}",
				icon=sorted(track["album"]["images"], key=lambda di: di.get("height", 0), reverse=True)[0]["url"] if "album" in track and track["album"].get("images") else None,
				artist=track.get("artists") and track["artists"][0]["name"],
				id=track["id"],
				duration=dur,
			)
			out.append(temp)
		return out, total
	def get_spotify_playlist(self, url):
		item = url.split("?", 1)[0]
		# Spotify playlist searches contain up to 100 items each
		if "playlist" in item:
			url = item[item.index("playlist"):]
			url = url[url.index("/") + 1:]
			key = url.split("/", 1)[0]
			url = f"https://api.spotify.com/v1/playlists/{key}/tracks?type=track,episode"
			page = 100
		# Spotify album searches contain up to 50 items each
		elif "album" in item:
			url = item[item.index("album"):]
			url = url[url.index("/") + 1:]
			key = url.split("/", 1)[0]
			url = f"https://api.spotify.com/v1/albums/{key}/tracks?type=track,episode"
			page = 50
		# Single track links also supported
		elif "track" in item:
			url = item[item.index("track"):]
			url = url[url.index("/") + 1:]
			key = url.split("/", 1)[0]
			url = f"https://api.spotify.com/v1/tracks/{key}"
			page = 1
		# Single episode links also supported
		elif "episode" in item:
			url = item[item.index("episode"):]
			url = url[url.index("/") + 1:]
			key = url.split("/", 1)[0]
			url = f"https://api.spotify.com/v1/episodes/{key}"
			page = 1
		else:
			raise TypeError("Unsupported Spotify URL.")
		if page == 1:
			return self.get_spotify_part(url)[0]
		search = f"{url}&offset=0&limit={page}"
		entries, count = self.get_spotify_part(search)
		if count > page:
			futs = deque()
			for curr in range(page, count, page):
				search = f"{url}&offset={curr}&limit={page}"
				fut = esubmit(self.get_spotify_part, search)
				futs.append(fut)
				time.sleep(0.0625)
			while futs:
				entries.extend(futs.popleft().result()[0])
		v_id = None
		for x in ("?highlight=spotify:track:", "&highlight=spotify:track:"):
			if x in url:
				v_id = url[url.index(x) + len(x):]
				v_id = v_id.split("&", 1)[0]
				break
		if v_id:
			entries = deque(entries)
			for i, e in enumerate(entries):
				if v_id == e.get("id"):
					entries.rotate(-i)
					break
			entries = list(entries)
		return entries

	def ydl_errors(self, s):
		return "this video has been removed" not in s and "private video" not in s and "has been terminated" not in s and "Video unavailable" not in s

	def item_yt(self, item):
		video = next(iter(item.values()))
		if "videoId" not in video:
			return
		try:
			dur = time_parse(video["lengthText"]["simpleText"])
		except KeyError:
			dur = None
		try:
			title = video["title"]["runs"][0]["text"]
		except KeyError:
			title = video["title"]["simpleText"]
		try:
			tn = video["thumbnail"]
		except KeyError:
			thumbnail = None
		else:
			if type(tn) is dict:
				thumbnail = sorted(tn["thumbnails"], key=lambda t: t.get("width", 0) * t.get("height", 0))[-1]["url"]
			else:
				thumbnail = tn
		try:
			views = int(video["viewCountText"]["simpleText"].replace(",", "").replace("views", "").replace(" ", ""))
		except (KeyError, ValueError):
			views = 0
		return cdict(
			name=title,
			url=f"https://www.youtube.com/watch?v={video['videoId']}",
			duration=dur,
			icon=thumbnail,
			views=views,
		)
	def parse_yt(self, s):
		data = orjson.loads(s)
		results = alist()
		try:
			pages = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
		except KeyError:
			pages = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
		for page in pages:
			try:
				items = next(iter(page.values()))["contents"]
			except KeyError:
				continue
			for item in items:
				if "promoted" not in next(iter(item)).casefold():
					entry = self.item_yt(item)
					if entry is not None:
						results.append(entry)
		return sorted(results, key=lambda entry: entry.views, reverse=True)
	def ytsearch(self, query, skip=False, count=1):
		out = alist()
		if not skip:
			try:
				resp = self.extract_info(f"ytsearch{count}:" + query, process=False)
				if resp.get("_type", None) == "url":
					resp = self.extract_info(resp["url"], process=True)
			except Exception:
				print_exc()
			else:
				if resp.get("_type", None) == "playlist":
					entries = resp["entries"]
				else:
					entries = [resp]
				for entry in entries:
					if "title" in entry:
						title = entry["title"]
					else:
						title = entry["url"].rsplit("/", 1)[-1]
						if "." in title:
							title = title[:title.rindex(".")]
					url = entry.get("webpage_url", entry.get("url", entry.get("id")))
					if not url:
						continue
					if entry.get("duration"):
						dur = float(entry["duration"])
					else:
						dur = None
					temp = cdict(name=title, url=url, duration=dur)
					if not is_url(url):
						if entry.get("ie_key", "").casefold() == "youtube":
							temp["url"] = f"https://youtu.be/{url}"
					out.append(temp)
		if not out:
			url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
			resp = Request(url, headers=self.youtube_header, timeout=12)
			result = None
			s = resp
			assert isinstance(s, bytes)
			with suppress(ValueError):
				with suppress(ValueError):
					s = s[s.index(b"// scraper_data_begin") + 21:s.rindex(b"// scraper_data_end")]
				s = s[s.index(b"var ytInitialData = ") + 20:]
				s = s[:s.index(b";</script>")]
				result = self.parse_yt(s)
			with suppress(ValueError):
				s = s[s.index(b'window["ytInitialData"] = ') + 26:]
				s = s[:s.index(b'window["ytInitialPlayerResponse"] = null;')]
				s = s[:s.rindex(b";")]
				result = self.parse_yt(s)
			if result is not None:
				q = to_alphanumeric(full_prune(query))
				high = alist()
				low = alist()
				for entry in result:
					if entry.duration:
						name = full_prune(entry.name)
						aname = to_alphanumeric(name)
						spl = aname.split()
						if ("remix" in q or "cover" in q) == ("remix" in spl or "cover" in spl) and (entry.duration < 960 or ("extended" in q or "hour" in q) == ("extended" in spl or "hour" in spl or "hours" in spl)):
							if string_similarity(aname, q) >= 0.5:
								high.append(entry)
								continue
					low.append(entry)

				def key(entry):
					coeff = string_similarity(to_alphanumeric(full_prune(entry.name)), q)
					if coeff < 0.5:
						coeff = 0
					return coeff

				out = sorted(high, key=key, reverse=True)
				out.extend(sorted(low, key=key, reverse=True))
		return out[:count]
	def scsearch(self, query, count=1):
		out = alist()
		resp = self.extract_info(f"scsearch{count}:" + query, process=False)
		if resp.get("_type", None) == "url":
			resp = self.extract_info(resp["url"], process=True)
		if resp.get("_type", None) == "playlist":
			entries = list(resp["entries"])
		else:
			entries = [resp]
		for entry in entries:
			if "title" in entry:
				title = entry["title"]
			else:
				title = entry["url"].rsplit("/", 1)[-1]
				if "." in title:
					title = title[:title.rindex(".")]
			url = entry.get("webpage_url", entry.get("url", entry.get("id")))
			if not url:
				continue
			if entry.get("duration"):
				dur = float(entry["duration"])
			else:
				dur = None
			temp = cdict(name=title, url=url, duration=dur)
			icon = get_best_icon(entry)
			if icon:
				temp.icon = icon
			out.append(temp)
			if len(out) >= count:
				break
		return out
	def spsearch(self, query, count=1):
		# Currently favour searching Bandcamp over Spotify due to downloaders for the former being more reliable.
		return self.bcsearch(query, count)
		# query = f"https://api.spotify.com/v1/search?type=track%2Cshow_audio%2Cepisode_audio&include_external=audio&limit={count}&q=" + quote_plus(query)
		# resp = self.session.get(query, headers=self.spotify_header, timeout=20).json()
		# if "tracks" not in resp:
		# 	print(resp)
		# 	return []
		# out = alist()
		# for track in resp["tracks"]["items"]:
		# 	try:
		# 		name = track.get("name", track["id"])
		# 	except LookupError:
		# 		continue
		# 	else:
		# 		item = f"https://open.spotify.com/track/{track['id']}"
		# 	out.append(cdict(
		# 		name=name,
		# 		url=item,
		# 	))
		# 	if len(out) >= count:
		# 		break
		# return out
	def bcsearch(self, query, count=1):
		query = "https://bandcamp.com/search?q=" + quote_plus(query) + "&item_type=t"
		content = self.session.get(query, headers=Request.header(), timeout=20).content
		assert isinstance(content, bytes)
		out = alist()
		try:
			content = content.split(b'<ul class="result-items">', 1)[1]
			tracks = content.split(b"<!-- search result type=")
			for track in tracks:
				entry = cdict()
				if track.startswith(b"track id=") or track.startswith(b"album id="):
					ttype = track[:5]
					try:
						track = track.split(b'<img src="', 1)[1]
						entry.icon = as_str(track[:track.index(b'">')])
					except ValueError:
						pass
					track = track.split(b'<div class="heading">', 1)[1]
					entry.name = html_decode(as_str(track.split(b">", 1)[1].split(b"<", 1)[0].strip()))
					entry.url = as_str(track.split(b'href="', 1)[1].split(b'"', 1)[0].split(b"?", 1)[0])
					if ttype == b"track":
						track = track.split(b'<div class="subhead">', 1)[1]
						artist = as_str(track.split(b"<", 1)[0].strip())
						entry.artist = html_decode(artist.rsplit("by ", 1)[-1])
						if out and out[-1].name == entry.name and out[-1].artist == entry.artist:
							# Bandcamp search results are often cluttered with duplicates, so we skip them.
							continue
						out.append(entry)
						if len(out) >= count:
							break
			return out
		except (LookupError, ValueError):
			return []

	def handle_path(self, path, entry=None):
		info = audio_meta(path)
		if entry is not None:
			entry["duration"] = info.get("duration")
			if info.name:
				entry["name"] = info.name
		return path, info.codec, info.duration, info.channels

	def handle_spotify(self, entry, url, fn):
		assert is_spotify_url(url), "Not a Spotify URL."
		# If we have the track artist, attempt to find a better match on Bandcamp. The Spotify downloaders are spotty (no pun intended), so those are kept as the fallback instead.
		artist = entry.get("artist")
		if not artist:
			res = self.get_spotify_playlist(url)
			if res:
				entry.update(res[0])
				artist = entry.get("artist")
		if artist:
			query = f"{artist}: {entry['name']}"
			results = self.bcsearch(query, count=3)
			final = None
			match = 0
			for entry2 in results:
				name1 = to_alphanumeric(full_prune(entry["name"]))
				name2 = to_alphanumeric(full_prune(entry2.name))
				artist1 = to_alphanumeric(full_prune(artist))
				artist2 = to_alphanumeric(full_prune(entry2.get("artist", "")))
				# If the Bandcamp search result is a good match, we use that instead.
				if (name_match := string_similarity(name1, name2)) >= 0.5 and (artist_match := string_similarity(artist1, artist2)) >= 0.5:
					coeff = name_match * 0.6 + artist_match * 0.4
					if coeff > match:
						final = entry2
						match = coeff
			if final:
				print(f"Matched Spotify track {url} to Bandcamp: {final.name} by {final.artist}, {match:.2%} match.")
				stream, cdc, dur, ac = self.get_audio(final, asap=True)
				if stream:
					entry.update(final)
					return stream, cdc, dur, ac
		r_mp3 = temporary_file("mp3")
		fn2 = self.run(f"get_audio_spotify({json_dumpstr(url)},{json_dumpstr(r_mp3)})", priority=True)
		return self.handle_path(fn2, entry)

	def handle_special_multiple(self, url):
		headers = fcdict(asyncio.run(attachment_cache.scan_headers(url)))
		match headers.get("content-type"):
			case "application/json":
				b = asyncio.run(attachment_cache.download(url))
				try:
					d = select_and_loads(b, safe=True)
				except orjson.JSONDecodeError:
					d = [url for url in as_str(b).splitlines() if is_url(url)]
					if not d:
						raise
					q = [dict(name=url.split("?", 1)[0].rsplit("/", 1)[-1], url=url) for url in d]
				else:
					q = d["queue"][:262144]
				return [cdict(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
			case "application/zip" | "application/gzip" | "application/x-gzip" | "application/zstd" | "application/vnd.rar" | "application/x-tar" | "application/x-tar+xz" | "application/x-7z-compressed":
				fp = asyncio.run(attachment_cache.download(url, read=True))
				path = fp.name
				assert os.path.exists(path), path
				files = extract_archive(path)
				output = []
				with niquests.Session() as session:
					for fi in files:
						with open(fi, "rb") as f:
							b = f.read()
						data = audio_meta(fi)
						if data.get("bitrate"):
							name = data.get("name") or url2fn(fi)
							resp2 = session.post(
								f"https://api.mizabot.xyz/upload?filename={name}",
								data=b,
							)
							url = resp2.text
							output.append(cdict(name=name, url=url, duration=data.get("duration")))
				return output

	def handle_special_audio(self, entry, url, fn):
		"""Handles special audio formats unsupported by FFmpeg, such as spotify URLs, spectrogram images, as well as ecdc, org, and midi files."""
		if is_spotify_url(url):
			return self.handle_spotify(entry, url, fn)
		if is_youtube_url(url):
			return
		header = Request.header()
		header["User-Agent"] = "Lavf/100.0.0"
		header["Accept"] = "audio/*, */*"
		header["Icy-Metadata"] = "1"
		try:
			resp = self.session.get(url, headers=header, stream=True)
		except niquests.exceptions.ConnectionError:
			try:
				return self.handle_path(url, entry)
			except Exception:
				return
		with resp:
			head = resp.headers
			changed = False
			if not entry.get("icon", None):
				head = fcdict(head)
				try:
					for keyword in ("thumbnail", "icon", "logo"):
						for k, v in head.items():
							if keyword in k:
								entry["icon"] = v
								changed = True
								raise StopIteration
				except StopIteration:
					pass
			for keyword in ("title", "name"):
				for k, v in head.items():
					if keyword in k:
						entry["name"] = v
						changed = True
			if changed:
				print(entry)
			ct = head.get("Content-Type", "").split(";", 1)[0]
			b = b""
			it = resp.iter_content(65536)
			if not ct or ct in ("application/octet-stream", "application/vnd.lotus-organizer"):
				b = next(it)
				ct = magic.from_buffer(b)

			def copy_to_file(fn2):
				nonlocal b
				with open(fn2, "wb") as f:
					while True:
						f.write(b)
						try:
							b = next(it)
						except StopIteration:
							break
						if not b:
							break
				return fn2

			left, right = ct.split("/", 1)[0], ct.split("/", 1)[-1]
			if left == "image":
				r_im = temporary_file(right)
				copy_to_file(r_im)
				args = [python, "png2wav.py", r_im, fn]
				print(args)
				res = subprocess.run(args, cwd="misc", stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
				if not os.path.exists(fn) or not os.path.getsize(fn):
					raise RuntimeError(as_str(res.stderr) or "Unable to locate converted file.")
				return self.handle_path(fn, entry)
			if ct in ("audio/x-ecdc", "audio/ecdc"):
				r_ecdc = temporary_file("ecdc")
				copy_to_file(r_ecdc)
				args1 = [compat_python, "misc/ecdc_stream.py", "-b", "0", "-d", r_ecdc]
				args2 = ["ffmpeg", "-v", "error", "-hide_banner", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "96k", "-vbr", "on", fn]
				print(args1, args2)
				res = PipedProcess(args1, args2, stderr=subprocess.PIPE).wait()
				if not os.path.exists(fn) or not os.path.getsize(fn):
					raise RuntimeError(as_str(res.stderr.read()) or "Unable to locate converted file.")
				return self.handle_path(fn, entry)
			if ct in ("audio/x-org", "audio/org"):
				r_org = temporary_file("org")
				r_opus = replace_ext(r_org, "opus")
				copy_to_file(r_org)
				args = ["hyperchoron", "-i", r_org, "-f", "opus", "-o", r_opus]
				print(args)
				res = subprocess.run(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
				if not os.path.exists(r_opus) or not os.path.getsize(r_opus):
					raise RuntimeError(as_str(res.stderr) or "Unable to locate converted file.")
				return self.handle_path(r_opus, entry)
			if ct in ("audio/x-midi", "audio/midi", "audio/sp-midi"):
				r_mid = temporary_file("mid")
				r_opus = replace_ext(r_mid, "opus")
				copy_to_file(r_mid)
				args = ["hyperchoron", "-i", r_mid, "-f", "opus", "-o", r_opus]
				print(args)
				res = subprocess.run(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
				if not os.path.exists(r_opus) or not os.path.getsize(r_opus):
					raise RuntimeError(as_str(res.stderr) or "Unable to locate converted file.")
				return self.handle_path(r_opus, entry)
			if ct in ("audio/x-scpls", "audio/pls", "audio/scpls"):
				entries = []
				lines = as_str(b).splitlines()
				for line in lines:
					k, v = line.split("=", 1)
					if k.lower().startswith("file") and k[-1].isnumeric():
						entries.append(v)
					if k.lower() == "title1":
						entry["name"] = v
					if k.lower() == "length1":
						dur = float(v)
						if dur > 0:
							entry["duration"] = dur
				if entries:
					fn = temporary_file("concat")
					with open(fn, "w", encoding="utf-8") as f:
						f.write("\n".join(f"file '{url}'" for url in entries))
					return self.handle_path(entries[0], entry)
			if left == "audio" or right in AUDIO_FORMS:
				try:
					return self.handle_path(url, entry)
				except Exception:
					return

	def get_audio(self, entry, asap=None, fmt=None, start=None, end=None, research=True):
		"""Gets a valid audio stream (URL or file) from a given entry, with optional trimming. In ASAP (as soon as possible) mode, prefer URLs."""
		url = entry.get("orig") or entry["url"]
		ext = fmt or "opus"
		fn = temporary_file(ext)
		d = entry.get("duration")
		if asap is None:
			asap = not d or d > 3840
		is_trim = start or end is not None
		special_checked = False
		if not fmt and asap or (not d or not isfinite(d)):
			# If format is not specified, try to stream the audio from URL if possible
			with tracebacksuppressor:
				stream, cdc, ac = get_best_audio(entry)
				if entry.get("duration") and cdc and not expired(stream) and (d is not None or asap):
					return stream, cdc, entry["duration"], ac
				if research:
					entry2 = self.search(url, force=True)[0]
					entry.update(entry2)
					stream, cdc, ac = get_best_audio(entry2)
					if entry.get("duration") and cdc and stream:
						return stream, cdc, entry["duration"], ac
				result = self.handle_special_audio(entry, url, fn)
				if result:
					return result
				if stream and not entry.get("duration"):
					return stream, cdc or "auto", entry.get("duration") or None, ac or 1
				special_checked = True
		if not special_checked:
			result = self.handle_special_audio(entry, url, fn)
			if result:
				if not fmt:
					return result
				# If format is specified, always produce a file, and allow trimming if necessary
				url, cdc, dur, ac = result
				tmpl = temporary_file()
				if is_trim:
					tmpl += f"~{start}-{end}"
				fn = f"{tmpl}.{fmt}"
				if os.path.exists(fn) and os.path.getsize(fn):
					return self.handle_path(fn, entry=None if is_trim else entry)
				fn2 = f"{tmpl}~.{fmt}"
				args = ["ffmpeg", "-v", "error", "-hide_banner", "-vn"]
				if start:
					args.extend(["-ss", str(start)])
				if end:
					args.extend(["-to", str(end)])
				args.extend(["-i", url, "-b:a", "160k", "-vbr", "on", fn2])
				print(args)
				res = subprocess.run(args, stderr=subprocess.PIPE)
				if not os.path.exists(fn2) or not os.path.getsize(fn2):
					raise RuntimeError(as_str(res.stderr) or "Unable to locate converted file.")
				rename(fn2, fn)
				return self.handle_path(fn, entry=None if is_trim else entry)
		if not fmt and (is_discord_attachment(url) or is_miza_url(url)):
			if discord_expired(url):
				# Just in case a Discord attachment expires in the short time between checking and downloading
				url = shorten_attachment(url, 0)
			return self.handle_path(url, entry)
		codec = "opus" if ext == "ogg" else ext
		target = fn
		if is_trim:
			fn = temporary_file(ext)
		# print("OUTTMPL:", fn)
		ydl_opts = dict(
			# Prefer selected codec, but fallback to best audio if not available
			format=f"bestaudio[vcodec=none][acodec={codec}][audio_channels=2]/worst[abr>=96][audio_channels=2]/bestaudio[audio_channels=2]/bestaudio/worst[abr>=64]/worst[acodec!=none][tbr>=480]/best",
			default_search="auto",
			source_address="0.0.0.0",
			remote_components=["ejs:github"],
			final_ext=ext,
			cachedir=TEMP_PATH,
			outtmpl=target,
			windowsfilenames=True,
			cookiesfrombrowser=["firefox"],
			postprocessors=[dict(
				# Use our custom FFmpeg audio convertor to ensure consistent audio codec, and allow trimming if necessary
				key="FFmpegCustomAudioConvertor",
				format=ext,
				codec=codec,
				start=start,
				end=end,
				final=fn,
			)]
		)
		if url.startswith("https://youtu.be/"):
			url2 = url.replace("https://youtu.be/", "https://www.youtube.com/watch?v=")
		else:
			url2 = url
		self.run(f"ytd.YoutubeDL({repr(ydl_opts)}).download({repr(url2)})")
		assert os.path.exists(fn) and os.path.getsize(fn), fn
		return self.handle_path(fn, entry=None if is_trim else entry)

	def preprocess(self, url, mode, count):
		output = deque()
		if is_url(url):
			if is_discord_attachment(url):
				if discord_expired(url):
					# For expired Discord attachments, shorten the URL using our reverse proxy and try that instead
					url = shorten_attachment(url, 0)
				temp = cdict(
					name=url2fn(url),
					url=url,
				)
				output.append(temp)
			elif "youtube.com/" in url or "youtu.be/" in url:
				p_id = None
				for x in ("?list=", "&list="):
					if x in url:
						p_id = url[url.index(x) + len(x):]
						p_id = p_id.split("&", 1)[0]
						break
				if p_id:
					v_id = None
					for x in ("?v=", "&v="):
						if x in url:
							v_id = url[url.index(x) + len(x):]
							v_id = v_id.split("&", 1)[0]
							break
					if p_id in ("LL", "LM", "WL", "RDMM"):
						if v_id:
							return [], f"https://youtu.be/{v_id}"
						raise PermissionError(url, "Sorry, this endpoint is intentionally blocked for privacy/performance.")
					with tracebacksuppressor:
						output.extend(self.get_youtube_playlist(p_id))
						# Scroll to highlighted entry if possible
						if v_id:
							for i, e in enumerate(output):
								if v_id in e.url:
									output.rotate(-i)
									break
			elif re.match(r"^https:\/\/soundcloud\.com\/[A-Za-z0-9]+\/sets\/", url) or re.match(r"^https:\/\/soundcloud\.com\/[A-Za-z0-9]+\/likes", url) or re.match(r"^https:\/\/api-v2\.soundcloud\.com\/users\/[0-9]+\/likes", url):
				with tracebacksuppressor:
					output.extend(self.get_soundcloud_playlist(url))
			elif is_spotify_url(url):
				raise NotImplementedError(url, "Spotify is currently unsupported!")
				# with tracebacksuppressor:
				# 	output.extend(self.get_spotify_playlist(url))
		else:
			urls = []
			if ":" not in url:
				if not mode and count >= 4:
					# If no search mode is specified, split the search into 4 parts: 2 for YouTube, 1 for SoundCloud, and 1 for Bandcamp
					half = ceil(count / 2)
					urls.append(f"ytsearch{half}:{url}")
					quarter = ceil((count - half) / 2)
					urls.append(f"scsearch{quarter}:{url}")
					remainder = count - half - quarter
					urls.append(f"bcsearch{remainder}:{url}")
				else:
					urls.append(f"{mode or 'yt'}search{count}:{url}")
			futs = deque()
			for url in urls:
				check, search = url.split(":", 1)
				for mode in ("ytsearch", "scsearch", "spsearch", "bcsearch"):
					if check == mode:
						fut = esubmit(getattr(self, mode), search, count=count)
						futs.append(fut)
						break
					elif check.startswith(mode) and check[len(mode):].isnumeric():
						fut = esubmit(getattr(self, mode), search, count=int(check[len(mode):]))
						futs.append(fut)
						break
			output.extend(itertools.chain.from_iterable(f.result() for f in futs))
		return list(output), url

	# Main extract function, able to extract from youtube playlists much faster than youtube-dl using youtube API, as well as ability to follow spotify links.
	def extract(self, url, mode=None, count=1):
		output, url = self.preprocess(url, mode=mode, count=count)
		# Only proceed if no items have already been found (from playlists in this case)
		if not len(output):
			if is_miza_attachment(url):
				output = self.handle_special_multiple(url)
				if output:
					return output
				return [cdict(
					name=url2fn(url),
					url=url,
					audio=url,
					video=url,
				)]
			resp = None
			# Otherwise call automatic extract_info function
			try:
				if not resp:
					if url.startswith("https://youtu.be/"):
						url2 = url.replace("https://youtu.be/", "https://www.youtube.com/watch?v=")
					else:
						url2 = url
					resp = self.extract_info(url2, process=False)
				if not resp:
					return []
				if isinstance(resp, str):
					assert is_url(resp), resp
					resp = dict(_type="url", url=resp)
				if resp.get("_type") == "url":
					resp = self.extract_info(resp["url"], process=True)
			except (RuntimeError, ytd.utils.DownloadError):
				output = self.handle_special_multiple(url)
				if output:
					return output
				entry = cdict(
					name=url2fn(url),
					url=url,
				)
				ext = url2ext(url)
				fn = temporary_file(ext)
				result = self.handle_special_audio(entry, url, fn)
				if not result:
					raise
				return [entry]
			if resp is None or not len(resp):
				raise LookupError(f"No results for {url}")
			# Check if result is a playlist
			if resp.get("_type") == "playlist":
				entries = list(resp["entries"])
				for i, entry in enumerate(entries):
					temp = None
					if not i:
						# Extract full data from first item only
						try:
							if "url" in entry:
								temp = self.extract(entry["url"])[0]
							elif "formats" in entry:
								temp = cdict({
									"name": resp["title"],
									"url": resp.get("webpage_url", url),
									"duration": None,
									"audio": get_best_audio(entry),
									"icon": get_best_icon(entry),
									"video": get_best_video(entry),
								})
						except Exception:
							print_exc()
							continue
					else:
						with tracebacksuppressor:
							if "title" in entry:
								title = entry["title"]
							else:
								title = entry["url"].rsplit("/", 1)[-1]
								if "." in title:
									title = title[:title.rindex(".")]
							try:
								dur = round_min(entry["duration"])
							except Exception:
								dur = None
							url = entry.get("webpage_url", entry.get("url", entry.get("id")))
							if not url:
								continue
							temp = {
								"name": title,
								"url": url,
								"duration": dur,
							}
							if not is_url(url):
								if entry.get("ie_key", "").casefold() == "youtube":
									temp["url"] = f"https://youtu.be/{url}"
					if temp:
						output.append(cdict(temp))
			else:
				if resp.get("ext", "unknown_video") == "unknown_video":
					output = self.handle_special_multiple(url)
					if output:
						return output
				# Single item results must contain full data, we take advantage of that here
				name = resp.get("title") or resp["webpage_url"].rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
				url = resp.get("webpage_url") or resp["url"]
				dur = resp.get("duration")
				temp = cdict({
					"name": name,
					"url": url,
					"duration": dur,
					"audio": get_best_audio(resp),
					"icon": get_best_icon(resp),
					"video": get_best_video(resp),
				})
				audio = temp.audio
				if not dur and "googlevideo" in audio[:64]:
					durstr = re.findall(r"[&?]dur=([0-9\.]+)", audio)
					if durstr:
						temp.duration = round_min(durstr[0])
				output.append(temp)
		return output

	def search_into(self, item, mode, count):
		temp = self.extract(item, mode=mode, count=count)
		for e in temp:
			url = unyt(e["url"])
			if url != e["url"]:
				e["url"], e["orig"] = url, e["url"]
		return temp

	# Performs a search, storing and using cached search results for efficiency.
	def search(self, item, force=False, mode=None, count=1):
		key = verify_search(item)
		# Force key to be a string, eliminating shenanigans where tuples are serialised as lists within the cache and then fail to retrieve due to hashability
		retrieval = json_dumpstr([key, mode, count])
		retrieve = self.search_cache._retrieve if force else self.search_cache.retrieve
		temp = retrieve(retrieval, self.search_into, item, mode, count)
		if not temp:
			raise FileNotFoundError(f'No results for {item}.')
		if not temp[0].get("duration"):
			self.get_audio(temp[0], asap=True, research=False)
		return temp

	def in_cache(self, item):
		item = verify_search(item)
		retrieval = json_dumpstr([item, None, 1])
		try:
			temp = self.search_cache[retrieval][0]
		except KeyError:
			return 0
		else:
			return 1 + (not expired(get_best_audio(temp)[0]))

	def close(self):
		for worker in self.workers:
			worker.terminate()
		self.workers.clear()