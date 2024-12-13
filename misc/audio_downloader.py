from collections import deque
from concurrent.futures import Future
from contextlib import suppress
from math import ceil, inf
import random
import time
from urllib.parse import quote_plus
from traceback import print_exc
import orjson
import requests
import yt_dlp as ytd
from misc.types import alist, as_str, cdict, full_prune, json_dumps, round_min, to_alphanumeric, tracebacksuppressor
from misc.smath import time_parse, fuzzy_substring
from misc.asyncs import esubmit
from misc.util import (
	python, shuffle, utc, proxy, leb128, verify_search,
	find_urls, is_url, is_discord_url, is_image, is_miza_url, is_youtube_url,
	EvalPipe, FileHashDict, Request, Semaphore, Cache
)

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
		thumbnails = entry["thumbnails"]
		if not thumbnails:
			raise KeyError(thumbnails)
	except KeyError:
		try:
			url = entry["webpage_url"]
		except KeyError:
			url = entry["url"]
		if not url:
			return ""
		if is_discord_url(url):
			if not is_image(url):
				return "https://cdn.discordapp.com/embed/avatars/0.png"
		if is_youtube_url(url):
			if "?v=" in url:
				vid = url.split("?v=", 1)[-1]
			else:
				vid = url.rsplit("/", 1)[-1].split("?", 1)[0]
			entry["thumbnail"] = f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
			return entry["thumbnail"]
		if is_miza_url(url):
			return "https://mizabot.xyz/static/mizaleaf.png"
		return url
	return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]
# Gets the best audio file download link for a queue entry.
def get_best_audio(entry):
	try:
		return entry["stream"]
	except KeyError:
		pass
	best = (-inf,)
	try:
		fmts = entry["formats"]
	except KeyError:
		fmts = ()
	try:
		url = entry["url"]
	except KeyError:
		url = entry["webpage_url"]
	replace = True
	for fmt in fmts:
		q = (
			fmt.get("acodec") in ("opus", "vorbis"),
			fmt.get("vcodec") in (None, "none"),
			-abs(fmt["audio_channels"] - 2) if isinstance(fmt.get("audio_channels"), (int, float)) else -inf,
			fmt["abr"] if isinstance(fmt.get("abr"), (int, float)) else -inf,
			fmt["tbr"] if not isinstance(fmt.get("abr"), (int, float)) and isinstance(fmt.get("tbr"), (int, float)) else -inf,
			fmt["asr"] if isinstance(fmt.get("asr"), (int, float)) else -inf,
		)
		q = fmt.get("abr", 0)
		if not isinstance(q, (int, float)):
			q = 0
		if q <= 0:
			if fmt.get("asr"):
				q = fmt["asr"] / 1000
			elif fmt.get("audio_channels"):
				q = fmt["audio_channels"]
		q = (fmt.get("acodec") in ("opus", "vorbis"), fmt.get("vcodec") in (None, "none"), fmt.get("tbr", 0) or q)
		u = as_str(fmt["url"])
		if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = Request(url)
		fmts = alist()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = as_str(resp[:resp.index(b'"')])
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = as_str(resp[:resp.index(b'</BaseURL>')])
				fmt = cdict(ytd.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_audio(entry)
	if not url:
		raise KeyError("URL not found.")
	return url
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
	try:
		url = entry["url"]
	except KeyError:
		url = entry["webpage_url"]
	replace = True
	for fmt in fmts:
		q = (
			fmt.get("vcodec") not in (None, "none"),
			fmt.get("protocol") != "m3u8_native" if not hq else False,
			-abs(fmt["fps"] - (90 if hq else 42)) if isinstance(fmt.get("fps"), (int, float)) else -inf,
			-abs(fmt["height"] - (1600 if hq else 720)) if isinstance(fmt.get("height"), (int, float)) else -inf,
			fmt["tbr"] if isinstance(fmt.get("tbr"), (int, float)) else -inf,
		)
		u = as_str(fmt["url"])
		if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
			replace = False
		if q > best or replace:
			best = q
			url = fmt["url"]
	if "dropbox.com" in url:
		if "?dl=0" in url:
			url = url.replace("?dl=0", "?dl=1")
	if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
		resp = Request(url)
		fmts = alist()
		with suppress(ValueError, KeyError):
			while True:
				search = b'<Representation id="'
				resp = resp[resp.index(search) + len(search):]
				f_id = as_str(resp[:resp.index(b'"')])
				search = b"><BaseURL>"
				resp = resp[resp.index(search) + len(search):]
				stream = as_str(resp[:resp.index(b'</BaseURL>')])
				fmt = cdict(ytd.extractor.youtube.YoutubeIE._formats[f_id])
				fmt.url = stream
				fmts.append(fmt)
		entry["formats"] = fmts
		return get_best_video(entry)
	if not url:
		raise KeyError("URL not found.")
	return url
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
				lyr = []
				for event in data["events"]:
					para = "".join(seg.get("utf8", "") for seg in event.get("segs", ()))
					lyr.append(para)
				lyrics = "".join(lyr).strip()
				if lyrics:
					print("lyrics_captions", lyrics)
					return lyrics
def simplify_entry(entry):
	if "title" in entry:
		title = entry["title"]
	else:
		title = entry["url"].rsplit("/", 1)[-1]
		if "." in title:
			title = title[:title.rindex(".")]
	url = entry.get("webpage_url", entry.get("url", entry.get("id")))
	if not url or entry.get("invalid"):
		return
	if not is_url(url):
		if entry.get("ie_key", "").casefold() == "youtube":
			url = f"https://youtu.be/{url}"
	if entry.get("duration"):
		dur = float(entry["duration"])
	else:
		dur = None
	temp = cdict(name=title, url=url, duration=dur)
	icon = get_best_icon(entry)
	if icon:
		temp["icon"] = icon
	if "formats" in entry:
		temp["stream"] = get_best_audio(entry)
		temp["video"] = get_best_video(entry)
	lyrics = get_best_lyrics(entry)
	if lyrics:
		temp["lyrics"] = lyrics
	return temp


class AudioDownloader:

	def __init__(self):
		self.worker_count = 3
		self.workers = alist()
		for i in range(self.worker_count):
			worker = EvalPipe.connect(
				[python, "-m", "misc.x_ytdl", str(8011 + i)],
				8011 + i,
				glob=globals(),
			)
			self.workers.append(worker)
		self.session = requests.Session()
		self.cache = FileHashDict(path="cache/ytdl.cache", automut=False, autosave=120)
		self.futs = [
			esubmit(self.set_cookie),
		]

	def submit(self, s):
		return self.workers.next().submit(s)

	def run(self, s):
		return self.workers.next().run(s)

	def extract_info(self, url, download=False, process=True):
		return self.run(f"ytdl.extract_info({json_dumps(url)},download={download},process={process})")

	def set_cookie(self):
		self.youtube_base = "CONSENT=YES+cb.20210328-17-p0.en+FX"
		s = self.session.get("https://www.youtube.com").text
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
		out = deque()
		for data in items:
			try:
				video = data["playlistVideoRenderer"]
			except KeyError:
				try:
					token = data["continuationItemRenderer"]["continuationEndpoint"]["continuationCommand"]["token"]
				except KeyError:
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
	# Returns a subsequent page of a youtube playlist from a page token.
	def get_youtube_continuation(self, token, ctx):
		for i in range(3):
			try:
				data = Request(
					"https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
					headers=self.youtube_header,
					method="POST",
					data=json_dumps(dict(
						context=ctx,
						continuation=token,
					)),
					json=True,
				)
			except Exception:
				print_exc()
				time.sleep(i)
		items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		return self.extract_playlist_items(items)
	# Async version of the previous function, used when possible to minimise thread pool wastage.
	async def get_youtube_continuation_async(self, token, ctx):
		data = await Request(
			"https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
			headers=self.youtube_header,
			method="POST",
			data=json_dumps(dict(
				context=ctx,
				continuation=token,
			)),
			json=True,
			aio=True,
		)
		items = data["onResponseReceivedActions"][0]["appendContinuationItemsAction"]["continuationItems"]
		return self.extract_playlist_items(items)
	# Generates a playlist continuation token purely from ID and page number.
	def produce_continuation(self, p, i):
		if not isinstance(p, (bytes, bytearray, memoryview)):
			p = str(p).encode("ascii")
		parts = []
		if i == 1:
			parts.append(b"\xe2\xa9\x85\xb2\x02a\x12$VL")
		else:
			parts.append(b"\xe2\xa9\x85\xb2\x02_\x12$VL")
		parts.append(p)
		if i == 1:
			parts.append(b"\x1a\x14")
		else:
			parts.append(b"\x1a\x12")
		import base64
		key = bytes((8, i, 0x7a, (i != 1) + 6)) + b"PT:" + base64.b64encode(b"\x08" + leb128(i * 100)).rstrip(b"=")
		obj = base64.b64encode(key).replace(b"=", b"%3D")
		parts.append(obj)
		parts.append(b"\x9a\x02\x22")
		parts.append(p)
		code = b"".join(parts)
		return base64.b64encode(code).replace(b"=", b"%3D").decode("ascii")
	# Returns a full youtube playlist.
	def get_youtube_playlist(self, p_id):
		resp = Request(f"https://www.youtube.com/playlist?list={p_id}", headers=self.youtube_header)
		client = ytd.extractor.youtube.INNERTUBE_CLIENTS["web"]["INNERTUBE_CONTEXT"]["client"].copy()
		try:
			ytcfg = resp[resp.index(b"ytcfg.set"):]
			ytcfg = ytcfg[:ytcfg.index(b";")]
			ytcfg = eval(ytcfg.split(b"(", 1)[-1].rsplit(b")", 1)[0], {}, {})[-1] + "&"
			end = "&"
			start = "client.name="
			cname = ytcfg[ytcfg.index(start) + len(start):]
			client["clientName"] = cname[:cname.index(end)]
			start = "client.version="
			cversion = ytcfg[ytcfg.index(start) + len(start):]
			client["clientVersion"] = cversion[:cversion.index(end)]
		except ValueError:
			pass
		context = dict(client=client)
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
		except:
			print(resp)
			raise
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
			for fut in futs:
				entries.extend(fut.result()[0])
		out = deque()
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
		if not self.spothead_sem.active:
			with self.spothead_sem:
				resp = proxy.content_or("https://open.spotify.com/get_access_token", headers=Request.header(), timeout=10)
		else:
			with self.spothead_sem:
				resp = self.session.get("https://open.spotify.com/get_access_token", headers=Request.header(), timeout=20)
		resp.raise_for_status()
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
			resp = proxy.content_or(url, headers=self.spotify_header, timeout=25)
			resp.raise_for_status()
		return self.export_spotify_part(resp.json())
	def export_spotify_part(self, d):
		out = deque()
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
		return entries

	def ydl_errors(self, s):
		return "this video has been removed" not in s and "private video" not in s and "has been terminated" not in s and ("Video unavailable" not in s or not self.backup_sem.active)

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
	def search_yt(self, query, skip=False, count=1):
		out = alist()
		if not skip:
			resp = self.extract_info(f"ytsearch{count}:" + query, process=False)
			if resp.get("_type", None) == "url":
				resp = self.extract_from(resp["url"])
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
				if not url or entry.get("invalid"):
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
							if fuzzy_substring(aname, q, match_length=False) >= 0.5:
								high.append(entry)
								continue
					low.append(entry)

				def key(entry):
					coeff = fuzzy_substring(to_alphanumeric(full_prune(entry.name)), q, match_length=False)
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
			resp = self.extract_from(resp["url"])
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
			if not url or entry.get("invalid"):
				continue
			if entry.get("duration"):
				dur = float(entry["duration"])
			else:
				dur = None
			temp = cdict(name=title, url=url, duration=dur)
			if not is_url(url):
				if entry.get("ie_key", "").casefold() == "youtube":
					temp["url"] = f"https://www.youtube.com/watch?v={url}"
			out.append(temp)
	def spsearch(self, query, count=1):
		query = f"https://api.spotify.com/v1/search?type=track%2Cshow_audio%2Cepisode_audio&include_external=audio&limit={count}&q=" + quote_plus(query)
		resp = self.session.get(query, headers=self.spotify_header, timeout=20).json()
		out = alist()
		for track in resp["tracks"]["items"]:
			try:
				name = track.get("name", track["id"])
			except LookupError:
				continue
			else:
				item = f"https://open.spotify.com/track/{track['id']}"
			out.append(cdict(
				name=name,
				url=item,
			))
			if len(out) >= count:
				break
		return out
	def bcsearch(self, query, count=1):
		query = "https://bandcamp.com/search?q=" + quote_plus(query)
		resp = self.session.get(query, timeout=20).content
		out = alist()
		try:
			resp = resp.split(b'<ul class="result-items">', 1)[1]
			tracks = resp.split(b"<!-- search result type=")
			entry = cdict()
			for track in tracks:
				if track.startswith(b"track id=") or track.startswith(b"album id="):
					ttype = track[:5]
					try:
						track = track.split(b'<img src="', 1)[1]
						entry.icon = track[:track.index(b'">')].decode("utf-8", "replace")
					except ValueError:
						pass
					track = track.split(b'<div class="heading">', 1)[1]
					entry.name = track.split(b">", 1)[1].split(b"<", 1)[0].strip().decode("utf-8", "replace")
					entry.url = track.split(b'href="', 1)[1].split(b'"', 1)[0].split(b"?", 1)[0].decode("utf-8", "replace")
					if ttype == b"track":
						out.append(entry)
						break
			if not out:
				raise LookupError(entry)
			return out
		except (LookupError, ValueError):
			return dict(_type="playlist", entries=[])

	# Performs a search, storing and using cached search results for efficiency.
	def search(self, item, force=False, mode=None, images=False, count=1, follow=True):
		item = verify_search(item)
		if mode is None and count == 1:
			try:
				return self.searched[item]
			except KeyError:
				pass
			try:
				output = self.searched.retrieve(item)
			except KeyError:
				pass
			else:
				esubmit(self.search, item, force=force, mode=mode, images=images, count=count)
				return output
		with self.semaphore:
			try:
				output = None
				if not output:
					output = self.extract(item, force, mode=mode, count=count)
				self.searched[item] = output
				return output
			except Exception as ex:
				print_exc()
				return repr(ex)