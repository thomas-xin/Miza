import io
import os
import subprocess
import sys
import time
from traceback import print_exc
import zipfile
import niquests
from PIL import Image
import psutil
from .util import is_url, is_discord_attachment, get_duration_2, TEMP_PATH, CODECS_INV
# Allow fallback (although not recommended as generally the up-to-date version is necessary for most sites)
try:
	import yt_dlp as ytd
except ImportError as ex:
	try:
		import youtube_dl as ytd
	except ImportError:
		raise ex
else:
	# Temporary monkey patch until https://github.com/yt-dlp/yt-dlp/pull/12075 is implemented to some degree
	import json
	import urllib
	from yt_dlp.utils import (
		ExtractorError,
		float_or_none,
		int_or_none,
		parse_qs,
		traverse_obj,
		truncate_string,
		try_get,
		unescapeHTML,
		update_url_query,
		url_or_none,
	)
	def _real_extract(self, url):
		slug, video_id = self._match_valid_url(url).group('slug', 'id')

		try:
			data = self._download_json(
				f'https://www.reddit.com/{slug}/.json', video_id, expected_status=403)
		except ExtractorError as e:
			if isinstance(e.cause, json.JSONDecodeError):
				if self._get_cookies('https://www.reddit.com/').get('reddit_session'):
					raise ExtractorError('Your IP address is unable to access the Reddit API', expected=True)
				self.raise_login_required('Account authentication is required')
			raise

		if traverse_obj(data, 'error') == 403:
			reason = data.get('reason')
			if reason == 'quarantined':
				self.raise_login_required('Quarantined subreddit; an account that has opted in is required')
			elif reason == 'private':
				self.raise_login_required('Private subreddit; an account that has been approved is required')
			else:
				raise ExtractorError(f'HTTP Error 403 Forbidden; reason given: {reason}')

		data = data[0]['data']['children'][0]['data']
		video_url = data['url']

		thumbnails = []

		def add_thumbnail(src):
			if not isinstance(src, dict):
				return
			thumbnail_url = url_or_none(src.get('url'))
			if not thumbnail_url:
				return
			thumbnails.append({
				'url': unescapeHTML(thumbnail_url),
				'width': int_or_none(src.get('width')),
				'height': int_or_none(src.get('height')),
				'http_headers': {'Accept': '*/*'},
			})

		for image in try_get(data, lambda x: x['preview']['images']) or []:
			if not isinstance(image, dict):
				continue
			add_thumbnail(image.get('source'))
			resolutions = image.get('resolutions')
			if isinstance(resolutions, list):
				for resolution in resolutions:
					add_thumbnail(resolution)

		info = {
			'thumbnails': thumbnails,
			'age_limit': {True: 18, False: 0}.get(data.get('over_18')),
			**traverse_obj(data, {
				'title': ('title', {truncate_string(left=72)}),
				'alt_title': ('title', {str}),
				'description': ('selftext', {str}, filter),
				'timestamp': ('created_utc', {float_or_none}),
				'uploader': ('author', {str}),
				'channel_id': ('subreddit', {str}),
				'like_count': ('ups', {int_or_none}),
				'dislike_count': ('downs', {int_or_none}),
				'comment_count': ('num_comments', {int_or_none}),
			}),
		}

		parsed_url = urllib.parse.urlparse(video_url)

		# Check for embeds in text posts, or else raise to avoid recursing into the same reddit URL
		if 'reddit.com' in parsed_url.netloc and f'/{video_id}/' in parsed_url.path:
			entries = []
			for media in traverse_obj(data, ('media_metadata', ...), expected_type=dict):
				if not media.get('id') or media.get('e') != 'RedditVideo':
					continue
				formats = []
				if media.get('hlsUrl'):
					formats.extend(self._extract_m3u8_formats(
						unescapeHTML(media['hlsUrl']), video_id, 'mp4', m3u8_id='hls', fatal=False))
				if media.get('dashUrl'):
					formats.extend(self._extract_mpd_formats(
						unescapeHTML(media['dashUrl']), video_id, mpd_id='dash', fatal=False))
				if formats:
					entries.append({
						'id': media['id'],
						'display_id': video_id,
						'formats': formats,
						**info,
					})
			if entries:
				return self.playlist_result(entries, video_id, **info)
			self.raise_no_formats('No media found', expected=True, video_id=video_id)
			return {**info, 'id': video_id}

		# Check if media is hosted on reddit:
		reddit_video = traverse_obj(data, (
			(None, ('crosspost_parent_list', ...)), ('secure_media', 'media'), 'reddit_video'), get_all=False)
		if reddit_video:
			playlist_urls = [
				try_get(reddit_video, lambda x: unescapeHTML(x[y]))
				for y in ('dash_url', 'hls_url')
			]

			# Update video_id
			display_id = video_id
			video_id = self._search_regex(
				r'https?://v\.redd\.it/(?P<id>[^/?#&]+)', reddit_video['fallback_url'],
				'video_id', default=display_id)

			dash_playlist_url = playlist_urls[0] or f'https://v.redd.it/{video_id}/DASHPlaylist.mpd'
			hls_playlist_url = playlist_urls[1] or f'https://v.redd.it/{video_id}/HLSPlaylist.m3u8'
			qs = traverse_obj(parse_qs(hls_playlist_url), {
				'f': ('f', 0, {lambda x: ','.join([x, 'subsAll']) if x else 'hd,subsAll'}),
			})
			hls_playlist_url = update_url_query(hls_playlist_url, qs)

			formats = [{
				'url': unescapeHTML(reddit_video['fallback_url']),
				'height': int_or_none(reddit_video.get('height')),
				'width': int_or_none(reddit_video.get('width')),
				'tbr': int_or_none(reddit_video.get('bitrate_kbps')),
				'acodec': 'none',
				'vcodec': 'h264',
				'ext': 'mp4',
				'format_id': 'fallback',
				'format_note': 'DASH video, mp4_dash',
			}]
			hls_fmts, subtitles = self._extract_m3u8_formats_and_subtitles(
				hls_playlist_url, display_id, 'mp4', m3u8_id='hls', fatal=False)
			formats.extend(hls_fmts)
			dash_fmts, dash_subs = self._extract_mpd_formats_and_subtitles(
				dash_playlist_url, display_id, mpd_id='dash', fatal=False)
			formats.extend(dash_fmts)
			self._merge_subtitles(dash_subs, target=subtitles)

			return {
				**info,
				'id': video_id,
				'display_id': display_id,
				'formats': formats,
				'subtitles': subtitles or self.extract_subtitles(video_id),
				'duration': int_or_none(reddit_video.get('duration')),
			}

		if parsed_url.netloc == 'v.redd.it':
			self.raise_no_formats('This video is processing', expected=True, video_id=video_id)
			return {
				**info,
				'id': parsed_url.path.split('/')[1],
				'display_id': video_id,
			}

		# Hosted on reddit as an image (most likely gif)
		if parsed_url.netloc == 'i.redd.it':
			formats = [{
				'url': video_url,
				'acodec': 'none',
				'ext': video_url.rsplit('.', 1)[-1],
				'format_id': video_url.rsplit('.', 1)[-1],
				# Must GET only accepting raw file content
				'http_headers': {'Accept': 'video/*,image/*'},
			}]
			return {
				**info,
				'id': parsed_url.path.split('/')[1],
				'display_id': video_id,
				'formats': formats,
			}

		# Not hosted on reddit, must continue extraction
		return {
			**info,
			'display_id': video_id,
			'_type': 'url_transparent',
			'url': video_url,
		}
	ytd.extractor.reddit.RedditIE._real_extract = _real_extract

	real_download = ytd.downloader.http.HttpFD.real_download
	def trial_download(self, filename, info_dict):
		proc = None
		try:
			url = info_dict["url"]
			if not is_url(url) or is_discord_attachment(url) or url.endswith(".ts"):
				raise ValueError
			print(url)
			t = time.time()
			args = ["streamshatter", url, "-c", TEMP_PATH, "-H", json.dumps(info_dict.get("http_headers", {})), "-l", "48", "-t", "30", filename]
			proc = psutil.Popen(args, stdin=subprocess.DEVNULL)
			proc.wait(timeout=32)
			elapsed = time.time() - t
			assert os.path.exists(filename) and os.path.getsize(filename)
		except ValueError:
			pass
		except Exception as ex:
			sys.stderr.write(f"{repr(ex)}\n")
			if proc and proc.is_running():
				proc.terminate()
		else:
			sys.stdout.write("\n")
			byte_counter = os.path.getsize(filename)
			self._hook_progress({
				'downloaded_bytes': byte_counter,
				'total_bytes': byte_counter,
				'filename': filename,
				'status': 'finished',
				'elapsed': elapsed,
				'ctx_id': info_dict.get('ctx_id'),
			}, info_dict)
			return True
		return real_download(self, filename, info_dict)
	ytd.downloader.http.HttpFD.real_download = trial_download
from .types import list_like, utc
from .util import Request, EvalPipe, esubmit, python, new_playwright_page

ts = 0
ytdl = None

def entry_from_ytdl(resp):
	return dict(
		name=resp.get("title"),
		url=resp.get("webpage_url") or resp.get("url"),
		thumbnails=resp.get("thumbnails") or [resp.get("thumbnail")],
		duration=resp.get("duration"),
	)

def extract_info(url, download=False, process=True):
	global ts, ytdl
	ts2 = utc()
	if not ytdl or ts2 - ts > 300:
		ts = ts2
		ydl_opts = {
			"quiet": 1,
			"format": "bestvideo+bestaudio/best",
			"overwrites": 1,
			"nocheckcertificate": 1,
			"no_call_home": 1,
			"nooverwrites": 1,
			"noplaylist": 1,
			"logtostderr": 0,
			"ignoreerrors": 0,
			"default_search": "auto",
			"source_address": "0.0.0.0",
			"cookiesfrombrowser": ["firefox"],
			"extractor_args": {
				"youtube": {
					"player_client": ["default", "web_safari"],
					"player_js_version": ["actual"]
				}
			}
		}
		ytdl = ytd.YoutubeDL(ydl_opts)
	resp = ytdl.extract_info(url, download=download, process=process)
	if "entries" in resp:
		entries = []
		for entry in resp["entries"]:
			if "entries" not in entry:
				entries.append(entry)
				continue
			elif not isinstance(entry["entries"], list_like):
				entry["entries"] = list(entry["entries"])
			entries.extend(entry["entries"])
		resp["entries"] = entries
	return resp

def get_audio_spotify(url, fn):
	downloader = "https://spotifymate.com"
	page = new_playwright_page()
	with page:
		page.goto(downloader)
		inputs = page.locator("#url")
		inputs.fill(url)
		page.locator("#send").click()
		success = page.locator(".is-success").first
		if not success:
			raise RuntimeError("Failed to download Spotify audio.")
		stream = success.get_attribute("href")
	if not stream:
		raise RuntimeError("Failed to download Spotify audio.")
	with niquests.get(stream, headers=Request.header(), stream=True) as resp:
		resp.raise_for_status()
		with open(fn, "wb") as f:
			for chunk in resp.iter_content(65536):
				f.write(chunk)
	assert os.path.exists(fn) and os.path.getsize(fn), "Failed to download Spotify audio."
	return fn

def get_full_storyboard(info):
	"""
	Extracts and processes the storyboard images from the provided video information.

	Args:
		info (dict): A dictionary containing video information, including formats, duration, and storyboard details.

	Returns:
		bytes: A byte buffer containing the storyboard images in a ZIP archive.
		str: The URL of the video thumbnail if the storyboard could not be extracted.

	The function performs the following steps:
		1. Extracts the storyboard format from the video information.
		2. Creates a ZIP archive to store the storyboard images.
		3. Calculates the number of images per storyboard and the duration of each fragment.
		4. Downloads the storyboard fragments concurrently.
		5. Processes each fragment to extract individual images and saves them in the ZIP archive.
	"""
	try:
		storyboards = [f for f in info["formats"] if (f.get("format") and "storyboard" in f.get("format")) or f.get("format_note") == "storyboard"]
		storyboard = sorted(storyboards, key=lambda x: x.get("height"))[-1]
	except LookupError:
		print_exc()
		return info["thumbnail"]
	b2 = io.BytesIO()
	with zipfile.ZipFile(b2, "w", compression=zipfile.ZIP_STORED) as z:
		images_per_storyboard = storyboard["rows"] * storyboard["columns"]
		duration = info["duration"]
		fragments = storyboard["fragments"]
		last_duration = duration - sum(frag["duration"] for frag in fragments[:-1])
		last_count = round(last_duration / fragments[0]["duration"])
		futs = [esubmit(
			niquests.get,
			frag["url"],
			headers=storyboard.get("http_headers") or Request.header(),
			stream=True,
		) for frag in fragments]
		timestamps = set()
		curr = 0
		for i, fut in enumerate(futs):
			if i == len(futs) - 1:
				count = last_count
			else:
				count = images_per_storyboard
			frag = fragments[i]
			with fut.result() as resp:
				if not resp.ok:
					return info["thumbnail"]
				im = Image.open(resp.raw)
				for index in range(count):
					curr2 = curr + index * frag["duration"] / count
					timestamp = round(curr2)
					# Skip duplicate timestamps
					if timestamp in timestamps:
						continue
					timestamps.add(timestamp)
					row = index // storyboard["columns"]
					col = index % storyboard["columns"]
					width = storyboard["width"]
					height = storyboard["height"]
					imc = im.crop((col * width, row * height, col * width + width, row * height + height))
					b = io.BytesIO()
					imc.save(b, "JPEG", quality=75)
					z.writestr(f"{timestamp}.jpg", b.getbuffer())
			curr += frag["duration"]
	return b2.getbuffer()

class FFmpegCustomVideoConvertorPP(ytd.postprocessor.FFmpegPostProcessor):
	"Replace the default FFmpegPostProcessor with one that supports seeking as well as custom video formats and codecs. Additionally supports adding a thumbnail to purely audio files as the cover image."

	def __init__(self, downloader=None, codec=None, format=None, start=None, end=None, thumbnail=None):
		super().__init__(downloader)
		self.codec = codec
		self.format = format
		self.start = start
		self.end = end
		self.thumbnail = thumbnail

	@ytd.postprocessor.PostProcessor._restrict_to(images=False)
	def run(self, info):
		filename, source_ext = info['filepath'], info['ext'].lower()
		if source_ext == self.format and self.start == self.end == None:  # noqa: E711
			return [], info
		name = filename.rsplit(".", 1)[0]
		if self.start is not None or self.end is not None:
			name += f"~{self.start}-{self.end}"
		if self.thumbnail:
			name += "~i"
		outpath = name + "." + self.format
		temp_path = filename.rsplit(".", 1)[0] + "~." + self.format
		before = []
		input_args = []
		if not self.codec and self.format == "mp4" and info.get("vcodec", "none") not in ("none", "png", "jpeg", "gif"):
			output_args = ["-f", self.format, "-c", "copy", "-loop", "0"]
			lightning = self.start is not None or self.end is not None
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
				temp_path = filename.rsplit(".", 1)[0] + f"~None~{self.end}~." + self.format
		elif self.format in ("avif", "webp", "gif", "apng"):
			if self.start is not None:
				input_args.extend(["-ss", str(self.start)])
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
			if self.format == "avif":
				codecs = ["-c:v", "libsvtav1"]
			elif self.format == "webp":
				codecs = ["-c:v", "libwebp"]
			else:
				codecs = []
			output_args = ["-f", self.format, *codecs, "-b:v", "2M", "-vbr", "on", "-an", "-loop", "0"]
			lightning = False
		else:
			fmt = CODECS_INV.get(self.format, self.format)
			if self.start is not None:
				input_args.extend(["-ss", str(self.start)])
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
			output_args = ["-f", fmt, "-c:v", self.codec, "-b:v", "2M", "-vbr", "on"]
			if self.format == "mp4":
				# MP4 supports just about any audio codec, but WebM and MKV do not. We assume the audio codec is not WMA, as it is highly unlikely any website would use it for streaming videos.
				output_args.extend(["-c:a", "copy"])
			else:
				output_args.extend(["-c:a", "libopus", "-b:a", "160k", "-ar", "48k"])
			lightning = False
		if self.thumbnail:
			before_filename = self.thumbnail
			before_args = ["-loop", "1"]
			output_args.extend("-shortest")
			before.append([before_filename, before_args])
		if not lightning:
			temp_path = outpath
		self.real_run_ffmpeg(
			[*before, [filename, input_args]],
			[[temp_path, output_args]],
		)
		if lightning:
			# Run lightning-trim script to trim the video, as simple FFmpeg trims often do not play properly (such as black screens at the start).
			# Lightning-trim computes the nearest keyframe after the start time, and forces a reencode for any cut-off frames occuring before said keyframe.
			start = str(self.start) if self.start is not None else "0"
			end = str(self.end) if self.end is not None else "86400"
			args = [python, "misc/lightning.py", temp_path, start, end, outpath]
			subprocess.run(args)
			return [temp_path], info
		return [], info

class FFmpegCustomAudioConvertorPP(ytd.postprocessor.FFmpegPostProcessor):
	"Replace the default FFmpegPostProcessor with one that supports seeking as well as custom audio codecs and formats."

	def __init__(self, downloader=None, codec=None, format=None, start=None, end=None):
		super().__init__(downloader)
		self.codec = codec
		self.format = format
		self.start = start
		self.end = end

	@ytd.postprocessor.PostProcessor._restrict_to(images=False)
	def run(self, info):
		filename, source_ext = info['filepath'], info['ext'].lower()
		dur, bps, cdc, ac = get_duration_2(filename)
		cbr = max(60, min(bps / 1000 / (0.875 if cdc == "mp3" else 0.75 if cdc == "aac" else 0.625), 256))
		if not dur or dur < 1920:
			mbr = 160
		elif dur < 3840:
			mbr = 128
		elif dur < 7680:
			mbr = 108
		else:
			mbr = 96
		print(dur, bps, cbr, mbr)
		if source_ext == self.format and self.start == self.end == None:  # noqa: E711
			return [], info
		name = filename.rsplit(".", 1)[0]
		if self.start is not None or self.end is not None:
			name += f"~{self.start}-{self.end}"
		outpath = name + "." + self.format
		source_codec = cdc
		A = ytd.postprocessor.ffmpeg.ACODECS
		acodec = A[self.codec][1] or A[self.codec][0]
		input_args = []
		output_args = []
		if self.start is not None:
			input_args.extend(["-ss", str(self.start)])
		if self.end is not None:
			input_args.extend(["-to", str(self.end)])
		if source_codec == self.codec and cbr <= mbr or isinstance(source_codec, str) and source_codec.startswith("pcm_") and self.codec == "wav":
			output_args.extend(["-c", "copy"])
		else:
			# Default to 192k for AAC, 160k for Opus, and 224k for MP3
			bitrate = min(mbr, cbr * (0.875 if self.codec == "mp3" else 0.75 if self.codec == "aac" else 0.625))
			# Default to 44100 Hz for MP3, 48000 Hz for everything else
			sample_rate = 44100 if self.codec == "mp3" else 48000
			if acodec == "wav":
				acodec = "pcm_s16le"
			output_args.extend(["-c:a", acodec, "-b:a", f"{bitrate}k", "-vbr", "on", "-ar", f"{sample_rate}"])
		self.real_run_ffmpeg(
			[[filename, input_args]],
			[[outpath, output_args]],
		)
		return [], info

for pp in (FFmpegCustomVideoConvertorPP, FFmpegCustomAudioConvertorPP):
	ytd.postprocessor.postprocessors.__dict__[pp.__name__] = pp
	ytd.postprocessor.postprocessors.value[pp.__name__] = pp


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	interface.start(background=False)