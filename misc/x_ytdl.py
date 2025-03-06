import io
import os
import subprocess
import sys
from traceback import print_exc
import zipfile
import niquests
from PIL import Image
# Allow fallback (although not recommended as generally the up-to-date version is necessary for most sites)
try:
	import yt_dlp as ytd
except ImportError as ex:
	try:
		import youtube_dl as ytd
	except ImportError:
		raise ex
from .types import list_like, utc
from .util import Request, EvalPipe, esubmit, python, new_playwright_page, CODECS

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
		}
		ytdl = ytd.YoutubeDL(ydl_opts)
	resp = ytdl.extract_info(url, download=download, process=process)
	if "entries" in resp and not isinstance(resp["entries"], list_like):
		resp["entries"] = list(resp["entries"])
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
		if self.format == "mp4" and info.get("vcodec", "none") not in ("none", "png", "jpeg", "gif"):
			output_args = ["-f", self.format, "-c", "copy"]
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
			output_args = ["-f", self.format, *codecs, "-b:v", "2M", "-an"]
			lightning = False
		else:
			fmt = CODECS.get(self.format, self.format)
			if self.start is not None:
				input_args.extend(["-ss", str(self.start)])
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
			output_args = ["-f", fmt, "-c:v", self.codec, "-b:v", "2M"]
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
		if source_ext == self.format and self.start == self.end == None:  # noqa: E711
			return [], info
		name = filename.rsplit(".", 1)[0]
		if self.start is not None or self.end is not None:
			name += f"~{self.start}-{self.end}"
		outpath = name + "." + self.format
		source_codec = self.get_audio_codec(filename)
		A = ytd.postprocessor.ffmpeg.ACODECS
		acodec = A[self.codec][1] or A[self.codec][0]
		input_args = []
		output_args = []
		if self.start is not None:
			input_args.extend(["-ss", str(self.start)])
		if self.end is not None:
			input_args.extend(["-to", str(self.end)])
		if source_codec == self.codec or isinstance(source_codec, str) and source_codec.startswith("pcm_") and self.codec == "wav":
			output_args.extend(["-c", "copy"])
		else:
			# Default to 192k for AAC, 160k for Opus, and 224k for MP3
			bitrate = 224 if self.codec == "mp3" else 192 if self.codec == "aac" else 160
			# Default to 44100 Hz for MP3, 48000 Hz for everything else
			sample_rate = 44100 if self.codec == "mp3" else 48000
			if acodec == "wav":
				acodec = "pcm_s16le"
			output_args.extend(["-c:a", acodec, "-b:a", f"{bitrate}k", "-ar", f"{sample_rate}"])
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