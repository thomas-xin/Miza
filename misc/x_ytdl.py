import io
import subprocess
import sys
from traceback import print_exc
import zipfile
import requests
from PIL import Image
try:
	import yt_dlp as ytd
except ImportError as ex:
	try:
		import youtube_dl as ytd
	except ImportError:
		raise ex
from .types import list_like
from .util import Request, EvalPipe, esubmit, python

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

def extract_info(url, download=False, process=True):
	resp = ytdl.extract_info(url, download=download, process=process)
	if "entries" in resp and not isinstance(resp["entries"], list_like):
		resp["entries"] = list(resp["entries"])
	return resp

def get_full_storyboard(info):
	"""
	Extracts and processes the storyboard images from the provided video information.

	Args:
		info (dict): A dictionary containing video information, including formats, duration, and storyboard details.

	Returns:
		bytes: A byte buffer containing the storyboard images in a ZIP archive. If an error occurs, returns the thumbnail image URL.

	Raises:
		LookupError: If no storyboard format is found in the provided video information.

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
			requests.get,
			frag["url"],
			headers=storyboard.get("http_headers") or Request.header(),
			stream=True,
		) for frag in fragments]
		curr = 0
		for i, fut in enumerate(futs):
			if i == len(futs) - 1:
				count = last_count
			else:
				count = images_per_storyboard
			frag = fragments[i]
			with fut.result() as resp:
				im = Image.open(resp.raw)
				for index in range(count):
					curr2 = curr + index * frag["duration"] / count
					row = index // storyboard["columns"]
					col = index % storyboard["columns"]
					width = storyboard["width"]
					height = storyboard["height"]
					imc = im.crop((col * width, row * height, col * width + width, row * height + height))
					b = io.BytesIO()
					imc.save(b, "JPEG", quality=75)
					z.writestr(f"{round(curr2)}.jpg", b.getbuffer())
			curr += frag["duration"]
	return b2.getbuffer()

class FFmpegCustomVideoConvertorPP(ytd.postprocessor.FFmpegPostProcessor):
	"Replace the default FFmpegPostProcessor with one that supports seeking as well as custom video formats and codecs."

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
		temp_path = filename.rsplit(".", 1)[0] + "~." + self.format
		input_args = []
		if self.format == "mp4":
			output_args = ["-f", self.format, "-c", "copy"]
			lightning = self.start is not None or self.end is not None
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
				temp_path = filename.rsplit(".", 1)[0] + f"~None~{self.end}~." + self.format
		else:
			if self.start is not None:
				input_args.extend(["-ss", str(self.start)])
			if self.end is not None:
				input_args.extend(["-to", str(self.end)])
			output_args = ["-f", self.format, "-c:v", self.codec, "-b:v", "3072k", "-c:a", "libopus", "-b:a", "160k"]
			lightning = False
		if not lightning:
			temp_path = outpath
		self.real_run_ffmpeg(
			[[filename, input_args]],
			[[temp_path, output_args]],
		)
		if lightning:
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
		if not input_args and source_codec == self.codec:
			output_args.extend(["-c", "copy"])
		else:
			output_args.extend(["-c:a", acodec, "-b:a", "160k"])
		self.real_run_ffmpeg(
			[[filename, input_args]],
			[[outpath, output_args]],
		)
		return [], info

ytd.postprocessor.__dict__["FFmpegCustomVideoConvertorPP"] = FFmpegCustomVideoConvertorPP
ytd.postprocessor.__dict__["FFmpegCustomAudioConvertorPP"] = FFmpegCustomAudioConvertorPP


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	interface.start(background=False)