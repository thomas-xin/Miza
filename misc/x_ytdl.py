import io
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
from .util import EvalPipe, esubmit

ydl_opts = {
	# "verbose": 1,
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
}
ytdl = ytd.YoutubeDL(ydl_opts)

def extract_info(url, download=False, process=True):
	resp = ytdl.extract_info(url, download=download, process=process)
	if "entries" in resp and not isinstance(resp["entries"], list_like):
		resp["entries"] = list(resp["entries"])
	return resp

def get_full_storyboard(info):
	try:
		storyboard = [f for f in info["formats"] if "storyboard" in f.get("format")][-1]
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
			headers=storyboard["http_headers"],
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

	def __init__(self, downloader=None, codec=None, format=None):
		super().__init__(downloader)
		self.codec = codec
		self.format = format

	@ytd.postprocessor.PostProcessor._restrict_to(images=False)
	def run(self, info):
		filename, source_ext = info['filepath'], info['ext'].lower()
		if source_ext == self.format:
			return [], info
		outpath = ytd.utils.replace_extension(filename, self.format, source_ext)
		if self.format == "mp4":
			self.run_ffmpeg(filename, outpath, ["-f", self.format, "-c", "copy"])
		else:
			self.run_ffmpeg(filename, outpath, ["-f", self.format, "-c:v", self.codec, "-b:v", "3072k", "-c:a", "libopus", "-b:a", "160k"])
		return [], info

class FFmpegCustomAudioConvertorPP(ytd.postprocessor.FFmpegPostProcessor):

	def __init__(self, downloader=None, codec=None, format=None):
		super().__init__(downloader)
		self.codec = codec
		self.format = format

	@ytd.postprocessor.PostProcessor._restrict_to(images=False)
	def run(self, info):
		filename, source_ext = info['filepath'], info['ext'].lower()
		if source_ext == self.format:
			return [], info
		source_codec = self.get_audio_codec(filename)
		outpath = ytd.utils.replace_extension(filename, self.format, source_ext)
		if source_codec == self.codec:
			self.run_ffmpeg(filename, outpath, ["-vn", "-f", self.format, "-c", "copy"])
		else:
			A = ytd.postprocessor.ffmpeg.ACODECS
			acodec = A[self.codec][1] or A[self.codec][0]
			self.run_ffmpeg(filename, outpath, ["-vn", "-f", self.format, "-c:a", acodec, "-b:a", "160k"])
		return [], info

ytd.postprocessor.__dict__["FFmpegCustomVideoConvertorPP"] = FFmpegCustomVideoConvertorPP
ytd.postprocessor.__dict__["FFmpegCustomAudioConvertorPP"] = FFmpegCustomAudioConvertorPP


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	interface.start(background=False)