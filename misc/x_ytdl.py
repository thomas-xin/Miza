import io
import sys
from traceback import print_exc
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
from .util import EvalPipe

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

def get_storyboard(info, pos=0):
	try:
		storyboard = [f for f in info["formats"] if "storyboard" in f.get("format")][-1]
	except LookupError:
		print_exc()
		return info["thumbnail"]
	images_per_storyboard = storyboard["rows"] * storyboard["columns"]
	duration = info["duration"]
	fragments = list(storyboard["fragments"])
	last_duration = duration - sum(frag["duration"] for frag in fragments[:-1])
	last_count = round(last_duration / fragments[0]["duration"])
	curr = 0
	while True:
		frag = fragments.pop(0)
		if not fragments or curr + frag["duration"] > pos:
			count = images_per_storyboard if fragments else last_count
			if count <= 1:
				return frag["url"]
			index = max(min(round((pos - curr) / frag["duration"] * count), count - 1), 0)
			row = index // storyboard["columns"]
			col = index % storyboard["columns"]
			width = storyboard["width"]
			height = storyboard["height"]
			with requests.get(frag["url"], headers=storyboard["http_headers"], stream=True) as resp:
				im = Image.open(resp.raw)
				im = im.crop((col * width, row * height, col * width + width, row * height + height))
				b = io.BytesIO()
				im.save(b, "JPEG", quality=75)
				return b.getbuffer()
		curr += frag["duration"]


if __name__ == "__main__":
	interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
	interface.start(background=False)