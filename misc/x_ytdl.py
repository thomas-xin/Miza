import sys
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

interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())
interface.start(background=False)