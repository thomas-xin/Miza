import sys
import yt_dlp as ytd
from misc.util import EvalPipe

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

interface = EvalPipe.listen(int(sys.argv[1]), glob=globals())