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


def new_playwright_page():
	global browser
	try:
		return browser.new_page()
	except NameError:
		from playwright.sync_api import sync_playwright
		sp = sync_playwright().start()
		browser = sp.firefox.launch()
	return browser.new_page()

def get_spotify(url):
	page = new_playwright_page()
	try:
		page.goto("https://spotifydown.com")
		page.evaluate('document.getElementsByClassName("searchInput")[0].value = "https://open.spotify.com/album/3whFAKu7WRtkGPRSoCRrva"')
		page.evaluate('document.getElementsByClassName("hover:bg-button-active")[0].click()')
	finally:
		page.close()