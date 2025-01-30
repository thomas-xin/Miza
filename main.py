# ruff: noqa: E402
import sys
import subprocess
try:
	import pynvml
except ImportError:
	subprocess.run([sys.executable, "-m", "pip", "install", "pynvml", "--upgrade", "--user"])
import pynvml
try:
	pynvml.nvmlInit()
	DC = pynvml.nvmlDeviceGetCount()
except Exception:
	DC = 0
import os
import json


AUTH = {
	"active_categories": ["MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "WEBHOOK", "FUN", "OWNER"],
	"prefix": "~",
	"slash_commands": False,
	"webserver_address": "0.0.0.0",
	"webserver_port": "",
	"name": "Miza",
	"discord_id": "",
	"discord_token": "",
	"discord_secret": "",
	"owner_id": [],
	"rapidapi_key": "",
	"rapidapi_secret": "",
	"alexflipnote_key": "",
	"giphy_key": "",
	"huggingface_key": "",
	"openai_key": "",
	"openrouter_key": "",
	"backup_path": "",
	"cache_path": "",
	"temp_path": "",
	"default_personality": "",
	"ai_features": bool(DC),
}
modified = False
# Makes sure an authentication file exists.
if not os.path.exists("auth.json") or not os.path.getsize("auth.json"):
	print("Authentication file not found. Generating empty template...")
	orig = ()
else:
	with open("auth.json", "rb") as f:
		orig = json.load(f)
	if orig.get("ai_features") and not orig.get("default_personality"):
		AUTH["default_personality"] = ""
	AUTH.update(orig)
if set(AUTH).difference(orig):
	with open("auth.json", "w", encoding="utf-8") as f:
		json.dump(AUTH, f, indent="\t")
	if "discord_token" not in orig:
		token = input("auth.json generated. Please fill in discord_token and restart bot when done. ")
		if not token:
			raise SystemExit
		AUTH["discord_token"] = token
	print("auth.json updated. Make sure to check empty fields!")


if not AUTH.get("ai_features"):
	os.environ["AI_FEATURES"] = ""

# Loads the install_update module, which makes sure all required libraries are installed to their required versions.
import install_update
from install_update import python, traceback

import time
import datetime
import psutil
import subprocess
ffmpeg = "./ffmpeg"
print("Verifying FFmpeg installation...")

if os.name == "nt":
	try:
		os.system("color")
	except Exception:
		traceback.print_exc()
	# with requests.get("https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", stream=True) as resp:
	try:
		# v = resp.url.rsplit("/", 1)[-1].split("-", 1)[-1].rsplit(".", 1)[0].split("-", 1)[0]
		v = "6.0"
		r = subprocess.run(ffmpeg, stderr=subprocess.PIPE)
		s = r.stderr[:r.stderr.index(b"\n")].decode("utf-8", "replace").strip().lower()
		if s.startswith("ffmpeg"):
			s = s[6:].lstrip()
		if s.startswith("version"):
			s = s[7:].lstrip()
		s = s.split("-", 1)[0]
		# if s != v:
			# print(f"FFmpeg version outdated ({v} > {s})")
			# raise FileNotFoundError
		print(f"FFmpeg version {s} found; skipping installation...")
	except FileNotFoundError:
		print(f"Downloading FFmpeg version {v}...")
		subprocess.run([
			sys.executable, "downloader.py", "-c",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164383061585970/c.b",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164384059838594/c.b",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164384571527218/c.b",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164385146155018/c.b",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164385754337391/c.b",
			"https://cdn.discordapp.com/attachments/886856504135802890/1084164386593181736/c.b",
			"../cache/ffmpeg.zip",
		], cwd="misc")
		import zipfile
		print("Download complete; extracting new FFmpeg installation...")
		f = "cache/ffmpeg.zip"
		with zipfile.ZipFile(f) as z:
			names = [name for name in z.namelist() if "bin/" in name and (".exe" in name or ".dll" in name)]
			for i, name in enumerate(names):
				print(f"{i}/{len(names)}")
				fn = name.rsplit("/", 1)[-1]
				with open(fn, "wb") as y:
					with z.open(name, "r") as x:
						while True:
							b = x.read(1048576)
							if not b:
								break
							y.write(b)
		print("FFmpeg extraction complete.")
		os.remove(f)
	if not os.path.exists("misc/poppler/pdftocairo.exe"):
		print("Downloading Poppler version 21.10.0...")
		try:
			os.mkdir("misc/poppler")
		except FileExistsError:
			pass
		subprocess.run([sys.executable, "downloader.py", "https://cdn.discordapp.com/attachments/1091275350740320258/1107280656347705404/poppler.zip", "../cache/poppler.zip"], cwd="misc")
		import zipfile
		f = "cache/poppler.zip"
		print("Download complete; extracting new Poppler installation...")
		if os.path.exists(f):
			with zipfile.ZipFile(f) as z:
				z.extractall("misc/poppler")
			print("Poppler extraction complete.")
			os.remove(f)
else:
	try:
		subprocess.run(ffmpeg)
	except FileNotFoundError:
		print("Downloading FFmpeg...")
		subprocess.run(("wget", "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"))
		print("Download complete; extracting new FFmpeg installation...")
		os.mkdir(".temp")
		subprocess.run(("tar", "-xf", "ffmpeg-release-amd64-static.tar.xz", "-C", ".temp"))
		fi = os.listdir(".temp")[0]
		os.rename(f".temp/{fi}/ffmpeg", "ffmpeg")
		os.rename(f".temp/{fi}/ffprobe", "ffprobe")
		os.rename(f".temp/{fi}/qt-faststart", "qt-faststart")
		subprocess.run(("rm", "-rf", ".temp"))


# Repeatedly attempts to delete a file, waiting 1 second between attempts.
def delete(f):
	while os.path.exists(f):
		try:
			os.remove(f)
			return
		except Exception:
			traceback.print_exc()
		time.sleep(1)

sd = "shutdown.tmp"
rs = "restart.tmp"
hb = "heartbeat.tmp"
hb_ack = "heartbeat_ack.tmp"

delete(sd)
delete("log.txt")


try:
	# Main watchdog loop.
	att = 0
	while not os.path.exists(sd):
		delete(rs)
		delete(hb)
		proc = psutil.Popen([python, "bot.py"])
		start = time.time()
		print("Bot started with PID \033[1;34;40m" + str(proc.pid) + "\033[1;37;40m.")
		time.sleep(12)
		try:
			alive = True
			was_alive = True
			if proc.is_running():
				print("\033[1;32;40mHeartbeat started\033[1;37;40m.")
				while alive and proc.is_running():
					if not os.path.exists(hb):
						if os.path.exists(hb_ack):
							os.rename(hb_ack, hb)
						else:
							with open(hb, "wb"):
								pass
					print(
						"\033[1;36;40m Heartbeat at "
						+ str(datetime.datetime.now())
						+ "\033[1;37;40m."
					)
					for i in range(64):
						time.sleep(0.5)
						ld = os.listdir()
						if rs in ld or sd in ld:
							alive = False
							print("\033[1;31;40mSignal received! Exiting...\033[1;37;40m")
							break
					if alive and os.path.exists(hb):
						print("\033[1;31;40mHeartbeat missed! Exiting...\033[1;37;40m")
						break
				was_alive = proc.is_running()
				children = list(proc.children(recursive=True))
				try:
					proc.terminate()
					try:
						proc.wait(timeout=2)
					except psutil.TimeoutExpired:
						proc.kill()
				except psutil.NoSuchProcess:
					pass
				for child in children:
					try:
						child.terminate()
						try:
							child.wait(timeout=2)
						except psutil.TimeoutExpired:
							child.kill()
					except Exception:
						traceback.print_exc()
				if os.path.exists(sd):
					break
			if time.time() - start < 60:
				att += 1
			else:
				att = 0
			if att > 16:
				print("\033[1;31;40mBot crashed 16 times in a row. Waiting 5 minutes before trying again.\033[1;37;40m")
				time.sleep(300)
				att = 0
			if not alive:
				print("\033[1;31;40mBot sent restart signal, advancing...\033[1;37;40m")
			elif was_alive:
				print("\033[1;31;40mBot failed to acknowledge heartbeat signal, restarting...\033[1;37;40m")
			else:
				print("\033[1;31;40mBot process disappeared, restarting...\033[1;37;40m")
		except KeyboardInterrupt:
			raise
		except Exception:
			traceback.print_exc()
		time.sleep(0.5)
		import importlib
		importlib.reload(install_update)
		time.sleep(0.5)
finally:
	while True:
		try:
			try:
				for child in proc.children():
					if child.is_running():
						child.terminate()
						try:
							child.wait(timeout=2)
						except psutil.TimeoutExpired:
							child.kill()
			except Exception:
				traceback.print_exc()
			if proc.is_running():
				proc.terminate()
				try:
					proc.wait(timeout=2)
				except psutil.TimeoutExpired:
					proc.kill()
			break
		except Exception:
			traceback.print_exc()

delete(sd)
delete(rs)
delete(hb)
delete(hb_ack)

print("Shutdown signal confirmed. Program will now terminate. ")
raise SystemExit
