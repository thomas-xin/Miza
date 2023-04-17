# Loads the install_update module, which makes sure all required libraries are installed to their required versions.
import install_update
from install_update import *

# Makes sure an authentication file exists.
if not os.path.exists("auth.json") or not os.path.getsize("auth.json"):
    print("Authentication file not found. Generating empty template...")
    d = {
        "active_categories": ["MAIN", "STRING", "ADMIN", "VOICE", "IMAGE", "WEBHOOK", "FUN", "OWNER"],
        "prefix": "~",
        "slash_commands": False,
        "webserver_address": "0.0.0.0",
        "webserver_port": "",
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
        "backup_path": "backup"
    }
    import json
    with open("auth.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent="\t")
    input("auth.json generated. Please fill in discord_token and restart bot when done.")
    raise SystemExit


import time, datetime, psutil, subprocess
ffmpeg = "./ffmpeg"
print("Verifying FFmpeg installation...")

if os.name == "nt":
    import requests
    try:
        os.system("color")
    except:
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
        if s != v:
            print(f"FFmpeg version outdated ({v} > {s})")
            raise FileNotFoundError
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
            "ffmpeg.zip",
        ], cwd="misc")
        import zipfile, io
        print("Download complete; extracting new FFmpeg installation...")
        f = "misc/ffmpeg.zip"
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
    if not os.path.exists("misc/poppler"):
        print("Downloading Poppler version 21.10.0...")
        os.mkdir("misc/poppler")
        subprocess.run([sys.executable, "downloader.py", "https://cdn.discordapp.com/attachments/731709481863479436/899556463016554496/Poppler.zip", "poppler.zip"], cwd="misc")
        import zipfile, io
        f = "misc/poppler.zip"
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
        print(f"Downloading FFmpeg...")
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
        except:
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
            if proc.is_running():
                print("\033[1;32;40mHeartbeat started\033[1;37;40m.")
                while alive:
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
                        time.sleep(0.25)
                        ld = os.listdir()
                        if rs in ld or sd in ld:
                            alive = False
                            break
                    if os.path.exists(hb):
                        break
                for child in proc.children(recursive=True):
                    try:
                        child.terminate()
                        try:
                            child.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            child.kill()
                    except:
                        traceback.print_exc()
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
                except psutil.NoSuchProcess:
                    pass
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
            if alive:
                print("\033[1;31;40mBot failed to acknowledge heartbeat signal, restarting...\033[1;37;40m")
            else:
                print("\033[1;31;40mBot sent restart signal, advancing...\033[1;37;40m")
        except KeyboardInterrupt:
            raise
        except:
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
            except:
                traceback.print_exc()
            if proc.is_running():
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except psutil.TimeoutExpired:
                    proc.kill()
            break
        except:
            traceback.print_exc()

delete(sd)
delete(rs)
delete(hb)
delete(hb_ack)

print("Shutdown signal confirmed. Program will now terminate. ")
raise SystemExit
