# Loads the install_update module, which makes sure all required libraries are installed to their required versions.
from install_update import *

# Makes sure an authentication file exists.
if not os.path.exists("auth.json") or not os.path.getsize("auth.json"):
    print("Authentication file not found. Generating empty template...")
    d = {
        "prefix": "~",
        "slash_commands": False,
        "webserver_port": "",
        "discord_token": "",
        "owner_id": [],
        "google_api_key": "",
        "rapidapi_key": "",
        "alexflipnote_key": "",
        "giphy_key": "",
        "papago_id": "",
        "papago_secret": "",
    }
    import json
    with open("auth.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)
    input("auth.json generated. Please fill in discord_token and restart bot when done.")
    raise SystemExit


import time, datetime, psutil

# Required on Windows to display terminal colour codes? 🤔
if os.name == "nt":
    try:
        os.system("color")
    except:
        traceback.print_exc()
    import requests, subprocess
    ffmpeg = "ffmpeg"
    print("Verifying FFmpeg installation...")
    with requests.get("https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", stream=True) as resp:
        try:
            v = resp.url.rsplit("/", 1)[-1].split("-", 1)[-1].rsplit(".", 1)[0].split("-", 1)[0]
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
            subprocess.run([sys.executable, "downloader.py", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", "ffmpeg.zip"], cwd="misc")
            import zipfile, io
            print("Download complete; extracting new FFmpeg installation...")
            f = "misc/ffmpeg.zip"
            with zipfile.ZipFile(f) as z:
                names = [name for name in z.namelist() if "/bin/" in name and ".exe" in name]
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
    if os.path.exists("misc") and not os.path.exists("misc/ffmpeg-c"):
        print("Downloading ffmpeg version 4.2.2...")
        os.mkdir("misc/ffmpeg-c")
        subprocess.run([sys.executable, "downloader.py", "https://drive.google.com/u/0/uc?export=download&confirm=QLKC&id=168rCEMiRXi9X_o3pVEl_2cVWTcYGgR4N", "ffmpeg-c.zip"], cwd="misc")
        import zipfile, io
        print("Download complete; extracting new FFmpeg installation...")
        f = "misc/ffmpeg-c.zip"
        with zipfile.ZipFile(f) as z:
            names = z.namelist()
            for i, name in enumerate(names):
                print(f"{i}/{len(names)}")
                fn = "misc/ffmpeg-c/" + name.rsplit("/", 1)[-1]
                with open(fn, "wb") as y:
                    with z.open(name, "r") as x:
                        while True:
                            b = x.read(1048576)
                            if not b:
                                break
                            y.write(b)
        print("FFmpeg extraction complete.")
        os.remove(f)



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


# Main watchdog loop.
att = 0
while not os.path.exists(sd):
    delete(rs)
    delete(hb)
    proc = psutil.Popen([python, "bot.py"], shell=True)
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
                for i in range(32):
                    time.sleep(0.25)
                    ld = os.listdir()
                    if rs in ld or sd in ld:
                        alive = False
                        break
                if os.path.exists(hb):
                    break
            for child in proc.children(recursive=True):
                try:
                    child.kill()
                except:
                    traceback.print_exc()
            try:
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

if proc.is_running():
    try:
        for child in proc.children():
            child.kill()
    except:
        traceback.print_exc()
    proc.kill()

delete(sd)
delete(rs)
delete(hb)
delete(hb_ack)

print("Shutdown signal confirmed. Program will now terminate. ")
raise SystemExit