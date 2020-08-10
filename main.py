# Loads the install_update module, which makes sure all required libraries are installed to their required versions.
from install_update import os, traceback, python


# Makes sure an authentication file exists.
if not os.path.exists("auth.json"):
    print("Authentication file not found. Generating empty template...")
    f = open("auth.json", "wb")
    d = {
        "discord_id": "",
        "discord_token": "",
        "owner_id": [],
        "google_api_key": "",
        "rapidapi_key": "",
        "genius_key": "",
        "papago_id": "",
        "papago_secret": "",
        "knack_id": "",
        "knack_secret": "",
    }
    s = "{\n" + repr(d).replace(" ", "").replace("'", '"').replace(",", ",\n")[1:-1] + "\n}"
    f.write(s.encode("utf-8"))
    f.close()
    input("auth.json generated. Please fill in discord_token and restart bot when done.")
    raise SystemExit


import time, datetime, psutil

# Required on Windows to display terminal colour codes? 🤔
try:
    os.system("color")
except:
    traceback.print_exc()


# Repeatedly attempts to delete a file, waiting 1 second between attempts.
def delete(f):
    while os.path.exists(f):
        try:
            os.remove(f)
            break
        except:
            traceback.print_exc()
        time.sleep(1)

sd = "shutdown.tmp"
rs = "restart.tmp"
hb = "heartbeat.tmp"

delete(sd)


# Main watchdog loop.
att = 0
while not os.path.exists(sd):
    delete(rs)
    delete(hb)
    proc = psutil.Popen([python, "bot.py"], shell=True)
    start = time.time()
    print("Bot started with PID \033[1;34;40m" + str(proc.pid) + "\033[1;37;40m.")
    time.sleep(8)
    if not proc.is_running():
        print("\033[1;31;40mBot closed without shutdown signal, restarting...\033[1;37;40m")
        continue
    try:
        print("\033[1;32;40mHeartbeat started\033[1;37;40m.")
        alive = True
        while alive:
            with open(hb, "wb"):
                pass
            print(
                "\033[1;36;40m Heartbeat at "
                + str(datetime.datetime.now())
                + "\033[1;37;40m."
            )
            for i in range(16):
                time.sleep(0.5)
                ld = os.listdir()
                if rs in ld or sd in ld:
                    alive = False
                    break
            if not alive or os.path.exists(hb):
                alive = False
                break
        found = True
        while found:
            found = False
            try:
                for child in proc.children():
                    child.kill()
                    found = True
            except psutil.NoSuchProcess:
                break
        while True:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                break
        if os.path.exists(sd):
            break
        if time.time() - start < 30:
            att += 1
        else:
            att = 0
        if att > 16:
            print("\033[1;31;40mBot crashed 16 times in a row. Waiting 5 minutes before trying again.\033[1;37;40m")
            time.sleep(300)
            att = 0
        print("\033[1;31;40mBot closed without shutdown signal, restarting...\033[1;37;40m")
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
    time.sleep(0.5)

if proc.is_running():
    try:
        for child in proc.children():
            child.kill()
        proc.kill()
    except:
        pass
    
delete(sd)
delete(rs)
delete(hb)
        
print("Shutdown signal confirmed. Program will now terminate. ")
raise SystemExit