import os, time, datetime, traceback, psutil

python = ("python3", "python")[os.name == "nt"]


def delete(f):
    while f in os.listdir():
        try:
            os.remove(f)
            break
        except:
            print(traceback.format_exc())
        time.sleep(1)

sd = "shutdown.json"
rs = "restart.json"
hb = "heartbeat.json"

delete(sd)
delete(rs)
delete(hb)

while not sd in os.listdir():
    delete(rs)
    delete(hb)
    proc = psutil.Popen([python, "bot.py"], shell=True)
    print("Bot started with PID " + str(proc.pid) + ".")
    time.sleep(8)
    try:
        print("Heartbeat started.")
        alive = True
        while alive:
            f = open(hb, "wb")
            f.close()
            print(
                "Heartbeat at "
                + str(datetime.datetime.now())
                + "."
            )
            for i in range(16):
                time.sleep(0.5)
                ld = os.listdir()
                if rs in ld or sd in ld:
                    alive = False
                    break
            if not alive or hb in os.listdir():
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
        print("Bot closed without shutdown signal, restarting...")
    except KeyboardInterrupt:
        raise
    except:
        print(traceback.format_exc())
    time.sleep(0.5)
    
delete(hb)
delete(rs)
delete(sd)
        
print("Shutdown signal confirmed. Press [ENTER] to close.")
input()
