import os, time, datetime, traceback, subprocess, psutil

Process = psutil.Process()


def delete(f):
    while f in os.listdir():
        try:
            os.remove(f)
        except:
            print(traceback.format_exc())
            time.sleep(1)

sd = "shutdown.json"
hb = "heartbeat.json"

delete(sd)
delete(hb)

while not sd in os.listdir():
    name = '"' + "bot.py: " + str(Process.pid) + '"'
    proc = None
    while proc is None:
        os.system("start " + name + " /abovenormal powershell -command python bot.py")
        time.sleep(1)
        for p in psutil.process_iter():
            if "powershell.exe" in p.name().lower():
                if time.time() - p.create_time() < 12:
                    proc = p
                    break
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
            time.sleep(2)
            if hb in os.listdir():
                alive = False
                break
        while True:
            try:
                for child in proc.children():
                    child.kill()
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
delete(sd)
        
print("Shutdown signal confirmed. Press [ENTER] to close.")
input()
