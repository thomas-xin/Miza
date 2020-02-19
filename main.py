import os, time, datetime, traceback, subprocess

kill_args = [
    "taskkill",
    "/f",
    "/fi",
]
    
def kill():
    for a in range(1, 3):
        while a:
            if a != 1:
                args = kill_args + ["windowtitle eq " + opt + name]
            else:
                args = kill_args + ["windowtitle eq " + name]
            a = 0
            out = subprocess.run(
                args=args,
                capture_output=True,
            )
            resp = out.stdout.decode("utf-8")
            print(resp)
            if "SUCCESS" in resp:
                a = 1
            time.sleep(0.001)

def delete(f):
    while f in os.listdir():
        try:
            os.remove(f)
        except:
            print(traceback.format_exc())
            time.sleep(1)
    
name = "C:\\WINDOWS\\system32\\WindowsPowerShell\\v1.0\\powershell.exe"
opt = "Select "
op = (
    "start powershell -NoExit -Command \"$OutputEncoding = "
    + "[Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8\"; .\\bot.bat"
)
#name = "C:\\WINDOWS\\system32\\cmd.exe"
#op = "start cmd /abovenormal /c bot.bat"

sd = "shutdown.json"
hb = "heartbeat.json"

kill()
delete(sd)
delete(hb)

while not sd in os.listdir():
    try:
        os.system(op)
        print("Bot started.")
        time.sleep(20)
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
            for i in range(4):
                time.sleep(0.25)
                if hb in os.listdir():
                    alive = False
                    break
        kill()
        print("Bot closed without shutdown signal, restarting...")
    except:
        print(traceback.format_exc())
    time.sleep(0.5)
    
kill()
delete(hb)
delete(sd)
        
print("Shutdown signal confirmed. Press [ENTER] to close.")
input()
