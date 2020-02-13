import os, time, traceback

def kill(count=1):
    for i in range(count):
        os.system(filt)

def delete(f):
    while f in os.listdir():
        try:
            os.remove(f)
        except:
            print(traceback.format_exc())
            time.sleep(1)
    
name = "C:\\WINDOWS\\system32\\WindowsPowerShell\\v1.0\\powershell.exe"
filt = "taskkill /f /fi \"windowtitle eq " + name + "\""

kill()
delete("shutdown")
delete("heartbeat")

while not "shutdown" in os.listdir():
    os.system("start powershell .\\bot.bat")
    print("Bot started.")
    time.sleep(30)
    print("Heartbeat started.")
    while not "heartbeat" in os.listdir():
        f = open("heartbeat", "wb")
        f.close()
        print("Heartbeat at " + str(round(time.time(), 3)) + ".")
        time.sleep(5)
        for i in range(5):
            if "heartbeat" in os.listdir():
                break
            time.sleep(1)
    kill(3)
    print("Bot closed without shutdown signal, restarting...")
    
kill(2)
delete("shutdown")
delete("heartbeat")
        
print("Shutdown signal confirmed. Press [ENTER] to close.")
input()
