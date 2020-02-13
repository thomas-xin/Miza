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
op = "start powershell -NoExit -Command \"$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8\"; .\\bot.bat"
#name = "C:\\WINDOWS\\system32\\cmd.exe"
#op = "start cmd /abovenormal /c bot.bat"
filt = "taskkill /f /fi \"windowtitle eq " + name + "\""

kill()
delete("shutdown")
delete("heartbeat")

while not "shutdown" in os.listdir():
    os.system(op)
    print("Bot started.")
    time.sleep(30)
    print("Heartbeat started.")
    alive = True
    while alive:
        f = open("heartbeat", "wb")
        f.close()
        print("Heartbeat at " + str(round(time.time(), 3)) + ".")
        time.sleep(2)
        for i in range(3):
            time.sleep(1)
            if "heartbeat" in os.listdir():
                alive = False
                break
    kill(3)
    print("Bot closed without shutdown signal, restarting...")
    
kill(2)
delete("shutdown")
delete("heartbeat")
        
print("Shutdown signal confirmed. Press [ENTER] to close.")
input()
