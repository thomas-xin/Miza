import os, time, traceback

while not "shutdown" in os.listdir():
    os.system("start cmd /c start.bat")
    time.sleep(30)
    while not "heartbeat" in os.listdir():
        f = open("heartbeat", "wb")
        f.close()
        print("Heartbeat at " + str(round(time.time(), 3)) + ".")
        time.sleep(10)
    print("Bot closed without shutdown signal, restarting...")
while "shutdown" in os.listdir():
    try:
        os.remove("shutdown")
    except:
        print(traceback.format_exc())
        time.sleep(1)
