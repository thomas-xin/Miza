import os, time

hb = "heartbeat.tmp"

while True:
    if os.path.exists(hb):
        try:
            os.remove(hb)
        except FileNotFoundError:
            pass
    time.sleep(0.5)