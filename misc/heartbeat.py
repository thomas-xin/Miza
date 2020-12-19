import os, time

hb = "heartbeat.tmp"
hb_ack = "heartbeat_ack.tmp"

while True:
    if os.path.exists(hb):
        try:
            os.rename(hb, hb_ack)
        except (FileNotFoundError, PermissionError):
            pass
    time.sleep(0.25)