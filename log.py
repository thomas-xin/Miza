import os


fn = "log.txt"
while True:
    f = open(fn, "rb")
    s = f.read().decode("utf-8")
    
