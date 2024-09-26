from misc.util import EvalPipe, Flush, aexec
import psutil, subprocess, time

proc = psutil.Popen("py ep-2.py", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
pipe = EvalPipe.from_proc(proc, glob=globals())

time.sleep(3)
print(pipe.run("123+456", timeout=1))
pipe.kill()