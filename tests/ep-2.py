from misc.util import EvalPipe, Flush

pipe = EvalPipe.from_stdin(glob=globals(), start=True)

pipe.join()