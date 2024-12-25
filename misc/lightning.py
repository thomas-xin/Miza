import subprocess
import os
import sys

def round_min(x):
	y = int(x)
	if x == y:
		return y
	return x

def strnum(num):
	return str(round_min(round(num, 6)))

def time_disp(s):
	s = float(s)
	output = strnum(s % 60)
	if len(output) < 2:
		output = "0" + output
	if s >= 60:
		temp = strnum((s // 60) % 60)
		if len(temp) < 2 and s >= 3600:
			temp = "0" + temp
		output = temp + ":" + output
		if s >= 3600:
			temp = strnum((s // 3600) % 24)
			if len(temp) < 2 and s >= 86400:
				temp = "0" + temp
			output = temp + ":" + output
			if s >= 86400:
				output = strnum(s // 86400) + ":" + output
	else:
		output = "0:" + output
	return output

def time_parse(ts):
	if ts == "N/A":
		return inf
	data = ts.split(":")
	if len(data) >= 5: 
		raise TypeError("Too many time arguments.")
	mults = (1, 60, 3600, 86400)
	return round_min(sum(float(count) * mult for count, mult in zip(data, reversed(mults[:len(data)]))))

if len(sys.argv) >= 4:
	fi, start, end = sys.argv[1:4]
	name, fmt = fi.rsplit(".", 1) if "." in fi else (fi, "mp4")
	if len(sys.argv) >= 5:
		fn = sys.argv[4]
		name, fmt = fn.rsplit(".", 1) if "." in fn else (fn, "mp4")
	else:
		fn = name + "~t" + "." + fmt
	start, end = time_parse(start), time_parse(end)
	easygui = None
else:
	import easygui
	fi = None
	if len(sys.argv) >= 2:
		fi = sys.argv[1]
	if not fi:
		fi = easygui.fileopenbox("Please select file to trim:")
	if not fi:
		raise SystemExit
	dur = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi]).strip().decode("ascii")
	if not dur or dur == "N/A":
		dur = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi]).strip().decode("ascii")
		if not dur or dur == "N/A":
			dur = 0
	fmts = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "format=format_name", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi]).strip().decode("ascii").replace("matroska", "mkv").split(",")
	name, ext = fi.rsplit(".", 1) if "." in fi else (fi, "mp4")
	if ext in fmts:
		fmt = ext
	else:
		fmt = fmts[0]
	fn = name + "~t" + "." + fmt
	text = easygui.textbox("Please enter start/end, and optionally name:", text=f"START: 0:00\nEND: {time_disp(dur)}\nNAME: {fn}")
	if not text:
		raise SystemExit
	try:
		lines = [line for line in text.splitlines() if line.strip()]
		start = time_parse(lines[0].split(":", 1)[-1])
		end = time_parse(lines[1].split(":", 1)[-1])
		fn = lines[2].split(":", 1)[-1].strip()
		if '"' in fn or "'" in fn:
			import ast
			fn = ast.literal_eval(fn)
		name, fmt = fn.rsplit(".", 1) if "." in fn else (fn, "mp4")
	except Exception:
		easygui.exceptionbox()
		raise SystemExit
while name and ".." in name:
	name = name.replace("..", "_")

frun = subprocess.run
subprocess.run = lambda proc, *args, **kwargs: print(proc) or frun(proc, *args, **kwargs)

fproc = subprocess.Popen
subprocess.Popen = lambda proc, *args, **kwargs: print(proc) or fproc(proc, *args, **kwargs)

try:
	proc = subprocess.Popen(["ffprobe", "-read_intervals", str(max(0, start - 30)) + "%+60", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi], stdout=subprocess.PIPE)

	key = prevkey = 0
	while key < start:
		prevkey = key
		key = float(proc.stdout.readline())

	proc.terminate()
	if os.path.exists(fn):
		os.remove(fn)

	if abs(start - key) < 1 / 30:
		starting = ["-ss", str(start)] if start >= 1 / 30 else []
		subprocess.run(["ffmpeg", "-y", "-hwaccel", "auto", *starting, "-to", str(end), "-i", fi, "-reset_timestamps", "1", "-c", "copy", fn])
	else:
		fa = name + "~0" + "." + "ts"
		starting = ["-ss", str(prevkey)] if prevkey else []
		starting2 = ["-ss", str(start - prevkey)] if start - prevkey else []
		aproc = subprocess.Popen(["ffmpeg", "-y", "-hwaccel", "auto", "-vn", *starting, "-i", fi, *starting2, "-to", str(end), "-c:a", "copy", fa])

		f1 = name + "~1" + "." + "ts"
		f2 = name + "~2" + "." + "ts"
		f3 = None
		cproc = subprocess.Popen(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi], stdout=subprocess.PIPE)
		fps = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1", "-i", fi]).strip().split(b"\n", 1)[0].decode("ascii")
		cdc = cproc.stdout.read().strip().split(b"\n", 1)[0].strip().decode("ascii")
		if cdc == "av1":
			cdc = "libsvtav1"
		if not fps or fps == "N/A":
			fps = 1000
		else:
			fps = eval(fps, {}, {})
		print("TRIM:", start, key, end, fps)
		starting = ["-ss", str(start)] if start else []
		vproc = subprocess.Popen(["ffmpeg", "-y", "-hwaccel", "auto", *starting, "-to", str(key - 1 / fps), "-an", "-i", fi, "-crf", "20", "-c:v", cdc, f1])
		starting = ["-ss", str(key + 1 / fps)] if key else []
		subprocess.run(["ffmpeg", "-y", "-hwaccel", "auto", *starting, "-to", str(end), "-an", "-i", fi, "-c:v", "copy", "-avoid_negative_ts", "1", f2])
		vproc.wait()

		if "://" in fi and os.path.exists(f2) and os.path.getsize(f2):
			dur = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", "-i", f2]).strip().decode("ascii")
			if not dur or dur == "N/A":
				dur = subprocess.check_output(["ffprobe", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", "-i", f2]).strip().decode("ascii")
			if dur != "N/A":
				dur = float(dur)
				# print("DUR:", dur, start, key, end)
				diff = dur - (end - key)
				if diff > 1 / fps:
					f3 = name + "~3" + "." + "ts"
					ndur = key - start - diff - 1 / fps
					print(f"Mismatch of {diff}s ({dur}/{end - key}), retrimming to {ndur}...")
					if abs(ndur) <= 1 / fps:
						f1, f2 = f2, None
					elif ndur <= 0:
						subprocess.run(["ffmpeg", "-y", "-hwaccel", "auto", "-an", "-i", f2, "-ss", str(-ndur), "-avoid_negative_ts", "1", "-c:v", cdc, "-crf", "20", f3])
						f1, f2, f3 = f3, None, f1
					else:
						subprocess.run(["ffmpeg", "-y", "-hwaccel", "auto", "-t", str(ndur), "-an", "-i", f1, "-c:v", "copy", f3])
						f1, f3 = f3, f1

		aproc.wait()

		if fa and not (os.path.exists(fa) and os.path.getsize(fa)):
			fa = None
		fc = None
		if f2 or fa:
			if f2:
				fc = name + "~c" + "." + "txt"
				with open(fc, "w", encoding="utf-8") as f:
					f.write("file " + repr(os.path.abspath(f1)) + "\n")
					f.write("file " + repr(os.path.abspath(f2)) + "\n")
				fin = ["-f", "concat", "-i", fc]
			else:
				fin = ["-i", f1]
			subprocess.run(["ffmpeg", "-y", "-hwaccel", "auto", "-safe", "0", *fin, *(("-i", fa) if fa else ()), "-c:v", "copy", "-c:a", "copy", fn])
		else:
			if os.path.exists(fn):
				os.remove(fn)
			os.rename(f1, fn)
		if os.path.exists(fn) and os.path.getsize(fn):
			for fd in (fa, f1, f2, f3, fc):
				if not fd:
					continue
				try:
					os.remove(fd)
				except Exception:
					pass
	if not os.path.exists(fn) or not os.path.getsize(fn):
		raise RuntimeError("Unable to save file. Please check log for info.")
except Exception:
	if not easygui:
		raise
	easygui.exceptionbox()