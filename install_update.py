# ruff: noqa: E401 E402 E731 F401
import sys, os, subprocess, traceback
from traceback import print_exc

# Required to open python on different operating systems
python = sys.executable


if sys.version_info[0] < 3:
	raise ImportError("Python 3 required.")

print("Loading and checking modules...")

with open("requirements.txt", "rb") as f:
	modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")

import importlib.metadata
x = sys.version_info[1]

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
	import collections
	for k in dir(collections.abc):
		setattr(collections, k, getattr(collections.abc, k))

installing = []
def install(m):
	installing.append(subprocess.Popen([python, "-m", "pip", "install", m, "--upgrade"]))
	if len(installing) > 8:
		installing.pop(0).wait()

def try_int(i):
	if isinstance(i, str) and not i.isnumeric():
		return i
	try:
		return int(i)
	except (TypeError, ValueError):
		return i

if os.name == "nt":
	modlist.append("wmi>=1.5.1")
if os.environ.get("AI_FEATURES", True):
	modlist.extend((
		# "accelerate>=0.22.0",
		# "clip-interrogator>=0.6.0",
		# "diffusers>=0.19.0",
		# "fasttext-langdetect>=1.0.5",
		"openai>=1.23.2",
		"opencv-python>=4.8.0.74",
		# "protobuf==3.20.3",
		"pytesseract>=0.3.10",
		# "replicate>=0.11.0",
		"safetensors>=0.3.1",
		# "sentencepiece>=0.1.99",
		# "sentence-transformers>=2.2.2",
		# "soundfile>=0.12.1",
		"tokenizers>=0.13.3",
		# "torch>=2.1.1",
		"transformers>=4.31.0",
		# "tomesd>=0.1.3",
	))

# Parsed requirements.txt
for mod in modlist:
	if mod:
		s = None
		try:
			name = mod
			version = None
			for op in (">=", "==", "<="):
				if op in mod:
					name, version = mod.split(op)
					break
			if name == "yt-dlp":
				raise StopIteration
			v = importlib.metadata.version(name)
			if version is not None:
				try:
					s = repr([try_int(i) for i in v.split(".")]) + op + repr([try_int(i) for i in version.split(".")])
					assert eval(s, {}, {})
				except TypeError:
					s = repr(v.split(".")) + op + repr(version.split("."))
					assert eval(s, {}, {})
		except Exception:
			# Modules may require an older version, replace current version if necessary
			if s:
				print(s)
			print_exc()
			inst = name
			if op in ("==", "<="):
				inst += "==" + version
			install(inst)

# Run pip on any modules that need installing
if installing:
	print("Installing missing or outdated modules, please wait...")
	subprocess.run([python, "-m", "pip", "install", "pip", "--upgrade"])
	for i in installing:
		i.wait()
try:
	importlib.metadata.version("colorspace")
except importlib.metadata.PackageNotFoundError:
	subprocess.run([python, "-m", "pip", "install", "git+https://github.com/retostauffer/python-colorspace"])

try:
	v = importlib.metadata.version("googletrans")
	assert v >= "4.0.0rc1"
except Exception:
	print_exc()
	subprocess.run([python, "-m", "pip", "install", "googletrans==4.0.0rc1", "--upgrade"])

# if os.environ.get("AI_FEATURES", True):
# 	try:
# 		assert importlib.metadata.version("encodec") >= "0.1.2a3"
# 	except (importlib.metadata.PackageNotFoundError, AssertionError):
# 		subprocess.run([python, "-m", "pip", "install", "git+https://github.com/facebookresearch/encodec", "--upgrade"])
# 	try:
# 		if sys.version_info.major == 3 and sys.version_info.minor >= 12:
# 			pass
# 		else:
# 			assert importlib.metadata.version("xformers") >= "0.0.25"
# 		assert importlib.metadata.version("torch") >= "2.2.2"
# 	except (importlib.metadata.PackageNotFoundError, AssertionError):
# 		subprocess.run([python, "-m", "pip", "install", "xformers", "--upgrade"])
# 		subprocess.run([python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cu121"])

if installing:
	subprocess.run([python, "-m", "pip", "install", "-r", "requirements.txt"])

print("Installer terminated.")
