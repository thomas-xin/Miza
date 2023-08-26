import sys, os, subprocess, traceback
from traceback import print_exc

# Required to open python on different operating systems
python = sys.executable


if sys.version_info[0] < 3:
    raise ImportError("Python 3 required.")

print("Loading and checking modules...")

with open("requirements.txt", "rb") as f:
    modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")

try:
    import pkg_resources
except:
    print_exc()
    subprocess.run(["pip", "install", "setuptools", "--upgrade", "--user"])
    import pkg_resources
x = sys.version_info[1]

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
	import collections
	for k in dir(collections.abc):
		setattr(collections, k, getattr(collections.abc, k))

installing = []
install = lambda m: installing.append(subprocess.Popen([python, "-m", "pip", "install", m, "--upgrade", "--user"]))

def try_int(i):
    if type(i) is str and not i.isnumeric():
        return i
    try:
        return int(i)
    except:
        return i

if os.environ.get("AI_FEATURES", True):
	modlist.extend((
		"accelerate>=0.22.0",
		"clip-interrogator>=0.6.0",
		"diffusers>=0.19.0",
		"openai>=0.27.8",
		"opencv-python>=4.8.0.74",
		"protobuf==3.20.3",
		"pytesseract>=0.3.10",
		"replicate>=0.11.0",
		"safetensors>=0.3.1",
		"sentencepiece>=0.1.99",
		"sentence-transformers>=2.2.2",
		"soundfile>=0.12.1",
		"tiktoken>=0.4.0",
		"tokenizers>=0.13.3",
		"torch>=2.0.1",
		"transformers>=4.31.0",
		"xformers>=0.0.21",
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
            v = pkg_resources.get_distribution(name).version
            if version is not None:
                try:
                    s = repr([try_int(i) for i in v.split(".")]) + op + repr([try_int(i) for i in version.split(".")])
                    assert eval(s, {}, {})
                except TypeError:
                    s = repr(v.split(".")) + op + repr(version.split("."))
                    assert eval(s, {}, {})
        except:
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
    subprocess.run([python, "-m", "pip", "install", "pip", "--upgrade", "--user"])
    for i in installing:
        i.wait()
try:
    pkg_resources.get_distribution("colorspace")
except pkg_resources.DistributionNotFound:
    subprocess.run([python, "-m", "pip", "install", "git+https://github.com/retostauffer/python-colorspace", "--user"])

# try:
    # v = pkg_resources.get_distribution("discord.py").version
    # assert v == "2.0.0a3575+g45d498c1"
# except:
    # print_exc()
    # subprocess.run([python, "-m", "pip", "install", "git+https://github.com/thomas-xin/discord.py.git", "--user"])

try:
    v = pkg_resources.get_distribution("googletrans").version
    assert v >= "4.0.0rc1"
except:
    print_exc()
    subprocess.run([python, "-m", "pip", "install", "googletrans==4.0.0rc1", "--upgrade", "--user"])

try:
    v = pkg_resources.get_distribution("httpx").version
    assert v >= "0.24.0"
except:
    print_exc()
    subprocess.run([python, "-m", "pip", "install", "httpx[http2]>=0.24.0", "--upgrade", "--user"])

if os.name == "nt" and os.environ.get("AI_FEATURES", True):
    try:
        pkg_resources.get_distribution("bitsandbytes")
    except:
        dist = pkg_resources.get_distribution("bitsandbytes-windows")
        fold = dist.module_path + "/bitsandbytes_windows-" + dist.version + ".dist-info"
        if os.path.exists(fold):
            os.rename(fold, fold.replace("_windows", ""))

if os.environ.get("AI_FEATURES", True):
	try:
		assert pkg_resources.get_distribution("encodec").version >= "0.1.2a3"
	except (pkg_resources.DistributionNotFound, AssertionError):
		subprocess.run([python, "-m", "pip", "install", "git+https://github.com/facebookresearch/encodec", "--user"])

print("Installer terminated.")
