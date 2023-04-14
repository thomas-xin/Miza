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

# Parse requirements.txt
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
                s = repr([try_int(i) for i in v.split(".")]) + op + repr([try_int(i) for i in version.split(".")])
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

try:
    v = pkg_resources.get_distribution("discord.py").version
    assert v == "2.0.0a3575+g45d498c1"
except:
    print_exc()
    subprocess.run([python, "-m", "pip", "install", "git+https://github.com/thomas-xin/discord.py.git", "--user"])

# try:
#     v = pkg_resources.get_distribution("googletrans").version
#     assert v >= "4.0.0rc1"
# except:
#     print_exc()
#     subprocess.run([python, "-m", "pip", "install", "googletrans==4.0.0rc1", "--upgrade", "--user"])

print("Installer terminated.")
