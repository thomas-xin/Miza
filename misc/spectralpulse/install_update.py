print("Loading and checking modules...")

import os, sys, subprocess, traceback

python = sys.executable
arg = sys.argv[0].replace("\\", "/")
if "/" in arg:
	PATH = os.path.join(os.getcwd(), arg.rsplit("/", 1)[0])
else:
	PATH = "."

with open(f"{PATH}/requirements.txt", "rb") as f:
    modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")


try:
    import pkg_resources
except ModuleNotFoundError:
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "--user", "setuptools"])
    import pkg_resources

installing = []
install = lambda m: installing.append(subprocess.Popen([python, "-m", "pip", "install", "--upgrade", m, "--user"]))

# Parse requirements.txt
for mod in modlist:
    if mod:
        try:
            name = mod
            version = None
            for op in (">=", "==", "<="):
                if op in mod:
                    name, version = mod.split(op)
                    break
            v = pkg_resources.get_distribution(name).version
            if version is not None:
                assert eval(repr(v) + op + repr(version), {}, {})
        except:
            # Modules may require an older version, replace current version if necessary
            traceback.print_exc()
            inst = name
            if op in ("==", "<="):
                inst += "==" + version
            install(inst)

try:
    v = pkg_resources.get_distribution("yt_dlp").version
    assert v >= "2022.5.18"
except:
    print_exc()
    subprocess.run([python, "-m", "pip", "install", "git+https://github.com/yt-dlp/yt-dlp.git", "--upgrade", "--user"])

# Run pip on any modules that need installing
if installing:
    print("Installing missing or outdated modules, please wait...")
    subprocess.Popen([python, "-m", "pip", "install", "--upgrade", "pip", "--user"]).wait()
    for i in installing:
        i.wait()
    print("Installer terminated.")