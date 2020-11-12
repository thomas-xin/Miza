import sys, os, subprocess, traceback

# Required to open python on different operating systems
python = ("python3", "python")[os.name == "nt"]


if sys.version_info[0] < 3:
    raise ImportError("Python 3 required.")

print("Loading and checking modules...")

with open("requirements.txt", "rb") as f:
    modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")

import pkg_resources

installing = []
install = lambda m: installing.append(subprocess.Popen(["python", "-m", "pip", "install", "--upgrade", m, "--user"]))

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

# Run pip on any modules that need installing
if installing:
    print("Installing missing or outdated modules, please wait...")
    subprocess.Popen([python, "-m", "pip", "install", "--upgrade", "pip", "--user"]).wait()
    for i in installing:
        i.wait()
    print("Installer terminated.")