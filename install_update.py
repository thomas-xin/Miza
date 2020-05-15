print("Loading and checking modules...")

f = open("requirements.txt", "rb")
modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")
f.close()

import os, subprocess, traceback, pkg_resources

python = ("python3", "python")[os.name == "nt"]

installing = []
install = lambda m: installing.append(subprocess.Popen(["python", "-m", "pip", "install", "--upgrade", m, "--user"]))

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
            traceback.print_exc()
            inst = name
            if op in ("==", "<="):
                inst += "==" + version
            install(inst)

if installing:
    print("Installing missing or outdated modules, please wait...")
    subprocess.Popen([python, "-m", "pip", "install", "--upgrade", "pip", "--user"]).wait()
    for i in installing:
        i.wait()
    print("Installer terminated.")