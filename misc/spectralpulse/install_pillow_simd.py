import sys, subprocess, traceback

print("Loading and checking modules...")

modlist = """
numpy>=1.20.1
psutil>=5.8.0
pygame>=2.0.1
youtube-dlc>=2020.11.11.post3
requests>=2.25.1
""".split("\n")

import pkg_resources, struct
x = sys.version_info[1]

installing = []
install = lambda m: installing.append(subprocess.Popen(["py", f"-3.{x}", "-m", "pip", "install", "--upgrade", m, "--user"]))

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
    subprocess.run(["py", f"-3.{x}", "-m", "pip", "install", "--upgrade", "pip", "--user"])
    for i in installing:
        i.wait()
try:
    pkg_resources.get_distribution("pillow")
except pkg_resources.DistributionNotFound:
    pass
else:
    subprocess.run(["py", f"-3.{x]}", "-m", "pip", "uninstall", "pillow", "-y"])
try:
    pkg_resources.get_distribution("pillow-simd")
except pkg_resources.DistributionNotFound:
    psize = struct.calcsize("P")
    if psize == 8:
        win = "win_amd64"
    else:
        win = "win32"
    subprocess.run(["py", f"-3.{x}", "-m", "pip", "install", f"https://download.lfd.uci.edu/pythonlibs/w4tscw6k/Pillow_SIMD-7.0.0.post3+avx2-cp3{x}-cp3{x}-{win}.whl", "--user"])
print("Installer terminated.")