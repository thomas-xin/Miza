import sys, os, subprocess, concurrent.futures


def remove_file(file):
    global count
    try:
        os.remove(file)
        count += 1
    except:
        raise

def remove_folder(folder):
    global count
    try:
        os.rmdir(folder)
        count += 1
    except:
        raise

def remove_folder_ex(folder, recursive=False):
    global count
    args = ["python", sys.argv[0], folder]
    if not recursive:
        args.insert(2, "-r")
    try:
        resp = subprocess.run(args, stdout=subprocess.PIPE)
        try:
            count += int(resp.stdout.split(None, 1)[0])
        except IndexError:
            pass
    except:
        raise

def delete_folder_folder(x, y):
    global count
    futs = set()
    for f in y:
        futs.add(exc.submit(remove_folder, x + "/" + f))
    for fut in futs:
        fut.result()


if __name__ == "__main__":

    recursive = True
    if len(sys.argv) > 1 and sys.argv[1] == "-r":
        recursive = False
        sys.argv.pop(0)
    exc = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    futs = set()
    count = 0

    if len(sys.argv) > 2:
        for folder in (f for f in sys.argv[1:] if os.path.exists(f)):
            futs.add(exc.submit(remove_folder_ex, folder, recursive=recursive))
        for fut in futs:
            fut.result()

    else:
        if len(sys.argv) == 2:
            folder = sys.argv[1]
        else:
            folder = input("Please input a file or folder to delete: ")

        for x, y, z in os.walk(folder, topdown=False):
            if len(z) >= 1024 and recursive:
                futs.add(exc.submit(remove_folder_ex, x))
            else:
                for f in z:
                    futs.add(exc.submit(remove_file, x + "/" + f))
        for fut in futs:
            fut.result()
        futs.clear()

        for x, y, z in os.walk(folder, topdown=False):
            futs.add(exc.submit(delete_folder_folder, x, y))

        for fut in futs:
            fut.result()

        if count or os.path.isdir(folder):
            remove_folder(folder)
        else:
            remove_file(folder)

    print(f"{count} file{'s' if count != 1 else ''} deleted.")