#!/usr/bin/python3

import os, sys, io, time, re, traceback, requests, urllib, numpy, blend_modes, subprocess, psutil
import PIL, concurrent.futures
from PIL import Image, ImageChops, ImageEnhance, ImageMath, ImageStat


exc = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def logging(func):
    def call(self, *args, file="log.txt", **kwargs):
        try:
            output = func(self, *args, **kwargs)
        except:
            f = open(file, "ab")
            f.write(traceback.format_exc().encode("utf-8"))
            f.close()
            raise
        return output
    return call


def rdhms(ts):
    data = ts.split(":")
    t = 0
    mult = 1
    while len(data):
        t += float(data[-1]) * mult
        data = data[:-1]
        if mult <= 60:
            mult *= 60
        elif mult <= 3600:
            mult *= 24
        elif len(data):
            raise TypeError("Too many time arguments.")
    return t


DOMAIN_FORMAT = re.compile(
    r"(?:^(\w{1,255}):(.{1,255})@|^)"
    r"(?:(?:(?=\S{0,253}(?:$|:))"
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:[a-z0-9]{1,63})))"
    r"|localhost)"
    r"(:\d{1,5})?",
    re.IGNORECASE
)
SCHEME_FORMAT = re.compile(
    r"^(http|hxxp|ftp|fxp)s?$",
    re.IGNORECASE
)

def isURL(url):
    url = url.strip()
    if url.startswith("<") and url[-1] == ">":
        url = url[1:-1]
    if not url:
        return None
    try:
        result = urllib.parse.urlparse(url)
    except:
        return False
    scheme = result.scheme
    domain = result.netloc
    if not scheme:
        return False
    if not re.fullmatch(SCHEME_FORMAT, scheme):
        return False
    if not domain:
        return False
    if not re.fullmatch(DOMAIN_FORMAT, domain):
        return False
    return True


from_colour = lambda colour, size=128, key=None: Image.fromarray(numpy.tile(numpy.array(colour, dtype=numpy.uint8), (size, size, 1)))


sizecheck = re.compile("[1-9][0-9]*x[0-9]+")

def video2img(url, maxsize, fps, out, size=None, dur=None, orig_fps=None):
    direct = any((size is None, dur is None, orig_fps is None))
    ts = round(time.time() * 1000)
    fn = "cache/" + str(ts)
    if direct:
        data = requests.get(url, timeout=8).content
        file = open(fn, "wb")
        try:
            file.write(data)
        except:
            file.close()
            raise
        file.close()
    try:
        if direct:
            command = ["ffprobe", "-hide_banner", fn]
            resp = bytes()
            for _ in range(3):
                try:
                    proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    fut = exc.submit(proc.communicate)
                    res = fut.result(timeout=12)
                    resp = bytes().join(res)
                    break
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            s = resp.decode("utf-8", "replace")
            if dur is None:
                i = s.index("Duration: ")
                d = s[i + 10:]
                i = 2147483647
                for c in ", \n\r":
                    try:
                        x = d.index(c)
                    except ValueError:
                        pass
                    else:
                        if x < i:
                            i = x
                dur = rdhms(d[:i])
            else:
                d = s
            if orig_fps is None:
                i = d.index(" fps")
                f = d[i - 5:i]
                while f[0] < "0" or f[0] > "9":
                    f = f[1:]
                orig_fps = float(f)
            if size is None:
                sfind = re.finditer(sizecheck, d)
                sizestr = next(sfind).group()
                size = [int(i) for i in sizestr.split("x")]
        fps = min(fps, 256 / dur)
        fn2 = fn + ".gif"
        f_in = fn if direct else url
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", f_in, "-fs", str(8388608 - 131072), "-an", "-vf"]
        vf = ""
        s = max(size)
        if s > maxsize:
            r = maxsize / s
            scale = int(size[0] * r)
            vf += "scale=" + str(scale) + ":-1:flags=lanczos,"
        vf += "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        command += [vf, "-loop", "0", "-r", str(fps), out]
        subprocess.check_output(command)
        if direct:
            os.remove(fn)
    except:
        if direct:
            try:
                os.remove(fn)
            except:
                pass
        raise

def create_gif(in_type, args, delay):
    ts = round(time.time() * 1000)
    out = "cache/" + str(ts) + ".gif"
    maxsize = 512
    if in_type == "video":
        video2img(args[0], maxsize, round(1000 / delay), out, args[1], args[2], args[3])
        return "$" + out
    images = args
    maxsize = int(min(maxsize, 32768 / len(images) ** 0.5))
    imgs = []
    for url in images:
        data = requests.get(url, timeout=8).content
        try:
            img = get_image(data, None)
        except (PIL.UnidentifiedImageError, OverflowError):
            if len(data) < 268435456:
                video2img(data, maxsize, round(1000 / delay), out)
                return "$" + out
            else:
                raise OverflowError("Max file size to load is 256MB.")
        else:
            if not imgs:
                size = max_size(img.width, img.height, maxsize)
            img = resize_to(img, *size, operation="hamming")
            if str(img.mode) != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-r", str(1000 / delay), "-pix_fmt", "rgb24", "-video_size", "x".join(str(i) for i in size), "-i", "-"]
    command += ["-fs", str(8388608 - 131072), "-an", "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", "-loop", "0", out]
    proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for img in imgs:
        b = numpy.array(img).tobytes()
        proc.stdin.write(b)
        time.sleep(0.02)
    proc.stdin.close()
    proc.wait()
    return "$" + out

def rainbow_gif(image, duration):
    ts = round(time.time() * 1000)
    out = "cache/" + str(ts) + ".gif"
    image = resize_max(image, 512, resample=Image.HAMMING)
    size = [image.width, image.height]
    if duration == 0:
        fps = 0
    else:
        fps = round(64 / abs(duration))
    rate = 4
    while fps > 16:
        fps >>= 1
        rate <<= 1
    if fps <= 0:
        raise ValueError("Invalid framerate value.")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-r", str(fps), "-pix_fmt", "rgb24", "-video_size", "x".join(str(i) for i in size), "-i", "-"]
    command += ["-fs", str(8388608 - 131072), "-an", "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", "-loop", "0", out]
    proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if str(image.mode) != "HSV":
        curr = image.convert("HSV")
        if str(image.mode) != "RGB":
            image = image.convert("RGB")
    else:
        curr, image = image, image.convert("RGB")
    channels = list(curr.split())
    if duration < 0:
        rate = -rate
    func = lambda x: (x + rate) & 255
    for i in range(0, 256, abs(rate)):
        if i:
            channels[0] = channels[0].point(func)
            image = Image.merge("HSV", channels).convert("RGB")
        b = numpy.array(image).tobytes()
        proc.stdin.write(b)
        time.sleep(0.02)
    proc.stdin.close()
    proc.wait()
    return "$" + out


def max_size(w, h, maxsize):
    s = max(w, h)
    if s > maxsize:
        r = maxsize / s
        w = int(w * r)
        h = int(h * r)
    return w, h

def resize_max(image, maxsize, resample=Image.LANCZOS, box=None, reducing_gap=None):
    w, h = max_size(image.width, image.height, maxsize)
    if w != image.width or h != image.height:
        image = image.resize([w, h], resample, box, reducing_gap)
    return image

resizers = {
    "sinc": Image.LANCZOS,
    "lanczos": Image.LANCZOS,
    "cubic": Image.BICUBIC,
    "bicubic": Image.BICUBIC,
    "hamming": Image.HAMMING,
    "linear": Image.BILINEAR,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "nearestneighbour": Image.NEAREST,
}

def resize_mult(image, x, y, operation):
    if x == y == 1:
        return image
    w = image.width * x
    h = image.height * y
    return resize_to(image, round(w), round(h), operation)

def resize_to(image, w, h, operation="auto"):
    if abs(w * h) > 16777216:
        raise OverflowError("Resulting image size too large.")
    if w == image.width and h == image.height:
        return image
    op = operation.lower().replace(" ", "").replace("_", "")
    if op in resizers:
        filt = resizers[op]
    elif op == "auto":
        m = min(abs(w), abs(h))
        n = min(image.width, image.height)
        if n > m:
            m = n
        if m <= 64:
            filt = Image.NEAREST
        elif m <= 256:
            filt = Image.HAMMING
        elif m <= 2048:
            filt = Image.LANCZOS
        elif m <= 3072:
            filt = Image.BICUBIC
        else:
            filt = Image.BILINEAR
    else:
        raise TypeError("Invalid image operation: \"" + op + '"')
    return image.resize([w, h], filt)


blenders = {
    "normal": "blend",
    "blt": "blend",
    "blit": "blend",
    "blend": "blend",
    "replace": "blend",
    "+": "add",
    "add": "add",
    "addition": "add",
    "-": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "subtraction": "subtract",
    "*": "multiply",
    "mul": "multiply",
    "mult": "multiply",
    "multiply": "multiply",
    "multiplication": "multiply",
    "/": blend_modes.divide,
    "div": blend_modes.divide,
    "divide": blend_modes.divide,
    "division": blend_modes.divide,
    "mod": "OP_X%Y",
    "modulo": "OP_X%Y",
    "%": "OP_X%Y",
    "and": "OP_X&Y",
    "&": "OP_X&Y",
    "or": "OP_X|Y",
    "|": "OP_X|Y",
    "xor": "OP_X^Y",
    "^": "OP_X^Y",
    "nand": "OP_255-(X&Y)",
    "~&": "OP_255-(X&Y)",
    "nor": "OP_255-(X|Y)",
    "~|": "OP_255-(X|Y)",
    "xnor": "OP_255-(X^Y)",
    "~^": "OP_255-(X^Y)",
    "xand": "OP_255-(X^Y)",
    "diff": "difference",
    "difference": "difference",
    "overlay": "overlay",
    "screen": "screen",
    "soft": "soft_light",
    "softlight": "soft_light",
    "hard": "hard_light",
    "hardlight": "hard_light",
    "lighter": "lighter",
    "lighten": "lighter",
    "darker": "darker",
    "darken": "darker",
    "extract": blend_modes.grain_extract,
    "grainextract": blend_modes.grain_extract,
    "merge": blend_modes.grain_merge,
    "grainmerge": blend_modes.grain_merge,
    "burn": "OP_255*(1-((255-Y)/X))",
    "colorburn": "OP_255*(1-((255-Y)/X))",
    "colourburn": "OP_255*(1-((255-Y)/X))",
    "linearburn": "OP_(X+Y)-255",
    "dodge": blend_modes.dodge,
    "colordodge": blend_modes.dodge,
    "colourdodge": blend_modes.dodge,
    "lineardodge": "add",
    "hue": "SP_HUE",
    "sat": "SP_SAT",
    "saturation": "SP_SAT",
    "lum": "SP_LUM",
    "luminosity": "SP_LUM",
}

def blend_op(image, url, operation, amount):
    op = operation.lower().replace(" ", "").replace("_", "")
    if op in blenders:
        filt = blenders[op]
    elif op == "auto":
        filt = "blend"
    else:
        raise TypeError("Invalid image operation: \"" + op + '"')
    image2 = get_image(url, url)
    if image2.width != image.width or image2.height != image.height:
        image2 = resize_to(image2, image.width, image.height, "auto")
    if type(filt) is not str:
        if str(image.mode) != "RGBA":
            image = image.convert("RGBA")
        if str(image2.mode) != "RGBA":
            image2 = image2.convert("RGBA")
        imgA = numpy.array(image).astype(float)
        imgB = numpy.array(image2).astype(float)
        out = Image.fromarray(numpy.uint8(filt(imgA, imgB, amount)))
    else:
        if filt == "blend":
            out = image2
        elif filt.startswith("OP_"):
            f = filt[3:]
            if str(image.mode) != str(image2.mode):
                if str(image.mode) != "RGBA":
                    image = image.convert("RGBA")
                if str(image2.mode) != "RGBA":
                    image2 = image2.convert("RGBA")
            mode = image.mode
            ch1 = image.split()
            ch2 = image2.split()
            c = len(ch1)
            ch3 = [ImageMath.eval(f, dict(X=ch1[i], Y=ch2[i])).convert("L") for i in range(3)]
            if c > 3:
                ch3.append(ImageMath.eval("max(X,Y)", dict(X=ch1[-1], Y=ch2[-1])).convert("L"))
            out = Image.merge(mode, ch3)
        elif filt.startswith("SP_"):
            f = filt[3:]
            if str(image.mode) == "RGBA":
                A1 = image.split()[-1]
            else:
                A1 = None
            if str(image2.mode) == "RGBA":
                A2 = image2.split()[-1]
            else:
                A2 = None
            if str(image.mode) != "HSV":
                image = image.convert("HSV")
            channels = list(image.split())
            if str(image2.mode) != "HSV":
                image2 = image2.convert("HSV")
            channels2 = list(image2.split())
            if f == "HUE":
                channels = [channels2[0], channels[1], channels[2]]
            elif f == "SAT":
                channels = [channels[0], channels2[1], channels[2]]
            elif f == "LUM":
                channels = [channels[0], channels[1], channels2[2]]
            out = Image.merge("RGB", channels)
            if A1 or A2:
                out = out.convert("RGBA")
                spl = list(out.split())
                if not A1:
                    A = A2
                elif not A2:
                    A = A1
                else:
                    A = ImageMath.eval("max(X,Y)", dict(X=A1, Y=A2)).convert("L")
                spl[-1] = A
        else:
            if str(image.mode) != str(image2.mode):
                if str(image.mode) != "RGBA":
                    image = image.convert("RGBA")
                if str(image2.mode) != "RGBA":
                    image2 = image2.convert("RGBA")
            filt = getattr(ImageChops, filt)
            out = filt(image, image2)
        if str(image.mode) != str(out.mode):
            if str(image.mode) != "RGBA":
                image = image.convert("RGBA")
            if str(out.mode) != "RGBA":
                out = out.convert("RGBA")
        out = ImageChops.blend(image, out, amount)
    return out

# def ColourDeficiency(image, operation, value):
#     pass

Enhance = lambda image, operation, value: getattr(ImageEnhance, operation)(image).enhance(value)

def hue_shift(image, value):
    if str(image.mode) == "RGBA":
        A = image.split()[-1]
    else:
        A = None
    if str(image.mode) != "HSV":
        image = image.convert("HSV")
    channels = list(image.split())
    value *= 256
    channels[0] = channels[0].point(lambda x: (x + value) % 256)
    image = Image.merge("HSV", channels)
    if A is not None:
        channels = list(image.convert("RGBA").split())
        channels[-1] = A
        image = Image.merge("RGBA", channels)
    else:
        image = image.convert("RGB")
    return image


def get_image(url, out):
    if type(url) not in (bytes, bytearray, io.BytesIO):
        if isURL(url):
            data = requests.get(url, timeout=8).content
        else:
            if os.path.getsize(url) > 67108864:
                raise OverflowError("Max file size to load is 64MB.")
            f = open(url, "rb")
            data = f.read()
            f.close()
            if out != url and out:
                try:
                    os.remove(url)
                except:
                    pass
    else:
        data = url
    if len(data) > 67108864:
        raise OverflowError("Max file size to load is 64MB.")
    return Image.open(io.BytesIO(data))


@logging
def evalImg(url, operation, args):
    ts = round(time.time() * 1000)
    out = "cache/" + str(ts) + ".png"
    args = eval(args)
    if operation != "$":
        image = get_image(url, out)
        f = getattr(image, operation, None)
        if f is None:
            new = eval(operation)(image, *args)
        else:
            new = f(*args)
    else:
        new = eval(url)(*args)
    if issubclass(type(new), Image.Image):
        new.save(out, "png")
        return repr([out])
    elif type(new) is str and new.startswith("$"):
        return repr([new[1:]])
    return repr(str(new).encode("utf-8"))


while True:
    try:
        args = eval(sys.stdin.readline()).decode("utf-8", "replace").replace("\n", "").split("`")
        resp = evalImg(*args)
        sys.stdout.write(repr(resp.encode("utf-8")) + "\n")
        sys.stdout.flush()
    except Exception as ex:
        sys.stdout.write(repr(ex) + "\n")
        sys.stdout.flush()
    time.sleep(0.01)
