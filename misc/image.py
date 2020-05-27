#!/usr/bin/python3

import os, sys, io, time, re, traceback, requests, urllib, numpy, blend_modes
from PIL import Image, ImageChops, ImageEnhance, ImageMath, ImageStat


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


from_colour = lambda colour, size=128: Image.fromarray(numpy.tile(numpy.array(colour, dtype=numpy.uint8), (size, size, 1)))


def resize_max(image, maxsize, resample=Image.LANCZOS, box=None, reducing_gap=None):
    w = image.width
    h = image.height
    s = max(w, h)
    if s > maxsize:
        r = maxsize / s
        w = int(w * r)
        h = int(h * r)
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
    if abs(w * h) > 16777216:
        raise OverflowError("Resulting image size too large.")
    return resize_to(image, round(w), round(h), operation)

def resize_to(image, w, h, operation):
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


def get_image(out, url):
    if isURL(url):
        data = requests.get(url, timeout=8).content
    else:
        if os.path.getsize(url) > 67108864:
            raise OverflowError("Max file size to load is 64MB.")
        f = open(url, "rb")
        data = f.read()
        f.close()
        if out != url:
            try:
                os.remove(url)
            except:
                pass
    if len(data) > 67108864:
        raise OverflowError("Max file size to load is 64MB.")
    return Image.open(io.BytesIO(data))


@logging
def evalImg(url, operation, args, key):
    out = "cache/" + key + ".png"
    args = eval(args)
    if operation != "$":
        image = get_image(out, url)
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
    return repr(str(new).encode("utf-8"))


while True:
    try:
        i = eval(sys.stdin.readline()).decode("utf-8", "replace").replace("\n", "").split("`")
        if len(i) <= 1:
            i.append("0")
        resp = evalImg(*i)
        sys.stdout.write(repr(resp.encode("utf-8")) + "\n")
        sys.stdout.flush()
    except Exception as ex:
        sys.stdout.write(repr(ex) + "\n")
        sys.stdout.flush()
    time.sleep(0.01)
