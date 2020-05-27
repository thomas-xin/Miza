import os, sys, requests, io, time, re, traceback, urllib, numpy, blend_modes
from PIL import Image, ImageStat, ImageEnhance


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
    "normal": blend_modes.normal,
    "blit": blend_modes.normal,
    "blend": blend_modes.normal,
    "replace": blend_modes.normal,
    "+": blend_modes.addition,
    "add": blend_modes.addition,
    "addition": blend_modes.addition,
    "-": blend_modes.subtract,
    "sub": blend_modes.subtract,
    "subtract": blend_modes.subtract,
    "subtraction": blend_modes.subtract,
    "*": blend_modes.multiply,
    "mul": blend_modes.multiply,
    "mult": blend_modes.multiply,
    "multiply": blend_modes.multiply,
    "multiplication": blend_modes.multiply,
    "/": blend_modes.divide,
    "div": blend_modes.divide,
    "divide": blend_modes.divide,
    "division": blend_modes.divide,
    "diff": blend_modes.difference,
    "difference": blend_modes.difference,
    "overlay": blend_modes.overlay,
    "soft": blend_modes.soft_light,
    "softlight": blend_modes.soft_light,
    "hard": blend_modes.hard_light,
    "hardlight": blend_modes.hard_light,
    "lighten": blend_modes.lighten_only,
    "lightenonly": blend_modes.lighten_only,
    "darken": blend_modes.darken_only,
    "darkenonly": blend_modes.darken_only,
    "extract": blend_modes.grain_extract,
    "grainextract": blend_modes.grain_extract,
    "merge": blend_modes.grain_merge,
    "grainmerge": blend_modes.grain_merge,
    "dodge": blend_modes.dodge,
}

def blend_op(image, url, operation, amount):
    op = operation.lower().replace(" ", "").replace("_", "").replace("color", "").replace("colour", "")
    if op in blenders:
        filt = blenders[op]
    elif op == "auto":
        filt = blend_modes.normal
    else:
        raise TypeError("Invalid image operation: \"" + op + '"')
    image2 = get_image(url, url)
    if image2.width != image.width or image2.height != image.height:
        image2 = resize_to(image2, image.width, image.height, "auto")
    if str(image.mode) != "RGBA":
        image = image.convert("RGBA")
    if str(image2.mode) != "RGBA":
        image2 = image2.convert("RGBA")
    imgA = numpy.array(image).astype(float)
    imgB = numpy.array(image2).astype(float)
    imgC = numpy.uint8(filt(imgA, imgB, amount))
    return Image.fromarray(imgC)


Enhance = lambda image, operation, value: getattr(ImageEnhance, operation)(image).enhance(value)


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
    image = get_image(out, url)
    f = getattr(image, operation, None)
    if f is None:
        new = eval(operation)(image, *args)
    else:
        new = f(*args)
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
