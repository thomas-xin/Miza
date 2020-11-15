#!/usr/bin/python3

import os, sys, io, time, concurrent.futures, subprocess, psutil, collections, traceback, re, requests, blend_modes, pdf2image, zipfile, contextlib
import numpy as np
import PIL
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter, ImageEnhance, ImageMath, ImageStat
from zipfile import ZipFile
import matplotlib.pyplot as plt

deque = collections.deque
suppress = contextlib.suppress

exc = concurrent.futures.ThreadPoolExecutor(max_workers=3)
start = time.time()
CACHE = {}
ANIM = False


# For debugging only
def file_print(*args, sep=" ", end="\n", prefix="", file="log.txt", **void):
    with open(file, "ab") as f:
        f.write((str(sep).join((i if type(i) is str else str(i)) for i in args) + str(end) + str(prefix)).encode("utf-8"))

def logging(func):
    def call(self, *args, file="log.txt", **kwargs):
        try:
            output = func(self, *args, **kwargs)
        except:
            file_print(traceback.format_exc(), file=file)
            raise
        return output
    return call


# Converts a time interval represented using days:hours:minutes:seconds, to a value in seconds.
def time_parse(ts):
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

# URL string detector
url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
is_url = lambda url: url_match.search(url)
discord_match = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/")
is_discord_url = lambda url: discord_match.findall(url)

fcache = "cache" if os.path.exists("cache") else "../cache"

def header():
    return {
        "DNT": "1",
        "user-agent": f"Mozilla/5.{(time.time_ns() // 1000) % 10}",
    }

def get_request(url):
    if is_discord_url(url) and "attachments/" in url[:64]:
        try:
            a_id = int(url.split("?", 1)[0].rsplit("/", 2)[-2])
        except ValueError:
            pass
        else:
            fn = f"{fcache}/attachment_{a_id}.bin"
            if os.path.exists(fn):
                with open(fn, "rb") as f:
                    file_print(f"Attachment {a_id} loaded from cache.")
                    return f.read()
    with requests.get(url, headers=header(), stream=True, timeout=12) as resp:
        return resp.content
    # resp = requests.get(url, headers=header(), stream=True, timeout=12)
    # return seq(resp)


from_colour = lambda colour, size=128, key=None: Image.new("RGB", (size, size), tuple(colour))


sizecheck = re.compile("[1-9][0-9]*x[0-9]+")
fpscheck = re.compile("[0-9]+ fps")

def video2img(url, maxsize, fps, out, size=None, dur=None, orig_fps=None, data=None):
    direct = any((size is None, dur is None, orig_fps is None))
    ts = time.time_ns() // 1000
    fn = "cache/" + str(ts)
    if direct:
        if data is None:
            data = get_request(url)
        with open(fn, "wb") as file:
            file.write(data if type(data) is bytes else data.read())
    try:
        if direct:
            command = ["ffprobe", "-hide_banner", fn]
            resp = bytes()
            # Up to 3 attempts to get video duration
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
                dur = time_parse(d[:i])
            else:
                d = s
            if orig_fps is None:
                f = re.findall(fpscheck, d)[0][:-4]
                orig_fps = float(f)
            if size is None:
                sfind = re.finditer(sizecheck, d)
                sizestr = next(sfind).group()
                size = [int(i) for i in sizestr.split("x")]
        fn2 = fn + ".gif"
        f_in = fn if direct else url
        command = ["ffmpeg", "-threads", "2", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", f_in, "-an", "-vf"]
        w, h = max_size(*size, maxsize)
        # Adjust FPS if duration is too long
        fps = min(fps, orig_fps)
        r2 = 2 ** 0.5
        rr2 = r2 ** 0.5
        while fps < 12:
            fps *= r2
            w /= rr2
            h /= rr2
        w = round(w)
        h = round(h)
        vf = ""
        if w != size[0]:
            vf += "scale=" + str(w) + ":-1:flags=lanczos,"
        vf += "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        command.extend([vf, "-loop", "0", "-r", str(fps), out])
        file_print(command)
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
    ts = time.time_ns() // 1000
    out = "cache/" + str(ts) + ".gif"
    maxsize = 960
    if in_type == "video":
        video2img(args[0], maxsize, round(1000 / delay), out, args[1], args[2], args[3])
        return "$" + out
    images = args
    # Detect if an image sequence or video is being inputted
    imgs = deque()
    for url in images:
        data = get_request(url)
        try:
            img = get_image(data, None)
        except (PIL.UnidentifiedImageError, OverflowError):
            if len(data) < 268435456:
                video2img(data, maxsize, round(1000 / delay), out, data=data)
                # $ symbol indicates to return directly
                return "$" + out
            else:
                raise OverflowError("Max file size to load is 256MB.")
        else:
            length = 0
            for f in range(2147483648):
                try:
                    img.seek(f)
                    length = f
                except EOFError:
                    break
            if length != 0:
                maxsize = int(min(maxsize, 32768 / (len(images) + length) ** 0.5))
                dur = img.info.get("duration")
                if dur:
                    delay = dur
            for f in range(2147483648):
                try:
                    img.seek(f)
                except EOFError:
                    break
                if not imgs:
                    size = max_size(img.width, img.height, maxsize)
                temp = resize_to(img, *size, operation="hamming")
                if str(temp.mode) != "RGBA":
                    temp = temp.convert("RGBA")
                imgs.append(temp)
    size = list(imgs[0].size)
    while size[0] * size[1] * len(imgs) > 8388608:
        size[0] /= 2 ** 0.5
        size[1] /= 2 ** 0.5
    size = [round(size[0]), round(size[1])]
    count = len(imgs)
    if imgs[0].size[0] != size[0]:
        imgs = (resize_to(img, *size, operation="hamming") for img in imgs)
    return dict(duration=delay * len(imgs), count=count, frames=imgs)

def rainbow_gif2(image, duration):
    total = 0
    for f in range(2147483648):
        try:
            image.seek(f)
        except EOFError:
            break
        total += max(image.info.get("duration", 0), 1 / 60)
    length = f
    loops = total / duration / 1000
    scale = 1
    while abs(loops * scale) < 1:
        scale *= 2
        if length * scale >= 64:
            loops = 1 if loops >= 0 else -1
            break
    loops = round(loops * scale) / scale
    if not loops:
        loops = 1 if loops >= 0 else -1
    maxsize = 960
    size = list(max_size(*image.size, maxsize))

    def rainbow_gif_iterator(image):
        for f in range(length * scale):
            image.seek(f % length)
            if str(image.mode) != "RGBA":
                temp = image.convert("RGBA")
            else:
                temp = image
            if temp.size[0] != size[0] or temp.size[1] != size[1]:
                temp = temp.resize(size, Image.HAMMING)
            A = temp.getchannel("A")
            channels = list(temp.convert("HSV").split())
            channels[0] = channels[0].point(lambda x: int(((f / length / scale * loops + x / 256) % 1) * 256))
            temp = Image.merge("HSV", channels).convert("RGB")
            temp.putalpha(A)
            yield temp

    return dict(duration=total * scale, count=length * scale, frames=rainbow_gif_iterator(image))

def rainbow_gif(image, duration):
    try:
        image.seek(1)
    except EOFError:
        image.seek(0)
    else:
        return rainbow_gif2(image, duration)
    ts = time.time_ns() // 1000
    image = resize_max(image, 960, resample=Image.HAMMING)
    size = list(image.size)
    if duration == 0:
        fps = 0
    else:
        fps = round(128 / abs(duration))
    rate = 2
    while fps > 24 and rate < 32:
        fps >>= 1
        rate <<= 1
    if fps <= 0:
        raise ValueError("Invalid framerate value.")
    # Make sure image is in RGB/HSV format
    if str(image.mode) != "HSV":
        curr = image.convert("HSV")
        if str(image.mode) == "RGBA":
            A = image.getchannel("A")
        else:
            A = None
    else:
        curr = image
        A = None
    channels = list(curr.split())
    if duration < 0:
        rate = -rate
    count = 256 // abs(rate)
    func = lambda x: (x + rate) & 255

    # Repeatedly hueshift image and return copies
    def rainbow_gif_iterator(image):
        for i in range(0, 256, abs(rate)):
            if i:
                channels[0] = channels[0].point(func)
                image = Image.merge("HSV", channels).convert("RGBA")
                if A is not None:
                    image.putalpha(A)
            yield image

    return dict(duration=1000 / fps * count, count=count, frames=rainbow_gif_iterator(image))


def spin_gif2(image, duration):
    total = 0
    for f in range(2147483648):
        try:
            image.seek(f)
        except EOFError:
            break
        total += max(image.info.get("duration", 0), 1 / 60)
    length = f
    loops = total / duration / 1000
    scale = 1
    while abs(loops * scale) < 1:
        scale *= 2
        if length * scale >= 64:
            loops = 1 if loops >= 0 else -1
            break
    loops = round(loops * scale) / scale
    if not loops:
        loops = 1 if loops >= 0 else -1
    maxsize = 960
    size = list(max_size(*image.size, maxsize))

    def spin_gif_iterator(image):
        for f in range(length * scale):
            image.seek(f % length)
            temp = image
            if temp.size[0] != size[0] or temp.size[1] != size[1]:
                temp = temp.resize(size, Image.HAMMING)
            temp = to_circle(temp.rotate(f * 360 / length / scale * loops))
            yield temp

    return dict(duration=total * scale, count=length * scale, frames=spin_gif_iterator(image))


def spin_gif(image, duration):
    try:
        image.seek(1)
    except EOFError:
        image.seek(0)
    else:
        return spin_gif2(image, duration)
    ts = time.time_ns() // 1000
    image = 960
    size = list(image.size)
    if duration == 0:
        fps = 0
    else:
        fps = round(64 / abs(duration))
    rate = 8
    while fps > 24 and rate < 32:
        fps >>= 1
        rate <<= 1
    if fps <= 0:
        raise ValueError("Invalid framerate value.")
    if duration < 0:
        rate = -rate
    count = 256 // abs(rate)

    # Repeatedly rotate image and return copies
    def spin_gif_iterator(image):
        for i in range(0, 256, abs(rate)):
            if i:
                im = image.rotate(i * 360 / 256)
            else:
                im = image
            yield to_circle(im)

    return dict(duration=1000 / fps * count, count=count, frames=spin_gif_iterator(image))


def to_square(image):
    w, h = image.size
    d = w - h
    if not d:
        return image
    if d > 0:
        return image.crop((d >> 1, 0, w - (1 + d >> 1), h))
    return image.crop((0, -d >> 1, w, h - (1 - d >> 1)))


CIRCLE_CACHE = {}

def to_circle(image):
    global CIRCLE_CACHE
    if str(image.mode) != "RGBA":
        image = to_square(image).convert("RGBA")
    try:
        image_map = CIRCLE_CACHE[image.size]
    except KeyError:
        image_map = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(image_map)
        draw.ellipse((0, 0, *image.size), outline=0, fill=(255,) * 4, width=0)
        CIRCLE_CACHE[image.size] = image_map
    return ImageChops.multiply(image, image_map)


def magik_gif2(image, cell_size, grid_distance, iterations):
    total = 0
    for f in range(2147483648):
        try:
            image.seek(f)
        except EOFError:
            break
        total += max(image.info.get("duration", 0), 1 / 60)
    length = f
    loops = total / 2 / 1000
    scale = 1
    while abs(loops * scale) < 1:
        scale *= 2
        if length * scale >= 32:
            loops = 1 if loops >= 0 else -1
            break
    loops = round(loops * scale) / scale
    if not loops:
        loops = 1 if loops >= 0 else -1
    maxsize = 960
    size = list(max_size(*image.size, maxsize))
    ts = time.time_ns() // 1000

    def magik_gif_iterator(image):
        for f in range(length * scale):
            np.random.seed(ts & 4294967295)
            image.seek(f % length)
            temp = image
            if temp.size[0] != size[0] or temp.size[1] != size[1]:
                temp = temp.resize(size, Image.HAMMING)
            for _ in range(int(31 * iterations * f / length / scale)):
                dst_grid = griddify(shape_to_rect(image.size), cell_size, cell_size)
                src_grid = distort_grid(dst_grid, grid_distance)
                mesh = grid_to_mesh(src_grid, dst_grid)
                temp = temp.transform(temp.size, Image.MESH, mesh, resample=Image.NEAREST)
            yield temp

    return dict(duration=total * scale, count=length * scale, frames=magik_gif_iterator(image))


def magik_gif(image, cell_size=7, grid_distance=23, iterations=1):
    try:
        image.seek(1)
    except EOFError:
        image.seek(0)
    else:
        return magik_gif2(image, cell_size, grid_distance, iterations)
    ts = time.time_ns() // 1000
    image = resize_max(image, 960, resample=Image.HAMMING)

    def magik_gif_iterator(image):
        yield image
        for _ in range(31):
            for _ in range(iterations):
                dst_grid = griddify(shape_to_rect(image.size), cell_size, cell_size)
                src_grid = distort_grid(dst_grid, grid_distance)
                mesh = grid_to_mesh(src_grid, dst_grid)
                image = image.transform(image.size, Image.MESH, mesh, resample=Image.NEAREST)
            yield image

    return dict(duration=2, count=32, frames=magik_gif_iterator(image))


def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True

def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
    assert(len(rect) == 4)
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
    assert(len(shape) == 2)
    return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = deque()
    for _ in range(h_div + 1):
        grid_vertex_matrix.append(deque())
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(-max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = deque()
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return list(mesh)

def magik(image, cell_size=7):
    dst_grid = griddify(shape_to_rect(image.size), cell_size, cell_size)
    src_grid = distort_grid(dst_grid, max(1, round(160 / cell_size)))
    mesh = grid_to_mesh(src_grid, dst_grid)
    return image.transform(image.size, Image.MESH, mesh, resample=Image.NEAREST)


blurs = {
    "box": ImageFilter.BoxBlur,
    "boxblur": ImageFilter.BoxBlur,
    "gaussian": ImageFilter.GaussianBlur,
    "gaussianblur": ImageFilter.GaussianBlur,
}

def blur(image, filt="box", radius=2):
    try:
        _filt = blurs[filt.replace("_", "").casefold()]
    except KeyError:
        raise TypeError(f'Invalid image operation: "{filt}"')
    return image.filter(_filt(radius))


def invert(image):
    if str(image.mode) == "RGBA":
        A = image.getchannel("A")
        image = image.convert("RGB")
    else:
        A = None
    image = ImageOps.invert(image)
    if A is not None:
        image.putalpha(A)
    return image


# Autodetect max image size, keeping aspect ratio
def max_size(w, h, maxsize):
    s = w * h
    m = (maxsize * maxsize << 1) / 3
    if s > m:
        r = (m / s) ** 0.5
        w = int(w * r)
        h = int(h * r)
    return w, h

def resize_max(image, maxsize, resample=Image.LANCZOS, box=None, reducing_gap=None):
    w, h = max_size(image.width, image.height, maxsize)
    if w != image.width or h != image.height:
        if type(resample) is str:
            image = resize_to(image, w, h, resample)
        else:
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
    op = operation.casefold().replace(" ", "").replace("_", "")
    if op in resizers:
        filt = resizers[op]
    elif op == "auto":
        # Choose resampling algorithm based on source/destination image sizes
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
        raise TypeError(f'Invalid image operation: "{op}"')
    if w < 0:
        w = -w
        image = ImageOps.mirror(image)
    if h < 0:
        h = -h
        image = ImageOps.flip(image)
    return image.resize([w, h], filt)


channel_map = {
    "alpha": -1,
    "a": -1,
    "red": 0,
    "r": 0,
    "green": 1,
    "g": 1,
    "blue": 2,
    "b": 2,
    "cyan": 3,
    "c": 3,
    "magenta": 4,
    "m": 4,
    "yellow": 5,
    "y": 5,
    "hue": 6,
    "h": 6,
    "saturation": 7,
    "sat": 7,
    "s": 7,
    "luminance": 8,
    "lum": 8,
    "l": 8,
    "v": 8
}

def fill_channels(image, colour, *channels):
    channels = list(channels)
    ops = {}
    for c in channels:
        try:
            cid = channel_map[c]
        except KeyError:
            if len(c) <= 1:
                raise TypeError("invalid colour identifier: " + c)
            channels.extend(c)
        else:
            ops[cid] = None
    ch = Image.new("L", image.size, colour)
    if "RGB" not in str(image.mode):
        image = image.convert("RGB")
    if -1 in ops:
        image.putalpha(ch)
    mode = image.mode
    rgb = False
    for i in range(3):
        if i in ops:
            rgb = True
    if rgb:
        spl = list(image.split())
        for i in range(3):
            if i in ops:
                spl[i] = ch
        image = Image.merge(mode, spl)
    cmy = False
    for i in range(3, 6):
        if i in ops:
            cmy = True
    if cmy:
        spl = list(ImageChops.invert(image).split())
        for i in range(3, 6):
            if i in ops:
                spl[i - 3] = ch
        image = ImageChops.invert(Image.merge(mode, spl))
    hsv = False
    for i in range(6, 9):
        if i in ops:
            hsv = True
    if hsv:
        if str(image.mode) == "RGBA":
            A = image.getchannel("A")
        else:
            A = None
        spl = list(image.convert("HSV").split())
        for i in range(6, 9):
            if i in ops:
                spl[i - 6] = ch
        image = Image.merge("HSV", spl).convert("RGB")
        if A is not None:
            image.putalpha(A)
    return image


# Image blend operations (this is a bit of a mess)
blenders = {
    "normal": "blend",
    "blt": "blend",
    "blit": "blend",
    "blend": "blend",
    "replace": "replace",
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
    "plusdarker": "OP_X+Y-255",
    "plusdarken": "OP_X+Y-255",
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
    "color": "SP_COL",
    "colour": "SP_COL",
}

def blend_op(image, url, operation, amount, recursive=True):
    op = operation.casefold().replace(" ", "").replace("_", "")
    if op in blenders:
        filt = blenders[op]
    elif op == "auto":
        filt = "blend"
    else:
        raise TypeError("Invalid image operation: \"" + op + '"')
    image2 = get_image(url, url)
    if recursive:
        if not globals()["ANIM"]:
            try:
                image2.seek(1)
            except EOFError:
                image2.seek(0)
            else:
                out = deque()
                total = 0
                for f in range(2147483648):
                    try:
                        image2.seek(f)
                    except EOFError:
                        break
                    total += max(image2.info.get("duration", 0), 1 / 60)
                    if str(image.mode) != "RGBA":
                        temp = image.convert("RGBA")
                    else:
                        temp = image
                    out.append(blend_op(temp, image2, operation, amount, recursive=False))
                return dict(duration=total, frames=out)
        try:
            n_frames = 1
            for f in range(CURRENT_FRAME + 1):
                try:
                    image2.seek(f)
                except EOFError:
                    break
                n_frames += 1
            image2.seek(CURRENT_FRAME % n_frames)
        except EOFError:
            image2.seek(0)
    if image2.width != image.width or image2.height != image.height:
        image2 = resize_to(image2, image.width, image.height, "auto")
    if type(filt) is not str:
        if str(image.mode) != "RGBA":
            image = image.convert("RGBA")
        if str(image2.mode) != "RGBA":
            image2 = image2.convert("RGBA")
        imgA = np.array(image).astype(float)
        imgB = np.array(image2).astype(float)
        out = Image.fromarray(np.uint8(filt(imgA, imgB, amount)))
    else:
        # Basic blend, use second image
        if filt in ("blend", "replace"):
            out = image2
        # Image operation, use ImageMath.eval
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
        # Special operation, use HSV channels
        elif filt.startswith("SP_"):
            f = filt[3:]
            if str(image.mode) == "RGBA":
                A1 = image.getchannel("A")
            else:
                A1 = None
            if str(image2.mode) == "RGBA":
                A2 = image2.getchannel("A")
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
            elif f == "COL":
                channels = [channels2[0], channels2[1], channels[2]]
            out = Image.merge("HSV", channels).convert("RGB")
            if A1 or A2:
                if not A1:
                    A = A2
                elif not A2:
                    A = A1
                else:
                    A = ImageMath.eval("max(X,Y)", dict(X=A1, Y=A2)).convert("L")
                out.putalpha(A)
        # Otherwise attempt to find as ImageChops filter
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
        if op == "blend":
            A = out.getchannel("A")
            A.point(lambda x: round(x * amount))
            out.putalpha(A)
            out = Image.alpha_composite(image, out)
        else:
            out = Image.blend(image, out, amount)
    return out


def remove_matte(image, colour):
    if str(image.mode) != "RGBA":
        image = image.convert("RGBA")
    arr = np.array(image).astype(np.float32)
    col = np.array(colour)
    t = len(col)
    for row in arr:
        for cell in row:
            r = min(1, np.min(cell[:t] / col))
            if r > 0:
                col = cell[:t] - r * col
                if max(col) > 0:
                    ratio = sum(cell) / max(col)
                    cell[:t] = np.clip(col * ratio, 0, 255)
                    cell[3] /= ratio
                else:
                    cell[3] = 0
    image = Image.fromarray(arr.astype(np.uint8))
    return image


def colour_deficiency(image, operation, value=None):
    if value is None:
        if operation == "protanopia":
            operation = "protan"
            value = 0.991
        elif operation == "protanomaly":
            operation = "protan"
            value = 0.516
        if operation == "deuteranopia":
            operation = "deutan"
            value = 0.93
        elif operation == "deuteranomaly":
            operation = "deutan"
            value = 0.458
        elif operation == "tritanopia":
            operation = "tritan"
            value = 0.96
        elif operation == "tritanomaly":
            operation = "tritan"
            value = 0.45
        elif operation == "monochromacy":
            operation = "achro"
            value = 1
        elif operation == "achromatopsia":
            operation = "achro"
            value = 1
        elif operation == "achromatonomaly":
            operation = "achro"
            value = 0.645
        else:
            value = 0.9
    if operation == "protan":
        redscale = [1 - 183 / 516 * value, 183 / 516 * value, 0]
        greenscale = [333 / 516 * value, 1 - 333 / 516 * value, 0]
        bluescale = [0, 125 / 516 * value, 1 - 125 / 516 * value]
    elif operation == "deutan":
        redscale = [1 - 200 / 458 * value, 200 / 458 * value, 0]
        greenscale = [258 / 458 * value, 1 - 258 / 458 * value, 0]
        bluescale = [0, 142 / 458 * value, 1 - 142 / 458 * value]
    elif operation == "tritan":
        redscale = [1 - 33 / 450 * value, 33 / 450 * value, 0]
        greenscale = [0, 1 - 267 / 450 * value, 267 / 450 * value]
        bluescale = [0, 183 / 450 * value, 1 - 183 / 450 * value]
    elif operation == "achro":
        redscale = [1 - 701 / 1000 * value, 587 / 1000 * value, 114 / 1000 * value]
        greenscale = [299 / 1000 * value, 1 - 413 / 1000 * value, 114 / 1000 * value]
        bluescale = [299 / 1000 * value, 587 / 1000 * value, 1 - 886 / 1000 * value]
    else:
        raise TypeError(f"Invalid filter {operation}.")
    ratios = [redscale, greenscale, bluescale]
    channels = list(image.split())
    out = [None] * len(channels)
    if len(out) == 4:
        out[-1] = channels[-1]
    for i_ratio, ratio in enumerate(ratios):
        for i_colour in range(3):
            if ratio[i_colour] != 0:
                if out[i_ratio] is None:
                    out[i_ratio] = channels[i_colour].point(lambda x: x * ratio[i_colour])
                else:
                    out[i_ratio] = ImageChops.add(out[i_ratio], channels[i_colour].point(lambda x: x * ratio[i_colour]))
    return Image.merge(image.mode, out)

Enhance = lambda image, operation, value: getattr(ImageEnhance, operation)(image).enhance(value)

# Hueshift image using HSV channels
def hue_shift(image, value):
    if str(image.mode) == "RGBA":
        A = image.getchannel("A")
    else:
        A = None
    if str(image.mode) != "HSV":
        image = image.convert("HSV")
    channels = list(image.split())
    value *= 256
    channels[0] = channels[0].point(lambda x: (x + value) % 256)
    image = Image.merge("HSV", channels).convert("RGB")
    if A is not None:
        image.putalpha(A)
    return image


# For the ~activity command.
special_colours = {
    "message": (0, 0, 1),
    "typing": (0, 1, 0),
    "command": (0, 1, 1),
    "reaction": (1, 1, 0),
    "misc": (1, 0, 0),
}

def plt_special(d, user=None, **void):
    hours = 168
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.rcParams["figure.dpi"] = 128
    plt.xlim(-hours, 0)
    temp = np.zeros(len(next(iter(d.values()))))
    width = hours / len(temp)
    domain = width * np.arange(-len(temp), 0)
    for k, v in d.items():
        plt.bar(domain, v, bottom=temp, color=special_colours.get(k, "k"), edgecolor="black", width=width, label=k)
        temp += np.array(v)
    plt.bar(list(range(-hours, 0)), np.ones(hours) * max(temp) / 512, edgecolor="black", color="k")
    if user:
        plt.title("Recent Discord Activity for " + user)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Action Count")
    plt.legend(loc="upper left")
    ts = time.time_ns() // 1000
    out = "cache/" + str(ts) + ".png"
    plt.savefig(out)
    plt.clf()
    return "$" + out


discord_emoji = re.compile("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/assets\\/[0-9A-Fa-f]+\\.svg")
is_discord_emoji = lambda url: discord_emoji.search(url)


def write_to(fn, data):
    with open(fn, "wb") as f:
        f.write(data)

def from_bytes(b, save=None):
    if b[:4] == b"<svg" or b[:5] == b"<?xml":
        resp = requests.post("https://www.svgtopng.me/api/svgtopng/upload-file", headers=header(), files={"files": ("temp.svg", b, "image/svg+xml"), "format": (None, "PNG"), "forceTransparentWhite": (None, "true"), "jpegQuality": (None, "256")})
        z = ZipFile(io.BytesIO(resp.content), compression=zipfile.ZIP_DEFLATED, strict_timestamps=False)
        data = z.open("temp.png").read()
        out = io.BytesIO(data)
        z.close()
        if save and data and not os.path.exists(save):
            exc.submit(write_to, save, data)
    elif b[:4] == b"%PDF":
        return ImageSequence(*pdf2image.convert_from_bytes(b, poppler_path="misc/poppler", use_pdftocairo=True))
    else:
        out = io.BytesIO(b) if type(b) is bytes else b
    try:
        return Image.open(out)
    except PIL.UnidentifiedImageError:
        file_print(b[:1024])
        raise


class seq(io.IOBase, collections.abc.MutableSequence, contextlib.AbstractContextManager):

    BUF = 262144

    def __init__(self, obj, filename=None):
        self.iter = None
        self.closer = getattr(obj, "close", None)
        if issubclass(type(obj), io.IOBase):
            if issubclass(type(obj), io.BytesIO):
                self.data = obj
            else:
                obj.seek(0)
                self.data = io.BytesIO(obj.read())
                obj.seek(0)
        elif issubclass(type(obj), bytes) or issubclass(type(obj), bytearray) or issubclass(type(obj), memoryview):
            self.data = io.BytesIO(obj)
        elif issubclass(type(obj), collections.abc.Iterator):
            self.iter = iter(obj)
            self.data = io.BytesIO()
            self.high = 0
        elif issubclass(type(obj), requests.models.Response):
            self.iter = obj.iter_content(self.BUF)
            self.data = io.BytesIO()
            self.high = 0
        else:
            raise TypeError(f"a bytes-like object is required, not '{type(obj)}'")
        self.filename = filename
        self.buffer = {}

    def __getitem__(self, k):
        if type(k) is slice:
            out = io.BytesIO()
            start = k.start or 0
            stop = k.stop or inf
            step = k.step or 1
            if step < 0:
                start, stop, step = stop + 1, start + 1, -step
                rev = True
            else:
                rev = False
            curr = start // self.BUF * self.BUF
            offs = start % self.BUF
            out.write(self.load(curr))
            curr += self.BUF
            while curr < stop:
                temp = self.load(curr)
                if not temp:
                    break
                out.write(temp)
                curr += self.BUF
            out.seek(0)
            return out.read()[k]
        base = k // self.BUF
        with suppress(KeyError):
            return self.load(base)[k % self.BUF]
        raise IndexError("seq index out of range")

    def __str__(self):
        if self.filename is None:
            return str(self.data)
        if self.filename:
            return f"<seq name='{self.filename}'>"
        return f"<seq object at {hex(id(self))}"

    def __iter__(self):
        i = 0
        while True:
            x = self[i]
            if x:
                yield x
            i += 1

    def __getattr__(self, k):
        if k in ("data", "filename"):
            return self.data
        return object.__getattribute__(self.data, k)

    close = lambda self: self.closer() if self.closer else None
    __exit__ = lambda self, *args: self.close()

    def load(self, k):
        with suppress(KeyError):
            return self.buffer[k]
        seek = getattr(self.data, "seek", None)
        if seek:
            if self.iter is not None and k + self.BUF >= self.high:
                seek(self.high)
                with suppress(StopIteration):
                    while k + self.BUF >= self.high:
                        temp = next(self.iter)
                        self.data.write(temp)
                        self.high += len(temp)
            seek(k)
            self.buffer[k] = self.data.read(self.BUF)
        else:
            with suppress(StopIteration):
                while self.high < k:
                    temp = next(self.data)
                    if not temp:
                        return b""
                    self.buffer[self.high] = temp
                    self.high += self.BUF
        return self.buffer.get(k, b"")


def ImageOpIterator(image, step, operation, ts):
    # Attempt to perform operation on all individual frames of .gif images
    for i, f in enumerate(range(0, 2147483648, step)):
        np.random.seed(ts & 4294967295)
        globals()["CURRENT_FRAME"] = i
        try:
            image.seek(f)
        except EOFError:
            break
        if str(image.mode) != "RGBA":
            temp = image.convert("RGBA")
        else:
            temp = image
        func = getattr(temp, operation, None)
        if func is None:
            res = eval(operation)(temp, *args)
        else:
            res = func(*args)
        yield res


class ImageSequence(Image.Image):

    def __init__(self, *images):
        self._images = [image.copy() for image in images]
        self._position = 0

    def seek(self, position):
        if position >= len(self._images):
            raise EOFError
        self._position = position
    
    def __getattr__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return getattr(self._images[self._position], key)


def get_image(url, out):
    if issubclass(type(url), Image.Image):
        return url
    if type(url) not in (bytes, bytearray, io.BytesIO, seq):
        save = None
        if url in CACHE:
            return CACHE[url]
        if is_url(url):
            data = None
            if is_discord_emoji(url):
                save = f"cache/emoji_{url.rsplit('/', 1)[-1].split('.', 1)[0]}"
                if os.path.exists(save):
                    with open(save, "rb") as f:
                        data = f.read()
                    file_print(f"Emoji {save} successfully loaded from cache.")
            if data is None:
                data = get_request(url)
            if len(data) > 8589934592:
                raise OverflowError("Max file size to load is 8GB.")
        else:
            if os.path.getsize(url) > 8589934592:
                raise OverflowError("Max file size to load is 8GB.")
            with open(url, "rb") as f:
                data = f.read()
            if out != url and out:
                try:
                    os.remove(url)
                except:
                    pass
        image = from_bytes(data, save)
        CACHE[url] = image
    else:
        if len(url) > 8589934592:
            raise OverflowError("Max file size to load is 8GB.")
        image = from_bytes(url)
    return image


# Main image operation function
@logging
def evalImg(url, operation, args):
    globals()["CURRENT_FRAME"] = 0
    ts = time.time_ns() // 1000
    out = "cache/" + str(ts) + ".png"
    args = eval(args)
    if operation != "$":
        if args and args[-1] == "-raw":
            args.pop(-1)
            image = get_request(url)
        else:
            image = get_image(url, out)
        # $%GIF%$ is a special case where the output is always a .gif image
        if args and args[-1] == "-gif":
            new = eval(operation)(image, *args[:-1])
        else:
            try:
                if args and args[0] == "-nogif":
                    args = args[1:]
                    raise EOFError
                image.seek(1)
            except EOFError:
                globals()["ANIM"] = False
                image.seek(0)
                func = getattr(image, operation, None)
                if func is None:
                    new = eval(operation)(image, *args)
                else:
                    new = func(*args)
            else:
                new = dict(frames=deque(), duration=0)
                globals()["ANIM"] = True
                for f in range(2147483648):
                    try:
                        image.seek(f)
                    except EOFError:
                        break
                    new["duration"] += max(image.info.get("duration", 0), 1 / 60)
                fps = 1000 * f / new["duration"]
                step = 1
                while f // step > 4096 and fps // step >= 24:
                    step += 1
                new["count"] = f // step
                new["frames"] = ImageOpIterator(image, step, operation=operation, ts=ts)
    else:
        new = eval(url)(*args)
    if type(new) is dict:
        duration = new["duration"]
        frames = new["frames"]
        if not frames:
            raise EOFError("No image output detected.")
        elif new["count"] == 1:
            new = frames[0]
        else:
            fps = 1000 * new["count"] / duration
            if issubclass(type(frames), collections.abc.Sequence):
                first = frames[0]
            else:
                it = iter(frames)
                first = next(it)

                def frameit():
                    with suppress(StopIteration):
                        while True:
                            yield next(it)

                frames = frameit()
            size = first.size
            out = "cache/" + str(ts) + ".gif"
            command = ["ffmpeg", "-threads", "2", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-r", str(fps), "-pix_fmt", "rgba", "-video_size", "x".join(str(i) for i in size), "-i", "-"]
            if new["count"] > 4096:
                vf = "split[s0][s1];[s0]palettegen=reserve_transparent=1:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle:alpha_threshold=128"
            else:
                vf = "split[s0][s1];[s0]palettegen=reserve_transparent=1:stats_mode=diff[p];[s1][p]paletteuse=diff_mode=rectangle:alpha_threshold=128"
            command.extend(["-gifflags", "-offsetting", "-an", "-vf", vf, "-loop", "0", out])
            file_print(command)
            file_print(new["count"])
            proc = psutil.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for frame in frames:
                if issubclass(type(frame), Image.Image):
                    if frame.size != size:
                        frame = frame.resize(size)
                    if str(frame.mode) != "RGBA":
                        frame = frame.convert("RGBA")
                    b = frame.tobytes()
                    # arr = np.array(frame)
                    # b = arr.tobytes()
                elif type(frame) is io.BytesIO:
                    b = frame.read()
                else:
                    b = frame
                proc.stdin.write(b)
                time.sleep(0.02)
            proc.stdin.close()
            proc.wait()
            return repr([out])
    if issubclass(type(new), Image.Image):
        new.save(out, "png")
        return repr([out])
    elif type(new) is str and new.startswith("$"):
        return repr([new[1:]])
    return repr(str(new).encode("utf-8"))


if __name__ == "__main__":
    while True:
        try:
            args = eval(sys.stdin.readline()).decode("utf-8", "replace").strip().split("`")
            resp = evalImg(*args)
            sys.stdout.write(repr(resp.encode("utf-8")) + "\n")
            sys.stdout.flush()
        except Exception as ex:
            # Exceptions are evaluated and handled by main process
            sys.stdout.write(repr(ex) + "\n")
            sys.stdout.flush()
        time.sleep(0.01)
        if time.time() - start > 3600:
            start = time.time()
            for img in CACHE.values():
                img.close()
            CACHE.clear()
