try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *
import flask
from flask import Flask
import werkzeug
from werkzeug.exceptions import HTTPException


HOST = "https://mizabot.xyz"
PORT = AUTH.get("webserver_port", 9801)
IND = "\x7f"


def send(*args, escape=True):
    try:
        s = " ".join(str(i) for i in args)
        if escape:
            s = "\x00" + s
        if s:
            if s[-1] != "\n":
                s += "\n"
            s = s.encode("utf-8")
            sys.stdout.buffer.write(s)
            sys.__stderr__.buffer.write(s)
            sys.__stderr__.flush()
    except OSError:
        psutil.Process().kill()

app = Flask(__name__, static_url_path="/static")
app.url_map.strict_slashes = False
# app.use_x_sendfile = True


@app.errorhandler(Exception)
def on_error(ex):
    send(repr(ex))
    # Redirect HTTP errors to http.cat, python exceptions go to code 500 (internal server error)
    if issubclass(type(ex), HTTPException):
        return flask.redirect(f"https://http.cat/{ex.code}")
    if issubclass(type(ex), FileNotFoundError):
        return flask.redirect("https://http.cat/404")
    if issubclass(type(ex), EOFError):
        return flask.redirect("https://http.cat/204")
    if issubclass(type(ex), PermissionError):
        return flask.redirect("https://http.cat/403")
    if issubclass(type(ex), TimeoutError) or issubclass(type(ex), concurrent.futures.TimeoutError):
        return flask.redirect("https://http.cat/504")
    if issubclass(type(ex), ConnectionError):
        return flask.redirect("https://http.cat/502")
    return flask.redirect("https://http.cat/500")


def create_etag(data):
    s = str(ihash(data[:128] + data[-128:]) + len(data) & 4294967295)
    return '"' + "0" * (10 - len(s)) + s + '"'


STATIC = {}
TZCACHE = {}
RESPONSES = {}
RESPONSES[0] = cdict(set_result=lambda *args: None)


PREVIEW = {}
prev_date = utc_dt().date()

@app.route("/preview/<path>", methods=["GET"])
@app.route("/view/<path>", methods=["GET"])
@app.route("/file/<path>", methods=["GET"])
@app.route("/files/<path>", methods=["GET"])
@app.route("/download/<path>", methods=["GET"])
@app.route("/preview/<path>/<path:filename>", methods=["GET"])
@app.route("/view/<path>/<path:filename>", methods=["GET"])
@app.route("/file/<path>/<path:filename>", methods=["GET"])
@app.route("/files/<path>/<path:filename>", methods=["GET"])
@app.route("/download/<path>/<path:filename>", methods=["GET"])
def get_file(path, filename=None):
    if path in ("hacks", "mods", "files", "download", "static"):
        send(flask.request.remote_addr + " was rickrolled 🙃")
        return flask.redirect("https://youtu.be/dQw4w9WgXcQ")
    orig_path = path
    ind = IND
    if path.startswith("~"):
        b = path.split(".", 1)[0].encode("utf-8") + b"==="
        if (len(b) - 1) & 3 == 0:
            b += b"="
        path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
    elif path.startswith("!"):
        ind = "!"
        path = path[1:]
    p = find_file(path, ind=ind)
    endpoint = flask.request.path[1:].split("/", 1)[0]
    down = flask.request.args.get("download", "false")
    download = down and down[0] not in "0fFnN" or endpoint == "download"
    if download:
        mime = MIMES.get(p.rsplit("/", 1)[-1].rsplit(".", 1)[-1])
    else:
        mime = get_mime(p)
    fn = p.rsplit("/", 1)[-1].split("~", 1)[-1].rstrip(IND)
    if endpoint.endswith("view") and mime.startswith("image/"):
        if os.path.getsize(p) > 1048576:
            if endpoint != "preview":
                og_image = flask.request.host_url + "preview/" + orig_path
                data = f"""<!DOCTYPE html>
<html>
<meta name="robots" content="noindex"><link rel="image_src" href="{og_image}">
<meta property="og:image" content="{og_image}" itemprop="image">
<meta property="og:url" content="{flask.request.host_url}view/{orig_path}">
<meta property="og:image:width" content="1280">
<meta property="og:type" content="website">
<meta http-equiv="refresh" content="0; URL={flask.request.host_url}files/{orig_path}">
</html>"""
                resp = flask.Response(data, mimetype="text/html")
                resp.headers.update(CHEADERS)
                resp.headers["ETag"] = create_etag(data)
                return resp
            if prev_date != utc_dt().date():
                PREVIEW.clear()
            elif path in PREVIEW:
                p = os.getcwd() + "/cache/" + PREVIEW[path]
            else:
                fmt = mime.rsplit('/', 1)[-1]
                if fmt != "gif":
                    fmt = "png"
                p2 = f"{path}~preview.{fmt}"
                p3 = os.getcwd() + "/cache/" + p2
                if not os.path.exists(p3):
                    args = ["ffmpeg", "-n", "-hide_banner", "-loglevel", "error", "-i", p, "-fs", "4194304", "-vf", "scale=320:-1", p3]
                    send(args)
                    proc = psutil.Popen(args)
                    proc.wait()
                PREVIEW[path] = p2
                p = p3
    elif endpoint == "download" and p[-1] != IND and not p.endswith(".zip"):
        size = os.path.getsize(p)
        if size > 16777216:
            fi = p.rsplit(".", 1)[0] + ".zip" + IND
            if not os.path.exists(fi):
                test = min(67108864, max(1048576, size >> 6))
                send(f"Testing {p} with {test} bytes...")
                with open(p, "rb") as f:
                    data = f.read(test)
                b = bytes2zip(data)
                r = len(b) / test
                if r < 0.75:
                    send(f"Zipping {p}...")
                    with ZipFile(fi, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as z:
                        z.write(p, arcname=filename or fn)
                    r = os.path.getsize(fi) / size
            else:
                r = os.path.getsize(fi) / size
            if r < 0.8:
                p = fi
            else:
                send(f"{p} has compression ratio {r}, skipping...")
    resp = flask.send_file(p, as_attachment=download, attachment_filename=filename or fn, mimetype=mime, conditional=True)
    resp.headers.update(CHEADERS)
    return resp


def fetch_static(path, ignore=False):
    while path.startswith("../"):
        path = path[3:]
    try:
        try:
            data = STATIC[path]
        except KeyError:
            with open(f"misc/{path}", "rb") as f:
                data = f.read()
            STATIC[path] = data
        fmt = path.rsplit(".", 1)[-1].casefold()
        try:
            mime = MIMES[fmt]
        except KeyError:
            mime = "text/html"
        return data, mime
    except:
        if not ignore:
            send(path)
            send(traceback.format_exc())
        raise

@app.route("/static/<filename>", methods=["GET"])
@app.route("/static/<path:filename>", methods=["GET"])
def get_static_file(filename):
    data, mime = fetch_static(filename)
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/static", methods=["DELETE"])
def clearcache():
    ip = flask.request.remote_addr
    if ip == "127.0.0.1":
        STATIC.clear()
        send("Webserver cache cleared.")
        return b"\xf0\x9f\x92\x9c"
    raise PermissionError


@app.route("/mizatlas", methods=["GET"])
@app.route("/mizatlas/<path:filename>", methods=["GET"])
def mizatlas(filename=None):
    data = None
    if filename:
        with suppress(FileNotFoundError):
            data, mime = fetch_static(f"mizatlas/{filename}")
    if not data:
        data, mime = fetch_static("mizatlas/index.html")
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp


@app.route("/models/<path:filename>", methods=["GET"])
def models(filename):
    data, mime = fetch_static(f"waifu2x/models/{filename}")
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/waifu2x/<path:filename>", methods=["GET"])
def waifu2x_ex(filename):
    data, mime = fetch_static("waifu2x/main.js")
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/waifu2x", methods=["GET"])
def waifu2x():
    source = flask.request.args.get("source")
    if source:
        if not is_url(source):
            raise FileNotFoundError
        if not regexp("https:\\/\\/images-ext-[0-9]+\\.discordapp\\.net\\/external\\/").match(source) and not source.startswith("https://media.discordapp.net/"):
            if not source.startswith(flask.request.host_url):
                t = ts_us()
                while t in RESPONSES:
                    t += 1
                RESPONSES[t] = fut = concurrent.futures.Future()
                send(f"!{t}\x7fbot.data.exec.proxy({repr(source)})", escape=False)
                j, after = fut.result()
                RESPONSES.pop(t, None)
                source = j["result"]
    if source:
        src = '<source src="' + source + '" id="imageIn"/>'
        model = '<span value="models/upconv_7/art/scale2.0x_model"/>'
    else:
        src = '<input type="file" id="imageIn" accept="image/png, image/jpeg"/>'
        model = '''<select id="modelName">
            <option value="models/upconv_7/art/scale2.0x_model">Waifu2x Upconv7 Art</option>
            <option value="models/upconv_7/art/noise0_scale2.0x_model">Waifu2x Upconv7 Art (noise0)</option>
            <option value="models/upconv_7/art/noise1_scale2.0x_model">Waifu2x Upconv7 Art (noise1)</option>
            <option value="models/upconv_7/art/noise2_scale2.0x_model">Waifu2x Upconv7 Art (noise2)</option>
            <option value="models/upconv_7/art/noise3_scale2.0x_model">Waifu2x Upconv7 Art (noise3)</option>
            <option value="models/upconv_7/art/scale2.0x_model">Waifu2x Upconv7 Photo</option>
            <option value="models/upconv_7/art/noise0_scale2.0x_model">Waifu2x Upconv7 Photo (noise0)</option>
            <option value="models/upconv_7/art/noise1_scale2.0x_model">Waifu2x Upconv7 Photo (noise1)</option>
            <option value="models/upconv_7/art/noise2_scale2.0x_model">Waifu2x Upconv7 Photo (noise2)</option>
            <option value="models/upconv_7/art/noise3_scale2.0x_model">Waifu2x Upconv7 Photo (noise3)</option>
        </select>'''
    data = f"""<!DOCTYPE html>
<html>
    <meta property="og:image" content="{source}">""" + """
    <style>
        .center {
            margin: 0;
            position: absolute;
            top: 50%;
            left: 50%;
            -ms-transform: translate(-50%, -50%);
            transform: translate(-50%, -50%);
            color: white;
        }
    </style>""" + f"""
	<body>
		<p>file</p>
        {src}
		<p>model</p>
		{model}
		<p>Model data from <a href="https://github.com/nagadomi/waifu2x/">nagadomi waifu2x</a></p>
		<button id="runButton">Run</button>
		<button id="cancelButton">Cancel</button>
		<p id="statusDiv">JS not loaded yet...</p>
		<p>experimental. exposure to high amounts of data may result in hazardous levels of memory usage, which may result in system OOM.</p>
		<p>view</p>
		<canvas id="canvas" width="400" height="400"></canvas>
		<script src="{flask.request.base_url}/main.js"></script>
	</body>
</html>"""
    resp = flask.Response(data, mimetype="text/html")
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp


@app.route("/", methods=["GET", "POST"])
def home():
    data, mime = fetch_static("index.html")
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    data, mime = fetch_static("icon.ico")
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/ip", methods=["GET"])
def get_ip():
    data = json.dumps(dict(
        remote=flask.request.remote_addr,
        host=flask.request.host,
    ))
    resp = flask.Response(data, mimetype="application/json")
    resp.headers.update(SHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp


est_time = utc()
est_last = -inf

def estimate_life():
    global est_time, est_last
    with tracebacksuppressor:
        hosted = sorted(int(f[1:].split("~", 1)[0]) / 1e6 for f in os.listdir("cache") if f.startswith(IND))
        if not hosted:
            est_last = -inf
        ts = hosted[0]
        t = ts_us()
        while t in RESPONSES:
            t += 1
        RESPONSES[t] = fut = concurrent.futures.Future()
        send(f"!{t}\x7fbot.storage_ratio", escape=False)
        j, after = fut.result()
        RESPONSES.pop(t, None)
        last = (utc() - ts) / j.get("result", 1)
        send(last)
        est_time = utc() - last
        est_last = utc()

estimate_life_after = lambda t: time.sleep(t) or estimate_life()

create_future_ex(estimate_life_after, 10)


@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    global est_time
    ip = flask.request.remote_addr
    files = [file for file in flask.request.files.getlist("file") if file.filename]
    if not files:
        raise EOFError
    ts = time.time_ns() // 1000
    urls = deque()
    futs = deque()
    for file in files:
        fn = file.filename
        sfn = f"cache/{IND}{ts}~{fn}"
        futs.append(create_future_ex(file.save, sfn))
        href = f"/files/{ts}/{fn}"
        b = ts.bit_length() + 7 >> 3
        url = f"{HOST}/view/~" + as_str(base64.urlsafe_b64encode(ts.to_bytes(b, "big"))).rstrip("=")
        data = (href, url, sfn)
        urls.append(data)
        send(ip + "\t" + fn + "\t" + str(data))
        ts += 1
    s = """<!DOCTYPE html>
<html>
    <head>
        <style>
        body {
            text-align: center;
            font-family: 'Comic Sans MS';
        }
        img {
            margin-top: 32px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        a:link {
            color: #ffff00;
            text-decoration: none;
        }
        a:visited {
            color: #ffff00;
            text-decoration: none;
        }
        a:hover {
            color: #ff0000;
            text-decoration: underline;
        }
        a:active {
            color: #00ff00;
            text-decoration: underline;
        }
        </style>
    </head>
    <body style="background-color:black;">
        <h1 style="color:white;">Upload successful!</h1>"""
    with tracebacksuppressor:
        for fut in futs:
            fut.result()
        s += f"""
        <p style="color:#00ffff;">Total file size: {byte_scale(sum(os.path.getsize(f[2]) for f in urls))}B</p>
        <p style="color:#bf7fff;">Estimated file lifetime: {sec2time(utc() - est_time)}</p>"""
        for fi in urls:
            s += f'\n<a href="{fi[0]}">{fi[1]}<br></a>'
        preview = deque()
        for f in urls:
            mime = get_mime(f[2])
            if mime.startswith("image/"):
                preview.append(f'<img width="480" src="{f[0].replace("/view/", "/preview/")}" alt="{f[2].split("~", 1)[-1]}">')
            elif mime.startswith("audio/"):
                preview.append(f'<div align="center"><audio controls><source src="{f[0]}" type="{mime}"></audio></div>')
            elif mime.startswith("video/"):
                preview.append(f'<div align="center"><video width="480" controls><source src="{f[0]}" type="{mime}"></video></div>')
            elif mime.startswith("text/"):
                preview.append(f'<a href="{fi[0].replace("/view/", "/files/")}">{fi[1].replace("/view/", "/files/")}</a>')
            else:
                preview.append(f'<a href="{fi[0].replace("/view/", "/download/")}">{fi[1].replace("/view/", "/files/")}</a>')
    if not preview:
        preview.append(f'<img src="{flask.request.host_url}static/hug.gif" alt="Miza-Dottie-Hug" style="width:14.2857%;height:14.2857%;">')
    s += "\n" + "\n".join(preview)
    s += """
        <p><a href="/upload">Click here to upload another file!</a></p>
    </body>
</html>"""
    return s

@app.route("/upload", methods=["GET", "POST"])
def upload():
    global est_last
    ip = flask.request.remote_addr
    colour = hex(colour2raw(hue2colour(xrand(1536))))[2:].upper()
    if utc() - est_last > 1800:
        est_last = utc()
        create_future_ex(estimate_life)
    data = f"""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Files</title>
        <meta content="Files" property="og:title">
        <meta content="Upload a file here!" property="og:description">
        <meta content="{flask.request.url}" property="og:url">
        <meta property="og:image" content="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif">
        <meta content="#BF7FFF" data-react-helmet="true" name="theme-color">
        <link href="https://unpkg.com/boxicons@2.0.7/css/boxicons.min.css" rel="stylesheet">
    </head>""" + """
    <style>
        body {
            background-image: url('""" + flask.request.host_url + """static/spiral.gif');
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
            font-family: 'Comic Sans MS';
        }
        .select {
            vertical-align: center;
            background: transparent;
            color: white;
            width: 100%;
            height: 100%;
            font-weight: 400;
            align-items: center;
        }
        .center {
            margin: 0;
            position: absolute;
            top: 50%;
            left: 50%;
            -ms-transform: translate(-50%, -50%);
            transform: translate(-50%, -50%);
        }""" + f"""
    </style>
    <body>
        <link href="/static/hamburger.css" rel="stylesheet">
        <div class="hamburger">
            <input
                type="checkbox"
                title="Toggle menu"
            />
            <div class="items select">
                <a href="/" data-popup="Home"><img
                    src="{flask.request.host_url}static/avatar-rainbow.gif"
                /></a>
                <a href="/mizatlas" data-popup="Command Atlas"><img
                    src="{flask.request.host_url}static/background-rainbow.gif"
                /></a>
                <a href="/upload" data-popup="File Host"><img
                    src="{flask.request.host_url}static/sky-rainbow.gif"
                /></a>
                <a 
                    href="/time"
                    data-popup="Clock"
                    class='bx bx-time'></a>
            </div>
            <div class="hambg"></div>
        </div>
        <div class="center">
            <h1 align="center" style="color:white;">Upload a file here!</h1>
            <form action="/upload_file" method="POST" enctype="multipart/form-data">
                <input style="color:white;" type="file" name="file" multiple/>
                <input type="submit"/>
            </form>
        </div>
    </body>
</html>"""
    resp = flask.Response(data, mimetype="text/html")
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp


geo_sem = Semaphore(90, 256, rate_limit=60)
geo_count = 0

def get_geo(ip):
    global geo_count
    try:
        resp = TZCACHE[ip]
    except KeyError:
        if geo_count & 1:
            url = f"http://ip-api.com/json/{ip}?fields=256"
        else:
            url = f"https://pro.ip-api.com/json/{ip}?fields=256&key=test-demo-pro"
        geo_count += 1
        with geo_sem:
            resp = requests.get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}", "Origin": "https://members.ip-api.com"})
        resp.raise_for_status()
        TZCACHE[ip] = resp = resp.json()
        send(ip + "\t" + "\t".join(resp.values()))
    return resp

@app.route("/time", methods=["GET", "POST"])
@app.route("/timezone", methods=["GET", "POST"])
@app.route("/timezones", methods=["GET", "POST"])
def timezone():
    ip = flask.request.remote_addr
    try:
        data = get_geo(ip)
        tz = data["timezone"]
        dt = datetime.datetime.now(pytz.timezone(tz))
        colour = hex(colour2raw(hue2colour(xrand(1536))))[2:].upper()
        html = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Timezones</title>
        <meta content="Timezones" property="og:title">
        <meta content="Find your current timezone here!" property="og:description">
        <meta content=\"""" + flask.request.url + """\" property="og:url">
        <meta property="og:image" content="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif">
        <meta content="#""" + colour + """\" data-react-helmet="true" name="theme-color">
        <meta http-equiv="refresh" content="60">
        <link rel="stylesheet" type="text/css" href="/static/timezonestyles.css">
        <link href="https://unpkg.com/boxicons@2.0.7/css/boxicons.min.css" rel="stylesheet">
    </head>
    <body>""" + f"""
        <link href="/static/hamburger.css" rel="stylesheet">
        <div class="hamburger">
            <input
                type="checkbox"
                title="Toggle menu"
            />
            <div class="items select">
                <a href="/" data-popup="Home"><img
                    src="{flask.request.host_url}static/avatar-rainbow.gif"
                /></a>
                <a href="/mizatlas" data-popup="Command Atlas"><img
                    src="{flask.request.host_url}static/background-rainbow.gif"
                /></a>
                <a href="/upload" data-popup="File Host"><img
                    src="{flask.request.host_url}static/sky-rainbow.gif"
                /></a>
                <a 
                    href="/time"
                    data-popup="Clock"
                    class='bx bx-time'></a>
            </div>
            <div class="hambg"></div>
        </div>
        <div class="main">
            <h3>Estimated time:</h3>
            <h1>""" + str(dt) + """</h1>
            <h2>Detected timezone: """ + tz + f"""</h2>
            <p class="align_left">
                <a class="glow" href="/time">Refresh</a>
            </p>
            <p class="align_right">
                <a class="glow" href="/">Home</a>
            </p>
        </div>
    <img class="border" src="{flask.request.host_url}static/sky-rainbow.gif" alt="Miza-Sky" style="width:14.2857%;height:14.2857%;">
    </body>
</html>"""
        return html
    except KeyError:
        return flask.redirect("https://http.cat/417")
    except:
        send(traceback.format_exc())
        raise


@app.route("/eval/<string:token>/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
@app.route("/exec/<string:token>/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
def execute(token, content):
    if token != AUTH.get("discord_token"):
        return flask.redirect("https://http.cat/401")
    content = urllib.parse.unquote(flask.request.full_path.rstrip("?").lstrip("/").split("/", 2)[-1])
    t = ts_us()
    while t in RESPONSES:
        t += 1
    RESPONSES[t] = fut = concurrent.futures.Future()
    send(f"!{t}\x7f{content}", escape=False)
    j, after = fut.result()
    RESPONSES.pop(t, None)
    return j


@app.route("/command/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
@app.route("/commands/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
def command(content):
    ip = flask.request.remote_addr
    if ip == "127.0.0.1":
        t, after = content.split("\x7f", 1)
        t = int(t)
        after = float(after)
        j = flask.request.get_json(force=True)
        if t in RESPONSES:
            RESPONSES[t].set_result((j, after))
            send(j)
            return b"\xf0\x9f\x92\x9c"
    content = urllib.parse.unquote(flask.request.full_path.rstrip("?").lstrip("/").split("/", 1)[-1])
    data = get_geo(ip)
    tz = data["timezone"]
    if " " not in content:
        content += " "
    t = ts_us()
    while t in RESPONSES:
        t += 1
    RESPONSES[t] = fut = concurrent.futures.Future()
    send(f"~{t}\x7f{ip}\x7f{tz}\x7f{content}", escape=False)
    j, after = fut.result(timeout=420)
    RESPONSES.pop(t, None)
    response = flask.Response(json.dumps(j), mimetype="application/json")
    a = after - utc()
    if a > 0:
        response.headers["Retry-After"] = a
    return response


@app.route("/cat", methods=["GET"])
@app.route("/cats", methods=["GET"])
def cat():
    t = ts_us()
    while t in RESPONSES:
        t += 1
    RESPONSES[t] = fut = concurrent.futures.Future()
    send(f"!{t}\x7fbot.commands.cat[0](bot, None, 'v')", escape=False)
    j, after = fut.result()
    RESPONSES.pop(t, None)
    url = j["result"]
    if fcdict(flask.request.headers).get("Accept") == "application/json":
        return url
    refresh = float(flask.request.args.get("refresh", 60))
    return f"""<!DOCTYPE html>
<html>
<head>
<meta property="og:image" content="{url}">
<meta http-equiv="refresh" content="{refresh}; URL={flask.request.url}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>""" + """
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  margin-top: auto;
  margin-bottom: auto;
}
.center {
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}""" + f"""
</style>
</head>
<body style="background-color:black;">
<img src="{url}" class="center">
</body>
</html>"""

@app.route("/dog", methods=["GET"])
@app.route("/dogs", methods=["GET"])
def dog():
    t = ts_us()
    while t in RESPONSES:
        t += 1
    RESPONSES[t] = fut = concurrent.futures.Future()
    send(f"!{t}\x7fbot.commands.dog[0](bot, None, 'v')", escape=False)
    j, after = fut.result()
    RESPONSES.pop(t, None)
    url = j["result"]
    refresh = float(flask.request.args.get("refresh", 60))
    if fcdict(flask.request.headers).get("Accept") == "application/json":
        return url
    return f"""<!DOCTYPE html>
<html>
<head>
<meta property="og:image" content="{url}">
<meta http-equiv="refresh" content="{refresh}; URL={flask.request.url}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>""" + """
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  margin-top: auto;
  margin-bottom: auto;
}
.center {
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}""" + f"""
</style>
</head>
<body style="background-color:black;">
<img src="{url}" class="center">
</body>
</html>"""


HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Server": "Miza",
    "Vary": "Accept-Encoding",
    "Accept-Ranges": "bytes",
    "Access-Control-Allow-Origin": "*",
}

CHEADERS = {"Cache-Control": "public, max-age=3600, stale-while-revalidate=1073741824, stale-if-error=1073741824"}
SHEADERS = {"Cache-Control": "public, max-age=30, stale-while-revalidate=1073741824, stale-if-error=1073741824"}

@app.after_request
def custom_header(response):
    response.headers.update(HEADERS)
    return response


def ensure_parent(proc, parent):
    while True:
        if not parent.is_running():
            psutil.Process().kill()
        # t = ts_us()
        # while t in RESPONSES:
        #     t += 1
        # RESPONSES[t] = fut = concurrent.futures.Future()
        # send(f"!{t}\x7f{content}", escape=False)
        # j, after = fut.result()
        # RESPONSES.pop(t, None)
        # send(f"!{t}\x7fGC.__setitem__({proc.pid}, {len(gc.get_objects())})", escape=False)
        time.sleep(6)

if __name__ == "__main__":
    pid = os.getpid()
    ppid = os.getppid()
    send(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
    proc = psutil.Process(pid)
    parent = psutil.Process(ppid)
    create_thread(ensure_parent, proc, parent)
    app.run("0.0.0.0", PORT)