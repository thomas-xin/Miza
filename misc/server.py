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
            sys.stdout.write(s)
            sys.__stderr__.write(s)
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
    s = str(ihash(data[:128] + data[-128:]) % 4294967296)
    return repr("0" * (10 - len(s)) + s)


STATIC = {}
TZCACHE = {}
RESPONSES = {}


def find_file(path):
    # if no file name is inputted, return no content
    if not path:
        raise EOFError
    # do not include "." in the path name
    path = path.rsplit(".", 1)[0]
    fn = f"{IND}{path}"
    for file in reversed(os.listdir("cache")):
        # file cache is stored as "{timestamp}~{name}", search for file via timestamp
        if file.rsplit(".", 1)[0].split("~", 1)[0] == fn:
            return os.getcwd() + "/cache/" + file
    raise FileNotFoundError(path)


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
    orig_path = path
    if path.startswith("~"):
        path = str(int.from_bytes(base64.urlsafe_b64decode(path.encode("utf-8") + b"==="), "big"))
    p = find_file(path)
    endpoint = flask.request.path[1:].split("/", 1)[0]
    down = flask.request.args.get("download", "false")
    download = down and down[0] not in "0fFnN" or endpoint == "download"
    if download:
        mime = MIMES.get(p.rsplit("/", 1)[-1].rsplit(".", 1)[-1])
    else:
        mime = get_mime(p)
    fn = p.rsplit('/', 1)[-1].split('~', 1)[-1]
    send(p, fn, mime)
    if endpoint.endswith("view") and mime.startswith("image/"):
        if os.path.getsize(p) > 262144:
            if endpoint != "preview":
                og_image = flask.request.host_url + "preview/" + orig_path
                return f"""<!DOCTYPE html>
<html>
<meta name="robots" content="noindex"><link rel="image_src" href="{og_image}">
<meta property="og:image" itemprop="image" content="{og_image}">
<meta property="og:url" content="{flask.request.host_url}view/{orig_path}">
<meta property="og:image:width" content="1280">
<meta property="og:type" content="website">
<meta http-equiv="refresh" content="0; URL={flask.request.host_url}files/{orig_path}" />
</html>"""
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
    resp = flask.send_file(p, as_attachment=download, attachment_filename=filename or fn, mimetype=mime)
    resp.headers.update(CHEADERS)
    send(resp)
    return resp


def fetch_static(path):
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
        send(path)
        send(traceback.format_exc())
        raise

@app.route("/static/<filename>", methods=["GET"])
@app.route("/static/<path:filename>", methods=["GET"])
def get_static_file(filename):
    data, mime = fetch_static(filename)
    send("static/" + filename, mime)
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
    data, mime = fetch_static("mizatlas/index.html")
    send("mizatlas/index.html", mime)
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp


@app.route("/", methods=["GET", "POST"])
def home():
    data, mime = fetch_static("index.html")
    send("index.html", mime)
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
    resp.headers["ETag"] = create_etag(data)
    return resp

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    data, mime = fetch_static("icon.ico")
    send("icon.ico", mime)
    resp = flask.Response(data, mimetype=mime)
    resp.headers.update(CHEADERS)
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
        url = f"{HOST}{href}"
        urls.append((href, url, sfn))
        send(ip + "\t" + fn + "\t" + url)
        ts += 1
    s = """<!DOCTYPE html>
<html>
    <head>
        <style>
        h1 {text-align: center;}
        p {text-align: center;}
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
        for fi, fut in zip(urls, futs):
            fut.result()
            s += f'\n<p><a href="{fi[0]}">{fi[1]}</a></p>'
    s += f"""
        <p style="color:cyan;">Total file size: {byte_scale(sum(os.path.getsize(f[2]) for f in urls))}B</p>
        <p style="color:orange;">Estimated file lifetime: {sec2time(utc() - est_time)}</p>
        <img src="{flask.request.host_url}static/hug.gif" alt="Miza-Dottie-Hug" style="width:14.2857%;height:14.2857%;">
        <p><a href="/upload">Click here to upload another file!</a></p>
    </body>
</html>"""
    return s

@app.route("/upload", methods=["GET", "POST"])
def upload():
    global est_last
    ip = flask.request.remote_addr
    send(ip + "/upload\n")
    colour = hex(colour2raw(hue2colour(xrand(1536))))[2:].upper()
    if utc() - est_last > 1800:
        est_last = utc()
        create_future_ex(estimate_life)
    return f"""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Files</title>
        <meta content="Files" property="og:title">
        <meta content="Upload a file here!" property="og:description">
        <meta content="{flask.request.url}" property="og:url">
        <meta content="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif" property="og:image">
        <meta content="#""" + colour + """\" data-react-helmet="true" name="theme-color">
    </head>
    <style>
        body {
            background-image: url('""" + flask.request.host_url + """static/spiral.gif');
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
        }
    </style>
    <body>
        <form action="/upload_file" method="POST" enctype="multipart/form-data">
            <input style="color:white;" type="file" name="file" multiple/>
            <input type="submit"/>
        </form>
    </body>
</html>"""


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
        <meta content="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif" property="og:image">
        <meta content="#""" + colour + """\" data-react-helmet="true" name="theme-color">
        <meta http-equiv="refresh" content="60">
        <link rel="stylesheet" type="text/css" href="/static/timezonestyles.css" />
    </head>
    <body>
        <div>
        <h3>Estimated time:</h3>
        <h1>""" + str(dt) + """</h1>
        <h2>Detected timezone: """ + tz + f"""</h2>
        <p class="align_left">
            <a href="/time">Refresh</a>
        </p>
        <p class="align_right">
            <a href="/">Home</a>
        </p>
        </div>
    <img src="{flask.request.host_url}static/sky-rainbow.gif" alt="Miza-Sky" style="width:14.2857%;height:14.2857%;">
    </body>
</html>
        """
        return html
    except KeyError:
        send(traceback.format_exc())
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


cat_t = utc()
def get_cats():
    global cats
    with open("saves/imagepools/cats", "rb") as f:
        s = f.read()
    cats = select_and_loads(s)
    return cats
cats = get_cats()

@app.route("/cat", methods=["GET"])
@app.route("/cats", methods=["GET"])
def cat():
    global cats, cat_t
    if utc() - cat_t > 300:
        cat_t = utc()
        create_future_ex(get_cats)
    return flask.redirect(choice(cats))

dog_t = utc()
def get_dogs():
    global dogs
    with open("saves/imagepools/dogs", "rb") as f:
        s = f.read()
    dogs = select_and_loads(s)
    return dogs
dogs = get_dogs()

@app.route("/dog", methods=["GET"])
@app.route("/dogs", methods=["GET"])
def dog():
    global dogs, dog_t
    if utc() - dog_t > 300:
        create_future_ex(get_dogs)
    return flask.redirect(choice(dogs))


HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Server": "Miza",
    "Vary": "Accept-Encoding",
    "Accept-Ranges": "bytes",
    "Access-Control-Allow-Origin": "*",
}

CHEADERS = cdict(
    {"Cache-Control": "public, max-age=3600, stale-while-revalidate=1073741824, stale-if-error=1073741824"}
)

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