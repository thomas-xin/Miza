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

get_mime = lambda path: magic.from_file(path, mime=True)

@app.route("/file/<path>", methods=["GET"])
@app.route("/files/<path>", methods=["GET"])
def get_file(path):
    p = find_file(path)
    down = flask.request.args.get("download", "false")
    download = down and down[0] not in "0fFnN"
    if download:
        mime = MIMES.get(p.rsplit("/", 1)[-1].rsplit(".", 1)[-1])
    else:
        mime = get_mime(p)
    send(p, mime)
    return flask.send_file(p, as_attachment=download, mimetype=mime)

@app.route("/file/<path>/<path:filename>", methods=["GET"])
@app.route("/files/<path>/<path:filename>", methods=["GET"])
def get_file_ex(path, filename):
    p = find_file(path)
    down = flask.request.args.get("download", "false")
    download = down and down[0] not in "0fFnN"
    if download:
        mime = MIMES.get(p.rsplit("/", 1)[-1].rsplit(".", 1)[-1])
    else:
        mime = get_mime(p)
    send(p, mime)
    return flask.send_file(p, as_attachment=download, attachment_filename=filename, mimetype=mime)


MIMES = dict(
    css="text/css",
    json="application/json",
    js="application/javascript",
    txt="text/plain",
    html="text/html",
    ico="image/x-icon",
    png="image/png",
    jpg="image/jpeg",
    gif="image/gif",
    webp="image/webp",
    mp3="audio/mpeg",
    ogg="audio/ogg",
    opus="audio/opus",
    flac="audio/flac",
    wav="audio/x-wav",
    mp4="video/mp4",
)

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
    return flask.Response(data, mimetype=mime)

@app.route("/static", methods=["DELETE"])
def clearcache():
    ip = flask.request.remote_addr
    if ip == "127.0.0.1":
        STATIC.clear()
        send("Webserver cache cleared.")
        return b"\xf0\x9f\x92\x9c"
    raise PermissionError


@app.route("/mizatlas/<path:filename>", methods=["GET"])
def atlas(filename):
    # if filename == "run":
    #     content = flask.request.args.get("command", "")
    #     ip = flask.request.remote_addr
    #     resp = get_geo(ip)
    #     data = resp["data"]["geo"]
    #     tz = data["timezone"]
    #     t = ts_us()
    #     if " " not in content:
    #         content += " "
    #     RESPONSES[t] = fut = concurrent.futures.Future()
    #     sys.__stderr__.write(f"~{t}\x7f{ip}\x7f{tz}\x7f{content}\n")
    #     j, after = fut.result(timeout=420)
    #     RESPONSES.pop(t, None)
    #     response = flask.Response(json.dumps(j), mimetype="application/json")
    #     a = after - utc()
    #     if a > 0:
    #         response.headers["Retry-After"] = a
    #     return response
    try:
        data, mime = fetch_static("mizatlas/" + filename)
    except FileNotFoundError:
        data, mime = fetch_static("mizatlas/index.html")
    send("mizatlas/" + filename, mime)
    return flask.Response(data, mimetype=mime)

@app.route("/mizatlas", methods=["GET"])
def mizatlas():
    data, mime = fetch_static("mizatlas/index.html")
    send("mizatlas/index.html", mime)
    return flask.Response(data, mimetype=mime)
#     return f"""<!DOCTYPE html>
# <html lang="en">
#     <head>
#         <meta charset="utf-8"/><link rel="icon" href="/mizatlas/favicon.ico"/>
#         <meta content="MizAtlas" property="og:title">
#         <meta content="A Miza [command tester]({flask.request.url}/run) and [atlas]({flask.request.url}/atlas)!" property="og:description">
#         <meta content="{flask.request.url}" property="og:url">
#         <meta content="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif" property="og:image">
#         <meta name="viewport" content="width=device-width,initial-scale=1"/>
#         <meta content="bf7fff" data-react-helmet="true" name="theme-color">
#         <meta name="description" content="MizAtlas"/><link rel="apple-touch-icon" href="/mizatlas/logo192.png"/>
#         <link rel="stylesheet" href="/mizatlas/styles.css"/><link rel="manifest" href="/mizatlas/manifest.json"/>
#         <title>MizAtlas</title>
#     </head>
#     <body>
#         <noscript>You need to enable JavaScript to run this app.</noscript>
#         <div id="root"></div><script>
#             """ + """!function(e){function t(t){for(var n,a,i=t[0],c=t[1],l=t[2],f=0,p=[];f<i.length;f++)a=i[f],Object.prototype.hasOwnProperty.call(o,a)&&o[a]&&p.push(o[a][0]),o[a]=0;for(n in c)Object.prototype.hasOwnProperty.call(c,n)&&(e[n]=c[n]);for(s&&s(t);p.length;)p.shift()();return u.push.apply(u,l||[]),r()}function r(){for(var e,t=0;t<u.length;t++){for(var r=u[t],n=!0,i=1;i<r.length;i++){var c=r[i];0!==o[c]&&(n=!1)}n&&(u.splice(t--,1),e=a(a.s=r[0]))}return e}var n={},o={1:0},u=[];function a(t){if(n[t])return n[t].exports;var r=n[t]={i:t,l:!1,exports:{}};return e[t].call(r.exports,r,r.exports,a),r.l=!0,r.exports}a.e=function(e){var t=[],r=o[e];if(0!==r)if(r)t.push(r[2]);else{var n=new Promise((function(t,n){r=o[e]=[t,n]}));t.push(r[2]=n);var u,i=document.createElement("script");i.charset="utf-8",i.timeout=120,a.nc&&i.setAttribute("nonce",a.nc),i.src=function(e){return a.p+"static/js/"+({}[e]||e)+"."+{3:"c65330d6"}[e]+".chunk.js"}(e);var c=new Error;u=function(t){i.onerror=i.onload=null,clearTimeout(l);var r=o[e];if(0!==r){if(r){var n=t&&("load"===t.type?"missing":t.type),u=t&&t.target&&t.target.src;c.message="Loading chunk "+e+" failed.\n("+n+": "+u+")",c.name="ChunkLoadError",c.type=n,c.request=u,r[1](c)}o[e]=void 0}};var l=setTimeout((function(){u({type:"timeout",target:i})}),12e4);i.onerror=i.onload=u,document.head.appendChild(i)}return Promise.all(t)},a.m=e,a.c=n,a.d=function(e,t,r){a.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:r})},a.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},a.t=function(e,t){if(1&t&&(e=a(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var r=Object.create(null);if(a.r(r),Object.defineProperty(r,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var n in e)a.d(r,n,function(t){return e[t]}.bind(null,n));return r},a.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return a.d(t,"a",t),t},a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},a.p="/mizatlas/",a.oe=function(e){throw console.error(e),e};var i=this.webpackJsonpmizatlas=this.webpackJsonpmizatlas||[],c=i.push.bind(i);i.push=t,i=i.slice();for(var l=0;l<i.length;l++)t(i[l]);var s=c;r()}([])
#         </script>
#         <script src="/mizatlas/static/js/2.baca3994.chunk.js"></script>
#         <script src="/mizatlas/static/js/main.0d3f8e6b.chunk.js"></script>
#     </body>
# </html>"""


@app.route("/", methods=["GET", "POST"])
def home():
    data, mime = fetch_static("index.html")
    send("index.html", mime)
    return flask.Response(data, mimetype=mime)

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    data, mime = fetch_static("icon.ico")
    send("icon.ico", mime)
    return flask.Response(data, mimetype=mime)


@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    ip = flask.request.remote_addr
    f = flask.request.files["file"]
    ts = time.time_ns() // 1000
    fn = f.filename
    f.save(f"cache/{IND}{ts}~{fn}")
    href = f"/files/{ts}/{fn}"
    url = f"{flask.request.host}{href}"
    send(ip + "\t" + fn + "\t" + url)
    return """<!DOCTYPE html>
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
        <h1 style="color:white;">File uploaded successfully!</h1>
        <p><a href=\"""" + href + f"""\">{url}</a></p>
        <img src="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/hug.gif" alt="Miza-Dottie-Hug" style="width:14.2857%;height:14.2857%;">
        <p><a href="/upload">Click here to upload another file!</a></p>
    </body>
</html>"""

@app.route("/upload", methods=["GET", "POST"])
def upload():
    ip = flask.request.remote_addr
    send(ip + "/upload\n")
    colour = hex(colour2raw(hue2colour(xrand(1536))))[2:].upper()
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
            background-image: url('https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/spiral.gif');
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
        }
    </style>
    <body>
        <form action="/upload_file" method="POST" enctype="multipart/form-data">
            <input style="color:white;" type="file" name="file" />
            <input type="submit"/>
        </form>
    </body>
</html>"""


def get_geo(ip):
    try:
        resp = TZCACHE[ip]
    except KeyError:
        url = f"https://tools.keycdn.com/geo.json?host={ip}"
        resp = requests.get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}"}).json()
        TZCACHE[resp["data"]["geo"]["ip"]] = resp
    return resp

@app.route("/time", methods=["GET", "POST"])
@app.route("/timezone", methods=["GET", "POST"])
@app.route("/timezones", methods=["GET", "POST"])
def timezone():
    ip = flask.request.remote_addr
    try:
        resp = get_geo(ip)
        data = resp["data"]["geo"]
        tz = data["timezone"]
        dt = datetime.datetime.now(pytz.timezone(tz))
        send(ip + "\t" + str(dt) + "\t" + tz)
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
        <link rel="stylesheet" type="text/css" href="/static/timezonestyles.css" />
    </head>
    <body>
        <div>
        <h3>Estimated time:</h3>
        <h1>""" + str(dt) + """</h1>
        <h2>Detected timezone: """ + tz + """</h2>
        <p class="align_left">
            <a href="/time">Refresh</a>
        </p>
        <p class="align_right">
            <a href="/">Home</a>
        </p>
        </div>
    <img src="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif" alt="Miza-Sky" style="width:14.2857%;height:14.2857%;">
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
    resp = get_geo(ip)
    data = resp["data"]["geo"]
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

@app.after_request
def custom_header(response):
    response.headers.update(HEADERS)
    return response


def ensure_parent(proc, parent):
    while True:
        if not parent.is_running():
            proc.kill()
        t = ts_us()
        while t in RESPONSES:
            t += 1
        RESPONSES[t] = fut = concurrent.futures.Future()
        send(f"!{t}\x7f{content}", escape=False)
        j, after = fut.result()
        RESPONSES.pop(t, None)
        send(f"!{t}\x7fGC.__setitem__({proc.pid}, {len(gc.get_objects())})", escape=False)
        time.sleep(6)

if __name__ == "__main__":
    pid = os.getpid()
    ppid = os.getppid()
    send(f"Webserver starting on port {PORT}, with PID {pid} and parent PID {ppid}...")
    proc = psutil.Process(pid)
    parent = psutil.Process(ppid)
    create_thread(ensure_parent, proc, parent)
    app.run("0.0.0.0", PORT)