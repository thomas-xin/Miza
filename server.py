from common import *
import flask
from flask import Flask
from werkzeug.exceptions import HTTPException

PORT = 9801
IND = ""


sys.stderr = sys.stdout

app = Flask(__name__, static_url_path="/static")
app.url_map.strict_slashes = False
# app.use_x_sendfile = True


@app.errorhandler(Exception)
def on_error(ex):
    sys.__stderr__.write("\x00" + repr(ex) + "\n")
    # Redirect HTTP errors to http.cat, python exceptions go to code 500 (internal server error)
    if issubclass(type(ex), HTTPException):
        return flask.redirect(f"https://http.cat/{ex.code}")
    if issubclass(type(ex), FileNotFoundError):
        return flask.redirect("https://http.cat/404")
    if issubclass(type(ex), TimeoutError) or issubclass(type(ex), concurrent.futures.TimeoutError):
        return flask.redirect("https://http.cat/504")
    if issubclass(type(ex), ConnectionError):
        return flask.redirect("https://http.cat/502")
    return flask.redirect("https://http.cat/500")


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
            out = "cache/" + file
            return out
    raise FileNotFoundError

@app.route("/file/<path>", methods=["GET"])
@app.route("/files/<path>", methods=["GET"])
def get_file(path):
    try:
        mime = MIMES.get(path.rsplit(".", 1)[-1])
        down = flask.request.args.get("download", "false")
        download = down and down[0] not in "0fFnN"
        return flask.send_file(find_file(path), as_attachment=download, mimetype=mime)
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")

@app.route("/file/<path>/<path:filename>", methods=["GET"])
@app.route("/files/<path>/<path:filename>", methods=["GET"])
def get_file_ex(path, filename):
    try:
        mime = MIMES.get(filename.rsplit(".", 1)[-1])
        down = flask.request.args.get("download", "false")
        download = down and down[0] not in "0fFnN"
        return flask.send_file(find_file(path), as_attachment=download, attachment_filename=filename, mimetype=mime)
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")


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
    wav="audio/wav",
    mp4="video/mp4",
)

STATIC = {}

def fetch_static(path):
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
        sys.__stderr__.write("\x00" + path + "\n\x00" + traceback.format_exc())
        raise

@app.route("/static/<filename>", methods=["GET"])
@app.route("/static/<path:filename>", methods=["GET"])
def get_static_file(filename):
    try:
        data, mime = fetch_static(filename)
        return flask.Response(data, mimetype=mime)
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")

@app.route("/", methods=["GET", "POST"])
def home():
    data, mime = fetch_static("index.html")
    return flask.Response(data, mimetype=mime)

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    data, mime = fetch_static("icon.ico")
    return flask.Response(data, mimetype=mime)


TZCACHE = {}

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
        sys.__stderr__.write(ip + "\t" + str(dt) + "\t" + tz + "\n")
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
  <style>
  img {
    margin-top: 32px;
    border: 10px solid transparent;
    padding: 0px;
    border-image: url('https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/border.gif') 20% round;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  </style>
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
  <img src="https://raw.githubusercontent.com/thomas-xin/Miza/master/misc/sky-rainbow.gif" alt="Miza-Sky" style="width:25%;height:25%;">
  </body>
</html>
        """
        return html
    except KeyError:
        sys.__stderr__.write("\x00" + traceback.format_exc())
        return flask.redirect("https://http.cat/417")
    except:
        sys.__stderr__.write("\x00" + traceback.format_exc())
        raise


RESPONSES = {}

@app.route("/command/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
@app.route("/commands/<path:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
def command(content):
    ip = flask.request.remote_addr
    if ip == "127.0.0.1":
        t, after = content.split("\x7f", 1)
        t = int(t)
        after = float(after)
        j = flask.request.get_json(force=True)
        RESPONSES[t].set_result((j, after))
        sys.__stderr__.write("\x00" + str(j) + "\n")
        return b"\xf0\x9f\x92\x9c"
    resp = get_geo(ip)
    data = resp["data"]["geo"]
    tz = data["timezone"]
    t = int(utc() * 1000)
    if " " not in content:
        content += " "
    RESPONSES[t] = fut = concurrent.futures.Future()
    sys.__stderr__.write(f"~{t}\x7f{ip}\x7f{tz}\x7f{content}\n")
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


if __name__ == "__main__":
    app.run("0.0.0.0", PORT)
