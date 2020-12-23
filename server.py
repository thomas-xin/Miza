from common import *
import flask
from flask import Flask
from werkzeug.exceptions import HTTPException

PORT = 9801
IND = ""


sys.stderr = sys.stdout

app = Flask(__name__)
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
    if issubclass(type(ex), TimeoutError):
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

@app.route("/file/<path>/<filename>", methods=["GET"])
@app.route("/files/<path>/<filename>", methods=["GET"])
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
    ogg="audio/vorbis",
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

@app.route("/static/<string:path>", methods=["GET"])
def get_static_file(path):
    try:
        data, mime = fetch_static(path)
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
    <meta charset=\"utf-8\">
    <title>Timezones</title>
    <meta content="Timezones" property="og:title">
    <meta content="Find your current timezone here!" property="og:description">
    <meta content=\"""" + flask.request.url + """\" property="og:url">
    <meta content="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/b9573a17-63e8-4ec1-9c97-2bd9a1e9b515/de1q8ky-037a9e0a-debc-47a3-bb3c-217715f13f50.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3sicGF0aCI6IlwvZlwvYjk1NzNhMTctNjNlOC00ZWMxLTljOTctMmJkOWExZTliNTE1XC9kZTFxOGt5LTAzN2E5ZTBhLWRlYmMtNDdhMy1iYjNjLTIxNzcxNWYxM2Y1MC5wbmcifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.lHV-a85lkq2WLXt5-ZXR5Gai9LBgF7-lhJCeRCMOZmI" property="og:image">
    <meta content="#""" + colour + """\" data-react-helmet="true" name="theme-color">
    <link rel=\"stylesheet\" type=\"text/css\" href=\"/static/timezonestyles.css\" />
  </head>
  <body>
    <div>
      <h3>Estimated time:</h3>
      <h1>""" + str(dt) + """</h1>
      <h2>Detected timezone: """ + tz + """</h2>
      <p>
        <a href=\"/time\">Refresh</a>
      </p>
      <p>
        <a href=\"/\">Home</a>
      </p>
    </div>
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

@app.route("/command/<string:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
@app.route("/commands/<string:content>", methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"])
def command(content):
    ip = flask.request.remote_addr
    if ip == "127.0.0.1":
        t = int(content)
        j = flask.request.get_json(force=True)
        while len(RESPONSES) >= 256:
            RESPONSES.pop(next(iter(RESPONSES)), None)
        RESPONSES[t] = j
        sys.__stderr__.write("\x00" + str(j) + "\n")
        return b"\xf0\x9f\x92\x9c"
    resp = get_geo(ip)
    data = resp["data"]["geo"]
    tz = data["timezone"]
    t = int(utc() * 1000)
    sys.__stderr__.write(f"~{t}\x7f{ip}\x7f{tz}\x7f{content}\n")
    for i in range(360):
        if t in RESPONSES:
            return flask.Response(json.dumps(RESPONSES.pop(t)), mimetype="application/json")
        time.sleep(0.1)
    raise TimeoutError


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


if __name__ == "__main__":
    app.run("0.0.0.0", PORT)
