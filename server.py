from common import *
import flask
from flask import Flask
from werkzeug.exceptions import HTTPException

PORT = 9801
IND = ""


app = Flask(__name__)
# app.use_x_sendfile = True


@app.errorhandler(Exception)
def on_error(ex):
    sys.stderr.write(repr(ex) + "\n")
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

@app.route("/files/<path>", methods=["GET"])
def get_file(path):
    try:
        return flask.send_file(find_file(path), as_attachment=bool(flask.request.args.get("download")))
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")

@app.route("/files/<path>/<filename>", methods=["GET"])
def get_file_ex(path, filename):
    try:
        return flask.send_file(find_file(path), as_attachment=bool(flask.request.args.get("download")), attachment_filename=filename)
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")


MIMES = dict(
    css="text/css",
    json="application/json",
    js="application/javascript",
    ico="image/x-icon",
    png="image/png",
    jpg="image/jpeg",
    gif="image/gif",
    webp="image/webp",
    mp3="audio/mpeg",
    ogg="audio/vorbis",
    opus="audio/opus",
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
        sys.stderr.write(path + "\n")
        traceback.print_exc()
        raise

@app.route("/static/<path>", methods=["GET"])
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

timezones = {}
@app.route("/timezone", methods=["GET", "POST"])
def timezone():
    ip = flask.request.remote_addr
    try:
        try:
            resp = timezones[ip]
        except KeyError:
            url = f"https://tools.keycdn.com/geo.json?host={ip}"
            resp = requests.get(url, headers={"DNT": "1", "User-Agent": f"Mozilla/5.{ip[-1]}"}).json()
            timezones[resp["data"]["geo"]["ip"]] = resp
        data = resp["data"]["geo"]
        tz = data["timezone"]
        dt = datetime.datetime.now(pytz.timezone(tz))
        sys.stderr.write(ip + "\t" + str(dt) + "\t" + tz + "\n")
        html = """<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <title>timezone page</title>  
    <link rel=\"stylesheet\" type=\"text/css\" href=\"/static/timezonestyles.css\" />
  </head>
  <body>
    <div>
      <h3>Estimated time:</h3>
      <h1>""" + str(dt) + """</h1>
      <h2>Detected timezone: """ + tz + """</h2>
      <p>
        <a href=\"/timezone\">Refresh</a>
      </p>
    </div>
  </body>
</html>
        """
        return html
    except KeyError:
        traceback.print_exc()
        return flask.redirect("https://http.cat/417")
    except:
        traceback.print_exc()
        raise


cat_t = utc()
def get_cats():
    global cats
    with open("saves/imagepools/cats", "rb") as f:
        s = f.read()
    cats = select_and_loads(s)
    return cats
cats = get_cats()

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

@app.route("/dogs", methods=["GET"])
def dog():
    global dogs, dog_t
    if utc() - dog_t > 300:
        create_future_ex(get_dogs)
    return flask.redirect(choice(dogs))


if __name__ == "__main__":
    app.run("0.0.0.0", PORT)
