import os, sys, flask, requests, datetime, pytz, traceback
from flask import Flask
from werkzeug.exceptions import HTTPException

PORT = 9801
IND = ""


app = Flask(__name__)
# app.use_x_sendfile = True


@app.errorhandler(Exception)
def on_error(ex):
    print(repr(ex))
    # Redirect HTTP errors to http.cat, python exceptions go to code 500 (internal server error)
    if issubclass(type(ex), HTTPException):
        return flask.redirect(f"https://http.cat/{ex.code}")
    return flask.redirect("https://http.cat/500")

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return flask.send_file("misc/icon.ico")

def find_file(path):
    # if no file name is inputted, return no content
    if not path:
        raise EOFError
    # do not include "." in the path name
    path = path.rsplit(".", 1)[0]
    fn = f"{IND}{path}"
    for file in os.listdir("cache"):
        # file cache is stored as "{timestamp}~{name}", search for file via timestamp
        if file.rsplit(".", 1)[0].split("~", 1)[0] == fn:
            out = "cache/" + file
            print(out)
            return out
    raise FileNotFoundError

@app.route("/files/<path>", methods=["GET"])
def get_file(path):
    print(flask.request.remote_addr, path)
    try:
        return flask.send_file(find_file(path), as_attachment=bool(flask.request.args.get("download")))
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")

@app.route("/files/<path>/<filename>", methods=["GET"])
def get_file_ex(path, filename):
    print(flask.request.remote_addr, path)
    try:
        return flask.send_file(find_file(path), as_attachment=bool(flask.request.args.get("download")), attachment_filename=filename)
    except EOFError:
        return flask.redirect("https://http.cat/204")
    except FileNotFoundError:
        return flask.redirect("https://http.cat/404")

@app.route("/", methods=["GET", "POST"])
def home():
    # basic endpoint for the port; return the request's remote (external) IP address
    return flask.request.remote_addr

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
        html = "\n".join((
            "<!DOCTYPE html>",
            "<html>",
            "<body>",
            f'<h1 style="text-align:center;">Estimated time: {dt}</h1>',
            f'<h2 style="text-align:center;">Detected timezone: {tz}</h2>',
            f'<p style="text-align:center;"><a href="http://{flask.request.host}/timezone">Refresh</a></p>',
            "</body>",
            "</html>",
        ))
        return html
    except KeyError:
        traceback.print_exc()
        return flask.redirect("https://http.cat/417")
    except:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    app.run("0.0.0.0", PORT)