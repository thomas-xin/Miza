import os, flask
from flask import Flask
from werkzeug.exceptions import HTTPException

PORT = 9801
IND = ""


app = Flask(__name__)

@app.errorhandler(Exception)
def on_error(ex):
    print(repr(ex))
    if issubclass(type(ex), HTTPException):
        return flask.redirect(f"https://http.cat/{ex.code}")
    return flask.redirect("https://http.cat/500")

@app.route("/files/<path>", methods=["GET"])
def get_file(path):
    if not path:
        return flask.redirect("https://http.cat/204")
    path = path.rsplit(".", 1)[0]
    print(flask.request.remote_addr, path)
    fn = f"{IND}{path}"
    print(fn)
    for file in os.listdir("cache"):
        if file.split("~", 1)[0] == fn:
            return flask.send_file("cache/" + file, as_attachment=False, attachment_filename=file.split("~", 1)[-1])
    return flask.redirect("https://http.cat/404")

@app.route("/", methods=["GET", "POST"])
def get_ip():
    return flask.request.remote_addr

app.run("0.0.0.0", PORT)