import discord, asyncio, ast, urllib, json
from googletrans import Translator
from smath import *


class PapagoTrans:
    class PapagoOutput:
        def __init__(self, text):
            self.text = text

    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec

    def translate(self, string, dest, source="en"):
        url = "https://openapi.naver.com/v1/papago/n2mt"
        enc = urllib.parse.quote(string)
        data = "source=" + source + "&target=" + dest + "&text=" + enc
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", self.id)
        req.add_header("X-Naver-Client-Secret", self.secret)
        print(req, url, data)
        resp = urllib.request.urlopen(req, data=data.encode("utf-8"))
        if resp.getcode() != 200:
            raise ConnectionError("Error " + str(resp.getcode()))
        read = resp.read().decode("utf-8")
        r = json.loads(read)
        t = r["message"]["result"]["translatedText"]
        output = self.PapagoOutput(t)
        return output


f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
translators = {
    "Google Translate": Translator(["translate.google.com"]),
    "Papago": PapagoTrans(auth["papago_id"], auth["papago_secret"]),
}


def _c2e(string, em1, em2):
    chars = {
        " ": [0, 0, 0, 0, 0],
        "_": [0, 0, 0, 0, 7],
        "!": [2, 2, 2, 0, 2],
        '"': [5, 5, 0, 0, 0],
        "#": [10, 31, 10, 31, 10],
        "$": [7, 10, 6, 5, 14],
        "?": [3, 4, 2, 0, 2],
        "%": [5, 1, 2, 4, 5],
        "&": [4, 10, 4, 10, 7],
        "'": [2, 2, 0, 0, 0],
        "(": [2, 4, 4, 4, 2],
        ")": [2, 1, 1, 1, 2],
        "[": [6, 4, 4, 4, 6],
        "]": [3, 1, 1, 1, 3],
        "|": [2, 2, 2, 2, 2],
        "*": [21, 14, 4, 14, 21],
        "+": [0, 2, 7, 2, 0],
        "=": [0, 7, 0, 7, 0],
        ",": [0, 0, 3, 3, 4],
        "-": [0, 0, 7, 0, 0],
        ".": [0, 0, 3, 3, 0],
        "/": [1, 1, 2, 4, 4],
        "\\": [4, 4, 2, 1, 1],
        "@": [14, 17, 17, 17, 14],
        "0": [7, 5, 5, 5, 7],
        "1": [3, 1, 1, 1, 1],
        "2": [7, 1, 7, 4, 7],
        "3": [7, 1, 7, 1, 7],
        "4": [5, 5, 7, 1, 1],
        "5": [7, 4, 7, 1, 7],
        "6": [7, 4, 7, 5, 7],
        "7": [7, 5, 1, 1, 1],
        "8": [7, 5, 7, 5, 7],
        "9": [7, 5, 7, 1, 7],
        "A": [2, 5, 7, 5, 5],
        "B": [6, 5, 7, 5, 6],
        "C": [3, 4, 4, 4, 3],
        "D": [6, 5, 5, 5, 6],
        "E": [7, 4, 7, 4, 7],
        "F": [7, 4, 7, 4, 4],
        "G": [7, 4, 5, 5, 7],
        "H": [5, 5, 7, 5, 5],
        "I": [7, 2, 2, 2, 7],
        "J": [7, 1, 1, 5, 7],
        "K": [5, 5, 6, 5, 5],
        "L": [4, 4, 4, 4, 7],
        "M": [17, 27, 21, 17, 17],
        "N": [9, 13, 15, 11, 9],
        "O": [2, 5, 5, 5, 2],
        "P": [7, 5, 7, 4, 4],
        "Q": [4, 10, 10, 10, 5],
        "R": [6, 5, 7, 6, 5],
        "S": [3, 4, 7, 1, 6],
        "T": [7, 2, 2, 2, 2],
        "U": [5, 5, 5, 5, 7],
        "V": [5, 5, 5, 5, 2],
        "W": [17, 17, 21, 21, 10],
        "X": [5, 5, 2, 5, 5],
        "Y": [5, 5, 2, 2, 2],
        "Z": [7, 1, 2, 4, 7],
    }
    printed = [""] * 7
    string = string.upper()
    for i in range(len(string)):
        curr = string[i]
        data = chars.get(curr, [15] * 5)
        size = max(1, max(data))
        lim = max(2, trunc(log(size, 2))) + 1
        printed[0] += em2 * (lim + 1)
        printed[6] += em2 * (lim + 1)
        if len(data) == 5:
            for y in range(5):
                printed[y + 1] += em2
                for p in range(lim):
                    if data[y] & (1 << (lim - 1 - p)):
                        printed[y + 1] += em1
                    else:
                        printed[y + 1] += em2
        for x in range(len(printed)):
            printed[x] += em2
    output = "\n".join(printed)
    print("[" + em1 + "]", "[" + em2 + "]")
    if len(em1) == len(em2) == 1:
        output = "```\n" + output + "```"
    return output


class math:
    is_command = True

    def __init__(self):
        self.name = ["python"]
        self.minm = 0
        self.desc = "Evaluates a math formula using Python syntax."
        self.usag = "<function>"

    async def __call__(self, _vars, argv, channel, **extra):
        tm = time.time()
        f = argv
        _vars.plt.clf()
        if not len(f):
            raise EOFError("Function is empty.")
        terr = TimeoutError
        returns = [terr]
        doParallel(_vars.doMath, [f, returns])
        while returns[0] == terr and time.time() < tm + _vars.timeout:
            time.sleep(0.1)
        if returns[0] == terr:
            raise TimeoutError("Request timed out.")
        _vars.updateGlobals()
        if _vars.fig.get_axes():
            fn = "cache/temp.png"
            _vars.plt.savefig(fn, bbox_inches="tight")
            f = discord.File(fn)
            await channel.send(file=f)
            return
        else:
            answer = returns[0]
            if answer is None:
                return "```\n" + argv + " successfully executed!```"
            elif "\nError" in answer:
                return "```" + answer + "\n```"
            else:
                return "```\n" + argv + " = " + str(answer) + "\n```"


class uni2hex:
    is_command = True

    def __init__(self):
        self.name = ["u2h"]
        self.minm = 0
        self.desc = "Converts unicode text to hexadecimal numbers."
        self.usag = "<string>"

    async def __call__(self, argv, **extra):
        b = bytes(argv, "utf-8")
        return "```\n" + bytes2Hex(b) + "```"


class hex2uni:
    is_command = True

    def __init__(self):
        self.name = ["h2u"]
        self.minm = 0
        self.desc = "Converts hexadecimal numbers to unicode text."
        self.usag = "<string>"

    async def __call__(self, argv, **extra):
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return "```\n" + b.decode("utf-8") + "```"


class char2emoj:
    is_command = True

    def __init__(self):
        self.name = ["c2e"]
        self.minm = 0
        self.desc = "Makes emoji blocks using a string."
        self.usag = "<string>"

    async def __call__(self, args, **extra):
        return _c2e(*args[:3])


class translate:
    is_command = True

    def __init__(self):
        self.name = ["tr"]
        self.minm = 0
        self.desc = "Translates a string into another language."
        self.usag = "<0:language> <1:string> <verbose:(?v)> <translator:(?g)>"

    async def __call__(self, args, flags, user, **extra):
        dest = args[0]
        string = " ".join(args[1:])
        detected = translators["Google Translate"].detect(string)
        source = detected.lang
        trans = ["Papago", "Google Translate"]
        if "g" in flags:
            trans = trans[::-1]
        if "v" in flags:
            count = 2
        else:
            count = 1
        response = "**" + user.name + "**:"
        print(string, dest, source)
        for i in range(count):
            for t in trans:
                try:
                    if "papago" in t:
                        if "-" in dest:
                            dest = dest[:-2] + dest[-2:].upper()
                    else:
                        dest = dest.lower()
                    output = translators[t].translate(string, dest, source)
                    output = output.text
                    print(output + "\n\n")
                    response += "\n" + output + "  `" + t + "`"
                    source, dest = dest, source
                    break
                except:
                    if t == trans[-1]:
                        raise
        return response
