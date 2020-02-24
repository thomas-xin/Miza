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
    "Papago": PapagoTrans(auth["papago_id"], auth["papago_secret"])
}


class math:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["python", "py", "sympy", "m"]
        self.min_level = 0
        self.description = "Evaluates a math formula."
        self.usage = "<function> <verbose(?v)> <rationalize(?r)>"

    async def __call__(self, _vars, argv, channel, flags, guild, **void):
        tm = time.time()
        f = argv
        if not len(f):
            raise IndexError("Function is empty.")
        r = "r" in flags
        p = flags.get("v", 0) * 2 + 1 << 6
        resp = await _vars.solveMath(f, guild, p, r)
        if type(resp) is dict and "file" in resp:
            f = discord.File(resp["file"])
            return {"file": f}
        return "```py\n" + str(f) + " = " + "\n".join(str(i) for i in resp) + "```"


class uni2hex:
    is_command = True

    def __init__(self):
        self.name = ["u2h"]
        self.min_level = 0
        self.description = "Converts unicode text to hexadecimal numbers."
        self.usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        b = bytes(argv, "utf-8")
        return "```fix\n" + bytes2Hex(b) + "```"


class hex2uni:
    is_command = True

    def __init__(self):
        self.name = ["h2u"]
        self.min_level = 0
        self.description = "Converts hexadecimal numbers to unicode text."
        self.usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return "```fix\n" + b.decode("utf-8") + "```"


class uniFmt:
    is_command = True

    def __init__(self):
        self.name = ["fancy", "fancyText"]
        self.min_level = 0
        self.description = "Creates a representation of a text string using unicode fonts."
        self.usage = "<0:font_id> <1:string>"

    async def __call__(self, args, guild, **void):
        if len(args) < 2:
            raise IndexError("Input string is empty.")
        i = await self._vars.evalMath(args[0], guild)
        return "```fix\n" + uniStr(" ".join(args[1:]), i) + "```"


def getTranslate(translator, string, dest, source):
    try:
        resp = translator.translate(string, dest, source)
        return resp
    except Exception as ex:
        print(traceback.format_exc())
        return ex


class translate:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["tr"]
        self.min_level = 0
        self.description = "Translates a string into another language."
        self.usage = "<0:language> <1:string> <verbose(?v)> <google(?g)>"

    async def __call__(self, args, flags, user, **void):
        dest = args[0]
        string = " ".join(args[1:])
        detected = translators["Google Translate"].detect(string)
        source = detected.lang
        trans = ["Papago", "Google Translate"]
        if "g" in flags:
            trans = trans[::-1]
        if "v" in flags:
            count = 2
            end = "\nDetected language: **" + str(source) + "**"
        else:
            count = 1
            end = ""
        response = "**" + user.name + "**:"
        print(string, dest, source)
        for i in range(count):
            for t in trans:
                try:
                    dest = dest[:2] + dest[2:].upper()
                    returns = [None]
                    doParallel(getTranslate, [translators[t], string, dest, source], returns)
                    while returns[0] is None:
                        await asyncio.sleep(0.5)
                    output = returns[0]
                    ex = issubclass(output.__class__, Exception)
                    try:
                        ex = issubclass(output, Exception)
                    except TypeError:
                        pass
                    if ex:
                        raise output
                    output = output.text
                    response += "\n" + output + "  `" + t + "`"
                    source, dest = dest, source
                    break
                except:
                    if t == trans[-1] and i == count - 1:
                        raise
        return response + end
