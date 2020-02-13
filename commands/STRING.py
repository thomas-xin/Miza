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
    "Papago": PapagoTrans(auth["papago_id"], auth["papago_secret"])}


class math:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["python"]
        self.min_level = 0
        self.description = "Evaluates a math formula using Python syntax."
        self.usage = "<function>"

    async def __call__(self, _vars, argv, channel, flags, **extra):
        tm = time.time()
        f = argv
        _vars.plt.clf()
        if not len(f):
            raise EOFError("Function is empty.")
        terr = self
        returns = [BaseException]
        doParallel(_vars.doMath, [f, returns])
        while returns[0] is BaseException and time.time() < tm + _vars.timeout / 2:
            await asyncio.sleep(0.1)
        if returns[0] == BaseException:
            raise TimeoutError("Request timed out.")
        _vars.updateGlobals()
        if _vars.fig.get_axes():
            fn = "cache/temp.png"
            _vars.plt.savefig(fn, bbox_inches="tight")
            f = discord.File(fn)
            return {"file": f}
        else:
            answer = returns[0]
            if answer is None:
                if "h" in flags:
                    return
                return "```python\n" + argv + " successfully executed!```"
            elif "\nError: " in answer:
                return "```python" + answer + "\n```", 1
            else:
                return "```python\n" + argv + " = " + str(answer) + "\n```"


class uni2hex:
    is_command = True

    def __init__(self):
        self.name = ["u2h"]
        self.min_level = 0
        self.description = "Converts unicode text to hexadecimal numbers."
        self.usage = "<string>"

    async def __call__(self, argv, **extra):
        if not argv:
            raise ValueError("Input string is empty.")
        b = bytes(argv, "utf-8")
        return "```fix\n" + bytes2Hex(b) + "```"


class hex2uni:
    is_command = True

    def __init__(self):
        self.name = ["h2u"]
        self.min_level = 0
        self.description = "Converts hexadecimal numbers to unicode text."
        self.usage = "<string>"

    async def __call__(self, argv, **extra):
        if not argv:
            raise ValueError("Input string is empty.")
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return "```fix\n" + b.decode("utf-8") + "```"


class translate:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["tr"]
        self.min_level = 0
        self.description = "Translates a string into another language."
        self.usage = "<0:language> <1:string> <verbose(?v)> <translator(?g)>"

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
            end = "\nDetected language: **" + str(source) + "**"
        else:
            count = 1
            end = ""
        response = "**" + user.name + "**:"
        #print(string, dest, source)
        for i in range(count):
            for t in trans:
                try:
                    dest = dest[:2] + dest[2:].upper()
                    output = translators[t].translate(string, dest, source)
                    output = output.text
                    response += "\n" + output + "  `" + t + "`"
                    source, dest = dest, source
                    break
                except Exception as ex:
                    print(ex)
                    if t == trans[-1]:
                        raise
        return response + end
