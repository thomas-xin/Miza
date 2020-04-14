from googletrans import Translator
try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class PapagoTrans:
    class PapagoOutput:
        def __init__(self, text):
            self.text = text

    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec

    def translate(self, string, dest, source="en"):
        if dest == source:
            raise ValueError("Source language is the same as destination.")
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


translators = {
    "Google Translate": Translator(["translate.google.com"]),
}
f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    translators["Papago"] = PapagoTrans(auth["papago_id"], auth["papago_secret"])
except KeyError:
    translators["Papago"] = freeClass(
        translate=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to use Papago Translate.")'),
    )
    print("WARNING: papago_id/papago_secret not found. Unable to use Papago Translate.")
try:
    rapidapi_key = auth["rapidapi_key"]
except:
    rapidapi_key = None
    print("WARNING: rapidapi_key not found. Unable to search Urban Dictionary.")


def getTranslate(translator, string, dest, source):
    try:
        resp = translator.translate(string, dest, source)
        return resp
    except Exception as ex:
        print(traceback.format_exc())
        return ex


class Translate(Command):
    time_consuming = True
    name = ["TR"]
    min_level = 0
    description = "Translates a string into another language."
    usage = "<0:language> <1:string> <verbose(?v)> <papago(?p)>"
    flags = "pv"

    async def __call__(self, args, flags, user, **void):
        dest = args[0]
        string = " ".join(args[1:])
        detected = translators["Google Translate"].detect(string)
        source = detected.lang
        trans = ["Google Translate", "Papago"]
        if "p" in flags:
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


class Math(Command):
    time_consuming = True
    name = ["Python", "PY", "Sympy", "M", "Calc"]
    min_level = 0
    description = "Evaluates a math formula."
    usage = "<function> <verbose(?v)> <rationalize(?r)>"
    flags = "rv"

    async def __call__(self, _vars, argv, channel, flags, guild, **void):
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


class Uni2Hex(Command):
    name = ["U2H"]
    min_level = 0
    description = "Converts unicode text to hexadecimal numbers."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        b = bytes(argv, "utf-8")
        return "```fix\n" + bytes2Hex(b) + "```"


class Hex2Uni(Command):
    name = ["H2U"]
    min_level = 0
    description = "Converts hexadecimal numbers to unicode text."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return "```fix\n" + b.decode("utf-8") + "```"


class ID2Time(Command):
    name = ["I2T"]
    min_level = 0
    description = "Converts a discord ID to its corresponding UTC time."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        argv = verifyID(argv)
        return "```fix\n" + str(snowflake_time(argv)) + "```"


class Time2ID(Command):
    name = ["T2I"]
    min_level = 0
    description = "Converts a UTC time to its corresponding discord ID."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        argv = tparser.parse(argv)
        return "```fix\n" + str(time_snowflake(argv)) + "```"


class UniFmt(Command):
    name = ["Fancy", "FancyText"]
    min_level = 0
    description = "Creates a representation of a text string using unicode fonts."
    usage = "<0:font_id> <1:string>"

    async def __call__(self, args, guild, **void):
        if len(args) < 2:
            raise IndexError("Input string is empty.")
        i = await self._vars.evalMath(args[0], guild)
        return "```fix\n" + uniStr(" ".join(args[1:]), i) + "```"


class OwOify(Command):
    omap = {
        "n": "ny",
        "N": "NY",
        "r": "w",
        "R": "W",
        "l": "w",
        "L": "W",
    }
    otrans = "".maketrans(omap)
    name = ["OwO"]
    min_level = 0
    description = "owo-ifies text."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        return "```fix\n" + argv.translate(self.otrans) + "```"


class Time(Command):
    name = ["UTC", "GMT"]
    min_level = 0
    description = "Shows the current time in a certain timezone."
    usage = "<offset_hours[0]>"

    async def __call__(self, argv, guild, **void):
        if argv:
            h = await self._vars.evalMath(argv, guild)
        else:
            h = 0
        hrs = datetime.timedelta(hours=h)
        t = datetime.datetime.utcnow() + hrs
        s = str(h)
        if not s.startswith("-"):
            s = "+" + s
        return (
            "```ini\nCurrent time at UTC/GMT" + s 
            + ": [" + str(t) + "].```"
        )


class UrbanDictionary(Command):
    time_consuming = True
    header = {
	"x-rapidapi-host": "mashape-community-urban-dictionary.p.rapidapi.com",
	"x-rapidapi-key": rapidapi_key,
    }
    name = ["Urban"]
    min_level = 0
    description = "Searches Urban Dictionary for an item."
    usage = "<string> <verbose(?v)>"
    flags = "v"

    async def __call__(self, argv, flags, **void):
        url = (
            "https://mashape-community-urban-dictionary.p.rapidapi.com/define?term="
            + argv.replace(" ", "%20")
        )
        returns = [None]
        doParallel(
            funcSafe,
            [requests.get, url],
            returns,
            {"headers": self.header}
        )
        while returns[0] is None:
            await asyncio.sleep(0.4)
        if type(returns[0]) is str:
            resp = evalEX(returns[0])
        else:
            resp = returns[0][-1]
        s = resp.content
        resp.close()
        try:
            d = json.loads(s)
        except:
            d = eval(s, {}, infinum)
        l = d["list"]
        if not l:
            raise LookupError("No results for " + argv + ".")
        l.sort(
            key=lambda e: scaleRatio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
            reverse=True,
        )
        if "v" in flags:
            output = (
                "```ini\n[" + noHighlight(argv) + "]\n"
                + "\n".join(
                    "[" + str(i + 1) + "] " + l[i].get(
                        "definition",
                        "",
                    ).replace("\n", " ").replace("\r", "") for i in range(
                        min(len(l), 1 + 2 * flags["v"])
                    )
                )
                + "```"
            )
        else:
            output = (
                "```ini\n[" + noHighlight(argv) + "]\n"
                + l[0].get("definition", "") + "```"
            )
        return output