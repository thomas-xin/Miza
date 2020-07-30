try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

from googletrans import Translator


# This is a bit of a mess
class PapagoTrans:

    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec

    def translate(self, string, dest, source="en"):
        if dest == source:
            raise ValueError("Source language is the same as destination.")
        url = "https://openapi.naver.com/v1/papago/n2mt"
        enc = verifyURL(string)
        url += "?source=" + source + "&target=" + dest + "&text=" + enc
        headers = {
            "X-Naver-Client-Id": self.id,
            "X-Naver-Client-Secret": self.secret,
        }
        print(url, headers)
        resp = Request(url, headers=headers, timeout=16)
        r = json.loads(resp)
        t = r["message"]["result"]["translatedText"]
        output = cdict(text=t)
        return output


translators = {"Google Translate": Translator(["translate.google.com"])}
with open("auth.json") as f:
    auth = ast.literal_eval(f.read())
try:
    translators["Papago"] = PapagoTrans(auth["papago_id"], auth["papago_secret"])
except KeyError:
    translators["Papago"] = cdict(
        translate=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to use Papago Translate.")'),
    )
    print("WARNING: papago_id/papago_secret not found. Unable to use Papago Translate.")
try:
    rapidapi_key = auth["rapidapi_key"]
    if not rapidapi_key:
        raise
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
    no_parse = True
    rate_limit = 2

    async def __call__(self, channel, args, flags, user, **void):
        if not args:
            raise ArgumentError("Input string is empty.")
        dest = args[0]
        string = " ".join(args[1:])
        fut = create_task(channel.trigger_typing())
        detected = await create_future(translators["Google Translate"].detect, string)
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
        response = "**" + str(user) + "**:"
        print(string, dest, source)
        # Attempt to use all available translators if possible
        for i in range(count):
            for t in trans:
                try:
                    resp = await create_future(getTranslate, translators[t], string, dest, source)
                    try:
                        output = resp.text
                    except AttributeError:
                        output = str(resp)
                    response += "\n" + output + "  `" + t + "`"
                    source, dest = dest, source
                    break
                except:
                    if t == trans[-1] and i == count - 1:
                        raise
        await fut
        return response + end    


class Math(Command):
    _timeout_ = 4
    name = ["PY", "Sympy", "M", "Calc"]
    min_level = 0
    description = "Evaluates a math formula."
    usage = "<function> <verbose(?v)> <rationalize(?r)>"
    flags = "rv"
    rate_limit = 0.5
    typing = True

    async def __call__(self, bot, argv, channel, flags, user, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        r = "r" in flags
        p = flags.get("v", 0) * 2 + 1 << 6
        resp = await bot.solveMath(argv, user, p, r, timeout=24)
        # Determine whether output is a direct answer or a file
        if type(resp) is dict and "file" in resp:
            await channel.trigger_typing()
            fn = resp["file"]
            f = discord.File(fn)
            await sendFile(channel, "", f, filename=fn, best=True)
            return
        return "```py\n" + str(argv) + " = " + "\n".join(str(i) for i in resp) + "```"


class Uni2Hex(Command):
    name = ["U2H", "HexEncode"]
    min_level = 0
    description = "Converts unicode text to hexadecimal numbers."
    usage = "<string>"
    no_parse = True

    async def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        b = bytes(argv, "utf-8")
        return "```fix\n" + bytes2Hex(b) + "```"


class Hex2Uni(Command):
    name = ["H2U", "HexDecode"]
    min_level = 0
    description = "Converts hexadecimal numbers to unicode text."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return "```fix\n" + b.decode("utf-8", "replace") + "```"


class ID2Time(Command):
    name = ["I2T", "CreateTime", "Timestamp"]
    min_level = 0
    description = "Converts a discord ID to its corresponding UTC time."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = verifyID(argv)
        return "```fix\n" + str(snowflake_time(argv)) + "```"


class Time2ID(Command):
    name = ["T2I", "RTimestamp"]
    min_level = 0
    description = "Converts a UTC time to its corresponding discord ID."
    usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = tzparse(argv)
        return "```fix\n" + str(time_snowflake(argv)) + "```"


class Fancy(Command):
    name = ["FancyText"]
    min_level = 0
    description = "Creates translations of a string using unicode fonts."
    usage = "<string>"
    no_parse = True

    async def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        emb = discord.Embed(colour=randColour())
        emb.set_author(name=argv)
        for i in range(len(UNIFMTS) - 1):
            s = uniStr(argv, i)
            if i == len(UNIFMTS) - 2:
                s = s[::-1]
            emb.add_field(name="Font " + str(i + 1), value="```" + "fix" * (i & 1) + "\n" + s + "```")
        # Only return embed if it can be sent
        if len(emb) > 6000:
            return "\n\n".join(f.name + "\n" + noCodeBox(f.value) for f in emb.fields)
        return dict(embed=emb)


class Zalgo(Command):
    name = ["ZalgoText"]
    min_level = 0
    description = "Generates random combining accent symbols between characters in a string."
    usage = "<string>"
    no_parse = True
    # This is a bit unintuitive with the character IDs
    nums = numpy.concatenate([numpy.arange(11) + 7616, numpy.arange(4) + 65056, numpy.arange(112) + 768])
    chrs = [chr(n) for n in nums]
    randz = lambda self: random.choice(self.chrs)
    zalgo = lambda self, s, x: "".join("".join(self.randz() + "\u200b" for i in range(x + 1 >> 1)) + c + "\u200a" + "".join(self.randz() + "\u200b" for i in range(x >> 1)) for c in s)

    async def __call__(self, channel, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        emb = discord.Embed(colour=randColour())
        emb.set_author(name=argv)
        for i in (1, 2, 3, 4, 5, 6, 7, 8):
            emb.add_field(name="Level " + str(i), value="```" + "fix" * (i & 1) + "\n" + self.zalgo(argv, i) + "```")
        # Discord often removes combining characters past a limit, so messages longer than 6000 characters may be sent, test this by attempting to send
        try:
            await channel.send(embed=emb)
        except discord.HTTPException:
            return "\n\n".join(f.name + "\n" + noCodeBox(f.value) for f in emb.fields)


class OwOify(Command):
    omap = {
        "r": "w",
        "R": "W",
        "l": "w",
        "L": "W",
    }
    otrans = "".maketrans(omap)
    name = ["UwU", "OwO", "UWUify"]
    min_level = 0
    description = "Applies the owo/uwu text filter to a string."
    usage = "<string> <aggressive(?a)> <basic(?b)>"
    flags = "ab"
    no_parse = True

    async def __call__(self, argv, flags, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        out = argv.translate(self.otrans)
        temp = None
        if "a" in flags or "b" not in flags:
            temp = list(out)
            for i, c in enumerate(out):
                if i > 0 and c in "yY" and out[i - 1] not in "wW \n\t":
                    if c.isupper():
                        temp[i] = "W" + c.lower()
                    else:
                        temp[i] = "w" + c
                if i < len(out) - 1 and c in "nN" and out[i + 1] not in "yY \n\t":
                    temp[i] = c + "y"
            if "a" in flags and "b" not in flags:
                for i, c in enumerate(out):
                    if i > 0 and c in "aeiouAEIOU" and out[i - 1] not in "wW \n\t":
                        if c.isupper():
                            temp[i] = "W" + c.lower()
                        else:
                            temp[i] = "w" + c
        if temp is not None:
            out = "".join(temp)
            if "a" in flags:
                for c in " \n\t":
                    if c in out:
                        spl = out.split(c)
                        for i, w in enumerate(spl):
                            if w.lower().startswith("th"):
                                spl[i] = "D" if w[0].isupper() else "d" + w[2:]
                            elif "th" in w:
                                spl[i] = w.replace("th", "ff")
                        out = c.join(spl)
        return "```fix\n" + out + "```"


class AltCaps(Command):
    min_level = 0
    description = "Alternates the capitalization on characters in a string."
    usage = "<string>"
    no_parse = True

    async def __call__(self, argv, **void):
        a = argv[::2].lower()
        b = argv[1::2].upper()
        if len(a) > len(b):
            c = a[-1]
            a = a[:-1]
        else:
            c = ""
        if argv[0].isupper():
            a, b = b, a
        return "".join(i[0] + i[1] for i in zip(a, b)) + c


class Time(Command):
    name = ["UTC", "GMT", "T"]
    min_level = 0
    description = "Shows the current time in a certain timezone."
    usage = "<offset_hours[0]>"

    async def __call__(self, name, argv, args, user, **void):
        s = 0
        # Only check for timezones if the command was called with alias "t" or "time"
        if args and name in "time":
            for a in (args[0], args[-1]):
                tz = a.lower()
                if tz in TIMEZONES:
                    s = get_timezone(tz)
                    argv = argv.replace(a, "")
                    break
        elif name in TIMEZONES:
            s = TIMEZONES.get(name, 0)
        t = utc_dt()
        if argv:
            h = await self.bot.evalMath(argv, user)
        else:
            h = 0
        if h or s:
            t += datetime.timedelta(hours=h, seconds=s)
        hrs = roundMin(h + s / 3600)
        if hrs >= 0:
            hrs = "+" + str(hrs)
        else:
            hrs = str(hrs)
        return (
            "```ini\nCurrent time at UTC/GMT" + hrs
            + ": [" + str(t) + "].```"
        )


class Follow(Command):
    name = ["FollowURL", "Redirect"]
    min_level = 0
    description = "Follows a discord message link and/or finds URLs in a string."
    rate_limit = 0.125
    
    async def __call__(self, argv, **void):
        urls = await self.bot.followURL(argv, allow=True)
        if not urls:
            raise FileNotFoundError("No valid URLs detected.")
        return "\n".join(urls)


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
    rate_limit = 2
    typing = True

    async def __call__(self, channel, argv, flags, **void):
        url = (
            "https://mashape-community-urban-dictionary.p.rapidapi.com/define?term="
            + argv.replace(" ", "%20")
        )
        fut = create_task(channel.trigger_typing())
        s = await Request(url, headers=self.header, timeout=16, aio=True)
        # eval is often better at json decoding than json.loads for some reason, this usage isn't 100% safe though
        try:
            d = json.loads(s)
        except:
            d = eval(s, {}, eval_const)
        l = d["list"]
        if not l:
            await fut
            raise LookupError("No results for " + argv + ".")
        l.sort(
            key=lambda e: scaleRatio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
            reverse=True,
        )
        if "v" in flags:
            output = (
                "```ini\n[" + noHighlight(argv) + "]\n"
                + clrHighlight("\n".join(
                    "[" + str(i + 1) + "] " + l[i].get(
                        "definition", "",
                    ).replace("\n", " ").replace("\r", "") for i in range(
                        min(len(l), 1 + 2 * flags["v"])
                    )
                )).replace("#", "♯")
                + "```"
            )
        else:
            output = (
                "```ini\n[" + noHighlight(argv) + "]\n"
                + clrHighlight(l[0].get("definition", "")).replace("#", "♯") + "```"
            )
        await fut
        return output