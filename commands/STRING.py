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
        enc = verify_url(string)
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
        print_exc()
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
        with discord.context_managers.Typing(channel):
            detected = await create_future(translators["Google Translate"].detect, string, timeout=20)
            source = detected.lang
            trans = ["Google Translate", "Papago"]
            if "p" in flags:
                trans = trans[::-1]
            if "v" in flags:
                count = 2
                end = f"\nDetected language: {bold(source)}"
            else:
                count = 1
                end = ""
            response = bold(user) + ":"
            print(string, dest, source)
            # Attempt to use all available translators if possible
            for i in range(count):
                for t in trans:
                    try:
                        resp = await create_future(getTranslate, translators[t], string, dest, source, timeout=20)
                        try:
                            output = resp.text
                        except AttributeError:
                            output = resp
                        response += f"\n{output} `{t}`"
                        source, dest = dest, source
                        break
                    except:
                        if t == trans[-1] and i == count - 1:
                            raise
        return response + end    


class Math(Command):
    _timeout_ = 4
    name = ["M", "PY", "Sympy", "Plot", "Calc"]
    min_level = 0
    description = "Evaluates a math formula."
    usage = "<function> <verbose(?v)> <rationalize(?r)>"
    flags = "rv"
    rate_limit = 0.5
    typing = True

    async def __call__(self, bot, argv, name, channel, flags, user, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        r = "r" in flags
        p = flags.get("v", 0) * 2 + 1 << 6
        if name == "plot" and not argv.lower().startswith("plot"):
            argv = f"plot({argv})"
        resp = await bot.solve_math(argv, user, p, r, timeout=24)
        # Determine whether output is a direct answer or a file
        if type(resp) is dict and "file" in resp:
            await channel.trigger_typing()
            fn = resp["file"]
            f = discord.File(fn)
            await send_with_file(channel, "", f, filename=fn, best=True)
            return
        answer = "\n".join(str(i) for i in resp)
        return py_md(f"{argv} = {answer}")


class Uni2Hex(Command):
    name = ["U2H", "HexEncode"]
    min_level = 0
    description = "Converts unicode text to hexadecimal numbers."
    usage = "<string>"
    no_parse = True

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        b = bytes(argv, "utf-8")
        return fix_md(bytes2hex(b))


class Hex2Uni(Command):
    name = ["H2U", "HexDecode"]
    min_level = 0
    description = "Converts hexadecimal numbers to unicode text."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        b = hex2Bytes(argv.replace("0x", "").replace(" ", ""))
        return fix_md(b.decode("utf-8", "replace"))


class ID2Time(Command):
    name = ["I2T", "CreateTime", "Timestamp"]
    min_level = 0
    description = "Converts a discord ID to its corresponding UTC time."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = verify_id(argv)
        return fix_md(snowflake_time(argv))


class Time2ID(Command):
    name = ["T2I", "RTimestamp"]
    min_level = 0
    description = "Converts a UTC time to its corresponding discord ID."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = tzparse(argv)
        return fix_md(time_snowflake(argv))


class SHA256(Command):
    name = ["SHA"]
    min_level = 0
    description = "Computes the SHA256 hash of a string."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        result = bytes2hex(hashlib.sha256(argv.encode("utf-8")).digest())
        return fix_md(result)


class Fancy(Command):
    name = ["FancyText"]
    min_level = 0
    description = "Creates translations of a string using unicode fonts."
    usage = "<string>"
    no_parse = True

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(name=argv)
        for i in range(len(UNIFMTS) - 1):
            s = uni_str(argv, i)
            if i == len(UNIFMTS) - 2:
                s = s[::-1]
            emb.add_field(name=f"Font {i + 1}", value=(fix_md if i & 1 else code_md)(s))
        # Only return embed if it can be sent
        if len(emb) > 6000:
            return "\n\n".join(f"{f.name}\n{strip_code_box(f.value)}" for f in emb.fields)
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
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(name=argv)
        for i in range(1, 9):
            emb.add_field(name=f"Level {i}", value=(fix_md if i & 1 else code_md)(self.zalgo(argv, i)))
        # Discord often removes combining characters past a limit, so messages longer than 6000 characters may be sent, test this by attempting to send
        try:
            await channel.send(embed=emb)
        except discord.HTTPException:
            return "\n\n".join(f"{f.name}\n{strip_code_box(f.value)}" for f in emb.fields)


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

    def __call__(self, argv, flags, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        out = argv.translate(self.otrans)
        temp = None
        if "a" in flags or "b" not in flags:
            temp = list(out)
            for i, c in enumerate(out):
                if i > 0 and c in "yY" and out[i - 1].casefold() not in "aeiouyw \n\t":
                    if c.isupper():
                        temp[i] = "W" + c.casefold()
                    else:
                        temp[i] = "w" + c
                if i < len(out) - 1 and c in "nN" and out[i + 1].casefold() in "aeiou":
                    temp[i] = c + "y"
            if "a" in flags and "b" not in flags:
                out = "".join(temp)
                temp = list(out)
                for i, c in enumerate(out):
                    if i > 0 and c.casefold() in "aeiou" and out[i - 1].casefold() not in "aeiouyw \n\t":
                        if c.isupper():
                            temp[i] = "W" + c.casefold()
                        else:
                            temp[i] = "w" + c
        if temp is not None:
            out = "".join(temp)
            if "a" in flags:
                for c in " \n\t":
                    if c in out:
                        spl = out.split(c)
                        for i, w in enumerate(spl):
                            if w.casefold().startswith("th"):
                                spl[i] = ("D" if w[0].isupper() else "d") + w[2:]
                            elif "th" in w:
                                spl[i] = w.replace("th", "ff")
                        out = c.join(spl)
        return fix_md(out)


class AltCaps(Command):
    min_level = 0
    description = "Alternates the capitalization on characters in a string."
    usage = "<string>"
    no_parse = True

    def __call__(self, argv, **void):
        i = argv[0].isupper()
        a = argv[i::2].casefold()
        b = argv[1 - i::2].upper()
        if i:
            a, b = b, a
        if len(a) > len(b):
            c = a[-1]
            a = a[:-1]
        else:
            c = ""
        return fix_md("".join(i[0] + i[1] for i in zip(a, b)) + c)


class Time(Command):
    name = ["UTC", "GMT", "T"]
    min_level = 0
    description = "Shows the current time in a certain timezone."
    usage = "<offset_hours[0]>"

    async def __call__(self, name, argv, args, user, **void):
        s = 0
        # Only check for timezones if the command was called with alias "t" or "time"
        if args and name in "time":
            for arg in (args[0], args[-1]):
                a = arg
                h = 0
                for op in "+-":
                    try:
                        i = arg.index(op)
                    except ValueError:
                        continue
                    a = arg[:i]
                    h += float(arg[i:])
                tz = a.casefold()
                if tz in TIMEZONES:
                    s = get_timezone(tz)
                    argv = argv.replace(arg, "")
                    break
                h = 0
            s += h * 3600
        elif name in TIMEZONES:
            s = TIMEZONES.get(name, 0)
        t = utc_dt()
        if argv:
            h = await self.bot.eval_math(argv, user)
        else:
            h = 0
        if h or s:
            t += datetime.timedelta(hours=h, seconds=s)
        hrs = round_min(h + s / 3600)
        if hrs >= 0:
            hrs = "+" + str(hrs)
        else:
            hrs = str(hrs)
        return ini_md(f"Current time at UTC/GMT+{hrs}: {sqr_md(t)}.")


class TimeCalc(Command):
    name = ["TimeDifference"]
    min_level = 0
    description = "Computes the difference between two times, or the Unix timestamp of a datetime string."
    usage = "<0:time1> <1:time2[0]>"
    no_parse = True

    def __call__(self, argv, user, **void):
        if not argv:
            timestamps = [utc()]
        else:
            if "," in argv:
                spl = argv.split(",")
            elif "-" in argv:
                spl = argv.split("-")
            else:
                spl = [argv]
            timestamps = [utc_ts(tzparse(t)) for t in spl]
        if len(timestamps) == 1:
            out = f"{round_min(timestamps[0])} ({datetime.datetime.utcfromtimestamp(timestamps[0])} UTC)"
        else:
            out = sec2time(max(timestamps) - min(timestamps))
        return code_md(out)


class Follow(Command):
    name = ["follow_url", "Redirect"]
    min_level = 0
    description = "Follows a discord message link and/or finds URLs in a string."
    rate_limit = 1
    
    async def __call__(self, argv, **void):
        urls = find_urls(argv)
        out = set()
        for url in urls:
            if is_discord_message_link(url):
                temp = await self.bot.follow_url(url, allow=True)
            else:
                data = await Request(url, decode=True, aio=True)
                temp = find_urls(data)
            out.update(temp)
        if not out:
            raise FileNotFoundError("No valid URLs detected.")
        return f"`Detected {len(out)} url{'s' if len(out) != 1 else ''}:`\n" + "\n".join(out)


class Match(Command):
    name = ["RE", "RegEx", "RexExp", "GREP"]
    min_level = 0
    description = "matches two strings using Linux-style RegExp, or computes the match ratio of two strings."
    rate_limit = 0.125
    no_parse = True
    
    async def __call__(self, args, name, **void):
        if len(args) < 2:
            raise ArgumentError("Please enter two or more strings to match.")
        if name == "match":
            regex = None
            for i in (1, -1):
                s = args[i]
                if len(s) >= 2 and s[0] == s[1] == "/":
                    if regex:
                        raise ArgumentError("Cannot match two Regular Expressions.")
                    regex = s[1:-1]
                    args.pop(i)
        else:
            regex = args.pop(0)
        if regex:
            temp = await create_future(re.findall, regex, " ".join(args))
            match = "\n".join(sqr_md(i) for i in temp)
        else:
            search = args.pop(0)
            s = " ".join(args)
            match = (
                sqr_md(round_min(round(fuzzy_substring(search, s) * 100, 6))) + "% literal match,\n"
                + sqr_md(round_min(round(fuzzy_substring(search.casefold(), s.casefold()) * 100, 6))) + "% case-insensitive match,\n"
                + sqr_md(round_min(round(fuzzy_substring(full_prune(search), full_prune(s)) * 100, 6))) + "% unicode mapping match."
            )
        return ini_md(match)


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
        with discord.context_managers.Typing(channel):
            s = await Request(url, headers=self.header, timeout=16, aio=True)
            d = eval_json(s)
            l = d["list"]
            if not l:
                raise LookupError(f"No results for {argv}.")
            l.sort(
                key=lambda e: scaleRatio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
                reverse=True,
            )
            if "v" in flags:
                output = (
                    "```ini\n[" + no_md(argv) + "]\n"
                    + clr_md("\n".join(
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
                    "```ini\n[" + no_md(argv) + "]\n"
                    + clr_md(l[0].get("definition", "")).replace("#", "♯") + "```"
                )
        return output