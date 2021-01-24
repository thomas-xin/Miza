try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

print = PRINT

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

try:
    translators["Papago"] = PapagoTrans(AUTH["papago_id"], AUTH["papago_secret"])
except KeyError:
    translators["Papago"] = cdict(
        translate=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to use Papago Translate.")'),
    )
    print("WARNING: papago_id/papago_secret not found. Unable to use Papago Translate.")
try:
    rapidapi_key = AUTH["rapidapi_key"]
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
    description = "Translates a string into another language."
    usage = "<0:language> <1:string> <verbose{?v}>? <papago{?p}>?"
    flags = "pv"
    no_parse = True
    rate_limit = (2, 7)
    slash = True

    async def __call__(self, channel, args, argv, flags, user, message, **void):
        if not args:
            raise ArgumentError("Input string is empty.")
        dest = args[0]
        string = argv[len(dest):].strip()
        with discord.context_managers.Typing(channel):
            detected = await create_future(translators["Google Translate"].detect, string, timeout=20)
            source = detected.lang
            trans = ["Google Translate", "Papago"]
            if "p" in flags:
                trans = trans[::-1]
            if "v" in flags:
                count = 2
                end = f"Detected language: {source}"
            else:
                count = 1
                end = None
            used = None
            response = ""
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
                        if not used:
                            used = t
                        response += f"\n{output}"
                        source, dest = dest, source
                        break
                    except:
                        if t == trans[-1] and i == count - 1:
                            raise
            if end:
                footer = dict(text=f"{used}\n{end}")
            elif used:
                footer = dict(text=used)
            else:
                footer = None
        self.bot.send_as_embeds(channel, response, author=get_author(user), footer=footer, reference=message)


class Math(Command):
    _timeout_ = 4
    name = ["🔢", "M", "PY", "Sympy", "Plot", "Calc"]
    alias = name + ["Plot3d"]
    description = "Evaluates a math formula."
    usage = "<string> <verbose{?v}>? <rationalize{?r}>? <show_variables{?l}>? <clear_variables{?c}>?"
    flags = "rvlcd"
    rate_limit = (0.5, 5)
    typing = True
    slash = True

    async def __call__(self, bot, argv, name, channel, guild, flags, user, **void):
        if "l" in flags:
            var = bot.data.variables.get(user.id, {})
            if not var:
                return ini_md(f"No currently assigned variables for {sqr_md(user)}.")
            return f"Currently assigned variables for {user}:\n" + ini_md(iter2str(var))
        if "c" in flags or "d" in flags:
            bot.data.variables.pop(user.id, None)
            return italics(css_md(f"Successfully cleared all variables for {sqr_md(user)}."))
        if not argv:
            raise ArgumentError(f"Input string is empty. Use {bot.get_prefix(guild)}math help for help.")
        r = "r" in flags
        p = flags.get("v", 0) * 2 + 1 << 7
        var = None
        if "plot" in name and not argv.lower().startswith("plot"):
            argv = f"{name}({argv})"
        elif name.startswith("m"):
            for equals in ("=", ":="):
                if equals in argv:
                    ii = argv.index(equals)
                    for i, c in enumerate(argv):
                        if i >= ii:
                            temp = argv[i + len(equals):]
                            if temp.startswith("="):
                                break
                            check = argv[:i].strip().replace(" ", "")
                            if check.isnumeric():
                                break
                            var = check
                            argv = temp.strip()
                            break
                        elif not (c.isalnum() or c in " _"):
                            break
                    if var is not None:
                        break
        resp = await bot.solve_math(argv, p, r, timeout=24, variables=bot.data.variables.get(user.id))
        # Determine whether output is a direct answer or a file
        if type(resp) is dict and "file" in resp:
            await channel.trigger_typing()
            fn = resp["file"]
            f = CompatFile(fn)
            await bot.send_with_file(channel, "", f, filename=fn, best=True)
            return
        answer = "\n".join(str(i) for i in resp)
        if var is not None:
            env = bot.data.variables.setdefault(user.id, {})
            env[var] = resp[0]
            while len(env) > 64:
                env.pop(next(iter(env)))
            bot.data.variables.update(user.id)
            return css_md(f"Variable {sqr_md(var)} set to {sqr_md(resp[0])}.")
        if argv.lower() == "help":
            return answer
        return py_md(f"{argv} = {answer}")


class UpdateVariables(Database):
    name = "variables"
    user = True


class Uni2Hex(Command):
    name = ["U2H", "HexEncode"]
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
    description = "Converts hexadecimal numbers to unicode text."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        b = hex2bytes(argv.replace("0x", "").replace(" ", ""))
        return fix_md(as_str(b))


class ID2Time(Command):
    name = ["I2T", "CreateTime", "Timestamp"]
    description = "Converts a discord ID to its corresponding UTC time."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = verify_id(argv)
        return fix_md(snowflake_time(argv))


class Time2ID(Command):
    name = ["T2I", "RTimestamp"]
    description = "Converts a UTC time to its corresponding discord ID."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        argv = tzparse(argv)
        return fix_md(time_snowflake(argv))


class SHA256(Command):
    name = ["SHA"]
    description = "Computes the SHA256 hash of a string."
    usage = "<string>"

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        result = bytes2hex(hashlib.sha256(argv.encode("utf-8")).digest())
        return fix_md(result)


class Fancy(Command):
    name = ["FancyText"]
    description = "Creates translations of a string using unicode fonts."
    usage = "<string>"
    no_parse = True
    slash = True

    def __call__(self, channel, argv, message, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        fields = deque()
        for i in range(len(UNIFMTS) - 1):
            s = uni_str(argv, i)
            if i == len(UNIFMTS) - 2:
                s = s[::-1]
            fields.append((f"Font {i + 1}", s + "\n"))
        self.bot.send_as_embeds(channel, fields=fields, author=dict(name=lim_str(argv, 256)), reference=message)


class Zalgo(Command):
    name = ["Chaos", "ZalgoText"]
    description = "Generates random combining accent symbols between characters in a string."
    usage = "<string>"
    no_parse = True
    slash = True
    chrs = [chr(n) for n in zalgo_map]
    randz = lambda self: choice(self.chrs)
    def zalgo(self, s, x):
        if unfont(s) == s:
            return "".join(c + self.randz() for c in s)
        return s[0] + "".join("".join(self.randz() + "\u200b" for i in range(x + 1 >> 1)) + c + "\u200a" + "".join(self.randz() + "\u200b" for i in range(x >> 1)) for c in s[1:])

    async def __call__(self, channel, argv, message, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        fields = deque()
        for i in range(1, 9):
            s = self.zalgo(argv, i)
            fields.append((f"Level {i}", s + "\n"))
        self.bot.send_as_embeds(channel, fields=fields, author=dict(name=lim_str(argv, 256)), reference=message)


class Format(Command):
    name = ["FormatText"]
    description = "Creates neatly fomatted text using combining unicode characters."
    usage = "<string>"
    no_parse = True
    slash = True
    formats = "".join(chr(i) for i in (0x30a, 0x325, 0x303, 0x330, 0x30c, 0x32d, 0x33d, 0x353, 0x35b, 0x20f0))

    def __call__(self, channel, argv, message, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        fields = deque()
        for i, f in enumerate(self.formats):
            s = "".join(c + f for c in argv)
            fields.append((f"Format {i}", s + "\n"))
        s = "".join("_" if c in " _" else c if c in "gjpqy" else c + chr(818) for c in argv)
        fields.append((f"Format {i + 1}", s))
        self.bot.send_as_embeds(channel, fields=fields, author=dict(name=lim_str(argv, 256)), reference=message)


class UnFancy(Command):
    name = ["UnFormat", "UnZalgo"]
    description = "Removes unicode formatting and diacritic characters from inputted text."
    usage = "<string>"
    slash = True

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        return fix_md(argv)


class OwOify(Command):
    omap = {
        "r": "w",
        "R": "W",
        "l": "w",
        "L": "W",
    }
    otrans = "".maketrans(omap)
    name = ["UwU", "OwO", "UwUify"]
    description = "Applies the owo/uwu text filter to a string."
    usage = "<string> <aggressive{?a}>? <basic{?b}>?"
    flags = "ab"
    no_parse = True

    def __call__(self, argv, flags, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        out = argv.translate(self.otrans)
        temp = None
        if "a" in flags:
            out = out.replace("v", "w").replace("V", "W")
        if "a" in flags or "b" not in flags:
            temp = list(out)
            for i, c in enumerate(out):
                if i > 0 and c in "yY" and out[i - 1].casefold() not in "aeioucdhvwy \n\t":
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
                    if i > 0 and c.casefold() in "aeiou" and out[i - 1].casefold() not in "aeioucdhvwy \n\t":
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
    description = "Alternates the capitalization on characters in a string."
    usage = "<string>"
    no_parse = True

    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
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


class Say(Command):
    description = "Repeats a message that the user provides."
    usage = "<string>"
    no_parse = True
    
    def __call__(self, argv, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        return argv.content


# Char2Emoj, a simple script to convert a string into a block of text
def _c2e(string, em1, em2):
    chars = {
        " ": [0, 0, 0, 0, 0],
        "_": [0, 0, 0, 0, 7],
        "!": [2, 2, 2, 0, 2],
        '"': [5, 5, 0, 0, 0],
        ":": [0, 2, 0, 2, 0],
        ";": [0, 2, 0, 2, 4],
        "~": [0, 5, 7, 2, 0],
        "#": [10, 31, 10, 31, 10],
        "$": [7, 10, 6, 5, 14],
        "?": [6, 1, 2, 0, 2],
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
    # I don't quite remember how this algorithm worked lol
    printed = [""] * 7
    string = string.upper()
    for i in range(len(string)):
        curr = string[i]
        data = chars.get(curr, [15] * 5)
        size = max(1, max(data))
        lim = max(2, int(log(size, 2))) + 1
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
    if len(em1) + len(em2) > 2 and ":" in em1 + em2:
        return output
    return fix_md(output)


class Char2Emoj(Command):
    name = ["C2E"]
    description = "Makes emoji blocks using a string."
    usage = "<0:string> <1:emoji_1> <2:emoji_2>"
    slash = True

    def __call__(self, args, **extra):
        try:
            if len(args) != 3:
                raise IndexError
            for i in range(1, 3):
                if args[i][0] == ":" and args[i][-1] != ":":
                    args[i] = "<" + args[i] + ">"
            return _c2e(*args[:3])
        except IndexError:
            raise ArgumentError(
                "Exactly 3 arguments are required for this command.\n"
                + "Place quotes around arguments containing spaces as required."
            )


class Time(Command):
    name = ["🕰️", "⏰", "⏲️", "UTC", "GMT", "T", "EstimateTime", "EstimateTimezone"]
    description = "Shows the current time at a certain GMT/UTC offset, or the current time for a user. Be sure to check out ⟨WEBSERVER⟩/time!"
    usage = "<offset_hours|user>?"
    slash = True

    async def __call__(self, name, channel, guild, argv, args, user, **void):
        u = user
        s = 0
        # Only check for timezones if the command was called with alias "estimate_time", "estimate_timezone", "t", or "time"
        if "estimate" in name:
            if argv:
                try:
                    if not argv.isnumeric():
                        raise KeyError
                    user = self.bot.cache.guilds[int(argv)]
                except KeyError:
                    try:
                        user = self.bot.cache.channels[verify_id(argv)]
                    except KeyError:
                        user = await self.bot.fetch_user_member(argv, guild)
            argv = None
        if args and name in "time":
            try:
                i = None
                with suppress(ValueError):
                    i = argv.index("-")
                with suppress(ValueError):
                    j = argv.index("+")
                    if i is None:
                        i = j
                    else:
                        i = min(i, j)
                if i is not None:
                    s = as_timezone(argv[:i])
                    argv = argv[i:]
                else:
                    s = as_timezone(argv)
                    argv = "0"
            except KeyError:
                user = await self.bot.fetch_user_member(argv, guild)
                argv = None
        elif name in TIMEZONES:
            s = TIMEZONES.get(name, 0)
        t = utc_dt()
        estimated = None
        if argv:
            h = await self.bot.eval_math(argv)
        elif "estimate" in name:
            if is_channel(user):
                h = self.bot.data.users.estimate_timezone("#" + str(user.id))
            else:
                h = self.bot.data.users.estimate_timezone(user.id)
            estimated = True
        elif name in "time":
            h = self.bot.data.users.get_timezone(user.id)
            if h is None:
                h = self.bot.data.users.estimate_timezone(user.id)
                estimated = True
            else:
                estimated = False
        else:
            h = 0
        if h or s:
            t += datetime.timedelta(hours=h, seconds=s)
        hrs = round_min(h + s / 3600)
        if hrs >= 0:
            hrs = "+" + str(hrs)
        out = f"Current time at UTC/GMT{hrs}: {sqr_md(t)}."
        if estimated:
            out += f"\nUsing timezone automatically estimated from {sqr_md(user)}'s discord activity."
        elif estimated is not None:
            out += f"\nUsing timezone assigned by {sqr_md(user)}."
        return ini_md(out)


class Timezone(Command):
    description = "Shows the current time in a certain timezone. Be sure to check out ⟨WEBSERVER⟩/time!"
    usage = "<timezone> <list{?l}>?"

    async def __call__(self, channel, argv, message, **void):
        if argv.startswith("-l") or argv.startswith("list"):
            fields = deque()
            for k, v in COUNTRIES.items():
                fields.append((k, ", ".join(v), False))
            self.bot.send_as_embeds(channel, description=f"[Click here to find your timezone]({self.bot.webserver}/time)", title="Timezone list", fields=fields, author=get_author(self.bot.user), reference=message)
            return
        secs = as_timezone(argv)
        t = utc_dt() + datetime.timedelta(seconds=secs)
        h = round_min(secs / 3600)
        if not h < 0:
            h = "+" + str(h)
        return ini_md(f"Current time at UTC/GMT{h}: {sqr_md(t)}.")


class TimeCalc(Command):
    name = ["TimeDifference", "TimeDiff", "TimeSum", "TimeAdd"]
    description = "Computes the sum or difference between two times, or the Unix timestamp of a datetime string."
    usage = "<0:time1> [|,] <1:time2>?"
    no_parse = True

    def __call__(self, argv, user, name, **void):
        if not argv:
            timestamps = [utc()]
        else:
            if "|" in argv:
                spl = argv.split("|")
            elif "," in argv:
                spl = argv.split(",")
            else:
                spl = [argv]
            timestamps = [utc_ts(tzparse(t)) for t in spl]
        if len(timestamps) == 1:
            out = f"{round_min(timestamps[0])} ({datetime.datetime.utcfromtimestamp(timestamps[0])} UTC)"
        elif "sum" not in name and "add" not in name:
            out = time_diff(max(timestamps), min(timestamps))
        else:
            out = time_sum(*timestamps)
        return code_md(out)


class Identify(Command):
    name = ["📂", "Magic", "Mime", "FileType"]
    description = "Detects the type, mime, and optionally details of an input file."
    usage = "<url>*"
    rate_limit = (2, 7)
    mime = magic.Magic(mime=True, mime_encoding=True)

    def probe(self, url):
        command = ["ffprobe", "-hide_banner", url]
        resp = None
        for _ in loop(3):
            try:
                proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                fut = create_future_ex(proc.communicate, timeout=8)
                res = fut.result(timeout=8)
                resp = b"\n".join(res)
                break
            except:
                with suppress():
                    proc.kill()
                print_exc()
        if not resp:
            raise RuntimeError
        return as_str(resp)

    def identify(self, url):
        out = deque()
        with requests.get(url, headers=Request.header(), stream=True) as resp:
            head = fcdict(resp.headers)
            it = resp.iter_content(262144)
            data = next(it)
        out.append(code_md(magic.from_buffer(data)))
        mimedata = self.mime.from_buffer(data).replace("; ", "\n")
        mime = mimedata.split("\n", 1)[0].split("/", 1)
        if mime == ["text", "plain"]:
            if "Content-Type" in head:
                ctype = head["Content-Type"]
                spl = ctype.split("/")
                if spl[-1].casefold() != "octet-stream":
                    mimedata = ctype + "\n" + mimedata.split("\n", 1)[-1]
                    mime = spl
        mimedata = "mimetype: " + mimedata
        if "Content-Length" in head:
            fs = head['Content-Length']
        elif len(data) < 131072:
            fs = len(data)
        else:
            fs = None
        if fs is not None:
            mimedata = f"filesize: {byte_scale(int(fs))}B\n" + mimedata
        out.append(fix_md(mimedata))
        with tracebacksuppressor:
            resp = self.probe(url)
            if mime[0] == "image" and mime[1] != "gif":
                search = "Video:"
                spl = regexp(r"\([^)]+\)").sub("", resp[resp.index(search) + len(search):].split("\n", 1)[0].strip()).split(", ")
                out.append(code_md(f"Codec: {spl[1]}\nSize: {spl[2].split(None, 1)[0]}"))
            elif mime[0] == "video" or mime[1] == "gif":
                search = "Duration:"
                resp = resp[resp.index(search) + len(search):]
                dur = time_disp(time_parse(resp[:resp.index(",")]), False)
                search = "bitrate:"
                resp = resp[resp.index(search) + len(search):]
                bps = resp.split("\n", 1)[0].strip().rstrip("b/s").casefold()
                mult = 1
                if bps.endswith("k"):
                    mult = 10 ** 3
                elif bps.endswith("m"):
                    mult = 10 ** 6
                elif bps.endswith("g"):
                    mult = 10 ** 9
                bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
                s = f"Duration: {dur}\nBitrate: {bps}"
                search = "Video:"
                try:
                    resp = resp[resp.index(search) + len(search):]
                except ValueError:
                    pass
                else:
                    spl = regexp(r"\([^)]+\)").sub("", resp.split("\n", 1)[0].strip()).split(", ")
                    s += f"\nCodec: {spl[1]}\nSize: {spl[2].split(None, 1)[0]}"
                    for i in spl[3:]:
                        if i.endswith(" fps"):
                            s += f"\nFPS: {i[:-4]}"
                            break
                out.append(code_md(s))
                search = "Audio:"
                try:
                    resp = resp[resp.index(search) + len(search):]
                except ValueError:
                    pass
                else:
                    spl = regexp(r"\([^)]+\)").sub("", resp.split("\n", 1)[0].strip()).split(", ")
                    fmt = spl[0]
                    sr = spl[1].split(None, 1)[0]
                    s = f"Audio format: {fmt}\nAudio sample rate: {sr}"
                    if len(spl) > 2:
                        s += f"\nAudio channel: {spl[2]}"
                        if len(spl) > 4:
                            bps = spl[4].rstrip("b/s").casefold()
                            mult = 1
                            if bps.endswith("k"):
                                mult = 10 ** 3
                            elif bps.endswith("m"):
                                mult = 10 ** 6
                            elif bps.endswith("g"):
                                mult = 10 ** 9
                            bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
                            s += f"\nAudio bitrate: {bps}"
                    out.append(code_md(s))
            elif mime[0] == "audio":
                search = "Duration:"
                resp = resp[resp.index(search) + len(search):]
                dur = time_disp(time_parse(resp[:resp.index(",")]), False)
                search = "Audio:"
                spl = regexp(r"\([^)]+\)").sub("", resp[resp.index(search) + len(search):].split("\n", 1)[0].strip()).split(", ")
                s = f"Duration: {dur}\nFormat: {spl[0]}\nSample rate: {spl[1].split(None, 1)[0]}"
                if len(spl) > 2:
                    s += f"\nChannel: {spl[2]}"
                    if len(spl) > 4:
                        bps = spl[4].rstrip("b/s").casefold()
                        mult = 1
                        if bps.endswith("k"):
                            mult = 10 ** 3
                        elif bps.endswith("m"):
                            mult = 10 ** 6
                        elif bps.endswith("g"):
                            mult = 10 ** 9
                        bps = byte_scale(int(bps.split(None, 1)[0]) * mult, ratio=1000) + "bps"
                        s += f"\nBitrate: {bps}"
                out.append(code_md(s))
        return "".join(out)

    async def __call__(self, bot, channel, argv, user, message, **void):
        argv += " ".join(best_url(a) for a in message.attachments)
        urls = await bot.follow_url(argv, allow=True, images=False)
        urls = set(urls)
        names = [url.rsplit("/", 1)[-1].rsplit("?", 1)[0] for url in urls]
        futs = [create_future(self.identify, url) for url in urls]
        fields = deque()
        for name, fut in zip(names, futs):
            resp = await fut
            fields.append((name, resp))
        if not fields:
            raise FileNotFoundError("Please input a file by URL or attachment.")
        title = f"{len(fields)} file{'s' if len(fields) != 1 else ''} identified"
        await bot.send_as_embeds(channel, title=title, author=get_author(user), fields=sorted(fields))


class Follow(Command):
    name = ["🚶", "Follow_URL", "Redirect"]
    description = "Follows a discord message link and/or finds URLs in a string."
    usage = "<url>*"
    rate_limit = (1, 5)
    slash = True

    async def __call__(self, channel, argv, message, **void):
        urls = find_urls(argv)
        out = set()
        for url in urls:
            if is_discord_message_link(url):
                temp = await self.bot.follow_url(url, allow=True)
            else:
                data = await self.bot.get_request(url)
                temp = find_urls(as_str(data))
            out.update(temp)
        if not out:
            raise FileNotFoundError("No valid URLs detected.")
        output = f"`Detected {len(out)} url{'s' if len(out) != 1 else ''}:`\n" + "\n".join(out)
        if len(output) > 2000 and len(output) < 54000:
            self.bot.send_as_embeds(channel, output, reference=message)
        else:
            return escape_everyone(output)


class Match(Command):
    name = ["RE", "RegEx", "RexExp", "GREP"]
    description = "matches two strings using Linux-style RegExp, or computes the match ratio of two strings."
    usage = "<0:string1> <1:string2>?"
    rate_limit = (0.5, 2)
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


class Ask(Command):
    description = "Ask me any question, and I'll answer it!"
    usage = "<string>"
    flags = "h"
    no_parse = True
    rate_limit = (0.5, 1)
    slash = True

    async def __call__(self, message, channel, user, argv, flags=(), **void):
        bot = self.bot
        guild = getattr(channel, "guild", None)
        q = single_space(full_prune(argv)).strip().translate(bot.mtrans).replace("?", "\u200b").strip("\u200b")
        if not q:
            raise ArgumentError(choice("Sorry, didn't see that, was that a question? 🤔", "Ay, speak up, I don't bite! :3"))
        out = None
        count = bot.data.users.get(user.id, {}).get("last_talk", 0)
        add_dict(bot.data.users, {user.id: {"last_talk": 1, "last_mention": 1}})
        bot.data.users[user.id]["last_used"] = utc()
        await bot.seen(user, event="misc", raw="Talking to me")
        # print(user, q)
        if q == "why":
            out = "Because! :3"
        elif q == "what":
            out = "Nothing! 🙃"
        elif q == "who":
            out = "Who, me?"
        elif q == "when":
            out = "Right now!"
        elif q == "where":
            out = "Here, dummy!"
        elif q[:3] == "how" and not q[3:4].isalpha():
            await send_with_reply(channel, "h" not in flags and message, "https://imgur.com/gallery/8cfRt")
            return
        elif (q.startswith("what's ") or q.startswith("whats ") or q.startswith("what is ")) and is_numeric(q):
            q = q[5:]
            q = q[q.index(" ") + 1:]
            try:
                if 0 <= q.rfind("<") < q.find(">"):
                    q = verify_id(q[q.rindex("<") + 1:q.index(">")])
                num = int(q)
            except ValueError:
                for _math in bot.commands.math:
                    answer = await _math(bot, q, "ask", channel, guild, {}, user)
                    if answer:
                        await send_with_reply(channel, "h" not in flags and message, answer)
                return
            else:
                if bot.in_cache(num) and "info" in bot.commands:
                    for _info in bot.commands.info:
                        await _info(num, None, "info", guild, channel, bot, user, "")
                    return
                resp = await bot.solve_math(f"factorize {num}", timeout=20)
                factors = safe_eval(resp[0])
                out = f"{num}'s factors are `{', '.join(str(i) for i in factors)}`. If you'd like more information, try {bot.get_prefix(guild)}math!"
        elif q.startswith("who's ") or q.startswith("whos ") or q.startswith("who is "):
            q = q[4:]
            q = q[q.index(" ") + 1:]
            if "info" in bot.commands:
                for _info in bot.commands.info:
                    await _info(q, None, "info", guild, channel, bot, user, "")
                return
        elif random.random() < math.atan(count / 7) / 4:
            if guild:
                bots = [member for member in guild.members if member.bot and member.id != bot.id]
            answers = ("Ay, I'm busy, ask me later!", "¯\_(ツ)_/¯", "Hm, I dunno, have you tried asking Google?")
            if bots:
                answers += (f"🥱 I'm tired... go ask {user_mention(choice(bots).id)}...",)
            await send_with_reply(channel, "h" not in flags and message, choice(answers))
            return
        elif q.startswith("why "):
            out = alist(
                "Why not?",
                "It's obvious, isn't? 😏",
                "Meh, does it matter?",
                "Why do you think?",
            )[ihash(q)]
        else:
            out = alist(
                "Yes :3",
                "Totally!",
                "Maybe?",
                "Definitely!",
                "Of course!",
                "Perhaps?",
                "Maybe not...",
                "Probably not?",
                "Nah",
                "Don't think so...",
            )[ihash(q)]
        if not out:
            raise RuntimeError("Unable to construct a valid response.")
        q = q.replace("am i", "are y\uf000ou").replace("i am", "y\uf000ou are")
        q = replace_map(q, {
            "yourself": "myself",
            "your ": "my ",
            "are you": "am I",
            "you are": "I am",
            "you're": "i'm",
            "you'll": "i'll",
        })
        res = alist(q.split())
        for sym in "!.,":
            if sym in q:
                for word, rep in {"you": "I", "me": "you", "i": "I"}.items():
                    res.replace(word + sym, rep + sym)
        q = " ".join(res.replace("you", "I").replace("i", "you").replace("me", "you").replace("i", "I").replace("i'm", "I'm").replace("i'll", "I'll"))
        if "dailies" in bot.data:
            bot.data.dailies.progress_quests(user, "talk")
        await send_with_reply(channel, "h" not in flags and message, escape_everyone(f"\xad{q[0].upper() + q[1:]}? {out}".replace("\uf000", "")))


class Urban(Command):
    time_consuming = True
    header = {
	"x-rapidapi-host": "mashape-community-urban-dictionary.p.rapidapi.com",
	"x-rapidapi-key": rapidapi_key,
    }
    name = ["📖", "UrbanDictionary"]
    description = "Searches Urban Dictionary for an item."
    usage = "<string> <verbose{?v}>*"
    flags = "v"
    rate_limit = (2, 8)
    typing = True
    slash = True

    async def __call__(self, channel, argv, flags, _timeout, **void):
        url = (
            "https://mashape-community-urban-dictionary.p.rapidapi.com/define?term="
            + argv.replace(" ", "%20")
        )
        with discord.context_managers.Typing(channel):
            d = await Request(url, headers=self.header, timeout=_timeout, json=True, aio=True)
            l = d["list"]
            if not l:
                raise LookupError(f"No results for {argv}.")
            l.sort(
                key=lambda e: scale_ratio(e.get("thumbs_up", 0), e.get("thumbs_down", 0)),
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