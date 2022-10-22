print = PRINT

import csv, knackpy
from prettytable import PrettyTable as ptable
from tsc_utils.flags import address_to_flag, flag_to_address
from tsc_utils.numbers import tsc_value_to_num, num_to_tsc_value

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class DouClub:

    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec
        self.time = utc()
        self.knack = knackpy.App(app_id=self.id, api_key=self.secret)
        create_future_ex(self.pull)

    def pull(self):
        with tracebacksuppressor:
            # print("Pulling Doukutsu Club...")
            self.data = self.knack.get("object_1")
            self.time = utc()

    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull, timeout=60)
            self.time = utc()

    def search(self, query):
        # This string search algorithm could be better
        output = []
        query = query.casefold()
        qlist = set(query.split())
        for res in self.data:
            author = res["Author"][0]["identifier"]
            name = res["Title"]
            description = res["Description"]
            s = (name, description, author)
            if not all(any(q in v for v in s) for q in qlist):
                continue
            url = f"https://doukutsuclub.knack.com/database#search-database/mod-details/{res['id']}"
            output.append({
                "author": author,
                "name": name,
                "description": description,
                "url": url,
            })
        return output

try:
    douclub = DouClub(AUTH["knack_id"], AUTH["knack_secret"])
except KeyError:
    douclub = cdict(
        search=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to search Doukutsu Club.")'),
        update=lambda: None,
        pull=lambda: None,
    )


async def searchForums(query):
    url = f"https://forum.cavestory.org/search/320966/?q={url_parse(query)}"
    s = await Request(url, aio=True, timeout=16, ssl=False, decode=True)
    output = []
    i = 0
    while i < len(s):
        # HTML is a mess
        try:
            search = '<li class="block-row block-row--separated  js-inlineModContainer" data-author="'
            s = s[s.index(search) + len(search):]
        except ValueError:
            break
        j = s.index('">')
        curr = {"author": s[:j]}
        s = s[s.index('<h3 class="contentRow-title">'):]
        search = '<a href="/'
        s = s[s.index(search) + len(search):]
        j = s.index('">')
        curr["url"] = 'https://www.cavestory.org/forums/' + s[:j].lstrip("/")
        s = s[j + 2:]
        j = s.index('</a>')
        curr["name"] = s[:j]
        search = '<div class="contentRow-snippet">'
        s = s[s.index(search) + len(search):]
        j = s.index('</div>')
        curr["description"] = s[:j]
        for elem in curr:
            temp = curr[elem].replace('<em class="textHighlight">', "**").replace("</em>", "**")
            temp = html_decode(temp)
            curr[elem] = temp
        output.append(curr)
    return output


class SheetPull:

    def __init__(self, url):
        self.url = url
        self.time = utc()
        create_future_ex(self.pull)

    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull, timeout=60)
            self.time = utc()

    def pull(self):
        with tracebacksuppressor:
            # print("Pulling Spreadsheet...")
            url = self.url
            text = Request(url, timeout=32, decode=True)
            data = text.split("\r\n")
            columns = 0
            sdata = [[], utc()]
            # Splits rows and colums into cells
            for i in range(len(data)):
                line = data[i]
                read = list(csv.reader(line))
                reli = []
                curr = ""
                for j in read:
                    if len(j) >= 2 and j[0] == j[1] == "":
                        if curr != "":
                            reli.append(curr)
                            curr = ""
                    else:
                        curr += "".join(j)
                if curr != "":
                    reli.append(curr)
                if len(reli):
                    columns = max(columns, len(reli))
                    sdata[0].append(reli)
                for line in range(len(sdata[0])):
                    while len(sdata[0][line]) < columns:
                        sdata[0][line].append(" ")
            self.data = sdata
            self.time = utc()

    def search(self, query, lim):
        output = []
        query = query.casefold()
        try:
            int(query)
            mode = 0
        except ValueError:
            mode = 1
        if not mode:
            for l in self.data[0]:
                if l[0] == query:
                    temp = [lim_line(e, lim) for e in l]
                    output.append(temp)
        else:
            qlist = set(query.split())
            for l in self.data[0]:
                if len(l) >= 3:
                    found = True
                    for q in qlist:
                        tag = False
                        for i in l:
                            if q in i.casefold():
                                tag = True
                                break
                        if not tag:
                            found = False
                            break
                    if found:
                        temp = [lim_line(e, lim) for e in l]
                        if temp[2].replace(" ", ""):
                            output.append(temp)
        return output


# URLs of Google Sheets .csv download links
entity_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0\
/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0"
)
tsc_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U\
/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0"
)


class CS_mem2flag(Command):
    name = ["CS_m2f"]
    description = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
    usage = "<0:address> <1:value(1)>?"
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        if len(args) < 2:
            num = 1
        else:
            num = await bot.eval_math(" ".join(args[1:]))
        return css_md("".join(address_to_flag(int(args[0], 16), num)))


class CS_flag2mem(Command):
    name = ["CS_f2m"]
    description = "Returns the memory offset and specific bit pointed to by a given flag number."
    usage = "<flag>"
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        flag = args[0]
        if len(flag) > 4:
            raise ValueError("Flag number should be no more than 4 characters long.")
        flag = flag.zfill(4)
        return css_md(str(flag_to_address(flag)))


class CS_num2val(Command):
    name = ["CS_n2v"]
    description = "Returns a TSC value representing the desired number, within a certain number of characters."
    usage = "<0:number> <1:length(4)>?"
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        if len(args) < 2:
            length = 4
        else:
            length = await bot.eval_math(" ".join(args[1:]))
        return css_md(str(num_to_tsc_value(int(args[0], 0), length)))


class CS_val2num(Command):
    name = ["CS_v2n"]
    description = "Returns the number encoded by a given TSC value."
    usage = "<tsc_value>"
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        return css_md(str(tsc_value_to_num(args[0])))


class CS_hex2xml(Command):
    time_consuming = True
    name = ["CS_h2x"]
    description = "Converts a given Cave Story hex patch to an xml file readable by Booster's Lab."
    usage = "<hex_data>"
    rate_limit = (3, 5)

    async def __call__(self, bot, argv, channel, message, **void):
        hacks = {}
        hack = argv.replace(" ", "").replace("`", "").strip("\n")
        while len(hack):
            # hack XML parser
            try:
                i = hack.index("0x")
            except ValueError:
                break
            hack = hack[i:]
            i = hack.index("\n")
            offs = hack[:i]
            hack = hack[i + 1:]
            try:
                i = hack.index("0x")
                curr = hack[:i]
                hack = hack[i:]
            except ValueError:
                curr = hack
                hack = ""
            curr = curr.replace(" ", "").replace("\n", "").replace("\r", "")
            n = 2
            curr = " ".join([curr[i:i + n] for i in range(0, len(curr), n)])
            if offs in hacks:
                hacks[offs] = curr + hacks[offs][len(curr):]
            else:
                hacks[offs] = curr
        # Generate hack template
        output = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<hack name="HEX PATCH">\n'
            + '\t<panel>\n'
            + '\t\t<panel title="Description">\n'
            + '\t\t</panel>\n'
            + '\t\t<field type="info">\n'
            + '\t\t\tHex patch converted by ' + bot.user.name + '.\n'
            + '\t\t</field>\n'
            + '\t\t<panel title="Data">\n'
            + '\t\t</panel>\n'
            + '\t\t<panel>\n'
        )
        col = 0
        for hack in sorted(hacks):
            n = 63
            p = hacks[hack]
            p = '\n\t\t\t\t'.join([p[i:i + n] for i in range(0, len(p), n)])
            output += (
                '\t\t\t<field type="data" offset="' + hack + '" col="' + str(col) + '">\n'
                + '\t\t\t\t' + p + '\n'
                + '\t\t\t</field>\n'
            )
            col = 1 + col & 3
        output += (
            '\t\t</panel>\n'
            + '\t</panel>\n'
            + '</hack>'
        )
        b = output.encode("utf-8")
        f = CompatFile(b, filename="patch.xml")
        create_task(bot.send_with_file(channel, "Patch successfully converted!", f, reference=message))


class CS_npc(Command):
    time_consuming = True
    description = "Searches the Cave Story NPC list for an NPC by name or ID."
    usage = "<query> <condensed{?c}>?"
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, bot, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(entity_list.search, argv, lim, timeout=8)
        # Sends multiple messages up to 20000 characters total
        if len(data):
            head = entity_list.data[0][1]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i * " "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output) < 20000 and len(output) > 1900:
                response = [f"Search results for `{argv}`:"]
                lines = output.splitlines()
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            return f"Search results for `{argv}`:\n{code_md(output)}"
        raise LookupError(f"No results for {argv}.")


class CS_flag(Command):
    name = ["CS_OOB", "CS_flags"]
    description = "Searches the Cave Story OOB flags list for a memory variable."
    usage = "<query> <condensed{?c}>?"
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(tsc_list.search, argv, lim, timeout=8)
        # Sends multiple messages up to 20000 characters total
        if len(data):
            head = tsc_list.data[0][0]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i * " "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output) < 20000 and len(output) > 1900:
                response = [f"Search results for `{argv}`:"]
                lines = output.splitlines()
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            return f"Search results for `{argv}`:\n{code_md(output)}"
        raise LookupError(f"No results for {argv}.")


class CS_mod(Command):
    time_consuming = True
    name = ["CS_search"]
    description = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
    usage = "<query>"
    no_parse = True
    rate_limit = (3, 7)

    async def __call__(self, channel, user, args, **void):
        argv = " ".join(args)
        fut = create_future(douclub.search, argv, timeout=8)
        try:
            data = await searchForums(argv)
        except ConnectionError as ex:
            if ex.errno != 404:
                raise
            data = []
        data += await fut
        if not data:
            raise LookupError(f"No results for {argv}.")
        description = f"Search results for `{argv}`:\n"
        fields = deque()
        for res in data:
            fields.append(dict(
                name=res["name"],
                value=res["url"] + "\n" + lim_str(res["description"], 128).replace("\n", " ") + f"\n> {res['author']}",
                inline=False,
            ))
        self.bot.send_as_embeds(channel, description=description, fields=fields, author=get_author(user))


class CS_Database(Database):
    name = "cs_database"
    no_file = True

    async def __call__(self, **void):
        entity_list.update()
        tsc_list.update()
        douclub.update()


class MathQuiz(Command):
    name = ["MathTest", "MQ"]
    min_level = 1
    description = "Starts a math quiz in the current channel."
    usage = "(easy|hard)? <disable{?d}>?"
    flags = "aed"
    rate_limit = 3

    async def __call__(self, channel, guild, flags, argv, **void):
        mathdb = self.bot.data.mathtest
        if "d" in flags:
            if channel.id in mathdb.data:
                mathdb.data.pop(channel.id)
            return italics(css_md(f"Disabled math quizzes for {sqr_md(channel)}."))
        if not argv:
            argv = "easy"
        elif argv not in ("easy", "hard"):
            raise TypeError("Invalid quiz mode.")
        mathdb.data[channel.id] = cdict(mode=argv, answer=None)
        return italics(css_md(f"Enabled {argv} math quiz for {sqr_md(channel)}."))


class UpdateMathTest(Database):
    name = "mathtest"
    no_file = True

    def __load__(self):
        s = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        ss = {str(i): s[i] for i in range(len(s))}
        ss["-"] = "⁻"
        self.sst = "".maketrans(ss)

    def format(self, x, y, op):
        length = 6
        xs = str(x)
        xs = " " * (length - len(xs)) + xs
        ys = str(y)
        ys = " " * (length - len(ys)) + ys
        return " " + xs + "\n" + op + ys

    def eqtrans(self, eq):
        return str(eq).replace("**", "^").replace("exp", "e^").replace("*", "∙")

    # Addition of 2 numbers less than 10000
    def addition(self):
        x = xrand(10000)
        y = xrand(10000)
        s = self.format(x, y, "+")
        return s, x + y

    # Subtraction of 2 numbers, result must be greater than or equal to 0
    def subtraction(self):
        x = xrand(12000)
        y = xrand(8000)
        if x < y:
            x, y = y, x
        s = self.format(x, y, "-")
        return s, x - y

    # Addition of 2 numbers 2~20
    def multiplication(self):
        x = xrand(2, 20)
        y = xrand(2, 20)
        s = self.format(x, y, "×")
        return s, x * y

    # Addition of 2 numbers 13~99
    def multiplication2(self):
        x = xrand(13, 100)
        y = xrand(13, 100)
        s = self.format(x, y, "×")
        return s, x * y

    # Division result between 2 and 13
    def division(self):
        y = xrand(2, 20)
        x = xrand(2, 14) * y
        s = self.format(x, y, "/")
        return s, x // y

    # Power of 2
    def exponentiation(self):
        x = xrand(2, 20)
        y = xrand(2, max(3, 14 / x))
        s = str(x) + "^" + str(y)
        return s, x ** y

    # Power of 2 or 3
    def exponentiation2(self):
        x = xrand(2, 4)
        if x == 2:
            y = xrand(7, 35)
        else:
            y = xrand(5, 11)
        s = str(x) + "^" + str(y)
        return s, x ** y

    # Square root result between 2 and 19
    def square_root(self):
        x = xrand(2, 20)
        y = x ** 2
        s = "√" + str(y)
        return s, x

    # Square root result between 21 and 99
    def square_root2(self):
        x = xrand(21, 1000)
        y = x ** 2
        s = "√" + str(y)
        return s, x

    # Scientific number form, exponent between -3 and 5
    def scientific(self):
        x = xrand(100, 10000)
        x /= 10 ** int(math.log10(x))
        y = xrand(-3, 6)
        s = str(x) + "×10^" + str(y)
        return s, round(x * 10 ** y, 9)

    # Like division but may result in a finite decimal
    def fraction(self):
        y = choice([2, 4, 5, 10])
        x = xrand(3, 20)
        mult = xrand(4) + 1
        y *= mult
        x *= mult
        s = self.format(x, y, "/")
        return s, round(x / y, 9)

    # An infinite recurring decimal number of up to 3 digits
    def recurring(self):
        x = "".join(str(xrand(10)) for _ in loop(xrand(2, 4)))
        s = "0." + "".join(x[i % len(x)] for i in range(28)) + "..."
        ans = "0.[" + x + "]"
        return s, ans

    # Quadratic equation with a = 1
    def equation(self):
        a = xrand(1, 10)
        b = xrand(1, 10)
        if xrand(2):
            a = -a
        if xrand(2):
            b = -b
        bx = -a - b
        cx = a * b
        s = "x^2 "
        if bx:
            s += ("+", "-")[bx < 0] + " " + (str(abs(bx))) * (abs(bx) != 1) +  "x "
        s += ("+", "-")[cx < 0] + " " + str(abs(cx)) + " = 0"
        return s, [a, b]

    # Quadratic equation with all values up to 13
    async def equation2(self):
        a = xrand(1, 14)
        b = xrand(1, 14)
        c = xrand(1, 14)
        d = xrand(1, 14)
        if xrand(2):
            a = -a
        if xrand(2):
            b = -b
        if xrand(2):
            c = -c
        if xrand(2):
            d = -d
        st = "(" + str(a) + "*x+" + str(b) + ")*(" + str(c) + "*x+" + str(d) + ")"
        a = [-sympy.Number(b) / a, -sympy.Number(d) / c]
        q = await create_future(sympy.expand, st, timeout=8)
        q = self.eqtrans(q).replace("∙", "") + " = 0"
        return q, a

    # A derivative or integral
    async def calculus(self):
        amount = xrand(2, 5)
        s = []
        for i in range(amount):
            t = xrand(3)
            if t == 0:
                a = xrand(1, 7)
                e = xrand(-3, 8)
                if xrand(2):
                    a = -a
                s.append(str(a) + "x^(" + str(e) + ")")
            elif t == 1:
                a = xrand(5)
                if a <= 1:
                    a = "e"
                s.append("+-"[xrand(2)] + str(a) + "^x")
            elif t == 2:
                a = xrand(6)
                if a < 1:
                    a = 1
                if xrand(2):
                    a = -a
                op = ["sin", "cos", "tan", "sec", "csc", "cot", "log"]
                s.append(str(a) + "*" + choice(op) + "(x)")
        st = ""
        for i in s:
            if st and i[0] not in "+-":
                st += "+"
            st += i
        ans = await self.bot.solve_math(st, xrand(2147483648), 0, 1)
        a = ans[0]
        q = self.eqtrans(a)
        if xrand(2):
            q = "Dₓ " + q
            op = sympy.diff
        else:
            q = "∫ " + q
            op = sympy.integrate
        a = await create_future(op, a, timeout=8)
        return q, a

    # Selects a random math question based on difficulty.
    async def generateMathQuestion(self, mode):
        easy = (
            self.addition,
            self.subtraction,
            self.multiplication,
            self.division,
            self.exponentiation,
            self.square_root,
            self.scientific,
            self.fraction,
            self.recurring,
            self.equation,
        )
        hard = (
            self.multiplication2,
            self.exponentiation2,
            self.square_root2,
            self.equation2,
            self.calculus,
        )
        modes = {"easy": easy, "hard": hard}
        qa = choice(modes[mode])()
        if awaitable(qa):
            return await qa
        return qa

    async def newQuestion(self, channel):
        q, a = await self.generateMathQuestion(self.data[channel.id].mode)
        msg = "```\n" + q + "```"
        self.data[channel.id].answer = a
        await channel.send(msg)

    async def __call__(self):
        bot = self.bot
        for c_id in self.data:
            if self.data[c_id].answer is None:
                self.data[c_id].answer = nan
                channel = await bot.fetch_channel(c_id)
                await self.newQuestion(channel)

    messages = cdict(
        correct=[
            "Great work!",
            "Very nice!",
            "Congrats!",
            "Nice job! Keep going!",
            "That is correct!",
            "Bullseye!",
        ],
        incorrect=[
            "Aw, close, keep trying!",
            "Oops, not quite, try again!",
        ],
    )

    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if channel.id in self.data:
            if message.author.id != bot.id:
                msg = message.content.strip("|").strip("`")
                if not msg or msg.casefold() != msg:
                    return
                # Ignore commented messages
                if msg.startswith("#") or msg.startswith("//") or msg.startswith("\\"):
                    return
                try:
                    x = await bot.solve_math(msg, message.author, 0, 1)
                    x = await create_future(sympy.sympify, x[0], timeout=6)
                except:
                    return
                correct = False
                a = self.data[channel.id].answer
                if type(a) is list:
                    if x in a:
                        correct = True
                else:
                    a = await create_future(sympy.sympify, a, timeout=6)
                    d = await create_future(sympy.Add, x, -a, timeout=12)
                    z = await create_future(sympy.simplify, d, timeout=18)
                    correct = z == 0
                if correct:
                    create_task(self.newQuestion(channel))
                    pull = self.messages.correct
                else:
                    pull = self.messages.incorrect
                high = (len(pull) - 1) ** 2
                i = isqrt(random.randint(0, high))
                await channel.send(pull[i])


class Wav2Png(Command):
    _timeout_ = 15
    name = ["Png2Wav", "Png2Mp3"]
    description = "Runs wav2png on the input URL. See https://github.com/thomas-xin/Audio-Image-Converter for more info, or to run it yourself!"
    usage = "<0:search_links>"
    rate_limit = (9, 30)
    typing = True

    async def __call__(self, bot, channel, message, argv, name, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            raise ArgumentError("Input string is empty.")
        urls = await bot.follow_url(argv, allow=True, images=False)
        if not urls or not urls[0]:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        fn = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        ts = ts_us()
        ext = "png" if name == "wav2png" else "wav"
        dest = f"cache/&{ts}." + ext
        w2p = "wav2png" if name == "wav2png" else "png2wav"
        args = [python, w2p + ".py", url, "../" + dest]
        with discord.context_managers.Typing(channel):
            print(args)
            proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd() + "/misc", stdout=subprocess.DEVNULL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3200)
            except (T0, T1, T2):
                with tracebacksuppressor:
                    force_kill(proc)
                raise
        await bot.send_with_file(channel, "", dest, filename=fn + "." + ext, reference=message)


class SpectralPulse(Command):
    _timeout_ = 150
    description = "Runs SpectralPulse on the input URL. Operates on a global queue system. See https://github.com/thomas-xin/SpectralPulse for more info, or to run it yourself!"
    usage = "<0:search_links>"
    rate_limit = (12, 60)
    typing = True
    spec_sem = Semaphore(1, 256, rate_limit=1)

    async def __call__(self, bot, channel, message, args, **void):
        for a in message.attachments:
            args.insert(0, a.url)
        if not args:
            raise ArgumentError("Input string is empty.")
        urls = await bot.follow_url(args.pop(0), allow=True, images=False)
        if not urls or not urls[0]:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        kwargs = {
            "-size": "[1280,720]",
            "-fps": "30",
            "-sample_rate": "48000",
            "-amplitude": "0.1",
            "-smudge_ratio": "0.9",
            "-speed": "2",
            "-lower_bound": "A0",
            "-higher_bound": "F#10",
            "-particles": "piano",
            "-skip": "true",
            "-display": "false",
            "-render": "true",
            "-play": "false",
            "-image": "true",
        }
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("-"):
                if arg in {
                    "-size", "-fps", "-sample_rate", "-amplitude",
                    "-smudge_ratio", "-speed", "-lower_bound", "-higher_bound",
                    "-particles", "-skip", "-display", "-render", "-play", "-image",
                    "-dest",
                    "-width", "-height",
                }:
                    kwargs[arg] = args[i + 1].replace("\xad", "#")
                    i += 1
            i += 1
        if "-width" in kwargs and "-height" in kwargs:
            kwargs["-size"] = f'({kwargs["-width"]},{kwargs["-height"]})'
        name = kwargs.get("-dest") or url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        n1 = name + ".mp4"
        n2 = name + ".png"
        ts = ts_us()
        dest = f"cache/&{ts}"
        fn1 = dest + ".mp4"
        fn2 = dest + ".png"
        args = [
            python, "misc/spectralpulse/main.py",
            *itertools.chain(*kwargs.items()),
            "-dest", dest, url,
        ]
        with discord.context_managers.Typing(channel):
            if self.spec_sem.is_busy() and not getattr(message, "simulated", False):
                await send_with_react(channel, italics(ini_md(f"SpectralPulse: {sqr_md(url)} enqueued in position {sqr_md(self.spec_sem.passive + 1)}.")), reacts="❎", reference=message)
            async with self.spec_sem:
                print(args)
                proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd(), stdout=subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3200)
                except (T0, T1, T2):
                    with tracebacksuppressor:
                        force_kill(proc)
                    raise
                for ext in ("pcm", "riff"):
                    await create_future(os.remove, f"{dest}.{ext}")
        await bot.send_with_file(channel, "", fn1, filename=n1, reference=message)
        if kwargs.get("-image") in ("true", "True"):
            await bot.send_with_file(channel, "", fn2, filename=n2, reference=message)


class StableDiffusion(Command):
    _timeout_ = 150
    name = ["Art", "AIArt"]
    description = "Runs a Stable Diffusion AI art generator on the input prompt or image. Operates on a global queue system, and must be installed separately from https://github.com/bes-dev/stable_diffusion.openvino, extracted into the misc folder. Accepts appropriate keyword arguments."
    usage = "<0:prompt>"
    rate_limit = (12, 60)
    typing = True
    sdiff_sem = Semaphore(1, 256, rate_limit=1)
    cache = {}

    async def stable_diffusion_deepai(self, prompt):
        resp = await create_future(
            requests.post,
            "https://api.deepai.org/api/text2img",
            files=dict(
                text=prompt,
            ),
            headers={
                "api-key": "quickstart-QUdJIGlzIGNvbWluZy4uLi4K",
            },
        )
        if resp.status_code in range(200, 400):
            print(resp.text)
            url = resp.json()["output_url"]
            b = await self.bot.get_request(url)
            image = Image.open(io.BytesIO(b))
            ims = [
                image.crop((0, 0, 512, 512)),
                image.crop((512, 0, 1024, 512)),
                image.crop((512, 512, 1024, 1024)),
                image.crop((0, 512, 512, 1024)),
            ]
            self.cache.setdefault(prompt, []).extend(ims)
            return shuffle(self.cache[prompt])
        print(ConnectionError(resp.status_code, resp.text))
        return ()

    async def __call__(self, bot, channel, message, args, **void):
        for a in message.attachments:
            args.insert(0, a.url)
        if not args:
            raise ArgumentError("Input string is empty.")
        req = " ".join(args)
        url = None
        rems = deque()
        kwargs = {
            "--num-inference-steps": "24",
            "--guidance-scale": "7.5",
            "--eta": "0.8",
        }
        specified = set()
        kwarg = ""
        for arg in args:
            if kwarg:
                # if kwarg == "--model":
                #     kwargs[kwarg] = arg
                if kwarg == "--seed":
                    kwargs[kwarg] = arg
                elif kwarg in ("--num-inference-steps", "--ddim_steps"):
                    kwarg = "--num-inference-steps"
                    kwargs[kwarg] = str(max(1, min(64, int(arg))))
                elif kwarg in ("--guidance-scale", "--scale"):
                    kwarg = "--guidance-scale"
                    kwargs[kwarg] = str(max(0, min(100, float(arg))))
                elif kwarg == "--eta":
                    kwargs[kwarg] = str(max(0, min(1, float(arg))))
                # elif kwarg in ("--tokenizer", "--tokeniser"):
                #     kwargs["--tokenizer"] = arg
                elif kwarg == "--prompt":
                    kwargs[kwarg] = arg
                elif kwarg == "--strength":
                    kwargs[kwarg] = str(max(0, min(1, float(arg))))
                # elif kwargs == "--mask":
                #     kwargs[kwarg] = arg
                specified = kwarg
                kwarg = ""
                continue
            if arg.startswith("--"):
                kwarg = arg
                continue
            urls = await bot.follow_url(arg, allow=True, images=True)
            if not urls:
                rems.append(arg)
            elif not url:
                url = urls[0]
        prompt = " ".join(rems).strip()
        if not prompt:
            if not url:
                raise ArgumentError("Please input a valid prompt.")
            processor = await create_future(TrOCRProcessor.from_pretrained, "nlpconnect/vit-gpt2-image-captioning")
            model = await create_future(VisionEncoderDecoderModel.from_pretrained, "nlpconnect/vit-gpt2-image-captioning")
            b = await bot.get_request(url)
            image = Image.open(io.BytesIO(b)).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = await create_future(model.generate, pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            prompt = generated_text.strip()
            if not prompt:
                prompt = "art"
        req = prompt
        if url:
            if req:
                req += " "
            req += url
        if specified:
            req += " ".join(f"{k} {v}" for k, v in kwargs.items() if k in specified)
        fn = None
        if not specified or not os.path.exists("misc/stable_diffusion.openvino"):
            if self.cache.get(prompt):
                fn = self.cache[prompt].pop(0).tobytes()
                if not self.cache[prompt]:
                    create_task(self.stable_diffusion_deepai(prompt))
            else:
                with discord.context_managers.Typing(channel):
                    ims = await self.stable_diffusion_deepai(prompt)
                    if ims:
                        fn = ims.pop(0).tobytes()
        if not fn:
            args = [
                "py",
                "-3.9",
                "demo.py",
            ]
            if prompt and "--prompt" not in kwargs:
                args.extend((
                    "--prompt",
                    prompt,
                ))
            if url:
                b = await bot.get_request(url)
                fn = "misc/stable_diffusion.openvino/input.png"
                with open(fn, "wb") as f:
                    f.write(b)
                args.extend((
                    "--init-image",
                    "input.png",
                ))
                if "--strength" not in kwargs:
                    args.extend((
                        "--strength",
                        "0.75",
                    ))
            for k, v in kwargs.items():
                args.extend((k, v))
            with discord.context_managers.Typing(channel):
                if self.sdiff_sem.is_busy() and not getattr(message, "simulated", False):
                    await send_with_react(channel, italics(ini_md(f"StableDiffusion: {sqr_md(req)} enqueued in position {sqr_md(self.sdiff_sem.passive + 1)}.")), reacts="❎", reference=message)
                async with self.sdiff_sem:
                    print(args)
                    proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd() + "/misc/stable_diffusion.openvino", stdout=subprocess.DEVNULL)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3200)
                    except (T0, T1, T2):
                        with tracebacksuppressor:
                            force_kill(proc)
                        raise
            fn = "misc/stable_diffusion.openvino/output.png"
        await bot.send_with_file(channel, "", fn, filename=prompt + ".png", reference=message)


class DeviantArt(Command):
    server_only = True
    min_level = 2
    description = "Subscribes to a DeviantArt Gallery, reposting links to all new posts."
    usage = "(add|remove)? <url> <reversed{?r}>?"
    flags = "raed"
    rate_limit = 4

    async def __call__(self, argv, flags, channel, guild, bot, **void):
        data = bot.data.deviantart
        update = bot.data.deviantart.update
        if not argv:
            assigned = data.get(channel.id, ())
            if not assigned:
                return ini_md(f"No currently subscribed DeviantArt Galleries for {sqr_md(channel)}.")
            if "d" in flags:
                data.pop(channel.id, None)
                return css_md(f"Successfully removed all DeviantArt Gallery subscriptions from {sqr_md(channel)}.")
            return f"Currently subscribed DeviantArt Galleries for {sqr_md(channel)}:{ini_md(iter2str(assigned, key=lambda x: x['user']))}"
        urls = await bot.follow_url(argv, images=False, allow=True)
        if not urls:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        if "deviantart.com" not in url:
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        # Parse DeviantArt gallery URls
        url = url[url.index("deviantart.com") + 15:]
        spl = url.split("/")
        user = spl[0]
        if spl[1] != "gallery":
            raise ArgumentError("Only Gallery URLs are supported.")
        content = spl[2].split("&", 1)[0]
        folder = no_md(spl[-1].split("&", 1)[0])
        # Gallery may be an ID or "all"
        try:
            content = int(content)
        except (ValueError, TypeError):
            if content in (user, "all"):
                content = user
            else:
                raise TypeError("Invalid Gallery type.")
        if content in self.data.get(channel.id, {}):
            raise KeyError(f"Already subscribed to {user}: {folder}")
        if "d" in flags:
            try:
                data.get(channel.id).pop(content)
            except KeyError:
                raise KeyError(f"Not currently subscribed to {user}: {folder}")
            else:
                if channel.id in data and not data[channel.id]:
                    data.pop(channel.id)
                return css_md(f"Successfully unsubscribed from {sqr_md(user)}: {sqr_md(folder)}.")
        set_dict(data, channel.id, {}).__setitem__(content, {"user": user, "type": "gallery", "reversed": ("r" in flags), "entries": {}})
        update(channel.id)
        out = f"Successfully subscribed to {sqr_md(user)}: {sqr_md(folder)}"
        if "r" in flags:
            out += ", posting in reverse order"
        return css_md(out + ".")


class UpdateDeviantArt(Database):
    name = "deviantart"

    async def processPart(self, found, c_id):
        bot = self.bot
        try:
            channel = await bot.fetch_channel(c_id)
        except (LookupError, discord.NotFound):
            self.data.pop(c_id, None)
            return
        try:
            assigned = self.data.get(c_id)
            if assigned is None:
                return
            embs = deque()
            for content in assigned:
                if content not in found:
                    continue
                items = found[content]
                entries = assigned[content]["entries"]
                new = tuple(items)
                orig = tuple(entries)
                # O(n) comparison
                if assigned[content].get("reversed", False):
                    it = reversed(new)
                else:
                    it = new
                for i in it:
                    if i not in entries:
                        entries[i] = True
                        self.update(c_id)
                        home = "https://www.deviantart.com/" + items[i][2]
                        emb = discord.Embed(
                            colour=discord.Colour(1),
                            description="*🔔 New Deviation from " + items[i][2] + " 🔔*\n" + items[i][0],
                        ).set_image(url=items[i][1]).set_author(name=items[i][2], url=home, icon_url=items[i][3])
                        embs.append(emb)
                for i in orig:
                    if i not in items:
                        entries.pop(i)
                        self.update(c_id)
        except:
            print(found)
            print_exc()
        else:
            bot.send_embeds(channel, embs)

    async def fetch_gallery(self, folder, username):
        base = "https://www.deviantart.com/_napi/da-user-profile/api/gallery/contents?username="
        # "all" galleries require different URL options
        if type(folder) is str:
            f_id = "&all_folder=true&mode=oldest"
        else:
            f_id = "&folderid=" + str(folder)
        url = base + username + f_id
        # Binary search algorithm to improve search time for entire galleries to O(log n)
        maxitems = 2147483647
        r = 0
        t = utc()
        found = {}
        futs = deque()
        page = 24
        # Begin with quaternary search (powers of 4) to estimate lowest power of 2 greater than or equal to gallery page count
        with suppress(StopIteration):
            for i in range(2 + int(math.log2(maxitems / page))):
                curr = 1 << i
                search = url + f"&offset={curr * page}&limit={page}"
                futs.append((curr, Request(search, timeout=20, json=True, aio=True)))
                if i & 1:
                    for x, fut in futs:
                        try:
                            resp = await fut
                        except ConnectionError as ex:
                            if ex.errno >= 500:
                                return
                            raise
                        if resp.get("results"):
                            found[x] = resp
                        if not resp.get("hasMore"):
                            curr = x
                            raise StopIteration
                r += 1
        # Once the end has been reached, use binary search to estimate the page count again, being off by at most 8 pages
        check = 1 << max(0, i - 2)
        while check > 4:
            x = curr - check
            search = url + f"&offset={x * page}&limit={page}"
            resp = await Request(search, json=True, aio=True)
            if resp.get("results"):
                found[x] = resp
            r += 1
            if not resp.get("hasMore"):
                curr = x
            check >>= 1
        futs = deque()
        for i in range(curr + 1):
            if i not in found:
                search = url + f"&offset={i * page}&limit={page}"
                futs.append((i, Request(search, json=True, aio=True)))
                r += 1
        for x, fut in futs:
            resp = await fut
            if resp.get("results"):
                found[x] = resp
        # Collect all page results into a single list
        results = (resp.get("results", ()) for resp in found.values())
        items = {}
        for res in itertools.chain(*results):
            deviation = res["deviation"]
            media = deviation["media"]
            prettyName = media["prettyName"]
            orig = media["baseUri"]
            extra = ""
            token = "?token=" + media["token"][0]
            # Attempt to find largest available format for media
            for t in reversed(media["types"]):
                if t["t"].casefold() == "fullview":
                    if "c" in t:
                        extra = "/" + t["c"].replace("<prettyName>", prettyName)
                        break
            image_url = orig + extra + token
            items[deviation["deviationId"]] = (deviation["url"], image_url, deviation["author"]["username"], deviation["author"]["usericon"])
        return items

    async def __call__(self):
        t = set_dict(self.__dict__, "time", 0)
        # Fetches once every 5 minutes
        if utc() - t < 300:
            return
        self.time = inf
        conts = {i: a[i]["user"] for a in tuple(self.data.values()) for i in a}
        total = {}
        attempts, successes = 0, 0
        for folder, username in conts.items():
            with tracebacksuppressor:
                items = await self.fetch_gallery(folder, username)
                if items:
                    total[folder] = items
        for c_id in tuple(self.data):
            create_task(self.processPart(total, c_id))
        self.time = utc()