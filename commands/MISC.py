try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import csv, knackpy
from prettytable import PrettyTable as ptable

knackpy.__builtins__["print"] = lambda *args, **kwargs: None


class DouClub:
    
    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec
        self.time = 0
        self.pull()

    def pull(self):
        try:
            knackpy.__builtins__["print"] = lambda *args, **kwargs: None
            # print("Pulling Doukutsu Club...")
            kn = knackpy.Knack(obj="object_1", app_id=self.id, api_key=self.secret)
            self.data = kn.data
            self.time = utc()
        except:
            print(traceback.format_exc())
    
    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull)
            self.time = utc()

    def search(self, query):
        output = []
        query = query.lower()
        for l in self.data:
            found = True
            qlist = query.split(" ")
            for q in qlist:
                tag = False
                for k in l:
                    i = str(l[k])
                    if q in i.lower():
                        tag = True
                        break
                if not tag:
                    found = False
                    break
            if found:
                output.append({
                    "author": l["Author"],
                    "name": l["Title"],
                    "description": l["Description"],
                    "url": (
                        "https://doukutsuclub.knack.com/database#search-database/mod-details/"
                        + l["id"] + "/"
                    ),
                })
        return output

f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    douclub = DouClub(auth["knack_id"], auth["knack_secret"])
except KeyError:
    douclub = cdict(
        search=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to search Doukutsu Club.")'),
        update=lambda: None
    )
    print("WARNING: knack_id/knack_secret not found. Unable to search Doukutsu Club.")


async def searchForums(query):
    url = (
        "https://www.cavestory.org/forums/search/1/?q=" + query.replace(" ", "+")
        + "&t=post&c[child_nodes]=1&c[nodes][0]=33&o=date&g=1"
    )
    s = await Request(url, aio=True, timeout=16, decode=True)
    output = []
    i = 0
    while i < len(s):
        try:
            search = '<li class="block-row block-row--separated  js-inlineModContainer" data-author="'
            s = s[s.index(search) + len(search):]
        except ValueError:
            break
        j = s.index('">')
        curr = {"author": s[:j]}
        s = s[s.index('<h3 class="contentRow-title">'):]
        search = '<a href="/forums/'
        s = s[s.index(search) + len(search):]
        j = s.index('">')
        curr["url"] = 'https://www.cavestory.org/forums/' + s[:j]
        s = s[j + 2:]
        j = s.index('</a>')
        curr["name"] = s[:j]
        search = '<div class="contentRow-snippet">'
        s = s[s.index(search) + len(search):]
        j = s.index('</div>')
        curr["description"] = s[:j]
        for elem in curr:
            temp = curr[elem].replace('<em class="textHighlight">', "").replace('</em>', "")
            temp = htmlDecode(temp)
            curr[elem] = temp
        output.append(curr)
    return output
    

class SheetPull:
    
    def __init__(self, url):
        self.url = url
        self.time = 0
        self.pull()

    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull)
            self.time = utc()

    def pull(self):
        try:
            # print("Pulling Spreadsheet...")
            url = self.url
            text = Request(url, timeout=32, decode=True)
            data = text.split("\r\n")
            columns = 0
            sdata = [[], utc()]
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
        except:
            print(traceback.format_exc())

    def search(self, query, lim):
        output = []
        query = query.lower()
        try:
            int(query)
            mode = 0
        except ValueError:
            mode = 1
        if not mode:
            for l in self.data[0]:
                if l[0] == query:
                    temp = [limLine(e, lim) for e in l]
                    output.append(temp)
        else:
            qlist = query.split(" ")
            for q in qlist:
                for l in self.data[0]:
                    if len(l) >= 3:
                        found = False
                        for i in l:
                            if q in i.lower():
                                found = True
                        if found:
                            temp = [limLine(e, lim) for e in l]
                            if temp[2].replace(" ", ""):
                                output.append(temp)
        return output


entity_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0\
/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0"
)
tsc_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U\
/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0"
)


def _n2f(n):
    flag = int(n)
    offset = max(0, (999 - flag) // 1000)
    flag += offset * 1000
    output = ""
    for i in range(0, 3):
        a = 10 ** i
        b = flag // a
        char = b % 10
        char += 48
        output += chr(char)
    char = flag // 1000
    char += 48
    char -= offset
    try:
        return chr(char) + output[::-1]
    except ValueError:
        return "(0x" + hex((char + 256) & 255).upper()[2:] + ")" + output[::-1]


def _m2f(mem, val):
    val1 = mem
    val2 = val & 4294967295
    curr = 0
    result = ""
    while val2:
        difference = int(val1, 16) - 4840864 + curr / 8
        flag = difference * 8
        output = _n2f(flag)
        if val2 & 1:
            operation = "+"
        else:
            operation = "-"
        output = "<FL" + operation + output
        result += output
        val2 >>= 1
        curr += 1
    return result


class CS_mem2flag(Command):
    name = ["CS_m2f"]
    min_level = 0
    description = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
    usage = "<0:address> <1:value[1]>"
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        if len(args) < 2:
            return "```css\n" + _m2f(args[0], 1) + "```"
        num = await bot.evalMath(" ".join(args[1:]), user)
        return "```css\n" + _m2f(args[0], num) + "```"


class CS_hex2xml(Command):
    time_consuming = True
    name = ["CS_h2x"]
    min_level = 0
    description = "Converts a given Cave Story hex patch to an xml file readable by Booster's Lab."
    usage = "<hex_data>"
    rate_limit = 3

    async def __call__(self, client, argv, channel, **void):
        hacks = {}
        hack = argv.replace(" ", "").replace("`", "").strip("\n")
        while len(hack):
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
        output = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<hack name="HEX PATCH">\n'
            + '\t<panel>\n'
            + '\t\t<panel title="Description">\n'
            + '\t\t</panel>\n'
            + '\t\t<field type="info">\n'
            + '\t\t\tHex patch converted by ' + client.user.name + '.\n'
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
        data = await create_future(bytes, output, "utf-8")
        b = io.BytesIO(data)
        f = discord.File(b, filename="patch.xml")
        create_task(sendFile(channel, "Patch successfully converted!", f))


class CS_npc(Command):
    time_consuming = True
    min_level = 0
    description = "Searches the Cave Story NPC list for an NPC by name or ID."
    usage = "<query> <condensed(?c)>"
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, bot, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(entity_list.search, argv, lim)
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
                response = ["Search results for **" + argv + "**:"]
                lines = output.split("\n")
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            else:
                return "Search results for **" + argv + "**:\n```\n" + output + "```"
        else:
            raise LookupError("No results found for " + argv + ".")


class CS_tsc(Command):
    min_level = 0
    description = "Searches the Cave Story OOB flags list for a memory variable."
    usage = "<query> <condensed(?c)>"
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(tsc_list.search, argv, lim)
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
                response = ["Search results for **" + argv + "**:"]
                lines = output.split("\n")
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            else:
                return "Search results for **" + argv + "**:\n```\n" + output + "```"
        else:
            raise LookupError("No results found for " + argv + ".")


class CS_mod(Command):
    time_consuming = True
    name = ["CS_search"]
    min_level = 0
    description = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
    usage = "<query>"
    no_parse = True
    rate_limit = 3

    async def __call__(self, args, **void):
        argv = " ".join(args)
        data = await searchForums(argv)
        data += await create_future(douclub.search, argv)
        if len(data):
            response = "Search results for **" + argv + "**:\n"
            for l in data:
                line = (
                    "\n<" + str(l["url"]) + ">\n"
                    + "```css\nName: [" + noHighlight(l["name"])
                    + "]\nAuthor: [" + noHighlight(l["author"].strip(" "))
                    + "]\n" + limStr(l["description"].replace("\n", " "), 128)
                    + "```\r"
                )
                response += line
            if len(response) < 20000 and len(response) > 1900:
                output = response.split("\r")
                response = []
                curr = ""
                for line in output:
                    if len(curr) + len(line) > 1900:
                        response.append(curr)
                        curr = line
                    else:
                        curr += line
            return response
        else:
            raise LookupError("No results found for " + argv + ".")

    async def _ready_(self, **void):
        knackpy.__builtins__["print"] = lambda *args, **kwargs: None


class CS_Database(Database):
    name = "cs_database"
    no_file = True

    async def __call__(self, **void):
        entity_list.update()
        tsc_list.update()
        douclub.update()


class Dogpile(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically imitate users when 3+ of the same messages are posted in a row."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"
    rate_limit = 0.5

    async def __call__(self, flags, guild, **void):
        update = self.data.dogpiles.update
        bot = self.bot
        following = bot.data.dogpiles
        curr = following.get(guild.id, False)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
                update()
            return "```css\nDisabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        elif "e" in flags or "a" in flags:
            following[guild.id] = True
            update()
            return "```css\nEnabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        else:
            return (
                "```ini\nDogpile imitating is currently " + "not " * (not curr)
                + "enabled in [" + noHighlight(guild.name) + "].```"
            )


class UpdateDogpiles(Database):
    name = "dogpiles"

    def __load__(self):
        self.msgFollow = {}

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None or not orig:
            return
        g_id = message.guild.id
        following = self.data
        if g_id in following:
            u_id = message.author.id
            c_id = message.channel.id
            if not edit:
                if following[g_id]:
                    checker = orig
                    curr = self.msgFollow.get(c_id)
                    if curr is None:
                        curr = [checker, 1, 0]
                        self.msgFollow[c_id] = curr
                    elif checker == curr[0] and u_id != curr[2]:
                        curr[1] += 1
                        if curr[1] >= 3:
                            curr[1] = xrand(-3) + 1
                            if len(checker):
                                create_task(message.channel.send(checker))
                    else:
                        if len(checker) > 100:
                            checker = ""
                        curr[0] = checker
                        curr[1] = xrand(-1, 2)
                    curr[2] = u_id
                    #print(curr)


class MathQuiz(Command):
    name = ["MathTest"]
    min_level = 1
    description = "Starts a math quiz in the current channel."
    usage = "<mode(easy)(hard)> <disable(?d)>"
    flags = "aed"
    rate_limit = 3

    async def __call__(self, channel, guild, flags, argv, **void):
        if not self.bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server for this command.")
        mathdb = self.bot.database.mathtest
        if "d" in flags:
            if channel.id in mathdb.data:
                mathdb.data.pop(channel.id)
            return "```css\nDisabled math quizzes for " + sbHighlight(channel.name) + ".```"
        if not argv:
            argv = "easy"
        elif argv not in ("easy", "hard"):
            raise TypeError("Invalid quiz mode.")
        mathdb.data[channel.id] = cdict(mode=argv, answer=None)
        return "```css\nEnabled " + argv + " math quiz for " + sbHighlight(channel.name) + ".```"


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
    
    def addition(self):
        x = xrand(100, 10000)
        y = xrand(100, 10000)
        s = self.format(x, y, "+")
        return s, x + y
    
    def subtraction(self):
        x = xrand(100, 12000)
        y = xrand(100, 8000)
        if x < y:
            x, y = y, x
        s = self.format(x, y, "-")
        return s, x - y

    def multiplication(self):
        x = xrand(2, 20)
        y = xrand(2, 20)
        s = self.format(x, y, "×")
        return s, x * y

    def multiplication2(self):
        x = xrand(13, 100)
        y = xrand(13, 100)
        s = self.format(x, y, "×")
        return s, x * y

    def division(self):
        y = xrand(2, 20)
        x = xrand(2, 14) * y
        s = self.format(x, y, "/")
        return s, x // y

    def exponentiation(self):
        x = xrand(2, 20)
        y = xrand(2, max(3, 14 / x))
        s = str(x) + "^" + str(y)
        return s, x ** y

    def exponentiation2(self):
        x = xrand(2, 4)
        if x == 2:
            y = xrand(7, 35)
        else:
            y = xrand(5, 11)
        s = str(x) + "^" + str(y)
        return s, x ** y
        
    def square_root(self):
        x = xrand(2, 20)
        y = x ** 2
        s = "√" + str(y)
        return s, x

    def square_root2(self):
        x = xrand(21, 1000)
        y = x ** 2
        s = "√" + str(y)
        return s, x
        
    def scientific(self):
        x = xrand(100, 10000)
        x /= 10 ** int(math.log10(x))
        y = xrand(-3, 6)
        s = str(x) + "×10^" + str(y)
        return s, round(x * 10 ** y, 9)
        
    def fraction(self):
        y = random.choice([2, 4, 5, 10])
        x = xrand(3, 20)
        mult = xrand(4) + 1
        y *= mult
        x *= mult
        s = self.format(x, y, "/")
        return s, round(x / y, 9)

    def recurring(self):
        x = "".join(str(xrand(10)) for _ in loop(xrand(2, 4)))
        s = "0." + "".join(x[i % len(x)] for i in range(28)) + "..."
        ans = "0.[" + x + "]"
        return s, ans

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
        q = await create_future(sympy.expand, st)
        q = self.eqtrans(q).replace("∙", "") + " = 0"
        return q, a

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
                s.append(str(a) + "*" + random.choice(op) + "(x)")
        st = ""
        for i in s:
            if st and i[0] not in "+-":
                st += "+"
            st += i
        ans = await self.bot.solveMath(st, xrand(2147483648), 0, 1)
        a = ans[0]
        q = self.eqtrans(a)
        if xrand(2):
            q = "Dₓ " + q
            op = sympy.diff
        else:
            q = "∫ " + q
            op = sympy.integrate
        a = await create_future(op, a)
        return q, a

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
        qa = random.choice(modes[mode])()
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

    async def _nocommand_(self, message, **void):
        bot = self.bot
        channel = message.channel
        if channel.id in self.data:
            if message.author.id != bot.client.user.id:
                msg = message.content.strip("|").strip("`")
                if not msg or msg.lower() != msg:
                    return
                if msg.startswith("#") or msg.startswith("//") or msg.startswith("\\"):
                    return
                try:
                    x = await bot.solveMath(msg, message.author, 0, 1)
                    x = await create_future(sympy.sympify, x[0])
                except:
                    return
                correct = False
                a = self.data[channel.id].answer
                if type(a) is list:
                    if x in a:
                        correct = True
                else:
                    a = await create_future(sympy.sympify, a)
                    d = await create_future(sympy.Add, x, -a)
                    z = await create_future(sympy.simplify, d)
                    correct = z == 0
                if correct:
                    create_task(self.newQuestion(channel))
                    await channel.send("Great work!")
                else:
                    await channel.send("Oops! Not quite, try again!")


class DeviantArt(Command):
    server_only = True
    min_level = 2
    description = "Subscribes to a DeviantArt Gallery, reposting links to all new posts."
    usage = "<reversed(?r)> <disable(?d)>"
    flags = "raed"
    rate_limit = 4

    async def __call__(self, argv, flags, channel, guild, bot, **void):
        if not bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server to subscribe to DeviantArt Galleries.")
        data = bot.data.deviantart
        update = bot.database.deviantart.update
        if not argv:
            assigned = data.get(channel.id, ())
            if not assigned:
                return "```ini\nNo currently subscribed DeviantArt Galleries for [#" + noHighlight(channel) + "].```"
            if "d" in flags:
                try:
                    data.pop(channel.id)
                except KeyError:
                    pass
                return "```css\nSuccessfully removed all DeviantArt Gallery subscriptions from [#" + noHighlight(channel) + "].```"
            return "Currently subscribed DeviantArt Galleries for [#" + noHighlight(channel) + "]:```ini" + strIter(assigned, key=lambda x: x["user"]) + "```"
        urls = await bot.followURL(argv)
        if not urls:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        if "deviantart.com" not in url:
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        url = url[url.index("deviantart.com") + 15:]
        spl = url.split("/")
        user = spl[0]
        if spl[1] != "gallery":
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        content = spl[2].split("&")[0]
        folder = noHighlight(spl[-1].split("&")[0])
        try:
            content = int(content)
        except (ValueError, TypeError):
            if content in (user, "all"):
                content = user
            else:
                raise TypeError("Invalid Gallery type.")
        if content in self.data.get(channel.id, {}):
            raise KeyError("Already subscribed to " + user + ": " + folder)
        if "d" in flags:
            try:
                data.get(channel.id).pop(content)
            except KeyError:
                raise KeyError("Not currently subscribed to " + user + ": " + folder)
            else:
                if channel.id in data and not data[channel.id]:
                    data.pop(channel.id)
                update()
                return "```css\nSuccessfully unsubscribed from " + sbHighlight(user) + ": " + sbHighlight(folder) + ".```"
        setDict(data, channel.id, {}).__setitem__(content, {"user": user, "type": "gallery", "reversed": ("r" in flags), "entries": {}})
        update()
        return "```css\nSuccessfully subscribed to " + sbHighlight(user) + ": " + sbHighlight(folder) + ", posting in reverse order" * ("r" in flags) + ".```"


class UpdateDeviantArt(Database):
    name = "deviantart"

    async def processPart(self, found, c_id):
        bot = self.bot
        try:
            channel = await bot.fetch_channel(c_id)
            if not bot.isTrusted(channel.guild.id):
                raise LookupError
        except LookupError:
            self.data.pop(c_id)
            return
        try:
            assigned = self.data.get(c_id)
            if assigned is None:
                return
            embs = deque()
            for content in assigned:
                items = found[content]
                entries = assigned[content]["entries"]
                new = tuple(items)
                orig = tuple(entries)
                if hash(tuple(sorted(new))) != hash(tuple(sorted(orig))):
                    if assigned[content].get("reversed", False):
                        it = reversed(new)
                    else:
                        it = new
                    for i in it:
                        if i not in entries:
                            entries[i] = True
                            self.update()
                            home = "https://www.deviantart.com/" + items[i][2]
                            emb = discord.Embed(
                                colour=discord.Colour(1),
                                description="🔔 New Deviation from " + items[i][2] + " 🔔\n" + items[i][0],
                            ).set_image(url=items[i][1]).set_author(name=items[i][2], url=home, icon_url=items[i][3])
                            embs.append(emb)
                    for i in orig:
                        if i not in items:
                            entries.pop(i)
                            self.update()
        except:
            print(traceback.format_exc())
        else:
            bot.embedSender(channel, embs)

    async def __call__(self):
        t = setDict(self.__dict__, "time", 0)
        if utc() - t < 300:
            return
        self.time = inf
        conts = {i: a[i]["user"] for a in tuple(self.data.values()) for i in a}
        found = {}
        base = "https://www.deviantart.com/_napi/da-user-profile/api/gallery/contents?limit=24&username="
        attempts, successes = 0, 0
        for content, user in conts.items():
            if type(content) is str:
                f_id = "&all_folder=true&mode=oldest"
            else:
                f_id = "&folderid=" + str(content)
            items = {}
            try:
                url = base + user + f_id + "&offset="
                for i in range(0, 13824, 24):
                    req = url + str(i)
                    # print(req)
                    attempts += 1
                    resp = await Request(req, timeout=16, aio=True)
                    try:
                        d = json.loads(resp)
                    except:
                        d = eval(resp, {}, eval_const)
                    for res in d["results"]:
                        deviation = res["deviation"]
                        media = deviation["media"]
                        prettyName = media["prettyName"]
                        orig = media["baseUri"]
                        extra = ""
                        token = "?token=" + media["token"][0]
                        for t in reversed(media["types"]):
                            if t["t"].lower() == "fullview":
                                if "c" in t:
                                    extra = "/" + t["c"].replace("<prettyName>", prettyName)
                                    break
                        image_url = orig + extra + token
                        items[deviation["deviationId"]] = (deviation["url"], image_url, deviation["author"]["username"], deviation["author"]["usericon"])
                    successes += 1
                    if not d.get("hasMore", None):
                        break
            except:
                print(traceback.format_exc())
            else:
                found[content] = items
        # if attempts:
        #     print(successes, "of", attempts, "DeviantArt requests executed successfully.")
        for c_id in tuple(self.data):
            create_task(self.processPart(found, c_id))
        self.time = utc()