import requests, csv, time, knackpy, ast, discord, urllib, asyncio, os, ffmpy
from prettytable import PrettyTable as ptable
from smath import *


class urlBypass(urllib.request.FancyURLopener):
    version = "Mozilla/5." + str(xrand(1, 10))


class DouClub:
    
    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec
        self.pull()

    def pull(self):
        kn = knackpy.Knack(obj="object_1", app_id=self.id, api_key=self.secret)
        self.data = [kn.data, time.time()]

    def search(self, query):
        if time.time() - self.data[1] > 720:
            doParallel(self.pull)
        output = []
        query = query.lower()
        for l in self.data[0]:
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
douclub = DouClub(auth["knack_id"], auth["knack_secret"])


def searchForums(query):
        
    url = (
        "https://www.cavestory.org/forums/search/1/?q=" + query.replace(" ", "+")
        + "&t=post&c[child_nodes]=1&c[nodes][0]=33&o=date&g=1"
        )
    opener = urlBypass()
    resp = opener.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error " + str(resp.getcode()))
    s = resp.read().decode("utf-8")

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
            while len(temp) > 7:
                try:
                    i = temp.index("&#")
                    if temp[i + 2] == "x":
                        h = "0x"
                        p = i + 3
                    else:
                        h = ""
                        p = i + 2
                    for a in range(4):
                        if temp[p + a] == ";":
                            v = int(h + temp[p:p + a])
                            break
                    c = chr(v)
                    temp = temp[:i] + c + temp[p + a + 1:]
                except ValueError:
                    break
            curr[elem] = temp
        output.append(curr)
    return output
    

class SheetPull:
    
    def __init__(self, url):
        self.url = url
        self.pull()

    def pull(self):
        url = self.url
        text = requests.get(url).text
        data = text.split("\r\n")
        columns = 0
        sdata = [[], time.time()]
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

    def search(self, query, lim):
        if time.time() - self.data[1] > 60:
            doParallel(self.pull)
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
                            output.append(temp)
        return output


entity_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYH\
hC56mPXCnCcsDjxit0/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0"
)
tsc_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i\
-rjRjgMW04Gdz9EO6U/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0"
)


##def orgConv(org, wave100, fmt="mp3"):
##    resp = opener.open(url)
##    if resp.getcode() != 200:
##        raise ConnectionError("Error " + str(resp.getcode()))
##    data = list(resp.read())
##    for i in range(len(data) >> 8):
##        tempf = open("cache/wave100/wave"+str(i)+".wav","wb")
##        outA = [82,73,70,70,36,1,0,0,87,65,86,69,102,109,116,32,16,0,0,0,1,0,1,0,172,130,0,0,
##                88,5,1,0,2,0,16,0,100,97,116,97,0,1,0,0]
##        outA += data[i << 8:i * 256 + 256]
##        for f in range(44, len(outA)):
##            if outA[f] < 0:
##                outA[f] += 256
##        outF = bytes(outA)
##        tempf.write(outF)
##        tempf.close
##    os.system("org2xm cache/temp.org 
##    return


def orgConv(org, wave, fmt):
    try:
        try:
            os.remove("cache/temp.org")
        except FileNotFoundError:
            pass
        try:
            os.remove("cache/temp.xm")
        except FileNotFoundError:
            pass
        opener = urlBypass()
        opener.retrieve(wave, "cache/temp.dat")
        opener.retrieve(org, "cache/temp.org")
        os.system("org2xm cache/temp.org cache/temp.dat")
        fi = "cache/temp.xm"
        t = time.time()
        while time.time() - t < 12:
            time.sleep(0.01)
            if "temp.xm" in os.listdir("cache"):
                try:
                    f = open(fi, "rb")
                    f.read(32)
                    f.close()
                    break
                except Exception as ex:
                    print(repr(ex))                
                    pass
        if fmt != "xm":
            fn = "cache/temp." + fmt
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
            ff = ffmpy.FFmpeg(
                global_options=["-y", "-hide_banner", "-loglevel panic"],
                inputs={fi: None},
                outputs={"160k": "-b:a", fn: None},
                )
            ff.run()
        else:
            fn = fi
        return fn
    except Exception as ex:
        return repr(ex)


class cs_org2xm:
    is_command = True
    fmts = [
        "mp3",
        "ogg",
        "xm",
        ]

    def __init__(self):
        self.name = ["org2xm", "convert_org"]
        self.min_level = 0
        self.description = "Converts a .org file to another file format."
        self.usage = "<0:org_url{attached_file}> <2:wave_url[]> <1:out_format[xm]>"

    async def __call__(self, args, _vars, message, channel, **void):
        if len(message.attachments):
            org = message.attachments[0].url
            args = [""] + args
        else:
            org = args[0]
        if len(args) > 2:
            wave = _vars.verifyURL(args[1])
        else:
            wave = "https://cdn.discordapp.com/attachments/313292557603962881/674183355972976660/ORG210EN.DAT"
            #wave = "https://cdn.discordapp.com/attachments/317898572458754049/674166849763672064/wave100"
        if len(args) > 1:
            fmt = args[-1]
        else:
            fmt = "xm"
        if fmt not in self.fmts:
            raise TypeError(fmt + " is not a supported output format.")
        returns = [None]
        doParallel(orgConv, [org, wave, fmt], returns)
        t = time.time()
        while returns[0] is None and time.time() - t < _vars.timeout - 1:
            await asyncio.sleep(0.01)
        fn = returns[0]
        if fn is None:
            raise TimeoutError("Request timed out.")
        try:
            f = discord.File(fn)
        except:
            raise eval(fn)
        return {
            "content": "Org successfully converted!",
            "file": f,
            }


def _m2f(mem, val):
    val1 = mem
    val2 = val
    curr = 0
    result = ""
    while val2:
        difference = int(val1, 16) - 4840864 + curr / 8
        flag = difference * 8
        offset = max(0, int((-flag + 999.9) / 1000))
        flag += offset * 1000
        output = ""
        for i in range(0, 3):
            a = 10 ** i
            b = int(flag / a)
            char = b % 10
            char += 48
            output += chr(char)
        char = int(flag / 1000)
        char += 48
        char -= offset
        if val2 & 1:
            operation = "+"
        else:
            operation = "-"
        try:
            output += chr(char)
            output = "<FL" + operation + output[::-1]
        except ValueError:
            output = "<FL" + operation + "(0x" + hex((char + 256) & 255).upper()[2:] + ")" + output[::-1]
        result += output
        val2 >>= 1
        curr += 1
    return result


class cs_mem2flag:
    is_command = True

    def __init__(self):
        self.name = ["cs_m2f"]
        self.min_level = 0
        self.description = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
        self.usage = "<0:address> <1:value[1]>"

    async def __call__(self, _vars, args, **void):
        if len(args) < 2:
            return "```css\n" + _m2f(args[0], 1) + "```"
        return "```css\n" + _m2f(args[0], _vars.evalMath(" ".join(args[1:]))) + "```"


class cs_hex2xml:
    is_command = True

    def __init__(self):
        self.name = ["cs_h2x"]
        self.min_level = 0
        self.description = "Converts a given Cave Story hex patch to an xml file readable by Booster's Lab."
        self.usage = "<hex_data>"

    async def __call__(self, argv, channel, **void):
        hacks = {}
        hack = argv.replace(" ", "").replace("`", "")
        while len(hack):
            try:
                i = hack.index("0x")
            except ValueError:
                break
            hack = hack[i:]
            i = hack.index("\n")
            offs = hack[:i]
            hack = hack[i+1:]
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
            + '\t\t\tHex patch converted by Miza.\n'
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
        fn = "cache/temp.xml"
        f = open(fn, "w")
        f.write(output)
        f.close()
        f = discord.File(fn)
        print(fn)
        return {
            "content": "Hack successfully converted!",
            "file": f,
            }


class cs_npc:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Searches the Cave Story NPC list for an NPC by name or ID."
        self.usage = "<query>"

    async def __call__(self, _vars, args, flags, **void):
        lim = ("c" in flags) * 40 + 20
        argv = " ".join(args)
        data = entity_list.search(argv, lim)
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
                curr = "```css\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```css\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            else:
                return "Search results for **" + argv + "**:\n```\n" + output + "```"
        else:
            raise EOFError("No results found for " + uniStr(argv) + ".")


class cs_tsc:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Searches the Cave Story OOB flags list for a memory variable."
        self.usage = "<query>"

    async def __call__(self, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = tsc_list.search(argv, lim)
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
                curr = "```css\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```css\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            else:
                return "Search results for **" + argv + "**:\n```\n" + output + "```"
        else:
            raise EOFError("No results found for " + uniStr(argv) + ".")


class cs_mod:
    is_command = True

    def __init__(self):
        self.name = ["cs_search"]
        self.min_level = 0
        self.description = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
        self.usage = "<query>"

    async def __call__(self, args, **void):
        argv = " ".join(args)
        resp = [None]
        doParallel(searchForums, [argv], resp)
        data = douclub.search(argv)
        t = time.time()
        while resp[0] is None and time.time() - t < 5:
            await asyncio.sleep(0.01)
        if resp[0] is not None:
            data += resp[0]
        print(data)
        if len(data):
            response = "Search results for **" + argv + "**:\n"
            for l in data:
                line = (
                    "\n<" + str(l["url"]) + ">\n"
                    + "```css\nName: " + uniStr(l["name"])
                    + "\nAuthor: " + uniStr(l["author"])
                    + "\n" + limStr(l["description"].replace("\n", " "), 128)
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
            raise EOFError("No results found for " + uniStr(argv) + ".")
