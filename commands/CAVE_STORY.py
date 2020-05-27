try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import csv, knackpy, ffmpy
from prettytable import PrettyTable as ptable

FFRuntimeError = ffmpy.FFRuntimeError
getattr(knackpy, "__builtins__", {})["print"] = print
getattr(ffmpy, "__builtins__", {})["print"] = print


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
            create_future(self.pull)
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
try:
    douclub = DouClub(auth["knack_id"], auth["knack_secret"])
except KeyError:
    douclub = freeClass(
        search=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to use Doukutsu Club.")'),
    )
    print("WARNING: knack_id/knack_secret not found. Unable to use Doukutsu Club.")


def searchForums(query):
        
    url = (
        "https://www.cavestory.org/forums/search/1/?q=" + query.replace(" ", "+")
        + "&t=post&c[child_nodes]=1&c[nodes][0]=33&o=date&g=1"
    )
    s = urlOpen(url).read().decode("utf-8", "replace")
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
            create_future(self.pull)
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


def getDuration(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]
    try:
        output = subprocess.check_output(command).decode("utf-8", "replace")
    except:
        print(traceback.format_exc())
        output = "N/A"
    try:
        i = output.index("\r")
        output = output[:i]
    except ValueError:
        output = "N/A"
    if output == "N/A":
        n = 0
    else:
        n = roundMin(float(output))
    return max(1 / (1 << 24), n)


def orgConv(org, wave, fmt, key="temp", fl=8388608):
    try:
        try:
            os.remove("cache/" + key + ".org")
        except FileNotFoundError:
            pass
        try:
            os.remove("cache/" + key + ".xm")
        except FileNotFoundError:
            pass
        opener = urlBypass()
        opener.retrieve(org, "cache/" + key + ".org")
        if wave is not None:
            opener.retrieve(wave, "cache/" + key + ".dat")
            com = "org2xm ../cache/" + key + ".org ../cache/" + key + ".dat"
        else:
            com = "org2xm ../cache/" + key + ".org ORG210EN.DAT"
        os.chdir("misc")
        try:
            os.system(com)
            os.chdir("..")
        except:
            os.chdir("..")
            raise
        fi = "cache/" + key + ".xm"
        t = time.time()
        while time.time() - t < 12:
            time.sleep(0.2)
            if "&" + key + ".xm" in os.listdir("cache"):
                try:
                    if os.path.getsize(fi) >= 32:
                        break
                except Exception as ex:
                    print(repr(ex))                
                    pass
        if time.time() - t >= 12:
            try:
                os.remove("cache/" + key + ".org")
            except (FileNotFoundError, PermissionError):
                pass
            try:
                os.remove("cache/" + key + ".xm")
            except (FileNotFoundError, PermissionError):
                pass
            raise TimeoutError("Request timed out.")
        if fmt != "xm":
            fn = "cache/" + key + "." + fmt
            try:
                os.remove(fn)
            except (FileNotFoundError, PermissionError):
                pass
            dur = getDuration(fi)
            br = max(32, min(256, floor(((fl - 262144) / dur / 128) / 4) * 4))
            ff = ffmpy.FFmpeg(
                global_options=["-y", "-hide_banner", "-loglevel panic"],
                inputs={fi: None},
                outputs={str(br) + "k": "-vn -b:a", fn: None},
            )
            ff.run()
            try:
                os.remove("cache/" + key + ".org")
            except (FileNotFoundError, PermissionError):
                pass
            try:
                os.remove("cache/" + key + ".xm")
            except (FileNotFoundError, PermissionError):
                pass
        else:
            fn = fi
        return [fn]
    except Exception as ex:
        print(traceback.format_exc())
        return ex


class CS_org2xm(Command):
    time_consuming = True
    fmts = [
        "mp3",
        "ogg",
        "xm",
        "webm"
        "wav"
    ]
    name = ["CS_o2x", "Org2xm", "Convert_org"]
    min_level = 0
    description = "Converts a .org file to another file format."
    usage = "<0:org_url{attached_file}> <2:wave_url[]> <1:out_format[xm]>"
    rate_limit = 5

    async def __call__(self, args, bot, message, channel, guild, **void):
        if not bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server to convert .org files.")
        if len(message.attachments):
            org = message.attachments[0].url
            args = [""] + args
        else:
            org = await bot.followURL(verifyURL(args[0]))
        if len(args) > 2:
            wave = await bot.followURL(verifyURL(args[1]))
        else:
            wave = None
            #wave = "https://cdn.discordapp.com/attachments/313292557603962881/674183355972976660/ORG210EN.DAT"
            #wave = "https://cdn.discordapp.com/attachments/317898572458754049/674166849763672064/wave100"
        if len(args) > 1:
            fmt = args[-1]
        else:
            fmt = "xm"
        if fmt not in self.fmts:
            raise TypeError(fmt + " is not a supported output format.")
        name = org.split("/")[-1].replace(".org", "") + "." + fmt
        fn = await create_future(
            orgConv,
            org,
            wave,
            fmt,
            "%" + str(guild.id),
            guild.filesize_limit
        )
        if issubclass(type(fn), Exception):
            raise fn
        f = discord.File(fn[0], filename=name)
        create_task(sendFile(channel, "Org successfully converted!", f, fn[0]))


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

    async def __call__(self, bot, args, guild, **void):
        if len(args) < 2:
            return "```css\n" + _m2f(args[0], 1) + "```"
        num = await bot.evalMath(" ".join(args[1:]), guild.id)
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
    rate_limit = 2

    async def __call__(self, args, **void):
        argv = " ".join(args)
        data = await create_future(searchForums, argv)
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
