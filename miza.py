import discord,pygame,urllib.request,nekos,ast,sys,asyncio,datetime,requests,json,csv
from prettytable import PrettyTable as ptable
from matplotlib import pyplot as plt
from googletrans import Translator
from smath import *
import Char2Emoj
import op_items_generator
import mem2flag

client = discord.Client(
    max_messages=2000,
    activity=discord.Activity(name="Magic"),
    )

class _globals:
    def __init__(self):
        self.owner_id = 201548633244565504
        self.TIMEOUT_DELAY = 10
        self.op = self.urlBypass()
        self.disabled = [
            "__",
            "pygame",
            "open",
            "import",
            "urllib",
            "discord",
            "http",
            "exec",
            "eval",
            "locals",
            "globals",
            "vars",
            "client",
            "doMath",
            "sys",
            "async",
            "copyGlobals",
            "_init",
            "compile",
            "threading",
            "input",
            "exit",
            "quit",
            "getattr",
            ]
        self.image_forms = [
            "gif",
            "png",
            "bmp",
            "jpg",
            "jpeg",
            "tiff",
            ]
        self.commands = {
            "help":[-inf,"Shows a list of usable commands.","`~help l(0) v(?v)`"],
            "math":[0,"Evaluates a math function or formula.","`~math f`"],
            "join":[0,"Joins the current voice channel.","`~join`"],
            "play":[0,"Plays an audio file from a link, into a voice channel.","`~play l`"],
            "q":[0,"Retrieves a list of currently playing audio.","`~q`"],
            "rem":[0,"Removes an entry in the audio queue.","`~rem n(0)`"],
            "neko":[1,"",""],
            "lewd_neko":[1,"",""],
            "lewd":[1,"",""],
            "uni2hex":[0,"Converts unicode text to hexadecimal numbers.","`~uni2hex a`"],
            "hex2uni":[0,"Converts hexadecimal numbers to unicode text.","`~hex2uni b`"],
            "translate":[0,"Translates a string into another language.","`~translate l(en) s`"],
            "translate2":[0,"Translates a string into another language and then back.","`~translate2 l(en) s`"],
            "char2emoj":[0,"Makes emoji blocks using strings.","`~char2emoj s a b`"],
            "cs_mem2flag":[0,"Returns the TSC representation of a Cave Story flag \
corresponding to a certain memory address and value.","`~cs_mem2flag m n(1)`"],
            "cs_npc":[0,"Searches the Cave Story NPC list for an NPC by name or ID.","`~cs_npc x`"],
            "cs_tsc":[0,"Searches the Cave Story OOB flags list for a memory variable.","`~cs_tsc x`"],
            "mc_ench":[0,"Returns a Minecraft command that generates an item with target enchants.","`~mc_ench i e`"],
            "img_scale2x":[0,"Performs Scale2x on an image from an embed link.","`~img_scale2x l`"],
            "clearcache":[1,"Clears all cached data.","`~clearcache`"],
            "changeperms":[0,"Changes a user's permissions.","`~changeperms u n s(0)`"],
            "purge":[1,"Deletes a number of messages from bot in current channel.","`~purge c(1) h(?h)`"],
            "purgeU":[3,"Deletes a number of messages from a user in current channel.","`~purgeU u c(1) h(?h)`"],
            "purgeA":[3,"Deletes a number of messages from all users in current channel.","`~purgeA c(1) h(?h)`"],
            "ban":[3,"Bans a user for a certain amount of hours, with an optional message.","`~ban u t(0) m()`"],
            "shutdown":[3,"Shuts down the bot.","`~shutdown`"],
            
            "tr":[0,"",""],
            "tr2":[0,"",""],
            "perms":[0,"",""],
            }
        self.lastCheck = time.time()
        self.queue = []
    class urlBypass(urllib.request.FancyURLopener):
        version = "Mozilla/5.2"

bar_plot = plt.bar
plot_points = plt.scatter
plot = plt.plot
plotY = plt.semilogy
plotX = plt.semilogx
plotXY = plt.loglog
fig = plt.figure()

class PapagoTrans:
    def __init__(self,c_id,c_sec):
        self.id = c_id
        self.secret = c_sec
    def translate(self,string,source,dest):
        url = "https://openapi.naver.com/v1/papago/n2mt"
        enc = urllib.parse.quote(string)
        data = "source="+source+"&target="+dest+"&text="+enc
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id",self.id)
        req.add_header("X-Naver-Client-Secret",self.secret)
        resp = urllib.request.urlopen(req,data=data.encode("utf-8"),timeout=_vars.TIMEOUT_DELAY/2)
        if resp.getcode() != 200:
            raise ConnectionError("Error "+str(resp.getcode()))
        r = resp.read().decode("utf-8")
        try:
            r = json.loads(r)
        except:
            pass
        output = r
        return output["message"]["result"]["translatedText"]

def pull_douclub(argv):
    c_id = _vars.auth["knack_id"]
    c_sec = _vars.auth["knack_secret"]
    baseurl = "https://api.knack.com/v1/pages/"#scene_xx/views/view_yy/records"
    items = argv.replace(" ","%20").lower()
    baseurl = "https://doukutsuclub.knack.com/database#search-database/?view_22_search="
    url = baseurl+items+"&view_22_page=1"
    print(url)
    resp = _vars.op.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    s = resp.read().decode("utf-8")

    return s
        
class SheetPull:
    def __init__(self,url):
        text = requests.get(url).text
        data = text.split("\r\n")
        columns = 0
        self.data = []
        for i in range(len(data)):
            line = data[i]
            read = list(csv.reader(line))
            reli = []
            curr = ""
            for j in read:
                if len(j)>=2 and j[0]==j[1]=="":
                    if curr != "":
                        reli.append(curr)
                        curr = ""
                else:
                    curr += "".join(j)
            if curr != "":
                reli.append(curr)
            if len(reli):
                columns = max(columns,len(reli))
                self.data.append(reli)
            for line in range(len(self.data)):
                while len(self.data[line]) < columns:
                    self.data[line].append(" "*line)
    def search(self,query,lim):
        output = []
        query = query.lower()
        try:
            int(query)
            mode = 0
        except:
            mode = 1
        if not mode:
            for l in self.data:
                if l[0] == query:
                    temp = [limLine(e,lim) for e in l]
                    output.append(temp)
        else:
            qlist = query.split(" ")
            for q in qlist:
                for l in self.data:
                    if len(l) >= 3:
                        for i in l:
                            found = False
                            if q in i.lower():
                                found = True
                            if found:
                                temp = [limLine(e,lim) for e in l]
                                output.append(temp)
        return output

def verifyCommand(func):
    _f = func.lower()
    for d in _vars.disabled:
        if d in _f:
            raise PermissionError("Issued command is not enabled.")
    return func

def verifyURL(_f):
    _f = _f.replace("<","").replace(">","").replace("|","").replace("*","").replace("_","").replace("`","")
    return _f
    
def pull_e621(argv):
    items = argv.replace(" ","%20").lower()
    baseurl = "https://e621.net/post/index/"
    url = baseurl+"1/"+items
    resp = _vars.op.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    s = resp.read().decode("utf-8")
    
    ind = s.index('class="next_page" rel="next"')
    s = s[ind-90:ind]
    d = s.split(" ")
    i = -1
    while True:
        if "</a>" in d[i]:
            break
        i -= 1
    u = d[i][:-4]
    u = u[u.index(">")+1:]
    v1 = xrand(1,int(u))

    url = baseurl+str(v1)+"/"+items
    resp = _vars.op.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    s = resp.read().decode("utf-8")

    try:
        limit = s.index('class="next_page" rel="next"')
        s = s[:limit]
    except:
        pass

    search = '<a href="/post/show/'
    sources = []
    while True:
        try:
            ind1 = s.index(search)
            s = s[ind1+len(search):]
            ind2 = s.index('"')
            target = s[:ind2]
            try:
                sources.append(int(target))
            except:
                pass
        except:
            break
    v2 = xrand(len(sources))
    x = sources[v2]
    url = "https://e621.net/post/show/"+str(x)
    resp = _vars.op.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    s = resp.read().decode("utf-8")

    search = '<a href="https://static1.e621.net/data/'
    ind1 = s.index(search)
    s = s[ind1+9:]
    ind2 = s.index('"')
    s = s[:ind2]
    url = s
    return [url,v1,v2]

def pull_rule34(argv):
    items = argv.split("_")
    for i in range(len(items)):
        items[i] = items[i][0].upper()+items[i][1:].lower()
    items = "_".join(items)
    items = argv.split(" ")
    for i in range(len(items)):
        items[i] = items[i][0].upper()+items[i][1:]
    items = "%20".join(items)
    baseurl = "https://rule34.paheal.net/post/list/"
    try:
        url = baseurl+items+"/1"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req,timeout=_vars.TIMEOUT_DELAY/2)
    except:
        url = baseurl+items.upper()+"/1"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req,timeout=_vars.TIMEOUT_DELAY/2)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    
    s = resp.read().decode("utf-8")
    try:
        ind = s.index('">Last</a><br>')
        s = s[ind-5:ind]
        v1 = xrand(1,int(s.split("/")[-1]))
        url = url[:-1]+str(v1)
        
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req,timeout=_vars.TIMEOUT_DELAY/2)
        if resp.getcode() != 200:
            raise ConnectionError("Error "+str(resp.getcode()))
        s = resp.read().decode("utf-8")
    except:
        pass
    try:
        limit = s.index("class=''>Images</h3><div class='blockbody'>")
        s = s[limit:]
        limit = s.index("</div></div></section>")
        s = s[:limit]
    except:
        pass

    search = 'href="'
    sources = []
    while True:
        try:
            ind1 = s.index(search)
            s = s[ind1+len(search):]
            ind2 = s.index('"')
            target = s[:ind2]
            if not "." in target:
                continue
            elif ".js" in target:
                continue
            found = False
            for i in image_forms:
                if i in target:
                    found = True
            if target[0]=="h" and found:
                sources.append(target)
        except:
            break
    v2 = xrand(len(sources))
    url = sources[v2]
    return [url,v1,v2]

def searchRandomNSFW(argv):
    nsfw = {
        0:pull_rule34,
        1:pull_e621,
        }
    data = None
    while True:
        l = list(nsfw)
        if not l:
            break
        r = xrand(len(l))
        f = nsfw[r]
        nsfw.pop(r)
        try:
            data = f(argv)
        except:
            continue
        break
    if data is None:
        raise EOFError("Unable to locate any search results.")
    return data

def copyGlobals():
    g = dict(globals())
    for i in _vars.disabled:
        try:
            g.pop(i)
        except:
            pass
    return g

def doMath(_f,returns):
    try:
        try:
            answer = eval(_f,_vars.stored_vars)
        except:
            exec(_f,_vars.stored_vars)
            answer = None
    except Exception as ex:
        answer = "\nError: "+repr(ex)
    if answer is not None:
        answer = str(answer)
    returns[0] = answer

def getUserPerms(guild,user):
    u_id = user.id
    if guild:
        g_id = guild.id
        g_perm = _vars.perms.get(g_id,{})
        _vars.perms[g_id] = g_perm
        if u_id == _vars.owner_id:
            u_perm = inf
            g_perm[u_id] = inf
        else:
            u_perm = g_perm.get(u_id,0)
    else:
        g_perm = {}
        u_perm = 1
    return g_perm,u_perm

async def processMessage(message):
    global client
    queue = _vars.queue
    perms = _vars.perms
    bans = _vars.bans
    commands = _vars.commands
    stored_vars = _vars.stored_vars
    msg = message.content
    if msg[:2] == "> ":
        msg = msg[2:]
    elif msg[:2]=="||" and msg[-2:]=="||":
        msg = msg[2:-2]
    msg = msg.replace("`","")
    user = message.author
    guild = message.guild
    u_id = user.id
    g_perm,u_perm = getUserPerms(guild,user)
    ch = message.channel
    if msg[0]=="~" and msg[1]!="~":
        comm = msg[1:]
        for command in commands:
            check = comm[:len(command)]
            argv = comm[len(command):]
            if check==command and (len(comm)==len(command) or comm[len(command)]==" "):
                print(user.name+" ("+str(u_id)+") issued command "+msg)
                req = commands[command][0]
                if req <= u_perm:
                    dtime = datetime.datetime.utcnow().timestamp()
                    try:
                        nextw = argv.index(" ")
                        argv = argv[nextw+1:]
                    except:
                        pass
                    response = None
                    try:
                        if command == "help":
                            less = 0
                            show = []
                            for com in commands:
                                comrep = commands[com]
                                if com in argv:
                                    less = -1
                                    newstr = "\n`"+com+"`\nEffect: \
"+comrep[1]+"\nUsage: "+comrep[2]+"\nRequired permission level: **__"+str(comrep[0])+"__**"
                                    if (not len(show)) or len(show[-1])<len(newstr):
                                        show = [newstr]
                            if less == 0:
                                if "?v" in argv.lower():
                                    less = 0
                                else:
                                    less = 1
                            if less >= 0:
                                for com in commands:
                                    comrep = commands[com]
                                    if comrep[0] <= u_perm:
                                        if comrep[1] != "":
                                            if less:
                                                show.append("`"+comrep[2]+"`")
                                            else:
                                                show.append("\n`"+com+"`\nEffect: \
"+comrep[1]+"\nUsage: "+comrep[2])
                                response = "Commands for **"+user.name+"**:\n"+"\n".join(show)
                            else:
                                response = "\n".join(show)
                        elif command == "clearcache":
                            _vars.stored_vars = copyGlobals()
                            response = "```\nCache cleared!\n```"
                        elif command == "math":
                            _tm = time.time()
                            plt.clf()
                            _f = verifyCommand(argv)
                            returns = [doMath]
                            doParallel(doMath,[_f,returns])
                            while returns[0]==doMath and time.time()<_tm+_vars.TIMEOUT_DELAY:
                                await asyncio.sleep(.1)
                            if fig.get_axes():
                                fn = "cache/temp.png"
                                plt.savefig(fn,bbox_inches="tight")
                                _f = discord.File(fn)
                                await ch.send(file=_f)
                            else:
                                answer = returns[0]
                                if answer == doMath:
                                    response = "```\nError: Timed out.\n```"
                                if answer is None:
                                    if len(argv):
                                        response = "```\n"+argv+" successfully executed!\n```"
                                    else:
                                        response = "```\nError: function is empty.\n```"
                                elif "\nError" in answer:
                                    response = "```"+answer+"\n```"
                                else:
                                    response = "```\n"+argv+" = "+str(answer)+"\n```"
                                stored_vars.update(globals())
                        elif command == "join":
                            voice = message.author.voice
                            vc = voice.channel
                            await vc.connect(timeout=_vars.TIMEOUT_DELAY,reconnect=True)
                        elif command == "play":
                            url = verifyURL(argv)
                            queue.append(url)
                            response = "```\nAdded "+url+" to the queue!\n```"
                        elif command == "q":
                            resp = ""
                            for i in range(len(queue)):
                                resp += "\n"+str(i)+": "+queue[i]
                            response = "```\nQueue: "+resp+"\n```"
                        elif "perms" in command:
                            try:
                                _p1 = argv.index(" ")
                                _user = argv[:_p1]
                            except:
                                _p1 = None
                                _user = argv
                                if _user == "":
                                    _user = u_id
                            try:
                                _user = int(_user)
                            except:
                                _user = int(_user.replace("<","").replace("@","").replace("!","").replace(">",""))
                            if _p1 is not None:
                                _val = argv[_p1+1:]
                                _val = float(eval(verifyCommand(_val),stored_vars))
                                orig = g_perm.get(_user,0)
                                requ = max(_val,orig,1)+1
                                u_target = await client.fetch_user(_user)
                                if u_perm>=requ:
                                    g_perm[_user] = _val
                                    _f = open("perms.json","w")
                                    _f.write(str(perms))
                                    _f.close()
                                    response = "Changed permissions for **"+u_target.name+"** in \
**"+guild.name+"** from **__"+expNum(orig,12,4)+"__** to **__"+expNum(_val,12,4)+"__**."
                                else:
                                    response = "```\nError:\n```\nInsufficient priviliges to change permissions for \
**"+u_target.name+"** in **"+guild.name+"** from **__"+expNum(orig,12,4)+"__** to \
**__"+expNum(_val,12,4)+"__**.\nRequired level: \
**__"+expNum(requ,12,4)+"__**, Current level: **__"+expNum(u_perm,12,4)+"__**"
                            else:
                                u_target = await client.fetch_user(_user)
                                _val = g_perm.get(_user,0)
                                response = "Current permissions for **"+u_target.name+"** in \
**"+guild.name+"**: **__"+expNum(_val,12,4)+"__**"
                        elif command == "uni2hex":
                            b = bytes(argv,"utf-8")
                            response = bytes2Hex(b)
                        elif command == "hex2uni":
                            b = hex2Bytes(argv)
                            response = b.decode("utf-8")
                        elif command in ["translate","tr","translate2","tr2"]:
                            try:
                                spl = argv.index(" ")
                                dest = argv[:spl+1].replace(" ","")
                                string = argv[spl+1:]
                            except:
                                dest = "en"
                                string = argv
                            if "?g " in string:
                                skip = True
                                string = string.replace("?g ","")
                            elif " ?g" in string:
                                skip = True
                                string = string.replace(" ?g","")
                            else:
                                skip = False
                            source = _vars.tr[0].detect(string).lang
                            try:
                                if skip:
                                    raise
                                if "-" in dest:
                                    dest = dest[:-2]+dest[-2:].upper()
                                tr = _vars.tr[1].translate(string,source,dest)
                                m1 = "`Papago`"
                            except:
                                dest = dest.lower()
                                tr = _vars.tr[0].translate(string,dest,source).text
                                m1 = "`Google Translate`"
                            if "2" in command:
                                try:
                                    if skip:
                                        raise
                                    if "-" in dest:
                                        dest = dest[:-2]+dest[-2:].upper()
                                    tr2 = _vars.tr[1].translate(tr,dest,source)
                                    m2 = "`Papago`"
                                except:
                                    dest = dest.lower()
                                    tr2 = _vars.tr[0].translate(tr,source,dest).text
                                    m2 = "`Google Translate`"
                            else:
                                tr2 = ""
                                m2 = ""
                            response = "**"+user.name+"**:\n"+tr+"  "+m1+"\n"*(len(tr2)>0)+tr2+"  "+m2
                        elif command == "cs_mem2flag":
                            try:
                                spl = argv.index(" ")
                                mem = argv[:spl+1].replace(" ","")
                                val = argv[spl+1:]
                            except:
                                val = "1"
                                mem = argv
                            response = mem2flag.mem2flag(mem,int(eval(verifyCommand(val),stored_vars)))
                        elif command == "cs_npc":
                            if "?s " in argv:
                                argv = argv.replace("?s ","")
                                lim = 20
                            elif " ?s" in argv:
                                argv = argv.replace(" ?s","")
                                lim = 20
                            else:
                                lim = 64
                            data = _vars.ent.search(argv,lim)
                            if len(data):
                                head = _vars.ent.data[1]
                                for i in range(len(head)):
                                    if head[i] == "":
                                        head[i] = i*" "
                                table = ptable(head)
                                for line in data:
                                    table.add_row(line)
                                output = str(table)
                                if len(output)<10000 and len(output)>1900:
                                    response = ["Search results for **"+argv+"**:"]
                                    lines = output.split("\n")
                                    curr = "```\n"
                                    for line in lines:
                                        if len(curr)+len(line) > 1900:
                                            response.append(curr+"```")
                                            curr = "```\n"
                                        curr += line+"\n"
                                    response.append(curr+"```")
                                else:
                                    response = "Search results for **"+argv+"**:\n```\n"+output+"```"
                            else:
                                response = "No results found for **"+argv+"**."
                        elif command == "cs_tsc":
                            if "?s " in argv:
                                argv = argv.replace("?s ","")
                                lim = 20
                            elif " ?s" in argv:
                                argv = argv.replace(" ?s","")
                                lim = 20
                            else:
                                lim = 64
                            data = _vars.tsc.search(argv,lim)
                            if len(data):
                                head = _vars.tsc.data[0]
                                for i in range(len(head)):
                                    if head[i] == "":
                                        head[i] = i*" "
                                table = ptable(head)
                                for line in data:
                                    table.add_row(line)
                                output = str(table)
                                if len(output)<10000 and len(output)>1900:
                                    response = ["Search results for **"+argv+"**:"]
                                    lines = output.split("\n")
                                    curr = "```\n"
                                    for line in lines:
                                        if len(curr)+len(line) > 1900:
                                            response.append(curr+"```")
                                            curr = "```\n"
                                        curr += line+"\n"
                                    response.append(curr+"```")
                                else:
                                    response = "Search results for **"+argv+"**:\n```\n"+output+"```"
                            else:
                                response = "No results found for **"+argv+"**."
                        elif command == "char2emoj":
                            _p1 = argv.index(" ")
                            _str = argv[:_p1]
                            _part = argv[_p1+1:]
                            _p2 = _part.index(" ")
                            _c1 = _part[:_p2]
                            _c2 = _part[_p2+1:]
                            response = Char2Emoj.convertString(_str.upper(),_c1,_c2)
                        elif command == "mc_ench":
                            spl = argv.index(" ")
                            item = argv[:spl+1]
                            enchants = ast.literal_eval(argv[spl+1:])
                            response = op_items_generator.generateEnchant(item,enchants)
                        elif command == "scale2x":
                            url = verifyURL(argv)
                            spl = url.split(".")
                            ext = spl[-1]
                            fn = "cache/temp."+ext
                            resp = opener.open(url)
                            data = resp.read()
                            _f = open(fn,"wb")
                            _f.write(data)
                            _f.close()
                            img1 = pygame.image.load(fn)
                            img2 = pygame.transform.scale2x(img1)
                            pygame.image.save(img2,fn)
                            _f = discord.File(fn)
                            await ch.send("Scaled image:",file=_f)
                        elif command in ["neko","lewd_neko","lewd"]:
                            if "lewd" in command:
                                valid = False
                                try:
                                    valid = message.channel.is_nsfw()
                                except AttributeError:
                                    valid = True
                            else:
                                valid = True
                            if valid:
                                if "?l " in argv:
                                    link = True
                                    argv = argv.replace("?l ","")
                                elif " ?l" in argv:
                                    link = True
                                    argv = argv.replace(" ?l","")
                                else:
                                    link = False
                                text = ""
                                if command == "neko":
                                    if "gif" in argv:
                                        url = nekos.img("ngif")
                                    else:
                                        url = nekos.img("neko")
                                elif command == "lewd_neko":
                                    if "gif" in argv:
                                        url = nekos.img("nsfw_neko_gif")
                                    else:
                                        url = nekos.img("lewd")
                                else:
                                    objs = searchRandomNSFW(argv)
                                    if link:
                                        text = "\nImage **__"+str(objs[2])+"__** on page **__"+str(objs[1])+"__**"
                                    url = objs[0]
                                if link:
                                    text = "Pulled from "+url+text
                                emb = discord.Embed(url=url)
                                emb.set_image(url=url)
                                print(url)
                                #response = url
                                await ch.send(text,embed=emb)
                            else:
                                response = "```\nError:\n```\nThis command is only available in **NSFW** channels."
                        elif command=="purge" or command=="purgeU" or command=="purgeA":
                            if "?h " in argv:
                                hidden = True
                                argv = argv.replace("?h ","")
                            elif " ?h" in argv:
                                hidden = True
                                argv = argv.replace(" ?h","")
                            else:
                                hidden = False
                            if command == "purge":
                                cnt = argv
                                target = client.user.id
                            elif command == "purgeU":
                                spl = argv.index(" ")
                                _user = argv[:spl+1].replace(" ","")
                                try:
                                    _user = int(_user)
                                except:
                                    _user = int(_user.replace("<","").replace("@","").replace("!","").replace(">",""))
                                cnt = argv[spl+1:]
                                target = _user
                            elif command == "purgeA":
                                cnt = argv
                                target = None
                            if not len(cnt):
                                cnt = 1
                            cnt = ceil(mpf(cnt))
                            hist = await message.channel.history(limit=64).flatten()
                            delM = []
                            deleted = 0
                            for m in hist:
                                if cnt <= 0:
                                    break
                                m_id = m.author.id
                                if target is None or m_id==target:
                                    delM.append(m)
                                    cnt -= 1
                            try:
                                await message.channel.delete_messages(delM)
                                deleted = len(delM)
                            except:
                                deleted = 0
                                for m in delM:
                                    try:
                                        await m.delete()
                                        deleted += 1
                                    except:
                                        pass
                            if not hidden:
                                response = "Deleted **__"+str(deleted)+"__** message"+"s"*(deleted!=1)+"!"
                        elif command == "ban":
                            try:
                                spl = argv.index(" ")
                                target = argv[:spl+1]
                                argv = argv[spl+1:]
                                try:
                                    spl = argv.index(" ")
                                    tm = argv[:spl+1]
                                    msg = argv[spl+1:]
                                except:
                                    tm = argv
                                    msg = None
                            except:
                                target = argv
                                tm = "0"
                                msg = None
                            try:
                                target = int(target)
                            except:
                                target = int(target.replace("<","").replace("@","").replace("!","").replace(">",""))
                            tm = eval(verifyCommand(tm),stored_vars)
                            etime = tm*3600+dtime
                            u_target = await client.fetch_user(target)
                            g_id = message.channel.guild.id
                            is_banned = bans.get(g_id,{}).get(target,None)
                            if is_banned is not None:
                                is_banned = is_banned[0]-dtime
                            bans[g_id] = {target:[etime,message.channel.id]}
                            _f = open("bans.json","w")
                            _f.write(str(bans))
                            _f.close()
                            if tm >= 0:
                                await ch.guild.ban(u_target,reason=msg,delete_message_days=0)
                            if is_banned:
                                response = "Updated ban for **"+u_target.name+"** from \
**__"+expNum(is_banned/3600,16,8)+"__** hours to **__"+expNum(tm,16,8)+"__** hours."
                            elif tm >= 0:
                                response = "**"+u_target.name+"** has been banned from \
**"+message.channel.guild.name+"** for **__"+expNum(tm,16,8)+"__** hours."
                        elif command == "shutdown":
                            await ch.send("Shutting down... :wave:")
                            for vc in client.voice_clients:
                                await vc.disconnect(force=True)
                            await client.close()
                            sys.exit()
                            quit()
                        else:
                            response = "```\nError: Unimplemented command "+command+"\n```"
                        if response is not None:
                            if len(response) < 65536:
                                print(response)
                            else:
                                print("[RESPONSE OVER 64KB]")
                            if type(response) is list:
                                for r in response:
                                    await ch.send(r)
                            else:
                                await ch.send(response)
                    except discord.HTTPException:
                        try:
                            fn = "cache/temp.txt"
                            _f = open(fn,"wb")
                            _f.write(bytes(response,"utf-8"))
                            _f.close()
                            _f = discord.File(fn)
                            await ch.send("Response too long for message.",file=_f)
                        except:
                            raise
                    except Exception as ex:
                        rep = repr(ex)
                        if len(rep) > 1900:
                            await ch.send("```\nError: Error message too long.\n```")
                        else:
                            await ch.send("```\nError: "+rep+"\n```")
                    return
                else:
                    await ch.send(
                        "```\nError:\n```\nInsufficient priviliges for command "+command+"\
.\nRequred level: **__"+expNum(req)+"__**, Current level: **__"+expNum(u_perm)+"__**")
                    return
    msg = message.content
    if msg == "<@!"+str(client.user.id)+">":
        if not u_perm < 0:
            await ch.send("Hi, did you require my services for anything?")
        else:
            await ch.send("Sorry, you are currently not permitted to request my services.")

@client.event
async def on_ready():
    print("Successfully connected as "+str(client.user))
    print("Servers: ")
    for guild in client.guilds:
        print(guild)
    await handleUpdate()
##    print("Users: ")
##    for guild in client.guilds:
##        print(guild.members)

async def handleUpdate(force=False):
    global client,_vars
    if force or time.time()-_vars.lastCheck>.5:
        _vars.lastCheck = time.time()
        dtime = datetime.datetime.utcnow().timestamp()
        bans = _vars.bans
        if bans:
            for g in bans:
                bl = list(bans[g])
                for b in bl:
                    if dtime >= bans[g][b][0]:
                        u_target = await client.fetch_user(b)
                        g_target = await client.fetch_guild(g)
                        c_target = await client.fetch_channel(bans[g][b][1])
                        bans[g].pop(b)
                        try:
                            await g_target.unban(u_target)
                            await c_target.send("**"+u_target.name+"** has been unbanned from **"+g_target.name+"**.")
                        except:
                            await c_target.send("Unable to unban **"+u_target.name+"** from **"+g_target.name+"**.")
            f = open("bans.json","w")
            f.write(str(bans))
            f.close()
        for vc in client.voice_clients:
            membs = vc.channel.members
            cnt = len(membs)
            if not cnt > 1:
                await vc.disconnect(force=False)

async def checkDelete(message,reaction,user):
    u_perm = getUserPerms(message.guild,user)[1]
    if not u_perm < 0:
        if user.id != client.user.id:
            if message.author.id == client.user.id:
                s = str(reaction)
                if s in "❌✖❎":
                    try:
                        temp = message.content
                        await message.delete()
                        print(temp+" deleted by "+user.name)
                    except:
                        pass
            await handleUpdate()

@client.event
async def on_reaction_add(reaction,user):
    message = reaction.message
    await checkDelete(message,reaction,user)
        
@client.event
async def on_raw_reaction_add(payload):
    try:
        channel = await client.fetch_channel(payload.channel_id)
        user = await client.fetch_user(payload.user_id)
        message = await channel.fetch_message(payload.message_id)
    except:
        return
    reaction = payload.emoji
    await checkDelete(message,reaction,user)
    
@client.event
async def on_typing(channel,user,when):
    await handleUpdate()

@client.event
async def on_voice_state_update(member,before,after):
    await handleUpdate()

async def handleMessage(message):
    msg = message.content
    user = message.author
    u_id = user.id
    u_perm = _vars.perms.get(u_id,0)
    if not len(msg)>1:
        return
    elif u_id==client.user.id:
        try:
            await message.add_reaction("❎")
        except:
            raise
    try:
        await asyncio.wait_for(processMessage(message),timeout=_vars.TIMEOUT_DELAY)
    except Exception as ex:
        await message.channel.send("```\nError: "+repr(ex)+"\n```")
    return
        
@client.event
async def on_message(message):
    await handleUpdate()
    await handleMessage(message)
    await handleUpdate(True)

@client.event
async def on_message_edit(before,after):
    message = after
    await handleUpdate()
    await handleMessage(message)
    await handleUpdate(True)

@client.event
async def on_raw_message_edit(payload):
    message = None
    if payload.cached_message is None:
        try:
            channel = await client.fetch_channel(payload.data["channel_id"])
            message = await channel.fetch_message(payload.message_id)
        except:
            for guild in client.guilds:
                for channel in guild.text_channels:
                    try:
                        message = await channel.fetch_message(payload.message_id)
                    except:
                        pass
    if message:
        await handleUpdate()
        await handleMessage(message)
        await handleUpdate(True)

def _init():
    global _vars
    try:
        _f = open("perms.json")
        perms = eval(_f.read())
    except:
        perms = {}
        _f = open("perms.json","w")
        _f.write(str(perms))
    _f.close()

    try:
        _f = open("bans.json")
        bans = eval(_f.read())
    except:
        bans = {}
        _f = open("bans.json","w")
        _f.write(str(bans))
    _f.close()
    _vars = _globals()
    _vars.auth = {}
    try:
        _f = open("auth.json")
        data = ast.literal_eval(_f.read())
        _f.close()
        _vars.token = data["discord_token"]
        print("Attempting to authorize with token "+_vars.token+":")
        _vars.auth["papago_id"] = data["papago_id"]
        _vars.auth["papago_secret"] = data["papago_secret"]
    except:
        print("Unable to load bot tokens.")
        raise
    _vars.tr = [Translator(["translate.google.com"]),PapagoTrans(
        _vars.auth["papago_id"],
        _vars.auth["papago_secret"],
        )]
    _vars.ent = SheetPull("https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYH\
hC56mPXCnCcsDjxit0/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0")
    _vars.tsc = SheetPull("https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i\
-rjRjgMW04Gdz9EO6U/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0")
    _vars.perms = perms
    _vars.bans = bans
    _vars.stored_vars = copyGlobals()

if __name__ == "__main__":
    _init()
    try:
        client.run(_vars.token)
    except:
        print("Unable to authorize bot tokens.")
        raise
