import discord,ast,os,sys,asyncio,datetime,requests,json,csv,shlex
import urllib.request
from matplotlib import pyplot as plt
from googletrans import Translator
from smath import *

client = discord.Client(
    max_messages=2000,
    activity=discord.Activity(name="Magic"),
    )

bar_plot = plt.bar
plot_points = plt.scatter
plot = plt.plot
plotY = plt.semilogy
plotX = plt.semilogx
plotXY = plt.loglog
fig = plt.figure()

class _globals:
    owner_id = 201548633244565504
    timeout = 10
    disabled = [
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
        "os",
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
        "requests",
        "commands"
        ]
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
    class PapagoTrans:
        class PapagoOutput:
            def __init__(self,text):
                self.text = text
        def __init__(self,c_id,c_sec):
            self.id = c_id
            self.secret = c_sec
        def translate(self,string,dest,source="en"):
            url = "https://openapi.naver.com/v1/papago/n2mt"
            enc = urllib.parse.quote(string)
            data = "source="+source+"&target="+dest+"&text="+enc
            req = urllib.request.Request(url)
            req.add_header("X-Naver-Client-Id",self.id)
            req.add_header("X-Naver-Client-Secret",self.secret)
            print(req,url,data)
            resp = urllib.request.urlopen(req,data=data.encode("utf-8"),timeout=_globals.timeout/2)
            if resp.getcode() != 200:
                raise ConnectionError("Error "+str(resp.getcode()))
            r = resp.read().decode("utf-8")
            try:
                r = json.loads(r)
            except:
                pass
            text = r["message"]["result"]["translatedText"]
            output = self.PapagoOutput(text)
            return output
    def __init__(self):
        self.lastCheck = time.time()
        self.queue = []
        try:
            f = open("perms.json")
            self.perms = eval(f.read())
            f.close()
        except:
            self.perms = {}
        try:
            f = open("bans.json")
            self.bans = eval(f.read())
            f.close()
        except:
            self.bans = {}
        try:
            f = open("enabled.json")
            self.enabled = eval(f.read())
            f.close()
        except:
            self.enabled = {}            
        self.update()
        self.fig = fig
        self.plt = plt
        self.auth = {}
        f = open("auth.json")
        data = ast.literal_eval(f.read())
        f.close()
        self.token = data["discord_token"]
        print("Attempting to authorize with token "+self.token+":")
        self.auth["papago_id"] = data["papago_id"]
        self.auth["papago_secret"] = data["papago_secret"]
        self.translators = {"Google Translate":Translator(["translate.google.com"]),
                            "Papago":self.PapagoTrans(self.auth["papago_id"],self.auth["papago_secret"])}
        self.ent = self.SheetPull("https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYH\
hC56mPXCnCcsDjxit0/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0")
        self.tsc = self.SheetPull("https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i\
-rjRjgMW04Gdz9EO6U/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0")
        comstr = "commands_"
        files = [f for f in os.listdir('.') if f[-3:]==".py" and comstr in f]
        self.categories = {}
        for f in files:
            module = f[:-3]
            category = module.replace(comstr,"")
            exec("import "+module+" as _vars_",globals())
            commands = []
            vd = _vars_.__dict__
            for k in vd:
                var = vd[k]
                try:
                    assert(var.is_command)
                    obj = var()
                    obj.__name__ = var.__name__
                    obj.name.append(obj.__name__)
                    commands.append(obj)
                except:
                    pass
            self.categories[category] = commands
        self.resetGlobals()
    def update(self):
        f = open("perms.json","w")
        f.write(str(self.perms))
        f.close()
        f = open("bans.json","w")
        f.write(str(self.bans))
        f.close()
        f = open("enabled.json","w")
        f.write(str(self.enabled))
        f.close()
    def verifyID(self,value):
        return int(str(value).replace("<","").replace(">","").replace("@","").replace("!",""))
    def getPerms(self,user,guild):
        try:
            u_id = user.id
        except:
            u_id = user
        if guild:
            g_id = guild.id
            g_perm = self.perms.get(g_id,{})
            self.perms[g_id] = g_perm
            if u_id == self.owner_id:
                u_perm = inf
                g_perm[u_id] = inf
            else:
                u_perm = g_perm.get(u_id,0)
        else:
            u_perm = 1
        return u_perm
    def resetGlobals(self):
        self.stored_vars = dict(globals())
        for i in self.disabled:
            try:
                self.stored_vars.pop(i)
            except:
                pass
        return self.stored_vars
    def updateGlobals(self):
        temp = dict(globals())
        for i in self.disabled:
            try:
                temp.pop(i)
            except:
                pass
        self.stored_vars.update(temp)
        return self.stored_vars
    def verifyCommand(self,func):
        f = func.lower()
        for d in self.disabled:
            if d in f:
                raise PermissionError("Issued command is not enabled.")
        return func
    def verifyURL(self,_f):
        _f = _f.replace("<","").replace(">","").replace("|","").replace("*","").replace("_","").replace("`","")
        return _f
    def doMath(self,f,returns):
        try:
            self.verifyCommand(f)
            try:
                answer = eval(f,self.stored_vars)
            except:
                exec(f,self.stored_vars)
                answer = None
        except Exception as ex:
            answer = "\nError: "+repr(ex)
        if answer is not None:
            answer = str(answer)
        returns[0] = answer
    def evalMath(self,f):
        self.verifyCommand(f)
        return eval(f)

async def processMessage(message):
    global client
    perms = _vars.perms
    bans = _vars.bans
    categories = _vars.categories
    stored_vars = _vars.stored_vars
    msg = message.content
    if msg[:2] == "> ":
        msg = msg[2:]
    elif msg[:2]=="||" and msg[-2:]=="||":
        msg = msg[2:-2]
    msg = msg.replace("`","")
    user = message.author
    guild = message.guild
    g_id = guild.id
    u_id = user.id
    try:
        enabled = _vars.enabled[g_id]
    except:
        enabled = _vars.enabled[g_id] = ["math","admin"]
        _vars.update()
    u_perm = _vars.getPerms(user.id,guild)
    ch = channel = message.channel
    if msg[0]=="~" and msg[1]!="~":
        comm = msg[1:]
        commands = []
        for catg in categories:
            if catg in enabled or catg == "main":
                commands += categories[catg]
        for command in commands:
            for alias in command.name:
                length = len(alias)
                check = comm[:length]
                argv = comm[length:]
                print(alias)
                if check==alias and (len(comm)==length or comm[length]==" " or comm[length]=="?"):
                    print(user.name+" ("+str(u_id)+") issued command "+msg)
                    req = command.minm
                    if not req > u_perm:
                        try:
                            if argv:
                                while argv[0] == " ":
                                    argv = argv[1:]
                            flags = {}
                            for c in range(26):
                                char = chr(c+97)
                                flag = "?"+char
                                for r in (flag.lower(),flag.upper()):
                                    if len(argv)>=2 and r in argv:
                                        for check in (r+" "," "+r):
                                            if check in argv:
                                                argv = argv.replace(check,"")
                                                flags[char] = True
                                        if argv == flag:
                                            argv = ""
                                            flags[char] = True
                                        if len(argv)>=2:
                                            if not char in flags:
                                                i = argv.index(r)
                                                if i==0 or argv[i-1]==" " or argv[i-2]=="?":
                                                    if argv[i+2]==" " or argv[i+2]=="?":
                                                        argv = argv[:i]+argv[i+2:]
                                                        flags[char] = True
                            args = shlex.split(argv.replace("<","'").replace(">","'"))
                            response = await command(
                                client=client,      #for interfacing with discord
                                _vars=_vars,        #for interfacing with bot's database
                                argv=argv,          #raw text arguments
                                args=args,          #split text arguments
                                flags=flags,        #special flags
                                user=user,          #user that invoked the command
                                message=message,    #message data
                                channel=channel,    #channel data
                                guild=guild,        #guild data
                                name=alias,         #alias the command was called as
                                )
                            if response is not None:
                                if len(response) < 65536:
                                    print(response)
                                else:
                                    print("[RESPONSE OVER 64KB]")
                                if type(response) is list:
                                    for r in response:
                                        await channel.send(r)
                                else:
                                    await channel.send(response)
                        except discord.HTTPException:
                            try:
                                fn = "cache/temp.txt"
                                _f = open(fn,"wb")
                                _f.write(bytes(response,"utf-8"))
                                _f.close()
                                _f = discord.File(fn)
                                await channel.send("Response too long for message.",file=_f)
                            except:
                                raise
                        except Exception as ex:
                            rep = repr(ex)
                            if len(rep) > 1900:
                                await channel.send("```\nError: Error message too long.\n```")
                            else:
                                await channel.send("```\nError: "+rep+"\n```")
                            raise
                        return
                    else:
                        await channel.send(
                            "```\nError:\n```\nInsufficient priviliges for command "+command+"\
    .\nRequred level: **__"+expNum(req)+"__**, Current level: **__"+expNum(u_perm)+"__**")
                        return
    msg = message.content
    if msg == "<@!"+str(client.user.id)+">":
        if not u_perm < 0:
            await channel.send("Hi, did you require my services for anything?")
        else:
            await channel.send("Sorry, you are currently not permitted to request my services.")

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
    u_perm = _vars.getPerms(user.id,message.guild)
    check = False
    if not u_perm < 1:
        check = True
    else:
        for reaction in message.reactions:
            async for u in reaction.users():
                if u.id == client.user.id:
                    check = True
    if check:
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
async def on_raw_message_delete(payload):
    await handleUpdate()
    
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
        if "Error: " in msg or "Commands for " in msg:
            try:
                await message.add_reaction("❎")
            except:
                raise
    try:
        await asyncio.wait_for(processMessage(message),timeout=_vars.timeout)
    except Exception as ex:
        await message.channel.send("```\nError: "+repr(ex)+"\n```")
        raise
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

if __name__ == "__main__":
    _vars = _globals()
    client.run(_vars.token)
