import discord,pygame,urllib.request,nekos,ast,sys,asyncio,datetime
from googletrans import Translator
from smath import *
import Char2Emoj
import op_items_generator
import mem2flag

class MyOpener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

translator = Translator(["translate.google.com"])
opener = MyOpener()
client = discord.Client()

f = open("auth.json")
token = ast.literal_eval(f.read())["token"]
f.close()

f = open("perms.json")
perms = ast.literal_eval(f.read())
f.close()

commands = {
    "help":[0,"Shows a list of usable commands.","`~help`"],
    "math":[0,"Evaluates a math function or formula.","`~math f`"],
    "play":[0,"Plays an audio file from a link, into a voice channel.","`~play l`"],
    "neko":[1,"",""],
    "lewd_neko":[2,"",""],
    "changeperms":[1,"Changes a user's permissions.","`~changeperms u n`"],
    "translate":[0,"Translates a string into another language.","`~translate l(en) s`"],
    "translate2":[0,"Translates a string into another language and then back.","`~translate2 l(en) s`"],
    "char2emoj":[1,"Makes emoji blocks using strings.","`~char2emoj s a b`"],
    "mem2flag":[0,"Returns the TSC representation of a Cave Story flag \
corresponding to a certain memory address and value.","`~mem2flag m n(1)`"],
    "mcenchant":[0,"Returns a Minecraft command that generates an item with target enchants.","`~mcenchant i e`"],
    "scale2x":[0,"Performs Scale2x on an image from an embed link.","`~scale2x l`"],
    "purge":[1,"Deletes all messages from bot in current channel, within a timeframe.","`~purge t(10)`"],
    "purgeU":[2,"Deletes all messages from a user in current channel, within a timeframe.","`~purgeU u t(10)`"],
    "purgeA":[2,"Deletes all messages from all users in current channel, within a timeframe.","`~purgeA t(10)`"],
    "shutdown":[1,"Shuts down the bot.","`~shutdown`"],
    
    "tr":[0,"Short for translate.","`~tr l(en) s`"],
    "tr2":[0,"Short for translate2.","`~tr2 l(en) s)`"],
}
    
def sendMessage(responses,*args,**kwargs):
    var = (args,kwargs)
    print(var)
    doParallel(responses.append,[var])

def processMessage(message,responses):
    global client,linkfile,perms,commands,translator
    msg = message.content
    user = message.author
    u_id = user.id
    u_perm = perms.get(u_id,0)
    if msg[0]=="~" and msg[1]!="~":
        comm = msg[1:]
        for command in commands:
            check = comm[:len(command)]
            argv = comm[len(command):]
            if check==command and (len(comm)==len(command) or comm[len(command)]==" "):
                print(user.display_name+" ("+str(u_id)+") executed command "+msg)
                req = commands[command][0]
                if req <= u_perm:
                    try:
                        nextw = argv.index(" ")
                        argv = argv[nextw+1:]
                    except:
                        pass
                    try:
                        if command == "math":
                            answer = eval(argv.replace("__",""))
                            if answer is None:
                                sendMessage(responses,argv+" successfully executed!")
                            else:
                                sendMessage(responses,argv+" = "+str(answer))
                        elif command == "help":
                            show = []
                            for com in commands:
                                comrep = commands[com]
                                if comrep[0] <= u_perm:
                                    if comrep[1] != "":
                                        show.append("\nCommand: `"+com+"`\nDescription: "+comrep[1]+"\nUsage: "+comrep[2])
                            sendMessage(responses,"Commands for **"+user.display_name+"**:\n"+"\n".join(show))
                        elif command == "changeperms":
                            _p1 = argv.index(" ")
                            _user = int(argv[:_p1])
                            _val = int(argv[_p1+1:])
                            orig = perms.get(_user,0)
                            requ = max(_val,orig)
                            if u_perm>=requ:
                                perms[_user] = _val
                                f = open("perms.json","w")
                                f.write(str(perms))
                                f.close()
                                sendMessage(responses,"Changed permissions for "+str(_user)+" from "+str(orig)+" to "+str(_val)+".")
                            else:
                                sendMessage(
                                    responses,
                                    "Error: Insufficient priviliges to change permissions for \
"+str(_user)+" from "+str(orig)+" to "+str(_val)+".\nRequired level: **"+str(requ)+"**, Current level: **"+str(u_perm)+"**")
                        elif command=="translate" or command=="tr":
                            try:
                                spl = argv.index(" ")
                                language = argv[:spl+1].replace(" ","")
                                string = argv[spl+1:]
                            except:
                                language = "en"
                                string = argv
                            tr = translator.translate(string,dest=language)
                            sendMessage(responses,tr.text)
                        elif command == "translate2" or command=="tr2":
                            try:
                                spl = argv.index(" ")
                                language = argv[:spl+1].replace(" ","")
                                string = argv[spl+1:]
                            except:
                                language = "en"
                                string = argv
                            tr = translator.translate(string,dest=language)
                            string2 = tr.text
                            tr2 = translator.translate(string2,dest=tr.src)
                            sendMessage(responses,string2)
                            sendMessage(responses,tr2.text)
                        elif command == "mem2flag":
                            try:
                                spl = argv.index(" ")
                                mem = argv[:spl+1].replace(" ","")
                                val = argv[spl+1:]
                            except:
                                val = "1"
                                mem = argv
                            text = mem2flag.mem2flag(mem,int(val))
                            sendMessage(responses,text)
                        elif command == "char2emoj":
                            _p1 = argv.index(" ")
                            _str = argv[:_p1]
                            _part = argv[_p1+1:]
                            _p2 = _part.index(" ")
                            _c1 = _part[:_p2]
                            _c2 = _part[_p2+1:]
                            resp = Char2Emoj.convertString(_str,_c1,_c2)
                            sendMessage(responses,resp)
                        elif command == "mcenchant":
                            spl = argv.index(" ")
                            item = argv[:spl+1]
                            enchants = ast.literal_eval(argv[spl+1:])
                            response = op_items_generator.generateEnchant(item,enchants)
                            sendMessage(responses,response)
                        elif command == "scale2x":
                            url = argv
                            spl = url.split(".")
                            ext = spl[-1]
                            fn = "cache/temp."+ext
                            response = opener.open(url)#urllib.request.urlopen(url)
                            data = response.read()
                            f = open(fn,"wb")
                            f.write(data)
                            f.close()
                            img1 = pygame.image.load(fn)
                            img2 = pygame.transform.scale2x(img1)
                            pygame.image.save(img2,fn)
                            f = discord.File(fn)
                            sendMessage(responses,"Scaled image:",file=f)
                        elif command == "neko":
                            url = nekos.img("neko")
                            emb = discord.Embed(url=url)
                            emb.set_image(url=url)
                            sendMessage(responses,embed=emb)
                        elif command == "lewd_neko":
                            valid = False
                            try:
                                valid = message.channel.is_nsfw()
                            except AttributeError:
                                valid = True
                            if valid:
                                url = nekos.img("lewd")
                                emb = discord.Embed(url=url)
                                emb.set_image(url=url)
                                sendMessage(responses,embed=emb)
                            else:
                                sendMessage(responses,"Error: This command is only available in NSFW channels.")
                        elif command == "purge":
                            responses.append({"purge":argv})
                        elif command == "purgeU":
                            spl = argv.index(" ")
                            user = argv[:spl+1].replace(" ","")
                            delay = argv[spl+1:]
                            responses.append({"purge":[user,delay]})
                        elif command == "purgeA":
                            responses.append({"purge":[None,argv]})
                        elif command == "shutdown":
                            responses.append({"shutdown":1})
                            sendMessage(responses,"Shutting down... :wave:")
                            sys.exit()
                            quit()
                        else:
                            sendMessage(responses,"Error: Unimplemented command "+command)
                    except Exception as ex:
                        sendMessage(responses,"Error: "+repr(ex))
                    return
                else:
                    sendMessage(
                        responses,
                        "Error: Insufficient priviliges for command "+command+"\
.\nRequred level: **"+str(req)+"**, Current level: **"+str(u_perm)+"**")
                    return
    responses.append(None)

@client.event
async def on_ready():
    print("Connected as "+str(client.user))
    print("Servers: ")
    for guild in client.guilds:
        print(guild)
##    print("Users: ")
##    for guild in client.guilds:
##        print(guild.members)

@client.event
async def on_message(message):
    msg = message.content
    user = message.author
    u_id = user.id
    u_perm = perms.get(u_id,0)
    if not len(msg)>1 or u_id==client.user.id:
        return
    try:
        responses = []
        doParallel(processMessage,[message,responses])
        t = time.time()
        while not len(responses):
            await asyncio.sleep(.1)
            if time.time()-t > 3:
                t = inf
                await message.channel.send("Processing...")
        check = responses[0]
        if check is None:
            return
        elif type(check) is dict:
            for c in check:
                if c == "purge":
                    tm = check[c]
                    target = client.user.id
                    if type(tm) is list:
                        if tm[0] is None:
                            target = None
                        else:
                            target = int(tm[0])
                        tm = tm[1]
                    if not len(tm):
                        tm = 10
                    tm = datetime.datetime.utcnow().timestamp()-float(tm)
                    hist = await message.channel.history(limit=128).flatten()
                    deleted = 0
                    for m in hist:
                        m_id = m.author.id
                        m_time = m.created_at.timestamp()
                        if (m_id==target or target==None) and m_time>=tm:
                            await m.delete()
                            deleted += 1
                    await message.channel.send("Deleted "+str(deleted)+" message"+"s"*(deleted!=1)+"!")
                elif c == "shutdown":
                    await message.channel.send("Shutting down... :wave:")
            return
        for m in responses:
            args = m[0]
            kwargs = m[1]
            await message.channel.send(*args,**kwargs)
    except Exception as ex:
        await message.channel.send("Error: "+repr(ex))

@client.event
async def on_message_edit(before,after):
    message = after
    msg = message.content
    user = message.author
    u_id = user.id
    u_perm = perms.get(u_id,0)
    if not len(msg)>1 or u_id==client.user.id:
        return
    try:
        responses = []
        doParallel(processMessage,[message,responses])
        t = time.time()
        while not len(responses):
            await asyncio.sleep(.1)
            if time.time()-t > 3:
                t = inf
                await message.channel.send("Processing...")
        check = responses[0]
        if check is None:
            return
        elif type(check) is dict:
            for c in check:
                if c == "purge":
                    t_check = time.time()-check[c]
                    await message.channel.purge(check=lambda x: x.author==client.user and x.created_at.timestamp()>=t_check)
                    await message.channel.send("Deleted all messages since "+str(t_check))
                elif c == "shutdown":
                    await message.channel.send("Shutting down... :wave:")
            return
        for m in responses:
            args = m[0]
            kwargs = m[1]
            await message.channel.send(*args,**kwargs)
    except Exception as ex:
        await message.channel.send("Error: "+repr(ex))
   
client.run(token)
