import discord,pygame,urllib.request,nekos,ast,sys,asyncio,datetime,signal
from matplotlib import pyplot as plt
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
bar_plot = plt.bar
plot_points = plt.scatter
plot_spectrogram = plt.specgram
plot = plt.plot
plotY = plt.semilogy
plotX = plt.semilogx
plotXY = plt.loglog
fig = plt.figure()

owner_id = 201548633244565504

commands = {
    "help":[-inf,"Shows a list of usable commands.","`~help l(0)`"],
    "math":[0,"Evaluates a math function or formula.","`~math f`"],
    "play":[0,"Plays an audio file from a link, into a voice channel.","`~play l`"],
    "neko":[1,"",""],
    "lewd_neko":[1,"",""],
    "uni2hex":[0,"Converts unicode text to hexadecimal numbers.","`~uni2hex a`"],
    "hex2uni":[0,"Converts hexadecimal numbers to unicode text.","`~hex2uni b`"],
    "translate":[0,"Translates a string into another language.","`~translate l(en) s`"],
    "translate2":[0,"Translates a string into another language and then back.","`~translate2 l(en) s`"],
    "char2emoj":[0,"Makes emoji blocks using strings.","`~char2emoj s a b`"],
    "mem2flag":[0,"Returns the TSC representation of a Cave Story flag \
corresponding to a certain memory address and value.","`~mem2flag m n(1)`"],
    "mcenchant":[0,"Returns a Minecraft command that generates an item with target enchants.","`~mcenchant i e`"],
    "scale2x":[0,"Performs Scale2x on an image from an embed link.","`~scale2x l`"],
    "clear_cache":[1,"Clears all cached data.","`~clear_cache`"],
    "changeperms":[2,"Changes a user's permissions.","`~changeperms u n`"],
    "purge":[1,"Deletes a number of messages from bot in current channel.","`~purge c(1)`"],
    "purgeU":[3,"Deletes a number of messages from a user in current channel.","`~purgeU u c(1)`"],
    "purgeA":[3,"Deletes a number of messages from all users in current channel.","`~purgeA c(1)`"],
    "ban":[3,"Bans a user for an amount of time, with an optional message.","`~ban u t(0) m()`"],
    "shutdown":[3,"Shuts down the bot.","`~shutdown`"],
    
    "tr":[0,"",""],
    "tr2":[0,"",""],
}

def verifyCommand(_f):
    er = "ERR_NOT_ENABLED"
    _f = _f.replace("__",er).replace("pygame",er).replace("open",er).replace("import",er).replace("urllib",er)
    _f = _f.replace("discord",er).replace("sys",er)
    if er in _f:
        raise PermissionError("Issued command is not enabled.")
    return _f

def verifyURL(_f):
    _f = _f.replace("<","").replace(">","").replace("|","").replace("*","").replace("_","").replace("`","")
    return _f

def doMath(_f,returns):
    global stored_vars
    if not _f:
        returns[0] = None
        return
    try:
        try:
            answer = eval(_f,stored_vars)
        except:
            exec(_f,stored_vars)
            answer = None
    except Exception as ex:
        answer = "\nError: "+repr(ex)
    returns[0] = answer

async def processMessage(message,responses):
    global client,linkfile,perms,commands,translator,stored_vars
    msg = message.content
    if msg[:2] == "> ":
        msg = msg[2:]
    elif msg[:2]=="||" and msg[-2:]=="||":
        msg = msg[2:-2]
    msg = msg.replace("`","")
    user = message.author
    u_id = user.id
    if u_id == owner_id:
        u_perm = inf
        perms[u_id] = inf
    else:
        u_perm = perms.get(u_id,0)
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
                            show = []
                            if argv.lower() == "less" or argv.lower() == "l":
                                less = True
                            else:
                                less = False
                                for com in commands:
                                    comrep = commands[com]
                                    if com in argv:
                                        less = -1
                                        show.append("\nCommand: `"+com+"`\nEffect: \
"+comrep[1]+"\nUsage: "+comrep[2])
                            if less >= 0:
                                for com in commands:
                                    comrep = commands[com]
                                    if comrep[0] <= u_perm:
                                        if comrep[1] != "":
                                            if less:
                                                show.append("Command: `"+comrep[2]+"`")
                                            else:
                                                show.append("\nCommand: `"+com+"`\nEffect: \
"+comrep[1]+"\nUsage: "+comrep[2])
                                response = "Commands for **"+user.name+"**:\n"+"\n".join(show)
                            else:
                                response = "\n".join(show)
                        elif command == "clear_cache":
                            stored_vars = dict(globals())
                            response = "Cache cleared!"
                        elif command == "math":
                            _tm = time.time()
                            plt.clf()
                            _f = verifyCommand(argv)
                            returns = [doMath]
                            doParallel(doMath,[_f,returns])
                            while returns[0] == doMath and time.time() < _tm+5:
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
                                else:
                                    response = "```\n"+argv+" = "+str(answer)+"\n```"
                                stored_vars.update(globals())
                        elif command == "changeperms":
                            _p1 = argv.index(" ")
                            _user = argv[:_p1]
                            try:
                                _user = int(_user)
                            except:
                                _user = int(_user[3:-1])
                            _val = argv[_p1+1:]
                            _val = float(eval(verifyCommand(_val),stored_vars))
                            orig = perms.get(_user,0)
                            requ = max(_val,orig)
                            u_target = await client.fetch_user(_user)
                            if u_perm>=requ+1:
                                perms[_user] = _val
                                _f = open("perms.json","w")
                                _f.write(str(perms))
                                _f.close()
                                response = "Changed permissions for **"+u_target.name+"** from **\
"+expNum(orig,12,4)+"** to **"+expNum(_val,12,4)+"**."
                            else:
                                response = "Error: Insufficient priviliges to change permissions for \
**"+u_target.name+"** from **"+expNum(orig,12,4)+"** to **"+expNum(_val,12,4)+"**.\nRequired level: \
**"+expNum(requ+1,12,4)+"\**, Current level: **"+expNum(u_perm,12,4)+"**"
                        elif command == "uni2hex":
                            b = bytes(argv,"utf-8")
                            response = bytes2Hex(b)
                        elif command == "hex2uni":
                            b = hex2Bytes(argv)
                            response = b.decode("utf-8")
                        elif command=="translate" or command=="tr":
                            try:
                                spl = argv.index(" ")
                                language = argv[:spl+1].replace(" ","")
                                string = argv[spl+1:]
                            except:
                                language = "en"
                                string = argv
                            tr = translator.translate(string,dest=language)
                            response = user.name+": "+tr.text
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
                            response = "**"+user.name+"**: "+string2+"\n**"+user.name+"**: "+tr2.text
                        elif command == "mem2flag":
                            try:
                                spl = argv.index(" ")
                                mem = argv[:spl+1].replace(" ","")
                                val = argv[spl+1:]
                            except:
                                val = "1"
                                mem = argv
                            response = mem2flag.mem2flag(mem,int(eval(verifyCommand(val),stored_vars)))
                        elif command == "char2emoj":
                            _p1 = argv.index(" ")
                            _str = argv[:_p1]
                            _part = argv[_p1+1:]
                            _p2 = _part.index(" ")
                            _c1 = _part[:_p2]
                            _c2 = _part[_p2+1:]
                            response = Char2Emoj.convertString(_str.upper(),_c1,_c2)
                        elif command == "mcenchant":
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
                        elif command == "neko":
                            url = nekos.img("neko")
                            emb = discord.Embed(url=url)
                            emb.set_image(url=url)
                            await ch.send(embed=emb)
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
                                await ch.send(embed=emb)
                            else:
                                response = "Error: This command is only available in **NSFW** channels."
                        elif command=="purge" or command=="purgeU" or command=="purgeA":
                            if command == "purge":
                                cnt = argv
                                target = client.user.id
                            elif command == "purgeU":
                                spl = argv.index(" ")
                                _user = argv[:spl+1].replace(" ","")
                                try:
                                    _user = int(_user)
                                except:
                                    _user = int(_user[3:-1])
                                cnt = argv[spl+1:]
                                target = _user
                            elif command == "purgeA":
                                cnt = argv
                                target = None
                            if not len(cnt):
                                cnt = 1
                            cnt = ceil(dec(cnt))
                            hist = await message.channel.history(limit=64).flatten()
                            deleted = 0
                            for m in hist:
                                if cnt <= 0:
                                    break
                                m_id = m.author.id
                                if m_id==target or target==None:
                                    try:
                                        await m.delete()
                                        deleted += 1
                                    except:
                                        pass
                                cnt -= 1
                            response = "Deleted **"+str(deleted)+"** message"+"s"*(deleted!=1)+"!"
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
                                target = int(target[3:-1])
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
**"+expNum(is_banned/3600,16,5)+"** hours to **"+expNum(tm*24,16,5)+"** hours."
                            elif tm >= 0:
                                response = "**"+u_target.name+"** has been banned from \
**"+message.channel.guild.name+"** for **"+expNum(tm*24,16,5)+"** hours."
                        elif command == "shutdown":
                            await ch.send("Shutting down... :wave:")
                            sys.exit()
                            quit()
                        else:
                            response = "```\nError: Unimplemented command "+command+"\n```"
                        if response is not None:
                            print(response)
                            await ch.send(response)
                    except discord.HTTPException:
                        try:
                            fn = "cache/temp.txt"
                            _f = open(fn,"w")
                            _f.write(response)
                            _f.close()
                            _f = discord.File(fn)
                            await ch.send("Response too long for message.",file=_f)
                        except:
                            raise
                    except Exception as ex:
                        await ch.send("```\nError: "+repr(ex)+"\n```")
                    return
                else:
                    await ch.send(
                        "Error: Insufficient priviliges for command "+command+"\
.\nRequred level: **"+expNum(req)+"**, Current level: **"+expNum(u_perm)+"**")
                    return
    responses.append(None)

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

async def handleUpdate():
    global client,lastCheck
    if time.time()-lastCheck > 1:
        lastCheck = time.time()
        dtime = datetime.datetime.utcnow().timestamp()
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
        
@client.event
async def on_typing(channel,user,when):
    await handleUpdate()

async def handleMessage(message):
    msg = message.content
    user = message.author
    u_id = user.id
    u_perm = perms.get(u_id,0)
    if not len(msg)>1 or u_id==client.user.id:
        return
    try:
        responses = []
        await asyncio.wait_for(processMessage(message,responses),timeout=5)
    except Exception as ex:
        await message.channel.send("```\nError: "+repr(ex)+"\n```")
    return
        
@client.event
async def on_message(message):
    await handleMessage(message)
    await handleUpdate()

@client.event
async def on_message_edit(before,after):
    await handleMessage(after)
    await handleUpdate()

if __name__ == "__main__":
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
    
    try:
        _f = open("auth.json")
        token = ast.literal_eval(_f.read())["token"]
        print("Attempting to authorize with token "+token+":")
        _f.close()
        stored_vars = dict(globals())
        lastCheck = time.time()
        client.run(token)
    except:
        print("Unable to authorize bot token.")
        raise
