from smath import *

class help:
    is_command = True
    def __init__(self):
        self.name = ["?"]
        self.minm = -inf
        self.desc = "Shows a list of usable commands."
        self.usag = '<command:[]> <verbose:(?v)>'
    async def __call__(self,_vars,args,user,guild,flags,**void):
        try:
            enabled = _vars.enabled[guild.id]
        except:
            enabled = _vars.enabled[guild.id] = ["string","admin"]
            _vars.update()
        categories = _vars.categories
        commands = []
        for catg in categories:
            if catg in enabled or catg == "main":
                commands += categories[catg]
        u_perm = _vars.getPerms(user,guild)
        verb = "v" in flags
        argv = " ".join(args)
        show = []
        for a in args:
            if a in categories and (a in enabled or a=="main"):
                show.append("\nCommands for **"+user.name+"** in **"+guild.name+"** in category **"+a+"**:")
                for com in categories[a]:
                    name = com.__name__
                    minm = com.minm
                    desc = com.desc
                    usag = com.usag
                    if minm>u_perm:
                        continue
                    newstr = "\n`"+name+"`\nAliases: "+str(com.name)+"\nEffect: "+desc+"\nUsage: "+usag+"\nRequired permission level: **__"+str(minm)+"__**"
                    show.append(newstr)
        if not show:
            for com in commands:
                name = com.__name__
                minm = com.minm
                desc = com.desc
                usag = com.usag
                if minm>u_perm:
                    continue
                found = False
                for n in com.name:
                    if n in argv:
                        found = True
                if found:
                    newstr = "\n`"+name+"`\nAliases: "+str(com.name)+"\nEffect: "+desc+"\nUsage: "+usag+"\nRequired permission level: **__"+str(minm)+"__**"
                    if (not len(show)) or len(show[-1])<len(newstr):
                        show = [newstr]
        if not show:
            for com in commands:
                name = com.__name__
                minm = com.minm
                desc = com.desc
                usag = com.usag
                if not minm>u_perm:
                    if desc != "":
                        if not verb:
                            show.append("`"+name+" "+usag+"`")
                        else:
                            show.append("\n`"+com.__name__+"`\nEffect: "+com.desc+"\nUsage: "+name+" "+usag)
            return "Commands for **"+user.name+"** in **"+guild.name+"**:\n"+"\n".join(show)
        return "\n".join(show)
        
class clearCache:
    is_command = True
    def __init__(self):
        self.name = ["cc"]
        self.minm = 1
        self.desc = "Clears all cached data."
        self.usag = ''
    async def __call__(self,client,_vars,**void):
        client.clear()
        _vars.resetGlobals()
        return "```\nCache cleared!```"

class changePerms:
    is_command = True
    def __init__(self):
        self.name = ["perms","perm","changeperm","changePerm","changeperms"]
        self.minm = -inf
        self.desc = "Shows or changes a user's permission level."
        self.usag = '<0:user:{self}> <1:value{curr}>'
    async def __call__(self,client,_vars,args,user,guild,**void):
        if len(args) < 2:
            if len(args) < 1:
                t_user = user
            else:
                t_user = await client.fetch_user(_vars.verifyID(args[0]))
            print(t_user)
            t_perm = _vars.getPerms(t_user.id,guild)
        else:
            c_perm = _vars.evalMath(" ".join(args[1:]))
            s_user = user
            s_perm = _vars.getPerms(s_user.id,guild)
            t_user = await client.fetch_user(_vars.verifyID(args[0]))
            t_perm = _vars.getPerms(t_user.id,guild)
            m_perm = max(t_perm,c_perm,1)+1
            if not(s_perm==m_perm) and not(s_perm<m_perm):
                g_perm = _vars.perms.get(guild.id,{})
                g_perm.update({t_user.id:c_perm})
                _vars.perms[guild.id] = g_perm
                _vars.update()
                return "Changed permissions for **"+t_user.name+"** in **"+guild.name+"** from \
**__"+expNum(t_perm,12,4)+"__** to **__"+expNum(c_perm,12,4)+"__**."
            else:
                return "```\nError:\n```\nInsufficient priviliges to change permissions for \
**"+t_user.name+"** in **"+guild.name+"** from **__"+expNum(t_perm,12,4)+"__** to \
**__"+expNum(c_perm,12,4)+"__**.\nRequired level: \
**__"+expNum(m_perm,12,4)+"__**, Current level: **__"+expNum(s_perm,12,4)+"__**"
        return "Current permissions for **"+t_user.name+"** in **"+guild.name+"**: **__"+expNum(t_perm,12,4)+"__**"
    
class enableCommand:
    is_command = True
    def __init__(self):
        self.name = ["ec","enable"]
        self.minm = 3
        self.desc = "Shows, enables, or disables a command category in the current server."
        self.usag = '<command:{all}> <show:[?s]> <enable:(?e)> <disable:(?d)>'
    async def __call__(self,client,_vars,argv,flags,guild,**void):
        catg = argv.lower()
        print(catg)
        if not catg:
            if "e" in flags:
                categories = list(_vars.categories)
                categories.remove("main")
                _vars.enabled[guild.id] = categories
                _vars.update()
                return "Enabled all command categories in **"+guild.name+"**."
            if "d" in flags:
                _vars.enabled[guild.id] = []
                _vars.update()
                return "Disabled all command categories in **"+guild.name+"**."
            return "Currently enabled command categories in **"+guild.name+"**:\n\
```\n"+str(["main"]+_vars.enabled.get(guild.id,["math","admin"]))+"```"
        else:
            if not catg in _vars.categories:
                return "Error: Unknown command category **"+argv+"**."
            else:
                try:
                    enabled = _vars.enabled[guild.id]
                except:
                    enabled = {}
                    _vars.enabled[guild.id] = enabled
                if "e" in flags:
                    if catg in enabled:
                        return "Error: command category **"+catg+"\
** is already enabled in **"+guild.name+"**."
                    enabled.append(catg)
                    _vars.update()
                    return "Enabled command category **"+catg+"** in **"+guild.name+"**."
                if "d" in flags:
                    if catg not in enabled:
                        return "Error: command category **"+catg+"\
** is not currently enabled in **"+guild.name+"**."
                    enabled.remove(catg)
                    _vars.update()
                    return "Disabled command category **"+catg+"** in **"+guild.name+"**."
                return "Command category **"+catg+"** is currently\
"+" not"*(catg not in enabled)+" enabled in **"+guild.name+"**."
        
class shutdown:
    is_command = True
    def __init__(self):
        self.name = ["gtfo"]
        self.minm = inf
        self.desc = "Shuts down the bot."
        self.usag = ''
    async def __call__(self,client,channel,**void):
        await channel.send("Shutting down... :wave:")
        for vc in client.voice_clients:
            await vc.disconnect(force=True)
        await client.close()
        sys.exit()
        quit()
