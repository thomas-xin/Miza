import ast,copy
from smath import *

class text2048:
    is_command = True
    directions = ["⬅","⬆","➡","⬇️","↩"]
    numScore = lambda x: x*2**(x+1)
    def __init__(self):
        self.name = ["2048","text_2048"]
        self.minm = 1
        self.desc = "Plays a game of 2048 using reactions."
        self.usag = '<board_size:[4]> <public:(?p)> <easy_mode:(?e)>'
    async def nextIter(self,message,gamestate,username,direction,mode):
        width = len(gamestate[-1])
        a = 0
        i = direction
        if i == 4:
            if mode != 1:
                return
            gamestate = gamestate[::-1]
        gamestate[1] = copy.deepcopy(gamestate[0])
        if i!=4 and i is not None:
            a = 1
            for w in range(width-1):
                if i&1==0:
                    z = (i^2)-1
                    for x in range(width):
                        for y in range(width):
                            if x-z>=0 and x-z<width:
                                if gamestate[0][x][y] != 0:
                                    if gamestate[0][x-z][y] == 0:
                                        gamestate[0][x-z][y] = gamestate[0][x][y]
                                        gamestate[0][x][y] = 0
                                        a = 0
                                    elif gamestate[0][x-z][y] == gamestate[0][x][y]:
                                        gamestate[0][x-z][y] += 1
                                        gamestate[0][x][y] = 0
                                        a = 0
                else:
                    z = (i^2)-2
                    for x in range(width):
                        for y in range(width):
                            if y-z>=0 and y-z<width:
                                if gamestate[0][x][y] != 0:
                                    if gamestate[0][x][y-z] == 0:
                                        gamestate[0][x][y-z] = gamestate[0][x][y]
                                        gamestate[0][x][y] = 0
                                        a = 0
                                    elif gamestate[0][x][y-z] == gamestate[0][x][y]:
                                        gamestate[0][x][y-z] += 1
                                        gamestate[0][x][y] = 0
                                        a = 0
        if not a:
            if i!=4:
                self.spawn(gamestate[0],1)
            gsr = str(gamestate).replace("[","A").replace("]","B").replace(",","C").replace(" ","")
            orig = message.content.split("\n")[0].split("-")
            last = "-".join(orig[:-1])
            text = last+"-"+gsr+"\n"
            score = 0
            for y in range(width):
                text += "+---"*width+"+\n"
                for x in range(width):
                    n = gamestate[0][x][y]
                    if n:
                        score += self.numScore(n-1)
                    if n > 6:
                        text += "|"+str(1<<n)
                    elif n > 3:
                        text += "| "+str(1<<n)
                    elif n > 0:
                        text += "| "+str(1<<n)+" "
                    else:
                        text += "|   "
                text += "|\n"
            text += "+"+"---+"*width+"\nPlayer: "+username+"\nScore: "+str(score)+"```"
            print(text)
            await message.edit(content=text)
    def spawn(self,gamestate,count=1):
        width = len(gamestate)
        i = 0
        while i < count:
            v = (not xrand(4))+1
            x = xrand(width)
            y = xrand(width)
            if gamestate[x][y] == 0:
                gamestate[x][y] = v
                i += 1
    async def _callback_(self,client,message,reaction,argv,user,perm,vals,**void):
        u_id,mode = [int(x) for x in vals.split("_")]
        if reaction is not None and u_id!=user.id and u_id!=0 and perm<3:
            return
        gamestate = ast.literal_eval(argv.replace("A","[").replace("B","]").replace("C",","))
        if reaction is not None:
            try:
                reaction = self.directions.index(str(reaction))
            except:
                return
        else:
            for react in self.directions:
                if mode==1 or react!="↩":
                    await message.add_reaction(react)
            self.spawn(gamestate[0],1)
        if u_id == 0:
            username = "@everyone"
        else:
            if user.id != u_id:
                u = await client.fetch_user(u_id)
                username = u.name
            else:
                username = user.name
        await self.nextIter(message,gamestate,username,reaction,mode)
    async def __call__(self,_vars,argv,user,flags,**void):
        try:
            if not len(argv.replace(" ","")):
                size = 4
            else:
                size = int(_vars.evalMath(argv))
                assert(size>1)
        except:
            raise ValueError("Invalid board size.")
        if size > 12:
            raise OverflowError("Board size too large.")
        if "p" in flags:
            u_id = 0
        else:
            u_id = user.id
        if "e" in flags:
            mode = 1
        else:
            mode = 0
        gamestate = [[[0 for y in range(size)] for x in range(size)]]*2
        gsr = str(gamestate).replace("[","A").replace("]","B").replace(",","C").replace(" ","")
        text = "```callback-game-text2048-"+str(u_id)+"_"+str(mode)+"-"+gsr+"\nStarting Game...```"
        return text
        
