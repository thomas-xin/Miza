import ast, copy
from smath import *


class text2048:
    is_command = True
    directions = ["⬅", "⬆", "➡", "⬇️", "↩"]
    numScore = lambda self, x: x * 2 ** (x + 1)

    def __init__(self):
        self.name = ["2048", "text_2048"]
        self.min_level = 1
        self.description = "Plays a game of 2048 using reactions."
        self.usage = "<board_size:[4]> <public:(?p)> <insanity_mode:(?i)> <easy_mode:(?e)> <specials_enabled:(?s)>"

    async def nextIter(self, message, gamestate, username, direction, mode):
        width = len(gamestate[-1])
        a = 0
        i = direction
        if i == 4:
            if not mode & 1:
                return
            gamestate = gamestate[::-1]
        for z in range(len(gamestate)):
            for x in range(width):
                for y in range(width):
                    if gamestate[z][x][y] < 0:
                        gamestate[z][x][y] = 0
        gamestate[1] = copy.deepcopy(gamestate[0])
        tiles = gamestate[0]
        if i != 4 and i is not None:
            a = 1
            for w in range(width - 1):
                if i & 1 == 0:
                    z = (i ^ 2) - 1
                    for x in range(width):
                        for y in range(width):
                            if x - z >= 0 and x - z < width:
                                if tiles[x][y] > 0:
                                    if tiles[x - z][y] <= 0:
                                        tiles[x - z][y] = tiles[x][y]
                                        tiles[x][y] = 0
                                        a = 0
                                    elif type(tiles[x][y]) is float:
                                        if type(tiles[x - z][y]) is int:
                                            tiles[x - z][y] += round(tiles[x][y] * 10)
                                            tiles[x][y] = 0
                                        else:
                                            tiles[x - z][y] += tiles[x][y]
                                            tiles[x][y] = 0
                                        a = 0
                                    elif type(tiles[x - z][y]) is float:
                                        tiles[x - z][y] = round(tiles[x - z][y] * 10) + tiles[x][y]
                                        tiles[x][y] = 0
                                        a = 0
                                    elif tiles[x - z][y] == tiles[x][y]:
                                        tiles[x - z][y] += 1
                                        tiles[x][y] = 0
                                        a = 0
                else:
                    z = (i ^ 2) - 2
                    for x in range(width):
                        for y in range(width):
                            if y - z >= 0 and y - z < width:
                                if tiles[x][y] > 0:
                                    if tiles[x][y - z] <= 0:
                                        tiles[x][y - z] = tiles[x][y]
                                        tiles[x][y] = 0
                                        a = 0
                                    elif type(tiles[x][y]) is float:
                                        if type(tiles[x][y - z]) is int:
                                            tiles[x][y - z] += round(tiles[x][y] * 10)
                                            tiles[x][y] = 0
                                        else:
                                            tiles[x][y - z] += tiles[x][y]
                                            tiles[x][y] = 0
                                        a = 0
                                    elif type(tiles[x][y - z]) is float:
                                        tiles[x][y - z] = round(tiles[x][y - z] * 10) + tiles[x][y]
                                        tiles[x][y] = 0
                                        a = 0
                                    elif tiles[x][y - z] == tiles[x][y]:
                                        tiles[x][y - z] += 1
                                        tiles[x][y] = 0
                                        a = 0
        if not a:
            if i != 4:
                self.spawn(gamestate[0], mode, 1)
            gsr = str(gamestate).replace("[", "A").replace("]", "B").replace(",", "C").replace("-", "D").replace(" ", "")
            orig = message.content.split("\n")[0].split("-")
            last = "-".join(orig[:-1])
            text = last + "-" + gsr + "\n"
            score = 0
            largest = numpy.max(gamestate[0])
            size = max(3, int(1 + math.log10(2 ** largest)))
            for y in range(width):
                text += ("+" + "-" * size) * width + "+\n"
                for x in range(width):
                    n = gamestate[0][x][y]
                    if type(n) is int and n > 0:
                        score += self.numScore(n - 1)
                    if n <= 0:
                        num = ""
                    elif type(n) is float:
                        num = "×" + str(1 << round(n * 10))
                    else:
                        num = str(1 << n)
                    empty = size - len(num)
                    text += "|" + " " * (empty + 1 >> 1) + num + " " * (empty >> 1)
                text += "|\n"
            text += ("+" + "-" * size) * width + "+" + "\nPlayer: " + username + "\nScore: " + str(score) + "```"
            doParallel(print, [text])
            await message.edit(content=text)

    def spawn(self, gamestate, mode, count=1):
        width = len(gamestate)
        isZero = count <= 0
        count *= width ** 2 / 16
        if count != int(count):
            count = int(count) + round(frand(count - int(count)))
        if count < 1 and not isZero:
            count = 1
        largest = numpy.max(gamestate[0])
        attempts = 0
        i = 0
        while i < count and attempts < 256:
            attempts += 1
            v = (not xrand(4)) + 1
            if mode & 4:
                v += max(0, xrand(largest) - 1)
            if mode & 2 and not xrand(16):
                v = int(sqrt(max(1, v))) / 10
            x = xrand(width)
            y = xrand(width)
            if gamestate[x][y] <= 0:
                gamestate[x][y] = v
                i += 1

    async def _callback_(self, client, message, reaction, argv, user, perm, vals, **void):
        u_id, mode = [int(x) for x in vals.split("_")]
        if reaction is not None and u_id != user.id and u_id != 0 and perm < 3:
            return
        gamestate = ast.literal_eval(argv.replace("A", "[").replace("B", "]").replace("C", ",").replace("D", "-"))
        if reaction is not None:
            try:
                reaction = self.directions.index(str(reaction))
            except IndexError:
                return
        else:
            for react in self.directions:
                if mode & 1 or react != "↩":
                    await message.add_reaction(react)
            self.spawn(gamestate[0], mode, 1)
        if u_id == 0:
            username = "@everyone"
        else:
            if user.id != u_id:
                u = await client.fetch_user(u_id)
                username = u.name
            else:
                username = user.name
        await self.nextIter(message, gamestate, username, reaction, mode)

    async def __call__(self, _vars, argv, user, flags, **void):
        try:
            if not len(argv.replace(" ", "")):
                size = 4
            else:
                size = int(_vars.evalMath(argv))
                if not size > 1:
                    raise
        except:
            raise ValueError("Invalid board size.")
        if size > 11:
            raise OverflowError("Board size too large.")
        if "p" in flags:
            u_id = 0
        else:
            u_id = user.id
        mode = 0
        if "i" in flags:
            mode |= 4
        if "s" in flags:
            mode |= 2
        if "e" in flags:
            mode |= 1
        gamestate = [[[0 for y in range(size)] for x in range(size)]] * 2
        gsr = str(gamestate).replace("[", "A").replace("]", "B").replace(",", "C").replace("-", "D").replace(" ", "")
        text = "```callback-game-text2048-" + str(u_id) + "_" + str(mode) + "-" + gsr + "\nStarting Game...```"
        return text
