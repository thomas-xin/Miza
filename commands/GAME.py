try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from smath import *


class Text2048:
    is_command = True
    time_consuming = True
    directions = {
        b'\xe2\xac\x85': [0, 0],
        b'\xe2\xac\x86': [0, 1],
        b'\xe2\x9e\xa1': [0, 2],
        b'\xe2\xac\x87': [0, 3],
        b'\xe2\x86\xa9': [1, 4],
        b'\xe2\x86\x96': [16, 5],
        b'\xe2\x86\x97': [16, 6],
        b'\xe2\x86\x98': [16, 7],
        b'\xe2\x86\x99': [16, 8],
        b'\xe2\x86\x94': [16, 9],
        b'\xe2\x86\x95': [16, 10],
        b'\xf0\x9f\x94\x84': [16, 11],
        b'\xf0\x9f\x92\xa0': [16, 12],
        b'\xf0\x9f\x92\xaf': [16, 13],
    }
    multis = {
        5: [0, 1],
        6: [1, 2],
        7: [2, 3],
        8: [3, 0],
        9: [0, 2],
        10: [1, 3],
        11: [0,1,2,3],
        12: [i for i in range(16)],
        13: [i for i in range(100)],
    }
    numScore = lambda y, x: x * 2 ** (x + 1)

    def __init__(self):
        self.name = ["2048", "Text_2048"]
        self.min_level = 0
        self.description = "Plays a game of 2048 using reactions."
        self.usage = (
            "<board_size[4]> <verbose(?v)> <special_tiles(?s)> <public(?p)> "
            + "<insanity_mode(?i)> <special_controls(?c)> <easy_mode(?e)>"
        )

    def shiftTile(self, tiles, p1, p2):
        print(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        if tiles[x2][y2] <= 0:
            tiles[x2][y2] = tiles[x1][y1]
        elif type(tiles[x1][y1]) is float:
            if type(tiles[x2][y2]) is int:
                tiles[x2][y2] += round(tiles[x1][y1] * 10)
            else:
                tiles[x2][y2] += tiles[x1][y1]
        elif type(tiles[x2][y2]) is float:
            tiles[x2][y2] = round(tiles[x2][y2] * 10) + tiles[x1][y1]
        elif tiles[x2][y2] == tiles[x1][y1]:
            tiles[x2][y2] += 1
        else:
            return
        tiles[x1][y1] = 0

    def moveTiles(self, gamestate, direction, copy=False):
        if copy:
            tiles = copy.deepcopy(gamestate[0])
        else:
            tiles = gamestate[0]
        width = len(tiles)
        i = direction & 3
        if i & 2:
            r = reversed(range(width))
            d = -1
        else:
            r = range(width)
            d = 1
        a = 1
        for w in range(width - 1):
            changed = False
            if not i & 1:
                for x in r:
                    z = x - d
                    if z in r:
                        for y in r:
                            if tiles[x][y] > 0:
                                self.shiftTile(tiles, (x, y), (z, y))
                                changed = True
            else:
                for y in r:
                    z = y - d
                    if z in r:
                        for x in r:
                            if tiles[x][y] > 0:
                                self.shiftTile(tiles, (x, y), (x, z))
                                changed = True
            if not changed:
                break
            a = 0
        return tiles, a

    def randomSpam(self, gamestate, mode, pool, returns):
        gamestate[1] = gamestate[0]
        a = 1
        moved = {}
        shuffle(pool)
        while pool:
            move = pool[0]
            if not move in moved:
                gamestate[0], b = self.moveTiles(gamestate, move)
                self.spawn(gamestate[0], mode, 1)
                a &= b
                if b:
                    moved[pool[0]] = True
                    if len(moved) >= 4:
                        break
                else:
                    moved = {}
            pool = pool[1:]
        returns[0] = (gamestate, a)
                                        
    async def nextIter(self, message, gamestate, username, direction, mode):
        width = len(gamestate[-1])
        i = direction
        for z in range(len(gamestate)):
            for x in range(width):
                for y in range(width):
                    if gamestate[z][x][y] < 0:
                        gamestate[z][x][y] = 0
        if i == 4:
            a = gamestate[0] == gamestate[1]
            gamestate = gamestate[::-1]
        elif i is None:
            a = 0
        else:
            if i < 4:
                gamestate[1] = gamestate[0]
                gamestate[0], a = self.moveTiles(gamestate, i)
                self.spawn(gamestate[0], mode, 1)
            else:
                pool = list(self.multis[i])
                returns = [None]
                t = time.time()
                doParallel(self.randomSpam, [gamestate, mode, pool, returns])
                while returns[0] is None and time.time() - t < self._vars.timeout / 3:
                    await asyncio.sleep(0.2)
                if returns[0] is None:
                    return
                self.gamestate, a = returns[0]
        if not a:
            gsr = str(gamestate).replace("[", "A").replace("]", "B").replace(",", "C").replace("-", "D").replace(" ", "")
            orig = "\n".join(message.content.split("\n")[:1 + ("\n" == message.content[3])]).split("-")
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
            text += (
                ("+" + "-" * size) * width + "+" + "\nPlayer: "
                + username + "\nScore: " + str(score) + "```"
            )
            #print(text)
            await message.edit(content=text)
        elif not mode & 1:
            count = 0
            for x in range(width):
                for y in range(width):
                    if gamestate[0][x][y] > 0:
                        count += 1
            if count >= width ** 2:
                a = 1
                for i in range(4):
                    dump, b = self.moveTiles(gamestate, i, copy=True)
                    a &= b
                if a:
                    await message.clear_reactions()
                    gameover = ["🇬","🇦","🇲","🇪","⬛","🇴","🇻","3️⃣","🇷"]
                    for g in gameover:
                        await message.add_reaction(g)

    def spawn(self, gamestate, mode, count=1):
        width = len(gamestate)
        if count <= 0:
            return
        count *= width ** 2 / 16
        if count != int(count):
            count = int(count) + round(frand(count - int(count)))
        count = max(count, 1)
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

    async def _callback_(self, _vars, message, reaction, argv, user, perm, vals, **void):
        print(user, message, reaction, argv)
        u_id, mode = [int(x) for x in vals.split("_")]
        if reaction is not None and u_id != user.id and u_id != 0 and perm < 3:
            return
        gamestate = ast.literal_eval(argv.replace("A", "[").replace("B", "]").replace("C", ",").replace("D", "-"))
        if reaction is not None:
            reac = reaction
            if not reac in self.directions:
                return
            r = self.directions[reac]
            if not (r[0] & mode or r[0] == 0):
                return
            reaction = r[1]
        else:
            for react in self.directions:
                rval = self.directions[react][0]
                if rval & mode or rval == 0:
                    await message.add_reaction(react.decode("utf-8"))
            self.spawn(gamestate[0], mode, 1)
        if u_id == 0:
            username = "＠everyone"
        else:
            if user.id != u_id:
                u = await _vars.fetch_user(u_id)
                username = u.name
            else:
                username = user.name
        await self.nextIter(message, gamestate, username, reaction, mode)

    async def __call__(self, _vars, argv, user, flags, guild, **void):
        try:
            if not len(argv.replace(" ", "")):
                size = 4
            else:
                ans = await _vars.evalMath(argv, guild)
                size = int(ans)
                if not size > 1:
                    raise IndexError
        except:
            raise ValueError("Invalid board size.")
        if size > 11:
            raise OverflowError("Board size too large.")
        if "p" in flags:
            u_id = 0
        else:
            u_id = user.id
        mode = 0
        if "c" in flags:
            mode |= 16
        if "v" in flags:
            mode |= 8
        if "i" in flags:
            mode |= 4
        if "s" in flags:
            mode |= 2
        if "e" in flags:
            mode |= 1
        gamestate = [[[0 for y in range(size)] for x in range(size)]] * 2
        gsr = str(gamestate).replace("[", "A").replace("]", "B").replace(",", "C").replace("-", "D").replace(" ", "")
        text = (
            "```" + "\n" * (mode & 8 != 0) + "callback-game-text2048-"
            + str(u_id) + "_" + str(mode) + "-" + gsr + "\nStarting Game...```"
        )
        return text
