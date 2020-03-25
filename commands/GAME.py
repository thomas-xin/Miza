import discord
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
        self.flags = "vspice"

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
        for _ in loop(width - 1):
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
                    _, b = self.moveTiles(gamestate, i, copy=True)
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


class MimicConfig:
    is_command = True

    def __init__(self):
        self.name = ["PluralConfig", "MC", "PC"]
        self.min_level = 0
        self.description = "Modifies an existing webhook mimic's attributes."
        self.usage = "<0:mimic_id> <1:option(prefix)([name][username][nickname])([avatar][icon][url])([status][description])(gender)(birthday)> <2:new>"
    
    async def __call__(self, _vars, user, perm, flags, args, **void):
        mimicdb = _vars.data["mimics"]
        update = _vars.database["mimics"].update
        m_id = "&" + str(_vars.verifyID(args.pop(0)))
        if m_id not in mimicdb:
            raise LookupError("Target mimic ID not found.")
        if perm is not nan:
            mimics = mimicdb.setdefault(user.id, {})
            found = False
            for prefix in mimics:
                for mid in mimics[prefix]:
                    if mid == m_id:
                        found = True
            if not found:
                raise PermissionError("Target mimic does not belong to you.")
        else:
            mimics = mimicdb[mimicdb[m_id].u_id]
            found = True
        opt = args.pop(0)
        if args:
            new = " ".join(args)
        else:
            new = None
        mimic = mimicdb[m_id]
        if opt in ("name", "username", "nickname"):
            setting = "name"
        elif opt in ("avatar", "icon", "url"):
            setting = "url"
        elif opt in ("status", "description"):
            setting = "description"
        elif opt in ("gender", "birthday", "prefix"):
            setting = opt
        else:
            raise TypeError("Invalid target attribute.")
        if new is None:
            return (
                "```css\nCurrent " + setting + " for " 
                + uniStr(mimic.name) + ": " + str(mimic[setting]) + ".```"
            )
        if setting == "birthday":
            new = tparser.parse(new)
        elif setting == "prefix":
            if len(new) > 16:
                raise OverflowError("Must be 16 or fewer in length.")
            for prefix in mimics:
                for mid in mimics[prefix]:
                    if mid == m_id:
                        mimics[prefix].remove(m_id)
            if new in mimics:
                mimics[new].append(m_id)
            else:
                mimics[new] = hlist([m_id])
        elif setting != "description":
            if len(new) > 256:
                raise OverflowError("Must be 256 or fewer in length.")
        mimic[setting] = new
        update()
        return (
            "```css\nChanged " + setting + " for " 
            + uniStr(mimic.name) + " to " + str(new) + ".```"
        )


class Mimic:
    is_command = True

    def __init__(self):
        self.name = ["RolePlay", "Plural", "RP"]
        self.min_level = 0
        self.description = "Spawns a webhook mimic with an optional username and icon URL, or lists all mimics with their respective prefixes."
        self.usage = "<0:prefix> <1:user[]> <1:name[]> <2:url[]> <disable(?d)>"
        self.flags = "ed"
    
    async def __call__(self, _vars, message, user, flags, args, argv, **void):
        mimicdb = _vars.data["mimics"]
        mimics = mimicdb.setdefault(user.id, {})
        update = _vars.database["mimics"].update
        if not argv:
            if "d" in flags:
                _vars.data["mimics"].pop(user.id)
                update()
                return (
                    "```css\nSuccessfully removed all webhook mimics for "
                    + uniStr(user) + ".```"
                )
            if not mimics:
                return (
                    "```css\nNo webhook mimics currently enabled for "
                    + uniStr(user) + ".```"
                )
            key = lambda x: "⟨" + ", ".join(i + ": " + str(_vars.data["mimics"][i].name) for i in iter(x)) + "⟩"
            return (
                "Currently enabled webhook mimics for **"
                + str(user) + "**: ```ini\n"
                + strIter(mimics, key=key) + "```"
            )
        u_id = user.id
        prefix = args.pop(0)
        if not prefix:
            raise IndexError("Prefix must not be empty.")
        if "d" in flags:
            try:
                mlist = mimics[prefix]
            except KeyError:
                raise TypeError("Please enter prefix of mimic to delete.")
            if len(mlist):
                m_id = mlist.popleft()
                mimic = _vars.data["mimics"].pop(m_id)
            else:
                mimicdb.pop(prefix)
                update()
                raise KeyError("Unable to find webhook mimic.")
            if not mlist:
                mimics.pop(prefix)
            update()
            return (
                "```css\nSuccessfully removed webhook mimic " + uniStr(mimic.name)
                + " for " + uniStr(user) + ".```"
            )
        ctime = datetime.datetime.utcnow()
        mid = discord.utils.time_snowflake(ctime)
        m_id = "&" + str(mid)
        mimic = None
        if len(args):
            if len(args) > 1:
                url = args[-1]
                name = " ".join(args[:-1])
            else:
                mim = 0
                try:
                    mim = _vars.verifyID(args[-1])
                    user = await _vars.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    name = user.name
                    url = str(user.avatar_url)
                except:
                    try:
                        mimi = _vars.get_mimic(mim)
                        mimic = copy.deepcopy(mimi)
                        mimic.id = m_id
                        mimic.u_id = u_id
                        mimic.prefix = prefix
                        mimic.count = mimic.total = 0
                        mimic.created_at = ctime
                    except:
                        name = args[0]
                        url = "https://cdn.discordapp.com/embed/avatars/0.png"
        else:
            name = user.name
            url = str(user.avatar_url)
        while m_id in mimics:
            mid += 1
            m_id = "&" + str(mid)
        if mimic is None:
            mimic = freeClass(
                id=m_id,
                u_id=u_id,
                prefix=prefix,
                name=name,
                url=url,
                description="",
                gender="N/A",
                birthday=ctime,
                created_at=ctime,
                count=0,
                total=0,
            )
        mimicdb[m_id] = mimic
        if prefix in mimics:
            mimics[prefix].append(m_id)
        else:
            mimics[prefix] = hlist([m_id])
        update()
        return (
            "```css\nSuccessfully added webhook mimic " + uniStr(name)
            + " with prefix " + prefix + " and ID " + m_id + ".```"
        )


class updateMimics:
    is_database = True
    name = "mimics"
    user = True

    def __init__(self):
        pass

    async def _nocommand_(self, message, **void):
        user = message.author
        if user.id in self.data:
            _vars = self._vars
            perm = _vars.getPerms(user.id, message.guild)
            admin = not inf > perm
            if message.guild is not None:
                try:
                    enabled = _vars.data["enabled"][message.channel.id]
                except KeyError:
                    enabled = ()
            else:
                enabled = list(_vars.categories)
            if admin or "game" in enabled:
                database = self.data[user.id]
                msg = message.content
                try:
                    channel = message.channel
                    w = await _vars.ensureWebhook(channel)
                    found = False
                    for line in msg.split("\n"):
                        if len(line) > 2 and " " in line:
                            i = line.index(" ")
                            prefix = line[:i]
                            line = line[i + 1:].strip(" ")
                            if prefix in database:
                                mimics = database[prefix]
                                if mimics:
                                    for m in mimics:
                                        mimic = self.data[m]
                                        await w.send(line, username=mimic.name, avatar_url=mimic.url)
                                        mimic.count += 1
                                        mimic.total += len(line)
                                    if not found:
                                        await _vars.silentDelete(message)
                                    found = True
                        elif not found:
                            break
                except Exception as ex:
                    await channel.send(repr(ex))

    async def __call__(self):
        pass