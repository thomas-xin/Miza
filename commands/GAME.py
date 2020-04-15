try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *


class Text2048(Command):
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
    name = ["2048", "Text_2048"]
    min_level = 0
    description = "Plays a game of 2048 using reactions."
    usage = (
        "<board_size[4]>  <show_debug(?z)> <special_tiles(?s)> <public(?p)> "
        + "<insanity_mode(?i)> <special_controls(?c)> <easy_mode(?e)>"
    )
    flags = "vzpice"

    def shiftTile(self, tiles, p1, p2):
        # print(p1, p2)
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
            return False
        tiles[x1][y1] = 0
        return True

    def moveTiles(self, gamestate, direction):
        tiles = gamestate[0]
        width = len(tiles)
        i = direction & 3
        if i & 2:
            r = range(width - 1, -1, -1)
            d = -1
        else:
            r = range(width)
            d = 1
        # print(direction, i, list(r), d)
        a = 1
        for _ in loop(width - 1):
            changed = False
            if not i & 1:
                for x in r:
                    z = x - d
                    if z in r:
                        for y in r:
                            if tiles[x][y] > 0:
                                changed |= self.shiftTile(tiles, (x, y), (z, y))
            else:
                for y in r:
                    z = y - d
                    if z in r:
                        for x in r:
                            if tiles[x][y] > 0:
                                changed |= self.shiftTile(tiles, (x, y), (x, z))
            if not changed:
                break
            a = 0
        return tiles, a

    def randomSpam(self, gamestate, mode, pool, returns):
        gamestate[1] = gamestate[0]
        a = i = 1
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
            if not i % 20:
                time.sleep(0.01)
            i += 1
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
            # print(text)
            await message.edit(content=text)
        elif not mode & 1:
            count = 0
            for x in range(width):
                for y in range(width):
                    if gamestate[0][x][y] > 0:
                        count += 1
            if count >= width ** 2:
                gamecopy = list(gamestate)
                gamecopy[0] = [list(l) for l in gamestate[0]]
                a = 1
                for i in range(4):
                    try:
                        gamecopy, b = self.moveTiles(gamecopy, i)
                        a &= b
                    except TypeError:
                        pass
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
        # print(user, message, reaction, argv)
        u_id, mode = [int(x) for x in vals.split("_")]
        if reaction is not None and u_id != user.id and u_id != 0 and perm < 3:
            return
        gamestate = ast.literal_eval(argv.replace("A", "[").replace("B", "]").replace("C", ",").replace("D", "-"))
        if reaction is not None:
            reac = reaction
            if not reac in self.directions:
                return
            r = self.directions[reac]
            if not (r[0] & mode or not r[0]):
                return
            reaction = r[1]
        else:
            for react in self.directions:
                rval = self.directions[react][0]
                if rval & mode or not rval:
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
        if "z" in flags:
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

    
class Dogpile(Command):
    server_only = True
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically imitate users when 3+ of the same messages are posted in a row."
    usage = "<enable(?e)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, flags, guild, **void):
        update = self.data.dogpiles.update
        _vars = self._vars
        following = _vars.data.dogpiles
        curr = following.get(guild.id, False)
        if "d" in flags:
            if guild.id in following:
                following.pop(guild.id)
                update()
            return "```css\nDisabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        elif "e" in flags or "a" in flags:
            following[guild.id] = True
            update()
            return "```css\nEnabled dogpile imitating for [" + noHighlight(guild.name) + "].```"
        else:
            return (
                "```ini\nDogpile imitating is currently " + "not " * (not curr)
                + "enabled in [" + noHighlight(guild.name) + "].```"
            )


class MathQuiz(Command):
    name = ["MathTest"]
    min_level = 1
    description = "Starts a math quiz in the current channel."
    usage = "<mode(easy)(hard)> <disable(?d)>"
    flags = "aed"

    async def __call__(self, channel, flags, argv, **void):
        mathdb = self._vars.database.mathtest
        if "d" in flags:
            if channel.id in mathdb.data:
                mathdb.data.pop(channel.id)
            return "```css\nDisabled math quizzes for " + sbHighlight(channel.name) + ".```"
        if not argv:
            argv = "easy"
        elif argv not in ("easy", "hard"):
            raise TypeError("Invalid quiz mode.")
        mathdb.data[channel.id] = freeClass(mode=argv, answer=None)
        return "```css\nEnabled " + argv + " math quiz for " + sbHighlight(channel.name) + ".```"


class MimicConfig(Command):
    name = ["PluralConfig", "RPConfig"]
    min_level = 0
    description = "Modifies an existing webhook mimic's attributes."
    usage = (
        "<0:mimic_id> <1:option(prefix)([name][username][nickname])([avatar][icon][url])"
        + "([status][description])(gender)(birthday)> <2:new>"
    )
    
    async def __call__(self, _vars, user, perm, flags, args, **void):
        update = self.data.mimics.update
        mimicdb = _vars.data.mimics
        m_id = "&" + str(verifyID(args.pop(0)))
        if m_id not in mimicdb:
            raise LookupError("Target mimic ID not found.")
        if not isnan(perm):
            mimics = mimicdb.setdefault(user.id, {})
            found = 0
            for prefix in mimics:
                found += mimics.count(m_id)
            if not found:
                raise PermissionError("Target mimic does not belong to you.")
        else:
            mimics = mimicdb[mimicdb[m_id].u_id]
            found = True
        mimic = mimicdb[m_id]
        opt = args.pop(0).lower()
        if args:
            new = " ".join(args)
        else:
            new = None
        if opt in ("name", "username", "nickname"):
            setting = "name"
        elif opt in ("avatar", "icon", "url"):
            setting = "url"
        elif opt in ("status", "description"):
            setting = "description"
        elif opt in ("gender", "birthday", "prefix"):
            setting = opt
        elif opt in ("auto", "copy", "user", "auto"):
            setting = "auto"
        else:
            raise TypeError("Invalid target attribute.")
        if new is None:
            return (
                "```ini\nCurrent " + setting + " for [" 
                + noHighlight(mimic.name) + "]: [" + noHighlight(mimic[setting]) + "].```"
            )
        if setting == "birthday":
            new = tparser.parse(new).timestamp()
        elif setting == "prefix":
            if len(new) > 16:
                raise OverflowError("Must be 16 or fewer in length.")
            for prefix in mimics:
                try:
                    mimics[prefix].remove(m_id)
                except (ValueError, IndexError):
                    pass
            if new in mimics:
                mimics[new].append(m_id)
            else:
                mimics[new] = [m_id]
        elif setting == "url":
            new = await _vars.followURL(verifyURL(new))
        elif setting == "auto":
            if new.lower() in ("none", "null", "0", "false", "f"):
                new = None
            else:
                mim = None
                try:
                    mim = verifyID(new)
                    user = await _vars.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    new = user.id
                except:
                    try:
                        mimi = _vars.get_mimic(mim, user)
                        new = mimi.id
                    except:
                        raise LookupError("Target user or mimic ID not found.")
        elif setting != "description":
            if len(new) > 256:
                raise OverflowError("Must be 256 or fewer in length.")
        name = mimic.name
        mimic[setting] = new
        update()
        return (
            "```css\nChanged " + setting + " for [" 
            + noHighlight(name) + "] to [" + noHighlight(new) + "].```"
        )


class Mimic(Command):
    name = ["RolePlay", "Plural", "RP", "RPCreate"]
    min_level = 0
    description = "Spawns a webhook mimic with an optional username and icon URL, or lists all mimics with their respective prefixes."
    usage = "<0:prefix> <1:user[]> <1:name[]> <2:url[]> <disable(?d)>"
    flags = "aed"
    
    async def __call__(self, _vars, message, user, perm, flags, args, argv, **void):
        update = self.data.mimics.update
        mimicdb = _vars.data.mimics
        if len(args) == 1 and "d" not in flags:
            user = await _vars.fetch_user(verifyID(argv))
        mimics = mimicdb.setdefault(user.id, {})
        if not argv or (len(args) == 1 and "d" not in flags):
            if "d" in flags:
                mimicdb.pop(user.id)
                update()
                return (
                    "```css\nSuccessfully removed all webhook mimics for ["
                    + noHighlight(user) + "].```"
                )
            for k in tuple(mimics):
                if not mimics[k]:
                    mimics.pop(k)
                    update()
            if not mimics:
                return (
                    "```ini\nNo webhook mimics currently enabled for ["
                    + noHighlight(user) + "].```"
                )
            key = lambda x: limStr("⟨" + ", ".join(i + ": " + (str(noHighlight(mimicdb[i].name)), "[<@" + str(getattr(mimicdb[i], "auto", "None")) + ">]")[bool(getattr(mimicdb[i], "auto", None))] for i in iter(x)) + "⟩", 1900 / len(mimics))
            return (
                "Currently enabled webhook mimics for **"
                + discord.utils.escape_markdown(str(user)) + "**: ```ini\n"
                + strIter(mimics, key=key) + "```"
            )
        u_id = user.id
        prefix = args.pop(0)
        if not prefix:
            raise IndexError("Prefix must not be empty.")
        if len(prefix) > 16:
            raise OverflowError("Must be 16 or fewer in length.")
        if "d" in flags:
            try:
                mlist = mimics[prefix]
                if mlist is None:
                    raise KeyError
                if len(mlist):
                    m_id = mlist.pop(0)
                    mimic = mimicdb.pop(m_id)
                else:
                    mimics.pop(prefix)
                    update()
                    raise KeyError("Unable to find webhook mimic.")
                if not mlist:
                    mimics.pop(prefix)
            except KeyError:
                mimic = _vars.get_mimic(prefix, user)
                if not isnan(perm) and mimic.u_id != user.id:
                    raise PermissionError("Target mimic does not belong to you.")
                mimics = mimicdb[mimic.u_id]
                user = await _vars.fetch_user(mimic.u_id)
                m_id = mimic.id
                for prefix in mimics:
                    try:
                        mimics[prefix].remove(m_id)
                    except (ValueError, IndexError):
                        pass
                mimicdb.pop(mimic.id)
            update()
            return (
                "```css\nSuccessfully removed webhook mimic [" + mimic.name
                + "] for [" + noHighlight(user) + "].```"
            )
        if sum(len(i) for i in iter(mimics.values())) >= 256:
            raise OverflowError(
                "Mimic list for " + str(user)
                + " has reached the maximum of 256 items. "
                + "Please remove an item to add another."
            )
        dop = None
        utcn = datetime.datetime.utcnow()
        mid = discord.utils.time_snowflake(utcn)
        ctime = utcn.timestamp()
        m_id = "&" + str(mid)
        mimic = None
        if len(args):
            if len(args) > 1:
                url = await _vars.followURL(verifyURL(args[-1]))
                name = " ".join(args[:-1])
            else:
                mim = 0
                try:
                    mim = verifyID(args[-1])
                    user = await _vars.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    dop = user.id
                    name = user.name
                    url = strURL(user.avatar_url)
                except:
                    try:
                        mimi = _vars.get_mimic(mim, user)
                        dop = mimi.id
                        mimic = copy.deepcopy(mimi)
                        mimic.id = m_id
                        mimic.u_id = u_id
                        mimic.prefix = prefix
                        mimic.count = mimic.total = 0
                        mimic.created_at = ctime
                        mimic.auto = dop
                    except:
                        name = args[0]
                        url = "https://cdn.discordapp.com/embed/avatars/0.png"
        else:
            name = user.name
            url = strURL(user.avatar_url)
        while m_id in mimics:
            mid += 1
            m_id = "&" + str(mid)
        if mimic is None:
            mimic = freeClass(
                id=m_id,
                u_id=u_id,
                prefix=prefix,
                auto=dop,
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
            mimics[prefix] = [m_id]
        update()
        return (
            "```css\nSuccessfully added webhook mimic [" + mimic.name
            + "] with prefix [" + mimic.prefix + "] and ID [" + mimic.id + "]"
            + (", bound to user [<" + "@" * (type(dop) is int) + str(dop) + ">]") * (dop is not None) + ".```"
        )


class RPSend(Command):
    name = ["MimicSend", "PluralSend"]
    min_level = 0
    description = "Sends a message using a webhook mimic, to the target channel."
    usage = "<0:mimic> <1:channel> <2:string>"

    async def __call__(self, _vars, user, perm, args, **void):
        update = self.data.mimics.update
        mimicdb = _vars.data.mimics
        mimics = mimicdb.setdefault(user.id, {})
        prefix = args.pop(0)
        c_id = verifyID(args.pop(0))
        channel = await _vars.fetch_channel(c_id)
        guild = channel.guild
        w = await _vars.ensureWebhook(channel)
        msg = " ".join(args)
        if not msg:
            raise IndexError("Message is empty.")
        perm = _vars.getPerms(user.id, guild)
        try:
            mlist = mimics[prefix]
            if mlist is None:
                raise KeyError
            m = [_vars.get_mimic(verifyID(p)) for p in mlist]
        except KeyError:
            mimic = _vars.get_mimic(verifyID(prefix))
            if not isnan(perm) and mimic.u_id != user.id:
                raise PermissionError("Target mimic does not belong to you.")
            m = [mimic]
        admin = not inf > perm
        try:
            enabled = _vars.data.enabled[channel.id]
        except KeyError:
            enabled = ()
        if not admin and "game" not in enabled:
            raise PermissionError("Not permitted to send into target channel.")
        for mimic in m:
            await _vars.database.mimics.updateMimic(mimic, guild)
            name = mimic.name
            url = mimic.url
            await w.send(msg, username=name, avatar_url=url)
            mimic.count += 1
            mimic.total += len(msg)


class UpdateMimics(Database):
    name = "mimics"
    user = True

    async def _nocommand_(self, message, **void):
        if not message.content:
            return
        user = message.author
        if user.id in self.data:
            _vars = self._vars
            perm = _vars.getPerms(user.id, message.guild)
            admin = not inf > perm
            if message.guild is not None:
                try:
                    enabled = _vars.data.enabled[message.channel.id]
                except KeyError:
                    enabled = ()
            else:
                enabled = list(_vars.categories)
            if admin or "game" in enabled:
                database = self.data[user.id]
                msg = message.content
                try:
                    sending = hlist()
                    channel = message.channel
                    for line in msg.split("\n"):
                        found = False
                        if len(line) > 2 and " " in line:
                            i = line.index(" ")
                            prefix = line[:i]
                            if prefix in database:
                                mimics = database[prefix]
                                if mimics:
                                    line = line[i + 1:].strip(" ")
                                    for m in mimics:
                                        sending.append(freeClass(m_id=m, msg=line))
                                    found = True
                        if not sending:
                            break
                        if not found:
                            sending[-1].msg += "\n" + line
                    if sending:
                        create_task(_vars.silentDelete(message))
                        w = await _vars.ensureWebhook(channel)
                        for k in sending:
                            mimic = self.data[k.m_id]
                            await self.updateMimic(mimic, guild=message.guild)
                            name = mimic.name
                            url = mimic.url
                            await w.send(k.msg, username=name, avatar_url=url)
                            mimic.count += 1
                            mimic.total += len(k.msg)
                except Exception as ex:
                    await channel.send(repr(ex))

    async def updateMimic(self, mimic, guild=None, it=None):
        if mimic.setdefault("auto", None):
            _vars = self._vars
            mim = 0
            try:
                mim = verifyID(mimic.auto)
                if guild is not None:
                    user = guild.get_member(mim)
                if user is None:
                    user = await _vars.fetch_user(mim)
                if user is None:
                    raise LookupError
                mimic.name = user.display_name
                mimic.url = str(user.avatar_url)
            except (discord.NotFound, LookupError):
                try:
                    mimi = _vars.get_mimic(mim)
                    if it is None:
                        it = {}
                    elif mim in it:
                        raise RecursionError("Infinite recursive loop detected.")
                    it[mim] = True
                    if not len(it) & 255:
                        await asyncio.sleep(0.2)
                    await self.updateMimic(mimi, guild=guild, it=it)
                    mimic.name = mimi.name
                    mimic.url = mimi.url
                except LookupError:
                    mimic.name = str(mimic.auto)
                    mimic.url = "https://cdn.discordapp.com/embed/avatars/0.png"

    async def __call__(self):
        if self.busy:
            return
        self.busy = True
        try:
            i = 1
            for m_id in tuple(self.data):
                if type(m_id) is str:
                    mimic = self.data[m_id]
                    try:
                        if mimic.u_id not in self.data or mimic.id not in self.data[mimic.u_id][mimic.prefix]:
                            self.data.pop(m_id)
                            self.update()
                    except:
                        self.data.pop(m_id)
                        self.update()
                if not i % 8191:
                    await asyncio.sleep(0.45)
                i += 1
        except:
            print(traceback.format_exc())
        await asyncio.sleep(2)
        self.busy = False


class UpdateDogpiles(Database):
    name = "dogpiles"

    def __init__(self, *args):
        self.msgFollow = {}
        super().__init__(*args)

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None or not orig:
            return
        g_id = message.guild.id
        following = self.data
        if g_id in following:
            u_id = message.author.id
            c_id = message.channel.id
            if not edit:
                if following[g_id]:
                    checker = orig
                    curr = self.msgFollow.get(c_id)
                    if curr is None:
                        curr = [checker, 1, 0]
                        self.msgFollow[c_id] = curr
                    elif checker == curr[0] and u_id != curr[2]:
                        curr[1] += 1
                        if curr[1] >= 3:
                            curr[1] = xrand(-3) + 1
                            if len(checker):
                                create_task(message.channel.send(checker))
                    else:
                        if len(checker) > 100:
                            checker = ""
                        curr[0] = checker
                        curr[1] = xrand(-1, 2)
                    curr[2] = u_id
                    #print(curr)


class UpdateMathTest(Database):
    name = "mathtest"
    no_file = True

    def __init__(self, *args):
        s = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        ss = {str(i): s[i] for i in range(len(s))}
        ss["-"] = "⁻"
        self.sst = "".maketrans(ss)
        super().__init__(*args)

    def format(self, x, y, op):
        length = 6
        xs = str(x)
        xs = " " * (length - len(xs)) + xs
        ys = str(y)
        ys = " " * (length - len(ys)) + ys
        return " " + xs + "\n" + op + ys

    def eqtrans(self, eq):
        return str(eq).replace("**", "^").replace("exp", "e^").replace("*", "∙")
    
    def addition(self):
        x = xrand(100, 10000)
        y = xrand(100, 10000)
        s = self.format(x, y, "+")
        return s, x + y
    
    def subtraction(self):
        x = xrand(100, 12000)
        y = xrand(100, 8000)
        if x < y:
            x, y = y, x
        s = self.format(x, y, "-")
        return s, x - y

    def multiplication(self):
        x = xrand(2, 20)
        y = xrand(2, 20)
        s = self.format(x, y, "×")
        return s, x * y

    def multiplication2(self):
        x = xrand(13, 100)
        y = xrand(13, 100)
        s = self.format(x, y, "×")
        return s, x * y

    def division(self):
        y = xrand(2, 20)
        x = xrand(2, 14) * y
        s = self.format(x, y, "÷")
        return s, x // y

    def exponentiation(self):
        x = xrand(2, 20)
        y = xrand(2, max(3, 14 / x))
        s = str(x) + str(y).translate(self.sst)
        return s, x ** y

    def exponentiation2(self):
        x = xrand(2, 4)
        if x == 2:
            y = xrand(7, 35)
        else:
            y = xrand(5, 11)
        s = str(x) + str(y).translate(self.sst)
        return s, x ** y
        
    def square_root(self):
        x = xrand(2, 20)
        y = x ** 2
        s = "√" + str(y)
        return s, x

    def square_root2(self):
        x = xrand(21, 1000)
        y = x ** 2
        s = "√" + str(y)
        return s, x
        
    def scientific(self):
        x = xrand(100, 10000)
        x /= 10 ** int(math.log10(x))
        y = xrand(-3, 6)
        s = str(x) + "×10" + str(y).translate(self.sst)
        return s, round(x * 10 ** y, 9)
        
    def fraction(self):
        y = random.choice([2, 4, 5, 10])
        x = xrand(3, 20)
        mult = xrand(4) + 1
        y *= mult
        x *= mult
        s = self.format(x, y, "÷")
        return s, round(x / y, 9)
        
    def equation(self):
        a = xrand(1, 10)
        b = xrand(1, 10)
        if xrand(2):
            a = -a
        if xrand(2):
            b = -b
        bx = -a - b
        cx = a * b
        s = "x² "
        if bx:
            s += ("+", "-")[bx < 0] + " " + (str(abs(bx))) * (abs(bx) != 1) +  "x "
        s += ("+", "-")[cx < 0] + " " + str(abs(cx)) + " = 0"
        return s, [a, b]

    async def equation2(self):
        a = xrand(1, 14)
        b = xrand(1, 14)
        c = xrand(1, 14)
        d = xrand(1, 14)
        if xrand(2):
            a = -a
        if xrand(2):
            b = -b
        if xrand(2):
            c = -c
        if xrand(2):
            d = -d
        st = "(" + str(a) + "*x+" + str(b) + ")*(" + str(c) + "*x+" + str(d) + ")"
        a = [-b / a, -d / c]
        returns = [None]
        doParallel(sympy.expand, [st], returns)
        while returns[0] is None:
            await asyncio.sleep(0.2)
        q = self.eqtrans(returns[0]).replace("^2", "²").replace("∙", "") + " = 0"
        return q, a

    async def calculus(self):
        amount = xrand(2, 5)
        s = []
        for i in range(amount):
            t = xrand(3)
            if t == 0:
                a = xrand(1, 7)
                e = xrand(-3, 8)
                if xrand(2):
                    a = -a
                s.append(str(a) + "x^(" + str(e) + ")")
            elif t == 1:
                a = xrand(5)
                if a <= 1:
                    a = "e"
                s.append("+-"[xrand(2)] + str(a) + "^x")
            elif t == 2:
                a = xrand(6)
                if a < 1:
                    a = 1
                if xrand(2):
                    a = -a
                op = ["sin", "cos", "tan", "sec", "csc", "cot", "log"]
                s.append(str(a) + "*" + random.choice(op) + "(x)")
        st = ""
        for i in s:
            if st and i[0] not in "+-":
                st += "+"
            st += i
        ans = await self._vars.solveMath(st, -1, 2, 1)
        a = ans[0]
        q = self.eqtrans(a)
        if xrand(2):
            q = "Dₓ " + q
            op = sympy.diff
        else:
            q = "∫ " + q
            op = sympy.integrate
        returns = [None]
        doParallel(op, [a], returns)
        while returns[0] is None:
            await asyncio.sleep(0.2)
        a = returns[0]
        return q, a

    async def generateMathQuestion(self, mode):
        easy = (
            self.addition,
            self.subtraction,
            self.multiplication,
            self.division,
            self.exponentiation,
            self.square_root,
            self.scientific,
            self.fraction,
            self.equation,
        )
        hard = (
            self.multiplication2,
            self.exponentiation2,
            self.square_root2,
            self.equation2,
            self.calculus,
        )
        modes = {"easy": easy, "hard": hard}
        qa = random.choice(modes[mode])()
        if asyncio.iscoroutine(qa):
            return await qa
        return qa

    async def newQuestion(self, channel):
        q, a = await self.generateMathQuestion(self.data[channel.id].mode)
        msg = "```\n" + q + "```"
        self.data[channel.id].answer = a
        await channel.send(msg)

    async def __call__(self):
        _vars = self._vars
        for c_id in self.data:
            if self.data[c_id].answer is None:
                self.data[c_id].answer = nan
                channel = await _vars.fetch_channel(c_id)
                await self.newQuestion(channel)

    async def _nocommand_(self, message, **void):
        _vars = self._vars
        channel = message.channel
        if channel.id in self.data:
            if message.author.id != _vars.client.user.id:
                msg = message.content.strip("|").strip("`")
                if not msg or msg.lower() != msg:
                    return
                if msg.startswith("#") or msg.startswith("//") or msg.startswith("\\"):
                    return
                try:
                    x = await _vars.solveMath(msg, getattr(channel, "guild", None), 2, 1)
                    returns = [None]
                    doParallel(sympy.sympify, [x[0]], returns)
                    while returns[0] is None:
                        await asyncio.sleep(0.2)
                    x = returns[0]
                except:
                    return
                correct = False
                a = self.data[channel.id].answer
                if type(a) is list:
                    if x in a:
                        correct = True
                else:
                    returns = [None]
                    doParallel(sympy.sympify, [a], returns)
                    while returns[0] is None:
                        await asyncio.sleep(0.2)
                    a = returns[0]
                    returns = [None]
                    doParallel(sympy.simplify, [x - a], returns)
                    while returns[0] is None:
                        await asyncio.sleep(0.2)
                    if returns[0] == 0:
                        correct = True
                if correct:
                    create_task(self.newQuestion(channel))
                    await channel.send("Great work!")
                else:
                    await channel.send("Oops! Not quite, try again!")