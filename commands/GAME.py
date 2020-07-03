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
    numScore = lambda y, x=0: x * 2 ** (x + 1)
    name = ["2048", "Text_2048"]
    min_level = 0
    description = "Plays a game of 2048 using reactions."
    usage = (
        "<board_size[4]>  <show_debug(?z)> <special_tiles(?s)> <public(?p)> "
        + "<insanity_mode(?i)> <special_controls(?c)> <easy_mode(?e)>"
    )
    flags = "zspice"
    rate_limit = (1, 2)

    def shiftTile(self, tiles, p1, p2):
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

    def randomSpam(self, gamestate, mode, pool):
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
        return gamestate, a
                                        
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
                self.gamestate, a = await create_future(self.randomSpam, gamestate, mode, pool)
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
                    try:
                        await message.clear_reactions()
                    except discord.Forbidden:
                        pass
                    gameover = ["🇬","🇦","🇲","🇪","⬛","🇴","🇻","3️⃣","🇷"]
                    for g in gameover:
                        create_task(message.add_reaction(g))
                        await asyncio.sleep(0.5)

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

    async def _callback_(self, bot, message, reaction, argv, user, perm, vals, **void):
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
                    create_task(message.add_reaction(react.decode("utf-8")))
            self.spawn(gamestate[0], mode, 1)
        if u_id == 0:
            username = "＠everyone"
        else:
            if user.id != u_id:
                u = await bot.fetch_user(u_id)
                username = u.name
            else:
                username = user.name
        await self.nextIter(message, gamestate, username, reaction, mode)

    async def __call__(self, bot, argv, user, flags, guild, **void):
        try:
            if not len(argv.replace(" ", "")):
                size = 4
            else:
                ans = await bot.evalMath(argv, guild)
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
        if "z" in flags:
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


class MimicConfig(Command):
    name = ["PluralConfig", "RPConfig"]
    min_level = 0
    description = "Modifies an existing webhook mimic's attributes."
    usage = (
        "<0:mimic_id> <1:option(prefix)(name[])(avatar)"
        + "(description)(gender)(birthday)> <2:new>"
    )
    no_parse = True
    rate_limit = 1
    
    async def __call__(self, bot, user, perm, flags, args, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        m_id = "&" + str(verifyID(args.pop(0)))
        if m_id not in mimicdb:
            raise LookupError("Target mimic ID not found.")
        if not isnan(perm):
            mimics = setDict(mimicdb, user.id, {})
            found = 0
            for prefix in mimics:
                found += mimics[prefix].count(m_id)
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
        if opt in ("name", "username", "nickname", ""):
            setting = "name"
        elif opt in ("avatar", "icon", "url", "pfp", "image"):
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
            new = utc_ts(tzparse(new))
        elif setting == "name":
            if len(new) > 80:
                raise OverflowError("Prefix must be 80 or fewer in length.")
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
            new = await bot.followURL(verifyURL(new), best=True)
        elif setting == "auto":
            if new.lower() in ("none", "null", "0", "false", "f"):
                new = None
            else:
                mim = None
                try:
                    mim = verifyID(new)
                    user = await bot.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    new = user.id
                except:
                    try:
                        mimi = bot.get_mimic(mim, user)
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
    usage = "<0:prefix> <1:user[]> <1:name[]> <2:url[]> <disable(?d)> <debug(?z)>"
    flags = "aedzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1, 2)
    
    async def __call__(self, bot, message, user, perm, flags, args, argv, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        if len(args) == 1 and "d" not in flags:
            user = await bot.fetch_user(verifyID(argv))
        mimics = setDict(mimicdb, user.id, {})
        if not argv or (len(args) == 1 and "d" not in flags):
            if "d" in flags:
                if "f" not in flags:
                    response = uniStr(
                        "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                        + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                    )
                    return ("```asciidoc\n[" + response + "]```")
                mimicdb.pop(user.id)
                update()
                return (
                    "```css\nSuccessfully removed all webhook mimics for ["
                    + noHighlight(user) + "].```"
                )
            return (
                "```" + "\n" * ("z" in flags) + "callback-game-mimic-"
                + str(user.id) + "_0"
                + "-\nLoading Mimic database...```"
            )
        u_id = user.id
        prefix = args.pop(0)
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
                mimic = bot.get_mimic(prefix, user)
                if not isnan(perm) and mimic.u_id != user.id:
                    raise PermissionError("Target mimic does not belong to you.")
                mimics = mimicdb[mimic.u_id]
                user = await bot.fetch_user(mimic.u_id)
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
        if not prefix:
            raise IndexError("Prefix must not be empty.")
        if len(prefix) > 16:
            raise OverflowError("Prefix must be 16 or fewer in length.")
        if " " in prefix:
            raise TypeError("Prefix must not contain spaces.")
        if sum(len(i) for i in iter(mimics.values())) >= 32768:
            raise OverflowError(
                "Mimic list for " + str(user)
                + " has reached the maximum of 32768 items. "
                + "Please remove an item to add another."
            )
        dop = None
        utcn = utc_dt()
        mid = discord.utils.time_snowflake(utcn)
        ctime = utc()
        m_id = "&" + str(mid)
        mimic = None
        if len(args):
            if len(args) > 1:
                url = await bot.followURL(verifyURL(args[-1]), best=True)
                name = " ".join(args[:-1])
            else:
                mim = 0
                try:
                    mim = verifyID(args[-1])
                    user = await bot.fetch_user(mim)
                    if user is None:
                        raise EOFError
                    dop = user.id
                    name = user.name
                    url = bestURL(user)
                except:
                    try:
                        mimi = bot.get_mimic(mim, user)
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
            url = bestURL(user)
        if len(name) > 80:
            raise OverflowError("Prefix must be 80 or fewer in length.")
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

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm <= inf:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        user = await bot.fetch_user(u_id)
        mimics = mimicdb.get(user.id, {})
        for k in tuple(mimics):
            if not mimics[k]:
                mimics.pop(k)
                update()
        page = 24
        last = max(0, len(mimics) - page)
        if reaction is not None:
            i = self.directions.index(reaction)
            if i == 0:
                new = 0
            elif i == 1:
                new = max(0, pos - page)
            elif i == 2:
                new = min(last, pos + page)
            elif i == 3:
                new = last
            else:
                new = pos
            pos = new
        content = message.content
        if not content:
            content = message.embeds[0].description
        i = content.index("callback")
        content = content[:i] + (
            "callback-game-mimic-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not mimics:
            content += "No currently enabled webhook mimics for " + str(user).replace("`", "") + ".```"
            msg = ""
        else:
            content += str(len(mimics)) + " currently enabled webhook mimics for " + str(user).replace("`", "") + ":```"
            key = lambda x: limStr("⟨" + ", ".join(i + ": " + (str(noHighlight(mimicdb[i].name)), "[<@" + str(getattr(mimicdb[i], "auto", "None")) + ">]")[bool(getattr(mimicdb[i], "auto", None))] for i in iter(x)) + "⟩", 1900 / len(mimics))
            msg = "```ini\n" + strIter({k: mimics[k] for k in sorted(mimics)[pos:pos + page]}, key=key) + "```"
        emb = discord.Embed(
            description=content + msg,
            colour=randColour(),
        )
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        more = len(mimics) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class MimicSend(Command):
    name = ["RPSend", "PluralSend"]
    min_level = 0
    description = "Sends a message using a webhook mimic, to the target channel."
    usage = "<0:mimic> <1:channel> <2:string>"
    no_parse = True
    rate_limit = 0.5

    async def __call__(self, bot, channel, user, perm, args, **void):
        update = self.data.mimics.update
        mimicdb = bot.data.mimics
        mimics = setDict(mimicdb, user.id, {})
        prefix = args.pop(0)
        c_id = verifyID(args.pop(0))
        channel = await bot.fetch_channel(c_id)
        guild = channel.guild
        w = await bot.ensureWebhook(channel)
        msg = " ".join(args)
        if not msg:
            raise IndexError("Message is empty.")
        perm = bot.getPerms(user.id, guild)
        try:
            mlist = mimics[prefix]
            if mlist is None:
                raise KeyError
            m = [bot.get_mimic(verifyID(p)) for p in mlist]
        except KeyError:
            mimic = bot.get_mimic(verifyID(prefix))
            if not isnan(perm) and mimic.u_id != user.id:
                raise PermissionError("Target mimic does not belong to you.")
            m = [mimic]
        admin = not inf > perm
        try:
            enabled = bot.data.enabled[channel.id]
        except KeyError:
            enabled = ()
        if not admin and "game" not in enabled:
            raise PermissionError("Not permitted to send into target channel.")
        msg = escape_everyone(msg)
        for mimic in m:
            await bot.database.mimics.updateMimic(mimic, guild)
            name = mimic.name
            url = mimic.url
            try:
                await waitOnNone(w.send(msg, username=name, avatar_url=url))
            except (discord.NotFound, discord.InvalidArgument, discord.Forbidden):
                w = await bot.ensureWebhook(channel, force=True)
                await waitOnNone(w.send(msg, username=name, avatar_url=url))
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
            bot = self.bot
            perm = bot.getPerms(user.id, message.guild)
            admin = not inf > perm
            if message.guild is not None:
                try:
                    enabled = bot.data.enabled[message.channel.id]
                except KeyError:
                    enabled = ()
            else:
                enabled = list(bot.categories)
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
                        create_task(bot.silentDelete(message))
                        w = await bot.ensureWebhook(channel)
                        for k in sending:
                            mimic = self.data[k.m_id]
                            await self.updateMimic(mimic, guild=message.guild)
                            name = mimic.name
                            url = mimic.url
                            msg = escape_everyone(k.msg)
                            try:
                                await waitOnNone(w.send(msg, username=name, avatar_url=url))
                            except (discord.NotFound, discord.InvalidArgument, discord.Forbidden):
                                w = await bot.ensureWebhook(channel, force=True)
                                await waitOnNone(w.send(msg, username=name, avatar_url=url))
                            mimic.count += 1
                            mimic.total += len(k.msg)
                except Exception as ex:
                    await sendReact(channel, "```py\n" + repr(ex) + "```", reacts="❎")

    async def updateMimic(self, mimic, guild=None, it=None):
        if setDict(mimic, "auto", None):
            bot = self.bot
            mim = 0
            try:
                mim = verifyID(mimic.auto)
                if guild is not None:
                    user = guild.get_member(mim)
                if user is None:
                    user = await bot.fetch_user(mim)
                if user is None:
                    raise LookupError
                mimic.name = user.display_name
                mimic.url = bestURL(user)
            except (discord.NotFound, LookupError):
                try:
                    mimi = bot.get_mimic(mim)
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