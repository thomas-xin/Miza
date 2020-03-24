import discord, nekos, requests
try:
    from smath import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from smath import *


class IMG:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.min_display = "0~2"
        self.description = "Sends an image in the current chat from a list."
        self.usage = "<tags[]> <url[]> <verbose(?v)> <random(?r)> <enable(?e)> <disable(?d)> <hide(?h)>"
        self.flags = "vredh"

    async def __call__(self, flags, args, argv, guild, perm, **void):
        update = self.data["images"].update
        _vars = self._vars
        imglists = _vars.data["images"]
        images = imglists.get(guild.id, {})
        if "e" in flags or "d" in flags:
            req = 2
            if perm < req:
                reason = "to change image list for " + uniStr(guild.name)
                self.permError(perm, req, reason)
            if "e" in flags:
                if len(images) > 64:
                    raise OverflowError(
                        "Image list for " + uniStr(guild.name)
                        + " has reached the maximum of 64 items. "
                        + "Please remove an item to add another."
                    )
                key = args[0].lower()
                if len(key) > 64:
                    raise OverflowError("Image tag too long.")
                url = verifyURL(args[1])
                images[key] = url
                sort(images)
                imglists[guild.id] = images
                update()
                if not "h" in flags:
                    return (
                        "```css\nSuccessfully added " + uniStr(key)
                        + " to the image list for " + uniStr(guild.name) + ".```"
                    )
            if not args:
                imglists[guild.id] = {}
                update()
                return (
                    "```css\nSuccessfully removed all images from the image list for "
                    + uniStr(guild.name) + ".```"
                )
            key = args[0].lower()
            images.pop(key)
            imglists[guild.id] = images
            update()
            return (
                "```css\nSuccessfully removed " + uniStr(key)
                + " from the image list for " + uniStr(guild.name) + ".```"
            )
        if not argv and not "r" in flags:
            if images:
                if "v" in flags:
                    key = lambda x: x
                else:
                    key = lambda x: limStr(x, 32)
                return (
                    "Available images in **" + guild.name
                    + "**: ```ini\n" + strIter(images, key=key).replace("'", '"') + "```"
                )
            return (
                "```css\nImage list for " + uniStr(guild.name)
                + " is currently empty.```"
            )
        sources = []
        for tag in args:
            t = tag.lower()
            if t in images:
                sources.append(images[t])
        r = flags.get("r", 0)
        for _ in loop(r):
            sources.append(images[tuple(images)[xrand(len(images))]])
        if not len(sources):
            raise LookupError("Target image " + str(argv) + " not found. Use img for list.")
        v = xrand(len(sources))
        url = sources[v]
        if "v" in flags:
            return url
        emb = discord.Embed(
            url=url,
            colour=_vars.randColour(),
        )
        emb.set_image(url=url)
        return {
            "embed": emb
        }


def _c2e(string, em1, em2):
    chars = {
        " ": [0, 0, 0, 0, 0],
        "_": [0, 0, 0, 0, 7],
        "!": [2, 2, 2, 0, 2],
        '"': [5, 5, 0, 0, 0],
        ":": [0, 2, 0, 2, 0],
        ";": [0, 2, 0, 2, 4],
        "~": [0, 5, 7, 2, 0],
        "#": [10, 31, 10, 31, 10],
        "$": [7, 10, 6, 5, 14],
        "?": [6, 1, 2, 0, 2],
        "%": [5, 1, 2, 4, 5],
        "&": [4, 10, 4, 10, 7],
        "'": [2, 2, 0, 0, 0],
        "(": [2, 4, 4, 4, 2],
        ")": [2, 1, 1, 1, 2],
        "[": [6, 4, 4, 4, 6],
        "]": [3, 1, 1, 1, 3],
        "|": [2, 2, 2, 2, 2],
        "*": [21, 14, 4, 14, 21],
        "+": [0, 2, 7, 2, 0],
        "=": [0, 7, 0, 7, 0],
        ",": [0, 0, 3, 3, 4],
        "-": [0, 0, 7, 0, 0],
        ".": [0, 0, 3, 3, 0],
        "/": [1, 1, 2, 4, 4],
        "\\": [4, 4, 2, 1, 1],
        "@": [14, 17, 17, 17, 14],
        "0": [7, 5, 5, 5, 7],
        "1": [3, 1, 1, 1, 1],
        "2": [7, 1, 7, 4, 7],
        "3": [7, 1, 7, 1, 7],
        "4": [5, 5, 7, 1, 1],
        "5": [7, 4, 7, 1, 7],
        "6": [7, 4, 7, 5, 7],
        "7": [7, 5, 1, 1, 1],
        "8": [7, 5, 7, 5, 7],
        "9": [7, 5, 7, 1, 7],
        "A": [2, 5, 7, 5, 5],
        "B": [6, 5, 7, 5, 6],
        "C": [3, 4, 4, 4, 3],
        "D": [6, 5, 5, 5, 6],
        "E": [7, 4, 7, 4, 7],
        "F": [7, 4, 7, 4, 4],
        "G": [7, 4, 5, 5, 7],
        "H": [5, 5, 7, 5, 5],
        "I": [7, 2, 2, 2, 7],
        "J": [7, 1, 1, 5, 7],
        "K": [5, 5, 6, 5, 5],
        "L": [4, 4, 4, 4, 7],
        "M": [17, 27, 21, 17, 17],
        "N": [9, 13, 15, 11, 9],
        "O": [2, 5, 5, 5, 2],
        "P": [7, 5, 7, 4, 4],
        "Q": [4, 10, 10, 10, 5],
        "R": [6, 5, 7, 6, 5],
        "S": [3, 4, 7, 1, 6],
        "T": [7, 2, 2, 2, 2],
        "U": [5, 5, 5, 5, 7],
        "V": [5, 5, 5, 5, 2],
        "W": [17, 17, 21, 21, 10],
        "X": [5, 5, 2, 5, 5],
        "Y": [5, 5, 2, 2, 2],
        "Z": [7, 1, 2, 4, 7],
    }
    printed = [""] * 7
    string = string.upper()
    for i in range(len(string)):
        curr = string[i]
        data = chars.get(curr, [15] * 5)
        size = max(1, max(data))
        lim = max(2, int(log(size, 2))) + 1
        printed[0] += em2 * (lim + 1)
        printed[6] += em2 * (lim + 1)
        if len(data) == 5:
            for y in range(5):
                printed[y + 1] += em2
                for p in range(lim):
                    if data[y] & (1 << (lim - 1 - p)):
                        printed[y + 1] += em1
                    else:
                        printed[y + 1] += em2
        for x in range(len(printed)):
            printed[x] += em2
    output = "\n".join(printed)
    print("[" + em1 + "]", "[" + em2 + "]")
    if len(em1) + len(em2) > 2 and ":" in em1 + em2:
        return output
    return "```fix\n" + output + "```"


class Char2Emoj:
    is_command = True

    def __init__(self):
        self.name = ["C2E"]
        self.min_level = 0
        self.description = "Makes emoji blocks using a string."
        self.usage = "<0:string> <1:emoji_1> <2:emoji_2>"

    async def __call__(self, args, **extra):
        try:
            if len(args) != 3:
                raise IndexError
            for i in range(1,3):
                if args[i][0] == ":" and args[i][-1] != ":":
                    args[i] = "<" + args[i] + ">"
            return _c2e(*args[:3])
        except IndexError:
            raise IndexError(
                "Exactly 3 arguments are required for this command.\n"
                + "Place <> around arguments containing spaces as required."
            )


f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    cat_key = auth["cat_api_key"]
except:
    cat_key = None
    print("WARNING: cat_api_key not found. Unable to use API to pull cat images.")


class Cat:
    is_command = True
    if cat_key:
        header = {"x-api-key": cat_key}
    else:
        header = None

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Pulls a random image from thecatapi.com or cdn.nekos.life/meow, and embeds it."
        self.usage = "<verbose(?v)>"
        self.flags = "v"

    async def __call__(self, channel, flags, **void):
        if not self.header or random.random() > 0.9375:
            url = nekos.cat()
        else:
            for _ in loop(8):
                returns = [None]
                doParallel(
                    funcSafe,
                    [requests.get, "https://api.thecatapi.com/v1/images/search"],
                    returns,
                    {"headers": self.header},
                )
                while returns[0] is None:
                    await asyncio.sleep(0.5)
                if type(returns[0]) is str:
                    print(eval, returns[0])
                    raise eval(returns[0])
                resp = returns[0][-1]
                try:
                    d = json.loads(resp.content)
                except:
                    d = eval(resp.content, {}, {})
                try:
                    if type(d) is list:
                        d = random.choice(d)
                    url = d["url"]
                    break
                except KeyError:
                    await asyncio.sleep(0.5)
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=self._vars.randColour(),
        )
        emb.set_image(url=url)
        print(url)
        create_task(channel.send(embed=emb))


class Dog:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Pulls a random image from images.dog.ceo and embeds it."
        self.usage = "<verbose(?v)>"
        self.flags = "v"

    async def __call__(self, channel, flags, **void):
        for _ in loop(8):
            returns = [None]
            doParallel(funcSafe, [urlOpen, "https://dog.ceo/api/breeds/image/random"], returns)
            while returns[0] is None:
                await asyncio.sleep(0.5)
            if type(returns[0]) is str:
                raise eval(returns[0])
            resp = returns[0][-1]
            s = resp.read()
            resp.close()
            try:
                d = json.loads(s)
            except:
                d = eval(s, {}, {})
            try:
                if type(d) is list:
                    d = random.choice(d)
                url = d["message"]
                break
            except KeyError:
                await asyncio.sleep(0.5)
        url = url.replace("\\", "/")
        while "///" in url:
            url = url.replace("///", "//")
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=self._vars.randColour(),
        )
        emb.set_image(url=url)
        print(url)
        create_task(channel.send(embed=emb))


class MimicConfig:
    is_command = True

    def __init__(self):
        self.name = ["PluralConfig"]
        self.min_level = 0
        self.description = "Modifies an existing webhook mimic's attributes."
        self.usage = "<0:mimic_id> <1:option(prefix)([name][username][nickname])([avatar][icon][url])([status][description])(gender)(birthday)> <2:new>"
    
    async def __call__(self, _vars, user, perm, flags, args, **void):
        mimicdb = _vars.data["mimics"]
        mimics = mimicdb.setdefault(user.id, {})
        update = _vars.database["mimics"].update
        m_id = "&" + str(_vars.verifyID(args.pop(0)))
        if m_id not in mimicdb:
            raise LookupError("Target mimic ID not found.")
        found = False
        for prefix in mimics:
            for mid in mimics[prefix]:
                if mid == m_id:
                    found = True
        if not found and perm is not nan:
            raise PermissionError("Target mimic does not belong to you.")
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
            if not found:
                raise PermissionError("Target mimic does not belong to you.")
            for prefix in mimics:
                for mid in mimics[prefix]:
                    if mid == m_id:
                        mimics[prefix].remove(m_id)
            if new in mimics:
                mimics[new].append(m_id)
            else:
                mimics[new] = hlist([m_id])
        mimic[setting] = new
        update()
        return (
            "```css\nChanged " + setting + " for " 
            + uniStr(mimic.name) + " to " + str(new) + ".```"
        )


class Mimic:
    is_command = True

    def __init__(self):
        self.name = ["RolePlay", "Plural"]
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
        prefix = args[0]
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
        args.pop(0)
        if len(args):
            if len(args) > 1:
                url = args[-1]
                name = " ".join(args[:-1])
            else:
                try:
                    user = await _vars.fetch_user(_vars.verifyID(argv))
                    if user is None:
                        raise EOFError
                    name = user.name
                    url = str(user.avatar_url)
                except:
                    name = args[0]
                    url = "https://cdn.discordapp.com/embed/avatars/0.png"
        else:
            name = user.name
            url = str(user.avatar_url)
        mid = discord.utils.time_snowflake(ctime)
        m_id = "&" + str(mid)
        while m_id in mimics:
            mid += 1
            m_id = "&" + str(mid)
        mimic = freeClass(
            id=m_id,
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
            database = self.data[user.id]
            msg = message.content
            found = False
            for line in msg.split("\n"):
                if len(line) > 2 and " " in line:
                    i = line.index(" ")
                    prefix = line[:i]
                    line = line[i + 1:].strip(" ")
                    if prefix in database:
                        mimics = database[prefix]
                        if mimics:
                            channel = message.channel
                            try:
                                _vars = self._vars
                                returns = _vars.returns()
                                create_task(_vars.parasync(_vars.ensureWebhook(channel), returns))
                                _vars.logDelete(message.id)
                                if not found:
                                    await _vars.silentDelete(message)
                                    found = True
                                while not returns.data:
                                    await asyncio.sleep(0.3)
                                r = returns.data
                                if type(r) is str:
                                    raise eval(r)
                                w = r.data
                                for m in mimics:
                                    mimic = self.data[m]
                                    await w.send(line, username=mimic.name, avatar_url=mimic.url)
                                    mimic.count += 1
                                    mimic.total += len(line)
                            except Exception as ex:
                                await channel.send(repr(ex))
                elif not found:
                    break

    async def __call__(self):
        pass


class updateImages:
    is_database = True
    name = "images"

    def __init__(self):
        pass

    async def __call__(self):
        pass
