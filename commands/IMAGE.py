import discord, nekos, requests
from smath import *


class IMG:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.min_display = "0~2"
        self.description = "Sends an image in the current chat from a list."
        self.usage = "<tags[]> <url[]> <verbose(?v)> <random(?r)> <enable(?e)> <disable(?d)> <hide(?h)>"

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
                return (
                    "Available images in **" + guild.name
                    + "**: ```ini\n" + strIter(images).replace("'", '"') + "```"
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
        for i in range(r):
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

    async def __call__(self, channel, flags, **void):
        if not self.header or random.random() > 0.9375:
            url = nekos.cat()
        else:
            for attempts in range(8):
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
        asyncio.create_task(channel.send(embed=emb))


class Dog:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Pulls a random image from images.dog.ceo and embeds it."
        self.usage = "<verbose(?v)>"

    async def __call__(self, channel, flags, **void):
        for attempts in range(8):
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
        url = url.replace("\\", "/").replace("///", "//")
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=self._vars.randColour(),
        )
        emb.set_image(url=url)
        print(url)
        asyncio.create_task(channel.send(embed=emb))


class updateImages:
    is_update = True
    name = "images"

    def __init__(self):
        pass

    async def __call__(self):
        pass
