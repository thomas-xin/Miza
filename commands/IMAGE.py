import discord
from smath import *


class IMG:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
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
                raise PermissionError(
                    "Insufficient priviliges to change image list for " + uniStr(guild.name)
                    + ".\nRequred level: " + uniStr(req)
                    + ", Current level: " + uniStr(perm) + "."
                )
            if "e" in flags:
                key = args[0].lower()
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
            return (
                "Available images in **" + guild.name
                + "**: ```ini\n" + str(list(images)).replace("'", '"') + "```"
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
            raise EOFError("Target image not found. Use ~img for list.")
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


class OwOify:
    is_command = True

    omap = {
        "n": "ny",
        "N": "NY",
        "r": "w",
        "R": "W",
        "l": "w",
        "L": "W",
    }
    otrans = "".maketrans(omap)

    def __init__(self):
        self.name = ["OwO"]
        self.min_level = 0
        self.description = "owo-ifies text."
        self.usage = "<string>"

    async def __call__(self, argv, **void):
        if not argv:
            raise IndexError("Input string is empty.")
        return "```fix\n" + argv.translate(self.otrans) + "```"


class updateImages:
    is_update = True
    name = "images"

    def __init__(self):
        pass

    async def __call__(self):
        pass
