import discord
from smath import *


images = {
    "https://media.discordapp.net/attachments/500919580596764673/642515924578205696/HOW.gif": ["how"],
    "https://cdn.discordapp.com/attachments/430005329984487434/654326757654134785/4aa217e.gif": ["ded", "4aa217e"],
    "https://media.discordapp.net/attachments/543449246221860885/651594997925281803/parrot.gif": ["parrot"],
    "https://cdn.discordapp.com/attachments/320915703102177293/671169395019612170/dolphino.gif": ["dolphin"],
    "https://cdn.discordapp.com/attachments/320915703102177293/671167411914801182/curiouseal.gif": ["seal"],
    }


class img:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Sends an image in the current chat from a list."
        self.usage = "<target> <verbose:(?v)> <list:(?l)>"

    async def __call__(self, flags, channel, argv, **void):
        if "l" in flags:
            return "Available images: ```\n" + str([images[i][0] for i in images]) + "```"
        check = argv.lower()
        sources = []
        for url in images:
            for alias in images[url]:
                counter = check.split(alias)
                sources += [url for i in range(len(counter) - 1)]
        if not len(sources):
            raise EOFError("Target image not found. Use ?l for list.")
        v = xrand(len(sources))
        url = sources[v]
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)
