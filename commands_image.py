import discord
from smath import *


images = {
    "https://media.discordapp.net/attachments/500919580596764673/642515924578205696/HOW.gif": ["how"],
    "https://cdn.discordapp.com/attachments/430005329984487434/654326757654134785/4aa217e.gif": ["ded", "4aa217e"],
    "https://media.discordapp.net/attachments/543449246221860885/651594997925281803/parrot.gif": ["parrot"],
    "https://cdn.discordapp.com/attachments/313292557603962881/671873663120703508/dolphino.gif": ["dolphin"],
    "https://cdn.discordapp.com/attachments/313292557603962881/671873485722746890/curiouseal.gif": ["seal"],
    "https://cdn.discordapp.com/attachments/292453316099702786/673082811464286228/HW.gif": ["hw"],
    "https://media.discordapp.net/attachments/528720242336333837/663588773283627018/WTF.gif": ["wtf"],
    "https://cdn.discordapp.com/attachments/317898572458754049/674495774591156252/ok.gif": ["ok"],
    "https://cdn.discordapp.com/attachments/317898572458754049/674497566188109824/hug.gif": ["hug", "cuddle"],
    "https://cdn.discordapp.com/attachments/317898572458754049/674497584160833536/munch.gif": ["munch", "moonch"],
    "https://cdn.discordapp.com/attachments/317898572458754049/674497598949818378/no.gif": ["no"],
    "https://cdn.discordapp.com/attachments/664992327701495828/674517702697680896/cave_story_modding.png": ["modding"],
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
            return "Available images: ```css\n" + str([images[i][0] for i in images]) + "```"
        check = argv.lower()
        sources = []
        for url in images:
            for alias in images[url]:
                counter = len(check.split(alias)) - 1 + ("r" in flags)
                sources += [url for i in range(counter)]
        if not len(sources):
            raise EOFError("Target image not found. Use ?l for list.")
        v = xrand(len(sources))
        url = sources[v]
        if "v" in flags:
            return url
        emb = discord.Embed(
            url=url,
            colour=colour2Raw(colourCalculation(xrand(1536))),
            )
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)
