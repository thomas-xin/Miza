import discord


class how:
    is_command = True

    def __init__(self):
        self.name = ["how.gif"]
        self.minm = 0
        self.desc = "Sends HOW.gif in the current chat."
        self.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        url = "https://media.discordapp.net/attachments/500919580596764673/642515924578205696/HOW.gif"
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)

        
class ded:
    is_command = True

    def __init__(self):
        self.name = ["ded.gif","4aa217e","4aa217e.gif"]
        self.minm = 0
        self.desc = "Sends 4aa217e.gif in the current chat."
        self.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        url = "https://cdn.discordapp.com/attachments/430005329984487434/654326757654134785/4aa217e.gif"
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)


class parrot:
    is_command = True

    def __init__(self):
        self.name = ["parrot.gif"]
        self.minm = 0
        self.desc = "Sends parrot.gif in the current chat."
        self.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        url = "https://media.discordapp.net/attachments/543449246221860885/651594997925281803/parrot.gif"
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)


class dolphino:
    is_command = True

    def __init__(self):
        self.name = ["dolphino.gif","dolphin.gif","dolphin"]
        self.minm = 0
        self.desc = "Sends dolphino.gif in the current chat."
        self.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        url = "https://cdn.discordapp.com/attachments/320915703102177293/671169395019612170/dolphino.gif"
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)


class curiouseal:
    is_command = True

    def __init__(self):
        self.name = ["curiouseal.gif","seal.gif","seal"]
        self.minm = 0
        self.desc = "Sends curiouseal.gif in the current chat."
        self.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        url = "https://cdn.discordapp.com/attachments/320915703102177293/671167411914801182/curiouseal.gif"
        if "v" in flags:
            return url
        emb = discord.Embed(url=url)
        emb.set_image(url=url)
        print(url)
        await channel.send(embed=emb)
