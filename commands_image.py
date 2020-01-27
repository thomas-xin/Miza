import discord
from smath import *

class image_sender:
    is_command = true

    def __init__(self, name, minm, url, desc):
        self.name = name
        self.minm = minm
        self.url = url
        self.desc = desc
        seld.usag = "<verbose:(?v)>"

    async def __call__(self, flags, channel, **void):
        if "v" in flags:
            return self.url
        emb = discord.Embed(url=self.url)
        emb.set_image(url=self.url)
        print(self.url)
        await channel.send(embed=emb)

class how(image_sender):
    def __init__(self):
        image_sender.__init__(self,
                              ["how.gif"],
                              0,
                              "https://media.discordapp.net/attachments/500919580596764673/642515924578205696/HOW.gif",
                              "Sends HOW.gif in the current chat."
                              )


        
class ded(image_sender):
    def __init__(self):
        image_sender.__init__(self,
                              ["ded.gif","4aa217e","4aa217e.gif"],
                              0,
                              "https://cdn.discordapp.com/attachments/430005329984487434/654326757654134785/4aa217e.gif",
                              "Sends 4aa217e.gif in the current chat."
                              )


class parrot(image_sender):
    def __init__(self):
        image_sender.__init__(self,
                              ["parrot.gif"],
                              0,
                              "https://media.discordapp.net/attachments/543449246221860885/651594997925281803/parrot.gif",
                              "Sends parrot.gif in the current chat."
                              )


class dolphino(image_sender):
    def __init__(self):
        image_sender.__init__(self,
                              ["dolphino.gif","dolphin.gif","dolphin"],
                              0,
                              "https://cdn.discordapp.com/attachments/320915703102177293/671169395019612170/dolphino.gif",
                              "Sends dolphino.gif in the current chat."
                              )


class curiouseal(image_sender):
    def __init__(self):
        image_sender.__init__(self,
                              ["curiouseal.gif","seal.gif","seal"],
                              0,
                              "https://cdn.discordapp.com/attachments/320915703102177293/671167411914801182/curiouseal.gif",
                              "Sends curiouseal.gif in the current chat."
                              )
