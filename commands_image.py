import discord
from smath import *


class img:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Sends an image in the current chat from a list."
        self.usage = "<tags> <url[]> <verbose(?v)> <random(?r)> <enable(?e)> <disable(?d)>"

    async def __call__(self, flags, args, argv, guild, perm, _vars, **void):
        images = _vars.imglists.get(guild.id, {})
        if "e" in flags or "d" in flags:
            if perm < 3:
                raise PermissionError(
                    "Insufficient priviliges to change image list for " + uniStr(guild.name)
                    + ".\nRequred level: " + uniStr(3)
                    + ", Current level: " + uniStr(perm) + "."
                    )
            if "e" in flags:
                key = args[0].lower()
                url = _vars.verifyURL(args[1])
                images[key] = url
                _vars.imglists[guild.id] = images
                _vars.update()
                return (
                    "```css\nSuccessfully added " + uniStr(key)
                    + " to the image list for " + uniStr(guild.name) + ".```"
                    )
            if not args:
                _vars.imglists[guild.id] = {}
                _vars.update()
                return (
                    "```css\nSuccessfully removed all images from the image list for "
                    + uniStr(guild.name) + ".```"
                    )
            key = args[0].lower()
            images.pop(key)
            _vars.imglists[guild.id] = images
            _vars.update()
            return (
                "```css\nSuccessfully removed " + uniStr(key)
                + " from the image list for " + uniStr(guild.name) + ".```"
                )
        if not argv and not "r" in flags:
            return "Available images in **" + guild.name + "**: ```css\n" + str(list(images)) + "```"
        sources = []
        for tag in args:
            t = tag.lower()
            if t in images:
                sources.append(images[t])
        r = flags.get("r", 0)
        for i in range(r):
            sources.append(images[tuple(images)[xrand(len(images))]])
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
        return {
            "embed": emb
            }
