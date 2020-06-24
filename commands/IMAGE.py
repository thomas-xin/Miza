try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import youtube_dl, nekos

getattr(youtube_dl, "__builtins__", {})["print"] = print

ydl_opts = {
    "quiet": 1,
    "format": "bestvideo/best",
    "nocheckcertificate": 1,
    "no_call_home": 1,
    "nooverwrites": 1,
    "noplaylist": 1,
    "logtostderr": 0,
    "ignoreerrors": 0,
    "default_search": "auto",
    "source_address": "0.0.0.0",
}
downloader = youtube_dl.YoutubeDL(ydl_opts)

def get_video(url, fps):
    entry = downloader.extract_info(url, download=False)
    best = 0
    size = None
    dur = None
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ""
    for fmt in fmts:
        q = fmt.get("width", 0)
        if type(q) is not int:
            q = 0
        if abs(q - 512) < abs(best - 512):
            best = q
            url = fmt["url"]
            size = [fmt["width"], fmt["height"]]
            dur = fmt.get("duration", entry.get("duration"))
            fps = fmt.get("fps", entry.get("fps"))
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
    return url, size, dur, fps


class IMG(Command):
    min_level = 0
    min_display = "0~2"
    description = "Sends an image in the current chat from a list."
    usage = "<tags[]> <url[]> <verbose(?v)> <random(?r)> <add(?a)> <delete(?d)> <hide(?h)> <debug(?z)>"
    flags = "vraedhzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']

    async def __call__(self, bot, flags, args, argv, user, guild, perm, **void):
        update = self.data.images.update
        imglists = bot.data.images
        images = imglists.get(guild.id, {})
        if "a" in flags or "e" in flags or "d" in flags:
            req = 2
            if perm < req:
                reason = "to change image list for " + guild.name
                raise self.permError(perm, req, reason)
            if "a" in flags or "e" in flags:
                lim = 64 << bot.isTrusted(guild.id) * 2 + 1
                if len(images) > lim:
                    raise OverflowError(
                        "Image list for " + guild.name
                        + " has reached the maximum of " + str(lim) + " items. "
                        + "Please remove an item to add another."
                    )
                key = " ".join(args[:-1]).lower()
                if len(key) > 128:
                    raise ArgumentError("Image tag too long.")
                url = await bot.followURL(verifyURL(args[-1]), best=True)
                if len(url) > 1024:
                    raise ArgumentError("Image url too long.")
                images[key] = url
                images = {i: images[i] for i in sorted(images)}
                imglists[guild.id] = images
                update()
                if not "h" in flags:
                    return (
                        "```css\nSuccessfully added [" + noHighlight(key)
                        + "] to the image list for [" + noHighlight(guild.name) + "].```"
                    )
            if not args:
                if "f" not in flags:
                    response = uniStr(
                        "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                        + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                    )
                    return ("```asciidoc\n[" + response + "]```")
                imglists[guild.id] = {}
                update()
                return (
                    "```css\nSuccessfully removed all images from the image list for ["
                    + noHighlight(guild.name) + "].```"
                )
            key = argv.lower()
            images.pop(key)
            imglists[guild.id] = images
            update()
            return (
                "```css\nSuccessfully removed [" + noHighlight(key)
                + "] from the image list for [" + noHighlight(guild.name) + "].```"
            )
        if not argv and not "r" in flags:
            return (
                "```" + "\n" * ("z" in flags) + "callback-image-img-"
                + str(user.id) + "_0"
                + "-\nLoading Image database...```"
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
            raise LookupError("Target image " + argv + " not found. Use img for list.")
        v = xrand(len(sources))
        url = sources[v]
        if "v" in flags:
            return url
        emb = discord.Embed(
            url=url,
            colour=randColour(),
        )
        emb.set_image(url=url)
        return {
            "embed": emb
        }

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        imglists = bot.data.images
        images = imglists.get(guild.id, {})
        page = 12
        last = max(0, len(images) - page)
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
            "callback-image-img-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not images:
            content += "Image list for " + str(guild).replace("`", "") + " is currently empty.```"
            msg = ""
        else:
            content += str(len(images)) + " images currently assigned for " + str(guild).replace("`", "") + ":```"
            msg = "```ini\n" + strIter({k: "\n" + images[k] for k in tuple(images)[pos:pos + page]}) + "```"
        emb = discord.Embed(
            description=content + msg,
            colour=randColour(),
        )
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        more = len(images) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class React(Command):
    server_only = True
    name = ["AutoReact"]
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically assign a reaction to messages containing the substring."
    usage = "<0:react_to[]> <1:react_data[]> <disable(?d)> <debug(?z)>"
    flags = "aedzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1, 2)

    async def __call__(self, bot, flags, guild, message, user, argv, args, **void):
        update = self.data.reacts.update
        following = bot.data.reacts
        curr = setDict(following, guild.id, multiDict())
        if type(curr) is not multiDict:
            following[guild.id] = curr = multiDict(curr)
        if not argv:
            if "d" in flags:
                if "f" not in flags:
                    response = uniStr(
                        "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                        + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                    )
                    return ("```asciidoc\n[" + response + "]```")
                if guild.id in following:
                    following.pop(guild.id)
                    update()
                return "```css\nRemoved all auto reacts for [" + noHighlight(guild.name) + "].```"
            return (
                "```" + "\n" * ("z" in flags) + "callback-image-react-"
                + str(user.id) + "_0"
                + "-\nLoading React database...```"
            )
        if "d" in flags:
            a = reconstitute(argv).lower()
            if a in curr:
                curr.pop(a)
                update()
                return (
                    "```css\nRemoved [" + noHighlight(a) + "] from the auto react list for ["
                    + noHighlight(guild.name) + "].```"
                )
            else:
                raise LookupError(str(a) + " is not in the auto react list.")
        lim = 64 << bot.isTrusted(guild.id) * 2 + 1
        if curr.count() >= lim:
            raise OverflowError(
                "React list for " + guild.name
                + " has reached the maximum of " + str(lim) + " items. "
                + "Please remove an item to add another."
            )
        a = reconstitute(" ".join(args[:-1])).lower()[:64]
        try:
            e_id = int(args[-1])
        except:
            emoji = args[-1]
        else:
            emoji = await bot.fetch_emoji(e_id)
        await message.add_reaction(emoji)
        curr.append(a, str(emoji))
        following[guild.id] = multiDict({i: curr[i] for i in sorted(curr)})
        update()
        return (
            "```css\nAdded [" + noHighlight(a) + "] ➡️ [" + noHighlight(args[-1]) + "] to the auto react list for ["
            + noHighlight(guild.name) + "].```"
        )
    
    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        following = bot.data.reacts
        curr = following.get(guild.id, multiDict())
        page = 16
        last = max(0, len(curr) - page)
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
            "callback-image-react-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += "No currently assigned auto reactions for " + str(guild).replace("`", "") + ".```"
            msg = ""
        else:
            content += str(len(curr)) + " auto reactions currently assigned for " + str(guild).replace("`", "") + ":```"
            key = lambda x: "\n" + ", ".join(x)
            msg = "```ini\n" + strIter({k: curr[k] for k in tuple(curr)[pos:pos + page]}, key=key) + "```"
        emb = discord.Embed(
            description=content + msg,
            colour=randColour(),
        )
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class CreateEmoji(Command):
    server_only = True
    name = ["EmojiCreate", "EmojiCopy", "CopyEmoji", "Emoji"]
    min_level = 2
    description = "Creates a custom emoji from a URL or attached file."
    usage = "<1:name> <0:url{attached_file}>"
    flags = "ae"
    no_parse = True
    rate_limit = (3, 4)

    async def __call__(self, bot, guild, message, args, argv, **void):
        if message.attachments:
            args = [bestURL(a) for a in message.attachments] + args
            argv = " ".join(bestURL(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            raise ArgumentError("Please enter URL, emoji, or attached file to add.")
        url = args.pop(-1)
        url = await bot.followURL(url, best=True)
        if not isURL(url):
            emojis = findEmojis(argv)
            if not emojis:
                emojis = findEmojis(url)
                if not emojis:
                    raise ArgumentError("Please input an image by URL or attachment.")
                s = emojis[0]
                name = url[:url.index(s)].strip()
            else:
                s = emojis[0]
                name = argv[:argv.index(s)].strip()
            s = s[2:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            url = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
        else:
            name = " ".join(args).strip()
        if not name:
            name = "emoji_" + str(len(guild.emojis))
        print(name, url)
        resp = await create_future(requests.get, url, headers={"user-agent": "Mozilla/5." + str(xrand(1, 10))}, timeout=8)
        image = resp.content
        if len(image) > 67108864:
            raise OverflowError("Max file size to load is 64MB.")
        if len(image) > 262144:
            path = "cache/" + str(guild.id) + ".png"
            f = await create_future(open, path, "wb")
            await create_future(f.write, image)
            await create_future(f.close)
            resp = await imageProc(path, "resize_max", [128], guild.id)
            fn = resp[0]
            f = await create_future(open, fn, "rb")
            image = await create_future(f.read)
            create_future_ex(f.close)
        emoji = await guild.create_custom_emoji(image=image, name=name, reason="CreateEmoji command")
        await message.add_reaction(emoji)
        return (
           "```css\nSuccessfully created emoji [" + noHighlight(emoji) + "] in ["
            + noHighlight(guild.name) + "].```"
        )


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


class Char2Emoj(Command):
    name = ["C2E"]
    min_level = 0
    description = "Makes emoji blocks using a string."
    usage = "<0:string> <1:emoji_1> <2:emoji_2>"

    async def __call__(self, args, **extra):
        try:
            if len(args) != 3:
                raise IndexError
            for i in range(1, 3):
                if args[i][0] == ":" and args[i][-1] != ":":
                    args[i] = "<" + args[i] + ">"
            return await create_future(_c2e, *args[:3])
        except IndexError:
            raise ArgumentError(
                "Exactly 3 arguments are required for this command.\n"
                + "Place quotes around arguments containing spaces as required."
            )


async def get_image(bot, message, args, argv, ext="png"):
    if message.attachments:
        args = [bestURL(a) for a in message.attachments] + args
        argv = " ".join(bestURL(a) for a in message.attachments) + " " * bool(argv) + argv
    if not args:
        raise ArgumentError("Please input an image by URL or attachment.")
    url = args.pop(0)
    url = await bot.followURL(url, best=True)
    if not isURL(url):
        emojis = findEmojis(argv)
        if not emojis:
            emojis = findEmojis(url)
            if not emojis:
                raise ArgumentError("Please input an image by URL or attachment.")
            s = emojis[0]
            value = url[url.index(s) + len(s):].strip()
        else:
            s = emojis[0]
            value = argv[argv.index(s) + len(s):].strip()
        s = s[2:]
        i = s.index(":")
        e_id = s[i + 1:s.rindex(">")]
        url = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
    else:
        value = " ".join(args).strip()
    if not value:
        value = 2
    else:
        value = await bot.evalMath(value, message.guild.id)
        if not value >= -16 or not value <= 16:
            raise OverflowError("Maximum multiplier input is 16.")
    try:
        name = url[url.rindex("/") + 1:]
        if not name:
            raise ValueError
        if "." in name:
            name = name[:name.rindex(".")]
    except ValueError:
        name = "unknown"
    if "." not in name:
        name += "." + ext
    return name, value, url


class Saturate(Command):
    name = ["Saturation", "ImageSaturate"]
    min_level = 0
    description = "Changes colour saturation of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv)
        resp = await imageProc(url, "Enhance", ["Color", value], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Contrast(Command):
    name = ["ImageContrast"]
    min_level = 0
    description = "Changes colour contrast of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv)
        resp = await imageProc(url, "Enhance", ["Contrast", value], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Brightness(Command):
    name = ["Brighten", "ImageBrightness"]
    min_level = 0
    description = "Changes colour brightness of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv)
        resp = await imageProc(url, "Enhance", ["Brightness", value], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Sharpness(Command):
    name = ["Sharpen", "ImageSharpness"]
    min_level = 0
    description = "Changes colour sharpness of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv)
        resp = await imageProc(url, "Enhance", ["Sharpness", value], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class HueShift(Command):
    name = ["Hue"]
    min_level = 0
    description = "Changes colour hue of supplied image."
    usage = "<0:url{attached_file}> <1:adjustment[0.5]>"
    no_parse = True
    rate_limit = (2, 3)

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv)
        resp = await imageProc(url, "hue_shift", [value], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Colour(Command):
    name = ["RGB", "HSV", "CMY", "LAB", "LUV", "XYZ", "Color"]
    min_level = 0
    description = "Creates a 128x128 image filled with the target colour."
    usage = "<Colour>"
    no_parse = True
    rate_limit = (1, 2)
    flags = "v"
    trans = {
        "hsv": hsv_to_rgb,
        "cmy": cmy_to_rgb,
        "lab": lab_to_rgb,
        "luv": luv_to_rgb,
        "xyz": xyz_to_rgb,
    }

    async def __call__(self, bot, guild, channel, name, argv, **void):
        argv = argv.replace("#", "").replace(",", " ").strip()
        if " " in argv:
            channels = [min(255, max(0, int(round(float(i.strip()))))) for i in argv.split(" ")[:5] if i.strip()]
            if len(channels) not in (3, 4):
                raise ArgumentError("Please input 3 or 4 channels for colour input.")
        else:
            try:
                raw = int(argv, 16)
                if len(argv) <= 6:
                    channels = [raw >> 16 & 255, raw >> 8 & 255, raw & 255]
                elif len(argv) <= 8:
                    channels = [raw >> 16 & 255, raw >> 8 & 255, raw & 255, raw >> 24 & 255]
                else:
                    raise ValueError
            except ValueError:
                raise ArgumentError("Please input a valid hex colour.")
        if name in self.trans:
            if name in "lab luv":
                adj = channels
            else:
                adj = [x / 255 for x in channels]
            channels = [round(x * 255) for x in self.trans[name](adj)]
        adj = [x / 255 for x in channels]
        msg = (
            "```ini\nHEX colour code: " + sbHighlight(bytes(channels).hex().upper())
            + "\nDEC colour code: " + sbHighlight(colour2Raw(channels))
            + "\nRGB values: " + str(channels)
            + "\nHSV values: " + sbHighlight(", ".join(str(round(x * 255)) for x in rgb_to_hsv(adj)))
            + "\nCMY values: " + sbHighlight(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
            + "\nLAB values: " + sbHighlight(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
            + "\nLUV values: " + sbHighlight(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
            + "\nXYZ values: " + sbHighlight(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
            + "```"
        )
        resp = await imageProc("from_colour", "$", [channels], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename="colour.png")
        await sendFile(channel, msg, f, filename=fn, best=True)


class Rainbow(Command):
    name = ["RainbowGIF"]
    min_level = 0
    description = "Creates a .gif image from repeatedly hueshifting supplied image."
    usage = "<0:url{attached_file}> <1:duration[2]>"
    no_parse = True
    rate_limit = (5, 8)
    _timeout_ = 3

    async def __call__(self, bot, guild, message, args, argv, **void):
        name, value, url = await get_image(bot, message, args, argv, ext="gif")
        resp = await imageProc(url, "rainbow_gif", [value], guild.id, timeout=32)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class CreateGIF(Command):
    name = ["Animate", "GIF"]
    min_level = 0
    description = "Combines multiple supplied images, and/or optionally a video, into an animated .gif image."
    usage = "<0*:urls{attached_files}> <-2:framerate_setting(?r)> <-1:framerate[16]>"
    no_parse = True
    rate_limit = (8, 12)
    _timeout_ = 5
    flags = "r"

    async def __call__(self, bot, guild, message, flags, args, **void):
        if not bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server to create GIF images.")
        if message.attachments:
            args += [bestURL(a) for a in message.attachments]
        if not args:
            raise ArgumentError("Please input images by URL or attachment.")
        if "r" in flags:
            fr = args.pop(-1)
            rate = await bot.evalMath(fr, guild)
        else:
            rate = 16
        if rate <= 0:
            args = args[:1]
            rate = 1
        delay = round(1000 / rate)
        if delay <= 0:
            args = args[-1:]
            delay = 1000
        if delay > 16777215:
            raise OverflowError("GIF image framerate too low.")
        video = None
        for i, url in enumerate(args):
            url = await bot.followURL(url, best=True)
            if "discord" not in url and "channels" not in url:
                url, size, dur, fps = await create_future(get_video, url, 16)
                if size and dur and fps:
                    video = (url, size, dur, fps)
            if not isURL(url):
                raise ArgumentError("Invalid URL detected: \"" + url + '"')
            args[i] = url
        name = "unknown.gif"
        if video is not None:
            resp = await imageProc("create_gif", "$", ["video", video, delay], guild.id, timeout=64)
        else:
            resp = await imageProc("create_gif", "$", ["image", args, delay], guild.id, timeout=64)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Resize(Command):
    name = ["ImageScale", "Scale", "Rescale", "ImageResize"]
    min_level = 0
    description = "Changes size of supplied image, using an optional scaling operation."
    usage = "<0:url{attached_file}> <1:x_multiplier[0.5]> <2:y_multiplier[x]> <3:operation[auto](?l)>"
    no_parse = True
    rate_limit = 3
    flags = "l"

    async def __call__(self, bot, guild, message, flags, args, argv, **void):
        if message.attachments:
            args = [bestURL(a) for a in message.attachments] + args
            argv = " ".join(bestURL(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            if "l" in flags:
                return (
                    "```ini\nAvailable scaling operations: ["
                    + "nearest, linear, hamming, bicubic, lanczos, auto]```"
                )
            raise ArgumentError("Please input an image by URL or attachment.")
        url = args.pop(0)
        url = await bot.followURL(url, best=True)
        if not isURL(url):
            emojis = findEmojis(argv) + findEmojis(url)
            if not emojis:
                raise ArgumentError("Please input an image by URL or attachment.")
            s = emojis[0]
            value = argv[argv.index(s) + len(s):].strip()
            s = s[2:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            url = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
        else:
            value = " ".join(args).strip()
        if not value:
            x = y = 0.5
            op = "auto"
        else:
            value = value.replace("x", " ").replace("X", " ").replace("*", " ").replace("×", " ")
            try:
                spl = shlex.split(value)
            except ValueError:
                spl = value.split()
            x = await bot.evalMath(spl.pop(0), message.guild.id)
            if spl:
                y = await bot.evalMath(spl.pop(0), message.guild.id)
            else:
                y = x
            for value in (x, y):
                if not value >= -16 or not value <= 16:
                    raise OverflowError("Maximum multiplier input is 16.")
            if spl:
                op = " ".join(spl)
            else:
                op = "auto"
        try:
            name = url[url.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        if "." not in name:
            name += ".png"
        resp = await imageProc(url, "resize_mult", [x, y, op], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


class Magik(Command):
    name = ["ImageMagik", "IMGMagik"]
    min_level = 0
    description = "Applies the Magik filter to an image."
    usage = "<0:url{attached_file}>"
    no_parse = True
    rate_limit = (4, 6)
    _timeout_ = 2

    async def __call__(self, bot, guild, channel, message, args, argv, **void):
        if message.attachments:
            args = [a.url for a in message.attachments] + args
            argv = " ".join(a.url for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            raise ArgumentError("Please input an image by URL or attachment.")
        url = args.pop(0)
        url = await bot.followURL(url)
        if not isURL(url):
            emojis = findEmojis(argv) + findEmojis(url)
            if not emojis:
                raise ArgumentError("Please input an image by URL or attachment.")
            s = emojis[0]
            s = s[2:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            url = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
        try:
            name = url[url.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        if "." not in name:
            name += ".png"
        if "cdn.discord" not in url:
            resp = await imageProc(url, "resize_to", [512, 512, "hamming"], guild.id)
            fn = resp[0]
            f = discord.File(fn, filename=name)
            msg = await channel.send(file=f)
            url = msg.attachments[0].url
        else:
            msg = None
        try:
            resp = await create_future(requests.get, "https://api.alexflipnote.dev/filter/magik?image=" + url, timeout=8)
        except:
            if msg is not None:
                await bot.silentDelete(msg)
            raise
        if msg is not None:
            create_task(bot.silentDelete(msg))
        b = resp.content
        f = discord.File(io.BytesIO(b), filename=name)
        await sendFile(message.channel, "", f)


class Blend(Command):
    name = ["ImageBlend", "ImageOP"]
    min_level = 0
    description = "Combines the two supplied images, using an optional blend operation."
    usage = "<0:url1{attached_file}> <1:url2{attached_file}> <2:operation[replace](?l)> <3:opacity[1]>"
    no_parse = True
    rate_limit = (3, 5)
    flags = "l"

    async def __call__(self, bot, guild, message, flags, args, argv, **void):
        if message.attachments:
            args = [bestURL(a) for a in message.attachments] + args
            argv = " ".join(bestURL(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            if "l" in flags:
                return (
                    "```ini\nAvailable blend operations: ["
                    + "replace, add, sub, mul, div, mod, and, or, xor, nand, nor, xnor, "
                    + "difference, overlay, screen, soft, hard, lighten, darken, "
                    + "burn, linearburn, dodge, lineardodge, hue, sat, lum, extract, merge]```"
                )
            raise ArgumentError("Please input an image by URL or attachment.")
        url1 = args.pop(0)
        url1 = await bot.followURL(url1, best=True)
        url2 = args.pop(0)
        url2 = await bot.followURL(url2, best=True)
        fromA = False
        if not isURL(url1):
            emojis = findEmojis(argv)
            if not emojis:
                emojis = findEmojis(url1)
                if not emojis:
                    raise ArgumentError("Please input an image by URL or attachment.")
                s = emojis[0]
                argv = url1[url1.index(s) + len(s):].strip()
            else:
                s = emojis[0]
                argv = argv[argv.index(s) + len(s):].strip()
            s = s[2:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            url1 = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
            fromA = True
        if not isURL(url2):
            emojis = findEmojis(argv)
            if not emojis:
                emojis = findEmojis(url2)
                if not emojis:
                    raise ArgumentError("Please input an image by URL or attachment.")
                s = emojis[0]
                argv = url2[url2.index(s) + len(s):].strip()
            else:
                s = emojis[0]
                argv = argv[argv.index(s) + len(s):].strip()
            s = s[2:]
            i = s.index(":")
            e_id = s[i + 1:s.rindex(">")]
            url1 = "https://cdn.discordapp.com/emojis/" + e_id + ".png?v=1"
            fromA = True
        if fromA:
            value = argv
        else:
            value = " ".join(args).strip()
        if not value:
            opacity = 1
            operation = "replace"
        else:
            try:
                spl = shlex.split(value)
            except ValueError:
                spl = value.split()
            operation = spl.pop(0)
            if spl:
                opacity = await bot.evalMath(spl.pop(-1), message.guild.id)
            else:
                opacity = 1
            if not opacity >= -16 or not opacity <= 16:
                raise OverflowError("Maximum multiplier input is 16.")
            if spl:
                operation += " ".join(spl)
            if not operation:
                operation = "replace"
        try:
            name = url1[url1.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        if "." not in name:
            name += ".png"
        resp = await imageProc(url1, "blend_op", [url2, operation, opacity], guild.id)
        fn = resp[0]
        f = discord.File(fn, filename=name)
        await sendFile(message.channel, "", f, filename=fn)


f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    cat_key = auth["cat_api_key"]
except:
    cat_key = None
    print("WARNING: cat_api_key not found. Unable to use Cat API to pull cat images.")


class Cat(Command):
    is_command = True
    if cat_key:
        header = {"x-api-key": cat_key}
    else:
        header = None
    min_level = 0
    description = "Pulls a random image from thecatapi.com, api.alexflipnote.dev/cats, or cdn.nekos.life/meow, and embeds it."
    usage = "<verbose(?v)>"
    flags = "v"
    rate_limit = 0.25

    def __load__(self):
        self.buffer = deque()
        create_future_ex(self.refill_buffer, 64)
    
    def fetch_one(self):
        if not self.header or random.random() > 2 / 3:
            if random.random() > 2 / 3:
                url = nekos.cat()
            else:
                for _ in loop(8):
                    resp = requests.get(
                        "https://api.alexflipnote.dev/cats",
                        headers=self.header,
                        timeout=8,
                    )
                    try:
                        d = json.loads(resp.content)
                    except:
                        d = eval(resp.content, {}, eval_const)
                    try:
                        if type(d) is list:
                            d = random.choice(d)
                        url = d["file"]
                        break
                    except KeyError:
                        pass
                time.sleep(0.25)
        else:
            for _ in loop(8):
                resp = requests.get(
                    "https://api.thecatapi.com/v1/images/search",
                    headers=self.header,
                    timeout=8,
                )
                try:
                    d = json.loads(resp.content)
                except:
                    d = eval(resp.content, {}, eval_const)
                try:
                    if type(d) is list:
                        d = random.choice(d)
                    url = d["url"]
                    break
                except KeyError:
                    pass
            time.sleep(0.25)
        return url

    def refill_buffer(self, amount):
        while len(self.buffer) < amount + 1:
            url = self.fetch_one()
            self.buffer.append(url)

    def get_buffer(self, amount):
        if len(self.buffer) < amount + 1:
            create_future_ex(self.refill_buffer(amount << 1))
            return self.fetch_one()
        return self.buffer.popleft()

    async def __call__(self, channel, flags, **void):
        url = await create_future(self.get_buffer, 32)
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=randColour(),
        )
        emb.set_image(url=url)
        print(url)
        return {
            "embed": emb
        }


class Dog(Command):
    min_level = 0
    description = "Pulls a random image from api.alexflipnote.dev/dogs or images.dog.ceo and embeds it."
    usage = "<verbose(?v)>"
    flags = "v"
    rate_limit = 0.25

    def __load__(self):
        self.buffer = deque()
        create_future_ex(self.refill_buffer, 64)

    def fetch_one(self):
        for _ in loop(8):
            x = random.random() > 2 / 3
            if x:
                resp = requests.get(
                    "https://api.alexflipnote.dev/dogs",
                    timeout=8,
                )
            else:
                resp = requests.get(
                    "https://dog.ceo/api/breeds/image/random",
                    timeout=8,
                )
            try:
                d = json.loads(resp.content)
            except:
                d = eval(resp.content, {}, eval_const)
            try:
                if type(d) is list:
                    d = random.choice(d)
                url = d["file" if x else "message"]
                break
            except KeyError:
                pass
        time.sleep(0.25)
        url = url.replace("\\", "/")
        while "///" in url:
            url = url.replace("///", "//")
        return url

    def refill_buffer(self, amount):
        while len(self.buffer) < amount + 1:
            url = self.fetch_one()
            self.buffer.append(url)

    def get_buffer(self, amount):
        if len(self.buffer) < amount + 1:
            create_future_ex(self.refill_buffer(amount << 1))
            return self.fetch_one()
        return self.buffer.popleft()

    async def __call__(self, channel, flags, **void):
        url = await create_future(self.get_buffer, 32)
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=randColour(),
        )
        emb.set_image(url=url)
        print(url)
        return {
            "embed": emb
        }


class DeviantArt(Command):
    server_only = True
    min_level = 2
    description = "Subscribes to a DeviantArt Gallery, reposting links to all new posts."
    usage = "<reversed(?r)> <disable(?d)>"
    flags = "raed"
    rate_limit = 4

    async def __call__(self, argv, flags, channel, guild, bot, **void):
        if not bot.isTrusted(guild.id):
            raise PermissionError("Must be in a trusted server to subscribe to DeviantArt Galleries.")
        data = bot.data.deviantart
        update = bot.database.deviantart.update
        if not argv:
            assigned = data.get(channel.id, ())
            if not assigned:
                return "```ini\nNo currently subscribed DeviantArt Galleries for [#" + noHighlight(channel) + "].```"
            if "d" in flags:
                try:
                    data.pop(channel.id)
                except KeyError:
                    pass
                return "```css\nSuccessfully removed all DeviantArt Gallery subscriptions from [#" + noHighlight(channel) + "].```"
            return "Currently subscribed DeviantArt Galleries for [#" + noHighlight(channel) + "]:```ini" + strIter(assigned, key=lambda x: x["user"]) + "```"
        url = await bot.followURL(argv)
        if not isURL(url):
            raise ArgumentError("Please input a valid URL.")
        if "deviantart.com" not in url:
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        url = url[url.index("deviantart.com") + 15:]
        spl = url.split("/")
        user = spl[0]
        if spl[1] != "gallery":
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        content = spl[2].split("&")[0]
        folder = noHighlight(spl[-1].split("&")[0])
        try:
            content = int(content)
        except (ValueError, TypeError):
            if content in (user, "all"):
                content = user
            else:
                raise TypeError("Invalid Gallery type.")
        if content in self.data.get(channel.id, {}):
            raise KeyError("Already subscribed to " + user + ": " + folder)
        if "d" in flags:
            try:
                data.get(channel.id).pop(content)
            except KeyError:
                raise KeyError("Not currently subscribed to " + user + ": " + folder)
            else:
                if channel.id in data and not data[channel.id]:
                    data.pop(channel.id)
                update()
                return "```css\nSuccessfully unsubscribed from " + sbHighlight(user) + ": " + sbHighlight(folder) + ".```"
        setDict(data, channel.id, {}).__setitem__(content, {"user": user, "type": "gallery", "reversed": ("r" in flags), "entries": {}})
        update()
        return "```css\nSuccessfully subscribed to " + sbHighlight(user) + ": " + sbHighlight(folder) + ", posting in reverse order" * ("r" in flags) + ".```"


class UpdateDeviantArt(Database):
    name = "deviantart"

    async def processPart(self, found, c_id):
        bot = self.bot
        try:
            channel = await bot.fetch_channel(c_id)
            if not bot.isTrusted(channel.guild.id):
                raise LookupError
        except LookupError:
            self.data.pop(c_id)
            return
        try:
            assigned = self.data[c_id]
            for content in assigned:
                items = found[content]
                entries = assigned[content]["entries"]
                new = tuple(items)
                orig = tuple(entries)
                if hash(tuple(sorted(new))) != hash(tuple(sorted(orig))):
                    if assigned[content].get("reversed", False):
                        it = reversed(new)
                    else:
                        it = new
                    for i in it:
                        if i not in entries:
                            entries[i] = True
                            self.update()
                            home = "https://www.deviantart.com/" + items[i][2]
                            emb = discord.Embed(
                                colour=discord.Colour(1),
                                description="🔔 New Deviation from " + items[i][2] + " 🔔\n" + items[i][0],
                            ).set_image(url=items[i][1]).set_author(name=items[i][2], url=home, icon_url=items[i][3])
                            await channel.send(embed=emb)
                    for i in orig:
                        if i not in items:
                            entries.pop(i)
                            self.update()
        except:
            print(traceback.format_exc())

    async def __call__(self):
        t = setDict(self.__dict__, "time", 0)
        if utc() - t < 300:
            return
        self.time = inf
        conts = {i: a[i]["user"] for a in tuple(self.data.values()) for i in a}
        found = {}
        base = "https://www.deviantart.com/_napi/da-user-profile/api/gallery/contents?limit=24&username="
        for content, user in conts.items():
            if type(content) is str:
                f_id = "&all_folder=true&mode=oldest"
            else:
                f_id = "&folderid=" + str(content)
            items = {}
            try:
                url = base + user + f_id + "&offset="
                for i in range(0, 13824, 24):
                    req = url + str(i)
                    print(req)
                    resp = await create_future(requests.get, req, timeout=8)
                    try:
                        d = json.loads(resp.content)
                    except:
                        d = eval(resp.content, {}, eval_const)
                    for res in d["results"]:
                        deviation = res["deviation"]
                        media = deviation["media"]
                        prettyName = media["prettyName"]
                        orig = media["baseUri"]
                        extra = ""
                        token = "?token=" + media["token"][0]
                        for t in reversed(media["types"]):
                            if t["t"].lower() == "fullview":
                                if "c" in t:
                                    extra = "/" + t["c"].replace("<prettyName>", prettyName)
                                    break
                        image_url = orig + extra + token
                        items[deviation["deviationId"]] = (deviation["url"], image_url, deviation["author"]["username"], deviation["author"]["usericon"])
                    if not d.get("hasMore", None):
                        break
            except:
                print(traceback.format_exc())
            found[content] = items
        for c_id in tuple(self.data):
            create_task(self.processPart(found, c_id))
        self.time = utc()


class UpdateImages(Database):
    name = "images"


class UpdateReacts(Database):
    name = "reacts"

    async def _nocommand_(self, text, edit, orig, message, **void):
        if message.guild is None or not orig:
            return
        g_id = message.guild.id
        data = self.data
        if g_id in data:
            try:
                following = self.data[g_id]
                if type(following) != multiDict:
                    following = self.data[g_id] = multiDict(following)
                reacting = {}
                for k in following:
                    if is_alphanumeric(k) and " " not in k:
                        words = text.split(" ")
                    else:
                        words = message.content.lower()
                    if k in words:
                        emojis = following[k]
                        reacting[words.index(k) / len(words)] = emojis
                for r in sorted(list(reacting)):
                    for react in reacting[r]:
                        await message.add_reaction(react)
            except ZeroDivisionError:
                pass
            except:
                print(traceback.format_exc())