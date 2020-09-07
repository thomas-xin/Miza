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
        # Attempt to get as close to width 512 as possible for download
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

    async def __call__(self, bot, flags, args, argv, user, message, channel, guild, perm, **void):
        update = self.data.images.update
        imglists = bot.data.images
        images = imglists.get(guild.id, {})
        if "a" in flags or "e" in flags or "d" in flags:
            if message.attachments:
                args = [best_url(a) for a in message.attachments] + args
                argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
            req = 2
            if perm < req:
                reason = "to change image list for " + guild.name
                raise self.perm_error(perm, req, reason)
            if "a" in flags or "e" in flags:
                lim = 64 << bot.is_trusted(guild.id) * 2 + 1
                if len(images) > lim:
                    raise OverflowError(f"Image list for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
                key = " ".join(args[:-1]).casefold()
                if len(key) > 2000:
                    raise ArgumentError("Image tag too long.")
                elif not key:
                    raise ArgumentError("Image tag must not be empty.")
                urls = await bot.follow_url(args[-1], best=True, allow=True, limit=1)
                url = urls[0]
                if len(url) > 2000:
                    raise ArgumentError("Image url too long.")
                images[key] = url
                images = {i: images[i] for i in sorted(images)}
                imglists[guild.id] = images
                update()
                if not "h" in flags:
                    return css_md(f"Successfully added {sqr_md(key)} to the image list for {sqr_md(guild)}.")
            if not args:
                # This deletes all images for the current guild
                if "f" not in flags and len(images) > 1:
                    return css_md(sqr_md(f"WARNING: {len(images)} IMAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."))
                imglists[guild.id] = {}
                update()
                return italics(css_md(f"Successfully removed all {sqr_md(len(images))} images from the image list for {sqr_md(guild)}."))
            key = argv.casefold()
            images.pop(key)
            imglists[guild.id] = images
            update()
            return italics(css_md(f"Successfully removed {sqr_md(key)} from the image list for {sqr_md(guild)}."))
        if not argv and not "r" in flags:
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-image-img-"
                + str(user.id) + "_0"
                + "-\nLoading Image database...```*"
            )
        sources = alist()
        for tag in args:
            t = tag.casefold()
            if t in images:
                sources.append(images[t])
        r = flags.get("r", 0)
        for _ in loop(r):
            sources.append(choice(tuple(images)))
        if not len(sources):
            raise LookupError(f"Target image {argv} not found. Use img for list.")
        url = choice(sources)
        if "v" in flags:
            return url
        emb = discord.Embed(
            url=url,
            colour=rand_colour(),
        )
        emb.set_image(url=url)
        bot.send_embeds(channel, embed=emb)

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
        content = "*```" + "\n" * ("\n" in content[:i]) + (
            "callback-image-img-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not images:
            content += f"Image list for {str(guild).replace('`', '')} is currently empty.```*"
            msg = ""
        else:
            content += f"{len(images)} images currently assigned for {str(guild).replace('`', '')}:```*"
            msg = "```ini\n" + iter2str({k: "\n" + images[k] for k in tuple(images)[pos:pos + page]}) + "```"
        emb = discord.Embed(
            description=content + msg,
            colour=rand_colour(),
        )
        emb.set_author(**get_author(user))
        more = len(images) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
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
        curr = set_dict(following, guild.id, mdict())
        if type(curr) is not mdict:
            following[guild.id] = curr = mdict(curr)
        if not argv:
            if "d" in flags:
                # This deletes all auto reacts for the current guild
                if "f" not in flags and len(curr) > 1:
                    return css_md(sqr_md(f"WARNING: {len(curr)} ITEMS TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."))
                if guild.id in following:
                    following.pop(guild.id)
                    update()
                return italics(css_md(f"Successfully removed all {sqr_md(len(curr))} auto reacts for {sqr_md(guild)}."))
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-image-react-"
                + str(user.id) + "_0"
                + "-\nLoading React database...```*"
            )
        if "d" in flags:
            a = unicode_prune(argv).casefold()
            if a in curr:
                curr.pop(a)
                update()
                return italics(css_md(f"Removed {sqr_md(a)} from the auto react list for {sqr_md(guild)}."))
            else:
                raise LookupError(f"{a} is not in the auto react list.")
        lim = 64 << bot.is_trusted(guild.id) * 2 + 1
        if curr.count() >= lim:
            raise OverflowError(f"React list for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
        # Limit substring length to 64
        a = unicode_prune(" ".join(args[:-1])).casefold()[:64]
        try:
            e_id = int(args[-1])
        except:
            emoji = args[-1]
        else:
            emoji = await bot.fetch_emoji(e_id)
        # This reaction indicates that the emoji was valid
        await message.add_reaction(emoji)
        curr.append(a, str(emoji))
        following[guild.id] = mdict({i: curr[i] for i in sorted(curr)})
        update()
        return css_md(f"Added {sqr_md(a)} ➡️ {sqr_md(args[-1])} to the auto react list for {sqr_md(guild)}.")
    
    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        following = bot.data.reacts
        curr = following.get(guild.id, mdict())
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
        content = "*```" + "\n" * ("\n" in content[:i]) + (
            "callback-image-react-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not curr:
            content += f"No currently assigned auto reactions for {str(guild).replace('`', '')}.```*"
            msg = ""
        else:
            content += f"{len(curr)} auto reactions currently assigned for {str(guild).replace('`', '')}:```*"
            key = lambda x: "\n" + ", ".join(x)
            msg = "```ini\n" + iter2str({k: curr[k] for k in tuple(curr)[pos:pos + page]}, key=key) + "```"
        emb = discord.Embed(
            description=content + msg,
            colour=rand_colour(),
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
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
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, args, argv, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            raise ArgumentError("Please enter URL, emoji, or attached file to add.")
        with discord.context_managers.Typing(channel):
            url = args.pop(-1)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError("Please input an image by URL or attachment.")
            url = urls[0]
            name = " ".join(args).strip()
            if not name:
                name = "emoji_" + str(len(guild.emojis))
            print(name, url)
            resp = await Request(url, timeout=12, aio=True)
            image = resp
            if len(image) > 67108864:
                raise OverflowError("Max file size to load is 64MB.")
            if len(image) > 262144:
                path = "cache/" + str(guild.id)
                f = await create_future(open, path, "wb", timeout=18)
                await create_future(f.write, image, timeout=18)
                await create_future(f.close, timeout=18)
                try:
                    resp = await process_image(path, "resize_max", [128], guild, timeout=32)
                except:
                    with suppress():
                        os.remove(path)
                    raise
                else:
                    fn = resp[0]
                    f = await create_future(open, fn, "rb", timeout=18)
                    image = await create_future(f.read, timeout=18)
                    create_future_ex(f.close, timeout=18)
                    with suppress():
                        os.remove(fn)
                with suppress():
                    os.remove(path)
            emoji = await guild.create_custom_emoji(image=image, name=name, reason="CreateEmoji command")
            # This reaction indicates the emoji was created successfully
            await message.add_reaction(emoji)
        return css_md(f"Successfully created emoji {sqr_md(emoji)} for {sqr_md(guild)}.")


async def get_image(bot, user, message, args, argv, ext="png"):
    # Take input from any attachments, or otherwise the message contents
    if message.attachments:
        args = [best_url(a) for a in message.attachments] + args
        argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
    if not args:
        raise ArgumentError("Please input an image by URL or attachment.")
    url = args.pop(0)
    urls = await bot.follow_url(url, best=True, allow=True, limit=1)
    if not urls:
        urls = await bot.follow_to_image(argv)
        if not urls:
            urls = await bot.follow_to_image(url)
            if not urls:
                raise ArgumentError("Please input an image by URL or attachment.")
    url = urls[0]
    value = " ".join(args).strip()
    if not value:
        value = 2
    else:
        value = await bot.eval_math(value, user)
        if not abs(value) <= 64:
            raise OverflowError("Maximum multiplier input is 64.")
    # Try and find a good name for the output image
    try:
        name = url[url.rindex("/") + 1:]
        if not name:
            raise ValueError
        if "." in name:
            name = name[:name.rindex(".")]
    except ValueError:
        name = "unknown"
    if not name.endswith("." + ext):
        name += "." + ext
    return name, value, url


class Saturate(Command):
    name = ["Saturation", "ImageSaturate"]
    min_level = 0
    description = "Changes colour saturation of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Color", value], user, timeout=28)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Contrast(Command):
    name = ["ImageContrast"]
    min_level = 0
    description = "Changes colour contrast of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Contrast", value], user, timeout=28)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Brightness(Command):
    name = ["Brighten", "ImageBrightness"]
    min_level = 0
    description = "Changes colour brightness of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Brightness", value], user, timeout=28)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Sharpness(Command):
    name = ["Sharpen", "ImageSharpness"]
    min_level = 0
    description = "Changes colour sharpness of supplied image."
    usage = "<0:url{attached_file}> <1:multiplier[2]>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Sharpness", value], user, timeout=28)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class HueShift(Command):
    name = ["Hue"]
    min_level = 0
    description = "Changes colour hue of supplied image."
    usage = "<0:url{attached_file}> <1:adjustment[0.5]>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "hue_shift", [value], user, timeout=32)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Invert(Command):
    name = ["Negate"]
    min_level = 0
    description = "Inverts supplied image."
    usage = "<0:url{attached_file}>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "ImageOps.invert", [], user, timeout=24)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class GreyScale(Command):
    name = ["GrayScale"]
    min_level = 0
    description = "Greyscales supplied image."
    usage = "<0:url{attached_file}>"
    no_parse = True
    rate_limit = (2, 3)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "ImageOps.grayscale", [], user, timeout=24)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Magik(Command):
    min_level = 0
    description = "Applies the Magik image filter to supplied image."
    usage = "<0:url{attached_file}>"
    no_parse = True
    rate_limit = (3, 4)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik", [], user, timeout=40)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


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
    typing = True

    async def __call__(self, bot, user, channel, name, argv, **void):
        argv = single_space(argv.replace("#", "").replace(",", " ")).strip()
        # Try to parse as colour tuple first
        if " " in argv:
            channels = [min(255, max(0, int(round(float(i.strip()))))) for i in argv.split(" ")[:5] if i]
            if len(channels) not in (3, 4):
                raise ArgumentError("Please input 3 or 4 channels for colour input.")
        else:
            # Try to parse as hex colour value
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
        # Any exceptions encountered during colour transformations will immediately terminate the command
        msg = ini_md(
            "HEX colour code: " + sqr_md(bytes(channels).hex().upper())
            + "\nDEC colour code: " + sqr_md(colour2raw(channels))
            + "\nRGB values: " + str(channels)
            + "\nHSV values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsv(adj)))
            + "\nCMY values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
            + "\nLAB values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
            + "\nLUV values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
            + "\nXYZ values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
        )
        with discord.context_managers.Typing(channel):
            resp = await process_image("from_colour", "$", [channels], user)
            fn = resp[0]
            f = discord.File(fn, filename="colour.png")
        await send_with_file(channel, msg, f, filename=fn, best=True)


class Rainbow(Command):
    name = ["RainbowGIF"]
    min_level = 0
    description = "Creates a .gif image from repeatedly hueshifting supplied image."
    usage = "<0:url{attached_file}> <1:duration[2]>"
    no_parse = True
    rate_limit = (5, 8)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "rainbow_gif", [value, "-gif"], user, timeout=40)
            fn = resp[0]
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Spin(Command):
    name = ["RainbowGIF"]
    min_level = 0
    description = "Creates a .gif image from repeatedly rotating supplied image."
    usage = "<0:url{attached_file}> <1:duration[2]>"
    no_parse = True
    rate_limit = (5, 8)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "spin_gif", [value, "-gif"], user, timeout=40)
            fn = resp[0]
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class CreateGIF(Command):
    name = ["Animate", "GIF"]
    min_level = 0
    description = "Combines multiple supplied images, and/or optionally a video, into an animated .gif image."
    usage = "<0*:urls{attached_files}> <-2:framerate_setting(?r)> <-1:framerate[16]>"
    no_parse = True
    rate_limit = (8, 24)
    _timeout_ = 5
    flags = "r"
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args += [best_url(a) for a in message.attachments]
        if not args:
            raise ArgumentError("Please input images by URL or attachment.")
        if "r" in flags:
            fr = args.pop(-1)
            rate = await bot.eval_math(fr, user)
        else:
            rate = 16
        # Validate framerate values to prevent issues further down the line
        if rate <= 0:
            args = args[:1]
            rate = 1
        delay = round(1000 / rate)
        if delay <= 0:
            args = args[-1:]
            delay = 1000
        if delay >= 16777216:
            raise OverflowError("GIF image framerate too low.")
        with discord.context_managers.Typing(channel):
            video = None
            for i, url in enumerate(args):
                urls = await bot.follow_url(url, best=True, allow=True, limit=1)
                url = urls[0]
                if "discord" not in url and "channels" not in url:
                    url, size, dur, fps = await create_future(get_video, url, 16, timeout=60)
                    if size and dur and fps:
                        video = (url, size, dur, fps)
                if not url:
                    raise ArgumentError(f'Invalid URL detected: "{url}".')
                args[i] = url
            name = "unknown.gif"
            if video is not None:
                resp = await process_image("create_gif", "$", ["video", video, delay], user, timeout=96)
            else:
                resp = await process_image("create_gif", "$", ["image", args, delay], user, timeout=96)
            fn = resp[0]
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Resize(Command):
    name = ["ImageScale", "Scale", "Rescale", "ImageResize"]
    min_level = 0
    description = "Changes size of supplied image, using an optional scaling operation."
    usage = "<0:url{attached_file}> <1:x_multiplier[0.5]> <2:y_multiplier[x]> <3:operation[auto](?l)>"
    no_parse = True
    rate_limit = 3
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            if "l" in flags:
                return ini_md("Available scaling operations: [nearest, linear, hamming, bicubic, lanczos, auto]")
            raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError("Please input an image by URL or attachment.")
            url = urls[0]
            value = " ".join(args).strip()
            if not value:
                x = y = 0.5
                op = "auto"
            else:
                # Parse width and height multipliers
                value = value.replace("x", " ").replace("X", " ").replace("*", " ").replace("×", " ")
                try:
                    spl = shlex.split(value)
                except ValueError:
                    spl = value.split()
                x = await bot.eval_math(spl.pop(0), user)
                if spl:
                    y = await bot.eval_math(spl.pop(0), user)
                else:
                    y = x
                for value in (x, y):
                    if not value >= -32 or not value <= 32:
                        raise OverflowError("Maximum multiplier input is 32.")
                if spl:
                    op = " ".join(spl)
                else:
                    op = "auto"
            # Try and find a good name for the output image
            try:
                name = url[url.rindex("/") + 1:]
                if not name:
                    raise ValueError
                if "." in name:
                    name = name[:name.rindex(".")]
            except ValueError:
                name = "unknown"
            if not name.endswith(".png"):
                name += ".png"
            resp = await process_image(url, "resize_mult", [x, y, op], user, timeout=36)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Fill(Command):
    name = ["ImageFill", "FillChannel", "FillImage"]
    min_level = 0
    description = "Fills an optional amount of channels in the target image with an optional value."
    usage = "<0:url{attached_file}> <1*:channels(r)(g)(b)(c)(m)(y)(h)(s)(v)(a)> <-1:value[0]>"
    no_parse = True
    rate_limit = 3
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError("Please input an image by URL or attachment.")
            url = urls[0]
            if is_numeric(args[-1]):
                value = await bot.eval_math(args.pop(-1), user)
                if type(value) is float:
                    if abs(value) <= 1:
                        value = round(value * 255)
                    else:
                        raise ValueError("invalid non-integer input value.")
            else:
                value = 255
            if not args:
                args = "rgb"
            # Try and find a good name for the output image
            try:
                name = url[url.rindex("/") + 1:]
                if not name:
                    raise ValueError
                if "." in name:
                    name = name[:name.rindex(".")]
            except ValueError:
                name = "unknown"
            if not name.endswith(".png"):
                name += ".png"
            resp = await process_image(url, "fill_channels", [value, *args], user, timeout=36)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Blend(Command):
    name = ["ImageBlend", "ImageOP"]
    min_level = 0
    description = "Combines the two supplied images, using an optional blend operation."
    usage = "<0:url1{attached_file}> <1:url2{attached_file}> <2:operation[replace](?l)> <3:opacity[0.5][1]>"
    no_parse = True
    rate_limit = (3, 5)
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            if "l" in flags:
                return ini_md(
                    "Available blend operations: ["
                    + "replace, add, sub, mul, div, mod, and, or, xor, nand, nor, xnor, "
                    + "difference, overlay, screen, soft, hard, lighten, darken, "
                    + "burn, linearburn, dodge, lineardodge, hue, sat, lum, extract, merge]"
                )
            raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            urls = await bot.follow_url(args.pop(0), best=True, allow=True, limit=1)
            if urls:
                url1 = urls[0]
            else:
                url1 = None
            urls = await bot.follow_url(args.pop(0), best=True, allow=True, limit=1)
            if urls:
                url2 = urls[0]
            else:
                url1 = None
            fromA = False
            if not url1 or not url2:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError("Please input an image by URL or attachment.")
                if not url1:
                    url1 = urls.pop(0)
                if not url2:
                    url2 = urls.pop(0)
            if fromA:
                value = argv
            else:
                value = " ".join(args).strip()
            if not value:
                opacity = 0.5
                operation = "replace"
            else:
                try:
                    spl = shlex.split(value)
                except ValueError:
                    spl = value.split()
                operation = spl.pop(0)
                if spl:
                    opacity = await bot.eval_math(spl.pop(-1), user)
                else:
                    opacity = 1
                if not opacity >= -256 or not opacity <= 256:
                    raise OverflowError("Maximum multiplier input is 256.")
                if spl:
                    operation += " ".join(spl)
                if not operation:
                    operation = "replace"
            # Try and find a good name for the output image
            try:
                name = url1[url1.rindex("/") + 1:]
                if not name:
                    raise ValueError
                if "." in name:
                    name = name[:name.rindex(".")]
            except ValueError:
                name = "unknown"
            if not name.endswith(".png"):
                name += ".png"
            resp = await process_image(url1, "blend_op", [url2, operation, opacity], user, timeout=32)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
            f = discord.File(fn, filename=name)
        await send_with_file(message.channel, "", f, filename=fn)


class Cat(Command):
    min_level = 0
    description = "Pulls a random image from thecatapi.com, api.alexflipnote.dev/cats, or cdn.nekos.life/meow, and embeds it."
    usage = "<verbose(?v)>"
    flags = "v"
    rate_limit = 0.25

    def __load__(self):
        self.buffer = deque()
        self.found = cdict()
        self.refilling = False
        create_task(self.refill_buffer(128))

    # Fetches one image from random pool
    async def fetch_one(self):
        if random.random() > 2 / 3:
            if random.random() > 2 / 3:
                x = 0
                url = await create_future(nekos.cat, timeout=8)
            else:
                x = 1
        else:
            x = 2
        if x:
            if x == 1:
                resp = await Request("https://api.alexflipnote.dev/cats", aio=True)
            else:
                resp = await Request("https://api.thecatapi.com/v1/images/search", aio=True)
            d = eval_json(resp)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 else "url"]
        return url

    # Refills image buffer to a certain amount
    async def refill_buffer(self, amount):
        try:
            while len(self.buffer) < amount + 1:
                futs = [create_task(self.fetch_one()) for _ in loop(8)]
                out = deque()
                for fut in futs:
                    with tracebacksuppressor:
                        res = await fut
                        out.append(res)
                self.buffer.extend(out)
                time.sleep(0.25)
        except:
            self.refilling = False
            raise
        self.refilling = False

    # Grabs next image from buffer, allocating when necessary
    async def get_buffer(self, amount):
        if len(self.buffer) < amount + 1:
            if not self.refilling:
                self.refilling = True
                create_task(self.refill_buffer(amount << 1))
            if len(self.found) >= 4096:
                return choice(tuple(self.found))
            if not self.buffer:
                return await create_future(nekos.cat, timeout=8)
        url = self.buffer.popleft()
        self.found[url] = True
        return url

    async def __call__(self, channel, flags, **void):
        url = await self.get_buffer(64)
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=rand_colour(),
        )
        emb.set_image(url=url)
        self.bot.send_embeds(channel, embed=emb)


class Dog(Command):
    min_level = 0
    description = "Pulls a random image from images.dog.ceo, api.alexflipnote.dev/dogs, or cdn.nekos.life/woof, and embeds it."
    usage = "<verbose(?v)>"
    flags = "v"
    rate_limit = 0.25

    def __load__(self):
        self.buffer = deque()
        self.found = cdict()
        self.refilling = False
        create_task(self.refill_buffer(128))

    # Fetches one image from random pool
    async def fetch_one(self):
        if random.random() > 2 / 3:
            if random.random() > 2 / 3:
                x = 0
                url = await create_future(nekos.img, "woof", timeout=8)
            else:
                x = 1
        else:
            x = 2
        if x:
            if x == 1:
                resp = await Request("https://api.alexflipnote.dev/dogs", aio=True)
            else:
                resp = await Request("https://dog.ceo/api/breeds/image/random", aio=True)
            d = eval_json(resp)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 else "message"]
            url = url.replace("\\", "/")
            while "///" in url:
                url = url.replace("///", "//")
        return url

    # Refills image buffer to a certain amount
    async def refill_buffer(self, amount):
        try:
            while len(self.buffer) < amount + 1:
                futs = [create_task(self.fetch_one()) for _ in loop(8)]
                out = deque()
                for fut in futs:
                    with tracebacksuppressor:
                        res = await fut
                        out.append(res)
                self.buffer.extend(out)
                time.sleep(0.25)
        except:
            self.refilling = False
            raise
        self.refilling = False

    # Grabs next image from buffer, allocating when necessary
    async def get_buffer(self, amount):
        if len(self.buffer) < amount + 1:
            if not self.refilling:
                self.refilling = True
                create_task(self.refill_buffer(amount << 1))
            if len(self.found) >= 4096:
                return choice(tuple(self.found))
            if not self.buffer:
                return await create_future(nekos.img, "woof", timeout=8)
        url = self.buffer.popleft()
        self.found[url] = True
        return url

    async def __call__(self, channel, flags, **void):
        url = await self.get_buffer(64)
        if "v" in flags:
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=rand_colour(),
        )
        emb.set_image(url=url)
        self.bot.send_embeds(channel, embed=emb)


class _8Ball(Command):
    min_level = 0
    description = "Pulls a random image from cdn.nekos.life/8ball, and embeds it."

    def __load__(self):
        self.buffer = deque()
        self.found = cdict()
        self.refilling = False
        create_future_ex(self.refill_buffer, 128)

    # Fetches one image from random pool
    fetch_one = lambda self: nekos.img("8ball")

    # Refills image buffer to a certain amount
    def refill_buffer(self, amount):
        try:
            while len(self.buffer) < amount + 1:
                url = self.fetch_one()
                self.buffer.append(url)
                time.sleep(0.25)
        except:
            self.refilling = False
            raise
        self.refilling = False

    # Grabs next image from buffer, allocating when necessary
    def get_buffer(self, amount):
        if len(self.buffer) < amount + 1:
            if not self.refilling:
                self.refilling = True
                create_future_ex(self.refill_buffer, amount << 1)
            if not self.buffer:
                return choice(tuple(self.found))
        url = self.buffer.popleft()
        self.found[url] = True
        return url

    def __call__(self, channel, **void):
        url = self.get_buffer(64)
        emb = discord.Embed(
            url=url,
            colour=rand_colour(),
        )
        emb.set_image(url=url)
        self.bot.send_embeds(channel, embed=emb)


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
            with tracebacksuppressor(ZeroDivisionError):
                following = self.data[g_id]
                if type(following) != mdict:
                    following = self.data[g_id] = mdict(following)
                reacting = {}
                for k in following:
                    if is_alphanumeric(k) and " " not in k:
                        words = text.split(" ")
                    else:
                        words = message.content.casefold()
                    if k in words:
                        emojis = following[k]
                        # Store position for each keyword found
                        reacting[words.index(k) / len(words)] = emojis
                # Reactions sorted by their order of appearance in the message
                for r in sorted(list(reacting)):
                    for react in reacting[r]:
                        try:
                            await message.add_reaction(react)
                        except discord.HTTPException as ex:
                            if "10014" in repr(ex):
                                emojis.remove(react)