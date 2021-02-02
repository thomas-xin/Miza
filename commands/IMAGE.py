try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

print = PRINT

import youtube_dlc, nekos
youtube_dl = youtube_dlc

getattr(youtube_dl, "__builtins__", {})["print"] = print


try:
    alexflipnote_key = AUTH["alexflipnote_key"]
    if not alexflipnote_key:
        raise
except:
    alexflipnote_key = None
    print("WARNING: alexflipnote_key not found. Unable to use API to generate images.")


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
    try:
        entry = downloader.extract_info(url, download=False)
    except:
        print_exc()
        return url, None, None, None
    best = 0
    size = None
    dur = None
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ""
    for fmt in fmts:
        q = fmt.get("height", 0)
        if type(q) is not int:
            q = 0
        # Attempt to get as close to 720p as possible for download
        if abs(q - 720) < abs(best - 720):
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
    min_display = "0~2"
    description = "Sends an image in the current chat from a list."
    usage = "(add|delete)? <0:tags>* <1:url>? <verbose{?v}|hide{?h}>?"
    flags = "vraedhzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    slash = True

    async def __call__(self, bot, flags, args, argv, user, message, channel, guild, perm, **void):
        update = self.data.images.update
        imglists = bot.data.images
        images = imglists.get(guild.id, {})
        if "a" in flags or "e" in flags or "d" in flags:
            if message.attachments:
                args.extend(best_url(a) for a in message.attachments)
                argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
            req = 2
            if perm < req:
                reason = "to change image list for " + guild.name
                raise self.perm_error(perm, req, reason)
            if "a" in flags or "e" in flags:
                lim = 256 << bot.is_trusted(guild.id) * 2 + 1
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
                if not "h" in flags:
                    return css_md(f"Successfully added {sqr_md(key)} to the image list for {sqr_md(guild)}.")
            if not args:
                # This deletes all images for the current guild
                if "f" not in flags and len(images) > 1:
                    return css_md(sqr_md(f"WARNING: {len(images)} IMAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."))
                imglists[guild.id] = {}
                return italics(css_md(f"Successfully removed all {sqr_md(len(images))} images from the image list for {sqr_md(guild)}."))
            key = argv.casefold()
            images.pop(key)
            imglists[guild.id] = images
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
            return escape_everyone(url)
        bot.send_as_embeds(channel, image=url, colour=xrand(1536), reference=message)

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(images) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.5)


class React(Command):
    server_only = True
    name = ["AutoReact"]
    min_level = 2
    description = "Causes ⟨MIZA⟩ to automatically assign a reaction to messages containing the substring."
    usage = "<0:react_to>? <1:react_data>? <disable{?d}>?"
    flags = "aedzf"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = (1, 2)
    slash = True

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
                return italics(css_md(f"Successfully removed all {sqr_md(len(curr))} auto reacts for {sqr_md(guild)}."))
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-image-react-"
                + str(user.id) + "_0"
                + "-\nLoading React database...```*"
            )
        if "d" in flags:
            a = full_prune(args[0])
            if a in curr:
                curr.pop(a)
                update(guild.id)
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
        emoji = str(emoji)
        # This reaction indicates that the emoji was valid
        await message.add_reaction(emoji)
        curr.append(a, emoji)
        following[guild.id] = mdict({i: curr[i] for i in sorted(curr)})
        return css_md(f"Added {sqr_md(a)} ➡️ {sqr_md(emoji)} to the auto react list for {sqr_md(guild)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(curr) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(as_str(react)))
                await asyncio.sleep(0.5)


class CreateEmoji(Command):
    server_only = True
    name = ["EmojiCreate", "EmojiCopy", "CopyEmoji", "Emoji"]
    min_level = 2
    description = "Creates a custom emoji from a URL or attached file."
    usage = "<1:name>+ <0:url>"
    flags = "aed"
    no_parse = True
    rate_limit = (3, 6)
    _timeout_ = 3
    typing = True
    slash = ("Emoji",)

    async def __call__(self, bot, user, guild, channel, message, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args.extend(best_url(a) for a in message.attachments)
            argv += " " * bool(argv) + " ".join(best_url(a) for a in message.attachments)
        if not args:
            raise ArgumentError("Please enter URL, emoji, or attached file to add.")
        with discord.context_managers.Typing(channel):
            try:
                url = args.pop(-1)
                urls = await bot.follow_url(url, best=True, allow=True, limit=1)
                if not urls:
                    urls = await bot.follow_to_image(argv)
                    if not urls:
                        urls = await bot.follow_to_image(url)
                        if not urls:
                            raise ArgumentError("Please input an image by URL or attachment.")
                url = urls[0]
            except ArgumentError:
                if not argv:
                    url = None
                    try:
                        url = await bot.get_last_image(message.channel)
                    except FileNotFoundError:
                        raise ArgumentError("Please input an image by URL or attachment.")
            name = " ".join(args).strip()
            if not name:
                name = "emoji_" + str(len(guild.emojis))
            print(name, url)
            image = resp = await bot.get_request(url)
            if len(image) > 67108864:
                raise OverflowError("Max file size to load is 64MB.")
            if len(image) > 262144 or not is_image(url):
                ts = ts_us()
                path = "cache/" + str(ts)
                f = await create_future(open, path, "wb", timeout=18)
                await create_future(f.write, image, timeout=18)
                await create_future(f.close, timeout=18)
                try:
                    resp = await process_image(path, "resize_max", [128], timeout=_timeout)
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


async def get_image(bot, user, message, args, argv, default=2, ext="png"):
    try:
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
    except ArgumentError:
        if not argv:
            url = None
            try:
                url = await bot.get_last_image(message.channel)
            except FileNotFoundError:
                raise ArgumentError("Please input an image by URL or attachment.")
    value = " ".join(args).strip()
    if not value:
        value = default
    else:
        value = await bot.eval_math(value)
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
    description = "Changes colour saturation of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Color", value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Contrast(Command):
    name = ["ImageContrast"]
    description = "Changes colour contrast of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Contrast", value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Brightness(Command):
    name = ["Brighten", "ImageBrightness"]
    description = "Changes colour brightness of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Brightness", value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Sharpness(Command):
    name = ["Sharpen", "ImageSharpness"]
    description = "Changes colour sharpness of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "Enhance", ["Sharpness", value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class HueShift(Command):
    name = ["Hue"]
    description = "Changes colour hue of supplied image."
    usage = "<0:url> <1:adjustment(0.5)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, default=0.5)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "hue_shift", [value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Blur(Command):
    name = ["Gaussian", "GaussianBlur"]
    description = "Applies Gaussian Blur to supplied image."
    usage = "<0:url> <1:radius(8)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, default=8)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "blur", ["gaussian", value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class ColourDeficiency(Command):
    name = ["ColorBlind", "ColourBlind", "ColorBlindness", "ColourBlindness", "ColorDeficiency"]
    alias = name + ["Protanopia", "Protanomaly", "Deuteranopia", "Deuteranomaly", "Tritanopia", "Tritanomaly", "Achromatopsia", "Achromatonomaly"]
    description = "Applies a colourblindness filter to the target image."
    usage = "<0:url> (protanopia|protanomaly|deuteranopia|deuteranomaly|tritanopia|tritanomaly|achromatopsia|achromatonomaly)? <1:ratio(0.9)>?"
    no_parse = True
    rate_limit = (3, 7)
    _timeout_ = 3.5
    typing = True

    async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        try:
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
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        if "color" not in name and "colour" not in name:
            operation = name
        elif args:
            operation = args.pop(0).casefold()
        else:
            operation = "deuteranomaly"
        value = " ".join(args).strip()
        if not value:
            value = None
        else:
            value = await bot.eval_math(value)
            if not abs(value) <= 2:
                raise OverflowError("Maximum multiplier input is 2.")
        # Try and find a good name for the output image
        try:
            name = url[url.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        ext = "png"
        if not name.endswith("." + ext):
            name += "." + ext
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "colour_deficiency", [operation, value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class RemoveMatte(Command):
    name = ["RemoveColor", "RemoveColour"]
    description = "Removes a colour from the supplied image."
    usage = "<0:url> <colour(#FFFFFF)>?"
    no_parse = True
    rate_limit = (4, 9)
    _timeout_ = 4.5
    typing = True

    async def __call__(self, bot, user, channel, message, name, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        try:
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
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        colour = parse_colour(" ".join(args), default=(255,) * 3)
        # Try and find a good name for the output image
        try:
            name = url[url.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        ext = "png"
        if not name.endswith("." + ext):
            name += "." + ext
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "remove_matte", [colour], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Invert(Command):
    name = ["Negate"]
    description = "Inverts supplied image."
    usage = "<url>"
    no_parse = True
    rate_limit = (2, 4.5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "invert", [], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class GreyScale(Command):
    name = ["GrayScale"]
    description = "Greyscales supplied image."
    usage = "<url>"
    no_parse = True
    rate_limit = (2, 4.5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "greyscale", [], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Magik(Command):
    description = "Applies the Magik image filter to supplied image."
    usage = "<0:url> <cell_size(7)>?"
    no_parse = True
    rate_limit = (3, 7)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, default=7)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik", [value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Colour(Command):
    name = ["RGB", "HSV", "CMY", "LAB", "LUV", "XYZ", "Color"]
    description = "Creates a 128x128 image filled with the target colour."
    usage = "<colour>"
    no_parse = True
    rate_limit = (1, 3)
    flags = "v"
    trans = {
        "hsv": hsv_to_rgb,
        "cmy": cmy_to_rgb,
        "lab": lab_to_rgb,
        "luv": luv_to_rgb,
        "xyz": xyz_to_rgb,
    }
    typing = True
    slash = True

    async def __call__(self, bot, user, channel, name, argv, **void):
        channels = parse_colour(argv)
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
            resp = await process_image("from_colour", "$", [channels])
            fn = resp[0]
            f = CompatFile(fn, filename="colour.png")
        await bot.send_with_file(channel, msg, f, filename=fn, best=True)


class Average(Command):
    name = ["AverageColour"]
    description = "Computes the average pixel colour in RGB for the supplied image."
    usage = "<url>"
    no_parse = True
    rate_limit = (2, 6)
    _timeout_ = 2
    typing = True

    async def __call__(self, bot, channel, user, message, argv, args, **void):
        if message.attachments:
            args = [worst_url(a) for a in message.attachments] + args
            argv = " ".join(worst_url(a) for a in message.attachments) + " " * bool(argv) + argv
        try:
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
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            colour = await bot.data.colours.get(url, threshold=False)
            channels = raw2colour(colour)
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
            resp = await process_image("from_colour", "$", [channels])
            fn = resp[0]
            f = CompatFile(fn, filename="colour.png")
        await bot.send_with_file(channel, msg, f, filename=fn, best=True)
        # return css_md("#" + bytes2hex(bytes(raw2colour(colour)), space=False))


class Rainbow(Command):
    name = ["RainbowGIF"]
    description = "Creates a .gif image from repeatedly hueshifting supplied image."
    usage = "<0:url> <1:duration(2)>?"
    no_parse = True
    rate_limit = (5, 12)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "rainbow_gif", [value, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Scroll(Command):
    name = ["Parallax", "Roll", "Offset", "ScrollGIF"]
    description = "Creates a .gif image from repeatedly shifting supplied image in a specified direction."
    usage = "<0:url> <1:direction(left)>? <2:duration(2)>? <3:fps(25)>?"
    no_parse = True
    rate_limit = (5, 11)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        try:
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
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        if args:
            direction = args.pop(0)
        else:
            direction = "LEFT"
        if args:
            duration = await bot.eval_math(args.pop(0))
        else:
            duration = 2
        if args:
            fps = await bot.eval_math(" ".join(args))
            fps = round(fps)
            if fps <= 0:
                raise ValueError("FPS value must be positive.")
            elif fps > 64:
                raise OverflowError("Maximum FPS value is 64.")
        else:
            fps = 25
        try:
            name = url[url.rindex("/") + 1:]
            if not name:
                raise ValueError
            if "." in name:
                name = name[:name.rindex(".")]
        except ValueError:
            name = "unknown"
        if not name.endswith(".gif"):
            name += ".gif"
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "scroll_gif", [direction, duration, fps, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Spin(Command):
    name = ["SpinGIF"]
    description = "Creates a .gif image from repeatedly rotating supplied image."
    usage = "<0:url> <1:duration(2)>?"
    no_parse = True
    rate_limit = (5, 11)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "spin_gif", [value, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class GMagik(Command):
    name = ["MagikGIF"]
    description = "Repeatedly applies the Magik image filter to supplied image."
    usage = "<0:url> <cell_size(7)>?"
    no_parse = True
    rate_limit = (7, 13)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik_gif", [abs(value), max(1, round(160 / abs(value))), "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Liquefy(Command):
    name = ["LiquidGIF"]
    description = "Repeatedly applies slight distortion to supplied image."
    usage = "<0:url> <cell_size(12)>?"
    no_parse = True
    rate_limit = (7, 14)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik_gif", [abs(value), 2, 2, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class CreateGIF(Command):
    name = ["Animate", "GIF"]
    description = "Combines multiple supplied images, and/or optionally a video, into an animated .gif image."
    usage = "<0:url>+ <-2:framerate_setting{?r}>? <-1:framerate(16)>?"
    no_parse = True
    rate_limit = (8, 24)
    _timeout_ = 10
    flags = "r"
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args += [best_url(a) for a in message.attachments]
        try:
            if not args:
                raise ArgumentError("Please input images by URL or attachment.")
        except ArgumentError:
            if not args:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        if "r" in flags:
            fr = args.pop(-1)
            rate = await bot.eval_math(fr)
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
                resp = await process_image("create_gif", "$", ["video", video, delay], timeout=_timeout)
            else:
                resp = await process_image("create_gif", "$", ["image", args, delay], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Resize(Command):
    name = ["ImageScale", "Scale", "Rescale", "ImageResize"]
    description = "Changes size of supplied image, using an optional scaling operation."
    usage = "<0:url> <1:x_multiplier(0.5)>? <2:y_multiplier(x)>? (nearest|linear|hamming|bicubic|lanczos|auto)?"
    no_parse = True
    rate_limit = (3, 6)
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args or argv == "list":
            if "l" in flags or argv == "list":
                return ini_md("Available scaling operations: [nearest, linear, hamming, bicubic, lanczos, auto]")
            # raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            try:
                url = args.pop(0)
                urls = await bot.follow_url(url, best=True, allow=True, limit=1)
                if not urls:
                    urls = await bot.follow_to_image(argv)
                    if not urls:
                        urls = await bot.follow_to_image(url)
                        if not urls:
                            raise ArgumentError("Please input an image by URL or attachment.")
                url = urls[0]
            except ArgumentError:
                if not argv:
                    url = None
                    try:
                        url = await bot.get_last_image(message.channel)
                    except FileNotFoundError:
                        raise ArgumentError("Please input an image by URL or attachment.")
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
                x = await bot.eval_math(spl.pop(0))
                if spl:
                    y = await bot.eval_math(spl.pop(0))
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
            resp = await process_image(url, "resize_mult", [x, y, op], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Fill(Command):
    name = ["ImageFill", "FillChannel", "FillImage"]
    description = "Fills an optional amount of channels in the target image with an optional value."
    usage = "<0:url> [rgbcmyhsva]* <-1:value(0)>?"
    no_parse = True
    rate_limit = (3, 6)
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        try:
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
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
        with discord.context_managers.Typing(channel):
            if is_numeric(args[-1]):
                value = await bot.eval_math(args.pop(-1))
                if type(value) is not int:
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
            resp = await process_image(url, "fill_channels", [value, *args], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Blend(Command):
    name = ["ImageBlend", "ImageOP"]
    description = "Combines the two supplied images, using an optional blend operation."
    usage = "<0:url1> <1:url2> (replace|add|sub|mul|div|mod|and|or|xor|nand|nor|xnor|difference|overlay|screen|soft|hard|lighten|darken|plusdarken|overflow|burn|linearburn|dodge|hue|sat|lum|colour|extract|merge)? <3:opacity(0.5|1)>?"
    no_parse = True
    rate_limit = (3, 8)
    flags = "l"
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args or argv == "list":
            if "l" in flags or argv == "list":
                return ini_md(
                    "Available blend operations: ["
                    + "replace, add, sub, mul, div, mod, and, or, xor, nand, nor, xnor, "
                    + "difference, overlay, screen, soft, hard, lighten, darken, plusdarken, overflow, "
                    + "burn, linearburn, dodge, lineardodge, hue, sat, lum, colour, extract, merge]"
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
                    urls = await bot.follow_to_image(argv)
                    if not urls:
                        raise ArgumentError("Please input an image by URL or attachment.")
                if type(urls) not in (list, alist):
                    urls = alist(urls)
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
                    opacity = await bot.eval_math(spl.pop(-1))
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
            resp = await process_image(url1, "blend_op", [url2, operation, opacity], timeout=_timeout)
            print(resp)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(message.channel, "", fn, filename=name)


class Cat(ImagePool, Command):
    description = "Pulls a random image from thecatapi.com, api.alexflipnote.dev/cats, or cdn.nekos.life/meow, and embeds it. Be sure to check out ⟨WEBSERVER⟩/cats!"
    database = "cats"
    name = ["🐱", "Meow", "Kitty", "Kitten"]
    slash = True

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
            if x == 1 and alexflipnote_key:
                d = await Request("https://api.alexflipnote.dev/cats", headers={"Authorization": alexflipnote_key}, json=True, aio=True)
            else:
                d = await Request("https://api.thecatapi.com/v1/images/search", json=True, aio=True)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 and alexflipnote_key else "url"]
        return url


class Dog(ImagePool, Command):
    description = "Pulls a random image from images.dog.ceo, api.alexflipnote.dev/dogs, or cdn.nekos.life/woof, and embeds it. Be sure to check out ⟨WEBSERVER⟩/dogs!"
    database = "dogs"
    name = ["🐶", "Woof", "Doggy", "Doggo"]
    slash = True

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
            if x == 1 and alexflipnote_key:
                d = await Request("https://api.alexflipnote.dev/dogs", headers={"Authorization": alexflipnote_key}, json=True, aio=True)
            else:
                d = await Request("https://dog.ceo/api/breeds/image/random", json=True, aio=True)
            if type(d) is list:
                d = choice(d)
            url = d["file" if x == 1 and alexflipnote_key else "message"]
            if "\\" in url:
                url = url.replace("\\/", "/").replace("\\", "/")
            while "///" in url:
                url = url.replace("///", "//")
        return url


class Muffin(ImagePool, Command):
    name = ["🧁", "Muffins"]
    description = "Muffin time! What more is there to say? :D"
    database = "muffins"

    async def fetch_one(self):
        if xrand(3):
            s = await Request(f"https://www.gettyimages.co.uk/photos/muffin?page={random.randint(1, 100)}", decode=True, aio=True)
            url = "https://media.gettyimages.com/photos/"
            spl = s.split(url)[1:]
            imageset = {url + i.split('"', 1)[0].split("?", 1)[0] for i in spl}
        else:
            d = await Request(f"https://unsplash.com/napi/search/photos?query=muffin&per_page=20&page={random.randint(1, 19)}", json=True, aio=True)
            imageset = {result["urls"]["raw"] for result in d["results"]}
        return imageset


class XKCD(ImagePool, Command):
    description = "Pulls a random image from xkcd.com and embeds it."
    database = "xkcd"

    async def fetch_one(self):
        s = await Request("https://c.xkcd.com/random/comic", decode=True, aio=True)
        search = "Image URL (for hotlinking/embedding): "
        s = s[s.index(search) + len(search):]
        url = s[:s.index("<")].strip()
        return url


class UpdateImagePools(Database):
    name = "imagepools"
    loading = {}
    sem = Semaphore(8, 2, rate_limit=1)
    no_delete = True

    async def load_until(self, key, func, threshold):
        data = set_dict(self.data, key, alist())
        for i in range(threshold << 1):
            if len(data) > threshold:
                break
            with tracebacksuppressor:
                out = await func()
                if type(out) is str:
                    out = (out,)
                for url in out:
                    if url not in data:
                        if i & 1:
                            data.appendleft(url)
                        else:
                            data.append(url)
                        self.update(key)
        data.uniq(sorted=None)

    async def proc(self, key, func):
        async with self.sem:
            data = set_dict(self.data, key, alist())
            out = await func()
            if out not in data:
                data.add(out)
                self.update(key)
            return out

    async def get(self, key, func, threshold=1024):
        if not self.loading.get(key):
            self.loading[key] = True
            create_task(self.load_until(key, func, threshold))
        data = set_dict(self.data, key, alist())
        if len(data) < threshold >> 1 or len(data) < threshold and xrand(2):
            out = await func()
            if type(out) is not str:
                out = next(iter(out))
            if out not in data:
                data.add(out)
                self.update(key)
            return out
        if not self.sem.is_busy():
            create_task(self.proc(key, func))
        return choice(data)


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
                        words = text.split()
                    else:
                        words = full_prune(message.content)
                    if k in words:
                        emojis = following[k]
                        # Store position for each keyword found
                        reacting[words.index(k) / len(words)] = emojis
                # Reactions sorted by their order of appearance in the message
                for r in sorted(reacting):
                    for react in reacting[r]:
                        try:
                            await message.add_reaction(react)
                        except discord.HTTPException as ex:
                            if "10014" in repr(ex):
                                emojis.remove(react)