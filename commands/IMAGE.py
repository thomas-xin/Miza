try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *

print = PRINT

import youtube_dl
# youtube_dl = youtube_dlc

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

def get_video(url, fps=None):
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
    usage = "(add|delete)? <0:tags>* <1:url>? <verbose{?v}|delete{?x}|hide{?h}>?"
    flags = "vraedhzfx"
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
                    return css_md(sqr_md(f"WARNING: {len(images)} IMAGES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True)
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
        if "x" in flags:
            create_task(bot.silent_delete(message))
        if "v" in flags:
            msg = escape_roles(url)
        else:
            msg = None
        url2 = await bot.get_proxy_url(message.author)
        colour = await create_future(bot.get_colour, message.author)
        emb = discord.Embed(colour=colour, url=url).set_image(url=url)
        await bot.send_as_webhook(channel, msg, embed=emb, username=message.author.display_name, avatar_url=url2)

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
            msg = ini_md(iter2str({k: "\n" + images[k] for k in tuple(images)[pos:pos + page]}))
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(images) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
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
                            raise ArgumentError
                url = urls[0]
            except ArgumentError:
                if not argv:
                    url = None
                    try:
                        url = await bot.get_last_image(message.channel)
                    except FileNotFoundError:
                        raise ArgumentError("Please input an image by URL or attachment.")
                else:
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


class ScanEmoji(Command):
    name = ["EmojiScan", "ScanEmojis"]
    min_level = 1
    description = "Scans all the emojis in the current server for potential issues."
    no_parse = True
    rate_limit = (4, 7)
    _timeout_ = 4
    typing = True

    ffprobe_start = (
        "ffprobe",
        "-v",
        "error",
        "-hide_banner",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
    )

    async def __call__(self, bot, guild, channel, message, **void):
        # fut = create_task(send_with_reply(channel, message, "Emoji scan initiated. Delete the original message at any point in time to cancel."))
        p = bot.get_prefix(guild)
        found = 0
        with discord.context_managers.Typing(channel):
            for emoji in sorted(guild.emojis, key=lambda e: e.id):
                url = str(emoji.url)
                resp = await create_future(subprocess.run, self.ffprobe_start + (url,), stdout=subprocess.PIPE)
                width, height = map(int, resp.stdout.splitlines())
                if width < 128 or height < 128:
                    found += 1
                    w, h = width, height
                    while w < 128 or h < 128:
                        w, h = w << 1, h << 1
                    colour = await create_future(bot.get_colour, url)
                    bot.send_as_embeds(
                        channel,
                        description=f"{emoji} is {width}×{height}, which is below the recommended discord emoji size, and may appear blurry when scaled by Discord. Scaling the image using {p}resize, with filters `nearest`, `scale2x` or `lanczos` is advised.",
                        fields=(("Example", f"{p}resize {emoji} {w}×{h} scale2x"),),
                        title=f"⚠ Issue {found} ⚠",
                        colour=discord.Colour(colour),
                    )
        if not found:
            return css_md(f"No emoji issues found for {guild}.")


async def get_image(bot, user, message, args, argv, default=2, raw=False, ext="png"):
    try:
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args:
            raise ArgumentError
        url = args.pop(0)
        urls = await bot.follow_url(url, best=True, allow=True, limit=1)
        if not urls:
            urls = await bot.follow_to_image(argv)
            if not urls:
                urls = await bot.follow_to_image(url)
                if not urls:
                    raise ArgumentError
        url = urls[0]
    except ArgumentError:
        if not argv:
            url = None
            try:
                url = await bot.get_last_image(message.channel)
            except FileNotFoundError:
                raise ArgumentError("Please input an image by URL or attachment.")
        else:
            raise ArgumentError("Please input an image by URL or attachment.")
    value = " ".join(args).strip()
    if not value:
        value = default
    elif not raw:
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


class Saturation(Command):
    name = ["Saturate", "ImageSaturate"]
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Brightness(Command):
    name = ["Brighten", "Lighten", "Lightness", "ImageBrightness"]
    description = "Changes colour brightness of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "brightness", [value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Luminance(Command):
    name = ["Luminosity", "ImageLuminance"]
    description = "Changes colour luminance of supplied image."
    usage = "<0:url> <1:multiplier(2)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "luminance", [value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
                raise ArgumentError
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError
            url = urls[0]
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
                raise ArgumentError
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError
            url = urls[0]
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Laplacian(Command):
    name = ["EdgeDetect", "Edges"]
    description = "Applies the Laplacian edge-detect algorithm to the image."
    usage = "<url>"
    no_parse = True
    rate_limit = (2, 4.5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "laplacian", [], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class ColourSpace(Command):
    name = ["ColorSpace"]
    description = "Changes the colour space of the supplied image."
    usage = "<url> <2:source(rgb)>? <1:dest(hsv)>?"
    no_parse = True
    rate_limit = (3, 6.5)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, raw=True, default="")
        spl = value.rsplit(None, 1)
        if not spl:
            source = "rgb"
            dest = "hsv"
        elif len(spl) == 1:
            source = "rgb"
            dest = spl[0].casefold()
        else:
            source, dest = (i.casefold() for i in spl)
        if source == dest:
            raise TypeError("Colour spaces must be different.")
        for i in (source, dest):
            if i not in ("rgb", "cmy", "hsv", "hsl", "hsi", "hcl", "lab", "luv", "xyz"):
                raise TypeError(f"Invalid colour space {i}.")
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "colourspace", [source, dest], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Magik(Command):
    name = ["Distort"]
    description = "Applies the Magik image filter to supplied image."
    usage = "<0:url> <cell_count(7)>?"
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Colour(Command):
    name = ["RGB", "HSV", "HSL", "CMY", "LAB", "LUV", "XYZ", "Color"]
    description = "Creates a 128x128 image filled with the target colour."
    usage = "<colour>"
    no_parse = True
    rate_limit = (1, 3)
    flags = "v"
    trans = {
        "hsv": hsv_to_rgb,
        "hsl": hsl_to_rgb,
        "cmy": cmy_to_rgb,
        "lab": lab_to_rgb,
        "luv": luv_to_rgb,
        "xyz": xyz_to_rgb,
    }
    typing = True
    slash = True

    async def __call__(self, bot, user, message, channel, name, argv, **void):
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
            + "\nRGB values: " + str(channels if type(channels) is list else list(channels))
            + "\nHSV values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsv(adj)))
            + "\nHSL values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsl(adj)))
            + "\nCMY values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
            + "\nLAB values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
            + "\nLUV values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
            + "\nXYZ values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
        )
        with discord.context_managers.Typing(channel):
            resp = await process_image("from_colour", "$", [channels])
            fn = resp[0]
            f = CompatFile(fn, filename="colour.png")
        await bot.send_with_file(channel, msg, f, filename=fn, best=True, reference=message)


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
                raise ArgumentError
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError
            url = urls[0]
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
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
                + "\nHSL values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_hsl(adj)))
                + "\nCMY values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_cmy(adj)))
                + "\nLAB values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_lab(adj)))
                + "\nLUV values: " + sqr_md(", ".join(str(round(x)) for x in rgb_to_luv(adj)))
                + "\nXYZ values: " + sqr_md(", ".join(str(round(x * 255)) for x in rgb_to_xyz(adj)))
            )
            resp = await process_image("from_colour", "$", [channels])
            fn = resp[0]
            f = CompatFile(fn, filename="colour.png")
        await bot.send_with_file(channel, msg, f, filename=fn, best=True, reference=message)
        # return css_md("#" + bytes2hex(bytes(raw2colour(colour)), space=False))


class QR(Command):
    name = ["RainbowQR"]
    description = "Creates a QR code image from an input string, optionally adding a rainbow swirl effect."
    usage = "<string>"
    no_parse = True
    rate_limit = (3, 7)
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, message, channel, argv, name, _timeout, **void):
        if not argv:
            raise ArgumentError("Input string is empty.")
        with discord.context_managers.Typing(channel):
            resp = await process_image("to_qr", "$", [argv, "rainbow" in name], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(channel, "", fn, filename="QR." + ("gif" if "rainbow" in name else "png"), reference=message)


class Rainbow(Command):
    name = ["RainbowGIF"]
    description = "Creates a .gif image from repeatedly hueshifting supplied image."
    usage = "<0:url> <1:duration(2)>?"
    no_parse = True
    rate_limit = (5, 12)
    _timeout_ = 8
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "rainbow_gif", [value, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Scroll(Command):
    name = ["Parallax", "Offset", "ScrollGIF"]
    description = "Creates a .gif image from repeatedly shifting supplied image in a specified direction."
    usage = "<0:url> <1:direction(left)>? <2:duration(2)>? <3:fps(25)>?"
    no_parse = True
    rate_limit = (5, 11)
    _timeout_ = 8
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        try:
            if message.attachments:
                args = [best_url(a) for a in message.attachments] + args
                argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
            if not args:
                raise ArgumentError
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError
            url = urls[0]
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Spin(Command):
    name = ["SpinGIF"]
    description = "Creates a .gif image from repeatedly rotating supplied image."
    usage = "<0:url> <1:duration(2)>?"
    no_parse = True
    rate_limit = (5, 11)
    _timeout_ = 8
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            # -gif signals to image subprocess that the output is always a .gif image
            resp = await process_image(url, "spin_gif", [value, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class GMagik(Command):
    name = ["MagikGIF"]
    description = "Repeatedly applies the Magik image filter to supplied image."
    usage = "<0:url> <cell_size(7)>?"
    no_parse = True
    rate_limit = (7, 13)
    _timeout_ = 8
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, ext="gif")
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik_gif", [abs(value), "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Liquefy(Command):
    name = ["LiquidGIF"]
    description = "Repeatedly applies slight distortion to supplied image."
    usage = "<0:url> <cell_count(32)>?"
    no_parse = True
    rate_limit = (7, 14)
    _timeout_ = 8
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, default=32, ext="gif")
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "magik_gif", [abs(value), 2, "-gif"], timeout=_timeout)
            fn = resp[0]
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class CreateGIF(Command):
    name = ["Animate", "GIF"]
    description = "Combines multiple supplied images, and/or optionally a video, into an animated .gif image."
    usage = "<0:url>+ <-2:framerate_setting{?r}>? <-1:framerate(16)>?"
    no_parse = True
    rate_limit = (8, 24)
    _timeout_ = 20
    flags = "r"
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args += [best_url(a) for a in message.attachments]
        try:
            if not args:
                raise ArgumentError
        except ArgumentError:
            if not args:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
                raise ArgumentError("Please input an image by URL or attachment.")
        if "r" in flags:
            fr = args.pop(-1)
            rate = await bot.eval_math(fr)
        else:
            rate = None
        # Validate framerate values to prevent issues further down the line
        if rate and rate <= 0:
            args = args[:1]
            rate = 1
        delay = round(1000 / rate) if rate else None
        if delay and delay <= 0:
            args = args[-1:]
            delay = 1000
        elif delay and delay >= 16777216:
            raise OverflowError("GIF image framerate too low.")
        with discord.context_managers.Typing(channel):
            video = None
            for i, url in enumerate(args):
                urls = await bot.follow_url(url, best=True, allow=True, limit=1)
                url = urls[0]
                if "discord" not in url and "channels" not in url:
                    with tracebacksuppressor:
                        url, size, dur, fps = await create_future(get_video, url, None, timeout=60)
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Resize(Command):
    name = ["ImageScale", "Scale", "Rescale", "ImageResize"]
    description = "Changes size of supplied image, using an optional scaling operation."
    usage = "<0:url> <1:x_multiplier(1)>? <2:y_multiplier(x)>? (nearest|linear|hamming|bicubic|lanczos|scale2x|auto)?"
    no_parse = True
    rate_limit = (3, 6)
    flags = "l"
    _timeout_ = 4
    typing = True

    async def __call__(self, bot, user, guild, channel, message, flags, args, argv, _timeout, **void):
        # Take input from any attachments, or otherwise the message contents
        if message.attachments:
            args = [best_url(a) for a in message.attachments] + args
            argv = " ".join(best_url(a) for a in message.attachments) + " " * bool(argv) + argv
        if not args or argv == "list":
            if "l" in flags or argv == "list":
                return ini_md("Available scaling operations: [nearest, linear, hamming, bicubic, lanczos, scale2x, auto]")
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
                            raise ArgumentError
                url = urls[0]
            except ArgumentError:
                if not argv:
                    url = None
                    try:
                        url = await bot.get_last_image(message.channel)
                    except FileNotFoundError:
                        raise ArgumentError("Please input an image by URL or attachment.")
                else:
                    raise ArgumentError("Please input an image by URL or attachment.")
            value = " ".join(args).strip()
            func = "resize_mult"
            if not value:
                x = y = 1
                op = "auto"
            else:
                # Parse width and height multipliers
                if "x" in value[:-1] or "X" in value or "*" in value or "×" in value:
                    func = "resize_to"
                    value = value.replace("x", " ").replace("X", " ").replace("*", " ").replace("×", " ")
                else:
                    value = value.replace(":", " ")
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
                    if op == "scale2":
                        op = "scale2x"
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
            resp = await process_image(url, func, [x, y, op], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Rotate(Command):
    name = ["Orientate", "Orientation", "Transpose"]
    description = "Rotates an image."
    usage = "<0:url> <1:angle(90)>?"
    no_parse = True
    rate_limit = (2, 5)
    _timeout_ = 3
    typing = True

    async def __call__(self, bot, user, channel, message, args, argv, _timeout, **void):
        name, value, url = await get_image(bot, user, message, args, argv, default=90, raw=True)
        value = await bot.eval_math(value)
        with discord.context_managers.Typing(channel):
            resp = await process_image(url, "rotate_to", [value], timeout=_timeout)
            fn = resp[0]
            if fn.endswith(".gif"):
                if not name.endswith(".gif"):
                    if "." in name:
                        name = name[:name.rindex(".")]
                    name += ".gif"
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


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
                raise ArgumentError
            url = args.pop(0)
            urls = await bot.follow_url(url, best=True, allow=True, limit=1)
            if not urls:
                urls = await bot.follow_to_image(argv)
                if not urls:
                    urls = await bot.follow_to_image(url)
                    if not urls:
                        raise ArgumentError
            url = urls[0]
        except ArgumentError:
            if not argv:
                url = None
                try:
                    url = await bot.get_last_image(message.channel)
                except FileNotFoundError:
                    raise ArgumentError("Please input an image by URL or attachment.")
            else:
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Blend(Command):
    name = ["ImageBlend", "ImageOP"]
    description = "Combines the two supplied images, using an optional blend operation."
    usage = "<0:url1> <1:url2> (replace|add|sub|mul|div|mod|and|or|xor|nand|nor|xnor|difference|overlay|screen|soft|hard|lighten|darken|plusdarken|overflow|lighting|burn|linearburn|dodge|hue|sat|lum|colour|extract|merge)? <3:opacity(0.5|1)>?"
    no_parse = True
    rate_limit = (3, 8)
    flags = "l"
    _timeout_ = 7
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
                    + "difference, overlay, screen, soft, hard, lighten, darken, plusdarken, overflow, lighting, "
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
        await bot.send_with_file(channel, "", fn, filename=name, reference=message)


class Waifu2x(Command):
    description = "Resizes the target image using the popular Waifu2x AI algorithm."
    usage = "<url> <api{?a}>"
    no_parse = True
    rate_limit = (5, 10)
    flags = "l"
    _timeout_ = 5
    typing = True

    async def __call__(self, bot, user, message, channel, args, argv, flags, **void):
        name, value, url = await get_image(bot, user, message, args, argv, raw=True, default="")
        if "a" not in flags:
            return self.bot.webserver + "/waifu2x?source=" + url
        with discord.context_managers.Typing(channel):
            mime = await create_future(bot.detect_mime, url)
            image = None
            if "image/png" not in mime:
                if "image/jpg" not in mime:
                    if "image/jpeg" not in mime:
                        resp = await process_image(url, "resize_mult", ["-nogif", 1, 1, "auto"], timeout=60)
                        with open(resp[0], "rb") as f:
                            image = await create_future(f.read)
                        ext = "png"
                    else:
                        ext = "jpeg"
                else:
                    ext = "jpg"
            else:
                ext = "png"
            if not image:
                image = await Request(url, timeout=20, aio=True)
            data = await create_future(
                Request,
                "https://api.alcaamado.es/api/v1/waifu2x/convert",
                files={
                    "denoise": (None, "1"),
                    "scale": (None, "true"),
                    "file": (f"file.{ext}", image),
                },
                _timeout_=22,
                method="post",
                json=True,
            )
            for i in range(60):
                async with delay(0.75):
                    img = await Request(
                        f"https://api.alcaamado.es/api/v1/waifu2x/get?hash={data['hash']}",
                        headers=dict(Accept="application/json, text/plain, */*"),
                        timeout=60,
                        json=True,
                        aio=True,
                    )
                    if img.get("image"):
                        break
            if not img.get("image"):
                raise FileNotFoundError("image file not found")
            image = await create_future(base64.b64decode, img["image"])
        await bot.send_with_file(channel, "", file=image, filename=name, reference=message)


class UpdateImages(Database):
    name = "images"