try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import nekos, rule34, pybooru

# This entire module is a mess, I honestly don't care for it much
e_loop = asyncio.new_event_loop()
asyncio.set_event_loop(e_loop)
rule34_sync = rule34.Sync()
booruSites = list(pybooru.resources.SITE_LIST.keys())
LOG = False


def pull_e621(argv, delay=5):
    try:
        v1, v2 = 1, 1
        items = argv.replace(" ", "%20").casefold()
        baseurl = "https://e621.net/post/index/"
        url = baseurl + "1/" + items
        s = Request(url, decode=True)
        with suppress(ValueError):
            ind = s.index('class="next_page" rel="next"')
            s = s[ind - 90:ind]
            d = s.split(" ")
            i = -1
            while True:
                if "</a>" in d[i]:
                    break
                i -= 1
            u = d[i][:-4]
            u = u[u.index(">") + 1:]
            v1 = xrand(1, int(u))

            url = baseurl + str(v1) + "/" + items
            s = Request(url, decode=True)

        with suppress(ValueError):
            limit = s.index('class="next_page" rel="next"')
            s = s[:limit]

        search = '<a href="/post/show/'
        sources = []
        with suppress(ValueError):
            while True:
                ind1 = s.index(search)
                s = s[ind1 + len(search):]
                ind2 = s.index('"')
                target = s[:ind2]
                with suppress(ValueError):
                    sources.append(int(target))
        x = None
        while not x:
            v2 = xrand(len(sources))
            x = sources[v2]
            found = False
            url = "https://e621.net/post/show/" + str(x)
            s = Request(url, decode=True)
            search = '<a href="https://static1.e621.net/data/'
            ind1 = s.index(search)
            s = s[ind1 + 9:]
            ind2 = s.index('"')
            s = s[:ind2]
            url = s
            if is_image(url) is not None:
                found = True
            if not found:
                x = None
        return [url, v1, v2 + 1]
    except:
        if LOG:
            print_exc()


def pull_booru(argv, delay=5):
    client = pybooru.Moebooru(choice(tuple(booruSites)))
    try:
        posts = client.post_list(tags=argv, random=True, limit=16)
        if not posts:
            raise EOFError
        i = xrand(len(posts))
        url = posts[i]["file_url"]
        return [url, 1, i + 1]
    except:
        if LOG:
            print_exc()


def pull_rule34_xxx(argv, delay=5):
    v1, v2 = 1, 1
    try:
        t = utc()
        while utc() - t < delay:
            with suppress(TimeoutError):
                sources = rule34_sync.getImages(
                    tags=argv,
                    fuzzy=True,
                    randomPID=True,
                    singlePage=True
                )
                break
        if sources:
            attempts = 0
            while attempts < 256:
                v2 = xrand(len(sources))
                url = sources[v2].file_url
                if is_image(url) is not None:
                    break
                attempts += 1
            if attempts >= 256:
                raise TimeoutError
            v1 = 1
            return [url, v1, v2 + 1]
        else:
            raise EOFError
    except:
        if LOG:
            print_exc()


def pull_rule34_paheal(argv, delay=5):
    try:
        v1, v2 = 1, 1
        items = argv.casefold().split(" ")
        if not len(argv.replace(" ", "")):
            tagsearch = [chr(i + 65) for i in range(26)]
        else:
            tagsearch = []
            for i in items:
                if i[0] not in tagsearch:
                    tagsearch.append(i[0])
        rx = xrand(len(tagsearch))
        baseurl = "https://rule34.paheal.net/tags/alphabetic?starts_with="
        url = baseurl + tagsearch[rx] + "&mincount=1"
        s = Request(url, decode=True)
        tags = s.split("href='/post/list/")[1:]
        valid = []
        for i in range(len(tags)):
            ind = tags[i].index("/")
            tags[i] = tags[i][:ind]
            t = tags[i].casefold()
            for a in items:
                if a in t or a[:-1] == t:
                    valid.append(tags[i])
        rx = xrand(len(valid))
        items = valid[rx]

        baseurl = "https://rule34.paheal.net/post/list/"
        try:
            s = Request(baseurl + items + "/1", decode=True)
        except ConnectionError:
            s = Request(baseurl + items.upper() + "/1", decode=True)
        with suppress(ValueError):
            ind = s.index('">Last</a><br>')
            s = s[ind - 5:ind]
            v1 = xrand(1, int(s.split("/")[-1]))
            url = url[:-1] + str(v1)

            s = Request(url, decode=True)
        with suppress(ValueError):
            limit = s.index("class=''>Images</h3><div class='blockbody'>")
            s = s[limit:]
            limit = s.index("</div></div></section>")
            s = s[:limit]

        search = 'href="'
        sources = []
        with suppress(ValueError):
            while True:
                ind1 = s.index(search)
                s = s[ind1 + len(search):]
                ind2 = s.index('"')
                target = s[:ind2]
                if not "." in target:
                    continue
                elif ".js" in target:
                    continue
                found = is_image(target) is not None
                if target[0] == "h" and found:
                    sources.append(target)
        v2 = xrand(len(sources))
        url = sources[v2]
        return [url, v1, v2 + 1]
    except:
        if LOG:
            print_exc()


async def searchRandomNSFW(argv, delay=10):
    funcs = [
        pull_booru,
        pull_rule34_paheal,
        pull_rule34_xxx,
        pull_e621,
    ]
    data = [create_future(f, argv, delay - 3) for f in funcs]
    out = deque()
    for fut in data:
        with tracebacksuppressor:
            temp = await fut
            if temp:
                out.append(temp)
    print(out)
    if not out:
        raise LookupError(f"No results for {argv}.")
    item = choice(out)
    return item


neko_tags = {
    "feet": True,
    "yuri": True,
    "trap": True,
    "futanari": True,
    "hololewd": True,
    "lewdkemo": True,
    "solog": True,
    "feetg": True,
    "cum": True,
    "erokemo": True,
    "les": True,
    "wallpaper": True,
    "lewdk": True,
    "ngif": False,
    "tickle": False,
    "lewd": True,
    "feed": False,
    "gecg": False,
    "eroyuri": True,
    "eron": True,
    "cum_jpg": True,
    "bj": True,
    "nsfw_neko_gif": True,
    "solo": True,
    "kemonomimi": True,
    "nsfw_avatar": True,
    "gasm": False,
    "poke": False,
    "anal": True,
    "slap": False,
    "hentai": True,
    "avatar": False,
    "erofeet": True,
    "holo": True,
    "keta": True,
    "blowjob": True,
    "pussy": True,
    "tits": True,
    "holoero": True,
    "lizard": False,
    "pussy_jpg": True,
    "pwankg": True,
    "classic": True,
    "kuni": True,
    "waifu": False,
    "pat": False,
    "8ball": False,
    "kiss": False,
    "femdom": True,
    "neko": False,
    "spank": True,
    "cuddle": False,
    "erok": True,
    "fox_girl": False,
    "boobs": True,
    "random_hentai_gif": True,
    "hug": False,
    "ero": True,
    "smug": False,
    "goose": False,
    "baka": False,
    "cat": False,
    "gif": False,
    "woof": False,
    "404": 2,
}


def is_nsfw(channel):
    try:
        return channel.is_nsfw()
    except AttributeError:
        return True


class Neko(Command):
    min_level = 0
    description = "Pulls a random image from nekos.life and embeds it."
    usage = "<tags[neko]> <random(?r)> <verbose(?v)> <list(?l)>"
    flags = "lrv"
    rate_limit = 0.5

    async def __call__(self, args, argv, flags, channel, **void):
        isNSFW = is_nsfw(channel)
        if "l" in flags:
            available = []
            text = "Available tags in **" + channel.name + "**:\n```ini\n"
            for key in neko_tags:
                if isNSFW or not neko_tags[key] == True:
                    available.append(key)
            text += str(sorted(available)) + "```"
            return text
        tagNSFW = False
        selected = []
        for tag in args:
            tag = tag.replace(",", "").casefold()
            if tag in neko_tags:
                if neko_tags.get(tag, 0) == True:
                    tagNSFW = True
                    if not isNSFW:
                        raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels.")
                selected.append(tag)
        for _ in loop(flags.get("r", 0)):
            possible = [i for i in neko_tags if neko_tags[i] <= isNSFW]
            selected.append(possible[xrand(len(possible))])
        if not selected:
            if not argv:
                url = nekos.img("neko")
            else:
                raise LookupError(f"Search tag {argv} not found. Use ?l for list.")
        else:
            v = xrand(len(selected))
            get = selected[v]
            if get == "gif":
                if tagNSFW:
                    url = nekos.img("nsfw_neko_gif")
                else:
                    url = nekos.img("ngif")
            elif get == "cat":
                url = nekos.cat()
            elif get == "404":
                url = nekos.img("smallboobs")
            else:
                url = nekos.img(get)
        if "v" in flags:
            return url
        self.bot.send_as_embeds(channel, image=url, colour=xrand(1536))


class Lewd(Command):
    time_consuming = True
    _timeout_ = 2
    name = ["nsfw"]
    min_level = 1
    description = "Pulls a random image from a search on Rule34 and e621, and embeds it."
    usage = "<query> <verbose(?v)>"
    flags = "v"
    no_parse = True
    rate_limit = 1

    async def __call__(self, args, flags, channel, **void):
        if not is_nsfw(channel):
            raise PermissionError(f"This command is only available in {uni_str('NSFW')} channels.")
        objs = await searchRandomNSFW(" ".join(args), 12)
        url = objs[0]
        if "v" in flags:
            text = (
                "Pulled from " + url
                + "\nImage **__" + str(objs[2])
                + "__** on page **__" + str(objs[1])
                + "__**"
            )
            return text
        self.bot.send_as_embeds(channel, image=url, colour=xrand(1536))