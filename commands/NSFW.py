import nekos, rule34, discord
from smath import *

image_forms = [
    ".gif",
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tiff",
]


def pull_e621(argv, data, thr, delay=5):
    try:
        v1, v2 = 1, 1
        opener = urlBypass()
        items = argv.replace(" ", "%20").lower()
        baseurl = "https://e621.net/post/index/"
        url = baseurl + "1/" + items
        resp = opener.open(url)
        if resp.getcode() != 200:
            raise ConnectionError("Error " + str(resp.getcode()))
        s = resp.read().decode("utf-8")

        try:
            ind = s.index('class="next_page" rel="next"')
            s = s[ind - 90 : ind]
            d = s.split(" ")
            i = -1
            while True:
                if "</a>" in d[i]:
                    break
                i -= 1
            u = d[i][:-4]
            u = u[u.index(">") + 1 :]
            v1 = xrand(1, int(u))

            url = baseurl + str(v1) + "/" + items
            resp = opener.open(url)
            if resp.getcode() != 200:
                raise ConnectionError("Error " + str(resp.getcode()))
            s = resp.read().decode("utf-8")
        except ValueError:
            pass

        try:
            limit = s.index('class="next_page" rel="next"')
            s = s[:limit]
        except ValueError:
            pass

        search = '<a href="/post/show/'
        sources = []
        while True:
            try:
                ind1 = s.index(search)
                s = s[ind1 + len(search) :]
                ind2 = s.index('"')
                target = s[:ind2]
                try:
                    sources.append(int(target))
                except ValueError:
                    pass
            except ValueError:
                break
        x = None
        while not x:
            v2 = xrand(len(sources))
            x = sources[v2]
            found = False
            url = "https://e621.net/post/show/" + str(x)
            resp = opener.open(url)
            if resp.getcode() != 200:
                raise ConnectionError("Error " + str(resp.getcode()))
            s = resp.read().decode("utf-8")
            search = '<a href="https://static1.e621.net/data/'
            ind1 = s.index(search)
            s = s[ind1 + 9 :]
            ind2 = s.index('"')
            s = s[:ind2]
            url = s
            for i in image_forms:
                if i in url:
                    found = True
            if not found:
                x = None
        data[thr] = [url, v1, v2 + 1]
    except:
        data[thr] = 0
    print(data)


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
rule34_sync = rule34.Sync()


def pull_rule34_xxx(argv, data, thr, delay=5):
    v1, v2 = 1, 1
    try:
        t = time.time()
        while time.time() - t < delay:
            try:
                sources = rule34_sync.getImages(
                    tags=argv,
                    fuzzy=True,
                    randomPID=True,
                    singlePage=True
                )
                break
            except TimeoutError:
                pass
        if sources:
            attempts = 0
            while attempts < 1000:
                v2 = xrand(len(sources))
                url = sources[v2].file_url
                found = False
                for i in image_forms:
                    if i in url:
                        found = True
                        break
                if found:
                    break
                attempts += 1
            if attempts >= 1000:
                raise
            v1 = 1
            data[thr] = [url, v1, v2 + 1]
        else:
            raise
    except:
        data[thr] = 0
    print(data)


def pull_rule34_paheal(argv, data, thr, delay=5):
    try:
        v1, v2 = 1, 1
        items = argv.lower().split(" ")
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
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=delay)
        if resp.getcode() != 200:
            raise ConnectionError("Error " + str(resp.getcode()))
        s = resp.read().decode("utf-8")
        tags = s.split("href='/post/list/")[1:]
        valid = []
        for i in range(len(tags)):
            ind = tags[i].index("/")
            tags[i] = tags[i][:ind]
            t = tags[i].lower()
            for a in items:
                if a in t or a[:-1] == t:
                    valid.append(tags[i])
        rx = xrand(len(valid))
        items = valid[rx]

        baseurl = "https://rule34.paheal.net/post/list/"
        try:
            url = baseurl + items + "/1"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=delay)
        except ConnectionError:
            url = baseurl + items.upper() + "/1"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=delay)
        if resp.getcode() != 200:
            raise ConnectionError("Error " + str(resp.getcode()))

        s = resp.read().decode("utf-8")
        try:
            ind = s.index('">Last</a><br>')
            s = s[ind - 5 : ind]
            v1 = xrand(1, int(s.split("/")[-1]))
            url = url[:-1] + str(v1)

            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=delay)
            if resp.getcode() != 200:
                raise ConnectionError("Error " + str(resp.getcode()))
            s = resp.read().decode("utf-8")
        except ValueError:
            pass
        try:
            limit = s.index("class=''>Images</h3><div class='blockbody'>")
            s = s[limit:]
            limit = s.index("</div></div></section>")
            s = s[:limit]
        except ValueError:
            pass

        search = 'href="'
        sources = []
        while True:
            try:
                ind1 = s.index(search)
                s = s[ind1 + len(search) :]
                ind2 = s.index('"')
                target = s[:ind2]
                if not "." in target:
                    continue
                elif ".js" in target:
                    continue
                found = False
                for i in image_forms:
                    if i in target:
                        found = True
                if target[0] == "h" and found:
                    sources.append(target)
            except ValueError:
                break
        v2 = xrand(len(sources))
        url = sources[v2]
        data[thr] = [url, v1, v2 + 1]
    except:
        data[thr] = 0
    print(data)


async def searchRandomNSFW(argv, delay=9):
    t = time.time()
    funcs = [pull_e621, pull_rule34_paheal, pull_rule34_xxx]
    data = [None for i in funcs]
    for i in range(len(funcs)):
        doParallel(funcs[i], [argv, data, i, delay - 3])
    while None in data and time.time() - t < delay:
        await asyncio.sleep(0.6)
    data = [i for i in data if i]
    i = xrand(len(data))
    if not len(data) or data[i] is None:
        raise EOFError("Unable to locate any search results for " + uniStr(argv) + ".")
    return data[i]


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
    "404": 2,
}


def is_nsfw(channel):
    try:
        return channel.is_nsfw()
    except AttributeError:
        return True


class Neko:
    is_command = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Pulls a random image from nekos.life and embeds it."
        self.usage = "<tags[neko]> <random(?r)> <verbose(?v)> <list(?l)>"

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
            tag = tag.replace(",", "").lower()
            if tag in neko_tags:
                if neko_tags.get(tag, 0) == True:
                    tagNSFW = True
                    if not isNSFW:
                        raise PermissionError(
                            "This command is only available in " + uniStr("NSFW") + " channels."
                            )
                selected.append(tag)
        for x in range(flags.get("r", 0)):
            possible = [i for i in neko_tags if neko_tags[i] <= isNSFW]
            selected.append(possible[xrand(len(possible))])
        if not selected:
            if not argv:
                url = nekos.img("neko")
            else:
                raise EOFError(
                    "Search tag " + uniStr(argv) + " not found. Use ?l for list.")
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
            text = "Pulled from " + url
            return text
        emb = discord.Embed(
            url=url,
            colour=self._vars.randColour(),
        )
        emb.set_image(url=url)
        print(url)
        asyncio.create_task(channel.send(embed=emb))


class Lewd:
    is_command = True
    time_consuming = True

    def __init__(self):
        self.name = ["nsfw"]
        self.min_level = 1
        self.description = "Pulls a random image from a search on Rule34 and e621, and embeds it."
        self.usage = "<query> <verbose(?v)>"

    async def __call__(self, _vars, args, flags, channel, **void):
        if not is_nsfw(channel):
            raise PermissionError("This command is only available in NSFW channels.")
        objs = await searchRandomNSFW(" ".join(args), _vars.timeout-1)
        url = objs[0]
        if "v" in flags:
            text = (
                "Pulled from " + url
                + "\nImage **__" + str(objs[2])
                + "__** on page **__" + str(objs[1])
                + "__**"
            )
            return text
        emb = discord.Embed(
            url=url,
            colour=_vars.randColour(),
        )
        emb.set_image(url=url)
        print(url)
        asyncio.create_task(channel.send(embed=emb))
