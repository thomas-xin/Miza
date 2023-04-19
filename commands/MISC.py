print = PRINT

import csv, knackpy
from prettytable import PrettyTable as ptable
from tsc_utils.flags import address_to_flag, flag_to_address
from tsc_utils.numbers import tsc_value_to_num, num_to_tsc_value


class DouClub:

    def __init__(self, c_id, c_sec):
        self.id = c_id
        self.secret = c_sec
        self.time = utc()
        self.knack = knackpy.App(app_id=self.id, api_key=self.secret)
        create_future_ex(self.pull)

    def pull(self):
        with tracebacksuppressor:
            # print("Pulling Doukutsu Club...")
            self.data = self.knack.get("object_1")
            self.time = utc()

    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull, timeout=60)
            self.time = utc()

    def search(self, query):
        # This string search algorithm could be better
        output = []
        query = query.casefold()
        qlist = set(query.split())
        for res in self.data:
            author = res["Author"][0]["identifier"]
            name = res["Title"]
            description = res["Description"]
            s = (name, description, author)
            if not all(any(q in v for v in s) for q in qlist):
                continue
            url = f"https://doukutsuclub.knack.com/database#search-database/mod-details/{res['id']}"
            output.append({
                "author": author,
                "name": name,
                "description": description,
                "url": url,
            })
        return output

douclub = None


async def searchForums(query):
    url = f"https://forum.cavestory.org/search/320966/?q={url_parse(query)}"
    s = await Request(url, aio=True, timeout=16, ssl=False, decode=True)
    output = []
    i = 0
    while i < len(s):
        # HTML is a mess
        try:
            search = '<li class="block-row block-row--separated  js-inlineModContainer" data-author="'
            s = s[s.index(search) + len(search):]
        except ValueError:
            break
        j = s.index('">')
        curr = {"author": s[:j]}
        s = s[s.index('<h3 class="contentRow-title">'):]
        search = '<a href="/'
        s = s[s.index(search) + len(search):]
        j = s.index('">')
        curr["url"] = 'https://www.cavestory.org/forums/' + s[:j].lstrip("/")
        s = s[j + 2:]
        j = s.index('</a>')
        curr["name"] = s[:j]
        search = '<div class="contentRow-snippet">'
        s = s[s.index(search) + len(search):]
        j = s.index('</div>')
        curr["description"] = s[:j]
        for elem in curr:
            temp = curr[elem].replace('<em class="textHighlight">', "**").replace("</em>", "**")
            temp = html_decode(temp)
            curr[elem] = temp
        output.append(curr)
    return output


class SheetPull:

    def __init__(self, url):
        self.url = url
        self.time = utc()
        create_future_ex(self.pull)

    def update(self):
        if utc() - self.time > 720:
            create_future_ex(self.pull, timeout=60)
            self.time = utc()

    def pull(self):
        with tracebacksuppressor:
            # print("Pulling Spreadsheet...")
            url = self.url
            text = Request(url, timeout=32, decode=True)
            data = text.split("\r\n")
            columns = 0
            sdata = [[], utc()]
            # Splits rows and colums into cells
            for i in range(len(data)):
                line = data[i]
                read = list(csv.reader(line))
                reli = []
                curr = ""
                for j in read:
                    if len(j) >= 2 and j[0] == j[1] == "":
                        if curr != "":
                            reli.append(curr)
                            curr = ""
                    else:
                        curr += "".join(j)
                if curr != "":
                    reli.append(curr)
                if len(reli):
                    columns = max(columns, len(reli))
                    sdata[0].append(reli)
                for line in range(len(sdata[0])):
                    while len(sdata[0][line]) < columns:
                        sdata[0][line].append(" ")
            self.data = sdata
            self.time = utc()

    def search(self, query, lim):
        output = []
        query = query.casefold()
        try:
            int(query)
            mode = 0
        except ValueError:
            mode = 1
        if not mode:
            for l in self.data[0]:
                if l[0] == query:
                    temp = [lim_line(e, lim) for e in l]
                    output.append(temp)
        else:
            qlist = set(query.split())
            for l in self.data[0]:
                if len(l) >= 3:
                    found = True
                    for q in qlist:
                        tag = False
                        for i in l:
                            if q in i.casefold():
                                tag = True
                                break
                        if not tag:
                            found = False
                            break
                    if found:
                        temp = [lim_line(e, lim) for e in l]
                        if temp[2].replace(" ", ""):
                            output.append(temp)
        return output


# URLs of Google Sheets .csv download links
entity_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0\
/export?format=csv&id=12iC9uRGNZ2MnrhpS4s_KvIRYHhC56mPXCnCcsDjxit0&gid=0"
)
tsc_list = SheetPull(
    "https://docs.google.com/spreadsheets/d/11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U\
/export?format=csv&id=11LL7T_jDPcWuhkJycsEoBGa9i-rjRjgMW04Gdz9EO6U&gid=0"
)


class CS_mem2flag(Command):
    name = ["mem2flag", "m2f", "CS_m2f"]
    description = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
    usage = "<0:address> <1:value(1)>?"
    example = ("cs_m2f 49e6e8 123",)
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        if len(args) < 2:
            num = 1
        else:
            num = await bot.eval_math(" ".join(args[1:]))
        return css_md("".join(address_to_flag(int(args[0], 16), num)))


class CS_flag2mem(Command):
    name = ["flag2mem", "f2m", "CS_f2m"]
    description = "Returns the memory offset and specific bit pointed to by a given flag number."
    usage = "<flag>"
    example = ("cs_f2m A036",)
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        flag = args[0]
        if len(flag) > 4:
            raise ValueError("Flag number should be no more than 4 characters long.")
        flag = flag.zfill(4)
        return css_md(str(flag_to_address(flag)))


class CS_num2val(Command):
    name = ["num2val", "n2v", "CS_n2v"]
    description = "Returns a TSC value representing the desired number, within a certain number of characters."
    usage = "<0:number> <1:length(4)>?"
    example = ("cs_n2v 12345",)
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        if len(args) < 2:
            length = 4
        else:
            length = await bot.eval_math(" ".join(args[1:]))
        return css_md(str(num_to_tsc_value(int(args[0], 0), length)))


class CS_val2num(Command):
    name = ["val2num", "v2n", "CS_v2n"]
    description = "Returns the number encoded by a given TSC value."
    usage = "<tsc_value>"
    example = ("cs_v2n CCCC",)
    rate_limit = 1

    async def __call__(self, bot, args, user, **void):
        return css_md(str(tsc_value_to_num(args[0])))


class CS_hex2xml(Command):
    time_consuming = True
    name = ["hex2xml", "h2x", "CS_h2x"]
    description = "Converts a given Cave Story hex patch to an xml file readable by Booster's Lab."
    usage = "<hex_data>"
    example = ("cs_h2x 0x481D27 C3",)
    rate_limit = (3, 5)

    async def __call__(self, bot, argv, channel, message, **void):
        hacks = {}
        hack = argv.replace(" ", "").replace("`", "").strip("\n")
        while len(hack):
            # hack XML parser
            try:
                i = hack.index("0x")
            except ValueError:
                break
            hack = hack[i:]
            i = hack.index("\n")
            offs = hack[:i]
            hack = hack[i + 1:]
            try:
                i = hack.index("0x")
                curr = hack[:i]
                hack = hack[i:]
            except ValueError:
                curr = hack
                hack = ""
            curr = curr.replace(" ", "").replace("\n", "").replace("\r", "")
            n = 2
            curr = " ".join([curr[i:i + n] for i in range(0, len(curr), n)])
            if offs in hacks:
                hacks[offs] = curr + hacks[offs][len(curr):]
            else:
                hacks[offs] = curr
        # Generate hack template
        output = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<hack name="HEX PATCH">\n'
            + '\t<panel>\n'
            + '\t\t<panel title="Description">\n'
            + '\t\t</panel>\n'
            + '\t\t<field type="info">\n'
            + '\t\t\tHex patch converted by ' + bot.user.name + '.\n'
            + '\t\t</field>\n'
            + '\t\t<panel title="Data">\n'
            + '\t\t</panel>\n'
            + '\t\t<panel>\n'
        )
        col = 0
        for hack in sorted(hacks):
            n = 63
            p = hacks[hack]
            p = '\n\t\t\t\t'.join([p[i:i + n] for i in range(0, len(p), n)])
            output += (
                '\t\t\t<field type="data" offset="' + hack + '" col="' + str(col) + '">\n'
                + '\t\t\t\t' + p + '\n'
                + '\t\t\t</field>\n'
            )
            col = 1 + col & 3
        output += (
            '\t\t</panel>\n'
            + '\t</panel>\n'
            + '</hack>'
        )
        b = output.encode("utf-8")
        f = CompatFile(b, filename="patch.xml")
        create_task(bot.send_with_file(channel, "Patch successfully converted!", f, reference=message))


class CS_npc(Command):
    time_consuming = True
    name = ["npc"]
    description = "Searches the Cave Story NPC list for an NPC by name or ID."
    usage = "<query> <condensed{?c}>?"
    example = ("cs_npc misery",)
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, bot, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(entity_list.search, argv, lim, timeout=8)
        # Sends multiple messages up to 20000 characters total
        if len(data):
            head = entity_list.data[0][1]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i * " "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output) < 20000 and len(output) > 1900:
                response = [f"Search results for `{argv}`:"]
                lines = output.splitlines()
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            return f"Search results for `{argv}`:\n{code_md(output)}"
        raise LookupError(f"No results for {argv}.")


class CS_flag(Command):
    name = ["OOB", "CS_OOB", "CS_flags"]
    description = "Searches the Cave Story OOB flags list for a memory variable."
    usage = "<query> <condensed{?c}>?"
    example = ("cs_oob key",)
    flags = "c"
    no_parse = True
    rate_limit = 2

    async def __call__(self, args, flags, **void):
        lim = ("c" not in flags) * 40 + 20
        argv = " ".join(args)
        data = await create_future(tsc_list.search, argv, lim, timeout=8)
        # Sends multiple messages up to 20000 characters total
        if len(data):
            head = tsc_list.data[0][0]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i * " "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output) < 20000 and len(output) > 1900:
                response = [f"Search results for `{argv}`:"]
                lines = output.splitlines()
                curr = "```\n"
                for line in lines:
                    if len(curr) + len(line) > 1900:
                        response.append(curr + "```")
                        curr = "```\n"
                    if len(line):
                        curr += line + "\n"
                response.append(curr + "```")
                return response
            return f"Search results for `{argv}`:\n{code_md(output)}"
        raise LookupError(f"No results for {argv}.")


class CS_mod(Command):
    time_consuming = True
    name = ["CS_search"]
    description = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
    usage = "<query>"
    example = ("cs_mod critter",)
    no_parse = True
    rate_limit = (3, 7)

    async def __call__(self, channel, user, args, **void):
        argv = " ".join(args)
        fut = create_future(douclub.search, argv, timeout=8)
        try:
            data = await searchForums(argv)
        except ConnectionError as ex:
            if ex.errno != 404:
                raise
            data = []
        data += await fut
        if not data:
            raise LookupError(f"No results for {argv}.")
        description = f"Search results for `{argv}`:\n"
        fields = deque()
        for res in data:
            fields.append(dict(
                name=res["name"],
                value=res["url"] + "\n" + lim_str(res["description"], 128).replace("\n", " ") + f"\n> {res['author']}",
                inline=False,
            ))
        self.bot.send_as_embeds(channel, description=description, fields=fields, author=get_author(user))


class CS_Database(Database):
    name = "cs_database"
    no_file = True

    async def __call__(self, **void):
        entity_list.update()
        tsc_list.update()
        douclub.update()


class Wav2Png(Command):
    _timeout_ = 15
    name = ["Png2Wav", "Png2Mp3"]
    description = "Runs wav2png on the input URL. See https://github.com/thomas-xin/Audio-Image-Converter for more info, or to run it yourself!"
    usage = "<0:search_links>"
    example = ("wav2png https://www.youtube.com/watch?v=IgOci6JXPIc", "png2wav https://mizabot.xyz/favicon")
    rate_limit = (20, 30)
    typing = True

    async def __call__(self, bot, channel, message, argv, name, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            raise ArgumentError("Input string is empty.")
        urls = await bot.follow_url(argv, allow=True, images=False)
        if not urls or not urls[0]:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        fn = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        ts = ts_us()
        ext = "png" if name == "wav2png" else "wav"
        dest = f"cache/&{ts}." + ext
        w2p = "wav2png" if name == "wav2png" else "png2wav"
        args = [python, w2p + ".py", url, "../" + dest]
        with discord.context_managers.Typing(channel):
            print(args)
            proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd() + "/misc", stdout=subprocess.DEVNULL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3200)
            except (T0, T1, T2):
                with tracebacksuppressor:
                    force_kill(proc)
                raise
        await bot.send_with_file(channel, "", dest, filename=fn + "." + ext, reference=message)


class SpectralPulse(Command):
    _timeout_ = 150
    description = "Runs SpectralPulse on the input URL. Operates on a global queue system. See https://github.com/thomas-xin/SpectralPulse for more info, or to run it yourself!"
    usage = "<0:search_links>"
    example = ("spectralpulse https://www.youtube.com/watch?v=IgOci6JXPIc", "spectralpulse -fps 60 https://www.youtube.com/watch?v=kJQP7kiw5Fk")
    rate_limit = (120, 180)
    typing = True
    spec_sem = Semaphore(1, 256, rate_limit=1)

    async def __call__(self, bot, channel, message, args, **void):
        for a in message.attachments:
            args.insert(0, a.url)
        if not args:
            raise ArgumentError("Input string is empty.")
        urls = await bot.follow_url(args.pop(0), allow=True, images=False)
        if not urls or not urls[0]:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        kwargs = {
            "-size": "[1280,720]",
            "-fps": "30",
            "-sample_rate": "48000",
            "-amplitude": "0.1",
            "-smudge_ratio": "0.875",
            "-speed": "2",
            "-lower_bound": "A0",
            "-higher_bound": "F#10",
            "-particles": "piano",
            "-skip": "true",
            "-display": "false",
            "-render": "true",
            "-play": "false",
            "-image": "true",
        }
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("-"):
                if arg in {
                    "-size", "-fps", "-sample_rate", "-amplitude",
                    "-smudge_ratio", "-speed", "-lower_bound", "-higher_bound",
                    "-particles", "-skip", "-display", "-render", "-play", "-image",
                    "-dest",
                    "-width", "-height",
                }:
                    kwargs[arg] = args[i + 1].replace("\xad", "#")
                    i += 1
            i += 1
        if "-width" in kwargs and "-height" in kwargs:
            kwargs["-size"] = f'({kwargs["-width"]},{kwargs["-height"]})'
        name = kwargs.get("-dest") or url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
        n1 = name + ".mp4"
        n2 = name + ".png"
        ts = ts_us()
        dest = f"cache/&{ts}"
        fn1 = dest + ".mp4"
        fn2 = dest + ".png"
        args = [
            python, "misc/spectralpulse/main.py",
            *itertools.chain(*kwargs.items()),
            "-dest", dest, url,
        ]
        with discord.context_managers.Typing(channel):
            if self.spec_sem.is_busy() and not getattr(message, "simulated", False):
                await send_with_react(channel, italics(ini_md(f"SpectralPulse: {sqr_md(url)} enqueued in position {sqr_md(self.spec_sem.passive + 1)}.")), reacts="❎", reference=message)
            async with self.spec_sem:
                print(args)
                proc = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd(), stdout=subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3200)
                except (T0, T1, T2):
                    with tracebacksuppressor:
                        force_kill(proc)
                    raise
                for ext in ("pcm", "riff"):
                    await create_future(os.remove, f"{dest}.{ext}")
        await bot.send_with_file(channel, "", fn1, filename=n1, reference=message)
        if kwargs.get("-image") in ("true", "True"):
            await bot.send_with_file(channel, "", fn2, filename=n2, reference=message)


class MCEnchant(Command):
    name = ["Enchant", "GenerateEnchant"]
    description = "Given an item and custom enchant values, generates a Minecraft /give command. An old misc command brought back."
    usage = "<item> (<enchantment>|<level>)*"
    example = ("enchant diamond_sword sharpness 8, fire_aspect 3, sweeping", "enchant diamond_axe mending unbreaking XI silk_touch", "enchant netherite_shovel efficiency 2000 looting vanishing_curse")
    rate_limit = (4, 5)

    def __call__(self, args, **void):
        import enchant_generator
        if not args:
            raise ArgumentError("Input string is empty.")
        item = args.pop(0)
        return fix_md(enchant_generator.generate_enchant(item, args))


class BTD6Paragon(Command):
    name = ["Paragon", "GenerateParagon"]
    description = "Given a tower and provided parameters, generates a list of Bloons TD 6 optimised paragon sacrifices. Parameters are \"p\" for pops, \"g\" for cash generated, \"t\" for Geraldo totems, and \"l\" for additional tower limit."
    usage = "<tower> <sacrifices>* <parameters>*"
    example = ("paragon dartmonkey 520 050 2*025", "paragon boat 400000p 30l", "paragon monkey_ace 2x500 2x050 2x005")
    rate_limit = (4, 5)

    def __call__(self, args, **void):
        import paragon_calc
        if not args:
            raise ArgumentError("Input string is empty.")
        return "\xad" + paragon_calc.parse(args)


class DeviantArt(Command):
    server_only = True
    name = ["DASubscribe"]
    min_level = 2
    description = "Subscribes to a DeviantArt Gallery, reposting links to all new posts."
    usage = "(add|remove)? <url> <reversed{?r}>?"
    flags = "raed"
    rate_limit = 4

    async def __call__(self, argv, flags, channel, guild, bot, **void):
        data = bot.data.deviantart
        update = bot.data.deviantart.update
        if not argv:
            assigned = data.get(channel.id, ())
            if not assigned:
                return ini_md(f"No currently subscribed DeviantArt Galleries for {sqr_md(channel)}.")
            if "d" in flags:
                data.pop(channel.id, None)
                return css_md(f"Successfully removed all DeviantArt Gallery subscriptions from {sqr_md(channel)}.")
            return f"Currently subscribed DeviantArt Galleries for {sqr_md(channel)}:{ini_md(iter2str(assigned, key=lambda x: x['user']))}"
        urls = await bot.follow_url(argv, images=False, allow=True)
        if not urls:
            raise ArgumentError("Please input a valid URL.")
        url = urls[0]
        if "deviantart.com" not in url:
            raise ArgumentError("Please input a DeviantArt Gallery URL.")
        # Parse DeviantArt gallery URls
        url = url[url.index("deviantart.com") + 15:]
        spl = url.split("/")
        user = spl[0]
        if spl[1] != "gallery":
            raise ArgumentError("Only Gallery URLs are supported.")
        content = spl[2].split("&", 1)[0]
        folder = no_md(spl[-1].split("&", 1)[0])
        # Gallery may be an ID or "all"
        try:
            content = int(content)
        except (ValueError, TypeError):
            if content in (user, "all"):
                content = user
            else:
                raise TypeError("Invalid Gallery type.")
        if content in self.data.get(channel.id, {}):
            raise KeyError(f"Already subscribed to {user}: {folder}")
        if "d" in flags:
            try:
                data.get(channel.id).pop(content)
            except KeyError:
                raise KeyError(f"Not currently subscribed to {user}: {folder}")
            else:
                if channel.id in data and not data[channel.id]:
                    data.pop(channel.id)
                return css_md(f"Successfully unsubscribed from {sqr_md(user)}: {sqr_md(folder)}.")
        set_dict(data, channel.id, {}).__setitem__(content, {"user": user, "type": "gallery", "reversed": ("r" in flags), "entries": {}})
        update(channel.id)
        out = f"Successfully subscribed to {sqr_md(user)}: {sqr_md(folder)}"
        if "r" in flags:
            out += ", posting in reverse order"
        return css_md(out + ".")


class UpdateDeviantArt(Database):
    name = "deviantart"

    async def processPart(self, found, c_id):
        bot = self.bot
        try:
            channel = await bot.fetch_channel(c_id)
        except (LookupError, discord.NotFound):
            self.data.pop(c_id, None)
            return
        try:
            assigned = self.data.get(c_id)
            if assigned is None:
                return
            embs = deque()
            for content in assigned:
                if content not in found:
                    continue
                items = found[content]
                entries = assigned[content]["entries"]
                new = tuple(items)
                orig = tuple(entries)
                # O(n) comparison
                if assigned[content].get("reversed", False):
                    it = reversed(new)
                else:
                    it = new
                for i in it:
                    if i not in entries:
                        entries[i] = True
                        self.update(c_id)
                        home = "https://www.deviantart.com/" + items[i][2]
                        emb = discord.Embed(
                            colour=discord.Colour(1),
                            description="*🔔 New Deviation from " + items[i][2] + " 🔔*\n" + items[i][0],
                        ).set_image(url=items[i][1]).set_author(name=items[i][2], url=home, icon_url=items[i][3])
                        embs.append(emb)
                for i in orig:
                    if i not in items:
                        entries.pop(i)
                        self.update(c_id)
        except:
            print(found)
            print_exc()
        else:
            bot.send_embeds(channel, embs)

    async def fetch_gallery(self, folder, username):
        base = "https://www.deviantart.com/_napi/da-user-profile/api/gallery/contents?username="
        # "all" galleries require different URL options
        if type(folder) is str:
            f_id = "&all_folder=true&mode=oldest"
        else:
            f_id = "&folderid=" + str(folder)
        url = base + username + f_id
        # Binary search algorithm to improve search time for entire galleries to O(log n)
        maxitems = 2147483647
        r = 0
        t = utc()
        found = {}
        futs = deque()
        page = 24
        # Begin with quaternary search (powers of 4) to estimate lowest power of 2 greater than or equal to gallery page count
        with suppress(StopIteration):
            for i in range(2 + int(math.log2(maxitems / page))):
                curr = 1 << i
                search = url + f"&offset={curr * page}&limit={page}"
                futs.append((curr, Request(search, timeout=20, json=True, aio=True)))
                if i & 1:
                    for x, fut in futs:
                        try:
                            resp = await fut
                        except ConnectionError as ex:
                            if ex.errno >= 500:
                                return
                            raise
                        if resp.get("results"):
                            found[x] = resp
                        if not resp.get("hasMore"):
                            curr = x
                            raise StopIteration
                r += 1
        # Once the end has been reached, use binary search to estimate the page count again, being off by at most 8 pages
        check = 1 << max(0, i - 2)
        while check > 4:
            x = curr - check
            search = url + f"&offset={x * page}&limit={page}"
            resp = await Request(search, json=True, aio=True)
            if resp.get("results"):
                found[x] = resp
            r += 1
            if not resp.get("hasMore"):
                curr = x
            check >>= 1
        futs = deque()
        for i in range(curr + 1):
            if i not in found:
                search = url + f"&offset={i * page}&limit={page}"
                futs.append((i, Request(search, json=True, aio=True)))
                r += 1
        for x, fut in futs:
            resp = await fut
            if resp.get("results"):
                found[x] = resp
        # Collect all page results into a single list
        results = (resp.get("results", ()) for resp in found.values())
        items = {}
        for res in itertools.chain(*results):
            deviation = res["deviation"]
            media = deviation["media"]
            prettyName = media["prettyName"]
            orig = media["baseUri"]
            extra = ""
            token = "?token=" + media["token"][0]
            # Attempt to find largest available format for media
            for t in reversed(media["types"]):
                if t["t"].casefold() == "fullview":
                    if "c" in t:
                        extra = "/" + t["c"].replace("<prettyName>", prettyName)
                        break
            image_url = orig + extra + token
            items[deviation["deviationId"]] = (deviation["url"], image_url, deviation["author"]["username"], deviation["author"]["usericon"])
        return items

    async def __call__(self):
        t = set_dict(self.__dict__, "time", 0)
        # Fetches once every 5 minutes
        if utc() - t < 300:
            return
        self.time = inf
        conts = {i: a[i]["user"] for a in tuple(self.data.values()) for i in a}
        total = {}
        attempts, successes = 0, 0
        for folder, username in conts.items():
            with tracebacksuppressor:
                items = await self.fetch_gallery(folder, username)
                if items:
                    total[folder] = items
        for c_id in tuple(self.data):
            create_task(self.processPart(total, c_id))
        self.time = utc()


while True:
    try:
        douclub = DouClub(AUTH["knack_id"], AUTH["knack_secret"])
    except KeyError:
        douclub = cdict(
            search=lambda *void1, **void2: exec('raise FileNotFoundError("Unable to search Doukutsu Club.")'),
            update=lambda: None,
            pull=lambda: None,
        )
    except:
        print_exc()
        time.sleep(30)
        continue
    break
