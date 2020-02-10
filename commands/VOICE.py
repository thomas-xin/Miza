import youtube_dl, asyncio, discord, time, os, urllib, json, copy, traceback
from subprocess import check_output, CalledProcessError, STDOUT
from smath import *


def getDuration(filename):
    command = [
        'ffprobe', 
        '-v', 
        'error', 
        '-show_entries', 
        'format=duration', 
        '-of', 
        'default=noprint_wrappers=1:nokey=1', 
        filename,
      ]
    try:
        output = check_output(command, stderr=STDOUT).decode()
    except CalledProcessError as e:
        output = e.output.decode()
    try:
        i = output.index("\r")
        output = output[:i]
    except ValueError:
        pass
    if output == "N/A":
        n = 0
    else:
        n = roundMin(float(output))
    #print(n)
    return max(1 / (1 << 24), n)


async def forceJoin(guild, channel, user, client, _vars):
    found = False
    if guild.id not in _vars.queue:
        for func in _vars.categories["voice"]:
            if "join" in func.name:
                try:
                    await func(client=client, user=user, _vars=_vars, channel=channel, guild=guild)
                except discord.ClientException:
                    pass
                except AttributeError:
                    pass
    try:
        auds = _vars.queue[guild.id]
        auds.channel = channel.id
    except KeyError:
        raise LookupError("Voice channel not found.")
    return auds


class urlBypass(urllib.request.FancyURLopener):
    version = "Mozilla/6." + str(xrand(10))

    
class videoDownloader:
    
    opener = urlBypass()
    
    ydl_opts = {
        "quiet": 1,
        "verbose": 0,
        "format": "bestaudio/best",
        "noplaylist": 1,
        "call_home": 1,
        "nooverwrites": 1,
        "ignoreerrors": 0,
        "source_address": "0.0.0.0",
        "default_search": "auto",
        }

    def __init__(self):
        self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
        self.lastsearch = 0
        self.requests = 0

    def extract(self, item, force):
        resp = self.downloader.extract_info(item, download=False, process=False)
        if resp.get("_type", None) == "url":
            resp = self.downloader.extract_info(resp["url"], download=False, process=False)
        if resp is None or not len(resp):
            raise EOFError("No search results found.")
        output = []
        if resp.get("_type", None) == "playlist":
            entries = list(resp["entries"])
            if force or len(entries) <= 1:
                for entry in entries:
                    data = self.downloader.extract_info(entry["id"], download=False, process=False)
                    output.append({
                        "id": data["id"],
                        "name": data["title"],
                        "url": data["webpage_url"],
                        "duration": data["duration"],
                        })
            else:
                for entry in entries:
                    dur = "duration" in entry
                    temp = {
                        "id": entry["id"],
                        "name": entry["title"],
                        "url": entry["url"],
                        "duration": entry.get("duration", 60),
                        }
                    if not dur:
                        temp["research"] = True
                    output.append(temp)
        else:
            dur = "duration" in resp
            temp = {
                "id": resp["id"],
                "name": resp["title"],
                "url": resp["webpage_url"],
                "duration": resp.get("duration", 60),
                }
            if not dur:
                temp["research"] = True
            output.append(temp)
        return output

    def search(self, item, force=False):
        item = item.strip("<>").replace("\n", "")
        while self.requests > 4:
            time.sleep(0.01)
        if time.time() - self.lastsearch > 1800:
            self.lastsearch = time.time()
            self.searched = {}
        if item in self.searched:
            return self.searched[item]
        try:
            self.requests += 1
            self.searched[item] = output = self.extract(item, force)
            self.requests = max(self.requests - 1, 0)
            return output
        except Exception as ex:
            self.requests = max(self.requests - 1, 0)
            return str(ex)
        
    def downloadSingle(self, i, durc=None):
        new_opts = dict(self.ydl_opts)
        fn = "cache/" + i["id"].replace("@", "") + ".mp3"
        new_opts["outtmpl"] = fn
        downloader = youtube_dl.YoutubeDL(new_opts)
        try:
            downloader.download([i["url"]])
            if durc is not None:
                durc[0] = getDuration(fn)
        except:
            i["id"] = ""
            print(traceback.format_exc())
        return True

    def extractSingle(self, i):
        item = i["url"]
        while self.requests > 4:
            time.sleep(0.01)
        if time.time() - self.lastsearch > 1800:
            self.lastsearch = time.time()
            self.searched = {}
        if item in self.searched:
            i["duration"] = self.searched[item]["duration"]
            i["url"] = self.searched[item]["webpage_url"]
            return True
        try:
            self.requests += 1
            data = self.downloader.extract_info(item, download=False, process=False)
            i["duration"] = data["duration"]
            i["url"] = data["webpage_url"]
            self.requests = max(self.requests - 1, 0)
        except:
            self.requests = max(self.requests - 1, 0)
            i["id"] = ""
            print(traceback.format_exc())
        return True

    def getDuration(self, filename):
        return getDuration(filename)


downloader = videoDownloader()


class queue:
    is_command = True
    server_only = True
    ytdl = downloader

    def __init__(self):
        self.name = ["q", "play", "playing", "np", "p"]
        self.min_level = 0
        self.description = "Shows the music queue, or plays a song in voice."
        self.usage = "<link[]> <verbose(?v)> <hide(?h)>"

    async def __call__(self, client, user, _vars, argv, channel, guild, flags, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        elapsed = auds.stats["position"]
        q = _vars.queue[guild.id].queue
        if not len(argv.replace(" ", "")):
            if not len(q):
                return "```css\nQueue for " + uniStr(guild.name) + " is currently empty. ```", 1
            if "v" in flags:
                if auds.stats["loop"]:
                    totalTime = inf
                else:
                    totalTime = -elapsed
                    for e in q:
                        totalTime += e["duration"]
                cnt = len(q)
                info = (
                    "`" + uniStr(cnt) + " item" + "s" * (cnt != 1) + ", estimated total duration: "
                    + uniStr(sec2Time(totalTime / auds.speed)) + "`"
                    )
            else:
                info = ""
            currTime = 0
            show = ""
            for i in range(len(q)):
                curr = "\n"
                e = q[i]
                curr += " " * (int(math.log10(len(q))) - int(math.log10(max(1, i))))
                curr += "[" + uniStr(i) + "] "
                if "v" in flags:
                    curr += (
                        uniStr(noSquareBrackets(e["name"])) + ", URL: [" + e["url"] + "]"
                        + ", Duration: " + uniStr(sec2Time(e["duration"]))
                        + ", Added by: " + uniStr(e["added by"])
                        )
                else:
                    curr += limStr(uniStr(noSquareBrackets(e["name"])), 48)
                estim = currTime - elapsed
                if estim > 0:
                    if i <= 1 or not auds.stats["shuffle"]:
                        curr += ", Time until playing: " + uniStr(sec2Time(estim / auds.speed))
                    else:
                        curr += ", Time until playing: (" + uniStr(sec2Time(estim / auds.speed)) + ")"
                else:
                    curr += ", Remaining time: " + uniStr(sec2Time((estim + e["duration"]) / auds.speed))
                if len(show) + len(info) + len(curr) < 1800:
                    show += curr
                else:
                    show += uniStr("\nAnd " + str(len(q) - i) + " more...", 1)
                    break
                if i <= 1 or not auds.stats["shuffle"]:
                    currTime += e["duration"]
            duration = q[0]["duration"]
            sym = "⬜⬛"
            barsize = 16 * (1 + ("v" in flags))
            r = round(min(1, elapsed / duration) * barsize)
            bar = sym[0] * r + sym[1] * (barsize - r)
            countstr = "Currently playing " + uniStr(noSquareBrackets(q[0]["name"])) + "\n"
            countstr += (
                "(" + uniStr(dhms(elapsed))
                + "/" + uniStr(dhms(duration)) + ") "
                )
            countstr += bar + "\n"
            return (
                "Queue for **" + guild.name + "**: "
                + info + "\n```css\n"
                + countstr + show + "```", 1
                )
        else:
            auds.preparing = True
            output = [None]
            doParallel(self.ytdl.search, [argv], output)
            await channel.trigger_typing()
            while output[0] is None:
                await asyncio.sleep(0.01)
            res = output[0]
            if type(res) is str:
                raise ConnectionError(res)
            elif type(res) is dict:
                return (
                    "```css\nAdding " + uniStr(res["count"]) + " tracks...```", 1
                    )
            dur = 0
            added = []
            names = []
            for e in res:
                name = e["name"]
                url = e["url"]
                duration = e["duration"]
                v_id = e["id"]
                temp = {
                    "name": name,
                    "url": url,
                    "duration": duration,
                    "added by": user.name,
                    "u_id": user.id,
                    "id": v_id,
                    "skips": [],
                    }
                if "research" in e:
                    temp["research"] = True
                added.append(temp)
                if not dur:
                    dur = duration
                names.append(name)
            total_duration = 0
            for e in q:
                total_duration += e["duration"]
            total_duration = max((total_duration - elapsed) / auds.speed, dur / 128 + frand(0.5) + 2)
            q += added
            if not len(names):
                raise EOFError("No results for " + str(argv) + ".")
            if "v" in flags:
                names = [subDict(i, "id") for i in added]
            elif len(names) == 1:
                names = names[0]
            if not "h" in flags:
                return (
                    "```css\n🎶 Added " + noSquareBrackets(uniStr(names))
                    + " to the queue! Estimated time until playing: "
                    + uniStr(sec2Time(total_duration)) + ". 🎶```", 1
                    )


class playlist:
    is_command = True
    server_only = True
    ytdl = downloader

    def __init__(self):
        self.name = ["defaultplaylist", "pl"]
        self.min_level = 0
        self.description = "Shows, appends, or removes from the default playlist."
        self.usage = "<link[]> <remove(?d)> <verbose(?v)>"

    async def __call__(self, user, argv, _vars, guild, flags, channel, perm, **void):
        if argv or "d" in flags:
            req = 2
            if perm < req:
                raise PermissionError(
                    "Insufficient privileges to modify default playlist for " + uniStr(guild.name)
                    + ". Required level: " + uniStr(req) + ", Current level: " + uniStr(perm) + "."
                    )
        pl = _vars.playlists.setdefault(guild.id, [])
        if not argv:
            if "d" in flags:
                _vars.playlists[guild.id] = []
                doParallel(_vars.update)
                return "```css\nRemoved all entries from the default playlist for " + uniStr(guild.name) + ".```"
            if "v" in flags:
                return (
                    "Current default playlist for **" + guild.name + "**: ```json\n"
                    + str(pl).replace("'", '"') + "```"
                    )
            else:
                items = []
                for i in pl:
                    items.append(limStr(noSquareBrackets(i["name"]), 32))
                s = ""
                for i in range(len(items)):
                    s += " " * (int(math.log10(len(items))) - int(math.log10(max(1, i))))
                    s += "[" + uniStr(i) + "] "
                    s += items[i] + "\n"
            return (
                "Current default playlist for **" + guild.name + "**: ```ini\n"
                + s + "```"
                )
        if "d" in flags:
            i = _vars.evalMath(argv)
            temp = pl[i]
            pl.pop(i)
            doParallel(_vars.update)
            return (
                "```css\nRemoved " + uniStr(noSquareBrackets(temp["name"])) + " from the default playlist for "
                + uniStr(guild.name) + ".```"
                )
        output = [None]
        doParallel(self.ytdl.search, [argv, True], output)
        await channel.trigger_typing()
        while output[0] is None:
            await asyncio.sleep(0.01)
        res = output[0]
        if type(res) is str:
            raise ConnectionError(res)
        names = []
        for e in res:
            name = e["name"]
            names.append(noSquareBrackets(name))
            pl.append({
                "name": name,
                "url": e["url"],
                "duration": e["duration"],
                "id": e["id"],
                })
        if len(names):
            pl.sort(key=lambda x: x["name"][0].lower())
            doParallel(_vars.update)
            return "```css\nAdded " + uniStr(names) + " to the default playlist for " + uniStr(guild.name) + ".```"
        

class join:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["summon", "connect"]
        self.min_level = 0
        self.description = "Summons the bot into a voice channel."
        self.usage = ""

    async def __call__(self, client, user, _vars, channel, guild, **void):
        voice = user.voice
        vc = voice.channel
        if guild.id not in _vars.queue:
            await channel.trigger_typing()
            _vars.queue[guild.id] = _vars.createPlayer(channel.id)
        try:
            joined = True
            await vc.connect(timeout=_vars.timeout, reconnect=True)
        except discord.ClientException:
            joined = False
        for user in guild.members:
            if user.id == client.user.id:
                asyncio.create_task(user.edit(mute=False,deafen=False))
        if joined:
            return (
                "```css\n🎵 Successfully connected to " + uniStr(vc.name)
                + " in " + uniStr(guild.name) + ". 🎵```", 1
                )


class leave:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = ["quit", "dc", "disconnect"]
        self.min_level = 1
        self.description = "Leaves a voice channel."
        self.usage = ""

    async def __call__(self, user, client, _vars, guild, **void):
        error = None
        try:
            _vars.queue.pop(guild.id)
        except KeyError:
            error = LookupError("Unable to find connected channel.")
        found = False
        for vclient in client.voice_clients:
            if guild.id == vclient.channel.guild.id:
                await vclient.disconnect(force=False)
                return "```css\n🎵 Successfully disconnected from " + uniStr(guild.name) + ". 🎵```", 1
        error = LookupError("Unable to find connected channel.")
        if error is not None:
            raise error


class remove:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["rem", "skip", "s"]
        self.min_level = 0
        self.description = "Removes an entry from the voice channel queue."
        self.usage = "<0:queue_position[0]> <force(?f)>"

    async def __call__(self, client, user, _vars, args, argv, guild, flags, **void):
        found = False
        if guild.id not in _vars.queue:
            raise LookupError("Currently not playing in a voice channel.")
        auds = _vars.queue[guild.id]
        s_perm = _vars.getPerms(user, guild)
        min_level = 1
        if "f" in flags and s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to force skip. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(min_level) + "."
                )
        if not argv:
            elems = [0]
        elif ":" in argv or ".." in argv:
            l = argv.replace("...", ":").replace("..", ":").split(":")
            if len(l) > 2:
                raise ValueError("Too many arguments for range input.")
            if l[0]:
                left = round(_vars.evalMath(l[0]))
            else:
                left = 0
            if l[1]:
                right = round(_vars.evalMath(l[1]))
            else:
                right = len(auds.queue)
            elems = xrange(left, right)
        else:
            elems = [round(_vars.evalMath(i)) for i in args]
        if not "f" in flags:
            valid = True
            for e in elems:
                if not isValid(e):
                    valid = False
                    break
            if not valid:
                elems = range(len(auds.queue))
        members = 0
        for vc in client.voice_clients:
            if vc.channel.guild.id == guild.id:
                for memb in vc.channel.members:
                    if not memb.bot:
                        members += 1
        required = 1 + members >> 1
        response = "```css\n"
        for pos in elems:
            try:
                if not isValid(pos):
                    if "f" in flags:
                        auds.queue = []
                        auds.new()
                        #print("Stopped audio playback in " + guild.name)
                        return "```fix\nRemoved all items from the queue.```"
                    raise LookupError
                curr = auds.queue[pos]
            except LookupError:
                raise IndexError("Entry " + uniStr(pos) + " is out of range.")
            if type(curr["skips"]) is list:
                if "f" in flags or user.id == curr["u_id"]:
                    curr["skips"] = None
                elif user.id not in curr["skips"]:
                    curr["skips"].append(user.id)
            else:
                curr["skips"] = None
            if curr["skips"] is not None:
                response += (
                    "Voted to remove " + uniStr(noSquareBrackets(curr["name"]))
                    + " from the queue.\nCurrent vote count: "
                    + uniStr(len(curr["skips"])) + ", required vote count: " + uniStr(required) + ".\n"
                    )
        q = auds.queue
        i = 0
        while i < len(q):
            song = q[i]
            if song["skips"] is None or len(song["skips"]) >= required:
                if i == 0:
                    auds.advance(False)
                    auds.new()
                    #print("Stopped audio playback in " + guild.name)
                else:
                    q.pop(i)
                response += uniStr(noSquareBrackets(song["name"])) + " has been removed from the queue.\n"
                continue
            else:
                i += 1
        return response + "```", 1


class pause:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["resume", "unpause"]
        self.min_level = 1
        self.description = "Pauses or resumes audio playing."
        self.usage = ""

    async def __call__(self, _vars, name, guild, client, user, channel, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if not auds.paused > 1:
            auds.paused = name == "pause"
        return "```css\nSuccessfully " + name + "d audio playback in " + uniStr(guild.name) + ".```"


class seek:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Seeks to a position in the current audio file."
        self.usage = "<pos[0]>"

    async def __call__(self, argv, _vars, guild, client, user, channel, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        pos = 0
        if argv:
            data = argv.split(":")
            mult = 1
            while len(data):
                pos += _vars.evalMath(data[-1]) * mult
                data = data[:-1]
                if mult <= 60:
                    mult *= 60
                elif mult <= 3600:
                    mult *= 24
                elif len(data):
                    raise ValueError("Too many time arguments.")
        pos = auds.seek(pos)
        return (
            "```css\nSuccessfully moved audio position to "
            + uniStr(sec2Time(pos)) + ".```"
            )


class dump:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = []
        self.min_level = 2
        self.description = "Dumps or loads the currently playing audio queue state."
        self.usage = "<data[]> <append(?a)> <hide(?h)>"

    async def __call__(self, guild, channel, user, client, _vars, argv, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if not argv and not len(message.attachments):
            q = copy.deepcopy(auds.queue)
            for e in q:
                e["id"] = e["id"].replace("@", "")
                e.pop("added by")
                e.pop("u_id")
                e.pop("skips")
            d = {
                "stats": auds.stats,
                "queue": q,
                }
            return "Queue data for **" + guild.name + "**:\n```json\n" + json.dumps(d) + "\n```"
        try:
            opener = urlBypass()
            if len(message.attachments):
                url = message.attachments[0].url
            else:
                url = _vars.verifyURL(argv)
            resp = opener.open(url)
            rescode = resp.getcode()
            if rescode != 200:
                raise ConnectionError(rescode)
            s = resp.read().decode("utf-8")
            s = s[s.index("{"):]
            if s[-4:] == "\n```":
                s = s[:-4]
        except:
            s = argv
        d = json.loads(s)
        q = d["queue"]
        for e in q:
            e["added by"] = user.name
            e["u_id"] = user.id
            e["skips"] = []
        if not "a" in flags:
            #print("Stopped audio playback in " + guild.name)
            auds.new()
            auds.queue = q
            for k in d["stats"]:
                if k not in auds.stats:
                    d["stats"].pop(k)
                d["stats"][k] = float(d["stats"][k])
            auds.stats.update(d["stats"])
            if not "h" in flags:
                return "```css\nSuccessfully reinstated audio queue for " + uniStr(guild.name) + ".```"
        auds.queue += q
        auds.stats = d["stats"]
        if not "h" in flags:
            return "```css\nSuccessfully appended dump to queue for " + uniStr(guild.name) + ".```"
            

class volume:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["vol", "audio", "v", "opt", "set"]
        self.min_level = 0
        self.description = "Changes the current audio settings for this server."
        self.usage = (
            "<value[]> <volume()(?v)> <reverb(?r)> <speed(?s)> <pitch(?p)> "
            + "<bassboost(?b)> <reverbdelay(?d)> <loop(?l)> <shuffle(?x)> <clear(?c)>"
            )

    async def __call__(self, client, channel, user, guild, _vars, flags, argv, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if "c" in flags:
            op = None
        elif "v" in flags:
            op = "volume"
        elif "s" in flags:
            op = "speed"
        elif "p" in flags:
            op = "pitch"
        elif "b" in flags:
            op = "bassboost"
        elif "d" in flags:
            op = "reverbdelay"
        elif "r" in flags:
            op = "reverb"
        elif "l" in flags:
            op = "loop"
        elif "x" in flags:
            op = "shuffle"
        else:
            op = "settings"
        if not argv and op is not None:
            if op == "settings":
                return (
                    "Current audio settings for **" + guild.name + "**:\n```json\n"
                    + str(auds.stats).replace("'", '"') + "```"
                    )
            orig = _vars.queue[guild.id].stats[op]
            if op in "loop shuffle":
                num = bool(orig)
            else:
                num = round(100. * orig, 9)
            return (
                "```css\nCurrent audio " + op + " in " + uniStr(guild.name)
                + ": " + uniStr(num) + ".```"
                )
        if op == "settings":
            op = "volume"
        s_perm = _vars.getPerms(user, guild)
        if s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to change audio settings. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(1) + "."
                )
        if op is None:
            pos = auds.stats["position"]
            auds.stats = dict(auds.defaults)
            auds.new(auds.file, pos)
            return (
                "```css\nSuccessfully reset all audio settings for "
                + uniStr(guild.name) + ".```"
                )
        origVol = _vars.queue[guild.id].stats
        val = roundMin(float(_vars.evalMath(argv) / 100))
        orig = round(origVol[op] * 100, 9)
        new = round(val * 100, 9)
        if op in "loop shuffle":
            origVol[op] = new = bool(val)
            orig = bool(orig)
        else:
            origVol[op] = val
        if op in "speed pitch":
            auds.new(auds.file, auds.stats["position"])
        return (
            "```css\nChanged audio " + op + " in " + uniStr(guild.name)
            + " from " + uniStr(orig)
            + " to " + uniStr(new) + ".```"
            )


class randomize:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["shuffle"]
        self.min_level = 1
        self.description = "Shuffles the audio queue."
        self.usage = ""

    async def __call__(self, guild, channel, user, client, _vars, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if len(auds.queue):
            auds.queue = [auds.queue[0]] + shuffle(auds.queue[1:])
        return "```css\nSuccessfully shuffled audio queue for " + uniStr(guild.name) + ".```"


class unmute:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = ["unmuteall"]
        self.min_level = 2
        self.description = "Disables server mute for all members."
        self.usage = ""

    async def __call__(self, guild, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                asyncio.create_task(user.edit(mute=False, deafen=False))
        return "```css\nSuccessfully unmuted all users in voice channels in " + uniStr(guild.name) + ".```"
