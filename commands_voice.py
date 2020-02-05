import youtube_dl, asyncio, discord, time, os, urllib, json, copy
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
    n = float(output)
    print(n)
    return n


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
    version = "Mozilla/6.0"

    
class videoDownloader:
    ydl_opts = {
        "quiet": 1,
        "verbose": 0,
        "format": "bestaudio/best",
        "noplaylist": 1,
        "call_home": 1,
        "nooverwrites": 1,
        "ignoreerrors": 1,
        "source_address": "0.0.0.0",
        "default_search": "auto",
        }

    opener = urlBypass()

    def __init__(self):
        self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
        self.lastsearch = 0
        self.requests = 0

    def search(self, item):
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
            try:
                pl = self.downloader.extract_info(item, False)
                self.requests = max(self.requests - 1, 0)
            except:
                raise
            if "direct" in pl:
                resp = self.opener.open(item)
                rescode = resp.getcode()
                if rescode != 200:
                    raise ConnectionError(rescode)
                header = dict(resp.headers.items())
                duration = float(header["Content-Length"]) / 16384
                hh = hex(hash(item)).replace("-", "").replace("0x", "")
                output = [{
                    "name": pl["title"],
                    "url": pl["webpage_url"],
                    "duration": duration,
                    "id": hh,
                    }]
            elif "entries" in pl:
                output = []
                for e in pl["entries"]:
                    output.append({
                        "name": e["title"],
                        "url": e["webpage_url"],
                        "duration": e["duration"],
                        "id": e["id"],
                        })
            else:
                output = [{
                    "name": pl["title"],
                    "url": pl["webpage_url"],
                    "duration": pl["duration"],
                    "id": pl["id"],
                    }]
            self.searched[item] = output
            return output
        except Exception as ex:
            return str(ex)
        
    def download(self, item, i_id, durc=None):
        new_opts = dict(self.ydl_opts)
        fn = "cache/" + i_id.replace("@", "") + ".mp3"
        new_opts["outtmpl"] = fn
        downloader = youtube_dl.YoutubeDL(new_opts)
        try:
            downloader.download([item])
            if durc is not None:
                durc[0] = getDuration(fn)
        except Exception as ex:
            print(repr(ex))

    def getDuration(self, filename):
        return getDuration(filename)


downloader = videoDownloader()


class queue:
    is_command = True
    server_only = True
    ytdl = downloader

    def __init__(self):
        self.name = ["q", "qlist", "play", "playing", "np", "p"]
        self.min_level = 0
        self.description = "Shows the music queue, or plays a song in voice."
        self.usage = "<link:[]> <verbose:(?v)>"

    async def __call__(self, client, user, _vars, argv, channel, guild, flags, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        elapsed = auds.readpos / 50
        q = _vars.queue[guild.id].queue
        if not len(argv.replace(" ", "")):
            if not len(q):
                return "```css\nQueue for " + uniStr(guild.name) + " is currently empty. ```"
            totalTime = -elapsed
            for e in q:
                totalTime += e["duration"]
            if "v" in flags:
                cnt = uniStr(len(q))
                info = (
                    "`" + cnt + " item" + "s" * (cnt != 1) + ", estimated total duration: "
                    + uniStr(" ".join(timeConv(totalTime))) + "`"
                    )
            else:
                info = ""
            currTime = 0
            showing = True
            show = ""
            for i in range(len(q)):
                if showing:
                    curr = "\n"
                    e = q[i]
                    curr += " " * (int(math.log10(len(q))) - int(math.log10(max(1, i))))
                    curr += "【" + uniStr(i) + "】 "
                    if "v" in flags:
                        curr += (
                            uniStr(e["name"]) + ", URL: " + e["url"]
                            + ", Duration: " + uniStr(" ".join(timeConv(e["duration"])))
                            + ", Added by: " + uniStr(e["added by"])
                            )
                    else:
                        curr += limStr(uniStr(e["name"]), 48)
                    estim = currTime - elapsed
                    if estim > 0:
                        curr += ", Time until playing: " + uniStr(" ".join(timeConv(estim)))
                    else:
                        curr += ", Remaining time: " + uniStr(" ".join(timeConv(estim + e["duration"])))
                    if len(show) + len(info) + len(curr) < 1800:
                        show += curr
                    else:
                        show += uniStr("\nAnd " + str(len(q) - i) + " more...", 1)
                        showing = False
                currTime += e["duration"]
            duration = q[0]["duration"]
            sym = "⬜⬛"
            count = 16 * (1 + ("v" in flags))
            r = round(elapsed / duration * count)
            countstr = "Currently playing " + uniStr(q[0]["name"]) + ", "
            countstr += uniStr(dhms(elapsed)) + "/" + uniStr(dhms(duration)) + ", "
            countstr += sym[0] * r + sym[1] * (count - r) + "\n"
            return (
                "Queue for **" + guild.name + "**: "
                + info + "\n```css\n"
                + countstr + show + "```"
                )
        else:
            output = [None]
            doParallel(self.ytdl.search, [argv], output)
            while output[0] is None:
                await asyncio.sleep(0.01)
            res = output[0]
            if type(res) is str:
                raise ConnectionError(res)
            dur = 0
            added = []
            names = []
            for e in res:
                name = e["name"]
                url = e["url"]
                duration = e["duration"]
                v_id = e["id"]
                added.append({
                    "name": name,
                    "url": url,
                    "duration": duration,
                    "added by": user.name,
                    "u_id": user.id,
                    "id": v_id,
                    "skips": [],
                    })
                if not dur:
                    dur = duration
                names.append(name)
            total_duration = 0
            for e in q:
                total_duration += e["duration"]
            total_duration = max(total_duration - elapsed, dur / 128 + frand(0.5) + 2)
            q += added
            if not len(names):
                raise EOFError("No results for " + str(argv) + ".")
            if "v" in flags:
                names = [subDict(i, "id") for i in added]
            elif len(names) == 1:
                names = names[0]
            if not "h" in flags:
                return (
                    "```css\n🎶 Added " + uniStr(names) + " to the queue! Estimated time until playing: "
                    + uniStr(" ".join(timeConv(total_duration))) + ". 🎶```"
                    )


class playlist:
    is_command = True
    server_only = True
    ytdl = downloader

    def __init__(self):
        self.name = ["defaultplaylist", "pl"]
        self.min_level = 2
        self.description = "Shows, appends, or removes from the default playlist."
        self.usage = "<link:[]> <remove:(?r)>"

    async def __call__(self, user, argv, _vars, guild, flags, **void):
        if not argv:
            if "r" in flags:
                _vars.playlists[guild.id] = []
                _vars.update()
                return "```css\nRemoved all entries from the default playlist for " + uniStr(guild.name) + ".```"
            return (
                "```css\nCurrent default playlist for " + uniStr(guild.name) + ": "
                + str(_vars.playlists.get(guild.id, [])) + ".```"
                )
        curr = _vars.playlists.get(guild.id, [])
        if "r" in flags:
            i = _vars.evalMath(argv)
            temp = curr[i]
            curr.pop(i)
            _vars.playlists[guild.id] = curr
            _vars.update()
            return "```css\nRemoved " + uniStr(temp["name"]) + " from the default playlist for " + uniStr(guild.name) + ".```"
        output = [None]
        doParallel(self.ytdl.search, [argv], output)
        while output[0] is None:
            await asyncio.sleep(0.01)
        res = output[0]
        if type(res) is str:
            raise ConnectionError(res)
        names = []
        for e in res:
            name = e["name"]
            names.append(name)
            url = e["url"]
            if "r" in flags:
                for p in curr:
                    if p["url"] == url:
                        curr.remove(p)
                        break
            else:
                curr.append({
                    "name": name,
                    "url": url,
                    "duration": e["duration"],
                    "id": e["id"],
                    })
        if len(names):
            _vars.playlists[guild.id] = curr
            _vars.update()
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
                + " in " + uniStr(guild.name) + ". 🎵```"
                )


class leave:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["quit","dc","disconnect"]
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
                return "```css\n🎵 Successfully disconnected from " + uniStr(guild.name) + ". 🎵```"
        error = LookupError("Unable to find connected channel.")
        if error is not None:
            raise error


class remove:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["rem", "skip"]
        self.min_level = 0
        self.description = "Removes an entry from the voice channel queue."
        self.usage = "<0:queue_position[0]> <force:(?f)>"

    async def __call__(self, client, user, _vars, argv, guild, flags, **void):
        found = False
        if guild.id not in _vars.queue:
            raise LookupError("Currently not playing in a voice channel.")
        s_perm = _vars.getPerms(user, guild)
        min_level = 1
        if "f" in flags and s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to force skip. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(min_level) + "."
                )
        if not argv:
            pos = 0
        else:
            pos = _vars.evalMath(argv)
        try:
            if not isValid(pos):
                if "f" in flags:
                    _vars.queue[guild.id].queue = []
                    print("Stopped audio playback in " + guild.name)
                    for vc in client.voice_clients:
                        if vc.channel.guild.id == guild.id:
                            vc.stop()
                    return "```css\nRemoved all items from the queue.```"
                raise LookupError
            curr = _vars.queue[guild.id].queue[pos]
        except LookupError:
            raise IndexError("Entry " + uniStr(pos) + " is out of range.")
        if type(curr["skips"]) is not tuple:
            if "f" in flags or user.id == curr["u_id"]:
                curr["skips"] = ()
            elif user.id not in curr["skips"]:
                curr["skips"].append(user.id)
        members = 0
        for vc in client.voice_clients:
            if vc.channel.guild.id == guild.id:
                for memb in vc.channel.members:
                    if not memb.bot:
                        members += 1
        required = 1 + members >> 1
        if type(curr["skips"]) is tuple:
            response = "```\n"
        else:
            response = (
                "```css\nVoted to remove " + uniStr(curr["name"]) + " from the queue.\nCurrent vote count: "
                + uniStr(len(curr["skips"])) + ", required vote count: " + uniStr(required) + "."
                )
        skipped = False
        q = _vars.queue[guild.id].queue
        i = 0
        while i < len(q):
            song = q[i]
            if len(song["skips"]) >= required or type(song["skips"]) is tuple:
                q.pop(i)
                response += "\n" + uniStr(song["name"]) + " has been removed from the queue."
                if i == 0:
                    print("Stopped audio playback in " + guild.name)
                    for vc in client.voice_clients:
                        if vc.channel.guild.id == guild.id:
                            vc.stop()
                continue
            i += 1
        return response + "```"


class pause:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["resume"]
        self.min_level = 1
        self.description = "Pauses or resumes audio playing."
        self.usage = ""

    async def __call__(self, _vars, name, guild, client, user, channel, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        auds.paused = name == "pause"
        return "```css\nSuccessfully " + name + "d audio playback in " + uniStr(guild.name) + ".```"


class dump:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Dumps or restores the currently playing audio queue state."
        self.usage = "<data:[]>"

    async def __call__(self, guild, channel, user, client, _vars, argv, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if not argv:
            q = copy.deepcopy(auds.queue)
            for e in q:
                e["id"] = e["id"].replace("@", "")
            d = {
                "stats": auds.stats,
                "queue": q,
                }
            return "Queue data for " + uniStr(guild.name) + ":\n```css\n" + json.dumps(d) + "\n```"
        try:
            opener = urlBypass()
            resp = opener.open(_vars.verifyURL(argv))
            rescode = resp.getcode()
            if rescode != 200:
                raise ConnectionError(rescode)
            s = resp.read().decode("utf-8")
            s = s[s.index("{"):]
            if s[-4:] == "\n```":
                s = s[:-4]
            d = json.loads(s)
        except:
            d = json.loads(argv)
        print("Stopped audio playback in " + guild.name)
        for vc in client.voice_clients:
            if vc.channel.guild.id == guild.id:
                vc.stop()
        auds.queue = d["queue"]
        auds.stats = d["stats"]
        return "```css\nSuccessfully reinstated audio queue for " + uniStr(guild.name) + ".```"
            

class volume:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["vol", "audio", "v"]
        self.min_level = 0
        self.description = "Changes the current playing volume in this server."
        self.usage = "<value> <reverb(?r)> <pitch(?p)> <bassboost(?b)> <delay(?d)>"

    async def __call__(self, client, channel, user, guild, _vars, flags, argv, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if "p" in flags:
            op = "pitch"
        elif "b" in flags:
            op = "bassboost"
        elif "d" in flags:
            op = "delay"
        elif "r" in flags:
            op = "reverb"
        else:
            op = "volume"
        if not len(argv.replace(" ", "")):
            num = round(100. * _vars.queue[guild.id].stats[op], 8)
            return (
                "```css\nCurrent audio " + op + " in " + uniStr(guild.name)
                + ": " + uniStr(num) + ".```"
                )
        s_perm = _vars.getPerms(user, guild)
        if s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to change volume. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(1) + "."
                )
        origVol = _vars.queue[guild.id].stats
        val = roundMin(float(_vars.evalMath(argv) / 100))
        orig = origVol[op]
        origVol[op] = val
        return (
            "```css\nChanged audio " + op + " in " + uniStr(guild.name)
            + " from " + uniStr(round(100. * orig, 8))
            + " to " + uniStr(round(100. * val, 8)) + ".```"
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

    def __init__(self):
        self.name = ["unmuteall"]
        self.min_level = 2
        self.description = "Disables server mute for all members."
        self.usage = "<link:[]> <verbose:(?v)>"

    async def __call__(self, guild, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                asyncio.create_task(user.edit(mute=False,deafen=False))
        return "```css\nSuccessfully unmuted all users in voice channels in " + uniStr(guild.name) + ".```"
