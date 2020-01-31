import youtube_dl, asyncio, discord, time
from smath import *


class youtubeDownloader:
    ydl_opts_1 = {
        "quiet": 1,
        "simulate": 1,
        "default_search": "auto",
        "skip_download": 1,
        "writeinfojson": 1,
        }
    ydl_opts_2 = {
        "quiet": 1,
        "format": "bestaudio/best",
        "outtmpl": "/cache/%(id)s.mp3",
        }

    def __init__(self):
        self.searcher = youtube_dl.YoutubeDL(self.ydl_opts_1)
        self.searcher.add_default_info_extractors()
        self.downloader = youtube_dl.YoutubeDL(self.ydl_opts_2)
        self.downloader.add_default_info_extractors()

    def search(self, item):
        try:
            return self.searcher.extract_info(item.strip("<>"))
        except youtube_dl.utils.DownloadError:
            return {}
        
    def download(self, item):
        try:
            self.downloader.download([item])
        except youtube_dl.utils.DownloadError:
            pass


class queue:
    is_command = True
    server_only = True
    ytdl = youtubeDownloader()

    def __init__(self):
        self.name = ["q", "qlist", "play", "playing", "np"]
        self.min_level = 0
        self.description = "Shows the music queue, or plays a song in voice."
        self.usage = "<link:[]> <verbose:(?v)>"

    async def __call__(self, client, user, _vars, argv, channel, guild, flags, **void):
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
            _vars.queue[guild.id]["channel"] = channel.id
        except KeyError:
            raise KeyError("Voice channel not found.")
        if not len(argv.replace(" ", "")):
            q = _vars.queue[guild.id]["queue"]
            if not len(q):
                return "```\nQueue for " + uniStr(guild.name) + " is currently empty. ```"
            t = time.time()
            origTime = q[0].get("start_time", t)
            currTime = 0
            show = ""
            for i in range(len(q)):
                curr = "\n"
                e = q[i]
                curr += "【" + uniStr(i) + "】 "
                if "v" in flags:
                    curr += (
                        uniStr(e["name"]) + ", URL: " + e["url"]
                        + ", Duration: " + uniStr(" ".join(timeConv(e["duration"])))
                        + ", Added by: " + uniStr(e["added by"])
                        )
                else:
                    curr += limStr(uniStr(e["name"]), 48)
                estim = currTime + origTime - t
                currTime += e["duration"]
                if estim > 0:
                    curr += ", Time until playing: " + uniStr(" ".join(timeConv(estim)))
                else:
                    curr += ", Remaining time: " + uniStr(" ".join(timeConv(estim + e["duration"])))
                if len(show) + len(curr) < 1900:
                    show += curr
                else:
                    show += uniStr("\nAnd more...", 1)
                    break
            return (
                "Currently playing in **" + guild.name + "**:\n"
                + "```\n" + show + "```"
                )
        else:
            output = [None]
            doParallel(self.ytdl.search, [argv], output)
            while output[0] is None:
                await asyncio.sleep(0.01)
            res = output[0]
            #print(res)
            try:
                entries = res["entries"]
            except KeyError:
                if "uploader" in res:
                    entries = [res]
                else:
                    entries = []
            dur = 0
            added = []
            names = []
            for e in entries:
                name = e["title"]
                url = e["webpage_url"]
                duration = e["duration"]
                v_id = e["id"]
                added.append({
                    "name": name,
                    "url": url,
                    "duration": duration,
                    "added by": user.name,
                    "id": v_id,
                    "skips": [],
                    })
                if not dur:
                    dur = duration
                names.append(name)
            total_duration = 0
            for e in _vars.queue[guild.id]["queue"]:
                if "start_time" in e:
                    total_duration += e["duration"] + e["start_time"] - time.time()
                else:
                    total_duration += e["duration"]
            total_duration = max(total_duration, dur / 128 + frand(0.5) + 1.5)
            _vars.queue[guild.id]["queue"] += added
            if not len(names):
                raise EOFError("No results for " + str(argv) + ".")
            if "v" in flags:
                names = [subDict(i, "id") for i in added]
            elif len(names) == 1:
                names = names[0]
            if not "h" in flags:
                return (
                    "```\n🎶 Added " + str(names) + " to the queue! Estimated time until playing: "
                    + uniStr(" ".join(timeConv(total_duration))) + ". 🎶```"
                    )


class playlist:
    is_command = True
    server_only = True
    ytdl = youtubeDownloader()

    def __init__(self):
        self.name = ["defaultplaylist"]
        self.min_level = 2
        self.description = "Shows, appends, or removes from the default playlist."
        self.usage = "<link:[]> <remove:(?r)>"

    async def __call__(self, user, argv, _vars, guild, flags, **void):
        if not len(argv.replace(" ","")):
            if "r" in flags:
                _vars.playlists[guild.id] = []
                _vars.update()
                return "```\nRemoved all entries from the default playlist for " + uniStr(guild.name) + ".```"
            return (
                "```\nCurrent default playlist for " + uniStr(guild.name) + ": "
                + str(_vars.playlists.get(guild.id, [])) + ".```"
                )
        output = [None]
        doParallel(self.ytdl.search, [argv], output)
        while output[0] is None:
            await asyncio.sleep(0.01)
        res = output[0]
        #print(res)
        try:
            entries = res["entries"]
        except KeyError:
            if "uploader" in res:
                entries = [res]
            else:
                entries = []
        curr = _vars.playlists.get(guild.id, [])
        names = []
        for e in entries:
            name = e["title"]
            names.append(name)
            url = e["webpage_url"]
            if "r" in flags:
                for p in curr:
                    if p["url"] == url:
                        curr.remove(p)
                        break
            else:
                curr.append({
                    "name": e["title"],
                    "url": url,
                    "duration": e["duration"],
                    "id": e["id"],
                    })
        if len(names):
            _vars.playlists[guild.id] = curr
            _vars.update()
            if "r" in flags:
                return "```\nRemoved " + str(names) + " from the default playlist for " + uniStr(guild.name) + ".```"
            return "```\nAdded " + str(names) + " to the default playlist for " + uniStr(guild.name) + ".```"
        

class join:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["summon"]
        self.min_level = 0
        self.description = "Summons the bot into a voice channel."
        self.usage = ""

    async def __call__(self, client, user, _vars, channel, guild, **void):
        voice = user.voice
        vc = voice.channel
        try:
            joined = True
            await vc.connect(timeout=_vars.timeout, reconnect=True)
        except discord.ClientException:
            joined = False
        for user in guild.members:
            if user.id == client.user.id:
                asyncio.create_task(user.edit(mute=False,deafen=False))
        if guild.id not in _vars.queue:
            _vars.queue[guild.id] = {"channel": channel.id, "queue": []}
        if joined:
            return (
                "```\n🎵 Successfully connected to " + uniStr(vc.name)
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
        found = False
        for vclient in client.voice_clients:
            if guild.id == vclient.channel.guild.id:
                _vars.queue.pop(guild.id)
                await vclient.disconnect(force=False)
                return "```\n🎵 Successfully disconnected from " + uniStr(guild.name) + ". 🎵```"
        raise EOFError("Unable to find connected channel.")


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
            raise EOFError("Currently not playing in a voice channel.")
        s_perm = _vars.getPerms(user, guild)
        if "f" in flags and s_perm < 2:
            raise PermissionError(
                "Insufficient permissions to force skip. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(2) + "."
                )
        if not len(argv.replace(" ","")):
            pos = 0
        else:
            pos = _vars.evalMath(argv)
        try:
            if not isValid(pos):
                if "f" in flags:
                    _vars.queue[guild.id]["queue"] = []
                    print("Stopped audio playback in " + guild.name)
                    for vc in client.voice_clients:
                        if vc.channel.guild.id == guild.id:
                            vc.stop()
                    return "```\nRemoved all items from the queue.```"
                raise LookupError
            curr = _vars.queue[guild.id]["queue"][pos]
        except LookupError:
            raise IndexError("Entry " + uniStr(pos) + " is out of range.")
        if user.id not in curr["skips"] and type(curr["skips"]) is not tuple:
            if "f" in flags:
                curr["skips"] = ()
            else:
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
                "```\nVoted to remove " + uniStr(curr["name"]) + " from the queue.\nCurrent vote count: "
                + uniStr(len(curr["skips"])) + ", required vote count: " + uniStr(required) + "."
                )
        skipped = False
        i = 0
        while i < len(_vars.queue[guild.id]["queue"]):
            song = _vars.queue[guild.id]["queue"][i]
            if len(song["skips"]) >= required or type(song["skips"]) is tuple:
                _vars.queue[guild.id]["queue"].pop(i)
                response += "\n" + uniStr(song["name"]) + " has been removed from the queue."
                if i == 0:
                    print("Stopped audio playback in " + guild.name)
                    for vc in client.voice_clients:
                        if vc.channel.guild.id == guild.id:
                            vc.stop()
                continue
            i += 1
        return response + "```"


class volume:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["vol"]
        self.min_level = 0
        self.description = "Changes the current playing volume in this server."
        self.usage = "<volume>"

    async def __call__(self, user, guild, _vars, argv, **void):
        if not len(argv.replace(" ", "")):
            return (
                "```\nCurrent playing volume in " + uniStr(guild.name)
                + ": " + uniStr(100. * _vars.volumes.get(guild.id, 1)) + ".```"
                )
        s_perm = _vars.getPerms(user, guild)
        if s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to change volume. Current permission level: " + uniStr(s_perm)
                + ", required permission level: " + uniStr(1) + "."
                )
        origVol = _vars.volumes.get(guild.id, 1)
        val = roundMin(float(_vars.evalMath(argv) / 100))
        _vars.volumes[guild.id] = val
        return (
            "```\nChanged playing volume in " + uniStr(guild.name)
            + " from " + uniStr(100. * origVol)
            + " to " + uniStr(100. * val) + ".```"
            )


class unmute:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["unmuteall"]
        self.min_level = 3
        self.description = "Disables server mute for all members."
        self.usage = "<link:[]> <verbose:(?v)>"

    async def __call__(self, guild, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                asyncio.create_task(user.edit(mute=False,deafen=False))
        return "```\nSuccessfully unmuted all users in voice channels in " + uniStr(guild.name) + ".```"
