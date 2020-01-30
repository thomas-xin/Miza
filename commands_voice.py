import youtube_dl, asyncio, discord
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
        "outtmpl": "/cache/temp.%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    def __init__(self):
        self.searcher = youtube_dl.YoutubeDL(self.ydl_opts_1)
        self.searcher.add_default_info_extractors()
        self.downloader = youtube_dl.YoutubeDL(self.ydl_opts_2)
        self.downloader.add_default_info_extractors()

    def search(self, item):
        try:
            return self.searcher.extract_info(item)
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

    def dictRemove(self, d, key):
        output = dict(d)
        try:
            key[0]
        except TypeError:
            key = [key]
        for k in key:
            try:
                output.pop(k)
            except KeyError:
                pass
        return output

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
                        await func(user=user, _vars=_vars, channel=channel, guild=guild)
                    except discord.ClientException:
                        pass
                    except AttributeError:
                        pass
        _vars.queue[guild.id]["channel"] = channel.id
        if not len(argv.replace(" ", "")):
            q = _vars.queue[guild.id]["queue"]
            if "v" in flags:
                show = "\n".join([str(i) + ": " + str(self.dictRemove(q[i], ("id", "skips"))) for i in range(len(q))])
            else:
                show = "\n".join([str(i) + ": " + str(q[i]["name"]) for i in range(len(q))])
            return (
                "Currently playing in " + uniStr(guild.name) + ":\n"
                + "```\n" + show + "```"
                )
        else:
            output = [None]
            doParallel(self.ytdl.search, [argv], output)
            while output[0] is None:
                await asyncio.sleep(0.2)
            res = output[0]
            #print(res)
            try:
                entries = res["entries"]
            except KeyError:
                if "uploader" in res:
                    entries = [res]
                else:
                    entries = []
            added = []
            names = []
            for e in entries:
                name = e["title"]
                url = e["webpage_url"]
                duration = e["duration"]
                v_id = e["id"]
                added.append({"name": name, "url": url, "duration": duration, "added by": user.name, "id": v_id, "skips": []})
                names.append(name)
            _vars.queue[guild.id]["queue"] += added
            if not len(names):
                raise EOFError("No results for " + str(argv) + ".")
            if "v" in flags:
                names = [self.dictRemove(i, "id") for i in added]
            elif len(names) == 1:
                names = names[0]
            return (
                "```\n🎶 Added " + str(names) + " to the queue! 🎶```"
                )
        return


class join:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["summon"]
        self.min_level = 0
        self.description = "Summons the bot into a voice channel."
        self.usage = ""

    async def __call__(self, user, _vars, channel, guild, **void):
        voice = user.voice
        vc = voice.channel
        await vc.connect(timeout=_vars.timeout, reconnect=True)
        if guild.id not in _vars.queue:
            _vars.queue[guild.id] = {"channel": channel.id, "queue": []}
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
        curr = _vars.queue[guild.id]["queue"][pos]
        if user.id not in curr["skips"]:
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
            response = "```\n" + uniStr(curr["name"]) + " has been forcefully removed from the queue."
        else:
            response = (
                "```\nSuccessfully voted to remove " + uniStr(curr["name"]) + " from the queue.\nCurrent vote count: "
                + uniStr(len(curr["skips"])) + ", required vote count: " + uniStr(required) + "."
                )
        skipped = False
        i = 0
        while i < len(_vars.queue[guild.id]["queue"]):
            song = _vars.queue[guild.id]["queue"][i]
            if len(song["skips"]) >= required or type(song["skips"]) is tuple:
                _vars.queue[guild.id]["queue"].pop(i)
                response += "\n" + uniStr(curr["name"]) + " has been removed from the queue."
                if i == 0:
                    print("Stopped audio playback in " + guild.name)
                    for vc in client.voice_clients:
                        if vc.channel.guild.id == guild.id:
                            vc.stop()
                continue
            i += 1
        return response + "```"
