import youtube_dl, asyncio, discord, time, os, urllib, json, copy, traceback
from subprocess import check_output, CalledProcessError, STDOUT
from scipy.signal import butter, sosfilt
from smath import *


class customAudio(discord.AudioSource):
    
    length = 1920
    empty = numpy.zeros(length >> 1, float)
    bass = butter(2, 1/7, btype="low", output="sos")
    treble = butter(2, 1/7, btype="high", output="sos")
    filt = butter(1, 1/4, btype="low", output="sos")
    defaults = {
            "volume": 1,
            "reverb": 0,
            "pitch": 0,
            "speed": 1,
            "bassboost": 0,
            "chorus": 0,
            "loop": False,
            "shuffle": False,
            "quiet": False,
            "position": 0,
            }

    def __init__(self, channel, _vars):
        try:
            self.stats = dict(self.defaults)
            self.new()
            self.queue = []
            self.channel = channel
            self.buffer = []
            self.feedback = None
            self.bassadj = None
            self.prev = None
            self.searching = False
            self.preparing = False
            self.player = None
            self._vars = _vars
            _vars.queue[channel.guild.id] = self
        except:
            print(traceback.format_exc())

    def new(self, source=None, pos=0):
        self.reverse = self.stats["speed"] < 0
        self.speed = max(0.01, abs(self.stats["speed"]))
        if self.speed == 0.01:
            self.speed = 1
            self.paused = 2
        else:
            self.paused = False
        self.stats["position"] = pos
        self.is_playing = source is not None
        if getattr(self, "source", None) is not None:
            try:
                self.source.cleanup()
            except:
                print(traceback.format_exc())
        if source is not None:
            if not isValid(self.stats["pitch"] * self.stats["speed"] * self.stats["chorus"]):
                self.source = None
                self.file = None
                return
            d = {"source": source}
            pitchscale = 2 ** (self.stats["pitch"] / 12)
            chorus = min(32, abs(self.stats["chorus"]))
            if pitchscale != 1 or self.stats["speed"] != 1:
                speed = self.speed / pitchscale
                speed = max(0.005, speed)
                opts = ""
                while speed > 1.8:
                    opts += "atempo=1.8,"
                    speed /= 1.8
                while speed < 0.6:
                    opts += "atempo=0.6,"
                    speed /= 0.6
                opts += "atempo=" + str(speed)
                d["options"] = "-af " + opts
            else:
                d["options"] = ""
            if pitchscale != 1:
                d["options"] += ",asetrate=r=" + str(48000 * pitchscale)
            if self.reverse:
                d["options"] += ",areverse"
            if chorus:
                if not d["options"]:
                    d["options"] = "-af "
                else:
                    d["options"] += ","
                A = ""
                B = ""
                C = ""
                D = ""
                for i in range(ceil(chorus)):
                    neg = ((i & 1) << 1) - 1
                    i = 1 + i >> 1
                    i *= chorus / ceil(chorus)
                    if i:
                        A += "|"
                        B += "|"
                        C += "|"
                        D += "|"
                    delay = (25 + i * tau * neg) % 39 + 18
                    A += str(round(delay, 3))
                    decay = (0.125 + i * 0.03 * neg) % 0.25 + 0.25
                    B += str(round(decay, 3))
                    speed = (2 + i * 0.61 * neg) % 4.5 + 0.5
                    C += str(round(speed, 3))
                    depth = (1.5 + i * 0.43 * neg) % 4 + 0.5
                    D += str(round(depth, 3))
                b = 0.5 / sqrt(ceil(chorus + 1))
                d["options"] += (
                    "\"chorus=0.5:" + str(round(b, 3)) + ":"
                    + A + ":"
                    + B + ":"
                    + C + ":"
                    + D + "\""
                    )
            if pos != 0:
                d["before_options"] = "-ss " + str(pos)
            print(d)
            self.is_loading = True
            self.source = discord.FFmpegPCMAudio(**d)
            self.file = source
        else:
            self.source = None
            self.file = None
        self.is_loading = False
        self.stats["position"] = pos
        if pos == 0:
            if self.reverse and len(self.queue):
                self.stats["position"] = self.queue[0]["duration"]

    def seek(self, pos):
        duration = self.queue[0]["duration"]
        pos = max(0, pos)
        if pos >= duration:
            self.new()
            return duration
        self.new(self.file, pos)
        self.stats["position"] = pos
        return self.stats["position"]

    def advance(self, loop=True):
        q = self.queue
        if len(q):
            if self.stats["loop"]:
                temp = q[0]
            self.prev = q[0]["id"]
            q.pop(0)
            if self.stats["shuffle"]:
                if len(q):
                    t2 = q[0]
                    q.pop(0)
                    shuffle(q)
                    q.insert(0, t2)
            if self.stats["loop"] and loop:
                temp["id"] = temp["id"]
                if "download" in temp:
                    temp.pop("download")
                q.append(temp)
            self.preparing = False
            return len(q)
        if self.player:
            self.player["time"] = 0

    async def updatePlayer(self):
        curr = self.player
        self.stats["quiet"] &= 1
        if curr is not None:
            self.stats["quiet"] |= 2
            try:
                if not curr["message"].content:
                    raise EOFError
            except:
                self.player = None
                print(traceback.format_exc())
            if time.time() > curr["time"]:
                curr["time"] = inf
                try:
                    await self._vars.reactCallback(curr["message"], "❎", self._vars.client.user)
                except discord.NotFound:
                    self.player = None
                    print(traceback.format_exc())
        
    def read(self):
        try:
            if self.is_loading:
                self.is_playing = True
                raise EOFError
            if self.paused:
                self.is_playing = True
                raise EOFError
            temp = self.source.read()
            if not len(temp):
                sendUpdateRequest(True)
                raise EOFError
            self.stats["position"] = round(
                self.stats["position"] + self.speed / 50 * (self.reverse * -2 + 1), 4
                )
            self.is_playing = True
        except:
            if not self.paused and not self.is_loading:
                if self.is_playing:
                    self._vars.doUpdate |= True
                self.new()
            temp = numpy.zeros(self.length, numpy.uint16).tobytes()
        try:
            volume = self.stats["volume"]
            reverb = self.stats["reverb"]
            pitch = self.stats["pitch"]
            bassboost = self.stats["bassboost"]
            chorus = self.stats["chorus"]
            delay = 16 #min(400, max(2, round(sndset["reverbdelay"] * 5)))
            if volume == 1 and reverb == pitch == bassboost == chorus == 0:
                self.buffer = []
                self.feedback = None
                self.bassadj = None
                return temp
            array = numpy.frombuffer(temp, dtype=numpy.int16).astype(float)
            size = self.length >> 1
            if not isValid(volume * reverb * bassboost * pitch):
                array = numpy.random.rand(self.length) * 65536 - 32768
            elif volume != 1 or chorus:
                try:
                    array *= volume * (bool(chorus) + 1)
                except:
                    array = numpy.random.rand(self.length) * 65536 - 32768
            left, right = array[::2], array[1::2]
            if bassboost:
                try:
                    lbass = numpy.array(left)
                    rbass = numpy.array(right)
                    if self.bassadj is not None:
                        if bassboost > 0:
                            filt = self.bass
                        else:
                            filt = self.treble
                        left += sosfilt(filt, numpy.concatenate((self.bassadj[0], left)))[size-16:-16] * bassboost
                        right += sosfilt(filt, numpy.concatenate((self.bassadj[1], right)))[size-16:-16] * bassboost
                    self.bassadj = [lbass, rbass]
                except:
                    print(traceback.format_exc())
            else:
                self.bassadj = None
            if reverb:
                try:
                    if not len(self.buffer):
                        self.buffer = [[self.empty] * 2] * delay
                    r = 18
                    p1 = round(size * (0.5 - 2 / r))
                    p2 = round(size * (0.5 - 1 / r))
                    p3 = round(size * 0.5)
                    p4 = round(size * (0.5 + 1 / r))
                    p5 = round(size * (0.5 + 2 / r))
                    lfeed = (
                        + numpy.concatenate((self.buffer[0][0][p1:], self.buffer[1][0][:p1])) / 24
                        + numpy.concatenate((self.buffer[0][0][p2:], self.buffer[1][0][:p2])) / 12
                        + numpy.concatenate((self.buffer[0][0][p3:], self.buffer[1][0][:p3])) * 0.75
                        + numpy.concatenate((self.buffer[0][0][p4:], self.buffer[1][0][:p4])) / 12
                        + numpy.concatenate((self.buffer[0][0][p5:], self.buffer[1][0][:p5])) / 24
                        ) * reverb
                    rfeed = (
                        + numpy.concatenate((self.buffer[0][1][p1:], self.buffer[1][1][:p1])) / 24
                        + numpy.concatenate((self.buffer[0][1][p2:], self.buffer[1][1][:p2])) / 12
                        + numpy.concatenate((self.buffer[0][1][p3:], self.buffer[1][1][:p3])) * 0.75
                        + numpy.concatenate((self.buffer[0][1][p4:], self.buffer[1][1][:p4])) / 12
                        + numpy.concatenate((self.buffer[0][1][p5:], self.buffer[1][1][:p5])) / 24
                        ) * reverb
                    if self.feedback is not None:
                        left -= sosfilt(self.filt, numpy.concatenate((self.feedback[0], lfeed)))[size-16:-16]
                        right -= sosfilt(self.filt, numpy.concatenate((self.feedback[1], rfeed)))[size-16:-16]
                    self.feedback = (lfeed, rfeed)
                    #array = numpy.convolve(array, resizeVector(self.buffer[0], len(array) * 2))
                    a = 1 / 16
                    b = 1 - a
                    self.buffer.append((left * a + right * b, left * b + right * a))
                except:
                    print(traceback.format_exc())
                self.buffer = self.buffer[-delay:]
            else:
                self.buffer = []
                self.feedback = None
            array = numpy.stack((left, right), axis=-1).flatten()
            numpy.clip(array, -32767, 32767, out=array)
            temp = array.astype(numpy.int16).tobytes()
        except Exception as ex:
            print(traceback.format_exc())
        return temp

    def is_opus(self):
        return False

    def cleanup(self):
        if getattr(self, "source", None) is not None:
            return self.source.cleanup()


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
        auds.channel = channel
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
        fn = "cache/" + i["id"] + ".mp3"
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

    async def __call__(self, client, user, _vars, argv, channel, guild, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if auds.player:
            flags["h"] = 1
            try:
                await message.delete()
            except discord.NotFound:
                pass
        elapsed = auds.stats["position"]
        q = _vars.queue[guild.id].queue
        if not len(argv.replace(" ", "")):
            if not len(q):
                return "```css\nQueue for " + uniStr(guild.name) + " is currently empty. ```", 1
            if "v" in flags:
                if auds.stats["loop"]:
                    totalTime = inf
                else:
                    if auds.reverse and len(auds.queue):
                        totalTime = elapsed - auds.queue[0]["duration"]
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
                if auds.reverse and len(auds.queue):
                    estim = currTime + elapsed - auds.queue[0]["duration"]
                else:
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
            if auds.reverse and len(auds.queue):
                total_duration += elapsed - auds.queue[0]["duration"]
            else:
                total_duration -= elapsed
            total_duration = max(total_duration / auds.speed, dur / 128 + frand(0.5) + 2)
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
            _vars.queue[guild.id] = customAudio(channel, _vars)
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
                await vclient.disconnect(force=True)
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
        self.usage = "<0:queue_position[0]> <force(?f)> <vote(?v)> <hide(?h)>"

    async def __call__(self, client, user, _vars, args, argv, guild, flags, message, **void):
        found = False
        if guild.id not in _vars.queue:
            raise LookupError("Currently not playing in a voice channel.")
        auds = _vars.queue[guild.id]
        if auds.player:
            flags["h"] = 1
            try:
                await message.delete()
            except discord.NotFound:
                pass
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
                if "f" in flags or user.id == curr["u_id"] and not "v" in flags:
                    curr["skips"] = None
                elif user.id not in curr["skips"]:
                    curr["skips"].append(user.id)
            elif "v" in flags:
                curr["skips"] = [user.id]
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
        if not "h" in flags:
            return response + "```", 1


class pause:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["resume", "unpause"]
        self.min_level = 1
        self.description = "Pauses or resumes audio playing."
        self.usage = ""

    async def __call__(self, _vars, name, guild, client, user, channel, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if not auds.paused > 1:
            auds.paused = name == "pause"
        if auds.player:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
            return (
                "```css\nSuccessfully " + name + "d audio playback in "
                + uniStr(guild.name) + ".```"
                )


class seek:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Seeks to a position in the current audio file."
        self.usage = "<pos[0]>"

    async def __call__(self, argv, _vars, guild, client, user, channel, message, **void):
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
        if auds.player:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
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
                e["id"] = e["id"]
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
                if k in "loop shuffle quiet":
                    d["stats"][k] = bool(d["stats"][k])
                else:
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
            "<value[]> <volume()(?v)> <speed(?s)> <pitch(?p)> <bassboost(?b)> <reverb(?r)> <chorus(?c)>"
            + " <loop(?l)> <shuffle(?x)> <quiet(?q)> <disable_all(?d)>"
            )

    async def __call__(self, client, channel, user, guild, _vars, flags, argv, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if "d" in flags:
            op = None
        elif "v" in flags:
            op = "volume"
        elif "s" in flags:
            op = "speed"
        elif "p" in flags:
            op = "pitch"
        elif "b" in flags:
            op = "bassboost"
        elif "c" in flags:
            op = "chorus"
        elif "r" in flags:
            op = "reverb"
        elif "l" in flags:
            op = "loop"
        elif "x" in flags:
            op = "shuffle"
        elif "q" in flags:
            op = "quiet"
        else:
            op = "settings"
        if not argv and op is not None:
            if op == "settings":
                return (
                    "Current audio settings for **" + guild.name + "**:\n```json\n"
                    + str(auds.stats).replace("'", '"') + "```"
                    )
            orig = _vars.queue[guild.id].stats[op]
            if op in "loop shuffle quiet":
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
            if auds.player:
                try:
                    await message.delete()
                except discord.NotFound:
                    pass
            else:
                return (
                    "```css\nSuccessfully reset all audio settings for "
                    + uniStr(guild.name) + ".```"
                    )
        origVol = _vars.queue[guild.id].stats
        val = roundMin(float(_vars.evalMath(argv) / 100))
        orig = round(origVol[op] * 100, 9)
        new = round(val * 100, 9)
        if op in "loop shuffle quiet":
            origVol[op] = new = bool(val)
            orig = bool(orig)
        else:
            origVol[op] = val
        if op in "speed pitch chorus":
            auds.new(auds.file, auds.stats["position"])
        if auds.player:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
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


class player:
    is_command = True
    server_only = True
    time_consuming = True
    buttons = {
	"⏸️": 0,
	"🔄": 1,
	"🔀": 2,
	"⏮️": 3,
	"⏭️": 4,
        "🔊": 5,
        "🥁": 6,
        "📉": 7,
        "📊": 8,
        "⏪": 9,
        "⏩": 10,
        "⏫": 11,
        "⏬": 12,
        "♻️": 13,
	"⏏️": 14,
        "🚫": 15,
        }
    barsize = 28

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Creates a virtual audio player for the current server."
        self.usage = "<verbose:(?v)>"

    def showCurr(self, auds):
        q = auds.queue
        if q:
            output = "Playing " + uniStr(q[0]["name"]) + ", "
            output += uniStr(len(q)) + " item" + "s" * (len(q) != 1) + " total "
        else:
            output = "Queue is currently empty. "
        if auds.stats["loop"]:
            output += "🔄"
        if auds.stats["shuffle"]:
            output += "🔀"
        output += "\n"
        v = abs(auds.stats["volume"])
        if v <= 0.5:
            output += "🔉"
        elif v <= 1:
            output += "🔊"
        else:
            output += "📢"
        b = auds.stats["bassboost"]
        if abs(b) > 1 / 3:
            if b > 0:
                output += "🥁"
            else:
                output += "🎻"
        r = auds.stats["reverb"]
        if r:
            output += "📉"
        c = auds.stats["chorus"]
        if c:
            output += "📊"
        s = auds.stats["speed"]
        if s < 0:
            output += "⏪"
        elif s > 1:
            output += "⏩"
        elif s > 0 and s < 1:
            output += "🐌"
        p = auds.stats["pitch"]
        if p > 0:
            output += "⏫"
        elif p < 0:
            output += "⏬"
        output += "\n"
        if auds.paused or not auds.stats["speed"]:
            output += "⏸️"
        elif auds.stats["speed"] > 0:
            output += "▶️"
        else:
            output += "◀️"
        output += (
            " (" + uniStr(dhms(auds.stats["position"]))
            + "/" + uniStr(dhms(q[0]["duration"])) + ") "
            )
        sym = "⬜⬛"
        r = round(min(1, auds.stats["position"] / q[0]["duration"]) * self.barsize)
        output += sym[0] * r + sym[1] * (self.barsize - r)
        return output

    async def _callback_(self, message, guild, channel, reaction, _vars, perm, **void):
        if perm < 1 or not guild.id in _vars.queue:
            return
        auds = _vars.queue[guild.id]
        if reaction is not None and (auds.player is None or auds.player["message"].id != message.id):
            await message.clear_reactions()
            return
        orig = "\n".join(message.content.split("\n")[:1 + ("\n" == message.content[3])]) + "\n"
        if reaction is None:
            for b in self.buttons:
                await message.add_reaction(b)
        else:
            if type(reaction) is str:
                emoji = reaction
            else:
                emoji = reaction.emoji
            if emoji in self.buttons:
                i = self.buttons[emoji]
                if i == 0:
                    auds.paused ^= 1
                elif i == 1:
                    auds.stats["loop"] = bool(auds.stats["loop"] ^ 1)
                elif i == 2:
                    auds.stats["shuffle"] = bool(auds.stats["shuffle"] ^ 1)
                elif i == 3 or i == 4:
                    if i == 3:
                        pos = 0
                    else:
                        pos = inf
                    dur = auds.queue[0]["duration"]
                    auds.seek(pos)
                    auds.stats["position"] = dur
                elif i == 5:
                    v = abs(auds.stats["volume"])
                    if v < 0.25 or v >= 2:
                        v = 0.25
                    elif v < 0.5:
                        v = 0.5
                    elif v < 1:
                        v = 1
                    else:
                        v = 2
                    auds.stats["volume"] = v
                elif i == 6:
                    b = auds.stats["bassboost"]
                    if abs(b) < 1 / 3:
                        b = 1
                    elif b < 0:
                        b = 0
                    else:
                        b = -1
                    auds.stats["bassboost"] = b
                elif i == 7:
                    r = auds.stats["reverb"]
                    if r:
                        r = 0
                    else:
                        r = 0.5
                    auds.stats["reverb"] = r
                elif i == 8:
                    c = abs(auds.stats["chorus"])
                    if c:
                        c = 0
                    else:
                        c = 1 / 3
                    auds.stats["chorus"] = c
                    auds.new(auds.file, auds.stats["position"])
                elif i == 9 or i == 10:
                    s = (i * 2 - 19) * 2 / 11
                    auds.stats["speed"] = round(auds.stats["speed"] + s, 5)
                    auds.new(auds.file, auds.stats["position"])
                elif i == 11 or i == 12:
                    p = i * 2 - 23
                    auds.stats["pitch"] -= p
                    auds.new(auds.file, auds.stats["position"])
                elif i == 13:
                    pos = auds.stats["position"]
                    auds.stats = dict(auds.defaults)
                    auds.new(auds.file, pos)
                elif i == 14:
                    auds.dead = True
                    auds.player = None
                    await message.delete()
                    return
                else:
                    auds.player = None
                    await message.delete()
                    return
        text = orig + self.showCurr(auds) + "```"
        await message.edit(
            content=text,
            )
        maxdel = auds.queue[0]["duration"] - auds.stats["position"] + 1
        delay = max(6, min(maxdel, auds.queue[0]["duration"] / self.barsize / abs(auds.stats["speed"])))
        if auds.paused:
            delay = inf
        auds.player = {
            "time": time.time() + delay,
            "message": message,
            }
        #print(text)

    async def __call__(self, channel, user, client, _vars, flags, **void):
        auds = await forceJoin(channel.guild, channel, user, client, _vars)
        auds.stats["quiet"] |= 2
        text = (
            "```" + "\n" * ("v" in flags) + "callback-voice-player-_\n"
            + "Initializing virtual audio player...```"
            )
        auds.player = {
            "message": await channel.send(text),
            "time": inf,
            }
