import discord, urllib, json, youtube_dl
from smath import *

youtube_dl.__builtins__["print"] = print


class customAudio(discord.AudioSource):
    
    length = 1920
    empty = numpy.zeros(length >> 1, float)
    filt = signal.butter(1, 0.125, btype="low", output="sos")
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
            self.paused = False
            self.stats = dict(self.defaults)
            self.new()
            self.queue = hlist()
            self.channel = channel
            self.buffer = []
            self.feedback = None
            self.bassadj = None
            self.prev = None
            self.searching = False
            self.preparing = False
            self.player = None
            self._vars = _vars
            _vars.updaters["playlists"].audio[channel.guild.id] = self
        except:
            print(traceback.format_exc())

    def __str__(self):
        classname = self.__class__.replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return (
            "<" + classname + " object at " + hex(id(self)).upper() + ">: {"
            + "\"queue\": " + str(self.queue)
            + ", \"stats\": " + str(self.stats)
            + ", \"player\": " + str(self.player)
            + "}"
        )

    def new(self, source=None, pos=0):
        self.reverse = self.stats["speed"] < 0
        self.speed = max(0.005, abs(self.stats["speed"]))
        if self.speed == 0.005:
            self.speed = 1
            self.paused |= 2
        else:
            self.paused &= -3
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
                #br = getBitrate(source)
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
            d["options"] = d["options"].strip(" ")
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
        if self.source is not None and self.player:
            self.player["time"] = 2

    def seek(self, pos):
        duration = self.queue[0]["duration"]
        pos = max(0, pos)
        if pos >= duration:
            self.new()
            return duration
        self.new(self.file, pos)
        self.stats["position"] = pos
        return self.stats["position"]

    def advance(self, loop=True, shuffled=True):
        q = self.queue
        if q:
            if self.stats["loop"]:
                temp = q[0]
            self.prev = q[0]["id"]
            q.pop(0)
            if shuffled and self.stats["shuffle"]:
                if len(q) > 1:
                    temp = q.popleft()
                    shuffle(q)
                    q.appendleft(temp)
            if self.stats["loop"] and loop:
                temp["id"] = temp["id"]
                if "download" in temp:
                    temp.pop("download")
                q.append(temp)
            self.preparing = False
            self.queue = q
        if self.player:
            self.player["time"] = 2

    async def updatePlayer(self):
        curr = self.player
        self.stats["quiet"] &= -3
        if curr is not None:
            if curr["type"]:
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
        q = self.stats["quiet"]
        if q == bool(q):
            self.stats["quiet"] = bool(q)
        
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
                updateQueues.sendUpdateRequest(self, force=True)
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
            delay = 16
            if volume == 1 and reverb == pitch == bassboost == chorus == 0:
                self.buffer = []
                self.feedback = None
                self.bassadj = None
                return temp
            array = numpy.frombuffer(temp, dtype=numpy.int16).astype(float)
            size = self.length >> 1
            if abs(volume) > 1 << 32:
                volume = nan
            if abs(reverb) > 1 << 32:
                reverb = nan
            if abs(bassboost) > 1 << 32:
                bassboost = nan
            if abs(pitch) > 1 << 32:
                pitch = nan
            if not isValid(volume * reverb * bassboost * pitch):
                array = numpy.random.rand(self.length) * 65536 - 32768
            elif volume != 1 or chorus:
                try:
                    array *= volume * 2 * (bool(chorus) + 1)
                except:
                    array = numpy.random.rand(self.length) * 65536 - 32768
            left, right = array[::2], array[1::2]
            if bassboost:
                try:
                    lbass = numpy.array(left)
                    rbass = numpy.array(right)
                    if self.bassadj is not None:
                        x = float(sqrt(abs(bassboost)))
                        f = min(8, 2 + round(x))
                        if bassboost > 0:
                            g = max(1 / 128, (1 - x / 64) / 9)
                            filt = signal.butter(f, g, btype="low", output="sos")
                        else:
                            g = min(127 / 128, (1 + x / 64) / 9)
                            filt = signal.butter(f, g, btype="high", output="sos")
                        left += signal.sosfilt(
                            filt,
                            numpy.concatenate((self.bassadj[0], left))
                        )[size-16:-16] * bassboost
                        right += signal.sosfilt(
                            filt,
                            numpy.concatenate((self.bassadj[1], right))
                        )[size-16:-16] * bassboost
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
                        left -= signal.sosfilt(self.filt, numpy.concatenate((self.feedback[0], lfeed)))[size-16:-16]
                        right -= signal.sosfilt(self.filt, numpy.concatenate((self.feedback[1], rfeed)))[size-16:-16]
                    self.feedback = (lfeed, rfeed)
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


async def createPlayer(auds, p_type=0, verbose=False):
    auds.stats["quiet"] |= 2 * p_type
    text = (
        "```" + "\n" * verbose + "callback-voice-player-" + str(int(bool(p_type)))
        + "\nInitializing virtual audio player...```"
    )
    await auds.channel.send(text)
    await auds.updatePlayer()


def getDuration(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]
    try:
        output = subprocess.check_output(command).decode()
    except:
        print(traceback.format_exc())
        output = "N/A"
    try:
        i = output.index("\r")
        output = output[:i]
    except ValueError:
        output = "N/A"
    if output == "N/A":
        n = 0
    else:
        n = roundMin(float(output))
    return max(1 / (1 << 24), n)


def getBitrate(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]
    try:
        output = subprocess.check_output(command).decode()
    except:
        print(traceback.format_exc())
        output = "N/A"
    print(output)
    try:
        i = output.index("\r")
        output = output[:i]
    except ValueError:
        output = "N/A"
    if output == "N/A":
        n = 0
    else:
        n = roundMin(float(output))
    return max(1 / (1 << 24), n)


async def forceJoin(guild, channel, user, client, _vars):
    found = False
    if guild.id not in _vars.updaters["playlists"].audio:
        for func in _vars.categories["voice"]:
            if "Join" in func.name:
                try:
                    await func(client=client, user=user, _vars=_vars, channel=channel, guild=guild)
                except discord.ClientException:
                    pass
                except AttributeError:
                    pass
    try:
        auds = _vars.updaters["playlists"].audio[guild.id]
        auds.channel = channel
    except KeyError:
        raise LookupError("Voice channel not found.")
    return auds

    
class videoDownloader:
    
    opener = urlBypass()
    
    ydl_opts = {
        "quiet": 1,
        "format": "bestaudio/best",
        "call_home": 1,
        "nooverwrites": 1,
        "noplaylist": 1,
        "ignoreerrors": 0,
        "source_address": "0.0.0.0",
        "default_search": "auto",
    }

    def __init__(self):
        self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
        self.lastsearch = 0
        self.requests = 0

    def extract(self, item, force):
        try:
            output = []
            r = None
            if "open.spotify.com" in item or force == "spotify":
                try:
                    op = urlBypass()
                    for i in range(4):
                        try:
                            r = op.open(item)
                            rescode = r.getcode()
                            if rescode != 200:
                                raise ConnectionError(rescode)
                            continue
                        except:
                            time.sleep(1)
                    if force == "spotify":
                        time.sleep(frand(2))
                        it = '<meta property="music:duration" content="'
                        s = ""
                        while not it in s:
                            s += r.read(256).decode("utf-8")
                        s += r.read(256).decode("utf-8")
                        temp = s[s.index(it) + len(it):]
                        duration = temp[:temp.index('" />')]
                        t = '<meta name="description" content="'
                        s = s[s.index(t) + len(t):]
                        item = htmlDecode(s[:s.index('" />')]).replace(", a song by ", " ~ ").replace(" on Spotify", "").strip(" ")
                        temp = {
                            "id": hex(abs(hash(item))).replace("0x", ""),
                            "name": item,
                            "url": "ytsearch: " + item,
                            "duration": int(duration),
                            "research": True,
                            }
                        sys.stdout.write(repr(temp) + "\n")
                        output.append(temp)
                    else:
                        it = '<meta property="og:type" content="'
                        s = ""
                        while not it in s:
                            s += r.read(512).decode("utf-8")
                        s += r.read(256).decode("utf-8")
                        temp = s[s.index(it) + len(it):]
                        ptype = temp[:temp.index('" />')]
                        sys.stdout.write(ptype + "\n")
                        if "album" in ptype or "playlist" in ptype:
                            s += r.read().decode("utf-8")
                            output = []
                            while s:
                                try:
                                    i = s.index('<meta property="music:song')
                                    s = s[i:]
                                    it = 'content="'
                                    i = s.index(it) + len(it)
                                    s = s[i:]
                                    if s[:5] == ":disc":
                                        continue
                                    x = s[:s.index('" />')]
                                    if len(x) > 12:
                                        output.append([None])
                                        doParallel(self.extract, [x, "spotify"], output[-1], state=2)
                                except ValueError:
                                    break
                            while [None] in output:
                                time.sleep(0.1)
                            outlist = [i[-1] for i in output if i[-1] != 0]
                            output = []
                            for i in outlist:
                                output += i
                            sys.stdout.write(repr(output) + "\n\n")
                        else:
                            t = '<meta name="description" content="'
                            s = s[s.index(t) + len(t):]
                            item = htmlDecode(s[:s.index('" />')]).replace(" on Spotify", "")
                            sys.stdout.write(item + "\n")
                except urllib.error.URLError:
                    pass
            if r is not None:
                r.close()
            if not len(output) and force != "spotify":
                resp = self.downloader.extract_info(item, download=False, process=False)
                if resp.get("_type", None) == "url":
                    resp = self.downloader.extract_info(resp["url"], download=False, process=False)
                if resp is None or not len(resp):
                    raise EOFError("No search results found.")
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
                            try:
                                found = "duration" in entry
                                if "title" in entry:
                                    title = entry["title"]
                                else:
                                    title = entry["url"].split("/")[-1]
                                    found = False
                                temp = {
                                    "id": entry["id"],
                                    "name": title,
                                    "url": entry["url"],
                                    "duration": entry.get("duration", 60),
                                }
                                if not found:
                                    temp["research"] = True
                                output.append(temp)
                            except:
                                print(traceback.format_exc())
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
        except:
            if force != "spotify":
                raise
            print(traceback.format_exc())
            return 0

    def search(self, item, force=False):
        item = item.strip("< >\r\n\t")
        while self.requests > 4:
            time.sleep(0.1)
        if time.time() - self.lastsearch > 86400:
            self.lastsearch = time.time()
            self.searched = {}
        self.lastsearch = time.time()
        if item in self.searched:
            return self.searched[item]
        try:
            self.requests += 1
            self.searched[item] = output = self.extract(item, force)
            self.requests = max(self.requests - 1, 0)
            return output
        except Exception as ex:
            print(traceback.format_exc())
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
            time.sleep(0.1)
        if time.time() - self.lastsearch > 1800:
            self.lastsearch = time.time()
            self.searched = {}
        self.lastsearch = time.time()
        if item in self.searched:
            it = self.searched[item][-1]
            i["name"] = it["name"]
            i["duration"] = it["duration"]
            i["url"] = it["url"]
            return True
        try:
            self.requests += 1
            data = self.downloader.extract_info(item, download=False, process=True)
            if "entries" in data:
                data = data["entries"][-1]
            self.searched[item] = data = [{
                "id": data["id"],
                "name": data["title"],
                "duration": data["duration"],
                "url": data["webpage_url"],
            }]
            it = data[-1]
            i["name"] = it["name"]
            i["duration"] = it["duration"]
            i["url"] = it["url"]
            self.requests = max(self.requests - 1, 0)
        except:
            self.requests = max(self.requests - 1, 0)
            i["id"] = ""
            print(traceback.format_exc())
        return True

    def getDuration(self, filename):
        return getDuration(filename)


async def downloadTextFile(url):
    
    def dreader(file):
        try:
            s = resp.read().decode("utf-8")
            resp.close()
            return [s]
        except Exception as ex:
            print(traceback.format_exc())
            return repr(ex)

    resp = urlOpen(url)
    returns = [None]
    doParallel(dreader, [resp], returns)
    while returns[0] is None:
        await asyncio.sleep(0.3)
    resp = returns[0]
    if type(resp) is str:
        raise eval(resp)
    return resp[0]


ytdl = videoDownloader()


class Queue:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Q", "Play", "Playing", "NP", "P"]
        self.min_level = 0
        self.description = "Shows the music queue, or plays a song in voice."
        self.usage = "<link[]> <verbose(?v)> <hide(?h)>"

    async def __call__(self, client, user, _vars, argv, channel, guild, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if auds.stats["quiet"] & 2:
            flags["h"] = 1
            try:
                await message.delete()
            except discord.NotFound:
                pass
        elapsed = auds.stats["position"]
        q = auds.queue
        if not len(argv.replace(" ", "")):
            v = "v" in flags
            if not len(q):
                return "```css\nQueue for " + uniStr(guild.name) + " is currently empty. ```", 1
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
                uniStr(cnt) + " item" + "s" * (cnt != 1) + ", estimated total duration: "
                + uniStr(sec2Time(totalTime / auds.speed)) + "\n"
            )
            duration = q[0]["duration"]
            sym = "⬜⬛"
            barsize = 24
            r = round(min(1, elapsed / duration) * barsize)
            bar = sym[0] * r + sym[1] * (barsize - r)
            countstr = "Currently playing [" + q[0]["name"] + "](" + q[0]["url"] + ")\n"
            countstr += (
                "`(" + uniStr(dhms(elapsed))
                + "/" + uniStr(dhms(duration)) + ") "
            )
            countstr += bar + "`\n"
            embed=discord.Embed(
                title=" ",
                description=info + countstr,
                colour=_vars.randColour(),
            )
            embed.set_author(name="Queue for " + uniStr(guild.name) + ":")
            embstr = ""
            embcnt = 0
            currTime = 0
            for i in range(len(q)):
                if i >= len(q):
                    break
                e = q[i]
                curr = "`"
                curr += " " * (int(math.log10(len(q))) - int(math.log10(max(1, i))))
                curr += "【" + uniStr(i) + "】` "
                curr += "[" + limStr(noHighlight(e["name"]), 64 + 192 * v) + "](" + e["url"] + ")```css\n"
                if v:
                    curr += (
                        "Duration: " + uniStr(sec2Time(e["duration"]))
                        + ", Added by: " + uniStr(e["added by"]) + "\n"
                    )
                if auds.reverse and len(auds.queue):
                    estim = currTime + elapsed - auds.queue[0]["duration"]
                else:
                    estim = currTime - elapsed
                if estim > 0:
                    curr += "Time until playing: "
                    estimate = uniStr(sec2Time(estim / auds.speed))
                    if i <= 1 or not auds.stats["shuffle"]:
                        curr += estimate
                    else:
                        curr += "(" + estimate + ")"
                else:
                    curr += "Remaining time: " + uniStr(sec2Time((estim + e["duration"]) / auds.speed))
                curr += "```\n"
                if len(embstr) + len(curr) < 1024:
                    embstr += curr
                elif embcnt < v * 4:
                    embed.add_field(
                        name="Page " + uniStr(1 + embcnt),
                        value=embstr,
                        inline=False,
                    )
                    embcnt += 1
                    embstr = curr
                else:
                    embed.set_footer(
                        text=uniStr("And " + str(len(q) - i) + " more...", 1),
                    )
                    break
                if i <= 1 or not auds.stats["shuffle"]:
                    currTime += e["duration"]
            embed.add_field(
                name="Page " + uniStr(1 + embcnt),
                value=embstr,
                inline=False,
            )
            return {
                "embed": embed,
            }
        else:
            auds.preparing = True
            output = [None]
            doParallel(ytdl.search, [argv], output)
            if not "h" in flags:
                await channel.trigger_typing()
            while output[0] is None:
                await asyncio.sleep(0.3)
            res = output[0]
            if type(res) is str:
                raise ConnectionError(res)
            dur = 0
            added = deque()
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
                total_duration += elapsed - q[0]["duration"]
            else:
                total_duration -= elapsed
            total_duration = max(total_duration / auds.speed, float(dur / 128 + frand(0.5) + 2))
            if auds.stats["shuffle"]:
                added = shuffle(added)
            auds.queue.extend(added)
            if not len(names):
                raise EOFError("No results for " + str(argv) + ".")
            if "v" in flags:
                names = uniStr(hlist(subDict(i, "id") for i in added))
            elif len(names) == 1:
                names = uniStr(names[0])
            elif len(names) >= 4:
                names = uniStr(len(names)) + " items"
            if not "h" in flags:
                return (
                    "```css\n🎶 Added " + noHighlight(names)
                    + " to the queue! Estimated time until playing: "
                    + uniStr(sec2Time(total_duration)) + ". 🎶```", 1
                )


class Playlist:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["DefaultPlaylist", "PL"]
        self.min_level = 0
        self.description = "Shows, appends, or removes from the default playlist."
        self.usage = "<link[]> <remove(?d)> <verbose(?v)>"

    async def __call__(self, user, argv, guild, flags, channel, perm, **void):
        update = self.data["playlists"].update
        _vars = self._vars
        pl = _vars.data["playlists"]
        if argv or "d" in flags:
            req = 2
            if perm < req:
                raise PermissionError(
                    "Insufficient privileges to modify default playlist for "
                    + uniStr(guild.name) + ". Required level: "
                    + uniStr(req) + ", Current level: " + uniStr(perm) + "."
                )
        pl = pl.setdefault(guild.id, [])
        if not argv:
            if "d" in flags:
                pl[guild.id] = []
                update()
                return (
                    "```css\nRemoved all entries from the default playlist for "
                    + uniStr(guild.name) + ".```"
                )
            if "v" in flags:
                return (
                    "Current default playlist for **" + guild.name + "**: ```json\n"
                    + str(pl).replace("'", '"') + "```"
                )
            else:
                items = []
                for i in pl:
                    items.append(limStr(noHighlight(i["name"]), 32))
                s = ""
                for i in range(len(items)):
                    s += " " * (int(math.log10(len(items))) - int(math.log10(max(1, i))))
                    s += "[" + uniStr(i) + "] "
                    s += items[i] + "\n"
            if not s:
                return (
                    "```css\nDefault playlist for " + uniStr(guild.name)
                    + " is currently empty.```"
                )
            return (
                "Current default playlist for **" + guild.name + "**: ```ini\n"
                + s + "```"
            )
        if "d" in flags:
            i = await _vars.evalMath(argv, guild.id)
            temp = pl[i]
            pl.pop(i)
            update()
            return (
                "```css\nRemoved " + uniStr(noHighlight(temp["name"]))
                + " from the default playlist for "
                + uniStr(guild.name) + ".```"
            )
        if len(pl) >= 128:
            raise OverflowError(
                "Playlist size for " + uniStr(guild.name)
                + " has reached the maximum of 128 items. "
                + "Please remove an item to add another."
            )
        output = [None]
        doParallel(ytdl.search, [argv, True], output)
        await channel.trigger_typing()
        while output[0] is None:
            await asyncio.sleep(0.3)
        res = output[0]
        if type(res) is str:
            raise ConnectionError(res)
        names = []
        for e in res:
            name = e["name"]
            names.append(noHighlight(name))
            pl.append({
                "name": name,
                "url": e["url"],
                "duration": e["duration"],
                "id": e["id"],
            })
        if len(names):
            pl.sort(key=lambda x: x["name"].lower())
            update()
            return (
                "```css\nAdded " + uniStr(names)
                + " to the default playlist for "
                + uniStr(guild.name) + ".```"
            )
        

class Join:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Summon", "Connect"]
        self.min_level = 0
        self.description = "Summons the bot into a voice channel."
        self.usage = ""

    async def __call__(self, client, user, _vars, channel, guild, **void):
        voice = user.voice
        vc = voice.channel
        if guild.id not in _vars.updaters["playlists"].audio:
            await channel.trigger_typing()
            _vars.updaters["playlists"].audio[guild.id] = customAudio(channel, _vars)
        try:
            joined = True
            await vc.connect(timeout=30, reconnect=True)
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


class Leave:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = ["Quit", "DC", "Disconnect"]
        self.min_level = 1
        self.description = "Leaves a voice channel."
        self.usage = ""

    async def __call__(self, user, client, _vars, guild, **void):
        error = None
        try:
            _vars.updaters["playlists"].audio.pop(guild.id)
        except KeyError:
            error = LookupError("Unable to find connected channel.")
        found = False
        for vclient in client.voice_clients:
            if guild.id == vclient.channel.guild.id:
                await vclient.disconnect(force=True)
                return (
                    "```css\n🎵 Successfully disconnected from " 
                    + uniStr(guild.name) + ". 🎵```", 1
                )
        error = LookupError("Unable to find connected channel.")
        if error is not None:
            raise error


class Skip:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Remove", "Rem", "S"]
        self.min_level = 0
        self.description = "Removes an entry from the voice channel queue."
        self.usage = "<0:queue_position[0]> <force(?f)> <vote(?v)> <hide(?h)>"

    async def __call__(self, client, user, _vars, args, argv, guild, flags, message, **void):
        found = False
        if guild.id not in _vars.updaters["playlists"].audio:
            raise LookupError("Currently not playing in a voice channel.")
        auds = _vars.updaters["playlists"].audio[guild.id]
        if auds.stats["quiet"] & 2:
            flags["h"] = 1
            try:
                await message.delete()
            except discord.NotFound:
                pass
        s_perm = _vars.getPerms(user, guild)
        min_level = 1
        if "f" in flags and s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to force skip. Current permission level: "
                + uniStr(s_perm) + ", required permission level: "
                + uniStr(min_level) + "."
            )
        if not argv:
            elems = [0]
        elif ":" in argv or ".." in argv:
            l = argv.replace("...", ":").replace("..", ":").split(":")
            it = None
            if len(l) > 3:
                raise ValueError("Too many arguments for range input.")
            elif len(l) > 2:
                num = await _vars.evalMath(l[0], guild.id)
                it = int(round(float(num)))
            if l[0]:
                num = await _vars.evalMath(l[0], guild.id)
                if num > len(auds.queue):
                    num = len(auds.queue)
                else:
                    num = round(num) % len(auds.queue)
                left = num
            else:
                left = 0
            if l[1]:
                num = await _vars.evalMath(l[1], guild.id)
                if num > len(auds.queue):
                    num = len(auds.queue)
                else:
                    num = round(num) % len(auds.queue)
                right = num
            else:
                right = len(auds.queue)
            elems = xrange(left, right, it)
        else:
            elems = [0 for i in args]
            for i in range(len(args)):
                elems[i] = await _vars.evalMath(args[i], guild.id)
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
                        auds.queue.clear()
                        auds.new()
                        return "```fix\nRemoved all items from the queue.```"
                    raise LookupError
                curr = auds.queue[pos]
            except LookupError:
                response += repr(IndexError("Entry " + uniStr(pos) + " is out of range."))
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
                if len(response) > 1200:
                    response = limStr(response, 1200)
                else:
                    response += (
                        "Voted to remove " + uniStr(noHighlight(curr["name"]))
                        + " from the queue.\nCurrent vote count: "
                        + uniStr(len(curr["skips"])) + ", required vote count: "
                        + uniStr(required) + ".\n"
                    )
        pops = deque()
        count = 0
        i = 1
        while i < len(auds.queue):
            q = auds.queue
            song = q[i]
            if song["skips"] is None or len(song["skips"]) >= required:
                if count <= 3:
                    q.pop(i)
                else:
                    pops.append(i)
                    i += 1
                if count < 4:
                    response += (
                        uniStr(noHighlight(song["name"]))
                        + " has been removed from the queue.\n"
                    )
                count += 1
            else:
                i += 1
        auds.queue.pops(pops)
        if auds.queue:
            song = auds.queue[0]
            if song["skips"] is None or len(song["skips"]) >= required:
                doParallel(auds.advance, [False, not count])
                auds.new()
                response += (
                    uniStr(noHighlight(song["name"]))
                    + " has been removed from the queue.\n"
                )
                count += 1
        if not "h" in flags:
            if count >= 4:
                return (
                    "```css\n" + uniStr(count)
                    + " items have been removed from the queue.```"
                )
            return response + "```", 1


class Pause:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Resume", "Unpause", "Stop"]
        self.min_level = 1
        self.description = "Pauses, stops, or resumes audio playing."
        self.usage = ""

    async def __call__(self, _vars, name, guild, client, user, channel, message, **void):
        name = name.lower()
        auds = await forceJoin(guild, channel, user, client, _vars)
        if name == "stop":
            auds.seek(0)
        if not auds.paused > 1:
            auds.paused = name in ("pause", "stop")
        if auds.player is not None:
            auds.player["time"] = 1
        if auds.stats["quiet"] & 2:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
            past = name + "pe" * (name == "stop") + "d"
            return (
                "```css\nSuccessfully " + past + " audio playback in "
                + uniStr(guild.name) + ".```"
            )


class Seek:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Seeks to a position in the current audio file."
        self.usage = "<position[0]>"

    async def __call__(self, argv, _vars, guild, client, user, channel, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        pos = 0
        if argv:
            data = argv.split(":")
            mult = 1
            while len(data):
                pos += await _vars.evalMath(data[-1], guild.id) * mult
                data = data[:-1]
                if mult <= 60:
                    mult *= 60
                elif mult <= 3600:
                    mult *= 24
                elif len(data):
                    raise ValueError("Too many time arguments.")
        pos = auds.seek(pos)
        if auds.player is not None:
            auds.player["time"] = 1
        if auds.stats["quiet"] & 2:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
            return (
                "```css\nSuccessfully moved audio position to "
                + uniStr(sec2Time(pos)) + ".```"
            )


def getDump(auds, guild):
    try:
        lim = 32767
        if len(auds.queue) > lim:
            raise OverflowError(
                "Too many items in queue (" + uniStr(len(auds.queue))
                + " > " + uniStr(lim) + ")."
            )
        q = copy.deepcopy(list(auds.queue))
        s = copy.deepcopy(auds.stats)
        for e in q:
            if "download" in e:
                e.pop("download")
            e.pop("added by")
            e.pop("u_id")
            e.pop("skips")
            if random.random() > 0.99:
                time.sleep(0.001)
        d = {
            "stats": s,
            "queue": q,
        }
        d["stats"].pop("position")
        return ["Queue data for **" + guild.name + "**:\n```json\n" + json.dumps(d) + "\n```"]
    except Exception as ex:
        print(traceback.format_exc())
        return repr(ex)


class Dump:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Dumps or loads the currently playing audio queue state."
        self.usage = "<data{attached_file}> <append(?a)> <hide(?h)>"

    async def __call__(self, guild, channel, user, client, _vars, argv, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if not argv and not len(message.attachments):
            returns = [None]
            doParallel(getDump, [auds, guild], returns)
            while returns[0] is None:
                await asyncio.sleep(0.3)
            resp = returns[0]
            if type(resp) is str:
                raise eval(resp)
            return resp[0]
        try:
            if len(message.attachments):
                url = message.attachments[0].url
            else:
                url = verifyURL(argv)
            s = await downloadTextFile(url)
            s = s[s.index("{"):]
            if s[-4:] == "\n```":
                s = s[:-4]
        except:
            s = argv
            print(traceback.format_exc())
        d = json.loads(s.strip("\n"))
        q = d["queue"]
        for e in q:
            e["added by"] = user.name
            e["u_id"] = user.id
            e["skips"] = []
        if auds.player is not None:
            auds.player["time"] = 1
        if auds.stats["shuffle"]:
            shuffle(q)
        if not "a" in flags:
            auds.new()
            del auds.queue
            auds.queue = hlist(q)
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
        if len(auds.queue) > 1000:
            doParallel(auds.queue.extend, [q])
        else:
            auds.queue.extend(q)
        auds.stats = d["stats"]
        if not "h" in flags:
            return "```css\nSuccessfully appended dump to queue for " + uniStr(guild.name) + ".```"
            

class Volume:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = ["Vol", "Audio", "V", "Opt", "Set"]
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
            orig = _vars.updaters["playlists"].audio[guild.id].stats[op]
            if op in "loop shuffle quiet":
                num = bool(orig)
            else:
                num = round(100 * orig, 9)
            return (
                "```css\nCurrent audio " + op
                + " state" * (type(orig) is bool)
                + " in " + uniStr(guild.name)
                + ": " + uniStr(num) + ".```"
            )
        if op == "settings":
            op = "volume"
        s_perm = _vars.getPerms(user, guild)
        if s_perm < 1:
            raise PermissionError(
                "Insufficient permissions to change audio settings. Current permission level: "
                + uniStr(s_perm) + ", required permission level: " + uniStr(1) + "."
            )
        if op is None:
            pos = auds.stats["position"]
            auds.stats = dict(auds.defaults)
            auds.new(auds.file, pos)
            if auds.stats["quiet"] & 2:
                try:
                    await message.delete()
                except discord.NotFound:
                    pass
            else:
                return (
                    "```css\nSuccessfully reset all audio settings for "
                    + uniStr(guild.name) + ".```"
                )
        origVol = _vars.updaters["playlists"].audio[guild.id].stats
        num = await _vars.evalMath(argv, guild.id)
        val = roundMin(float(num / 100))
        orig = round(origVol[op] * 100, 9)
        new = round(val * 100, 9)
        if op in "loop shuffle quiet":
            origVol[op] = new = bool(val)
            orig = bool(orig)
        else:
            origVol[op] = val
        if op in "speed pitch chorus":
            auds.new(auds.file, auds.stats["position"])
        if auds.stats["quiet"] & 2:
            try:
                await message.delete()
            except discord.NotFound:
                pass
        else:
            return (
                "```css\nChanged audio " + op
                + " state" * (type(orig) is bool)
                + " in " + uniStr(guild.name)
                + " from " + uniStr(orig)
                + " to " + uniStr(new) + ".```"
            )


class Shuffle:
    is_command = True
    server_only = True

    def __init__(self):
        self.name = []
        self.min_level = 1
        self.description = "Shuffles the audio queue."
        self.usage = ""

    async def __call__(self, guild, channel, user, client, _vars, **void):
        auds = await forceJoin(guild, channel, user, client, _vars)
        if len(auds.queue) > 1:
            for i in range(3):
                try:
                    auds.queue[i].pop("download")
                except KeyError:
                    pass
            shuffle(auds.queue)
            auds.seek(inf)
        return (
            "```css\nSuccessfully shuffled audio queue for "
            + uniStr(guild.name) + ".```"
        )


class Unmute:
    is_command = True
    server_only = True
    time_consuming = True

    def __init__(self):
        self.name = ["Unmuteall"]
        self.min_level = 2
        self.description = "Disables server mute for all members."
        self.usage = ""

    async def __call__(self, guild, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                asyncio.create_task(user.edit(mute=False, deafen=False))
        return (
            "```css\nSuccessfully unmuted all users in voice channels in "
            + uniStr(guild.name) + ".```"
        )


class Player:
    is_command = True
    server_only = True
    time_consuming = True
    buttons = {
	b'\xe2\x8f\xb8': 0,
	b'\xf0\x9f\x94\x84': 1,
	b'\xf0\x9f\x94\x80': 2,
	b'\xe2\x8f\xae': 3,
	b'\xe2\x8f\xad': 4,
        b'\xf0\x9f\x94\x8a': 5,
        b'\xf0\x9f\xa5\x81': 6,
        b'\xf0\x9f\x93\x89': 7,
        b'\xf0\x9f\x93\x8a': 8,
        b'\xe2\x8f\xaa': 9,
        b'\xe2\x8f\xa9': 10,
        b'\xe2\x8f\xab': 11,
        b'\xe2\x8f\xac': 12,
        b'\xe2\x99\xbb': 13,
	b'\xe2\x8f\x8f': 14,
        b'\xe2\x9b\x94': 15,
        }
    barsize = 24

    def __init__(self):
        self.name = []
        self.min_level = 0
        self.description = "Creates an auto-updating virtual audio player for the current server."
        self.usage = "<verbose(?v)> <controllable(?c)> <disable(?d)>"

    def showCurr(self, auds):
        q = auds.queue
        if q:
            s = q[0]["skips"]
            if s is not None:
                skips = len(s)
            else:
                skips = 0
            output = "Playing " + uniStr(q[0]["name"]) + ", "
            output += uniStr(len(q)) + " item" + "s" * (len(q) != 1) + " total "
            output += skips * "🚫"
        else:
            output = "Queue is currently empty. "
        if auds.stats["loop"]:
            output += "🔄"
        if auds.stats["shuffle"]:
            output += "🔀"
        output += "\n"
        v = abs(auds.stats["volume"])
        if v == 0:
            output += "🔇"
        if v <= 0.5:
            output += "🔉"
        elif v <= 1:
            output += "🔊"
        elif v <= 5:
            output += "📢"
        else:
            output += "🌪️"
        b = auds.stats["bassboost"]
        if abs(b) > 1 / 3:
            if abs(b) > 5:
                output += "💥"
            elif b > 0:
                output += "🥁"
            else:
                output += "🎻"
        r = auds.stats["reverb"]
        if r:
            if abs(r) >= 1:
                output += "📈"
            else:
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
        if q:
            p = [auds.stats["position"], q[0]["duration"]]
        else:
            p = [0, 0.25]
        output += (
            " (" + uniStr(dhms(p[0]))
            + "/" + uniStr(dhms(p[1])) + ") "
        )
        sym = "⬜⬛"
        r = round(min(1, p[0] / p[1]) * self.barsize)
        output += sym[0] * r + sym[1] * (self.barsize - r)
        return output

    async def _callback_(self, message, guild, channel, reaction, _vars, perm, vals, **void):
        if message is None:
            return
        if not guild.id in _vars.updaters["playlists"].audio:
            await message.clear_reactions()
            return
        auds = _vars.updaters["playlists"].audio[guild.id]
        if reaction is None:
            auds.player = {
                "time": inf,
                "message": message,
                "type": int(vals),
                "events": 0,
            }
            if vals:
                auds.stats["quiet"] |= 2
        elif auds.player is None or auds.player["message"].id != message.id:
            await message.clear_reactions()
            return
        if perm < 1:
            return
        orig = "\n".join(message.content.split("\n")[:1 + ("\n" == message.content[3])]) + "\n"
        if reaction is None and auds.player["type"]:
            for b in self.buttons:
                await message.add_reaction(b.decode("utf-8"))
        else:
            if not auds.player["type"]:
                emoji = bytes()
            elif type(reaction) is bytes:
                emoji = reaction
            else:
                try:
                    emoji = reaction.emoji
                except:
                    emoji = str(reaction)
            if type(emoji) is str:
                emoji = reaction.encode("utf-8")
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
                    if pos:
                        return
                elif i == 5:
                    v = abs(auds.stats["volume"])
                    if v < 0.25 or v >= 2:
                        v = 1 / 3
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
                    try:
                        await message.delete()
                    except discord.NotFound:
                        pass
                    return
                else:
                    auds.player = None
                    try:
                        await message.delete()
                    except discord.NotFound:
                        pass
                    return
        text = orig + self.showCurr(auds) + "```"
        last = message.channel.last_message
        if last is not None and (auds.player["type"] or message.id == last.id):
            auds.player["events"] += 1
            await message.edit(
                content=text,
            )
        else:
            auds.player["time"] = inf
            auds.player["events"] += 2
            channel = message.channel
            temp = message
            message = await channel.send(
                content=text,
            )
            auds.player["message"] = message
            try:
                await temp.delete()
            except (TypeError, discord.NotFound):
                pass
        if auds.queue and not auds.paused & 1:
            maxdel = auds.queue[0]["duration"] - auds.stats["position"] + 2
            delay = min(maxdel, auds.queue[0]["duration"] / self.barsize / abs(auds.stats["speed"]))
            if delay > 20:
                delay = 20
            elif delay < 6:
                delay = 6
        else:
            delay = inf
        auds.player["time"] = time.time() + delay

    async def __call__(self, channel, user, client, _vars, flags, perm, **void):
        auds = await forceJoin(channel.guild, channel, user, client, _vars)
        if "c" in flags or auds.stats["quiet"] & 2:
            req = 1
            if perm < req:
                if auds.stats["quiet"] & 2:
                    if "d" in flags:
                        reason = "delete"
                    else:
                        reason = "override"
                else:
                    reason = "create controllable"
                raise PermissionError(
                    "Insufficient privileges to " + reason
                    + " virtual audio player for " + uniStr(guild.name)
                    + ". Required level: " + uniStr(req)
                    + ", Current level: " + uniStr(perm) + "."
                )
        if "d" in flags:
            auds.player = None
            return (
                "```css\nSuccessfully disabled active virtual audio players in "
                + uniStr(channel.guild.name) + ".```"
            )
        await createPlayer(auds, p_type="c" in flags, verbose="v" in flags)


class updateQueues:
    is_update = True
    name = "playlists"

    def __init__(self):
        self.audio = {}
        self.audiocache = {}
        self.clearAudioCache()

    def sendUpdateRequest(self, *args, **void):
        self._vars.doUpdate = True

    async def research(self, auds):
        if auds.searching >= 1:
            return
        auds.searching += 1
        searched = 0
        q = auds.queue
        for i in q:
            if searched >= 32:
                break
            if "research" in i:
                try:
                    i.pop("research")
                    returns = [None]
                    t = time.time()
                    doParallel(ytdl.extractSingle, [i], returns)
                    while returns[0] is None and time.time() - t < 10:
                        await asyncio.sleep(0.6)
                    print(i["name"])
                    searched += 1
                except:
                    print(traceback.format_exc())
                    break
            if random.random() > 0.99:
                await asyncio.sleep(0.4)
        await asyncio.sleep(2)
        auds.searching = max(auds.searching - 1, 0)

    def clearAudioCache(self):
        _vars = self._vars
        pl = self.data
        should_cache = {}
        for g in pl:
            for i in pl[g]:
                s = i["id"] + ".mp3"
                should_cache[s] = True
        for path in os.listdir("cache/"):
            found = False
            if "mp3" in path:
                for i in should_cache:
                    if i in path:
                        found = True
                        break
            if not found:
                try:
                    os.remove("cache/" + path)
                except Exception as ex:
                    print(traceback.format_exc())

    async def __call__(self, **void):
        while self.busy:
            await asyncio.sleep(0.3)
        self.busy = True
        try:
            _vars = self._vars
            pl = self.data
            client = _vars.client
            should_cache = {}
            for g in pl:
                for i in pl[g]:
                    should_cache[i["id"]] = True
            for vc in client.voice_clients:
                if not vc.is_connected():
                    continue
                channel = vc.channel
                guild = channel.guild
                try:
                    auds = self.audio[guild.id]
                    playing = auds.is_playing and vc.is_playing() or auds.is_loading
                    membs = [m for m in channel.members if m.id != client.user.id]
                    cnt = len(membs)
                except KeyError:
                    continue
                if not cnt or getattr(auds, "dead", 0):
                    try:
                        channel = auds.channel
                        self.audio.pop(guild.id)
                        msg = (
                            "```css\n🎵 Successfully disconnected from "
                            + uniStr(guild.name) + ". 🎵```"
                        )
                        sent = await channel.send(msg)
                        await sent.add_reaction("❎")
                    except KeyError:
                        pass
                    await vc.disconnect(force=True)
                else:
                    try:
                        asyncio.create_task(auds.updatePlayer())
                        try:
                            q = auds.queue
                        except NameError:
                            continue
                        asyncio.create_task(self.research(auds))
                        dels = deque()
                        i = 0
                        for i in range(len(q)):
                            if i >= len(q) or i > 10000:
                                break
                            e = q[i]
                            e_id = e["id"]
                            if not e_id:
                                dels.append(i)
                                continue
                            if e_id in self.audiocache:
                                e["duration"] = self.audiocache[e_id][0]
                        if len(dels) > 1:
                            q.delitems(dels)
                        elif len(dels):
                            q.pop(dels[0])
                        if len(q):
                            for i in range(2):
                                if i < len(q):
                                    e_id = q[i]["id"]
                                    should_cache[e_id] = True
                                    if not q[i].get("download", 0):
                                        q[i]["download"] = 1
                                        if e_id not in self.audiocache:
                                            search = e_id + ".mp3"
                                            found = False
                                            for path in os.listdir("cache"):
                                                if search in path:
                                                    found = True
                                            if not found:
                                                durc = [q[i]["duration"]]
                                                self.audiocache[e_id] = durc
                                                doParallel(
                                                    ytdl.downloadSingle,
                                                    [q[i], durc],
                                                    state=2
                                                )
                                            else:
                                                q[i]["duration"] = ytdl.getDuration("cache/" + search)
                            if not q[0].get("download", 0) > 1 and not playing:
                                try:
                                    path = "cache/" + q[0]["id"] + ".mp3"
                                    f = open(path, "rb")
                                    minl = 64
                                    b = f.read(minl)
                                    f.close()
                                    if len(b) < minl:
                                        raise FileNotFoundError
                                    q[0]["download"] = 2
                                    name = q[0]["name"]
                                    added_by = q[0]["added by"]
                                    auds = self.audio[guild.id]
                                    auds.new(path)
                                    if not vc.is_playing():
                                        vc.play(auds, after=self.sendUpdateRequest)
                                    if not auds.stats["quiet"]:
                                        channel = auds.channel
                                        sent = await channel.send(
                                            "```css\n🎵 Now playing "
                                            + uniStr(noHighlight(name))
                                            + ", added by " + uniStr(added_by) + "! 🎵```"
                                        )
                                        await sent.add_reaction("❎")
                                except FileNotFoundError:
                                    pass
                                auds.preparing = False
                            elif not playing and auds.source is None:
                                doParallel(auds.advance)
                        if not len(q) and not auds.preparing:
                            t = pl.get(guild.id, ())
                            if len(t):
                                d = None
                                while d is None or d["id"] == auds.prev:
                                    p = t[xrand(len(t))]
                                    d = {
                                        "name": p["name"],
                                        "url": p["url"],
                                        "duration": p["duration"],
                                        "added by": client.user.name,
                                        "u_id": client.user.id,
                                        "id": p["id"],
                                        "skips": (),
                                    }
                                    if len(t) <= 1:
                                        break
                                q.append(d)
                    except KeyError as ex:
                        print(traceback.format_exc())
                l = list(self.audiocache)
                for i in l:
                    if not i in should_cache:
                        path = "cache/" + i + ".mp3"
                        try:
                            os.remove(path)
                            self.audiocache.pop(i)
                        except PermissionError:
                            pass
                        except FileNotFoundError:
                            self.audiocache.pop(i)
        except:
            print(traceback.format_exc())
        self.busy = False
