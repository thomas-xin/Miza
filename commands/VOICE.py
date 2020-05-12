try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import youtube_dl, pytube, ffmpy, samplerate
from bs4 import BeautifulSoup

getattr(youtube_dl, "__builtins__", {})["print"] = print
getattr(ffmpy, "__builtins__", {})["print"] = print

SAMPLE_RATE = 48000


async def createPlayer(auds, p_type=0, verbose=False):
    auds.stats.quiet |= 2 * p_type
    text = (
        "```" + "\n" * verbose + "callback-voice-player-" + str(int(bool(p_type)))
        + "\nInitializing virtual audio player...```"
    )
    await auds.channel.send(text)
    await auds.updatePlayer()


# def gethash(entry):
#     return entry.setdefault("hash", shash(entry.url))

# def sethash(entry):
#     entry.hash = shash(entry.url)
#     return entry.hash


def getDuration(filename):
    command = ["ffprobe", filename]
    resp = bytes()
    for _ in loop(3):
        try:
            proc = psutil.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            resp = bytes().join(proc.communicate())
            break
        except:
            print(traceback.format_exc())
    s = resp.decode("utf-8")
    try:
        i = s.index("Duration: ")
        d = s[i + 10:]
        i = inf
        for c in ", \n\r":
            try:
                x = d.index(c)
            except ValueError:
                pass
            else:
                if x < i:
                    i = x
        dur = rdhms(d[:i])
    except:
        print(s)
        print(traceback.format_exc())
        return "300"
    return dur


def pytube2Dict(url):
    if not url.startswith("https://www.youtube.com/"):
        if not url.startswith("http://youtu.be/"):
            if isURL(url):
                raise youtube_dl.DownloadError("Not a youtube link.")
            url = "https://www.youtube.com/watch?v=" + url
    # print(url)
    for _ in loop(3):
        try:
            resp = pytube.YouTube(url)
            break
        except pytube.exceptions.RegexMatchError:
            raise youtube_dl.DownloadError("Invalid single youtube link.")
        except KeyError as ex:
            resp = ex
    if issubclass(type(resp), Exception):
        raise resp
    entry = {
        "formats": [
            {
                "abr": 0,
                "vcodec": stream.video_codec,
                "url": stream.url,
            } for stream in resp.streams.fmt_streams
        ],
        "duration": resp.length,
    }
    for i in range(len(entry["formats"])):
        stream = resp.streams.fmt_streams[i]
        try:
            abr = stream.abr.lower()
        except AttributeError:
            abr = 0
        abr = str(abr)
        if abr.endswith("kbps"):
            abr = float(abr[:-4])
        elif abr.endswith("mbps"):
            abr = float(abr[:-4]) * 1024
        elif abr.endswith("bps"):
            abr = float(abr[:-3]) / 1024
        else:
            try:
                abr = float(abr)
            except:
                print(traceback.format_exc())
                continue
        entry["formats"][i]["abr"] = abr
    return entry


def getBestAudio(entry):
    best = -1
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ()
    try:
        url = entry["webpage_url"]
    except KeyError:
        url = entry["url"]
    for fmt in fmts:
        q = fmt.get("abr", 0)
        if type(q) is not int:
            q = 0
        vcodec = fmt.get("vcodec", "none")
        if vcodec not in (None, "none"):
            q -= 1
        if q > best:
            best = q
            url = fmt["url"]
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
    return url


async def forceJoin(guild, channel, user, client, bot, preparing=False):
    if guild.id not in bot.database.playlists.audio:
        for func in bot.commands.connect:
            try:
                await func(user=user, channel=channel)
            except (discord.ClientException, AttributeError):
                pass
    try:
        auds = bot.database.playlists.audio[guild.id]
        auds.channel = channel
        auds.preparing = preparing
    except KeyError:
        raise LookupError("Voice channel not found.")
    return auds


async def downloadTextFile(url, bot):
    
    def dreader(file):
        s = resp.read().decode("utf-8")
        resp.close()
        return s

    url = await bot.followURL(url)
    resp = urlOpen(url)
    return create_future(dreader, resp)


def isAlone(auds, user):
    for m in auds.vc.channel.members:
        if m.id != user.id and not m.bot:
            return False
    return True


def ensure_url(url):
    if url.startswith("ytsearch:"):
        url = "https://www.youtube.com/results?search_query=" + verifyURL(url[9:])
    return url


class customAudio(discord.AudioSource):
    
    length = round(SAMPLE_RATE / 25)
    empty = numpy.zeros(length >> 1, float)
    emptybuff = numpy.zeros(length, numpy.uint16).tobytes()
    filt = signal.butter(1, 0.125, btype="low", output="sos")
    #fff = numpy.abs(numpy.fft.fftfreq(SAMPLE_RATE / 50, 1/SAMPLE_RATE))[:ceil(SAMPLE_RATE / 100 + 1)]
    static = lambda self, *args: numpy.random.rand(self.length) * 65536 - 32768
    defaults = {
        "volume": 1,
        "reverb": 0,
        "pitch": 0,
        "speed": 1,
        "pan": 1,
        "bassboost": 0,
        "compressor": 0,
        "chorus": 0,
        "resample": 0,
        "loop": False,
        "shuffle": False,
        "quiet": False,
        "position": 0,
    }

    def __init__(self, channel, vc, bot):
        try:
            self.paused = False
            self.stats = freeClass(**self.defaults)
            self.source = None
            self.channel = channel
            self.vc = vc
            # self.fftrans = list(range(len(self.fff)))
            # self.cpitch = 0
            self.temp_buffer = [numpy.zeros(0, dtype=float) for _ in loop(2)]
            self.buffer = []
            self.reverse = False
            self.feedback = None
            self.bassadj = None
            self.bufadj = None
            self.prev = None
            self.refilling = 0
            self.reading = 0
            self.has_read = False
            self.searching = False
            self.preparing = True
            self.player = None
            self.timeout = time.time()
            self.lastsent = 0
            self.lastEnd = 0
            self.pausec = False
            self.curr_timeout = 0
            self.bot = bot
            self.new(update=False)
            self.queue = AudioQueue(auds=self)
            bot.database.playlists.audio[vc.guild.id] = self
        except:
            print(traceback.format_exc())

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return (
            "<" + classname + " object at " + hex(id(self)).upper() + ">: {"
            + "\"vc\": " + str(self.vc)
            + ", \"queue\": " + str(self.queue)
            + ", \"stats\": " + str(self.stats)
            + ", \"source\": " + str(self.source)
            + "}"
        )

    def stop(self):
        self.refilling = False
        if getattr(self, "source", None) is None:
            return
        create_future_ex(self.source.close)
        self.source = None

    def new(self, source=None, pos=0, update=True):
        self.speed = abs(self.stats.speed)
        self.is_playing = source is not None
        if source is not None:
            new_source = None
            try:
                new_source = source.create_reader(pos, auds=self)
            except OverflowError:
                source = None
            else:
                print(new_source)
                self.preparing = False
                self.is_playing = True
                self.has_read = False
            self.stop()
            self.source = new_source
            self.file = source
        else:
            self.stop()
            self.file = None
        self.stats.position = pos
        if pos == 0:
            if self.reverse and len(self.queue):
                self.stats.position = float(self.queue[0].duration)
        if self.source is not None and self.player:
            self.player.time = 1 + time.time()
        if self.speed < 0.005:
            self.speed = 1
            self.paused |= 2
        else:
            self.paused &= -3
        if update:
            self.update()
            self.queue.update_play()
        self.refilling = 0

    def seek(self, pos):
        duration = float(self.queue[0].duration)
        pos = max(0, pos)
        if (pos >= duration and not self.reverse) or (pos <= 0 and self.reverse):
            create_future(self.new, update=True)
            return duration
        create_future(self.new, self.file, pos, update=False)
        self.stats.position = pos
        return self.stats.position

    def update(self, *void1, **void2):
        vc = self.vc
        guild = vc.guild
        g = guild.id
        if hasattr(self, "dead"):
            try:
                self.bot.database.playlists.audio.pop(g)
            except KeyError:
                pass
            try:
                self.bot.database.playlists.connecting.pop(g)
            except KeyError:
                pass
            create_task(vc.disconnect())
            if self.dead is not None:
                self.dead = None
                try:
                    msg = (
                        "```css\n🎵 Successfully disconnected from ["
                        + noHighlight(guild.name) + "]. 🎵```"
                    )
                    create_task(sendReact(
                        self.channel,
                        msg,
                        reacts=["❎"],
                    ))
                except KeyError:
                    pass
                self.stop()
            return
        if not hasattr(vc, "channel"):
            self.dead = True
            return
        m = guild.get_member(self.bot.client.user.id)
        if m is None:
            self.dead = True
            return
        q = self.queue
        if not vc.is_playing():
            try:
                if q and not self.pausec and self.source is not None:
                    vc.play(self, after=self.queue.advance)
                self.att = 0
            except:
                self.att = getattr(self, "att", 0) + 1
                if self.att > 5:
                    self.dead = True
                    return
        cnt = sum(1 for m in vc.channel.members if not m.bot)
        if not cnt:
            if self.timeout < time.time() - 20:
                self.dead = True
                return
            elif self.timeout < time.time() - 10:
                if guild.afk_channel is not None:
                    if guild.afk_channel.id != vc.channel.id:
                        create_task(self.move_unmute(vc, guild.afk_channel))
                    else:
                        cnt = 0
                        ch = None
                        for channel in guild.voice_channels:
                            if channel.id != guild.afk_channel.id:
                                c = sum(1 for m in channel.members if not m.bot)
                                if c > cnt:
                                    cnt = c
                                    ch = channel
                        if ch:
                            create_task(vc.move_to(ch))
        else:
            self.timeout = time.time()
        if m.voice is not None:
            if m.voice.deaf or m.voice.mute or m.voice.afk:
                create_task(m.edit(mute=False, deafen=False))
        if not (vc.is_connected() or self.bot.database.playlists.is_connecting(vc.guild.id)):
            create_task(self.reconnect())
        else:
            self.att = 0
            create_future(self.queue.update_load)

    async def move_unmute(self, vc, channel):
        await vc.move_to(channel)
        await channel.guild.get_member(self.bot.client.user.id).edit(mute=False, deafen=False)

    async def reconnect(self):
        try:
            if hasattr(self, "dead") or self.vc.is_connected():
                return
            bot.database.playlists.connecting[self.vc.guild.id] = True
            self.att = getattr(self, "att", 0) + 1
            self.vc = await self.vc.channel.connect(timeout=30, reconnect=False)
            user = self.vc.guild.get_member(self.bot.client.user.id)
            if getattr(user, "voice", None) is not None:
                if user.voice.deaf or user.voice.mute or user.voice.afk:
                    create_task(user.edit(mute=False, deafen=False))
            self.att = 0
        except (discord.Forbidden):
            self.dead = True
        except:
            print(traceback.format_exc())
            if self.att > 5:
                self.dead = True
        try:
            bot.database.playlists.connecting.pop(self.vc.guild.id)
        except:
            print(traceback.format_exc())

    async def updatePlayer(self):
        curr = self.player
        self.stats.quiet &= -3
        if curr is not None:
            if curr.type:
                self.stats.quiet |= 2
            try:
                if not curr.message.content:
                    raise EOFError
            except:
                self.player = None
                print(traceback.format_exc())
            if time.time() > curr.time:
                curr.time = inf
                try:
                    await self.bot.reactCallback(curr.message, "❎", self.bot.client.user)
                except discord.NotFound:
                    self.player = None
                    print(traceback.format_exc())
        q = self.stats.quiet
        if q == bool(q):
            self.stats.quiet = bool(q)

    def refill_buffer(self):
        empty = False
        size = self.length >> 1
        volume = self.stats.volume
        reverb = self.stats.reverb
        pitch = self.stats.pitch
        bassboost = self.stats.bassboost
        chorus = self.stats.chorus
        if self.stats.resample >= 2400:
            resample = 1
        else:
            resample = 2 ** (self.stats.resample / 12)
        if abs(volume) >= 1 << 31:
            volume = nan
        if abs(reverb) >= 1 << 31:
            volume = nan
        if abs(bassboost) >= 1 << 31:
            volume = nan
        if abs(pitch) >= 1 << 31:
            volume = nan
        if abs(resample) >= 1 << 31:
            volume = nan
        buflen = size
        if resample != 1:
            buflen = max(1, round_random(resample * buflen))
        new_buf = [numpy.zeros(0, dtype=float) for _ in loop(2)]
        found = False
        while len(self.temp_buffer[0]) + len(new_buf[0]) < buflen:
            # print(len(self.temp_buffer[0]) + len(new_buf[0]))
            try:
                if self.queue.loading or self.paused:
                    self.is_playing = True
                    raise EOFError
                try:
                    source = self.source
                    if source is None:
                        raise StopIteration
                    temp = source.read(discord.opus.Encoder.FRAME_SIZE)
                    if not temp:
                        raise StopIteration
                    found = True
                except (StopIteration, ValueError):
                    empty = True
                    raise EOFError
                except:
                    empty = True
                    print(traceback.format_exc())
                    raise EOFError
                if not empty:
                    self.stats.position = round(
                        self.stats.position + self.speed / 50 * (self.reverse * -2 + 1),
                        4,
                    )
                    self.has_read = True
                    self.curr_timeout = 0
                self.is_playing = True
            except EOFError:
                if self.source is not None and self.source.closed:
                    self.source = None
                if (empty or not self.paused) and not self.queue.loading:
                    queueable = (self.queue or self.bot.data.playlists.get(self.vc.guild.id, None))
                    if self.queue and not self.queue[0].get("played", False):
                        if not found and not self.queue.loading:
                            self.refilling = 2
                            create_future(self.queue.advance)
                            return
                    elif empty and queueable and self.source is not None:
                        if time.time() - self.lastEnd > 0.5:
                            if self.reverse:
                                ended = self.stats.position <= 0.5
                            else:
                                ended = ceil(self.stats.position) >= float(self.queue[0].duration) - 0.5
                            if self.curr_timeout and time.time() - self.curr_timeout > 0.5 or ended:
                                if not found:
                                    self.lastEnd = time.time()
                                    if not self.has_read or not self.queue:
                                        print("Advanced.")
                                        self.refilling = 2
                                        if self.queue:
                                            self.queue[0].url = ""
                                        create_future(self.new)
                                        self.preparing = False
                                        return
                                    else:
                                        self.stats.position = round(
                                            self.stats.position + self.speed * resample / 6.25 * (self.reverse * -2 + 1),
                                            4,
                                        )
                                        if not ended:
                                            self.refilling = 2
                                            self.seek(self.stats.position)
                                            return
                                        else:
                                            print("Advanced.")
                                            self.refilling = 2
                                            create_future(self.new)
                                            self.preparing = False
                                            return
                            elif self.curr_timeout == 0:
                                self.curr_timeout = time.time()
                    elif not queueable:
                        self.curr_timeout = 0
                        self.vc.stop()
                        return
                temp = self.emptybuff
                # print(traceback.format_exc())
            try:
                if not isValid(volume):
                    array = self.static()
                else:
                    array = numpy.frombuffer(temp, dtype=numpy.int16).astype(float)
                    if volume != 1 or chorus:
                        try:
                            if chorus:
                                volume *= 2
                            array *= volume * (bool(chorus) + 1)
                        except:
                            array = self.static()
                for i in range(2):
                    new_buf[i] = numpy.concatenate([new_buf[i], array[i::2]])
            except:
                print(traceback.format_exc())
        while self.reading or self.refilling > 1:
            time.sleep(0.03)
        self.refilling = 2
        for i in range(2):
            self.temp_buffer[i] = numpy.concatenate([self.temp_buffer[i], new_buf[i]])
        self.refilling = 0

    def read(self):
        size = self.length >> 1
        reverb = self.stats.reverb
        bassboost = self.stats.bassboost
        if self.stats.resample >= 2400:
            resample = 1
        else:
            resample = 2 ** (self.stats.resample / 12)
        delay = 16
        buflen = size
        if resample != 1:
            buflen = max(1, round_random(resample * buflen))
        if self.refilling > 1 or self.queue.loading:
            return self.emptybuff
        if len(self.temp_buffer[0]) < buflen:
            if not self.refilling:
                self.refilling = 1
                self.refill_buffer()
            else:
                return self.emptybuff
        try:
            self.reading = 1
            lbuf, self.temp_buffer[0] = numpy.hsplit(self.temp_buffer[0], [buflen])
            rbuf, self.temp_buffer[1] = numpy.hsplit(self.temp_buffer[1], [buflen])
            self.reading = 0
            if resample != 1:
                if self.bufadj is not None:
                    ltemp = numpy.concatenate((self.bufadj[0], lbuf))
                    rtemp = numpy.concatenate((self.bufadj[1], rbuf))
                    try:
                        left = samplerate.resample(ltemp, 2 * size / len(ltemp), converter_type="sinc_fastest")[-size - 16:-16]
                        right = samplerate.resample(rtemp, 2 * size / len(rtemp), converter_type="sinc_fastest")[-size - 16:-16]
                    except (ZeroDivisionError, samplerate.exceptions.ResamplingError):
                        left, right = ltemp, rtemp
                else:
                    left, right = lbuf, rbuf
                self.bufadj = [lbuf, rbuf]
            else:
                left, right = lbuf, rbuf
            if not len(left) or not len(right):
                left = numpy.zeros(size, dtype=float)
                right = numpy.zeros(size, dtype=float)
            elif len(left) != size or len(right) != size:
                left = numpy.interp([i * len(left) / size for i in range(size)], list(range(len(left))), left)
                right = numpy.interp([i * len(right) / size for i in range(size)], list(range(len(right))), right)
            # if detune:
            #     if self.cpitch != detune:
            #         self.cpitch = detune
            #         self.fftrans = numpy.clip(self.fff * 2 ** (detune / 12) / 50, 1, len(self.fff) - 1)
            #         self.fftrans[0] = 0
            #     lft, rft = numpy.fft.rfft(left), numpy.fft.rfft(right)
            #     s = len(lft) + len(rft) >> 1
            #     temp = numpy.zeros(s, dtype=complex)
            #     lsh, rsh = temp, numpy.array(temp)
            #     for i in range(s):
            #         j = self.fftrans[i]
            #         x = int(j)
            #         y = min(x + 1, s - 1)
            #         z = j % 1
            #         lsh[x] += lft[i] * (1 - z)
            #         lsh[y] += lft[i] * z
            #         rsh[x] += rft[i] * (1 - z)
            #         rsh[y] += rft[i] * z
            #     left, right = numpy.fft.irfft(lsh), numpy.fft.irfft(rsh)
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
                        )[-size - 16:-16] * bassboost
                        right += signal.sosfilt(
                            filt,
                            numpy.concatenate((self.bassadj[1], right))
                        )[-size - 16:-16] * bassboost
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
                    # p2 = round(size * (0.5 - 1 / r))
                    p3 = round(size * 0.5)
                    # p4 = round(size * (0.5 + 1 / r))
                    p5 = round(size * (0.5 + 2 / r))
                    lfeed = (
                        + numpy.concatenate((self.buffer[0][0][p1:], self.buffer[1][0][:p1])) / 8
                        # + numpy.concatenate((self.buffer[0][0][p2:], self.buffer[1][0][:p2])) / 12
                        + numpy.concatenate((self.buffer[0][0][p3:], self.buffer[1][0][:p3])) * 0.75
                        # + numpy.concatenate((self.buffer[0][0][p4:], self.buffer[1][0][:p4])) / 12
                        + numpy.concatenate((self.buffer[0][0][p5:], self.buffer[1][0][:p5])) / 8
                    ) * reverb
                    rfeed = (
                        + numpy.concatenate((self.buffer[0][1][p1:], self.buffer[1][1][:p1])) / 8
                        # + numpy.concatenate((self.buffer[0][1][p2:], self.buffer[1][1][:p2])) / 12
                        + numpy.concatenate((self.buffer[0][1][p3:], self.buffer[1][1][:p3])) * 0.75
                        # + numpy.concatenate((self.buffer[0][1][p4:], self.buffer[1][1][:p4])) / 12
                        + numpy.concatenate((self.buffer[0][1][p5:], self.buffer[1][1][:p5])) / 8
                    ) * reverb
                    if self.feedback is not None:
                        left -= signal.sosfilt(self.filt, numpy.concatenate((self.feedback[0], lfeed)))[-size - 16:-16]
                        right -= signal.sosfilt(self.filt, numpy.concatenate((self.feedback[1], rfeed)))[-size - 16:-16]
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
            self.pausec = self.paused and not (max(temp) or min(temp))
            if self.pausec:
                self.vc.stop()
        except:
            self.reading = 0
            print(traceback.format_exc())
        return temp

    is_opus = lambda self: False
    cleanup = lambda self: None


class AudioQueue(hlist):

    def __init__(self, *args, auds=None, **kwargs):
        super().__init__(*args, **kwargs)
        if auds is not None or not hasattr(self, "auds"):
            self.auds = auds
            self.bot = auds.bot
            self.vc = auds.vc
        self.prev = ""
        self.lastsent = 0

    def update_load(self):
        q = self
        if q:
            if len(q) > 65536 + 2048:
                q.__init__(q[-65535:].appendleft(q[0]))
            elif len(q) > 65536:
                q.rotate(-1)
                while len(q) > 65536:
                    q.pop()
                q.rotate(1)
            dels = deque()
            for i in range(len(q)):
                if i >= len(q) or i > 8191:
                    break
                e = q[i]
                if i < 3:
                    if not e.get("stream", None):
                        create_future(ytdl.getStream, e)
                        break
                if not e.url:
                    if not self.auds.stats.quiet:
                        create_task(sendReact(
                            self.auds.channel,
                            "```ini\nA problem occurred while loading " + sbHighlight(e.name)
                            + ", and it has been removed from the queue as a result.```",
                            reacts=["❎"],
                        ))
                    dels.append(i)
                    continue
            if len(dels) > 2:
                q.pops(dels)
            elif dels:
                while dels:
                    q.pop(dels.popleft())
            if q:
                self.update_play()
        create_task(self.auds.updatePlayer())

    def advance(self, looped=True, shuffled=True):
        q = self
        s = self.auds.stats
        if q:
            if s.loop:
                temp = q[0]
            self.prev = q[0]["url"]
            q.popleft()
            if s.shuffle and shuffled:
                if len(q) > 1:
                    temp = q.popleft()
                    shuffle(q)
                    q.appendleft(temp)
            if s.loop and looped:
                try:
                    temp.pop("played")
                except (KeyError, IndexError):
                    pass
                q.append(temp)
        if not (q or self.auds.preparing):
            t = self.bot.data.playlists.get(self.vc.guild.id, ())
            if t:
                while True:
                    p = random.choice(t)
                    if len(t) > 1 and p["url"] == self.prev:
                        continue
                    d = {
                        "name": p["name"],
                        "url": p["url"],
                        "duration": p["duration"],
                        "u_id": self.bot.client.user.id,
                        "skips": (),
                        "research": True,
                    }
                    break
                q.append(freeClass(d))
        if self.auds.player:
            self.auds.player.time = 1 + time.time()
        if q:
            self.update_play()
        
    def update_play(self):
        auds = self.auds
        q = self
        if q:
            if not q[0].get("played", False):
                if q[0].get("stream", None) not in (None, "none"):
                    q[0].played = True
                    if not auds.stats.quiet:
                        if time.time() - self.lastsent > 1:
                            try:
                                u = self.bot.cache.users[q[0].u_id]
                                name = u.display_name
                            except KeyError:
                                name = "Deleted User"
                            msg = (
                                "```ini\n🎵 Now playing "
                                + sbHighlight(q[0].name)
                                + ", added by " + sbHighlight(name) + "! 🎵```"
                            )
                            create_task(sendReact(
                                auds.channel,
                                msg,
                                reacts=["❎"],
                            ))
                            self.lastsent = time.time()
                    self.loading = True
                    if "research" in q[0]:
                        q[0].pop("research")
                        ytdl.extractSingle(q[0])
                    source = ytdl.getStream(q[0])
                    print(q[0])
                    try:
                        auds.new(source)
                        self.loading = False
                    except:
                        self.loading = False
                        print(traceback.format_exc())
                        raise
                    auds.preparing = False
            elif auds.source is None and not self.loading and not auds.preparing:
                print("Queue Advanced.")
                self.advance()

    def enqueue(self, items, position):
        initialize = not len(self)
        if position == -1:
            self.extend(items)
        else:
            self.rotate(-position)
            self.extend(items)
            self.rotate(len(items) + position)
        if initialize:
            self.update_play()
        return self



class PCMFile:
    
    def __init__(self, fn):
        self.file = fn
        self.proc = None
        self.time = time.time()
        self.loading = False
        self.expired = False
    
    def load(self, stream):
        if self.loading:
            return
        self.loading = True
        ff = ffmpy.FFmpeg(
            global_options=["-y", "-hide_banner", "-loglevel error", "-vn"],
            inputs={stream: None},
            outputs={"s16le": "-f", str(SAMPLE_RATE): "-ar", "2": "-ac", "cache/" + self.file: None},
        )
        cmd = shlex.split(ff.cmd)
        print(cmd)
        self.proc = None
        try:
            self.proc = psutil.Popen(cmd, stderr=subprocess.PIPE)
            print(self.proc)
            fl = 0
            while fl < 65536:
                if not self.proc.is_running():
                    err = self.proc.stderr.read()
                    if err:
                        ex = RuntimeError(err)
                    else:
                        ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
                    raise ex
                time.sleep(0.1)
                try:
                    fl = os.path.getsize("cache/" + self.file)
                except FileNotFoundError:
                    fl = 0
            self.time = time.time()
        except:
            try:
                ytdl.cache.pop(self.file)
            except KeyError:
                pass
            if self.proc is not None:
                try:
                    self.proc.kill()
                except:
                    print(traceback.format_exc())
            raise

    def update(self):
        if time.time() - self.time > 18000:
            self.expired = True
            try:
                ytdl.cache.pop(self.file)
            except KeyError:
                pass
            for _ in loop(16):
                try:
                    os.remove("cache/" + self.file)
                except FileNotFoundError:
                    break
                except PermissionError:
                    time.sleep(5)
                except:
                    print(traceback.format_exc())
                    time.sleep(5)
            try:
                self.proc.kill()
            except:
                pass

    def open(self):
        self.update()
        if self.proc is None:
            raise ProcessLookupError
        return open("cache/" + self.file, "rb")

    def create_reader(self, pos=0, auds=None):
        stats = auds.stats
        auds.reverse = stats.speed < 0
        if auds.speed < 0.005:
            auds.speed = 1
            auds.paused |= 2
        else:
            auds.paused &= -3
        stats.position = pos
        if not isValid(stats.pitch * stats.speed):
            raise OverflowError
        # if auds.reverse:
        #     start = 0
        #     end = pos
        # else:
        start = pos
        end = float(auds.queue[0].duration) + 0.5
        pitchscale = 2 ** (stats.pitch / 12)
        if stats.resample >= 2400:
            pitchscale *= 2 ** (stats.resample / 12)
        chorus = min(16, abs(stats.chorus))
        if pitchscale != 1 or stats.speed != 1:
            speed = abs(stats.speed) / pitchscale
            if stats.resample >= 2400:
                speed *= 2 ** (stats.resample / 12)
            if round(speed, 9) != 1:
                speed = max(0.005, speed)
                if speed >= 64:
                    raise OverflowError
                opts = ""
                while speed > 2:
                    opts += "atempo=2,"
                    speed /= 2
                while speed < 0.5:
                    opts += "atempo=0.5,"
                    speed /= 0.5
                opts += "atempo=" + str(speed)
                options = "-af " + opts
            else:
                options = "-af "
        else:
            options = ""
        if pitchscale != 1:
            if abs(pitchscale) >= 64:
                raise OverflowError
            if options and options[-1] != " ":
                options += ","
            options += "asetrate=r=" + str(SAMPLE_RATE * pitchscale)
        if auds.reverse:
            if options and options[-1] != " ":
                options += ","
            options += "areverse"
        if chorus:
            if not options:
                options = "-af "
            else:
                options += ","
            A = ""
            B = ""
            C = ""
            D = ""
            for i in range(ceil(chorus)):
                neg = ((i & 1) << 1) - 1
                i = 1 + i >> 1
                i *= stats.chorus / ceil(chorus)
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
                depth = (i * 0.43 * neg) % 4 + 0.5
                D += str(round(depth, 3))
            b = 0.5 / sqrt(ceil(chorus + 1))
            options += (
                "\"chorus=0.5:" + str(round(b, 3)) + ":"
                + A + ":"
                + B + ":"
                + C + ":"
                + D + "\""
            )
        if stats.compressor:
            comp = min(8000, abs(stats.compressor + sgn(stats.compressor)))
            while abs(comp) > 1:
                if not options:
                    options = "-af "
                else:
                    options += ","
                c = min(20, comp)
                try:
                    comp /= c
                except ZeroDivisionError:
                    comp = 1
                mult = str(round(math.sqrt(c), 4))
                options += (
                    "acompressor=mode=" + ("downward", "upward")[stats.compressor < 0]
                    + ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
                )
        if stats.pan != 1:
            pan = min(10000, max(-10000, stats.pan))
            while abs(abs(pan) - 1) > 0.001:
                if not options:
                    options = "-af "
                else:
                    options += ","
                p = max(-10, min(10, pan))
                try:
                    pan /= p
                except ZeroDivisionError:
                    pan = 1
                options += (
                    "extrastereo=m=" + str(p) + ":c=0,volume=" + str(1 / max(1, round(math.sqrt(abs(p)), 4)))
                )
        options = options.strip()
        if self.proc.is_running():
            if "-af" in options:
                args = [
                    "ffmpeg",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-i",
                    "pipe:0",
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                ]
            else:
                args = [
                    "ffmpeg",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-i",
                    "pipe:0",
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-f",
                    "s16le",
                    "-acodec",
                    "copy",
                ]
            args += shlex.split(options) + ["-loglevel", "error", "pipe:1"]
            player = BufferedAudioReader(self, args)
            create_future(player.run)
            return player
        else:
            if "-af" in options:
                args = [
                    "ffmpeg",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-i",
                    "cache/" + self.file,
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                ]
            else:
                args = [
                    "ffmpeg",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-i",
                    "cache/" + self.file,
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-f",
                    "s16le",
                    "-acodec",
                    "copy",
                ]
            args += shlex.split(options) + ["-loglevel", "error", "pipe:1"]
            player = LoadedAudioReader(self, args)
            create_future(player.run)
            return player


class LoadedAudioReader(discord.AudioSource):

    def __init__(self, file, args):
        print(args)
        self.closed = False
        self.proc = psutil.Popen(args, stdout=subprocess.PIPE)
        print(self.proc)

    read = lambda self, *args, **kwargs: self.proc.stdout.read(*args, **kwargs)
    run = lambda self: None
    process = read

    def close(self):
        self.closed = True
        try:
            self.proc.kill()
        except:
            pass

    is_opus = lambda self: False
    cleanup = close


class BufferedAudioReader(discord.AudioSource):

    def __init__(self, file, args):
        print(args)
        self.closed = False
        self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        print(self.proc)
        self.stream = file.open()
        self.buffer = bytes()
        self.loader = file.proc

    def read(self, size=0):
        while len(self.buffer) < size:
            b = self.process(8192)
            if not b:
                raise EOFError
            self.buffer += b
        temp = self.buffer[:size]
        self.buffer = self.buffer[size:]
        if len(self.buffer) < 65536:
            self.buffer += self.process(65536)
        return temp

    def run(self):
        while True:
            b = bytes()
            try:
                b = self.stream.read(65536)
                if not b:
                    raise EOFError
                self.proc.stdin.write(b)
                self.proc.stdin.flush()
            except (ValueError, EOFError, IndexError, BrokenPipeError):
                if not self.loader.is_running():
                    break
                time.sleep(0.1)
        self.proc.stdin.close()

    process = lambda self, *args, **kwargs: self.proc.stdout.read(*args, **kwargs)

    def close(self):
        self.closed = True
        try:
            self.stream.close()
        except:
            pass
        try:
            self.proc.kill()
        except:
            pass

    is_opus = lambda self: False
    cleanup = close

    
class videoDownloader:
    
    opener = urlBypass()
    _globals = globals()
    ydl_opts = {
        # "verbose": 1,
        "quiet": 1,
        "format": "bestaudio/best",
        "nocheckcertificate": 1,
        "no_call_home": 1,
        "nooverwrites": 1,
        "noplaylist": 1,
        "logtostderr": 0,
        "ignoreerrors": 0,
        "default_search": "auto",
        "source_address": "0.0.0.0",
    }

    def __init__(self):
        self.lastclear = 0
        self.downloading = {}
        self.cache = {}
        self.searched = {}
        self.requests = 0
        self.update_dl()

    def update_dl(self):
        if time.time() - self.lastclear > 3600:
            self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
            self.lastclear = time.time()

    def extract_true(self, url):
        if isURL(url):
            return self.downloader.extract_info(url, download=False, process=False)
        return self.downloader.extract_info(url, download=False, process=False)
        # pyt = create_future_ex(pytube2Dict, url)
        # resp = self.extract_info(url, search=False)
        # try:
        #     res = pyt.result(timeout=10)
        #     resp["formats"], resp["duration"] = res["formats"], res["duration"]
        # except youtube_dl.DownloadError:
        #     pass
        # except:
        #     print(traceback.format_exc())
        # if issubclass(type(data), Exception):
        #     raise data
        # return resp

    def extract(self, item, force=False, count=1, search=True):
        self.update_dl()
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
                                r.close()
                                raise ConnectionError(rescode)
                            continue
                        except:
                            time.sleep(i + 1)
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
                        search = "ytsearch:" + item.replace(":", "-")
                        temp = {
                            # "hash": shash(search),
                            "name": item,
                            "url": search,
                            "duration": float(duration),
                            "research": True,
                            }
                        sys.stdout.write(repr(temp) + "\n")
                        output.append(freeClass(temp))
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
                                        output.append(create_future_ex(self.extract, x, "spotify"))
                                except ValueError:
                                    break
                            outlist = []
                            for i in output:
                                if i:
                                    outlist += i.result()
                            output = outlist
                            # sys.stdout.write(repr(outlist) + "\n\n")
                        else:
                            t = '<meta name="description" content="'
                            s = s[s.index(t) + len(t):]
                            item = htmlDecode(s[:s.index('" />')]).replace(" on Spotify", "")
                            # sys.stdout.write(item + "\n")
                except urllib.error.URLError:
                    pass
            if r is not None:
                r.close()
            if not len(output) and force != "spotify":
                resp = self.extract_info(item, count, search=search)
                if resp.get("_type", None) == "url":
                    resp = self.extract_true(resp["url"])
                if resp is None or not len(resp):
                    raise EOFError("No search results found.")
                if resp.get("_type", None) == "playlist":
                    entries = list(resp["entries"])
                    if force or len(entries) <= 1:
                        for entry in entries:
                            data = self.extract_true(entry["url"])
                            temp = {
                                "name": data["title"],
                                "url": data["webpage_url"],
                                "duration": float(data["duration"]),
                            }
                            output.append(freeClass(temp))
                    else:
                        for i, entry in enumerate(entries):
                            if not i:
                                temp = self.extract(entry["url"], search=False)[0]
                            else:
                                try:
                                    found = True
                                    if "title" in entry:
                                        title = entry["title"]
                                    else:
                                        title = entry["url"].split("/")[-1]
                                        found = False
                                    if "duration" in entry:
                                        dur = float(entry["duration"])
                                    else:
                                        dur = None
                                    url = entry.get("webpage_url", entry.get("url", entry["id"]))
                                    temp = {
                                        # "hash": shash(entry["url"]),
                                        "name": title,
                                        "url": url,
                                        "duration": dur,
                                    }
                                    if not isURL(url):
                                        if entry.get("ie_key", "").lower() == "youtube":
                                            temp["url"] = "https://www.youtube.com/watch?v=" + url
                                    if dur is None:
                                        temp["duration"] = "300"
                                    temp["research"] = True
                                except:
                                    print(traceback.format_exc())
                            output.append(freeClass(temp))
                else:
                    found = "duration" in resp
                    if found:
                        dur = resp["duration"]
                    else:
                        dur = None
                    temp = {
                        # "hash": shash(resp["webpage_url"]),
                        "name": resp["title"],
                        "url": resp["webpage_url"],
                        "duration": dur,
                        "stream": getBestAudio(resp),
                    }
                    if dur is None:
                        temp["duration"] = getDuration(temp["stream"])
                    output.append(freeClass(temp))
            return output
        except:
            if force != "spotify":
                raise
            print(traceback.format_exc())
            return 0

    def extract_info(self, item, count=1, search=False):
        self.update_dl()
        if search and not item.startswith("ytsearch:") and not isURL(item):
            item = item.replace(":", "-")
            if count == 1:
                c = ""
            else:
                c = str(count)
            exc = ""
            try:
                return self.downloader.extract_info("ytsearch" + c + ":" + item, download=False, process=False)
            except Exception as ex:
                exc = repr(ex)
            try:
                return self.downloader.extract_info("scsearch" + c + ":" + item, download=False, process=False)
            except Exception as ex:
                raise ConnectionError(exc + repr(ex))
        if isURL(item) or not search:
            return self.extract_true(item)
        return self.downloader.extract_info(item, download=False, process=False)

    def search(self, item, force=False):
        item = verifySearch(item)
        while self.requests > 4:
            time.sleep(0.1)
        if item in self.searched:
            if time.time() - self.searched[item].t < 18000:
                # self.searched[item].t = time.time()
                return self.searched[item].data
            else:
                self.searched.pop(item)
        while len(self.searched) > 262144:
            self.searched.pop(next(iter(self.searched)))
        self.update_dl()
        try:
            self.requests += 1
            # print(item)
            obj = freeClass(t=time.time())
            obj.data = output = self.extract(item, force)
            self.searched[item] = obj
            self.requests = max(self.requests - 1, 0)
            return output
        except Exception as ex:
            print(traceback.format_exc())
            self.requests = max(self.requests - 1, 0)
            return repr(ex)
        
    def getStream(self, entry, force=False, download=True):
        stream = entry.get("stream", None)
        if stream == "none" and not force:
            return None
        entry["stream"] = "none"
        if stream in (None, "none") or stream.startswith("https://cf-hls-media.sndcdn.com/"):
            data = self.extract(entry["url"], search=False)
            stream = data[0].setdefault("stream", data[0].url)
        # print(stream)
        h = shash(entry["url"])
        fn = h + ".pcm"
        if fn in self.cache or not download:
            entry["stream"] = stream
            return self.cache.get(fn, None)
        try:
            self.cache[fn] = f = PCMFile(fn)
            f.load(stream)
            entry["stream"] = stream
            return f
        except:
            print(traceback.format_exc())
            entry["url"] = ""
    
    def download_file(self, url, fmt="ogg", fl=8388608):
        self.update_dl()
        fn = "cache/&" + str(time_snowflake(datetime.datetime.utcnow())) + "." + fmt
        info = self.extract(url)[0]
        stream = self.getStream(info, force=True, download=False)
        duration = getDuration(stream)
        if type(duration) is str:
            dur = 960
        else:
            dur = min(960, duration)
        br = max(32, min(256, floor(((fl - 262144) / dur / 128) / 4) * 4))
        # print(br)
        ff = ffmpy.FFmpeg(
            global_options=["-hide_banner", "-loglevel error", "-vn"],
            inputs={stream: None},
            outputs={str(dur): "-to", str(br) + "k": "-b:a", fn: None},
        )
        ff.run()
        return fn, info["name"] + "." + fmt

    def extractSingle(self, i):
        item = i.url
        while self.requests > 4:
            time.sleep(0.1)
        if item in self.searched:
            if time.time() - self.searched[item].t < 18000:
                # self.searched[item].t = time.time()
                it = self.searched[item].data[0]
                i.name = it.name
                i.duration = it.duration
                i.url = it.url
                # i.hash = gethash(it)
                return True
            else:
                self.searched.pop(item)
        while len(self.searched) > 262144:
            self.searched.pop(next(iter(self.searched)))
        self.update_dl()
        try:
            self.requests += 1
            data = self.extract_true(item)
            if "entries" in data:
                data = data["entries"][0]
            obj = freeClass(t=time.time())
            obj.data = out = [freeClass(
                name=data["title"],
                url=data["webpage_url"],
                stream=getBestAudio(data),
            )]
            try:
                out[0].duration = data["duration"]
            except KeyError:
                out[0].research = True
            self.searched[item] = obj
            it = out[0]
            i.name = it.name
            i.duration = it.get("duration", "300")
            i.url = it.url
            # sethash(i)
            self.requests = max(self.requests - 1, 0)
        except:
            self.requests = max(self.requests - 1, 0)
            i.url = ""
            print(traceback.format_exc())
        return True

ytdl = videoDownloader()


class Queue(Command):
    server_only = True
    name = ["Q", "Play", "P"]
    alias = name + ["LS"]
    min_level = 0
    description = "Shows the music queue, or plays a song in voice."
    usage = "<search_link[]> <verbose(?v)> <hide(?h)> <force(?f)> <budge(?b)>"
    flags = "hvfbz"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac']

    async def __call__(self, bot, client, user, perm, message, channel, guild, flags, name, argv, **void):
        if not argv:
            if message.attachments:
                argv = message.attachments[0].url
        if not argv:
            auds = await forceJoin(guild, channel, user, client, bot)
            elapsed = auds.stats.position
            q = auds.queue
            v = "v" in flags
            if not v and len(q) and auds.paused & 1 and "p" in name:
                auds.paused &= -2
                auds.pausec = False
                auds.preparing = False
                if auds.stats.position <= 0:
                    if auds.queue:
                        try:
                            auds.queue[0].pop("played")
                        except (KeyError, IndexError):
                            pass
                create_future(auds.update)
                return "```css\nSuccessfully resumed audio playback in [" + noHighlight(guild.name) + "].```", 1
            if not len(q):
                auds.preparing = False
                create_future(auds.update)
                return "```ini\nQueue for [" + noHighlight(guild.name) + "] is currently empty. ```", 1
            return (
                "```" + "\n" * ("z" in flags) + "callback-voice-queue-"
                + str(user.id) + "_0_" + str(int(v))
                + "-\nQueue for " + guild.name.replace("`", "") + ":```"
            )
        future = wrap_future(create_task(forceJoin(guild, channel, user, client, bot, preparing=True)))
        argv = await bot.followURL(argv)
        resp = await create_future(ytdl.search, argv)
        auds = await future
        if "f" in flags or "b" in flags:
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to force play while other users are in voice")
        if auds.stats.quiet & 2:
            flags.setdefault("h", 1)
        elapsed = auds.stats.position
        q = auds.queue
        if type(resp) is str:
            raise evalEX(resp)
        added = deque()
        names = []
        for e in resp:
            name = e.name
            url = e.url
            temp = {
                # "hash": e.hash,
                "name": name,
                "url": url,
                "duration": e.get("duration", "300"),
                "u_id": user.id,
                "skips": [],
            }
            if "research" in e:
                temp["research"] = True
            added.append(freeClass(temp))
            names.append(noHighlight(name))
        if "b" not in flags:
            total_duration = 0
            for e in q:
                total_duration += float(e.duration)
            if auds.reverse and len(auds.queue):
                total_duration += elapsed - float(q[0].duration)
            else:
                total_duration -= elapsed
        if auds.stats.shuffle:
            added = shuffle(added)
        tdur = 3
        if "f" in flags:
            for i in range(3):
                try:
                    auds.queue[i].pop("played")
                except (KeyError, IndexError):
                    pass
            auds.queue.enqueue(added, 0)
            create_future(auds.new)
            total_duration = tdur
        elif "b" in flags:
            auds.queue.enqueue(added, 1)
            total_duration = q[0].duration
        else:
            auds.queue.enqueue(added, -1)
            total_duration = max(total_duration / auds.speed, tdur)
        if not names:
            raise LookupError("No results for " + str(argv) + ".")
        if "v" in flags:
            names = noHighlight(hlist(i.name + ": " + dhms(i.duration) for i in added))
        elif len(names) == 1:
            names = names[0]
        else:
            names = str(len(names)) + " items"
        if "h" not in flags:
            return (
                "```css\n🎶 Added [" + names
                + "] to the queue! Estimated time until playing: ["
                + sec2Time(total_duration) + "]. 🎶```", 1
            )

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos, v = [int(i) for i in vals.split("_")]
        # print(vals, reaction, message)
        if reaction is not None and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        auds = await forceJoin(guild, message.channel, user, bot.client, bot)
        q = auds.queue
        last = max(0, len(q) - 10)
        if reaction is not None:
            i = self.directions.index(reaction)
            if i == 0:
                new = 0
            elif i == 1:
                new = max(0, pos - 10)
            elif i == 2:
                new = min(last, pos + 10)
            else:
                new = last
            if new == pos:
                return
            pos = new
        content = message.content
        if not content:
            content = message.embeds[0].description
        i = content.index("callback")
        content = content[:i] + (
            "callback-voice-queue-"
            + str(u_id) + "_" + str(pos) + "_" + str(int(v))
            + "-\nQueue for " + guild.name.replace("`", "") + ":\n"
        )
        elapsed = auds.stats.position
        startTime = 0
        if auds.stats.loop:
            totalTime = inf
        else:
            if auds.reverse and len(auds.queue):
                totalTime = elapsed - float(auds.queue[0].duration)
            else:
                totalTime = -elapsed
            i = 0
            for e in q:
                totalTime += float(e.duration)
                if i < pos:
                    startTime += float(e.duration)
                if not 1 + i & 4095:
                    await asyncio.sleep(0.2)
                i += 1
        cnt = len(q)
        info = (
            str(cnt) + " item" + "s" * (cnt != 1) + ", estimated total duration: "
            + sec2Time(totalTime / auds.speed) + "```"
        )
        duration = float(q[0].duration)
        sym = "⬜⬛"
        barsize = 24
        r = round(min(1, elapsed / duration) * barsize)
        bar = sym[0] * r + sym[1] * (barsize - r)
        countstr = "Currently playing [" + discord.utils.escape_markdown(q[0].name) + "](" + q[0].url + ")\n"
        countstr += (
            "`(" + uniStr(dhms(elapsed))
            + "/" + uniStr(dhms(duration)) + ") "
        )
        countstr += bar + "`\n"
        emb = discord.Embed(
            description=content + info + countstr,
            colour=randColour(),
        )
        user = await bot.fetch_user(u_id)
        url = strURL(user.avatar_url)
        for size in ("?size=1024", "?size=2048"):
            if url.endswith(size):
                url = url[:-len(size)] + "?size=4096"
        emb.set_author(name=str(user), url=url, icon_url=url)
        embstr = ""
        currTime = startTime
        i = pos
        while i < min(pos + 10, len(q)):
            e = q[i]
            curr = "`"
            curr += " " * (int(math.log10(len(q))) - int(math.log10(max(1, i))))
            curr += "【" + str(i) + "】` ["
            curr += discord.utils.escape_markdown(limStr(noHighlight(e.name), 64))
            curr += "](" + ensure_url(e.url) + ") `("
            curr += dhms(e.duration) + ")`"
            if v:
                try:
                    u = bot.cache.users[e.u_id]
                    name = u.display_name
                except KeyError:
                    try:
                        u = await bot.fetch_user(e.u_id)
                        name = u.display_name
                    except:
                        print(traceback.format_exc())
                        name = "Deleted User"
                curr += "\n```css\n" + sbHighlight(name) + "\n"
            if auds.reverse and len(auds.queue):
                estim = currTime + elapsed - float(auds.queue[0].duration)
            else:
                estim = currTime - elapsed
            if v:
                if estim > 0:
                    curr += "Time until playing: "
                    estimate = sec2Time(estim / auds.speed)
                    if i <= 1 or not auds.stats.shuffle:
                        curr += "[" + estimate + "]"
                    else:
                        curr += "{" + estimate + "}"
                else:
                    curr += "Remaining time: [" + sec2Time((estim + float(e.duration)) / auds.speed) + "]"
                curr += "```"
            curr += "\n"
            if len(embstr) + len(curr) > 2048 - len(emb.description):
                break
            embstr += curr
            if i <= 1 or not auds.stats.shuffle:
                currTime += float(e.duration)
            if not 1 + 1 & 4095:
                await asyncio.sleep(0.3)
            i += 1
        emb.description += embstr
        if pos != last:
            emb.set_footer(
                text=uniStr("And ", 1) + str(len(q) - i) + uniStr(" more...", 1),
            )
        await message.edit(content=None, embed=emb)
        if reaction is None:
            for react in self.directions:
                await message.add_reaction(react.decode("utf-8"))


class Playlist(Command):
    server_only = True
    name = ["DefaultPlaylist", "PL"]
    min_level = 0
    min_display = "0~2"
    description = "Shows, appends, or removes from the default playlist."
    usage = "<search_link[]> <remove(?d)> <verbose(?v)>"
    flags = "aedv"

    async def __call__(self, user, argv, guild, flags, channel, perm, **void):
        update = self.bot.database.playlists.update
        bot = self.bot
        pl = bot.data.playlists
        if argv or "d" in flags:
            req = 2
            if perm < req:
                reason = (
                    "to modify default playlist for "
                    + guild.name
                )
                self.permError(perm, req, reason)
        pl = pl.setdefault(guild.id, [])
        if not argv:
            if "d" in flags:
                pl[guild.id] = []
                update()
                return (
                    "```css\nRemoved all entries from the default playlist for "
                    + sbHighlight(guild) + ".```"
                )
            if not pl:
                return (
                    "```ini\nDefault playlist for " + sbHighlight(guild)
                    + " is currently empty.```"
                )
            if "v" in flags:
                key = lambda x: noHighlight(x)
                s = strIter(pl, key=key).replace("'", '"')
            else:
                key = lambda x: limStr(noHighlight(x["name"]), 1900 / len(pl) - 10)
                s = strIter(pl, key=key)
            return (
                "Current default playlist for **" + discord.utils.escape_markdown(guild.name)
                + "**: ```ini\n" + s + "```"
            )
        if "d" in flags:
            i = await bot.evalMath(argv, guild.id)
            temp = pl[i]
            pl.pop(i)
            update()
            return (
                "```css\nRemoved " + sbHighlight(temp.name)
                + " from the default playlist for "
                + sbHighlight(guild.name) + "```"
            )
        if len(pl) >= 64:
            raise OverflowError(
                "Playlist size for " + guild.name
                + " has reached the maximum of 64 items. "
                + "Please remove an item to add another."
            )
        argv = await bot.followURL(argv)
        resp = await create_future(ytdl.search, argv)
        if type(resp) is str:
            raise evalEX(resp)
        names = []
        for e in resp:
            name = e.name
            names.append(noHighlight(name))
            pl.append({
                "name": name,
                "url": e.url,
                "duration": e.duration,
            })
        if not names:
            raise LookupError("No results for " + argv + ".")
        pl.sort(key=lambda x: x["name"].lower())
        update()
        return (
            "```css\nAdded " + sbHighlight(", ".join(names))
            + " to the default playlist for "
            + sbHighlight(guild.name) + ".```"
        )
        

class Connect(Command):
    server_only = True
    name = ["Summon", "Join", "DC", "Disconnect", "Move", "Reconnect"]
    min_level = 0
    description = "Summons the bot into a voice channel."
    usage = "<channel{curr}(0)>"

    async def __call__(self, user, channel, name="join", argv="", **void):
        bot = self.bot
        client = bot.client
        if name in ("dc", "disconnect"):
            vc_ = None
        elif argv or name == "move":
            c_id = verifyID(argv)
            if not c_id > 0:
                vc_ = None
            else:
                vc_ = await bot.fetch_channel(c_id)
        else:
            voice = user.voice
            if voice is None:
                raise LookupError("Unable to find voice channel.")
            vc_ = voice.channel
        connecting = bot.database.playlists.connecting
        if vc_ is None:
            guild = channel.guild
        else:
            guild = vc_.guild
        perm = bot.getPerms(user, guild)
        if perm < 0:
            self.permError(perm, 0, "for command " + self.name + " in " + str(guild))
        if vc_ is None:
            try:
                auds = bot.database.playlists.audio[guild.id]
            except KeyError:
                raise LookupError("Unable to find connected channel.")
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to disconnect while other users are in voice")
            auds.dead = True
            try:
                connecting.pop(guild.id)
            except KeyError:
                pass
            await bot.database.playlists(guild=guild)
            return
        joined = False
        for vc in client.voice_clients:
            if vc.guild.id == guild.id:
                joined = True
                if vc.channel.id != vc_.id:
                    connecting[guild.id] = time.time()
                    try:
                        await vc.move_to(vc_)
                        connecting[guild.id] = 0
                    except:
                        connecting[guild.id] = 0
                        raise
                break
        if not joined:
            connecting[guild.id] = time.time()
            vc = freeClass(is_connected = lambda: False)
            t = time.time()
            while not vc.is_connected() and time.time() - t < 12:
                try:
                    vc = await vc_.connect(timeout=30, reconnect=False)
                    for _ in loop(8):
                        if vc.is_connected():
                            break
                        await asyncio.sleep(0.5)
                except discord.ClientException:
                    print(traceback.format_exc())
                    await asyncio.sleep(1)
            if isinstance(vc, freeClass):
                connecting[guild.id] = 0
                raise ConnectionError("Unable to connect to voice channel.")
        if guild.id not in bot.database.playlists.audio:
            await channel.trigger_typing()
            bot.database.playlists.audio[guild.id] = auds = customAudio(channel, vc, bot)
        try:
            joined = connecting.pop(guild.id)
        except KeyError:
            joined = False
        for user in guild.members:
            if user.id == client.user.id:
                if hasattr(user, "voice") and user.voice is not None:
                    if not (user.voice.deaf or user.voice.mute or user.voice.afk):
                        break
                create_task(user.edit(mute=False, deafen=False))
                break
        if joined:
            await bot.database.playlists(guild=guild)
            return (
                "```css\n🎵 Successfully connected to [#" + noHighlight(vc_.name)
                + "] in [" + noHighlight(guild.name) + "]. 🎵```", 1
            )


class Skip(Command):
    server_only = True
    name = ["Remove", "Rem", "S", "SK", "ClearQueue", "Clear", "CQ"]
    min_level = 0
    min_display = "0~1"
    description = "Removes an entry or range of entries from the voice channel queue."
    usage = "<0:queue_position[0]> <force(?f)> <vote(?v)> <hide(?h)>"
    flags = "fhv"

    async def __call__(self, client, user, perm, bot, name, args, argv, guild, flags, message, **void):
        if guild.id not in bot.database.playlists.audio:
            raise LookupError("Currently not playing in a voice channel.")
        auds = bot.database.playlists.audio[guild.id]
        if name.lower().startswith("c"):
            argv = "inf"
            args = [argv]
            flags["f"] = True
        if "f" in flags:
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to force skip while other users are in voice")
        if not argv:
            elems = [0]
        elif ":" in argv or ".." in argv:
            while "..." in argv:
                argv = argv.replace("...", "..")
            l = argv.replace("..", ":").split(":")
            it = None
            if len(l) > 3:
                raise ValueError("Too many arguments for range input.")
            elif len(l) > 2:
                num = await bot.evalMath(l[0], guild.id)
                it = int(round(float(num)))
            if l[0]:
                num = await bot.evalMath(l[0], guild.id)
                if num > len(auds.queue):
                    num = len(auds.queue)
                else:
                    num = round(num) % len(auds.queue)
                left = num
            else:
                left = 0
            if l[1]:
                num = await bot.evalMath(l[1], guild.id)
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
                elems[i] = await bot.evalMath(args[i], guild.id)
        if not "f" in flags:
            valid = True
            for e in elems:
                if not isValid(e):
                    valid = False
                    break
            if not valid:
                elems = range(len(auds.queue))
        members = sum(1 for m in auds.vc.channel.members if not m.bot)
        required = 1 + members >> 1
        response = "```css\n"
        i = 1
        for pos in elems:
            try:
                if not isValid(pos):
                    if "f" in flags:
                        auds.queue.clear()
                        create_future(auds.new)
                        if "h" not in flags:
                            return "```fix\nRemoved all items from the queue.```", 1
                        return
                    raise LookupError
                curr = auds.queue[pos]
            except LookupError:
                response += "\n" + repr(IndexError("Entry " + str(pos) + " is out of range."))
                continue
            if type(curr.skips) is list:
                if "f" in flags or user.id == curr["u_id"] and not "v" in flags:
                    curr.skips = None
                elif user.id not in curr.skips:
                    curr.skips.append(user.id)
            elif "v" in flags:
                curr.skips = [user.id]
            else:
                curr.skips = None
            if curr.skips is not None:
                if len(response) > 1200:
                    response = limStr(response, 1200)
                else:
                    response += (
                        "Voted to remove [" + noHighlight(curr.name)
                        + "] from the queue.\nCurrent vote count: ["
                        + str(len(curr.skips)) + "], required vote count: ["
                        + str(required) + "].\n"
                    )
            if not i & 2047:
                await asyncio.sleep(0.2)
            i += 1
        pops = deque()
        count = 0
        i = 1
        while i < len(auds.queue):
            q = auds.queue
            song = q[i]
            if song.skips is None or len(song.skips) >= required:
                if count <= 3:
                    q.pop(i)
                else:
                    pops.append(i)
                    i += 1
                if count < 4:
                    response += (
                        "[" + noHighlight(song.name)
                        + "] has been removed from the queue.\n"
                    )
                count += 1
            else:
                i += 1
        if pops:
            auds.queue.pops(pops)
        if auds.queue:
            song = auds.queue[0]
            if song.skips is None or len(song.skips) >= required:
                auds.queue.popleft()
                create_future(auds.new)
                auds.preparing = False
                if count < 4:
                    response += (
                        "[" + noHighlight(song.name)
                        + "] has been removed from the queue.\n"
                    )
                count += 1
        if "h" not in flags:
            if count >= 4:
                return (
                    "```css\n[" + str(count)
                    + "] items have been removed from the queue.```"
                )
            return response + "```", 1


class Pause(Command):
    server_only = True
    name = ["Resume", "Unpause", "Stop"]
    min_level = 0
    min_display = "0~1"
    description = "Pauses, stops, or resumes audio playing."
    usage = "<hide(?h)>"
    flags = "h"

    async def __call__(self, bot, name, guild, client, user, perm, channel, flags, **void):
        name = name.lower()
        auds = await forceJoin(guild, channel, user, client, bot)
        auds.preparing = False
        if name in ("pause", "stop"):
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to " + name + " while other users are in voice")
        elif auds.stats.position <= 0:
            if auds.queue and "played" in auds.queue[0]:
                auds.queue[0].pop("played")
        if name == "stop":
            auds.seek(0)
        if not auds.paused > 1:
            auds.paused = name in ("pause", "stop")
            auds.pausec = False
        if auds.player is not None:
            auds.player.time = 1 + time.time()
        await bot.database.playlists(guild=guild)
        if "h" not in flags:
            past = name + "pe" * (name == "stop") + "d"
            return (
                "```css\nSuccessfully " + past + " audio playback in ["
                + noHighlight(guild.name) + "].```", 1
            )


class Seek(Command):
    server_only = True
    name = ["Replay"]
    min_level = 0
    min_display = "0~1"
    description = "Seeks to a position in the current audio file."
    usage = "<position[0]> <hide(?h)>"
    flags = "h"

    async def __call__(self, argv, bot, guild, client, user, perm, channel, name, flags, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if not isAlone(auds, user) and perm < 1:
            self.permError(perm, 1, "to seek while other users are in voice")
        if name == "replay":
            num = 0
        else:
            orig = auds.stats.position
            expr = argv
            _op = None
            for operator in ("+=", "-=", "*=", "/=", "%="):
                if expr.startswith(operator):
                    expr = expr[2:].strip(" ")
                    _op = operator[0]
            num = await bot.evalTime(expr, guild)
            if _op is not None:
                num = eval(str(orig) + _op + str(num), {}, infinum)
        pos = auds.seek(num)
        if auds.player is not None:
            auds.player.time = 1 + time.time()
        if "h" not in flags:
            return (
                "```css\nSuccessfully moved audio position to ["
                + noHighlight(sec2Time(pos)) + "].```", 1
            )


def getDump(auds):

    copyDict = lambda item: {"name": item.name, "url": item.url, "duration": item.duration}

    lim = 32768
    if len(auds.queue) > lim:
        raise OverflowError(
            "Too many items in queue (" + str(len(auds.queue))
            + " > " + str(lim) + ")."
        )
    q = [copyDict(item) for item in auds.queue if random.random() < 0.99 or not time.sleep(0.01)]
    s = dict(auds.stats)
    d = {
        "stats": s,
        "queue": q,
    }
    d["stats"].pop("position")
    return json.dumps(d)


class Dump(Command):
    server_only = True
    time_consuming = True
    name = ["Save", "Load", "Dujmpö"]
    min_level = 0
    min_display = "0~1"
    description = "Saves or loads the currently playing audio queue state."
    usage = "<data{attached_file}> <append(?a)> <hide(?h)>"
    flags = "ah"

    async def __call__(self, guild, channel, user, client, bot, perm, name, argv, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if not argv and not len(message.attachments) or name.lower() == "save":
            if name.lower() == "load":
                raise EOFError("Please input a file, URL or json data to load.")
            resp = await create_future(getDump, auds)
            f = discord.File(io.BytesIO(bytes(resp, "utf-8")), filename="dump.json")
            create_task(sendFile(channel, "Queue data for **" + guild.name + "**:", f))
            return
        if not isAlone(auds, user) and perm < 1:
            self.permError(perm, 1, "to load new queue while other users are in voice")
        try:
            if len(message.attachments):
                url = message.attachments[0].url
            else:
                url = verifyURL(argv)
            f = await downloadTextFile(url, bot)
            s = await f
            s = s[s.index("{"):]
            if s[-4:] == "\n```":
                s = s[:-4]
        except:
            s = argv
            print(traceback.format_exc())
        d = json.loads(s.strip("\n"))
        q = d["queue"]
        for i in range(len(q)):
            e = q[i] = freeClass(q[i])
            e.u_id = user.id
            e.skips = []
            if not 1 + i & 2047:
                await asyncio.sleep(0.2)
        if auds.player is not None:
            auds.player.time = 1 + time.time()
        if auds.stats.shuffle:
            shuffle(q)
        for k in d["stats"]:
            if k not in auds.stats:
                d["stats"].pop(k)
            if k in "loop shuffle quiet":
                d["stats"][k] = bool(d["stats"][k])
            else:
                d["stats"][k] = float(d["stats"][k])
        if "a" not in flags:
            await create_future(auds.new)
            auds.preparing = True
            auds.queue.clear()
            auds.queue.__init__(q)
            auds.stats.update(d["stats"])
            create_future(auds.update)
            if "h" not in flags:
                return (
                    "```css\nSuccessfully loaded audio queue data for [" 
                    + noHighlight(guild.name) + "].```", 1
                )
        auds.queue.enqueue(q, -1)
        auds.stats.update(d["stats"])
        if "h" not in flags:
            return (
                "```css\nSuccessfully appended loaded data to queue for [" 
                + noHighlight(guild.name) + "].```", 1
            )
            

class AudioSettings(Command):
    server_only = True
    aliasMap = {
        "Volume": "volume",
        "Speed": "speed",
        "Pitch": "pitch",
        "Pan": "pan",
        "BassBoost": "bassboost",
        "Reverb": "reverb",
        "Compressor": "compressor",
        "Chorus": "chorus",
        "NightCore": "resample",
        "Resample": "resample",
        "LoopQueue": "loop",
        "ShuffleQueue": "shuffle",
        "Quiet": "quiet",
        "Reset": "reset",
    }
    aliasExt = {
        "AudioSettings": None,
        "Audio": None,
        "A": None,
        "Vol": "volume",
        "V": "volume",
        "SP": "speed",
        "PI": "pitch",
        "PN": "pan",
        "BB": "bassboost",
        "RV": "reverb",
        "CO": "compressor",
        "CH": "chorus",
        "NC": "resample",
        "LQ": "loop",
        "SQ": "shuffle",
    }

    def __init__(self, *args):
        self.alias = list(self.aliasMap) + list(self.aliasExt)[1:]
        self.name = list(self.aliasMap)
        self.min_level = 0
        self.min_display = "0~1"
        self.description = "Changes the current audio settings for this server."
        self.usage = (
            "<value[]> <volume()(?v)> <speed(?s)> <pitch(?p)> <pan(?e)> <bassboost(?b)> <reverb(?r)> <compressor(?c)>"
            + " <chorus(?u)> <nightcore(?n)> <loop(?l)> <shuffle(?x)> <quiet(?q)> <disable_all(?d)> <hide(?h)>"
        )
        self.flags = "vspbrcnlxqdh"
        self.map = {k.lower(): self.aliasMap[k] for k in self.aliasMap}
        addDict(self.map, {k.lower(): self.aliasExt[k] for k in self.aliasExt})
        super().__init__(*args)

    async def __call__(self, client, channel, user, guild, bot, flags, name, argv, perm, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        ops = hlist()
        op1 = self.map[name]
        if op1 == "reset":
            flags.clear()
            flags["d"] = True
        elif op1 is not None:
            ops.append(op1)
        disable = "d" in flags
        if "v" in flags:
            ops.append("volume")
        if "s" in flags:
            ops.append("speed")
        if "p" in flags:
            ops.append("pitch")
        if "e" in flags:
            ops.append("pan")
        if "b" in flags:
            ops.append("bassboost")
        if "r" in flags:
            ops.append("reverb")
        if "c" in flags:
            ops.append("compressor")
        if "u" in flags:
            ops.append("chorus")
        if "n" in flags:
            ops.append("resample")
        if "l" in flags:
            ops.append("loop")
        if "x" in flags:
            ops.append("shuffle")
        if "q" in flags:
            ops.append("quiet")
        if not disable and not argv and (len(ops) != 1 or ops[-1] not in "loop shuffle quiet"):
            if len(ops) == 1:
                op = ops[0]
            else:
                key = lambda x: (round(x * 100, 9), x)[type(x) is bool]
                d = dict(auds.stats)
                try:
                    d.pop("position")
                except KeyError:
                    pass
                return (
                    "Current audio settings for **" + discord.utils.escape_markdown(guild.name) + "**:\n```ini\n"
                    + strIter(d, key=key) + "```"
                )
            orig = bot.database.playlists.audio[guild.id].stats[op]
            num = round(100 * orig, 9)
            return (
                "```css\nCurrent audio " + op
                + " setting in [" + noHighlight(guild.name)
                + "]: [" + str(num) + "].```"
            )
        if not isAlone(auds, user) and perm < 1:
            self.permError(perm, 1, "to modify audio settings while other users are in voice")
        if not ops:
            if disable:
                pos = auds.stats.position
                auds.stats = freeClass(auds.defaults)
                if auds.queue:
                    await create_future(auds.new, auds.file, pos)
                return (
                    "```css\nSuccessfully reset all audio settings for ["
                    + noHighlight(guild.name) + "].```"
                )
            else:
                ops.append("volume")
        s = ""
        for op in ops:
            if type(op) is str and op in "loop shuffle quiet" and not argv:
                argv = str(not bot.database.playlists.audio[guild.id].stats[op])
            if disable:
                val = auds.defaults[op]
                if type(val) is not bool:
                    val *= 100
                argv = str(val)
            origVol = bot.database.playlists.audio[guild.id].stats
            _op = None
            for operator in ("+=", "-=", "*=", "/=", "%="):
                if argv.startswith(operator):
                    argv = argv[2:].strip(" ")
                    _op = operator[0]
            num = await bot.evalMath(argv, guild.id)
            orig = round(origVol[op] * 100, 9)
            if _op is not None:
                num = eval(str(orig) + _op + str(num), {}, infinum)
            val = roundMin(float(num / 100))
            new = round(val * 100, 9)
            if op in "loop shuffle quiet":
                origVol[op] = new = bool(val)
                orig = bool(orig)
            else:
                origVol[op] = val
            if auds.queue and (op in "speed pitch pan compressor chorus" or op == "resample" and max(orig, new) >= 240000):
                await create_future(auds.new, auds.file, auds.stats.position)
            s += (
                "\nChanged audio " + str(op)
                + " setting from [" + str(orig)
                + "] to [" + str(new) + "]."
            )
        if "h" not in flags:
            return "```css" + s + "```", 1


class Rotate(Command):
    server_only = True
    name = ["Jump"]
    min_level = 0
    min_display = "0~1"
    description = "Rotates the queue to the left by a certain amount of steps."
    usage = "<position> <hide(?h)>"
    flags = "h"

    async def __call__(self, perm, argv, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        amount = await bot.evalMath(argv, guild.id)
        if len(auds.queue) > 1 and amount:
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to rotate queue while other users are in voice")
            for i in range(3):
                try:
                    auds.queue[i].pop("played")
                except (KeyError, IndexError):
                    pass
            auds.queue.rotate(-amount)
            auds.seek(inf)
        if "h" not in flags:
            return (
                "```css\nSuccessfully rotated queue ["
                + str(amount) + "] step"
                + "s" * (amount != 1) + ".```", 1
            )


class Shuffle(Command):
    server_only = True
    min_level = 0
    min_display = "0~1"
    description = "Shuffles the audio queue."
    usage = "<hide(?h)>"
    flags = "h"

    async def __call__(self, perm, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if len(auds.queue) > 1:
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to shuffle queue while other users are in voice")
            for i in range(3):
                try:
                    auds.queue[i].pop("played")
                except (KeyError, IndexError):
                    pass
            shuffle(auds.queue)
            auds.seek(inf)
        if "h" not in flags:
            return (
                "```css\nSuccessfully shuffled queue for ["
                + noHighlight(guild.name) + "].```", 1
            )


class Reverse(Command):
    server_only = True
    min_level = 0
    min_display = "0~1"
    description = "Reverses the audio queue direction."
    usage = "<hide(?h)>"
    flags = "h"

    async def __call__(self, perm, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if len(auds.queue) > 1:
            if not isAlone(auds, user) and perm < 1:
                self.permError(perm, 1, "to reverse queue while other users are in voice")
            for i in range(1, 3):
                try:
                    auds.queue[i].pop("played")
                except (KeyError, IndexError):
                    pass
            reverse(auds.queue)
            auds.queue.rotate(-1)
        if "h" not in flags:
            return (
                "```css\nSuccessfully reversed queue for ["
                + noHighlight(guild.name) + "].```", 1
            )


class Unmute(Command):
    server_only = True
    time_consuming = True
    name = ["Unmuteall"]
    min_level = 3
    description = "Disables server mute for all members."
    usage = "<hide(?h)>"
    flags = "h"

    async def __call__(self, guild, flags, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                if user.voice is not None:
                    if user.voice.deaf or user.voice.mute or user.voice.afk:
                        create_task(user.edit(mute=False, deafen=False))
        if "h" not in flags:
            return (
                "```css\nSuccessfully unmuted all users in voice channels in ["
                + noHighlight(guild.name) + "].```", 1
            )


class VoiceNuke(Command):
    server_only = True
    time_consuming = True
    min_level = 3
    description = "Removes all users from voice channels in the current server."
    usage = "<hide(?h)>"
    flags = "h"

    async def __call__(self, guild, flags, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                if user.voice is not None:
                    create_task(user.move_to(None))
        if "h" not in flags:
            return (
                "```css\nSuccessfully removed all users in voice channels in ["
                + noHighlight(guild.name) + "].```", 1
            )


class Player(Command):
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
    name = ["NP", "NowPlaying", "Playing"]
    min_level = 0
    min_display = "0~3"
    description = "Creates an auto-updating virtual audio player for the current server."
    usage = "<controllable(?c)> <disable(?d)> <show_debug(?z)>"
    flags = "cdez"

    def showCurr(self, auds):
        q = auds.queue
        if q:
            s = q[0].skips
            if s is not None:
                skips = len(s)
            else:
                skips = 0
            output = "Playing " + noHighlight(q[0].name) + ", "
            output += str(len(q)) + " item" + "s" * (len(q) != 1) + " total "
            output += skips * "🚫"
        else:
            output = "Queue is currently empty. "
        if auds.stats.loop:
            output += "🔄"
        if auds.stats.shuffle:
            output += "🔀"
        if auds.stats.quiet:
            output += "🔕"
        output += "\n"
        v = abs(auds.stats.volume)
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
        b = auds.stats.bassboost
        if abs(b) > 1 / 3:
            if abs(b) > 5:
                output += "💥"
            elif b > 0:
                output += "🥁"
            else:
                output += "🎻"
        r = auds.stats.reverb
        if r:
            if abs(r) >= 1:
                output += "📈"
            else:
                output += "📉"
        c = auds.stats.chorus
        if c:
            output += "📊"
        s = auds.stats.speed * 2 ** (auds.stats.resample / 12)
        if s < 0:
            output += "⏪"
        elif s > 1:
            output += "⏩"
        elif s > 0 and s < 1:
            output += "🐌"
        p = auds.stats.pitch + auds.stats.resample
        if p > 0:
            output += "⏫"
        elif p < 0:
            output += "⏬"
        output += "\n"
        if auds.paused or not auds.stats.speed:
            output += "⏸️"
        elif auds.stats.speed > 0:
            output += "▶️"
        else:
            output += "◀️"
        if q:
            p = [auds.stats.position, float(q[0].duration)]
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

    async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, **void):
        if message is None:
            return
        if not guild.id in bot.database.playlists.audio:
            try:
                await message.clear_reactions()
            except (discord.NotFound, discord.Forbidden):
                pass
            return
        auds = bot.database.playlists.audio[guild.id]
        if reaction is None:
            auds.player = freeClass(
                time=inf,
                message=message,
                type=int(vals),
                events=0,
            )
            if auds.player.type:
                auds.stats.quiet |= 2
        elif auds.player is None or auds.player.message.id != message.id:
            try:
                await message.clear_reactions()
            except (discord.NotFound, discord.Forbidden):
                pass
            return
        if perm < 1:
            return
        orig = "\n".join(message.content.split("\n")[:1 + ("\n" == message.content[3])]) + "\n"
        if reaction is None and auds.player.type:
            for b in self.buttons:
                await message.add_reaction(b.decode("utf-8"))
        else:
            if not auds.player.type:
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
                    auds.stats.loop = bool(auds.stats.loop ^ 1)
                elif i == 2:
                    auds.stats.shuffle = bool(auds.stats.shuffle ^ 1)
                elif i == 3 or i == 4:
                    if i == 3:
                        pos = 0
                    else:
                        pos = inf
                    auds.seek(pos)
                    if pos:
                        return
                elif i == 5:
                    v = abs(auds.stats.volume)
                    if v < 0.25 or v >= 2:
                        v = 1 / 3
                    elif v < 1:
                        v = 1
                    else:
                        v = 2
                    auds.stats.volume = v
                elif i == 6:
                    b = auds.stats.bassboost
                    if abs(b) < 1 / 3:
                        b = 1
                    elif b < 0:
                        b = 0
                    else:
                        b = -1
                    auds.stats.bassboost = b
                elif i == 7:
                    r = auds.stats.reverb
                    if r:
                        r = 0
                    else:
                        r = 0.5
                    auds.stats.reverb = r
                elif i == 8:
                    c = abs(auds.stats.chorus)
                    if c:
                        c = 0
                    else:
                        c = 1 / 3
                    auds.stats.chorus = c
                    await create_future(auds.new, auds.file, auds.stats.position)
                elif i == 9 or i == 10:
                    s = (i * 2 - 19) * 2 / 11
                    auds.stats.speed = round(auds.stats.speed + s, 5)
                    await create_future(auds.new, auds.file, auds.stats.position)
                elif i == 11 or i == 12:
                    p = i * 2 - 23
                    auds.stats.pitch -= p
                    await create_future(auds.new, auds.file, auds.stats.position)
                elif i == 13:
                    pos = auds.stats.position
                    auds.stats = freeClass(auds.defaults)
                    await create_future(auds.new, auds.file, pos)
                elif i == 14:
                    auds.dead = True
                    auds.player = None
                    await bot.silentDelete(message)
                    return
                else:
                    auds.player = None
                    await bot.silentDelete(message)
                    return
        text = limStr(orig + self.showCurr(auds) + "```", 2000)
        last = message.channel.last_message
        if last is not None and (auds.player.type or message.id == last.id):
            auds.player.events += 1
            await message.edit(
                content=text,
            )
        else:
            auds.player.time = inf
            auds.player.events += 2
            channel = message.channel
            temp = message
            message = await channel.send(
                content=text,
            )
            auds.player.message = message
            await bot.silentDelete(temp, no_log=True)
        if auds.queue and not auds.paused & 1:
            maxdel = float(auds.queue[0].duration) - auds.stats.position + 2
            delay = min(maxdel, float(auds.queue[0].duration) / self.barsize / abs(auds.stats.speed))
            if delay > 20:
                delay = 20
            elif delay < 6:
                delay = 6
        else:
            delay = inf
        auds.player.time = time.time() + delay

    async def __call__(self, guild, channel, user, client, bot, flags, perm, **void):
        auds = await forceJoin(channel.guild, channel, user, client, bot)
        if "c" in flags or auds.stats.quiet & 2:
            req = 3
            if perm < req:
                if auds.stats.quiet & 2:
                    if "d" in flags:
                        reason = "delete"
                    else:
                        reason = "override"
                else:
                    reason = "create controllable"
                self.permError(perm, req, "to " + reason + " virtual audio player for " + noHighlight(guild.name))
        if "d" in flags:
            auds.player = None
            return (
                "```css\nDisabled virtual audio players in ["
                + noHighlight(channel.guild.name) + "].```"
            )
        await createPlayer(auds, p_type="c" in flags, verbose="z" in flags)


f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    genius_key = auth["genius_key"]
except:
    genius_key = None
    print("WARNING: genius_key not found. Unable to use API to search song lyrics.")


def get_lyrics(item):
    item = verifySearch(item)
    url = "https://api.genius.com/search"
    for i in range(3):
        header = {"user-agent": "Mozilla/5." + str(xrand(1, 10)), "Authorization": "Bearer " + genius_key}
        data = {"q": item}
        resp = requests.get(url, data=data, headers=header)
        rdata = json.loads(resp.content)
        hits = rdata["response"]["hits"]
        name = None
        path = None
        for i in hits:
            try:
                name = i["result"]["title"]
                path = i["result"]["api_path"]
                break
            except KeyError:
                pass
        if not (path and name):
            break
        page = requests.get("https://genius.com" + path, headers=header)
        html = BeautifulSoup(page.text, "html.parser")
        lyricobj = html.find('div', class_='lyrics')
        if lyricobj is not None:
            lyrics = lyricobj.get_text()
            return name, lyrics
        if i < 2:
            time.sleep(1)
    raise LookupError("No results for " + item + ".")


class Lyrics(Command):
    time_consuming = True
    name = ["SongLyrics"]
    min_level = 0
    description = "Searches genius.com for lyrics of a song."
    usage = "<0:search_link{queue}> <verbose(?v)>"
    flags = "v"

    async def __call__(self, bot, channel, message, argv, flags, user, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            try:
                auds = await forceJoin(channel.guild, channel, user, bot.client, bot)
                if not auds.queue:
                    raise EOFError
                argv = auds.queue[0].name
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        name, lyrics = await create_future(get_lyrics, argv)
        text = lyrics.strip()
        msg = "Lyrics for **" + discord.utils.escape_markdown(name) + "**:"
        if "v" not in flags:
            return limStr(msg + "```ini\n" + text + "```", 2000)
        emb = discord.Embed(colour=randColour())
        emb.set_author(name=name)
        curr = ""
        i = 1
        for para in text.split("\n\n"):
            if len(curr) + len(para) > 1000:
                emb.add_field(name="Page " + str(i), value="```ini\n" + curr + "```", inline=False)
                curr = para
                i += 1
            else:
                if curr:
                    curr += "\n\n"
                curr += para
        if curr:
            emb.add_field(name="Page " + str(i), value="```ini\n" + curr + "```", inline=False)
        return {
            "embed": emb
        }


class Download(Command):
    time_consuming = True
    _timeout_ = 8
    name = ["Search", "YTDL", "Youtube_DL", "Convert"]
    min_level = 0
    description = "Searches and/or downloads a song from a YouTube/SoundCloud query or link."
    usage = "<0:search_link{queue}> <-1:out_format[ogg]> <verbose(?v)> <show_debug(?z)>"
    flags = "vz"

    async def __call__(self, bot, channel, message, argv, flags, user, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            try:
                auds = await forceJoin(channel.guild, channel, user, bot.client, bot)
                if not auds.queue:
                    raise EOFError
                res = [{"name": e.name, "url": e.url} for e in auds.queue[:10]]
                fmt = "ogg"
                end = "Current items in queue for " + channel.guild.name + ":```"
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        else:
            if " " in argv:
                try:
                    spl = shlex.split(argv)
                except ValueError:
                    spl = argv.split(" ")
                if len(spl) > 1:
                    fmt = spl[-1]
                    if fmt.startswith("."):
                        fmt = fmt[1:]
                    if fmt not in ("mp3", "ogg", "webm", "wav"):
                        fmt = "ogg"
                    else:
                        argv = " ".join(spl[:-1])
                else:
                    fmt = "ogg"
            else:
                fmt = "ogg"
            # print(argv, fmt)
            argv = verifySearch(argv)
            res = []
            if isURL(argv):
                argv = await bot.followURL(argv)
                data = await create_future(ytdl.extract, argv)
                res += data
            if not res:
                sc = min(4, flags.get("v", 0) + 1)
                yt = min(6, sc << 1)
                searches = ["ytsearch" + str(yt), "scsearch" + str(sc)]
                returns = []
                for r in range(2):
                    returns.append(create_future(
                        ytdl.downloader.extract_info,
                        searches[r] + ":" + argv.replace(":", "~"),
                        download=False,
                        process=r,
                    ))
                returns = await recursiveCoro(returns)
                for r in returns:
                    try:
                        if type(r) is not str:
                            data = r
                            for e in data["entries"]:
                                if "webpage_url" in e:
                                    if "title" in e:
                                        res.append({
                                            "name": e["title"],
                                            "url": e["webpage_url"],
                                        })
                                    else:
                                        res.append({
                                            "name": e["id"],
                                            "url": e["webpage_url"],
                                        })
                                else:
                                    if e["ie_key"].lower() == "youtube":
                                        res.append({
                                            "name": e["title"],
                                            "url": "https://www.youtube.com/watch?v=" + e["url"],
                                        })
                        else:
                            print(r)
                    except:
                        print(r)
                        print(traceback.format_exc())
            if not res:
                raise LookupError("No results for " + argv + ".")
            res = res[:10]
            end = "Search results for " + argv + ":```"
        url_list = bytes2Hex(bytes(str([e["url"] for e in res]), "utf-8"), space=False)
        msg = (
            "```" + "\n" * ("z" in flags) + "callback-voice-download-" + str(user.id) 
            + "_" + str(len(res)) + "_" + url_list + "_" + fmt + "\n" + end
        )
        emb = discord.Embed(colour=randColour())
        url = strURL(user.avatar_url)
        for size in ("?size=1024", "?size=2048"):
            if url.endswith(size):
                url = url[:-len(size)] + "?size=4096"
        emb.set_author(name=str(user), url=url, icon_url=url)
        emb.description = "\n".join(
            ["`【" + str(i) + "】` [" + discord.utils.escape_markdown(e["name"] + "](" + ensure_url(e["url"]) + ")") for i in range(len(res)) for e in [res[i]]]
        )
        sent = await message.channel.send(
            msg,
            embed=emb,
        )
        for i in range(len(res)):
            await sent.add_reaction(str(i) + b"\xef\xb8\x8f\xe2\x83\xa3".decode("utf-8"))
        # await sent.add_reaction("❎")

    async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, user, **void):
        if reaction is None or user.id == bot.client.user.id:
            return
        spl = vals.split("_")
        u_id = int(spl[0])
        if user.id == u_id or not perm < 3:
            if b"\xef\xb8\x8f\xe2\x83\xa3" in reaction:
                num = int(reaction.decode("utf-8")[0])
                if num <= int(spl[1]):
                    data = ast.literal_eval(hex2Bytes(spl[2]).decode("utf-8"))
                    url = data[num]
                    if guild is None:
                        fl = 8388608
                    else:
                        fl = guild.filesize_limit
                    create_task(channel.trigger_typing())
                    await message.edit(
                        content="```ini\nDownloading [" + noHighlight(ensure_url(url)) + "]...```",
                        embed=None,
                    )
                    fn, out = await create_future(
                        ytdl.download_file,
                        url,
                        spl[3],
                        fl,
                    )
                    f = discord.File(fn, out)
                    create_task(message.edit(
                        content="```ini\nUploading [" + noHighlight(out) + "]...```",
                        embed=None,
                    ))
                    await sendFile(
                        channel=channel,
                        msg="",
                        file=f,
                        filename=fn
                    )
                    create_task(bot.silentDelete(message))


class UpdateQueues(Database):
    name = "playlists"

    def __init__(self, *args):
        self.audio = {}
        self.audiocache = {}
        self.connecting = {}
        super().__init__(*args)
        pl = self.data
        for g in pl:
            for i in range(len(pl[g])):
                e = pl[g][i]
                if type(e) is dict:
                    pl[g][i] = freeClass(e)

    def is_connecting(self, g):
        if g in self.connecting:
            if time.time() - self.connecting[g] < 12:
                return True
            self.connecting.pop(g)
        return False

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
                    await create_future(ytdl.extractSingle, i)
                    print(i.name)
                    searched += 1
                except:
                    print(traceback.format_exc())
                    break
            if random.random() > 0.99:
                await asyncio.sleep(0.4)
        await asyncio.sleep(2)
        auds.searching = max(auds.searching - 1, 0)

    async def _typing_(self, channel, user, **void):
        if not hasattr(channel, "guild") or channel.guild is None:
            return
        if channel.guild.id in self.audio and user.id != self.bot.client.user.id:
            auds = self.audio[channel.guild.id]
            if auds.player is not None and channel.id == auds.channel.id:
                t = time.time() + 10
                if auds.player.time < t:
                    auds.player.time = t

    async def _send_(self, message, **void):
        if message.guild.id in self.audio and message.author.id != self.bot.client.user.id:
            auds = self.audio[message.guild.id]
            if auds.player is not None and message.channel.id == auds.channel.id:
                t = time.time() + 10
                if auds.player.time < t:
                    auds.player.time = t

    async def _dc(self, member):
        try:
            await member.move_to(None)
        except discord.Forbidden:
            pass
        except:
            print(traceback.format_exc())

    async def __call__(self, guild=None, **void):
        if self.busy > 1:
            return
        self.busy += 1
        bot = self.bot
        client = bot.client
        try:
            if guild is not None:
                g = guild
                if not self.is_connecting(g.id) and g.id not in self.audio:
                    for c in g.voice_channels:
                        for m in c.members:
                            if m.id == client.user.id:
                                create_task(self._dc(m))
                else:
                    m = g.get_member(client.user.id)
                    if m.voice is not None:
                        if m.voice.deaf or m.voice.mute or m.voice.afk:
                            create_task(m.edit(mute=False, deafen=False))
            else:
                for vc in client.voice_clients:
                    if not self.is_connecting(vc.guild.id) and vc.guild.id not in self.audio:
                        create_task(vc.disconnect(force=True))
                for g in client.guilds:
                    if not self.is_connecting(g.id) and g.id not in self.audio:
                        for c in g.voice_channels:
                            for m in c.members:
                                if m.id == client.user.id:
                                    create_task(self._dc(m))
                    else:
                        m = g.get_member(client.user.id)
                        if m.voice is not None:
                            if m.voice.deaf or m.voice.mute or m.voice.afk:
                                create_task(m.edit(mute=False, deafen=False))
        except:
            print(traceback.format_exc())
        if guild is not None:
            if guild.id in self.audio:
                auds = self.audio[guild.id]
                create_future(auds.update)
        else:
            a = 1
            for g in tuple(self.audio):
                try:
                    auds = self.audio[g]
                    create_future(auds.update)
                    create_task(self.research(auds))
                except:
                    print(traceback.format_exc())
                if not a & 15:
                    await asyncio.sleep(0.2)
                a += 1
            # t = time.time()
            # i = 1
            # for path in os.listdir("cache"):
            #     fn = "cache/" + path
            #     if path.startswith("&"):
            #         if t - os.path.getmtime(fn) > 3600:
            #             os.remove(fn)
            #     if not i & 1023:
            #         await asyncio.sleep(0.2)
            #         t = time.time()
            #     i += 1
            await asyncio.sleep(0.5)
            for item in tuple(ytdl.cache.values()):
                item.update()
        self.busy = max(0, self.busy - 1)
