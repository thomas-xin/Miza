try:
    from common import *
except ModuleNotFoundError:
    import os
    os.chdir("..")
    from common import *

import youtube_dl, pytube
from bs4 import BeautifulSoup

getattr(youtube_dl, "__builtins__", {})["print"] = print

SAMPLE_RATE = 48000


f = open("auth.json")
auth = ast.literal_eval(f.read())
f.close()
try:
    genius_key = auth["genius_key"]
    if not genius_key:
        raise
except:
    genius_key = None
    print("WARNING: genius_key not found. Unable to use API to search song lyrics.")
try:
    google_api_key = auth["google_api_key"]
    if not google_api_key:
        raise
except:
    google_api_key = None
    print("WARNING: google_api_key not found. Unable to use API to search youtube playlists.")


async def createPlayer(auds, p_type=0, verbose=False):
    auds.stats.quiet |= 2 * p_type
    text = (
        "```" + "\n" * verbose + "callback-voice-player-" + str(int(bool(p_type)))
        + "\nInitializing virtual audio player...```"
    )
    await auds.channel.send(text)
    await auds.updatePlayer()


e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else 300)


def getDuration(filename):
    command = ["ffprobe", "-hide_banner", filename]
    resp = None
    for _ in loop(3):
        try:
            proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            fut = create_future_ex(proc.communicate)
            res = fut.result(timeout=2)
            resp = bytes().join(res)
            break
        except:
            try:
                proc.kill()
            except:
                pass
            print(traceback.format_exc())
    if not resp:
        return None
    s = resp.decode("utf-8", "replace")
    try:
        i = s.index("Duration: ")
        d = s[i + 10:]
        i = 2147483647
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


def getBestIcon(entry):
    try:
        return entry["thumbnail"]
    except KeyError:
        try:
            return entry["thumbnails"][0]["url"]
        except LookupError:
            try:
                url = entry["webpage_url"]
            except KeyError:
                url = entry["url"]
            if "discord" in url and "attachments/" in url:
                if not is_image(url):
                    return "https://cdn.discordapp.com/embed/avatars/0.png"
            return url


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
        raise LookupError("Unable to find voice channel.")
    return auds


def isAlone(auds, user):
    for m in auds.vc.channel.members:
        if m.id != user.id and not m.bot:
            return False
    return True


def ensure_url(url):
    if url.startswith("ytsearch:"):
        url = "https://www.youtube.com/results?search_query=" + verifyURL(url[9:])
    return url


class CustomAudio(discord.AudioSource):
    
    length = round(SAMPLE_RATE / 25)
    # empty = numpy.zeros(length >> 1, float)
    # emptybuff = numpy.zeros(length, numpy.uint16).tobytes()
    # filt = signal.butter(1, 0.125, btype="low", output="sos")
    # #fff = numpy.abs(numpy.fft.fftfreq(SAMPLE_RATE / 50, 1/SAMPLE_RATE))[:ceil(SAMPLE_RATE / 100 + 1)]
    # static = lambda self, *args: numpy.random.rand(self.length) * 65536 - 32768
    emptyopus = b"\xfc\xff\xfe"
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
        "repeat": False,
        "shuffle": False,
        "quiet": False,
        "stay": False,
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
            self.reading = 0
            self.has_read = False
            self.searching = False
            self.preparing = True
            self.player = None
            self.timeout = utc()
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
            "<" + classname + " object at " + hex(id(self)).upper().replace("X", "x") + ">: {"
            + "\"vc\": " + str(self.vc)
            + ", \"queue\": " + str(self.queue)
            + ", \"stats\": " + str(self.stats)
            + ", \"source\": " + str(self.source)
            + "}"
        )

    def ensure_play(self):
        try:
            self.vc.play(self, after=self.update)
        except discord.ClientException:
            pass
        except:
            print(traceback.format_exc())

    def stop(self):
        if getattr(self, "source", None) is None:
            return
        if not self.source.closed:
            create_future_ex(self.source.close)
        self.source = None

    def new(self, source=None, pos=0, update=True):
        self.speed = abs(self.stats.speed)
        self.is_playing = source is not None
        if source is not None:
            new_source = None
            try:
                self.stats.position = 0
                new_source = source.create_reader(pos, auds=self)
            except OverflowError:
                source = None
            else:
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
                self.stats.position = e_dur(self.queue[0].duration)
        if self.source is not None and self.player:
            self.player.time = 1 + utc()
        if self.speed < 0.005:
            self.speed = 1
            self.paused |= 2
        else:
            self.paused &= -3
        if update:
            self.update()
            self.queue.update_play()

    def seek(self, pos):
        duration = e_dur(self.queue[0].duration)
        pos = max(0, pos)
        if (pos >= duration and not self.reverse) or (pos <= 0 and self.reverse):
            create_future_ex(self.new, update=True)
            return duration
        create_future_ex(self.new, self.file, pos, update=False)
        self.stats.position = pos
        return self.stats.position

    announce = lambda self, *args, **kwargs: create_task(sendReact(self.channel, *args, reacts="❎", **kwargs))

    def kill(self, reason=""):
        self.dead = None
        g = self.vc.guild.id
        try:
            self.bot.database.playlists.audio.pop(g)
        except KeyError:
            pass
        try:
            self.bot.database.playlists.connecting.pop(g)
        except KeyError:
            pass
        try:
            if not reason:
                reason = (
                    "```css\n🎵 Successfully disconnected from ["
                    + noHighlight(self.vc.guild.name) + "]. 🎵```"
                )
            self.announce(reason)
        except LookupError:
            pass
        self.stop()

    def update(self, *void1, **void2):
        vc = self.vc
        guild = vc.guild
        if hasattr(self, "dead"):
            create_task(vc.disconnect())
            if self.dead is not None:
                self.kill()
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
                    vc.play(self, after=self.update)
                self.att = 0
            except:
                print(traceback.format_exc())
                if getattr(self, "att", 0) <= 0:
                    self.att = utc()
                elif utc() - self.att > 10:
                    self.dead = True
                    return
        if self.stats.stay:
            cnt = inf
        else:
            cnt = sum(1 for m in vc.channel.members if not m.bot)
        if not cnt:
            if self.timeout < utc() - 20:
                self.dead = True
                return
            elif self.timeout < utc() - 10:
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
                            self.announce(
                                "```ini\n🎵 Detected " + sbHighlight(cnt) + " user" + "s" * (cnt != 1)
                                + " in [#" + noHighlight(ch) + "], moving... 🎵```"
                            )
                            create_task(vc.move_to(ch))
        else:
            self.timeout = utc()
        if m.voice is not None:
            if m.voice.deaf or m.voice.mute or m.voice.afk:
                create_task(m.edit(mute=False, deafen=False))
        if not (vc.is_connected() or self.bot.database.playlists.is_connecting(vc.id)):
            create_task(self.reconnect())
        else:
            self.att = 0
            create_future_ex(self.queue.update_load)

    async def move_unmute(self, vc, channel):
        await vc.move_to(channel)
        await channel.guild.get_member(self.bot.client.user.id).edit(mute=False, deafen=False)

    async def reconnect(self):
        try:
            if hasattr(self, "dead") or self.vc.is_connected():
                return
            self.bot.database.playlists.connecting[self.vc.guild.id] = True
            if getattr(self, "att", 0) <= 0:
                self.att = utc()
            self.vc = await self.vc.channel.connect(timeout=30, reconnect=False)
            user = self.vc.guild.get_member(self.bot.client.user.id)
            if getattr(user, "voice", None) is not None:
                if user.voice.deaf or user.voice.mute or user.voice.afk:
                    create_task(user.edit(mute=False, deafen=False))
            self.att = 0
        except discord.Forbidden:
            print(traceback.format_exc())
            self.dead = True
        except discord.ClientException:
            self.att = utc()
        except:
            print(traceback.format_exc())
            if getattr(self, "att", 0) > 0 and utc() - self.att > 10:
                self.dead = True
        try:
            self.bot.database.playlists.connecting.pop(self.vc.guild.id)
        except:
            pass

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
            if utc() > curr.time:
                curr.time = inf
                try:
                    await self.bot.reactCallback(curr.message, "❎", self.bot.client.user)
                except discord.NotFound:
                    self.player = None
                    print(traceback.format_exc())
        q = self.stats.quiet
        if q == bool(q):
            self.stats.quiet = bool(q)

    def construct_options(self, full=True):
        stats = self.stats
        pitchscale = 2 ** ((stats.pitch + stats.resample) / 12)
        chorus = min(16, abs(stats.chorus))
        reverb = stats.reverb
        if reverb:
            args = ["-i", "misc/SNB3,0all.wav"]
        else:
            args = []
        options = []
        if self.reverse:
            options.append("areverse")
        if pitchscale != 1 or stats.speed != 1:
            speed = abs(stats.speed) / pitchscale
            speed *= 2 ** (stats.resample / 12)
            if round(speed, 9) != 1:
                speed = max(0.005, speed)
                if speed >= 64:
                    raise OverflowError
                opts = ""
                while speed > 3:
                    opts += "atempo=3,"
                    speed /= 3
                while speed < 0.5:
                    opts += "atempo=0.5,"
                    speed /= 0.5
                opts += "atempo=" + str(speed)
                options.append(opts)
        if pitchscale != 1:
            if abs(pitchscale) >= 64:
                raise OverflowError
            if full:
                options.append("aresample=" + str(SAMPLE_RATE))
            options.append("asetrate=" + str(SAMPLE_RATE * pitchscale))
        if chorus:
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
                depth = (i * 0.43 * neg) % max(4, stats.chorus) + 0.5
                D += str(round(depth, 3))
            b = 0.5 / sqrt(ceil(chorus + 1))
            options.append(
                "\"chorus=0.5:" + str(round(b, 3)) + ":"
                + A + ":"
                + B + ":"
                + C + ":"
                + D + "\""
            )
            options.append("volume=2")
        if stats.compressor:
            comp = min(8000, abs(stats.compressor + sgn(stats.compressor)))
            while abs(comp) > 1:
                c = min(20, comp)
                try:
                    comp /= c
                except ZeroDivisionError:
                    comp = 1
                mult = str(round(math.sqrt(c), 4))
                options.append(
                    "acompressor=mode=" + ("upward" if stats.compressor < 0 else "downward")
                    + ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
                )
        if stats.bassboost:
            opt = "anequalizer="
            width = 4096
            x = round(sqrt(1 + abs(stats.bassboost)), 5)
            coeff = width * max(0.03125, (0.25 / x))
            ch = " f=" + str(coeff if stats.bassboost > 0 else width - coeff) + " w=" + str(coeff / 2) + " g=" + str(max(0.5, min(48, 2 * math.log2(x * 5))))
            opt += "c0" + ch + "|c1" + ch
            options.append(opt)
        if reverb:
            coeff = abs(reverb)
            wet = min(2, coeff) / 2
            if wet != 1:
                options.append("asplit[2]")
            options.append("volume=1.2")
            options.append("afir=dry=10:wet=10")
            if wet != 1:
                dry = 1 - wet
                options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(wet, 6)))
            if coeff > 1:
                decay = str(round(1 - 4 / (3 + coeff), 4))
                options.append("aecho=1:1:479|613:" + decay + "|" + decay)
        if stats.pan != 1:
            pan = min(10000, max(-10000, stats.pan))
            while abs(abs(pan) - 1) > 0.001:
                p = max(-10, min(10, pan))
                try:
                    pan /= p
                except ZeroDivisionError:
                    pan = 1
                options.append("extrastereo=m=" + str(p) + ":c=0")
                v = 1 / max(1, round(math.sqrt(abs(p)), 4))
                if v != 1:
                    options.append("volume=" + str(v))
        if stats.volume != 1:# and full:
            options.append("volume=" + str(round(stats.volume, 7)))
        if options:
            options.append("asoftclip=atan")
            args.append(("-af", "-filter_complex")[bool(reverb)])
            args.append(",".join(options))
        return args

    def read(self):
        try:
            found = empty = False
            if self.queue.loading or self.paused:
                self.is_playing = True
                raise EOFError
            try:
                source = self.source
                if source is None:
                    raise StopIteration
                temp = source.read()
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
                        if self.source is not None:
                            self.source.advanced = True
                        create_future_ex(self.queue.advance)
                elif empty and queueable and self.source is not None:
                    if utc() - self.lastEnd > 0.5:
                        if self.reverse:
                            ended = self.stats.position <= 0.5
                        else:
                            ended = ceil(self.stats.position) >= e_dur(self.queue[0].duration) - 0.5
                        if self.curr_timeout and utc() - self.curr_timeout > 0.5 or ended:
                            if not found:
                                self.lastEnd = utc()
                                if not self.has_read or not self.queue:
                                    if self.queue:
                                        self.queue[0].url = ""
                                    self.source.advanced = True
                                    create_future_ex(self.queue.update_play)
                                    self.preparing = False
                                else:
                                    self.source.advanced = True
                                    create_future_ex(self.queue.update_play)
                                    self.preparing = False
                        elif self.curr_timeout == 0:
                            self.curr_timeout = utc()
                elif (empty and not queueable) or self.pausec:
                    self.curr_timeout = 0
                    self.vc.stop()
            temp = self.emptyopus
            self.pausec = self.paused & 1
        else:
            self.pausec = False
        return temp

    is_opus = lambda self: True
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
        self.loading = False

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
                if i < 2:
                    if not e.get("stream", None):
                        if not i:
                            callback = self.update_play
                        else:
                            callback = None
                        create_future_ex(ytdl.getStream, e, callback=callback)
                        break
                if "file" in e:
                    e["file"].ensure_time()
                if not e.url:
                    if not self.auds.stats.quiet:
                        self.auds.announce(
                            "```ini\nA problem occurred while loading " + sbHighlight(e.name)
                            + ", and it has been removed from the queue as a result.```"
                        )
                    dels.append(i)
                    continue
            if len(dels) > 2:
                q.pops(dels)
            elif dels:
                while dels:
                    q.pop(dels.popleft())
            self.advance(process=False)
        create_task(self.auds.updatePlayer())

    def advance(self, looped=True, repeated=True, shuffled=True, process=True):
        q = self
        s = self.auds.stats
        if q and process:
            if q[0].get("played"):
                self.prev = q[0]["url"]
                try:
                    q[0].pop("played")
                except (KeyError, IndexError):
                    pass
                if not (s.repeat and repeated):
                    if s.loop:
                        temp = q[0]
                    q.popleft()
                    if s.shuffle and shuffled:
                        if len(q) > 1:
                            temp = q.popleft()
                            shuffle(q)
                            q.appendleft(temp)
                    if s.loop and looped:
                        q.append(temp)
                if self.auds.player:
                    self.auds.player.time = 1 + utc()
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
                    self.auds.player.time = 1 + utc()
        self.update_play()
        
    def update_play(self):
        auds = self.auds
        q = self
        if q:
            if (auds.source is None or auds.source.closed or auds.source.advanced) and not q[0].get("played", False):
                if q[0].get("stream", None) not in (None, "none"):
                    q[0].played = True
                    if not auds.stats.quiet:
                        if utc() - self.lastsent > 1:
                            try:
                                u = self.bot.cache.users[q[0].u_id]
                                name = u.display_name
                            except KeyError:
                                name = "Deleted User"
                            auds.announce(
                                "```ini\n🎵 Now playing "
                                + sbHighlight(q[0].name)
                                + ", added by " + sbHighlight(name) + "! 🎵```"
                            )
                            self.lastsent = utc()
                    self.loading = True
                    source = ytdl.getStream(q[0])
                    try:
                        auds.new(source)
                        self.loading = False
                        create_future_ex(auds.ensure_play)
                    except:
                        self.loading = False
                        print(traceback.format_exc())
                        raise
                    auds.preparing = False
            elif auds.source is None and not self.loading and not auds.preparing:
                self.advance()
            elif auds.source is not None and auds.source.advanced:
                auds.source.advanced = False
                auds.source.closed = True
                self.advance()
        else:
            if auds.source is None or auds.source.closed or auds.source.advanced:
                auds.vc.stop()
                auds.source = None

    def enqueue(self, items, position):
        if not self:
            self.__init__(items)
            self.auds.source = None
            create_future_ex(self.advance, process=False)
            return self
        if position == -1:
            self.extend(items)
        else:
            self.rotate(-position)
            self.extend(items)
            self.rotate(len(items) + position)
        return self


def org2xm(org, dat=None):
    if not org or type(org) is not bytes:
        if not isURL(org):
            raise TypeError("Invalid input URL.")
        org = verifyURL(org)
        data = None
        resp = None
        try:
            resp = requests.get(org, timeout=8, stream=True)
            it = resp.iter_content(4096)
            b = bytes()
            while len(b) < 4:
                b += next(it)
            if not b.startswith(b"Org-"):
                raise ValueError("Invalid file header.")
            data = b + resp.content
            resp.close()
        except:
            if resp is not None:
                try:
                    resp.close()
                except:
                    pass
            raise
        if not data:
            raise FileNotFoundError("Error downloading file content.")
    else:
        if not org.startswith(b"Org-"):
            raise ValueError("Invalid file header.")
        data = org
    compat = not data.startswith(b"Org-02")
    ts = round(utc() * 1000)
    r_org = "cache/" + str(ts) + ".org"
    f = open(r_org, "wb")
    f.write(data)
    f.close()
    r_dat = "cache/" + str(ts) + ".dat"
    orig = False
    if dat is not None and isURL(dat):
        dat = verifyURL(dat)
        f = open(r_dat, "wb")
        dat = Request(dat)
        f.write(dat)
        f.close()
    else:
        if type(dat) is bytes and dat:
            f = open(r_dat, "wb")
            f.write(dat)
            f.close()
        else:
            r_dat = "misc/ORG210EN.DAT"
            orig = True
    args = ["misc/org2xm.exe", r_org, r_dat]
    if compat:
        args.append("c")
    subprocess.check_output(args)
    r_xm = "cache/" + str(ts) + ".xm"
    if str(ts) + ".xm" not in os.listdir("cache"):
        raise FileNotFoundError("Unable to locate converted file.")
    if not os.path.getsize(r_xm):
        raise RuntimeError("Converted file is empty.")
    for f in (r_org, r_dat)[:2 - orig]:
        try:
            os.remove(f)
        except (PermissionError, FileNotFoundError):
            pass
    return r_xm


class PCMFile:
    
    def __init__(self, fn):
        self.file = fn
        self.proc = None
        self.loading = False
        self.expired = False
        self.buffered = False
        self.loaded = False
        self.readers = {}
        self.assign = deque()
        self.ensure_time()

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return "<" + classname + " object " + self.file + " at " + hex(id(self)).upper().replace("X", "x") + ">"
    
    def load(self, stream, check_fmt=False, force=False):
        if self.loading and not force:
            return
        self.stream = stream
        self.loading = True
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-bufsize", "65536", "cache/" + self.file]
        self.proc = None
        try:
            self.proc = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            fl = 0
            while fl < 65536:
                if not self.proc.is_running():
                    if check_fmt:
                        try:
                            xm = org2xm(stream)
                        except ValueError:
                            pass
                        else:
                            return self.load(xm, check_fmt=False, force=True)
                    err = self.proc.stderr.read().decode("utf-8", "replace")
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
            self.buffered = True
            self.ensure_time()
            print(self.file, "buffered", fl)
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
        if ytdl.bot is not None and "videoplayback" in stream:
            try:
                i = stream.index("&ip=") + 4
            except ValueError:
                pass
            else:
                ip = stream[i:].split("&")[0]
                ytdl.bot.updateIP(ip)
        return self

    ensure_time = lambda self: setattr(self, "time", utc())

    def update(self):
        if self.loaded:
            dur = self.duration()
            if dur is not None:
                for e in self.assign:
                    e["duration"] = dur
                self.assign.clear()
        elif self.buffered and not self.proc.is_running():
            if not self.loaded:
                self.loaded = True
                if not isURL(self.stream):
                    for _ in loop(3):
                        try:
                            os.remove(self.stream)
                            break
                        except (PermissionError, FileNotFoundError):
                            time.sleep(0.5)
                try:
                    fl = os.path.getsize("cache/" + self.file)
                except FileNotFoundError:
                    fl = 0
                print(self.file, "loaded", fl)
        if self.readers:
            self.ensure_time()
            return
        if utc() - self.time > 1800:
            try:
                fl = os.path.getsize("cache/" + self.file)
            except FileNotFoundError:
                fl = 0
                if self.buffered:
                    self.time = -inf
            ft = 9000 / (math.log2(fl / 134217728 + 1) + 1)
            if utc() - self.time > ft:
                self.destroy()

    def open(self):
        self.ensure_time()
        if self.proc is None:
            raise ProcessLookupError
        return open("cache/" + self.file, "rb")

    def destroy(self):
        self.expired = True
        try:
            self.proc.kill()
        except:
            pass
        for _ in loop(8):
            try:
                os.remove("cache/" + self.file)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                self.ensure_time()
                return
            except:
                print(traceback.format_exc())
                time.sleep(5)
        try:
            ytdl.cache.pop(self.file)
        except KeyError:
            pass
        print(self.file, "deleted.")

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
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2"]
        if pos:
            arg = "-to" if auds.reverse else "-ss"
            args += [arg, str(pos)]
        args.append("-i")
        if self.loaded:
            buff = False
            args.insert(1, "-nostdin")
            args.append("cache/" + self.file)
        else:
            buff = True
            args.append("pipe:0")
        options = auds.construct_options(full=False)
        args.extend(options)
        args.extend(("-map_metadata", "-1", "-f", "opus", "-c:a", "libopus", "-ar", str(SAMPLE_RATE), "-ac", "2", "-b:a", "196608", "-bufsize", "8192", "pipe:1"))
        key = auds.vc.guild.id
        self.readers[key] = True
        callback = lambda: self.readers.pop(key) if key in self.readers else None
        if buff:
            player = BufferedAudioReader(self, args, callback=callback)
        else:
            player = LoadedAudioReader(self, args, callback=callback)
        return player.start()
        create_future_ex(player.run)
        return player

    duration = lambda self: self.dur if getattr(self, "dur", None) is not None else setDict(self.__dict__, "dur", os.path.getsize("cache/" + self.file) / 48000 / 4 if self.loaded else getDuration(self.stream), ignore=True)


class LoadedAudioReader(discord.AudioSource):

    def __init__(self, file, args, callback=None):
        self.closed = False
        self.advanced = False
        self.proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        self._packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
        self.file = file
        self.buffer = None
        self.callback = callback
    
    def read(self):
        if self.buffer:
            b, self.buffer = self.buffer, None
            return b
        return next(self._packet_iter)
    
    def start(self):
        self.buffer = None
        self.buffer = self.read()
        return self

    def close(self, *void1, **void2):
        self.closed = True
        try:
            self.proc.kill()
        except:
            pass
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close


class BufferedAudioReader(discord.AudioSource):

    def __init__(self, file, args, callback=None):
        self.closed = False
        self.advanced = False
        self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
        self.file = file
        self.stream = file.open()
        self.buffer = None
        self.callback = callback
        self.full = False

    def read(self):
        if self.buffer:
            b, self.buffer = self.buffer, None
            return b
        return next(self._packet_iter)

    def run(self):
        while True:
            b = bytes()
            try:
                b = self.stream.read(65536)
                if not b:
                    raise EOFError
                self.proc.stdin.write(b)
                self.proc.stdin.flush()
            except (ValueError, EOFError):
                if self.file.loaded:
                    break
                time.sleep(0.1)
        self.full = True
        self.proc.stdin.close()
    
    def start(self):
        create_future_ex(self.run)
        self.buffer = None
        self.buffer = self.read()
        return self

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
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close

    
class AudioDownloader:
    
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
        self.bot = None
        self.lastclear = 0
        self.downloading = {}
        self.cache = {}
        self.searched = {}
        self.requests = 0
        self.update_dl()
        self.setup_pages()

    def setup_pages(self):
        resp = Request("https://raw.githubusercontent.com/Quihico/handy.stuff/master/yt.pagetokens.x10", timeout=64, decode=True)
        page10 = resp.split("\n")
        self.yt_pages = [page10[i] for i in range(0, len(page10), 5)]

    def update_dl(self):
        if utc() - self.lastclear > 720:
            self.lastclear = utc()
            self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
            self.spotify_headers = deque({"authorization": "Bearer " + json.loads(Request("https://open.spotify.com/get_access_token")[:512])["accessToken"]} for _ in loop(8))

    def from_pytube(self, url):
        url = verifyURL(url)
        if not url.startswith("https://www.youtube.com/"):
            if not url.startswith("http://youtu.be/"):
                if isURL(url):
                    raise youtube_dl.DownloadError("Not a youtube link.")
                url = "https://www.youtube.com/watch?v=" + url
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
            "thumbnail": getattr(resp, "thumbnail_url", None),
        }
        for i in range(len(entry["formats"])):
            stream = resp.streams.fmt_streams[i]
            try:
                abr = stream.abr.lower()
            except AttributeError:
                abr = "0"
            if type(abr) is not str:
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

    def get_spotify_part(self, url):
        out = deque()
        self.spotify_headers.rotate()
        resp = Request(url, headers=self.spotify_headers[0])
        try:
            d = json.loads(resp)
        except:
            d = eval(resp, {}, eval_const)
        try:
            items = d["items"]
            total = d.get("total", 0)
        except KeyError:
            if "type" in d:
                items = (d,)
                total = 1
            else:
                items = []
        for item in items:
            try:
                track = item["track"]
            except KeyError:
                try:
                    track = item["episode"]
                except KeyError:
                    if "id" in item:
                        track = item
                    else:
                        continue
            name = track.get("name", track["id"])
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            dur = track.get("duration_ms")
            if dur:
                dur /= 1000
            temp = freeClass(
                name=name,
                url="ytsearch:" + (name + " ~ " + artists).replace(":", "-"),
                duration=dur,
                research=True,
            )
            out.append(temp)
        return out, total

    def get_youtube_part(self, url):
        out = deque()
        resp = Request(url)
        try:
            d = json.loads(resp)
        except:
            d = eval(resp, {}, eval_const)
        try:
            items = d["items"]
            total = d.get("pageInfo", {}).get("totalResults", 0)
        except KeyError:
            raise
        for item in items:
            try:
                snip = item["snippet"]
                v_id = snip["resourceId"]["videoId"]
            except KeyError:
                continue
            name = snip.get("title", v_id)
            url = "https://www.youtube.com/watch?v=" + v_id
            temp = freeClass(
                name=name,
                url=url,
                duration=None,
                research=True,
            )
            out.append(temp)
        return out, total

    spotifyFind = re.compile("(play|open|api)\\.spotify\\.com")

    def extract_true(self, url):
        while not isURL(url):
            resp = self.extract_from(url)
            if "entries" in resp:
                resp = resp["entries"][0]
            if "duration" in resp and "formats" in resp:
                return resp
            try:
                url = resp["webpage_url"]
            except KeyError:
                try:
                    url = resp["url"]
                except KeyError:
                    url = resp["id"]
        try:
            return self.downloader.extract_info(url, download=False, process=True)
        except youtube_dl.DownloadError as ex:
            if "429" in str(ex):
                try:
                    return self.from_pytube(url)
                except youtube_dl.DownloadError:
                    raise FileNotFoundError("Unable to fetch audio data.")
            raise
    
    def extract_from(self, url):
        try:
            return self.downloader.extract_info(url, download=False, process=False)
        except youtube_dl.DownloadError as ex:
            if "429" in str(ex):
                if isURL(url):
                    try:
                        return self.from_pytube(url)
                    except youtube_dl.DownloadError:
                        raise FileNotFoundError("Unable to fetch audio data.")
            raise

    def extract_info(self, item, count=1, search=False):
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
            return self.extract_from(item)
        return self.downloader.extract_info(item, download=False, process=False)

    def extract(self, item, force=False, count=1, search=True):
        try:
            page = None
            output = deque()
            if google_api_key and ("youtube.com" in item or "youtu.be/" in item):
                p_id = None
                for x in ("?list=", "&list="):
                    if x in item:
                        p_id = item[item.index(x) + len(x):]
                        p_id = p_id.split("&")[0]
                        break
                if p_id:
                    url = "https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&key=" + google_api_key + "&playlistId=" + p_id
                    page = 50
                if page:
                    futs = deque()
                    maxitems = 5000
                    for i, curr in enumerate(range(0, maxitems, page)):
                        if curr >= maxitems:
                            break
                        search = url + "&pageToken=" + self.yt_pages[i]
                        fut = create_future_ex(self.get_youtube_part, search)
                        futs.append(fut)
                        if not math.log2(i + 4) % 1 or not 4 + i & 15:
                            while futs:
                                fut = futs.popleft()
                                res = fut.result()
                                if not i:
                                    maxitems = res[1] + page
                                if not res[0]:
                                    maxitems = 0
                                    futs.clear()
                                    break
                                output += res[0]
                        else:
                            time.sleep(0.03125)
                    while futs:
                        output += futs.popleft().result()[0]
            if re.search(self.spotifyFind, item):
                if "playlist" in item:
                    url = item[item.index("playlist"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/")[0]
                    url = "https://api.spotify.com/v1/playlists/" + str(key) + "/tracks?type=track,episode"
                    page = 100
                elif "album" in item:
                    url = item[item.index("album"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/")[0]
                    url = "https://api.spotify.com/v1/albums/" + str(key) + "/tracks?type=track,episode"
                    page = 50
                elif "track" in item:
                    url = item[item.index("track"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/")[0]
                    url = "https://api.spotify.com/v1/tracks/" + str(key)
                    page = 1
                else:
                    raise TypeError("Unsupported Spotify URL.")
                if page == 1:
                    output += self.get_spotify_part(url)
                else:
                    futs = deque()
                    maxitems = 10000
                    for i, curr in enumerate(range(0, maxitems, page)):
                        if curr >= maxitems:
                            break
                        search = url + "&offset=" + str(curr) + "&limit=" + str(page)
                        fut = create_future_ex(self.get_spotify_part, search)
                        futs.append(fut)
                        if not math.log2(i + 1) % 1 or not i & 7:
                            while futs:
                                fut = futs.popleft()
                                res = fut.result()
                                if not i:
                                    maxitems = res[1] + page
                                if not res[0]:
                                    maxitems = 0
                                    futs.clear()
                                    break
                                output += res[0]
                        else:
                            time.sleep(0.125)
                    while futs:
                        output += futs.popleft().result()[0]
            if not len(output):
                if isURL(item):
                    url = verifyURL(item)
                    if url.endswith(".json") or url.endswith(".txt"):
                        s = Request(url)
                        if len(s) > 8388608:
                            raise OverflowError("Playlist entity data too large.")
                        s = s[s.index(b"{"):s.rindex(b"}") + 1]
                        d = json.loads(s)
                        q = d["queue"]
                        return [freeClass(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
                resp = self.extract_info(item, count, search=search)
                if resp.get("_type", None) == "url":
                    resp = self.extract_from(resp["url"])
                if resp is None or not len(resp):
                    raise EOFError("No search results found.")
                if resp.get("_type", None) == "playlist":
                    entries = list(resp["entries"])
                    if force or len(entries) <= 1:
                        for entry in entries:
                            data = self.extract_from(entry["url"])
                            temp = {
                                "name": data["title"],
                                "url": data["webpage_url"],
                                "duration": float(data["duration"]),
                                "stream": getBestAudio(resp),
                                "icon": getBestIcon(resp),
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
                        "icon": getBestIcon(resp),
                    }
                    # if dur is None:
                    #     temp["duration"] = getDuration(temp["stream"])
                    output.append(freeClass(temp))
            return output
        except:
            if force != "spotify":
                raise
            print(traceback.format_exc())
            return 0

    def search(self, item, force=False):
        item = verifySearch(item)
        while self.requests > 4:
            time.sleep(0.1)
        if item in self.searched:
            if utc() - self.searched[item].t < 18000:
                # self.searched[item].t = utc()
                return self.searched[item].data
            else:
                self.searched.pop(item)
        while len(self.searched) > 262144:
            self.searched.pop(next(iter(self.searched)))
        try:
            self.requests += 1
            obj = freeClass(t=utc())
            obj.data = output = self.extract(item, force)
            self.searched[item] = obj
            self.requests = max(self.requests - 1, 0)
            return output
        except Exception as ex:
            print(traceback.format_exc())
            self.requests = max(self.requests - 1, 0)
            return repr(ex)
        
    def getStream(self, entry, force=False, download=True, callback=None):
        stream = entry.get("stream", None)
        icon = entry.get("icon", None)
        if stream == "none" and not force:
            return None
        entry["stream"] = "none"
        if "research" in entry:
            try:
                self.extractSingle(entry)
                try:
                    entry.pop("research")
                except KeyError:
                    pass
            except:
                print(traceback.format_exc())
                try:
                    entry.pop("research")
                except KeyError:
                    pass
                raise
        if stream in (None, "none") or stream.startswith("https://cf-hls-media.sndcdn.com/"):
            data = self.extract(entry["url"], search=False)
            stream = setDict(data[0], "stream", data[0].url)
            icon = setDict(data[0], "icon", data[0].url)
        h = shash(entry["url"])
        fn = h + ".pcm"
        if fn in self.cache or not download:
            entry["stream"] = stream
            entry["icon"] = icon
            if callback is not None:
                create_future_ex(callback)
            f = self.cache.get(fn, None)
            if f is not None:
                entry["file"] = f
                if f.loaded:
                    entry["duration"] = f.duration()
                else:
                    f.assign.append(entry)
                f.ensure_time()
            return f
        try:
            self.cache[fn] = f = PCMFile(fn)
            f.load(stream, check_fmt=entry.get("duration") is None)
            dur = entry.get("duration", None)
            f.assign.append(entry)
            entry["stream"] = stream
            entry["icon"] = icon
            entry["file"] = f
            f.ensure_time()
            if callback is not None:
                create_future_ex(callback)
            return f
        except:
            print(traceback.format_exc())
            entry["url"] = ""
    
    def download_file(self, url, fmt="ogg", auds=None, fl=8388608):
        fn = "cache/&" + str(time_snowflake(utc_dt())) + "." + fmt
        info = self.extract(url)[0]
        self.getStream(info, force=True, download=False)
        stream = info["stream"]
        if not stream:
            raise LookupError("No stream URLs found for " + url)
        duration = getDuration(stream)
        if type(duration) not in (int, float):
            dur = 960
        else:
            dur = duration
        fs = fl - 131072
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-vn", "-i", stream]
        if auds is not None:
            args += auds.construct_options(full=True)
            dur /= auds.stats.speed / 2 ** (auds.stats.resample / 12)
        if dur > 960:
            dur = 960
        br = max(32, min(256, floor(((fs - 131072) / dur / 128) / 4) * 4)) * 1024
        args += ["-ar", "48000", "-b:a", str(br), "-fs", str(fs), fn]
        try:
            subprocess.check_output(args)
        except subprocess.CalledProcessError:
            try:
                xm = org2xm(stream)
            except ValueError:
                pass
            else:
                args[8] = xm
                subprocess.check_output(args)
                try:
                    os.remove(xm)
                except (PermissionError, FileNotFoundError):
                    pass
                return fn, info["name"] + "." + fmt
            raise
        return fn, info["name"] + "." + fmt

    def extractSingle(self, i):
        item = i.url
        while self.requests > 4:
            time.sleep(0.1)
        if item in self.searched:
            if utc() - self.searched[item].t < 18000:
                # self.searched[item].t = utc()
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
        try:
            self.requests += 1
            data = self.extract_true(item)
            if "entries" in data:
                data = data["entries"][0]
            obj = freeClass(t=utc())
            obj.data = out = [freeClass(
                name=data["title"],
                url=data["webpage_url"],
                stream=getBestAudio(data),
                icon=getBestIcon(data),
            )]
            try:
                out[0].duration = data["duration"]
            except KeyError:
                out[0].research = True
            self.searched[item] = obj
            it = out[0]
            i.name = it.name
            i.duration = it.get("duration")
            i.url = it.url
            # sethash(i)
            self.requests = max(self.requests - 1, 0)
        except:
            self.requests = max(self.requests - 1, 0)
            i.url = ""
            print(traceback.format_exc())
        return True

ytdl = AudioDownloader()


class Queue(Command):
    server_only = True
    name = ["Q", "Play", "Enqueue", "P"]
    alias = name + ["LS"]
    min_level = 0
    description = "Shows the music queue, or plays a song in voice."
    usage = "<search_link[]> <verbose(?v)> <hide(?h)> <force(?f)> <budge(?b)> <debug(?z)>"
    flags = "hvfbz"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    _timeout_ = 2
    rate_limit = (0.5, 1.5)

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
                create_future_ex(auds.queue.update_play)
                create_future_ex(auds.ensure_play)
                return "```css\nSuccessfully resumed audio playback in [" + noHighlight(guild.name) + "].```", 1
            if not len(q):
                auds.preparing = False
                create_future_ex(auds.update)
            return (
                "```" + "\n" * ("z" in flags) + "callback-voice-queue-"
                + str(user.id) + "_0_" + str(int(v))
                + "-\nLoading Queue...```"
            )
        try:
            auds = bot.database.playlists.audio[guild.id]
            future = None
        except KeyError:
            future = wrap_future(create_task(forceJoin(guild, channel, user, client, bot, preparing=True)))
        future2 = create_task(channel.trigger_typing())
        if isURL(argv):
            argv = await bot.followURL(argv)
        resp = await create_future(ytdl.search, argv)
        if future is not None:
            auds = await future
        await future2
        if "f" in flags or "b" in flags:
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to force play while other users are in voice")
        if auds.stats.quiet & 2:
            setDict(flags, "h", 1)
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
                "duration": e.get("duration"),
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
                total_duration += e_dur(e.duration)
            if auds.reverse and len(auds.queue):
                total_duration += elapsed - e_dur(q[0].duration)
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
            create_future_ex(auds.new)
            total_duration = tdur
        elif "b" in flags:
            auds.queue.enqueue(added, 1)
            total_duration = max(3, e_dur(q[0].duration) - elapsed if q else 0)
        else:
            auds.queue.enqueue(added, -1)
            total_duration = max(total_duration / auds.speed, tdur)
        if not names:
            raise LookupError("No results for " + str(argv) + ".")
        if "v" in flags:
            names = noHighlight(hlist(i.name + ": " + dhms(e_dur(i.duration)) for i in added))
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
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        user = await bot.fetch_user(u_id)
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
            elif i == 3:
                new = last
            else:
                new = pos
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
        if not q:
            totalTime = 0
        elif auds.stats.loop or auds.stats.repeat:
            totalTime = inf
        else:
            if auds.reverse and q:
                totalTime = elapsed - e_dur(q[0].duration)
            else:
                totalTime = -elapsed
            i = 0
            for e in q:
                totalTime += e_dur(e.duration)
                if i < pos:
                    startTime += e_dur(e.duration)
                if not 1 + i & 4095:
                    await asyncio.sleep(0.2)
                i += 1
        cnt = len(q)
        info = (
            str(cnt) + " item" + "s" * (cnt != 1) + ", estimated total duration: "
            + sec2Time(totalTime / auds.speed) + "```"
        )
        if not q:
            duration = 0
        else:
            duration = e_dur(q[0].duration)
        sym = "⬜⬛"
        barsize = 24
        if not elapsed or not duration:
            r = 0
        else:
            r = round(min(1, elapsed / duration) * barsize)
        bar = sym[0] * r + sym[1] * (barsize - r)
        if not q:
            countstr = "Queue is currently empty.\n"
        else:
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
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        if q:
            icon = q[0].get("icon", "")
        else:
            icon = ""
        emb.set_thumbnail(url=icon)
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
            curr += dhms(e_dur(e.duration)) + ")`"
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
                estim = currTime + elapsed - e_dur(auds.queue[0].duration)
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
                    curr += "Remaining time: [" + sec2Time((estim + e_dur(e.duration)) / auds.speed) + "]"
                curr += "```"
            curr += "\n"
            if len(embstr) + len(curr) > 2048 - len(emb.description):
                break
            embstr += curr
            if i <= 1 or not auds.stats.shuffle:
                currTime += e_dur(e.duration)
            if not 1 + 1 & 4095:
                await asyncio.sleep(0.3)
            i += 1
        emb.description += embstr
        more = len(q) - i
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)


class Playlist(Command):
    server_only = True
    name = ["DefaultPlaylist", "PL"]
    min_level = 0
    min_display = "0~2"
    description = "Shows, appends, or removes from the default playlist."
    usage = "<search_link[]> <remove(?d)> <debug(?z)>"
    flags = "aedzf"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 0.5

    async def __call__(self, user, argv, guild, flags, channel, perm, **void):
        update = self.bot.database.playlists.update
        bot = self.bot
        if argv or "d" in flags:
            req = 2
            if perm < req:
                reason = (
                    "to modify default playlist for "
                    + guild.name
                )
                raise self.permError(perm, req, reason)
        pl = setDict(bot.data.playlists, guild.id, [])
        if not argv:
            if "d" in flags:
                if "f" not in flags:
                    response = uniStr(
                        "WARNING: POTENTIALLY DANGEROUS COMMAND ENTERED. "
                        + "REPEAT COMMAND WITH \"?F\" FLAG TO CONFIRM."
                    )
                    return ("```asciidoc\n[" + response + "]```")
                pl[guild.id].clear()
                pl.pop(guild.id)
                update()
                return (
                    "```css\nRemoved all entries from the default playlist for "
                    + sbHighlight(guild) + ".```"
                )
            return (
                "```" + "\n" * ("z" in flags) + "callback-voice-playlist-"
                + str(user.id) + "_0"
                + "-\nLoading Playlist database...```"
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
        lim = 8 << self.bot.isTrusted(guild.id) * 2 + 1
        if len(pl) >= lim:
            raise OverflowError(
                "Playlist size for " + guild.name
                + " has reached the maximum of " + str(lim) + " items. "
                + "Please remove an item to add another."
            )
        if isURL(argv):
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
    
    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_")]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        guild = message.guild
        user = await bot.fetch_user(u_id)
        pl = bot.data.playlists.get(guild.id, [])
        page = 12
        last = max(0, len(pl) - page)
        if reaction is not None:
            i = self.directions.index(reaction)
            if i == 0:
                new = 0
            elif i == 1:
                new = max(0, pos - page)
            elif i == 2:
                new = min(last, pos + page)
            elif i == 3:
                new = last
            else:
                new = pos
            pos = new
        content = message.content
        if not content:
            content = message.embeds[0].description
        i = content.index("callback")
        content = content[:i] + (
            "callback-voice-playlist-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not pl:
            content += "No currently enabled default playlist for " + str(guild).replace("`", "") + ".```"
            msg = ""
        else:
            content += str(len(pl)) + " items in default playlist for " + str(guild).replace("`", "") + ":```"
            key = lambda x: limStr(sbHighlight(x["name"]) + "(" + x["url"] + ")", 1900 / page)
            msg = strIter(pl[pos:pos + page], key=key, offset=pos, left="`【", right="】`")
        emb = discord.Embed(
            description=content + msg,
            colour=randColour(),
        )
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        more = len(pl) - pos - page
        if more > 0:
            emb.set_footer(
                text=uniStr("And ", 1) + str(more) + uniStr(" more...", 1),
            )
        create_task(message.edit(content=None, embed=emb))
        if reaction is None:
            for react in self.directions:
                create_task(message.add_reaction(react.decode("utf-8")))
                await asyncio.sleep(0.5)
        

class Connect(Command):
    server_only = True
    name = ["Summon", "Join", "DC", "Disconnect", "FuckOff", "Move", "Reconnect"]
    min_level = 0
    description = "Summons the bot into a voice channel."
    usage = "<channel{curr}(0)>"
    rate_limit = 3

    async def __call__(self, user, channel, name="join", argv="", **void):
        bot = self.bot
        client = bot.client
        if name in ("dc", "disconnect", "fuckoff"):
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
            raise self.permError(perm, 0, "for command " + self.name + " in " + str(guild))
        if vc_ is None:
            try:
                auds = bot.database.playlists.audio[guild.id]
            except KeyError:
                raise LookupError("Unable to find voice channel.")
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to disconnect while other users are in voice")
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
                    connecting[guild.id] = utc()
                    try:
                        await vc.move_to(vc_)
                        connecting[guild.id] = 0
                    except:
                        connecting[guild.id] = 0
                        raise
                break
        if not joined:
            connecting[guild.id] = utc()
            vc = freeClass(is_connected = lambda: False)
            t = utc()
            while not vc.is_connected() and utc() - t < 8:
                try:
                    vc = await vc_.connect(timeout=30, reconnect=False)
                    for _ in loop(12):
                        if vc.is_connected():
                            break
                        await asyncio.sleep(1 / 3)
                except discord.ClientException:
                    print(traceback.format_exc())
                    await asyncio.sleep(1)
            if isinstance(vc, freeClass):
                connecting[guild.id] = 0
                raise ConnectionError("Unable to connect to voice channel.")
        if guild.id not in bot.database.playlists.audio:
            bot.database.playlists.audio[guild.id] = auds = CustomAudio(channel, vc, bot)
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
            create_task(bot.database.playlists(guild=guild))
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
    rate_limit = (0.5, 1.5)

    async def __call__(self, client, user, perm, bot, name, args, argv, guild, flags, message, **void):
        if guild.id not in bot.database.playlists.audio:
            raise LookupError("Currently not playing in a voice channel.")
        auds = bot.database.playlists.audio[guild.id]
        if name.startswith("c"):
            argv = "inf"
            args = [argv]
            flags["f"] = True
        if "f" in flags:
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to force skip while other users are in voice")
        count = len(auds.queue)
        if not count:
            raise IndexError("Queue is currently empty.")
        if not argv:
            elems = [0]
        elif ":" in argv or ".." in argv:
            while "..." in argv:
                argv = argv.replace("...", "..")
            l = argv.replace("..", ":").split(":")
            it = None
            if len(l) > 3:
                raise ArgumentError("Too many arguments for range input.")
            elif len(l) > 2:
                num = await bot.evalMath(l[0], guild.id)
                it = int(round(float(num)))
            if l[0]:
                num = await bot.evalMath(l[0], guild.id)
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                left = num
            else:
                left = 0
            if l[1]:
                num = await bot.evalMath(l[1], guild.id)
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                right = num
            else:
                right = count
            elems = xrange(left, right, it)
        else:
            elems = [0] * len(args)
            for i in range(len(args)):
                elems[i] = await bot.evalMath(args[i], guild.id)
        if not "f" in flags:
            valid = True
            for e in elems:
                if not isValid(e):
                    valid = False
                    break
            if not valid:
                elems = range(count)
        members = sum(1 for m in auds.vc.channel.members if not m.bot)
        required = 1 + members >> 1
        response = "```css\n"
        i = 1
        for pos in elems:
            pos = float(pos)
            try:
                if not isValid(pos):
                    if "f" in flags:
                        auds.queue.clear()
                        create_future_ex(auds.new)
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
                song.played = True
                auds.preparing = False
                if auds.source is not None:
                    auds.source.advanced = True
                auds.queue.advance()
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
        auds = await forceJoin(guild, channel, user, client, bot)
        auds.preparing = False
        if name in ("pause", "stop"):
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to " + name + " while other users are in voice")
        elif auds.stats.position <= 0:
            if auds.queue and "played" in auds.queue[0]:
                auds.queue[0].pop("played")
        if name == "stop":
            auds.seek(0)
        if not auds.paused > 1:
            auds.paused = auds.pausec = name in ("pause", "stop")
            if auds.paused:
                create_future_ex(auds.vc.stop)
        if not auds.paused:
            create_future_ex(auds.queue.update_play)
            create_future_ex(auds.ensure_play)
        if auds.player is not None:
            auds.player.time = 1 + utc()
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
    rate_limit = (0.5, 2.5)

    async def __call__(self, argv, bot, guild, client, user, perm, channel, name, flags, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if not isAlone(auds, user) and perm < 1:
            raise self.permError(perm, 1, "to seek while other users are in voice")
        if name == "replay":
            num = 0
        else:
            orig = auds.stats.position
            expr = argv
            num = await bot.evalTime(expr, guild, orig)
        pos = auds.seek(num)
        if auds.player is not None:
            auds.player.time = 1 + utc()
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
    q = [copyDict(item) for item in auds.queue]
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
    name = ["Save", "Load"]
    alias = name + ["Dujmpö"]
    min_level = 0
    min_display = "0~1"
    description = "Saves or loads the currently playing audio queue state."
    usage = "<data{attached_file}> <append(?a)> <hide(?h)>"
    flags = "ah"
    rate_limit = (1, 2)

    async def __call__(self, guild, channel, user, client, bot, perm, name, argv, flags, message, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if not argv and not len(message.attachments) or name == "save":
            if name == "load":
                raise ArgumentError("Please input a file, URL or json data to load.")
            fut = create_task(channel.trigger_typing())
            resp = await create_future(getDump, auds)
            f = discord.File(io.BytesIO(bytes(resp, "utf-8")), filename="dump.json")
            await fut
            create_task(sendFile(channel, "Queue data for **" + guild.name + "**:", f))
            return
        if not isAlone(auds, user) and perm < 1:
            raise self.permError(perm, 1, "to load new queue while other users are in voice")
        try:
            if len(message.attachments):
                url = message.attachments[0].url
            else:
                url = verifyURL(argv)
            url = await bot.followURL(url)
            s = await create_future(Request, url)
            s = s[s.index(b"{"):]
            if s[-4:] == b"\n```":
                s = s[:-4]
        except:
            s = argv
            print(traceback.format_exc())
        d = json.loads(s.strip())
        fut = create_task(channel.trigger_typing())
        q = d["queue"]
        for i in range(len(q)):
            e = q[i] = freeClass(q[i])
            e.u_id = user.id
            e.skips = []
            if not 1 + i & 2047:
                await asyncio.sleep(0.2)
        if auds.player is not None:
            auds.player.time = 1 + utc()
        if auds.stats.shuffle:
            shuffle(q)
        for k in d["stats"]:
            if k not in auds.stats:
                d["stats"].pop(k)
            if k in "loop shuffle quiet":
                d["stats"][k] = bool(d["stats"][k])
            else:
                d["stats"][k] = float(d["stats"][k])
        await fut
        if "a" not in flags:
            await create_future(auds.new)
            auds.preparing = True
            auds.queue.clear()
            auds.stats.update(d["stats"])
            auds.queue.enqueue(q, -1)
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
        "Repeat": "repeat",
        "ShuffleQueue": "shuffle",
        "Quiet": "quiet",
        "Reset": "reset",
        "Stay": "stay",
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
        "LoopOne": "repeat",
        "L1": "repeat",
        "SQ": "shuffle",
        "24/7": "stay",
    }
    rate_limit = (0.5, 3)

    def __init__(self, *args):
        self.alias = list(self.aliasMap) + list(self.aliasExt)[1:]
        self.name = list(self.aliasMap)
        self.min_level = 0
        self.min_display = "0~1"
        self.description = "Changes the current audio settings for this server."
        self.usage = (
            "<value[]> <volume()(?v)> <speed(?s)> <pitch(?p)> <pan(?e)> <bassboost(?b)> <reverb(?r)> <compressor(?c)>"
            + " <chorus(?u)> <nightcore(?n)> <loop(?l)> <repeat(?1)> <shuffle(?x)> <quiet(?q)> <stay(?t)> <disable_all(?d)> <hide(?h)>"
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
        if "1" in flags:
            ops.append("repeat")
        if "x" in flags:
            ops.append("shuffle")
        if "q" in flags:
            ops.append("quiet")
        if "t" in flags:
            ops.append("stay")
        if not disable and not argv and (len(ops) != 1 or ops[-1] not in "loop repeat shuffle quiet stay"):
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
            raise self.permError(perm, 1, "to modify audio settings while other users are in voice")
        if not ops:
            if disable:
                pos = auds.stats.position
                res = False
                for k, v in auds.defaults.items():
                    if k != "volume" and auds.stats.get(k) != v:
                        res = True
                        break
                auds.stats = freeClass(auds.defaults)
                if auds.queue and res:
                    await create_future(auds.new, auds.file, pos)
                return (
                    "```css\nSuccessfully reset all audio settings for ["
                    + noHighlight(guild.name) + "].```"
                )
            else:
                ops.append("volume")
        s = ""
        for op in ops:
            # if op not in "volume loop repeat shuffle quiet reset":
            #     if not bot.isTrusted(guild.id):
            #         raise PermissionError("Must be in a trusted server for targeted audio setting.")
            if type(op) is str and op in "loop repeat shuffle quiet stay" and not argv:
                argv = str(not bot.database.playlists.audio[guild.id].stats[op])
            if disable:
                val = auds.defaults[op]
                if type(val) is not bool:
                    val *= 100
                argv = str(val)
            origStats = bot.database.playlists.audio[guild.id].stats
            orig = round(origStats[op] * 100, 9)
            num = await bot.evalMath(argv, guild.id, orig)
            val = roundMin(float(num / 100))
            new = round(num, 9)
            if op in "loop repeat shuffle quiet stay":
                origStats[op] = new = bool(val)
                orig = bool(orig)
            else:
                origStats[op] = val
            if auds.queue:# and op in "speed pitch pan bassboost reverb compressor chorus resample":
                await create_future(auds.new, auds.file, auds.stats.position)
            s += (
                "\nChanged audio {" + str(op)
                + "} setting from [" + str(orig)
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
    rate_limit = 5

    async def __call__(self, perm, argv, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        amount = await bot.evalMath(argv, guild.id)
        if len(auds.queue) > 1 and amount:
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to rotate queue while other users are in voice")
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
    rate_limit = 5

    async def __call__(self, perm, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if len(auds.queue) > 1:
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to shuffle queue while other users are in voice")
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
    rate_limit = 5

    async def __call__(self, perm, flags, guild, channel, user, client, bot, **void):
        auds = await forceJoin(guild, channel, user, client, bot)
        if len(auds.queue) > 1:
            if not isAlone(auds, user) and perm < 1:
                raise self.permError(perm, 1, "to reverse queue while other users are in voice")
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
    rate_limit = 10

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
    rate_limit = 10

    async def __call__(self, guild, flags, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                if user.id != self.bot.client.user.id:
                    if user.voice is not None:
                        create_task(user.move_to(None))
        if "h" not in flags:
            return (
                "```css\nSuccessfully removed all users in voice channels in ["
                + noHighlight(guild.name) + "].```", 1
            )


class Player(Command):
    server_only = True
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
    rate_limit = 1

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
        if abs(b) > 1 / 6:
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
        u = auds.stats.chorus
        if u:
            output += "📊"
        c = auds.stats.compressor
        if c:
            output += "🗜️"
        e = auds.stats.pan
        if abs(e - 1) > 0.25:
            output += "♒"
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
            p = [auds.stats.position, e_dur(q[0].duration)]
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
                create_task(message.add_reaction(b.decode("utf-8")))
                await asyncio.sleep(0.5)
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
            maxdel = e_dur(auds.queue[0].duration) - auds.stats.position + 2
            delay = min(maxdel, e_dur(auds.queue[0].duration) / self.barsize / abs(auds.stats.speed))
            if delay > 20:
                delay = 20
            elif delay < 6:
                delay = 6
        else:
            delay = inf
        auds.player.time = utc() + delay

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
                raise self.permError(perm, req, "to " + reason + " virtual audio player for " + noHighlight(guild.name))
        if "d" in flags:
            auds.player = None
            return (
                "```css\nDisabled virtual audio players in ["
                + noHighlight(channel.guild.name) + "].```"
            )
        await createPlayer(auds, p_type="c" in flags, verbose="z" in flags)


def extract_lyrics(s):
    s = s[s.index("JSON.parse(") + len("JSON.parse("):]
    s = s[:s.index("</script>")]
    if "window.__" in s:
        s = s[:s.index("window.__")]
    s = s[:s.rindex(");")]
    data = ast.literal_eval(s)
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        d = eval(data, {}, eval_const)
    lyrics = d["songPage"]["lyricsData"]["body"]["children"][0]["children"]
    newline = True
    output = ""
    while lyrics:
        line = lyrics.pop(0)
        if type(line) is str:
            if line:
                if line.startswith("["):
                    output += "\n"
                    newline = False
                if "]" in line:
                    if line == "]":
                        if output.endswith(" ") or output.endswith("\n"):
                            output = output[:-1]
                    newline = True
                output += line + ("\n" if newline else (" " if not line.endswith(" ") else ""))
        elif type(line) is dict:
            if "children" in line:
                lyrics = line["children"] + lyrics
    return output


def get_lyrics(item):
    url = "https://api.genius.com/search"
    for i in range(2):
        header = {"Authorization": "Bearer " + genius_key}
        if i == 0:
            search = item
        else:
            search = "".join(shuffle(item.split()))
        data = {"q": search}
        resp = Request(url, data=data, headers=header)
        rdata = json.loads(resp)
        hits = rdata["response"]["hits"]
        name = None
        path = None
        for h in hits:
            try:
                name = h["result"]["title"]
                path = h["result"]["api_path"]
                break
            except KeyError:
                print(traceback.format_exc())
        if path and name:
            s = "https://genius.com" + path
            page = Request(s, headers=header, decode=True)
            text = page
            html = BeautifulSoup(text, "html.parser")
            lyricobj = html.find('div', class_='lyrics')
            if lyricobj is not None:
                lyrics = lyricobj.get_text().strip()
                print("lyrics_html", s)
                return name, lyrics
            try:
                lyrics = extract_lyrics(text).strip()
                print("lyrics_json", s)
                return name, lyrics
            except:
                if i:
                    raise
                print(traceback.format_exc())
                print(s)
                print(text)
    raise LookupError("No results for " + item + ".")


class Lyrics(Command):
    time_consuming = True
    name = ["SongLyrics"]
    min_level = 0
    description = "Searches genius.com for lyrics of a song."
    usage = "<0:search_link{queue}> <verbose(?v)>"
    flags = "v"
    lyric_trans = re.compile(
        (
            "[([]+"
            "(((official|full|demo|original|extended) *)?"
            "((version|ver.?) *)?"
            "((w\\/)?"
            "(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq) *)?"
            "((album|video|audio|cover|remix) *)?"
            "(upload|reupload|version|ver.?)?"
            "|(feat|ft)"
            ".+)"
            "[)\\]]+"
        ),
        flags=re.I,
    )
    rate_limit = (2, 3)

    async def __call__(self, bot, channel, message, argv, flags, user, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            try:
                auds = bot.database.playlists.audio[guild.id]
                if not auds.queue:
                    raise LookupError
                argv = auds.queue[0].name
            except LookupError:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        if isURL(argv):
            argv = await bot.followURL(argv)
            resp = await create_future(ytdl.search, argv)
            search = resp[0].name
        else:
            search = argv
        search = search.translate(self.bot.mtrans)
        item = verifySearch(to_alphanumeric(re.sub(self.lyric_trans, "", search)))
        if not item:
            item = verifySearch(to_alphanumeric(search))
            if not item:
                item = search
        name, lyrics = await create_future(get_lyrics, item)
        text = clrHighlight(lyrics.strip()).replace("#", "♯")
        msg = "Lyrics for **" + discord.utils.escape_markdown(name) + "**:"
        s = msg + "```ini\n" + text + "```"
        if "v" not in flags and len(s) <= 2000:
            return s
        title = "Lyrics for " + name + ":"
        if len(text) > 6000:
            return (title + "\n\n" + text).strip()
        emb = discord.Embed(colour=randColour())
        emb.set_author(name=title)
        curr = ""
        paragraphs = [p + "\n\n" for p in text.split("\n\n")]
        while paragraphs:
            para = paragraphs.pop(0)
            if not emb.description and len(curr) + len(para) > 2000:
                if len(para) <= 2000:
                    emb.description = "```ini\n" + curr.strip() + "```"
                    curr = para
                else:
                    p = [i + "\n" for i in para.split("\n")]
                    if len(p) <= 1:
                        p = [i + "" for i in para.split()]
                        if len(p) <= 1:
                            p = list(para)
                    paragraphs = p + paragraphs
            elif emb.description and len(curr) + len(para) > 1000:
                if len(para) <= 1000:
                    emb.add_field(name="Page " + str(len(emb.fields) + 2), value="```ini\n" + curr.strip() + "```", inline=False)
                    curr = para
                else:
                    p = [i + "\n" for i in para.split("\n")]
                    if len(p) <= 1:
                        p = [i + "" for i in para.split()]
                        if len(p) <= 1:
                            p = list(para)
                    paragraphs = p + paragraphs
            else:
                curr += para
        if curr:
            if emb.description:
                emb.add_field(name="Page " + str(len(emb.fields) + 2), value="```ini\n" + curr.strip() + "```", inline=False)
            else:
                emb.description = "```ini\n" + curr.strip() + "```"
        try:
            if len(emb) > 6000:
                raise discord.HTTPException
            bot.embedSender(channel, emb)
        except discord.HTTPException:
            print(traceback.format_exc())
            return (title + "\n\n" + emb.description + "\n\n".join(noCodeBox(f.value) for f in emb.fields)).strip()


class Download(Command):
    time_consuming = True
    _timeout_ = 8
    name = ["Search", "YTDL", "Youtube_DL", "AF", "AudioFilter", "ConvertORG", "Org2xm", "Convert"]
    min_level = 0
    description = "Searches and/or downloads a song from a YouTube/SoundCloud query or audio file link."
    usage = "<0:search_link{queue}> <-1:out_format[ogg]> <apply_settings(?a)> <verbose_search(?v)> <show_debug(?z)>"
    flags = "avz"
    rate_limit = (7, 12)

    async def __call__(self, bot, channel, guild, message, name, argv, flags, user, **void):
        if name in ("af", "audiofilter"):
            setDict(flags, "a", 1)
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            try:
                auds = await forceJoin(guild, channel, user, bot.client, bot)
                if not auds.queue:
                    raise EOFError
                res = [{"name": e.name, "url": e.url} for e in auds.queue[:10]]
                fmt = "ogg"
                end = "Current items in queue for " + guild.name + ":"
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        else:
            if " " in argv:
                try:
                    spl = shlex.split(argv)
                except ValueError:
                    spl = argv.split(" ")
                if len(spl) >= 1:
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
                    except:
                        print(r)
                        print(traceback.format_exc())
            if not res:
                raise LookupError("No results for " + argv + ".")
            res = res[:10]
            end = "Search results for " + argv + ":"
        a = flags.get("a", 0)
        end += "\nDestination format: {." + fmt + "}"
        if a:
            end += ", Audio settings: {ON}"
        end += "```"
        url_bytes = bytes(repr([e["url"] for e in res]), "utf-8")
        url_enc = bytes2B64(url_bytes, True).decode("utf-8", "replace")
        msg = (
            "```" + "\n" * ("z" in flags) + "callback-voice-download-" + str(user.id) 
            + "_" + str(len(res)) + "_" + fmt + "_" + str(int(bool(a))) + "-" + url_enc + "\n" + end
        )
        emb = discord.Embed(colour=randColour())
        url = bestURL(user)
        emb.set_author(name=str(user), url=url, icon_url=url)
        emb.description = "\n".join(
            ["`【" + str(i) + "】` [" + discord.utils.escape_markdown(e["name"] + "](" + ensure_url(e["url"]) + ")") for i in range(len(res)) for e in [res[i]]]
        )
        sent = await channel.send(
            msg,
            embed=emb,
        )
        for i in range(len(res)):
            create_task(sent.add_reaction(str(i) + b"\xef\xb8\x8f\xe2\x83\xa3".decode("utf-8")))
            await asyncio.sleep(0.5)
        # await sent.add_reaction("❎")

    async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, argv, user, **void):
        if reaction is None or user.id == bot.client.user.id:
            return
        spl = vals.split("_")
        u_id = int(spl[0])
        if user.id == u_id or not perm < 3:
            if b"\xef\xb8\x8f\xe2\x83\xa3" in reaction:
                num = int(reaction.decode("utf-8")[0])
                if num <= int(spl[1]):
                    data = ast.literal_eval(b642Bytes(argv, True).decode("utf-8", "replace"))
                    url = data[num]
                    if guild is None:
                        fl = 8388608
                    else:
                        fl = guild.filesize_limit
                    create_task(channel.trigger_typing())
                    create_task(message.edit(
                        content="```ini\nDownloading [" + noHighlight(ensure_url(url)) + "]...```",
                        embed=None,
                    ))
                    try:
                        if int(spl[3]):
                            auds = bot.database.playlists.audio[guild.id]
                        else:
                            auds = None
                    except LookupError:
                        auds = None
                    fn, out = await create_future(
                        ytdl.download_file,
                        url,
                        fmt=spl[2],
                        auds=auds,
                        fl=fl,
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
                        filename=fn,
                    )
                    try:
                        create_future_ex(os.remove, fn)
                    except (PermissionError, FileNotFoundError):
                        pass
                    create_task(bot.silentDelete(message, no_log=True))


class UpdateQueues(Database):
    name = "playlists"

    def __load__(self):
        self.audio = {}
        self.audiocache = {}
        self.connecting = {}
        pl = self.data
        for g in pl:
            for i in range(len(pl[g])):
                e = pl[g][i]
                if type(e) is dict:
                    pl[g][i] = freeClass(e)

    def is_connecting(self, g):
        if g in self.connecting:
            if utc() - self.connecting[g] < 12:
                return True
            self.connecting.pop(g)
        return False

    async def research(self, auds):
        if auds.searching >= 1:
            return
        auds.searching += 1
        searched = 0
        q = auds.queue
        for i, e in enumerate(q):
            if searched >= 32 or i >= 128:
                break
            if "research" in e:
                try:
                    await create_future(ytdl.extractSingle, e)
                    try:
                        e.pop("research")
                    except KeyError:
                        pass
                    searched += 1
                except:
                    try:
                        e.pop("research")
                    except KeyError:
                        pass
                    print(traceback.format_exc())
                    break
            if not 1 + i & 7:
                await asyncio.sleep(0.4)
        await asyncio.sleep(2)
        auds.searching = max(auds.searching - 1, 0)

    async def _typing_(self, channel, user, **void):
        if not hasattr(channel, "guild") or channel.guild is None:
            return
        if channel.guild.id in self.audio and user.id != self.bot.client.user.id:
            auds = self.audio[channel.guild.id]
            if auds.player is not None and channel.id == auds.channel.id:
                t = utc() + 10
                if auds.player.time < t:
                    auds.player.time = t

    async def _send_(self, message, **void):
        if message.guild.id in self.audio and message.author.id != self.bot.client.user.id:
            auds = self.audio[message.guild.id]
            if auds.player is not None and message.channel.id == auds.channel.id:
                t = utc() + 10
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
                create_future_ex(auds.update)
        else:
            a = 1
            for g in tuple(self.audio):
                try:
                    auds = self.audio[g]
                    create_future_ex(auds.update)
                    create_task(self.research(auds))
                except:
                    print(traceback.format_exc())
                if not a & 15:
                    await asyncio.sleep(0.2)
                a += 1
            # t = utc()
            # i = 1
            # for path in os.listdir("cache"):
            #     fn = "cache/" + path
            #     if path.startswith("&"):
            #         if t - os.path.getmtime(fn) > 3600:
            #             os.remove(fn)
            #     if not i & 1023:
            #         await asyncio.sleep(0.2)
            #         t = utc()
            #     i += 1
            await asyncio.sleep(0.5)
            for item in tuple(ytdl.cache.values()):
                await create_future(item.update)
        create_future_ex(ytdl.update_dl)
        self.busy = max(0, self.busy - 1)

    def _announce_(self, *args, **kwargs):
        for auds in self.audio.values():
            auds.announce(*args, **kwargs)

    def _destroy_(self, **void):
        for auds in self.audio.values():
            auds.kill()

    def _ready_(self, bot, **void):
        ytdl.bot = bot