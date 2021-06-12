print = PRINT

import youtube_dl
from bs4 import BeautifulSoup
# youtube_dl = youtube_dlc

# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000


try:
    genius_key = AUTH["genius_key"]
    if not genius_key:
        raise
except:
    genius_key = None
    print("WARNING: genius_key not found. Unable to use API to search song lyrics.")
try:
    google_api_key = AUTH["google_api_key"]
    if not google_api_key:
        raise
except:
    google_api_key = None
    print("WARNING: google_api_key not found. Unable to use API to search youtube playlists.")


async def create_player(auds, p_type=0, verbose=False):
    auds.stats.quiet |= 2 * p_type
    # Set callback message for updating audio player
    text = (
        "```" + "\n" * verbose + "callback-voice-player-" + str(int(bool(p_type)))
        + "\nInitializing virtual audio player...```"
    )
    await auds.text.send(text)
    await auds.update_player()


# Gets estimated duration from duration stored in queue entry
e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else 300)


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration(filename, _timeout=12):
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        filename,
    )
    resp = None
    try:
        proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        fut = create_future_ex(proc.wait, timeout=_timeout)
        res = fut.result(timeout=_timeout)
        resp = proc.stdout.read().split()
    except:
        with suppress():
            proc.kill()
        with suppress():
            resp = proc.stdout.read().split()
        print_exc()
    try:
        dur = float(resp[0])
    except (IndexError, ValueError, TypeError):
        dur = None
    bps = None
    if resp and len(resp) > 1:
        with suppress(ValueError):
            bps = float(resp[1])
    return dur, bps

DUR_CACHE = {}

def get_duration(filename):
    if filename:
        with suppress(KeyError):
            return DUR_CACHE[filename]
        dur, bps = _get_duration(filename, 4)
        if not dur and is_url(filename):
            with requests.get(filename, headers=Request.header(), stream=True) as resp:
                head = fcdict(resp.headers)
                if "Content-Length" not in head:
                    dur = _get_duration(filename, 20)[0]
                    DUR_CACHE[filename] = dur
                    return dur
                if bps:
                    print(head, bps, sep="\n")
                    return (int(head["Content-Length"]) << 3) / bps
                ctype = [e.strip() for e in head.get("Content-Type", "").split(";") if "/" in e][0]
                if ctype.split("/", 1)[0] not in ("audio", "video") or ctype == "audio/midi":
                    DUR_CACHE[filename] = nan
                    return nan
                it = resp.iter_content(65536)
                data = next(it)
            ident = str(magic.from_buffer(data))
            print(head, ident, sep="\n")
            try:
                bitrate = regexp("[0-9.]+\\s.?bps").findall(ident)[0].casefold()
            except IndexError:
                dur = _get_duration(filename, 16)[0]
                DUR_CACHE[filename] = dur
                return dur
            bps, key = bitrate.split(None, 1)
            bps = float(bps)
            if key.startswith("k"):
                bps *= 1e3
            elif key.startswith("m"):
                bps *= 1e6
            elif key.startswith("g"):
                bps *= 1e9
            dur = (int(head["Content-Length"]) << 3) / bps
        DUR_CACHE[filename] = dur
        return dur


# Gets the best icon/thumbnail for a queue entry.
def get_best_icon(entry):
    with suppress(KeyError):
        return entry["icon"]
    with suppress(KeyError):
        return entry["thumbnail"]
    try:
        thumbnails = entry["thumbnails"]
    except KeyError:
        try:
            url = entry["webpage_url"]
        except KeyError:
            url = entry["url"]
        if is_discord_url(url):
            if not is_image(url):
                return "https://cdn.discordapp.com/embed/avatars/0.png"
        if ytdl.bot.is_webserver_url(url):
            return ytdl.bot.raw_github + "/master/misc/sky-rainbow.gif"
            # return ytdl.bot.webserver + "/static/sky-rainbow.gif"
        return url
    return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]


# Gets the best audio file download link for a queue entry.
def get_best_audio(entry):
    with suppress(KeyError):
        return entry["stream"]
    best = -inf
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ()
    try:
        url = entry["webpage_url"]
    except KeyError:
        url = entry.get("url")
    replace = True
    for fmt in fmts:
        q = fmt.get("abr", 0)
        if type(q) is not int:
            q = 0
        vcodec = fmt.get("vcodec", "none")
        if vcodec not in (None, "none"):
            q -= 1
        u = as_str(fmt["url"])
        if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
            replace = False
        if q > best or replace:
            best = q
            url = fmt["url"]
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
    if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
        resp = Request(url)
        fmts = alist()
        with suppress(ValueError, KeyError):
            while True:
                search = b'<Representation id="'
                resp = resp[resp.index(search) + len(search):]
                f_id = as_str(resp[:resp.index(b'"')])
                search = b"><BaseURL>"
                resp = resp[resp.index(search) + len(search):]
                stream = as_str(resp[:resp.index(b'</BaseURL>')])
                fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
                fmt.url = stream
                fmts.append(fmt)
        entry["formats"] = fmts
        return get_best_audio(entry)
    if not url:
        raise KeyError("URL not found.")
    return url


# Gets the best video file download link for a queue entry.
def get_best_video(entry):
    with suppress(KeyError):
        return entry["stream"]
    best = -inf
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ()
    try:
        url = entry["webpage_url"]
    except KeyError:
        url = entry.get("url")
    replace = True
    for fmt in fmts:
        q = fmt.get("height", 0)
        if type(q) is not int:
            q = 0
        vcodec = fmt.get("vcodec", "none")
        if vcodec in (None, "none"):
            q -= 1
        u = as_str(fmt["url"])
        if not u.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
            replace = False
        if q > best or replace:
            best = q
            url = fmt["url"]
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
    if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
        resp = Request(url)
        fmts = alist()
        with suppress(ValueError, KeyError):
            while True:
                search = b'<Representation id="'
                resp = resp[resp.index(search) + len(search):]
                f_id = as_str(resp[:resp.index(b'"')])
                search = b"><BaseURL>"
                resp = resp[resp.index(search) + len(search):]
                stream = as_str(resp[:resp.index(b'</BaseURL>')])
                fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
                fmt.url = stream
                fmts.append(fmt)
        entry["formats"] = fmts
        return get_best_video(entry)
    if not url:
        raise KeyError("URL not found.")
    return url


# Joins a voice channel and returns the associated audio player.
async def auto_join(guild, channel, user, bot, preparing=False, vc=None):
    if type(channel) in (str, int):
        channel = await bot.fetch_channel(channel)
    if guild.id not in bot.data.audio.players:
        for func in bot.commands.connect:
            try:
                await func(user=user, channel=channel, vc=vc)
            except (discord.ClientException, AttributeError):
                pass
    try:
        auds = bot.data.audio.players[guild.id]
    except KeyError:
        raise LookupError("Unable to find voice channel.")
    auds.text = channel
    return auds


# Helper function to save all items in a queue
copy_entry = lambda item: {"name": item.name, "url": item.url, "duration": item.get("duration")}


async def disconnect_members(bot, guild, members, channel=None):
    if bot.id in (member.id for member in members):
        with suppress(KeyError):
            auds = bot.data.audio.players[guild.id]
            await create_future(auds.kill)
    futs = [create_task(member.move_to(None)) for member in members]
    for fut in futs:
        await fut


# Checks if the user is alone in voice chat (excluding bots).
def is_alone(auds, user):
    for m in auds.channel.members:
        if m.id != user.id and not m.bot:
            return False
    return True


# Replaces youtube search queries in youtube-dl with actual youtube search links.
def ensure_url(url):
    if url.startswith("ytsearch:"):
        url = f"https://www.youtube.com/results?search_query={verify_url(url[9:])}"
    return url


# This messy regex helps identify and remove certain words in song titles
lyric_trans = re.compile(
    (
        "[([]+"
        "(((official|full|demo|original|extended) *)?"
        "((version|ver.?) *)?"
        "((w\\/)?"
        "(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq|english) *)?"
        "((album|video|audio|cover|remix) *)?"
        "(upload|reupload|version|ver.?)?"
        "|(feat|ft)"
        ".+)"
        "[)\\]]+"
    ),
    flags=re.I,
)


# Audio player that wraps discord audio sources, contains a queue, and also manages audio settings.
class CustomAudio(collections.abc.Hashable):

    # Default player settings
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
        "bitrate": 1966.08,
        "loop": False,
        "repeat": False,
        "shuffle": False,
        "quiet": False,
        "stay": False,
    }
    paused = False
    source = None
    next = None
    timeout = 0
    seek_pos = 0
    ts = None

    def __init__(self, text=None):
        with tracebacksuppressor:
            # Class instance variables
            self.bot = bot
            self.stats = cdict(self.defaults)
            self.text = text
            self.fut = concurrent.futures.Future()
            self.acsi = None
            self.args = []
            self.queue = AudioQueue()
            self.queue._init_()
            self.queue.auds = self
            self.semaphore = Semaphore(1, 4, rate_limit=1 / 8)
            self.announcer = Semaphore(1, 1, rate_limit=1 / 3)
            self.search_sem = Semaphore(1, 1, rate_limit=1)

    def join(self, channel=None):
        if channel:
            if getattr(channel, "guild", None):
                self.guild = guild = channel.guild
                if not bot.is_trusted(guild):
                    self.queue.maxitems = 8192
                bot.data.audio.players[guild.id] = self
                self.stats.update(bot.data.audiosettings.get(guild.id, {}))
            create_future_ex(self.connect_to, channel)
            self.timeout = utc()

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: " + "{" + f'"acsi": {self.acsi}, "queue": {len(self.queue)}, "stats": {self.stats}, "source": {self.source}' + "}"

    __hash__ = lambda self: self.guild.id

    def __getattr__(self, key):
        if key in ("reverse", "speed", "epos", "pos"):
            return self.__getattribute__("_" + key)()
        try:
            return self.__getattribute__(key)
        except AttributeError:
            pass
        try:
            return getattr(self.__getattribute__("queue"), key)
        except AttributeError:
            pass
        self.fut.result(timeout=12)
        return getattr(self.__getattribute__("acsi"), key)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(("reverse", "speed", "epos", "pos"))
        data.update(dir(self.acsi))
        data.update(dir(self.queue))
        return data

    def has_options(self):
        stats = self.stats
        return stats.volume != 1 or stats.reverb != 0 or stats.pitch != 0 or stats.speed != 1 or stats.pan != 1 or stats.bassboost != 0 or stats.compressor != 0 or stats.chorus != 0 or stats.resample != 0

    def get_dump(self, position=False, js=False):
        with self.semaphore:
            lim = 1024
            q = [copy_entry(item) for item in self.queue.verify()]
            s = {k: (v if not isinstance(v, mpf) else str(v) if len(str(v)) > 16 else float(v)) for k, v in self.stats.items()}
            d = {
                "stats": s,
                "queue": q,
            }
            if position:
                d["pos"] = self.pos
            if js:
                d = json.dumps(d).encode("utf-8")
                if len(d) > 2097152:
                    d = bytes2zip(d)
                    return d, "dump.zip"
                return d, "dump.json"
                # if len(q) > lim:
                #     s = pickle.dumps(d)
                #     if len(s) > 262144:
                #         return encrypt(bytes2zip(s)), "dump.bin"
                #     return encrypt(s), "dump.bin"
                # return json.dumps(d).encode("utf-8"), "dump.json"
            return d, None

    def _reverse(self):
        return self.stats.speed < 0

    def _speed(self):
        return abs(self.stats.speed)

    def _epos(self):
        self.fut.result(timeout=12)
        pos = self.acsi.pos
        if not pos[1] and self.queue:
            dur = e_dur(self.queue[0].get("duration"))
            return min(dur, pos[0]), dur
        elif pos[1] is None:
            return 0, 0
        return pos

    def _pos(self):
        return self.epos[0]

    def skip(self):
        self.acsi.skip()

    def clear_source(self):
        if self.source:
            create_future_ex(self.acsi.clear_source)
        self.source = None
    
    def clear_next(self):
        if self.next:
            create_future_ex(self.acsi.clear_next)
        self.next = None

    def reset(self, start=True):
        self.acsi.clear()
        self.source = self.next = None
        if start:
            self.queue.update_load()

    def pause(self, unpause=False):
        if unpause and self.paused:
            self.paused = False
            self.acsi.resume()
            self.queue.update_load()
        if not self.paused:
            self.paused = True
            self.acsi.pause()

    def resume(self):
        if self.paused:
            self.paused = False
            self.acsi.resume()
        else:
            self.acsi.read()
        self.queue.update_load()

    # Stops currently playing source, closing it if possible.
    def stop(self):
        self.acsi.stop()
        self.paused = True
        return self.reset(start=False)

    # Loads and plays a new audio source, with current settings and optional song init position.
    def play(self, source=None, pos=0, update=True):
        self.seek_pos = 0
        if source is not None:
            self.source = source
            src = None
            try:
                # This call may take a while depending on the time taken by FFmpeg to start outputting
                source.readable.result(timeout=12)
                src = source.create_reader(pos, auds=self)
            except OverflowError:
                self.clear_source()
            else:
                if src:
                    # Only stop and replace audio source when the next one is buffered successfully and readable
                    self.acsi.play(src, after=self.queue.advance)
        else:
            self.stop()

    def enqueue(self, source):
        self.next = source
        with tracebacksuppressor:
            source.readable.result(timeout=12)
            src = source.create_reader(0, auds=self)
            self.acsi.enqueue(src, after=self.queue.advance)

    # Seeks current song position.
    def seek(self, pos):
        duration = self.epos[1]
        pos = max(0, pos)
        # Skip current song if position is out of range
        if (pos >= duration and not self.reverse) or (pos <= 0 and self.reverse):
            self.skip()
            return duration
        self.play(self.source, pos, update=False)
        return pos

    # Sends a deletable message to the audio player's text channel.
    def announce(self, *args, aio=False, dump=False, **kwargs):
        if self.queue and dump and (len(self.queue) > 1 or self.queue[0].get("skips") != ()):
            resp, fn = self.get_dump(js=True)
            f = CompatFile(io.BytesIO(resp), filename=fn)
        else:
            f = None
        if aio:
            return create_task(send_with_react(self.text, *args, file=f, reacts="❎", **kwargs))
        with self.announcer:
            return await_fut(send_with_react(self.text, *args, file=f, reacts="❎", **kwargs))

    # Kills this audio player, stopping audio playback. Will cause bot to leave voice upon next update event.
    def kill(self, reason=None):
        self.acsi.kill()
        self.bot.data.audio.players.pop(self.guild.id, None)
        with suppress(LookupError):
            if reason is None:
                reason = css_md(f"🎵 Successfully disconnected from {sqr_md(self.guild)}. 🎵")
            if reason:
                self.announce(reason, dump=True)

    # Update event, ensures audio is playing correctly and moves, leaves, or rejoins voice when necessary.
    def update(self, *void1, **void2):
        with tracebacksuppressor:
            self.fut.result(timeout=12)
            guild = self.guild
            if self.stats.stay:
                cnt = inf
            else:
                cnt = sum(1 for m in self.acsi.channel.members if not m.bot)
            if not cnt:
                # Timeout for leaving is 120 seconds
                if self.timeout < utc() - 120:
                    return self.kill(css_md(f"🎵 Automatically disconnected from {sqr_md(guild)}: All channels empty. 🎵"))
                perms = self.acsi.channel.permissions_for(guild.me)
                if not perms.connect or not perms.speak:
                    return self.kill(css_md(f"🎵 Automatically disconnected from {sqr_md(guild)}: No permission to connect/speak in {sqr_md(self.acsi.channel)}. 🎵"))
                # If idle for more than 10 seconds, attempt to find members in other voice channels
                elif self.timeout < utc() - 10:
                    if guild.afk_channel and (guild.afk_channel.id != self.acsi.channel.id and guild.afk_channel.permissions_for(guild.me).connect):
                        await_fut(self.move_unmute(self.acsi, guild.afk_channel))
                    else:
                        cnt = 0
                        ch = None
                        for channel in guild.voice_channels:
                            if not guild.afk_channel or channel.id != guild.afk_channel.id:
                                c = sum(1 for m in channel.members if not m.bot)
                                if c > cnt:
                                    cnt = c
                                    ch = channel
                        if ch:
                            with tracebacksuppressor(SemaphoreOverflowError):
                                await_fut(self.acsi.move_to(ch))
                                self.announce(ini_md(f"🎵 Detected {sqr_md(cnt)} user{'s' if cnt != 1 else ''} in {sqr_md(ch)}, automatically joined! 🎵"), aio=False)
            else:
                self.timeout = utc()
            self.queue.update_load()

    # Moves to the target channel, unmuting self afterwards.
    async def move_unmute(self, vc, channel):
        await vc.move_to(channel)
        await channel.guild.me.edit(mute=False, deafen=False)

    def connect_to(self, channel=None):
        if not self.acsi:
            try:
                self.acsi = AudioClientSubInterface(channel)
                self.fut.set_result(self.acsi)
            except Exception as ex:
                self.fut.set_exception(ex)
        self.queue._init_(auds=self)
        self.timeout = utc()
        return self.acsi

    # Constructs array of FFmpeg options using the audio settings.
    def construct_options(self, full=True):
        stats = self.stats
        # Pitch setting is in semitones, so frequency is on an exponential scale
        pitchscale = 2 ** ((stats.pitch + stats.resample) / 12)
        chorus = min(16, abs(stats.chorus))
        reverb = stats.reverb
        volume = stats.volume
        # FIR sample for reverb
        if reverb:
            args = ["-i", "misc/SNB3,0all.wav"]
        else:
            args = []
        options = deque()
        # This must be first, else the filter will not initialize properly
        if not is_finite(stats.compressor):
            options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
        # Reverses song, this may be very resource consuming
        if self.stats.speed < 0:
            options.append("areverse")
        # Adjusts song tempo relative to speed, pitch, and nightcore settings
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
        # Adjusts resample to match song pitch
        if pitchscale != 1:
            if abs(pitchscale) >= 64:
                raise OverflowError
            if full:
                options.append("aresample=" + str(SAMPLE_RATE))
            options.append("asetrate=" + str(SAMPLE_RATE * pitchscale))
        # Chorus setting, this is a bit of a mess
        if chorus:
            A = B = C = D = ""
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
                "chorus=0.5:" + str(round(b, 3)) + ":"
                + A + ":"
                + B + ":"
                + C + ":"
                + D
            )
            volume *= 2
        # Compressor setting, this needs a bit of tweaking perhaps
        if stats.compressor:
            comp = min(8000, abs(stats.compressor * 10 + sgn(stats.compressor)))
            while abs(comp) > 1:
                c = min(20, comp)
                try:
                    comp /= c
                except ZeroDivisionError:
                    comp = 1
                mult = str(round((c * math.sqrt(2)) ** 0.5, 4))
                options.append(
                    "acompressor=mode=" + ("upward" if stats.compressor < 0 else "downward")
                    + ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
                )
        # Bassboost setting, the ratio is currently very unintuitive and definitely needs tweaking
        if stats.bassboost:
            opt = "anequalizer="
            width = 4096
            x = round(sqrt(1 + abs(stats.bassboost)), 5)
            coeff = width * max(0.03125, (0.25 / x))
            ch = " f=" + str(coeff if stats.bassboost > 0 else width - coeff) + " w=" + str(coeff / 2) + " g=" + str(max(0.5, min(48, 4 * math.log2(x * 5))))
            opt += "c0" + ch + "|c1" + ch
            options.append(opt)
        # Reverb setting, using afir and aecho FFmpeg filters.
        if reverb:
            coeff = abs(reverb)
            wet = min(3, coeff) / 3
            # Split audio into 2 inputs if wet setting is between 0 and 1, one input passes through FIR filter
            if wet != 1:
                options.append("asplit[2]")
            volume *= 1.2
            options.append("afir=dry=10:wet=10")
            # Must include amix if asplit is used
            if wet != 1:
                dry = 1 - wet
                options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(wet, 6)))
            if coeff > 1:
                decay = round(1 - 4 / (3 + coeff), 4)
                options.append(f"aecho=1:1:400|600:{decay}|{decay / 2}")
                if decay >= 0.25:
                    options.append(f"aecho=1:1:800|1100:{decay / 4}|{decay / 8}")
        # Pan setting, uses extrastereo and volume filters to balance
        if stats.pan != 1:
            pan = min(10000, max(-10000, stats.pan))
            while abs(abs(pan) - 1) > 0.001:
                p = max(-10, min(10, pan))
                try:
                    pan /= p
                except ZeroDivisionError:
                    pan = 1
                options.append("extrastereo=m=" + str(p) + ":c=0")
                volume *= 1 / max(1, round(math.sqrt(abs(p)), 4))
        if volume != 1:
            options.append("volume=" + str(round(volume, 7)))
        # Soft clip audio using atan, reverb filter requires -filter_complex rather than -af option
        if options:
            if stats.compressor:
                options.append("alimiter")
            elif volume > 1:
                options.append("asoftclip=atan")
            args.append(("-af", "-filter_complex")[bool(reverb)])
            args.append(",".join(options))
        # print(args)
        return args


# Manages the audio queue. Has optimized insertion/removal on both ends, and linear time lookup. One instance of this class is created per audio player.
class AudioQueue(alist):

    maxitems = 262144

    def _init_(self, auds=None):
        self.lastsent = 0
        self.loading = False
        self.playlist = None
        self.sem = Semaphore(1, 0)
        self.wait = concurrent.futures.Future()
        if auds:
            self.auds = auds
            self.bot = auds.bot
            self.acsi = auds.acsi
            self.wait.set_result(auds)

    def announce_play(self, e):
        auds = self.auds
        if not auds.stats.quiet:
            if utc() - self.lastsent > 1:
                try:
                    u = self.bot.cache.users[e.u_id]
                    name = u.display_name
                except KeyError:
                    name = "Deleted User"
                self.lastsent = utc()
                auds.announce(italics(ini_md(f"🎵 Now playing {sqr_md(e.name)}, added by {sqr_md(name)}! 🎵")))

    def start_queue(self):
        if self.sem.is_busy():
            return
        with tracebacksuppressor:
            auds = self.auds
            if not auds.source and self:
                e = self[0]
                source = None
                with self.sem:
                    with tracebacksuppressor:
                        source = ytdl.get_stream(e, force=True)
                if not source:
                    if self.sem.is_busy():
                        self.sem.wait()
                    return self.update_load()
                with self.sem:
                    self.announce_play(e)
                    self.auds.play(source, pos=auds.seek_pos)
            if not auds.next and auds.source and len(self) > 1:
                with self.sem:
                    e = self[1]
                    source = ytdl.get_stream(e)
                    if source:
                        auds.enqueue(source)

    # Update queue, loading all file streams that would be played soon
    def update_load(self):
        self.wait.result()
        q = self
        if q:
            dels = deque()
            for i, e in enumerate(q):
                if i >= len(q) or i > 64:
                    break
                if "file" in e:
                    e.file.ensure_time()
                if not e.url:
                    if not self.auds.stats.quiet:
                        self.auds.announce(ini_md(f"A problem occured while loading {sqr_md(e.name)}, and it has been automatically removed from the queue."))
                    dels.append(i)
                    continue
            q.pops(dels)
        if not q:
            if self.auds.next:
                self.auds.clear_next()
            if self.auds.source:
                self.auds.stop()
        elif not self.auds.paused:
            self.start_queue()

    # Advances queue when applicable, taking into account loop/repeat/shuffle settings.
    def advance(self, looped=True, repeated=True, shuffled=True):
        self.auds.source = self.auds.next
        self.auds.next = None
        q = self
        s = self.auds.stats
        if q:
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
        # If no queue entries found but there is a default playlist assigned, load a random entry from that
        if not q:
            if not self.playlist:
                t = self.bot.data.playlists.get(self.auds.guild.id, ())
                if t:
                    self.playlist = shuffle(t)
            if self.playlist:
                p = self.playlist.pop()
                e = cdict(p)
                e.u_id = self.bot.id
                e.skips = ()
                ytdl.get_stream(e)
                q.appendleft(e)
        elif self.auds.source:
            self.announce_play(self[0])
        self.update_load()

    def verify(self):
        if len(self) > self.maxitems + 2048:
            self.__init__(self[1 - self.maxitems:].appendleft(self[0]), fromarray=True)
        elif len(self) > self.maxitems:
            self.rotate(-1)
            while len(self) > self.maxitems:
                self.pop()
            self.rotate(1)
        return self

    # Enqueue items at target position, starting audio playback if queue was previously empty.
    def enqueue(self, items, position):
        with self.auds.semaphore:
            if len(items) > self.maxitems:
                items = items[:self.maxitems]
            if not self:
                self.auds.clear_source()
            if position == -1 or not self:
                self.extend(items)
            else:
                if position < 1:
                    self.auds.reset(start=False)
                elif position < 2:
                    self.auds.clear_next()
                self.rotate(-position)
                self.extend(items)
                self.rotate(len(items) + position)
            self.verify()
            create_future_ex(self.update_load, timeout=120)
            return self


# runs org2xm on a file, with an optional custom sample bank.
def org2xm(org, dat=None):
    if os.name != "nt":
        raise OSError("org2xm is only available on Windows.")
    if not org or type(org) is not bytes:
        if not is_url(org):
            raise TypeError("Invalid input URL.")
        org = verify_url(org)
        data = Request(org)
        if not data:
            raise FileNotFoundError("Error downloading file content.")
    else:
        if not org.startswith(b"Org-"):
            raise ValueError("Invalid file header.")
        data = org
    # Set compatibility option if file is not of org2 format.
    compat = not data.startswith(b"Org-02")
    ts = ts_us()
    # Write org data to file.
    r_org = "cache/" + str(ts) + ".org"
    with open(r_org, "wb") as f:
        f.write(data)
    r_dat = "cache/" + str(ts) + ".dat"
    orig = False
    # Load custom sample bank if specified
    if dat is not None and is_url(dat):
        dat = verify_url(dat)
        with open(r_dat, "wb") as f:
            dat = Request(dat)
            f.write(dat)
    else:
        if type(dat) is bytes and dat:
            with open(r_dat, "wb") as f:
                f.write(dat)
        else:
            r_dat = "misc/ORG210EN.DAT"
            orig = True
    args = ["misc/org2xm.exe", r_org, r_dat]
    if compat:
        args.append("c")
    subprocess.check_output(args)
    r_xm = f"cache/{ts}.xm"
    if not os.path.exists(r_xm):
        raise FileNotFoundError("Unable to locate converted file.")
    if not os.path.getsize(r_xm):
        raise RuntimeError("Converted file is empty.")
    for f in (r_org, r_dat)[:2 - orig]:
        with suppress():
            os.remove(f)
    return r_xm

def mid2mp3(mid):
    url = Request(
        "https://hostfast.onlineconverter.com/file/send",
        files={
            "class": (None, "audio"),
            "from": (None, "midi"),
            "to": (None, "mp3"),
            "source": (None, "file"),
            "file": mid,
            "audio_quality": (None, "192"),
        },
        method="post",
        decode=True,
    )
    fn = url.rsplit("/", 1)[-1].strip("\x00")
    for i in range(360):
        with delay(1):
            test = Request(f"https://hostfast.onlineconverter.com/file/{fn}")
            if test == b"d":
                break
    ts = ts_us()
    r_mp3 = f"cache/{ts}.mp3"
    with open(r_mp3, "wb") as f:
        f.write(Request(f"https://hostfast.onlineconverter.com/file/{fn}/download"))
    return r_mp3

def png2wav(png):
    ts = ts_us()
    r_png = f"cache/{ts}"
    r_wav = f"cache/{ts}.wav"
    args = ["py", f"-3.{sys.version_info[1]}", "png2wav.py", "../" + r_png, "../" + r_wav]
    with open(r_png, "wb") as f:
        f.write(png)
    print(args)
    subprocess.run(args, cwd="misc", stderr=subprocess.PIPE)
    # while True:
    #     if os.path.exists(r_wav) and os.path.getsize(r_wav) >= 96000:
    #         break
    #     if not proc.is_running():
    #         raise RuntimeError(as_str(proc.stderr.read()))
    return r_wav


CONVERTERS = {
    b"MThd": mid2mp3,
    b"Org-": org2xm,
}

def select_and_convert(stream):
    with requests.get(stream, headers=Request.header(), timeout=8, stream=True) as resp:
        it = resp.iter_content(4096)
        b = bytes()
        while len(b) < 4:
            b += next(it)
        try:
            convert = CONVERTERS[b[:4]]
        except KeyError:
            convert = png2wav
            # raise ValueError("Invalid file header.")
        b += resp.content
    return convert(b)


class AudioFileLink:

    seekable = True
    live = False
    dur = None

    def __init__(self, fn, stream=None, wasfile=None):
        self.fn = self.file = fn
        self.stream = stream
        self.streaming = concurrent.futures.Future()
        if stream:
            self.streaming.set_result(stream)
        self.readable = concurrent.futures.Future()
        if wasfile:
            self.readable.set_result(self)
        source = bot.audio.submit(f"!AudioFile({repr(fn)},{repr(stream)},{repr(wasfile)})")
        self.assign = deque()
        # print("Loaded source", source)

    def __getattr__(self, k):
        if k == "__await__":
            raise AttributeError(k)
        try:
            return object.__getattribute__(self, k)
        except AttributeError:
            pass
        if not bot.audio:
            raise AttributeError("Audio client not active.")
        return bot.audio.submit(f"cache['{self.fn}'].{k}")

    def load(self, stream=None, check_fmt=False, force=False, webpage_url=None, live=False, seekable=True, duration=None):
        if stream:
            self.stream = stream
        try:
            self.streaming.set_result(stream)
        except concurrent.futures.InvalidStateError:
            self.streaming = concurrent.futures.Future()
            self.streaming.set_result(stream)
        self.live = live
        self.seekable = seekable
        self.webpage_url = webpage_url
        bot.audio.submit(f"!cache['{self.fn}'].load(" + ",".join(repr(i) for i in (stream, check_fmt, force, webpage_url, live, seekable, duration)) + ")")
        if duration:
            self.dur = duration
        try:
            self.readable.set_result(self)
        except concurrent.futures.InvalidStateError:
            pass

    def create_reader(self, pos, auds=None):
        if auds is not None:
            if auds.paused or abs(auds.stats.speed) < 0.005:
                return
            ident = cdict(stats=auds.stats, args=[], guild_id=auds.guild.id)
            options=auds.construct_options(full=self.live)
        else:
            ident = None
            options = ()
        ts = ts_us()
        bot.audio.submit(f"!cache['{self.fn}'].create_reader({repr(pos)},{repr(ident)},{repr(options)},{ts})")
        return ts

    def duration(self):
        if not self.dur:
            self.dur = bot.audio.submit(f"cache['{self.fn}'].duration()")
        return self.dur

    def ensure_time(self):
        try:
            return bot.audio.submit(f"cache['{self.fn}'].ensure_time()")
        except KeyError:
            ytdl.cache.pop(self.fn, None)

    def update(self):
        # Newly loaded files have their duration estimates copied to all queue entries containing them
        if self.loaded:
            if not self.wasfile:
                dur = self.duration()
                if dur is not None:
                    for e in self.assign:
                        e["duration"] = dur
                    self.assign.clear()
        return bot.audio.submit(f"cache['{self.fn}'].update()")

    def destroy(self):
        bot.audio.submit(f"cache['{self.fn}'].destroy()")
        ytdl.cache.pop(self.fn, None)


class AudioClientSubInterface:

    afters = {}

    @classmethod
    def from_guild(cls, guild):
        if guild.me.voice:
            c_id = bot.audio.submit(f"getattr(getattr(client.get_guild({guild.id}).voice_client, 'channel', None), 'id', None)")
            if c_id:
                self = cls()
                self.guild = guild
                self.channel = bot.get_channel(c_id)
                bot.audio.clients[guild.id] = self
                return self

    @classmethod
    def after(cls, key):
        cls.afters[key]()

    @property
    def pos(self):
        return self.bot.audio.submit(f"AP.from_guild({self.guild.id}).pos")

    def __init__(self, channel=None, reconnect=True):
        self.bot = bot
        self.user = bot.user
        if channel:
            self.channel = channel
            self.guild = channel.guild
            bot.audio.submit(f"!await AP.join({channel.id})")
            bot.audio.clients[self.guild.id] = self

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: " + "{" + f'"guild": {self.guild}, "pos": {self.pos}' + "}"

    def __getattr__(self, k):
        if k in ("__await__"):
            raise AttributeError(k)
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        if not bot.audio:
            raise AttributeError("Audio client not active.")
        return bot.audio.submit(f"AP.from_guild({self.guild.id}).{k}")

    def enqueue(self, src, after=None):
        if src is None:
            return
        key = ts_us()
        if after:
            self.afters[key] = after
            return create_future_ex(bot.audio.submit, f"AP.from_guild({self.guild.id}).enqueue(players[{repr(src)}], after=lambda *args: submit('VOICE.ACSI.after({key})'))")
        return create_future_ex(bot.audio.submit, f"AP.from_guild({self.guild.id}).enqueue(players[{repr(src)}])")

    def play(self, src, after=None):
        if src is None:
            return
        key = ts_us()
        if after:
            self.afters[key] = after
            return bot.audio.submit(f"AP.from_guild({self.guild.id}).play(players[{repr(src)}], after=lambda *args: submit('VOICE.ACSI.after({key})'))")
        return bot.audio.submit(f"AP.from_guild({self.guild.id}).play(players[{repr(src)}])")

    def connect(self, reconnect=True, timeout=60):
        return create_future(bot.audio.submit, f"!await AP.from_guild({self.guild.id}).connect(reconnect={reconnect}, timeout={timeout})")

    async def disconnect(self, force=False):
        await create_future(bot.audio.submit, f"!await AP.from_guild({self.guild.id}).disconnect(force={force})")
        bot.audio.clients.pop(self.guild.id)
        self.channel = None

    async def move_to(self, channel=None):
        if not channel:
            return await self.disconnect(force=True)
        await create_future(bot.audio.submit, f"!await AP.from_guild({self.guild.id}).move_to(client.get_channel({channel.id}))")
        self.channel = channel

ACSI = AudioClientSubInterface

for attr in ("read", "skip", "stop", "pause", "resume", "clear_source", "clear_next", "clear", "kill", "is_connected", "is_paused", "is_playing"):
    setattr(ACSI, attr, eval("""lambda self: bot.audio.submit(f"AP.from_guild({self.guild.id}).""" + f"""{attr}()")"""))


# Manages all audio searching and downloading.
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
    youtube_x = 0
    youtube_dl_x = 0
    spotify_x = 0
    other_x = 0

    def __init__(self):
        self.bot = None
        self.lastclear = 0
        self.downloading = cdict()
        self.cache = cdict()
        self.searched = cdict()
        self.semaphore = Semaphore(4, 128)
        self.download_sem = Semaphore(8, 64, rate_limit=0.5)
        self.keepvid_failed = 0

    def __load__(self, **void):
        print("Initializing audio downloader keys...")
        fut = create_future_ex(self.update_dl)
        self.setup_pages()
        self.set_cookie()
        return fut.result()

    # Fetches youtube playlist page codes, split into pages of 50 items
    def setup_pages(self):
        with open("misc/page_tokens.txt", "r", encoding="utf-8") as f:
            s = f.read()
        page10 = s.splitlines()
        self.yt_pages = {i * 10: page10[i] for i in range(len(page10))}
        # self.yt_pages = [page10[i] for i in range(0, len(page10), 5)]
    
    def set_cookie(self):
        self.youtube_base = "CONSENT=YES+cb.20210328-17-p0.en+FX"
        resp = requests.get("https://www.youtube.com").text
        if "<title>Before you continue to YouTube</title>" in resp:
            resp = resp.split('<input type="hidden" name="v" value="', 1)[-1]
            resp = resp[:resp.index('">')].rsplit("+", 1)[0]
            self.youtube_base = f"CONSENT=YES+{resp}"
            self.youtube_x += 1

    def youtube_header(self):
        if self.youtube_base:
            return dict(cookie=self.youtube_base + "%03d" % random.randint(0, 999) + ";")
        return {}

    # Initializes youtube_dl object as well as spotify tokens, every 720 seconds.
    def update_dl(self):
        if utc() - self.lastclear > 720:
            self.lastclear = utc()
            with tracebacksuppressor:
                self.youtube_dl_x += 1
                self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
                self.spotify_x += 1
                token = await_fut(aretry(Request, "https://open.spotify.com/get_access_token", aio=True, attempts=8, delay=0.5))
                self.spotify_header = {"authorization": f"Bearer {json.loads(token[:512])['accessToken']}"}
                self.other_x += 1
                if self.keepvid_failed < 4:
                    try:
                        resp = Request("https://keepv.id", timeout=16)
                        search = b"<script>apikey='"
                        resp = resp[resp.rindex(search) + len(search):]
                        search = b";sid='"
                        resp = resp[resp.index(search) + len(search):]
                        self.keepvid_token = as_str(resp[:resp.index(b"';</script>")])
                        self.keepvid_failed = 0
                    except:
                        if not self.keepvid_failed:
                            print_exc()
                        self.keepvid_failed += 1

    # Gets data from yt-download.org, keepv.id, or y2mate.guru, adjusts the format to ensure compatibility with results from youtube-dl. Used as backup.
    def extract_backup(self, url, video=False):
        url = verify_url(url)
        if is_url(url) and not is_youtube_url(url):
            raise TypeError("Not a youtube link.")
        excs = alist()
        if ":" in url:
            url = url.rsplit("/", 1)[-1].split("v=", 1)[-1].split("&", 1)[0]
        webpage_url = f"https://www.youtube.com/watch?v={url}"
        resp = None
        try:
            yt_url = f"https://www.yt-download.org/file/mp3/{url}"
            if video:
                v_url = f"https://www.yt-download.org/file/mp4/{url}"
                self.other_x += 1
                fut = create_future_ex(Request, v_url, timeout=16)
            self.other_x += 1
            resp = Request(yt_url, timeout=16)
            search = b'<img class="h-20 w-20 md:h-48 md:w-48 mt-0 md:mt-12 lg:mt-0 rounded-full mx-auto md:mx-0 md:mr-6" src="'
            resp = resp[resp.index(search) + len(search):]
            thumbnail = as_str(resp[:resp.index(b'"')])
            search = b'<h2 class="text-lg text-teal-600 font-bold m-2 text-center">'
            resp = resp[resp.index(search) + len(search):]
            title = html_decode(as_str(resp[:resp.index(b"</h2>")]))
            resp = resp[resp.index(f'<a href="https://www.yt-download.org/download/{url}/mp3/192'.encode("utf-8")) + 9:]
            stream = as_str(resp[:resp.index(b'"')])
            resp = resp[:resp.index(b"</a>")]
            search = b'<div class="text-shadow-1">'
            fs = parse_fs(resp[resp.rindex(search) + len(search):resp.rindex(b"</div>")])
            dur = fs / 192000 * 8
            entry = {
                "formats": [
                    {
                        "abr": 192,
                        "url": stream,
                    },
                ],
                "duration": dur,
                "thumbnail": thumbnail,
                "title": title,
                "webpage_url": webpage_url,
            }
            if video:
                with tracebacksuppressor:
                    resp = fut.result()
                    while True:
                        try:
                            resp = resp[resp.index(f'<a href="https://www.yt-download.org/download/{url}/mp4/'.encode("utf-8")) + 9:]
                            stream = as_str(resp[:resp.index(b'"')])
                            search = b'<div class="text-shadow-1">'
                            height = int(resp[resp.rindex(search) + len(search):resp.rindex(b"</div>")].strip().rstrip("p"))
                        except ValueError:
                            break
                        else:
                            entry["formats"].append(dict(
                                abr=1,
                                url=stream,
                                height=height,
                            ))
            print("Successfully resolved with yt-download.")
            return entry
        except Exception as ex:
            if resp:
                try:
                    search = b'<h3 class="text-center text-xl">'
                    resp = resp[resp.index(search) + len(search):]
                    resp = resp[:resp.index(b"<")]
                except ValueError:
                    pass
                else:
                    excs.append(as_str(resp))
            excs.append(ex)
        try:
            self.other_x += 1
            resp = Request(
                "https://keepv.id",
                headers={"Accept": "*/*", "Cookie": "PHPSESSID=" + self.keepvid_token, "X-Requested-With": "XMLHttpRequest"},
                data=(("url", webpage_url), ("sid", self.keepvid_token)),
                method="POST",
                timeout=16,
            )
            search = b'<h2 class="mb-3">'
            resp = resp[resp.index(search) + len(search):]
            title = html_decode(as_str(resp[:resp.index(b"</h3>")]))
            search = b'<img src="'
            resp = resp[resp.index(search) + len(search):]
            thumbnail = as_str(resp[:resp.index(b'"')])
            entry = {
                "formats": [],
                "thumbnail": thumbnail,
                "title": title,
                "webpage_url": webpage_url,
            }
            with suppress(ValueError):
                search = b"Download Video</a><br>"
                resp = resp[resp.index(search) + len(search):]
                search = b"Duration: "
                resp = resp[resp.index(search) + len(search):]
                entry["duration"] = dur = time_parse(as_str(resp[:resp.index(b"<br><br>")]))
            resp = resp[resp.index(b"<body>") + 6:]
            while resp:
                try:
                    search = b"<tr><td>"
                    resp = resp[resp.index(search) + len(search):]
                except ValueError:
                    break
                fmt = as_str(resp[:resp.index(b"</td>")])
                if not fmt:
                    form = {}
                elif "x" in fmt:
                    form = dict(zip(("width", "height"), (int(i) for i in fmt.split("x", 1))))
                else:
                    form = dict(abr=int(fmt.casefold().rstrip("kmgbps")))
                search = b' shadow vdlbtn" href="'
                resp = resp[resp.index(search) + len(search):]
                stream = resp[:resp.index(b'"')]
                form["url"] = stream
                entry["formats"].append(form)
            if not entry["formats"]:
                raise FileNotFoundError
            print("Successfully resolved with keepv.id.")
            return entry
        except Exception as ex:
            if resp:
                if b"our system was not able to detect any video at the adress you provided." in resp:
                    excs.append("our system was not able to detect any video at the adress you provided.")
                else:
                    excs.append(resp)
            excs.append(ex)
        try:
            resp = None
            self.other_x += 1
            resp = data = Request("https://y2mate.guru/api/convert", data={"url": webpage_url}, method="POST", json=True, timeout=16)
            meta = data["meta"]
            entry = {
                "formats": [
                    {
                        "height": stream.get("height", 0),
                        "abr": stream.get("quality", 0),
                        "url": stream["url"],
                    } for stream in data["url"] if "url" in stream and stream.get("audio")
                ],
                "thumbnail": data.get("thumb"),
                "title": meta["title"],
                "webpage_url": meta["source"],
            }
            if meta.get("duration"):
                entry["duration"] = time_parse(meta["duration"])
            if not entry["formats"]:
                raise FileNotFoundError
            print("Successfully resolved with y2mate.")
            return entry
        except Exception as ex:
            if resp:
                excs.append(resp)
            excs.append(ex)
            print("\n\n".join(as_str(e) for e in excs))
            raise

    # Returns part of a spotify playlist.
    def get_spotify_part(self, url):
        out = deque()
        self.spotify_x += 1
        d = Request(url, headers=self.spotify_header, json=True)
        with suppress(KeyError):
            d = d["tracks"]
        try:
            items = d["items"]
            total = d.get("total", 0)
        except KeyError:
            if "type" in d:
                items = (d,)
                total = 1
            else:
                items = []
                total = 0
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
            temp = cdict(
                name=name,
                url="ytsearch:" + "".join(c if c.isascii() and c != ":" else "_" for c in f"{name} ~ {artists}"),
                id=track["id"],
                duration=dur,
                research=True,
            )
            out.append(temp)
        return out, total

    # Returns part of a youtube playlist.
    def get_youtube_part(self, url):
        out = deque()
        self.youtube_x += 1
        d = Request(url, json=True)
        items = d["items"]
        total = d.get("pageInfo", {}).get("totalResults", 0)
        for item in items:
            try:
                snip = item["snippet"]
                v_id = snip["resourceId"]["videoId"]
            except KeyError:
                continue
            name = snip.get("title", v_id)
            url = f"https://www.youtube.com/watch?v={v_id}"
            temp = cdict(
                name=name,
                url=url,
                duration=None,
                research=True,
            )
            out.append(temp)
        return out, total

    # Returns a full youtube playlist.
    def get_youtube_playlist(self, p_id):
        out = deque()
        self.youtube_x += 1
        resp = Request(f"https://www.youtube.com/playlist?list={p_id}", headers=self.youtube_header())
        try:
            search = b'window["ytInitialData"] = '
            try:
                resp = resp[resp.index(search) + len(search):]
            except ValueError:
                search = b"var ytInitialData = "
                resp = resp[resp.index(search) + len(search):]
            try:
                resp = resp[:resp.index(b'window["ytInitialPlayerResponse"] = null;')]
                resp = resp[:resp.rindex(b";")]
            except ValueError:
                try:
                    resp = resp[:resp.index(b";</script><title>")]
                except:
                    resp = resp[:resp.index(b';</script><link rel="')]
            data = eval_json(resp)
        except:
            print(resp)
            raise
        count = int(data["sidebar"]["playlistSidebarRenderer"]["items"][0]["playlistSidebarPrimaryInfoRenderer"]["stats"][0]["runs"][0]["text"].replace(",", ""))
        for part in data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"]["contents"]:
            try:
                video = part["playlistVideoRenderer"]
            except KeyError:
                if "continuationItemRenderer" not in part:
                    print(part)
                continue
            v_id = video['videoId']
            try:
                dur = round_min(float(video["lengthSeconds"]))
            except (KeyError, ValueError):
                try:
                    dur = time_parse(video["lengthText"]["simpleText"])
                except KeyError:
                    dur = None
            temp = cdict(
                name=video["title"]["runs"][0]["text"],
                url=f"https://www.youtube.com/watch?v={v_id}",
                duration=dur,
                thumbnail=f"https://i.ytimg.com/vi/{v_id}/maxresdefault.jpg",
            )
            out.append(temp)
        if count > 100:
            url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&key={google_api_key}&playlistId={p_id}"
            page = 50
            futs = deque()
            for curr in range(100, page * ceil(count / page), page):
                search = f"{url}&pageToken={self.yt_pages[curr]}"
                futs.append(create_future_ex(self.get_youtube_part, search))
            for fut in futs:
                out.extend(fut.result()[0])
        return out

    def ydl_errors(self, s):
        return "this video has been removed" not in s and "private video" not in s and "has been terminated" not in s

    blocked_yt = False

    # Repeatedly makes calls to youtube-dl until there is no more data to be collected.
    def extract_true(self, url):
        while not is_url(url):
            with suppress(NotImplementedError):
                return self.search_yt(regexp("ytsearch[0-9]*:").sub("", url, 1))[0]
            resp = self.extract_from(url)
            if "entries" in resp:
                resp = next(iter(resp["entries"]))
            if "duration" in resp and "formats" in resp:
                out = cdict(
                    name=resp["title"],
                    url=resp["webpage_url"],
                    duration=resp["duration"],
                    stream=get_best_audio(resp),
                    icon=get_best_icon(resp),
                    video=get_best_video(resp),
                )
                stream = out.stream
                if "googlevideo" in stream[:64]:
                    durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
                    if durstr:
                        out.duration = round_min(durstr[0])
                return out
            try:
                url = resp["webpage_url"]
            except KeyError:
                try:
                    url = resp["url"]
                except KeyError:
                    url = resp["id"]
        if is_discord_url(url):
            title = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if title.rsplit(".", 1)[-1] in ("ogg", "webm", "mp4", "avi", "mov"):
                url2 = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
                with requests.get(url2, headers=Request.header(), stream=True) as resp:
                    if resp.status_code in range(200, 400):
                        url = url2
            if "." in title:
                title = title[:title.rindex(".")]
            return dict(url=url, name=title, direct=True)
        if self.bot.is_webserver_url(url):
            spl = url[8:].split("/")
            if spl[1] in ("preview", "view", "file", "files", "download"):
                path = spl[2]
                orig_path = path
                ind = "\x7f"
                if path.startswith("~"):
                    b = path.split(".", 1)[0].encode("utf-8") + b"=="
                    if (len(b) - 1) & 3 == 0:
                        b += b"="
                    path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
                elif path.startswith("!"):
                    ind = "!"
                    path = path[1:]
                p = find_file(path, ind=ind)
                fn = urllib.parse.unquote(p.rsplit("/", 1)[-1].split("~", 1)[-1].rsplit(".", 1)[0])
                url = self.bot.raw_webserver + "/files/" + orig_path
                return dict(url=url, name=fn, direct=True)
        try:
            if self.blocked_yt > utc():
                raise PermissionError
            self.youtube_dl_x += 1
            entries = self.downloader.extract_info(url, download=False, process=True)
        except Exception as ex:
            s = str(ex).casefold()
            if type(ex) is not youtube_dl.DownloadError or self.ydl_errors(s):
                if "429" in s:
                    self.blocked_yt = utc() + 3600
                try:
                    entries = self.extract_backup(url)
                except (TypeError, youtube_dl.DownloadError):
                    raise FileNotFoundError("Unable to fetch audio data.")
            else:
                raise
        if "entries" in entries:
            entries = entries["entries"]
        else:
            entries = [entries]
        out = deque()
        for entry in entries:
            temp = cdict(
                name=entry["title"],
                url=entry["webpage_url"],
                duration=entry.get("duration"),
                stream=get_best_audio(entry),
                icon=get_best_icon(entry),
                video=get_best_video(entry),
            )
            stream = temp.stream
            if "googlevideo" in stream[:64]:
                durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
                if durstr:
                    temp.duration = round_min(durstr[0])
            if not temp.duration:
                temp.research = True
            out.append(temp)
        return out

    # Extracts audio information from a single URL.
    def extract_from(self, url):
        if is_discord_url(url):
            title = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if title.rsplit(".", 1)[-1] in ("ogg", "webm", "mp4", "avi", "mov"):
                url2 = url.replace("/cdn.discordapp.com/", "/media.discordapp.net/")
                with requests.get(url2, headers=Request.header(), stream=True) as resp:
                    if resp.status_code in range(200, 400):
                        url = url2
            if "." in title:
                title = title[:title.rindex(".")]
            return dict(url=url, webpage_url=url, title=title, direct=True)
        if self.bot.is_webserver_url(url):
            spl = url[8:].split("/")
            if spl[1] in ("preview", "view", "file", "files", "download"):
                url2 = url
                path = spl[2]
                orig_path = path
                ind = "\x7f"
                if path.startswith("~"):
                    b = path.split(".", 1)[0].encode("utf-8") + b"=="
                    if (len(b) - 1) & 3 == 0:
                        b += b"="
                    path = str(int.from_bytes(base64.urlsafe_b64decode(b), "big"))
                elif path.startswith("!"):
                    ind = "!"
                    path = path[1:]
                p = find_file(path, ind=ind)
                fn = urllib.parse.unquote(p.rsplit("/", 1)[-1].split("~", 1)[-1].rsplit(".", 1)[0])
                url = self.bot.raw_webserver + "/files/" + orig_path
                return dict(url=url, webpage_url=url2, title=fn, direct=True)
        try:
            if self.blocked_yt > utc():
                raise PermissionError
            self.youtube_dl_x += 1
            return self.downloader.extract_info(url, download=False, process=False)
        except Exception as ex:
            s = str(ex).casefold()
            if type(ex) is not youtube_dl.DownloadError or self.ydl_errors(s):
                if "429" in s:
                    self.blocked_yt = utc() + 3600
                if is_url(url):
                    try:
                        return self.extract_backup(url)
                    except (TypeError, youtube_dl.DownloadError):
                        raise FileNotFoundError("Unable to fetch audio data.")
            raise

    # Extracts info from a URL or search, adjusting accordingly.
    def extract_info(self, item, count=1, search=False, mode=None):
        if (mode or search) and not item.startswith("ytsearch:") and not is_url(item):
            if count == 1:
                c = ""
            else:
                c = count
            item = item.replace(":", "-")
            if mode:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"{mode}search{c}:{item}", download=False, process=False)
            exc = ""
            try:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"ytsearch{c}:{item}", download=False, process=False)
            except Exception as ex:
                exc = repr(ex)
            try:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"scsearch{c}:{item}", download=False, process=False)
            except Exception as ex:
                raise ConnectionError(exc + repr(ex))
        if is_url(item) or not search:
            return self.extract_from(item)
        self.youtube_dl_x += 1
        return self.downloader.extract_info(item, download=False, process=False)

    # Main extract function, able to extract from youtube playlists much faster than youtube-dl using youtube API, as well as ability to follow spotify links.
    def extract(self, item, force=False, count=1, mode=None, search=True):
        try:
            page = None
            output = deque()
            if google_api_key and ("youtube.com" in item or "youtu.be/" in item):
                p_id = None
                for x in ("?list=", "&list="):
                    if x in item:
                        p_id = item[item.index(x) + len(x):]
                        p_id = p_id.split("&", 1)[0]
                        break
                if p_id:
                    with tracebacksuppressor:
                        output.extend(self.get_youtube_playlist(p_id))
                        # Scroll to highlighted entry if possible
                        v_id = None
                        for x in ("?v=", "&v="):
                            if x in item:
                                v_id = item[item.index(x) + len(x):]
                                v_id = v_id.split("&", 1)[0]
                                break
                        if v_id:
                            for i, e in enumerate(output):
                                if v_id in e.url:
                                    output.rotate(-i)
                                    break
                        return output
            elif regexp("(play|open|api)\\.spotify\\.com").search(item):
                # Spotify playlist searches contain up to 100 items each
                if "playlist" in item:
                    url = item[item.index("playlist"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/playlists/{key}/tracks?type=track,episode"
                    page = 100
                # Spotify album searches contain up to 50 items each
                elif "album" in item:
                    url = item[item.index("album"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/albums/{key}/tracks?type=track,episode"
                    page = 50
                # Single track links also supported
                elif "track" in item:
                    url = item[item.index("track"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/tracks/{key}"
                    page = 1
                else:
                    raise TypeError("Unsupported Spotify URL.")
                if page == 1:
                    output.extend(self.get_spotify_part(url)[0])
                else:
                    futs = deque()
                    maxitems = 10000
                    # Optimized searching with lookaheads
                    for i, curr in enumerate(range(0, maxitems, page)):
                        with delay(0.03125):
                            if curr >= maxitems:
                                break
                            search = f"{url}&offset={curr}&limit={page}"
                            fut = create_future_ex(self.get_spotify_part, search, timeout=90)
                            print("Sent 1 spotify search.")
                            futs.append(fut)
                            if not (i < 1 or math.log2(i + 1) % 1) or not i & 7:
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
                    while futs:
                        output.extend(futs.popleft().result()[0])
                    # Scroll to highlighted entry if possible
                    v_id = None
                    for x in ("?highlight=spotify:track:", "&highlight=spotify:track:"):
                        if x in item:
                            v_id = item[item.index(x) + len(x):]
                            v_id = v_id.split("&", 1)[0]
                            break
                    if v_id:
                        for i, e in enumerate(output):
                            if v_id == e.get("id"):
                                output.rotate(-i)
                                break
            # Only proceed if no items have already been found (from playlists in this case)
            if not len(output):
                # Allow loading of files output by ~dump
                if is_url(item):
                    url = verify_url(item)
                    if url[-5:] == ".json" or url[-4:] in (".txt", ".bin", ".zip"):
                        s = await_fut(self.bot.get_request(url))
                        d = select_and_loads(s, size=268435456)
                        q = d["queue"][:262144]
                        return [cdict(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
                elif mode in (None, "yt"):
                    with suppress(NotImplementedError):
                        res = self.search_yt(item)
                        if res:
                            return res[:count]
                # Otherwise call automatic extract_info function
                resp = self.extract_info(item, count, search=search, mode=mode)
                if resp.get("_type", None) == "url":
                    resp = self.extract_from(resp["url"])
                if resp is None or not len(resp):
                    raise LookupError(f"No results for {item}")
                # Check if result is a playlist
                if resp.get("_type", None) == "playlist":
                    entries = list(resp["entries"])
                    if force or len(entries) <= 1:
                        for entry in entries:
                            # Extract full data if playlist only contains 1 item
                            try:
                                data = self.extract_from(entry["url"])
                            except KeyError:
                                temp = cdict({
                                    "name": resp["title"],
                                    "url": resp["webpage_url"],
                                    "duration": inf,
                                    "stream": get_best_audio(entry),
                                    "icon": get_best_icon(entry),
                                    "video": get_best_video(entry),
                                })
                            else:
                                temp = cdict({
                                    "name": data["title"],
                                    "url": data["webpage_url"],
                                    "duration": round_min(data["duration"]),
                                    "stream": get_best_audio(resp),
                                    "icon": get_best_icon(resp),
                                    "video": get_best_video(resp),
                                })
                            stream = temp.stream
                            if "googlevideo" in stream[:64]:
                                durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
                                if durstr:
                                    temp.duration = round_min(durstr[0])
                            output.append(temp)
                    else:
                        for i, entry in enumerate(entries):
                            if not i:
                                # Extract full data from first item only
                                temp = self.extract(entry["url"], search=False)[0]
                            else:
                                # Get as much data as possible from all other items, set "research" flag to have bot lazily extract more info in background
                                with tracebacksuppressor:
                                    found = True
                                    if "title" in entry:
                                        title = entry["title"]
                                    else:
                                        title = entry["url"].rsplit("/", 1)[-1]
                                        if "." in title:
                                            title = title[:title.rindex(".")]
                                        found = False
                                    if entry.get("duration") is not None:
                                        dur = float(entry["duration"])
                                    else:
                                        dur = None
                                    url = entry.get("webpage_url", entry.get("url", entry.get("id")))
                                    if not url:
                                        continue
                                    temp = {
                                        "name": title,
                                        "url": url,
                                        "duration": dur,
                                    }
                                    if not is_url(url):
                                        if entry.get("ie_key", "").casefold() == "youtube":
                                            temp["url"] = f"https://www.youtube.com/watch?v={url}"
                                    temp["research"] = True
                            output.append(cdict(temp))
                else:
                    # Single item results must contain full data, we take advantage of that here
                    found = "duration" in resp
                    if found:
                        dur = resp["duration"]
                    else:
                        dur = None
                    temp = cdict({
                        "name": resp["title"],
                        "url": resp["webpage_url"],
                        "duration": dur,
                        "stream": get_best_audio(resp),
                        "icon": get_best_icon(resp),
                        "video": get_best_video(resp),
                    })
                    stream = temp.stream
                    if "googlevideo" in stream[:64]:
                        durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
                        if durstr:
                            temp.duration = round_min(durstr[0])
                    output.append(temp)
            return output
        except:
            if force != "spotify":
                raise
            print_exc()
            return 0

    def item_yt(self, item):
        video = next(iter(item.values()))
        if "videoId" not in video:
            return
        try:
            dur = time_parse(video["lengthText"]["simpleText"])
        except KeyError:
            dur = None
        try:
            title = video["title"]["runs"][0]["text"]
        except KeyError:
            title = video["title"]["simpleText"]
        try:
            tn = video["thumbnail"]
        except KeyError:
            thumbnail = None
        else:
            if type(tn) is dict:
                thumbnail = sorted(tn["thumbnails"], key=lambda t: t.get("width", 0) * t.get("height", 0))[-1]["url"]
            else:
                thumbnail = tn
        try:
            views = int(video["viewCountText"]["simpleText"].replace(",", "").replace("views", "").replace(" ", ""))
        except (KeyError, ValueError):
            views = 0
        return cdict(
            name=video["title"]["runs"][0]["text"],
            url=f"https://www.youtube.com/watch?v={video['videoId']}",
            duration=dur,
            icon=thumbnail,
            views=views,
        )

    def parse_yt(self, s):
        data = eval_json(s)
        results = alist()
        try:
            pages = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
        except KeyError:
            pages = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
        for page in pages:
            try:
                items = next(iter(page.values()))["contents"]
            except KeyError:
                continue
            for item in items:
                if "promoted" not in next(iter(item)).casefold():
                    entry = self.item_yt(item)
                    if entry is not None:
                        results.append(entry)
        return sorted(results, key=lambda entry: entry.views, reverse=True)

    def search_yt(self, query):
        out = None
        with tracebacksuppressor:
            resp = self.extract_info(query)
            if resp.get("_type", None) == "url":
                resp = self.extract_from(resp["url"])
            if resp.get("_type", None) == "playlist":
                entries = list(resp["entries"])
            else:
                entries = [resp]
            out = alist()
            for entry in entries:
                with tracebacksuppressor:
                    found = True
                    if "title" in entry:
                        title = entry["title"]
                    else:
                        title = entry["url"].rsplit("/", 1)[-1]
                        if "." in title:
                            title = title[:title.rindex(".")]
                        found = False
                    if "duration" in entry:
                        dur = float(entry["duration"])
                    else:
                        dur = None
                    url = entry.get("webpage_url", entry.get("url", entry.get("id")))
                    if not url:
                        continue
                    temp = cdict(name=title, url=url, duration=dur)
                    if not is_url(url):
                        if entry.get("ie_key", "").casefold() == "youtube":
                            temp["url"] = f"https://www.youtube.com/watch?v={url}"
                    temp["research"] = True
                    out.append(temp)
        if not out:
            url = f"https://www.youtube.com/results?search_query={verify_url(query)}"
            self.youtube_x += 1
            resp = Request(url, headers=self.youtube_header(), timeout=12)
            result = None
            with suppress(ValueError):
                s = resp[resp.index(b"// scraper_data_begin") + 21:resp.rindex(b"// scraper_data_end")]
                s = s[s.index(b"var ytInitialData = ") + 20:s.rindex(b";")]
                result = self.parse_yt(s)
            with suppress(ValueError):
                s = resp[resp.index(b'window["ytInitialData"] = ') + 26:]
                s = s[:s.index(b'window["ytInitialPlayerResponse"] = null;')]
                s = s[:s.rindex(b";")]
                result = self.parse_yt(s)
            if result is not None:
                q = to_alphanumeric(full_prune(query))
                high = alist()
                low = alist()
                for entry in result:
                    if entry.duration:
                        name = full_prune(entry.name)
                        aname = to_alphanumeric(name)
                        spl = aname.split()
                        if entry.duration < 960 or "extended" in q or "hour" in q or "extended" not in spl and "hour" not in spl and "hours" not in spl:
                            if fuzzy_substring(aname, q, match_length=False) >= 0.5:
                                high.append(entry)
                                continue
                    low.append(entry)

                def key(entry):
                    coeff = fuzzy_substring(to_alphanumeric(full_prune(entry.name)), q, match_length=False)
                    if coeff < 0.5:
                        coeff = 0
                    return coeff

                out = sorted(high, key=key, reverse=True)
                out.extend(sorted(low, key=key, reverse=True))
            if not out and len(query) < 16:
                self.failed_yt = utc() + 180
                print(query)
        return out

    # Performs a search, storing and using cached search results for efficiency.
    def search(self, item, force=False, mode=None, count=1):
        item = verify_search(item)
        if mode is None and count == 1 and item in self.searched:
            if utc() - self.searched[item].t < 18000:
                return self.searched[item].data
            else:
                self.searched.pop(item)
        while len(self.searched) > 262144:
            self.searched.pop(next(iter(self.searched)))
        with self.semaphore:
            try:
                obj = cdict(t=utc())
                obj.data = output = self.extract(item, force, mode=mode, count=count)
                if obj.data:
                    self.searched[item] = obj
                return output
            except Exception as ex:
                print_exc()
                return repr(ex)

    # Gets the stream URL of a queue entry, starting download when applicable.
    def get_stream(self, entry, video=False, force=False, download=True, callback=None):
        if not entry.get("url"):
            raise FileNotFoundError
        try:
            entry.update(self.searched[entry["url"]])
        except KeyError:
            pass
        if video:
            stream = entry.get("video", None)
        else:
            stream = entry.get("stream", None)
        icon = entry.get("icon", None)
        # Use SHA-256 hash of URL to avoid filename conflicts
        h = shash(entry["url"])
        if type(download) is str:
            fn = "~" + h + download
        else:
            fn = "~" + h + ".opus"
        # Use cached file if one already exists
        if self.cache.get(fn) or not download:
            if video:
                entry["video"] = stream
            else:
                entry["stream"] = stream
            entry["icon"] = icon
            # Files may have a callback set for when they are loaded
            if callback is not None:
                create_future_ex(callback)
            f = self.cache.get(fn)
            if f is not None:
                entry["file"] = f
                # Assign file duration estimate to queue entry
                # This could be done better, this current implementation is technically not thread-safe
                if f.loaded:
                    entry["duration"] = f.duration()
                else:
                    f.assign.append(entry)
                # Touch file to indicate usage
                f.ensure_time()
                f.readable.result(timeout=16)
            if f or (not force and not download):
                return f
        # "none" indicates stream is currently loading
        if stream == "none" and not force:
            return
        entry["stream"] = "none"
        searched = False
        # If "research" tag is set, entry does not contain full data and requires another search
        if "research" in entry:
            try:
                self.extract_single(entry)
                searched = True
                entry.pop("research", None)
            except:
                print_exc()
                entry.pop("research", None)
                entry["url"] = ""
                raise
            else:
                stream = entry.get("stream", None)
                icon = entry.get("icon", None)
        # If stream is still not found or is a soundcloud audio fragment playlist file, perform secondary youtube-dl search
        if stream in (None, "none"):
            data = self.search(entry["url"])
            if type(data) is str:
                try:
                    data = evalEX(data)
                except:
                    entry["url"] = ""
                    raise
            stream = set_dict(data[0], "stream", data[0].url)
            icon = set_dict(data[0], "icon", data[0].url)
            entry.update(data[0])
        elif not searched and (stream.startswith("ytsearch:") or stream.startswith("https://cf-hls-media.sndcdn.com/") or expired(stream)):
            data = self.extract(entry["url"])
            stream = set_dict(data[0], "stream", data[0].url)
            icon = set_dict(data[0], "icon", data[0].url)
            entry.update(data[0])
        # Otherwise attempt to start file download
        try:
            if stream.startswith("ytsearch:") or stream in (None, "none"):
                self.extract_single(entry, force=True)
                stream = entry.get("stream")
                if stream in (None, "none"):
                    raise FileNotFoundError("Unable to locate appropriate file stream.")
            entry["stream"] = stream
            entry["icon"] = icon
            if "googlevideo" in stream[:64]:
                durstr = regexp("[&?]dur=([0-9\\.]+)").findall(stream)
                if durstr:
                    entry["duration"] = round_min(durstr[0])
            if not entry.get("duration"):
                entry["duration"] = get_duration(stream)
            # print(entry.url, entry.duration)
            with suppress(KeyError):
                self.searched[entry["url"]]["duration"] = entry["duration"]
            if not download:
                return entry
            self.cache[fn] = f = AudioFileLink(fn)
            if type(download) is str:
                live = False
            else:
                live = not entry.get("duration") or entry["duration"] > 960
            seekable = not entry.get("duration") or entry["duration"] < inf
            cf = isnan(entry.get("duration") or nan) or not (stream.startswith("https://cf-hls-media.sndcdn.com/") or is_youtube_stream(stream))
            try:
                f.load(stream, check_fmt=cf, webpage_url=entry["url"], live=live, seekable=seekable, duration=entry.get("duration"))
            except:
                self.cache.pop(fn, None)
                raise
            # Assign file duration estimate to queue entry
            f.assign.append(entry)
            entry["file"] = f
            f.ensure_time()
            # Files may have a callback set for when they are loaded
            if callback is not None:
                create_future(callback)
            return f
        except:
            # Remove entry URL if loading failed
            print_exc()
            entry["url"] = ""

    emptybuff = b"\x00" * (48000 * 2 * 2)
    # codec_map = {}
    # For ~download
    def download_file(self, url, fmt, start=None, end=None, auds=None, ts=None, copy=False, ar=SAMPLE_RATE, ac=2, container=None, child=False, silenceremove=False):
        if child:
            ctx = emptyctx
        else:
            ctx = self.download_sem
        with ctx:
            # Select a filename based on current time to avoid conflicts
            if fmt[:3] == "mid":
                mid = True
                fmt = "mp3"
            else:
                mid = False
            videos = ("webm", "mkv", "f4v", "flv", "mov", "qt", "wmv", "mp4", "m4v", "mpv", "gif")
            if type(url) is str:
                urls = (url,)
            else:
                urls = url
            vst = deque()
            ast = deque()
            if not ts:
                ts = ts_us()
            outf = None
            for url in urls:
                if len(ast) > 1 and not vst:
                    ast.append(url)
                    continue
                try:
                    res = self.search(url)
                    if type(res) is str:
                        raise evalex(res)
                    info = res[0]
                except:
                    print(url)
                    print_exc()
                    continue
                self.get_stream(info, video=fmt in videos, force=True, download=False)
                if not outf:
                    outf = f"{info['name']}.{fmt}"
                    outft = outf.translate(filetrans)
                    if child:
                        fn = f"cache/C{ts}~{outft}"
                    else:
                        fn = f"cache/\x7f{ts}~{outft}"
                if vst or fmt in videos:
                    vst.append(info["video"])
                ast.append(info)
            if not ast and not vst:
                raise LookupError(f"No stream URLs found for {url}")
            ffmpeg = "ffmpeg"
            if len(ast) <= 1:
                if ast and not is_youtube_stream(ast[0]["stream"]):
                    ffmpeg = "misc/ffmpeg-c/ffmpeg.exe"
            args = alist((ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets", "-y"))
            if vst:
                if len(vst) > 1:
                    codec_map = {}
                    codecs = {}
                    for url in vst:
                        try:
                            codec = codec_map[url]
                        except KeyError:
                            codec = as_str(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "default=nokey=1:noprint_wrappers=1", url])).strip()
                            print(codec)
                            codec_map[url] = codec
                        add_dict(codecs, {codec: 1})
                    if len(codecs) > 1:
                        maxcodec = max(codecs.values())
                        selcodec = [k for k, v in codecs.items() if v >= maxcodec][0]
                        t = ts
                        for i, url in enumerate(vst):
                            if codec_map[url] != selcodec:
                                t += 1
                                vst[i] = self.download_file(url, selcodec, auds=auds, ts=t, silenceremove=silenceremove)[0].rsplit("/", 1)[-1]
                    vsc = "\n".join(f"file '{i}'" for i in vst)
                    vsf = f"cache/{ts}~video.concat"
                    with open(vsf, "w", encoding="utf-8") as f:
                        f.write(vsc)
                else:
                    vsf = vsc = vst[0]
            if len(ast) > 1:
                args.extend(("-f", "s16le", "-ar", "48k", "-ac", "2"))
                asf = "-"
            else:
                asf = asc = ast[0]["stream"]
            if not vst:
                args.append("-vn")
            elif fmt == "gif":
                args.append("-an")
            if str(start) != "None":
                start = round_min(float(start))
                args.extend(("-ss", str(start)))
            else:
                start = 0
            if str(end) != "None":
                end = round_min(min(float(end), 86400))
                args.extend(("-to", str(end)))
            else:
                end = None
                if len(ast) == 1:
                    args.extend(("-to", "86400"))
            if vst and vsf != asf:
                args.extend(("-i", vsf))
                if start:
                    args.extend(("-ss", str(start)))
                if end is not None:
                    args.extend(("-to", str(end)))
            args.extend(("-i", asf, "-map_metadata", "-1"))
            if auds:
                args.extend(auds.construct_options(full=True))
            if silenceremove and len(ast) == 1:
                args.extend(("-af", "silenceremove=start_periods=1:start_duration=0.015625:start_threshold=-50dB:start_silence=0.015625:stop_periods=-9000:stop_threshold=-50dB:window=0.015625"))
            br = 196608
            if auds and br > auds.stats.bitrate:
                br = max(4096, auds.stats.bitrate)
            sr = str(SAMPLE_RATE)
            ac = "2"
            if fmt in ("vox", "adpcm"):
                args.extend(("-acodec", "adpcm_ms"))
                fmt = "wav" if fmt == "adpcm" else "vox"
                outf = f"{info['name']}.{fmt}"
                fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
            elif fmt == "pcm":
                fmt = "s16le"
            elif fmt == "mp2":
                br = round(br / 64000) * 64000
                if not br:
                    br = 64000
            elif fmt == "aac":
                fmt = "adts"
            elif fmt == "8bit":
                container = "wav"
                fmt = "pcm_u8"
                sr = "24k"
                ac = "1"
                br = "256"
                outf = f"{info['name']}.wav"
                fn = f"cache/\x7f{ts}~" + outf.translate(filetrans)
            if ast:
                args.extend(("-ar", sr, "-ac", ac, "-b:a", str(br)))
            if copy:
                args.extend(("-c", "copy", fn))
            elif container:
                args.extend(("-f", container, "-c", fmt, "-strict", "-2", fn))
            else:
                args.extend(("-f", fmt, fn))
            print(args)
            try:
                if len(ast) > 1:
                    proc = psutil.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    for t, info in enumerate(ast, ts + 1):
                        cfn = None
                        fut = create_future_ex(proc.stdin.write, self.emptybuff)
                        if type(info) is not str:
                            url = info.get("url")
                        else:
                            url = info
                        try:
                            cfn = self.download_file(url, "pcm", auds=None, ts=t, child=True, silenceremove=silenceremove)[0]
                        except:
                            print_exc()
                        fut.result()
                        if cfn and os.path.exists(cfn):
                            if os.path.getsize(cfn):
                                with open(cfn, "rb") as f:
                                    while True:
                                        b = f.read(1048576)
                                        if not b:
                                            break
                                        proc.stdin.write(b)
                            create_future_ex(os.remove, cfn)
                    proc.stdin.close()
                    proc.wait()
                else:
                    resp = subprocess.run(args, stderr=subprocess.PIPE)
                    resp.check_returncode()
            except subprocess.CalledProcessError as ex:
                # Attempt to convert file from org if FFmpeg failed
                try:
                    url = ast[0]
                    if type(url) is not str:
                        url = url["url"]
                    new = select_and_convert(url)
                except ValueError:
                    if resp.stderr:
                        raise RuntimeError(*ex.args, resp.stderr)
                    raise ex
                # Re-estimate duration if file was successfully converted from org
                args[args.index("-i") + 1] = new
                try:
                    resp = subprocess.run(args, stderr=subprocess.PIPE)
                    resp.check_returncode()
                except subprocess.CalledProcessError as ex:
                    if resp.stderr:
                        raise RuntimeError(*ex.args, resp.stderr)
                    raise ex
                if not is_url(new):
                    with suppress():
                        os.remove(new)
            # with tracebacksuppressor:
            #     if len(ast) > 1:
            #         os.remove(asf)
            #     if len(vst) > 1:
            #         os.remove(vsf)
            if end:
                odur = end - start
                if odur:
                    dur = e_dur(get_duration(fn))
                    if dur < odur - 1:
                        ts += 1
                        fn, fn2 = f"cache/\x7f{ts}~{outft}", fn
                        times = ceil(odur / dur)
                        loopf = f"cache/{ts - 1}~loop.txt"
                        with open(loopf, "w", encoding="utf-8") as f:
                            f.write(f"file '{fn2.split('/', 1)[-1]}'\n" * times)
                        args = [
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "auto",
                            "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets",
                            "-protocol_whitelist", "concat,tls,tcp,file,http,https",
                            "-to", str(odur), "-f", "concat", "-safe", "0",
                            "-i", loopf, "-c", "copy", fn
                        ]
                        print(args)
                        try:
                            resp = subprocess.run(args)
                            resp.check_returncode()
                        except subprocess.CalledProcessError as ex:
                            if resp.stderr:
                                raise RuntimeError(*ex.args, resp.stderr)
                            raise ex
                        with suppress():
                            if loopf:
                                os.remove(loopf)
                            os.remove(fn2)
            if not mid:
                return fn, outf
            self.other_x += 1
            with open(fn, "rb") as f:
                resp = Request(
                    "https://cts.ofoct.com/upload.php",
                    method="post",
                    files={"myfile": ("temp.mp3", f)},
                    timeout=32,
                    decode=True
                )
                resp_fn = literal_eval(resp)[0]
            url = f"https://cts.ofoct.com/convert-file_v2.php?cid=audio2midi&output=MID&tmpfpath={resp_fn}&row=file1&sourcename=temp.ogg&rowid=file1"
            # print(url)
            with suppress():
                os.remove(fn)
            self.other_x += 1
            resp = Request(url, timeout=720)
            self.other_x += 1
            out = Request(f"https://cts.ofoct.com/get-file.php?type=get&genfpath=/tmp/{resp_fn}.mid", timeout=32)
            return out, outf[:-4] + ".mid"

    # Extracts full data for a single entry. Uses cached results for optimization.
    def extract_single(self, i, force=False):
        item = i.url
        if not force:
            if item in self.searched and not item.startswith("ytsearch:"):
                if utc() - self.searched[item].t < 18000:
                    it = self.searched[item].data[0]
                    i.update(it)
                    if i.get("stream") not in (None, "none"):
                        return True
                else:
                    self.searched.pop(item, None)
            while len(self.searched) > 262144:
                self.searched.pop(next(iter(self.searched)))
        with self.semaphore:
            try:
                data = self.extract_true(item)
                if "entries" in data:
                    data = data["entries"][0]
                elif not issubclass(type(data), collections.abc.Mapping):
                    data = data[0]
                if data.get("research"):
                    data = self.extract_true(data["url"])[0]
                obj = cdict(t=utc())
                obj.data = out = [cdict(
                    name=data.get("title") or data.get("name"),
                    url=data.get("webpage_url") or data.get("url"),
                    stream=data.get("stream") or get_best_audio(data),
                    icon=data.get("icon") or get_best_icon(data),
                    video=data.get("video") or get_best_video(data),
                )]
                try:
                    out[0].duration = data["duration"]
                except KeyError:
                    out[0].research = True
                self.searched[item] = obj
                it = out[0]
                i.update(it)
            except:
                i.url = ""
                print_exc(item)
                return False
        return True

ytdl = AudioDownloader()


class Queue(Command):
    server_only = True
    name = ["▶️", "P", "Q", "Play", "Enqueue"]
    alias = name + ["LS"]
    description = "Shows the music queue, or plays a song in voice."
    usage = "<search_links>* <force{?f}|budge{?b}|random{?r}|verbose{?v}|hide{?h}>*"
    flags = "hvfbrz"
    no_parse = True
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    _timeout_ = 2
    rate_limit = (0.5, 3)
    typing = True
    slash = ("Play", "Queue")

    async def __call__(self, bot, user, perm, message, channel, guild, flags, name, argv, **void):
        # This command is a bit of a mess
        argv += " ".join(best_url(a) for a in message.attachments)
        if not argv:
            auds = await auto_join(guild, channel, user, bot)
            q = auds.queue
            v = "v" in flags
            if not v and len(q) and auds.paused & 1 and "p" in name:
                auds.resume()
                create_future_ex(auds.queue.update_load, timeout=120)
                return css_md(f"Successfully resumed audio playback in {sqr_md(guild)}."), 1
            if not len(q):
                auds.preparing = False
                create_future_ex(auds.update, timeout=180)
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-voice-queue-"
                + str(user.id) + "_0_" + str(int(v))
                + "-\nLoading Queue...```*"
            )
        # Get audio player as fast as possible, scheduling it to join asynchronously if necessary
        try:
            auds = bot.data.audio.players[guild.id]
            auds.channel = channel
            future = None
        except KeyError:
            future = create_task(auto_join(guild, channel, user, bot, preparing=True))
        # Start typing event asynchronously to avoid delays
        async with discord.context_managers.Typing(channel):
            # Perform search concurrently, may contain multiple URLs
            out = None
            urls = await bot.follow_url(argv, allow=True, images=False)
            if urls:
                if len(urls) == 1:
                    argv = urls[0]
                else:
                    out = [create_future(ytdl.search, url) for url in urls]
            if out is None:
                resp = await create_future(ytdl.search, argv, timeout=180)
            else:
                resp = deque()
                for fut in out:
                    temp = await fut
                    # Ignore errors when searching with multiple URLs
                    if type(temp) not in (str, bytes):
                        resp.extend(temp)
            # Wait for audio player to finish loading if necessary
            if future is not None:
                auds = await future
        if "f" in flags or "b" in flags:
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to force play while other users are in voice")
        if auds.stats.quiet & 2:
            set_dict(flags, "h", 1)
        elapsed, length = auds.epos
        q = auds.queue
        # Raise exceptions returned by searches
        if type(resp) is str:
            raise evalEX(resp)
        if "r" in flags:
            resp = (choice(resp),)
        # Assign search results to queue entries
        added = deque()
        names = []
        for i, e in enumerate(resp, 1):
            if i > 262144:
                break
            try:
                name = e.name
            except:
                print(e)
                raise
            url = e.url
            temp = {
                # "hash": e.hash,
                "name": name,
                "url": url,
                "duration": e.get("duration"),
                "u_id": user.id,
                "skips": deque(),
            }
            if "research" in e:
                temp["research"] = True
            added.append(cdict(temp))
            names.append(no_md(name))
            if not i & 16383:
                await asyncio.sleep(0.2)
        # Prepare to enqueue entries
        if "b" not in flags:
            total_duration = 0
            for e in q:
                total_duration += e_dur(e.duration)
            if auds.reverse and auds.queue:
                total_duration += elapsed - length
            else:
                total_duration -= elapsed
        if auds.stats.shuffle:
            added = shuffle(added)
        tdur = 3
        if "f" in flags:
            # Force play moves currently playing item, which means we have to unset the "played" tag from currently playing entries to allow them to play again
            auds.queue.enqueue(added, 0)
            total_duration = tdur
        elif "b" in flags:
            auds.queue.enqueue(added, 1)
            total_duration = max(3, length - elapsed if q else 0)
        else:
            auds.queue.enqueue(added, -1)
            total_duration = max(total_duration / auds.speed, tdur)
        if not names:
            raise LookupError(f"No results for {argv}.")
        if "v" in flags:
            names = no_md(alist(i.name + ": " + time_disp(e_dur(i.duration)) for i in added))
        elif len(names) == 1:
            names = names[0]
        else:
            names = f"{len(names)} items"
        if "h" not in flags:
            return css_md(f"🎶 Added {sqr_md(names)} to the queue! Estimated time until playing: {sqr_md(time_until(utc() + total_duration))}. 🎶"), 1

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos, v = [int(i) for i in vals.split("_", 2)]
        if reaction not in (None, self.directions[-1]) and u_id != user.id and perm < 3:
            return
        if reaction not in self.directions and reaction is not None:
            return
        user = await bot.fetch_user(u_id)
        guild = message.guild
        auds = await auto_join(guild, message.channel, user, bot)
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
        content = "*```" + "\n" * ("\n" in content[:i]) + (
            "callback-voice-queue-"
            + str(u_id) + "_" + str(pos) + "_" + str(int(v))
            + "-\nQueue for " + guild.name.replace("`", "") + ": "
        )
        elapsed, length = auds.epos
        startTime = 0
        if not q:
            totalTime = 0
        elif auds.stats.loop or auds.stats.repeat:
            totalTime = inf
        else:
            if auds.reverse and q:
                totalTime = elapsed - length
            else:
                totalTime = -elapsed
            i = 0
            for e in q:
                totalTime += e_dur(e.duration)
                if i < pos:
                    startTime += e_dur(e.duration)
                if not 1 + i & 32767:
                    await asyncio.sleep(0.1)
                i += 1
        cnt = len(q)
        info = (
            str(cnt) + " item" + "s" * (cnt != 1) + "\nEstimated total duration: "
            + time_until(utc() + totalTime / auds.speed) + "```*"
        )
        if not q:
            duration = 0
        else:
            duration = length
        if duration == 0:
            elapsed = 0
            duration = 0.0001
        bar = await bot.create_progress_bar(18, elapsed / duration)
        if not q:
            countstr = "Queue is currently empty.\n"
        else:
            countstr = f'{"[`" + no_md(q[0].name) + "`]"}({q[0].url})'
        countstr += f"` ({uni_str(time_disp(elapsed))}/{uni_str(time_disp(duration))})`\n{bar}\n"
        emb = discord.Embed(
            description=content + info + countstr,
            colour=rand_colour(),
        )
        emb.set_author(**get_author(user))
        if q:
            icon = q[0].get("icon", "")
        else:
            icon = ""
        if icon:
            emb.set_thumbnail(url=icon)
        async with auds.semaphore:
            embstr = ""
            currTime = startTime
            i = pos
            maxlen = 48 - int(math.log10(len(q))) if q else 48
            while i < min(pos + 10, len(q)):
                e = q[i]
                space = int(math.log10(len(q))) - int(math.log10(max(1, i)))
                curr = "`" + " " * space
                ename = no_md(e.name)
                curr += f'【{i}】 `{"[`" + no_md(lim_str(ename + " " * (maxlen - len(ename)), maxlen)) + "`]"}({ensure_url(e.url)})` ({time_disp(e_dur(e.duration))})`'
                if v:
                    try:
                        u = bot.cache.users[e.u_id]
                        name = u.display_name
                    except KeyError:
                        name = "Deleted User"
                        with suppress():
                            u = await bot.fetch_user(e.u_id)
                            name = u.display_name
                    curr += "\n" + css_md(sqr_md(name))
                if auds.reverse and len(auds.queue):
                    estim = currTime + elapsed - length
                else:
                    estim = currTime - elapsed
                if v:
                    if estim > 0:
                        curr += "Time until playing: "
                        estimate = time_until(utc() + estim / auds.speed)
                        if i <= 1 or not auds.stats.shuffle:
                            curr += "[" + estimate + "]"
                        else:
                            curr += "{" + estimate + "}"
                    else:
                        curr += "Remaining time: [" + time_until(utc() + (estim + e_dur(e.duration)) / auds.speed) + "]"
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
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if reaction is None:
            for react in self.directions:
                await message.add_reaction(as_str(react))


class Playlist(Command):
    server_only = True
    name = ["DefaultPlaylist", "PL"]
    min_display = "0~2"
    description = "Shows, appends, or removes from the default playlist."
    usage = "(add|remove)? <search_links>*"
    flags = "aedzf"
    directions = [b'\xe2\x8f\xab', b'\xf0\x9f\x94\xbc', b'\xf0\x9f\x94\xbd', b'\xe2\x8f\xac', b'\xf0\x9f\x94\x84']
    rate_limit = 0.5
    typing = True
    slash = True

    async def __call__(self, user, argv, guild, flags, channel, perm, **void):
        update = self.bot.data.playlists.update
        bot = self.bot
        if argv or "d" in flags:
            req = 2
            if perm < req:
                reason = f"to modify default playlist for {guild.name}"
                raise self.perm_error(perm, req, reason)
        pl = set_dict(bot.data.playlists, guild.id, [])
        if not argv:
            if "d" in flags:
                # This deletes all default playlist entries for the current guild
                if "f" not in flags and len(pl) > 1:
                    return css_md(sqr_md(f"WARNING: {len(pl)} ENTRIES TARGETED. REPEAT COMMAND WITH ?F FLAG TO CONFIRM."), force=True)
                bot.data.playlists.pop(guild.id)
                return italics(css_md(f"Successfully removed all {sqr_md(len(pl))} entries from the default playlist for {sqr_md(guild)}."))
            # Set callback message for scrollable list
            return (
                "*```" + "\n" * ("z" in flags) + "callback-voice-playlist-"
                + str(user.id) + "_0"
                + "-\nLoading Playlist database...```*"
            )
        if "d" in flags:
            # Can only remove by index atm
            i = await bot.eval_math(argv)
            temp = pl[i]
            pl.pop(i)
            update(guild.id)
            return italics(css_md(f"Removed {sqr_md(temp.name)} from the default playlist for {sqr_md(guild)}."))
        lim = 8 << self.bot.is_trusted(guild.id) * 2 + 1
        if len(pl) >= lim:
            raise OverflowError(f"Playlist for {guild} has reached the maximum of {lim} items. Please remove an item to add another.")
        urls = await bot.follow_url(argv, allow=True, images=False)
        if urls:
            argv = urls[0]
        async with discord.context_managers.Typing(channel):
        # Unlike ~queue this command only supports a single URL/search
            resp = await create_future(ytdl.search, argv, timeout=180)
        if type(resp) is str:
            raise evalEX(resp)
        # Assign search results to default playlist entries
        names = []
        for e in resp:
            name = e.name
            names.append(no_md(name))
            pl.append(cdict(
                name=name,
                url=e.url,
                duration=e.duration,
            ))
        if not names:
            raise LookupError(f"No results for {argv}.")
        pl.sort(key=lambda x: x["name"].casefold())
        update(guild.id)
        stuff = str(len(names)) + " items" if len(names) > 3 else ', '.join(names)
        return css_md(f"Added {sqr_md(stuff)} to the default playlist for {sqr_md(guild)}.")

    async def _callback_(self, bot, message, reaction, user, perm, vals, **void):
        u_id, pos = [int(i) for i in vals.split("_", 1)]
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
        content = "*```" + "\n" * ("\n" in content[:i]) + (
            "callback-voice-playlist-"
            + str(u_id) + "_" + str(pos)
            + "-\n"
        )
        if not pl:
            content += f"No currently enabled default playlist for {str(guild).replace('`', '')}.```*"
            msg = ""
        else:
            pl.sort(key=lambda x: x["name"].casefold())
            content += f"{len(pl)} items in default playlist for {str(guild).replace('`', '')}:```*"
            key = lambda x: lim_str(sqr_md(x["name"]) + "(" + x["url"] + ")", 1900 / page)
            msg = iter2str(pl[pos:pos + page], key=key, offset=pos, left="`【", right="】`")
        colour = await self.bot.data.colours.get(to_png_ex(guild.icon_url))
        emb = discord.Embed(
            description=content + msg,
            colour=colour,
        )
        emb.set_author(**get_author(user))
        more = len(pl) - pos - page
        if more > 0:
            emb.set_footer(text=f"{uni_str('And', 1)} {more} {uni_str('more...', 1)}")
        create_task(message.edit(content=None, embed=emb, allowed_mentions=discord.AllowedMentions.none()))
        if reaction is None:
            for react in self.directions:
                await message.add_reaction(as_str(react))


class Connect(Command):
    server_only = True
    name = ["📲", "🎤", "🎵", "🎶", "📴", "📛", "Summon", "Join", "DC", "Disconnect", "Leave", "Move", "Reconnect"]
    # Because Rythm also has this alias :P
    alias = name + ["Yeet", "FuckOff"]
    description = "Summons the bot into a voice channel."
    usage = "<channel>?"
    rate_limit = (3, 4)
    slash = ("Join", "Leave")

    async def __call__(self, user, channel, name="join", argv="", vc=None, **void):
        bot = self.bot
        joining = False
        if name in ("dc", "disconnect", "leave", "yeet", "fuckoff", "📴", "📛"):
            vc_ = None
        elif argv or name == "move":
            c_id = verify_id(argv)
            if not c_id > 0:
                vc_ = None
            else:
                vc_ = await bot.fetch_channel(c_id)
        else:
            # If voice channel is already selected, use that
            if vc is not None:
                vc_ = vc
            else:
                # Otherwise attempt to match user's currently connected voice channel
                voice = user.voice
                member = user.guild.me
                if voice is None:
                    # Otherwise attempt to find closest voice channel to current text channel
                    catg = channel.category
                    if catg is not None:
                        channels = catg.voice_channels
                    else:
                        channels = None
                    if not channels:
                        pos = 0 if channel.category is None else channel.category.position
                        # Sort by distance from text channel
                        channels = sorted(tuple(channel for channel in channel.guild.voice_channels if channel.permissions_for(member).connect and channel.permissions_for(member).speak and channel.permissions_for(member).use_voice_activation), key=lambda channel: (abs(pos - (channel.position if channel.category is None else channel.category.position)), abs(channel.position)))
                    if channels:
                        vc_ = channels[0]
                    else:
                        raise LookupError("Unable to find voice channel.")
                else:
                    vc_ = voice.channel
        # target guild may be different from source guild
        if vc_ is None:
            guild = channel.guild
        else:
            guild = vc_.guild
        # Use permission level in target guild to make sure user is able to perform command
        perm = bot.get_perms(user, guild)
        if perm < 0:
            raise self.perm_error(perm, 0, f"for command {self.name} in {guild}")
        # If no voice channel is selected, perform disconnect
        if vc_ is None:
            if argv:
                if perm < 2:
                    raise self.perm_error(perm, 2, f"for command {self.name} in {guild}")
                u_id = verify_id(argv)
                try:
                    t_user = await bot.fetch_user_member(u_id, guild)
                except (LookupError):
                    t_role = guild.get_role(u_id)
                    if t_role is None:
                        raise LookupError(f"No results for {u_id}.")
                    members = [member for member in t_role.members if member.voice is not None]
                    if not members:
                        return code_md("No members to disconnect.")
                    await disconnect_members(bot, guild, members)
                    if len(members) == 1:
                        return ini_md(f"Disconnected {sqr_md(members[0])} from {sqr_md(guild)}."), 1
                    return ini_md(f"Disconnected {sqr_md(str(members) + ' members')} from {sqr_md(guild)}."), 1
                member = guild.get_member(t_user.id)
                if not member or member.voice is None:
                    return code_md("No members to disconnect.")
                await disconnect_members(bot, guild, (member,))
                return ini_md(f"Disconnected {sqr_md(member)} from {sqr_md(guild)}."), 1
            try:
                auds = bot.data.audio.players[guild.id]
            except KeyError:
                raise LookupError("Unable to find voice channel.")
            auds.text = channel
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to disconnect while other users are in voice")
            return await create_future(auds.kill)
        if not vc_.permissions_for(guild.me).connect:
            raise ConnectionError("Insufficient permissions to connect to voice channel.")
        if vc_.permissions_for(guild.me).manage_channels:
            if guild.id in bot.data.audio.players:
                br = round(bot.data.audio.players[guild.id].stats.bitrate * 100)
            else:
                br = 196608
            bitrate = min(br, guild.bitrate_limit)
            if vc_.bitrate < bitrate:
                await vc_.edit(bitrate=bitrate, reason="I deliver maximum quality audio only! :3")
        # Create audio source if none already exists
        if guild.id not in bot.data.audio.players:
            globals()["bot"] = bot
            auds = CustomAudio(channel)
            auds.join(vc_)
            joining = True
        else:
            auds = bot.data.audio.players[guild.id]
        if guild.me.voice is None:
            await bot.wait_for("voice_state_update", check=lambda member, before, after: member.id == bot.id and after, timeout=16)
        member = guild.me
        if getattr(member, "voice", None) is not None:
            if member.voice.deaf or member.voice.mute or member.voice.afk:
                create_task(member.edit(mute=False, deafen=False))
        if joining:
            # Send update event to bot audio database upon joining
            create_task(bot.data.audio(guild=guild))
            return css_md(f"🎵 Successfully connected to {sqr_md(vc_)} in {sqr_md(guild)}. 🎵"), 1


class Skip(Command):
    server_only = True
    name = ["⏭", "🚫", "S", "SK", "FS", "CQ", "ForceSkip", "Remove", "Rem", "ClearQueue", "Clear"]
    min_display = "0~1"
    description = "Removes an entry or range of entries from the voice channel queue."
    usage = "<queue_positions(0)>* <force{?f}|vote{?v}|hide{?h}>*"
    flags = "fhv"
    rate_limit = (0.5, 3)
    slash = True

    async def __call__(self, bot, user, perm, name, args, argv, guild, flags, message, **void):
        if guild.id not in bot.data.audio.players:
            raise LookupError("Currently not playing in a voice channel.")
        auds = bot.data.audio.players[guild.id]
        auds.text = message.channel
        # ~clear is an alias for ~skip -f inf
        if name.startswith("c"):
            argv = "inf"
            args = [argv]
        if name[0] in "rcf" or "f" in flags:
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to force skip while other users are in voice")
        count = len(auds.queue)
        if not count:
            raise IndexError("Queue is currently empty.")
        # Default to currently playing item
        if not argv:
            elems = [0]
        # Accept range/slice inputs
        elif ":" in argv or ".." in argv:
            while "..." in argv:
                argv = regexp("\\.{3,}").sub("..", argv)
            l = argv.replace("..", ":").split(":")
            it = None
            if len(l) > 3:
                raise ArgumentError("Too many arguments for range input.")
            elif len(l) > 2:
                num = await bot.eval_math(l[0])
                it = int(round(num))
            if l[0]:
                num = await bot.eval_math(l[0])
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                left = num
            else:
                left = 0
            if l[1]:
                num = await bot.eval_math(l[1])
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                right = num
            else:
                right = count
            elems = xrange(left, right, it)
        else:
            # Accept multiple single indices
            elems = [0] * len(args)
            for i in range(len(args)):
                elems[i] = await bot.eval_math(args[i])
        if not "f" in flags:
            valid = True
            for e in elems:
                if not is_finite(e):
                    valid = False
                    break
            if not valid:
                elems = range(count)
        async with auds.semaphore:
            # Calculate required vote count based on amount of non-bot members in voice
            members = sum(1 for m in auds.acsi.channel.members if not m.bot)
            required = 1 + members >> 1
            response = ""
            i = 1
            for pos in elems:
                pos = float(pos)
                try:
                    # If infinite entries are selected and force flag is set, remove all items
                    if not is_finite(pos):
                        if name[0] in "rcf" or "f" in flags:
                            auds.queue.clear()
                            await create_future(auds.reset, start=False)
                            if "h" not in flags:
                                return italics(fix_md("Removed all items from the queue.")), 1
                            return
                        raise LookupError
                    curr = auds.queue[pos]
                except LookupError:
                    response += "\n" + repr(IndexError(f"Entry {pos} is out of range."))
                    continue
                # Add skips if voting
                if issubclass(type(curr.skips), collections.abc.MutableSequence):
                    if name[0] in "rcf" or "f" in flags or user.id == curr["u_id"] and not "v" in flags:
                        curr.skips = None
                    elif user.id not in curr.skips:
                        curr.skips.append(user.id)
                elif "v" in flags:
                    curr.skips = deque([user.id])
                else:
                    curr.skips = None
                if curr.skips is not None:
                    if len(response) > 1200:
                        response = lim_str(response, 1200)
                    else:
                        response += f"Voted to remove {sqr_md(curr.name)}: {sqr_md(str(len(curr.skips)) + '/' + str(required))}.\n"
                if not i & 8191:
                    await asyncio.sleep(0.2)
                i += 1
            # Get list of items to remove from the queue, based on whether they have sufficient amount of skips
            pops = set()
            count = 0
            i = 1
            while i < len(auds.queue):
                q = auds.queue
                song = q[i]
                if song.skips is None or len(song.skips) >= required:
                    if count <= 3:
                        q.pop(i)
                    else:
                        pops.add(i)
                        i += 1
                    if count < 4:
                        response += f"{sqr_md(song.name)} has been removed from the queue.\n"
                    count += 1
                else:
                    i += 1
            if pops:
                if 1 in pops:
                    auds.clear_next()
                auds.queue.pops(pops)
            if auds.queue:
                # If first item is skipped, advance queue and update audio player
                song = auds.queue[0]
                if song.skips is None or len(song.skips) >= required:
                    await create_future(auds.skip)
                    if count < 4:
                        response += f"{sqr_md(song.name)} has been removed from the queue.\n"
                    count += 1
            if "h" not in flags:
                if count >= 4:
                    return italics(css_md(f"{sqr_md(count)} items have been removed from the queue."))
                return css_md(response), 1


class Pause(Command):
    server_only = True
    name = ["⏸️", "⏯️", "⏹️", "Resume", "Unpause", "Stop"]
    min_display = "0~1"
    description = "Pauses, stops, or resumes audio playing."
    usage = "<hide{?h}>?"
    flags = "h"
    rate_limit = (0.5, 3)
    slash = True

    async def __call__(self, bot, name, guild, user, perm, channel, flags, **void):
        auds = await auto_join(guild, channel, user, bot)
        if name in ("pause", "stop", "⏸️", "⏯️", "⏹️"):
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, f"to {name} while other users are in voice")
        if name in ("resume", "unpause"):
            await create_future(auds.resume)
            word = name + "d"
        elif name in ("⏹️", "stop"):
            await create_future(auds.stop)
            word = "stopped"
        elif name in ("⏸️", "pause"):
            await create_future(auds.pause)
            word = "paused"
        else:
            await create_future(auds.pause, unpause=True)
            word = "paused" if auds.paused else "resumed"
        if "h" not in flags:
            return italics(css_md(f"Successfully {word} audio playback in {sqr_md(guild)}.")), 1


class Seek(Command):
    server_only = True
    name = ["↔️", "Replay"]
    min_display = "0~1"
    description = "Seeks to a position in the current audio file."
    usage = "<position(0)>? <hide{?h}>?"
    flags = "h"
    rate_limit = (0.5, 3)
    slash = True

    async def __call__(self, argv, bot, guild, user, perm, channel, name, flags, **void):
        auds = await auto_join(guild, channel, user, bot)
        if not is_alone(auds, user) and perm < 1:
            raise self.perm_error(perm, 1, "to seek while other users are in voice")
        # ~replay always seeks to position 0
        if name == "replay":
            num = 0
        elif not argv:
            return ini_md(f"Current audio position: {sqr_md(sec2time(auds.pos))}."), 1
        else:
            # ~seek takes an optional time input
            orig = auds.pos
            expr = argv
            num = await bot.eval_time(expr, orig)
        pos = await create_future(auds.seek, num)
        if "h" not in flags:
            return italics(css_md(f"Successfully moved audio position to {sqr_md(sec2time(pos))}.")), 1


class Dump(Command):
    server_only = True
    time_consuming = True
    name = ["Save", "Load"]
    alias = name + ["Dujmpö"]
    min_display = "0~1"
    description = "Saves or loads the currently playing audio queue state."
    usage = "<data>? <append{?a}|song_positions{?x}|hide{?h}>*"
    flags = "ahx"
    rate_limit = (1, 2)
    slash = True

    async def __call__(self, guild, channel, user, bot, perm, name, argv, flags, message, vc=None, **void):
        auds = await auto_join(guild, channel, user, bot, vc=vc)
        # ~save is the same as ~dump without an argument
        if argv == "" and not message.attachments or name == "save":
            if name == "load":
                raise ArgumentError("Please input a file or URL to load.")
            async with discord.context_managers.Typing(channel):
                resp, fn = await create_future(auds.get_dump, "x" in flags, js=True, timeout=18)
                f = CompatFile(io.BytesIO(resp), filename=fn)
            create_task(bot.send_with_file(channel, f"Queue data for {bold(str(guild))}:", f, reference=message))
            return
        if not is_alone(auds, user) and perm < 1:
            raise self.perm_error(perm, 1, "to load new queue while other users are in voice")
        if type(argv) is str:
            if message.attachments:
                url = message.attachments[0].url
            else:
                url = argv
            urls = await bot.follow_url(argv, allow=True, images=False)
            url = urls[0]
            s = await self.bot.get_request(url)
            d = await create_future(select_and_loads, s, size=268435456)
        else:
            # Queue may already be in dict form if loaded from database
            d = argv
        if type(d) is list:
            d = dict(queue=d, stats={})
        q = d["queue"][:262144]
        async with discord.context_managers.Typing(channel):
            # Copy items and cast to cdict queue entries
            for i, e in enumerate(q, 1):
                if type(e) is not cdict:
                    e = q[i - 1] = cdict(e)
                e.u_id = user.id
                e.skips = deque()
                if i > 2:
                    e.research = True
                if not i & 8191:
                    await asyncio.sleep(0.1)
            # Shuffle newly loaded dump if autoshuffle is on
            if auds.stats.shuffle and not vc:
                shuffle(q)
            for k, v in deque(d["stats"].items()):
                if k not in auds.stats:
                    d["stats"].pop(k, None)
                if k in "loop repeat shuffle quiet stay":
                    d["stats"][k] = bool(v)
                elif isinstance(v, str):
                    d["stats"][k] = mpf(v)
        if "a" not in flags:
            # Basic dump, replaces current queue
            if auds.queue:
                auds.queue.clear()
            auds.stats.update(d["stats"])
            auds.seek_pos = d.get("pos", 0)
            auds.queue.enqueue(q, -1)
            if "h" not in flags:
                return italics(css_md(f"Successfully loaded audio data for {sqr_md(guild)}.")), 1
        else:
            # append dump, adds items without replacing
            auds.queue.enqueue(q, -1)
            auds.stats.update(d["stats"])
            if "h" not in flags:
                return italics(css_md(f"Successfully appended loaded data to queue for {sqr_md(guild)}.")), 1


class AudioSettings(Command):
    server_only = True
    # Aliases are a mess lol
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
        "Bitrate": "bitrate",
        "LoopQueue": "loop",
        "Repeat": "repeat",
        "ShuffleQueue": "shuffle",
        "Quiet": "quiet",
        "Stay": "stay",
        "Reset": "reset",
    }
    aliasExt = {
        "AudioSettings": None,
        "Audio": None,
        "A": None,
        "Vol": "volume",
        "V": "volume",
        "🔉": "volume",
        "🔊": "volume",
        "📢": "volume",
        "SP": "speed",
        "⏩": "speed",
        "rewind": "rewind",
        "⏪": "rewind",
        "PI": "pitch",
        "↕️": "pitch",
        "PN": "pan",
        "BB": "bassboost",
        "🥁": "bassboost",
        "RV": "reverb",
        "📉": "reverb",
        "CO": "compressor",
        "🗜": "compressor",
        "CH": "chorus",
        "📊": "chorus",
        "NC": "resample",
        "Rate": "bitrate",
        "BPS": "bitrate",
        "BR": "bitrate",
        "LQ": "loop",
        "🔁": "loop",
        "LoopOne": "repeat",
        "🔂": "repeat",
        "L1": "repeat",
        "SQ": "shuffle",
        "🤫": "quiet",
        "🔕": "quiet",
        "24/7": "stay",
        "♻": "reset",
    }
    rate_limit = (0.5, 5)

    def __init__(self, *args):
        self.alias = list(self.aliasMap) + list(self.aliasExt)[1:]
        self.name = list(self.aliasMap)
        self.min_display = "0~2"
        self.description = "Changes the current audio settings for this server."
        self.usage = "<value>? <volume(?v)|speed(?s)|pitch(?p)|pan(?e)|bassboost(?b)|reverb(?r)|compressor(?c)|chorus(?u)|nightcore(?n)|bitrate(?i)|loop(?l)|repeat(?1)|shuffle(?x)|quiet(?q)|stay(?t)|force_permanent(?f)|disable(?d)|hide(?h)>*"
        self.flags = "vspebrcunil1xqtfdh"
        self.map = {k.casefold(): self.aliasMap[k] for k in self.aliasMap}
        add_dict(self.map, {k.casefold(): self.aliasExt[k] for k in self.aliasExt})
        super().__init__(*args)

    async def __call__(self, bot, channel, user, guild, flags, name, argv, perm, **void):
        auds = await auto_join(guild, channel, user, bot)
        ops = alist()
        op1 = self.map[name]
        if op1 == "reset":
            flags.clear()
            flags["d"] = True
        elif op1 is not None:
            ops.append(op1)
        disable = "d" in flags
        # yanderedev code moment 🙃🙃🙃
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
        if "i" in flags:
            ops.append("bitrate")
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
        # If no number input given, show audio setting
        if not disable and not argv and (len(ops) != 1 or ops[-1] not in "rewind loop repeat shuffle quiet stay"):
            if len(ops) == 1:
                op = ops[0]
            else:
                key = lambda x: x if type(x) is bool else round_min(100 * x)
                d = dict(auds.stats)
                d.pop("position", None)
                return f"Current audio settings for **{escape_markdown(guild.name)}**:\n{ini_md(iter2str(d, key=key))}"
            orig = auds.stats[op]
            num = round_min(100 * orig)
            return css_md(f"Current audio {op} setting in {sqr_md(guild)}: [{num}].")
        if not is_alone(auds, user) and perm < 1:
            raise self.perm_error(perm, 1, "to modify audio settings while other users are in voice")
        # No audio setting selected
        if not ops:
            if disable:
                # Disables all audio settings
                pos = auds.pos
                res = False
                for k, v in auds.defaults.items():
                    if k != "volume" and auds.stats.get(k) != v:
                        res = True
                        break
                auds.stats = cdict(auds.defaults)
                if "f" in flags:
                    bot.data.audiosettings.pop(guild.id, None)
                if auds.queue and res:
                    await create_future(auds.play, auds.source, pos, timeout=18)
                succ = "Permanently" if "f" in flags else "Successfully"
                return italics(css_md(f"{succ} reset all audio settings for {sqr_md(guild)}."))
            else:
                # Default to volume
                ops.append("volume")
        s = ""
        for op in ops:
            # These audio settings automatically invert when used
            if type(op) is str:
                if op in "loop repeat shuffle quiet stay" and not argv:
                    argv = str(not auds.stats[op])
                elif op == "rewind":
                    argv = "100"
            if op == "rewind":
                op = "speed"
                argv = "- " + argv
            # This disables one or more audio settings
            if disable:
                val = auds.defaults[op]
                if type(val) is not bool:
                    val *= 100
                argv = str(val)
            # Values should be scaled by 100 to indicate percentage
            origStats = auds.stats
            orig = round_min(origStats[op] * 100)
            num = await bot.eval_math(argv, orig)
            new = round_min(num)
            val = round_min(num / 100)
            if op in "loop repeat shuffle quiet stay":
                origStats[op] = new = bool(val)
                orig = bool(orig)
                if "f" in flags:
                    bot.data.audiosettings.setdefault(guild.id, {})[op] = new
                    bot.data.audiosettings.update(guild.id)
            else:
                if op == "bitrate":
                    if val > 1966.08:
                        raise PermissionError("Maximum allowed bitrate is 196608.")
                    elif val < 5.12:
                        raise ValueError("Bitrate must be equal to or above 512.")
                elif op == "speed":
                    if abs(val * 2 ** (origStats.get("resample", 0) / 12)) > 16:
                        raise OverflowError("Maximum speed is 1600%.")
                elif op == "resample":
                    if abs(origStats.get("speed", 1) * 2 ** (val / 12)) > 16:
                        raise OverflowError("Maximum speed is 1600%.")
                origStats[op] = val
                if "f" in flags:
                    bot.data.audiosettings.setdefault(guild.id, {})[op] = val
                    bot.data.audiosettings.update(guild.id)
            if auds.queue:
                if type(op) is str and op not in "loop repeat shuffle quiet stay":
                    # Attempt to adjust audio setting by re-initializing FFmpeg player
                    try:
                        await create_future(auds.play, auds.source, auds.pos, timeout=12)
                    except (TimeoutError, asyncio.exceptions.TimeoutError, concurrent.futures.TimeoutError):
                        if auds.source:
                            print(auds.args)
                        await create_future(auds.stop, timeout=18)
                        raise RuntimeError("Unable to adjust audio setting.")
            changed = "Permanently changed" if "f" in flags else "Changed"
            s += f"\n{changed} audio {op} setting from {sqr_md(orig)} to {sqr_md(new)}."
        if "h" not in flags:
            return css_md(s), 1


class Roll(Command):
    server_only = True
    name = ["🔄", "Jump"]
    min_display = "0~1"
    description = "Rotates the queue to the left by a certain amount of steps."
    usage = "<position>? <hide{?h}>?"
    flags = "h"
    rate_limit = (4, 9)

    async def __call__(self, perm, argv, flags, guild, channel, user, bot, **void):
        auds = await auto_join(guild, channel, user, bot)
        if not argv:
            amount = 1
        else:
            amount = await bot.eval_math(argv)
        if len(auds.queue) > 1 and amount:
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to rotate queue while other users are in voice")
            async with auds.semaphore:
                # Clear "played" tag of current item
                auds.queue.rotate(-amount)
                await create_future(auds.reset)
        if "h" not in flags:
            return italics(css_md(f"Successfully rotated queue [{amount}] step{'s' if amount != 1 else ''}.")), 1


class Shuffle(Command):
    server_only = True
    name = ["🔀"]
    min_display = "0~1"
    description = "Shuffles the audio queue."
    usage = "<force_full_shuffle{?f}|hide{?h}>*"
    flags = "fsh"
    rate_limit = (4, 9)
    slash = True

    async def __call__(self, perm, flags, guild, channel, user, bot, **void):
        auds = await auto_join(guild, channel, user, bot)
        if len(auds.queue) > 1:
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to shuffle queue while other users are in voice")
            async with auds.semaphore:
                if "f" in flags or "s" in flags:
                    # Clear "played" tag of current item
                    shuffle(auds.queue)
                    await create_future(auds.reset)
                else:
                    temp = auds.queue.popleft()
                    shuffle(auds.queue)
                    auds.queue.appendleft(temp)
        if "h" not in flags:
            return italics(css_md(f"Successfully shuffled queue for {sqr_md(guild)}.")), 1


class Reverse(Command):
    server_only = True
    min_display = "0~1"
    description = "Reverses the audio queue direction."
    usage = "<hide{?h}>?"
    flags = "h"
    rate_limit = (4, 9)

    async def __call__(self, perm, flags, guild, channel, user, bot, **void):
        auds = await auto_join(guild, channel, user, bot)
        if len(auds.queue) > 1:
            if not is_alone(auds, user) and perm < 1:
                raise self.perm_error(perm, 1, "to reverse queue while other users are in voice")
            async with auds.semaphore:
                reverse(auds.queue)
                auds.queue.rotate(-1)
        if "h" not in flags:
            return italics(css_md(f"Successfully reversed queue for {sqr_md(guild)}.")), 1


class UnmuteAll(Command):
    server_only = True
    time_consuming = True
    min_level = 3
    description = "Disables server mute/deafen for all members."
    usage = "<hide{?h}>?"
    flags = "h"
    rate_limit = 10

    async def __call__(self, guild, flags, **void):
        for vc in guild.voice_channels:
            for user in vc.members:
                if user.voice is not None:
                    if user.voice.deaf or user.voice.mute or user.voice.afk:
                        create_task(user.edit(mute=False, deafen=False))
        if "h" not in flags:
            return italics(css_md(f"Successfully unmuted all users in voice channels in {sqr_md(guild)}.")), 1


class VoiceNuke(Command):
    server_only = True
    time_consuming = True
    min_level = 3
    name = ["☢️"]
    description = "Removes all users from voice channels in the current server."
    usage = "<hide{?h}>?"
    flags = "h"
    rate_limit = 10

    async def __call__(self, guild, flags, **void):
        connected = set()
        for vc in guild.voice_channels:
            for user in vc.members:
                if user.id != self.bot.id:
                    if user.voice is not None:
                        connected.add(user)
        await disconnect_members(self.bot, guild, connected)
        if "h" not in flags:
            return italics(css_md(f"Successfully removed all users from voice channels in {sqr_md(guild)}.")), 1


class Radio(Command):
    name = ["FM"]
    description = "Searches for a radio station livestream on http://worldradiomap.com that can be played on ⟨MIZA⟩."
    usage = "<0:country>? <2:state>? <1:city>?"
    rate_limit = (2, 6)
    slash = True
    countries = fcdict()

    def country_repr(self, c):
        out = io.StringIO()
        start = None
        for w in c.split("_"):
            if len(w) > 1:
                if start:
                    out.write("_")
                if len(w) > 3 or not start:
                    if len(w) < 3:
                        out.write(w.upper())
                    else:
                        out.write(w.capitalize())
                else:
                    out.write(w.lower())
            else:
                out.write(w.upper())
            start = True
        out.seek(0)
        return out.read().strip("_")

    def get_countries(self):
        with tracebacksuppressor:
            resp = Request("http://worldradiomap.com", timeout=24)
            search = b'<option value="selector/_blank.htm">- Select a country -</option>'
            resp = resp[resp.index(search) + len(search):]
            resp = resp[:resp.index(b"</select>")]
            with suppress(ValueError):
                while True:
                    search = b'<option value="'
                    resp = resp[resp.index(search) + len(search):]
                    search = b'">'
                    href = as_str(resp[:resp.index(search)])
                    if not href.startswith("http"):
                        href = "http://worldradiomap.com/" + href.lstrip("/")
                    if href.endswith(".htm"):
                        href = href[:-4]
                    resp = resp[resp.index(search) + len(search):]
                    country = single_space(as_str(resp[:resp.index(b"</option>")]).replace(".", " ")).replace(" ", "_")
                    try:
                        self.countries[country].url = href
                    except KeyError:
                        self.countries[country] = cdict(url=href, cities=fcdict(), states=False)
                    data = self.countries[country]
                    alias = href.rsplit("/", 1)[-1].split("_", 1)[-1]
                    self.countries[alias] = data

                    def get_cities(country):
                        resp = Request(country.url, decode=True)
                        search = '<img src="'
                        resp = resp[resp.index(search) + len(search):]
                        icon, resp = resp.split('"', 1)
                        icon = icon.replace("../", "http://worldradiomap.com/")
                        country.icon = icon
                        search = '<option selected value="_blank.htm">- Select a city -</option>'
                        try:
                            resp = resp[resp.index(search) + len(search):]
                        except ValueError:
                            search = '<option selected value="_blank.htm">- State -</option>'
                            resp = resp[resp.index(search) + len(search):]
                            country.states = True
                            with suppress(ValueError):
                                while True:
                                    search = '<option value="'
                                    resp = resp[resp.index(search) + len(search):]
                                    search = '">'
                                    href = as_str(resp[:resp.index(search)])
                                    if not href.startswith("http"):
                                        href = "http://worldradiomap.com/selector/" + href
                                    if href.endswith(".htm"):
                                        href = href[:-4]
                                    search = "<!--"
                                    resp = resp[resp.index(search) + len(search):]
                                    city = single_space(resp[:resp.index("-->")].replace(".", " ")).replace(" ", "_")
                                    country.cities[city] = cdict(url=href, cities=fcdict(), icon=icon, states=False, get_cities=get_cities)
                        else:
                            resp = resp[:resp.index("</select>")]
                            with suppress(ValueError):
                                while True:
                                    search = '<option value="'
                                    resp = resp[resp.index(search) + len(search):]
                                    search = '">'
                                    href = as_str(resp[:resp.index(search)])
                                    if href.startswith("../"):
                                        href = "http://worldradiomap.com/" + href[3:]
                                    if href.endswith(".htm"):
                                        href = href[:-4]
                                    resp = resp[resp.index(search) + len(search):]
                                    city = single_space(resp[:resp.index("</option>")].replace(".", " ")).replace(" ", "_")
                                    country.cities[city] = href
                        return country

                    data.get_cities = get_cities

        return self.countries

    async def __call__(self, bot, channel, message, args, **void):
        if not self.countries:
            await create_future(self.get_countries)
        path = deque()
        if not args:
            fields = msdict()
            for country in self.countries:
                if len(country) > 2:
                    fields.add(country[0].upper(), self.country_repr(country))
            return bot.send_as_embeds(channel, title="Available countries", fields={k: "\n".join(v) for k, v in fields.items()}, author=get_author(bot.user), reference=message)
        c = args.pop(0)
        if c not in self.countries:
            await create_future(self.get_countries)
            if c not in self.countries:
                raise LookupError(f"Country {c} not found.")
        path.append(c)
        country = self.countries[c]
        if not country.cities:
            await create_future(country.get_cities, country)
        if not args:
            fields = msdict()
            desc = deque()
            for city in country.cities:
                desc.append(self.country_repr(city))
            t = "states" if country.states else "cities"
            return bot.send_as_embeds(channel, title=f"Available {t} in {self.country_repr(c)}", thumbnail=country.icon, description="\n".join(desc), author=get_author(bot.user), reference=message)
        c = args.pop(0)
        if c not in country.cities:
            await create_future(country.get_cities, country)
            if c not in country.cities:
                raise LookupError(f"City/State {c} not found.")
        path.append(c)
        city = country.cities[c]
        if type(city) is not str:
            state = city
            if not state.cities:
                await create_future(state.get_cities, state)
            if not args:
                fields = msdict()
                desc = deque()
                for city in state.cities:
                    desc.append(self.country_repr(city))
                return bot.send_as_embeds(channel, title=f"Available cities in {self.country_repr(c)}", thumbnail=country.icon, description="\n".join(desc), author=get_author(bot.user), reference=message)
            c = args.pop(0)
            if c not in state.cities:
                await create_future(state.get_cities, state)
                if c not in state.cities:
                    raise LookupError(f"City {c} not found.")
            path.append(c)
            city = state.cities[c]
        resp = await Request(city, aio=True)
        title = "Radio stations in " + ", ".join(self.country_repr(c) for c in reversed(path)) + ", by frequency (MHz)"
        fields = deque()
        search = b"<div class=exp>Click on the radio station name to listen online"
        resp = as_str(resp[resp.index(search) + len(search):resp.index(b"</p></div><!--end rightcontent-->")])
        for section in resp.split("<td class=tr31><b>")[1:]:
            scale = section[section.index("</b>,") + 5:section.index("Hz</td>")].upper()
            coeff = 0.000001
            if "M" in scale:
                coeff = 1
            elif "K" in scale:
                coeff = 0.001
            with tracebacksuppressor:
                while True:
                    search = "<td class=freq>"
                    search2 = "<td class=dxfreq>"
                    i = j = inf
                    with suppress(ValueError):
                        i = section.index(search) + len(search)
                    with suppress(ValueError):
                        j = section.index(search2) + len(search2)
                    if i > j:
                        i = j
                    if type(i) is not int:
                        break
                    section = section[i:]
                    freq = round_min(round(float(section[:section.index("<")].replace("&nbsp;", "").strip()) * coeff, 6))
                    field = [freq, ""]
                    curr, section = section.split("</tr>", 1)
                    for station in regexp('(?:<td class=(?:dx)?fsta2?>|\s{2,})<a href="').split(curr)[1:]:
                        if field[1]:
                            field[1] += "\n"
                        href, station = station.split('"', 1)
                        if not href.startswith("http"):
                            href = "http://worldradiomap.com/" + href.lstrip("/")
                            if href.endswith(".htm"):
                                href = href[:-4]
                        search = "class=station>"
                        station = station[station.index(search) + len(search):]
                        name = station[:station.index("<")]
                        field[1] += f"[{name.strip()}]({href.strip()})"
                    fields.append(field)
        return bot.send_as_embeds(channel, title=title, thumbnail=country.icon, fields=sorted(fields), author=get_author(bot.user), reference=message)


# This whole thing is a mess, I can't be bothered cleaning this up lol
# class Player(Command):
#     server_only = True
#     buttons = {
# 	b'\xe2\x8f\xb8': 0,
# 	b'\xf0\x9f\x94\x84': 1,
# 	b'\xf0\x9f\x94\x80': 2,
# 	b'\xe2\x8f\xae': 3,
# 	b'\xe2\x8f\xad': 4,
#         b'\xf0\x9f\x94\x8a': 5,
#         b'\xf0\x9f\xa5\x81': 6,
#         b'\xf0\x9f\x93\x89': 7,
#         b'\xf0\x9f\x93\x8a': 8,
#         b'\xe2\x8f\xaa': 9,
#         b'\xe2\x8f\xa9': 10,
#         b'\xe2\x8f\xab': 11,
#         b'\xe2\x8f\xac': 12,
#         b'\xe2\x99\xbb': 13,
# 	b'\xe2\x8f\x8f': 14,
#         b'\xe2\x9b\x94': 15,
#         }
#     barsize = 24
#     name = ["NP", "NowPlaying", "Playing"]
#     min_display = "0~3"
#     description = "Creates an auto-updating virtual audio player for the current server."
#     usage = "<enable{?e}|disable{?d}|controllable{?c}>*"
#     flags = "cdez"
#     rate_limit = (2, 7)

#     async def showCurr(self, auds):
#         q = auds.queue
#         if q:
#             s = q[0].skips
#             if s is not None:
#                 skips = len(s)
#             else:
#                 skips = 0
#             output = "Playing " + str(len(q)) + " item" + "s" * (len(q) != 1) + " "
#             output += skips * "🚫"
#         else:
#             output = "Queue is currently empty. "
#         if auds.stats.loop:
#             output += "🔄"
#         if auds.stats.shuffle:
#             output += "🔀"
#         if auds.stats.quiet:
#             output += "🔕"
#         if q:
#             p = auds.epos
#         else:
#             p = [0, 1]
#         output += "```"
#         output += await self.bot.create_progress_bar(18, p[0] / p[1])
#         if q:
#             output += "\n[`" + no_md(q[0].name) + "`](" + ensure_url(q[0].url) + ")"
#         output += "\n`"
#         if auds.paused or not auds.stats.speed:
#             output += "⏸️"
#         elif auds.stats.speed > 0:
#             output += "▶️"
#         else:
#             output += "◀️"
#         if q:
#             p = auds.epos
#         else:
#             p = [0, 0.25]
#         output += (
#             " (" + time_disp(p[0])
#             + "/" + time_disp(p[1]) + ")`\n"
#         )
#         if auds.has_options():
#             v = abs(auds.stats.volume)
#             if v == 0:
#                 output += "🔇"
#             if v <= 0.5:
#                 output += "🔉"
#             elif v <= 1:
#                 output += "🔊"
#             elif v <= 5:
#                 output += "📢"
#             else:
#                 output += "🌪️"
#             b = auds.stats.bassboost
#             if abs(b) > 1 / 6:
#                 if abs(b) > 5:
#                     output += "💥"
#                 elif b > 0:
#                     output += "🥁"
#                 else:
#                     output += "🎻"
#             r = auds.stats.reverb
#             if r:
#                 if abs(r) >= 1:
#                     output += "📈"
#                 else:
#                     output += "📉"
#             u = auds.stats.chorus
#             if u:
#                 output += "📊"
#             c = auds.stats.compressor
#             if c:
#                 output += "🗜️"
#             e = auds.stats.pan
#             if abs(e - 1) > 0.25:
#                 output += "♒"
#             s = auds.stats.speed * 2 ** (auds.stats.resample / 12)
#             if s < 0:
#                 output += "⏪"
#             elif s > 1:
#                 output += "⏩"
#             elif s > 0 and s < 1:
#                 output += "🐌"
#             p = auds.stats.pitch + auds.stats.resample
#             if p > 0:
#                 output += "⏫"
#             elif p < 0:
#                 output += "⏬"
#         return output

#     async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, **void):
#         if message is None:
#             return
#         if not guild.id in bot.data.audio.players:
#             with suppress(discord.NotFound, discord.Forbidden):
#                 await message.clear_reactions()
#             return
#         auds = bot.data.audio.players[guild.id]
#         if reaction is None:
#             auds.player = cdict(
#                 time=inf,
#                 message=message,
#                 type=int(vals),
#                 events=0,
#             )
#             if auds.player.type:
#                 auds.stats.quiet |= 2
#         elif auds.player is None or auds.player.message.id != message.id:
#             with suppress(discord.NotFound, discord.Forbidden):
#                 await message.clear_reactions()
#             return
#         if perm < 1:
#             return
#         if message.content:
#             content = message.content
#         else:
#             content = message.embeds[0].description
#         orig = "\n".join(content.splitlines()[:1 + ("\n" == content[3])]) + "\n"
#         if reaction is None and auds.player.type:
#             for b in self.buttons:
#                 async with delay(0.5):
#                     create_task(message.add_reaction(as_str(b)))
#         else:
#             if not auds.player.type:
#                 emoji = bytes()
#             elif type(reaction) is bytes:
#                 emoji = reaction
#             else:
#                 try:
#                     emoji = reaction.emoji
#                 except:
#                     emoji = str(reaction)
#             if type(emoji) is str:
#                 emoji = reaction.encode("utf-8")
#             if emoji in self.buttons:
#                 i = self.buttons[emoji]
#                 if i == 0:
#                     auds.paused ^= 1
#                 elif i == 1:
#                     auds.stats.loop = bool(auds.stats.loop ^ 1)
#                 elif i == 2:
#                     auds.stats.shuffle = bool(auds.stats.shuffle ^ 1)
#                 elif i == 3 or i == 4:
#                     if i == 3:
#                         pos = 0
#                     else:
#                         pos = inf
#                     auds.seek(pos)
#                     if pos:
#                         return
#                 elif i == 5:
#                     v = abs(auds.stats.volume)
#                     if v < 0.25 or v >= 2:
#                         v = 1 / 3
#                     elif v < 1:
#                         v = 1
#                     else:
#                         v = 2
#                     auds.stats.volume = v
#                 elif i == 6:
#                     b = auds.stats.bassboost
#                     if abs(b) < 1 / 3:
#                         b = 1
#                     elif b < 0:
#                         b = 0
#                     else:
#                         b = -1
#                     auds.stats.bassboost = b
#                 elif i == 7:
#                     r = auds.stats.reverb
#                     if r:
#                         r = 0
#                     else:
#                         r = 0.5
#                     auds.stats.reverb = r
#                 elif i == 8:
#                     c = abs(auds.stats.chorus)
#                     if c:
#                         c = 0
#                     else:
#                         c = 1 / 3
#                     auds.stats.chorus = c
#                     await create_future(auds.play, auds.source, auds.pos, timeout=18)
#                 elif i == 9 or i == 10:
#                     s = (i * 2 - 19) * 2 / 11
#                     auds.stats.speed = round(auds.stats.speed + s, 5)
#                     await create_future(auds.play, auds.source, auds.pos, timeout=18)
#                 elif i == 11 or i == 12:
#                     p = i * 2 - 23
#                     auds.stats.pitch -= p
#                     await create_future(auds.play, auds.source, auds.pos, timeout=18)
#                 elif i == 13:
#                     pos = auds.pos
#                     auds.stats = cdict(auds.defaults)
#                     await create_future(auds.play, auds.source, pos, timeout=18)
#                 elif i == 14:
#                     auds.dead = True
#                     auds.player = None
#                     await bot.silent_delete(message)
#                     return
#                 else:
#                     auds.player = None
#                     await bot.silent_delete(message)
#                     return
#         other = await self.showCurr(auds)
#         text = lim_str(orig + other, 2000)
#         last = message.channel.last_message
#         emb = discord.Embed(
#             description=text,
#             colour=rand_colour(),
#             timestamp=utc_dt(),
#         ).set_author(**get_author(self.bot.user))
#         if last is not None and (auds.player.type or message.id == last.id):
#             auds.player.events += 1
#             await message.edit(
#                 content=None,
#                 embed=emb,
#             )
#         else:
#             auds.player.time = inf
#             auds.player.events += 2
#             channel = message.channel
#             temp = message
#             message = await channel.send(
#                 content=None,
#                 embed=emb,
#             )
#             auds.player.message = message
#             await bot.silent_delete(temp, no_log=True)
#         if auds.queue and not auds.paused & 1:
#             p = auds.epos
#             maxdel = p[1] - p[0] + 2
#             delay = min(maxdel, p[1] / self.barsize / abs(auds.stats.speed))
#             if delay > 10:
#                 delay = 10
#             elif delay < 5:
#                 delay = 5
#         else:
#             delay = inf
#         auds.player.time = utc() + delay

#     async def __call__(self, guild, channel, user, bot, flags, perm, **void):
#         auds = await auto_join(channel.guild, channel, user, bot)
#         if "c" in flags or auds.stats.quiet & 2:
#             req = 3
#             if perm < req:
#                 if auds.stats.quiet & 2:
#                     if "d" in flags:
#                         reason = "delete"
#                     else:
#                         reason = "override"
#                 else:
#                     reason = "create controllable"
#                 raise self.perm_error(perm, req, f"to {reason} virtual audio player for {guild}")
#         if "d" in flags:
#             auds.player = None
#             return italics(css_md(f"Disabled virtual audio players in {sqr_md(channel.guild)}.")), 1
#         await create_player(auds, p_type="c" in flags, verbose="z" in flags)


# Small helper function to fetch song lyrics from json data, because sometimes genius.com refuses to include it in the HTML
def extract_lyrics(s):
    s = s[s.index("JSON.parse(") + len("JSON.parse("):]
    s = s[:s.index("</script>")]
    if "window.__" in s:
        s = s[:s.index("window.__")]
    s = s[:s.rindex(");")]
    data = literal_eval(s)
    d = eval_json(data)
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
                # This is a mess, the children objects may or may not represent single lines
                lyrics = line["children"] + lyrics
    return output


# Main helper function to fetch song lyrics from genius.com searches
async def get_lyrics(item):
    url = "https://api.genius.com/search"
    for i in range(2):
        header = {"Authorization": f"Bearer {genius_key}"}
        if i == 0:
            search = item
        else:
            search = "".join(shuffle(item.split()))
        data = {"q": search}
        resp = await Request(url, data=data, headers=header, aio=True, timeout=18)
        rdata = await create_future(json.loads, resp, timeout=18)
        hits = rdata["response"]["hits"]
        name = None
        path = None
        for h in hits:
            with tracebacksuppressor:
                name = h["result"]["title"]
                path = h["result"]["api_path"]
                break
        if path and name:
            s = "https://genius.com" + path
            page = await Request(s, headers=header, decode=True, aio=True)
            text = page
            html = await create_future(BeautifulSoup, text, "html.parser", timeout=18)
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
                print_exc()
                print(s)
                print(text)
    raise LookupError(f"No results for {item}.")


class Lyrics(Command):
    time_consuming = True
    name = ["SongLyrics"]
    description = "Searches genius.com for lyrics of a song."
    usage = "<search_link>* <verbose{?v}>?"
    flags = "v"
    rate_limit = (2, 6)
    typing = True

    async def __call__(self, bot, guild, channel, message, argv, flags, user, **void):
        for a in message.attachments:
            argv = a.url + " " + argv
        if not argv:
            try:
                auds = bot.data.audio.players[guild.id]
                if not auds.queue:
                    raise LookupError
                argv = auds.queue[0].name
            except LookupError:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        # Extract song name if input is a URL, otherwise search song name directly
        urls = await bot.follow_url(argv, allow=True, images=False)
        if urls:
            resp = await create_future(ytdl.search, urls[0], timeout=18)
            search = resp[0].name
        else:
            search = argv
        search = search.translate(self.bot.mtrans)
        # Attempt to find best query based on the song name
        item = verify_search(to_alphanumeric(lyric_trans.sub("", search)))
        ic = item.casefold()
        if ic.endswith(" with lyrics"):
            item = item[:-len(" with lyrics")]
        elif ic.endswith(" lyrics"):
            item = item[:-len(" lyrics")]
        elif ic.endswith(" acoustic"):
            item = item[:-len(" acoustic")]
        item = item.rsplit(" ft ", 1)[0].strip()
        if not item:
            item = verify_search(to_alphanumeric(search))
            if not item:
                item = search
        with discord.context_managers.Typing(channel):
            name, lyrics = await get_lyrics(item)
        # Escape colour markdown because that will interfere with the colours we want
        text = clr_md(lyrics.strip()).replace("#", "♯")
        msg = f"Lyrics for **{escape_markdown(name)}**:"
        s = msg + ini_md(text)
        # Directly return lyrics in a code box if it fits
        if "v" not in flags and len(s) <= 2000:
            return s
        title = f"Lyrics for {name}:"
        if len(text) > 54000:
            return (title + "\n\n" + text).strip()
        bot.send_as_embeds(channel, text, author=dict(name=title), colour=(1024, 128), md=ini_md, reference=message)


class Download(Command):
    time_consuming = True
    _timeout_ = 75
    name = ["📥", "Search", "YTDL", "Youtube_DL", "AF", "AudioFilter", "Trim", "Concat", "Concatenate", "ConvertORG", "Org2xm", "Convert"]
    description = "Searches and/or downloads a song from a YouTube/SoundCloud query or audio file link."
    usage = "<0:search_links>* <trim{?t}>? <-3:trim_start|->? <-2:trim_end|->? <-1:out_format(mp4)>? <concatenate{?c}|remove_silence{?r}|apply_settings{?a}|verbose_search{?v}>*"
    flags = "avtzcr"
    rate_limit = (7, 16)
    typing = True
    slash = True

    async def __call__(self, bot, channel, guild, message, name, argv, flags, user, **void):
        fmt = default_fmt = "mp3"
        if name in ("af", "audiofilter"):
            set_dict(flags, "a", 1)
        # Prioritize attachments in message
        for a in message.attachments:
            argv = a.url + " " + argv
        direct = getattr(message, "simulated", None)
        concat = "concat" in name or "c" in flags
        start = end = None
        # Attempt to download items in queue if no search query provided
        if not argv:
            try:
                auds = await auto_join(guild, channel, user, bot)
                if not auds.queue:
                    raise EOFError
                res = [{"name": e.name, "url": e.url} for e in auds.queue[:10]]
                fmt = "mp3"
                desc = f"Current items in queue for {guild}:"
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        else:
            # Parse search query, detecting file format selection if possible
            if " " in argv:
                try:
                    spl = smart_split(argv)
                except ValueError:
                    spl = argv.split(" ")
                if len(spl) >= 1:
                    fmt = spl[-1].lstrip(".")
                    if fmt.casefold() not in ("mp3", "ogg", "opus", "m4a", "flac", "wav", "wma", "mp2", "weba", "vox", "adpcm", "pcm", "8bit", "mid", "midi", "webm", "mp4", "avi", "mov", "m4v", "mkv", "f4v", "flv", "wmv", "gif"):
                        fmt = default_fmt
                    else:
                        if spl[-2] in ("as", "to"):
                            spl.pop(-1)
                        argv = " ".join(spl[:-1])
            if name == "trim" or "t" in flags:
                try:
                    argv, start, end = argv.rsplit(None, 2)
                except ValueError:
                    raise ArgumentError("Please input search term followed by trim start and end.")
                if start == "-":
                    start = None
                else:
                    start = await bot.eval_time(start)
                if end == "-":
                    end = None
                else:
                    end = await bot.eval_time(end)
            argv = verify_search(argv)
            res = []
            # Input may be a URL or set of URLs, in which case we attempt to find the first one
            urls = await bot.follow_url(argv, allow=True, images=False)
            if urls:
                if not concat:
                    urls = (urls[0],)
                futs = deque()
                for e in urls:
                    futs.append(create_future(ytdl.extract, e, timeout=120))
                for fut in futs:
                    temp = await fut
                    res.extend(temp)
                direct = len(res) == 1 or concat
            if not res:
                # 2 youtube results per soundcloud result, increased with verbose flag
                sc = min(4, flags.get("v", 0) + 1)
                yt = min(6, sc << 1)
                res = []
                temp = await create_future(ytdl.search, argv, mode="yt", count=yt)
                res.extend(temp)
                temp = await create_future(ytdl.search, argv, mode="sc", count=sc)
                res.extend(temp)
            if not res:
                raise LookupError(f"No results for {argv}.")
            if not concat:
                res = res[:10]
            desc = f"Search results for {argv}:"
        a = flags.get("a", 0)
        b = flags.get("r", 0)
        if concat:
            entry = (e["url"] for e in res) if concat else res[0]["url"]
            print(entry)
            with discord.context_managers.Typing(channel):
                try:
                    if a:
                        auds = bot.data.audio.players[guild.id]
                    else:
                        auds = None
                except LookupError:
                    auds = None
                f, out = await create_future(
                    ytdl.download_file,
                    entry,
                    fmt=fmt,
                    start=start,
                    end=end,
                    auds=auds,
                    silenceremove=b,
                )
                create_task(bot.send_with_file(
                    channel=channel,
                    msg="",
                    file=f,
                    filename=out,
                    rename=False,
                    reference=message
                ))
                return
        desc += "\nDestination format: {." + fmt + "}"
        if start is not None or end is not None:
            desc += f"\nTrim: [{'-' if start is None else start} ~> {'-' if end is None else end}]"
        if b:
            desc += ", Silence remover: {ON}"
        if a:
            desc += ", Audio settings: {ON}"
        desc += "```*"
        # Encode URL list into bytes and then custom base64 representation, hide in code box header
        url_bytes = bytes(repr([e["url"] for e in res]), "utf-8")
        url_enc = as_str(bytes2b64(url_bytes, True))
        vals = f"{user.id}_{len(res)}_{fmt}_{int(bool(a))}_{start}_{end}_{int(bool(b))}"
        msg = "*```" + "\n" * ("z" in flags) + "callback-voice-download-" + vals + "-" + url_enc + "\n" + desc
        emb = discord.Embed(colour=rand_colour())
        emb.set_author(**get_author(user))
        emb.description = "\n".join(f"`【{i}】` [{escape_markdown(e['name'])}]({ensure_url(e['url'])})" for i, e in enumerate(res))
        sent = await send_with_reply(channel, message, msg, embed=emb)
        if direct:
            # Automatically proceed to download and convert immediately
            create_task(self._callback_(
                message=sent,
                guild=guild,
                channel=channel,
                reaction=b"0\xef\xb8\x8f\xe2\x83\xa3",
                bot=bot,
                perm=3,
                vals=vals,
                argv=url_enc,
                user=user
            ))
        else:
            # Add reaction numbers corresponding to search results for selection
            for i in range(len(res)):
                await sent.add_reaction(str(i) + as_str(b"\xef\xb8\x8f\xe2\x83\xa3"))
        # await sent.add_reaction("❎")

    async def _callback_(self, message, guild, channel, reaction, bot, perm, vals, argv, user, **void):
        if reaction is None or user.id == bot.id:
            return
        spl = vals.split("_")
        u_id = int(spl[0])
        if user.id == u_id or not perm < 3:
            # Make sure reaction is a valid number
            if b"\xef\xb8\x8f\xe2\x83\xa3" in reaction:
                with bot.ExceptionSender(channel):
                    # Make sure selected index is valid
                    num = int(as_str(reaction)[0])
                    if num < int(spl[1]):
                        # Reconstruct list of URLs from hidden encoded data
                        data = literal_eval(as_str(b642bytes(argv, True)))
                        url = data[num]
                        # Perform all these tasks asynchronously to save time
                        with discord.context_managers.Typing(channel):
                            fmt = spl[2]
                            try:
                                if int(spl[3]):
                                    auds = bot.data.audio.players[guild.id]
                                else:
                                    auds = None
                            except LookupError:
                                auds = None
                            silenceremove = False
                            try:
                                if int(spl[6]):
                                    silenceremove = True
                            except IndexError:
                                pass
                            start = end = None
                            if len(spl) >= 6:
                                start, end = spl[4:6]
                            if tuple(map(str, (start, end))) == ("None", "None") and not silenceremove and not auds and fmt in ("mp3", "opus", "ogg", "wav"):
                                content = bot.webserver + "/ytdl?fmt=" + fmt + "&view=" + url + "\n" + bot.webserver + "/ytdl?fmt=" + fmt + "&download=" + url
                                # if message.guild and message.guild.get_member(bot.client.user.id).permissions_in(message.channel).manage_messages:
                                #     create_task(message.clear_reactions())
                                return create_task(message.channel.send(content))
                            if len(data) <= 1:
                                create_task(message.edit(
                                    content=ini_md(f"Downloading and converting {sqr_md(ensure_url(url))}..."),
                                    embed=None,
                                ))
                            else:
                                message = await message.channel.send(
                                    ini_md(f"Downloading and converting {sqr_md(ensure_url(url))}..."),
                                )
                            f, out = await create_future(
                                ytdl.download_file,
                                url,
                                fmt=fmt,
                                start=start,
                                end=end,
                                auds=auds,
                                silenceremove=silenceremove,
                            )
                            create_task(message.edit(
                                content=css_md(f"Uploading {sqr_md(out)}..."),
                                embed=None,
                            ))
                            create_task(channel.trigger_typing())
                        reference = getattr(message, "reference", None)
                        if reference:
                            r_id = getattr(reference, "message_id", None) or getattr(reference, "id", None)
                            reference = bot.cache.messages.get(r_id)
                        resp = await bot.send_with_file(
                            channel=channel,
                            msg="",
                            file=f,
                            filename=out,
                            rename=False,
                            reference=reference,
                        )
                        if resp.attachments and type(f) is str:
                            create_future_ex(os.remove, f, timeout=18, priority=True)
                        create_task(bot.silent_delete(message, no_log=True))


class UpdateAudio(Database):
    name = "audio"

    def __load__(self):
        self.players = cdict()

    # Searches for and extracts incomplete queue entries
    async def research(self, auds):
        with tracebacksuppressor:
            if not auds.search_sem.is_busy():
                async with auds.search_sem:
                    searched = 0
                    q = auds.queue
                    async with delay(2):
                        for i, e in enumerate(q, 1):
                            if searched >= 32 or i > 128:
                                break
                            if "research" in e:
                                try:
                                    await create_future(ytdl.extract_single, e, timeout=18)
                                    e.pop("research", None)
                                    searched += 1
                                except:
                                    e.pop("research", None)
                                    print_exc()
                                    break
                                e.pop("id", None)
                            if "research" not in e and not e.get("duration") and "stream" in e:
                                e["duration"] = await create_future(get_duration, e["stream"])
                            if not i & 7:
                                await asyncio.sleep(0.4)

    # Delays audio player display message by 15 seconds when a user types in the target channel
    # async def _typing_(self, channel, user, **void):
    #     if getattr(channel, "guild", None) is None:
    #         return
    #     if channel.guild.id in self.players and user.id != self.bot.id:
    #         auds = self.players[channel.guild.id]
    #         if auds.player is not None and channel.id == auds.channel.id:
    #             t = utc() + 15
    #             if auds.player.time < t:
    #                 auds.player.time = t

    # Delays audio player display message by 10 seconds when a user sends a message in the target channel
    # async def _send_(self, message, **void):
    #     if message.guild.id in self.players and message.author.id != self.bot.id:
    #         auds = self.players[message.guild.id]
    #         if auds.player is not None and message.channel.id == auds.channel.id:
    #             t = utc() + 10
    #             if auds.player.time < t:
    #                 auds.player.time = t

    # Makes 1 attempt to disconnect a single member from voice.
    async def _dc(self, member):
        with tracebacksuppressor(discord.Forbidden):
            await member.move_to(None)

    def update_vc(self, guild):
        m = guild.me
        if m:
            if guild.id not in self.players:
                if m.voice is not None:
                    acsi = AudioClientSubInterface.from_guild(guild)
                    if acsi is not None:
                        return create_future(acsi.kill)
                    return guild.change_voice_state(channel=None)
            else:
                if m.voice is not None:
                    perm = m.permissions_in(m.voice.channel)
                    if perm.mute_members and perm.deafen_members:
                        if m.voice.deaf or m.voice.mute or m.voice.afk:
                            return m.edit(mute=False, deafen=False)
        return emptyfut

    # Updates all voice clients
    async def __call__(self, guild=None, **void):
        bot = self.bot
        with tracebacksuppressor(SemaphoreOverflowError):
            async with self._semaphore:
                # Ensure all voice clients are not muted, disconnect ones without matching audio players
                if guild is not None:
                    create_task(self.update_vc(guild))
                else:
                    [create_task(self.update_vc(g)) for g in bot.cache.guilds.values()]
            # Update audio players
            if guild is not None:
                if guild.id in self.players:
                    auds = self.players[guild.id]
                    create_future_ex(auds.update, priority=True)
            else:
                a = 1
                async with delay(0.5):
                    for g in tuple(self.players):
                        with tracebacksuppressor(KeyError):
                            auds = self.players[g]
                            create_future_ex(auds.update, priority=True)
                            create_task(self.research(auds))
                            if auds.queue and not auds.paused and "dailies" in bot.data:
                                if auds.ts is None:
                                    auds.ts = utc()
                                for member in auds.acsi.channel.members:
                                    if member.id != bot.id:
                                        vs = member.voice
                                        if vs is not None and not vs.deaf and not vs.self_deaf:
                                            bot.data.dailies.progress_quests(member, "music", utc() - auds.ts)
                                auds.ts = utc()
                            else:
                                auds.ts = None
                        if not a & 15:
                            await asyncio.sleep(0.2)
                        a += 1
                await create_future(bot.audio.submit, "ytdl.update()", priority=True)
            create_future_ex(ytdl.update_dl, priority=True)

    def _announce_(self, *args, **kwargs):
        for auds in self.players.values():
            create_future_ex(auds.announce, *args, aio=True, **kwargs)

    # Stores all currently playing audio data to temporary database when bot shuts down
    async def _destroy_(self, **void):
        for auds in tuple(self.players.values()):
            d, _ = await create_future(auds.get_dump, True)
            self.data[auds.acsi.channel.id] = {"dump": d, "channel": auds.text.id}
            await create_future(auds.kill, reason="")
            self.update(auds.acsi.channel.id)
        for file in tuple(ytdl.cache.values()):
            if not file.loaded:
                await create_future(file.destroy)
        await create_future(self.update, force=True, priority=True)

    # Restores all audio players from temporary database when applicable
    async def _bot_ready_(self, bot, **void):
        globals()["bot"] = bot
        ytdl.bot = bot
        create_future_ex(ytdl.__load__)
        try:
            await create_future(subprocess.check_output, "ffmpeg")
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            print("WARNING: FFmpeg not found. Unable to convert and play audio.")
        while bot.audio is None:
            await asyncio.sleep(0.5)
        await wrap_future(bot.audio.fut)
        count = 0
        for file in os.listdir("cache"):
            if file.startswith("~") and file not in ytdl.cache:
                ytdl.cache[file] = f = await create_future(AudioFileLink, file, "cache/" + file, wasfile=True)
                count += 1
        print(f"Successfully reinstated {count} audio file{'s' if count != 1 else ''}")
        for k, v in self.data.items():
            with tracebacksuppressor:
                vc = await bot.fetch_channel(k)
                channel = await bot.fetch_channel(v["channel"])
                guild = channel.guild
                bot = bot
                user = bot.user
                perm = inf
                name = "dump"
                argv = v["dump"]
                flags = "h"
                message = cdict(attachments=None)
                for dump in bot.commands.dump:
                    print("auto-loading queue of", len(argv["queue"]), "items to", guild)
                    create_task(dump(guild, channel, user, bot, perm, name, argv, flags, message, vc=vc))
        self.data.clear()


class UpdatePlaylists(Database):
    name = "playlists"


class UpdateAudioSettings(Database):
    name = "audiosettings"