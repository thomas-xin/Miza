try:
    from common import *
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath('..'))
    os.chdir("..")
    from common import *


# Audio sample rate for both converting and playing
SAMPLE_RATE = 48000


def send(*args, escape=True):
    s = " ".join(str(i) for i in args)
    if escape:
        s = "\x00" + s
    if s:
        if s[-1] != "\n":
            s += "\n"
        sys.__stdout__.write(s)
        sys.__stdout__.flush()

def request(s):
    with tracebacksuppressor:
        PORT = AUTH["webserver_port"]
        token = AUTH["discord_token"]
        return requests.get(f"http://127.0.0.1:{PORT}/eval/{token}/{url_parse(s)}").json()["result"]

def submit(s):
    b = "~" + repr(as_str(s).encode("utf-8")) + "\n"
    resp = sys.__stdout__.write(b)
    sys.__stdout__.flush()
    return resp

async def respond(s):
    k, c = as_str(s[1:]).split("~", 1)
    c = as_str(eval(c))
    try:
        if c.startswith("await "):
            resp = await eval(c[6:], client._globals)
        else:
            code = None
            try:
                code = compile(c, "miza", "eval")
            except SyntaxError:
                pass
            else:
                resp = await create_future(eval, code, client._globals, priority=True)
            if code is None:
                resp = await create_future(exec, c, client._globals, priority=True)
    except Exception as ex:
        sys.stdout.write(traceback.format_exc())
        await create_future(submit, f"bot.audio.returns[{k}].set_exception(pickle.loads({repr(pickle.dumps(ex))}))", priority=True)
        return
    res = repr(resp)
    if type(resp) not in (bool, int, float, str, bytes):
        try:
            compile(res, "miza2", "eval")
        except SyntaxError:
            res = repr(str(resp))
    await create_future(submit, f"bot.audio.returns[{k}].set_result({res})", priority=True)

async def communicate():
    print("Audio client successfully connected.")
    while True:
        with tracebacksuppressor:
            s = await create_future(sys.stdin.readline, priority=True)
            # send(s)
            if s.startswith("~"):
                create_task(respond(s))


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration(filename, _timeout=12):
    command = subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        filename
    ])
    resp = None
    try:
        proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        fut = create_future_ex(proc.wait, timeout=_timeout)
        res = fut.result(timeout=_timeout)
        resp = float(proc.stdout.read())
    except:
        with suppress():
            proc.kill()
        print_exc()
    return resp

def get_duration(filename):
    if filename:
        dur = _get_duration(filename, 4)
        if not dur and is_url(filename):
            with requests.get(filename, headers=Request.header(), stream=True) as resp:
                head = fcdict(resp.headers)
                if "Content-Length" not in head:
                    return _get_duration(filename, 20)
                it = resp.iter_content(65536)
                data = next(it)
            ident = str(magic.from_buffer(data))
            try:
                bitrate = regexp("[0-9]+\\s.bps").findall(ident)[0].casefold()
            except IndexError:
                return _get_duration(filename, 16)
            bps, key = bitrate.split(None, 1)
            if key.startswith("k"):
                bps *= 1e3
            elif key.startswith("m"):
                bps *= 1e6
            elif key.startswith("g"):
                bps *= 1e9
            return (int(head["Content-Length"]) << 3) / bps
        return dur


players = cdict()


class AudioPlayer(discord.AudioSource):

    vc = None
    # Empty opus packet data
    emptyopus = b"\xfc\xff\xfe"

    @classmethod
    async def join(cls, channel):
        channel = client.get_channel(verify_id(channel))
        self = cls(channel.guild)
        if not self.vc:
            if channel.guild.me.voice:
                await channel.guild.change_voice_state(channel=None)
            self.vc = await channel.connect(timeout=7, reconnect=True)
        players[channel.guild.id] = self

    @classmethod
    def from_guild(cls, guild):
        try:
            return players[verify_id(guild)]
        except KeyError:
            return None
        self = cls(guild)
        if self.vc:
            return self

    def __init__(self, guild=None):
        self.queue = deque(maxlen=2)
        if guild:
            self.vc = client.get_guild(verify_id(guild)).voice_client

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        if k == "pos":
            if not self.queue or not self.queue[0]:
                return 0, 0
            p = self.queue[0][0].pos / 50
            d = self.queue[0][0].duration() or inf
            return min(p, d), d
        try:
            return getattr(self.vc, k)
        except AttributeError:
            if not self.queue:
                raise
        return getattr(self.queue[0][0], k)

    def read(self):
        if not self.queue or not self.queue[0]:
            return self.emptyopus
        out = b""
        try:
            out = self.queue[0][0].read()
        except StopIteration:
            pass
        except:
            print_exc()
        if not out:
            entry = self.queue.popleft()
            create_future_ex(entry[0].close)
            after = entry[1]
            if callable(after):
                create_future_ex(after)
            if not self.queue:
                return self.emptyopus
            with tracebacksuppressor(StopIteration):
                out = self.queue[0][0].read()
        return out

    def play(self, source, after=None):
        if not self.queue:
            self.queue.append(None)
        elif self.queue[0]:
            create_future_ex(self.queue[0][0].close)
        self.queue[0] = (source, after)
        with tracebacksuppressor(RuntimeError, discord.ClientException):
            self.vc.play(self)

    def enqueue(self, source, after=None):
        if not self.queue:
            return self.play(source, after=None)
        if len(self.queue) < 2:
            self.queue.append(None)
        self.queue[1] = (source, after)

    def clear_source(self):
        if self.queue:
            self.queue[0][0].close()
            self.queue[0] = None

    def skip(self):
        if self.queue:
            entry = self.queue.popleft()
            create_future_ex(entry[0].close)
            after = entry[1]
            if callable(after):
                create_future_ex(after)

    def clear(self):
        for entry in self.queue:
            entry[0].close()
        self.queue.clear()

    def kill(self):
        create_task(self.vc.disconnect(force=True))
        self.clear()
        players.pop(self.guild.id, None)

    is_opus = lambda self: True
    cleanup = lambda self: None

AP = AudioPlayer


cache = cdict()

def update_cache():
    for item in tuple(cache.values()):
        item.update()

ytdl = cdict(update=update_cache, cache=cache)


# Represents a cached audio file in opus format. Executes and references FFmpeg processes loading the file.
class AudioFile:

    seekable = True
    live = False

    def __init__(self, fn, stream=None, wasfile=False):
        self.file = fn
        self.proc = None
        self.streaming = concurrent.futures.Future()
        self.readable = concurrent.futures.Future()
        if stream is not None:
            self.streaming.set_result(stream)
        self.stream = stream
        self.wasfile = False
        self.loading = self.buffered = self.loaded = wasfile
        if wasfile:
            self.proc = cdict(is_running=lambda: False, kill=lambda: None)
        self.expired = False
        self.readers = cdict()
        self.semaphore = Semaphore(1, 1, delay=5)
        self.ensure_time()
        self.webpage_url = None
        cache[fn] = self

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>"

    def load(self, stream=None, check_fmt=False, force=False, webpage_url=None, live=False, seekable=True):
        if live:
            self.loading = self.buffered = self.loaded = True
            self.live = self.stream = stream
            self.seekable = seekable
            self.proc = None
            return self
        if self.loading and not force:
            return self
        if stream is not None:
            self.stream = stream
            try:
                self.streaming.set_result(stream)
            except concurrent.futures.InvalidStateError:
                pass
        stream = self.stream
        if webpage_url is not None:
            self.webpage_url = webpage_url
        self.loading = True
        # Collects data from source, converts to 48khz 192kbps opus format, outputting to target file
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-f", "opus", "-c:a", "libopus", "-ar", str(SAMPLE_RATE), "-ac", "2", "-b:a", "196608", "cache/" + self.file]
        if "https://cf-hls-media.sndcdn.com/" not in stream:
            with suppress():
                fmt = as_str(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=nokey=1:noprint_wrappers=1", stream])).strip()
                if fmt == "opus":
                    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-c:a", "copy", "cache/" + self.file]
        self.proc = None
        try:
            try:
                self.proc = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except:
                print(cmd)
                raise
            fl = 0
            # Attempt to monitor status of output file
            while fl < 4096:
                with delay(0.1):
                    if not self.proc.is_running():
                        err = as_str(self.proc.stderr.read())
                        if self.webpage_url and ("Server returned 5XX Server Error reply" in err or "Server returned 404 Not Found" in err or "Server returned 403 Forbidden" in err):
                            with tracebacksuppressor:
                                if "https://cf-hls-media.sndcdn.com/" in stream:
                                    new_stream = request(f"VOICE.get_best_audio(VOICE.ytdl.extract_from({repr(self.webpage_url)}))")
                                else:
                                    new_stream = request(f"VOICE.get_best_audio(VOICE.ytdl.extract_backup({repr(self.webpage_url)}))")
                                print(err)
                                return self.load(new_stream, check_fmt=False, force=True)
                        if check_fmt:
                            new = None
                            with suppress(ValueError):
                                new = request(f"VOICE.select_and_convert({repr(stream)})")
                            if new is not None:
                                return self.load(new, check_fmt=False, force=True)
                        print(self.proc.args)
                        if err:
                            ex = RuntimeError(err)
                        else:
                            ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
                        self.readable.set_exception(ex)
                        raise ex
                try:
                    fl = os.path.getsize("cache/" + self.file)
                except FileNotFoundError:
                    fl = 0
            self.buffered = True
            self.ensure_time()
            # print(self.file, "buffered", fl)
        except Exception as ex:
            # File errored, remove from cache and kill corresponding FFmpeg process if possible
            ytdl.cache.pop(self.file, None)
            if self.proc is not None:
                with suppress():
                    self.proc.kill()
            with suppress():
                os.remove("cache/" + self.file)
            self.readable.set_exception(ex)
            raise
        self.readable.set_result(self)
        return self

    # Touch the file to update its cache time.
    ensure_time = lambda self: setattr(self, "time", utc())

    # Update event run on all cached files
    def update(self):
        if not self.live:
            # Check when file has been fully loaded
            if self.buffered and not self.proc.is_running():
                if not self.loaded:
                    self.loaded = True
                    if not is_url(self.stream):
                        retry(os.remove, self.stream, attempts=3, delay=0.5)
                    try:
                        fl = os.path.getsize("cache/" + self.file)
                    except FileNotFoundError:
                        fl = 0
                    # print(self.file, "loaded", fl)
            # Touch file if file is currently in use
        if self.readers:
            self.ensure_time()
            return
        # Remove any unused file that has been left for a long time
        if utc() - self.time > 24000:
            try:
                fl = os.path.getsize("cache/" + self.file)
            except FileNotFoundError:
                fl = 0
                if self.buffered:
                    self.time = -inf
            ft = 24000 / (math.log2(fl / 16777216 + 1) + 1)
            if ft > 86400:
                ft = 86400
            if utc() - self.time > ft:
                self.destroy()

    # Creates a reader object that either reads bytes or opus packets from the file.
    def open(self, key=None):
        self.ensure_time()
        if self.proc is None and not self.loaded:
            raise ProcessLookupError
        f = open("cache/" + self.file, "rb")
        it = discord.oggparse.OggStream(f).iter_packets()

        reader = cdict(
            pos=0,
            file=f,
            _read=lambda self, *args: f.read(args),
            closed=False,
            advanced=False,
            is_opus=lambda self: True,
            key=key,
            duration=self.duration,
        )

        def read():
            out = next(it)
            reader.pos += 1
            return out

        def close():
            reader.closed = True
            reader.file.close()
            players.pop(reader.key, None)

        reader.read = read
        reader.close = reader.cleanup = close
        return reader

    # Destroys the file object, killing associated FFmpeg process and removing from cache.
    def destroy(self):
        self.expired = True
        if self.proc and self.proc.is_running():
            with suppress():
                self.proc.kill()
        with suppress():
            with self.semaphore:
                if not self.live:
                    retry(os.remove, "cache/" + self.file, attempts=8, delay=5, exc=(FileNotFoundError,))
                # File is removed from cache data
                ytdl.cache.pop(self.file, None)
                # print(self.file, "deleted.")

    # Creates a reader, selecting from direct opus file, single piped FFmpeg, or double piped FFmpeg.
    def create_reader(self, pos=0, auds=None, options=None, key=None):
        if self.live:
            source = self.live
        else:
            source = "cache/" + self.file
            if not os.path.exists(source):
                self.readable.result(timeout=12)
                self.load(force=True)
        stats = auds.stats
        auds.reverse = stats.speed < 0
        auds.speed = abs(stats.speed)
        if auds.speed < 0.005:
            auds.speed = 1
        players[auds.guild_id]
        stats.position = pos
        if not is_finite(stats.pitch * stats.speed):
            raise OverflowError("Speed setting out of range.")
        # Construct FFmpeg options
        if options is None:
            options = auds.construct_options(full=self.live)
        if options or auds.reverse or pos or auds.stats.bitrate != 1966.08 or self.live:
            args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            if pos and self.seekable:
                arg = "-to" if auds.reverse else "-ss"
                args += [arg, str(pos)]
            args.append("-i")
            if self.loaded:
                buff = False
                args.insert(1, "-nostdin")
                args.append(source)
            else:
                buff = True
                args.append("pipe:0")
            if options or auds.stats.bitrate != 1966.08:
                br = 100 * auds.stats.bitrate
                sr = SAMPLE_RATE
                while br < 4096:
                    br *= 2
                    sr >>= 1
                if sr < 8000:
                    sr = 8000
                options.extend(("-f", "opus", "-c:a", "libopus", "-ar", str(sr), "-ac", "2", "-b:a", str(round_min(br)), "-bufsize", "8192"))
                if options:
                    args.extend(options)
            else:
                args.extend(("-f", "opus"))
                if not self.live:
                    args.extend(("-c:a", "copy"))
            args.append("pipe:1")
            g_id = auds.guild_id
            self.readers[g_id] = True
            callback = lambda: self.readers.pop(g_id, None)
            # print(args)
            if buff:
                self.readable.result()
                # Select buffered reader for files not yet fully loaded, convert while downloading
                player = BufferedAudioReader(self, args, callback=callback, key=key)
            else:
                # Select loaded reader for loaded files
                player = LoadedAudioReader(self, args, callback=callback, key=key)
            auds.args = args
            reader = player.start()
        else:
            auds.args.clear()
            # Select raw file stream for direct audio playback
            reader = self.open(key)
        reader.pos = pos * 50
        players[key] = reader
        return reader        

    # Audio duration estimation: Get values from file if possible, otherwise URL
    duration = lambda self: inf if not self.seekable else getattr(self, "dur", None) or set_dict(self.__dict__, "dur", get_duration("cache/" + self.file) if self.loaded and not self.live else get_duration(self.stream), ignore=True)


# Audio reader for fully loaded files. FFmpeg with single pipe for output.
class LoadedAudioReader(discord.AudioSource):

    def __init__(self, file, args, callback=None, key=None):
        self.closed = False
        self.advanced = False
        self.args = args
        self.proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
        self.file = file
        self.buffer = None
        self.callback = callback
        self.pos = 0
        self.key = key
        self.duration = file.duration

    def read(self):
        if self.buffer:
            b, self.buffer = self.buffer, None
            self.pos += 1
            return b
        for att in range(16):
            try:
                out = next(self.packet_iter)
            except OSError:
                self.proc = psutil.Popen(self.args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
                self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
            else:
                self.pos += 1
                return out

    def start(self):
        self.buffer = None
        with tracebacksuppressor():
            self.buffer = self.read()
        return self

    def close(self, *void1, **void2):
        self.closed = True
        with suppress():
            self.proc.kill()
        players.pop(self.key, None)
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close


# Audio player for audio files still being written to. Continuously reads and sends data to FFmpeg process, only terminating when file download is confirmed to be finished.
class BufferedAudioReader(discord.AudioSource):

    def __init__(self, file, args, callback=None, key=None):
        self.closed = False
        self.advanced = False
        self.proc = psutil.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.packet_iter = discord.oggparse.OggStream(self.proc.stdout).iter_packets()
        self.file = file
        self.stream = open("cache/" + file.file, "rb")
        self.buffer = None
        self.callback = callback
        self.full = False
        self.pos = 0
        self.key = key
        self.duration = file.duration

    def read(self):
        if self.buffer:
            b, self.buffer = self.buffer, None
            self.pos += 1
            return b
        out = next(self.packet_iter)
        self.pos += 1
        return out

    # Required loop running in background to feed data to FFmpeg
    def run(self):
        self.file.readable.result(timeout=86400)
        while True:
            b = bytes()
            try:
                b = self.stream.read(65536)
                if not b:
                    raise EOFError
                self.proc.stdin.write(b)
                self.proc.stdin.flush()
            except (ValueError, EOFError):
                # Only stop when file is confirmed to be finished
                if self.file.loaded or self.closed:
                    break
                time.sleep(0.1)
        self.full = True
        self.proc.stdin.close()

    def start(self):
        # Run loading loop in parallel thread obviously
        create_future_ex(self.run, timeout=86400, priority=True)
        self.buffer = None
        with tracebacksuppressor():
            self.buffer = self.read()
        return self

    def close(self):
        self.closed = True
        with suppress():
            self.stream.close()
        with suppress():
            self.proc.kill()
        players.pop(self.key, None)
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close


class AudioClient(discord.Client):

    intents = discord.Intents(
        guilds=True,
        members=False,
        bans=False,
        emojis=False,
        webhooks=False,
        voice_states=True,
        presences=False,
        messages=False,
        reactions=False,
        typing=False,
    )

    def __init__(self):
        super().__init__(
            max_messages=1,
            heartbeat_timeout=60,
            guild_ready_timeout=5,
            status=discord.Status.idle,
            guild_subscriptions=False,
            intents=self.intents,
        )
        self._globals = globals()

client = AudioClient()
client.http.user_agent = "Miza-Voice"


async def mobile_identify(self):
    """Sends the IDENTIFY packet."""
    print("Overriding with mobile status...")
    payload = {
        'op': self.IDENTIFY,
        'd': {
            'token': self.token,
            'properties': {
                '$os': 'Miza-OS',
                '$browser': 'Discord Android',
                '$device': 'Miza',
                '$referrer': '',
                '$referring_domain': ''
            },
            'compress': True,
            'large_threshold': 250,
            'guild_subscriptions': self._connection.guild_subscriptions,
            'v': 3
        }
    }

    if not self._connection.is_bot:
        payload['d']['synced_guilds'] = []

    if self.shard_id is not None and self.shard_count is not None:
        payload['d']['shard'] = [self.shard_id, self.shard_count]

    state = self._connection
    if state._activity is not None or state._status is not None:
        payload['d']['presence'] = {
            'status': state._status,
            'game': state._activity,
            'since': 0,
            'afk': False
        }

    if state._intents is not None:
        payload['d']['intents'] = state._intents.value

    await self.call_hooks('before_identify', self.shard_id, initial=self._initial_identify)
    await self.send_as_json(payload)

discord.gateway.DiscordWebSocket.identify = lambda self: mobile_identify(self)


async def kill():
    futs = deque()
    await client.change_presence(status=discord.Status.invisible)
    for vc in client.voice_clients:
        futs.append(create_task(vc.disconnect(force=True)))
    for fut in futs:
        await fut
    return await client.close()


@client.event
async def before_identify_hook(shard_id, initial=False):
    pass

@client.event
async def on_ready():
    create_task(communicate())


def ensure_parent(proc, parent):
    while True:
        if not parent.is_running():
            await_fut(kill())
            psutil.Process().kill()
        # submit(f"GC.__setitem__({proc.pid}, {len(gc.get_objects())})")
        time.sleep(6)


if __name__ == "__main__":
    pid = os.getpid()
    ppid = os.getppid()
    send(f"Audio client starting with PID {pid} and parent PID {ppid}...")
    proc = psutil.Process(pid)
    parent = psutil.Process(ppid)
    create_thread(ensure_parent, proc, parent)
    client.run(AUTH["discord_token"])